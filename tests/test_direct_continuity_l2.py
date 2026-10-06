"""Adversarial tests of the L2 direct continuity formulation.

Nothing here reads the elimination through the core's own assembly: the
windowed pairing is re-assembled in plain numpy from the documented
conventions (common Legendre window, endpoint traces on normal axes, signed
mapped components), and the transfer has to satisfy the defining identities
of the L2 elimination against it. Per element the stacked constraints have to
reproduce the window Grams through the shared-object columns and annihilate
the element-private columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product

import numpy as np
import numpy.typing as npt
import pytest
from fdg import (
    BasisSpecs,
    DirectDofMap,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    Mesh,
    compute_kform_boundary_trace_moments,
)
from fdg.enum_type import BasisType

FAMILIES = (
    BasisType.LAGRANGE_UNIFORM,
    BasisType.LAGRANGE_GAUSS,
    BasisType.LAGRANGE_GAUSS_LOBATTO,
    BasisType.LAGRANGE_CHEBYSHEV_GAUSS,
    BasisType.LEGENDRE,
    BasisType.BERNSTEIN,
)

# The map prunes coefficients below a relative roundoff threshold, so the
# identities hold to the roundoff of the pairing, not to the last bit.
IDENTITY_TOLERANCE = 1.0e-9


def grid_mesh(ndim: int, cells: int) -> Mesh:
    """Structured grid of ``cells`` cells per axis, densely numbered lattice points."""
    lattice = cells + 1
    corners = np.zeros(cells**ndim * 2**ndim, dtype=np.uint64)
    for element in range(cells**ndim):
        rest = element
        lower = []
        for _ in range(ndim):
            lower.append(rest % cells)
            rest //= cells
        for corner in range(2**ndim):
            point = 0
            for axis in range(ndim):
                point = point * lattice + lower[axis] + ((corner >> axis) & 1)
            corners[element * 2**ndim + corner] = point
    return Mesh.from_corners(ndim, corners)


def corner_fan_mesh(ndim: int, element_count: int) -> Mesh:
    """Elements that meet in one common corner point and share nothing else.

    The elements take the orthants of the even-parity code, so any two differ
    on at least two axes; a structured grid cannot produce this.
    """
    corners: list[int] = []
    seen: dict[tuple[int, ...], int] = {}
    for element in range(element_count):
        if ndim == 1:
            sides = [element & 1]
        else:
            rest = element
            sides = [0] * ndim
            for axis in range(1, ndim):
                sides[axis] = rest & 1
                rest >>= 1
            sides[0] = sum(sides[1:]) % 2
        for corner in range(2**ndim):
            key = tuple(sides[axis] + ((corner >> axis) & 1) for axis in range(ndim))
            seen.setdefault(key, len(seen))
            corners.append(seen[key])
    return Mesh.from_corners(ndim, np.array(corners, dtype=np.uint64))


def grid_mesh_relabelled(ndim: int, cells: int, permutation: Sequence[int]) -> Mesh:
    """:func:`grid_mesh` with its elements renumbered, a new element ID per old one."""
    lattice = cells + 1
    corners = np.zeros(cells**ndim * 2**ndim, dtype=np.uint64)
    for new_id, old_id in enumerate(permutation):
        rest = old_id
        lower = []
        for _ in range(ndim):
            lower.append(rest % cells)
            rest //= cells
        for corner in range(2**ndim):
            point = 0
            for axis in range(ndim):
                point = point * lattice + lower[axis] + ((corner >> axis) & 1)
            corners[new_id * 2**ndim + corner] = point
    return Mesh.from_corners(ndim, corners)


def space(
    orders: tuple[int, ...], family: BasisType = BasisType.LAGRANGE_GAUSS_LOBATTO
) -> FunctionSpace:
    """Tensor space carrying one basis specification of the given family per axis."""
    return FunctionSpace(*(BasisSpecs(family, order) for order in orders))


def specs_in(
    mesh: Mesh,
    orders: Sequence[tuple[int, ...]],
    kform_order: int = 0,
    family: BasisType = BasisType.LAGRANGE_GAUSS_LOBATTO,
) -> list[KFormSpecs]:
    """One specification per element, all on one family."""
    assert len(orders) == mesh.element_count
    return [KFormSpecs(kform_order, space(axes, family)) for axes in orders]


def map_of(
    mesh: Mesh,
    orders: Sequence[tuple[int, ...]],
    kform_order: int = 0,
    family: BasisType = BasisType.LAGRANGE_GAUSS_LOBATTO,
) -> DirectDofMap:
    """Build the direct map of one mesh and assert the structural invariants."""
    assert len(orders) == mesh.element_count
    dof_map = mesh.compute_kform_direct_dof_map(
        specs_in(mesh, orders, kform_order, family)
    )
    assert_transfer_is_conforming(dof_map, mesh.element_count)
    return dof_map


def assert_transfer_is_conforming(dof_map: DirectDofMap, element_count: int) -> None:
    """Assert the documented layout contract of the row-compressed transfer.

    Every array has the documented shape, the rows partition the entries in
    ascending global order without repeating an unknown, no coefficient is a
    stored zero or a non-finite value, every degree of freedom owns at least
    one entry, and the global space is no larger than the element space.
    """
    dofs = dof_map.element_dof_count
    globals_ = dof_map.global_dof_count
    offsets = dof_map.entry_offsets
    index = dof_map.entry_index
    values = dof_map.entry_value

    assert offsets.shape == (dofs + 1,)
    assert values.shape == (dof_map.entry_count,)
    assert index.shape == (dof_map.entry_count,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == dof_map.entry_count == index.size == values.size
    assert np.all(np.isfinite(values))
    assert np.all(values != 0.0)
    assert globals_ <= dofs

    element_offsets = dof_map.element_offsets
    assert element_offsets.shape == (element_count + 1,)
    assert int(element_offsets[0]) == 0
    assert int(element_offsets[-1]) == dofs
    assert np.all(np.diff(element_offsets) > 0)

    interior = dof_map.element_interior_offsets
    assert interior.shape == (element_count + 1,)
    assert np.all(np.diff(interior) >= 0)
    assert int(interior[-1]) == globals_
    # Object unknowns come first, then one private block per element.
    assert np.all(np.diff(interior) <= np.diff(element_offsets))

    counts = np.diff(offsets)
    assert np.all(counts >= 1), "a DoF without an entry is unreachable"
    for row in range(dofs):
        span = index[int(offsets[row]) : int(offsets[row + 1])]
        assert np.all(np.diff(span) > 0), f"row {row} is not strictly ascending"
    assert np.all(index >= 0)
    assert np.all(index < globals_)
    rows = np.repeat(np.arange(dofs), counts)
    order = np.lexsort((index, rows))
    assert not np.any(
        (rows[order][1:] == rows[order][:-1]) & (index[order][1:] == index[order][:-1])
    ), "a row repeats a global unknown"


def element_columns(dof_map: DirectDofMap, element: int) -> npt.NDArray[np.double]:
    """Dense element-local rows of the transfer over all global unknowns."""
    first = int(dof_map.element_offsets[element])
    last = int(dof_map.element_offsets[element + 1])
    counts = np.diff(dof_map.entry_offsets[first : last + 1])
    rows = np.repeat(np.arange(last - first, dtype=np.intp), counts)
    start = int(dof_map.entry_offsets[first])
    stop = int(dof_map.entry_offsets[last])
    columns = np.asarray(dof_map.entry_index[start:stop], dtype=np.intp)
    values = np.asarray(dof_map.entry_value[start:stop], dtype=np.double)
    dense = np.zeros((last - first, dof_map.global_dof_count))
    dense[rows, columns] = values
    return dense


def gauss_nodes(count: int) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Gauss-Legendre nodes and weights on the reference interval [-1, 1]."""
    if count <= 0:
        return np.zeros(0), np.zeros(0)
    return np.polynomial.legendre.leggauss(count)


def basis_table(
    family: BasisType,
    order: int,
    count: int,
    points: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """First ``count`` functions of the order-``order`` basis at ``points``."""
    if count == 0:
        return np.zeros((points.size, 0))
    return np.asarray(BasisSpecs(family, order).values(points))[:, :count]


def combination_of(total: int, order: int, index: int) -> tuple[int, ...]:
    """Covector axes of the lexicographically ``index``-th combination."""
    return list(combinations(range(total), order))[index]


def orientation_records(mesh: Mesh, mdim: int) -> dict[int, dict[int, tuple[int, ...]]]:
    """Signed one-based orientation record of every element at every object.

    Returns ``{object_id: {element_id: record}}`` over all objects of
    dimension ``mdim`` the elements carry, shared or boundary ones alike.
    """
    records: dict[int, dict[int, tuple[int, ...]]] = {}
    for element in range(mesh.element_count):
        for fixed in combinations(range(mesh.ndim), mesh.ndim - mdim):
            for signs in product((1, -1), repeat=len(fixed)):
                fixed_part = [sign * (axis + 1) for axis, sign in zip(fixed, signs)]
                object_id = mesh.element_object(element, *fixed_part)
                free_part = [axis + 1 for axis in range(mesh.ndim) if axis not in fixed]
                records.setdefault(object_id, {})[element] = tuple(fixed_part + free_part)
    return records


def mapped_axes_and_sign(
    record: Sequence[int], mdim: int, ndim: int, axes: Sequence[int]
) -> tuple[tuple[int, ...], float]:
    """Element axes and parity sign of one object-frame component."""
    fixed_count = ndim - mdim
    mapped = [abs(record[fixed_count + axis]) - 1 for axis in axes]
    flips = sum(record[fixed_count + axis] < 0 for axis in axes)
    swaps = 0
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            if mapped[i] > mapped[j]:
                swaps += 1
                mapped[i], mapped[j] = mapped[j], mapped[i]
    return tuple(mapped), -1.0 if (flips + swaps) % 2 else 1.0


class WindowedPairing:
    """Independent numpy assembly of the windowed boundary pairing.

    The constraint rows pair an element's mapped k-form trace against the
    common Legendre window of one shared object: Gauss integration over the
    free axes, endpoint traces on the fixed normal axes, and the parity sign
    of the mapped covector axes. Rows follow the component and tensor-digit
    conventions of the common boundary space; columns the element k-form
    layout (components lexicographic, axis 0 the slowest digit).
    """

    def __init__(self, mesh: Mesh, specs: Sequence[KFormSpecs], kform_order: int):
        self.mesh = mesh
        self.kform_order = kform_order
        self.families = [
            tuple(spec.base_space.basis_specs[axis].type for axis in range(mesh.ndim))
            for spec in specs
        ]
        self.orders = [tuple(spec.base_space.orders) for spec in specs]
        self.records = {
            mdim: orientation_records(mesh, mdim) for mdim in range(mesh.ndim)
        }

    def object_ids(self, mdim: int) -> list[int]:
        """Every object of one dimension, in the mesh's canonical order."""
        if mdim == 0:
            return list(range(self.mesh.point_count))
        return list(range(len(self.mesh.collections[mdim - 1])))

    def window_orders(self, mdim: int, object_id: int) -> list[int]:
        """Return the common window order per object axis: poorest incident element."""
        entries = self.records[mdim].get(object_id, {})
        merged = []
        for axis in range(mdim):
            backing = [
                self.orders[element][abs(record[self.mesh.ndim - mdim + axis]) - 1]
                for element, record in entries.items()
            ]
            merged.append(min(backing))
        return merged

    def window_rows(self, mdim: int, object_id: int) -> int:
        """Total window row count of one object, over all k-form components."""
        entries = self.records[mdim].get(object_id)
        if not entries:
            return 0
        window_orders = self.window_orders(mdim, object_id)
        total = 0
        for axes in combinations(range(mdim), self.kform_order):
            total += int(
                np.prod(
                    [
                        order if axis in axes else max(order - 1, 0)
                        for axis, order in enumerate(window_orders)
                    ]
                )
            )
        return total

    def block(
        self, element: int, mdim: int, object_id: int
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        """Constraint block and window Gram of one (element, object) pair.

        Returns ``(block, gram)`` sharing their row ordering; the block is
        window rows by element DoFs, the Gram the window's mass matrix. Both
        are empty when the window is.
        """
        ndim = self.mesh.ndim
        k = self.kform_order
        if k > mdim:
            return np.zeros((0, self.element_dofs(element))), np.zeros((0, 0))
        record = self.records[mdim][object_id][element]
        window_orders = self.window_orders(mdim, object_id)
        family = self.families[element]
        orders = self.orders[element]

        free_axis = [abs(record[ndim - mdim + axis]) - 1 for axis in range(mdim)]
        mirrored = [record[ndim - mdim + axis] < 0 for axis in range(mdim)]
        fixed_axis = [abs(record[axis]) - 1 for axis in range(ndim - mdim)]
        fixed_end = [1.0 if record[axis] > 0 else -1.0 for axis in range(ndim - mdim)]

        # Per object axis: quadrature exact for every product the pairing forms.
        nodes, weights = [], []
        for axis in range(mdim):
            degree = window_orders[axis] + orders[free_axis[axis]]
            axis_nodes, axis_weights = gauss_nodes((degree + 1) // 2 + 1)
            nodes.append(axis_nodes)
            weights.append(axis_weights)

        # Element-side tables per element axis: full basis on free and fixed
        # axes, at the object's quadrature points or the fixed endpoint.
        element_tables: dict[int, npt.NDArray[np.double]] = {}
        for axis in range(ndim):
            if axis in free_axis:
                slot = free_axis.index(axis)
                points = -nodes[slot] if mirrored[slot] else nodes[slot]
            else:
                points = np.array([fixed_end[fixed_axis.index(axis)]])
            element_tables[axis] = basis_table(
                family[axis], orders[axis], orders[axis] + 1, points
            )

        components = list(combinations(range(mdim), k))
        rows = 0
        row_blocks = []
        for axes in components:
            counts = [
                order if axis in axes else max(order - 1, 0)
                for axis, order in enumerate(window_orders)
            ]
            size = int(np.prod(counts)) if counts else 1
            row_blocks.append((axes, counts, rows, size))
            rows += size
        gram = np.zeros((rows, rows))
        if rows == 0:
            return np.zeros((0, self.element_dofs(element))), gram

        # Window value tables per object axis: all functions of the order-
        # (p - 1) Legendre basis on covector axes, the leading functions of
        # the order-p basis elsewhere.
        window_tables = []
        for axis in range(mdim):
            active_count = window_orders[axis]
            trimmed_count = max(window_orders[axis] - 1, 0)
            window_tables.append(
                (
                    basis_table(
                        BasisType.LEGENDRE,
                        window_orders[axis] - 1,
                        active_count,
                        nodes[axis],
                    ),
                    basis_table(
                        BasisType.LEGENDRE,
                        window_orders[axis],
                        trimmed_count,
                        nodes[axis],
                    ),
                )
            )

        # Row tensors: window values per component at every object point.
        point_counts = [axis_nodes.size for axis_nodes in nodes]
        point_count = int(np.prod(point_counts)) if mdim else 1
        weight_grid = np.ones(1)
        for axis in range(mdim):
            weight_grid = np.kron(weight_grid, weights[axis])

        row_values = np.zeros((point_count, rows))
        for axes, counts, row_start, size in row_blocks:
            for local, digit in enumerate(product(*(range(count) for count in counts))):
                value = np.ones(1)
                for axis in range(mdim):
                    table = window_tables[axis][0 if axis in axes else 1]
                    value = np.kron(value, table[:, digit[axis]])
                row_values[:, row_start + local] = value
            gram[row_start : row_start + size, row_start : row_start + size] = (
                self._tensor_gram(window_tables, axes, counts, weights)
            )

        block = np.zeros((rows, self.element_dofs(element)))
        for axes, counts, row_start, size in row_blocks:
            if size == 0:
                continue
            mapped, sign = mapped_axes_and_sign(record, mdim, ndim, axes)
            col_counts = [
                orders[axis] if axis in mapped else orders[axis] + 1
                for axis in range(ndim)
            ]
            columns = self._component_offset(mapped, self.orders[element])
            col_values = np.zeros((point_count, int(np.prod(col_counts))))
            for local, digit in enumerate(
                product(*(range(count) for count in col_counts))
            ):
                value = np.ones(1)
                for axis in range(ndim):
                    table = element_tables[axis]
                    if axis in mapped:
                        # A covector axis reads the order-(p - 1) basis.
                        table = self._lower_table(
                            family[axis],
                            orders[axis],
                            axis,
                            free_axis,
                            mirrored,
                            fixed_end,
                            fixed_axis,
                            nodes,
                        )
                    value = np.kron(value, table[:, digit[axis]])
                col_values[:, local] = value
            block[
                row_start : row_start + size, columns : columns + col_values.shape[1]
            ] = (
                sign
                * (row_values[:, row_start : row_start + size].T * weight_grid)
                @ col_values
            )
        return block, gram

    def element_dofs(self, element: int) -> int:
        """Local DoF count of one element."""
        k = self.kform_order
        total = 0
        for index in range(len(list(combinations(range(self.mesh.ndim), k)))):
            axes = combination_of(self.mesh.ndim, k, index)
            total += self._component_dofs(axes, self.orders[element])
        return total

    def _lower_table(
        self,
        family: BasisType,
        order: int,
        axis: int,
        free_axis: Sequence[int],
        mirrored: Sequence[bool],
        fixed_end: Sequence[float],
        fixed_axis: Sequence[int],
        nodes: Sequence[npt.NDArray[np.double]],
    ) -> npt.NDArray[np.double]:
        """Order-(p - 1) element basis at the trace points of one element axis."""
        if axis in free_axis:
            slot = free_axis.index(axis)
            points = -nodes[slot] if mirrored[slot] else nodes[slot]
        else:
            points = np.array([fixed_end[fixed_axis.index(axis)]])
        return basis_table(family, order - 1, order, points)

    def _component_offset(self, mapped: tuple[int, ...], orders: tuple[int, ...]) -> int:
        """First element DoF of the k-form component with the given covector axes."""
        offset = 0
        for index in range(len(list(combinations(range(self.mesh.ndim), len(mapped))))):
            axes = combination_of(self.mesh.ndim, len(mapped), index)
            if axes == mapped:
                return offset
            offset += self._component_dofs(axes, orders)
        raise AssertionError(f"component {mapped} not enumerated")

    def _component_dofs(self, axes: tuple[int, ...], orders: tuple[int, ...]) -> int:
        """Local DoF count of one element k-form component."""
        total = 1
        next_axis = 0
        for axis, order in enumerate(orders):
            active = next_axis < len(axes) and axes[next_axis] == axis
            total *= order if active else order + 1
            next_axis += active
        return total

    def _tensor_gram(
        self,
        window_tables: Sequence[tuple[npt.NDArray[np.double], npt.NDArray[np.double]]],
        axes: tuple[int, ...],
        counts: Sequence[int],
        weights: Sequence[npt.NDArray[np.double]],
    ) -> npt.NDArray[np.double]:
        """Gram of one component's window against itself."""
        gram = np.array([[1.0]])
        for axis, count in enumerate(counts):
            table = window_tables[axis][0 if axis in axes else 1][:, :count]
            gram = np.kron(gram, table.T @ (weights[axis][:, None] * table))
        return gram


def object_layout(pairing: WindowedPairing) -> tuple[dict[tuple[int, int], int], int]:
    """Canonical object-block offsets and the total object unknown count.

    The map numbers object unknowns per dimension ascending, each object's
    window block at its own offset; this mirrors that numbering independently.
    """
    offsets = {}
    base = 0
    for mdim in range(pairing.mesh.ndim):
        for object_id in pairing.object_ids(mdim):
            offsets[(mdim, object_id)] = base
            base += pairing.window_rows(mdim, object_id)
    return offsets, base


def assert_l2_elimination(
    mesh: Mesh, specs: Sequence[KFormSpecs], dof_map: DirectDofMap
) -> None:
    """Assert the defining identities of the L2 elimination for one map.

    With ``C_e`` the stacked windowed constraint rows of element ``e`` (as the
    independent pairing assembles them), the shared-object columns of the
    transfer have to reproduce the window Grams, ``C_e R = B_e``, and the
    element-private columns the orthogonal complement, ``C_e N = 0``. The
    private columns also have to stay orthonormal, and every object with a
    non-empty window has to be reached by every element carrying it.
    """
    kform_order = specs[0].order
    pairing = WindowedPairing(mesh, specs, kform_order)
    offsets, object_globals = object_layout(pairing)
    assert object_globals == int(dof_map.element_interior_offsets[0]), (
        "the object unknowns are not the canonical window blocks the map numbers first"
    )

    covered: set[tuple[int, int]] = set()
    for element in range(mesh.element_count):
        columns = element_columns(dof_map, element)
        shared = columns[:, :object_globals]
        first = int(dof_map.element_interior_offsets[element])
        last = int(dof_map.element_interior_offsets[element + 1])
        private = columns[:, first:last]
        if private.shape[1]:
            assert np.max(np.abs(private.T @ private - np.eye(private.shape[1]))) <= (
                IDENTITY_TOLERANCE
            ), f"element {element}: private columns are not orthonormal"
        for mdim in range(mesh.ndim):
            for object_id, entries in pairing.records[mdim].items():
                rows = pairing.window_rows(mdim, object_id)
                if element not in entries or rows == 0:
                    continue
                block, gram = pairing.block(element, mdim, object_id)
                start = offsets[(mdim, object_id)]
                hit = block @ shared[:, start : start + rows]
                assert np.max(np.abs(hit - gram)) <= IDENTITY_TOLERANCE, (
                    f"element {element}, object {object_id} of dimension {mdim}: "
                    f"C R deviates from the window Gram by "
                    f"{np.max(np.abs(hit - gram)):.3e}"
                )
                if private.shape[1]:
                    leftover = block @ private
                    assert np.max(np.abs(leftover)) <= IDENTITY_TOLERANCE, (
                        f"element {element}, object {object_id} of dimension "
                        f"{mdim}: C N deviates by {np.max(np.abs(leftover)):.3e}"
                    )
                covered.add((mdim, object_id))
    for mdim in range(mesh.ndim):
        if mdim < kform_order:
            continue
        for object_id, entries in pairing.records[mdim].items():
            if pairing.window_rows(mdim, object_id) > 0:
                assert (mdim, object_id) in covered, (
                    f"object {object_id} of dimension {mdim} owns unknowns but "
                    f"constrains nothing"
                )


def assert_kernel_matches_assembly(
    mesh: Mesh, specs: Sequence[KFormSpecs], kform_order: int
) -> None:
    """Cross-check the independent pairing against the public boundary kernel.

    The kernel's dense matrices share the window row order but group columns
    by boundary component; the permutation to element-local columns is rebuilt
    here, so a disagreement names the convention that moved.
    """
    pairing = WindowedPairing(mesh, specs, kform_order)
    integrations = [
        IntegrationSpace(
            *(IntegrationSpecs(2 * order + 2) for order in spec.base_space.orders)
        )
        for spec in specs
    ]
    for mdim in range(kform_order, mesh.ndim):
        for object_id, entries in pairing.records[mdim].items():
            if pairing.window_rows(mdim, object_id) == 0:
                continue
            elements = sorted(entries)
            _, _, matrices, _ = compute_kform_boundary_trace_moments(
                [specs[element] for element in elements],
                [list(entries[element]) for element in elements],
                [integrations[element] for element in elements],
                boundary_dimension=mdim,
            )
            for element in elements:
                block, _ = pairing.block(element, mdim, object_id)
                orders = pairing.orders[element]
                permutation: list[int] = []
                for axes in combinations(range(mdim), kform_order):
                    mapped, _ = mapped_axes_and_sign(
                        entries[element], mdim, mesh.ndim, axes
                    )
                    start = pairing._component_offset(mapped, orders)
                    permutation.extend(
                        range(start, start + pairing._component_dofs(mapped, orders))
                    )
                dense = np.asarray(matrices[elements.index(element)])
                assert len(permutation) == dense.shape[1], (
                    f"element {element}, object {object_id} of dimension {mdim}: "
                    f"kernel spans {dense.shape[1]} mapped columns, the "
                    f"assembly expects {len(permutation)}"
                )
                assert np.max(np.abs(block[:, permutation] - dense)) <= 1.0e-11, (
                    f"element {element}, object {object_id} of dimension {mdim}: "
                    f"kernel and assembly disagree"
                )


def sample_field(
    local: npt.NDArray[np.double],
    orders: tuple[int, ...],
    family: BasisType,
    count: int,
) -> npt.NDArray[np.double]:
    """Evaluate one element's coefficient vector on a shared reference grid."""
    nodes, _ = gauss_nodes(count)
    tables = [np.asarray(BasisSpecs(family, order).values(nodes)) for order in orders]
    field = local.reshape([order + 1 for order in orders])
    for axis in reversed(range(len(orders))):
        field = np.tensordot(tables[axis], field, axes=([1], [field.ndim - 1]))
    return field


def solve_direct(
    ndim: int,
    order: int,
    cells: int,
    family: BasisType,
) -> tuple[Mesh, list, npt.NDArray[np.double]]:
    """Solve the weak-bc Laplace problem through one direct map.

    Returns the mesh, the element maps, and the element-major local solution.
    """
    from examples.plot_multi_element_laplace_direct_continuity import (
        assemble_global_laplace,
        boundary_object_globals,
        element_solution,
        make_element_maps,
        make_mesh,
    )

    mesh = make_mesh(ndim, cells)
    maps = make_element_maps(ndim, order + 4, cells)
    base_space = FunctionSpace(*(BasisSpecs(family, order) for _ in range(ndim)))
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    transfer = mesh.compute_kform_direct_dof_map(element_specs)
    matrix, rhs = assemble_global_laplace(transfer, maps, element_specs, base_space)
    boundary, data = boundary_object_globals(transfer, mesh, maps, order, cells)
    solution = np.zeros(transfer.global_dof_count)
    solution[boundary] = data
    free = np.setdiff1d(np.arange(transfer.global_dof_count), boundary)
    reduced = matrix[np.ix_(free, free)]
    solution[free] = np.linalg.solve(
        reduced, rhs[free] - matrix[np.ix_(free, boundary)] @ data
    )
    return mesh, maps, element_solution(transfer, solution, len(maps))


def solve_hybridized_field(
    ndim: int,
    order: int,
    cells: int,
    family: BasisType,
) -> npt.NDArray[np.double]:
    """Solve the same problem through the hybridized path, element-major."""
    from fdg import (
        MeshGeometry,
        MeshKFormSpecs,
        laplace_stiffness,
        solve_hybridized,
    )
    from fdg.integration import projection_l2_dual

    from examples.plot_multi_element_laplace_direct_continuity import (
        make_element_maps,
        make_mesh,
        manufactured_solution,
        manufactured_source,
    )

    mesh = make_mesh(ndim, cells)
    maps = make_element_maps(ndim, order + 4, cells)
    base_space = FunctionSpace(*(BasisSpecs(family, order) for _ in range(ndim)))
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    dofs_per_element = int(np.sum(element_specs[0].component_dof_counts))
    rhs = np.zeros(len(maps) * dofs_per_element)
    for element_id, element_map in enumerate(maps):
        offset = element_id * dofs_per_element
        rhs[offset : offset + dofs_per_element] = projection_l2_dual(
            manufactured_source, base_space, element_map
        ).values.flatten()
    boundary_conditions = {
        int(object_id): manufactured_solution
        for _, object_id, _, _ in mesh.iterate_boundary(ndim - 1)
    }
    constraints, constraint_rhs = mesh.compute_kform_global_constraints(
        element_specs, maps, boundary_conditions
    )
    geometry = MeshGeometry.from_elements(*maps)
    structure = MeshKFormSpecs.from_space(ndim, [("u", 0)], base_space, len(maps))
    result = solve_hybridized(
        geometry, structure, rhs, constraints, constraint_rhs, laplace_stiffness
    )
    assert result.constraint_residual <= 1.0e-10
    return np.concatenate(result.element_dofs)


def uniform_orders(mesh: Mesh, order: int) -> list[tuple[int, ...]]:
    """Give every axis of every element the same order."""
    return [tuple([order] * mesh.ndim) for _ in range(mesh.element_count)]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("ndim,cells", [(1, 2), (2, 2), (3, 2)])
def test_elimination_reproduces_the_window_grams(
    ndim: int, cells: int, family: BasisType
) -> None:
    """Every family and k-form order satisfies C R = B and C N = 0 exactly."""
    mesh = grid_mesh(ndim, cells)
    for kform_order in range(ndim + 1):
        orders = uniform_orders(mesh, 2)
        dof_map = map_of(mesh, orders, kform_order, family)
        assert_l2_elimination(mesh, specs_in(mesh, orders, kform_order, family), dof_map)


@pytest.mark.parametrize("kform_order", [0, 1])
def test_mixed_orders_share_the_poorer_window(kform_order: int) -> None:
    """Elements of different orders eliminate against the same poorer window."""
    mesh = grid_mesh(2, 2)
    orders = [(1, 2), (3, 2), (2, 4), (2, 2)]
    dof_map = map_of(mesh, orders, kform_order)
    assert_l2_elimination(mesh, specs_in(mesh, orders, kform_order), dof_map)


@pytest.mark.parametrize(
    "orders", [[(1, 6), (6, 1), (1, 6), (6, 1)], [(6, 6), (1, 1), (1, 6), (6, 1)]]
)
def test_wide_order_gaps_eliminate_consistently(orders: list[tuple[int, ...]]) -> None:
    """A 1-vs-6 order gap across shared faces still satisfies the identities."""
    mesh = grid_mesh(2, 2)
    for kform_order in range(3):
        dof_map = map_of(mesh, orders, kform_order)
        assert_l2_elimination(mesh, specs_in(mesh, orders, kform_order), dof_map)


@pytest.mark.parametrize(
    "family", [BasisType.LAGRANGE_GAUSS, BasisType.LAGRANGE_CHEBYSHEV_GAUSS]
)
def test_wide_order_gaps_keep_significant_coefficients(family: BasisType) -> None:
    """Interior-node Lagrange families keep the identities at wide gaps too.

    The two interior-node Lagrange families evaluate a shared point only
    through integration, so their mapped coefficients decay smoothly instead
    of ending at the floor; a prune threshold of 2^-12 relative dropped them
    at the 1e-4 level and broke the window-Gram identity.
    """
    mesh = grid_mesh(2, 2)
    orders = [(5, 1), (1, 5), (2, 2), (2, 2)]
    dof_map = map_of(mesh, orders, 0, family)
    assert_l2_elimination(mesh, specs_in(mesh, orders, 0, family), dof_map)


def test_wide_order_gap_in_three_dimensions() -> None:
    """The same gap holds in three dimensions at every k-form order."""
    mesh = grid_mesh(3, 2)
    orders = uniform_orders(mesh, 2)
    orders[0] = (1, 6, 2)
    orders[7] = (6, 1, 2)
    for kform_order in range(4):
        dof_map = map_of(mesh, orders, kform_order)
        assert_l2_elimination(mesh, specs_in(mesh, orders, kform_order), dof_map)


def test_mixed_families_share_one_constraint_space() -> None:
    """Every family pairs against the same window, so any mix eliminates."""
    mesh = grid_mesh(2, 2)
    orders = uniform_orders(mesh, 3)
    for kform_order in range(3):
        specs = specs_in(mesh, orders, kform_order)
        specs[0] = KFormSpecs(kform_order, space((3, 3), BasisType.LEGENDRE))
        specs[1] = KFormSpecs(kform_order, space((3, 3), BasisType.BERNSTEIN))
        specs[2] = KFormSpecs(kform_order, space((3, 3), BasisType.LAGRANGE_UNIFORM))
        specs[3] = KFormSpecs(
            kform_order, space((3, 3), BasisType.LAGRANGE_CHEBYSHEV_GAUSS)
        )
        dof_map = mesh.compute_kform_direct_dof_map(specs)
        assert_transfer_is_conforming(dof_map, mesh.element_count)
        assert_l2_elimination(mesh, specs, dof_map)


@pytest.mark.parametrize("family", FAMILIES)
def test_corner_fan_shares_only_a_point(family: BasisType) -> None:
    """Elements meeting in one corner eliminate through the point windows."""
    for ndim, count in ((1, 2), (2, 2), (3, 4)):
        mesh = corner_fan_mesh(ndim, count)
        for kform_order in range(ndim + 1):
            orders = uniform_orders(mesh, 2)
            dof_map = map_of(mesh, orders, kform_order, family)
            assert_l2_elimination(
                mesh, specs_in(mesh, orders, kform_order, family), dof_map
            )


def test_a_single_element_still_carries_its_own_boundary() -> None:
    """No neighbours, yet the element's own objects own window unknowns."""
    for ndim in (1, 2, 3):
        mesh = grid_mesh(ndim, 1)
        for kform_order in range(ndim + 1):
            orders = uniform_orders(mesh, 3)
            dof_map = map_of(mesh, orders, kform_order)
            if kform_order < ndim:
                # Every object of the lone element is carried by it alone.
                assert int(dof_map.element_interior_offsets[0]) > 0
            assert_l2_elimination(mesh, specs_in(mesh, orders, kform_order), dof_map)


def test_nothing_shared_keeps_every_mode_element_private() -> None:
    """A k-form of full degree shares nothing even on a split mesh."""
    mesh = grid_mesh(1, 3)
    for family in FAMILIES:
        dof_map = map_of(mesh, uniform_orders(mesh, 3), 1, family)
        assert dof_map.global_dof_count == dof_map.element_dof_count
        for element in range(mesh.element_count):
            first = int(dof_map.element_interior_offsets[element])
            local = element_columns(dof_map, element)
            count = local.shape[0]
            assert np.array_equal(local[:, first : first + count], np.eye(count))
            outside = local.copy()
            outside[:, first : first + count] = 0.0
            assert not np.any(outside)


def test_a_one_dimensional_form_skips_every_point() -> None:
    """A 1-form on a line has no trace on points: nothing is shared."""
    mesh = grid_mesh(1, 3)
    dof_map = map_of(mesh, uniform_orders(mesh, 3), 1)
    assert dof_map.global_dof_count == dof_map.element_dof_count
    interior = np.asarray(dof_map.element_interior_offsets)
    assert int(interior[0]) == 0
    assert int(interior[-1]) == dof_map.global_dof_count


def test_declared_but_uncarried_points_own_nothing() -> None:
    """Points no element carries add no unknowns and no constraints."""
    base = grid_mesh(2, 2)
    mesh = Mesh.from_collections(2, base.point_count + 5, base.collections)
    orders = uniform_orders(mesh, 2)
    reference = base.compute_kform_direct_dof_map(specs_in(base, orders))
    dof_map = map_of(mesh, orders)
    assert dof_map.global_dof_count == reference.global_dof_count
    assert_l2_elimination(mesh, specs_in(mesh, orders), dof_map)


def test_a_top_form_is_purely_element_local() -> None:
    """A form of full degree traces on nothing and transfers as the identity."""
    mesh = grid_mesh(3, 2)
    orders = uniform_orders(mesh, 2)
    orders[3] = (4, 2, 3)
    dof_map = map_of(mesh, orders, 3)
    assert dof_map.global_dof_count == dof_map.element_dof_count
    for element in range(mesh.element_count):
        start = int(dof_map.element_interior_offsets[element])
        local = element_columns(dof_map, element)
        count = local.shape[0]
        assert np.array_equal(local[:, start : start + count], np.eye(count)), (
            f"element {element} does not transfer as the identity"
        )
        outside = local.copy()
        outside[:, start : start + count] = 0.0
        assert not np.any(outside), f"element {element} reaches a foreign global"


def test_relabelling_elements_keeps_the_shape_of_the_map() -> None:
    """The map's size may not depend on the mesh's element numbering."""
    mesh = grid_mesh(3, 2)
    orders = uniform_orders(mesh, 2)
    orders[0] = (4, 2, 3)
    reference = map_of(mesh, orders, 1)
    for permutation in ([1, 0, 3, 2, 5, 4, 7, 6], [7, 4, 1, 6, 3, 0, 5, 2]):
        relabelled = grid_mesh_relabelled(3, 2, permutation)
        dof_map = map_of(relabelled, [orders[old] for old in permutation], 1)
        assert dof_map.global_dof_count == reference.global_dof_count
        assert dof_map.element_dof_count == reference.element_dof_count
        assert abs(int(dof_map.entry_count) - int(reference.entry_count)) <= 2


def test_axis_permutation_keeps_the_shape_of_the_map() -> None:
    """Exchanging axes and orders together leaves the map's size unchanged."""
    cells = 3
    mesh = grid_mesh(2, cells)
    reference_orders = uniform_orders(mesh, 2)
    reference_orders[0] = (5, 2)
    reference = map_of(mesh, reference_orders)

    # New axis 0 reads the old axis 1 and vice versa, so the elements' local
    # frames transpose with the mesh.
    lattice = cells + 1
    corners = np.zeros(cells**2 * 4, dtype=np.uint64)
    for element in range(cells**2):
        for corner in range(4):
            point = 0
            for axis in range(2):
                rest = element
                for _ in range((1, 0)[axis]):
                    rest //= cells
                point = point * lattice + rest % cells + ((corner >> axis) & 1)
            corners[element * 4 + corner] = point
    swapped = Mesh.from_corners(2, corners)
    other = map_of(swapped, [axes[::-1] for axes in reference_orders])
    assert other.global_dof_count == reference.global_dof_count
    assert other.element_dof_count == reference.element_dof_count
    assert abs(int(other.entry_count) - int(reference.entry_count)) <= 2


def test_the_same_request_builds_byte_identical_arrays() -> None:
    """Two builds of one request agree bit for bit, not merely in shape."""
    mesh = grid_mesh(2, 2)
    orders = [(3, 2), (2, 3), (2, 2), (4, 2)]
    specs = specs_in(mesh, orders, 1, BasisType.LEGENDRE)
    first = mesh.compute_kform_direct_dof_map(specs)
    second = mesh.compute_kform_direct_dof_map(specs)
    for name in (
        "element_offsets",
        "element_interior_offsets",
        "entry_offsets",
        "entry_index",
        "entry_value",
    ):
        first_array = np.asarray(getattr(first, name))
        second_array = np.asarray(getattr(second, name))
        assert first_array.tobytes() == second_array.tobytes(), name


def test_pruning_leaves_no_garbage_behind() -> None:
    """Wide gaps and mixed families still store only finite, reachable weights."""
    mesh = grid_mesh(3, 2)
    orders = uniform_orders(mesh, 1)
    orders[4] = (6, 1, 6)
    specs = specs_in(mesh, orders, 0)
    specs[2] = KFormSpecs(0, space((1, 1, 1), BasisType.BERNSTEIN))
    dof_map = mesh.compute_kform_direct_dof_map(specs)
    assert_transfer_is_conforming(dof_map, mesh.element_count)
    values = np.asarray(dof_map.entry_value)
    assert np.all(np.abs(values) < 1.0e3)
    counts = np.diff(dof_map.entry_offsets)
    assert np.all(counts >= 1)
    assert int(dof_map.global_dof_count) <= int(dof_map.element_dof_count)


def test_a_zero_basis_order_is_rejected_wherever_it_appears() -> None:
    """Any axis without functions makes the elimination degenerate."""
    mesh = grid_mesh(2, 2)
    for position, axes in ((2, (0, 2)), (1, (2, 0)), (3, (0, 0))):
        orders = uniform_orders(mesh, 2)
        orders[position] = axes
        with pytest.raises(ValueError, match="needs a positive basis order"):
            mesh.compute_kform_direct_dof_map(specs_in(mesh, orders))
    line = grid_mesh(1, 2)
    with pytest.raises(ValueError, match="needs a positive basis order"):
        line.compute_kform_direct_dof_map(specs_in(line, [(0,), (2,)]))


def test_wrong_degree_and_dimension_are_still_rejected() -> None:
    """The degree, dimension, and type checks keep their documented errors."""
    mesh = grid_mesh(2, 2)
    orders = uniform_orders(mesh, 2)
    specs = specs_in(mesh, orders)
    specs[3] = KFormSpecs(1, space((2, 2)))
    with pytest.raises(ValueError, match="same k-form degree"):
        mesh.compute_kform_direct_dof_map(specs)
    mismatched = [KFormSpecs(0, space((2,))) for _ in range(mesh.element_count)]
    with pytest.raises(ValueError, match="mesh dimension"):
        mesh.compute_kform_direct_dof_map(mismatched)
    with pytest.raises(TypeError, match="KFormSpecs objects"):
        mesh.compute_kform_direct_dof_map(orders)


@pytest.mark.parametrize("family", [BasisType.LAGRANGE_GAUSS_LOBATTO, BasisType.LEGENDRE])
def test_kernel_rows_match_the_assembly(family: BasisType) -> None:
    """The public boundary kernel and the independent pairing agree."""
    mesh = grid_mesh(2, 2)
    orders = [(2, 3), (2, 2), (3, 2), (2, 2)]
    for kform_order in range(2):
        specs = specs_in(mesh, orders, kform_order, family)
        assert_kernel_matches_assembly(mesh, specs, kform_order)


def test_kernel_rows_match_the_assembly_for_a_top_order_form() -> None:
    """The kernel's component-column layout also matches at order one."""
    mesh = grid_mesh(2, 2)
    orders = uniform_orders(mesh, 2)
    specs = specs_in(mesh, orders, 1, BasisType.LEGENDRE)
    assert_kernel_matches_assembly(mesh, specs, 1)


@pytest.mark.parametrize(
    "family",
    [BasisType.LAGRANGE_GAUSS_LOBATTO, BasisType.LEGENDRE, BasisType.BERNSTEIN],
)
def test_two_families_solve_the_same_discrete_problem(family: BasisType) -> None:
    """The QR parametrization is a change of basis: the solution is not.

    Two families of the same order span one element space and eliminate
    against one window, so their Laplace solutions coincide as functions.
    """
    mesh, maps, reference_local = solve_direct(2, 2, 2, BasisType.LAGRANGE_GAUSS_LOBATTO)
    _, _, local = solve_direct(2, 2, 2, family)
    for element in range(mesh.element_count):
        dofs = int(np.prod([order + 1 for order in (2, 2)]))
        span = slice(element * dofs, (element + 1) * dofs)
        reference_field = sample_field(
            reference_local[span], (2, 2), BasisType.LAGRANGE_GAUSS_LOBATTO, 6
        )
        field = sample_field(local[span], (2, 2), family, 6)
        assert np.max(np.abs(field - reference_field)) <= 1.0e-10, (
            f"element {element}: {family} deviates from the reference solve "
            f"by {np.max(np.abs(field - reference_field)):.3e}"
        )


def test_direct_solve_matches_the_hybridized_field() -> None:
    """One case cross-checked against the hybridized solver, field level."""
    mesh, _, local = solve_direct(2, 2, 2, BasisType.LAGRANGE_GAUSS_LOBATTO)
    hybridized = solve_hybridized_field(2, 2, 2, BasisType.LAGRANGE_GAUSS_LOBATTO)
    for element in range(mesh.element_count):
        dofs = int(np.prod([order + 1 for order in (2, 2)]))
        span = slice(element * dofs, (element + 1) * dofs)
        field = sample_field(local[span], (2, 2), BasisType.LAGRANGE_GAUSS_LOBATTO, 6)
        reference = sample_field(
            hybridized[span], (2, 2), BasisType.LAGRANGE_GAUSS_LOBATTO, 6
        )
        assert np.max(np.abs(field - reference)) <= 1.0e-10, (
            f"element {element}: the direct solve deviates from the "
            f"hybridized one by {np.max(np.abs(field - reference)):.3e}"
        )
