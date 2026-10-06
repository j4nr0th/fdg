"""Adversarial tests of the direct element-to-global transfer.

Every case asserts a property any conforming map must satisfy, so a wrong
answer fails rather than merely differing from an expected one.
:func:`assert_windowed_projections_agree` checks the defining property of the
L2 elimination on random global vectors.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations, product
from math import comb

import numpy as np
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

# Families the boundary kernel evaluates exactly at shared objects for these orders.
EXACT_PROJECTION_FAMILIES = (
    "lagrange-uniform",
    "lagrange-gauss-lobatto",
    "legendre",
    "bernstein",
)


def grid_mesh(ndim: int, cells: int) -> Mesh:
    """Structured grid of ``cells`` cells per axis with densely numbered lattice points.

    Element ``e`` spans the cell whose lower corner is its base-``cells`` digit
    tuple, and corner ``c`` adds one to the axes whose bits are set.
    """
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


def grid_mesh_mirrored(
    ndim: int, cells: int, flipped: Mapping[int, tuple[int, ...]]
) -> Mesh:
    """:func:`grid_mesh` with the local axes of the given elements reversed.

    An element's orientation is read off its corner list, so this gives the same
    physical element a negative orientation without changing which objects it
    shares.
    """
    lattice = cells + 1
    corners = np.zeros(cells**ndim * 2**ndim, dtype=np.uint64)
    for element in range(cells**ndim):
        rest = element
        lower = []
        for _ in range(ndim):
            lower.append(rest % cells)
            rest //= cells
        sides = flipped.get(element, ())
        for corner in range(2**ndim):
            point = 0
            for axis in range(ndim):
                bit = (corner >> axis) & 1
                if axis in sides:
                    bit ^= 1
                point = point * lattice + lower[axis] + bit
            corners[element * 2**ndim + corner] = point
    return Mesh.from_corners(ndim, corners)


def grid_mesh_axes_swapped(ndim: int, cells: int, axes: tuple[int, ...]) -> Mesh:
    """:func:`grid_mesh` with its axes exchanged, old ``axes[a]`` read as axis ``a``."""
    lattice = cells + 1
    corners = np.zeros(cells**ndim * 2**ndim, dtype=np.uint64)
    for element in range(cells**ndim):
        for corner in range(2**ndim):
            point = 0
            for axis in range(ndim):
                rest = element
                for _ in range(axes[axis]):
                    rest //= cells
                point = point * lattice + rest % cells + ((corner >> axis) & 1)
            corners[element * 2**ndim + corner] = point
    return Mesh.from_corners(ndim, corners)


def corner_fan_mesh(ndim: int, element_count: int) -> Mesh:
    """Elements that meet in one common corner point and share nothing else.

    The elements take the orthants of the even-parity code, so any two differ on
    at least two axes. A structured grid cannot produce this: its elements
    meeting at an interior point also share a face through it.
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


def space(
    orders: tuple[int, ...], family: BasisType = BasisType.LAGRANGE_GAUSS_LOBATTO
) -> FunctionSpace:
    """Tensor space carrying one basis specification of the given family per axis."""
    return FunctionSpace(*(BasisSpecs(family, order) for order in orders))


def uniform_orders(mesh: Mesh, order: int) -> list[tuple[int, ...]]:
    """Give every axis of every element the same order."""
    return [tuple([order] * mesh.ndim) for _ in range(mesh.element_count)]


def specs_in(
    orders: Sequence[tuple[int, ...]],
    kform_order: int = 0,
    family: BasisType = BasisType.LAGRANGE_GAUSS_LOBATTO,
) -> list[KFormSpecs]:
    """One specification per element, all on one Lagrange family."""
    return [KFormSpecs(kform_order, space(axes, family)) for axes in orders]


def map_of(
    mesh: Mesh,
    orders: Sequence[tuple[int, ...]],
    kform_order: int = 0,
    family: BasisType = BasisType.LAGRANGE_GAUSS_LOBATTO,
) -> DirectDofMap:
    """Build the direct map of one mesh and assert that it is conforming."""
    assert len(orders) == mesh.element_count
    dof_map = mesh.compute_kform_direct_dof_map(specs_in(orders, kform_order, family))
    assert_conforming(dof_map, mesh.element_count)
    return dof_map


def assert_conforming(dof_map: DirectDofMap, element_count: int) -> None:
    """Assert the structural invariants any conforming transfer must satisfy."""
    dofs = dof_map.element_dof_count
    globals_ = dof_map.global_dof_count
    offsets = dof_map.entry_offsets
    index = dof_map.entry_index
    values = dof_map.entry_value

    assert offsets.shape == (dofs + 1,)
    assert values.shape == (dof_map.entry_count,)
    assert index.shape == (dof_map.entry_count,)
    assert int(offsets[0]) == 0
    assert np.all(np.diff(offsets) >= 0)
    assert int(offsets[-1]) == dof_map.entry_count == index.size
    assert np.all(index >= 0)
    assert np.all(index < globals_)
    # Every global unknown has to be reached, or the system has an empty row.
    assert np.unique(index).size == globals_
    assert globals_ <= dofs
    # A stored zero would be a different map with the same shape.
    assert np.all(np.isfinite(values))
    assert np.all(values != 0.0)

    element_offsets = dof_map.element_offsets
    assert element_offsets.shape == (element_count + 1,)
    assert int(element_offsets[0]) == 0
    assert int(element_offsets[-1]) == dofs
    assert np.all(np.diff(element_offsets) > 0)

    interior = dof_map.element_interior_offsets
    assert interior.shape == (element_count + 1,)
    assert np.all(np.diff(interior) >= 0)
    assert int(interior[-1]) == globals_

    # No element-local degree of freedom may reach one global degree of freedom twice.
    rows = np.repeat(np.arange(dofs), np.diff(offsets))
    order = np.lexsort((index, rows))
    sorted_rows = rows[order]
    sorted_index = index[order]
    assert not np.any(
        (sorted_rows[1:] == sorted_rows[:-1]) & (sorted_index[1:] == sorted_index[:-1])
    )


def assert_same_entry_budget(dof_map: DirectDofMap, reference: DirectDofMap) -> None:
    """Assert two structurally identical maps store nearly the same entry count.

    Coefficients within the block's relative roundoff threshold are pruned, so
    the count wobbles with the QR row ordering, never by structure.
    """
    budget = max(reference.entry_count, 100)
    assert abs(int(dof_map.entry_count) - int(reference.entry_count)) <= budget // 100


def orientation_record(mesh: Mesh, element: int, object_id: int, mdim: int) -> list[int]:
    """Build the signed one-based axis permutation mapping ``element`` onto ``object_id``.

    The first ``ndim - mdim`` entries name the fixed normal axes (positive for an
    end side, negative for a start side); the remaining entries map the free
    axes in the object's canonical order.
    """
    for fixed in combinations(range(mesh.ndim), mesh.ndim - mdim):
        for signs in product((1, -1), repeat=len(fixed)):
            fixed_part = [sign * (axis + 1) for axis, sign in zip(fixed, signs)]
            if mesh.element_object(element, *fixed_part) == object_id:
                free_part = [axis + 1 for axis in range(mesh.ndim) if axis not in fixed]
                return fixed_part + free_part
    pytest.fail(f"element {element} carries no side of object {object_id}")


def scattered_element_values(dof_map: DirectDofMap, u_global: np.ndarray) -> np.ndarray:
    """Apply the transfer to a random global vector, element by element."""
    u_element = np.zeros(dof_map.element_dof_count)
    for element in range(dof_map.element_offsets.size - 1):
        first = int(dof_map.element_offsets[element])
        for local in range(int(dof_map.element_offsets[element + 1]) - first):
            total = 0.0
            for entry in range(
                int(dof_map.entry_offsets[first + local]),
                int(dof_map.entry_offsets[first + local + 1]),
            ):
                total += dof_map.entry_value[entry] * u_global[dof_map.entry_index[entry]]
            u_element[first + local] = total
    return u_element


def assert_windowed_projections_agree(
    mesh: Mesh,
    specs: Sequence[KFormSpecs],
    dof_map: DirectDofMap,
) -> None:
    """Assert the defining property of the elimination for a random global vector.

    Every element's transferred trace must give the same window moments on all
    sides of a shared object. Only 0-forms are covered: the boundary kernel
    exposes its mapped trace columns in the element's own ordering only there,
    so the transferred element vector can be paired with the rows directly.
    """
    kform_order = specs[0].order
    assert kform_order == 0
    family = specs[0].base_space.basis_specs[0].type
    exact = family in EXACT_PROJECTION_FAMILIES
    # Otherwise the kernel's integration-node trace evaluation leaves a residual
    # (measured 5.1e-4 absolute at order 3); the map enforces moments to roundoff.
    tolerance = 1.0e-10 if exact else 1.0e-3
    rng = np.random.default_rng(1789)
    u_element = scattered_element_values(
        dof_map, rng.standard_normal(dof_map.global_dof_count)
    )
    integrations = [
        IntegrationSpace(
            *(IntegrationSpecs(2 * order + 2) for order in spec.base_space.orders)
        )
        for spec in specs
    ]

    for mdim in range(mesh.ndim):
        if mdim < kform_order:
            # A k-form pulls back to zero on an object of lower dimension:
            # the window is empty and the kernel rejects the request.
            continue
        for _, object_id, elements, _ in mesh.iterate_shared(mdim):
            records = [
                orientation_record(mesh, int(element), int(object_id), mdim)
                for element in elements
            ]
            result = compute_kform_boundary_trace_moments(
                [specs[int(element)] for element in elements],
                records,
                [integrations[int(element)] for element in elements],
                boundary_dimension=mdim,
                packed=True,
            )
            packed = result[3]
            assert packed is not None
            moments = []
            for side, element in enumerate(elements):
                offsets, index, values = (
                    np.asarray(packed[side][0]),
                    np.asarray(packed[side][3]),
                    np.asarray(packed[side][4]),
                )
                base = int(dof_map.element_offsets[int(element)])
                local = u_element[base + index]
                rows = offsets.size - 1
                moment = np.empty(rows)
                for row in range(rows):
                    span = range(int(offsets[row]), int(offsets[row + 1]))
                    moment[row] = float(np.dot(values[span], local[span]))
                moments.append(moment)
            for side in range(1, len(moments)):
                assert moments[side] == pytest.approx(moments[0], abs=tolerance), (
                    f"object {int(object_id)} of dimension {mdim}: elements "
                    f"{int(elements[0])} and {int(elements[side])} project "
                    f"{moments[side]} and {moments[0]}"
                )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("cells", [1, 2])
@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_and_kform_order_holds_the_invariants(
    ndim: int, cells: int, family: BasisType
) -> None:
    """No family and no k-form order may size the map wrongly or crash on it."""
    mesh = grid_mesh(ndim, cells)

    for kform_order in range(ndim + 1):
        map_of(mesh, uniform_orders(mesh, 2), kform_order, family)


@pytest.mark.parametrize(
    "orders",
    [
        (4, 2),
        (2, 4),
        (5, 2),
        (2, 5),
        (6, 3),
        (3, 6),
    ],
)
def test_anisotropic_element_keeps_every_node_on_one_global(
    orders: tuple[int, ...],
) -> None:
    """One element's axes disagreeing must not desynchronise a shared neighbour."""
    cells = 3
    mesh = grid_mesh(2, cells)
    pattern = uniform_orders(mesh, 2)

    # Positions 0 and 8 are corners, position 4 is the middle of the mesh.
    for position in range(mesh.element_count):
        mixed = list(pattern)
        mixed[position] = orders
        dof_map = map_of(mesh, mixed)
        assert_windowed_projections_agree(
            mesh, specs_in(mixed, 0, BasisType.LAGRANGE_GAUSS_LOBATTO), dof_map
        )


@pytest.mark.parametrize(
    "orders",
    [(4, 2, 2), (2, 4, 2), (2, 2, 4), (5, 2, 3), (3, 5, 2), (2, 2, 5)],
)
def test_anisotropic_element_in_three_dimensions(orders: tuple[int, ...]) -> None:
    """Anisotropy inside one element holds in three dimensions at every position."""
    mesh = grid_mesh(3, 2)

    for position in range(mesh.element_count):
        for kform_order in range(mesh.ndim + 1):
            mixed = uniform_orders(mesh, 2)
            mixed[position] = orders
            map_of(mesh, mixed, kform_order)


def test_anisotropic_element_in_the_middle_of_a_three_dimensional_mesh() -> None:
    """The one element of a 3x3x3 grid touching no outer face still shares correctly."""
    cells = 3
    mesh = grid_mesh(3, cells)
    mixed = uniform_orders(mesh, 2)
    mixed[13] = (4, 2, 3)

    dof_map = map_of(mesh, mixed)

    assert_windowed_projections_agree(
        mesh, specs_in(mixed, 0, BasisType.LAGRANGE_GAUSS_LOBATTO), dof_map
    )


def test_distinct_order_per_element() -> None:
    """Every element carrying an order of its own still has to give a valid map."""
    two_d = grid_mesh(2, 2)
    for kform_order in range(3):
        orders = [(2 + element, 5 - element) for element in range(4)]
        map_of(two_d, orders, kform_order)

    three_d = grid_mesh(3, 2)
    for kform_order in range(4):
        # Distinct orders per element, so any two neighbors disagree on a shared axis.
        orders = [
            (2 + element, 2, 1 + three_d.element_count - element)
            for element in range(three_d.element_count)
        ]
        map_of(three_d, orders, kform_order)


def test_checkerboard_orders() -> None:
    """Both orders on every interior object is what forces the per-axis minimum."""
    for low, high in ((2, 5), (3, 6)):
        two_d = grid_mesh(2, 3)
        for kform_order in range(3):
            orders = []
            for element in range(two_d.element_count):
                parity = (element % 3) + (element // 3)
                order = low if parity % 2 == 0 else high
                orders.append((order, order))
            map_of(two_d, orders, kform_order)

        three_d = grid_mesh(3, 2)
        for kform_order in range(4):
            orders = []
            for element in range(three_d.element_count):
                parity = (element % 2) + (element // 2) % 2 + (element // 4) % 2
                order = low if parity % 2 == 0 else high
                orders.append((order, order, order))
            map_of(three_d, orders, kform_order)


def test_rich_pair_next_to_poor() -> None:
    """A poor element seeing a rich one on both of its sides has to reach both blocks."""
    two_d = grid_mesh(2, 2)
    for kform_order in range(3):
        for pair in ((0, 1), (1, 3), (0, 3)):
            orders = uniform_orders(two_d, 2)
            orders[pair[0]] = (5, 4)
            orders[pair[1]] = (5, 4)
            map_of(two_d, orders, kform_order)

    three_d = grid_mesh(3, 2)
    for kform_order in range(4):
        # Elements 0 and 1 share a face, and element 6 touches both of them.
        orders = uniform_orders(three_d, 2)
        orders[0] = (4, 3, 2)
        orders[1] = (4, 3, 2)
        map_of(three_d, orders, kform_order)


@pytest.mark.parametrize("family", FAMILIES)
def test_extreme_order_ratios_drive_the_projection_path(family: BasisType) -> None:
    """A wide order gap still gives a valid map: the object takes the poorer order."""
    for ndim in (1, 2, 3):
        mesh = grid_mesh(ndim, 2)
        for kform_order in range(ndim + 1):
            for poor, rich in ((2, 6), (6, 2)):
                orders = uniform_orders(mesh, poor)
                orders[0] = (rich,) * ndim
                map_of(mesh, orders, kform_order, family)


def test_single_element_numbers_exactly_its_own_degrees_of_freedom() -> None:
    """With nothing shared, the global space is the element's own space."""
    for ndim in (1, 2, 3):
        mesh = grid_mesh(ndim, 1)
        for kform_order in range(ndim + 1):
            dof_map = map_of(mesh, uniform_orders(mesh, 3), kform_order)
            assert dof_map.global_dof_count == dof_map.element_dof_count
            # A window is weaker than the element, so a DoF may mix globals but owns one.
            assert np.all(np.diff(dof_map.entry_offsets) >= 1)


@pytest.mark.parametrize("family", FAMILIES)
def test_elements_meeting_in_one_point(family: BasisType) -> None:
    """Elements that share nothing but a corner still have to agree there."""
    for ndim in (1, 2, 3):
        count = 2 if ndim == 1 else 2 ** (ndim - 1)
        mesh = corner_fan_mesh(ndim, count)
        for kform_order in range(ndim + 1):
            map_of(mesh, [(2,) * ndim] * count, kform_order, family)


def test_relabelling_the_elements_keeps_the_size_of_the_map() -> None:
    """The size of the map may not depend on how the mesh numbers its elements."""
    mesh = grid_mesh(3, 2)
    reference_orders = uniform_orders(mesh, 2)
    reference_orders[0] = (4, 2, 3)
    reference = map_of(mesh, reference_orders, 1)

    permutations = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [7, 4, 1, 6, 3, 0, 5, 2],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [5, 7, 4, 6, 1, 3, 0, 2],
    ]
    for permutation in permutations:
        relabelled = grid_mesh_relabelled(3, 2, permutation)
        dof_map = map_of(relabelled, [reference_orders[old] for old in permutation], 1)
        assert dof_map.global_dof_count == reference.global_dof_count
        assert_same_entry_budget(dof_map, reference)
        assert dof_map.element_dof_count == reference.element_dof_count


def test_which_element_is_rich_does_not_change_the_size_of_the_map() -> None:
    """Only which objects the rich element meets matters, not its element ID."""
    mesh = grid_mesh(3, 2)
    counts = []
    for position in range(mesh.element_count):
        orders = uniform_orders(mesh, 2)
        orders[position] = (4, 2, 3)
        counts.append(map_of(mesh, orders, 1).global_dof_count)
    assert len(set(counts)) == 1


def test_exchanging_the_axes_exchanges_the_roles_of_the_orders() -> None:
    """Exchanging the axes and the orders together leaves the map the same size."""
    two_d = grid_mesh(2, 3)
    orders = uniform_orders(two_d, 2)
    orders[0] = (5, 2)
    reference = map_of(two_d, orders)
    swapped = grid_mesh_axes_swapped(2, 3, (1, 0))
    other = map_of(swapped, [axes[::-1] for axes in orders])

    assert other.global_dof_count == reference.global_dof_count
    assert_same_entry_budget(other, reference)

    three_d = grid_mesh(3, 2)
    triad = uniform_orders(three_d, 2)
    triad[0] = (3, 5, 2)
    reference = map_of(three_d, triad, 2)
    rotated = grid_mesh_axes_swapped(3, 2, (1, 2, 0))
    other = map_of(rotated, [axes[1:] + axes[:1] for axes in triad], 2)

    assert other.global_dof_count == reference.global_dof_count
    assert_same_entry_budget(other, reference)


def test_mirroring_an_element_keeps_the_size_of_the_map() -> None:
    """An element's orientation sign must not change the size of the map."""
    mesh = grid_mesh(3, 2)
    orders = uniform_orders(mesh, 2)
    orders[0] = (4, 2, 3)
    reference = map_of(mesh, orders, 1)

    for flipped in (
        {0: (0,)},
        {0: (2,)},
        {0: (0, 1, 2)},
        {3: (0,)},
        {0: (0, 2)},
        {5: (1,)},
    ):
        mirrored = grid_mesh_mirrored(3, 2, flipped)
        dof_map = map_of(mirrored, orders, 1)
        assert dof_map.global_dof_count == reference.global_dof_count
        assert_same_entry_budget(dof_map, reference)


def test_projected_dof_owns_a_weighted_combination() -> None:
    """A degree of freedom above the common order owns a weighted combination."""
    mesh = grid_mesh(2, 2)
    orders = uniform_orders(mesh, 3)
    orders[0] = (3, 5)
    dof_map = map_of(mesh, orders)

    assert np.any(np.diff(dof_map.entry_offsets) > 1)
    assert dof_map.entry_count > dof_map.element_dof_count
    # A genuine combination is not a relabelling of a signed one.
    assert np.any(np.abs(dof_map.entry_value) != pytest.approx(1.0))


@pytest.mark.parametrize("family", FAMILIES)
def test_window_matching_order_projects_correctly(family: BasisType) -> None:
    """Equal-order elements of any family transfer onto a consistent common space."""
    for ndim in (1, 2, 3):
        mesh = grid_mesh(ndim, 2)
        dof_map = map_of(mesh, uniform_orders(mesh, 3), 0, family)
        assert_windowed_projections_agree(
            mesh, specs_in(uniform_orders(mesh, 3), 0, family), dof_map
        )


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("cells", [1, 2])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_kform_globals_follow_the_window_formula(
    ndim: int, cells: int, order: int
) -> None:
    """Every k-form order counts its objects' windows exactly.

    An object of dimension ``d`` carries one Legendre window of size
    ``order`` on each covector axis of the form and ``max(order - 1, 0)``
    on every inactive axis, times ``C(d, k)`` components; the globals are
    the windows of all grid objects of dimension ``k`` and up.
    """
    mesh = grid_mesh(ndim, cells)
    for kform_order in range(ndim + 1):
        dof_map = map_of(mesh, uniform_orders(mesh, order), kform_order)
        expected = 0
        for d in range(kform_order, ndim + 1):
            objects = comb(ndim, d) * cells**d * (cells + 1) ** (ndim - d)
            window = 1
            for axis in range(d):
                window *= order if axis < kform_order else max(order - 1, 0)
            expected += objects * comb(d, kform_order) * window
        assert dof_map.global_dof_count == expected


@pytest.mark.parametrize(
    "family", [BasisType.LAGRANGE_UNIFORM, BasisType.LAGRANGE_GAUSS_LOBATTO]
)
@pytest.mark.parametrize("kform_order", [0, 1])
def test_first_order_nodal_families_transfer_a_signed_identity(
    family: BasisType,
    kform_order: int,
) -> None:
    """Order-one Lagrange nodes sit on the objects, so the transfer is +-1."""
    for ndim in (1, 2, 3):
        mesh = grid_mesh(ndim, 2)
        dof_map = map_of(mesh, uniform_orders(mesh, 1), kform_order, family)
        assert dof_map.entry_count == dof_map.element_dof_count
        assert np.all(np.diff(dof_map.entry_offsets) == 1)
        assert np.all(np.abs(dof_map.entry_value) == 1.0)


def test_a_top_form_has_no_shared_objects() -> None:
    """A form of full degree carries a covector on every axis, so shares nothing."""
    for ndim in (1, 2, 3):
        mesh = grid_mesh(ndim, 2)
        orders = uniform_orders(mesh, 2)
        orders[0] = (4, 2, 3)[:ndim]
        dof_map = map_of(mesh, orders, ndim)
        assert dof_map.global_dof_count == dof_map.element_dof_count
        assert int(dof_map.element_interior_offsets[0]) == 0
        assert np.all(np.diff(dof_map.entry_offsets) == 1)


@pytest.mark.parametrize(
    ("ndim", "cells", "orders"),
    [
        (1, 2, (2,)),
        (1, 3, (5,)),
        (2, 2, (2, 2)),
        (2, 2, (4, 2)),
        (2, 2, (2, 3)),
        (2, 3, (5, 2)),
        (3, 2, (2, 2, 2)),
        (3, 2, (4, 2, 2)),
        (3, 2, (2, 3, 3)),
        (3, 2, (5, 2, 3)),
        (3, 3, (4, 2, 2)),
    ],
)
def test_uniform_anisotropic_grid_counts_the_lattice(
    ndim: int, cells: int, orders: tuple[int, ...]
) -> None:
    """A scalar field on a uniform grid has one unknown per lattice point."""
    mesh = grid_mesh(ndim, cells)
    expected = 1
    for order in orders:
        expected *= cells * order + 1

    dof_map = map_of(mesh, [orders] * mesh.element_count)

    assert dof_map.global_dof_count == expected
    assert np.all(np.diff(dof_map.entry_offsets) >= 1)


def test_the_map_does_not_depend_on_how_many_times_it_is_built() -> None:
    """Two builds of the same request have to be identical, not merely equal in size."""
    mesh = grid_mesh(2, 3)
    orders = uniform_orders(mesh, 2)
    orders[4] = (4, 2)

    first = map_of(mesh, orders)
    second = map_of(mesh, orders)

    assert np.array_equal(first.entry_offsets, second.entry_offsets)
    assert np.array_equal(first.entry_index, second.entry_index)
    assert np.array_equal(first.entry_value, second.entry_value)


def test_mixed_families_in_one_mesh_are_accepted() -> None:
    """Every family pairs against the same Legendre windows, so they mix freely."""
    mesh = grid_mesh(2, 2)
    specs = specs_in(uniform_orders(mesh, 2))
    specs[0] = KFormSpecs(0, space((2, 2), BasisType.LEGENDRE))
    specs[3] = KFormSpecs(0, space((2, 2), BasisType.BERNSTEIN))

    dof_map = mesh.compute_kform_direct_dof_map(specs)
    assert_conforming(dof_map, mesh.element_count)
    assert np.all(np.isfinite(dof_map.entry_value))
    assert_windowed_projections_agree(mesh, specs, dof_map)


def test_a_zero_basis_order_is_rejected() -> None:
    """An axis without functions cannot carry a degree of freedom."""
    mesh = grid_mesh(2, 2)
    orders = uniform_orders(mesh, 2)
    orders[2] = (0, 2)

    with pytest.raises(ValueError, match="needs a positive basis order"):
        mesh.compute_kform_direct_dof_map(specs_in(orders))
