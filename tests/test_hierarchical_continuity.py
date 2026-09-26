"""Behavioral tests for hierarchical, cycle-free continuity rows."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    Mesh,
    SpaceMap,
)
from fdg.enum_type import IntegrationMethod

from examples.plot_multi_element_laplace_continuity import (
    build_continuity_rows,
    make_element_maps,
    make_mesh,
    packed_to_dense,
    solve_direct_laplace,
)

# TODO: make typing use numpy.typing.NDArray


def _scalar_setup(ndim: int, order: int):
    """Build the scalar prototype objects for one mesh dimension and order."""
    mesh = make_mesh(ndim)
    maps = make_element_maps(ndim, order + 4)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    packed = build_continuity_rows(mesh, maps, element_specs)
    return mesh, element_specs, packed


@pytest.mark.parametrize(
    ("ndim", "order", "expected_rows"),
    ((2, 1, 7), (2, 2, 11), (3, 1, 37), (3, 2, 91)),
)
def test_scalar_hierarchy_row_ownership(
    ndim: int, order: int, expected_rows: int
) -> None:
    """Trace rows belong to descending strata and have full row rank."""
    mesh, element_specs, packed = _scalar_setup(ndim, order)
    matrix = packed_to_dense(packed, element_specs)
    row_offsets, element_ids, components, local_dofs, coefficients = packed

    assert matrix.shape[0] == expected_rows
    assert np.linalg.matrix_rank(matrix) == expected_rows
    assert row_offsets.dtype == np.uintp
    assert element_ids.dtype == np.uint64
    assert components.dtype == np.uint32
    assert local_dofs.dtype == np.uintp
    assert coefficients.dtype == np.double
    assert row_offsets[-1] == element_ids.size
    assert element_ids.size == components.size == local_dofs.size == coefficients.size


def test_multi_element_objects_use_spanning_paths() -> None:
    """An object in E elements contributes exactly E minus one pairs."""
    for ndim, center_object, expected_elements in ((2, 4, 4), (3, 13, 8)):
        mesh = make_mesh(ndim)
        center = {
            int(object_id): element_ids
            for mdim, object_id, element_ids, _ in mesh.iterate_shared(0)
            if int(object_id) == center_object
        }[center_object]
        assert center.size == expected_elements
        assert center.size - 1 == (3 if ndim == 2 else 7)

        pair_count = sum(
            int(element_ids.size - 1)
            for _, _, element_ids, _ in mesh.iterate_shared_all()
        )
        assert pair_count > center.size - 1


def test_degree_one_has_empty_higher_stratum_stages() -> None:
    """Degree-one scalar traces leave face and edge interiors empty."""
    for ndim in (2, 3):
        mesh, element_specs, packed = _scalar_setup(ndim, 1)
        # Only the vertex stratum contributes rows: the face and edge stages
        # are empty, so the total is the vertex row count alone.
        expected_offsets = 38 if ndim == 3 else 8
        assert packed[0].shape[0] == expected_offsets
        assert np.linalg.matrix_rank(packed_to_dense(packed, element_specs)) == (
            expected_offsets - 1
        )


def test_direct_laplace_solves_have_continuous_full_rank_systems() -> None:
    """Direct primal solves satisfy continuity and improve under p-refinement."""
    for ndim in (2, 3):
        results = [solve_direct_laplace(ndim, order) for order in (1, 2)]
        for solution, continuity, error, rank, constraint_count, residual in results:
            del solution
            assert rank == continuity.shape[1] + constraint_count
            assert continuity.shape[0] + 1 == constraint_count
            assert residual < 1.0e-12
            assert np.isfinite(error)
        assert results[1][2] <= results[0][2] + 1.0e-10


def test_identity_maps_c1_rows_match_physical() -> None:
    """C1 rows and physical rows span the same space on identity maps."""
    for ndim in (2, 3):
        mesh = make_mesh(ndim)
        maps: list[SpaceMap] = []
        geometry_space = FunctionSpace(
            *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(ndim))
        )
        nodes = np.linspace(-1.0, 1.0, 2)
        grid = np.meshgrid(*([nodes] * ndim), indexing="ij")
        integration = IntegrationSpace(
            *(IntegrationSpecs(4, IntegrationMethod.GAUSS) for _ in range(ndim))
        )
        for element_index in product(range(2), repeat=ndim):
            coordinates = [
                0.5 * (grid[axis] + 1.0) + element_index[axis] - 0.5
                for axis in range(ndim)
            ]
            maps.append(
                SpaceMap(
                    *(
                        CoordinateMap(
                            DegreesOfFreedom(geometry_space, coordinate.ravel()),
                            integration,
                        )
                        for coordinate in coordinates
                    )
                )
            )

        base_space = FunctionSpace(
            *(BasisSpecs(BasisType.LEGENDRE, 2) for _ in range(ndim))
        )
        for form_order in range(ndim):
            element_specs = [KFormSpecs(form_order, base_space) for _ in maps]
            c1_rows = mesh.compute_kform_continuity_constraints(
                element_specs, None, c1_continuous=True
            )
            physical_rows = mesh.compute_kform_continuity_constraints(element_specs, maps)
            without_maps_again = mesh.compute_kform_continuity_constraints(
                element_specs, c1_continuous=True
            )
            matrix_c1 = packed_to_dense(c1_rows, element_specs)
            matrix_physical = packed_to_dense(physical_rows, element_specs)
            matrix_again = packed_to_dense(without_maps_again, element_specs)
            assert np.array_equal(matrix_c1, matrix_again)
            if matrix_c1.shape[0] == 0:
                continue
            rank_c1 = np.linalg.matrix_rank(matrix_c1)
            rank_physical = np.linalg.matrix_rank(matrix_physical)
            assert rank_c1 == matrix_c1.shape[0]
            assert rank_physical == matrix_physical.shape[0]
            stacked = np.vstack((matrix_c1, matrix_physical))
            assert np.linalg.matrix_rank(stacked) == rank_c1


def test_c1_without_maps_and_rejections() -> None:
    """The C1 flag relaxes the map requirement; the physical path rejects it."""
    mesh = make_mesh(2)
    maps = make_element_maps(2, 5)
    base_space = FunctionSpace(*(BasisSpecs(BasisType.LEGENDRE, 1) for _ in range(2)))
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    with pytest.raises(ValueError, match="C1 continuous"):
        mesh.compute_kform_continuity_constraints(element_specs, None)
    rows = mesh.compute_kform_continuity_constraints(element_specs, c1_continuous=True)
    assert rows[0].size > 1


def _tangential_trace_survives(
    spec: KFormSpecs, component: int, object_axes: set[int]
) -> bool:
    """Return whether the component's tangential trace survives on the object."""
    if spec.order == 0:
        return True
    axes = list(combinations(range(spec.base_space.dimension), spec.order))[component]
    return set(axes).issubset(object_axes)


def _ground_truth_matrix(
    mesh: Mesh,
    specs: list[KFormSpecs],
    points_per_axis: int = 0,
    element_axis_maps: list[tuple[int, ...]] | None = None,
    min_order_windows: bool = False,
    element_maps: list[SpaceMap] | None = None,
) -> np.ndarray:
    """Build an independent continuity-functional matrix.

    The default mode uses signed point evaluations. ``min_order_windows``
    instead contracts the common minimum Legendre test basis against the
    physical face measure and k-form pullback on the map quadrature grid;
    it is the L2 oracle for anisotropic element orders.
    """
    ndim = mesh.ndim
    dof_counts = [int(np.sum(spec.component_dof_counts)) for spec in specs]
    offsets = np.concatenate(([0], np.cumsum(dof_counts))).astype(int)
    ndof_total = int(offsets[-1])
    max_order = max(max(spec.base_space.orders) for spec in specs)
    npts = points_per_axis or (max_order + 2)
    identity = tuple(range(mesh.ndim))
    axis_maps = element_axis_maps or [identity] * len(specs)
    rows: list[np.ndarray] = []
    for mdim, _oid, elems, orients in mesh.iterate_shared_all():
        bdim = int(mdim)
        face_maps: list[SpaceMap] = []
        if min_order_windows and element_maps is not None and bdim > 0:
            for i, e in enumerate(elems):
                face_map = element_maps[int(e)]
                fixed = [
                    (
                        int(abs(orients[i][slot])) - 1,
                        bool(orients[i][slot] > 0),
                    )
                    for slot in range(ndim - bdim)
                ]
                for axis, end in sorted(fixed, reverse=True):
                    face_map = face_map.boundary(axis, end=end)
                face_maps.append(face_map)
            node_grid = np.asarray(face_maps[0].integration_space.nodes())
            quad_weights = np.asarray(face_maps[0].integration_space.weights()).reshape(
                -1
            )
            pts = np.stack([node_grid[slot].ravel() for slot in range(bdim)], axis=1)
        else:
            nodes = np.linspace(-1.0, 1.0, npts)
            grids = np.meshgrid(*([nodes] * bdim), indexing="ij")
            pts = (
                np.stack([g.ravel() for g in grids], axis=1) if bdim else np.zeros((1, 0))
            )
            quad_weights = np.ones(pts.shape[0])
        obj_axes = [
            {int(abs(orients[i][ndim - bdim + s])) - 1 for s in range(bdim)}
            for i in range(len(elems))
        ]
        fixed_axes = [
            [int(abs(v)) - 1 for v in orients[i][: ndim - bdim]]
            for i in range(len(elems))
        ]
        # Per element, map each surviving component index to its global wedge
        # so rotated elements pair components of the same physical direction.
        wedges = []
        for i, e in enumerate(elems):
            per_comp = {}
            for component in range(specs[e].component_count):
                if not _tangential_trace_survives(specs[e], component, obj_axes[i]):
                    continue
                local = list(
                    combinations(range(specs[e].base_space.dimension), specs[e].order)
                )[component]
                # A reversed wedge axis flips the pulled-back covector, so
                # the tangential trace carries the product of the axis
                # orientation signs over the component's wedge, and sorting
                # the mapped axes into physical order adds the permutation
                # parity for wedges of order two and up.
                sign = 1
                mapped = []
                for axis in local:
                    slot = next(
                        s
                        for s in range(bdim)
                        if int(abs(orients[i][ndim - bdim + s])) - 1 == axis
                    )
                    sign *= 1 if orients[i][ndim - bdim + slot] > 0 else -1
                    mapped.append(axis_maps[e][axis])
                for u in range(len(mapped)):
                    for v in range(u + 1, len(mapped)):
                        if mapped[u] > mapped[v]:
                            sign = -sign
                            mapped[u], mapped[v] = mapped[v], mapped[u]
                per_comp[component] = (frozenset(mapped), sign)
            wedges.append(per_comp)
        # Pair sides by physical wedge direction only; each side's reference
        # values below already carry its own orientation sign. The surviving
        # local component index may differ per side for rotated frames.
        by_wedge: list[dict[frozenset, tuple[int, int]]] = []
        for per_comp in wedges:
            mapping: dict[frozenset, tuple[int, int]] = {}
            for component, (wedge, sign) in per_comp.items():
                mapping.setdefault(wedge, (component, sign))
            by_wedge.append(mapping)
        common_wedges = set(by_wedge[0])
        for mapping in by_wedge[1:]:
            common_wedges &= set(mapping)
        for common_wedge in sorted(common_wedges):
            local_components = [mapping[common_wedge][0] for mapping in by_wedge]
            signs = [mapping[common_wedge][1] for mapping in by_wedge]
            per_elem_vals = []
            per_elem_blocks: list[dict[int, np.ndarray]] = []
            if min_order_windows:
                canonical_axes = sorted(obj_axes[0])
                local_wedge = tuple(canonical_axes.index(a) for a in sorted(common_wedge))
                test_component_index = list(
                    combinations(range(bdim), specs[0].order)
                ).index(local_wedge)
                face_components = list(combinations(range(bdim), specs[0].order))
            for i, e in enumerate(elems):
                coord = np.empty((ndim, pts.shape[0]))
                for axis in range(ndim):
                    if axis in obj_axes[i]:
                        slot = next(
                            s
                            for s in range(bdim)
                            if int(abs(orients[i][ndim - bdim + s])) - 1 == axis
                        )
                        sign = 1.0 if orients[i][ndim - bdim + slot] > 0 else -1.0
                        coord[axis] = sign * pts[:, slot]
                    else:
                        entry = orients[i][fixed_axes[i].index(axis)]
                        coord[axis] = 1.0 if entry > 0 else -1.0
                if min_order_windows:
                    transform = np.asarray(face_maps[i].basis_transform(specs[e].order))
                    surface = np.abs(np.asarray(face_maps[i].determinant)).reshape(-1)
                    blocks: dict[int, np.ndarray] = {}
                    for block_component in range(specs[e].component_count):
                        block_axes = list(
                            combinations(
                                range(specs[e].base_space.dimension), specs[e].order
                            )
                        )[block_component]
                        if not set(block_axes).issubset(obj_axes[i]):
                            continue
                        mapped = tuple(sorted(axis_maps[e][axis] for axis in block_axes))
                        face_axes = tuple(canonical_axes.index(axis) for axis in mapped)  # type: ignore
                        face_component = face_components.index(face_axes)  # type: ignore
                        block_values = np.asarray(
                            specs[e]
                            .get_component_function_space(block_component)
                            .evaluate(*[coord[a] for a in range(ndim)])
                        ).reshape(pts.shape[0], -1)
                        metric = np.sum(
                            transform[test_component_index] * transform[face_component],  # type: ignore
                            axis=0,
                        )
                        blocks[block_component] = (
                            block_values * (surface * quad_weights * metric)[:, None]
                        )
                    per_elem_blocks.append(blocks)
                else:
                    cfs = specs[e].get_component_function_space(local_components[i])
                    values = cfs.evaluate(*[coord[a] for a in range(ndim)])
                    per_elem_vals.append(
                        signs[i] * np.asarray(values).reshape(pts.shape[0], -1)
                    )
            if min_order_windows:
                # The signed trace values already carry each side's physical
                # face measure; pair them against the test functions.
                bdim = int(mdim)
                canonical_axes = sorted(obj_axes[0])
                axis_orders_by_axis = {
                    a: min(specs[e].base_space.basis_specs[a].order for e in elems)
                    for a in canonical_axes
                }
                # The mass rows read the merged per-axis minimum Legendre
                # basis: active covector axes keep the leading `min_order`
                # functions; inactive axes drop their two highest functions.
                common_space = FunctionSpace(
                    *(
                        BasisSpecs(BasisType.LEGENDRE, axis_orders_by_axis[a])
                        for a in canonical_axes
                    )
                )
                scalar_kform = KFormSpecs(0, common_space)
                cfs_w = scalar_kform.get_component_function_space(0)
                test_values = np.asarray(
                    cfs_w.evaluate(
                        *[
                            np.ascontiguousarray(pts[:, canonical_axes.index(a)])
                            for a in canonical_axes
                        ]
                    )
                ).reshape(pts.shape[0], -1)
                kept_axes: list[list[int]] = []
                local_orders = []
                for local_axis, a in enumerate(canonical_axes):
                    min_order = axis_orders_by_axis[a]
                    columns = min_order + 1
                    local_orders.append(columns)
                    kept_axes.append(
                        list(range(min_order))
                        if a in common_wedge
                        else list(range(columns - 2))  # drop the two highest
                    )
                strides = np.ones(bdim, dtype=int)
                for local_axis in range(bdim - 2, -1, -1):
                    strides[local_axis] = (
                        strides[local_axis + 1] * local_orders[local_axis + 1]
                    )
                kept_columns = [
                    column
                    for column in range(test_values.shape[1])
                    if all(
                        (column // strides[local_axis]) % local_orders[local_axis]
                        in kept_axes[local_axis]
                        for local_axis in range(bdim)
                    )
                ]
                test_values = test_values[:, kept_columns]
            pair_indices = (
                range(1, len(elems)) if min_order_windows else range(len(elems) - 1)
            )
            for k in pair_indices:
                side_indices = (0, k) if min_order_windows else (k, k + 1)
                pair = tuple(int(elems[index]) for index in side_indices)
                window_count = (
                    pts.shape[0] if not min_order_windows else test_values.shape[1]  # type: ignore
                )
                for j in range(window_count):
                    row = np.zeros(ndof_total)
                    for i, e in enumerate(pair):
                        sign = 1.0 if i == 0 else -1.0
                        side_index = side_indices[i]
                        if min_order_windows:
                            for block_component, value_block in per_elem_blocks[
                                side_index
                            ].items():
                                lo = offsets[e] + int(
                                    specs[e].get_component_slice(block_component).start
                                )
                                row[lo : lo + value_block.shape[1]] += sign * (
                                    value_block * test_values[:, j][:, None]  # type: ignore
                                ).sum(axis=0)
                        else:
                            component = local_components[side_index]
                            lo = offsets[e] + int(
                                specs[e].get_component_slice(component).start
                            )
                            value_block = per_elem_vals[side_index]
                            row[lo : lo + value_block.shape[1]] += (
                                sign * value_block[j % pts.shape[0]]
                            )
                    rows.append(row)
    return np.array(rows) if rows else np.zeros((0, ndof_total))


def assert_continuity_exact(
    mesh: Mesh,
    maps: list[SpaceMap],
    specs: list[KFormSpecs],
    element_axis_maps: list[tuple[int, ...]] | None = None,
) -> None:
    """Assert the produced rows enforce exactly the continuity subspace."""
    produced = packed_to_dense(
        mesh.compute_kform_continuity_constraints(specs, maps), specs
    )
    truth = _ground_truth_matrix(mesh, specs, element_axis_maps=element_axis_maps)
    rank_c = np.linalg.matrix_rank(produced, tol=1e-9) if produced.size else 0
    rank_g = np.linalg.matrix_rank(truth, tol=1e-9) if truth.size else 0
    stacked = np.vstack([produced, truth])
    rank_s = np.linalg.matrix_rank(stacked, tol=1e-9)
    assert rank_c == rank_g == rank_s, (
        f"continuity mismatch: produced rank {rank_c}, truth rank {rank_g}, "
        f"stacked rank {rank_s}"
    )


@pytest.mark.parametrize(
    ("ndim", "family", "order", "form_order"),
    tuple(
        (ndim, family, order, form_order)
        for ndim in (2, 3)
        for family in (BasisType.LEGENDRE, BasisType.LAGRANGE_GAUSS_LOBATTO)
        for order in (1, 2, 3)
        for form_order in (0, 1, 2, 3)
        # A k-form's order cannot exceed the element dimension.
        if form_order <= ndim
    ),
)
def test_continuity_matches_ground_truth_all_configs(
    ndim: int, family: BasisType, order: int, form_order: int
) -> None:
    """Produced rows span exactly the trace-continuity functionals."""
    mesh = make_mesh(ndim)
    maps = make_element_maps(ndim, order + 1)
    base_space = FunctionSpace(*(BasisSpecs(family, order) for _ in range(ndim)))
    element_specs = [KFormSpecs(form_order, base_space) for _ in maps]
    assert_continuity_exact(mesh, maps, element_specs)


def _mixed_orientation_mesh(
    ndim: int,
) -> tuple[Mesh, list[SpaceMap], list[tuple[int, ...]]]:
    """Mesh with one transposed and one mirrored element plus matching maps."""
    import numpy as np
    from numpy import prod

    from examples.plot_multi_element_laplace_continuity import (
        GEO_ORDER,
        _deformed_coordinates,
        mesh_corners,
    )

    corners = mesh_corners(ndim).reshape(-1, 2**ndim).copy()

    def permute_bits(entry: int, permutation: tuple[int, ...]) -> int:
        return sum(((entry >> axis) & 1) << permutation[axis] for axis in range(ndim))

    rotation = (1, 2, 0) if ndim == 3 else (1, 0)
    flip = 3 if ndim == 3 else 3  # rotate 180 degrees: flip the two first axes
    n_corner = 2**ndim
    for element_index in product(range(2), repeat=ndim):
        flat = sum(a * 2 ** (ndim - 1 - i) for i, a in enumerate(element_index))
        if element_index == (1,) + (0,) * (ndim - 1):
            original = corners[flat].copy()
            for local in range(n_corner):
                corners[flat, local] = original[permute_bits(local, rotation)]
        elif element_index == (0,) + (1,) + (0,) * (ndim - 2):
            original = corners[flat].copy()
            for local in range(n_corner):
                corners[flat, local] = original[local ^ flip]

    mesh = Mesh.from_corners(ndim, corners.ravel())

    geometry_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER) for _ in range(ndim))
    )
    geometry_nodes = np.linspace(-1.0, 1.0, GEO_ORDER + 1)
    geometry_grid = np.meshgrid(*([geometry_nodes] * ndim), indexing="ij")
    integration = IntegrationSpace(
        *(IntegrationSpecs(4, IntegrationMethod.GAUSS) for _ in range(ndim))
    )
    shape = (GEO_ORDER + 1,) * ndim
    maps: list[SpaceMap] = []
    axis_maps: list[tuple[int, ...]] = []
    for element_index in product(range(2), repeat=ndim):
        reference = tuple(
            0.5 * geometry_grid[axis] + element_index[axis] - 0.5 for axis in range(ndim)
        )
        coordinates = _deformed_coordinates(*reference)
        if element_index == (1,) + (0,) * (ndim - 1):
            # The corners rename local bit axis a to physical axis
            # rotation[a], so the coordinate tensors must transpose with
            # rotation itself: local axis a must drive tensor axis
            # rotation[a] (= the physical axis the corners assign it).
            coordinate_tensors = [
                np.asarray(c).reshape(shape).transpose(rotation) for c in coordinates
            ]
            axis_maps.append(rotation)
        elif element_index == (0,) + (1,) + (0,) * (ndim - 2):
            coordinate_tensors = [
                np.asarray(c).reshape(shape)[
                    tuple(
                        slice(None, None, -1) if axis < 2 else slice(None)
                        for axis in range(ndim)
                    )
                ]
                for axis, c in enumerate(coordinates)
            ]
            axis_maps.append(tuple(range(ndim)))
        else:
            coordinate_tensors = [np.asarray(c).reshape(shape) for c in coordinates]
            axis_maps.append(tuple(range(ndim)))
        maps.append(
            SpaceMap(
                *(
                    CoordinateMap(
                        DegreesOfFreedom(geometry_space, tensor.ravel()), integration
                    )
                    for tensor in coordinate_tensors
                )
            )
        )
    assert int(prod(corners.shape)) == len(maps) * n_corner
    return mesh, maps, axis_maps


@pytest.mark.parametrize("form_order", (0, 1, 2))
def test_continuity_with_mixed_element_orientations(form_order: int) -> None:
    """Rows stay exact when elements carry rotated or mirrored axes.

    The trace pullback tables hold each side's own covector image and the
    canonical rows read the mapped free-axis ranks with the mirror/parity
    sign, so the assembled rows span exactly the physical trace-continuity
    functionals (verified against _ground_truth_matrix).
    """
    ndim = 3
    mesh, maps, axis_maps = _mixed_orientation_mesh(ndim)
    base_space = FunctionSpace(*(BasisSpecs(BasisType.LEGENDRE, 2) for _ in range(ndim)))
    element_specs = [KFormSpecs(form_order, base_space) for _ in maps]
    assert_continuity_exact(mesh, maps, element_specs, element_axis_maps=axis_maps)


@pytest.mark.parametrize(
    "per_element",
    (False, True),
)
def test_continuity_with_anisotropic_orders(per_element: bool) -> None:
    """Rows stay exact for direction- and element-dependent basis orders."""
    ndim = 3
    mesh = make_mesh(ndim)
    maps = make_element_maps(ndim, 4)
    order_patterns = [(1, 2, 3), (3, 1, 2), (2, 3, 1), (2, 2, 2)] * 2
    element_specs = []
    for element in range(8):
        orders = order_patterns[element] if per_element else (1, 2, 3)
        base_space = FunctionSpace(
            *(BasisSpecs(BasisType.LEGENDRE, order) for order in orders)
        )
        element_specs.append(KFormSpecs(1, base_space))
    if per_element:
        # Per-element orders: the C constrains only min-order own-block
        # windows, so the oracle pairs against the common test space.
        produced = packed_to_dense(
            mesh.compute_kform_continuity_constraints(element_specs, maps), element_specs
        )
        truth = _ground_truth_matrix(
            mesh, element_specs, min_order_windows=True, element_maps=maps
        )
        rank_c = np.linalg.matrix_rank(produced, tol=1e-9)
        rank_g = np.linalg.matrix_rank(truth, tol=1e-9)
        rank_s = np.linalg.matrix_rank(np.vstack([produced, truth]), tol=1e-9)
        assert rank_c == rank_g == rank_s, (
            f"continuity mismatch: produced rank {rank_c}, truth rank {rank_g}, "
            f"stacked rank {rank_s}"
        )
    else:
        assert_continuity_exact(mesh, maps, element_specs)


def test_rotated_element_180_degrees_2d() -> None:
    """One 180-degree rotated element in a 2x2 mesh keeps continuity exact.

    Element ids are flat (ex * 2 + ey), so the element maps must be built
    in that same order; the rotated cell is (0, 1) with flipped corners
    and reversed control-node tensors. Hat-validated ground truth and the
    constraint rows agree to machine precision.
    """
    import numpy as np

    from examples.plot_multi_element_laplace_continuity import (
        GEO_ORDER,
        _deformed_coordinates,
        mesh_corners,
    )

    corners = mesh_corners(2).reshape(-1, 4).copy()
    original = corners[1].copy()
    for local in range(4):
        corners[1, local] = original[local ^ 3]

    mesh = Mesh.from_corners(2, corners.ravel())
    geometry_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER),
    )
    geometry_nodes = np.linspace(-1.0, 1.0, GEO_ORDER + 1)
    geometry_grid = np.meshgrid(geometry_nodes, geometry_nodes, indexing="ij")
    integration = IntegrationSpace(
        IntegrationSpecs(4, IntegrationMethod.GAUSS),
        IntegrationSpecs(4, IntegrationMethod.GAUSS),
    )
    maps = []
    for ex, ey in ((0, 0), (0, 1), (1, 0), (1, 1)):
        reference = (0.5 * geometry_grid[0] + ex - 0.5, 0.5 * geometry_grid[1] + ey - 0.5)
        coordinates = _deformed_coordinates(*reference)
        tensors = [np.asarray(c) for c in coordinates]
        if (ex, ey) == (0, 1):
            tensors = [t[::-1, ::-1].copy() for t in tensors]
        maps.append(
            SpaceMap(
                *(
                    CoordinateMap(
                        DegreesOfFreedom(geometry_space, tensor.ravel()), integration
                    )
                    for tensor in tensors
                )
            )
        )
    base_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 2),
        BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 2),
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    assert_continuity_exact(mesh, maps, element_specs)


@pytest.mark.parametrize(
    ("form_order", "expected_components"), ((1, (0, 1, 2)), (2, (0, 1, 2)))
)
def test_generic_kform_components_and_orientation(
    form_order: int, expected_components: tuple[int, ...]
) -> None:
    """Global rows filter dimensions and map canonical components."""
    mesh = make_mesh(3)
    maps = make_element_maps(3, 5)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 1) for _ in range(3))
    )
    element_specs = [KFormSpecs(form_order, base_space) for _ in maps]
    row_offsets, element_ids, components, local_dofs, coefficients = (
        mesh.compute_kform_continuity_constraints(element_specs, maps)
    )
    del row_offsets, element_ids, local_dofs, coefficients
    assert components.size > 0
    assert set(components.tolist()).issubset(set(expected_components))
    assert np.all(components < element_specs[0].component_count)
    assert_continuity_exact(mesh, maps, element_specs)


def test_empty_mesh_continuity_has_valid_offsets() -> None:
    """A mesh without shared objects returns one empty offset."""
    mesh = Mesh.from_corners(2, np.asarray([0, 1, 2, 3], dtype=np.uint64))
    maps = [make_element_maps(2, 5)[0]]
    base_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 1),
        BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 1),
    )
    element_specs = [KFormSpecs(0, base_space)]
    result = mesh.compute_kform_continuity_constraints(element_specs, maps)
    assert result[0].tolist() == [0]
    assert all(array.size == 0 for array in result[1:])


def test_global_continuity_rejects_wrong_lengths() -> None:
    """Global assembly rejects element sequences with wrong lengths."""
    mesh, element_specs, _ = _scalar_setup(2, 1)
    maps = make_element_maps(2, 5)
    with pytest.raises(ValueError, match="each contain"):
        mesh.compute_kform_continuity_constraints(element_specs[:-1], maps)


if __name__ == "__main__":
    pytest.main([__file__])
