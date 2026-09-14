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
    build_continuity_rows_reference,
    make_element_maps,
    make_mesh,
    packed_to_dense,
    solve_direct_laplace,
)


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
    spaces = mesh.kform_boundary_spaces(element_specs)

    assert matrix.shape[0] == expected_rows
    assert np.linalg.matrix_rank(matrix) == expected_rows
    assert row_offsets.dtype == np.uintp
    assert element_ids.dtype == np.uint64
    assert components.dtype == np.uint32
    assert local_dofs.dtype == np.uintp
    assert coefficients.dtype == np.double
    assert row_offsets[-1] == element_ids.size
    assert element_ids.size == components.size == local_dofs.size == coefficients.size

    stage_rows: list[int] = []
    for mdim in range(ndim - 1, -1, -1):
        count = 0
        for _, object_id, object_elements, _ in mesh.iterate_shared(mdim):
            for test_spec in spaces[mdim][int(object_id)]:
                if test_spec is None:
                    continue
                count += (object_elements.size - 1) * int(
                    np.sum(test_spec.component_dof_counts)
                )
        stage_rows.append(count)

    if ndim == 2 and order == 1:
        assert stage_rows == [0, 7]
    elif ndim == 2 and order == 2:
        assert stage_rows == [4, 7]
    elif ndim == 3 and order == 1:
        assert stage_rows == [0, 0, 37]
    else:
        assert stage_rows[0] == 12
        assert stage_rows[-1] == 37
        assert sum(stage_rows) == expected_rows


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
        spaces = mesh.kform_boundary_spaces(element_specs)
        for mdim in range(1, ndim):
            assert all(
                all(s is None for s in spaces[mdim][int(object_id)])
                for _, object_id, _, _ in mesh.iterate_shared(mdim)
            )
        expected_offsets = 38 if ndim == 3 else 8
        assert packed[0].shape[0] == expected_offsets


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


def _manual_test_orders(
    mesh: Mesh,
    element_specs: list[KFormSpecs],
    form_order: int,
) -> list[list[list[tuple[int, ...] | None]]]:
    """Reimplement the documented derivation order rule as a manual oracle.

    Returns per object dimension, object ID, and canonical component either
    the tuple of derived per-axis orders or ``None`` when the component has
    no rows.
    """
    ndim = mesh.ndim
    incidents_by_dimension: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    for mdim in range(ndim):
        incidents: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for iterator in (mesh.iterate_shared(mdim), mesh.iterate_boundary(mdim)):
            incidents.update(
                {
                    int(object_id): (element_ids, orientations)
                    for _, object_id, element_ids, orientations in iterator
                }
            )
        incidents_by_dimension.append(incidents)

    result: list[list[list[tuple[int, ...] | None]]] = []
    for mdim in range(ndim):
        objects: list[list[tuple[int, ...] | None]] = []
        object_count = (
            mesh.point_count if mdim == 0 else int(mesh.collections[mdim - 1].shape[0])
        )
        components = list(combinations(range(mdim), form_order))
        for object_id in range(object_count):
            component_orders: list[tuple[int, ...] | None] = [None] * len(components)
            incident = incidents_by_dimension[mdim].get(object_id)
            if incident is not None and mdim >= form_order:
                element_ids, orientations = incident
                fixed_count = ndim - mdim
                for index, active_axes in enumerate(components):
                    orders: list[int] = []
                    present = True
                    for canonical_axis in range(mdim):
                        mapped = [
                            abs(int(orientations[row][fixed_count + canonical_axis])) - 1
                            for row in range(len(element_ids))
                        ]
                        lowest = min(
                            element_specs[int(eid)].base_space.orders[axis]
                            for eid, axis in zip(element_ids, mapped, strict=True)
                        )
                        reduced = lowest - (0 if canonical_axis in active_axes else 2)
                        present = present and reduced >= 0
                        orders.append(reduced)
                    if present:
                        component_orders[index] = tuple(orders)
            objects.append(component_orders)
        result.append(objects)
    return result


@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("form_order", (0, 1, 2))
def test_auto_matches_manual_derivation(ndim: int, form_order: int) -> None:
    """Derived boundary spaces reproduce the documented min/minus-two rule."""
    if form_order > ndim:
        return
    mesh = make_mesh(ndim)
    maps = make_element_maps(ndim, form_order + 4)
    base_space = FunctionSpace(
        *(
            BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, form_order + 1)
            for _ in range(ndim)
        )
    )
    element_specs = [KFormSpecs(form_order, base_space) for _ in maps]
    spaces = mesh.kform_boundary_spaces(element_specs)
    manual = _manual_test_orders(mesh, element_specs, form_order)

    for mdim in range(ndim):
        for object_id in range(len(spaces[mdim])):
            expected = manual[mdim][object_id]
            actual = spaces[mdim][object_id]
            assert len(actual) == len(expected)
            for component, order_tuple in enumerate(expected):
                test_spec = actual[component]
                if order_tuple is None:
                    assert test_spec is None
                    continue
                assert test_spec is not None
                assert tuple(test_spec.base_space.orders) == order_tuple


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


def test_reference_builder_matches_c_rows() -> None:
    """The readable Python pairing loop produces the C-backed rows exactly."""
    for ndim in (2, 3):
        mesh = make_mesh(ndim)
        maps = make_element_maps(ndim, 5)
        base_space = FunctionSpace(
            *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 2) for _ in range(ndim))
        )
        element_specs = [KFormSpecs(0, base_space) for _ in maps]
        reference = build_continuity_rows_reference(mesh, maps, element_specs)
        produced = build_continuity_rows(mesh, maps, element_specs)
        assert np.array_equal(np.asarray(reference[0]), produced[0])
        for reference_array, produced_array in zip(reference[1:], produced[1:]):
            assert np.array_equal(np.asarray(reference_array), produced_array)


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
    matrix = packed_to_dense(
        mesh.compute_kform_continuity_constraints(element_specs, maps),
        element_specs,
    )
    assert np.linalg.matrix_rank(matrix) == matrix.shape[0]


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
