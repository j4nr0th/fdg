"""Behavioral tests for hierarchical, cycle-free continuity rows."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from fdg import BasisSpecs, BasisType, FunctionSpace, KFormSpecs

from examples.plot_multi_element_laplace_continuity import (
    build_continuity_rows,
    make_element_maps,
    make_mesh,
    make_test_specs,
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
    test_specs = make_test_specs(mesh, element_specs, 0, BasisType.LEGENDRE)
    packed = build_continuity_rows(mesh, maps, element_specs, test_specs)
    return mesh, element_specs, test_specs, packed


@pytest.mark.parametrize(
    ("ndim", "order", "expected_rows"),
    ((2, 1, 7), (2, 2, 11), (3, 1, 37), (3, 2, 91)),
)
def test_scalar_hierarchy_row_ownership(
    ndim: int, order: int, expected_rows: int
) -> None:
    """Trace rows belong to descending strata and have full row rank."""
    mesh, element_specs, test_specs, packed = _scalar_setup(ndim, order)
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

    stage_rows: list[int] = []
    for mdim in range(ndim - 1, -1, -1):
        count = 0
        for _, object_id, object_elements, _ in mesh.iterate_shared(mdim):
            for test_spec in test_specs[mdim][int(object_id)]:
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
        mesh, _, test_specs, packed = _scalar_setup(ndim, 1)
        for mdim in range(1, ndim):
            assert all(
                not test_specs[mdim][int(object_id)]
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


@pytest.mark.parametrize("ndim,order", ((2, 1), (2, 2), (3, 1), (3, 2)))
def test_dirichlet_rows_use_one_owner_per_boundary_object(ndim: int, order: int) -> None:
    """Each hierarchical boundary object contributes rows from one element."""
    mesh = make_mesh(ndim)
    maps = make_element_maps(ndim, order + 4)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    test_specs = make_test_specs(mesh, element_specs, 0, BasisType.LEGENDRE)
    boundary_data = {
        int(object_id): lambda *coordinates: np.ones_like(coordinates[0])
        for _, object_id, _, _ in mesh.iterate_boundary(ndim - 1)
    }
    packed, boundary_rhs = mesh.compute_kform_global_constraints(
        element_specs, maps, test_specs, boundary_data
    )
    row_offsets, element_ids, _, _, _ = packed
    shared_rows = build_continuity_rows(mesh, maps, element_specs, test_specs)[0].size - 1
    row = shared_rows
    for mdim, object_id, element_ids_object, _ in mesh.iterate_boundary_all():
        object_rows = sum(
            int(np.sum(test_spec.component_dof_counts))
            for test_spec in test_specs[mdim][int(object_id)]
        )
        if object_rows == 0:
            continue
        entries = element_ids[int(row_offsets[row]) : int(row_offsets[row + object_rows])]
        assert np.unique(entries).tolist() == [int(element_ids_object[0])]
        row += object_rows
    assert row == boundary_rhs.size == row_offsets.size - 1


def test_direct_laplace_dirichlet_solves_propagate_owner_values() -> None:
    """Owner rows plus retained continuity rows solve the strong BC problem."""
    for ndim in (2, 3):
        results = [solve_direct_laplace(ndim, order, "dirichlet") for order in (1, 2)]
        for solution, continuity, error, rank, constraint_count, residual in results:
            del solution
            assert rank == continuity.shape[1] + constraint_count
            assert constraint_count > continuity.shape[0]
            assert residual < 1.0e-12
            assert np.isfinite(error)
        assert results[1][2] <= results[0][2] + 1.0e-10


def _explicit_test_specs(mesh, form_order: int) -> list[list[list[KFormSpecs]]]:
    """Build valid hierarchical positive-order trace tests."""
    result: list[list[list[KFormSpecs]]] = []
    for mdim in range(mesh.ndim):
        object_count = (
            mesh.point_count if mdim == 0 else mesh.collections[mdim - 1].shape[0]
        )
        objects: list[list[KFormSpecs]] = []
        for _ in range(int(object_count)):
            component_specs: list[KFormSpecs] = []
            for active_axes in combinations(range(mdim), form_order):
                test_orders = tuple(
                    1 if axis in active_axes else -1 for axis in range(mdim)
                )
                if min(test_orders, default=0) < 0:
                    continue
                component_specs.append(
                    KFormSpecs(
                        form_order,
                        FunctionSpace(
                            *(
                                BasisSpecs(BasisType.BERNSTEIN, order)
                                for order in test_orders
                            )
                        ),
                    )
                )
            objects.append(component_specs)
        result.append(objects)
    return result


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
    test_specs = _explicit_test_specs(mesh, form_order)
    row_offsets, element_ids, components, local_dofs, coefficients = (
        mesh.compute_kform_continuity_constraints(element_specs, maps, test_specs)
    )
    del row_offsets, element_ids, local_dofs, coefficients
    assert components.size > 0
    assert set(components.tolist()).issubset(set(expected_components))
    assert np.all(components < element_specs[0].component_count)
    matrix = packed_to_dense(
        (mesh.compute_kform_continuity_constraints(element_specs, maps, test_specs)),
        element_specs,
    )
    assert np.linalg.matrix_rank(matrix) == matrix.shape[0]


def test_empty_mesh_continuity_has_valid_offsets() -> None:
    """A mesh without shared objects returns one empty offset."""
    from fdg import Mesh

    mesh = Mesh.from_corners(2, np.asarray([0, 1, 2, 3], dtype=np.uint64))
    maps = [make_element_maps(2, 5)[0]]
    base_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 1),
        BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 1),
    )
    element_specs = [KFormSpecs(0, base_space)]
    test_specs = [[[] for _ in range(mesh.point_count)] for _ in range(mesh.ndim)]
    result = mesh.compute_kform_continuity_constraints(element_specs, maps, test_specs)
    assert result[0].tolist() == [0]
    assert all(array.size == 0 for array in result[1:])


def test_global_continuity_rejects_wrong_lengths() -> None:
    """Global assembly rejects element sequences with wrong lengths."""
    mesh, element_specs, test_specs, _ = _scalar_setup(2, 1)
    maps = make_element_maps(2, 5)
    with pytest.raises(ValueError, match="each contain"):
        mesh.compute_kform_continuity_constraints(element_specs[:-1], maps, test_specs)


if __name__ == "__main__":
    pytest.main([__file__])
