"""Behavioral tests for global boundary and transformed trace constraints."""

from __future__ import annotations

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    BoundaryCondition,
    BoundaryPair,
    BoundaryPairGroup,
    FunctionSpace,
    KFormSpecs,
)
from fdg.boundary_conditions import _component_relation

from examples.plot_multi_element_laplace_continuity import (
    build_continuity_rows,
    make_element_maps,
    make_mesh,
    make_test_specs,
    packed_to_dense,
)


def _setup(ndim: int, order: int, form_order: int):
    """Build one conforming mapped mesh and explicit hierarchical tests."""
    mesh = make_mesh(ndim)
    maps = make_element_maps(ndim, order + 4)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(form_order, base_space) for _ in maps]
    test_specs = make_test_specs(mesh, element_specs, form_order, BasisType.LEGENDRE)
    return mesh, maps, element_specs, test_specs


def _paired_faces(mesh, axis: int = 0) -> tuple[int, int]:
    """Return one lower/upper outer-face pair perpendicular to ``axis``."""
    faces = mesh.iterate_boundary(mesh.ndim - 1)
    lower = next(
        (int(object_id), element_ids, orientations)
        for _, object_id, element_ids, orientations in faces
        if int(orientations[0, 0]) == -(axis + 1)
    )
    upper = next(
        (int(object_id), element_ids, orientations)
        for _, object_id, element_ids, orientations in faces
        if int(orientations[0, 0]) == axis + 1 and element_ids.size == lower[1].size
    )
    return lower[0], upper[0]


def test_adjacent_boundary_faces_deduplicate_intersection_rows() -> None:
    """Two prescribed faces on one element share one boundary-object row."""
    mesh, maps, element_specs, test_specs = _setup(2, 1, 0)
    faces = mesh.iterate_boundary(1)
    first = faces[0]
    second = next(item for item in faces[1:] if item[2][0] == first[2][0])
    condition = BoundaryCondition(
        (int(first[1]), int(second[1])),
        lambda *coordinates: np.ones_like(coordinates[0]),
    )
    result, rhs = mesh.compute_kform_global_constraints(
        element_specs, maps, test_specs, [condition]
    )
    shared_rows = build_continuity_rows(mesh, maps, element_specs, test_specs)[0].size - 1
    assert rhs.size - shared_rows == 3
    assert result[0].size == rhs.size + 1
    np.testing.assert_allclose(rhs[:shared_rows], 0.0)


def test_inconsistent_adjacent_boundary_data_is_rejected() -> None:
    """Conflicting face data cannot silently choose an intersection value."""
    mesh, maps, element_specs, test_specs = _setup(2, 1, 0)
    faces = mesh.iterate_boundary(1)
    first = faces[0]
    second = next(item for item in faces[1:] if item[2][0] == first[2][0])
    conditions = [
        BoundaryCondition(
            (int(first[1]),), lambda *coordinates: np.ones_like(coordinates[0])
        ),
        BoundaryCondition(
            (int(second[1]),), lambda *coordinates: 2.0 * np.ones_like(coordinates[0])
        ),
    ]
    with pytest.raises(ValueError, match="disagree"):
        mesh.compute_kform_global_constraints(element_specs, maps, test_specs, conditions)


def test_callable_boundary_data_supports_positive_k_forms() -> None:
    """Physical ambient component callables produce finite k-form RHS rows."""
    mesh, maps, element_specs, test_specs = _setup(2, 2, 1)
    face = int(mesh.iterate_boundary(1)[0][1])
    data = (
        lambda x, y: np.ones_like(x + y * 0.0),
        lambda x, y: np.zeros_like(x + y),
    )
    result, rhs = mesh.compute_kform_global_constraints(
        element_specs, maps, test_specs, {face: data}
    )
    shared_rows = build_continuity_rows(mesh, maps, element_specs, test_specs)[0].size - 1
    assert rhs.size > shared_rows
    assert result[0].size == rhs.size + 1
    assert np.isfinite(rhs).all()


def test_reversed_periodic_pair_expands_to_lower_strata() -> None:
    """A signed axis map adds transformed rows without boundary data."""
    mesh, maps, element_specs, test_specs = _setup(2, 2, 0)
    left, right = _paired_faces(mesh)
    result, rhs = mesh.compute_kform_global_constraints(
        element_specs,
        maps,
        test_specs,
        None,
        [BoundaryPair(left, right, (-1,))],
    )
    shared_rows = build_continuity_rows(mesh, maps, element_specs, test_specs)[0].size - 1
    assert rhs.size > shared_rows
    assert result[0].size == rhs.size + 1
    np.testing.assert_allclose(rhs, 0.0)


def test_axis_permutation_maps_positive_form_components() -> None:
    """Periodic axis permutations map k-form components and basis functions."""
    mesh, maps, element_specs, test_specs = _setup(3, 3, 1)
    left, right = _paired_faces(mesh)
    result, rhs = mesh.compute_kform_global_constraints(
        element_specs,
        maps,
        test_specs,
        None,
        [BoundaryPair(left, right, (2, -1))],
    )
    assert result[0].size > 1
    assert np.isfinite(result[4]).all()
    np.testing.assert_allclose(rhs, 0.0)


def test_invalid_periodic_axis_map_is_rejected() -> None:
    """Periodic relations require a signed permutation of boundary axes."""
    mesh, maps, element_specs, test_specs = _setup(2, 1, 0)
    left, right = _paired_faces(mesh)
    with pytest.raises(ValueError, match="signed permutation"):
        mesh.compute_kform_global_constraints(
            element_specs,
            maps,
            test_specs,
            None,
            [BoundaryPair(left, right, (0,))],
        )


def test_boundary_and_periodic_constraints_cannot_overlap() -> None:
    """One boundary object cannot be prescribed and periodic simultaneously."""
    mesh, maps, element_specs, test_specs = _setup(2, 1, 0)
    left, right = _paired_faces(mesh)
    with pytest.raises(ValueError, match="both prescribed and periodic"):
        mesh.compute_kform_global_constraints(
            element_specs,
            maps,
            test_specs,
            {left: lambda *coordinates: np.ones_like(coordinates[0])},
            [BoundaryPair(left, right, (1,))],
        )


def test_signed_axis_map_has_expected_kform_signs() -> None:
    """Signed permutations map components with their exterior-algebra sign."""
    assert _component_relation(2, 1, 0, (2, -1)) == (1, 1)
    assert _component_relation(2, 1, 1, (2, -1)) == (0, -1)
    assert _component_relation(2, 2, 0, (2, -1)) == (0, 1)


def _face_group(mesh, axis: int, side: int) -> list[int]:
    """Return boundary faces in caller-controlled deterministic order."""
    return sorted(
        int(object_id)
        for _, object_id, _, orientations in mesh.iterate_boundary(mesh.ndim - 1)
        if abs(int(orientations[0, 0])) - 1 == axis
        and (1 if orientations[0, 0] > 0 else -1) == side
    )


def test_grouped_periodic_faces_connect_all_cube_sides() -> None:
    """Three paired face groups support a fully periodic subdivided cube."""
    mesh, maps, element_specs, test_specs = _setup(3, 1, 0)
    groups = [
        BoundaryPairGroup(_face_group(mesh, axis, -1), _face_group(mesh, axis, 1), (1, 2))
        for axis in range(3)
    ]
    result, rhs = mesh.compute_kform_global_constraints(
        element_specs, maps, test_specs, None, groups
    )
    shared_rows = build_continuity_rows(mesh, maps, element_specs, test_specs)[0].size - 1
    assert result[0].size - 1 > shared_rows
    assert (
        np.linalg.matrix_rank(packed_to_dense(result, element_specs))
        == result[0].size - 1
    )
    np.testing.assert_allclose(rhs, 0.0)


def test_grouped_periodic_faces_require_equal_lengths() -> None:
    """A group cannot silently drop an unmatched face patch."""
    with pytest.raises(ValueError, match="equal lengths"):
        BoundaryPairGroup((1, 2), (3,), (1,))


def test_boundary_trace_batch_matches_one_sided_assembly() -> None:
    """The C batch path concatenates the established one-sided rows."""
    mesh, maps, element_specs, test_specs = _setup(2, 2, 0)
    faces = mesh.iterate_boundary(1)
    test_spec = test_specs[1][int(faces[0][1])][0]
    element_ids = [int(element_ids[0]) for _, _, element_ids, _ in faces]
    boundary_ids = [int(object_id) for _, object_id, _, _ in faces]
    batched = mesh.compute_kform_boundary_constraints_batch(
        test_spec,
        element_specs[0],
        [maps[element_id] for element_id in element_ids],
        element_ids,
        boundary_ids,
    )
    offsets, batched_elements, batched_components, batched_dofs, batched_coefficients = (
        batched
    )
    row = 0
    for element_id, boundary_id in zip(element_ids, boundary_ids, strict=True):
        local = mesh.compute_kform_boundary_constraints(
            test_spec,
            element_specs[0],
            maps[element_id],
            element_id,
            boundary_id,
        )
        local_offsets, local_components, local_dofs, local_coefficients = local
        local_rows = local_offsets.size - 1
        start = int(offsets[row])
        end = int(offsets[row + local_rows])
        assert end - start == local_components.size
        np.testing.assert_array_equal(batched_elements[start:end], element_id)
        np.testing.assert_array_equal(batched_components[start:end], local_components)
        np.testing.assert_array_equal(batched_dofs[start:end], local_dofs)
        np.testing.assert_allclose(batched_coefficients[start:end], local_coefficients)
        row += local_rows
    assert row == offsets.size - 1
