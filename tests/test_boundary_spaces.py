"""Tests for automatically derived boundary test spaces."""

from __future__ import annotations

import numpy as np
import pytest
from fdg import BasisSpecs, BasisType, FunctionSpace, KFormSpecs, Mesh


def _two_element_mesh(order_low: int, order_high: int, basis_type: BasisType):
    """Build a two-element 2D strip with mixed per-element orders."""
    corners = np.asarray([0, 1, 3, 4, 1, 2, 4, 5], dtype=np.uint64)
    mesh = Mesh.from_corners(2, corners)
    low_space = FunctionSpace(*(BasisSpecs(basis_type, order_low) for _ in range(2)))
    high_space = FunctionSpace(*(BasisSpecs(basis_type, order_high) for _ in range(2)))
    element_specs = [
        KFormSpecs(0, low_space),
        KFormSpecs(0, high_space),
    ]
    return mesh, element_specs


def test_per_axis_minimum_and_inactive_reduction() -> None:
    """Orders take the per-axis incident minimum, reduced by two on inactive axes."""
    mesh, element_specs = _two_element_mesh(2, 3, BasisType.LEGENDRE)
    spaces = mesh.kform_boundary_spaces(element_specs)
    # The single shared edge spans one axis; its scalar test takes the lower
    # incident order minus two.
    shared = mesh.iterate_shared(1)
    assert shared[0][2].size == 2
    edge_id = int(shared[0][1])
    edge_spaces = spaces[1][edge_id]
    assert len(edge_spaces) == 1
    assert edge_spaces[0] is not None
    assert tuple(edge_spaces[0].base_space.orders) == (0,)
    # Point objects keep their single point-value test component.
    for point_spaces in spaces[0]:
        assert len(point_spaces) == 1
        assert point_spaces[0] is not None
        assert point_spaces[0].base_space.orders == ()


def test_components_without_rows_hold_none() -> None:
    """A one-form on order-one elements reports every component absent."""
    mesh, _ = _two_element_mesh(1, 1, BasisType.LEGENDRE)
    form_space = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, 1), BasisSpecs(BasisType.LEGENDRE, 1)
    )
    element_specs = [KFormSpecs(1, form_space) for _ in range(2)]
    spaces = mesh.kform_boundary_spaces(element_specs)
    # Shared edges (mdim 1): the single tangential component stays at the
    # full order, so it is present.
    shared = mesh.iterate_shared(1)
    edge_spaces = spaces[1][int(shared[0][1])]
    assert edge_spaces[0] is not None
    assert tuple(edge_spaces[0].base_space.orders) == (1,)
    # Shared points (mdim 0) carry no one-form rows at all.
    assert not spaces[0][0]


def test_default_family_follows_lowest_order_element() -> None:
    """The derived family matches the incident element with fewer DoFs."""
    mesh, _ = _two_element_mesh(2, 3, BasisType.LEGENDRE)
    mixed = [
        KFormSpecs(
            0,
            FunctionSpace(
                BasisSpecs(BasisType.LEGENDRE, 2),
                BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 3),
            ),
        ),
        KFormSpecs(
            0,
            FunctionSpace(
                BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 3),
                BasisSpecs(BasisType.LEGENDRE, 3),
            ),
        ),
    ]
    spaces = mesh.kform_boundary_spaces(mixed)
    shared = mesh.iterate_shared(1)
    edge_id = int(shared[0][1])
    test_spec = spaces[1][edge_id][0]
    assert test_spec is not None
    # The edge spans one canonical axis; the minimum order (3) is a tie
    # between both elements, so the family comes from the lowest element.
    orders = test_spec.base_space.orders
    types = [spec.type for spec in test_spec.base_space.basis_specs]
    assert tuple(orders) == (1,)
    assert types == [BasisType.LAGRANGE_GAUSS_LOBATTO]


def test_basis_type_override_forces_one_family() -> None:
    """An explicit basis type overrides every derived axis."""
    mesh, element_specs = _two_element_mesh(2, 3, BasisType.LEGENDRE)
    spaces = mesh.kform_boundary_spaces(element_specs, basis_type=BasisType.BERNSTEIN)
    for dimension_spaces in spaces:
        for object_spaces in dimension_spaces:
            for test_spec in object_spaces:
                if test_spec is None:
                    continue
                assert all(
                    spec.type == BasisType.BERNSTEIN
                    for spec in test_spec.base_space.basis_specs
                )


def test_invalid_basis_type_is_rejected() -> None:
    """Unknown basis families fail before any derivation."""
    mesh, element_specs = _two_element_mesh(2, 2, BasisType.LEGENDRE)
    with pytest.raises(TypeError, match="BasisType"):
        mesh.kform_boundary_spaces(element_specs, basis_type=99)
    with pytest.raises(ValueError, match="basis family"):
        mesh.kform_boundary_spaces(element_specs, basis_type="not-a-family")


def test_wrong_element_spec_count_is_rejected() -> None:
    """The derivation requires one specification per mesh element."""
    mesh, element_specs = _two_element_mesh(2, 2, BasisType.LEGENDRE)
    with pytest.raises(ValueError, match="must contain 2"):
        mesh.kform_boundary_spaces(element_specs[:-1])


def test_continuity_supports_mixed_orders() -> None:
    """Mixed-order meshes produce full-rank continuity rows."""
    mesh, scalar_specs = _two_element_mesh(2, 2, BasisType.LEGENDRE)
    form_space = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, 2), BasisSpecs(BasisType.LEGENDRE, 2)
    )
    form_specs = [KFormSpecs(1, form_space) for _ in range(2)]

    for specs in (scalar_specs, form_specs):
        result = mesh.compute_kform_continuity_constraints(
            specs, None, c1_continuous=True
        )
        row_offsets, element_ids, components, local_dofs, coefficients = result
        if row_offsets.size == 1:
            continue
        counts = [int(np.sum(s.component_dof_counts)) for s in specs]
        matrix = np.zeros((row_offsets.size - 1, sum(counts)))
        element_starts = np.cumsum([0, *counts[:-1]])
        for row in range(row_offsets.size - 1):
            for index in range(int(row_offsets[row]), int(row_offsets[row + 1])):
                element = int(element_ids[index])
                component = int(components[index])
                column = (
                    element_starts[element]
                    + int(specs[element].get_component_slice(component).start)
                    + int(local_dofs[index])
                )
                matrix[row, column] += coefficients[index]
        assert np.linalg.matrix_rank(matrix, tol=1.0e-9) == matrix.shape[0]


if __name__ == "__main__":
    pytest.main([__file__])
