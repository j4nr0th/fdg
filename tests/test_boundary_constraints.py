"""Tests for common-boundary-space constraint mass matrices."""

from __future__ import annotations

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    SpaceMap,
)
from fdg._fdg import (
    compute_boundary_space_map_factors,
    compute_kform_boundary_mass_matrices,
)


def _element_spec(
    order: int,
    ndim: int = 2,
    form_order: int = 0,
    basis: BasisType = BasisType.LAGRANGE_GAUSS_LOBATTO,
) -> KFormSpecs:
    """One k-form of `form_order` on a uniform-order element."""
    space = FunctionSpace(*(BasisSpecs(basis, order) for _ in range(ndim)))
    return KFormSpecs(form_order, space)


def _integrations(orders: tuple[int, ...]) -> IntegrationSpace:
    """Gauss integration space with one order per axis."""
    return IntegrationSpace(
        *(IntegrationSpecs(order, IntegrationMethod.GAUSS) for order in orders)
    )


def _lgl_nodes(order: int) -> np.ndarray:
    """Gauss-Lobatto nodes of one order."""
    space = IntegrationSpace(IntegrationSpecs(order, IntegrationMethod.GAUSS_LOBATTO))
    return space.nodes()[0]


def _lagrange_matrix(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate Lagrange basis on `nodes` at `points`; shape (points, nodes)."""
    count = len(nodes)
    weights = np.empty(count)
    for j in range(count):
        diff = nodes[j] - np.delete(nodes, j)
        if np.any(diff == 0.0):
            raise ValueError("Lagrange nodes must be distinct.")
        weights[j] = 1.0 / np.prod(diff)
    matrix = np.empty((len(points), count))
    for i, point in enumerate(points):
        exact = np.nonzero(nodes == point)[0]
        if len(exact) > 0:
            row = np.zeros(count)
            row[exact[0]] = 1.0
            matrix[i] = row
            continue
        values = weights / (point - nodes)
        matrix[i] = values / values.sum()
    return matrix


def _orientation_record(*axes: int) -> list[int]:
    """Signed one-based orientation record."""
    return list(axes)


def test_common_space_merges_orders_and_rules() -> None:
    """The common link space takes the per-axis minimum order, maximum accuracy.

    The lowest order is the strongest link that never overconstrains: an
    element boundary cannot be constrained to a higher-order boundary
    solution, so higher-order traces conform in the L2 sense.
    """
    specs = [_element_spec(2), _element_spec(3)]
    orientations = [_orientation_record(1, 2), _orientation_record(-1, 2)]
    integrations = [_integrations((3, 3)), _integrations((4, 5))]

    common_specs, common_integration, _, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations
    )

    assert common_specs.order == 0
    assert common_integration.orders == (5,)
    types = [basis.type for basis in common_specs.base_space.basis_specs]
    assert all(t == BasisType.LEGENDRE for t in types)
    assert [basis.order for basis in common_specs.base_space.basis_specs] == [2]


def test_scalar_mass_matrix_matches_quadrature() -> None:
    """The order-zero mass block equals the integrated Legendre-Lagrange pairing."""
    order = 3
    integration_order = 4
    specs = [_element_spec(order), _element_spec(order)]
    orientations = [_orientation_record(1, 2), _orientation_record(-1, 2)]
    integrations = [_integrations((integration_order, integration_order))] * 2

    _, common_integration, matrices, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations
    )

    nodes = np.asarray(common_integration.nodes()[0])
    weights = np.asarray(common_integration.weights())
    legendre = np.polynomial.legendre.legvander(nodes, order)
    lagrange = _lagrange_matrix(_lgl_nodes(order), nodes)
    endpoint = _lagrange_matrix(_lgl_nodes(order), np.array([1.0]))[0]

    expected = np.empty((order + 1, (order + 1) ** 2))
    for row in range(order + 1):
        for normal in range(order + 1):
            block = (legendre[:, row] * weights) @ lagrange
            expected[row, normal * (order + 1) : (normal + 1) * (order + 1)] = (
                endpoint[normal] * block
            )

    np.testing.assert_allclose(matrices[0], expected, atol=1e-13)


def test_mirrored_orientation_matches_quadrature() -> None:
    """A reversed tangential axis reads mirrored nodes with matching sign."""
    order = 3
    specs = [_element_spec(order), _element_spec(order)]
    orientations = [_orientation_record(2, -1), _orientation_record(2, 1)]
    integrations = [_integrations((integration_order := 4, integration_order))] * 2

    _, common_integration, matrices, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations, boundary_dimension=1
    )

    nodes = np.asarray(common_integration.nodes()[0])
    weights = np.asarray(common_integration.weights())
    legendre = np.polynomial.legendre.legvander(nodes, order)
    lagrange = _lagrange_matrix(_lgl_nodes(order), -nodes)
    # Orientation (2, -1): element axis 0 is the tangential axis and element
    # axis 1 the normal one at +1, so columns enumerate tangential slow.
    endpoint = _lagrange_matrix(_lgl_nodes(order), np.array([1.0]))[0]

    expected = np.zeros((order + 1, (order + 1) ** 2))
    for row in range(order + 1):
        for tangential in range(order + 1):
            for normal in range(order + 1):
                expected[row, tangential * (order + 1) + normal] = endpoint[normal] * (
                    (legendre[:, row] * weights) @ lagrange[:, tangential]
                )

    np.testing.assert_allclose(matrices[0], expected, atol=1e-13)


def test_one_form_uses_lower_basis_and_orientation_sign() -> None:
    """Active covector axes read the order-reduced basis; signs follow orientation."""
    order = 3
    specs = [_element_spec(order, form_order=1), _element_spec(order, form_order=1)]
    orientations = [_orientation_record(1, 2), _orientation_record(-1, 2)]
    integrations = [_integrations((5, 5))] * 2

    _, common_integration, matrices, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations, boundary_dimension=1
    )

    nodes = np.asarray(common_integration.nodes()[0])
    weights = np.asarray(common_integration.weights())
    lower_legendre = np.polynomial.legendre.legvander(nodes, order - 1)
    lower_lagrange = _lagrange_matrix(_lgl_nodes(order - 1), nodes)
    endpoint = _lagrange_matrix(_lgl_nodes(order), np.array([1.0]))[0]

    block = np.empty((order, order * (order + 1)))
    for row in range(order):
        for normal in range(order + 1):
            block[row, normal * order : (normal + 1) * order] = (
                endpoint[normal] * (lower_legendre[:, row] * weights) @ lower_lagrange
            )

    np.testing.assert_allclose(matrices[0], block, atol=1e-13)
    # Element B's face sits at -1: the normal endpoint value carries the side,
    # and the mass matrices themselves stay side-sign free.
    endpoint_b = _lagrange_matrix(_lgl_nodes(order), np.array([-1.0]))[0]
    block_b = np.empty_like(block)
    for row in range(order):
        for normal in range(order + 1):
            block_b[row, normal * order : (normal + 1) * order] = (
                endpoint_b[normal] * (lower_legendre[:, row] * weights) @ lower_lagrange
            )

    np.testing.assert_allclose(matrices[1], block_b, atol=1e-13)
    assert matrices[0].shape == (order, order * (order + 1))


def test_axis_skip_removes_lowest_legendre_rows() -> None:
    """Skipping two test functions drops the lowest Legendre degrees."""
    order = 3
    specs = [_element_spec(order), _element_spec(order)]
    orientations = [_orientation_record(1, 2), _orientation_record(-1, 2)]
    integrations = [_integrations((5, 5))] * 2

    _, _, matrices, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations
    )
    _, _, skipped, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations, axis_skip=[2]
    )

    assert skipped[0].shape[0] == matrices[0].shape[0] - 2
    np.testing.assert_allclose(skipped[0], matrices[0][2:, :], atol=1e-13)
    np.testing.assert_allclose(skipped[1], matrices[1][2:, :], atol=1e-13)


def test_packed_rows_match_dense_matrix() -> None:
    """The COO packing reproduces the dense blocks row by row."""
    order = 3
    specs = [_element_spec(order), _element_spec(order)]
    orientations = [_orientation_record(1, 2), _orientation_record(-1, 2)]
    integrations = [_integrations((5, 5))] * 2

    _, _, matrices, packed = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations, packed=True
    )

    for element, matrix in enumerate(matrices):
        row_offsets, sides, components, local_dofs, coefficients = packed[element]
        for row in range(matrix.shape[0]):
            start, stop = row_offsets[row], row_offsets[row + 1]
            assert np.all(sides[start:stop] == element)
            row_entries = np.zeros(matrix.shape[1])
            for entry in range(start, stop):
                column = components[entry] * (order + 1) + local_dofs[entry]
                row_entries[column] += coefficients[entry]
            np.testing.assert_allclose(row_entries, matrix[row], atol=1e-13)


def test_three_dimensional_face_component_blocks() -> None:
    """A 3D face gives two one-form components with block-diagonal reference pairing."""
    order = 2
    specs = [
        _element_spec(order, ndim=3, form_order=1),
        _element_spec(order, ndim=3, form_order=1),
    ]
    orientations = [_orientation_record(3, 1, 2), _orientation_record(-3, 2, 1)]
    integrations = [_integrations((4, 4, 4))] * 2

    _, _, matrices, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations, boundary_dimension=2
    )

    matrix = matrices[0]
    component_rows = order * (order + 1)
    component_cols = order * (order + 1) ** 2
    assert matrix.shape == (2 * component_rows, 2 * component_cols)
    # Reference pairing is component-diagonal.
    np.testing.assert_allclose(matrix[:component_rows, component_cols:], 0.0, atol=1e-13)
    np.testing.assert_allclose(matrix[component_rows:, :component_cols], 0.0, atol=1e-13)


def test_shared_face_fields_annihilate() -> None:
    """Two elements carrying one face field produce cancelling mass rows."""
    order = 3
    specs = [_element_spec(order), _element_spec(order)]
    orientations = [_orientation_record(1, 2), _orientation_record(-1, 2)]
    integrations = [_integrations((5, 5))] * 2

    _, common_integration, matrices, _ = compute_kform_boundary_mass_matrices(
        specs, orientations, integrations
    )

    nodes = np.asarray(common_integration.nodes()[0])
    lagrange = _lagrange_matrix(_lgl_nodes(order), nodes)
    endpoint = _lagrange_matrix(_lgl_nodes(order), np.array([1.0]))[0]
    rng = np.random.default_rng(7)

    element_dofs = rng.random((order + 1, order + 1))
    trace = lagrange @ (element_dofs.T @ endpoint)

    # Element B carries the same face field: its normal axis uses the
    # endpoint at -1, so scale one normal function to unit endpoint value and
    # solve for the tangential coefficients of the trace.
    endpoint_b = _lagrange_matrix(_lgl_nodes(order), np.array([-1.0]))[0]
    normal_scales = np.zeros(order + 1)
    normal_scales[0] = 1.0 / endpoint_b[0]
    tangential, *_ = np.linalg.lstsq(lagrange, trace, rcond=None)
    dofs_b = np.outer(normal_scales, tangential)

    residual = matrices[0] @ element_dofs.ravel() - matrices[1] @ dofs_b.ravel()
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_curved_face_surface_measure() -> None:
    """Resampling a curved face map reproduces the analytic surface measure."""
    geometry_order = 2
    nodes = np.linspace(-1.0, 1.0, geometry_order + 1)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
    geometry_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, geometry_order) for _ in range(2))
    )
    int_space = _integrations((4, 4))
    map_x = CoordinateMap(
        DegreesOfFreedom(geometry_space, (grid_x + 0.3 * grid_y**2).ravel()), int_space
    )
    map_y = CoordinateMap(DegreesOfFreedom(geometry_space, grid_y.ravel()), int_space)
    space_map = SpaceMap(map_x, map_y)

    common = IntegrationSpace(IntegrationSpecs(5, IntegrationMethod.GAUSS))
    determinant, inverse_maps = compute_boundary_space_map_factors(
        space_map, _orientation_record(1, 2), common
    )

    nodes_face = np.asarray(common.nodes()[0])
    expected = np.sqrt((0.6 * nodes_face) ** 2 + 1.0)
    np.testing.assert_allclose(determinant, expected, atol=1e-12)
    assert inverse_maps.shape == (len(nodes_face), 1, 2)


if __name__ == "__main__":
    pytest.main([__file__])
