"""Check that mass matrix computations are correct."""

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    compute_mass_matrix,
)


@pytest.mark.parametrize(
    ("order_1,basis_type_1,order_2,basis_type_2"),
    (
        (1, BasisType.LEGENDRE, 1, BasisType.LEGENDRE),
        (3, BasisType.LEGENDRE, 3, BasisType.LEGENDRE),
        (2, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, 2, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
        (3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, 3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
        (2, BasisType.LEGENDRE, 3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
        (3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, 2, BasisType.LEGENDRE),
        (4, BasisType.LEGENDRE, 5, BasisType.BERNSTEIN),
    ),
)
def test_mass_matrix_1d(
    order_1: int, basis_type_1: BasisType, order_2: int, basis_type_2: BasisType
) -> None:
    """Check mass matrix computation in 1D."""
    int_space = IntegrationSpace(
        IntegrationSpecs(max(order_1, order_2) + 1, IntegrationMethod.GAUSS)
    )
    func_space_1 = FunctionSpace(BasisSpecs(basis_type_1, order_1))
    func_space_2 = FunctionSpace(BasisSpecs(basis_type_2, order_2))

    mass_matrix = compute_mass_matrix(func_space_1, func_space_2, int_space)

    int_nodes = int_space.nodes()[0, ...]
    int_weights = int_space.weights()

    vals_1 = func_space_1.evaluate(int_nodes)
    vals_2 = func_space_2.evaluate(int_nodes)

    expected_mass_matrix = np.sum(
        vals_1[..., None, :] * vals_2[..., :, None] * int_weights[:, None, None], axis=0
    )
    assert pytest.approx(mass_matrix) == expected_mass_matrix


@pytest.mark.parametrize(
    ("spec11", "spec12", "spec21", "spec22"),
    (
        (
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
        ),
        (
            BasisSpecs(BasisType.LEGENDRE, 2),
            BasisSpecs(BasisType.LEGENDRE, 3),
            BasisSpecs(BasisType.LEGENDRE, 2),
            BasisSpecs(BasisType.LEGENDRE, 3),
        ),
        (
            BasisSpecs(BasisType.LEGENDRE, 2),
            BasisSpecs(BasisType.BERNSTEIN, 3),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
            BasisSpecs(BasisType.LAGRNAGE_GAUSS_LOBATTO, 4),
        ),
    ),
)
def test_mass_matrix_2d(
    spec11: BasisSpecs, spec12: BasisSpecs, spec21: BasisSpecs, spec22: BasisSpecs
) -> None:
    """Check mass matrix computation in 2D."""
    order_1 = max(spec11.order, spec12.order)
    order_2 = max(spec21.order, spec22.order)

    int_space = IntegrationSpace(
        IntegrationSpecs(order_1 + 1, IntegrationMethod.GAUSS),
        IntegrationSpecs(order_2 + 1, IntegrationMethod.GAUSS),
    )
    func_space_1 = FunctionSpace(spec11, spec12)
    func_space_2 = FunctionSpace(spec21, spec22)

    mass_matrix = compute_mass_matrix(func_space_1, func_space_2, int_space)

    int_nodes_1 = int_space.nodes()[0, ...]
    int_nodes_2 = int_space.nodes()[1, ...]
    int_weights = int_space.weights()

    vals_1 = func_space_1.evaluate(int_nodes_1, int_nodes_2).reshape(
        int_weights.shape + (-1,)
    )
    vals_2 = func_space_2.evaluate(int_nodes_1, int_nodes_2).reshape(
        int_weights.shape + (-1,)
    )

    expected_mass_matrix = np.sum(
        vals_1[..., None, :] * vals_2[..., :, None] * int_weights[..., None, None],
        axis=tuple(range(int_weights.ndim)),
    )
    assert pytest.approx(mass_matrix) == expected_mass_matrix


@pytest.mark.parametrize(
    ("specs_11", "specs_12", "specs_13", "specs_21", "specs_22", "specs_23"),
    (
        (
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
            BasisSpecs(BasisType.LEGENDRE, 1),
        ),
        (
            BasisSpecs(BasisType.LEGENDRE, 2),
            BasisSpecs(BasisType.BERNSTEIN, 3),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
            BasisSpecs(BasisType.LAGRNAGE_GAUSS_LOBATTO, 4),
            BasisSpecs(BasisType.LEGENDRE, 2),
            BasisSpecs(BasisType.BERNSTEIN, 3),
        ),
        (
            BasisSpecs(BasisType.LEGENDRE, 2),
            BasisSpecs(BasisType.BERNSTEIN, 3),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
            BasisSpecs(BasisType.LAGRNAGE_GAUSS_LOBATTO, 4),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
            BasisSpecs(BasisType.LAGRNAGE_GAUSS_LOBATTO, 4),
        ),
    ),
)
def test_mass_matrix_3d(
    specs_11: BasisSpecs,
    specs_12: BasisSpecs,
    specs_13: BasisSpecs,
    specs_21: BasisSpecs,
    specs_22: BasisSpecs,
    specs_23: BasisSpecs,
) -> None:
    """Check mass matrix computation in 3D."""
    order_1 = max(specs_11.order, specs_21.order)
    order_2 = max(specs_12.order, specs_22.order)
    order_3 = max(specs_13.order, specs_23.order)

    int_space = IntegrationSpace(
        IntegrationSpecs(order_1 + 1, IntegrationMethod.GAUSS),
        IntegrationSpecs(order_2 + 1, IntegrationMethod.GAUSS),
        IntegrationSpecs(order_3 + 1, IntegrationMethod.GAUSS),
    )
    func_space_1 = FunctionSpace(specs_11, specs_12, specs_13)
    func_space_2 = FunctionSpace(specs_21, specs_22, specs_23)

    mass_matrix = compute_mass_matrix(func_space_1, func_space_2, int_space)

    int_nodes_1 = int_space.nodes()[0, ...]
    int_nodes_2 = int_space.nodes()[1, ...]
    int_nodes_3 = int_space.nodes()[2, ...]
    int_weights = int_space.weights()

    vals_1 = func_space_1.evaluate(int_nodes_1, int_nodes_2, int_nodes_3).reshape(
        int_weights.shape + (-1,)
    )
    vals_2 = func_space_2.evaluate(int_nodes_1, int_nodes_2, int_nodes_3).reshape(
        int_weights.shape + (-1,)
    )

    expected_mass_matrix = np.sum(
        vals_1[..., None, :] * vals_2[..., :, None] * int_weights[..., None, None],
        axis=tuple(range(int_weights.ndim)),
    )
    assert pytest.approx(mass_matrix) == expected_mass_matrix


if __name__ == "__main__":
    # test_mass_matrix_1d(2, BasisType.LEGENDRE, 3, BasisType.LEGENDRE)
    test_mass_matrix_2d(
        BasisSpecs(BasisType.LEGENDRE, 2),
        BasisSpecs(BasisType.BERNSTEIN, 3),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
        BasisSpecs(BasisType.LAGRNAGE_GAUSS_LOBATTO, 4),
    )
