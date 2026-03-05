"""Test degree of freedom code."""

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    compute_mass_matrix,
    reconstruct,
)
from fdg.integration import projection_l2_dual


def test_dofs_initialization():
    """Test initialization of DegreesOfFreedom."""
    basis_specs_1 = BasisSpecs(BasisType.LEGENDRE, 2)
    basis_specs_2 = BasisSpecs(BasisType.LEGENDRE, 4)
    basis_specs_3 = BasisSpecs(BasisType.LEGENDRE, 5)
    function_space = FunctionSpace(basis_specs_1, basis_specs_2, basis_specs_3)
    dofs = DegreesOfFreedom(function_space)

    zeros = np.zeros(tuple(x + 1 for x in function_space.orders))

    assert dofs.function_space == function_space
    assert dofs.n_dofs == zeros.size
    assert np.all(dofs.values == zeros)
    assert np.all(dofs.shape == zeros.shape)

    rng = np.random.default_rng(1234)
    randoms = rng.random(zeros.shape)
    dofs.values = randoms
    assert np.all(dofs.values == randoms)
    dofs.values = zeros.flat
    assert np.all(dofs.values == zeros)


@pytest.mark.parametrize(
    "orders",
    (
        (2, 2, 2),
        (3, 4, 5),
        (1, 2, 3),
    ),
)
def test_degrees_of_freedom(orders: tuple[int, ...]) -> None:
    """Check that we can compute dual degrees of freedom correctly, the reconstruct."""

    def test_function(*args):
        return sum(arg**order for arg, order in zip(args, orders))

    integration_space = IntegrationSpace(
        *[IntegrationSpecs(order + 2, IntegrationMethod.GAUSS) for order in orders]
    )
    function_space = FunctionSpace(
        *[BasisSpecs(BasisType.LEGENDRE, order) for order in orders]
    )

    dual_dofs = projection_l2_dual(test_function, function_space, integration_space)
    mass_matrix = compute_mass_matrix(function_space, function_space, integration_space)
    dofs_values = np.linalg.solve(mass_matrix, dual_dofs.values.flatten())
    dofs = DegreesOfFreedom(function_space, dofs_values)
    pts = np.meshgrid(*[np.linspace(-1, 1, num=5) for _ in range(len(orders))])
    reconstructed_function = reconstruct(dofs, *pts)
    expected_function = test_function(*pts)
    assert pytest.approx(reconstructed_function) == expected_function


@pytest.mark.parametrize(
    "orders",
    (
        (2, 2, 2),
        (3, 4, 5),
        (1, 2, 3, 4, 5),
    ),
)
def test_reconstruction_at_integration_nodes(orders: tuple[int, ...]) -> None:
    """Check that reconstruction at integration nodes works correctly."""
    integration_space = IntegrationSpace(
        *[IntegrationSpecs(order + 2, IntegrationMethod.GAUSS) for order in orders]
    )
    function_space = FunctionSpace(
        *[BasisSpecs(BasisType.LEGENDRE, order) for order in orders]
    )

    rng = np.random.default_rng(1234)
    dofs = DegreesOfFreedom(function_space)
    dofs.values = rng.random(dofs.shape)

    nodes = integration_space.nodes()
    expected_reconstruction = reconstruct(
        dofs, *[nodes[i, ...] for i in range(nodes.shape[0])]
    )
    test_reconstruction = dofs.reconstruct_at_integration_points(integration_space)
    assert pytest.approx(expected_reconstruction) == test_reconstruction


@pytest.mark.parametrize(
    "orders",
    (
        (2, 2, 2),
        (3, 4, 5),
        (1, 2, 3, 4, 5),
    ),
)
def test_reconstruction_lagrange_projection(orders: tuple[int, ...]) -> None:
    """Check that lagrange projection works as one would expect."""
    function_space = FunctionSpace(
        *[BasisSpecs(BasisType.LEGENDRE, order) for order in orders]
    )
    rng = np.random.default_rng(21598)
    int_orders = rng.integers(0, 6, len(orders))
    integration_space = IntegrationSpace(
        *[
            IntegrationSpecs(int(order + 2), IntegrationMethod.GAUSS_LOBATTO)
            for order in int_orders
        ]
    )

    dofs = DegreesOfFreedom(function_space)
    dofs.values = rng.random(dofs.shape)

    at_int_nodes = dofs.reconstruct_at_integration_points(integration_space)
    projected = dofs.lagrange_projection(integration_space.orders)

    assert pytest.approx(at_int_nodes) == projected.values


if __name__ == "__main__":
    for orders in (
        (2, 2, 2),
        (3, 4, 5),
        (1, 2, 3, 4, 5),
    ):
        test_reconstruction_lagrange_projection(orders)
