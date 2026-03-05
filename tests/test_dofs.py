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


_TEST_ORDERS = (
    (2, 2, 2),
    (3, 4, 5),
    (1, 2, 3, 4, 5),
)


@pytest.mark.parametrize("orders", _TEST_ORDERS)
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


@pytest.mark.parametrize("orders", _TEST_ORDERS)
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


@pytest.mark.parametrize("orders", _TEST_ORDERS)
@pytest.mark.parametrize("btype", BasisType)
def test_plane_projection(orders: tuple[int, ...], btype: BasisType) -> None:
    """Check the plane projection works correctly."""
    function_space = FunctionSpace(*[BasisSpecs(btype, order) for order in orders])
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

    reconstruction_grid = integration_space.nodes()

    # Make a few random projections
    for proj_val in rng.random(len(orders)):
        for idim in range(len(orders)):
            projection = dofs.plane_projection(idim, float(proj_val))
            proj_recon = reconstruct(
                projection, *reconstruction_grid[:idim], *reconstruction_grid[idim + 1 :]
            )
            real_recon = reconstruct(
                dofs,
                *reconstruction_grid[:idim],
                np.full_like(reconstruction_grid[idim], proj_val),
                *reconstruction_grid[idim + 1 :],
            )

            assert pytest.approx(proj_recon) == real_recon


@pytest.mark.parametrize("orders", _TEST_ORDERS)
@pytest.mark.parametrize("btype", BasisType)
def test_reverse_orientation(orders: tuple[int, ...], btype: BasisType) -> None:
    """Check the orientation reversal works correctly."""
    function_space = FunctionSpace(*[BasisSpecs(btype, order) for order in orders])
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

    reconstruction_grid = integration_space.nodes()

    # Make a few random projections
    for idim in range(len(orders)):
        reversal = dofs.reverse_orientation(idim)
        revs_recon = reconstruct(
            reversal,
            *reconstruction_grid[:idim],
            -reconstruction_grid[idim],
            *reconstruction_grid[idim + 1 :],
        )
        real_recon = reconstruct(dofs, *reconstruction_grid)

        assert pytest.approx(revs_recon) == real_recon


if __name__ == "__main__":
    for orders in _TEST_ORDERS:
        for btype in BasisType:
            test_reverse_orientation(orders, btype)
