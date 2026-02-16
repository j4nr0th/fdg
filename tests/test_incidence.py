"""Check that indicence matrices work correctly."""

import numpy as np
import numpy.typing as npt
import pytest
from interplib._interp import (
    BasisSpecs,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    incidence_matrix,
    incidence_operator,
)
from interplib.enum_type import BasisType, IntegrationMethod
from interplib.integration import projection_l2_primal


@pytest.mark.parametrize("btype", BasisType)
@pytest.mark.parametrize("order", (1, 2, 5, 10))
def test_incidence(order: int, btype: BasisType) -> None:
    """Check that applying an incidence matrix returns correct DoFs."""
    specs = BasisSpecs(btype, order)
    int_space = IntegrationSpace(IntegrationSpecs(order + 1, IntegrationMethod.GAUSS))
    func_space = FunctionSpace(specs)

    def real_function(*args):
        (x,) = args
        return x**order

    def real_derivative(*args):
        (x,) = args
        return order * x ** (order - 1)

    fn_dofs = projection_l2_primal(real_function, func_space, int_space)
    d_dofs = projection_l2_primal(real_derivative, func_space.lower_order(0), int_space)

    incidence = incidence_matrix(specs)
    computed = incidence @ fn_dofs.values
    expected = d_dofs.values

    assert pytest.approx(expected) == computed


@pytest.mark.parametrize("btype", BasisType)
@pytest.mark.parametrize("order", (1, 2, 5, 10))
def test_derivative_1d(order: int, btype: BasisType) -> None:
    """Check that applying an incidence matrix is same as derivative method."""
    specs = BasisSpecs(btype, order)
    func_space = FunctionSpace(specs)

    rng = np.random.default_rng(125)

    fn_dofs = DegreesOfFreedom(func_space)
    fn_dofs.values = rng.random(fn_dofs.shape)

    incidence = incidence_matrix(specs)
    # Apply on 1st axis
    expected = incidence @ fn_dofs.values
    computed = fn_dofs.derivative(0).values

    assert pytest.approx(expected) == computed

    # Use applying function instead of matrix
    applied = incidence_operator(fn_dofs.values, specs, axis=0)
    assert pytest.approx(expected) == applied


@pytest.mark.parametrize("btype1", BasisType)
@pytest.mark.parametrize("order1", (1, 2, 5, 10))
@pytest.mark.parametrize("btype2", BasisType)
@pytest.mark.parametrize("order2", (1, 2, 5, 10))
def test_derivative_2d(
    order1: int, btype1: BasisType, order2: int, btype2: BasisType
) -> None:
    """Check that applying an incidence matrix is same as derivative method."""
    func_space = FunctionSpace(BasisSpecs(btype1, order1), BasisSpecs(btype2, order2))

    rng = np.random.default_rng(125)

    fn_dofs = DegreesOfFreedom(func_space)
    fn_dofs.values = rng.random(fn_dofs.shape)

    v: list[npt.NDArray[np.float64]]
    # First dimension
    incidence = incidence_matrix(func_space.basis_specs[0])
    v = list()
    for j in range(fn_dofs.shape[1]):
        v.append(incidence @ fn_dofs.values[:, j])
    expected = np.stack(v, axis=1)
    computed = fn_dofs.derivative(0).values
    assert pytest.approx(expected) == computed

    # Use applying function instead of matrix
    applied = incidence_operator(fn_dofs.values, func_space.basis_specs[0], axis=0)
    assert pytest.approx(expected) == applied

    # Second dimension
    incidence = incidence_matrix(func_space.basis_specs[1])
    v = list()
    for i in range(fn_dofs.shape[0]):
        v.append(incidence @ fn_dofs.values[i, :])
    expected = np.stack(v, axis=0)
    computed = fn_dofs.derivative(1).values
    assert pytest.approx(expected) == computed

    # Use applying function instead of matrix
    applied = incidence_operator(fn_dofs.values, func_space.basis_specs[1], axis=1)
    assert pytest.approx(expected) == applied


@pytest.mark.parametrize(
    ("btype1", "btype2", "btype3"),
    (
        (BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
        (BasisType.LAGRANGE_CHEBYSHEV_GAUSS, BasisType.LEGENDRE, BasisType.BERNSTEIN),
        (
            BasisType.LAGRANGE_CHEBYSHEV_GAUSS,
            BasisType.LAGRNAGE_GAUSS_LOBATTO,
            BasisType.LAGRANGE_UNIFORM,
        ),
    ),
)
@pytest.mark.parametrize(
    ("order1", "order2", "order3"),
    (
        (1, 2, 4),
        (4, 4, 4),
        (10, 10, 4),
        (2, 4, 11),
    ),
)
def test_derivative_3d(
    order1: int,
    btype1: BasisType,
    order2: int,
    btype2: BasisType,
    order3: int,
    btype3: BasisType,
) -> None:
    """Check that applying an incidence matrix is same as derivative method."""
    func_space = FunctionSpace(
        BasisSpecs(btype1, order1), BasisSpecs(btype2, order2), BasisSpecs(btype3, order3)
    )

    rng = np.random.default_rng(125)

    fn_dofs = DegreesOfFreedom(func_space)
    fn_dofs.values = rng.random(fn_dofs.shape)

    v: list[npt.NDArray[np.float64]]
    u: list[npt.NDArray[np.float64]]
    # First dimension
    incidence = incidence_matrix(func_space.basis_specs[0])
    v = list()
    for j in range(fn_dofs.shape[1]):
        u = list()
        for k in range(fn_dofs.shape[2]):
            u.append(incidence @ fn_dofs.values[:, j, k])
        v.append(np.stack(u, axis=1))
    expected = np.stack(v, axis=1)
    computed = fn_dofs.derivative(0).values
    assert pytest.approx(expected) == computed

    # Use applying function instead of matrix
    applied = incidence_operator(fn_dofs.values, func_space.basis_specs[0], axis=0)
    assert pytest.approx(expected) == applied

    # Second dimension
    incidence = incidence_matrix(func_space.basis_specs[1])
    v = list()
    for i in range(fn_dofs.shape[0]):
        u = list()
        for k in range(fn_dofs.shape[2]):
            u.append(incidence @ fn_dofs.values[i, :, k])
        v.append(np.stack(u, axis=1))
    expected = np.stack(v, axis=0)
    computed = fn_dofs.derivative(1).values
    assert pytest.approx(expected) == computed

    # Use applying function instead of matrix
    applied = incidence_operator(fn_dofs.values, func_space.basis_specs[1], axis=1)
    assert pytest.approx(expected) == applied

    # Third dimension
    incidence = incidence_matrix(func_space.basis_specs[2])
    v = list()
    for i in range(fn_dofs.shape[0]):
        u = list()
        for j in range(fn_dofs.shape[1]):
            u.append(incidence @ fn_dofs.values[i, j, :])
        v.append(np.stack(u, axis=0))
    expected = np.stack(v, axis=0)
    computed = fn_dofs.derivative(2).values
    assert pytest.approx(expected) == computed

    # Use applying function instead of matrix
    applied = incidence_operator(fn_dofs.values, func_space.basis_specs[2], axis=2)
    assert pytest.approx(expected) == applied


if __name__ == "__main__":
    test_derivative_1d(5, BasisType.BERNSTEIN)
