"""Chech that function spaces work as expected."""

import numpy as np
import pytest
from fdg._fdg import (
    BasisSpecs,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
)


def test_1d_function_space():
    """Test creation of a 1D function space."""
    basis_specs = BasisSpecs("legendre", 3)
    fs = FunctionSpace(basis_specs)

    # Check properties
    assert fs.basis_specs == (basis_specs,)
    assert fs.dimension == 1
    assert fs.orders == (basis_specs.order,)

    # Check evaluation at a points
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, size=(3, 4, 2))

    values_fs = fs.evaluate(x)
    values_bs = basis_specs.values(x)
    # These should match exactly
    assert np.all(values_fs == values_bs)


def test_2d_function_space():
    """Test creation of a 2D function space."""
    basis_specs1 = BasisSpecs("legendre", 2)
    basis_specs2 = BasisSpecs("bernstein", 3)
    fs = FunctionSpace(basis_specs1, basis_specs2)

    # Check properties
    assert fs.basis_specs == (basis_specs1, basis_specs2)
    assert fs.dimension == 2
    assert fs.orders == (basis_specs1.order, basis_specs2.order)

    # Check evaluation at a points
    rng = np.random.default_rng(42)
    x1 = rng.uniform(-1, 1, size=(3, 4, 2))
    x2 = rng.uniform(-1, 1, size=(3, 4, 2))

    values_fs = fs.evaluate(x1, x2)
    values_bs1 = basis_specs1.values(x1)
    values_bs2 = basis_specs2.values(x2)
    # These should match exactly
    expected_values = values_bs2[..., None, :] * values_bs1[..., :, None]
    assert np.all(values_fs == expected_values)


def test_3d_function_space():
    """Test creation of a 3D function space."""
    basis_specs1 = BasisSpecs("legendre", 4)
    basis_specs2 = BasisSpecs("legendre", 3)
    basis_specs3 = BasisSpecs("legendre", 6)
    fs = FunctionSpace(basis_specs1, basis_specs2, basis_specs3)

    # Check properties
    assert fs.basis_specs == (basis_specs1, basis_specs2, basis_specs3)
    assert fs.dimension == 3
    assert fs.orders == (basis_specs1.order, basis_specs2.order, basis_specs3.order)

    # Check evaluation at a points
    rng = np.random.default_rng(42)
    x1 = rng.uniform(-1, 1, size=(2, 3, 3, 4, 2))
    x2 = rng.uniform(-1, 1, size=(2, 3, 3, 4, 2))
    x3 = rng.uniform(-1, 1, size=(2, 3, 3, 4, 2))

    values_fs = fs.evaluate(x1, x2, x3)
    values_bs1 = basis_specs1.values(x1)
    values_bs2 = basis_specs2.values(x2)
    values_bs3 = basis_specs3.values(x3)
    expected_values = (
        values_bs3[..., None, None, :]
        * values_bs2[..., None, :, None]
        * values_bs1[..., :, None, None]
    )

    # These may no longer match exactly due to floating point errors
    # since now order of multiplications may differ between C and Python.
    assert values_fs == pytest.approx(expected_values)


def test_function_space_boundary() -> None:
    """Check that a boundary space drops the selected reference dimension."""
    specs = (
        BasisSpecs("legendre", 2),
        BasisSpecs("bernstein", 3),
        BasisSpecs("lagrange-uniform", 4),
    )
    function_space = FunctionSpace(*specs)

    expected = (specs[1:], (specs[0], specs[2]), specs[:2])
    for idim, expected_specs in enumerate(expected):
        face_space = function_space.boundary(idim)
        assert face_space.basis_specs == expected_specs
        assert face_space.dimension == function_space.dimension - 1

    point_space = FunctionSpace(specs[0]).boundary(0)
    assert point_space.dimension == 0
    assert point_space.basis_specs == ()

    with pytest.raises(ValueError, match="out of bounds"):
        function_space.boundary(-1)
    with pytest.raises(ValueError, match="out of bounds"):
        function_space.boundary(function_space.dimension)
    with pytest.raises(ValueError, match="zero-dimensional"):
        FunctionSpace().boundary(0)


def test_integration_points() -> None:
    """Check that integration point values are same as those computed by evaluate."""
    basis_specs1 = BasisSpecs("legendre", 4)
    basis_specs2 = BasisSpecs("bernstein", 3)
    basis_specs3 = BasisSpecs("lagrange-uniform", 6)
    fs = FunctionSpace(basis_specs1, basis_specs2, basis_specs3)

    int_space = IntegrationSpace(
        IntegrationSpecs(5, "gauss"),
        IntegrationSpecs(4, "gauss-lobatto"),
        IntegrationSpecs(8, "gauss"),
    )

    nodes = int_space.nodes()
    expected_values = fs.evaluate(*(nodes[i] for i in range(nodes.shape[0])))
    computed_values = fs.values_at_integration_nodes(int_space)

    assert expected_values == pytest.approx(computed_values)


def test_bernstein_order_zero_derivatives_are_rejected():
    """Bernstein derivatives evaluate one order below, so order 0 would underflow."""
    with pytest.raises(ValueError, match="order of at least 1"):
        BasisSpecs("bernstein", 0).derivatives(np.zeros(2))


def test_order_zero_derivatives_remain_supported():
    """Order-0 derivatives of the other basis families keep working."""
    x = np.array([-0.5, 0.5])
    for basis_type in ("legendre", "lagrange-uniform"):
        np.testing.assert_array_equal(
            BasisSpecs(basis_type, 0).derivatives(x), np.zeros((2, 1))
        )


def test_empty_evaluation_points_are_rejected():
    """An empty point array underflows n_pos - 1 in the Lagrange evaluator."""
    empty = np.zeros(0)
    for basis_type in ("legendre", "bernstein", "lagrange-uniform"):
        specs = BasisSpecs(basis_type, 1)
        with pytest.raises(ValueError, match="non-empty"):
            specs.values(empty)
        with pytest.raises(ValueError, match="non-empty"):
            specs.derivatives(empty)


def test_dimension_upper_bound():
    """The option storage takes ndim of at most 63 as a precondition."""
    with pytest.raises(ValueError, match="at most 63"):
        FunctionSpace(*(BasisSpecs("legendre", 1) for _ in range(64)))


if __name__ == "__main__":
    test_1d_function_space()
    test_2d_function_space()
    test_3d_function_space()
    test_integration_points()
