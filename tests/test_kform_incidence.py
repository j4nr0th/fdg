"""Check that k-form incidence matrix is correctly implemented."""

import numpy as np
import pytest
from interplib._interp import (
    BasisSpecs,
    DegreesOfFreedom,
    FunctionSpace,
    compute_kform_incidence_matrix,
)
from interplib.enum_type import BasisType


@pytest.mark.parametrize(
    ("o1", "o2", "b1", "b2"),
    (
        (1, 1, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
        (3, 4, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
        (4, 3, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
        (3, 2, BasisType.LEGENDRE, BasisType.LAGRNAGE_GAUSS),
        (3, 3, BasisType.LAGRANGE_UNIFORM, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
    ),
)
def test_2d_derivatives(o1: int, o2: int, b1: BasisType, b2: BasisType) -> None:
    """Check that incidence matrices for k-forms in 2D work as intended."""
    # Init PRNG
    rng = np.random.default_rng(o1 + 2 * o2 + hash(b1) ** 2 + hash(b2) ** 2)
    # Create base function space
    base_space = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2))

    # Test 0 -> 1 derivative in 2D
    ## Create the 0-form
    form_0 = DegreesOfFreedom(base_space)
    form_0.values = rng.random(form_0.shape)

    ## Take the derivatives
    form_0_d0 = form_0.derivative(idim=0)
    form_0_d1 = form_0.derivative(idim=1)

    ## Try also using the incidence matrix instead
    e01_mat = compute_kform_incidence_matrix(base_space, 0)
    dofs_joined_01 = e01_mat @ form_0.values.flatten()

    ## Check that compute values are correct
    assert pytest.approx(dofs_joined_01[: form_0_d0.n_dofs]) == form_0_d0.values.flatten()
    assert pytest.approx(dofs_joined_01[form_0_d0.n_dofs :]) == form_0_d1.values.flatten()

    # Test 1 -> 2 derivative in 2D
    ## Create the 1-forms
    form_1_0 = DegreesOfFreedom(base_space.lower_order(idim=0))
    form_1_1 = DegreesOfFreedom(base_space.lower_order(idim=1))
    form_1_0.values = rng.random(form_1_0.shape)
    form_1_1.values = rng.random(form_1_1.shape)

    ## Take the derivatives
    form_1_0d = form_1_0.derivative(idim=1)
    form_1_1d = form_1_1.derivative(idim=0)
    assert form_1_0d.function_space == form_1_1d.function_space
    form_2 = DegreesOfFreedom(
        form_1_0d.function_space, -form_1_0d.values + form_1_1d.values
    )

    ## Try also using the incidence matrix instead
    e12_mat = compute_kform_incidence_matrix(base_space, 1)
    dofs_2 = e12_mat @ np.concatenate(
        (form_1_0.values.flatten(), form_1_1.values.flatten())
    )

    ## Check that computed values are correct
    assert pytest.approx(dofs_2) == form_2.values.flatten()


if __name__ == "__main__":
    test_2d_derivatives(
        1, 1, BasisType.LAGRANGE_UNIFORM, BasisType.LAGRANGE_CHEBYSHEV_GAUSS
    )
    for i, args in enumerate(
        (
            (1, 1, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
            (3, 4, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
            (4, 3, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
            (3, 2, BasisType.LEGENDRE, BasisType.LEGENDRE),
            (3, 3, BasisType.LAGRANGE_UNIFORM, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
        )
    ):
        print("Running test case", i, end=None)
        test_2d_derivatives(*args)
        print("Finished")
