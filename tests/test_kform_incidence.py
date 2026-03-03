"""Check that k-form incidence matrix and operator are correctly implemented."""

import numpy as np
import pytest
from fdg._fdg import (
    BasisSpecs,
    DegreesOfFreedom,
    FunctionSpace,
    KFormSpecs,
    compute_kform_incidence_matrix,
    incidence_kform_operator,
)
from fdg.enum_type import BasisType

_TEST_VALUES_2D = (
    (1, 1, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (3, 4, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (4, 3, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (3, 2, BasisType.LEGENDRE, BasisType.LAGRNAGE_GAUSS),
    (3, 3, BasisType.LAGRANGE_UNIFORM, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
)

_TEST_VALUES_3D = (
    (1, 1, 1, BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (3, 4, 5, BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (4, 3, 5, BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (3, 2, 4, BasisType.LEGENDRE, BasisType.LAGRNAGE_GAUSS, BasisType.LEGENDRE),
    (
        3,
        3,
        3,
        BasisType.LAGRANGE_UNIFORM,
        BasisType.LAGRANGE_CHEBYSHEV_GAUSS,
        BasisType.LAGRNAGE_GAUSS_LOBATTO,
    ),
)


@pytest.mark.parametrize(("o1", "o2", "b1", "b2"), _TEST_VALUES_2D)
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


@pytest.mark.parametrize(("o1", "o2", "o3", "b1", "b2", "b3"), _TEST_VALUES_3D)
def test_3d_derivatives(
    o1: int, o2: int, o3: int, b1: BasisType, b2: BasisType, b3: BasisType
) -> None:
    """Check that incidence matrices for k-forms in 3D work as intended."""
    # Init PRNG
    rng = np.random.default_rng(
        o1 + 2 * o2 + 3 * o3 + hash(b1) ** 2 + hash(b2) ** 2 + hash(b3) ** 2
    )
    # Create base function space
    base_space = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2), BasisSpecs(b3, o3))

    # Test 0 -> 1 derivative in 3D
    ## Create the 0-form
    form_0 = DegreesOfFreedom(base_space)
    form_0.values = rng.random(form_0.shape)

    ## Take the derivatives
    form_0_d0 = form_0.derivative(idim=0)
    form_0_d1 = form_0.derivative(idim=1)
    form_0_d2 = form_0.derivative(idim=2)

    ## Try also using the incidence matrix instead
    e01_mat = compute_kform_incidence_matrix(base_space, 0)
    dofs_joined_01 = e01_mat @ form_0.values.flatten()

    ## Check that compute values are correct
    assert pytest.approx(dofs_joined_01[: form_0_d0.n_dofs]) == form_0_d0.values.flatten()
    assert (
        pytest.approx(
            dofs_joined_01[form_0_d0.n_dofs : form_0_d0.n_dofs + form_0_d1.n_dofs]
        )
        == form_0_d1.values.flatten()
    )
    assert (
        pytest.approx(dofs_joined_01[form_0_d0.n_dofs + form_0_d1.n_dofs :])
        == form_0_d2.values.flatten()
    )

    # Test 1 -> 2 derivative in 3D
    ## Create the 1-forms
    form_1_0 = DegreesOfFreedom(base_space.lower_order(idim=0))
    form_1_1 = DegreesOfFreedom(base_space.lower_order(idim=1))
    form_1_2 = DegreesOfFreedom(base_space.lower_order(idim=2))
    form_1_0.values = rng.random(form_1_0.shape)
    form_1_1.values = rng.random(form_1_1.shape)
    form_1_2.values = rng.random(form_1_2.shape)

    ## Take the derivatives
    form_1_0d1 = form_1_0.derivative(idim=1)  # dy ^ dx
    form_1_0d2 = form_1_0.derivative(idim=2)  # dz ^ dx
    form_1_1d0 = form_1_1.derivative(idim=0)  # dx ^ dy
    form_1_1d2 = form_1_1.derivative(idim=2)  # dz ^ dy
    form_1_2d0 = form_1_2.derivative(idim=0)  # dx ^ dz
    form_1_2d1 = form_1_2.derivative(idim=1)  # dy ^ dz

    assert form_1_1d0.function_space == form_1_0d1.function_space
    form_1_d0 = DegreesOfFreedom(
        form_1_1d0.function_space, form_1_1d0.values - form_1_0d1.values
    )
    assert form_1_2d0.function_space == form_1_0d2.function_space
    form_1_d1 = DegreesOfFreedom(
        form_1_2d0.function_space, form_1_2d0.values - form_1_0d2.values
    )
    assert form_1_2d1.function_space == form_1_1d2.function_space
    form_1_d2 = DegreesOfFreedom(
        form_1_2d1.function_space, form_1_2d1.values - form_1_1d2.values
    )

    ## Try also using the incidence matrix instead
    e12_mat = compute_kform_incidence_matrix(base_space, 1)
    dofs_2 = e12_mat @ np.concatenate(
        (form_1_0.values.flatten(), form_1_1.values.flatten(), form_1_2.values.flatten())
    )

    ## Check that computed values are correct
    assert pytest.approx(dofs_2[: form_1_d0.n_dofs]) == form_1_d0.values.flatten()
    assert (
        pytest.approx(dofs_2[form_1_d0.n_dofs : form_1_d0.n_dofs + form_1_d1.n_dofs])
        == form_1_d1.values.flatten()
    )
    assert (
        pytest.approx(dofs_2[form_1_d0.n_dofs + form_1_d1.n_dofs :])
        == form_1_d2.values.flatten()
    )

    # Test 2 -> 3 derivative in 3D
    ## Create the 2-forms
    form_2_0 = DegreesOfFreedom(base_space.lower_order(idim=0).lower_order(idim=1))
    form_2_1 = DegreesOfFreedom(base_space.lower_order(idim=0).lower_order(idim=2))
    form_2_2 = DegreesOfFreedom(base_space.lower_order(idim=1).lower_order(idim=2))
    form_2_0.values = rng.random(form_2_0.shape)
    form_2_1.values = rng.random(form_2_1.shape)
    form_2_2.values = rng.random(form_2_2.shape)

    ## Take the derivatives
    form_2_0d2 = form_2_0.derivative(idim=2)  # dz ^ dx ^ dy (+)
    form_2_1d1 = form_2_1.derivative(idim=1)  # dy ^ dx ^ dz (-)
    form_2_2d0 = form_2_2.derivative(idim=0)  # dx ^ dy ^ dz (+)

    assert (
        form_2_0d2.function_space == form_2_1d1.function_space
        and form_2_1d1.function_space == form_2_2d0.function_space
    )
    form_2_d = DegreesOfFreedom(
        form_2_0d2.function_space,
        form_2_0d2.values - form_2_1d1.values + form_2_2d0.values,
    )

    ## Try also using the incidence matrix instead
    e23_mat = compute_kform_incidence_matrix(base_space, 2)
    dofs_3 = e23_mat @ np.concatenate(
        (form_2_0.values.flatten(), form_2_1.values.flatten(), form_2_2.values.flatten())
    )

    ## Check that computed values are correct
    assert pytest.approx(dofs_3) == form_2_d.values.flatten()


@pytest.mark.parametrize(("o1", "o2", "o3", "b1", "b2", "b3"), _TEST_VALUES_3D)
def test_3d_operator(
    o1: int, o2: int, o3: int, b1: BasisType, b2: BasisType, b3: BasisType
) -> None:
    """Check that operator matrices for k-forms in 3D work as intended."""
    # Init PRNG
    rng = np.random.default_rng(
        o1 + 2 * o2 + 3 * o3 + hash(b1) ** 2 + hash(b2) ** 2 + hash(b3) ** 2
    )
    # Create base function space
    base_space = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2), BasisSpecs(b3, o3))

    # Test 0 -> 1 derivative in 3D
    ## Create the 0-form
    specs_0 = KFormSpecs(0, base_space)
    form_0 = DegreesOfFreedom(base_space)
    form_0.values = rng.random(form_0.shape)

    ## Use the incidence matrix first
    e01_mat = compute_kform_incidence_matrix(base_space, 0)
    dofs_joined_01 = e01_mat @ form_0.values.flatten()

    # Try using the operator instead
    dofs_computed_01 = incidence_kform_operator(specs_0, form_0.values.flatten())

    ## Check that compute values are correct
    assert pytest.approx(dofs_joined_01) == dofs_computed_01

    # Test 1 -> 2 derivative in 3D
    ## Create the 1-forms
    specs_1 = KFormSpecs(1, base_space)
    form_1_0 = DegreesOfFreedom(base_space.lower_order(idim=0))
    form_1_1 = DegreesOfFreedom(base_space.lower_order(idim=1))
    form_1_2 = DegreesOfFreedom(base_space.lower_order(idim=2))
    form_1_0.values = rng.random(form_1_0.shape)
    form_1_1.values = rng.random(form_1_1.shape)
    form_1_2.values = rng.random(form_1_2.shape)

    ## Use the incidence matrix
    e12_mat = compute_kform_incidence_matrix(base_space, 1)
    dofs_1_flattened = np.concatenate(
        (form_1_0.values.flatten(), form_1_1.values.flatten(), form_1_2.values.flatten())
    )
    dofs_2 = e12_mat @ dofs_1_flattened

    # Try using the operator instead
    dofs_computed_12 = incidence_kform_operator(specs_1, dofs_1_flattened)

    ## Check that computed values are correct
    assert pytest.approx(dofs_2) == dofs_computed_12

    # Test 2 -> 3 derivative in 3D
    ## Create the 2-forms
    specs_2 = KFormSpecs(2, base_space)
    form_2_0 = DegreesOfFreedom(base_space.lower_order(idim=0).lower_order(idim=1))
    form_2_1 = DegreesOfFreedom(base_space.lower_order(idim=0).lower_order(idim=2))
    form_2_2 = DegreesOfFreedom(base_space.lower_order(idim=1).lower_order(idim=2))
    form_2_0.values = rng.random(form_2_0.shape)
    form_2_1.values = rng.random(form_2_1.shape)
    form_2_2.values = rng.random(form_2_2.shape)

    ## Use the incidence matrix
    e23_mat = compute_kform_incidence_matrix(base_space, 2)
    dofs_2_flattened = np.concatenate(
        (form_2_0.values.flatten(), form_2_1.values.flatten(), form_2_2.values.flatten())
    )
    dofs_3 = e23_mat @ dofs_2_flattened

    # Try using the operator instead
    dofs_computed_23 = incidence_kform_operator(specs_2, dofs_2_flattened)

    ## Check that computed values are correct
    assert pytest.approx(dofs_3) == dofs_computed_23


def _test_incidence_operators(specs_k: KFormSpecs, k: int) -> None:
    """Check that the k-form incidence operators work correctly."""
    # First assemble the incidence matrix
    e_mat = compute_kform_incidence_matrix(specs_k.base_space, k)

    eye_low = np.eye(e_mat.shape[1])
    eye_high = np.eye(e_mat.shape[0])

    # Check that applying it also works
    e = incidence_kform_operator(specs_k, eye_low, transpose=False, right=False)
    assert pytest.approx(e) == e_mat
    # print("C1")

    # Check that applying transpose also works
    et = incidence_kform_operator(specs_k, eye_high, transpose=True, right=False)
    assert pytest.approx(et) == e_mat.T
    # print("C2")

    # Check that right applying transpose also works
    ert = incidence_kform_operator(specs_k, eye_low, transpose=True, right=True)
    assert pytest.approx(ert) == e_mat.T
    # print("C3")

    # Check that right applying it also works
    er = incidence_kform_operator(specs_k, eye_high, transpose=False, right=True)
    assert pytest.approx(er) == e_mat
    # print("C4")


@pytest.mark.parametrize("o1", (1, 2, 4))
@pytest.mark.parametrize("b1", BasisType)
def test_1d_operator_matrix(o1: int, b1: BasisType) -> None:
    """Check that operator matrices for k-forms in 2D work as intended."""
    # Create base function space
    base_space = FunctionSpace(BasisSpecs(b1, o1))

    # Run through all k-forms until 2-form
    for k in range(0, 1):
        specs_k = KFormSpecs(k, base_space)
        _test_incidence_operators(specs_k, k)


@pytest.mark.parametrize(("o1", "o2", "b1", "b2"), _TEST_VALUES_2D)
def test_2d_operator_matrix(o1: int, o2: int, b1: BasisType, b2: BasisType) -> None:
    """Check that operator matrices for k-forms in 2D work as intended."""
    # Create base function space
    base_space = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2))

    # Run through all k-forms until 2-form
    for k in range(0, 2):
        specs_k = KFormSpecs(k, base_space)
        _test_incidence_operators(specs_k, k)


@pytest.mark.parametrize(("o1", "o2", "o3", "b1", "b2", "b3"), _TEST_VALUES_3D)
def test_3d_operator_matrix(
    o1: int, o2: int, o3: int, b1: BasisType, b2: BasisType, b3: BasisType
) -> None:
    """Check that operator matrices for k-forms in 3D work as intended."""
    # Create base function space
    base_space = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2), BasisSpecs(b3, o3))

    # Run through all k-forms until 3-form
    for k in range(0, 3):
        specs_k = KFormSpecs(k, base_space)
        _test_incidence_operators(specs_k, k)


if __name__ == "__main__":
    test_3d_operator_matrix(
        o1=2,
        o2=2,
        o3=2,
        b1=BasisType.LAGRANGE_UNIFORM,  # b1=BasisType.LEGENDRE,
        b2=BasisType.BERNSTEIN,  # b2=BasisType.LAGRNAGE_GAUSS,
        b3=BasisType.BERNSTEIN,  # b3=BasisType.LEGENDRE,
    )

    for o1 in (1, 2, 4):
        for b1 in BasisType:
            test_1d_operator_matrix(o1, b1)

    for args in _TEST_VALUES_2D:
        test_2d_operator_matrix(*args)

    test_3d_operator_matrix(
        3, 2, 4, BasisType.LAGRNAGE_GAUSS, BasisType.LEGENDRE, BasisType.LEGENDRE
    )
    for args3 in _TEST_VALUES_3D:
        test_3d_operator_matrix(*args3)
