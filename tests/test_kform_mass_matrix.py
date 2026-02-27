"""Real k-form mass matrix this time."""

from itertools import combinations

import numpy as np
import numpy.typing as npt
import pytest
from fdg._fdg import (
    BasisSpecs,
    CovectorBasis,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    SpaceMap,
    compute_kform_mass_matrix,
)
from fdg.domains import Line, Quad
from fdg.enum_type import BasisType


def component_inner_prod_mass_matrix_block(
    basis_left: CovectorBasis,
    basis_right: CovectorBasis,
    fn_space_left: FunctionSpace,
    fn_space_right: FunctionSpace,
    smap: SpaceMap,
) -> npt.NDArray[np.double]:
    """Compute an inner product mass matrix."""
    n = basis_left.ndim
    if n != basis_right.ndim:
        raise ValueError("Dimension basis must be equal.")
    if fn_space_right.dimension != n:
        raise ValueError("Input function space must have the same dimension basis.")
    if fn_space_left.dimension != n:
        raise ValueError("Outpu function space must have the same dimension basis.")
    if smap.input_dimensions != n:
        raise ValueError("Space map must have the same dimension basis.")

    k = basis_left.rank
    if k != basis_right.rank:
        raise ValueError("Rank of the basis must be equal.")

    # Prepare integration weights
    weights = smap.integration_space.weights()
    if k == 0:
        weights *= smap.determinant
    elif k == n:
        weights /= smap.determinant
    else:
        transformation_matrix = smap.basis_transform(k)

        weights_1 = transformation_matrix[basis_left.index, :, :]
        weights_2 = transformation_matrix[basis_right.index, :, :]
        tw = np.reshape(np.sum(weights_1 * weights_2, axis=0), weights.shape)
        weights *= smap.determinant * tw

    # Babe it's 4 p.m., time for your array flattening
    weights = weights.flatten()

    # Prepare function spaces
    fn_left = fn_space_left
    for i in range(n):
        if i in basis_left:
            fn_left = fn_left.lower_order(i)

    fn_right = fn_space_right
    for i in range(n):
        if i in basis_right:
            fn_right = fn_right.lower_order(i)

    # Get the basis
    bv_left = np.reshape(
        fn_left.values_at_integration_nodes(smap.integration_space), (weights.size, -1)
    )
    bv_right = np.reshape(
        fn_right.values_at_integration_nodes(smap.integration_space), (weights.size, -1)
    )

    # Compute the mass matrix in a big fat multiplication and sum
    return np.sum(
        bv_left[:, :, None] * bv_right[:, None, :] * weights[:, None, None], axis=0
    )


def iterate_kform_components(ndim: int, order: int):
    """Iterate over all components of a k-form.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    order : int
        Order of the k-form.

    Return
    ------
    Generator
        Generator which produces the basis of all k-form components.
    """
    for comb in combinations(range(ndim), order):
        yield CovectorBasis(ndim, *comb)


def compute_inner_prod_mass_matrix(
    smap: SpaceMap,
    order: int,
    fn_space_left: FunctionSpace,
    fn_space_right: FunctionSpace,
):
    """Compute the full inner-product mass matrix for a k-form."""
    blocks: list[list[npt.NDArray[np.double]]] = list()
    ndim = smap.input_dimensions

    for basis_left in iterate_kform_components(ndim, order):
        row: list[npt.NDArray[np.double]] = list()
        for basis_right in iterate_kform_components(ndim, order):
            row.append(
                component_inner_prod_mass_matrix_block(
                    basis_left, basis_right, fn_space_left, fn_space_right, smap
                )
            )
        blocks.append(row)
        del row

    return np.block(blocks)


def check_matrix_correctness(
    space_map: SpaceMap,
    order: int,
    fn_space_left: FunctionSpace,
    fn_space_right: FunctionSpace,
) -> None:
    """Assert the mass matrix is correctly computed for input parameters."""
    mat_computed = compute_kform_mass_matrix(
        space_map, order, fn_space_left, fn_space_right
    )
    mat_expected = compute_inner_prod_mass_matrix(
        space_map, order, fn_space_left, fn_space_right
    )
    assert (fn_space_left != fn_space_right) or pytest.approx(
        mat_computed
    ) == mat_computed.T
    assert pytest.approx(mat_expected) == mat_computed


@pytest.mark.parametrize(
    ("order_1", "order_2", "order_left", "order_right", "btype_1", "btype_2", "m"),
    (
        (6, 7, 2, 3, BasisType.BERNSTEIN, BasisType.BERNSTEIN, 1),
        (4, 5, 3, 3, BasisType.LEGENDRE, BasisType.LEGENDRE, 2),
        (4, 5, 2, 1, BasisType.BERNSTEIN, BasisType.LEGENDRE, 4),
        (6, 8, 6, 7, BasisType.LAGRANGE_UNIFORM, BasisType.LAGRNAGE_GAUSS, 5),
    ),
)
def test_1d_to_md(
    order_1: int,
    order_2: int,
    order_left: int,
    order_right: int,
    btype_1: BasisType,
    btype_2: BasisType,
    m: int,
) -> None:
    """Check that 1D to mD inner product is correct."""
    assert order_1 > 0 and order_2 > 0 and m > 0
    rng = np.random.default_rng(order_1**2 + 2 * order_2**2 + m)
    line = Line(
        *(
            0.1 * rng.random((order_1, m))
            + np.stack(m * [np.linspace(-1, +1, order_1)], axis=-1)
        )
    )

    fn_space_left = FunctionSpace(BasisSpecs(btype_1, order_left))
    fn_space_right = FunctionSpace(BasisSpecs(btype_2, order_right))

    int_space = IntegrationSpace(IntegrationSpecs(order_2))
    space_map = line(int_space)

    check_matrix_correctness(space_map, 0, fn_space_left, fn_space_right)
    check_matrix_correctness(space_map, 1, fn_space_left, fn_space_right)


@pytest.mark.parametrize(
    (
        "pts_h",
        "pts_v",
        "order_i1",
        "order_i2",
        "obl1",
        "obl2",
        "btl1",
        "btl2",
        "obr1",
        "obr2",
        "btr1",
        "btr2",
    ),
    (
        (
            2,
            3,
            6,
            6,
            2,
            3,
            BasisType.BERNSTEIN,
            BasisType.BERNSTEIN,
            3,
            4,
            BasisType.BERNSTEIN,
            BasisType.BERNSTEIN,
        ),
        (
            4,
            3,
            5,
            6,
            4,
            4,
            BasisType.LEGENDRE,
            BasisType.BERNSTEIN,
            4,
            4,
            BasisType.LEGENDRE,
            BasisType.BERNSTEIN,
        ),
    ),
)
def test_2d_to_2d(
    pts_h: int,
    pts_v: int,
    order_i1: int,
    order_i2: int,
    obl1: int,
    obl2: int,
    btl1: BasisType,
    btl2: BasisType,
    obr1: int,
    obr2: int,
    btr1: BasisType,
    btr2: BasisType,
) -> None:
    """Check that 2D to 2D mapping is correct."""
    assert pts_h > 1 and pts_v > 2 and order_i1 > 0 and order_i2 > 0
    rng = np.random.default_rng(pts_h**2 + 2 * pts_v**3 + order_i1 + 2 * order_i2 + 1)

    def perturbe_linspace(start, stop, nstep):
        """Perturbe linspace function a bit."""
        res = np.linspace(start, stop, nstep)
        res[1:-1] += rng.random(nstep - 2) * (stop - start) / (nstep - 1)
        return res

    quad = Quad(
        bottom=Line(*np.array((perturbe_linspace(-1, +1, pts_h), np.full(pts_h, -1))).T),
        right=Line(*np.array((np.full(pts_v, +1), perturbe_linspace(-1, +1, pts_v))).T),
        top=Line(*np.array((perturbe_linspace(+1, -1, pts_h), np.full(pts_h, +1))).T),
        left=Line(*np.array((np.full(pts_v, -1), perturbe_linspace(+1, -1, pts_v))).T),
    )

    fn_space_left = FunctionSpace(BasisSpecs(btl1, obl1), BasisSpecs(btl2, obl2))
    fn_space_right = FunctionSpace(BasisSpecs(btr1, obr1), BasisSpecs(btr2, obr2))

    int_space = IntegrationSpace(IntegrationSpecs(order_i1), IntegrationSpecs(order_i2))
    space_map = quad(int_space)

    check_matrix_correctness(space_map, 0, fn_space_left, fn_space_right)
    check_matrix_correctness(space_map, 2, fn_space_left, fn_space_right)
    check_matrix_correctness(space_map, 1, fn_space_left, fn_space_right)


@pytest.mark.parametrize(
    (
        "pts_h",
        "pts_v",
        "order_i1",
        "order_i2",
        "obl1",
        "obl2",
        "btl1",
        "btl2",
        "obr1",
        "obr2",
        "btr1",
        "btr2",
    ),
    (
        (
            2,
            3,
            6,
            6,
            2,
            3,
            BasisType.BERNSTEIN,
            BasisType.BERNSTEIN,
            3,
            4,
            BasisType.BERNSTEIN,
            BasisType.BERNSTEIN,
        ),
        (
            4,
            3,
            5,
            6,
            4,
            4,
            BasisType.LEGENDRE,
            BasisType.BERNSTEIN,
            4,
            4,
            BasisType.LEGENDRE,
            BasisType.BERNSTEIN,
        ),
    ),
)
def test_2d_to_3d(
    pts_h: int,
    pts_v: int,
    order_i1: int,
    order_i2: int,
    obl1: int,
    obl2: int,
    btl1: BasisType,
    btl2: BasisType,
    obr1: int,
    obr2: int,
    btr1: BasisType,
    btr2: BasisType,
) -> None:
    """Check that 2D to 3D mapping is correct."""
    assert pts_h > 1 and pts_v > 2 and order_i1 > 0 and order_i2 > 0
    rng = np.random.default_rng(pts_h**2 + 2 * pts_v**3 + order_i1 + 2 * order_i2 + 1)

    def perturbe_linspace(start, stop, nstep):
        """Perturbe linspace function a bit."""
        res = np.linspace(start, stop, nstep)
        res[1:-1] += rng.random(nstep - 2) * (stop - start) / (nstep - 1)
        return res

    quad = Quad(
        bottom=Line(
            *np.array(
                (
                    perturbe_linspace(-1, +1, pts_h),
                    np.full(pts_h, -1),
                    perturbe_linspace(-1, +1, pts_h),
                )
            ).T
        ),
        right=Line(
            *np.array(
                (
                    np.full(pts_v, +1),
                    perturbe_linspace(-1, +1, pts_v),
                    perturbe_linspace(+1, -1, pts_v),
                )
            ).T
        ),
        top=Line(
            *np.array(
                (
                    perturbe_linspace(+1, -1, pts_h),
                    np.full(pts_h, +1),
                    perturbe_linspace(-1, +1, pts_h),
                )
            ).T
        ),
        left=Line(
            *np.array(
                (
                    np.full(pts_v, -1),
                    perturbe_linspace(+1, -1, pts_v),
                    perturbe_linspace(+1, -1, pts_v),
                )
            ).T
        ),
    )

    fn_space_left = FunctionSpace(BasisSpecs(btl1, obl1), BasisSpecs(btl2, obl2))
    fn_space_right = FunctionSpace(BasisSpecs(btr1, obr1), BasisSpecs(btr2, obr2))

    int_space = IntegrationSpace(IntegrationSpecs(order_i1), IntegrationSpecs(order_i2))
    space_map = quad(int_space)

    check_matrix_correctness(space_map, 0, fn_space_left, fn_space_right)
    check_matrix_correctness(space_map, 2, fn_space_left, fn_space_right)
    check_matrix_correctness(space_map, 1, fn_space_left, fn_space_right)
