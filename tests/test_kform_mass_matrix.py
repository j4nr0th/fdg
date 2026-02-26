"""Real k-form mass matrix this time."""

import numpy as np
import pytest
from interplib._interp import (
    BasisSpecs,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    SpaceMap,
    compute_kform_mass_matrix,
)
from interplib.domains import Line, Quad
from interplib.enum_type import BasisType
from interplib.mass_matrix import compute_inner_prod_mass_matrix


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
