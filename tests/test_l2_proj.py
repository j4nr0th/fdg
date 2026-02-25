"""Check that L2 projection works correctly."""

from functools import partial

import numpy as np
import pytest
from interplib._interp import (
    BasisSpecs,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
)
from interplib.degrees_of_freedom import reconstruct
from interplib.domains import Quad
from interplib.enum_type import BasisType
from interplib.integration import projection_kform_l2_primal, projection_l2_primal


@pytest.mark.parametrize("order", (1, 2, 4, 10))
@pytest.mark.parametrize("btype", BasisType)
def test_projection_1d(order: int, btype: BasisType) -> None:
    """Check that projection of DoFs to the same space is identity."""
    fs = FunctionSpace(BasisSpecs(btype, order))

    dofs = DegreesOfFreedom(fs)

    rng = np.random.default_rng(129)
    dofs.values = rng.random(dofs.shape)

    int_space = IntegrationSpace(IntegrationSpecs(order + 2, method="gauss"))

    proj = projection_l2_primal(partial(reconstruct, dofs), fs, int_space)

    assert proj.values == pytest.approx(dofs.values)


@pytest.mark.parametrize(("o1", "o2"), ((1, 1), (2, 4), (10, 4)))
@pytest.mark.parametrize(
    ("b1", "b2"),
    (
        (BasisType.BERNSTEIN, BasisType.BERNSTEIN),
        (BasisType.LEGENDRE, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
    ),
)
def test_projection_2d(o1: int, b1: BasisType, o2: int, b2: BasisType) -> None:
    """Check that projection of DoFs to the same space is identity."""
    fs = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2))

    dofs = DegreesOfFreedom(fs)

    rng = np.random.default_rng(129)
    dofs.values = rng.random(dofs.shape)

    int_space = IntegrationSpace(
        IntegrationSpecs(o1 + 2, method="gauss"), IntegrationSpecs(o2 + 2, method="gauss")
    )

    proj = projection_l2_primal(partial(reconstruct, dofs), fs, int_space)

    assert proj.values == pytest.approx(dofs.values)


@pytest.mark.parametrize(("o1", "o2", "o3"), ((1, 1, 1), (2, 4, 10), (10, 4, 3)))
@pytest.mark.parametrize(
    ("b1", "b2", "b3"),
    (
        (BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.LAGRNAGE_GAUSS),
        (BasisType.LEGENDRE, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, BasisType.BERNSTEIN),
    ),
)
def test_projection_3d(
    o1: int, b1: BasisType, o2: int, b2: BasisType, o3: int, b3: BasisType
) -> None:
    """Check that projection of DoFs to the same space is identity."""
    fs = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2), BasisSpecs(b3, o3))

    dofs = DegreesOfFreedom(fs)

    rng = np.random.default_rng(129)
    dofs.values = rng.random(dofs.shape)

    int_space = IntegrationSpace(
        IntegrationSpecs(o1 + 2, method="gauss"),
        IntegrationSpecs(o2 + 2, method="gauss"),
        IntegrationSpecs(o3 + 2, method="gauss"),
    )

    proj = projection_l2_primal(partial(reconstruct, dofs), fs, int_space)

    assert proj.values == pytest.approx(dofs.values)


def test_deformed_2d_to_3d() -> None:
    """Check that projection of DoFs in deformed space works as expected."""

    def test_function(*args):
        x, y, z = args
        return x**2 + y * z - 1

    fs = FunctionSpace(
        BasisSpecs("bernstein", 3),
        BasisSpecs("bernstein", 3),
    )

    domain = Quad.from_corners(
        (-2, -1, -1),
        (+1.5, -1, +1),
        (+1, +1, +1),
        (-1, +1, -1),
    )

    int_space = IntegrationSpace(
        IntegrationSpecs(5, method="gauss"),
        IntegrationSpecs(5, method="gauss"),
    )
    smap = domain(int_space)

    proj = projection_l2_primal(test_function, fs, smap)

    r1, r2 = np.meshgrid(np.linspace(-1, +1, 11), np.linspace(-1, +1, 11))
    tx, ty, tz = domain.sample(r1, r2)

    assert test_function(tx, ty, tz) == pytest.approx(reconstruct(proj, r1, r2))


@pytest.mark.parametrize(("o1", "o2", "o3"), ((1, 1, 1), (2, 4, 6), (6, 4, 3)))
@pytest.mark.parametrize(
    ("b1", "b2", "b3"),
    (
        (BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.LAGRNAGE_GAUSS),
        (BasisType.LEGENDRE, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, BasisType.BERNSTEIN),
    ),
)
def test_projection_kform_3d(
    o1: int, b1: BasisType, o2: int, b2: BasisType, o3: int, b3: BasisType
) -> None:
    """Check that projection of DoFs to the same space is identity."""
    base_space = FunctionSpace(BasisSpecs(b1, o1), BasisSpecs(b2, o2), BasisSpecs(b3, o3))
    rng = np.random.default_rng(129)
    int_space = IntegrationSpace(
        IntegrationSpecs(o1 + 2, method="gauss"),
        IntegrationSpecs(o2 + 2, method="gauss"),
        IntegrationSpecs(o3 + 2, method="gauss"),
    )

    for k in range(0, 4):
        specs = KFormSpecs(k, base_space)
        components = [
            DegreesOfFreedom(specs.get_component_function_space(idx))
            for idx in range(specs.component_count)
        ]
        for c in components:
            c.values[:] = rng.random(c.shape)

        functions = [partial(reconstruct, c) for c in components]

        proj = projection_kform_l2_primal(functions, specs, int_space)

        for pv, c in zip(proj, components, strict=True):
            assert pytest.approx(pv) == c.values


if __name__ == "__main__":
    for o1, o2, o3 in ((1, 1, 1), (2, 4, 6), (6, 4, 3)):
        for b1, b2, b3 in (
            (BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.LAGRNAGE_GAUSS),
            (BasisType.LEGENDRE, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, BasisType.BERNSTEIN),
        ):
            test_projection_kform_3d(o1, b1, o2, b2, o3, b3)
