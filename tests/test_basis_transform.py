"""Test basis transformation computation that is further used in inner-products."""

import numpy as np
import pytest
from fdg._fdg import IntegrationSpace, IntegrationSpecs
from fdg.domains import Line, Quad


@pytest.mark.parametrize(
    ("order_1", "order_2", "m"),
    (
        (6, 7, 1),
        (4, 5, 2),
        (4, 5, 4),
        (6, 8, 5),
    ),
)
def test_1d_to_md(order_1: int, order_2: int, m: int) -> None:
    """Check that 1D to mD mapping is correct."""
    assert order_1 > 0 and order_2 > 0 and m > 0
    rng = np.random.default_rng(order_1**2 + 2 * order_2**2 + m)
    line = Line(
        *(
            0.1 * rng.random((order_1, m))
            + np.stack(m * [np.linspace(-1, +1, order_1)], axis=-1)
        )
    )

    int_space = IntegrationSpace(IntegrationSpecs(order_2))
    space_map = line(int_space)

    with pytest.raises(ValueError):
        space_map.basis_transform(0)  # Raises Value error

    transformation = space_map.basis_transform(1)  # Only 1-form
    # Shape is based on (in_basis, out_basis, int_pts)
    assert transformation.shape == (1, m, order_2 + 1)
    # Compute dx_i/dt
    derivatives = [
        dof.reconstruct_derivative_at_integration_points(int_space, (0,))
        for dof in line.dofs
    ]
    mag = sum(d**2 for d in derivatives)
    for i, d in enumerate(derivatives):
        assert (
            pytest.approx(d / mag) == transformation[0, i, :]
        )  # For the 1D -> MD case it's a bit more involved


@pytest.mark.parametrize(
    ("pts_h", "pts_v", "order_i1", "order_i2"),
    (
        (2, 3, 1, 2),
        (4, 5, 2, 1),
        (3, 3, 4, 4),
        (5, 5, 5, 4),
    ),
)
def test_2d_to_2d(pts_h: int, pts_v: int, order_i1: int, order_i2) -> None:
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

    int_space = IntegrationSpace(IntegrationSpecs(order_i1), IntegrationSpecs(order_i2))
    space_map = quad(int_space)

    with pytest.raises(ValueError):
        space_map.basis_transform(0)  # Raises Value error

    # Forward derivatives
    dx1dxi1 = (
        quad.dofs[0]
        .reconstruct_derivative_at_integration_points(int_space, (0,))
        .flatten()
    )
    dx1dxi2 = (
        quad.dofs[0]
        .reconstruct_derivative_at_integration_points(int_space, (1,))
        .flatten()
    )
    dx2dxi1 = (
        quad.dofs[1]
        .reconstruct_derivative_at_integration_points(int_space, (0,))
        .flatten()
    )
    dx2dxi2 = (
        quad.dofs[1]
        .reconstruct_derivative_at_integration_points(int_space, (1,))
        .flatten()
    )

    # Backward derivatives
    det = dx1dxi1 * dx2dxi2 - dx1dxi2 * dx2dxi1
    dxi1dx1 = dx2dxi2 / det
    dxi1dx2 = -dx1dxi2 / det
    dxi2dx1 = -dx2dxi1 / det
    dxi2dx2 = dx1dxi1 / det

    # With 2 input dimensions, we have both 1-forms and 2-forms
    transformation_1 = space_map.basis_transform(1)
    assert transformation_1.shape == (2, 2, int_space.weights().size)
    # Check 1-from factors
    assert pytest.approx(dxi1dx1) == transformation_1[0, 0, ...]
    assert pytest.approx(dxi1dx2) == transformation_1[0, 1, ...]
    assert pytest.approx(dxi2dx1) == transformation_1[1, 0, ...]
    assert pytest.approx(dxi2dx2) == transformation_1[1, 1, ...]

    transformation_2 = space_map.basis_transform(2)
    assert transformation_2.shape == (1, 1, int_space.weights().size)
    # Check 2-from factor
    assert (
        pytest.approx(1 / det) == transformation_2[0, 0, :]
    )  # Trivial answer for all n-forms in n-dimensional space


@pytest.mark.parametrize(
    ("pts_h", "pts_v", "order_i1", "order_i2"),
    (
        (2, 3, 1, 2),
        (4, 5, 2, 1),
        (3, 3, 4, 4),
        (5, 5, 5, 4),
    ),
)
def test_2d_to_3d(pts_h: int, pts_v: int, order_i1: int, order_i2: int) -> None:
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

    int_space = IntegrationSpace(IntegrationSpecs(order_i1), IntegrationSpecs(order_i2))
    space_map = quad(int_space)

    with pytest.raises(ValueError):
        space_map.basis_transform(0)  # Raises Value error

    # Forward derivatives
    dx1dxi1 = (
        quad.dofs[0]
        .reconstruct_derivative_at_integration_points(int_space, (0,))
        .flatten()
    )
    dx1dxi2 = (
        quad.dofs[0]
        .reconstruct_derivative_at_integration_points(int_space, (1,))
        .flatten()
    )
    dx2dxi1 = (
        quad.dofs[1]
        .reconstruct_derivative_at_integration_points(int_space, (0,))
        .flatten()
    )
    dx2dxi2 = (
        quad.dofs[1]
        .reconstruct_derivative_at_integration_points(int_space, (1,))
        .flatten()
    )
    dx3dxi1 = (
        quad.dofs[2]
        .reconstruct_derivative_at_integration_points(int_space, (0,))
        .flatten()
    )
    dx3dxi2 = (
        quad.dofs[2]
        .reconstruct_derivative_at_integration_points(int_space, (1,))
        .flatten()
    )

    # Pseudo-invert for backwrad derivatives
    matrices = np.array(((dx1dxi1, dx1dxi2), (dx2dxi1, dx2dxi2), (dx3dxi1, dx3dxi2)))
    inverses = np.zeros((matrices.shape[1], matrices.shape[0], matrices.shape[2]))
    for i in range(matrices.shape[2]):
        mat = matrices[:, :, i]
        inv = np.linalg.pinv(mat)
        inverses[:, :, i] = inv

    dxi1dx1 = inverses[0, 0, ...]
    dxi1dx2 = inverses[0, 1, ...]
    dxi1dx3 = inverses[0, 2, ...]
    dxi2dx1 = inverses[1, 0, ...]
    dxi2dx2 = inverses[1, 1, ...]
    dxi2dx3 = inverses[1, 2, ...]

    # With 2 input dimensions, we have both 1-forms and 2-forms
    transformation_1 = space_map.basis_transform(1)
    assert transformation_1.shape == (2, 3, int_space.weights().size)
    # Check 1-from factors
    assert pytest.approx(dxi1dx1) == transformation_1[0, 0, ...]
    assert pytest.approx(dxi1dx2) == transformation_1[0, 1, ...]
    assert pytest.approx(dxi1dx3) == transformation_1[0, 2, ...]
    assert pytest.approx(dxi2dx1) == transformation_1[1, 0, ...]
    assert pytest.approx(dxi2dx2) == transformation_1[1, 1, ...]
    assert pytest.approx(dxi2dx3) == transformation_1[1, 2, ...]

    transformation_2 = space_map.basis_transform(2)
    assert transformation_2.shape == (1, 3, int_space.weights().size)

    # This shit is kinda complicated to compute, but we got it

    # dxi1 ^ dxi2 = (dxi1/dx1 dx1 + dxi1/dx2 dx2 + dxi1/dx3 dx3) ^
    # ^ (dxi2/dx1 dx1 + dxi2/dx2 dx2 + dxi2/dx3 dx3) =
    # = (dxi1/dx1 * dxi2/dx2 - dxi1/dx2 * dxi2/dx1) dx1 ^ dx2 +
    # + (dxi1/dx1 * dxi2/dx3 - dxi1/dx3 * dxi2/dx1) dx1 ^ dx3 +
    # + (dxi1/dx2 * dxi2/dx3 - dxi1/dx3 * dxi2/dx2) dx2 ^ dx3

    c1 = dxi1dx1 * dxi2dx2 - dxi1dx2 * dxi2dx1
    c2 = dxi1dx1 * dxi2dx3 - dxi1dx3 * dxi2dx1
    c3 = dxi1dx2 * dxi2dx3 - dxi1dx3 * dxi2dx2

    # Check 2-from factor
    assert pytest.approx(c1) == transformation_2[0, 0, :]
    assert pytest.approx(c2) == transformation_2[0, 1, :]
    assert pytest.approx(c3) == transformation_2[0, 2, :]
