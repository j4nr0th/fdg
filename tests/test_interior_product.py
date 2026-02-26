"""Check that interior product is correctly compute (whatever that may mean)."""

import numpy as np
import pytest
from fdg._fdg import (
    BasisRegistry,
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationRegistry,
    IntegrationSpace,
    IntegrationSpecs,
    SpaceMap,
    compute_kform_interior_product_matrix,
)
from fdg.enum_type import BasisType

_TEST_CASES_2D = (
    (6, 7, 3, BasisType.LEGENDRE, 4, BasisType.BERNSTEIN),
    (5, 5, 2, BasisType.BERNSTEIN, 2, BasisType.BERNSTEIN),
    (4, 3, 5, BasisType.LAGRNAGE_GAUSS, 4, BasisType.LAGRANGE_UNIFORM),
)
_TEST_CASES_3D = (
    (6, 7, 5, 3, BasisType.LEGENDRE, 4, BasisType.BERNSTEIN, 4, BasisType.BERNSTEIN),
    (5, 5, 5, 2, BasisType.BERNSTEIN, 2, BasisType.BERNSTEIN, 2, BasisType.BERNSTEIN),
    (
        4,
        3,
        5,
        5,
        BasisType.LAGRNAGE_GAUSS,
        4,
        BasisType.LAGRANGE_UNIFORM,
        3,
        BasisType.LAGRANGE_CHEBYSHEV_GAUSS,
    ),
)


@pytest.mark.parametrize(("io1", "io2", "bo1", "bt1", "bo2", "bt2"), _TEST_CASES_2D)
def test_2d_1form(
    io1: int, io2: int, bo1: int, bt1: BasisType, bo2: int, bt2: BasisType
) -> None:
    """Check that 2D interior product of a 1-form works."""
    i_reg = IntegrationRegistry()
    b_reg = BasisRegistry()
    rng = np.random.default_rng(67)
    int_space = IntegrationSpace(IntegrationSpecs(io1), IntegrationSpecs(io2))
    func_space = FunctionSpace(BasisSpecs(bt1, bo1), BasisSpecs(bt2, bo2))

    x0_dofs = DegreesOfFreedom(func_space.lower_order(idim=0))
    x1_dofs = DegreesOfFreedom(func_space.lower_order(idim=1))
    x0_dofs.values = rng.random(x0_dofs.shape)
    x1_dofs.values = rng.random(x1_dofs.shape)
    space_map = SpaceMap(
        CoordinateMap(x0_dofs, int_space), CoordinateMap(x1_dofs, int_space)
    )

    int_weights = int_space.weights(i_reg) * space_map.determinant
    vx = rng.random(int_weights.shape)
    vy = rng.random(int_weights.shape)
    v_vals = np.array((vx, vy))

    u0_dofs = DegreesOfFreedom(func_space.lower_order(idim=0))
    u1_dofs = DegreesOfFreedom(func_space.lower_order(idim=1))
    u0_dofs.values = rng.random(u0_dofs.shape)
    u1_dofs.values = rng.random(u1_dofs.shape)
    u_dofs = np.concatenate((u0_dofs.values.flatten(), u1_dofs.values.flatten()))

    dxi1dx1 = space_map.inverse_map[..., 0, 0]
    dxi1dx2 = space_map.inverse_map[..., 0, 1]
    dxi2dx1 = space_map.inverse_map[..., 1, 0]
    dxi2dx2 = space_map.inverse_map[..., 1, 1]

    u0_vals = u0_dofs.reconstruct_at_integration_points(int_space, i_reg, b_reg)
    u1_vals = u1_dofs.reconstruct_at_integration_points(int_space, i_reg, b_reg)
    interprod_values = (vx * dxi1dx1 + vy * dxi1dx2) * u0_vals + (
        vx * dxi2dx1 + vy * dxi2dx2
    ) * u1_vals

    basis_0form = func_space.values_at_integration_nodes(
        int_space, integration_registry=i_reg, basis_registry=b_reg
    )
    dual_dof_values = np.sum(
        interprod_values[..., None, None] * int_weights[..., None, None] * basis_0form,
        axis=tuple(range(basis_0form.ndim - 2)),
    )

    interprod_mat = compute_kform_interior_product_matrix(
        space_map,
        1,
        func_space,
        func_space,
        v_vals,
        integration_registry=i_reg,
        basis_registry=b_reg,
    )
    computed_dual_dofs = np.reshape(interprod_mat @ u_dofs, (bo1 + 1, bo2 + 1))

    assert pytest.approx(computed_dual_dofs) == dual_dof_values


@pytest.mark.parametrize(("io1", "io2", "bo1", "bt1", "bo2", "bt2"), _TEST_CASES_2D)
def test_2d_2form(
    io1: int, io2: int, bo1: int, bt1: BasisType, bo2: int, bt2: BasisType
) -> None:
    """Check that 2D interior product of a 2-form works."""
    i_reg = IntegrationRegistry()
    b_reg = BasisRegistry()
    rng = np.random.default_rng(67)
    int_space = IntegrationSpace(IntegrationSpecs(io1), IntegrationSpecs(io2))
    func_space = FunctionSpace(BasisSpecs(bt1, bo1), BasisSpecs(bt2, bo2))

    x0_dofs = DegreesOfFreedom(func_space.lower_order(idim=0))
    x1_dofs = DegreesOfFreedom(func_space.lower_order(idim=1))
    x0_dofs.values = rng.random(x0_dofs.shape)
    x1_dofs.values = rng.random(x1_dofs.shape)
    space_map = SpaceMap(
        CoordinateMap(x0_dofs, int_space), CoordinateMap(x1_dofs, int_space)
    )

    int_weights = int_space.weights(i_reg) * space_map.determinant
    vx = rng.random(int_weights.shape)
    vy = rng.random(int_weights.shape)
    v_vals = np.array((vx, vy))

    u_dofs = DegreesOfFreedom(func_space.lower_order(idim=0).lower_order(idim=1))
    u_dofs.values = rng.random(u_dofs.shape)

    dxi0dx0 = space_map.inverse_map[..., 0, 0]
    dxi0dx1 = space_map.inverse_map[..., 0, 1]
    dxi1dx0 = space_map.inverse_map[..., 1, 0]
    dxi1dx1 = space_map.inverse_map[..., 1, 1]

    inv_det = dxi0dx0 * dxi1dx1 - dxi0dx1 * dxi1dx0

    u_vals = u_dofs.reconstruct_at_integration_points(int_space, i_reg, b_reg) * inv_det
    interprod_values_0_target = -u_vals * vy
    interprod_values_1_target = +u_vals * vx

    basis_1form_0 = func_space.lower_order(idim=0).values_at_integration_nodes(
        int_space, integration_registry=i_reg, basis_registry=b_reg
    )
    basis_1form_1 = func_space.lower_order(idim=1).values_at_integration_nodes(
        int_space, integration_registry=i_reg, basis_registry=b_reg
    )

    dual_dof_values_0 = np.sum(
        (interprod_values_0_target * dxi0dx0 + interprod_values_1_target * dxi0dx1)[
            ..., None, None
        ]
        * basis_1form_0
        * int_weights[..., None, None],
        axis=tuple(range(basis_1form_0.ndim - 2)),
    )
    dual_dof_values_1 = np.sum(
        (interprod_values_0_target * dxi1dx0 + interprod_values_1_target * dxi1dx1)[
            ..., None, None
        ]
        * int_weights[..., None, None]
        * basis_1form_1,
        axis=tuple(range(basis_1form_1.ndim - 2)),
    )

    interprod_mat = compute_kform_interior_product_matrix(
        space_map,
        2,
        func_space,
        func_space,
        v_vals,
        integration_registry=i_reg,
        basis_registry=b_reg,
    )
    computed_dual_dofs = interprod_mat @ u_dofs.values.flatten()

    flattened_dual_dofs = np.concatenate(
        (dual_dof_values_0.flatten(), dual_dof_values_1.flatten())
    )

    assert pytest.approx(computed_dual_dofs) == flattened_dual_dofs


@pytest.mark.parametrize(
    ("io1", "io2", "io3", "bo1", "bt1", "bo2", "bt2", "bo3", "bt3"), _TEST_CASES_3D
)
def test_3d_2form(
    io1: int,
    io2: int,
    io3: int,
    bo1: int,
    bt1: BasisType,
    bo2: int,
    bt2: BasisType,
    bo3: int,
    bt3: BasisType,
) -> None:
    """Check that 3D interior product of a 2-form works."""
    i_reg = IntegrationRegistry()
    b_reg = BasisRegistry()
    rng = np.random.default_rng(67)
    int_space = IntegrationSpace(
        IntegrationSpecs(io1), IntegrationSpecs(io2), IntegrationSpecs(io3)
    )
    func_space = FunctionSpace(
        BasisSpecs(bt1, bo1), BasisSpecs(bt2, bo2), BasisSpecs(bt3, bo3)
    )

    x0_dofs = DegreesOfFreedom(func_space.lower_order(idim=0))
    x1_dofs = DegreesOfFreedom(func_space.lower_order(idim=1))
    x2_dofs = DegreesOfFreedom(func_space.lower_order(idim=1))
    x0_dofs.values = rng.random(x0_dofs.shape)
    x1_dofs.values = rng.random(x1_dofs.shape)
    x2_dofs.values = rng.random(x2_dofs.shape)
    space_map = SpaceMap(
        CoordinateMap(x0_dofs, int_space),
        CoordinateMap(x1_dofs, int_space),
        CoordinateMap(x2_dofs, int_space),
    )

    int_weights = int_space.weights(i_reg) * space_map.determinant
    v0 = rng.random(int_weights.shape)
    v1 = rng.random(int_weights.shape)
    v2 = rng.random(int_weights.shape)
    v_vals = np.array((v0, v1, v2))

    u0_dofs = DegreesOfFreedom(func_space.lower_order(idim=0).lower_order(idim=1))
    u0_dofs.values = rng.random(u0_dofs.shape)
    u1_dofs = DegreesOfFreedom(func_space.lower_order(idim=0).lower_order(idim=2))
    u1_dofs.values = rng.random(u1_dofs.shape)
    u2_dofs = DegreesOfFreedom(func_space.lower_order(idim=1).lower_order(idim=2))
    u2_dofs.values = rng.random(u2_dofs.shape)

    dxi0dx0 = space_map.inverse_map[..., 0, 0]
    dxi0dx1 = space_map.inverse_map[..., 0, 1]
    dxi0dx2 = space_map.inverse_map[..., 0, 2]
    dxi1dx0 = space_map.inverse_map[..., 1, 0]
    dxi1dx1 = space_map.inverse_map[..., 1, 1]
    dxi1dx2 = space_map.inverse_map[..., 1, 2]
    dxi2dx0 = space_map.inverse_map[..., 2, 0]
    dxi2dx1 = space_map.inverse_map[..., 2, 1]
    dxi2dx2 = space_map.inverse_map[..., 2, 2]

    u0_vals = u0_dofs.reconstruct_at_integration_points(int_space, i_reg, b_reg)
    u1_vals = u1_dofs.reconstruct_at_integration_points(int_space, i_reg, b_reg)
    u2_vals = u2_dofs.reconstruct_at_integration_points(int_space, i_reg, b_reg)
    u0 = (
        u0_vals * (dxi0dx0 * dxi1dx1 - dxi0dx1 * dxi1dx0)
        + u1_vals * (dxi0dx0 * dxi2dx1 - dxi0dx1 * dxi2dx0)
        + u2_vals * (dxi1dx0 * dxi2dx1 - dxi1dx1 * dxi2dx0)
    )
    u1 = (
        u0_vals * (dxi0dx0 * dxi1dx2 - dxi0dx2 * dxi1dx0)
        + u1_vals * (dxi0dx0 * dxi2dx2 - dxi0dx2 * dxi2dx0)
        + u2_vals * (dxi1dx0 * dxi2dx2 - dxi1dx2 * dxi2dx0)
    )
    u2 = (
        u0_vals * (dxi0dx1 * dxi1dx2 - dxi0dx2 * dxi1dx1)
        + u1_vals * (dxi0dx1 * dxi2dx2 - dxi0dx2 * dxi2dx1)
        + u2_vals * (dxi1dx1 * dxi2dx2 - dxi1dx2 * dxi2dx1)
    )
    interprod_values_0_target = -(u0 * v1 + u1 * v2)
    interprod_values_1_target = u0 * v0 - u2 * v2
    interprod_values_2_target = u1 * v0 + u2 * v1

    basis_1form_0 = func_space.lower_order(idim=0).values_at_integration_nodes(
        int_space, integration_registry=i_reg, basis_registry=b_reg
    )
    basis_1form_1 = func_space.lower_order(idim=1).values_at_integration_nodes(
        int_space, integration_registry=i_reg, basis_registry=b_reg
    )
    basis_1form_2 = func_space.lower_order(idim=2).values_at_integration_nodes(
        int_space, integration_registry=i_reg, basis_registry=b_reg
    )

    dual_dof_values_0 = np.sum(
        (
            interprod_values_0_target * dxi0dx0
            + interprod_values_1_target * dxi0dx1
            + interprod_values_2_target * dxi0dx2
        )[..., None, None, None]
        * basis_1form_0
        * int_weights[..., None, None, None],
        axis=tuple(range(basis_1form_0.ndim - 3)),
    )
    dual_dof_values_1 = np.sum(
        (
            interprod_values_0_target * dxi1dx0
            + interprod_values_1_target * dxi1dx1
            + interprod_values_2_target * dxi1dx2
        )[..., None, None, None]
        * int_weights[..., None, None, None]
        * basis_1form_1,
        axis=tuple(range(basis_1form_1.ndim - 3)),
    )
    dual_dof_values_2 = np.sum(
        (
            interprod_values_0_target * dxi2dx0
            + interprod_values_1_target * dxi2dx1
            + interprod_values_2_target * dxi2dx2
        )[..., None, None, None]
        * int_weights[..., None, None, None]
        * basis_1form_2,
        axis=tuple(range(basis_1form_2.ndim - 3)),
    )

    interprod_mat = compute_kform_interior_product_matrix(
        space_map,
        2,
        func_space,
        func_space,
        v_vals,
        integration_registry=i_reg,
        basis_registry=b_reg,
    )
    u_dofs = np.concatenate(
        (u0_dofs.values.flatten(), u1_dofs.values.flatten(), u2_dofs.values.flatten())
    )
    computed_dual_dofs = interprod_mat @ u_dofs.flatten()

    flattened_dual_dofs = np.concatenate(
        (
            dual_dof_values_0.flatten(),
            dual_dof_values_1.flatten(),
            dual_dof_values_2.flatten(),
        )
    )

    assert pytest.approx(computed_dual_dofs) == flattened_dual_dofs


if __name__ == "__main__":
    for args in _TEST_CASES_3D:
        test_3d_2form(*args)
