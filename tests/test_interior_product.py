"""Check that interior product is correctly compute (whatever that may mean)."""

import numpy as np
import pytest
from interplib._interp import (
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
from interplib.enum_type import BasisType

_TEST_CASES_2D = (
    (6, 7, 3, BasisType.LEGENDRE, 4, BasisType.BERNSTEIN),
    (5, 5, 2, BasisType.BERNSTEIN, 2, BasisType.BERNSTEIN),
    (4, 3, 5, BasisType.LAGRNAGE_GAUSS, 4, BasisType.LAGRANGE_UNIFORM),
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
