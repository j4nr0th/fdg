"""Check that the gradien mass matrix works as one would expect."""

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    Line,
)
from fdg._fdg import (
    DegreesOfFreedom,
    compute_gradient_mass_matrix,
    compute_mass_matrix,
)
from fdg.degrees_of_freedom import reconstruct
from fdg.domains import Quad
from fdg.integration import projection_l2_dual, projection_l2_primal


@pytest.mark.parametrize("order", (1, 2, 4))
def test_gradient_1d(order: int):
    """Check that gradient on a line works."""

    def function(*args):
        (x,) = args
        return x**order

    def grad_x(*args):
        (x,) = args
        return order * x ** (order - 1)

    domain = Line((0,), (+1,), (+5,))
    k = 4
    func_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, order + k))
    test_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, order + k - 1))
    int_space = IntegrationSpace(IntegrationSpecs(order + k + 2, IntegrationMethod.GAUSS))

    smap = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, smap)
    grad_x_dofs = projection_l2_dual(grad_x, test_space, smap)

    grad_matrix_x = compute_gradient_mass_matrix(func_space, test_space, smap, 0, 0)
    comp_dual_x = grad_matrix_x @ base_dofs.values
    assert pytest.approx(grad_x_dofs.values) == comp_dual_x


# NOTE: this test can mathematically not work! If you have a 1D object in 2D space,
# you can only recover the tangential gradient along it (duh!)
def _check_gradient_1d_to_2d():
    """Check that gradient on a line works."""

    def function(*args):
        x, y = args
        return 3 * (x + y) ** 2 + 2 * (x + y) + 1

    def grad_x(*args):
        x, y = args
        return 6 * (x + y) + 2 * x

    def grad_y(*args):
        x, y = args
        return 6 * (x + y) + 2 * y

    domain = Line((0, 0), (1, 1))

    func_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 3))
    test_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 2))
    int_space = IntegrationSpace(IntegrationSpecs(4, IntegrationMethod.GAUSS))

    domain_map = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, domain_map)
    grad_x_dofs = projection_l2_dual(grad_x, test_space, domain_map)
    grad_y_dofs = projection_l2_dual(grad_y, test_space, domain_map)

    grad_matrix_x = compute_gradient_mass_matrix(func_space, test_space, domain_map, 0)
    comp_dual_x = grad_matrix_x @ base_dofs.values

    grad_matrix_y = compute_gradient_mass_matrix(func_space, test_space, domain_map, 1)
    comp_dual_y = grad_matrix_y @ base_dofs.values
    print(f"{base_dofs.values=}")
    print(f"{grad_matrix_x=}")
    print(f"{grad_matrix_y=}")
    print("Real x:", grad_x_dofs.values)
    print("Comp x:", comp_dual_x)
    print("Real y:", grad_y_dofs.values)
    print("Comp y:", comp_dual_y)
    assert pytest.approx(grad_x_dofs.values) == comp_dual_x
    assert pytest.approx(grad_y_dofs.values) == comp_dual_y


@pytest.mark.parametrize(("order1", "order2"), ((6, 6), (8, 6), (6, 9)))
def test_gradient_2d_to_2d_linear_map(order1: int, order2: int):
    """Check that gradient on a surface works."""

    def function(*args):
        x, y = args
        return 2 * (x + 1) ** 2 + 2 * y**2 + 1

    def grad_x(*args):
        x, y = args
        return 4 * (x + 1) + 0 * y

    def grad_y(*args):
        x, y = args
        return 4 * y + 0 * x + 0 * y

    # If we choose region that's too big or to deformed, we will not converge until
    # REALLY high orders.
    domain = Quad(
        bottom=Line((-1.5, -1), (+1, -2)),
        right=Line((+1, -2), (+1, +1)),
        top=Line((+1, +1), (-1, +1)),
        left=Line((-1, +1), (-1.5, -1)),
    ).subregion((-0.1, +0.1), (-0.1, +0.1))

    func_space = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, order1), BasisSpecs(BasisType.LEGENDRE, order2)
    )
    test_space_0 = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, order1 - 1),
        BasisSpecs(BasisType.LEGENDRE, order2),
    )
    test_space_1 = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, order1),
        BasisSpecs(BasisType.LEGENDRE, order2 - 1),
    )
    int_space = IntegrationSpace(
        IntegrationSpecs(order1 * 2, IntegrationMethod.GAUSS),
        IntegrationSpecs(order2 * 2, IntegrationMethod.GAUSS),
    )

    domain_map = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, domain_map)

    grad_matrix_x_0 = compute_gradient_mass_matrix(
        func_space, test_space_0, domain_map, 0, 0
    )
    grad_matrix_y_0 = compute_gradient_mass_matrix(
        func_space, test_space_0, domain_map, 0, 1
    )
    grad_matrix_x_1 = compute_gradient_mass_matrix(
        func_space, test_space_1, domain_map, 1, 0
    )
    grad_matrix_y_1 = compute_gradient_mass_matrix(
        func_space, test_space_1, domain_map, 1, 1
    )

    mass_matrix_0 = compute_mass_matrix(test_space_0, test_space_0, domain_map)
    mass_matrix_1 = compute_mass_matrix(test_space_1, test_space_1, domain_map)

    comp_x0 = DegreesOfFreedom(
        test_space_0,
        np.linalg.solve(mass_matrix_0, grad_matrix_x_0 @ base_dofs.values.flatten()),
    )
    comp_x1 = DegreesOfFreedom(
        test_space_1,
        np.linalg.solve(mass_matrix_1, grad_matrix_x_1 @ base_dofs.values.flatten()),
    )

    comp_y0 = DegreesOfFreedom(
        test_space_0,
        np.linalg.solve(mass_matrix_0, grad_matrix_y_0 @ base_dofs.values.flatten()),
    )
    comp_y1 = DegreesOfFreedom(
        test_space_1,
        np.linalg.solve(mass_matrix_1, grad_matrix_y_1 @ base_dofs.values.flatten()),
    )

    t1, t2 = np.meshgrid(np.linspace(-1, +1, 6), np.linspace(-1, +1, 6))

    real_grad_x = grad_x(*domain.sample(t1, t2))
    real_grad_y = grad_y(*domain.sample(t1, t2))

    recon_grad_x = reconstruct(comp_x0, t1, t2) + reconstruct(comp_x1, t1, t2)
    recon_grad_y = reconstruct(comp_y0, t1, t2) + reconstruct(comp_y1, t1, t2)

    # print("Err x:", real_grad_x - recon_grad_x)
    # print("Err y:", real_grad_y - recon_grad_y)

    # def rms(x):
    #     return np.sqrt(np.mean(np.square(x)))

    # return rms(real_grad_x - recon_grad_x), rms(real_grad_y - recon_grad_y)

    assert pytest.approx(real_grad_x) == recon_grad_x
    assert pytest.approx(real_grad_y) == recon_grad_y


# @pytest.mark.parametrize(("order1", "order2"), ((6, 6), (8, 6), (6, 9)))
def _test_gradient_2d_to_2d_deformed(order1: int, order2: int):
    """Check that gradient on a surface works."""

    def function(*args):
        x, y = args
        return 2 * (x + 1) ** 2 + 2 * y**2 + 1

    def grad_x(*args):
        x, y = args
        return 4 * (x + 1) + 0 * y

    def grad_y(*args):
        x, y = args
        return 4 * y + 0 * x + 0 * y

    # If we choose region that's too big, we will not converge until ridiculus orders
    domain = Quad(
        bottom=Line((-1.5, -1), (+1, -2)),
        right=Line((+1, -2), (0, +1), (+1, +1)),
        top=Line((+1, +1), (0, +0.5), (0, +1.5), (-1, +1)),
        left=Line((-1, +1), (-0.7, 0.2), (-1.5, -1)),
    ).subregion((-0.1, +0.1), (-0.1, +0.1))
    # domain = Quad(
    #     Line((+2, -3), (0, +1.5), (+1, +1)),
    #     Line((+1, +1), (-1, +4)),
    #     Line((-1, +4), (-0.7, 0.2), (-2, -1)),
    #     Line((-2, -1), (+2, -3)),
    # ).subregion((-0.1, +0.1), (-0.1, +0.1))

    func_space = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, order1), BasisSpecs(BasisType.LEGENDRE, order2)
    )
    test_space_0 = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, order1 - 1),
        BasisSpecs(BasisType.LEGENDRE, order2),
    )
    test_space_1 = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, order1),
        BasisSpecs(BasisType.LEGENDRE, order2 - 1),
    )
    int_space = IntegrationSpace(
        IntegrationSpecs(order1 * 2, IntegrationMethod.GAUSS),
        IntegrationSpecs(order2 * 2, IntegrationMethod.GAUSS),
    )

    domain_map = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, domain_map)

    grad_matrix_x_0 = compute_gradient_mass_matrix(
        func_space, test_space_0, domain_map, 0, 0
    )
    grad_matrix_y_0 = compute_gradient_mass_matrix(
        func_space, test_space_0, domain_map, 0, 1
    )
    grad_matrix_x_1 = compute_gradient_mass_matrix(
        func_space, test_space_1, domain_map, 1, 0
    )
    grad_matrix_y_1 = compute_gradient_mass_matrix(
        func_space, test_space_1, domain_map, 1, 1
    )

    mass_matrix_0 = compute_mass_matrix(test_space_0, test_space_0, domain_map)
    mass_matrix_1 = compute_mass_matrix(test_space_1, test_space_1, domain_map)

    comp_x0 = DegreesOfFreedom(
        test_space_0,
        np.linalg.solve(mass_matrix_0, grad_matrix_x_0 @ base_dofs.values.flatten()),
    )
    comp_x1 = DegreesOfFreedom(
        test_space_1,
        np.linalg.solve(mass_matrix_1, grad_matrix_x_1 @ base_dofs.values.flatten()),
    )

    comp_y0 = DegreesOfFreedom(
        test_space_0,
        np.linalg.solve(mass_matrix_0, grad_matrix_y_0 @ base_dofs.values.flatten()),
    )
    comp_y1 = DegreesOfFreedom(
        test_space_1,
        np.linalg.solve(mass_matrix_1, grad_matrix_y_1 @ base_dofs.values.flatten()),
    )

    t1, t2 = np.meshgrid(np.linspace(-1, +1, 6), np.linspace(-1, +1, 6))

    real_grad_x = grad_x(*domain.sample(t1, t2))
    real_grad_y = grad_y(*domain.sample(t1, t2))

    recon_grad_x = reconstruct(comp_x0, t1, t2) + reconstruct(comp_x1, t1, t2)
    recon_grad_y = reconstruct(comp_y0, t1, t2) + reconstruct(comp_y1, t1, t2)

    print("Err x:", real_grad_x - recon_grad_x)
    print("Err y:", real_grad_y - recon_grad_y)

    def rms(x):
        return np.sqrt(np.mean(np.square(x)))

    return rms(real_grad_x - recon_grad_x), rms(real_grad_y - recon_grad_y)

    # assert pytest.approx(real_grad_x) == recon_grad_x
    # assert pytest.approx(real_grad_y) == recon_grad_y


if __name__ == "__main__":
    order = np.arange(3, 13)
    ex = np.empty(order.size)
    ey = np.empty(order.size)

    for i, n in enumerate(order):
        ex[i], ey[i] = _test_gradient_2d_to_2d_deformed(int(n), int(n))

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    ax.semilogy(order, ex, label="$\\varepsilon_x$")
    ax.semilogy(order, ey, label="$\\varepsilon_y$")
    ax.legend()
    ax.grid()
    ax.set(xlabel="$p$", ylabel="$\\varepsilon_\\text{ rms }$")
    fig.tight_layout()
    plt.show()
