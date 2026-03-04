r"""
.. currentmodule:: fdg

N-D Poisson
===========

This example demonstrates system for N-dimensional mixed Poisson equation can be set up.

Mixed Poisson equation is defined in the weak form as:

.. math::
    :label: examples_nd_poisson_1

    \left( p^{(n - 1)}, q^{(n - 1)} \right)_\Omega + \left( \mathrm{d} p^{(n - 1)},
    u^{(n)} \right)_\Omega = \int_{\partial \Omega} p^{(n - 1)} \wedge \star u^{(n)}

.. math::
    :label: examples_nd_poisson_2

    \left( v^{(n)}, \mathrm{d} q^{(n - 1)} \right)_\Omega =
    \left( v^{(n)}, f^{(n)} \right)_\Omega

"""  # noqa: D205 D400

from time import perf_counter
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KForm,
    KFormSpecs,
    SpaceMap,
    compute_kform_mass_matrix,
    incidence_kform_operator,
    projection_kform_l2_dual,
    reconstruct,
    transform_kform_to_target,
)

# %%
#
# The manufactured solution for the general N-dimensional case uses
# the following for the manufactured solution:
#
# ..math::
#     :label: examples_nd_poisson_man_sol
#
#     u^{(n)}(x_1, \dots, x_n) = \left(\prod\limits_{i=1}^n \cos\left( \frac{\pi}{2} x_i
#     \right)\right) \mathrm{d} x_1 \wedge \dots \wedge \mathrm{d} x_n
#
#
# This gives the forcing function:
#
# ..math::
#     :label: examples_nd_poisson_man_for
#
#     f^{(n)}(x_1, \dots, x_n) = - n \left(\frac{\pi}{2}\right)^2 \left(
#     \prod\limits_{i=1}^n \cos\left( \frac{\pi}{2} x_i
#     \right)\right) \mathrm{d} x_1 \wedge \dots \wedge \mathrm{d} x_n
#
#

SCALE = 0.1


def manufactured_solution(*x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Exact manufactured solution."""
    res = np.cos(x[0] * np.pi / 2)
    for v in x[1:]:
        res *= np.cos(v * np.pi / 2)
    return res * SCALE


def manufactured_source_poisson(*x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Exact manufactured source term."""
    res = np.cos(x[0] * np.pi / 2)
    for v in x[1:]:
        res *= np.cos(v * np.pi / 2)
    res *= -((np.pi / 2) ** 2) * len(x)
    return res * SCALE


# %%
#
# First we need to define how we discretize our k-forms and integration. This means
# defining the order of basis and integration method, as well as their types.


def disturbed_mapping(
    c: float, idx: int, *x: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
    """Return a perturbed map, where the boundaries are not affected, but the interior is.

    Parameters
    ----------
    c : float
        Strenght of the disturbance.

    idx : int
        Index of the input to base the mapping on.

    *x : array
        Coordinates where the mapping should be computed.

    Returns
    -------
    array
        Input coordinate ``idx``, but somewhat.
    """
    base = x[idx]
    d = np.full_like(base, c)
    for v in x:
        d *= (1 - v**2) * np.sin(np.pi * v) * 0
    return base + d


def create_space_map(
    c: float, orders: Sequence[int], space: IntegrationSpace
) -> SpaceMap:
    """Create space map that are is disturbed."""
    func_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, order) for order in orders)
    )
    ndim = len(orders)
    points = np.meshgrid(
        *[np.linspace(-1, +1, order + 1) for order in orders], indexing="ij"
    )
    return SpaceMap(
        *[
            CoordinateMap(
                DegreesOfFreedom(func_space, disturbed_mapping(c, idim, *points)),
                space,
            )
            for idim in range(ndim)
        ]
    )


def compute_l2_error(
    order_integration: int,
    type_integration: IntegrationMethod,
    order_basis: int,
    type_basis: BasisType,
    ndim: int,
    dp: int,
) -> float:
    """Solve the N-dimensional Poisson equation and compute the L^2 error."""
    int_space = IntegrationSpace(
        *((IntegrationSpecs(order_integration, type_integration),) * ndim)
    )
    int_space_higher = IntegrationSpace(
        *((IntegrationSpecs(order_integration + dp, type_integration),) * ndim)
    )
    base_space = FunctionSpace(*((BasisSpecs(type_basis, order_basis),) * ndim))

    space_map = create_space_map(0.1, base_space.orders, int_space)
    space_map_high = create_space_map(0.1, base_space.orders, int_space_higher)

    specs_u = KFormSpecs(ndim, base_space)
    specs_q = KFormSpecs(ndim - 1, base_space)

    source_vals = projection_kform_l2_dual(
        [manufactured_source_poisson], specs_u, space_map_high
    )[0]

    mq = compute_kform_mass_matrix(space_map, ndim - 1, base_space, base_space)
    mu = compute_kform_mass_matrix(space_map, ndim, base_space, base_space)

    mu_e = incidence_kform_operator(specs_q, mu, right=True)
    et_mu = incidence_kform_operator(specs_q, mu, transpose=True)

    system_matrix = np.block(
        [
            [mq, et_mu],
            [mu_e, np.zeros_like(mu)],
        ]
    )
    rhs = np.concatenate((np.zeros(mq.shape[0]), source_vals.flatten()))

    solution_dofs = np.linalg.solve(system_matrix, rhs)

    sol_q = KForm(specs_q)
    sol_u = KForm(specs_u)

    sol_q.values[:] = solution_dofs[: mq.shape[0]]
    sol_u.values[:] = solution_dofs[mq.shape[0] :]
    u_dofs = DegreesOfFreedom(
        specs_u.get_component_function_space(0), sol_u.get_component_dofs(0)
    )
    # Compute the L^2 error

    # K-form computed values at integration nodes
    computed_values = transform_kform_to_target(
        ndim,
        space_map_high,
        [reconstruct(u_dofs, *space_map_high.integration_space.nodes())],
    )[0]
    # K-form exact values at integration nodes
    real_values = manufactured_solution(
        *[space_map_high.coordinate_map(idx).values for idx in range(ndim)]
    )

    err_l2 = np.sum(
        (computed_values - real_values) ** 2
        * space_map_high.determinant
        * space_map_high.integration_space.weights()
    )
    return float(np.sqrt(err_l2))


NDIM = 1
BTYPE = BasisType.BERNSTEIN
ITYPE = IntegrationMethod.GAUSS_LOBATTO
DP = 1

pvals = np.arange(1, 7)
evals = np.zeros(pvals.size)
tvals = np.zeros(pvals.size)
for ndim in range(1, 4):
    for ip, p in enumerate(pvals):
        pv = int(p)
        t0 = perf_counter()
        l2 = compute_l2_error(pv + 0, ITYPE, pv, BTYPE, ndim, DP)
        t1 = perf_counter()
        evals[ip] = l2
        tvals[ip] = t1 - t0
        # print(
        # f"Computed error for {ndim=} with {pv=}: {l2:.14e} (took {t1 - t0:g} seconds)"
        # )

    k1, k0 = np.polyfit(pvals, np.log(evals), deg=1)
    c = np.exp(k0)
    b = np.exp(k1)

    fig, ax = plt.subplots()

    ax.scatter(pvals, evals)
    ax.plot(
        pvals,
        c * b**pvals,
        linestyle="dashed",
        label=f"$\\varepsilon = {c:.2g} \\cdot {b:.2g}^{{p}}$",
    )
    ax.set(
        yscale="log",
        xlabel="$p$",
        ylabel="$\\left|\\left| \\varepsilon \\right|\\right|_{ L^2 }$",
    )
    ax.grid()
    ax.legend()
    ax.set_title(f"{ndim}-dimensional Poisson equation convergence")
    fig.tight_layout()

plt.show()
