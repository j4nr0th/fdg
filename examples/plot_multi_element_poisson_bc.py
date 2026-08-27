r"""
Multi-element mixed Poisson equation with strong and weak boundary conditions
=============================================================================

This example extends the multi-element mixed Poisson example by applying
boundary conditions on the outer boundary of the domain. The outer boundary is
*deformed*: the coordinate map no longer keeps the boundary fixed, so the
boundary faces are curved.

Two kinds of boundary conditions are demonstrated:

* a **strong** condition on the flux :math:`q`, enforced with Lagrange
  multipliers: the trace :math:`\operatorname{tr} q` on a face is prescribed
  by :meth:`Mesh.compute_kform_boundary_constraints`, exactly like the flux
  continuity between two elements;
* a **weak** condition on the solution :math:`u`, applied through the natural
  boundary term of the mixed formulation: the momentum equation is
  :math:`(p, q) + (\mathrm{d}p, u) = \int_{\partial\Omega} p \wedge \star u_D`,
  so prescribing :math:`u = u_D` weakly adds the boundary load
  :math:`\int_{F_e} u_D \operatorname{tr} p` to the momentum right-hand side.
  This is exactly the integral computed by :func:`compute_kform_boundary_load`.

The mixed Poisson equation is solved with the manufactured solution
:math:`u = \mathrm{SCALE}(\prod_i \cos(\pi x_i / 2) + 1)`, whose trace on the
deformed outer boundary is nonzero, with the flux
:math:`q = \star \mathrm{d}\star u` prescribed strongly on two faces and the
solution prescribed weakly on the remaining faces. Both a *neighboring* pair
of strong faces and an *opposite* pair are shown in two dimensions; because
the strong condition acts on the flux, neighboring strong faces constrain
disjoint components and do not over-constrain the system.
"""  # noqa: D205 D400

from itertools import combinations
from time import perf_counter
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyvista as pv
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
    Mesh,
    SpaceMap,
    compute_kform_boundary_load,
    compute_kform_mass_matrix,
    incidence_kform_operator,
    projection_kform_l2_dual,
    projection_kform_l2_primal,
    reconstruct,
    transform_kform_to_target,
)
from fdg.domains import _vtk_3d_indices
from fdg.integration import Integrable

# %%
#
# The manufactured solution is a product of cosines shifted by a constant, so
# it does not vanish on the outer boundary of the :math:`[-1, 1]^n` grid: its
# trace is the constant :math:`\mathrm{SCALE}` everywhere on the boundary and
# its normal flux is nonzero. The grids of hypercubes cover exactly that
# domain; both the interiors and the outer boundaries are deformed.
#
SCALE = 0.1

# Strength of the deformation and the order of the geometry basis.
DEFORM = 0.05
GEO_ORDER = 3


def manufactured_solution(*x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Exact manufactured solution."""
    res = np.cos(x[0] * np.pi / 2)
    for v in x[1:]:
        res *= np.cos(v * np.pi / 2)
    return (res + 1.0) * SCALE


def manufactured_source_laplacian(*x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Exact manufactured source term."""
    res = np.cos(x[0] * np.pi / 2)
    for v in x[1:]:
        res *= np.cos(v * np.pi / 2)
    res *= -((np.pi / 2) ** 2) * len(x)
    return res * SCALE


def deformed_map(*g: npt.NDArray[np.double]) -> tuple[npt.NDArray[np.double], ...]:
    """Deform grid coordinates, moving the outer boundary of the domain.

    The deformation does not vanish on the outer boundary, so every boundary
    face is curved. It is a diffeomorphism for ``DEFORM < 0.5``; the outer
    boundary is pushed outward by at most ``DEFORM * 2^ndim`` at the corners.
    """
    plus = np.ones_like(g[0])
    minus = np.ones_like(g[0])
    for v in g:
        plus *= 1.0 + v
        minus *= 1.0 - v
    return tuple(gi + DEFORM * (plus + minus) for gi in g)


def jacobian(*g: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Jacobian of deformed_map, with shape (..., ndim, ndim)."""
    g = np.broadcast_arrays(*g)
    ndim = len(g)
    J = np.zeros(g[0].shape + (ndim, ndim))
    for i in range(ndim):
        for k in range(ndim):
            plus = np.prod([1.0 + g[j] for j in range(ndim) if j != k], axis=0)
            minus = np.prod([1.0 - g[j] for j in range(ndim) if j != k], axis=0)
            J[..., i, k] = (1.0 if i == k else 0.0) + DEFORM * (plus - minus)
    return J


# %%
#
# The flux of the manufactured solution is :math:`q = \star \mathrm{d}\star u`.
# With the ordered basis :math:`(\mathrm{d}x_1, \dots, \mathrm{d}x_n)` and the
# Hodge star convention of the library, component ``c`` of the :math:`(n-1)`-
# form is the wedge of the ``c``-th combination of :math:`n-1` axes, and its
# coefficient is :math:`(-1)^m \partial_m u` where :math:`m` is the axis not
# contained in that combination.
#


def flux_components(ndim: int) -> list[Integrable]:
    """Return the flux components of the manufactured solution in ndim."""
    half = np.pi / 2.0

    def comp(axes: tuple[int, ...]) -> Integrable:
        missing = tuple(i for i in range(ndim) if i not in axes)[0]

        def value(*x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
            prod = np.ones_like(x[0])
            for i, xi in enumerate(x):
                if i != missing:
                    prod *= np.cos(xi * half)
            der = -half * np.sin(x[missing] * half)
            return SCALE * (1.0 if missing % 2 == 0 else -1.0) * der * prod

        return value

    return [comp(axes) for axes in combinations(range(ndim), ndim - 1)]


# %%
#
# First the topological mesh of the domain is built. The mesh has no geometry:
# it only stores the connections between the points, lines, faces and
# elements. The corner point IDs of the elements of the :math:`2^n` grid are
# computed below, with the point :math:`\sum_i \mathrm{idx}_i 3^i` being the
# grid node :math:`(\mathrm{idx}_0, \mathrm{idx}_1, \dots)`.
#


def grid_point(*idx: int) -> int:
    """Point ID of the grid node (idx[0], idx[1], ...) of the 2^ndim grid."""
    return sum(int(ix) * 3**d for d, ix in enumerate(idx))


def mesh_corners(ndim: int) -> npt.NDArray[np.uint64]:
    """Corner point IDs of all elements of the 2^ndim grid."""
    corners = []
    for element in range(1 << ndim):
        idx = tuple((element >> d) & 1 for d in range(ndim))
        for delta in range(1 << ndim):
            corners.append(
                grid_point(*(idx[d] + ((delta >> d) & 1) for d in range(ndim)))  # type: ignore
            )
    return np.array(corners, dtype=np.uint64)


# %%
#
# Mappings for every coordinate are collected into a joined :class:`SpaceMap`,
# which is then used to map :math:`k`-form components between the reference
# domain and the physical domain. The deformation moves the outer boundary, so
# the physical boundary is curved and every boundary face map carries the full
# surface geometry.
#


def element_map(
    idx: tuple[int, ...],
    basis: FunctionSpace,
    integration: IntegrationSpace,
) -> SpaceMap:
    """Deformed map of the element at grid position idx."""
    grids = np.meshgrid(
        *([np.linspace(-1.0, 1.0, GEO_ORDER + 1)] * len(idx)), indexing="ij"
    )
    deformed = deformed_map(*(0.5 * g + i - 0.5 for g, i in zip(grids, idx)))
    return SpaceMap(
        *(
            CoordinateMap(DegreesOfFreedom(basis, d.ravel()), integration)
            for d in deformed
        )
    )


def create_element_maps(ndim: int, integration: IntegrationSpace) -> list[SpaceMap]:
    """Deformed maps of all 2^ndim elements of the grid."""
    basis = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER) for _ in range(ndim))
    )
    return [
        element_map(tuple((e >> d) & 1 for d in range(ndim)), basis, integration)
        for e in range(1 << ndim)
    ]


# %%
#
# Along with the :class:`IntegrationSpace` objects that define integration we
# must define the discretization of the :math:`k`-forms using a :class:`FunctionSpace`
# object. With the base function space defined, the two :math:`k`-form
# specifications of the :math:`n`-dimensional mixed Poisson equation are
# created with :class:`KFormSpecs`: the :math:`(n - 1)`-form flux :math:`q`
# and the :math:`n`-form solution :math:`u`.
#


def create_kform_specs(
    type_basis: BasisType, order_basis: int, ndim: int
) -> tuple[KFormSpecs, KFormSpecs]:
    """Create the k-form specifications of the ndim-dimensional Poisson equation."""
    base_space = FunctionSpace(
        *(BasisSpecs(type_basis, order_basis) for _ in range(ndim))
    )
    specs_u = KFormSpecs(ndim, base_space)
    specs_q = KFormSpecs(ndim - 1, base_space)
    return specs_u, specs_q


# %%
# For the left side of the system the two mass matrices of the two
# :math:`k`-forms are computed, and the incidence operator is applied as
# needed. This produces the element block of the saddle-point system.
#


def assemble_element_lhs(
    sm: SpaceMap, specs_q: KFormSpecs, specs_u: KFormSpecs
) -> np.ndarray:
    """Assemble the element system matrix of the Poisson equation."""
    mq = compute_kform_mass_matrix(
        sm, specs_q.order, specs_q.base_space, specs_q.base_space
    )
    mu = compute_kform_mass_matrix(
        sm, specs_u.order, specs_u.base_space, specs_u.base_space
    )

    mu_e = incidence_kform_operator(specs_q, mu, right=True)
    et_mu = incidence_kform_operator(specs_q, mu, transpose=True)

    return np.block(
        [
            [mq, et_mu],
            [mu_e, np.zeros_like(mu)],
        ]
    )


# %%
# The right side of the Poisson equation is computed from the "dual projection"
# of the manufactured source term on the function space.
#


def assemble_element_rhs(
    specs_u: KFormSpecs, specs_q: KFormSpecs, sm_h: SpaceMap
) -> np.ndarray:
    """Assemble the element right-hand side of the Poisson equation."""
    source_vals = projection_kform_l2_dual(
        [manufactured_source_laplacian], specs_u, sm_h
    )[0]
    return np.concatenate(
        (
            np.zeros(int(np.sum(specs_q.component_dof_counts))),
            source_vals.flatten(),
        )
    )


# %%
#
# The boundary constraints returned by
# :meth:`Mesh.compute_kform_boundary_constraints` are packed sparse rows. For
# this example they are materialized into dense element operators, whose
# columns are the flattened degrees of freedom of the flux :math:`q`.
#


def packed_to_dense(
    result: tuple[np.ndarray, ...], element_spec: KFormSpecs
) -> np.ndarray:
    """Materialize packed boundary-constraint rows as a dense operator."""
    row_offsets, components, local_dofs, coefficients = result
    n_rows = row_offsets.size - 1
    n_dofs = int(np.sum(element_spec.component_dof_counts))
    matrix = np.zeros((n_rows, n_dofs))
    for row in range(n_rows):
        start, end = int(row_offsets[row]), int(row_offsets[row + 1])
        for i in range(start, end):
            column = int(
                element_spec.get_component_slice(int(components[i])).start
                + int(local_dofs[i])
            )
            matrix[row, column] += coefficients[i]
    return matrix


# %%
#
# Boundary faces are classified from their orientation record: the first entry
# is the signed fixed axis, whose sign is negative for the start side and
# positive for the end side, and whose absolute value minus one is the axis
# index.  A face with fixed axis ``a`` at side ``s`` has outward normal
# :math:`s \hat{x}_a`.  The example prescribes the flux strongly on the faces
# listed in ``strong`` and the solution weakly on all remaining boundary
# faces.
#


def face_axis_side(orient: npt.NDArray[np.int8]) -> tuple[int, int]:
    """Return (fixed axis index, side) of a boundary face orientation."""
    return abs(int(orient[0])) - 1, -1 if orient[0] < 0 else 1


def strong_constraint_row(
    mesh: Mesh,
    test_specs: KFormSpecs,
    specs_q: KFormSpecs,
    sm: SpaceMap,
    element_id: int,
    boundary_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Assemble the strong Neumann row and its prescribed data.

    The prescribed data is the trace of the exact flux: with the Lagrange
    multiplier enforcing :math:`\operatorname{tr} q = g_F`, tracing the exact
    flux avoids any manual sign bookkeeping of the outward normal.
    """
    t = packed_to_dense(
        mesh.compute_kform_boundary_constraints(
            test_specs, specs_q, sm, element_id, boundary_id
        ),
        specs_q,
    )
    q_exact = np.concatenate(
        [
            p.flatten()
            for p in projection_kform_l2_primal(
                flux_components(specs_q.dimension), specs_q, sm
            )
        ]
    )
    return t, t @ q_exact


# %%
# With all the small building blocks in place, the global saddle-point system
# can be assembled. The element blocks are placed on the diagonal, while the
# continuity constraints of the flux across the shared faces are added as rows
# of the constraint matrix :math:`C`, one Lagrange multiplier block per shared
# face, and the strong Neumann conditions on the selected outer faces are added
# as rows of the constraint matrix :math:`N` with the prescribed trace data.
# The weak Dirichlet condition contributes the boundary load
# :func:`compute_kform_boundary_load` to the momentum right-hand side.
#


def solve(
    mesh: Mesh,
    maps: list[SpaceMap],
    specs_q: KFormSpecs,
    specs_u: KFormSpecs,
    test_specs: KFormSpecs,
    strong: set[tuple[int, int]],
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Solve the mixed Poisson system with strong and weak boundary conditions.

    Parameters
    ----------
    strong : set of (axis, side)
        Outer boundary faces on which the flux trace is prescribed strongly.
        The remaining outer boundary faces get the weak Dirichlet condition.
    """
    nq = int(np.sum(specs_q.component_dof_counts))
    nu = int(np.sum(specs_u.component_dof_counts))
    blk = nq + nu
    nel = mesh.element_count
    nx = nel * blk

    lhs = np.zeros((nx, nx))
    rhs = np.zeros(nx)
    for e in range(nel):
        off = e * blk
        lhs[off : off + blk, off : off + blk] = assemble_element_lhs(
            maps[e], specs_q, specs_u
        )
        rhs[off + nq : off + blk] = assemble_element_rhs(specs_u, specs_q, maps[e])[nq:]

    constraints: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    n_lambda = 0
    for mdim, object_id, element_ids, _ in mesh.iterate_shared(mesh.ndim - 1):
        assert mdim == mesh.ndim - 1 and element_ids.size == 2
        t = [
            packed_to_dense(
                mesh.compute_kform_boundary_constraints(
                    test_specs, specs_q, maps[int(eid)], int(eid), int(object_id)
                ),
                specs_q,
            )
            for eid in element_ids
        ]
        constraints.append((int(element_ids[0]), int(element_ids[1]), t[0], t[1]))
        n_lambda += t[0].shape[0]

    c_matrix = np.zeros((n_lambda, nx))
    offset = 0
    for a, b, t_a, t_b in constraints:
        n_test = t_a.shape[0]
        c_matrix[offset : offset + n_test, a * blk : a * blk + nq] = t_a
        c_matrix[offset : offset + n_test, b * blk : b * blk + nq] = -t_b
        offset += n_test

    strong_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    n_strong = 0
    for mdim, object_id, element_ids, orientations in mesh.iterate_boundary(
        mesh.ndim - 1
    ):
        assert mdim == mesh.ndim - 1 and element_ids.size == 1
        element_id = int(element_ids[0])
        orient = orientations[0]
        if face_axis_side(orient) in strong:
            t, g = strong_constraint_row(
                mesh, test_specs, specs_q, maps[element_id], element_id, int(object_id)
            )
            strong_rows.append((element_id, t, g))
            n_strong += t.shape[0]
        else:
            rhs[element_id * blk : element_id * blk + nq] += compute_kform_boundary_load(
                test_specs,
                specs_q,
                maps[element_id],
                mesh.collections,
                mesh.point_count,
                element_id,
                int(object_id),
                manufactured_solution,
            )

    n_matrix = np.zeros((n_strong, nx))
    offset = 0
    for element_id, t, _ in strong_rows:
        n_test = t.shape[0]
        n_matrix[offset : offset + n_test, element_id * blk : element_id * blk + nq] = t
        offset += n_test

    system = np.block(
        [
            [lhs, c_matrix.T, n_matrix.T],
            [c_matrix, np.zeros((n_lambda, n_lambda)), np.zeros((n_lambda, n_strong))],
            [n_matrix, np.zeros((n_strong, n_lambda)), np.zeros((n_strong, n_strong))],
        ]
    )
    strong_rhs = np.concatenate([g for _, _, g in strong_rows])
    solution = np.linalg.solve(
        system, np.concatenate((rhs, np.zeros(n_lambda), strong_rhs))
    )

    q_dofs = [solution[e * blk : e * blk + nq] for e in range(nel)]
    u_dofs = [solution[e * blk + nq : e * blk + blk] for e in range(nel)]

    # Max absolute trace mismatch over all shared faces.
    continuity = max(
        float(np.max(np.abs(t_a @ q_dofs[a] - t_b @ q_dofs[b])))
        for a, b, t_a, t_b in constraints
    )
    return q_dofs, u_dofs, continuity


# %%
# To compute the :math:`L^2` error, the computed solution is reconstructed on
# every element, subtracted from the manufactured solution, and the square of
# the error is integrated over the element.
#


def reconstruct_element_error_l2(
    specs_u: KFormSpecs, u_dofs: np.ndarray, sm_h: SpaceMap
) -> float:
    """Reconstruct the solution on one element and return its squared L2 error."""
    sol_u = KForm(specs_u)
    sol_u.values[:] = u_dofs
    u_dofs_obj = DegreesOfFreedom(
        specs_u.get_component_function_space(0), sol_u.get_component_dofs(0)
    )
    computed_values = transform_kform_to_target(
        specs_u.order, sm_h, [reconstruct(u_dofs_obj, *sm_h.integration_space.nodes())]
    )[0]
    real_values = manufactured_solution(
        *[sm_h.coordinate_map(idx).values for idx in range(specs_u.order)]
    )
    return float(
        np.sum(
            (computed_values - real_values) ** 2
            * sm_h.determinant
            * sm_h.integration_space.weights()
        )
    )


# %%
# All the small building blocks discussed before can now be put together to
# form the error calculation function.
#


def compute_l2_error(
    order_integration: int,
    type_integration: IntegrationMethod,
    order_basis: int,
    type_basis: BasisType,
    dp: int,
    ndim: int,
    strong: set[tuple[int, int]],
) -> tuple[float, float, float]:
    """Solve the Poisson equation with BCs and compute the L2 error."""
    mesh = Mesh.from_corners(ndim, mesh_corners(ndim))

    integration = IntegrationSpace(
        *(IntegrationSpecs(order_integration, type_integration) for _ in range(ndim))
    )
    integration_high = IntegrationSpace(
        *(IntegrationSpecs(order_integration + dp, type_integration) for _ in range(ndim))
    )
    maps = create_element_maps(ndim, integration)
    maps_high = create_element_maps(ndim, integration_high)

    specs_u, specs_q = create_kform_specs(type_basis, order_basis, ndim)
    # The multiplier space on a shared face matches the flux trace: an
    # (ndim-1)-form on an (ndim-1)-dimensional face with the same basis type
    # and order as the element.
    test_specs = KFormSpecs(
        ndim - 1,
        FunctionSpace(*(BasisSpecs(type_basis, order_basis) for _ in range(ndim - 1))),
    )

    q_dofs, u_dofs, continuity = solve(mesh, maps, specs_q, specs_u, test_specs, strong)
    err_l2 = np.sqrt(
        sum(
            reconstruct_element_error_l2(specs_u, u_dofs[e], maps_high[e])
            for e in range(mesh.element_count)
        )
    )
    strong_residual = max(
        float(np.max(np.abs(t @ q_dofs[e] - g)))
        for e, t, g in strong_rows(mesh, maps, specs_q, test_specs, strong)
    )
    return float(err_l2), continuity, strong_residual


def strong_rows(
    mesh: Mesh,
    maps: list[SpaceMap],
    specs_q: KFormSpecs,
    test_specs: KFormSpecs,
    strong: set[tuple[int, int]],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Recompute the strong rows for residual reporting."""
    rows = []
    for mdim, object_id, element_ids, orientations in mesh.iterate_boundary(
        mesh.ndim - 1
    ):
        element_id = int(element_ids[0])
        if face_axis_side(orientations[0]) in strong:
            t, g = strong_constraint_row(
                mesh, test_specs, specs_q, maps[element_id], element_id, int(object_id)
            )
            rows.append((element_id, t, g))
    return rows


# %%
# For this test, we will use the Bernstein basis, the Gauss integration rule,
# and the order difference of 1 between the lower and higher order integration
# rules.
BTYPE = BasisType.BERNSTEIN
ITYPE = IntegrationMethod.GAUSS
DP = 1


def plot_convergence(
    pvals: npt.NDArray[np.intp], evals: npt.NDArray[np.double], title: str
) -> None:
    """Fit and plot the convergence of the L2 error over polynomial order."""
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
    ax.set_title(title)
    fig.tight_layout()

    plt.show()


# %%
# In two dimensions the grid consists of four unit squares. The error of the
# manufactured solution is measured for increasing polynomial order for two
# choices of the strong faces: the *neighboring* pair :math:`x = -1` and
# :math:`y = -1`, and the *opposite* pair :math:`x = -1` and :math:`x = +1`.
# In the mixed formulation the strong flux condition on neighboring faces
# constrains disjoint components of :math:`q` (the :math:`x`-face constrains
# the :math:`\mathrm{d}x`-component and the :math:`y`-face the
# :math:`\mathrm{d}y`-component), so neither configuration over-constrains the
# saddle-point system.
#
pvals = np.arange(1, 7)
evals_n = np.zeros(pvals.size)
evals_o = np.zeros(pvals.size)
strong_neighboring = {(0, -1), (1, -1)}
strong_opposite = {(0, -1), (0, 1)}
for ip, p in enumerate(pvals):
    p_ord = int(p)
    t0 = perf_counter()
    l2_n, cont_n, res_n = compute_l2_error(
        p_ord, ITYPE, p_ord, BTYPE, DP, 2, strong_neighboring
    )
    l2_o, cont_o, res_o = compute_l2_error(
        p_ord, ITYPE, p_ord, BTYPE, DP, 2, strong_opposite
    )
    t1 = perf_counter()
    evals_n[ip] = l2_n
    evals_o[ip] = l2_o
    print(
        f"p = {p_ord}: L2 error = {l2_n:.3e} / {l2_o:.3e} "
        f"(neighboring / opposite), strong residual = {res_n:.1e} / {res_o:.1e}, "
        f"continuity = {cont_n:.1e} / {cont_o:.1e}, time = {t1 - t0:.3f} s"
    )

plot_convergence(
    pvals,
    evals_n,
    "Poisson equation convergence, strong Neumann on x=-1, y=-1",
)
plot_convergence(
    pvals,
    evals_o,
    "Poisson equation convergence, strong Neumann on x=-1, x=+1",
)


# %%
# The same building blocks solve the three-dimensional Poisson equation on a
# 2x2x2 grid of eight hexahedral elements, covering :math:`[-1, 1]^3`, with
# strong flux conditions on the faces :math:`x = -1` and :math:`y = -1` and
# weak Dirichlet conditions on the remaining outer faces.
#
pvals_3d = np.arange(1, 6)
evals_3d = np.zeros(pvals_3d.size)
for ip, p in enumerate(pvals_3d):
    p_ord = int(p)
    t0 = perf_counter()
    l2, cont, res = compute_l2_error(
        p_ord, ITYPE, p_ord, BTYPE, DP, 3, strong_neighboring
    )
    t1 = perf_counter()
    evals_3d[ip] = l2
    print(
        f"p = {p_ord}: L2 error = {l2:.3e}, strong residual = {res:.1e}, "
        f"continuity = {cont:.1e}, time = {t1 - t0:.3f} s"
    )

plot_convergence(
    pvals_3d, evals_3d, "Multi-element (2x2x2 grid) Poisson equation convergence"
)
#
# The three-dimensional error converges exponentially as well: from
# :math:`7.5\times 10^{-2}` (p = 1) to :math:`2.5\times 10^{-4}` (p = 4), with
# the strong-condition residual and the flux-continuity residual at machine
# precision throughout.


# %%
# The highest-order three-dimensional solution is rendered with pyvista. The
# hexahedral mesh is built from the topological mesh as cubic Lagrange cells,
# whose curved edges and faces are rendered exactly, and the error against the
# manufactured solution is sampled on a uniform grid inside the deformed
# domain.
#


def hexa_mesh_vtk(mesh: Mesh) -> pv.UnstructuredGrid:
    """Build a pyvista grid of the hexahedral elements as curved Lagrange cells."""
    p = GEO_ORDER
    npts = 2 * p + 1
    nodes = np.linspace(-1.0, 1.0, npts)
    gx, gy, gz = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    xd, yd, zd = deformed_map(gx, gy, gz)
    points = np.stack(
        [xd.ravel(order="F"), yd.ravel(order="F"), zd.ravel(order="F")], axis=1
    )
    # _vtk_3d_indices maps every natural tensor-product point to its VTK local
    # index; scattering the point ids through it reorders them into the VTK
    # Lagrange-hexahedron numbering.
    vtk_idx = _vtk_3d_indices(p, p, p)
    cells = []
    for e in range(mesh.element_count):
        ix, iy, iz = (e & 1), ((e >> 1) & 1), ((e >> 2) & 1)
        a, b, c = np.meshgrid(
            np.arange(p + 1), np.arange(p + 1), np.arange(p + 1), indexing="ij"
        )
        global_ids = (
            (ix * p + a + npts * (iy * p + b + npts * (iz * p + c)))
            .ravel(order="F")
            .astype(np.intp)
        )
        vtk_order = np.empty_like(global_ids)
        vtk_order[vtk_idx] = global_ids
        cells.append(np.concatenate((np.array([(p + 1) ** 3], dtype=np.intp), vtk_order)))
    celltypes = np.full(len(cells), pv.CellType.LAGRANGE_HEXAHEDRON, dtype=np.uint8)
    return pv.UnstructuredGrid(np.concatenate(cells), celltypes, points)


def element_index(g: npt.NDArray[np.double]) -> npt.NDArray[np.intp]:
    """Grid element index of the points with grid coordinates g."""
    return np.clip(np.floor(g + 1.0), 0, 1).astype(np.intp)


def sample_error_grid(
    u_dofs: list[np.ndarray], specs_u: KFormSpecs, npts: int
) -> tuple[npt.NDArray[np.double], ...]:
    """Sample the pointwise error against the manufactured solution."""
    nodes = np.linspace(-1.0, 1.0, npts)
    gx, gy, gz = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    ex, ey, ez = (element_index(g) for g in (gx, gy, gz))
    ref = np.zeros_like(gx)
    for e, ue in enumerate(u_dofs):
        ix, iy, iz = (e & 1), ((e >> 1) & 1), ((e >> 2) & 1)
        mask = (ex == ix) & (ey == iy) & (ez == iz)
        dofs = DegreesOfFreedom(specs_u.get_component_function_space(0), ue)
        ref[mask] = reconstruct(
            dofs,
            2.0 * gx[mask] - (2 * ix - 1),
            2.0 * gy[mask] - (2 * iy - 1),
            2.0 * gz[mask] - (2 * iz - 1),
        )
    xd, yd, zd = deformed_map(gx, gy, gz)
    # The physical n-form density is the reference density divided by the
    # Jacobian determinant of the element map (math_background.rst).
    det = 0.125 * np.linalg.det(jacobian(gx, gy, gz))
    err = ref / det - manufactured_solution(xd, yd, zd)
    return xd, yd, zd, err


ORDER_VIS = 3
mesh_3d = Mesh.from_corners(3, mesh_corners(3))
integration_vis = IntegrationSpace(
    *(IntegrationSpecs(ORDER_VIS, ITYPE) for _ in range(3))
)
maps_vis = create_element_maps(3, integration_vis)
specs_u_vis, specs_q_vis = create_kform_specs(BTYPE, ORDER_VIS, 3)
test_specs_vis = KFormSpecs(
    2, FunctionSpace(*(BasisSpecs(BTYPE, ORDER_VIS) for _ in range(2)))
)
q_dofs_vis, u_dofs_vis, _ = solve(
    mesh_3d, maps_vis, specs_q_vis, specs_u_vis, test_specs_vis, strong_neighboring
)
xd_vis, yd_vis, zd_vis, err_vis = sample_error_grid(u_dofs_vis, specs_u_vis, 25)


# %%
# The error is shown as :math:`\log_{10}` of its absolute value on three
# orthogonal slices through the deformed domain, together with the edges of
# the hexahedral elements. Values below :math:`10^{-10}` of the maximum are
# clamped to the lowest color, so the colormap is not dominated by near-zero
# noise.
#
logerr = np.log10(np.abs(err_vis))
finite = np.isfinite(logerr)
hi = float(np.max(logerr[finite]))
lo = max(hi - 10.0, float(np.min(logerr[finite])))
logerr = np.clip(logerr, lo, hi)

image = pv.StructuredGrid(xd_vis, yd_vis, zd_vis)
image.point_data["logerr"] = logerr.ravel(order="F")

plotter = pv.Plotter()
for axis in range(3):
    normal = np.zeros(3)
    normal[axis] = 1.0
    plotter.add_mesh(
        cast(pv.DataSet, image.slice(normal=normal)),
        scalars="logerr",
        cmap="viridis",
        clim=(lo, hi),
        show_scalar_bar=(axis == 0),
        scalar_bar_args={"title": "log10 |u - u_exact|"},
    )
plotter.add_mesh(hexa_mesh_vtk(mesh_3d), style="wireframe", color="black")
plotter.camera_position = "iso"
plotter.show()


# %%
# The flux :math:`q` of the mixed formulation is a 2-form in three dimensions;
# on the uniform grid of this example its Hodge star is a vector field. The
# flux is transformed to the physical domain on a uniform grid of points and
# rendered as arrows, whose length and color encode its magnitude.
#


def flux_grid(
    q_dofs: list[np.ndarray], specs_q: KFormSpecs, npts: int
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Evaluate the physical flux vector on a uniform grid in the domain."""
    nodes = np.linspace(-1.0, 1.0, npts)
    gx, gy, gz = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    ex, ey, ez = (element_index(g) for g in (gx, gy, gz))
    vec = np.zeros(gx.shape + (3,))
    for e, qe in enumerate(q_dofs):
        ix, iy, iz = (e & 1), ((e >> 1) & 1), ((e >> 2) & 1)
        mask = (ex == ix) & (ey == iy) & (ez == iz)
        xi = (
            2.0 * gx[mask] - (2 * ix - 1),
            2.0 * gy[mask] - (2 * iy - 1),
            2.0 * gz[mask] - (2 * iz - 1),
        )
        g = (gx[mask], gy[mask], gz[mask])
        sol_q = KForm(specs_q)
        sol_q.values[:] = qe
        comps = []
        for c in range(specs_q.component_count):
            dofs = DegreesOfFreedom(
                specs_q.get_component_function_space(c), sol_q.get_component_dofs(c)
            )
            comps.append(reconstruct(dofs, *xi))
        # Physical components of the 2-form: the exterior power of the inverse
        # Jacobian of the element map (math_background.rst), evaluated
        # pointwise. The element map is xi -> phi(0.5 xi + idx - 0.5), so its
        # Jacobian is half the Jacobian of the grid deformation.
        Gfes = np.linalg.inv(0.5 * jacobian(*g))
        subsets = [(0, 1), (0, 2), (1, 2)]
        phys = np.zeros((3, len(gx[mask])))
        for b, (i, j) in enumerate(subsets):
            for a, (p, q) in enumerate(subsets):
                phys[b] += comps[a] * (
                    Gfes[..., p, i] * Gfes[..., q, j] - Gfes[..., p, j] * Gfes[..., q, i]
                )
        # Euclidean Hodge star of a 2-form in three dimensions:
        # star(dx1^dx2) = dx3, star(dx1^dx3) = -dx2, star(dx2^dx3) = dx1.
        vec[mask] = np.stack([phys[2], -phys[1], phys[0]], axis=1)
    px, py, pz = deformed_map(gx, gy, gz)
    origins = np.stack([px.ravel(), py.ravel(), pz.ravel()], axis=1)
    return origins, vec.reshape(-1, 3), np.linalg.norm(vec, axis=-1).ravel()


origins_vis, flux_vis, mag_vis = flux_grid(q_dofs_vis, specs_q_vis, 9)

arrows = pv.PolyData(origins_vis)
arrows["flux"] = flux_vis
arrows["mag"] = mag_vis

plotter_flux = pv.Plotter()
plotter_flux.add_mesh(
    cast(
        pv.DataSet, arrows.glyph(orient="flux", scale="mag", factor=0.25 / mag_vis.max())
    ),
    cmap="plasma",
    scalars="mag",
    scalar_bar_args={"title": "flux magnitude"},
)
plotter_flux.add_mesh(hexa_mesh_vtk(mesh_3d), style="wireframe", color="black")
plotter_flux.camera_position = "iso"
plotter_flux.show()
