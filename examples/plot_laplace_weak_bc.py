r"""
.. currentmodule:: fdg

Mixed Laplace equation with fully-weak Dirichlet boundary conditions.
=====================================================================

Solves the mixed Laplace equation on a single cubic element in three
dimensions with a *fully weak* Dirichlet boundary condition: the natural
boundary term :math:`B(p, u_D) = \int_{\partial\omega} p \wedge \star u_D`
prescribes the datum on **every** outer face through the right-hand side of
the momentum equation; no Lagrange multipliers are used anywhere.

The mixed system reads

.. math::

    (p, q) + (\mathrm{d}p, u) = B(p, u_D) \quad
    (v, \mathrm{d}q) = 0,

with the flux :math:`q` an :math:`(n-1)`-form and the solution :math:`u` an
:math:`n`-form. For Laplace the source vanishes (:math:`f = 0`) and the datum
gives :math:`u_D`. The boundary term is assembled with
:func:`compute_kform_boundary_load`, the general boundary-load interface: the
datum is passed as one callable per element-frame :math:`k`-form component
(here :math:`k = n`, a single component), each sampled at the canonical face
points. Because the load is a metric-free chain integral of the
reference-frame datum, no exactness is expected for arbitrary data at finite
order — but since the weak Dirichlet condition converges to its strong
counterpart, the computed solution approaches the harmonic extension of the
boundary data as the order :math:`p` increases.

Four geometries are compared on the reference cube :math:`[-1, 1]^3`, applied
as diagonal space maps:

1. the identity map (no metric contribution),
2. mirror maps that flip one or more axes, so the map Jacobian determinant is
   exactly :math:`-1` on every face while :math:`|\det J|` stays unity,
3. anisotropic scaling by three different positive factors,
4. a curved boundary-deforming map, where metric terms become point-dependent.

The convergence of the :math:`L^2` error against the harmonic extension of
the datum is reported for every geometry.
"""  # noqa: D205 D400

from time import perf_counter

import numpy as np
import pyvista as pv
import scipy.linalg
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
    reconstruct,
    transform_kform_to_target,
)
from fdg.domains import _vtk_3d_indices
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# %%
# Boundary data
# -------------
#
# The datum must be given as physical values of the potential. For the pure
# Laplace problem (:math:`f = 0`) the solution is the harmonic extension of
# the datum, so the datum itself is only recovered exactly when it is
# harmonic *and* lies in the discrete space. Use the fully three-dimensional
# harmonic field
#
# .. math::
#
#     u_D = \exp(x_0 / \sqrt{2})\,\cos(x_1 / 2)\,\cos(x_2 / 2),
#
# whose Laplacian vanishes identically (the separated frequencies satisfy
# :math:`c_0^2 = c_1^2 + c_2^2`). It is smooth but not polynomial, so the
# convergence under :math:`p`-refinement is exponential yet gradual, and no
# geometry reaches machine precision within the swept order range.

GEO_ORDER = 4
DEFORM = 0.05
NDIM = 3


def solution(*x):
    """Return the harmonic datum evaluated at physical ``x``."""
    return np.exp(x[0] / np.sqrt(2.0)) * np.cos(x[1] / 2.0) * np.cos(x[2] / 2.0)


# %%
# Geometry maps
# -------------
# All maps are maps of the reference cube onto the physical element. The four
# stages below are collected into callables returning the per-axis coordinate
# grids given the reference grid.


def _diag_coords(M):
    def coords(g):
        return tuple(float(M[d, d]) * g[d] for d in range(3))

    return coords


def curved_coords(g):
    """Displace all three axes by the smooth deformation bump."""
    plus = np.ones_like(g[0])
    minus = np.ones_like(g[0])
    for v in g:
        plus = plus * (1.0 + v)
        minus = minus * (1.0 - v)
    return tuple(g[d] + DEFORM * (plus + minus) for d in range(3))


GEOMETRIES = {
    "identity": _diag_coords(np.eye(3)),
    "flip-x (det = -1)": _diag_coords(np.diag([-1.0, 1.0, 1.0])),
    "flip-xyz (det = -1)": _diag_coords(np.diag([-1.0, -1.0, -1.0])),
    "flip-xy (det = +1)": _diag_coords(np.diag([-1.0, -1.0, 1.0])),
    "scaled (2, 3/2, 1/2)": _diag_coords(np.diag([2.0, 1.5, 0.5])),
    "curved": curved_coords,
}


# %%
# Element construction
# --------------------
# The mesh is a single element whose corner IDs cover the standard
# :math:`3^3` lattice block. A cubic Lagrange map realizes each geometry; its
# per-axis degrees of freedom store the mapped coordinates of the grid nodes.


corners = np.array(
    [sum(((d >> k) & 1) * 3**k for k in range(NDIM)) for d in range(8)],
    dtype=np.uint64,
)
mesh = Mesh.from_corners(3, corners)


def make_element_map(integ: IntegrationSpace, coords, geo_order: int) -> SpaceMap:
    """Build a cubic Lagrange element map from a coordinate callable."""
    nodes = np.linspace(-1.0, 1.0, geo_order + 1)
    grid = np.meshgrid(*([nodes] * 3), indexing="ij")
    basis = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, geo_order) for _ in range(3))
    )
    xs, ys, zs = coords(grid)
    dofs = [
        CoordinateMap(DegreesOfFreedom(basis, comp.ravel()), integ)
        for comp in (xs.copy(), ys.copy(), zs.copy())
    ]
    return SpaceMap(dofs[0], dofs[1], dofs[2])


# %%
# The mixed Laplace solver
# ------------------------
# The saddle-point system couples the flux and the solution of one element.
# The right-hand side receives the fully-weak Dirichlet load of every outer
# face, assembled with :func:`compute_kform_boundary_load`. This is the
# special case :math:`k = n` of the general boundary load: the datum is an
# :math:`n`-form with a single element-frame component, and the load pairs it
# against the trace of the :math:`(n-1)`-form flux test basis through the
# metric-free chain integral. The datum is still passed as the uniform
# *component list* the general interface expects — one callable per
# :math:`k`-form component (here exactly one) — so the same call pattern
# carries over verbatim to lower-degree data (e.g. a vector-flux Dirichlet
# condition in a vector Laplace solver). The reference-domain weights match
# the integration-by-parts structure of the incidence operators exactly, on
# any space map (identity, mirrors, scalings, and curved maps alike).
#
# The saddle matrix [[mq, et^T], [et, 0]] is symmetric with an
# invertible mq block and a zero right-hand side in the second block
# row, so the system reduces through the Schur complement of mq:
#
#     (et mq^{-1} et^T) u = et mq^{-1} b,
#     q = mq^{-1} (b - et^T u).
#
# Only the (smaller, symmetric) Schur complement and mq need dense
# factorizations, which is markedly cheaper than factoring the full
# saddle matrix.
#
# To solve this, Cholesky factorization cannot be used, as for some space
# maps the mass matrix is not SPD (it will be negative definite when
# determinant < 0). Instead we just fall back on using the LU factorization.


def solve_laplace(
    mesh: Mesh,
    space_map: SpaceMap,
    specs_q: KFormSpecs,
    specs_u: KFormSpecs,
    specs_test: KFormSpecs,
    gfun,
) -> tuple[KForm, KForm]:
    """Assemble and solve the fully-weak mixed Laplace system of one element."""
    nq = int(np.sum(specs_q.component_dof_counts))
    mq = np.asarray(
        compute_kform_mass_matrix(
            space_map, specs_q.order, specs_q.base_space, specs_q.base_space
        )
    )
    mu = np.asarray(
        compute_kform_mass_matrix(
            space_map, specs_u.order, specs_u.base_space, specs_u.base_space
        )
    )
    et = np.asarray(incidence_kform_operator(specs_q, mu, right=True))
    b = np.zeros(nq)
    for mdim, oid, eids, orientations in mesh.iterate_boundary(NDIM - 1):
        b += compute_kform_boundary_load(
            specs_test,
            specs_q,
            space_map,
            mesh.collections,
            mesh.point_count,
            0,
            int(oid),
            [gfun],  # one callable per k-form component: here the single n-form component
            surface_measure=False,
        )
    factor = scipy.linalg.lu_factor(mq)
    rhs_tilde = scipy.linalg.lu_solve(factor, b)
    # The Schur complement S = et mq^{-1} et^T; LU solving against the matrix
    # et.T yields all its columns at once.
    schur = et @ scipy.linalg.lu_solve(factor, et.T)
    u = KForm(specs_u)
    u.values[:] = scipy.linalg.lu_solve(scipy.linalg.lu_factor(schur), et @ rhs_tilde)
    q = KForm(specs_q)
    q.values[:] = scipy.linalg.lu_solve(factor, b - et.T @ u.values.flatten())
    return q, u


# %%
# Error measurement
# -----------------
# The datum is harmonic, so the exact state equals the datum itself. Since
# the datum is not polynomial, the error decays exponentially in the order
# without ever reaching machine precision within the swept range.


def l2_error_u(u: KForm, sm_h: SpaceMap, ref_fn) -> float:
    """Return the L2 difference between the computed solution and ``ref_fn``."""
    nodes = sm_h.integration_space.nodes()
    dof_obj = u.get_component(0)
    ur = reconstruct(dof_obj, *nodes)
    up = transform_kform_to_target(u.specs.order, sm_h, [ur])[0]
    real = ref_fn(*[np.asarray(sm_h.coordinate_map(d).values) for d in range(3)])
    weights = sm_h.integration_space.weights()
    det = np.abs(sm_h.determinant)
    diff = np.asarray(up) - real
    return float(np.sqrt(np.sum(diff**2 * det * weights)))


# %%
# Convergence study over every geometry
# -------------------------------------
# Each stage solves the same Laplace problem on its own geometry and prints
# the :math:`L^2` error of the solution against the datum. Order increments
# follow the refinements used in the other examples.

pvals = np.arange(1, 6)
results = {}
for label, coords in GEOMETRIES.items():
    evals = []
    t_first = perf_counter()
    for P in pvals:
        integ = IntegrationSpace(
            *(IntegrationSpecs(P + 6, IntegrationMethod.GAUSS) for _ in range(3))
        )
        sm = make_element_map(integ, coords, GEO_ORDER)
        bv = FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, P) for _ in range(3)))
        sq = KFormSpecs(2, bv)
        su = KFormSpecs(3, bv)
        ts = KFormSpecs(
            2, FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, P) for _ in range(2)))
        )
        _, u = solve_laplace(mesh, sm, sq, su, ts, solution)
        evals.append(l2_error_u(u, sm, solution))
        elapsed = perf_counter() - t_first
        print(f"   P={P}: L2 = {evals[-1]:.3e}   ({elapsed:.1f}s)", flush=True)
    results[label] = evals


# %%
# Convergence figure
# ------------------

fig, ax = plt.subplots()
for label, evals in results.items():
    ax.plot(pvals, np.asarray(evals), marker="o", label=label)
ax.set_yscale("log")
ax.set_xlabel("order $p$")
ax.set_ylabel(r"$\|u - u_D\|_{L^2}$")
ax.grid(True)
ax.legend()
fig.tight_layout()

# %%
# Rendering of the curved stage
# -----------------------------
# The highest-order curved solution is rendered on the *physical* geometry as
# a single VTK Lagrange hexahedron: the cell points are the element map
# evaluated at the uniform reference nodes that VTK expects, reordered into
# VTK's local numbering with ``_vtk_3d_indices``, and the solution is
# reconstructed at those same reference nodes.
#
# The VTK Lagrange hexahedron of order p expects the (p+1)^3 cell points
# at the uniform reference nodes, permuted into VTK's local numbering by
# ``_vtk_3d_indices``. Evaluate both the element map and the solution at
# exactly those nodes: the map is stored as Lagrange-uniform degrees of
# freedom, so ``reconstruct`` evaluates it in closed form anywhere.
#
# The physical n-form density is the reference density divided by the map
# Jacobian determinant (math_background.rst). The determinant is computed
# from a central-difference Jacobian of the mapped coordinates on a fine
# uniform grid, then interpolated onto the VTK nodes.


def render_curved(p: int, dp_int: int, dp_plot: int, geom) -> None:
    """Render the curved-stage solution on the deformed element."""
    integ = IntegrationSpace(
        *(IntegrationSpecs(p + dp_int, IntegrationMethod.GAUSS) for _ in range(3))
    )
    space_map = make_element_map(integ, geom, GEO_ORDER)
    func_space = FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, p) for _ in range(3)))
    _, u = solve_laplace(
        mesh=mesh,
        space_map=space_map,
        specs_q=KFormSpecs(2, func_space),
        specs_u=KFormSpecs(3, func_space),
        specs_test=KFormSpecs(
            2, FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, p) for _ in range(2)))
        ),
        gfun=solution,
    )
    dof_obj = u.get_component(0)
    # Create the reconstruction grid and reconstruct the values and positions.
    ref_grid = np.meshgrid(
        *(np.linspace(-1.0, 1.0, p + dp_plot) for _ in range(3)), indexing="ij"
    )
    u_ref = np.asarray(reconstruct(dof_obj, *ref_grid))
    # u_phys = transform_kform_to_target(u.specs.order, space_map, [u_ref])[0]
    # points = np.array(geom(ref_grid))

    geo_grid = np.meshgrid(*([np.linspace(-1.0, 1.0, GEO_ORDER + 1)] * 3), indexing="ij")
    basis_geo = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER) for _ in range(3))
    )
    map_dofs = [
        DegreesOfFreedom(basis_geo, np.asarray(v).ravel()) for v in geom(geo_grid)
    ]
    physical = [np.asarray(reconstruct(dof, *ref_grid)) for dof in map_dofs]
    points = np.stack([physical[d].ravel(order="F") for d in range(3)], axis=1)
    # Push the reference coordinates through the map to get physical coordinates.
    fine_axes = [np.linspace(-1.0, 1.0, 65) for _ in range(3)]
    fine_grid = np.meshgrid(*fine_axes, indexing="ij")
    fine_coords = geom(fine_grid)
    jac = np.empty((3, 3) + fine_grid[0].shape)
    for a in range(3):
        for b in range(3):
            jac[a, b] = np.gradient(fine_coords[a], fine_axes[b], axis=b, edge_order=2)
    det_fine = np.linalg.det(np.moveaxis(jac, (0, 1), (-2, -1)))
    det = RegularGridInterpolator(fine_axes, det_fine)(
        np.stack([g.reshape(-1) for g in ref_grid], axis=-1)
    ).reshape(ref_grid[0].shape)
    u_phys = u_ref / det
    # Scatter the natural tensor-product data into VTK's point ordering.
    n_points = points.shape[0]
    vtk_order = np.empty(n_points, dtype=np.intp)
    vtk_order[_vtk_3d_indices(*(s - 1 for s in u_phys.shape))] = np.arange(
        n_points, dtype=np.intp
    )
    cells = np.concatenate((np.array([n_points], dtype=np.intp), vtk_order))
    celltypes = np.array([pv.CellType.LAGRANGE_HEXAHEDRON], dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.point_data["u"] = u_phys.flatten()
    grid.plot(cmap="viridis")


render_curved(p=3, dp_int=6, dp_plot=10, geom=GEOMETRIES["curved"])
