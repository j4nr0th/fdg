"""Check the topological Mesh wrapper and its boundary-constraint method."""

import numpy as np
import pytest
from fdg import (
    Mesh,
    compute_kform_boundary_constraints,
    projection_kform_l2_dual,
    projection_kform_l2_primal,
    reconstruct,
    transform_kform_to_target,
)
from fdg._fdg import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KForm,
    KFormSpecs,
    SpaceMap,
    compute_kform_boundary_load,
    compute_kform_mass_matrix,
    incidence_kform_operator,
)
from fdg.enum_type import BasisType, IntegrationMethod
from fdg.integration import Integrable


def grid2(x: int, y: int) -> int:
    """Point ID of the (x, y) node of the 2x2 grid."""
    return x + 3 * y


CORNERS_2X2 = np.array(
    [
        grid2(0, 0),
        grid2(1, 0),
        grid2(0, 1),
        grid2(1, 1),  # element 0 (ix,iy)=(0,0)
        grid2(1, 0),
        grid2(2, 0),
        grid2(1, 1),
        grid2(2, 1),  # element 1 (1,0)
        grid2(0, 1),
        grid2(1, 1),
        grid2(0, 2),
        grid2(1, 2),  # element 2 (0,1)
        grid2(1, 1),
        grid2(2, 1),
        grid2(1, 2),
        grid2(2, 2),  # element 3 (1,1)
    ],
    dtype=np.uint64,
)

# Lines of the 2x2 grid: per element, x-parallel lines first, then y-parallel.
LINES_2X2 = np.array(
    [
        [0, 1],
        [3, 4],
        [0, 3],
        [1, 4],
        [1, 2],
        [4, 5],
        [2, 5],
        [6, 7],
        [3, 6],
        [4, 7],
        [7, 8],
        [5, 8],
    ],
    dtype=np.uint64,
)

# Element boundaries in {x-start, y-start, x-end, y-end} line-ID layout.
QUADS_2X2 = np.array(
    [[2, 0, 3, 1], [3, 4, 6, 5], [8, 1, 9, 7], [9, 5, 11, 10]],
    dtype=np.uint64,
)

GEOM_BASIS = FunctionSpace(
    BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
)


def _affine_map(element_id: int, integration: IntegrationSpace) -> SpaceMap:
    """Affine map of a 2x2-grid element onto its physical unit square."""
    corner = int(CORNERS_2X2[element_id * 4])
    ix, iy = corner % 3, corner // 3
    xi, eta = np.meshgrid(
        np.linspace(-1.0, 1.0, 2), np.linspace(-1.0, 1.0, 2), indexing="ij"
    )
    return SpaceMap(
        CoordinateMap(
            DegreesOfFreedom(GEOM_BASIS, (0.5 * xi + ix - 0.5).ravel()), integration
        ),
        CoordinateMap(
            DegreesOfFreedom(GEOM_BASIS, (0.5 * eta + iy - 0.5).ravel()), integration
        ),
    )


def _apply_boundary_rows(
    result: tuple[np.ndarray, ...], values: np.ndarray
) -> np.ndarray:
    """Apply packed boundary rows to one element's flattened DoFs."""
    row_offsets, _, local_dofs, coefficients = result
    return np.array(
        [
            np.dot(
                coefficients[row_offsets[row] : row_offsets[row + 1]],
                values[local_dofs[row_offsets[row] : row_offsets[row + 1]]],
            )
            for row in range(row_offsets.size - 1)
        ]
    )


def _constant_kform_values(specs: KFormSpecs) -> np.ndarray:
    """Return DoFs for a constant value in every k-form component."""
    return np.ones(int(np.sum(specs.component_dof_counts)))


def test_mesh_from_corners_2d_properties() -> None:
    """Corner-built meshes expose the correct counts and collections."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)
    assert mesh.ndim == 2
    assert mesh.point_count == 9
    assert mesh.element_count == 4

    collections = mesh.collections
    assert len(collections) == 2
    np.testing.assert_array_equal(collections[0], LINES_2X2)
    np.testing.assert_array_equal(collections[1], QUADS_2X2)


def test_mesh_iteration_counts() -> None:
    """Shared and boundary iterations report the expected objects."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)

    shared_lines = mesh.iterate_shared(1)
    assert len(shared_lines) == 4
    for mdim, _, element_ids, orientations in shared_lines:
        assert mdim == 1
        assert element_ids.size == 2
        assert orientations.shape == (2, 2)

    shared_points = mesh.iterate_shared(0)
    assert len(shared_points) == 5
    center = next(item for item in shared_points if item[2].size == 4)
    assert center[0] == 0
    assert int(center[1]) == 4
    np.testing.assert_array_equal(center[2], np.array([0, 1, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(
        center[3],
        np.array([[1, 2], [-1, 2], [1, -2], [-1, -2]], dtype=np.int8),
    )

    assert len(mesh.iterate_boundary(1)) == 8
    assert len(mesh.iterate_boundary(0)) == 8

    shared_all = mesh.iterate_shared_all()
    assert len(shared_all) == 9
    assert all(item[0] == 1 for item in shared_all[:4])
    assert all(item[0] == 0 for item in shared_all[4:])

    boundary_all = mesh.iterate_boundary_all()
    assert len(boundary_all) == 16
    assert all(item[0] == 1 for item in boundary_all[:8])
    assert all(item[0] == 0 for item in boundary_all[8:])


def test_mesh_element_object() -> None:
    """Element-local axis specifications resolve to global object IDs."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)

    # Corner point of element 0 at axis-0 start and axis-1 end: grid point (0, 1).
    assert mesh.element_object(0, [-1, 2]) == 3
    # y-start edge of element 0 is line 0.
    assert mesh.element_object(0, [0, -2]) == 0
    # The shared x-interface of elements 0 and 1 is the same line from both sides.
    assert mesh.element_object(0, [1, 0]) == mesh.element_object(1, [-1, 0])

    with pytest.raises(ValueError):
        mesh.element_object(0, [0, 0])
    with pytest.raises(ValueError):
        mesh.element_object(0, [0, -1])
    with pytest.raises(ValueError):
        mesh.element_object(4, [-1, -2])
    with pytest.raises(ValueError):
        mesh.element_object(0, [-1])


def test_mesh_from_corners_rejects_bad_input() -> None:
    """Invalid constructor arguments raise clear errors."""
    with pytest.raises(ValueError):
        Mesh.from_corners(0, CORNERS_2X2)
    with pytest.raises(ValueError):
        Mesh.from_corners(64, CORNERS_2X2)
    with pytest.raises(ValueError):
        Mesh.from_corners(2, CORNERS_2X2[:7])
    with pytest.raises(ValueError):
        Mesh.from_corners(2, np.array([], dtype=np.uint64))
    with pytest.raises(TypeError):
        Mesh()


def test_mesh_from_collections() -> None:
    """Collection-built meshes match the source collections and topology."""
    lines = np.array(
        [[0, 1], [0, 2], [2, 3], [1, 3], [2, 4], [4, 5], [3, 5]],
        dtype=np.uint64,
    )
    elements = np.array([[0, 1, 2, 3], [2, 4, 5, 6]], dtype=np.uint64)

    mesh = Mesh.from_collections(2, 6, (lines, elements))
    assert mesh.point_count == 6
    assert mesh.element_count == 2
    np.testing.assert_array_equal(mesh.collections[0], lines)
    np.testing.assert_array_equal(mesh.collections[1], elements)

    shared = mesh.iterate_shared(1)
    assert len(shared) == 1
    mdim, object_id, element_ids, orientations = shared[0]
    assert mdim == 1
    assert int(object_id) == 2
    np.testing.assert_array_equal(element_ids, np.array([0, 1], dtype=np.uint64))
    assert orientations.shape == (2, 2)


def test_mesh_boundary_constraints_matches_free_function() -> None:
    """The bound method returns bit-identical results to the free function."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)
    integration = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))
    element_specs = KFormSpecs(1, GEOM_BASIS)
    test_specs = KFormSpecs(1, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))

    shared = mesh.iterate_shared(1)[0]
    object_id = int(shared[1])
    element_id = int(shared[2][0])
    element_map = _affine_map(element_id, integration)

    bound = mesh.compute_kform_boundary_constraints(
        test_specs, element_specs, element_map, element_id, object_id
    )
    free = compute_kform_boundary_constraints(
        test_specs,
        element_specs,
        element_map,
        mesh.collections,
        mesh.point_count,
        element_id,
        object_id,
    )
    for expected, actual in zip(free, bound):
        np.testing.assert_array_equal(actual, expected)


def _grid_corners(ndim: int) -> np.ndarray:
    """Corner point IDs of all elements of the 2^ndim grid on [0, 2]^ndim."""
    corners: list[int] = []
    for element in range(1 << ndim):
        idx = tuple((element >> d) & 1 for d in range(ndim))
        for delta in range(1 << ndim):
            corners.append(sum((idx[d] + ((delta >> d) & 1)) * 3**d for d in range(ndim)))  # type: ignore
    return np.array(corners, dtype=np.uint64)


def _affine_map_ndim(
    ndim: int, element_id: int, integration: IntegrationSpace
) -> SpaceMap:
    """Affine map of a 2^ndim-grid element onto its physical unit hypercube."""
    corner = int(_grid_corners(ndim)[element_id * (1 << ndim)])
    idx = tuple((corner // 3**d) % 3 for d in range(ndim))
    grids = np.meshgrid(*([np.linspace(-1.0, 1.0, 2)] * ndim), indexing="ij")
    basis = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(ndim))
    )
    return SpaceMap(
        *(
            CoordinateMap(
                DegreesOfFreedom(basis, (0.5 * grids[d] + (idx[d] - 0.5)).ravel()),
                integration,
            )
            for d in range(ndim)
        )
    )


def _manufactured_poisson(
    ndim: int,
) -> tuple[Integrable, Integrable, list[Integrable]]:
    """Manufactured solution, source, and flux of the mixed Poisson equation.

    The flux is the Hodge star of the gradient: component ``c`` (the ``c``-th
    combination of ``ndim - 1`` axes) carries ``(-1)^m d/dx_m u`` where ``m``
    is the axis not contained in that combination.
    """
    half = np.pi / 2.0
    scale = 0.1

    def solution(*x: np.ndarray) -> np.ndarray:
        res = np.cos(x[0] * half)
        for v in x[1:]:
            res *= np.cos(v * half)
        return (res + 1.0) * scale

    def source(*x: np.ndarray) -> np.ndarray:
        res = np.cos(x[0] * half)
        for v in x[1:]:
            res *= np.cos(v * half)
        return res * (-(half**2) * ndim) * scale

    flux = []
    for missing in range(ndim):
        sign = 1.0 if missing % 2 == 0 else -1.0

        def component(missing: int = missing, sign: float = sign) -> Integrable:
            def value(*x: np.ndarray) -> np.ndarray:
                prod = np.ones_like(x[0])
                for i, xi in enumerate(x):
                    if i != missing:
                        prod *= np.cos(xi * half)
                return scale * sign * (-half * np.sin(x[missing] * half)) * prod

            return value

        flux.append(component())
    return solution, source, flux


def _packed_to_dense(
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


def _kform_specs(ndim: int, order: int) -> tuple[KFormSpecs, KFormSpecs]:
    """Create the (n-1)-form flux and n-form solution specs."""
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.BERNSTEIN, order) for _ in range(ndim))
    )
    return KFormSpecs(ndim - 1, base_space), KFormSpecs(ndim, base_space)


@pytest.mark.parametrize("ndim", range(1, 6))
def test_kform_boundary_load_chain_integral(ndim: int) -> None:
    """The boundary load sums to the element momentum boundary term.

    The momentum equation of the mixed formulation is
    ``(p, q) + (dp, u) = int_{dOmega} p wedge star u``. For the constant
    n-form ``u = 1 dV``, which the discrete space represents exactly on an
    affine element, the term ``(dp, u)`` reduces to the chain integral of
    ``p`` over the element boundary, so the sum of the boundary loads of all
    faces of one element must equal ``et_mu @ u_const`` exactly. This pins
    the normalization and the outward-orientation sign of the load in every
    dimension.
    """
    mesh = Mesh.from_corners(ndim, _grid_corners(ndim))
    integration = IntegrationSpace(
        *(IntegrationSpecs(3, IntegrationMethod.GAUSS) for _ in range(ndim))
    )
    sm = _affine_map_ndim(ndim, 0, integration)
    specs_q, specs_u = _kform_specs(ndim, 1)
    test_specs = KFormSpecs(
        ndim - 1,
        FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, 1) for _ in range(ndim - 1))),
    )

    u_const = projection_kform_l2_primal([lambda *x: np.ones_like(x[0])], specs_u, sm)[
        0
    ].flatten()
    mu = compute_kform_mass_matrix(
        sm, specs_u.order, specs_u.base_space, specs_u.base_space
    )
    et_mu = incidence_kform_operator(specs_q, mu, transpose=True)
    expected = et_mu @ u_const

    actual = np.zeros(int(np.sum(specs_q.component_dof_counts)))
    for mdim, object_id, element_ids, _ in mesh.iterate_boundary(ndim - 1):
        if 0 in element_ids:
            actual += compute_kform_boundary_load(
                test_specs,
                specs_q,
                sm,
                mesh.collections,
                mesh.point_count,
                0,
                int(object_id),
                lambda *x: np.ones_like(x[0]),
            )
    for mdim, object_id, element_ids, _ in mesh.iterate_shared(ndim - 1):
        if 0 in element_ids:
            actual += compute_kform_boundary_load(
                test_specs,
                specs_q,
                sm,
                mesh.collections,
                mesh.point_count,
                0,
                int(object_id),
                lambda *x: np.ones_like(x[0]),
            )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("ndim", range(1, 6))
def test_kform_boundary_constraints_strong_weak_solve(ndim: int) -> None:
    """Strong Neumann and weak Dirichlet BCs solve the mixed Poisson system.

    The flux trace is prescribed strongly on the faces with fixed axis 0 at
    the start side and axis 1 at the start side; the solution is prescribed
    weakly on the remaining outer faces. The saddle-point system must be
    square and full rank, the strong-Neumann and continuity residuals must
    vanish, and the L2 error must be finite.
    """
    mesh = Mesh.from_corners(ndim, _grid_corners(ndim))
    order = 1
    integration = IntegrationSpace(
        *(IntegrationSpecs(order + 1, IntegrationMethod.GAUSS) for _ in range(ndim))
    )
    maps = [_affine_map_ndim(ndim, e, integration) for e in range(1 << ndim)]
    specs_q, specs_u = _kform_specs(ndim, order)
    test_specs = KFormSpecs(
        ndim - 1,
        FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, order) for _ in range(ndim - 1))),
    )
    solution, source, flux = _manufactured_poisson(ndim)

    nq = int(np.sum(specs_q.component_dof_counts))
    nu = int(np.sum(specs_u.component_dof_counts))
    blk = nq + nu
    nel = 1 << ndim
    lhs = np.zeros((nel * blk, nel * blk))
    rhs = np.zeros(nel * blk)
    for e in range(nel):
        sm = maps[e]
        mq = compute_kform_mass_matrix(
            sm, specs_q.order, specs_q.base_space, specs_q.base_space
        )
        mu = compute_kform_mass_matrix(
            sm, specs_u.order, specs_u.base_space, specs_u.base_space
        )
        mu_e = incidence_kform_operator(specs_q, mu, right=True)
        et_mu = incidence_kform_operator(specs_q, mu, transpose=True)
        off = e * blk
        lhs[off : off + blk, off : off + blk] = np.block(
            [
                [mq, et_mu],
                [mu_e, np.zeros_like(mu)],
            ]
        )
        rhs[off + nq : off + blk] = projection_kform_l2_dual([source], specs_u, sm)[
            0
        ].flatten()

    c_rows = []
    for mdim, object_id, element_ids, _ in mesh.iterate_shared(ndim - 1):
        element_ids = element_ids.tolist()
        t = [
            _packed_to_dense(
                mesh.compute_kform_boundary_constraints(
                    test_specs, specs_q, maps[int(eid)], int(eid), int(object_id)
                ),
                specs_q,
            )
            for eid in element_ids
        ]
        c_rows.append((int(element_ids[0]), int(element_ids[1]), t[0], t[1]))

    strong_rows = []
    for mdim, object_id, element_ids, orientations in mesh.iterate_boundary(ndim - 1):
        element_id = int(element_ids[0])
        orient = orientations[0]
        axis = abs(int(orient[0])) - 1
        side = -1 if orient[0] < 0 else 1
        strong = (axis == 0 and side < 0) or (axis == 1 and side < 0)
        if strong:
            t = _packed_to_dense(
                mesh.compute_kform_boundary_constraints(
                    test_specs, specs_q, maps[element_id], element_id, int(object_id)
                ),
                specs_q,
            )
            q_exact = np.concatenate(
                [
                    p.flatten()
                    for p in projection_kform_l2_primal(flux, specs_q, maps[element_id])
                ]
            )
            strong_rows.append((element_id, t, t @ q_exact))
        else:
            rhs[element_id * blk : element_id * blk + nq] += compute_kform_boundary_load(
                test_specs,
                specs_q,
                maps[element_id],
                mesh.collections,
                mesh.point_count,
                element_id,
                int(object_id),
                solution,
            )

    n_lambda = sum(ta.shape[0] for _, _, ta, _ in c_rows)
    n_strong = sum(t.shape[0] for _, t, _ in strong_rows)
    nx = nel * blk
    system = np.zeros((nx + n_lambda + n_strong, nx + n_lambda + n_strong))
    system[:nx, :nx] = lhs
    row = 0
    for a, b, ta, tb in c_rows:
        n = ta.shape[0]
        system[nx + row : nx + row + n, a * blk : a * blk + nq] = ta
        system[nx + row : nx + row + n, b * blk : b * blk + nq] = -tb
        system[a * blk : a * blk + nq, nx + row : nx + row + n] = ta.T
        system[b * blk : b * blk + nq, nx + row : nx + row + n] = -tb.T
        row += n
    for e, t, _ in strong_rows:
        n = t.shape[0]
        system[nx + row : nx + row + n, e * blk : e * blk + nq] = t
        system[e * blk : e * blk + nq, nx + row : nx + row + n] = t.T
        row += n
    b_rhs = np.concatenate(
        (
            rhs,
            np.zeros(n_lambda),
            np.concatenate([g for _, _, g in strong_rows]),
        )
    )

    assert system.shape[0] == system.shape[1]
    assert np.linalg.matrix_rank(system) == system.shape[0]
    solution_vec = np.linalg.solve(system, b_rhs)
    q_dofs = [solution_vec[e * blk : e * blk + nq] for e in range(nel)]
    u_dofs = [solution_vec[e * blk + nq : e * blk + blk] for e in range(nel)]

    for e, t, g in strong_rows:
        np.testing.assert_allclose(t @ q_dofs[e], g, rtol=1e-10, atol=1e-10)
    for a, b, ta, tb in c_rows:
        np.testing.assert_allclose(ta @ q_dofs[a], tb @ q_dofs[b], rtol=1e-10, atol=1e-10)

    err = 0.0
    for e in range(nel):
        sm = maps[e]
        sol_u = KForm(specs_u)
        sol_u.values[:] = u_dofs[e]
        dof_obj = DegreesOfFreedom(
            specs_u.get_component_function_space(0), sol_u.get_component_dofs(0)
        )
        computed = transform_kform_to_target(
            specs_u.order, sm, [reconstruct(dof_obj, *sm.integration_space.nodes())]
        )[0]
        real = solution(*[sm.coordinate_map(i).values for i in range(specs_u.order)])
        err += np.sum(
            (computed - real) ** 2 * sm.determinant * sm.integration_space.weights()
        )
    assert float(np.sqrt(err)) < 0.3


def _single_element_corners(ndim: int) -> np.ndarray:
    """Corner point IDs of one element of the 2**ndim grid."""
    return np.array(
        [
            sum(((delta >> d) & 1) * 3**d for d in range(ndim))
            for delta in range(1 << ndim)
        ],
        dtype=np.uint64,
    )


def _element_map_ndim(
    ndim: int, integration: IntegrationSpace, deformed: bool
) -> SpaceMap:
    """Affine (deformed=False) or cubic boundary-deforming element map."""
    order = 3 if deformed else 1
    basis = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, order) for _ in range(ndim))
    )
    grid = np.meshgrid(*([np.linspace(-1.0, 1.0, order + 1)] * ndim), indexing="ij")
    deform = 0.05
    dofs = []
    for d in range(ndim):
        values = grid[d].astype(float)
        if deformed:
            plus = np.ones_like(values)
            minus = np.ones_like(values)
            for g in grid:
                plus *= 1.0 + g
                minus *= 1.0 - g
            values = values + deform * (plus + minus)
        dofs.append(DegreesOfFreedom(basis, values.ravel()))
    return SpaceMap(*(CoordinateMap(dofs[d], integration) for d in range(ndim)))


def _bernstein(deg: int, index: int, x):
    """Bernstein polynomial on [-1, 1], zero for index > deg."""
    from math import comb

    t = np.asarray(x, dtype=float)
    if index > deg:
        return np.zeros_like(t)
    return (
        comb(deg, index) * ((1.0 + t) / 2.0) ** index * ((1.0 - t) / 2.0) ** (deg - index)
    )


def _boundary_load_replica(
    specs_q: KFormSpecs,
    test_specs: KFormSpecs,
    orientation: list[int],
    xi_el: np.ndarray,
    weights: np.ndarray,
    data_values: np.ndarray,
) -> np.ndarray:
    """Independent assembly of the documented chain integral.

    Reproduces ``compute_kform_boundary_load`` without any library call:
    ``b[j] = s (-1)^a sum_p w_p data_p prod_a B_ja(xa)``, where the per-axis
    degrees are the volume orders reduced by one along the axes of the mapped
    flux component, digits run over sizes ``P + 1`` (inactive axis) or ``P``
    (active axis), and the last axis is fastest.
    """
    from itertools import combinations as _combinations

    ndim = xi_el.shape[0]
    form_degree = test_specs.order
    orders = [b.order for b in specs_q.base_space.basis_specs]
    free_recs = orientation[1:]
    sign = 1
    mapped = []
    for i in range(form_degree):
        rec = free_recs[i]
        axis = abs(rec) - 1
        mapped.append(axis)
        if rec < 0:
            sign = -sign
    for i in range(form_degree):
        for j in range(i + 1, form_degree):
            if mapped[i] > mapped[j]:
                sign = -sign
                mapped[i], mapped[j] = mapped[j], mapped[i]
    component = list(_combinations(range(ndim), form_degree)).index(tuple(mapped))
    sl = specs_q.get_component_slice(component)
    sigma = (1.0 if orientation[0] > 0 else -1.0) * (
        1.0 if (abs(orientation[0]) - 1) % 2 == 0 else -1.0
    )
    sizes = [orders[a] + (0 if a in set(mapped) else 1) for a in range(ndim)]
    mapped_set = set(mapped)
    npts = xi_el.shape[1]
    coefficients = np.zeros(sl.stop - sl.start)

    def walk(digits: list[int]) -> None:
        if len(digits) == ndim:
            values = np.ones(npts)
            for a in range(ndim):
                deg = orders[a] - (1 if a in mapped_set else 0)
                values = values * _bernstein(deg, digits[a], xi_el[a])
            dof = 0
            factor = 1
            for a in range(ndim - 1, -1, -1):
                dof += digits[a] * factor
                factor *= sizes[a]
            coefficients[dof] += sign * sigma * float(weights @ (data_values * values))
            return
        for digit in range(sizes[len(digits)]):
            walk(digits + [digit])

    walk([])
    result = np.zeros(int(np.sum(specs_q.component_dof_counts)))
    result[sl] += coefficients
    return result


@pytest.mark.parametrize("ndim", range(2, 6))
@pytest.mark.parametrize("deformed", [False, True])
def test_kform_boundary_load_round_trip(ndim: int, deformed: bool) -> None:
    """The weak-boundary machinery round-trips discrete face data exactly.

    Three identities are checked on every outer face of a single element:

    1. ``compute_kform_boundary_load`` equals an independent assembly of the
       documented chain integral (Bernstein products with trimmed degrees and
       outward-orientation signs), evaluated against an analytic datum.
    2. The library boundary mass matrix of the face equals the explicit
       quadrature Gram ``sum_p w_p / det_p N_i N_j`` (the top-form pairing on
       a mapped face carries the inverse surface Jacobian).
    3. Random face degrees of freedom are recovered from their explicit
       moments through that mass matrix: ``solve(M, moments) == m``, closing
       the "L2 projection then mass solve" round trip.
    """
    mesh = Mesh.from_corners(ndim, _single_element_corners(ndim))
    order = 3
    integration = IntegrationSpace(
        *(IntegrationSpecs(order + 2, IntegrationMethod.GAUSS) for _ in range(ndim))
    )
    sm = _element_map_ndim(ndim, integration, deformed)
    base_vol = FunctionSpace(
        *(BasisSpecs(BasisType.BERNSTEIN, order) for _ in range(ndim))
    )
    specs_q = KFormSpecs(ndim - 1, base_vol)
    test_specs = KFormSpecs(
        ndim - 1,
        FunctionSpace(
            *(BasisSpecs(BasisType.BERNSTEIN, order - 1) for _ in range(ndim - 1))
        ),
    )
    test_space = test_specs.get_component_function_space(0)
    rng = np.random.default_rng(31 + ndim)

    def analytic(*x: np.ndarray) -> np.ndarray:
        out = np.cos(np.pi * x[0] / 2.0)
        for v in x[1:]:
            out = out * np.cos(np.pi * v / 2.0)
        return 0.25 * out

    err_chain = err_mass = err_recover = 0.0
    seen: set[tuple[int, bool]] = set()
    for _, object_id, _, orientations in mesh.iterate_boundary(ndim - 1):
        rec = [int(v) for v in orientations[0]]
        key = (abs(rec[0]) - 1, rec[0] < 0)
        if key in seen:
            continue
        seen.add(key)
        axis, start_side = abs(rec[0]) - 1, rec[0] < 0
        fm = sm.boundary(axis, end=not start_side)
        nodes = np.asarray(fm.integration_space.nodes()).reshape(ndim - 1, -1)
        weights = np.asarray(fm.integration_space.weights()).reshape(-1)
        det = np.abs(np.asarray(fm.determinant)).reshape(-1)
        physical = np.stack(
            [np.asarray(fm.coordinate_map(i).values).reshape(-1) for i in range(ndim)]
        )
        xi_el = np.empty((ndim, nodes.shape[1]))
        xi_el[axis] = -1.0 if start_side else 1.0
        for j in range(ndim - 1):
            el_axis = abs(rec[1 + j]) - 1
            xi_el[el_axis] = nodes[j] if rec[1 + j] > 0 else -nodes[j]

        data_values = np.asarray(analytic(*physical)).reshape(-1)
        replicated = _boundary_load_replica(
            specs_q, test_specs, rec, xi_el, weights, data_values
        )
        loaded = compute_kform_boundary_load(
            test_specs,
            specs_q,
            sm,
            mesh.collections,
            mesh.point_count,
            0,
            int(object_id),
            analytic,
        )
        err_chain = max(err_chain, float(np.max(np.abs(replicated - loaded))))

        n_dofs = int(np.sum(test_specs.component_dof_counts))
        eye = np.eye(n_dofs)
        rows = np.stack(
            [
                np.asarray(
                    reconstruct(DegreesOfFreedom(test_space, eye[i]), *nodes)
                ).reshape(-1)
                for i in range(n_dofs)
            ]
        )
        gram_manual = ((weights / det) * rows) @ rows.T
        gram_library = np.asarray(
            compute_kform_mass_matrix(
                fm, test_specs.order, test_specs.base_space, test_specs.base_space
            )
        )
        err_mass = max(err_mass, float(np.max(np.abs(gram_library - gram_manual))))

        coeffs = rng.standard_normal(n_dofs)
        moments = ((weights / det) * (coeffs @ rows)) @ rows.T
        recovered = np.linalg.solve(gram_library, moments)
        err_recover = max(err_recover, float(np.max(np.abs(recovered - coeffs))))

    assert err_chain < 1e-12
    assert err_mass < 1e-12
    assert err_recover < 1e-10
