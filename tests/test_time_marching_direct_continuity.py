"""Tests for marching continuity-coupled fields through the direct map.

The direct map couples the elements' degrees of freedom through the shared
objects' unknowns, so the mass matrix of a march is no longer block diagonal:
it is the congruence of the per-element masses through the transfer. These
tests assemble that matrix with :func:`fdg.scatter_csc`, march with it, and
check the convergence and geometric properties the marcher promises.
"""

from __future__ import annotations

import numpy as np
from fdg import (
    BasisSpecs,
    BasisType,
    DirectDofMap,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    Mesh,
    MovingMesh,
    SpaceMap,
    march,
    projection_l2_primal,
    scatter_csc,
    stage_mass,
)
from fdg.moving_mesh import lie_derivative_operator
from hybsol import refined_solve

from examples.plot_direct_continuity_sparse import (
    block_owners,
    build_hybsol_system,
)
from examples.plot_multi_element_laplace_direct_continuity import (
    element_entries,
    make_element_maps,
    make_mesh,
)

CELLS = 2
ORDER = 4
ORDER_INTEGRATION = 2 * ORDER
GEO_ORDER = 2
DT = 0.1
SPEED = 0.3
VELOCITY = 0.5


def _field_setup() -> tuple[
    FunctionSpace, KFormSpecs, Mesh, DirectDofMap, list[SpaceMap]
]:
    """Return base space, specs, mesh, transfer, and element maps."""
    base_space = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, ORDER))
    specs = KFormSpecs(0, base_space)
    mesh = make_mesh(1, CELLS)
    transfer = mesh.compute_kform_direct_dof_map([specs] * CELLS)
    maps = make_element_maps(1, ORDER_INTEGRATION, CELLS)
    return base_space, specs, mesh, transfer, maps


def _integration() -> IntegrationSpace:
    """Return the shared integration space of the fields."""
    return IntegrationSpace(IntegrationSpecs(ORDER_INTEGRATION, IntegrationMethod.GAUSS))


def _element_view(transfer: DirectDofMap, y: np.ndarray, element: int) -> np.ndarray:
    """Gather one element's degrees of freedom from the global vector."""
    first = int(transfer.element_offsets[element])
    last = int(transfer.element_offsets[element + 1])
    offsets = np.asarray(transfer.entry_offsets[first : last + 1], np.intp)
    start = int(offsets[0])
    stop = int(offsets[-1])
    index = np.asarray(transfer.entry_index[start:stop], np.intp)
    values = np.asarray(transfer.entry_value[start:stop], np.double)
    contributions = values * y[index]
    return np.add.reduceat(contributions, offsets[:-1] - start)


def _global_scatter(
    transfer: DirectDofMap, element_vectors: list[np.ndarray]
) -> np.ndarray:
    """Scatter element vectors onto the global degrees of freedom."""
    result = np.zeros(int(transfer.global_dof_count))
    for element, y_e in enumerate(element_vectors):
        local, index, values = element_entries(transfer, element)
        np.add.at(result, index, values * y_e[local])
    return result


def _transfer_matrix(transfer: DirectDofMap) -> np.ndarray:
    """Return the stacked element-to-global transfer as a dense matrix."""
    matrix = np.zeros((int(transfer.element_dof_count), int(transfer.global_dof_count)))
    row = 0
    for element in range(CELLS):
        local, index, values = element_entries(transfer, element)
        np.add.at(matrix, (local + row, index), values)
        row += int(transfer.element_offsets[element + 1]) - int(
            transfer.element_offsets[element]
        )
    return matrix


def _project(function, maps: list[SpaceMap], base_space: FunctionSpace) -> np.ndarray:
    """Return the stacked element projections of a physical function."""
    return np.concatenate(
        [projection_l2_primal(function, base_space, smap).values for smap in maps]
    )


def _coupled_mass(transfer: DirectDofMap, specs: KFormSpecs, maps: list) -> np.ndarray:
    """Return the dense constrained mass of the element masses."""
    flat = np.concatenate([stage_mass(smap, specs).ravel() for smap in maps])
    return np.asarray(scatter_csc(transfer, flat).toarray())


def test_scatter_constrained_mass() -> None:
    """The scattered mass is the congruence of the per-element masses."""
    _base_space, specs, _mesh, transfer, _maps = _field_setup()
    rng = np.random.default_rng(7)
    n_e = int(sum(specs.component_dof_counts))
    blocks = []
    flat = []
    for _element in range(CELLS):
        draw = rng.standard_normal((n_e, n_e))
        block = 0.5 * (draw + draw.T) + n_e * np.eye(n_e)
        blocks.append(block)
        flat.append(block.ravel())
    scattered = np.asarray(scatter_csc(transfer, np.concatenate(flat)).toarray())

    reference = np.zeros((int(transfer.global_dof_count), int(transfer.global_dof_count)))
    for element, block in enumerate(blocks):
        local, index, values = element_entries(transfer, element)
        rows = np.repeat(index, index.size)
        cols = np.tile(index, index.size)
        kernel = values[:, None] * values[None, :] * block[np.ix_(local, local)]
        np.add.at(reference, (rows, cols), kernel.ravel())

    assert np.allclose(scattered, reference, rtol=0.0, atol=1e-12)


def test_transport_convergence_order() -> None:
    """Transport on a coupled mesh converges at the collocation order."""
    base_space, specs, _mesh, transfer, maps = _field_setup()
    nodes = np.ascontiguousarray(_integration().nodes()[0], np.double)
    field_velocity = np.ascontiguousarray(np.full(nodes.size, VELOCITY).reshape(1, -1))
    operators = [lie_derivative_operator(smap, specs, field_velocity) for smap in maps]

    def residual(y: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG001
        return -_global_scatter(
            transfer,
            [
                operator @ _element_view(transfer, y, element)
                for element, operator in enumerate(operators)
            ],
        )

    initial = _project(lambda x: np.exp(-(((x + 0.5) / 0.35) ** 2)), maps, base_space)
    y0, *_ = np.linalg.lstsq(_transfer_matrix(transfer), initial, rcond=None)
    mass = _coupled_mass(transfer, specs, maps)

    # Advection at positive velocity needs an inflow value. The leftmost
    # vertex unknown is the traced value at the point, so prescribing it
    # closes the problem and the march runs on the remaining unknowns.
    global_count = int(transfer.global_dof_count)
    free = np.setdiff1d(np.arange(global_count), [0])
    mass_free = mass[np.ix_(free, free)]

    def residual_free(y_free: np.ndarray, t: float) -> np.ndarray:
        """Return the residual of the full system on the free unknowns."""
        y = np.zeros(global_count)
        y[free] = y_free
        return residual(y, t)[free]

    final_time = 0.4

    def state_at(n_steps: int, stages: int) -> np.ndarray:
        return march(
            residual_free,
            y0[free],
            final_time / n_steps,
            n_steps,
            stages=stages,
            mass=mass_free,
        ).states[-1]

    for stages in (2, 3):
        reference = state_at(64, stages)
        errors = [
            float(np.linalg.norm(state_at(n_steps, stages) - reference))
            for n_steps in (8, 16)
        ]
        observed = np.log2(errors[0] / errors[1])
        assert observed >= 2 * stages - 0.7, (stages, errors, observed)


def _stage_index(mesh: MovingMesh, t: float) -> tuple[int, int]:
    """Return the step and stage of a stage time of the mesh."""
    times = np.array(
        [
            [mesh.stage_time(step, stage) for stage in range(mesh.stages)]
            for step in range(mesh.n_steps)
        ]
    )
    index = np.unravel_index(np.argmin(np.abs(times - t)), times.shape)
    return int(index[0]), int(index[1])


def test_free_stream_preservation_coupled() -> None:
    """A constant field is a fixed point of the coupled march on a moving mesh."""
    base_space, specs, _mesh, transfer, _maps = _field_setup()
    geometry_space = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER))

    def geometry_dofs(t: float) -> np.ndarray:
        """Return the translating geometry degrees of freedom of all elements."""
        corners = np.linspace(-1.0, 1.0, CELLS + 1) + SPEED * float(t)
        rows = [
            np.linspace(corners[e], corners[e + 1], GEO_ORDER + 1) for e in range(CELLS)
        ]
        return np.stack(rows).reshape(CELLS, 1, -1)

    mesh = MovingMesh(
        geometry_dofs,
        DT,
        8,
        element_count=CELLS,
        geometry_space=geometry_space,
        integration=_integration(),
        stages=2,
    )
    operators: dict[float, list[np.ndarray]] = {}

    def residual(y: np.ndarray, t: float) -> np.ndarray:
        """Return minus the mesh Lie derivative of the field."""
        key = float(t)
        if key not in operators:
            step, stage = _stage_index(mesh, t)
            operators[key] = [
                lie_derivative_operator(
                    smap, specs, np.ascontiguousarray(mesh.velocity(step, stage)[e])
                )
                for e, smap in enumerate(mesh.space_maps(step, stage))
            ]
        return -_global_scatter(
            transfer,
            [
                operator @ _element_view(transfer, y, element)
                for element, operator in enumerate(operators[key])
            ],
        )

    def mass_factory(t: float) -> np.ndarray:
        """Return the constrained mass at a stage time."""
        tolerance = 1.0e-9 * DT
        times = np.array(
            [
                [mesh.stage_time(step, stage) for stage in range(mesh.stages)]
                for step in range(mesh.n_steps)
            ]
        )
        matches = np.argwhere(np.abs(times - t) <= tolerance)
        if matches.size == 0:
            raise ValueError(f"Time {t} is not a stage time of the moving mesh.")
        step, stage = (int(value) for value in matches[0])
        return _coupled_mass(transfer, specs, mesh.space_maps(step, stage))

    constant = _project(
        lambda x: np.ones_like(np.asarray(x, dtype=np.double)),
        mesh.space_maps(0, 0),
        base_space,
    )
    y0, *_ = np.linalg.lstsq(_transfer_matrix(transfer), constant, rcond=None)

    result = march(residual, y0, DT, 8, stages=2, mass=mass_factory)

    assert np.max(np.abs(result.states[-1] - y0)) < 1.0e-10


def test_hybsol_mass_solver_agreement() -> None:
    """Stage solves through hybsol reproduce the dense march."""
    base_space, specs, mesh, transfer, maps = _field_setup()
    nodes = np.ascontiguousarray(_integration().nodes()[0], np.double)
    field_velocity = np.ascontiguousarray(np.full(nodes.size, VELOCITY).reshape(1, -1))
    operators = [lie_derivative_operator(smap, specs, field_velocity) for smap in maps]

    def residual(y: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG001
        return -_global_scatter(
            transfer,
            [
                operator @ _element_view(transfer, y, element)
                for element, operator in enumerate(operators)
            ],
        )

    initial = _project(lambda x: np.exp(-(((x + 0.5) / 0.35) ** 2)), maps, base_space)
    y0, *_ = np.linalg.lstsq(_transfer_matrix(transfer), initial, rcond=None)
    mass = _coupled_mass(transfer, specs, maps)
    global_count = int(transfer.global_dof_count)
    free = np.setdiff1d(np.arange(global_count), [0])
    mass_free = mass[np.ix_(free, free)]

    def residual_free(y_free: np.ndarray, t: float) -> np.ndarray:
        """Return the residual of the full system on the free unknowns."""
        y = np.zeros(global_count)
        y[free] = y_free
        return residual(y, t)[free]

    owners, _ = block_owners(transfer, mesh, ORDER, CELLS)
    flat = np.concatenate([stage_mass(smap, specs).ravel() for smap in maps])
    matrix = scatter_csc(transfer, flat)[free][:, free].tocsc()
    system, order = build_hybsol_system(matrix, owners, free, "element/object")
    assert system.is_valid()
    decomposition = system.decompose()
    unpermute = np.empty_like(order)
    unpermute[order] = np.arange(order.size)

    class _Refined:
        """Solve the reduced system in the block order of the factorization."""

        def solve(self, rhs: np.ndarray) -> np.ndarray:
            """Return the mass solve of one right-hand side, refined."""
            return refined_solve(system, decomposition, rhs[order], tolerance=1.0e-12)[
                unpermute
            ]

    solver = _Refined()
    reference = march(
        residual_free, y0[free], 0.4 / 16, 16, stages=2, mass=mass_free
    ).states[-1]
    hybsol = march(
        residual_free,
        y0[free],
        0.4 / 16,
        16,
        stages=2,
        mass_solver=lambda t: solver,  # noqa: ARG005
    ).states[-1]

    assert np.max(np.abs(reference - hybsol)) < 1.0e-9
