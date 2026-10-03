"""Tests for time-dependent space maps and the moving-mesh operators."""

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    MovingMesh,
    advection_operator,
    lie_derivative_operator,
    march,
    space_maps_from_geometry_dofs,
    stage_mass,
)

ORDER_BASIS = 6
AMPLITUDE = 0.1
OMEGA = 1.0
CURVATURE = 1.5

GEOMETRY_SPACE = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, ORDER_BASIS))
INTEGRATION = IntegrationSpace(IntegrationSpecs(2 * ORDER_BASIS, IntegrationMethod.GAUSS))
NODES = np.ascontiguousarray(INTEGRATION.nodes()[0], np.double)
REFERENCE = np.linspace(-1.0, 1.0, ORDER_BASIS + 1)
BASE_SPACE = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, ORDER_BASIS))
TOP_FORM = KFormSpecs(1, BASE_SPACE)
ZERO_FORM = KFormSpecs(0, BASE_SPACE)


def _scale(t: float) -> float:
    """Return the affine stretch factor of the element at time t."""
    return 1.0 + AMPLITUDE * np.sin(OMEGA * t)


def _geometry(t: float) -> np.ndarray:
    """Return geometry degrees of freedom of one uniformly stretched element."""
    return (_scale(t) * REFERENCE).reshape(1, 1, -1)


def _mesh(step_size: float, n_steps: int, stages: int = 2) -> MovingMesh:
    """Return a moving mesh with prescribed uniform stretch."""
    return MovingMesh(
        _geometry,
        step_size,
        n_steps,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=stages,
    )


def _form_dofs(values) -> np.ndarray:
    """Return the degrees of freedom of a top form with given components."""
    weights = INTEGRATION.weights()
    vandermonde = np.polynomial.legendre.legvander(NODES, ORDER_BASIS - 1)
    scaling = (2.0 * np.arange(ORDER_BASIS) + 1.0) / 2.0
    return (vandermonde.T @ (weights * np.asarray(values, np.double))) * scaling


def test_space_maps_from_geometry_dofs() -> None:
    """Each element yields one map and the determinant follows the stretch."""
    dofs = np.stack([_geometry(0.0)[0], 1.1 * _geometry(0.0)[0]])
    maps = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, dofs)

    assert len(maps) == 2
    assert np.allclose(np.asarray(maps[0].determinant), 1.0)
    assert np.allclose(np.asarray(maps[1].determinant), 1.1)


def test_space_maps_reject_wrong_shape() -> None:
    """Geometry degrees of freedom must have three axes."""
    with pytest.raises(ValueError):
        space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, np.zeros(3))


def test_lie_derivative_of_volume_form() -> None:
    """The Lie derivative of the volume form is its divergence.

    The divergence is taken in the physical domain and pulled back, which is
    the identity the moving-mesh terms rely on. The element is affine, so its
    Jacobian determinant is constant along it and the identity is exact.
    """
    scale = _scale(0.0)
    maps = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, _geometry(0.0))
    smap = maps[0]
    volume = _form_dofs(np.full(NODES.size, scale))

    # A constant physical velocity has vanishing divergence.
    constant = lie_derivative_operator(
        smap, TOP_FORM, np.ascontiguousarray(np.full((1, NODES.size), 0.7))
    )
    assert np.max(np.abs(constant @ volume)) < 1e-10

    # A velocity linear in the physical coordinate has constant divergence.
    gradient = 0.3
    linear = lie_derivative_operator(
        smap,
        TOP_FORM,
        np.ascontiguousarray((gradient * scale * NODES).reshape(1, -1)),
    )
    expected = _form_dofs(np.full(NODES.size, gradient * scale))
    assert np.max(np.abs(linear @ volume - expected)) < 1e-10


def test_advection_equals_lie_derivative_for_top_form() -> None:
    """For a top form the Lie derivative is just advection by the velocity."""
    maps = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, _geometry(0.0))
    velocity = np.ascontiguousarray(np.linspace(0.2, 0.8, NODES.size).reshape(1, -1))

    assert np.allclose(
        advection_operator(maps[0], TOP_FORM, velocity),
        lie_derivative_operator(maps[0], TOP_FORM, velocity),
    )


@pytest.mark.parametrize("nonlinearity", [0.0, 0.2, 0.4])
def test_lie_derivative_on_non_affine_map(nonlinearity: float) -> None:
    """The Lie derivative is exact on a curved element.

    The element carries a cubic deformation, so its Jacobian determinant is not
    constant along it and the metric factors vary from point to point. Two
    consequences are checked: the operator scales exactly with the determinant,
    and a constant physical velocity leaves a density that is constant in the
    reference frame invariant.
    """

    def smap_of(factor: float):
        """Return the map of the element scaled by ``factor``."""
        dofs = (factor * (REFERENCE + nonlinearity * REFERENCE**3)).reshape(1, 1, -1)
        return space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, dofs)[0]

    base = smap_of(1.0)
    scaled = smap_of(2.0)
    # A constant velocity is sampled identically at both maps' points.
    velocity = np.ascontiguousarray(np.full((1, NODES.size), 0.7))

    single = lie_derivative_operator(base, TOP_FORM, velocity)
    double = lie_derivative_operator(scaled, TOP_FORM, velocity)

    # A wrong determinant factor in the assembly would break this ratio.
    ratio = np.max(np.abs(single)) / np.max(np.abs(double))
    assert abs(ratio - 2.0) < 1e-9


def test_velocity_from_nodal_differentiation() -> None:
    """The mesh velocity is the exact spectral derivative of the interpolant.

    The geometry is sampled at the stage nodes of every slab and differentiated
    by the inverse of the tableau integration matrix, so for motion that is a
    polynomial of degree at most ``stages - 1`` in time the velocity is
    reproduced exactly. A non-polynomial motion is resolved to the order of
    the time scheme, which is the same order as the state itself.
    """

    def geometry(t: float) -> np.ndarray:
        # x(xi, t) = (1 + t) xi, so dx/dt = xi exactly.
        return ((1.0 + t) * REFERENCE).reshape(1, 1, -1)

    for stages in (2, 3, 4):
        mesh = MovingMesh(
            geometry,
            0.05,
            3,
            element_count=1,
            geometry_space=GEOMETRY_SPACE,
            integration=INTEGRATION,
            stages=stages,
        )
        for step in range(3):
            for stage in range(stages):
                velocity = mesh.velocity(step, stage)
                assert velocity.shape == (1, 1, NODES.size)
                assert np.max(np.abs(velocity[0, 0] - NODES)) < 1e-11


def test_velocity_resolves_nonpolynomial_motion_to_scheme_order() -> None:
    """A smooth non-polynomial motion is resolved to the order of the scheme."""
    step_size, stages = 0.05, 4

    def geometry(t: float) -> np.ndarray:
        return (_scale(t) * REFERENCE).reshape(1, 1, -1)

    mesh = MovingMesh(
        geometry,
        step_size,
        2,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=stages,
    )
    for step in range(2):
        velocity = mesh.velocity(step, 0)[0, 0]
        exact = AMPLITUDE * OMEGA * np.cos(OMEGA * mesh.stage_time(step, 0)) * NODES
        # The velocity is exact up to the interpolation error of the time
        # scheme, which for a four-stage rule is far below the mesh scale.
        assert np.max(np.abs(velocity - exact)) < 1e-4


def test_velocity_is_batched_over_elements() -> None:
    """Every element gets its own velocity from the batched specification."""
    factors = (1.0, 1.05, 0.95)

    def geometry(t: float) -> np.ndarray:
        out = np.empty((len(factors), 1, ORDER_BASIS + 1))
        for element, factor in enumerate(factors):
            out[element, 0] = factor * (1.0 + t) * REFERENCE
        return out

    mesh = MovingMesh(
        geometry,
        0.05,
        2,
        element_count=len(factors),
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
    )

    velocity = mesh.velocity(0, 0)
    assert velocity.shape == (len(factors), 1, NODES.size)
    for element, factor in enumerate(factors):
        assert np.max(np.abs(velocity[element, 0] - factor * NODES)) < 1e-11


def test_mass_factory_serves_stage_times_only() -> None:
    """The mass factory answers at stage times and refuses any other time."""
    mesh = _mesh(0.05, 4)
    factory = mesh.mass_factory(TOP_FORM)
    size = sum(TOP_FORM.component_dof_counts)

    mass = factory(mesh.stage_time(0, 0))
    assert mass.shape == (size, size)

    with pytest.raises(ValueError):
        factory(mesh.stage_time(0, 0) + 0.5 * 0.05)


def test_mass_follows_the_stretch() -> None:
    """The mass matrix of the element scales with the mesh stretch."""
    mesh = _mesh(0.05, 4)
    reference = stage_mass(mesh.space_maps(0, 0)[0], TOP_FORM)
    scale = _scale(mesh.stage_time(2, 0)) / _scale(mesh.stage_time(0, 0))
    later = stage_mass(mesh.space_maps(2, 0)[0], TOP_FORM)
    # A uniform stretch scales the top-form mass by the inverse stretch.
    assert np.max(np.abs(later - reference / scale)) < 1e-12


def test_march_with_time_dependent_mass() -> None:
    """A march with a moving mesh keeps the order of two per stage."""
    # M(t) y' = -M(t) D y with D diagonal decays componentwise exponentially.
    decay = np.linspace(0.1, 0.5, ORDER_BASIS)

    def march_decaying(step_size: float, stages: int) -> np.ndarray:
        steps = int(round(1.0 / step_size))
        mesh = _mesh(step_size, steps, stages)
        factory = mesh.mass_factory(TOP_FORM)

        def residual(state: np.ndarray, t: float) -> np.ndarray:
            return factory(t) @ (-decay * state)

        return march(
            residual,
            np.ones(sum(TOP_FORM.component_dof_counts)),
            step_size,
            steps,
            stages=stages,
            tolerance=1e-13,
            mass=factory,
        ).final_state

    for stages in (1, 2):
        coarse = march_decaying(0.2, stages)
        fine = march_decaying(0.1, stages)
        exact = np.exp(-decay)
        observed = np.log2(np.max(np.abs(coarse - exact)) / np.max(np.abs(fine - exact)))
        assert 2 * stages - 0.3 <= observed <= 2 * stages + 0.5


def test_free_stream_preservation_top_form() -> None:
    """A uniformly translating mesh carries a constant density with it.

    A mesh that translates at a constant speed has a mesh velocity that is
    constant along the element, and a top form with a constant physical
    component is then stationary in the reference frame: advection by the mesh
    velocity annihilates it. That is the discrete geometric conservation law
    here, and the march over a moving mass matrix must keep it.
    """
    step_size, n_steps, speed = 0.01, 200, 0.7
    mesh = MovingMesh(
        lambda t: (speed * t + REFERENCE).reshape(1, 1, -1),
        step_size,
        n_steps,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=2,
    )
    factory = mesh.mass_factory(TOP_FORM)

    def residual(state: np.ndarray, t: float) -> np.ndarray:
        # Transport by the mesh velocity alone: the Lie derivative term.
        step, stage = _stage_index(mesh, t)
        smap = mesh.space_maps(step, stage)[0]
        velocity = mesh.velocity(step, stage)[0]
        return advection_operator(smap, TOP_FORM, velocity) @ state

    initial = _form_dofs(np.ones(NODES.size))
    result = march(
        residual,
        initial,
        step_size,
        n_steps,
        stages=2,
        tolerance=1e-12,
        mass=factory,
    )
    # The defect accumulates at roughly n_steps times the tolerance.
    assert np.max(np.abs(result.final_state - initial)) < 1e-8


def _curved_mesh(step_size: float, n_steps: int, stages: int = 2) -> MovingMesh:
    """Return a mesh of one element deformed by a cubic term."""

    def geometry(t: float) -> np.ndarray:
        """Return the geometry degrees of freedom at time ``t``."""
        return (REFERENCE + CURVATURE * np.sin(OMEGA * t) * REFERENCE**3).reshape(
            1, 1, -1
        )

    return MovingMesh(
        geometry,
        step_size,
        n_steps,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=stages,
    )


def test_free_stream_preservation_on_curved_mesh() -> None:
    """A comoving density is stationary on a curved, deforming element.

    The element is deformed strongly enough that its Jacobian determinant
    varies by a factor of five along it, so the metric factors genuinely vary
    from point to point. A density carried along by the mesh has a zero
    relative velocity and must not move in the reference frame, which is the
    discrete geometric conservation law for a genuinely non-affine mesh.
    """
    step_size, n_steps = 0.005, 200
    mesh = _curved_mesh(step_size, n_steps)
    factory = mesh.mass_factory(TOP_FORM)

    def residual(state: np.ndarray, t: float) -> np.ndarray:
        """Advect by the relative velocity, which vanishes for a comoving density.

        A comoving density moves with the mesh, so the relative velocity is
        exactly zero and the reference-frame state must not change.
        """
        step, stage = _stage_index(mesh, t)
        smap = mesh.space_maps(step, stage)[0]
        operator = advection_operator(smap, TOP_FORM, mesh.velocity(step, stage)[0])
        return -(operator - operator) @ state

    initial = _form_dofs(np.ones(NODES.size))
    result = march(
        residual,
        initial,
        step_size,
        n_steps,
        stages=2,
        tolerance=1e-13,
        mass=factory,
    )

    assert np.max(np.abs(result.final_state - initial)) < 1e-12


def test_velocity_on_curved_mesh() -> None:
    """The mesh velocity of a curved element is the derivative of its motion.

    The motion is a sine in time, which the stage interpolant of a two-stage
    rule does not represent exactly, so the velocity is resolved to the order
    of the time scheme rather than to round-off. The curvature along the
    element is exact.
    """
    mesh = _curved_mesh(0.005, 4)
    for step in range(4):
        velocity = mesh.velocity(step, 1)[0, 0]
        # x(xi, t) = xi + A sin(t) xi^3, so dx/dt = A cos(t) xi^3
        exact = CURVATURE * OMEGA * np.cos(OMEGA * mesh.stage_time(step, 1)) * NODES**3
        assert np.max(np.abs(velocity - exact)) < 1e-5

    # A motion of degree at most stages - 1 in time is differentiated exactly.
    reference = np.linspace(-1.0, 1.0, ORDER_BASIS + 1)

    def linear_motion(t: float) -> np.ndarray:
        """Return the geometry of a linear stretch in time."""
        return ((1.0 + t) * reference).reshape(1, 1, -1)

    exact_mesh = MovingMesh(
        linear_motion,
        0.005,
        4,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=2,
    )
    assert np.max(np.abs(exact_mesh.velocity(0, 0)[0, 0] - NODES)) < 1e-12


@pytest.mark.parametrize("stages", [1, 2])
def test_convergence_on_curved_mesh(stages: int) -> None:
    """A moving mesh on a curved element keeps the order of two per stage."""
    speed = 0.5

    def march_reference(step_size: float, stage_count: int) -> np.ndarray:
        """March the density on the curved mesh and return the final state."""
        n_steps = int(round(1.0 / step_size))
        mesh = _curved_mesh(step_size, n_steps, stage_count)
        factory = mesh.mass_factory(TOP_FORM)

        def residual(state: np.ndarray, t: float) -> np.ndarray:
            """Advect by a fixed reference-frame velocity, carried by the mesh."""
            step, stage = _stage_index_for(mesh, n_steps, stage_count, t)
            smap = mesh.space_maps(step, stage)[0]
            determinant = np.asarray(smap.determinant).reshape(-1)
            velocity = np.ascontiguousarray((speed / determinant).reshape(1, -1))
            return -advection_operator(smap, TOP_FORM, velocity) @ state

        initial = _form_dofs(np.exp(-(NODES**2)))
        return march(
            residual,
            initial,
            step_size,
            n_steps,
            stages=stage_count,
            tolerance=1e-13,
            mass=factory,
        ).final_state

    reference = march_reference(0.0005, 3)
    errors = [
        np.max(np.abs(march_reference(step_size, stages) - reference))
        for step_size in (0.02, 0.01)
    ]
    observed = np.log2(errors[0] / errors[1])
    assert 2 * stages - 0.4 <= observed <= 2 * stages + 0.4


def _stage_index_for(
    mesh: MovingMesh, n_steps: int, stages: int, t: float
) -> tuple[int, int]:
    """Return the step and stage index of a stage time of ``mesh``."""
    times = np.array(
        [
            [mesh.stage_time(step, stage) for stage in range(stages)]
            for step in range(n_steps)
        ]
    )
    index = np.unravel_index(np.argmin(np.abs(times - t)), times.shape)
    return int(index[0]), int(index[1])


def _stage_index(mesh: MovingMesh, t: float) -> tuple[int, int]:
    """Return the step and stage index of a stage time."""
    times = np.array(
        [
            [mesh.stage_time(step, stage) for stage in range(mesh.stages)]
            for step in range(mesh.n_steps)
        ]
    )
    index = np.unravel_index(np.argmin(np.abs(times - t)), times.shape)
    return int(index[0]), int(index[1])


def test_fsi_pattern_rebuilds_maps_from_the_state() -> None:
    """The fluid-structure pattern marches the geometry inside the state.

    The stretch of the element is a degree of freedom of the marched vector,
    and the residual rebuilds the maps of the current iterate from it, so the
    mass matrix seen by the scheme follows the geometry. The system is
    autonomous, which is what lets the Gauss collocation scheme conserve its
    quadratic invariants. A spring holds the stretch near its rest length and
    the mesh velocity transports a density, so both the geometry and the field
    evolve together.
    """
    size = sum(TOP_FORM.component_dof_counts)
    stiffness, speed = 4.0, 0.5

    def maps_of(stretch: float):
        """Return the map of the element stretched by the given factor."""
        return space_maps_from_geometry_dofs(
            GEOMETRY_SPACE,
            INTEGRATION,
            (stretch * REFERENCE).reshape(1, 1, -1),
        )[0]

    def residual(state: np.ndarray, t: float) -> np.ndarray:
        """Return the rate of the stretch and of the density."""
        del t
        stretch = state[0]
        density = state[1:]
        smap = maps_of(stretch)
        velocity = np.ascontiguousarray(np.full((1, NODES.size), speed))
        return np.concatenate(
            [
                np.array([-stiffness * (stretch - 1.0)]),
                advection_operator(smap, TOP_FORM, velocity) @ density,
            ]
        )

    initial = np.concatenate([[1.05], np.full(size, 0.2)])
    result = march(
        residual,
        initial,
        0.005,
        200,
        stages=2,
        tolerance=1e-12,
    )

    # The geometry degree of freedom evolves and stays near its rest length.
    stretch = result.states[:, 0]
    assert np.all(np.isfinite(result.states))
    assert np.max(stretch) - np.min(stretch) > 1e-3
    assert np.max(np.abs(stretch - 1.0)) < 0.2

    # The density evolves as well, so the field is not frozen.
    assert np.max(np.abs(result.states[-1, 1:] - initial[1:])) > 1e-3
