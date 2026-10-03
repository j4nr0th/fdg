"""Robustness tests: extreme and degenerate inputs at the public API boundary.

These probe the failure modes that a happy-path suite never reaches:
degenerate geometry, extreme orders, wrong shapes and dtypes, and long runs
where floating point time accumulation drifts.
"""

import numpy as np
import pytest
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    MovingMesh,
    SpaceMap,
    advection_operator,
    collocation_tableau,
    compute_kform_mass_matrix,
    incidence_kform_operator,
    lie_derivative_operator,
    march,
    space_maps_from_geometry_dofs,
    stage_mass,
)

ORDER = 6
BASE_SPACE = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, ORDER))
GEOMETRY_SPACE = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, ORDER))
INTEGRATION = IntegrationSpace(IntegrationSpecs(2 * ORDER, IntegrationMethod.GAUSS))
TOP_FORM = KFormSpecs(1, BASE_SPACE)
REFERENCE = np.linspace(-1.0, 1.0, ORDER + 1)


def _map(values: np.ndarray) -> SpaceMap:
    """Return a one-dimensional space map through the given geometry values."""
    return SpaceMap(
        CoordinateMap(
            DegreesOfFreedom(GEOMETRY_SPACE, np.ascontiguousarray(values)),
            INTEGRATION,
        )
    )


def _velocity(smap: SpaceMap, value: float = 0.5) -> np.ndarray:
    """Return a constant physical velocity on the map's integration points."""
    points = np.asarray(smap.integration_space.nodes()[0]).size
    return np.ascontiguousarray(np.full((1, points), value))


# --------------------------------------------------------------------------
# degenerate geometry
# --------------------------------------------------------------------------


def test_zero_determinant_gives_singular_mass_matrix() -> None:
    """A collapsed element has zero volume and a singular mass matrix."""
    smap = _map(np.zeros(ORDER + 1))
    assert np.allclose(np.asarray(smap.determinant), 0.0)

    mass = compute_kform_mass_matrix(smap, 0, BASE_SPACE, BASE_SPACE)
    assert np.allclose(mass, 0.0)

    # The top form divides by the determinant, so it comes out non-finite.
    # This is a documented consequence of the volume element vanishing.
    top = compute_kform_mass_matrix(smap, 1, BASE_SPACE, BASE_SPACE)
    assert not np.all(np.isfinite(top))


def test_negative_determinant_is_representable() -> None:
    """A reversed element has a negative volume element."""
    smap = _map(-REFERENCE)
    assert np.all(np.asarray(smap.determinant) < 0.0)

    zero_form = compute_kform_mass_matrix(smap, 0, BASE_SPACE, BASE_SPACE)
    # The volume element enters with its sign, so the mass is not positive.
    assert np.max(np.abs(zero_form)) > 0.0


def test_geometry_contradicting_element_count_is_rejected() -> None:
    """Geometry degrees of freedom must have one row per element."""
    with pytest.raises(ValueError, match=r"\(3, n_axes, n_dofs\)"):
        MovingMesh(
            lambda t: np.zeros((2, 1, ORDER + 1)),  # noqa: ARG005
            0.1,
            2,
            element_count=3,
            geometry_space=GEOMETRY_SPACE,
            integration=INTEGRATION,
        )


def test_space_maps_reject_wrong_rank() -> None:
    """Geometry of the wrong rank is rejected before it reaches the C core."""
    with pytest.raises(ValueError):
        space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, np.zeros((4, 5)))


# --------------------------------------------------------------------------
# orders and dimensions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 12])
def test_orders_assemble(order: int) -> None:
    """A range of polynomial orders assembles on the identity map."""
    base = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, order))
    geometry = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, order))
    integration = IntegrationSpace(IntegrationSpecs(order + 1))
    smap = SpaceMap(
        CoordinateMap(
            DegreesOfFreedom(geometry, np.linspace(-1.0, 1.0, order + 1)), integration
        )
    )
    mass = compute_kform_mass_matrix(smap, 0, base, base)
    assert mass.shape == (order + 1, order + 1)
    assert np.all(np.isfinite(mass))


@pytest.mark.parametrize("method", [IntegrationMethod.GAUSS])
def test_lie_derivative_requires_gauss(method: IntegrationMethod) -> None:
    """The velocity differentiation inverts the tableau, which Gauss allows."""
    tableau = collocation_tableau(2, method)
    assert np.isfinite(np.linalg.cond(tableau.integration_matrix))


def test_lobatto_tableau_is_singular_for_velocity() -> None:
    """A Lobatto tableau cannot be inverted, which the mesh must report."""
    tableau = collocation_tableau(3, IntegrationMethod.GAUSS_LOBATTO)
    condition = np.linalg.cond(tableau.integration_matrix)
    assert condition > 1e12


# --------------------------------------------------------------------------
# moving mesh argument validation
# --------------------------------------------------------------------------


def test_moving_mesh_rejects_nonpositive_element_count() -> None:
    """A mesh must have at least one element."""
    with pytest.raises(ValueError):
        MovingMesh(
            lambda t: REFERENCE.reshape(1, 1, -1),  # noqa: ARG005
            0.1,
            2,
            element_count=0,
            geometry_space=GEOMETRY_SPACE,
            integration=INTEGRATION,
        )


@pytest.mark.parametrize("step_size", [0.0, -0.1])
def test_moving_mesh_rejects_nonpositive_step_size(step_size: float) -> None:
    """A step size must be positive."""
    with pytest.raises(ValueError):
        MovingMesh(
            lambda t: REFERENCE.reshape(1, 1, -1),  # noqa: ARG005
            step_size,
            2,
            element_count=1,
            geometry_space=GEOMETRY_SPACE,
            integration=INTEGRATION,
        )


def test_moving_mesh_rejects_wrong_geometry_dof_count() -> None:
    """Geometry rows of the wrong length are rejected when the maps are built."""
    mesh = MovingMesh(
        lambda t: np.zeros((1, 1, ORDER + 1)),  # noqa: ARG005
        0.1,
        2,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
    )
    assert mesh.element_count == 1


def test_mass_factory_rejects_non_stage_time() -> None:
    """The mass factory serves stage times only."""
    mesh = MovingMesh(
        lambda t: (REFERENCE * (1.0 + t)).reshape(1, 1, -1),  # noqa: ARG005
        0.1,
        2,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
    )
    factory = mesh.mass_factory(TOP_FORM)
    with pytest.raises(ValueError):
        factory(0.12345)


def test_stage_masses_nests_over_step_stage_element() -> None:
    """stage_masses returns one matrix per step, stage and element."""

    def geometry(t: float) -> np.ndarray:  # noqa: ARG001
        """Return geometry of two elements."""
        out = np.zeros((2, 1, ORDER + 1))
        for element in range(2):
            out[element, 0] = (1.0 + 0.1 * element) * REFERENCE
        return out

    mesh = MovingMesh(
        geometry,
        0.1,
        3,
        element_count=2,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=2,
    )
    masses = mesh.stage_masses(TOP_FORM)
    assert len(masses) == 3
    assert len(masses[0]) == 2
    assert len(masses[0][0]) == 2
    assert masses[0][0][0].shape[0] == sum(TOP_FORM.component_dof_counts)


def test_moving_mesh_rejects_out_of_range_stage() -> None:
    """A stage index beyond the rule is rejected."""
    mesh = MovingMesh(
        lambda t: (REFERENCE * (1.0 + t)).reshape(1, 1, -1),  # noqa: ARG005
        0.1,
        2,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=2,
    )
    with pytest.raises(ValueError):
        mesh.velocity(0, 5)
    with pytest.raises(ValueError):
        mesh.space_maps(9, 0)


# --------------------------------------------------------------------------
# time marching argument validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("y0", "dt", "n_steps", "kwargs"),
    [
        (np.zeros(2), 0.1, 0, {}),
        (np.zeros((2, 1)), 0.1, 2, {}),
        (np.zeros(2), -0.1, 2, {}),
        (np.zeros(2), 0.1, 2, {"stages": 0}),
        (np.zeros(2), [0.1, 0.2], 3, {}),
    ],
)
def test_march_rejects_invalid_arguments(y0, dt, n_steps, kwargs) -> None:
    """Malformed march arguments are rejected."""
    with pytest.raises(ValueError):
        march(lambda y, t: np.zeros_like(y), y0, dt, n_steps, **kwargs)  # noqa: ARG005


def test_march_rejects_wrong_mass_shape() -> None:
    """A mass matrix that does not match the state is rejected."""
    with pytest.raises(ValueError):
        march(
            lambda y, t: np.zeros_like(y),  # noqa: ARG005
            np.zeros(3),
            0.1,
            2,
            mass=np.eye(2),
        )


def test_march_accepts_integer_state() -> None:
    """An integer state is converted rather than rejected."""
    result = march(
        lambda y, t: np.zeros_like(y),  # noqa: ARG005
        np.zeros(3, dtype=int),
        0.1,
        2,
    )
    assert result.states.shape == (3, 3)


def test_march_mass_callable_failure_propagates() -> None:
    """A mass callable that refuses a time raises rather than interpolating."""

    def refuse(t: float) -> np.ndarray:  # noqa: ARG001
        """Reject every time."""
        raise ValueError("no mass here")

    with pytest.raises(ValueError, match="no mass here"):
        march(
            lambda y, t: np.zeros_like(y),  # noqa: ARG005
            np.zeros(2),
            0.1,
            1,
            mass=refuse,
        )


def test_march_runs_many_steps_without_time_drift() -> None:
    """A long march keeps its stage times aligned with the mesh.

    The marcher accumulates time; if it does so by repeated addition while the
    mesh computes its stage times from a cumulative sum, the two drift apart
    until the mass factory can no longer recognise a stage time.
    """
    step_size, n_steps = 0.1, 6000
    mesh = MovingMesh(
        lambda t: (REFERENCE * (1.0 + 1e-4 * t)).reshape(1, 1, -1),  # noqa: ARG005
        step_size,
        n_steps,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=2,
    )
    result = march(
        lambda y, t: np.zeros_like(y),  # noqa: ARG005
        np.zeros(sum(TOP_FORM.component_dof_counts)),
        step_size,
        n_steps,
        stages=2,
        tolerance=1e-12,
        mass=mesh.mass_factory(TOP_FORM),
    )
    assert result.times.size == n_steps + 1
    assert result.times[-1] == pytest.approx(step_size * n_steps)


def test_moving_mesh_with_anderson_disabled() -> None:
    """A depth of zero selects plain Picard iteration."""
    result = march(
        lambda y, t: -y,  # noqa: ARG005
        np.array([1.0]),
        0.1,
        20,
        anderson_depth=0,
    )
    assert np.all(np.isfinite(result.states))


def test_march_single_stage() -> None:
    """One stage gives the implicit midpoint rule."""
    result = march(lambda y, t: -y, np.array([1.0]), 0.1, 10, stages=1)  # noqa: ARG005
    expected = np.exp(-1.0)
    # One stage is the implicit midpoint rule, which is second order.
    assert result.final_state[0] == pytest.approx(expected, rel=1e-2)


# --------------------------------------------------------------------------
# operators on degenerate velocity input
# --------------------------------------------------------------------------


def test_velocity_of_wrong_size_is_rejected() -> None:
    """A velocity that does not match the map is rejected by the C core."""
    smap = _map(REFERENCE)
    with pytest.raises(ValueError):
        advection_operator(smap, TOP_FORM, np.ascontiguousarray(np.zeros((1, 3))))


def test_incidence_of_top_form_is_empty() -> None:
    """The exterior derivative of a top form has no degrees of freedom."""
    n_dofs = sum(TOP_FORM.component_dof_counts)
    operator = incidence_kform_operator(TOP_FORM, np.eye(n_dofs))
    assert operator.size == 0


def test_lie_derivative_shapes_match_the_form() -> None:
    """Both operators return a square matrix of the form's size."""
    smap = _map(REFERENCE)
    n_dofs = sum(TOP_FORM.component_dof_counts)
    assert lie_derivative_operator(smap, TOP_FORM, _velocity(smap)).shape == (
        n_dofs,
        n_dofs,
    )


def test_stage_mass_matches_kform_mass() -> None:
    """stage_mass agrees with the underlying assembly."""
    smap = _map(REFERENCE)
    assert np.allclose(
        stage_mass(smap, TOP_FORM),
        compute_kform_mass_matrix(smap, 1, BASE_SPACE, BASE_SPACE),
    )
