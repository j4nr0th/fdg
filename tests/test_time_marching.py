"""Tests for the geometric time marching module."""

import numpy as np
import pytest
from fdg import IntegrationMethod, IntegrationSpecs, collocation_tableau, march


def _oscillator(y: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG001
    """Right-hand side of the harmonic oscillator."""
    return np.array([y[1], -y[0]])


def _energy(y: np.ndarray, t: float) -> float:  # noqa: ARG001
    """Energy of the harmonic oscillator."""
    return 0.5 * (y[0] ** 2 + y[1] ** 2)


def test_tableau_row_sums_match_node_offsets() -> None:
    """Integrating the nodal basis over the full interval gives the node offset."""
    for method in (IntegrationMethod.GAUSS, IntegrationMethod.GAUSS_LOBATTO):
        for stages in range(1, 7):
            tableau = collocation_tableau(stages, method)
            assert np.allclose(
                tableau.integration_matrix.sum(axis=1), tableau.nodes + 1.0, atol=1e-12
            )
            assert np.isclose(tableau.weights.sum(), 2.0, atol=1e-12)


def test_weights_are_integrals_of_nodal_basis() -> None:
    """Rule weights equal the integrals of the nodal basis functions."""
    for method in (IntegrationMethod.GAUSS, IntegrationMethod.GAUSS_LOBATTO):
        for stages in range(1, 6):
            tableau = collocation_tableau(stages, method)
            fine = IntegrationSpecs(20, IntegrationMethod.GAUSS)
            nodes = fine.nodes()
            weights = fine.weights()

            # Evaluate the nodal basis on the fine rule with barycentric weights.
            bary = np.array(
                [
                    1.0
                    / np.prod([tableau.nodes[j] - n for n in tableau.nodes if n != node])
                    for j, node in enumerate(tableau.nodes)
                ]
            )
            basis = np.empty((stages, nodes.size))
            for j in range(stages):
                for i, n in enumerate(nodes):
                    if np.any(np.isclose(n, tableau.nodes)):
                        basis[j, i] = float(np.isclose(n, tableau.nodes[j]))
                    else:
                        numerators = np.array(
                            [bary[m] / (n - tableau.nodes[m]) for m in range(stages)]
                        )
                        basis[j, i] = numerators[j] / numerators.sum()

            reference = basis @ weights
            assert np.allclose(tableau.weights, reference, atol=1e-12)


def test_integration_matrix_exact_on_polynomials() -> None:
    """The integration matrix integrates polynomials of nodal degree exactly."""
    for method in (IntegrationMethod.GAUSS, IntegrationMethod.GAUSS_LOBATTO):
        for stages in range(1, 7):
            tableau = collocation_tableau(stages, method)
            rng = np.random.default_rng(stages)
            for _ in range(5):
                coefficients = rng.uniform(-1.0, 1.0, stages)
                values = np.polynomial.polynomial.polyval(tableau.nodes, coefficients)
                integral = np.polynomial.Polynomial(coefficients).integ()
                expected = np.array([integral(x) - integral(-1.0) for x in tableau.nodes])
                assert np.allclose(
                    tableau.integration_matrix @ values, expected, atol=1e-11
                )


def test_tableau_node_count_matches_stages() -> None:
    """A tableau has exactly one node per stage."""
    for stages in range(1, 6):
        tableau = collocation_tableau(stages)
        assert tableau.nodes.size == stages
        assert tableau.weights.size == stages
        assert tableau.integration_matrix.shape == (stages, stages)


def test_invalid_stage_count_raises() -> None:
    """A non-positive number of stages is rejected."""
    with pytest.raises(ValueError):
        collocation_tableau(0)


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_observed_order_matches_stages(stages: int) -> None:
    """The scheme converges with the expected order of two per stage."""
    errors = []
    for step_size in (1.0, 0.5, 0.25):
        result = march(
            lambda y, t: -y,  # noqa: ARG005
            np.array([1.0]),
            step_size,
            int(round(1.0 / step_size)),
            stages=stages,
            tolerance=1e-14,
        )
        errors.append(abs(result.final_state[0] - np.exp(-1.0)))

    for coarse, fine in zip(errors[:-1], errors[1:]):
        observed = np.log2(coarse / fine)
        assert 2 * stages - 0.3 <= observed <= 2 * stages + 0.5


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_quadratic_invariant_is_conserved(stages: int) -> None:
    """The energy of the oscillator is conserved up to round-off."""
    result = march(
        _oscillator,
        np.array([1.0, 0.0]),
        0.1,
        1000,
        stages=stages,
        tolerance=1e-13,
        invariant=_energy,
    )

    assert result.invariant_values is not None
    initial = result.invariant_values[0]
    assert np.max(np.abs(result.invariant_values - initial)) / initial < 1e-11


def test_linear_invariant_is_conserved() -> None:
    """A linear invariant of a linear system is conserved."""
    matrix = np.array([[1.0, 1.0], [-1.0, -1.0]])  # column sums vanish
    result = march(
        lambda y, t: matrix @ y,  # noqa: ARG005
        np.array([3.0, 4.0]),
        0.05,
        500,
        stages=2,
        tolerance=1e-13,
        invariant=lambda y, t: y[0] + y[1],  # noqa: ARG005
    )

    assert result.invariant_values is not None
    assert np.max(np.abs(result.invariant_values - 7.0)) < 1e-11


def test_mass_matrix_path_matches_explicit_solve() -> None:
    """Passing a mass matrix reproduces the trajectory of the divided residual."""
    mass = np.diag([2.0, 3.0])
    direct = march(_oscillator, np.array([1.0, 0.0]), 0.1, 50, stages=2, tolerance=1e-13)
    weighted = march(
        lambda y, t: mass @ _oscillator(y, t),
        np.array([1.0, 0.0]),
        0.1,
        50,
        stages=2,
        tolerance=1e-13,
        mass=mass,
    )

    assert np.allclose(direct.states, weighted.states, atol=1e-13)


def test_anderson_reduces_iteration_count() -> None:
    """Anderson acceleration needs fewer iterations than plain Picard."""

    def run(depth: int):
        result = march(
            lambda y, t: np.array([3.0 * y[1], -3.0 * y[0]]),  # noqa: ARG005
            np.array([1.0, 0.0]),
            1.0,
            50,
            stages=3,
            tolerance=1e-12,
            anderson_depth=depth,
        )
        return sum(result.iterations)

    assert run(5) < run(0)


def test_nonconvergence_raises() -> None:
    """A step that does not converge within the iteration budget raises."""
    with pytest.raises(RuntimeError, match="Time step 0"):
        march(
            lambda y, t: -1e6 * y,  # noqa: ARG005
            np.array([1.0]),
            1.0,
            3,
            max_iterations=5,
        )


def test_variable_step_sizes_and_times() -> None:
    """Step sizes may vary per step and the initial time is honoured."""
    sizes = np.array([0.1, 0.2, 0.4])
    result = march(
        lambda y, t: -y,  # noqa: ARG005
        np.array([1.0]),
        sizes,
        sizes.size,
        t0=1.0,
        stages=2,
        tolerance=1e-13,
    )

    assert np.allclose(result.times, 1.0 + np.concatenate(([0.0], np.cumsum(sizes))))
    expected = np.exp(-(result.times - 1.0))
    assert np.allclose(result.states[:, 0], expected, atol=1e-3)


def test_result_reports_final_state() -> None:
    """The final state of the result is the last accepted state."""
    result = march(_oscillator, np.array([1.0, 0.0]), 0.1, 10, stages=2)

    assert np.allclose(result.final_state, result.states[-1])
    assert result.invariant_values is None
    assert len(result.iterations) == 10
    assert len(result.residual_norms) == 10


@pytest.mark.parametrize(
    ("y0", "dt", "n_steps", "extra"),
    [
        (np.array([1.0, 0.0]), 0.1, 0, {}),
        (np.array([1.0, 0.0]), 0.1, 2, {"stages": 0}),
        (np.array([1.0, 0.0]), [0.1, 0.2, 0.3], 2, {}),
        (np.array([[1.0, 0.0]]), 0.1, 2, {}),
        (np.array([1.0, 0.0]), -0.1, 2, {}),
    ],
)
def test_invalid_inputs_raise(
    y0: np.ndarray, dt: object, n_steps: int, extra: dict
) -> None:
    """Invalid march arguments are rejected."""
    with pytest.raises(ValueError):
        march(_oscillator, y0, dt, n_steps, **extra)


def test_invalid_mass_shape_raises() -> None:
    """A mass matrix of the wrong shape is rejected."""
    with pytest.raises(ValueError):
        march(
            _oscillator,
            np.array([1.0, 0.0]),
            0.1,
            2,
            mass=np.eye(3),
        )


def test_residual_sees_exact_stage_times() -> None:
    """The residual is evaluated at the exact stage times of every slab."""
    seen: list[float] = []

    def probe(y: np.ndarray, t: float) -> np.ndarray:
        seen.append(float(t))
        return -y

    result = march(probe, np.array([1.0]), 0.1, 2, stages=3, tolerance=1e-12)
    tableau = collocation_tableau(3)

    expected = {
        round(float(t), 10)
        for base in (0.0, 0.1)
        for t in (base + 0.05 * (tableau.nodes + 1.0))
    }
    assert {round(t, 10) for t in seen} == expected
    assert result.states.shape == (3, 1)


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_time_dependent_forcing_keeps_order(stages: int) -> None:
    """A residual depending on time retains the order of two per stage."""
    decay = np.array([-1.0, -2.0])
    offset = np.array([1.0, 0.5])
    slope = np.array([0.7, -0.3])

    # Forcing chosen so the exact solution is known in closed form:
    #     y(t) = offset * exp(decay * t) * (1 + slope * t)
    def forcing(t: float) -> np.ndarray:
        return offset * slope * np.exp(decay * t)

    def exact(t: float) -> np.ndarray:
        return offset * np.exp(decay * t) * (1.0 + slope * t)

    errors = []
    for step_size in (0.2, 0.1):
        result = march(
            lambda y, t: decay * y + forcing(t),
            offset.copy(),
            step_size,
            int(round(1.0 / step_size)),
            stages=stages,
            tolerance=1e-13,
        )
        errors.append(np.max(np.abs(result.final_state - exact(1.0))))

    observed = np.log2(errors[0] / errors[1])
    assert 2 * stages - 0.3 <= observed <= 2 * stages + 0.5
