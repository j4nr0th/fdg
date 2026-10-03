r"""Geometric time marching by collocation of 0-forms in time.

Time is treated as an additional one-dimensional domain. On every time slab
the state is expanded in the nodal 0-form basis associated with an integration
rule of the library, which places the stage unknowns on the integration nodes
of that rule. The semi-discrete system

.. math::

    M \frac{\mathrm{d}y}{\mathrm{d}t} = r(y, t)

is integrated by evaluating the residual on those stages and solving the slab
system in its integral collocation form

.. math::

    Y_k = y_n + \frac{\Delta t}{2} \sum_j A_{kj} M^{-1} r(Y_j, t_j),
    \qquad
    y_{n+1} = y_n + \frac{\Delta t}{2} \sum_j b_j M^{-1} r(Y_j, t_j).

For Gauss nodes this is the :math:`s`-stage Gauss collocation implicit
Runge--Kutta method with :math:`s` stages. It is symplectic, of order
:math:`2s`, and conserves every quadratic invariant of the semi-discrete
system exactly in exact arithmetic; in floating point the conservation error of
a run grows like :math:`n_{\text{steps}} \cdot \text{tolerance}`. Gauss--Lobatto
nodes give the Lobatto collocation methods, which are supported but do not
carry the quadratic invariant guarantee.

A hybridized 1-form variant in time, which introduces trace unknowns on the
slab interfaces, is a planned sibling of this module and reuses the fixed-point
driver and the driver loop below.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import legvander
from scipy.linalg import lu_factor, lu_solve

from fdg._fdg import IntegrationSpecs
from fdg.enum_type import IntegrationMethod

#: LU factors as returned by :func:`scipy.linalg.lu_factor`.
MassFactors = tuple[npt.NDArray[np.double], npt.NDArray[np.int32]] | None


@dataclass(frozen=True)
class CollocationTableau:
    r"""Nodal collocation data of a time integration rule.

    Attributes
    ----------
    stages : int
        Number of stages, equal to the number of nodes.
    method : IntegrationMethod
        Method used for the integration rule the nodes come from.
    nodes : array
        Nodes :math:`\\tau_j` on the reference interval :math:`[-1, 1]`.
    weights : array
        Weights :math:`b_j` of the rule, which equal the integrals of the nodal
        basis functions over :math:`[-1, 1]`.
    integration_matrix : array
        Entries :math:`A_{kj}` equal the integral of the nodal basis function
        :math:`\\ell_j` over :math:`[-1, \\tau_k]`.
    """

    stages: int
    method: IntegrationMethod
    nodes: npt.NDArray[np.double]
    weights: npt.NDArray[np.double]
    integration_matrix: npt.NDArray[np.double]


@dataclass(frozen=True)
class MarchResult:
    """Result of a time march.

    Attributes
    ----------
    times : array
        Times of the initial state and of all step endpoints.
    states : array
        State at the initial time and at every step endpoint, one row per
        time.
    iterations : list of int
        Number of fixed-point iterations used per step.
    residual_norms : list of float
        Norm of the fixed-point defect at the accepted iterate of each step.
    invariant_values : array or None
        Values of the invariant at all times if one was passed to the march.
    """

    times: npt.NDArray[np.double]
    states: npt.NDArray[np.double]
    iterations: list[int]
    residual_norms: list[float]
    invariant_values: npt.NDArray[np.double] | None

    @property
    def final_state(self) -> npt.NDArray[np.double]:
        """State at the end of the march."""
        return self.states[-1]


def _integration_matrix(
    nodes: npt.NDArray[np.double], weights: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
    r"""Integrate the nodal basis functions from the left endpoint to the nodes.

    The nodal basis function :math:`\\ell_j` has the Legendre expansion
    :math:`c_{jm} = (2m + 1) w_j P_m(\\tau_j) / 2`, which is exact because the
    rule integrates polynomials of degree :math:`2s - 1`. Integrating the
    Legendre monomials with
    :math:`\\int_{-1}^{x} P_m(\\xi) \\, \\mathrm{d}\\xi =
    (P_{m+1}(x) - P_{m-1}(x)) / (2m + 1)` yields the requested integrals.

    Parameters
    ----------
    nodes : array
        Nodes :math:`\\tau_j` of the integration rule.
    weights : array
        Weights :math:`w_j` of the integration rule.

    Returns
    -------
    array
        Entries :math:`A_{kj}` of the integration matrix.
    """
    stages = nodes.size
    # Columns of the Vandermonde matrix are the Legendre polynomials of
    # increasing degree evaluated at the nodes; the last column is needed for
    # the primitive of the highest monomial.
    vander = legvander(nodes, stages)
    degrees = np.arange(stages)
    coefficients = 0.5 * (2.0 * degrees + 1.0) * vander[:, :stages] * weights[:, None]

    primitives = np.empty((stages, stages))
    primitives[:, 0] = nodes + 1.0
    denominators = 2.0 * np.arange(1, stages) + 1.0
    primitives[:, 1:] = (
        vander[:, 2 : stages + 1] - vander[:, : stages - 1]
    ) / denominators

    return primitives @ coefficients.T


def collocation_tableau(
    stages: int, method: IntegrationMethod = IntegrationMethod.GAUSS
) -> CollocationTableau:
    """Construct the collocation tableau of a time integration rule.

    Parameters
    ----------
    stages : int
        Number of collocation stages, which is the number of nodes of the rule.
    method : IntegrationMethod, default: "gauss"
        Method used for the integration rule. Gauss nodes give a symplectic
        method that conserves quadratic invariants, Gauss--Lobatto nodes do not.

    Returns
    -------
    CollocationTableau
        Tableau of the rule with its integration matrix.
    """
    if stages < 1:
        raise ValueError(f"Number of stages must be positive, got {stages}.")

    rule = IntegrationSpecs(stages - 1, method)
    nodes = np.ascontiguousarray(rule.nodes(), np.double)
    weights = np.ascontiguousarray(rule.weights(), np.double)

    return CollocationTableau(
        stages=stages,
        method=IntegrationMethod(method),
        nodes=nodes,
        weights=weights,
        integration_matrix=_integration_matrix(nodes, weights),
    )


def _stage_residuals(
    residual: Callable[[npt.NDArray[np.double], float], npt.NDArray[np.double]],
    stage_dofs: npt.NDArray[np.double],
    stage_times: npt.NDArray[np.double],
    mass_factors: MassFactors,
) -> npt.NDArray[np.double]:
    """Evaluate the residual on the stages and solve the mass matrix.

    Parameters
    ----------
    residual : callable
        Residual function of the semi-discrete system, see :func:`march`.
    stage_dofs : array
        Stage states with one row per stage.
    stage_times : array
        Times of the stages.
    mass_factors : tuple of arrays or None
        Factors of the factorization of the mass matrix, or None for an
        identity mass matrix.

    Returns
    -------
    array
        Right-hand side :math:`M^{-1} r` with one row per stage.
    """
    stages, n_dofs = stage_dofs.shape
    values = np.empty_like(stage_dofs)
    for k in range(stages):
        values[k] = residual(np.ascontiguousarray(stage_dofs[k]), float(stage_times[k]))

    if mass_factors is not None:
        # scipy.linalg.lu_solve takes a single right-hand side, so the stage
        # systems are solved one after another.
        for k in range(stages):
            values[k] = lu_solve(mass_factors, values[k])

    return values


def _anderson_solve(
    fixed_point: Callable[[npt.NDArray[np.double]], npt.NDArray[np.double]],
    initial: npt.NDArray[np.double],
    tolerance: float,
    max_iterations: int,
    depth: int,
) -> tuple[npt.NDArray[np.double], int, float]:
    """Solve a fixed-point equation with optional Anderson acceleration.

    Parameters
    ----------
    fixed_point : callable
        Fixed-point map applied to the iterate.
    initial : array
        Initial guess.
    tolerance : float
        Relative tolerance on the norm of the fixed-point defect.
    max_iterations : int
        Maximum number of applications of the fixed-point map.
    depth : int
        Depth of the Anderson window, zero for plain fixed-point iteration.

    Returns
    -------
    tuple
        Converged iterate, number of iterations, and final defect norm.
    """
    iterates: list[npt.NDArray[np.double]] = []
    defects: list[npt.NDArray[np.double]] = []
    iterate = initial
    norm = float("inf")

    for iteration in range(1, max_iterations + 1):
        image = fixed_point(iterate)
        defect = image - iterate
        norm = float(np.linalg.norm(defect))
        if norm <= tolerance * max(1.0, float(np.linalg.norm(iterate))):
            return image, iteration, norm

        if depth > 0:
            iterates.append(iterate)
            defects.append(defect)
            if len(iterates) > depth + 1:
                del iterates[0]
                del defects[0]
            if len(iterates) > 1:
                step_dofs = np.column_stack(
                    [iterates[i + 1] - iterates[i] for i in range(len(iterates) - 1)]
                )
                step_defects = np.column_stack(
                    [defects[i + 1] - defects[i] for i in range(len(defects) - 1)]
                )
                gamma = np.linalg.lstsq(step_defects, defect, rcond=None)[0]
                iterate = image - step_dofs @ gamma - step_defects @ gamma
                continue
        iterate = image

    raise RuntimeError(
        f"Fixed-point iteration did not converge after {max_iterations} iterations, "
        f"final defect norm {norm}."
    )


def march(
    residual: Callable[[npt.NDArray[np.double], float], npt.NDArray[np.double]],
    y0: npt.NDArray[np.double],
    dt: float | Sequence[float],
    n_steps: int,
    *,
    t0: float = 0.0,
    stages: int = 2,
    method: IntegrationMethod = IntegrationMethod.GAUSS,
    mass: npt.NDArray[np.double] | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
    anderson_depth: int = 4,
    invariant: Callable[[npt.NDArray[np.double], float], float] | None = None,
) -> MarchResult:
    r"""March a semi-discrete system in time with a geometric scheme.

    The system is given as :math:`M \\, \\mathrm{d}y / \\mathrm{d}t = r(y, t)`
    with an initial state. On every time slab the state is discretized as a
    0-form on the integration rule of the collocation tableau, and the slab
    system is solved by fixed-point iteration with Anderson acceleration. The
    residual is called repeatedly for the same arguments, so it must not have
    side effects. With Gauss rules all linear and quadratic invariants of the
    semi-discrete system are conserved up to the iteration tolerance and
    round-off; Gauss--Lobatto rules conserve only linear invariants.

    Parameters
    ----------
    residual : callable
        Function computing :math:`r(y, t)` for a state and a time.
    y0 : array
        Initial state at time :math:`t_0`.
    dt : float or sequence of float
        Step size, either the same for all steps or one value per step.
    n_steps : int
        Number of steps to take.
    t0 : float, default: 0.0
        Time of the initial state.
    stages : int, default: 2
        Number of collocation stages, equal to the accuracy knob of the scheme.
    method : IntegrationMethod, default: "gauss"
        Method of the integration rule in time.
    mass : array or None, default: None
        Dense mass matrix of the system, None for an identity mass matrix.
    tolerance : float, default: 1e-12
        Relative tolerance on the fixed-point defect of every step.
    max_iterations : int, default: 100
        Maximum number of fixed-point iterations per step.
    anderson_depth : int, default: 4
        Depth of the Anderson window, zero for plain fixed-point iteration.
    invariant : callable or None, default: None
        Function computing an invariant :math:`H(y, t)` for reporting.

    Returns
    -------
    MarchResult
        Times, states, and iteration diagnostics of the march.
    """
    initial = np.ascontiguousarray(y0, np.double)
    if initial.ndim != 1:
        raise ValueError(
            f"Initial state must be one-dimensional, got shape {initial.shape}."
        )
    if n_steps < 1:
        raise ValueError(f"Number of steps must be positive, got {n_steps}.")

    sizes = np.asarray(dt, np.double)
    if sizes.ndim == 0:
        sizes = np.full(n_steps, float(sizes))
    if sizes.shape != (n_steps,):
        raise ValueError(
            f"Step sizes must be a scalar or one per step, got shape {sizes.shape}."
        )
    if np.any(sizes <= 0.0):
        raise ValueError("Step sizes must be positive.")

    mass_factors: MassFactors = None
    if mass is not None:
        mass_matrix = np.ascontiguousarray(mass, np.double)
        if mass_matrix.shape != (initial.size, initial.size):
            raise ValueError(
                f"Mass matrix must have shape {(initial.size, initial.size)}, "
                f"got {mass_matrix.shape}."
            )
        lu, pivots = lu_factor(mass_matrix)
        mass_factors = (
            np.ascontiguousarray(lu, np.double),
            np.ascontiguousarray(pivots, np.int32),
        )

    tableau = collocation_tableau(stages, method)
    n_dofs = initial.size

    times = np.empty(n_steps + 1)
    states = np.empty((n_steps + 1, n_dofs))
    iterations: list[int] = []
    residual_norms: list[float] = []
    measure: Callable[[npt.NDArray[np.double], float], float] | None = invariant
    records = np.empty(n_steps + 1) if measure is not None else None

    state = initial.copy()
    time = float(t0)
    times[0] = time
    states[0] = state
    if measure is not None and records is not None:
        records[0] = float(measure(state, time))

    for step in range(n_steps):
        step_size = float(sizes[step])
        next_time = time + step_size
        scale = 0.5 * step_size
        stage_times = time + scale * (tableau.nodes + 1.0)

        def fixed_point(iterate: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
            """Apply the slab stage equations to a flattened stage vector."""
            stage_dofs = iterate.reshape(tableau.stages, n_dofs)
            rhs = _stage_residuals(residual, stage_dofs, stage_times, mass_factors)
            return (state + scale * (tableau.integration_matrix @ rhs)).ravel()

        try:
            converged, used, norm = _anderson_solve(
                fixed_point,
                np.tile(state, tableau.stages),
                tolerance,
                max_iterations,
                anderson_depth,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Time step {step} from {time} to {next_time} failed: {error}"
            ) from error

        rhs = _stage_residuals(
            residual, converged.reshape(tableau.stages, n_dofs), stage_times, mass_factors
        )
        state = state + scale * (tableau.weights @ rhs)
        time = next_time

        times[step + 1] = time
        states[step + 1] = state
        iterations.append(used)
        residual_norms.append(norm)
        if measure is not None and records is not None:
            records[step + 1] = float(measure(state, time))

    return MarchResult(
        times=times,
        states=states,
        iterations=iterations,
        residual_norms=residual_norms,
        invariant_values=records,
    )
