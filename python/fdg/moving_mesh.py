r"""Time-dependent space maps for moving-mesh and fluid-structure problems.

Degrees of freedom live on the fixed reference element, so a map that moves in
time does not change the basis: it changes the mass and the operators built from
it. For a map :math:`F(\xi, t)` with mesh velocity :math:`w = \partial_t F`, the
pullback identity

.. math::

    \partial_t (F_t^* \omega) = F_t^* \big(\partial_t \omega +
    \mathcal{L}_w \omega\big), \qquad
    \mathcal{L}_w = \mathrm{d}\,\iota_w + \iota_w\,\mathrm{d}

puts the mesh motion in exactly two places:

1. :math:`\mathcal{L}_w` on every :math:`k`-form field, which for a top form is
   advection by the relative velocity :math:`v - w`;
2. the mass, evaluated per stage as :math:`M(t)`.

No :math:`\dot M` is ever assembled. Under the marcher contract
:math:`M(t)\dot y = r(y, t)` the change of the volume element is carried entirely
by the stage values of :math:`M`, which makes the discrete geometric
conservation law structural: at :math:`v = w` the advection and Lie derivative
assemblies are identical and cancel exactly, so free-stream preservation holds to
the iteration tolerance.

Time dependence enters as geometry degrees of freedom sampled at the collocation
nodes of every slab, so the mesh velocity is a spectral derivative of that
interpolant. The tableau that integrates the state differentiates the geometry.

The velocity reaches the interior product as unscaled physical components, which
is that function's documented convention; the map supplies its own metric
factors.

The fluid-structure pattern reuses the same primitives with the geometry as part
of the state: append the geometry degrees of freedom to the marched vector and
rebuild the maps of the current iterate with
:func:`space_maps_from_geometry_dofs`. The coupled system is then autonomous, so
Gauss collocation conserves its quadratic invariants exactly.

.. _fdg_moving_mesh_scope:

Operator assembly
-----------------

:func:`lie_derivative_operator` and :func:`advection_operator` compose three
library operations whose contracts fix the assembly:

* :func:`~fdg.compute_kform_interior_product_matrix` returns the weak pairing
  ``(n_{k-1}, n_k)``, mapping the ``k``-form of ``basis_right`` onto the
  ``(k - 1)``-form of ``basis_left``;
* :func:`~fdg.incidence_kform_operator` applied from the right returns the
  ``(n_{k+1}, n_k)`` exterior derivative, so coefficients map as ``y @ D``. A
  form of the ambient dimension has an empty derivative;
* :func:`~fdg.compute_kform_mass_matrix` gives the mass of a form on the map.

Inverting the mass of the ``(k - 1)``-form test space makes the pairing strong;
the two terms then sit on opposite sides of it, ``D M_{k-1}^{-1} I`` for
:math:`\mathrm{d}\,\iota_w` and ``M_k^{-1} I D`` for :math:`\iota_w\,\mathrm{d}`.

Verified in one to four dimensions, all against closed forms built from the
library's own basis tables: every order assembles and annihilates the constant
mode; a zero-form gives :math:`w f'` to 1e-12 on a curved element; an axis-aligned
problem reproduces the one-dimensional operator to 1e-14; and the top form gives
the divergence to 1e-13 and equals advection exactly. The top-form density is the
physical divergence times the map determinant, because the interior product scales
its pairing by that determinant.

These hold to machine precision wherever the polynomial space represents the
answer; higher-degree input is projected, which is a property of the space.

:class:`MovingMesh` is dimension-agnostic. Geometry degrees of freedom carry one
row per axis and the integration space one rule per axis, and the velocity comes
back shaped ``(n_elements, n_axes, npts_0, ..., npts_{ndim-1})``, which is what
the operators expect per element.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fdg._fdg import (
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    KFormSpecs,
    SpaceMap,
    compute_kform_interior_product_matrix,
    compute_kform_mass_matrix,
    incidence_kform_operator,
)
from fdg.degrees_of_freedom import reconstruct
from fdg.enum_type import IntegrationMethod
from fdg.time_marching import collocation_tableau


def space_maps_from_geometry_dofs(
    geometry_space: FunctionSpace,
    integration: IntegrationSpace,
    geometry_dofs: npt.NDArray[np.double],
) -> list[SpaceMap]:
    """Build one space map per element from geometry degrees of freedom.

    Parameters
    ----------
    geometry_space : FunctionSpace
        Space of the geometry functions, shared by all axes and elements.
    integration : IntegrationSpace
        Integration space the maps are built on.
    geometry_dofs : array
        Geometry degrees of freedom of shape
        ``(n_elements, n_axes, n_geometry_dofs)``.

    Returns
    -------
    list of SpaceMap
        One map per element.

    Raises
    ------
    ValueError
        If the array does not have three axes.
    """
    dofs = np.asarray(geometry_dofs, np.double)
    if dofs.ndim != 3:
        raise ValueError(
            f"Geometry degrees of freedom must have shape (n_elements, n_axes, "
            f"n_dofs), got {dofs.shape}."
        )

    return [
        SpaceMap(
            *[
                CoordinateMap(
                    DegreesOfFreedom(
                        geometry_space,
                        np.ascontiguousarray(dofs[element, axis]),
                    ),
                    integration,
                )
                for axis in range(dofs.shape[1])
            ]
        )
        for element in range(dofs.shape[0])
    ]


def _mass(specs: KFormSpecs, smap: SpaceMap) -> npt.NDArray[np.double]:
    """Return the mass matrix of the given specs on an element map.

    Parameters
    ----------
    specs : KFormSpecs
        Specification of the form.
    smap : SpaceMap
        Map of the element.

    Returns
    -------
    array
        Mass matrix of the form on the element.
    """
    return compute_kform_mass_matrix(
        smap, specs.order, specs.base_space, specs.base_space
    )


def _velocity(
    smap: SpaceMap, velocity_points: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
    """Return the physical velocity components in the shape the interior product wants.

    That function applies the metric factors of the map itself, so the
    components pass through unscaled.

    Parameters
    ----------
    smap : SpaceMap
        Map providing the integration points.
    velocity_points : array
        Physical components of the vector field at the integration points of
        the map.

    Returns
    -------
    array
        Components shaped ``(n_axes, npts_0, ..., npts_k)``.

    Raises
    ------
    ValueError
        If a component does not have one value per integration point.
    """
    components = np.ascontiguousarray(velocity_points, np.double)
    expected = int(np.asarray(smap.determinant, np.double).size)
    # Accepted as (n_axes, npts_0, ..., npts_k) or as any array with one value
    # per integration point; neither is rescaled.
    per_component = components[0].size if components.ndim > 1 else components.size
    if components.ndim < 2:
        components = components.reshape(1, -1)
    if per_component != expected:
        raise ValueError(
            f"Velocity must have one value per integration point of the map: "
            f"expected {expected}, got {per_component}."
        )
    return components


def _incidence(specs: KFormSpecs) -> npt.NDArray[np.double]:
    """Return the exterior derivative of a k-form as a dense matrix.

    The matrix maps the degrees of freedom of the k-form to those of its
    (k + 1)-form derivative, so a coefficient vector transforms as ``y @ D``.
    Applying it from the right with an identity input returns that matrix
    directly. A form of the ambient dimension has an empty derivative, which is
    returned as an empty matrix.
    """
    order = int(specs.order)
    dimension = int(specs.base_space.dimension)
    if order == dimension:
        return np.zeros((0, int(sum(specs.component_dof_counts))))
    n_dofs = int(sum(KFormSpecs(order + 1, specs.base_space).component_dof_counts))
    return incidence_kform_operator(
        specs, np.ascontiguousarray(np.eye(n_dofs)), right=True
    )


def lie_derivative_operator(
    smap: SpaceMap,
    specs: KFormSpecs,
    velocity_points: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    r"""Assemble the Lie derivative of a form with respect to a mesh velocity.

    This is :math:`\mathcal{L}_w = \mathrm{d}\,\iota_w + \iota_w\,\mathrm{d}` for
    any form order. Each term puts the exterior derivative on the side of the
    strong-conversion mass matching its own form order.

    Parameters
    ----------
    smap : SpaceMap
        Map of the element.
    specs : KFormSpecs
        Specification of the form the operator acts on.
    velocity_points : array
        Physical components of the mesh velocity at the integration points
        of the map, shaped ``(n_axes, npts_0, ..., npts_k)``, passed through
        unscaled.

    Returns
    -------
    array
        Operator on the degrees of freedom of the form.

    Raises
    ------
    ValueError
        If the velocity does not have one value per integration point of the
        map.
    """
    order = int(specs.order)
    dimension = int(specs.dimension)
    basis = specs.base_space
    velocity = _velocity(smap, velocity_points)
    n_dofs = int(sum(specs.component_dof_counts))
    operator = np.zeros((n_dofs, n_dofs))

    # d iota_w: the weak pairing (n_{k-1}, n_k) made strong by the test mass.
    if order > 0:
        interior = compute_kform_interior_product_matrix(
            smap, order, basis, basis, velocity
        )
        test = KFormSpecs(order - 1, basis)
        # The exterior derivative acts on the (k - 1)-form, from the left.
        operator += _incidence(test) @ np.linalg.solve(_mass(test, smap), interior)

    # iota_w d: the trial space is the (k + 1)-form, so D composes from the right.
    # A form of the ambient dimension has an empty derivative, dropping the term.
    if order < dimension:
        upper_interior = compute_kform_interior_product_matrix(
            smap, order + 1, basis, basis, velocity
        )
        operator += np.linalg.solve(_mass(specs, smap), upper_interior) @ _incidence(
            specs
        )

    return operator


def advection_operator(
    smap: SpaceMap,
    specs: KFormSpecs,
    velocity_points: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    r"""Assemble the advection operator :math:`\mathrm{d}\,\iota_v` of a form.

    Parameters
    ----------
    smap : SpaceMap
        Map of the element.
    specs : KFormSpecs
        Specification of the form that is transported.
    velocity_points : array
        Physical components of the transport velocity at the integration
        points of the map, shaped ``(n_axes, npts_0, ..., npts_k)``, passed
        through unscaled.

    Returns
    -------
    array
        Operator on the degrees of freedom of the form.
    """
    order = int(specs.order)
    if order == 0:
        raise ValueError(
            "Advection by d iota_v is undefined for a zero-form, since the "
            "interior product has no form to contract."
        )
    basis = specs.base_space
    interior = compute_kform_interior_product_matrix(
        smap, order, basis, basis, _velocity(smap, velocity_points)
    )
    test = KFormSpecs(order - 1, basis)
    return _incidence(test) @ np.linalg.solve(_mass(test, smap), interior)


def stage_mass(smap: SpaceMap, specs: KFormSpecs) -> npt.NDArray[np.double]:
    """Return the mass matrix of the form on one element and stage.

    Parameters
    ----------
    smap : SpaceMap
        Map of the element at one stage.
    specs : KFormSpecs
        Specification of the form.

    Returns
    -------
    array
        Mass matrix of the form on the element.
    """
    return _mass(specs, smap)


@dataclass
class _StageGeometry:
    """Cached geometry of one element group at one stage."""

    maps: list[SpaceMap]
    velocity: npt.NDArray[np.double]


class MovingMesh:
    """Prescribed mesh motion sampled at the stage nodes of every slab.

    Parameters
    ----------
    geometry_dofs : callable
        Maps a time to the geometry degrees of freedom of all elements, of
        shape ``(n_elements, n_axes, n_geometry_dofs)``.
    dt : float or sequence of float
        Step size, either the same for all steps or one value per step.
    n_steps : int
        Number of steps the mesh is prepared for.
    element_count : int
        Number of elements the mesh has.
    geometry_space : FunctionSpace
        Space of the geometry functions, shared by all axes and elements.
    integration : IntegrationSpace
        Integration space the maps are built on.
    t0 : float, default: 0.0
        Time of the first step.
    stages : int, default: 2
        Number of collocation stages per slab, which must match the value
        passed to :func:`~fdg.march`.
    method : IntegrationMethod, default: "gauss"
        Method of the integration rule in time.

    Raises
    ------
    ValueError
        If the element count is not positive, a step size is not positive, or
        the geometry degrees of freedom at ``t0`` do not have the expected
        shape.
    """

    def __init__(
        self,
        geometry_dofs: Callable[[float], npt.NDArray[np.double]],
        dt: float | Sequence[float],
        n_steps: int,
        *,
        element_count: int,
        geometry_space: FunctionSpace,
        integration: IntegrationSpace,
        t0: float = 0.0,
        stages: int = 2,
        method: IntegrationMethod = IntegrationMethod.GAUSS,
    ) -> None:
        if element_count < 1:
            raise ValueError(f"Element count must be positive, got {element_count}.")
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

        start = np.ascontiguousarray(geometry_dofs(float(t0)), np.double)
        if start.ndim != 3 or start.shape[0] != element_count:
            raise ValueError(
                f"Geometry degrees of freedom must have shape "
                f"({element_count}, n_axes, n_dofs), got {start.shape}."
            )

        self._geometry_dofs = geometry_dofs
        self._geometry_space = geometry_space
        self._integration = integration
        self._element_count = element_count
        self._stages = stages
        self._sizes = sizes
        self._t0 = float(t0)
        self._tableau = collocation_tableau(stages, method)
        self._nodes = [
            np.ascontiguousarray(node, np.double) for node in self._integration.nodes()
        ]

        # Sample the geometry at every stage time and at every slab start. The
        # mesh motion is prescribed, so the start value is taken from the user
        # callable rather than accumulated, which keeps the rate exact instead
        # of degrading it by the quadrature-completion error of the state.
        self._stage_times = np.empty((n_steps, stages))
        self._stage_dofs = np.empty((n_steps, stages) + start.shape)
        # The time derivative of the geometry interpolant at the stage nodes.
        # The tableau integration matrix is exact on the nodal polynomial
        # space, so its inverse differentiates that interpolant exactly.
        inverse = np.linalg.inv(self._tableau.integration_matrix)
        self._stage_rates = np.empty_like(self._stage_dofs)

        for step in range(n_steps):
            begin = float(t0) + float(np.sum(sizes[:step]))
            scale = 0.5 * float(sizes[step])
            times = begin + scale * (self._tableau.nodes + 1.0)
            self._stage_times[step] = times
            for k in range(stages):
                self._stage_dofs[step, k] = self._geometry_dofs(float(times[k]))
            slab_start = self._geometry_dofs(begin)
            self._stage_rates[step] = (2.0 / float(sizes[step])) * np.einsum(
                "kj,j...->k...", inverse, self._stage_dofs[step] - slab_start
            )

        self._cache: dict[tuple[int, int], _StageGeometry] = {}

    @property
    def element_count(self) -> int:
        """Number of elements of the mesh."""
        return self._element_count

    @property
    def stages(self) -> int:
        """Number of collocation stages per slab."""
        return self._stages

    @property
    def n_steps(self) -> int:
        """Number of steps the mesh is prepared for."""
        return int(self._stage_times.shape[0])

    def stage_time(self, step: int, stage: int) -> float:
        """Return the time of one stage of one step.

        Parameters
        ----------
        step : int
            Index of the step.
        stage : int
            Index of the stage within the step.

        Returns
        -------
        float
            Time of the stage.
        """
        return float(self._stage_times[step, stage])

    def _stage(self, step: int, stage: int) -> _StageGeometry:
        """Return the cached geometry of one stage, building it on demand."""
        if not 0 <= step < self._stage_times.shape[0]:
            raise ValueError(f"Step index {step} is out of range.")
        if not 0 <= stage < self._stages:
            raise ValueError(f"Stage index {stage} is out of range.")

        key = (step, stage)
        if key not in self._cache:
            maps = space_maps_from_geometry_dofs(
                self._geometry_space, self._integration, self._stage_dofs[step, stage]
            )
            rates = self._stage_rates[step, stage]
            velocity = np.stack(
                [
                    np.stack(
                        [
                            reconstruct(
                                DegreesOfFreedom(
                                    self._geometry_space,
                                    np.ascontiguousarray(rates[element, axis]),
                                ),
                                *self._nodes,
                            )
                            for axis in range(rates.shape[1])
                        ]
                    )
                    for element in range(self._element_count)
                ]
            )
            self._cache[key] = _StageGeometry(maps=maps, velocity=velocity)
        return self._cache[key]

    def space_maps(self, step: int, stage: int) -> list[SpaceMap]:
        """Return the space maps of every element at one stage.

        Parameters
        ----------
        step : int
            Index of the step.
        stage : int
            Index of the stage within the step.

        Returns
        -------
        list of SpaceMap
            One map per element.
        """
        return self._stage(step, stage).maps

    def velocity(self, step: int, stage: int) -> npt.NDArray[np.double]:
        """Return the physical mesh velocity at the integration points.

        Parameters
        ----------
        step : int
            Index of the step.
        stage : int
            Index of the stage within the step.

        Returns
        -------
        array
            Components of the mesh velocity of shape
            ``(n_elements, n_axes, npts_0, ..., npts_{ndim-1})``.
        """
        return self._stage(step, stage).velocity

    def stage_masses(self, specs: KFormSpecs) -> list[list[list[npt.NDArray[np.double]]]]:
        """Return the per-element mass matrices at every stage.

        Parameters
        ----------
        specs : KFormSpecs
            Specification of the form whose mass matrix is needed.

        Returns
        -------
        list of list of list of array
            One entry per step, then per stage, then per element.
        """
        return [
            [
                [stage_mass(smap, specs) for smap in self.space_maps(step, stage)]
                for stage in range(self._stages)
            ]
            for step in range(self._stage_times.shape[0])
        ]

    def mass_factory(
        self, specs: KFormSpecs
    ) -> Callable[[float], npt.NDArray[np.double]]:
        """Return a callable giving the global mass matrix at a stage time.

        The callable serves the exact stage times and raises for anything else
        rather than interpolating silently. The matrices are block diagonal over
        the elements; hybridized coupling is not applied here.

        Parameters
        ----------
        specs : KFormSpecs
            Specification of the form whose mass matrix is needed.

        Returns
        -------
        callable
            Maps a stage time to the global mass matrix.
        """
        tolerance = 1e-9 * float(np.min(self._sizes))
        matrices = np.empty(
            (
                self._stage_times.shape[0],
                self._stages,
                self._element_count,
                self._stage_mass_size(specs),
                self._stage_mass_size(specs),
            ),
            dtype=object,
        )
        for step in range(self._stage_times.shape[0]):
            for stage in range(self._stages):
                matrices[step, stage] = [
                    stage_mass(smap, specs) for smap in self.space_maps(step, stage)
                ]

        def factory(t: float) -> npt.NDArray[np.double]:
            """Return the global mass matrix at the stage time ``t``."""
            matches = np.argwhere(np.abs(self._stage_times - t) <= tolerance)
            if matches.size == 0:
                raise ValueError(
                    f"Time {t} is not a stage time of this moving mesh; the "
                    f"mass matrix is only defined at the collocation nodes."
                )
            step, stage = matches[0]
            blocks = matrices[step, stage]
            size = blocks[0].shape[0]
            result = np.zeros((self._element_count * size, self._element_count * size))
            for element, block in enumerate(blocks):
                sl = slice(element * size, (element + 1) * size)
                result[sl, sl] = block
            return result

        return factory

    def _stage_mass_size(self, specs: KFormSpecs) -> int:
        """Return the number of degrees of freedom of the form per element."""
        return int(sum(specs.component_dof_counts))
