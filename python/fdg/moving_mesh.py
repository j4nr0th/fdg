r"""Time-dependent space maps for moving-mesh and fluid-structure problems.

Degrees of freedom in :mod:`fdg` live on the fixed reference element, so a
space map that moves in time does not change the basis: it changes the mass
matrix and the operators built from it. For a map :math:`F(\xi, t)` with mesh
velocity :math:`w = \partial_t F`, the pullback identity for time-dependent
diffeomorphisms reads

.. math::

    \partial_t (F_t^* \omega) = F_t^* \big(\partial_t \omega + \mathcal{L}_w
    \omega\big),
    \qquad
    \mathcal{L}_w = \mathrm{d}\,\iota_w + \iota_w\,\mathrm{d}.

The mesh motion therefore enters in exactly two ways:

1. through the Lie derivative :math:`\mathcal{L}_w` on every :math:`k`-form
   field of the state, which for a top form reduces to advection by the
   relative velocity :math:`v - w`, and
2. through the mass matrix, which becomes :math:`M(t)` and is evaluated at the
   stage times of every slab.

No explicit :math:`\dot M` is ever assembled: the marcher works with the
contract :math:`M(t) \dot y = r(y, t)`, so the change of the volume element is
carried entirely by the stage values of :math:`M`. The discrete geometric
conservation law is then structural rather than a condition to verify, since
with a fluid velocity equal to the mesh velocity the advection operator and
the Lie derivative operator are the same discrete assembly and cancel exactly,
so free-stream preservation holds to the iteration tolerance.

Time dependence is specified by geometry degrees of freedom sampled at the
collocation nodes of every slab, which makes the mesh velocity an exact
spectral derivative of that interpolant rather than a finite difference. The
same tableau that integrates the state differentiates the geometry.

The velocity is passed to the interior product as its physical components,
which is what that function documents; the map then applies its own metric
factors. Uniform stretch, translation and curved deformation are all covered
for the mesh motion itself; see the scope note below for the operators.

The fluid-structure pattern uses the same primitives with the geometry as part
of the state: the geometry degrees of freedom are appended to the marched
vector, and the residual rebuilds the maps of the current iterate with
:func:`space_maps_from_geometry_dofs`. The coupled system is autonomous in that
case, so its quadratic invariants are conserved exactly by the Gauss
collocation scheme.

.. _fdg_moving_mesh_scope:

Scope of the operators
----------------------

:func:`lie_derivative_operator` and :func:`advection_operator` are assembled and
verified for the top form of a one-dimensional element, which is the case the
moving-mesh march uses. The building blocks they compose are all available for every
``0 <= k <= ndim``, and the shapes work out in higher dimensions: an incidence
matrix built from ``incidence_kform_operator(specs, np.eye(n_k))`` is
``(n_{k+1}, n_k)``, an empty ``(0, n_k)`` when ``k == ndim``, and the interior
product returns the weak pairing ``(n_{k-1}, n_k)`` for a ``k`` form.

What is still open is the metric the strong form has to undo. Two facts are
established and are worth keeping:

* the incidence operator is exact on the standard degrees of freedom, in the
  sense that ``incidence_kform_operator(KFormSpecs(0, b), eye(n_0)) @ dofs(f)``
  reproduces ``dofs(f')`` to about 1e-13, and
* the interior product reproduces the weak pairing
  ``int psi_a (w . phi_b) dxi`` to about 1e-16 with the vector field supplied
  as unscaled physical components, which is what
  :func:`~fdg.compute_kform_interior_product_matrix` documents.

Turning the weak pairing into strong degrees of freedom inverts the row mass
that pairs with it, and that mass is where the determinant enters. On an affine
map the determinant is constant and any consistent choice gives the same
operator to machine precision; on a curved map the choices differ, and the one
that reproduces :math:`\mathrm{d}(c\,w)` has not yet been pinned down. Until it
is, these two operators are documented as one-dimensional and affine only,
rather than being reported as general.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fdg._fdg import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    KFormSpecs,
    SpaceMap,
    compute_kform_interior_product_matrix,
    compute_kform_mass_matrix,
    incidence_matrix,
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


def _raised_space(space: FunctionSpace, delta: int) -> FunctionSpace:
    """Return the space with every order shifted by ``delta``.

    Parameters
    ----------
    space : FunctionSpace
        Space to shift.
    delta : int
        Order increment applied to every dimension.

    Returns
    -------
    FunctionSpace
        Space with the shifted orders.
    """
    return FunctionSpace(
        *[BasisSpecs(spec.type, spec.order + delta) for spec in space.basis_specs]
    )


def _velocity(
    smap: SpaceMap, velocity_points: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
    """Return the physical velocity components in the expected shape.

    :func:`~fdg.compute_kform_interior_product_matrix` contracts the physical
    components of the vector field directly and applies the metric factors of
    the map itself, so the components are passed through unscaled.

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
        Components with the shape expected by
        :func:`~fdg.compute_kform_interior_product_matrix`.
    """
    components = np.ascontiguousarray(velocity_points, np.double)
    expected = np.asarray(smap.determinant, np.double).size
    flat = components.reshape(components.shape[0], -1) if components.ndim > 1 else None
    values = flat if flat is not None else components.reshape(1, -1)
    if values.shape[1] != expected:
        raise ValueError(
            f"Velocity must have one value per integration point of the map: "
            f"expected {expected}, got {values.shape[1]}."
        )
    return components


def lie_derivative_operator(
    smap: SpaceMap,
    specs: KFormSpecs,
    velocity_points: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    r"""Assemble the Lie derivative of a form with respect to a mesh velocity.

    This is :math:`\mathcal{L}_w = \mathrm{d}\,\iota_w + \iota_w\,\mathrm{d}`.

    Parameters
    ----------
    smap : SpaceMap
        Map of the element.
    specs : KFormSpecs
        Specification of the form the operator acts on.
    velocity_points : array
        Physical components of the mesh velocity at the integration points
        of the map, of shape ``(n_axes, n_points)``, passed through
        unscaled.

    Returns
    -------
    array
        Operator on the degrees of freedom of the form.

    Raises
    ------
    ValueError
        If the form is of the ambient dimension and its Lie derivative is
        requested for the second term only, or if the velocity shape does not
        match the map.
    """
    order = int(specs.order)
    dimension = int(specs.dimension)
    basis = specs.base_space

    # The interior product lowers the form order but keeps the space of the
    # form itself, so its test space is the base space and the exterior
    # derivative of the result is the incidence of that same base space.
    interior = compute_kform_interior_product_matrix(
        smap, max(order, 1), basis, basis, _velocity(smap, velocity_points)
    )
    first = incidence_matrix(basis.basis_specs[0]) @ np.linalg.solve(
        _mass(KFormSpecs(max(order - 1, 0), basis), smap), interior
    )

    # iota_w d is zero for a form of the ambient dimension, since its exterior
    # derivative vanishes.
    if order >= dimension:
        return first
    upper_basis = _raised_space(basis, 1)
    upper_interior = compute_kform_interior_product_matrix(
        smap, order + 1, basis, upper_basis, _velocity(smap, velocity_points)
    )
    second = np.linalg.solve(
        _mass(KFormSpecs(order, basis), smap), upper_interior
    ) @ incidence_matrix(basis.basis_specs[0])
    return first + second


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
        points of the map, of shape ``(n_axes, n_points)``, passed through
        unscaled.

    Returns
    -------
    array
        Operator on the degrees of freedom of the form.
    """
    order = int(specs.order)
    basis = specs.base_space
    interior = compute_kform_interior_product_matrix(
        smap, max(order, 1), basis, basis, _velocity(smap, velocity_points)
    )
    return incidence_matrix(basis.basis_specs[0]) @ np.linalg.solve(
        _mass(KFormSpecs(max(order - 1, 0), basis), smap), interior
    )


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
            ``(n_elements, n_axes, n_points)``.
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

        The returned callable serves the exact stage times of the mesh and
        nothing else, so it raises for any other time rather than
        interpolating silently. The matrices are block diagonal over the
        elements; the hybridized coupling of the existing constraint machinery
        composes with them but is not applied here.

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
