.. currentmodule:: fdg

.. _fdg_moving_mesh:

Moving Meshes
=============

Degrees of freedom in this library live on the fixed reference element, so a
space map that moves in time does not change the basis. It changes the mass
matrix and the operators built from it, and those changes enter a time-dependent
problem in exactly two places.

The Lie derivative
------------------

For a map :math:`F(\xi, t)` with mesh velocity :math:`w = \partial_t F`, the
pullback identity for time-dependent diffeomorphisms is

.. math::

    \partial_t (F_t^* \omega) = F_t^* \big(\partial_t \omega +
    \mathcal{L}_w \omega\big), \qquad
    \mathcal{L}_w = \mathrm{d}\,\iota_w + \iota_w\,\mathrm{d}.

The mesh motion therefore contributes :math:`\mathcal{L}_w` on every
:math:`k`-form field of the state, plus the mass evaluated per stage as
:math:`M(t)`. No :math:`\dot M` is ever assembled: under the marcher contract
:math:`M(t)\dot y = r(y, t)` the change of the volume element is carried entirely
by the stage values of :math:`M`.

For a form of the ambient dimension the exterior derivative vanishes, so
:math:`\mathcal{L}_w` reduces to advection by the relative velocity
:math:`v - w`. That makes the discrete geometric conservation law structural: at
:math:`v = w` the advection and Lie derivative assemblies are the same discrete
operator and cancel exactly, so a density carried along by the mesh does not
move in the reference frame to better than the solver tolerance.

.. autofunction:: lie_derivative_operator

.. autofunction:: advection_operator

.. autofunction:: space_maps_from_geometry_dofs

.. autofunction:: stage_mass

Prescribed motion
-----------------

Time dependence is specified as geometry degrees of freedom sampled at the
collocation nodes of every slab, which makes the mesh velocity a spectral
derivative of that interpolant rather than a finite difference. The same tableau
that integrates the state differentiates the geometry, so the two are treated
consistently. The velocity reaches the interior product as unscaled physical
components, which is that function's convention; the map applies its own metric
factors.

.. autoclass:: MovingMesh
   :members:

Fluid-structure coupling
------------------------

The fluid-structure pattern reuses the same primitives with the geometry as part
of the state: append the geometry degrees of freedom to the marched vector, and
have the residual rebuild the maps of the current iterate with
:func:`space_maps_from_geometry_dofs`. The resulting joint system is autonomous
in time, so the quadratic invariants of the coupled problem are conserved exactly
by the Gauss collocation scheme rather than to the truncation error of the
timestep.

:func:`MovingMesh.mass_factory` returns the matrices block diagonal over the
elements; the hybridized continuity constraints of :ref:`fdg_mesh` compose with
them but are not applied by this module.
