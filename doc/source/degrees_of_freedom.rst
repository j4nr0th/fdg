.. currentmodule:: fdg

.. _fdg_degrees_of_freedom:

Degrees of Freedom
==================

With a defined function space using :class:`FunctionSpace`, it is possible to now define a function
using a finite number of degrees of freedom (DoF). To help with that,
the :class:`DegreesOfFreedom` is provided. A new :class:`DegreesOfFreedom`
object is created by specifying the :class:`FunctionSpace` and optionally values of the
corresponding DoFs.

This type can be used to reconstruct the values of the function or its gradients,
thought they are never cached. The only exceptions are the partially cached methods
:meth:`DegreesOfFreedom.reconstruct_at_integration_points` and
:meth:`DegreesOfFreedom.reconstruct_derivative_at_integration_points`, which make use
of :class:`BasisRegistry` and :class:`IntegrationRegistry` to cache values of basis
functions at integration points.

.. autoclass:: DegreesOfFreedom

As a minor utility for reconstructing :class:`DegreesOfFreedom` at arbitrary points
:func:`reconstruct` is provided.

.. autoclass:: reconstruct
