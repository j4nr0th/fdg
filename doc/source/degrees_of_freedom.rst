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

Reconstruction
--------------

For a tensor-product basis :math:`\psi_{i_1 \dots i_N}(\vec{\xi}) =
\prod_k b^k_{i_k}(\xi_k)` (see :ref:`fdg_basis_functions`) the value of
the function with degrees of freedom :math:`c_{i_1 \dots i_N}` at a point
:math:`\vec{\xi}` is

.. math::

    f(\vec{\xi}) = \sum_{i_1, \dots, i_N}
    c_{i_1 \dots i_N}
    \prod_{k = 1}^{N} b^k_{i_k}(\xi_k) .

Evaluating the one-dimensional basis values (and, for the gradient, the
derivatives) at the nodes of an integration space and contracting with the
DoFs gives the reconstructed values at all integration points at once;
this is what the two cached methods do. Because the one-dimensional bases
are polynomial of degree :math:`o_k`, the reconstructed function is a
polynomial of degree :math:`o_k` in each reference coordinate, and a
derivative along dimension :math:`k` lowers the achievable order in that
dimension by one.

Derivatives and projections
---------------------------

- :meth:`DegreesOfFreedom.derivative` applies the one-dimensional
  incidence operator along the given reference dimension (see
  :ref:`fdg_incidence`) to the DoFs, producing the DoFs of the partial
  derivative :math:`\partial f / \partial \xi_k` in the space whose order
  is lowered by one in that dimension.

- :meth:`DegreesOfFreedom.plane_projection` evaluates the basis functions
  of dimension :math:`k` at the coordinate :math:`x`, i.e. it restricts the
  function to the plane :math:`\xi_k = x` and returns the DoFs of the
  restriction on that plane. This is the operation behind
  :meth:`SpaceMap.boundary`.

- :meth:`DegreesOfFreedom.reverse_orientation` maps the reference
  coordinate :math:`\xi_k \in [-1, +1]` to :math:`-\xi_k`, i.e. it reverses
  the orientation of the element along dimension :math:`k`. The DoFs are
  relabeled so that the represented function is unchanged: for nodal
  (Lagrange, Bernstein) bases the nodal values are reversed, for the
  Legendre basis the coefficients of odd order are negated, since
  :math:`P_k(-\xi) = (-1)^k P_k(\xi)`.

- :meth:`DegreesOfFreedom.lagrange_projection` re-expresses the function
  in a Lagrange basis of the given (or the smallest sufficient) orders by
  interpolation of nodal values, which is used when the function must be
  sampled at its nodal points.

.. autoclass:: DegreesOfFreedom

As a minor utility for reconstructing :class:`DegreesOfFreedom` at arbitrary points
:func:`reconstruct` is provided.

.. autoclass:: reconstruct
