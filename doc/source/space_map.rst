.. currentmodule:: fdg

.. _fdg_space_map:

Space Map
=========

Since the :ref:`fdg_basis_functions` and :ref:`fdg_integration` are both done on hypercube domain, where
each dimension goes from -1 to +1, this severely limits their usability on any deformed domain. As such
a mapping between the :math:`N`-dimensional reference domain and the target :math:`M`-dimensional (where
:math:`M \ge N`) target domain can be defined using :class:`SpaceMap` object. This mapping then determines
the way integration is done and how `k`-form components are mapped between the two domains.

The mapping is specified one target coordinate at the time, using :class:`CoordinateMap` objects. These
require the :class:`DegreesOfFreedom` as well as the :class:`IntegrationSpace` to define. This map is
used to store the values, as well as the derivatives of the coordinate mapping at all points in the integration
space.

.. note::

    All :class:`CoordinateMap` object that you want to use together for a complete :class:`SpaceMap` must use
    the same :class:`IntegrationSpace`.

The mapping, its Jacobian and the determinant
---------------------------------------------

A :class:`SpaceMap` represents the map

.. math::

    F : [-1, +1]^N \to \mathbb{R}^M, \qquad
    \vec{x}(\vec{\xi}) =
    \begin{bmatrix} x_1(\vec{\xi}) \\ \vdots \\ x_M(\vec{\xi}) \end{bmatrix},

where each coordinate :math:`x_m` is the function represented by the
degrees of freedom of the :math:`m`-th :class:`CoordinateMap`. The
Jacobian at an integration point is the :math:`M \times N` matrix of
reference derivatives

.. math::

    J_{mi} = \frac{\partial x_m}{\partial \xi_i},

computed from the gradients of the coordinate maps. From the QR
decomposition :math:`J = Q R` (computed with Givens rotations; the stored
factor is :math:`Q^T`, so :math:`J = Q^T R` holds) the library obtains

- the *determinant* :math:`\det J = \prod_i R_{ii}`, the product of the
  diagonal of :math:`R`. Because the rotations are chosen with a
  non-negative diagonal, this quantity is always :math:`\ge 0`: it is the
  *unsigned* volume factor :math:`|\det J|` used as the measure
  :math:`\mathrm{d}V = |\det J|\, \mathrm{d}\xi_1 \cdots \mathrm{d}\xi_N`
  everywhere integration is done (see :ref:`fdg_math_background`);

- the *inverse map* :math:`G = R_{11}^{-1} Q_{\text{top}}`, where
  :math:`R_{11}` is the upper-triangular :math:`N \times N` part of
  :math:`R` and :math:`Q_{\text{top}}` the first :math:`N` rows of
  :math:`Q^T`. It satisfies

  .. math::

      J\, G = I_N,

  i.e. it is a right inverse of the rectangular Jacobian. For
  :math:`M = N` it is the ordinary inverse :math:`J^{-1}`; for :math:`M >
  N` it is the (least-squares) pseudo-inverse. Its entries are the
  derivatives :math:`\partial \xi_i / \partial x_m` of the reference
  coordinates with respect to the physical ones, which are what the
  covariant transformation of :ref:`fdg_kform_transformations` needs.

The basis transform
-------------------

:meth:`SpaceMap.basis_transform` returns, for a form order :math:`k`, the
three-index array :math:`T^{(k)}_{ab}(\vec{\xi})` that expresses how the
reference wedge basis :math:`\mathrm{d}\xi_a` (component :math:`a` of the
reference :math:`k`-form) contributes to the physical wedge basis
:math:`\mathrm{d}x_b` (component :math:`b` of the physical :math:`k`-form)
at each integration point. It is the signed sum over permutations of
products of inverse-map entries

.. math::

    T^{(k)}_{ab} = \sum_{\sigma \in S_k} \operatorname{sgn}(\sigma)
    \prod_{l = 1}^{k} \frac{\partial \xi_{i_l}}{\partial x_{j_{\sigma(l)}}},
    \qquad a = \{i_1, \dots, i_k\},\; b = \{j_1, \dots, j_k\},

with the special cases :math:`T^{(1)}_{ab} = \partial \xi_a / \partial
x_b` and, for a square map, :math:`T^{(N)} = 1 / \det J`. This array
drives the k-form mass matrix (:ref:`fdg_inner_product`) and the k-form
transforms (:ref:`fdg_kform_transformations`).

Sampled space maps
------------------

:class:`SampledSpaceMap` provides a tensor-grid representation of a
:class:`SpaceMap` for visualization and for evaluating transformed fields on
user-selected reference-space points. Its constructor accepts one one-dimensional
sample array per reference dimension; the arrays may be non-uniform and may have
different lengths. The samples define the tensor grid, so the resulting arrays
have one axis per sample array.

The convenience constructor :meth:`SampledSpaceMap.on_uniform_grid` creates the
same representation for uniformly spaced points on :math:`[-1, +1]` from an
order per reference dimension. A sampled map cannot be used for integration.

For a sampled k-form, use :func:`transform_kform_to_target_sampled`. The function
accepts orders :math:`k \ge 1`; a 0-form is a scalar field and needs no coordinate
transformation, so its values should be used directly instead.

.. autoclass:: SampledSpaceMap

.. autofunction:: transform_kform_to_target_sampled


Restriction to boundaries
-------------------------

:meth:`SpaceMap.boundary` restricts the map to the face
:math:`\xi_{\mathrm{idim}} = \pm 1` by projecting each coordinate map with
:meth:`DegreesOfFreedom.plane_projection` and dropping the fixed axis from
the integration space. The result is a :class:`SpaceMap` from the
remaining :math:`N - 1` reference dimensions to the same physical space.
Its Jacobian is the :math:`M \times (N - 1)` *tangential* Jacobian of the
face, so its determinant is the *surface measure* of the face and its
inverse map provides the *tangential pullback* used by
:ref:`fdg_boundary_constraints` for trace constraints. Note that
:meth:`SpaceMap.boundary` does not apply any orientation sign: the sign
conventions of the element-face orientation are handled by the topology
and constraint code.

.. autoclass:: CoordinateMap

To specify the :class:`SpaceMap`, :class:`CoordinateMap` must be specified for each of the target domain
dimensions. With that, :class:`SpaceMap` can be used in place of :class:`IntegrationSpace` for many functions
that integrate quantities. It also contains both the Jacobian and its (pseudo-)inverse, along with the determinant.

.. autoclass:: SpaceMap
