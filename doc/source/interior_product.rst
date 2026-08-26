.. currentmodule:: fdg

.. _fdg_interior_product:

Interior Product
================

Besides :ref:`fdg_inner_product` and :ref:`fdg_incidence`, the last key operation to
deal with is the interior product. This involves applying a tangent vector field
(which may be the result of lowering a 1-form) to a :math:`k`-form. In the context
of FEM solvers, the actual form of this operation that is interesting taking an
inner product with a :math:`(k - 1)`-form weight. The mass matrix that is the result
of doing so is computed using :func:`compute_kform_interior_product_matrix`.

Mathematical formulation
------------------------

The interior product of a :math:`k`-form
:math:`\omega = \sum_{i_1 < \dots < i_k} \omega_{i_1 \dots i_k}\,
\mathrm{d}\xi_{i_1} \wedge \dots \wedge \mathrm{d}\xi_{i_k}` with a
vector field :math:`X = \sum_m X^m\, \partial / \partial x_m` is the
:math:`(k - 1)`-form that contracts the vector field with the first slot
of the form (see :ref:`fdg_math_background`),

.. math::

    \iota_X \omega = \sum_{i_1 < \dots < i_k} \sum_{j = 1}^{k}
    (-1)^{j - 1}\, X^{i_j}\, \omega_{i_1 \dots i_k}\,
    \mathrm{d}\xi_{i_1} \wedge \dots \wedge \widehat{\mathrm{d}\xi_{i_j}}
    \wedge \dots \wedge \mathrm{d}\xi_{i_k},

where the hat marks a removed factor. Each term removes one index of the
form and contracts it with the vector field, with the sign of the
permutation that brings the removed index to the front.

:func:`compute_kform_interior_product_matrix` assembles the matrix of the
Euclidean inner product of a :math:`(k - 1)`-form test with
:math:`\iota_X \omega` on the physical domain:

.. math::

    M_{\alpha i} = \int_{[-1,1]^N}
    \left\langle \phi^{(\text{test})}_\alpha,\,
    \iota_X \phi^{(\text{trial})}_i \right\rangle\,
    |\det J|\, \mathrm{d}\vec{\xi},

where the pointwise inner product contracts the physical components
(through the basis transforms :math:`\mathbf{T}^{(k-1)}` and
:math:`\mathbf{T}^{(k)}` of :meth:`SpaceMap.basis_transform` and the
physical vector-field components :math:`X^m` supplied as
``vector_field_components``). The rows are the degrees of freedom of the
:math:`(k - 1)`-form test space, the columns those of the :math:`k`-form
trial space; the sign :math:`(-1)^{j-1}` of each contraction appears in
the assembly. For the top form :math:`k = N` the :math:`1/\det J` factor
of :math:`T^{(N)}` cancels the volume element, so the integrand reduces
to the contraction with the vector field only.

.. autofunction:: compute_kform_interior_product_matrix
