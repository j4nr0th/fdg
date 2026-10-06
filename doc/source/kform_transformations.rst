.. currentmodule:: fdg

.. _fdg_kform_transformations:

Transforming :math:`k`-forms
============================

As was mentioned in the section about :ref:`fdg_kform_types`, the :math:`k`-forms
described by :class:`KForm` are defined on the reference domain. As such, they need
to be transformed to the physical domain to be able to meaningfully interpret them.

To support that, several functions are provided:

Transformation rules
--------------------

Let :math:`F : \vec{\xi} \mapsto \vec{x}` be the :ref:`space mapping
<fdg_space_map>` with Jacobian :math:`J` and inverse map :math:`G` with
entries :math:`G_{i m} = \partial \xi_i / \partial x_m` (for :math:`M >
N` the pseudo-inverse satisfying :math:`J G = I_N`). The components of a
geometric object transform according to its type:

**Contravariant** (tangent vectors, :func:`transform_contravariant_to_target`).
A vector field with reference components :math:`v_i` maps to the physical
components

.. math::

    v'_m = \sum_{i = 1}^{N} \frac{\partial x_m}{\partial \xi_i}\, v_i
    = (J v)_m .

This is the push-forward of the vector field. With the Euclidean metric of
:ref:`fdg_math_background`, the physical components of a :math:`1`-form
lower trivially to a tangent vector, which is why the contravariant
transform is the one used to obtain the vector field of a lowered
:math:`1`-form.

**Covariant** (:math:`1`-forms, :func:`transform_covariant_to_target`).
A :math:`1`-form with reference components :math:`w_i` maps to the
physical components

.. math::

    w'_m = \sum_{i = 1}^{N} \frac{\partial \xi_i}{\partial x_m}\, w_i
    = (G^T w)_m = (J^{-T} w)_m .

This is the pullback of the covector: it is the transpose of the inverse
of the Jacobian.

**General :math:`k`-forms** (:func:`transform_kform_to_target` and
:func:`transform_kform_component_to_target`). The components transform
with the :math:`k`-th exterior power of the covariant rule, i.e. through
the basis transform :math:`\mathbf{T}^{(k)}` of
:meth:`SpaceMap.basis_transform`. The physical component :math:`b` is

.. math::

    \omega'_b = \sum_{a} T^{(k)}_{ab}\, \omega_a,
    \qquad
    T^{(k)}_{ab} = \sum_{\sigma \in S_k} \operatorname{sgn}(\sigma)
    \prod_{l = 1}^{k} \frac{\partial \xi_{i_l}}{\partial x_{j_{\sigma(l)}}},

the signed sum of products of inverse-map entries over permutations (see
:ref:`fdg_math_background`). :func:`transform_kform_to_target` transforms
all components at once; :func:`transform_kform_component_to_target`
transforms a single component. The transformed values are sampled at the
integration points of the space map.

.. autofunction:: transform_kform_to_target

.. autofunction:: transform_kform_component_to_target

.. autofunction:: transform_covariant_to_target

.. autofunction:: transform_contravariant_to_target
