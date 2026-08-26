.. currentmodule:: fdg

.. _fdg_kform_types:

:math:`k`-form Type
===================

While :ref:`fdg_degrees_of_freedom` may be used to describe a function defined on a reference domain,
this is not very useful when dealing with :math:`k`-forms, since they have multiple components, with
their functions spaces being variations of the same base space, depending on their covector basis bundle.
As such, two types are provided:

- To specify and obtain information about :math:`k`-forms there is :class:`KFormSpecs` type.
- To hold and manipulate degrees of freedom of components there is :class:`KForm` type.

Component structure
-------------------

A :math:`k`-form on the :math:`N`-dimensional reference domain is a sum
over the :math:`\binom{N}{k}` wedge monomials (see
:ref:`fdg_math_background`),

.. math::

    \omega = \sum_{i_1 < \dots < i_k}
    \omega_{i_1 \dots i_k}\,
    \mathrm{d} \xi_{i_1} \wedge \dots \wedge \mathrm{d} \xi_{i_k}.

:class:`KFormSpecs` describes such a form by its *order* :math:`k` and its
*base space*: the :class:`FunctionSpace` used for the scalar components of
the :math:`0`-forms. The component with active axes
:math:`\{i_1, \dots, i_k\}` is expanded in the base space with the order
lowered by one on exactly those axes, so its degrees of freedom live in
the space returned by :meth:`KFormSpecs.get_component_function_space`; the
wedge monomial itself is returned by
:meth:`KFormSpecs.get_component_basis` as a :class:`CovectorBasis`.

The components are ordered lexicographically by their ascending axis sets
(:math:`\{1, 2\}, \{1, 3\}, \{2, 3\}, \dots`), which is the order of the
flattened degrees of freedom of a :class:`KForm`. The flattened array of a
:math:`k`-form is the concatenation of the component arrays in that order;
:meth:`KFormSpecs.get_component_slice` returns the slice of the flattened
array that corresponds to one component and
:meth:`KFormSpecs.component_dof_counts` the length of each slice.

The exterior derivative raises the order by one
(:math:`\mathrm{d}: \Omega^k \to \Omega^{k+1}`, see
:ref:`fdg_incidence`), the interior product with a vector field lowers it
by one (:ref:`fdg_interior_product`), and the Hodge star maps
:math:`\Omega^k \to \Omega^{N-k}` (:ref:`fdg_math_background`).

.. autoclass:: KFormSpecs

.. autoclass:: KForm
