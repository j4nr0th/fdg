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

.. autoclass:: KFormSpecs

.. autoclass:: KForm
