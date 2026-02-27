.. currentmodule:: fdg

.. _fdg_kform_transformations:

Transforming :math:`k`-forms
============================

As was mentioned in the section about :ref:`fdg_kform_types`, the :math:`k`-forms
described by :class:`KForm` are defined on the reference domain. As such, they need
to be transformed to the physical domain to be able to meaningfully interpret them.

To support that, several functions are provided:

.. autofunction:: transform_kform_to_target

.. autofunction:: transform_kform_component_to_target

.. autofunction:: transform_covariant_to_target

.. autofunction:: transform_contravariant_to_target
