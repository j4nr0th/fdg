.. currentmodule:: fdg

Covector Basis
==============

Since :math:`k`-forms use covector bundles as basis for different components,
utilities for dealing with these in Python are provided by the :class:`CovectorBasis`.
This type supports the wedge product using the XOR ``^`` operator, as well as the Hodge
using the inversion ``~`` operator.

The :class:`CovectorBasis` type is primarily intended to be used to help with sorting
and classifying different :math:`k`-form components.

.. autoclass:: CovectorBasis
