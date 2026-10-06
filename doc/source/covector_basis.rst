.. currentmodule:: fdg

.. _fdg_covector_basis:

Covector Basis
==============

Since :math:`k`-forms use covector bundles as basis for different components,
utilities for dealing with these in Python are provided by the :class:`CovectorBasis`.
This type supports the wedge product using the XOR ``^`` operator, as well as the Hodge
using the inversion ``~`` operator.

The :class:`CovectorBasis` type is primarily intended to be used to help with sorting
and classifying different :math:`k`-form components.

Representation and algebra
--------------------------

A wedge monomial :math:`\mathrm{d} x_{i_1} \wedge \dots \wedge
\mathrm{d} x_{i_k}` is represented as the set of its indices (a bit mask)
together with a *sign*: the parity of the permutation that orders the
index set (see :ref:`fdg_math_background`). The constructor takes the
dimension :math:`n` of the ambient space and the sorted, non-repeating
indices; the rank is the number of indices, i.e. the number of set bits.

The algebra on :class:`CovectorBasis` mirrors the exterior algebra:

- **Wedge product** (``^``). Merges the index sets; the sign is the parity
  of the permutation of the merged set, which the implementation computes
  by flipping the sign once for every index of the second factor that is
  smaller than an index of the first. If the index sets overlap, the
  product is the zero basis (:math:`\mathrm{d} x_i \wedge \mathrm{d} x_i
  = 0`).

- **Hodge star** (``~``). Replaces the index set by its complement within
  the dimension; the sign is the parity of the permutation that sorts the
  concatenation :math:`(i_1, \dots, i_k, j_1, \dots, j_{N-k})`, i.e. the
  Euclidean Hodge star of :ref:`fdg_math_background` with the ordered
  basis.

- **Comparison operators**. Bases of the same rank and dimension are
  ordered lexicographically by their index sets; the ordering reverses for
  ranks above :math:`n/2`, so that a basis and its Hodge dual sort
  adjacent to each other. This makes :class:`CovectorBasis` usable as a
  key when grouping :math:`k`-form components.

- :meth:`~CovectorBasis.normalize` splits the sign from the index set and
  returns it as a separate :math:`\pm 1` factor together with a
  sign-less basis.

.. autoclass:: CovectorBasis
