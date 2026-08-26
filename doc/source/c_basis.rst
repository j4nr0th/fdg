Basis Sets
==========

Basis function sets of various types (Legendre, Lagrange, Bernstein),
their precomputation at integration nodes, the basis set registry, and the
covector basis algebra used for differential forms.

A basis set of *order* :math:`o` holds the :math:`o + 1` basis functions
of degree :math:`o` evaluated (together with their first derivatives) at
the nodes of an integration rule; the tensor products of these one-
dimensional sets span the :ref:`fdg_basis_functions` function spaces. The
covector basis algebra encodes the wedge monomials
:math:`\mathrm{d} x_{i_1} \wedge \dots \wedge \mathrm{d} x_{i_k}` of the
:ref:`fdg_math_background` exterior algebra: each basis is a bit mask of
its indices together with a sign bit holding the permutation parity, and
the wedge product, Hodge star and ordering follow the sign conventions
described in :ref:`fdg_covector_basis`.

.. c:autodoc:: basis/basis_set.h basis/basis_bernstein.h basis/basis_lagrange.h basis/basis_legendre.h basis/covector_basis.h
