.. currentmodule:: fdg

.. _fdg_incidence:

Exterior Derivative
===================

To be able to solve differential equations at all, one must be capable of taking
the derivative of a :math:`k`-form. One way this can be done by explicitly evaluating
the so-called "incidence" matrix, then multiplying the vector with degrees of freedom
with it. Alternatively, one can instead just apply the effect of such a matrix to
the vector/matrix in question. This is typically the preferred way, since for many
types of basis the incidence matrix is sparse and the entries are values that are
very cheap to compute. Still, both options are available for use.

The one-dimensional operators
-----------------------------

On a single dimension the derivative acts on the degrees of freedom of the
basis: for a function :math:`f = \sum_k c_k \phi_k` of polynomial order
:math:`o`, the derivative :math:`f'` is a polynomial of order :math:`o -
1` whose coefficients are obtained by a fixed linear map, the
one-dimensional incidence operator. For the three basis families of
:ref:`fdg_basis_functions`:

**Legendre.** With :math:`f = \sum_k c_k P_k` the derivative is computed
from the identity

.. math::

    P'_n = \sum_{m = 0}^{\lfloor (n - 1) / 2 \rfloor}
    (2(n - 2m) - 1)\, P_{n - 2m - 1},

so the operator is upper-triangular with the entries
:math:`2(n - 2m) - 1`.

**Bernstein.** With :math:`f = \sum_k c_k B^n_k` the derivative has the
Bernstein coefficients :math:`\frac{n}{2}(c_{k+1} - c_k)` of degree
:math:`n - 1`; the factor :math:`1/2` is the chain rule of the
:math:`[-1, +1] \to [0, 1]` mapping.

**Lagrange.** The derivative degrees of freedom are the values of
:math:`f'` at the nodes of the order-:math:`(o - 1)` Lagrange set; the
operator is the derivative of the cardinal functions evaluated at those
nodes.

The scalar incidence functions :func:`incidence_matrix` (matrix form) and
:func:`incidence_operator` (operator form, applied along one axis of a
DoF array) apply exactly these maps.

The exterior derivative on :math:`k`-forms
------------------------------------------

For a :math:`k`-form :math:`\omega = \sum_{i_1 < \dots < i_k}
\omega_{i_1 \dots i_k}\, \mathrm{d}\xi_{i_1} \wedge \dots \wedge
\mathrm{d}\xi_{i_k}` the exterior derivative is (see
:ref:`fdg_math_background`)

.. math::

    \mathrm{d}\omega = \sum_{i_1 < \dots < i_k} \sum_{j}
    \frac{\partial \omega_{i_1 \dots i_k}}{\partial \xi_j}\,
    \mathrm{d}\xi_j \wedge \mathrm{d}\xi_{i_1} \wedge \dots \wedge
    \mathrm{d}\xi_{i_k}.

The component of the output :math:`(k + 1)`-form with index set
:math:`\{i_1, \dots, i_{j-1}, j, i_j, \dots, i_k\}` therefore receives
the derivative of the input component along axis :math:`j` with the sign
:math:`(-1)^{j-1}` of the permutation that moves :math:`\mathrm{d}\xi_j`
into place. The incidence matrix
:math:`\mathbb{E}^{(k+1,k)}` assembled by
:func:`compute_kform_incidence_matrix` is the block matrix of all these
one-dimensional operators: the block
:math:`(\text{output component}, \text{input component})` is non-zero
exactly when the output index set is the input set plus one axis, and
then holds the one-dimensional incidence operator of that axis times
:math:`(-1)^{j-1}`. The matrix maps the flattened degrees of freedom of
the :math:`k`-form (components lexicographically, see
:ref:`fdg_kform_types`) to those of its derivative.

For the k-form operator function, there are four ways the operator can be applied in.
These four ways are the result of being able to choose to whether or not to transpose
the incidence operator, and if it should be applied on the right.

+-----------------+--------------------------------------------------+-----------------------------------------------------------------+
|                 | ``transpose=False``                              | ``transpose=True``                                              |
+-----------------+--------------------------------------------------+-----------------------------------------------------------------+
| ``right=False`` | :math:`\mathbb{E}^{(k+1,k)} x^{(k)} = y^{(k+1)}` | :math:`\left(\mathbb{E}^{(k+1,k)}\right)^T x^{(k+1)} = y^{(k)}` |
+-----------------+--------------------------------------------------+-----------------------------------------------------------------+
| ``right=True``  | :math:`x^{(k+1)} \mathbb{E}^{(k+1,k)} = y^{(k)}` | :math:`x^{(k)} \left(\mathbb{E}^{(k+1,k)}\right)^T = y^{(k+1)}` |
+-----------------+--------------------------------------------------+-----------------------------------------------------------------+


.. autofunction:: incidence_matrix

.. autofunction:: incidence_operator

.. autofunction:: incidence_kform_operator

.. autofunction:: compute_kform_incidence_matrix
