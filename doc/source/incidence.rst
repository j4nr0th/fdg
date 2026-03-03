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
