Matrices
========

Dense row-major matrix helpers: QR decomposition, multiplication and back
substitution.

The QR decomposition uses Givens rotations. The accumulated rotation
matrix is stored such that :math:`A = Q^T R` holds (the stored factor is
:math:`Q^T`), and the diagonal of :math:`R` is non-negative. This is what
the space mapping code relies on: the determinant of a mapping is computed
as the product of the diagonal of :math:`R`, which is therefore the
unsigned volume factor :math:`|\det J|`, and the pseudo-inverse of a
rectangular Jacobian is obtained from :math:`R_{11}^{-1}` times the top
rows of the stored factor (see :ref:`fdg_space_map`).

.. c:autodoc:: operations/matrices.h
