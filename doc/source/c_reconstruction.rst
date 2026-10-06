Reconstruction
==============

Reconstruction of functions from degrees of freedom using tensor-product
basis sets evaluated at integration points.

The reconstructed value of a function at a point is the tensor-product
sum :math:`f(\vec{\xi}) = \sum_{i_1, \dots, i_N} c_{i_1 \dots i_N}
\prod_k b^k_{i_k}(\xi_k)` over the degrees of freedom; the gradient is
obtained by replacing the one-dimensional basis values with their
derivatives along the requested axes (see :ref:`fdg_degrees_of_freedom`
for the details and the derivative rules).

.. c:autodoc:: reconstruction/reconstruction.h
