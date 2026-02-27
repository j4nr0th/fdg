.. currentmodule:: fdg

.. _fdg_interior_product:

Interior Product
================

Besides :ref:`fdg_inner_product` and :ref:`fdg_incidence`, the last key operation to
deal with is the interior product. This involves applying a tangent vector field
(which may be the result of lowering a 1-form) to a :math:`k`-form. In the context
of FEM solvers, the actual form of this operation that is interesting taking an
inner product with a :math:`(k - 1)`-form weight. The mass matrix that is the result
of doing so is computed using :func:`compute_kform_interior_product_matrix`.

.. autofunction:: compute_kform_interior_product_matrix
