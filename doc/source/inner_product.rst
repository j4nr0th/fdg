.. currentmodule:: fdg

.. _fdg_inner_product:

Inner Product Mass Matrix
=========================

At the core of any FEM solver is the inner product of the trial solution with
the weight functions. The inner product is available for "raw" functions as well
as for :math:`k`-forms, with both having the option to either compute it by
evaluating a callable or instead returning a mass matrix that is the result
of factoring the values of degrees of freedom from the definition of these functions.

Functions Related
-----------------

.. autofunction:: projection_l2_dual

.. autofunction:: compute_mass_matrix


:math:`k`-forms Related
-----------------------

.. autofunction:: projection_kform_l2_dual

.. autofunction:: compute_kform_mass_matrix
