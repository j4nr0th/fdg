.. currentmodule:: fdg

.. _fdg_basis_functions:

Basis Functions
===============

Besides :ref:`fdg_integration`, another basic building block of dealing with
finite differential forms is basis functions. These determine values of the
degrees of freedom actually mean. For a demonstration of what types of basis
are available, see :ref:`sphx_glr_auto_examples_plot_basis_sets.py`.

Basis Specifications
--------------------

To specify the basis that are to be used, the :class:`BasisSpecs` type is used.
This specifies the order of the basis and their type (using :class:`BasisType`) for a single dimension.

.. autoclass:: BasisSpecs

.. autoclass:: BasisType
    :no-inherited-members:

While values of all basis functions can be computed with :class:`BasisSpecs`, these are not cached.
However, values of these bases at integration points are cached using :class:`BasisRegistry`. Similarly
to what was the case with :class:`IntegrationRegistry`, there is a default one provided that is
used if no other is specified.

.. autoclass:: BasisRegistry

.. autodata:: DEFAULT_BASIS_REGISTRY

    The :class:`BasisRegistry` registry used when another is not provided.

Function Spaces
---------------

Just as was the case with :class:`IntegrationSpecs`, when dealing with `N`-dimensional spaces
:class:`BasisSpecs` objects are bundled together into a :class:`FunctionSpace` objects. These
define the function space based on outer product of basis.

Given basis sets :math:`b^k = \left\{ b^k_1, \dots, b^k_{n_1} \right\}` for :math:`k = 1, \dots, N`,
the value of the different basis functions at point with position :math:`\vec{r} = \begin{bmatrix} x_1 \\ \vdots \\ x_N \end{bmatrix}` is
given by Equation :eq:`eq-outer-product-basis`. Based on this it is quite clear that the total number of basis functions in this case is
:math:`\prod\limits_{i=1}^N n_i`.

.. math::
    :label: eq-outer-product-basis

    \psi_{i_1, \dots, i_N} (\vec{r}) = \prod\limits_{j = 1}^N b^{j}_{i_1}(x_j)

.. autoclass:: FunctionSpace
