.. currentmodule:: fdg

.. _fdg_inner_product:

Inner Product Mass Matrix
=========================

At the core of any FEM solver is the inner product of the trial solution with
the weight functions. The inner product is available for "raw" functions as well
as for :math:`k`-forms, with both having the option to either compute it by
evaluating a callable or instead returning a mass matrix that is the result
of factoring the values of degrees of freedom from the definition of these functions.

Mathematical formulation
------------------------

All inner products are the Euclidean inner product on the physical domain
with the unsigned volume element :math:`\mathrm{d}V = |\det J|\,
\mathrm{d}\xi` (see :ref:`fdg_math_background`).

**Scalar functions.** For two functions :math:`f, g` expanded in the
tensor-product spaces :math:`V_{\text{in}}` and :math:`V_{\text{out}}`,
the mass matrix :func:`compute_mass_matrix` is

.. math::

    M_{ji} = \int_{F([-1,1]^N)} \phi^{\text{out}}_j(\vec{x})\,
    \phi^{\text{in}}_i(\vec{x})\, \mathrm{d}V
    = \int_{[-1,1]^N} \phi^{\text{out}}_j(\vec{\xi})\,
    \phi^{\text{in}}_i(\vec{\xi})\, |\det J|\, \mathrm{d}\vec{\xi}.

With an :class:`IntegrationSpace` instead of a :class:`SpaceMap`, the
integral is taken on the reference domain with unit weight. The matrix
maps the primal degrees of freedom of the input space to the dual degrees
of freedom of the output space (rows = output, columns = input).

**Gradients.** :func:`compute_gradient_mass_matrix` assembles the
:math:`x_m`-component of the gradient of an input function against an
output test function:

.. math::

    M_{ji} = \int_{[-1,1]^N}
    \frac{\partial \phi^{\text{in}}_i}{\partial \xi_{i_{\text{in}}}}\,
    \underbrace{\frac{\partial \xi_{i_{\text{in}}}}{\partial x_{i_{\text{out}}}}}_{G_{i_{\text{in}}, i_{\text{out}}}}\,
    \phi^{\text{out}}_j\, |\det J|\, \mathrm{d}\vec{\xi},

i.e. the reference derivative of the input basis is multiplied by the
corresponding entry of the inverse map (the chain rule
:math:`\partial / \partial x_m = \sum_i G_{im}\, \partial / \partial
\xi_i`), and the result is integrated with the volume element. Combined
with the inverse of the plain mass matrix it yields the dual-to-primal
gradient operator, as demonstrated in
:ref:`sphx_glr_auto_examples_plot_gradients.py`.

**:math:`k`-forms.** For two :math:`k`-forms with component expansions in
the spaces :math:`V^{\text{left}}` and :math:`V^{\text{right}}`,
:func:`compute_kform_mass_matrix` assembles

.. math::

    M_{ji} = \int_{[-1,1]^N} \sum_{m = 1}^{\binom{M}{k}}
    \left(T^{(k)} \omega^{\text{left}}\right)_m\,
    \left(T^{(k)} \omega^{\text{right}}\right)_m\,
    |\det J|\, \mathrm{d}\vec{\xi},

where :math:`T^{(k)}` is the basis transform of
:meth:`SpaceMap.basis_transform` that maps the reference components to the
physical ones. The integration weight of the block that couples the left
basis :math:`a` and the right basis :math:`b` is therefore

.. math::

    w_{ab} = |\det J| \sum_{m} T^{(k)}_{a m}\, T^{(k)}_{b m}.

The special cases follow from the special forms of :math:`T^{(k)}`:
for :math:`k = 0` (scalar functions) :math:`w = |\det J|`, and for
:math:`k = N` (top forms, a single component) :math:`T^{(N)} = 1/\det J`
so :math:`w = |\det J|^{-1}`. The component function spaces are the base
spaces lowered by one order on the active axes of each component (see
:ref:`fdg_kform_types`).

Functions Related
-----------------

.. autofunction:: projection_l2_dual

.. autofunction:: compute_mass_matrix


:math:`k`-forms Related
-----------------------

.. autofunction:: projection_kform_l2_dual

.. autofunction:: compute_kform_mass_matrix
