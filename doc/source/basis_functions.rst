.. currentmodule:: fdg

.. _fdg_basis_functions:

Basis Functions
===============

Besides :ref:`fdg_integration`, another basic building block of dealing with
finite differential forms is basis functions. These determine values of the
degrees of freedom actually mean. For a demonstration of what types of basis
are available, see :ref:`sphx_glr_auto_examples_plot_basis_sets.py`.

One-dimensional bases
---------------------

All bases are sets of :math:`n = o + 1` polynomial functions of degree
:math:`o` on the reference interval :math:`[-1, +1]`, one per degree of
freedom (:class:`BasisType`):

**Legendre** (``"legendre"``). The basis functions are the Legendre
polynomials :math:`\phi_k = P_k`, :math:`k = 0, \dots, o`, defined by
:math:`P_0 = 1`, :math:`P_1 = x` and Bonnet's recurrence

.. math::

    (k + 1)\, P_{k + 1}(x) = (2k + 1)\, x\, P_k(x) - k\, P_{k - 1}(x).

They are *not* normalized. The degrees of freedom are the coefficients of
the expansion :math:`f = \sum_k c_k P_k`. The derivatives satisfy the
recurrence :math:`P'_k = k\, P_{k-1} + x\, P'_{k-1}` and the identity

.. math::

    P'_n = \sum_{m = 0}^{\lfloor (n - 1) / 2 \rfloor}
    (2(n - 2m) - 1)\, P_{n - 2m - 1},

which is how the derivative degrees of freedom are computed.

**Bernstein** (``"bernstein"``). The Bernstein polynomials of degree
:math:`o`,

.. math::

    B^n_k(t) = \binom{n}{k}\, t^k (1 - t)^{n - k},
    \qquad t = \frac{x + 1}{2} \in [0, 1], \qquad k = 0, \dots, n,

with :math:`n = o`. Their derivatives satisfy

.. math::

    \frac{\mathrm{d}}{\mathrm{d}x} B^n_k = \frac{n}{2}
    \left(B^{n-1}_{k-1} - B^{n-1}_k\right),

so the derivative of :math:`f = \sum_k c_k B^n_k` has the Bernstein
coefficients :math:`\frac{n}{2}(c_{k+1} - c_k)` of degree :math:`n - 1`.

**Lagrange** (``"lagrange-uniform"``, ``"lagrange-gauss"``,
``"lagrange-gauss-lobatto"``, ``"lagrange-chebyshev-gauss"``). The
cardinal (nodal) functions with respect to a set of :math:`n = o + 1`
nodes :math:`x_k`,

.. math::

    \phi_k(x) = \prod_{\substack{j = 0 \\ j \ne k}}^{o}
    \frac{x - x_j}{x_k - x_j}, \qquad \phi_k(x_j) = \delta_{kj}.

The degrees of freedom are the nodal values :math:`c_k = f(x_k)`. The node
sets are, for :math:`k = 0, \dots, o`,

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Type
     - Nodes
   * - ``uniform``
     - :math:`x_k = \frac{2k}{o} - 1`
   * - ``gauss``
     - roots of :math:`P_{o+1}` (Gauss-Legendre nodes)
   * - ``gauss-lobatto``
     - Gauss-Lobatto nodes (includes :math:`\pm 1`)
   * - ``chebyshev-gauss``
     - :math:`x_k = -\cos\!\left(\frac{\pi (2k + 1)}{2(o + 1)}\right)`

The derivative degrees of freedom are the values of the derivative at the
nodes of the degree-:math:`o - 1` Lagrange set, obtained from the nodal
values by differentiating the cardinal functions.

The values (and first derivatives) of a basis set are tabulated at the
nodes of an integration rule and cached by the :class:`BasisRegistry`;
evaluating a basis on demand is also possible through
:meth:`BasisSpecs.values` and :meth:`BasisSpecs.derivatives`.

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
