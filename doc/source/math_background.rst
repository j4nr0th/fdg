.. _fdg_math_background:

Mathematical Background
=======================

This page collects the mathematical conventions used throughout ``fdg``.
Every other page in this documentation refers back to it. The conventions
match the implementation exactly: the formulas below are the ones the C core
and the Python bindings evaluate, including the sign conventions.

Reference Domain
----------------

All one-dimensional bases and quadrature rules live on the reference
interval

.. math::

    \xi \in [-1, +1] .

An :math:`N`-dimensional reference domain is the tensor product
:math:`[-1, +1]^N`; a point in it is written

.. math::

    \vec{\xi} = \begin{bmatrix} \xi_1 \\ \vdots \\ \xi_N \end{bmatrix},
    \qquad \xi_i \in [-1, +1] .

A deformed element is the image of this reference domain under a
:ref:`space mapping <fdg_space_map>`

.. math::

    F : [-1, +1]^N \to \mathbb{R}^M, \qquad \vec{x} = F(\vec{\xi}),

with :math:`M \ge N`. The reference coordinates are always denoted by
:math:`\vec{\xi}` and the physical coordinates by :math:`\vec{x}`.

Differential Forms
------------------

A *differential :math:`k`-form* on an :math:`N`-dimensional domain is a
smooth antisymmetric multilinear map that assigns to :math:`k` tangent
vectors a scalar. In a coordinate basis it is written as a sum over wedge
monomials of coordinate differentials,

.. math::

    \omega = \sum_{i_1 < \dots < i_k}
    \omega_{i_1 \dots i_k}
    \,\mathrm{d} x_{i_1} \wedge \dots \wedge \mathrm{d} x_{i_k},

where the sum runs over the :math:`\binom{N}{k}` strictly increasing index
sets :math:`\{i_1, \dots, i_k\} \subset \{1, \dots, N\}`. The scalar
functions :math:`\omega_{i_1 \dots i_k}` are the *components* of the form.
A :math:`0`-form is just a scalar function.

``fdg`` stores a :math:`k`-form as a collection of components. The
components are indexed *lexicographically* by their ascending axis sets:
:math:`\{1, 2\}` before :math:`\{1, 3\}` before :math:`\{2, 3\}`, and so
on. The number of components is the binomial coefficient
:math:`\binom{N}{k}`, which is what :attr:`KFormSpecs.component_count`
returns.

Each component is expanded in its own tensor-product function space. If the
base (0-form) space has polynomial order :math:`p_i` in dimension :math:`i`,
the space of the component with active axes :math:`\{i_1, \dots, i_k\}` has
order :math:`p_i` on the inactive axes and order :math:`p_i - 1` on the
active (wedged) axes. The number of degrees of freedom of one component is
therefore

.. math::

    \prod_{i = 1}^{N} \begin{cases}
        p_i,     & i \in \{i_1, \dots, i_k\} \\
        p_i + 1, & i \notin \{i_1, \dots, i_k\}
    \end{cases}

One degree of freedom less per active axis: the antisymmetry of the wedge
product makes the component vanish where two active coordinates coincide,
so no nodal values are needed on those planes.

The Wedge Product
-----------------

The wedge product of two forms is associative, bilinear and
antisymmetric,

.. math::

    \mathrm{d} x_i \wedge \mathrm{d} x_j = - \mathrm{d} x_j \wedge
    \mathrm{d} x_i, \qquad
    \mathrm{d} x_i \wedge \mathrm{d} x_i = 0 .

A wedge monomial :math:`\mathrm{d} x_{i_1} \wedge \dots \wedge
\mathrm{d} x_{i_k}` therefore carries a *sign*: the parity of the
permutation that orders its index set. ``fdg`` encodes a wedge monomial as
a :class:`CovectorBasis`: a bit mask of the component indices plus a sign
bit that stores this parity. The wedge product merges the two bit masks
and flips the sign once for every index of the second factor that must be
moved past a larger index of the first factor; if the masks overlap the
product is zero. This is exactly the permutation parity of the merged index
set. In the Python API the wedge product of two :class:`CovectorBasis`
objects is written with the XOR operator ``^``.

The Hodge Star
--------------

The Hodge star maps a :math:`k`-form to an :math:`(N - k)`-form. ``fdg``
uses the Euclidean Hodge star with respect to the ordered basis
:math:`(\mathrm{d} x_1, \dots, \mathrm{d} x_N)` and the volume form
:math:`\mathrm{d} x_1 \wedge \dots \wedge \mathrm{d} x_N`:

.. math::

    \star\left(\mathrm{d} x_{i_1} \wedge \dots \wedge \mathrm{d} x_{i_k}\right)
    = \operatorname{sgn}(\sigma)\,
    \mathrm{d} x_{j_1} \wedge \dots \wedge \mathrm{d} x_{j_{N-k}},

where :math:`\{j_1, \dots, j_{N-k}\}` is the complement of
:math:`\{i_1, \dots, i_k\}` in ascending order and
:math:`\operatorname{sgn}(\sigma)` is the parity of the permutation that
sorts :math:`(i_1, \dots, i_k, j_1, \dots, j_{N-k})` into
:math:`(1, \dots, N)`. No metric factors or extra powers of
:math:`(-1)^{k(N-k)}` are included. In the Python API the Hodge star of a
:class:`CovectorBasis` is written with the inversion operator ``~``. The
library uses the Hodge star primarily to classify and order components,
not to build the inner product.

The Exterior Derivative
-----------------------

The exterior derivative of a :math:`k`-form is the :math:`(k + 1)`-form

.. math::

    \mathrm{d}\omega = \sum_{i_1 < \dots < i_k} \sum_{j = 1}^{N}
    \frac{\partial \omega_{i_1 \dots i_k}}{\partial x_j}\,
    \mathrm{d} x_j \wedge \mathrm{d} x_{i_1} \wedge \dots \wedge
    \mathrm{d} x_{i_k} .

Moving :math:`\mathrm{d} x_j` into its sorted position contributes the sign
:math:`(-1)^m`, where :math:`m` is the number of indices
:math:`i_l < j`. In ``fdg`` the exterior derivative is applied to the
degrees of freedom: for each component of the input :math:`k`-form and each
axis :math:`j` not in its index set, the one-dimensional derivative operator
of the basis along that axis is applied and the result is placed into the
component of the output :math:`(k + 1)`-form whose index set contains the
input set plus :math:`j`, with the sign :math:`(-1)^m`. The resulting
matrix is the *incidence matrix* :math:`\mathbb{E}^{(k+1,k)}` of
:ref:`fdg_incidence`.

The Interior Product
--------------------

The interior product of a :math:`k`-form with a vector field
:math:`X = \sum_j X^j \, \partial / \partial x_j` is the :math:`(k - 1)`-form

.. math::

    (\iota_X \omega)_{i_1 \dots i_{k-1}}
    = \sum_{j = 1}^{k} (-1)^{j - 1}
    X^{i_j}\, \omega_{i_1 \dots i_{j-1} i_j i_j \dots i_{k-1}},

i.e. each index of the form is contracted with the vector field in turn,
with the sign of the permutation that removes it. It is the algebraic dual
of the wedge product:

.. math::

    \iota_X(\alpha \wedge \beta) = (\iota_X \alpha) \wedge \beta
    + (-1)^{\deg \alpha}\, \alpha \wedge (\iota_X \beta) .

The library computes the mass matrix of the inner product of a
:math:`(k - 1)`-form test with :math:`\iota_X \omega`
(:func:`compute_kform_interior_product_matrix`), see
:ref:`fdg_interior_product`.

Pullback and the Basis Transform
--------------------------------

Let :math:`F : \vec{\xi} \mapsto \vec{x}` be a :ref:`space mapping
<fdg_space_map>` with Jacobian

.. math::

    J_{mi} = \frac{\partial x_m}{\partial \xi_i},
    \qquad J = \begin{bmatrix}
        \dfrac{\partial x_1}{\partial \xi_1} & \cdots &
        \dfrac{\partial x_1}{\partial \xi_N} \\[2mm]
        \vdots & \ddots & \vdots \\[2mm]
        \dfrac{\partial x_M}{\partial \xi_1} & \cdots &
        \dfrac{\partial x_M}{\partial \xi_N}
    \end{bmatrix} \in \mathbb{R}^{M \times N}.

A :math:`1`-form on the physical domain pulls back component-wise with the
transpose of the inverse Jacobian,

.. math::

    F^*(\mathrm{d} x_m) = \sum_{i = 1}^{N}
    \frac{\partial \xi_i}{\partial x_m}\, \mathrm{d} \xi_i,

and a general :math:`k`-form pulls back with the :math:`k`-th exterior
power of this map, i.e. the wedge products of the rows are expanded and
collected. ``fdg`` evaluates this expansion explicitly: the *basis
transform* :math:`\mathbf{T}^{(k)}` returned by
:meth:`SpaceMap.basis_transform` is a three-index array
:math:`T^{(k)}_{ab}(\vec{\xi})` that gives the coefficient with which the
reference component :math:`a` (an ascending :math:`k`-subset of the
reference axes) contributes to the physical component :math:`b` (an
ascending :math:`k`-subset of the physical axes) at each integration point,

.. math::

    T^{(k)}_{ab} = \sum_{\sigma \in S_k} \operatorname{sgn}(\sigma)
    \prod_{l = 1}^{k} \frac{\partial \xi_{i_l}}{\partial x_{j_{\sigma(l)}}},

where :math:`a = \{i_1, \dots, i_k\}`, :math:`b = \{j_1, \dots, j_k\}` and
:math:`\partial \xi_i / \partial x_j` are the entries of the (pseudo-)
inverse of :math:`J`. The special cases are
:math:`T^{(1)}_{ab} = \partial \xi_a / \partial x_b` and, for a square map,
:math:`T^{(N)} = 1/\det J`. See :ref:`fdg_kform_transformations` for how
the components are transformed.

Inner Products
--------------

All inner products in ``fdg`` are taken on the physical domain with the
Euclidean metric: for two :math:`k`-forms with physical components
:math:`\omega_m` and :math:`\eta_m`,

.. math::

    (\omega, \eta) = \int_{F([-1,1]^N)} \sum_{m} \omega_m\, \eta_m\,
    \mathrm{d}V, \qquad \mathrm{d}V = |\det J|\, \mathrm{d}\xi_1 \cdots
    \mathrm{d}\xi_N .

Because the Jacobian determinant is stored as the product of the diagonal
of the QR factor :math:`R`, which is non-negative, ``fdg`` effectively
always integrates with the *unsigned* volume element :math:`|\det J|`. For
the :math:`0`-form mass matrix this gives the weight
:math:`|\det J|`; for the :math:`N`-form mass matrix the physical
components carry a factor :math:`1/\det J`, so the combined weight is
:math:`|\det J|^{-1}`; for intermediate orders the basis transform
:math:`\mathbf{T}^{(k)}` supplies the metric contraction. The details are
in :ref:`fdg_inner_product`. With the Euclidean metric, lowering the index
of a :math:`1`-form (or of a :math:`k`-form) is trivial: the physical
components are used directly as the components of the corresponding
tangent multivector, which is why the contravariant transform of
:ref:`fdg_kform_transformations` is the plain Jacobian.

Degrees of Freedom
------------------

A function in a tensor-product space is represented by its degrees of
freedom (DoFs), one per basis function. For a basis
:math:`\left\{b^k_1, \dots, b^k_{n_k}\right\}` in each dimension
:math:`k`, the DoF :math:`c_{i_1 \dots i_N}` multiplies the basis function

.. math::

    \psi_{i_1 \dots i_N}(\vec{\xi}) = \prod_{k = 1}^{N}
    b^k_{i_k}(\xi_k),

so the function is

.. math::

    f(\vec{\xi}) = \sum_{i_1 = 1}^{n_1} \cdots \sum_{i_N = 1}^{n_N}
    c_{i_1 \dots i_N}\, \psi_{i_1 \dots i_N}(\vec{\xi}) .

The one-dimensional bases are described in :ref:`fdg_basis_functions`.
Derivatives act on the DoFs through the one-dimensional *incidence
operators* of :ref:`fdg_incidence`.
