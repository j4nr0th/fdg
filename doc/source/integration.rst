.. currentmodule:: fdg

.. _fdg_integration:

Integration
===========

Many of the higher level functions need to integrate functions or differential
forms over a domain. As such numerical integration is one of the cornerstones of
this module. For an example of the different integration rules and their relative
performance see :ref:`sphx_glr_auto_examples_plot_integration_rules.py`.

Quadrature rules
----------------

All integration is done with tensor products of one-dimensional quadrature
rules on the reference interval :math:`[-1, +1]` (see
:ref:`fdg_math_background` for the reference-domain conventions). A
one-dimensional rule of *order* :math:`o` has :math:`o + 1` nodes
:math:`\xi_i` and weights :math:`w_i`, and approximates

.. math::

    \int_{-1}^{+1} f(\xi)\,\mathrm{d}\xi \approx
    \sum_{i = 0}^{o} w_i\, f(\xi_i) .

On an :math:`N`-dimensional tensor-product grid the integral becomes

.. math::

    \int_{[-1, +1]^N} f(\vec{\xi})\,\mathrm{d}\vec{\xi} \approx
    \sum_{i_1} \cdots \sum_{i_N}
    \left(\prod_{j = 1}^{N} w^{(j)}_{i_j}\right)
    f\!\left(\xi^{(1)}_{i_1}, \dots, \xi^{(N)}_{i_N}\right),

where the rule used in dimension :math:`j` is selected by the
:math:`j`-th :class:`IntegrationSpecs`. The weight of a grid point is the
product of the one-dimensional weights, which is what
:meth:`IntegrationSpace.weights` returns.

Two rule families are provided (:class:`IntegrationMethod`):

**Gauss-Legendre** (``"gauss"``). The :math:`o + 1` nodes are the roots of
the Legendre polynomial :math:`P_{o+1}` on :math:`[-1, +1]`, computed by
Newton iteration, and the weights are

.. math::

    w_i = \frac{2}{(o + 1)^2\, P_{o}\!\left(\xi_i\right)^2}
    \left(1 - \xi_i^2\right) .

The rule is exact for polynomials of degree up to
:math:`2(o + 1) - 1 = 2o + 1`, which is the value reported by
:attr:`IntegrationSpecs.accuracy`. The one- and two-point rules are
hard-coded: :math:`(\xi, w) = (0, 2)` and
:math:`\left(\pm 1/\sqrt{3}, 1\right)`.

**Gauss-Lobatto** (``"gauss-lobatto"``). The nodes include the endpoints
:math:`\pm 1`; the remaining :math:`o - 1` interior nodes are the roots of
:math:`P'_{o}` (the derivative of the Legendre polynomial of degree
:math:`o`). The endpoint weights are :math:`2 / (o(o + 1))` and the
interior weights are

.. math::

    w_i = \frac{2}{o(o + 1)\, P_{o}\!\left(\xi_i\right)^2} .

The rule is exact for polynomials of degree up to
:math:`2(o + 1) - 3 = 2o - 1`. The degenerate one-node rule
:math:`(\xi, w) = (0, 2)` coincides with the one-point Gauss rule and also
integrates degree 1 exactly, so its accuracy is 1.

Because an :math:`o`-th order rule has :math:`o + 1` nodes, the *accuracy*
:math:`a` needed to integrate a given polynomial degree exactly
determines the smallest sufficient order: :math:`o = \lceil (a - 1)/2
\rceil` for Gauss-Legendre and :math:`o = \lceil (a + 1)/2 \rceil` for
Gauss-Lobatto. This is the mapping used by the integration rule registry
when a rule for a requested accuracy is created.

Integration Specifications
--------------------------

The way the method of integration is specified in :math:`N`-dimensional space
is by "outer-product" grid, where for each dimension the method (as given by :class:`IntegrationMethod`) and order of
integration is specified using :class:`IntegrationSpecs`.

.. autoclass:: IntegrationMethod
    :no-inherited-members:

.. autoclass:: IntegrationSpecs

Since values of integration nodes and weights are ofter reused, they are not stored inside :class:`IntegrationSpecs` objects.
Instead, they are stored in :class:`IntegrationRegistry` objects. By default, the ``fdg`` module provides one already,
however any number of new registries can be created.

.. autoclass:: IntegrationRegistry


.. autodata:: DEFAULT_INTEGRATION_REGISTRY

    Default :class:`IntegrationRegistry` used by ``fdg`` unless another is provided.


Integration Space
-----------------

To define how an integration should be done on a :math:`N`-dimensional domain the individual specifications
for each dimension (given as :class:`IntegrationSpecs`) are bundled together into :class:`IntegrationSpace`.

.. autoclass:: IntegrationSpace
