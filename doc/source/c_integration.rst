Integration
===========

Gauss-Legendre and Gauss-Lobatto quadrature rules, their node and weight
computation, and the integration rule registry that caches them.

A rule of *order* :math:`o` has :math:`o + 1` nodes on :math:`[-1, +1]`.
The Gauss-Legendre nodes are the roots of the Legendre polynomial
:math:`P_{o+1}`; the rule integrates polynomials of degree up to
:math:`2o + 1` exactly. The Gauss-Lobatto nodes include the endpoints
:math:`\pm 1` with the interior nodes the roots of :math:`P'_o`; the rule
integrates polynomials of degree up to :math:`2o - 1` exactly. The
:attr:`~fdg.IntegrationSpecs.accuracy` reported for a rule is exactly
this maximum polynomial degree. See :ref:`fdg_integration` for the
formulas, the one- and two-point special cases, and the tensor-product
integration spaces.

.. c:autodoc:: integration/gauss_legendre.h integration/gauss_lobatto.h integration/integration_rules.h
