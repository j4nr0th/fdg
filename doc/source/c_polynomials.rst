Polynomials
===========

Evaluation and manipulation of the polynomial families used by the basis
functions: Bernstein, Legendre and Lagrange polynomials.

All polynomials live on the reference interval :math:`[-1, +1]`. The
Legendre polynomials :math:`P_k` satisfy :math:`P_0 = 1`, :math:`P_1 = x`
and Bonnet's recurrence, and are *not* normalized. The Bernstein
polynomials of degree :math:`n` are :math:`B^n_k(t) = \binom{n}{k} t^k
(1 - t)^{n-k}` with :math:`t = (x + 1)/2 \in [0, 1]`. The Lagrange
polynomials are the cardinal functions of a nodal set. The definitions,
recurrences and derivative identities are given in
:ref:`fdg_basis_functions`; the quadrature nodes that double as Lagrange
nodes are described in :ref:`fdg_integration`.

.. c:autodoc:: polynomials/bernstein.h polynomials/legendre.h polynomials/lagrange.h
