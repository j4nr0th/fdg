Constraints
===========

Assembly of trace constraints for differential forms on element interfaces,
in both reference space and physical space, including the packed
row representation of the resulting constraint matrices.

A trace constraint equates, for each test degree of freedom on a shared
face, the traces of the element :math:`k`-forms on both sides of the
interface. In reference space the coefficient of a side is the basis inner
product over the face quadrature, signed by the element orientation and
with the two sides entering with opposite signs. In physical space the
coefficients additionally carry the pullback of the physical :math:`k`-form
(through the face map's tangential pullback, see
:ref:`fdg_space_map`) and the unsigned surface measure
:math:`|\det J_{\text{face}}|`. The rows are returned in a packed
representation (row offsets, component, local DoF, coefficient) so that
callers can assemble their own global sparse matrices. The mathematical
construction is described in :ref:`fdg_boundary_constraints`.

.. c:autodoc:: constraints/constraints.h
