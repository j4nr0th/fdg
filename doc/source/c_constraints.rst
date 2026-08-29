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

The single-side physical-space assembly also provides a boundary *load*:
:c:func:`constraint_physical_side_load` integrates the components of a
:math:`k`-form datum (element-frame components sampled at the canonical face
points) against the trace of the element :math:`(k-1)`-form basis on a
codimension-1 face, for every datum degree :math:`k = 1, \dots, n`.  At
:math:`k = n` this is the weak Dirichlet boundary condition of the mixed
formulation.  Like the trace assembly it needs no surface weights or
pullback, because the load is a metric-free chain integral
(see :ref:`fdg_boundary_constraints`).

.. c:autodoc:: constraints/constraints.h
