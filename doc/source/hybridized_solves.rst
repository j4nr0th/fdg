.. currentmodule:: fdg

.. _fdg_hybridized_solves:

Hybridized Solves
=================

The trace constraints of a mesh turn one small problem per element into a
single global saddle-point system. Writing :math:`A` for the block diagonal
operator with one dense block per element and :math:`N` for the packed
constraint rows, the unknowns are the element degrees of freedom together with
one Lagrange multiplier per constraint row:

.. math::

    \begin{pmatrix} A & N \\ N^T & 0 \end{pmatrix}
    \begin{pmatrix} u \\ \lambda \end{pmatrix}
    = \begin{pmatrix} b \\ c \end{pmatrix}.

:func:`solve_hybridized` assembles that system as a block system and hands it
to the `hybsol <https://github.com/j4nr0th/hybridized-solver>`_ block solver.
No Schur complement is formed, and the multipliers come out of the same solve
rather than from a second, condensed system.

The block structure
-------------------

One block per element holds that element's operator. A constraint row can also
be *absorbed* into an element block: the row's equation takes a row of the
block and its multiplier a column, which borders the block into

.. math::

    \begin{pmatrix} A_e & C^T \\ C & 0 \end{pmatrix}.

That border is what makes a **singular** element operator usable. The 0-form
stiffness :math:`D^T M_1 D` has the constants in its nullspace, so on its own
it cannot lead a factorization; bordered by even one constraint row it becomes
nonsingular. Rows that no element can absorb stay in blocks of their own,
grouped per interface object, whose diagonals are structurally zero and are
filled once the element blocks have been eliminated.

Assembly is per element and per row, driven by the packed rows the mesh hands
out. See :ref:`fdg_boundary_constraints` for that format and
:ref:`fdg_mesh` for the topology it describes.

Ordering inside a block
-----------------------

hybsol factorizes without pivoting, so every leading principal minor of every
diagonal block has to be nonzero. Rows and columns inside an element block
therefore share one schedule: it walks the element's degrees of freedom and,
right after the last degree of freedom a row touches with a nonzero
coefficient, places that row's equation and its multiplier. A pure operator
prefix is then a proper principal submatrix of :math:`A_e`, and every border
sees a nonzero entry of its own row, which keeps the bordered minors away from
zero.

What the caller owes
--------------------

Every constraint row has to reach a block, and the rows absorbed into one
element must be linearly independent on that element -- otherwise that element
block is singular and hybsol raises
:exc:`hybsol.SingularSystemError`. A singular element operator needs at least
one absorbed row, and a 0-form problem needs at least one prescribed row per
connected component to gauge the solution; without one the system really is
singular and the solve returns garbage, which the residual guard in
:func:`solve_hybridized` rejects.

Element operators
-----------------

An element block is produced by a per-element callback, mirroring the loop the
C core runs: one call per element, receiving that element's k-form
specifications and space map. The constrained field has to come first in the
block, because the packed rows address that field. Two builders ship with the
module, :func:`laplace_stiffness` for a 0-form Laplacian and
:func:`mixed_block` for the mixed Poisson block
:math:`[[M_q, D^T], [D, 0]]`; any callable of the same shape is accepted.

The right-hand side is assembled by the caller. Boundary loads live on boundary
objects rather than on elements, so they have no place in a per-element
callback, and keeping them separate lets one set of element blocks serve
several right-hand sides.

The gallery examples
:ref:`sphx_glr_auto_examples_plot_multi_element_laplace_continuity.py` and
:ref:`sphx_glr_auto_examples_plot_multi_element_poisson_periodic.py` solve the
0-form and mixed problems through this entry point.

.. autofunction:: solve_hybridized

.. autofunction:: laplace_stiffness

.. autofunction:: mixed_block

.. autoclass:: HybridizedSolution

.. autoclass:: ElementBlockBuilder
