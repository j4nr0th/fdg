.. currentmodule:: fdg

.. _fdg_direct_continuity:

Direct Continuity
=================

Trace constraints can be built in two ways. The
:ref:`hybridized formulation <fdg_hybridized_solves>` *eliminates* the degrees
of freedom on a shared object and carries them as Lagrange multipliers; the
*direct* formulation introduces them explicitly and assembles the element
matrices straight onto them. Both enforce continuity exactly and differ in the
size and shape of the system.

The direct formulation lives in the C core at :file:`constraints/direct.h`
(documented at :doc:`c_constraints`) and is reached from Python through
:meth:`Mesh.compute_kform_direct_dof_map`, which returns a
:class:`DirectDofMap`.

.. _fdg_direct_two_formulations:

The two formulations
--------------------

The hybridized formulation keeps every element's local degrees of freedom and
adds one multiplier per constraint row, one row per shared-object degree of
freedom. The two traces stay separate variables tied together by the multiplier
rows, which makes the global operator a saddle-point system and is why
:func:`solve_hybridized` assembles it as a block system.

The direct formulation instead *identifies* the traces: every element-local
degree of freedom a shared object carries is replaced by that object's own
degree of freedom, so neighbouring elements write into one variable and
continuity is structural rather than enforced. There is no multiplier block.

.. _fdg_direct_decomposition:

The decomposition of the global space
-------------------------------------

In :math:`N` dimensions the global unknown vector splits into

- the degrees of freedom whose support reaches no shared object. They are
  private to their element and keep their local numbering.
- one block per shared object of every dimension :math:`N - 1` down to
  :math:`0`, holding exactly the degrees of freedom interior to that object:
  on a face those interior to the face, on an edge those interior to the edge,
  on a node the single node degree of freedom.

An object of dimension :math:`d` therefore owns a tensor-product space on its
own :math:`d` axes, taken per k-form component. A component whose k-form order
exceeds :math:`d` has no trace on that object, so it owns nothing for it and
the block is empty.

:meth:`Mesh.compute_kform_direct_dof_map` reports the element-private ranges in
``element_interior_offsets`` and the size of the whole vector in
``global_dof_count``. The object blocks precede the element-private ones.

Why this has the fewest unknowns
--------------------------------

The hybridized formulation counts

.. math::

   \underbrace{\sum_e \text{dofs}(e)}_{\text{all element DoFs}}
   + \underbrace{\sum_{\text{shared objects}} \text{constraint rows}}_{\text{multipliers}},

because the shared degrees of freedom are still present inside each element and
are duplicated once more as multipliers. The direct formulation counts

.. math::

   \underbrace{\sum_e \text{dofs interior to } e}_{\text{element-private}}
   + \underbrace{\sum_{\text{shared objects}} \text{block size of the object}}_{\text{shared DoFs}},

and that count is an upper bound: a degree of freedom that projects onto an
empty common space leaves the system entirely. No unknown is introduced that is
not a degree of freedom of the continuous field, which makes this the smallest
space that still enforces continuity exactly.

Which object owns a degree of freedom
-------------------------------------

Ownership is read off the k-form space itself, not configured. Component
:math:`I` of a :math:`k`-form is :math:`d\xi_I` tensored with the element's
function space: a covector axis :math:`a` of :math:`I` reads the order-one
basis (:math:`\texttt{order}` functions), every other axis the full basis
(:math:`\texttt{order} + 1` functions). Only the endpoint functions of an axis
live on an object, so component :math:`I` reaches the face perpendicular to axis
:math:`a` exactly when :math:`a` is not a covector axis of :math:`I` and its
digit is :math:`0` or :math:`\texttt{order}`.

Those faces name the smallest object carrying the degree of freedom, and the
degree of freedom is numbered on it. The transfer between the element's frame
and the object's frame also carries the component's orientation sign
(:math:`\pm 1`).

The common space of an object
-----------------------------

An object's block is the largest space every incident element can represent.
Per component, an axis carrying one of the component's covector axes keeps its
whole order-one space; an axis carrying none keeps only the functions *between*
the two endpoints, since the endpoint functions live on the faces perpendicular
to that axis. This is the windowed common test space the
:ref:`boundary constraints <fdg_boundary_constraints>` use, read in the
opposite direction.

The per-axis order is the minimum over the incident elements, taken over their
*free* axes -- the record positioning an object inside an element opens with the
axes it is pinned to -- so neither the order nor the choice of which element is
looked at first changes the result, except that a tie on the order keeps the
first incident element's basis family. A high-order neighbour is therefore never
over-constrained by its low-order neighbours.

The windowed transfer from an element's frame onto the object's is the
:math:`L_2` projection of the element's trace onto the common space: per axis
the mixed pairing divided by the common space's Gram matrix, at a
Gauss-Legendre rule of the larger of the two degrees. Where the element's window
already equals the common window the transfer short-circuits to the identity and
needs no quadrature, so the degree of freedom gets exactly one entry, of
coefficient :math:`+1` or :math:`-1`. Only a genuinely mismatched pair produces
the weighted combination of several.

A common window can also be *empty* -- a linear axis carries no function
between its endpoints -- and then the element's trace projects onto an empty
space, so its degree of freedom reaches no global degree of freedom and is
dropped from the system. The transfer still accounts for it, so the row
compression stays a valid partition and the degree of freedom owns an empty
range of ``entry_offsets``.

Preconditions
-------------

The numbering locates a degree of freedom by the *node* its digit sits on, so
it needs a nodal (Lagrange) basis family; orthogonal and Bernstein bases do not
localize and are rejected. In the C core this is a precondition of
:c:func:`direct_continuity_prepare`, which returns
:c:enumerator:`FDG_ERROR_NOT_IN_DOMAIN` for a non-nodal family or a
non-positive order on any axis of a shared object; from Python,
:meth:`Mesh.compute_kform_direct_dof_map` raises :exc:`ValueError` naming the
offending element, axis and basis type. See :ref:`fdg_basis_functions` for the
families the library provides.

.. note::

   Lagrange is the only family supported so far. Lifting that restriction --
   numbering a non-localizing family by its own decomposition rather than by
   node support -- is deferred to a later session; the check itself lives in
   :c:func:`direct_continuity_prepare` and in
   :meth:`Mesh.compute_kform_direct_dof_map`.

The remaining preconditions are the shape of ``element_specs``: exactly
``mesh.element_count`` entries, each describing the mesh dimension, all with the
same k-form degree, and each a :class:`KFormSpecs` instance. The order may
differ per element and per axis, since the common space takes the minimum over a
shared object's incident elements. A wrong length, dimension or degree raises
:exc:`ValueError`, a non-:class:`KFormSpecs` entry :exc:`TypeError`, and a
failure reported by the C core :exc:`RuntimeError`.

The :math:`L_2` projection needs a quadrature, so the C entry points borrow a
basis registry and an integration registry for the duration of the call. They
keep no lasting reference: a prepared plan holds no registry pointer, and every
caller-provided array is sized up front by
:c:func:`direct_continuity_layout`.

Choosing the direct path
------------------------

The direct path is a *numbering and assembly* choice, not a solve. Ask for the
map instead of the packed constraint rows, then scatter each element matrix
through it:

.. code-block:: python

    import numpy as np

    from fdg import BasisSpecs, BasisType, FunctionSpace, KFormSpecs, Mesh

    mesh = Mesh.from_corners(corners)                  # see Mesh.from_corners
    space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    specs = [KFormSpecs(0, space) for _ in range(mesh.element_count)]

    dof_map = mesh.compute_kform_direct_dof_map(specs)

    def block(dof):
        """Global indices and coefficients one element-local degree of freedom reaches."""
        first = int(dof_map.entry_offsets[dof])
        last = int(dof_map.entry_offsets[dof + 1])
        return dof_map.entry_index[first:last], dof_map.entry_value[first:last]

    size = int(dof_map.global_dof_count)
    operator = np.zeros((size, size))
    for element in range(mesh.element_count):
        matrix = laplace_stiffness(element, [specs[element]], maps[element])
        blocks = [block(dof) for dof in range(int(dof_map.element_offsets[element]),
                                             int(dof_map.element_offsets[element + 1]))]
        for local, (row_index, row_value) in enumerate(blocks):
            for column, (column_index, column_value) in enumerate(blocks):
                weights = row_value * matrix[local, column] * column_value
                np.add.at(operator,
                          (np.repeat(row_index, len(column_index)),
                           np.tile(column_index, len(row_index))),
                          weights)

Each element-local degree of freedom owns the contiguous range
``entry_offsets[dof: dof + 2]``; ``entry_index`` names the global degree of
freedom of each entry and ``entry_value`` its coefficient. Adding
:math:`v_i\, M_{ij}\, v_j` for every pair of entries reproduces the windowed
projection of an element's trace onto the shared object's common space, and is
exactly what :c:func:`direct_continuity_scatter` does in the C core.
:class:`DirectDofMap` reports the size of every array, so the loop needs no
counts of its own.

.. note::

   The direct formulation makes the *global* operator singular wherever the
   element operator is: a 0-form Laplacian still has constants in its
   nullspace. Prescribing boundary data or fixing one degree of freedom per
   connected component remains the caller's job; the
   :ref:`hybridized formulation <fdg_hybridized_solves>` gauges the solution
   with absorbed constraint rows instead.

The gallery example
:ref:`sphx_glr_auto_examples_plot_multi_element_laplace_direct_continuity.py`
solves the same 0-form Laplace problem on the same mesh through both
formulations and prints, per refinement level, the two unknown counts, their
ratio, and both errors against the analytic solution.

.. autoclass:: DirectDofMap
