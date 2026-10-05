.. currentmodule:: fdg

.. _fdg_direct_continuity:

Direct Continuity
=================

The :ref:`hybridized formulation <fdg_hybridized_solves>` eliminates the degrees of freedom on a shared object
into Lagrange multipliers; the *direct* formulation introduces them explicitly and assembles element matrices
straight onto them. Both enforce continuity exactly, with different system size and shape. The C core lives at
:file:`constraints/direct.h` (see :doc:`c_constraints`); Python reaches it through
:meth:`Mesh.compute_kform_direct_dof_map`, which returns a :class:`DirectDofMap`.

Hybridized elements keep all local degrees of freedom and gain one multiplier per shared-object constraint row,
so :func:`solve_hybridized` assembles a saddle-point block system. Direct, the traces are eliminated against
shared-object unknowns by an :math:`L_2` inner product -- the pairing the
:ref:`boundary constraints <fdg_boundary_constraints>` enforce: each element's constraint rows, stacked over its
shared objects, are eliminated by a QR of their transpose, leaving the element-to-global transfer. Direct and
hybridized solves of one problem agree to solver tolerance.

The decomposition of the global space
-------------------------------------

In :math:`N` dimensions the global unknown vector splits into

- one block per object of dimension :math:`N - 1` down to :math:`0` that the k-form traces (shared or on the
  mesh boundary), holding one unknown per test function of the object's common windowed Legendre space;
- one *free mode* per dimension of each element's constraint nullspace -- the trace components orthogonal to
  every shared object's window space, element-private and not a coordinate sub-range of the element's degrees
  of freedom.

:meth:`Mesh.compute_kform_direct_dof_map` reports the first element-private unknown per element in
``element_interior_offsets`` and the vector size in ``global_dof_count``; object blocks precede the
element-private ones. A component of order past an object's dimension has no trace on it; a block is empty when
no component traces the object or an axis window floors to a zero count.

Which object owns a degree of freedom
-------------------------------------

Per component of a :math:`k`-form, same window structure as the hybridized rows: an axis carrying one of the
component's covector axes reads the order-:math:`(q - 1)` basis of the common order :math:`q` (:math:`q`
functions); every other axis reads the leading :math:`q + 1 - 2` functions of the full basis, floored at zero.
The per-axis :math:`q` is the minimum over the incident elements' *free* axes (the orientation record opens with
the pinned axes); a high-order neighbour is never over-constrained by low-order ones.

The elimination
---------------

Over the element's shared objects in dimension-then-id order, let :math:`C_e` stack the boundary-mass pairing
rows (object window tests :math:`\times` element DoFs) and :math:`B_e` the block-diagonal stack of the objects'
test-space Gram matrices. The core QR-factorizes :math:`C_e^\top` and emits, per element degree of freedom, the
*constrained* part -- the object coefficients :math:`u_b` expressed in element DoFs, with
:math:`C_e\,R_{\text{map}} = B_e` exactly -- and the *free* part: one orthogonal mode per dimension of
:math:`\operatorname{null}(C_e)`, numbered from ``element_interior_offsets``. So
:math:`u_i = R_{\text{map}}\,u_b + w` over free-mode amplitudes :math:`w` parametrizes precisely the element
traces whose windowed projections onto each shared object agree across elements. :math:`C_e^\top` has full row
rank: the per-axis minima put the common polynomial space inside every incident element.

Coefficients at most :math:`2^{-40}` -- about :math:`10^{-12}` relative -- of the largest entry of their
element's part (constrained coefficients and free modes measured separately) are pruned as roundoff. Exact
:math:`\pm 1` transfers occur only for top-order forms
(:math:`k = N`): no object carries a trace, every degree of freedom stays element-private, the map is the
identity. Lower orders mix modes -- the 1-D hand case (two line elements, Legendre order 1, 0-form) yields 3
globals with entries :math:`0.5` and :math:`\pm 0.5` and no private DoFs; 3-D :math:`k = 1` Legendre mixes
:math:`\pm 1/4`. Projection values, not defects. Free modes are retained; nothing leaves the system.

The Gram matrix :math:`B_e` comes from :c:func:`constraint_boundary_mass_gram`, a public helper of
:file:`constraints/constraints.h` shared with the hybridized boundary-mass assembly; the test space is
hierarchic Legendre regardless of element family.

Any basis family; preconditions
-------------------------------

The elimination uses the :math:`L_2` pairing alone, so any element basis family works: Legendre, Gauss-Lobatto
and other Lagrange variants, Chebyshev, Bernstein (see :ref:`fdg_basis_functions`) all yield a valid
``DirectDofMap`` for the same mesh and specs. Only a zero basis order is rejected, on any family: no test
functions, empty common space, degenerate constraint. In the C core :c:func:`direct_continuity_prepare` returns
:c:enumerator:`FDG_ERROR_NOT_IN_DOMAIN` for a zero order on any axis of any element; from Python,
:meth:`Mesh.compute_kform_direct_dof_map` raises :exc:`ValueError` naming the offending element and axis.

``element_specs`` needs exactly ``mesh.element_count`` :class:`KFormSpecs` entries describing the mesh
dimension, all with the same k-form degree; orders may differ per element and per axis, since the common space
takes minima. A wrong length, dimension or degree raises :exc:`ValueError`, a non-:class:`KFormSpecs` entry
:exc:`TypeError`, a C-core failure :exc:`RuntimeError`.

The pairing needs quadrature and a Legendre test basis, so the C entry points borrow a basis and an integration
registry for the call; the Python method releases the plan before returning. Every caller array is sized up
front by :c:func:`direct_continuity_layout`.

Unknown count
-------------

Hybridized counts :math:`\sum_e \text{dofs}(e)` element unknowns plus one multiplier per shared constraint row.
Direct counts one unknown per common window function per object plus one free mode per element nullspace
dimension, bounded by :math:`\sum_e \text{dofs}(e)`: every element degree of freedom maps into the system,
shared windows only save unknowns. Nothing leaves the system; the free modes of an element whose window is empty
on some object remain. The gallery example prints 25 direct vs 63 hybridized unknowns for its 2-D order-2 case.

Choosing the direct path
------------------------

A *numbering and assembly* choice, not a solve: ask for the map instead of the packed constraint rows, then
scatter each element matrix through it.

.. code-block:: python

    import numpy as np
    from fdg import BasisSpecs, BasisType, FunctionSpace, KFormSpecs, Mesh
    mesh = Mesh.from_corners(corners)                  # see Mesh.from_corners
    space = FunctionSpace(*(BasisSpecs(BasisType.LEGENDRE, order) for _ in range(ndim)))
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
        matrix = element_stiffness(element)            # your local operator
        dofs = range(int(dof_map.element_offsets[element]), int(dof_map.element_offsets[element + 1]))
        blocks = [block(dof) for dof in dofs]
        for local, (row_index, row_value) in enumerate(blocks):
            for column, (column_index, column_value) in enumerate(blocks):
                kernel = row_value[:, None] * matrix[local, column] * column_value
                np.add.at(operator,
                          (np.repeat(row_index, len(column_index)),
                           np.tile(column_index, len(row_value))),
                          kernel.ravel())

Each element-local degree of freedom owns the contiguous range ``entry_offsets[dof: dof + 2]``;
``entry_index`` names the global degree of freedom of each entry, ``entry_value`` its coefficient. Adding
:math:`v_i\, M_{ij}\, v_j` for every entry pair assembles onto the object coefficients and the free modes --
exactly what :c:func:`direct_continuity_scatter` does in the C core. :class:`DirectDofMap` reports every array
size, so the loop needs no counts of its own.

.. note::

   The *global* operator stays as singular as the element operator: a 0-form Laplacian keeps constants in its
   nullspace. Prescribe boundary data or fix one degree of freedom per connected component; the
   :ref:`hybridized formulation <fdg_hybridized_solves>` gauges via absorbed constraint rows.

The gallery example
:ref:`sphx_glr_auto_examples_plot_multi_element_laplace_direct_continuity.py` solves the same 0-form Laplace
problem both ways and prints, per refinement level, the two unknown counts, their ratio, and both errors against
the analytic solution.

The map's transfer also feeds a standard sparse assembly: the ``T_i * M[i, j] * T_j``
products are collected as COO triplets and summed into a CSC matrix, the boundary
unknowns are eliminated by row and column slicing, and the reduced system goes to
``scipy.sparse.linalg.splu`` and to the hybsol block solver with one block per
element or shared object. The gallery example
:ref:`sphx_glr_auto_examples_plot_direct_continuity_sparse.py` times assemble,
factorize, and solve for the dense, SciPy, and hybsol paths across cell counts and
polynomial orders and reports the relative agreement of the solutions.

.. autoclass:: DirectDofMap
