.. currentmodule:: fdg

.. _fdg_boundary_constraints:

Boundary Constraints
====================

A boundary constraint compares the trace of an element k-form with a test
k-form on one of its boundaries.  The public function
:func:`compute_kform_boundary_constraints` constructs the rows for one
selected element and one selected mesh boundary object.

Mathematical construction
-------------------------

Let :math:`\widehat{F}` be a canonical reference boundary and let
:math:`F_e` be its realization on element :math:`e`.  The selected mesh object
provides an orientation record that identifies the fixed element axes and maps
the free canonical axes to the element axes.  The element map is restricted to
that boundary by repeated calls to :meth:`SpaceMap.boundary`.

For a k-form, the restricted map supplies the tangential pullback.  If
:math:`u_e` is represented by element degrees of freedom and :math:`v` is a
test k-form on the boundary, a row of the constraint operator represents

.. math::

   (v, \operatorname{tr}_e u_e)_{F_e}
   = \int_{\widehat{F}}
     v \mathbin{\cdot} \operatorname{tr}_e u_e\,
     |\det J_{F_e}|\,d\widehat{x}.

The pullback is the identity for scalar forms.  For positive form order it
maps only tangential components and includes the usual k-form basis
transformation.  The surface measure is positive; orientation signs are kept
in the component mapping and in the canonical boundary parameterization.

For two adjacent elements, let :math:`T_A` and :math:`T_B` be the operators
returned for the same mesh boundary object.  Continuity can be enforced by
assembling

.. math::

   T_A u_A - T_B u_B = 0.

It can also be checked after a solution has been computed by applying the two
operators separately and comparing their row values.  The test space must be
the same on both calls.  The mesh topology supplies the relative orientation,
so the two calls do not need to use the same local ordering of the boundary
axes.

Packed operator format
----------------------

The function returns four one-dimensional arrays:

``row_offsets``
    An array of length ``n_rows + 1``.  Entries for row ``i`` occupy the half
    open range ``row_offsets[i]:row_offsets[i + 1]``.

``components``
    Element k-form component index for each packed entry.

``local_dofs``
    DoF index inside the corresponding element component.

``coefficients``
    Physical trace inner-product coefficient for the entry.

For flattened element values ``u`` the value of row ``i`` is therefore

.. math::

   r_i(u) = \sum_{j=row\_offsets_i}^{row\_offsets_{i+1}-1}
     coefficients_j\,u[components_j, local\_dofs_j].

The arrays can be converted into a sparse matrix with one column per flattened
element DoF.  The Python API intentionally returns the packed representation
so callers can choose their own global DoF numbering and sparse matrix type.

Topology and dimensions
-----------------------

``collections`` is a tuple containing object-boundary arrays from dimension
one through the element dimension.  For a three-dimensional element, for
example, it contains line, face, and volume collections.  A collection for
objects of dimension ``d`` has shape ``(count, 2*d)`` and stores the IDs of its
lower and upper boundaries in each local axis.

``boundary_id`` is an ID in the collection matching the test space dimension:
points use the implicit point collection, lines use the one-dimensional
collection, faces use the two-dimensional collection, and so on.  The selected
object must be immersed in ``element_id``.  The wrapper rejects incompatible
orders and dimensions, non-unique topology, or a boundary that is not present
in the selected element.

A zero-dimensional :class:`FunctionSpace` is the canonical test space for a
point boundary:

.. code-block:: python

   point_test = KFormSpecs(0, FunctionSpace())

Example
-------

The gallery example :ref:`sphx_glr_auto_examples_plot_boundary_constraints.py`
constructs adjacent two-dimensional and three-dimensional elements.  It
prints the packed rows, materializes them with ``scipy.sparse``, and checks
polynomial traces on shared faces, lines, and points before plotting the
geometries.

.. autofunction:: compute_kform_boundary_constraints
