.. currentmodule:: fdg

.. _fdg_boundary_constraints:

Boundary Constraints
====================

A boundary row pairs the trace of an element k-form with a test k-form on one
of its boundaries.  The public function
:func:`compute_kform_boundary_mass_matrices` assembles, in one call, the
*per-object boundary mass matrices* of one shared mesh boundary object: one
matrix per incident element, whose rows are the windowed common Legendre test
space of the object and whose columns pair with the element trace degrees of
freedom.  The trace tables reuse the integration-independent endpoint basis
cache for the element axes fixed on the boundary.  Trace assembly needs basis
values (including the lower-order values used by positive-form components),
not basis derivatives; there is therefore no separate derivative cache in this
path.

Mathematical construction
-------------------------

Let :math:`\widehat{F}` be a canonical reference boundary and let
:math:`F_e` be its realization on element :math:`e`.  Each incident element
provides an orientation record that identifies the fixed element axes and maps
the free canonical axes to the element axes; the mesh reports such records for
every boundary object, for example through :meth:`Mesh.iterate_boundary`.
The element map is restricted to that boundary by repeated calls to
:meth:`SpaceMap.boundary`.

For a k-form, the restricted map supplies the tangential pullback.  If
:math:`u_e` is represented by element degrees of freedom and :math:`v` is a
row of the common test space, the corresponding row of the boundary mass
matrix represents

.. math::

   (v, \operatorname{tr}_e u_e)_{F_e}
   = \int_{\widehat{F}}
     v \mathbin{\cdot} \operatorname{tr}_e u_e\,
     |\det J_{F_e}|\,d\widehat{x}.

The pullback is the identity for scalar forms.  For positive form order it
maps only tangential components and includes the usual k-form basis
transformation.  The surface measure is positive; orientation signs are kept
in the component mapping and in the canonical boundary parameterization.

Common test space
-----------------

The rows of every returned matrix come from one common test space: on each
free axis the common basis order is the lowest order of the incident
elements, and an axis that carries no covector of a k-form component drops
its two highest basis functions (``SKIPPED_BASIS = 2``), reading the leading
ones.  Each constraint then pairs the trace against exactly the low-degree
functions the lower-order space can represent — the L2 projection of the
(higher-order) boundary solution onto that space — so traces of
higher-order neighbours never over-constrain the shared object.  Because
both incident elements pair their traces against the same rows, equal traces
produce equal pairing values.

Continuity between adjacent elements
------------------------------------

For two adjacent elements :math:`A` and :math:`B` sharing a boundary object,
continuity is the statement

.. math::

   T_A u_A - T_B u_B = 0,

where :math:`T_A` and :math:`T_B` are the trace pairings collected by the two
per-object mass matrices.  It is assembled globally with
:meth:`Mesh.compute_kform_continuity_constraints`, which walks the shared
objects from faces to points, pairs consecutive incident elements with
opposite signs, and derives the same windowed common test spaces internally.
Prescribed boundary data are appended to those rows by
:meth:`Mesh.compute_kform_global_constraints`.

Packed row format
-----------------

The function returns ``(common_specs, common_integration, matrices,
packed_rows)``: the merged common :class:`KFormSpecs`, the merged common
integration space, one dense matrix per incident element, and — with
``packed=True`` — one five-array tuple per element:

``row_offsets``
    An array of length ``n_rows + 1``.  Entries for row ``i`` occupy the half
    open range ``row_offsets[i]:row_offsets[i + 1]``.

``sides``
    Zero-based index of the incident element within the call, for each packed
    entry.

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

CSR conversion
--------------

For global element-major rows — as returned by
:meth:`Mesh.compute_kform_continuity_constraints` and
:meth:`Mesh.compute_kform_global_constraints` —
:func:`packed_kform_constraints_to_csr` converts the packed representation to
``(data, indices, indptr)`` arrays accepted directly by
``scipy.sparse.csr_matrix``.  The column indices include the element offset
and the flattened component offset, so callers do not need to materialize row
indices or perform per-entry index arithmetic in Python.


Prescribed rows
---------------

Inter-element constraints need two or more incident elements; strongly
prescribed boundary data instead constrains a single element's trace against
data. :func:`compute_kform_boundary_trace_moments` is the explicit
single-element route behind those rows: it shares the assembly core with
:func:`compute_kform_boundary_mass_matrices` and returns the same
``(common_specs, common_integration, matrices, packed)`` tuple for one
element. A datum :math:`g` on the face is bound by the moment rows

.. math::

   r_i(g) = \int_{\widehat{F}}
     v_i \mathbin{\cdot} \operatorname{tr} g\,
     |\det J_{F}|\,d\widehat{x},

so the right-hand side entry of row :math:`i` is the packed row applied to
the mapped data degrees of freedom. The boundary-condition assembler
(:meth:`Mesh.compute_kform_global_constraints`) builds its prescribed rows
through this interface.

Boundary load
-------------

The companion function :func:`compute_kform_boundary_load` assembles the
*chain integral* of a :math:`k`-form datum against the trace of the element
:math:`(k-1)`-form on a codimension-1 boundary face, for every datum degree
:math:`k = 1, \dots, n` (the datum order is ``element_spec.order + 1``).  For
a face with fixed normal axis :math:`a` at side :math:`s` (``-1`` for the
start side, ``+1`` for the end side) and reference face quadrature points
:math:`\widehat{x}_p` with weights :math:`w_p`, the load entries are

.. math::

   b_j = s\,o\,(-1)^{|\{i \in J_e : i < a\}|}
     \sum_p w_p\, u_{J_e \cup \{a\}}(\Phi_{F_e}(\widehat{x}_p))\,
     B_j(\widehat{x}_p),

where :math:`J_e` is the set of element-frame axes of the traced component,
:math:`o` is the orientation sign of the mapped component, :math:`B_j` runs
over the element :math:`(k-1)`-form basis of that component and
:math:`\Phi_{F_e}` is the restricted element map.  Only the datum component
whose axes are :math:`J_e \cup \{a\}` pairs with the traced component; all
other components contribute nothing.  The datum is passed as one callable per
:math:`k`-form component in element-frame component order (the same order as
:class:`KFormSpecs` components); each callable receives the physical
coordinates of the canonical face points and returns one value per point.
When :math:`k = n` there is a single component and a single callable is
accepted directly.

At :math:`k = n` the traced component is the volume form component and the
exponent reduces to :math:`a`, so the sign becomes :math:`s\,(-1)^a`, the
outward boundary orientation relative to the canonical face parameterization,
and the formula reduces to the scalar chain integral of previous releases.
Zero-form data (:math:`k = 0`) is not covered by the load; scalar boundary
values are imposed strongly with :meth:`Mesh.compute_kform_global_constraints`.

Unlike the trace constraint, the load is a metric-free chain integral: the
surface measure of the face cancels exactly against the coefficient scaling of
the pulled-back form, so no surface weights or pullback tensor enter the
assembly.

In the mixed Poisson formulation the natural boundary term of the momentum
equation is

.. math::

   \int_{\partial\Omega} p \wedge \star u_D
   = \sum_{F_e \subset \partial\Omega} \int_{F_e} u_D\,(\operatorname{tr} p),

which applies the Dirichlet condition :math:`u = u_D` weakly: the load of one
face with :math:`\mathrm{data} = u_D` is added to the momentum (flux) block of
the right-hand side.  The gallery example
:ref:`sphx_glr_auto_examples_plot_multi_element_poisson_bc.py` combines this
weak Dirichlet condition on some faces with the strong Neumann condition
:math:`\operatorname{tr} q = g` (enforced by Lagrange multipliers with
:meth:`Mesh.compute_kform_global_constraints`) on the remaining faces.

Example
-------

The gallery example :ref:`sphx_glr_auto_examples_plot_boundary_constraints.py`
constructs adjacent two-dimensional and three-dimensional elements that share
one edge and one face.  It prints the packed rows of the per-object boundary
mass matrices for every k-form order on the shared object, checks that
constant traces pair identically on both elements, and plots the geometries.

.. autofunction:: compute_kform_boundary_mass_matrices

.. autofunction:: compute_kform_boundary_trace_moments
