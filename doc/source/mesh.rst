.. currentmodule:: fdg

.. _fdg_mesh:

Mesh Topology
=============

A :class:`Mesh` holds the complete *topology* of a set of connected
:math:`N`-dimensional hypercube elements: the collections of every
topological object of every dimension (points, lines, faces, ...) and how
each object is immersed into the elements that contain it. The mesh carries
no geometry: points are identified by opaque IDs only, and the coordinates of
the elements come from the :ref:`space mapping <fdg_space_map>` of each
element. The primary use of the mesh is the generation of continuity
constraints between neighboring elements, see
:ref:`fdg_boundary_constraints`.

The implementation lives in the C core (:file:`topology/mesh.c`, documented
at :doc:`c_topology`); this page explains the concepts and the Python
binding.

Elements and their objects
--------------------------

A hypercube element in :math:`N` dimensions has :math:`2^N` corners, one per
plane combination of its axes. Every axis-aligned sub-object of dimension
:math:`m` inside the element is identified by two things:

- its *axis mask*: the set of the :math:`m` axes it spans, one of the
  :math:`\binom{N}{m}` masks; and
- its *planes*: for each of the :math:`N - m` fixed axes, whether the object
  sits at the start or the end plane of the element, one of
  :math:`2^{N - m}` positions.

The masks are enumerated in ascending numeric order, which is the
lexicographic order of the ascending axis sets used for the k-form
components in :ref:`fdg_math_background`; the positions are the mixed-radix
values of the fixed-axis plane indices, with the low axes least significant.
A corner of the element is the object of dimension zero at the plane
combination given by its corner coordinate, so its local index is the
mixed-radix corner index itself (bit :math:`a` of the index set = the corner
lies on the positive side of element axis :math:`a`).

Construction from corners
-------------------------

:meth:`Mesh.from_corners` builds the mesh from the corner point IDs of every
element: :math:`2^N` IDs per element, in the element's own frame, the corner
with local mixed-radix index :math:`k` having its point ID at
``corners[element * 2**N + k]``. A point that belongs to several elements is
shared by giving its ID as a corner of each of them.

For every element, the builder walks the objects of every dimension: for
each axis mask and each plane combination it collects the :math:`2^m` corner
IDs of the object and merges the object into the global mesh, keyed by its
*sorted* corner set. Two objects from different elements are the same global
object exactly when they consist of the same corner points. The points
themselves are implicit: point IDs are integers in
:math:`[0, \texttt{point\_count})`, and :meth:`Mesh.point_count` is one more
than the largest corner ID used. The corner data must come from a consistent
axis-aligned gluing: every object must be embedded identically in all of the
elements it is in. The input is validated as far as the merging machinery
allows; inconsistent gluing surfaces as an error from the immersion
computation.

The boundaries of an object are written exactly once, from the first element
that contains it, so the frame of the object — its set of spanning axes in
the ascending element-axis order — is canonical. The immersion computation
relies on this stability.

Collections and boundary slots
------------------------------

:meth:`Mesh.collections` returns one array per object dimension
:math:`d = 1, \dots, N` (copies of the internal data). The array of
dimension :math:`d` has shape ``(count, 2*d)``: object :math:`j` has
:math:`2d` boundary slots, slot :math:`i` being the boundary perpendicular
to axis :math:`i` at its *start* and slot :math:`i + d` the boundary at its
*end*. For :math:`d = 1` (lines) the boundaries are point IDs; otherwise
they are IDs into the collection of dimension :math:`d - 1`. The last
collection holds the :math:`N`-dimensional elements themselves, whose
boundaries are the element faces of dimension :math:`N - 1`.

The same layout is accepted by :meth:`Mesh.from_collections`, which takes
the collections of every dimension together with an explicit point count and
computes the immersion information from them.

Immersion and orientation
-------------------------

For every object the mesh records the elements that contain it and, for each
of them, an *orientation record*: :math:`N` signed, one-based entries. The
first :math:`N - m` entries are the *fixed axes*: the entry is positive when
the object sits at the end side of the axis and negative when it sits at the
start side, which identifies where the object is positioned in the element.
The remaining :math:`m` entries map the object's own local axes to the
element axes, with negative entries reversing the direction of the axis.

The iteration functions of the mesh return these records as an ``int8``
array of shape ``(element_count, ndim)``, one row per element ID. The
orientation of an object can differ between the elements that contain it;
this relative orientation is what the constraint assembly
(:ref:`fdg_boundary_constraints`) uses to match the traces of neighboring
elements.

Position lookup
---------------

:meth:`Mesh.element_object` resolves the global ID of the object at a given
position within one element. The position is specified with one signed,
one-based entry per element axis: zero when the object spans the axis,
:math:`-(j + 1)` when the object is perpendicular to axis :math:`j` at its
start, :math:`+(j + 1)` when it is perpendicular to it at its end; at least
one entry must be nonzero. The dimension of the found object is the number
of free (zero) entries, so a fully specified position identifies a point.
For example, ``(-1, +2, -3)`` in three dimensions names the point reached by
following element axis 0 to its start, axis 1 to its end and axis 2 to its
start.

The lookup descends the boundary chains of the collections: starting at the
element, each fixed axis crosses from the current object into the boundary
perpendicular to that axis at the requested side, so the cost is
proportional to the number of fixed axes.

Shared objects and the mesh boundary
------------------------------------

The mesh offers two iteration families:

- :meth:`Mesh.iterate_shared` and :meth:`Mesh.iterate_shared_all` visit the
  objects that are contained in at least two elements, with the element IDs
  sorted in ascending order. These are the objects on which continuity must
  be enforced. To build constraints without over-constraining, pair the
  consecutive elements of each shared object and constrain only the interior
  of the object, skipping its own boundaries.
- :meth:`Mesh.iterate_boundary` and :meth:`Mesh.iterate_boundary_all` visit
  the objects that lie on the outer boundary of the mesh: an object lies on
  the boundary when it is contained in a *boundary face*, an object of
  dimension :math:`N - 1` that is in exactly one element. Objects of
  dimension :math:`N - 1` on the boundary are therefore in exactly one
  element, while lower-dimensional objects on the boundary can still be in
  several elements — for example a point in the middle of a boundary edge of
  a two-dimensional mesh.

Each visited object is reported as a tuple
``(mdim, object_id, element_ids, orientations)``:

``mdim``
    Dimension of the object.

``object_id``
    Global ID of the object: a point ID for :math:`\texttt{mdim} = 0`,
    otherwise an index into the corresponding collection.

``element_ids``
    ``uint64`` array with the IDs of the elements the object is contained
    in, sorted in ascending order.

``orientations``
    ``int8`` array of shape ``(element_count, ndim)`` with the orientation
    records; row ``i`` belongs to ``element_ids[i]``.

The ``*_all`` variants iterate the object dimensions from :math:`N - 1`
down to zero. This is the order required for continuity constraints:
constraining the shared objects dimension by dimension, from the highest to
the lowest, ensures that no degree of freedom is constrained more than once.

Boundary constraints
--------------------

:meth:`Mesh.compute_kform_boundary_constraints` is identical to the free
function :func:`compute_kform_boundary_constraints`, except that the mesh
collections, the point count and the orientation of the selected boundary
within the selected element are taken from the mesh itself. See
:ref:`fdg_boundary_constraints` for the mathematical construction and the
packed format of the returned rows.

The gallery example
:ref:`sphx_glr_auto_examples_plot_multi_element_poisson.py` builds a mesh
with :meth:`Mesh.from_corners`, enumerates the shared faces with
:meth:`Mesh.iterate_shared` and assembles the flux continuity with
:meth:`Mesh.compute_kform_boundary_constraints`.

.. autoclass:: Mesh
