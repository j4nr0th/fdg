Topology
========

Topological mesh handling: collections of n-dimensional objects, their
boundaries, immersion of objects into elements, and iteration over element
boundaries.

The mesh stores, for every dimension :math:`d = 0, \dots, N`, the objects
of that dimension (points, lines, faces, ...) and how each object is
immersed into the elements that contain it. A collection of dimension
:math:`d` objects stores the boundary IDs of every object: object
:math:`j` has :math:`2d` boundary slots, slot :math:`i` being the
boundary perpendicular to axis :math:`i` at its *start* and slot
:math:`i + d` the boundary at its *end*. An immersion record stores, for
each occurrence of an object in an element, a signed one-based
orientation: the fixed axes identify where the object sits in the element
(negative = at the start side of the axis, positive = at the end), and the
remaining entries map the object's axes to the element axes, with negative
entries reversing the direction. The shared- and boundary-object iteration
functions in :file:`mesh.h` drive the assembly of the
:ref:`fdg_boundary_constraints` trace constraints: constraining the shared
objects dimension by dimension, from highest to lowest, ensures no degree
of freedom is constrained more than once.

.. c:autodoc:: topology/topology.h
.. c:autodoc:: topology/mesh.h
