.. currentmodule:: fdg

.. _fdg_domain_types:

Domain Types
============

To help with generating :class:`CoordinateMapping` and :class:`SpaceMap` objects
based on common domain boundaries, helper types are provided for deformed domains
of arbitrary reference dimension. :class:`Line` and :class:`Quad` remain convenient
constructors for one- and two-dimensional domains; :class:`Hypercube` assembles an
N-dimensional domain from opposite boundary pairs.

``Hypercube.from_boundary_points`` accepts one pair of tensor-product point arrays
per reference axis.  Each array has shape ``(n_0, ..., n_(N-2), n_physical)`` and
uses the remaining parent axes in ascending order.  The arrays are fitted with
uniform Lagrange interpolation, and neighboring boundaries must agree on every
shared trace.

.. autoclass:: Line

.. autoclass:: Quad

.. autoclass:: Hypercube
