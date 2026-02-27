.. currentmodule:: fdg

.. _fdg_space_map:

Space Map
=========

Since the :ref:`fdg_basis_functions` and :ref:`fdg_integration` are both done on hypercube domain, where
each dimension goes from -1 to +1, this severely limits their usability on any deformed domain. As such
a mapping between the :math:`N`-dimensional reference domain and the target :math:`M`-dimensional (where
:math:`M \ge N`) target domain can be defined using :class:`SpaceMap` object. This mapping then determines
the way integration is done and how `k`-form components are mapped between the two domains.

The mapping is specified one target coordinate at the time, using :class:`CoordinateMap` objects. These
require the :class:`DegreesOfFreedom` as well as the :class:`IntegrationSpace` to define. This map is
used to store the values, as well as the derivatives of the coordinate mapping at all points in the integration
space.

.. note::

    All :class:`CoordinateMap` object that you want to use together for a complete :class:`SpaceMap` must use
    the same :class:`IntegrationSpace`.

.. autoclass:: CoordinateMap

To specify the :class:`SpaceMap`, :class:`CoordinateMap` must be specified for each of the target domain
dimensions. With that, :class:`SpaceMap` can be used in place of :class:`IntegrationSpace` for many functions
that integrate quantities. It also contains both the Jacobian and its (pseudo-)inverse, along with the determinant.

.. autoclass:: SpaceMap
