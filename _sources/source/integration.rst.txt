.. currentmodule:: fdg

.. _fdg_integration:

Integration
===========

Many of the higher level functions need to integrate functions or differential
forms over a domain. As such numerical integration is one of the cornerstones of
this module. For an example of the different integration rules and their relative
performance see :ref:`sphx_glr_auto_examples_plot_integration_rules.py`.

Integration Specifications
--------------------------

The way the method of integration is specified in :math:`N`-dimensional space
is by "outer-product" grid, where for each dimension the method (as given by :class:`IntegrationMethod`) and order of
integration is specified using :class:`IntegrationSpecs`.

.. autoclass:: IntegrationMethod
    :no-inherited-members:

.. autoclass:: IntegrationSpecs

Since values of integration nodes and weights are ofter reused, they are not stored inside :class:`IntegrationSpecs` objects.
Instead, they are stored in :class:`IntegrationRegistry` objects. By default, the ``fdg`` module provides one already,
however any number of new registries can be created.

.. autoclass:: IntegrationRegistry


.. autodata:: DEFAULT_INTEGRATION_REGISTRY

    Default :class:`IntegrationRegistry` used by ``fdg`` unless another is provided.


Integration Space
-----------------

To define how an integration should be done on a :math:`N`-dimensional domain the individual specifications
for each dimension (given as :class:`IntegrationSpecs`) are bundled together into :class:`IntegrationSpace`.

.. autoclass:: IntegrationSpace
