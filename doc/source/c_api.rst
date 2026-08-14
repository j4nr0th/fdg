C API
=====

This section documents the C core of ``fdg``: the public functions, types
and macros declared in the headers under ``src/`` (excluding the Python
bindings in ``src/python/``). The core implements the low-level building
blocks used by the Python API: polynomial bases, quadrature rules,
topological mesh handling, constraint assembly and reconstruction.

The C core is written in C17 and has no dependency other than the ``cutl``
utility library. Error handling is done through status codes: most functions
return a status or result type such as :c:type:`fdg_result_t`,
:c:type:`topo_status_t` or :c:type:`constraint_status_t`, where the zero
value indicates success.

Functions marked :c:macro:`FDG_INTERNAL` are hidden from the shared library
symbol table; they are internal to the library but still usable by code that
links the static archive, such as the Python bindings.

.. toctree::
   :maxdepth: 2

   c_common
   c_basis
   c_polynomials
   c_integration
   c_matrices
   c_topology
   c_constraints
   c_reconstruction
