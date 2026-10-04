#ifndef FDG_PYTHON_DIRECT_CONTINUITY_H
#define FDG_PYTHON_DIRECT_CONTINUITY_H

#include "module.h"

typedef struct
{
    PyObject_HEAD;
    size_t global_dof_count;                 ///< Size of the global unknown vector.
    size_t element_dof_count;                ///< Sum of the elements' local DoF counts.
    size_t entry_count;                      ///< Nonzeros of the element-to-global transfer.
    PyArrayObject *element_offsets;          ///< Owned per-element local DoF offsets, int64.
    PyArrayObject *element_interior_offsets; ///< Owned per-element element-private DoF offsets, int64.
    PyArrayObject *entry_offsets;            ///< Owned row offsets of the transfer, int64.
    PyArrayObject *entry_index;              ///< Owned global DoF of every entry, int64.
    PyArrayObject *entry_value;              ///< Owned coefficient of every entry, double.
} direct_dof_map_object;

FDG_INTERNAL
extern PyType_Spec direct_dof_map_type_spec;

/**
 * @brief Build the direct element-to-global transfer of one mesh, bound as a Mesh method.
 *
 * @return A new DirectDofMap, or NULL with a Python exception set.
 */
FDG_INTERNAL
PyObject *mesh_compute_kform_direct_dof_map(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                            const Py_ssize_t nargs, PyObject *kwnames);

#endif // FDG_PYTHON_DIRECT_CONTINUITY_H
