//
// Created by jan on 2025-09-07.
//

#ifndef FDG_MODULE_H
#define FDG_MODULE_H

#include "../common/common_defines.h"
#include <cutl/allocators.h>

//  Python ssize define
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

//  Prevent numpy from being re-imported
#ifndef PY_LIMITED_API
#define PY_LIMITED_API 0x030A0000
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _fdg
#endif

#include <Python.h>
#include <numpy/ndarrayobject.h>

// This must be after the NumPy include
#include <cpyutl.h>

FDG_INTERNAL
extern cutl_allocator_t SYSTEM_ALLOCATOR;

FDG_INTERNAL
extern cutl_allocator_t PYTHON_ALLOCATOR;

FDG_INTERNAL
extern cutl_allocator_t OBJECT_ALLOCATOR;

#define FDG_MODULE_NAME "fdg._fdg"
#define FDG_TYPE_NAME(type_name) (FDG_MODULE_NAME "." type_name)

typedef struct
{
    // Integration
    PyTypeObject *integration_spec_type;
    PyTypeObject *integration_registry_type;
    PyTypeObject *integration_space_type;

    // Basis
    PyTypeObject *basis_registry_type;
    PyTypeObject *basis_spec_type;
    PyTypeObject *covector_basis_type;

    // Function Spaces
    PyTypeObject *function_space_type;

    // DoFs
    PyTypeObject *degrees_of_freedom_type;
    PyTypeObject *coordinate_mapping_type;
    PyTypeObject *space_mapping_type;

    // Default Registries
    PyObject *registry_integration;
    PyObject *registry_basis;

    // K-Forms
    PyTypeObject *kform_specs_type;
    PyTypeObject *kform_type;
} interplib_module_state_t;

FDG_INTERNAL
extern PyModuleDef interplib_module;

static inline const interplib_module_state_t *interplib_get_module_state(PyTypeObject *type)
{
    PyObject *const mod = PyType_GetModuleByDef(type, &interplib_module);
    if (!mod)
    {
        return NULL;
    }
    return PyModule_GetState(mod);
}

FDG_INTERNAL
int heap_type_traverse_type(PyObject *self, visitproc visit, void *arg);

#endif // FDG_MODULE_H
