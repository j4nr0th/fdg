//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _fdg
#include "module.h"

//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

// Internal C headers
#include "../common/error.h"
#include "../integration/gauss_lobatto.h"
#include "../polynomials/bernstein.h"
#include "../polynomials/lagrange.h"
#include "basis_objects.h"
#include "integration_objects.h"
#include "kform_objects.h"
#include "mappings.h"
#include "mass_matrices.h"

// Topology
#include "covector_basis.h"
#include "cpyutl.h"
#include "degrees_of_freedom.h"
#include "function_space_objects.h"
#include "incidence.h"
/**
 *
 *  ALLOCATORS
 *
 */

//  Magic numbers meant for checking with allocators that don't need to store
//  state.
enum
{
    SYSTEM_MAGIC = 0xBadBeef,
    PYTHON_MAGIC = 0x600dBeef,
};

static void *allocate_system(void *state, size_t size)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_RawMalloc(size);
}

static void *reallocate_system(void *state, void *ptr, size_t new_size)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_RawRealloc(ptr, new_size);
}

static void free_system(void *state, void *ptr)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    PyMem_RawFree(ptr);
}

FDG_INTERNAL
cutl_allocator_t SYSTEM_ALLOCATOR = {
    .allocate = allocate_system,
    .deallocate = free_system,
    .reallocate = reallocate_system,
    .state = (void *)SYSTEM_MAGIC,
};

static void *allocate_python(void *state, size_t size)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_Malloc(size);
}

static void *reallocate_python(void *state, void *ptr, size_t new_size)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_Realloc(ptr, new_size);
}

static void free_python(void *state, void *ptr)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    PyMem_Free(ptr);
}

FDG_INTERNAL
cutl_allocator_t PYTHON_ALLOCATOR = {
    .allocate = allocate_python,
    .deallocate = free_python,
    .reallocate = reallocate_python,
    .state = (void *)PYTHON_MAGIC,
};

static void free_module_state(void *module)
{
    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(module);
    *module_state = (interplib_module_state_t){};
}

static int interplib_add_types(PyObject *mod)
{
    if (PyArray_ImportNumPyAPI() < 0)
        return -1;

    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        return -1;
    }

    if ((module_state->integration_spec_type =
             cpyutl_add_type_from_spec_to_module(mod, &integration_specs_type_spec, NULL)) == NULL ||
        (module_state->integration_registry_type =
             cpyutl_add_type_from_spec_to_module(mod, &integration_registry_type_spec, NULL)) == NULL ||
        (module_state->basis_spec_type = cpyutl_add_type_from_spec_to_module(mod, &basis_specs_type_spec, NULL)) ==
            NULL ||
        (module_state->basis_registry_type =
             cpyutl_add_type_from_spec_to_module(mod, &basis_registry_type_specs, NULL)) == NULL ||
        (module_state->covector_basis_type =
             cpyutl_add_type_from_spec_to_module(mod, &covector_basis_type_spec, NULL)) == NULL ||
        (module_state->function_space_type =
             cpyutl_add_type_from_spec_to_module(mod, &function_space_type_spec, NULL)) == NULL ||
        (module_state->integration_space_type =
             cpyutl_add_type_from_spec_to_module(mod, &integration_space_type_spec, NULL)) == NULL ||
        (module_state->degrees_of_freedom_type =
             cpyutl_add_type_from_spec_to_module(mod, &degrees_of_freedom_type_spec, NULL)) == NULL ||
        (module_state->coordinate_mapping_type =
             cpyutl_add_type_from_spec_to_module(mod, &coordinate_map_type_spec, NULL)) == NULL ||
        (module_state->space_mapping_type = cpyutl_add_type_from_spec_to_module(mod, &space_map_type_spec, NULL)) ==
            NULL ||
        (module_state->kform_specs_type = cpyutl_add_type_from_spec_to_module(mod, &kform_spec_type_spec, NULL)) ==
            NULL ||
        (module_state->kform_type = cpyutl_add_type_from_spec_to_module(mod, &kform_type_spec, NULL)) == NULL)
    {
        return -1;
    }

    return 0;
}

static int interplib_add_functions(PyObject *mod)
{
    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        return -1;
    }

    if (PyModule_AddFunctions(mod, mass_matrices_methods) < 0 || PyModule_AddFunctions(mod, incidence_methods) < 0 ||
        PyModule_AddFunctions(mod, transformation_functions) < 0)
        return -1;

    return 0;
}

static int module_add_steal(PyObject *mod, const char *name, PyObject *obj)
{
    if (!obj)
        return -1;
    const int res = PyModule_AddObjectRef(mod, name, obj);
    Py_XDECREF(obj);
    return res;
}

static int interplib_add_registries(PyObject *mod)
{
    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        return -1;
    }

    // Add integration registry
    if (module_add_steal(mod, "DEFAULT_INTEGRATION_REGISTRY",
                         (module_state->registry_integration = (PyObject *)integration_registry_object_create(
                              module_state->integration_registry_type))) < 0)
        return -1;

    // Add basis registry
    if (module_add_steal(mod, "DEFAULT_BASIS_REGISTRY",
                         (module_state->registry_basis =
                              (PyObject *)basis_registry_object_create(module_state->basis_registry_type))) < 0)
        return -1;

    return 0;
}

PyModuleDef interplib_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = FDG_MODULE_NAME,
    .m_doc = "Internal C-extension implementing interpolation functions",
    .m_size = sizeof(interplib_module_state_t),
    .m_methods = NULL,
    .m_free = free_module_state,
    .m_slots =
        (PyModuleDef_Slot[]){
            {.slot = Py_mod_exec, .value = interplib_add_types},
            {.slot = Py_mod_exec, .value = interplib_add_functions},
            {.slot = Py_mod_exec, .value = interplib_add_registries},
            {.slot = Py_mod_multiple_interpreters, .value = Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
            {},
        },
};

PyMODINIT_FUNC PyInit__fdg(void)
{
    import_array();

    return PyModuleDef_Init(&interplib_module);
}

int heap_type_traverse_type(PyObject *self, const visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}
