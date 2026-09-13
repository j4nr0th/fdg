#ifndef FDG_ELEMENT_DATA_OBJECTS_H
#define FDG_ELEMENT_DATA_OBJECTS_H

#include "../data/element_dofs.h"
#include "../data/element_geometry.h"
#include "../data/element_kforms.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    element_geometry_t *data;  // Owned; created with SYSTEM_ALLOCATOR.
    PyObject **option_objects; // [2 * option_count] lazily built, owned refs:
                               // [2i + 0] FunctionSpace of the option.
                               // [2i + 1] IntegrationSpace of the option.
    int frozen;                // Set once a numpy view of the storage has been handed out.
} mesh_geometry_object;

typedef struct
{
    PyObject_HEAD;
    element_kforms_t *data;   // Owned; created with SYSTEM_ALLOCATOR.
    PyObject **space_objects; // [space_capacity] lazily built FunctionSpace per base space, owned refs.
    PyObject ***field_specs;  // [field_count][space_capacity] lazily built KFormSpecs, owned refs.
    unsigned space_capacity;  // Allocated length of space_objects and of every field_specs row.
    int frozen;               // Set once a numpy view of the storage has been handed out.
} element_kforms_object;

typedef struct
{
    PyObject_HEAD;
    element_dofs_t *data;      // Owned; created with SYSTEM_ALLOCATOR.
    PyObject **option_objects; // [option_count] lazily built FunctionSpace, owned refs.
    int frozen;                // Set once a numpy view of the storage has been handed out.
} element_dofs_object;

FDG_INTERNAL
extern PyType_Spec mesh_geometry_type_spec;

FDG_INTERNAL
extern PyType_Spec element_kforms_type_spec;

FDG_INTERNAL
extern PyType_Spec element_dofs_type_spec;

#endif // FDG_ELEMENT_DATA_OBJECTS_H
