#ifndef FDG_ELEMENT_DATA_OBJECTS_H
#define FDG_ELEMENT_DATA_OBJECTS_H

#include "../data/element_dofs.h"
#include "../data/element_kforms.h"
#include "../data/mesh_kform_specs.h"
#include "mappings.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    space_map_object **maps; // [allocated] owned refs to space_map_object, [0, count).
    Py_ssize_t count;
    Py_ssize_t allocated;
    unsigned input_dimensions;  // Reference dimensions shared by every stored map.
    unsigned output_dimensions; // Physical dimensions shared by every stored map.
} mesh_geometry_object;

typedef struct
{
    PyObject_HEAD;
    mesh_kform_specs_t *data; // Owned; created with SYSTEM_ALLOCATOR.
    PyObject **space_objects; // [space_capacity] lazily built FunctionSpace per base space, owned refs.
    PyObject ***field_specs;  // [field_count][space_capacity] lazily built KFormSpecs, owned refs.
    unsigned space_capacity;  // Allocated length of space_objects and of every field_specs row.
} mesh_kform_specs_object;

typedef struct
{
    PyObject_HEAD;
    PyObject *specs;        // Strong reference to the borrowed MeshKFormSpecs.
    element_kforms_t *data; // Owned; created with SYSTEM_ALLOCATOR.
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
extern PyType_Spec mesh_kform_specs_type_spec;

FDG_INTERNAL
extern PyType_Spec element_kforms_type_spec;

FDG_INTERNAL
extern PyType_Spec element_dofs_type_spec;

#endif // FDG_ELEMENT_DATA_OBJECTS_H
