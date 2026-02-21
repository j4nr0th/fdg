#ifndef INTERPLIB_KFORM_OBJECTS_H
#define INTERPLIB_KFORM_OBJECTS_H

#include "function_space_objects.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    function_space_object *function_space;
    unsigned order;
} kform_spec_object;

INTERPLIB_INTERNAL
extern PyType_Spec kform_spec_type_spec;
//
// typedef struct
// {
//     PyObject_VAR_HEAD;
//     kform_spec_object *specs;
//     size_t *component_offsets;
//     double values[];
// } kform_object;
//
// INTERPLIB_INTERNAL
// extern PyType_Spec kform_type_spec;

#endif // INTERPLIB_KFORM_OBJECTS_H
