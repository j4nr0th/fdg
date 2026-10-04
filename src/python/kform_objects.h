#ifndef FDG_KFORM_OBJECTS_H
#define FDG_KFORM_OBJECTS_H

#include "../kforms/kform_types.h"
#include "function_space_objects.h"
#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    function_space_object *function_space;
    unsigned order;
    size_t component_offsets[];
} kform_spec_object;

static inline kform_spec_t kform_specs_from_python(const kform_spec_object *this)
{
    // The variable-length tail of this object holds the component offsets, so Py_SIZE is the component count
    // plus one; the dimension lives on the base function space.
    const Py_ssize_t ndim = this->function_space ? Py_SIZE(this->function_space) : 0;
    return (kform_spec_t){
        .ndim = (unsigned)ndim,
        .order = this->order,
        .basis = ndim ? this->function_space->specs : NULL,
    };
}

FDG_INTERNAL
extern PyType_Spec kform_spec_type_spec;

typedef struct
{
    PyObject_VAR_HEAD;
    kform_spec_object *specs;
    double values[];
} kform_object;

FDG_INTERNAL
extern PyType_Spec kform_type_spec;

FDG_INTERNAL
kform_object *kform_object_create(PyTypeObject *type, kform_spec_object *spec, int zero_init);

#endif // FDG_KFORM_OBJECTS_H
