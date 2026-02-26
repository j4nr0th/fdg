#ifndef FDG_FUNCTION_SPACE_OBJECTS_H
#define FDG_FUNCTION_SPACE_OBJECTS_H

#include "../basis/basis_set.h"
#include "module.h"
#include <cutl/iterators/multidim_iteration.h>

typedef struct
{
    PyObject_VAR_HEAD;
    basis_spec_t specs[];
} function_space_object;

FDG_INTERNAL
extern PyType_Spec function_space_type_spec;

FDG_INTERNAL
function_space_object *function_space_object_create(PyTypeObject *type, unsigned n_basis,
                                                    const basis_spec_t FDG_ARRAY_ARG(specs, static n_basis));

FDG_INTERNAL
multidim_iterator_t *function_space_iterator(const function_space_object *space);

#endif // FDG_FUNCTION_SPACE_OBJECTS_H
