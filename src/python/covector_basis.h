#ifndef FDG_PYTHON_COVECTOR_BASIS_H
#define FDG_PYTHON_COVECTOR_BASIS_H

#include "../basis/basis_lagrange.h"
#include "../basis/covector_basis.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    const covector_basis_t basis;
} covector_basis_object;

FDG_INTERNAL
extern PyType_Spec covector_basis_type_spec;

FDG_INTERNAL
covector_basis_object *covector_basis_object_create(PyTypeObject *type, covector_basis_t basis);

FDG_INTERNAL
unsigned kform_basis_get_num_dofs(unsigned ndim, const basis_spec_t FDG_ARRAY_ARG(basis, static ndim), unsigned order,
                                  const uint8_t FDG_ARRAY_ARG(components, static order));

FDG_INTERNAL
void kform_basis_set_iterator(unsigned ndim, const basis_spec_t FDG_ARRAY_ARG(basis, static ndim), unsigned order,
                              const uint8_t FDG_ARRAY_ARG(components, static order), multidim_iterator_t *iter);

#endif // FDG_PYTHON_COVECTOR_BASIS_H
