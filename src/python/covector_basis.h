#ifndef INTERPLIB_PYTHON_COVECTOR_BASIS_H
#define INTERPLIB_PYTHON_COVECTOR_BASIS_H

#include "../../src/basis/covector_basis.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    const covector_basis_t basis;
} covector_basis_object;

INTERPLIB_INTERNAL
extern PyType_Spec covector_basis_type_spec;

INTERPLIB_INTERNAL
covector_basis_object *covector_basis_object_create(PyTypeObject *type, covector_basis_t basis);

INTERPLIB_INTERNAL
unsigned kform_basis_get_num_dofs(unsigned ndim, const basis_spec_t INTERPLIB_ARRAY_ARG(basis, static ndim),
                                  unsigned order, const uint8_t INTERPLIB_ARRAY_ARG(components, static order));

INTERPLIB_INTERNAL
void kform_basis_set_iterator(unsigned ndim, const basis_spec_t INTERPLIB_ARRAY_ARG(basis, static ndim), unsigned order,
                              const uint8_t INTERPLIB_ARRAY_ARG(components, static order), multidim_iterator_t *iter);

#endif // INTERPLIB_PYTHON_COVECTOR_BASIS_H
