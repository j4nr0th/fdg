#ifndef INTERPLIB_DEGREES_OF_FREEDOM_H
#define INTERPLIB_DEGREES_OF_FREEDOM_H

#include "../reconstruction/reconstruction.h"
#include "basis_objects.h"
#include "integration_objects.h"
#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned n_dims;
    basis_spec_t *basis_specs;
    double values[];
} dof_object;

INTERPLIB_INTERNAL
extern PyType_Spec degrees_of_freedom_type_spec;

INTERPLIB_INTERNAL
PyObject *dof_reconstruct_at_integration_points(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                Py_ssize_t nargs, const PyObject *kwnames);

INTERPLIB_INTERNAL
PyObject *dof_reconstruct_derivative_at_integration_points(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           const PyObject *kwnames);

INTERPLIB_INTERNAL
int *reconstruction_derivative_indices(unsigned ndim, PyObject *py_indices);

INTERPLIB_INTERNAL
int dof_reconstruction_state_init(const dof_object *this, const integration_space_object *integration_space,
                                  const integration_registry_object *python_integration_registry,
                                  const basis_registry_object *python_basis_registry,
                                  reconstruction_state_t *recon_state);

INTERPLIB_INTERNAL
void dof_reconstruction_state_release(reconstruction_state_t *recon_state, basis_set_registry_t *basis_registry,
                                      unsigned ndim, const basis_set_t *basis_sets[static ndim]);

#endif // INTERPLIB_DEGREES_OF_FREEDOM_H
