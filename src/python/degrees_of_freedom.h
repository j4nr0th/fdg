#ifndef FDG_DEGREES_OF_FREEDOM_H
#define FDG_DEGREES_OF_FREEDOM_H

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

FDG_INTERNAL
extern PyType_Spec degrees_of_freedom_type_spec;

FDG_INTERNAL
dof_object *dof_object_create(PyTypeObject *subtype, const unsigned ndim,
                              const basis_spec_t basis_specs_in[static ndim]);

/**
 * Project the degrees of freedom onto the plane `x[idim] == value`.
 *
 * Python wrapper around the plane projection used to construct boundary
 * spaces; requires `dofs->n_dims >= 1` and `idim < dofs->n_dims`.
 *
 * @param state Interpreter module state.
 * @param dofs Degrees of freedom to project.
 * @param idim Index of the fixed dimension.
 * @param value Position of the plane along the fixed dimension.
 * @return The projected degrees of freedom, or NULL with a Python exception set.
 */
FDG_INTERNAL
dof_object *dof_at_boundary_impl(const interplib_module_state_t *state, const dof_object *dofs, unsigned idim,
                                 double value);

FDG_INTERNAL
PyObject *dof_reconstruct_at_integration_points(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                Py_ssize_t nargs, const PyObject *kwnames);

FDG_INTERNAL
PyObject *dof_reconstruct_derivative_at_integration_points(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           const PyObject *kwnames);

FDG_INTERNAL
int *reconstruction_derivative_indices(unsigned ndim, PyObject *py_indices);

FDG_INTERNAL
int dof_reconstruction_state_init(const dof_object *this, unsigned ndim,
                                  const integration_spec_t integration_specs[const static ndim],
                                  const integration_registry_object *python_integration_registry,
                                  const basis_registry_object *python_basis_registry,
                                  reconstruction_state_t *recon_state);

FDG_INTERNAL
void dof_reconstruction_state_release(reconstruction_state_t *recon_state, basis_set_registry_t *basis_registry,
                                      unsigned ndim, const basis_set_t *basis_sets[static ndim]);

#endif // FDG_DEGREES_OF_FREEDOM_H
