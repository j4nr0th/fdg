#ifndef FDG_SAMPLED_SPACE_MAP_H
#define FDG_SAMPLED_SPACE_MAP_H
#include "mappings.h"
typedef struct
{
    PyObject_HEAD;
    unsigned ndim;
    unsigned coords;
    unsigned *orders;
    double *determinant;
    double *inverse_maps;
    double *positions;
    PyArrayObject **transformations;
} sampled_space_map_object;

FDG_INTERNAL
extern PyType_Spec sampled_space_map_type_spec;

FDG_INTERNAL
extern PyMethodDef sampled_space_map_methods[];

/**
 * Create a sampled space map from a space map.
 *
 * Interpolates the positions and transformations of the space map onto a tensor
 * grid using Lagrange nodal basis. If `samples` is NULL, a uniform grid with
 * the specified orders is used.
 *
 * @param type The Python type object for the sampled space map.
 * @param map The space map object to sample.
 * @param orders The orders of the output grid for each dimension.
 * @param samples Concatenated sample coordinates for each dimension, or NULL
 *                for a uniform grid on [-1, 1].
 * @param registry The integration rule registry to use for the sampling.
 * @return A new sampled space map object, or NULL on error.
 */
FDG_INTERNAL
sampled_space_map_object *sampled_space_map_create(PyTypeObject *type, space_map_object *map, const unsigned *orders,
                                                   const double *samples, integration_rule_registry_t *registry);

FDG_INTERNAL
PyObject *transform_kform_to_target_grid(PyObject *mod, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

#endif // FDG_SAMPLED_SPACE_MAP_H
