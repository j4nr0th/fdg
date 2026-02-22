#ifndef INTERPLIB_MAPPINGS_H
#define INTERPLIB_MAPPINGS_H

#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned ndim;
    integration_spec_t *int_specs;
    double values[];
} coordinate_map_object;

INTERPLIB_INTERNAL
extern PyType_Spec coordinate_map_type_spec;

INTERPLIB_INTERNAL
const double *coordinate_map_values(const coordinate_map_object *map);

INTERPLIB_INTERNAL
const double *coordinate_map_gradient(const coordinate_map_object *map, unsigned dim);

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned ndim;
    integration_spec_t *int_specs;
    double *determinant;
    double *inverse_maps;
    PyArrayObject **transformations;
    coordinate_map_object *maps[];
} space_map_object;

INTERPLIB_INTERNAL
extern PyType_Spec space_map_type_spec;

/**
 * Retrieves the pointer to the start of the inverse mapping data at a specific
 * integration point within a space map object.
 *
 * Rows of inverse mapping correspond to the reference dimensions, while
 * the columns correspond to the physical dimensions.
 *
 * @param map Pointer to the space_map_object that contains the mapping.
 * @param flat_index The flat index of the integration point for which the
 *                   inverse mapping data is needed.
 *
 * @return Pointer to the starting element of the inverse mapping data
 *         corresponding to the specified integration point.
 */
INTERPLIB_INTERNAL
const double *space_map_inverse_at_integration_point(const space_map_object *map, size_t flat_index);

INTERPLIB_INTERNAL
size_t space_map_inverse_size_per_integration_point(const space_map_object *map);

/**
 * Get the forward Jacobian derivative, meaning the derivative of the coordinate with respect to the input dimension.
 *
 * @param map Space map to use.
 * @param integration_point_index Flat index of the integration point at which to get the derivative.
 * @param idx_dim Index of the input dimension.
 * @param idx_coord Index of the coordinate.
 * @return Value of the specified forward derivative at the integration point.
 */
INTERPLIB_INTERNAL
double space_map_forward_derivative(const space_map_object *map, size_t integration_point_index, unsigned idx_dim,
                                    unsigned idx_coord);

/**
 * Get the backward Jacobian derivative, meaning the derivative of the input dimension with respect to the coordinate.
 *
 * @param map Space map to use.
 * @param integration_point_index Flat index of the integration point at which to get the derivative.
 * @param idx_dim Index of the input dimension.
 * @param idx_coord Index of the coordinate.
 * @return Value of the specified backward derivative at the integration point.
 */
INTERPLIB_INTERNAL
double space_map_backward_derivative(const space_map_object *map, size_t integration_point_index, unsigned idx_dim,
                                     unsigned idx_coord);

INTERPLIB_INTERNAL
extern PyMethodDef transformation_functions[];

INTERPLIB_INTERNAL
PyArrayObject *compute_basis_transform_impl(const space_map_object *map, const Py_ssize_t order);

#endif // INTERPLIB_MAPPINGS_H
