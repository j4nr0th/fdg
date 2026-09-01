#ifndef FDG_MAPPINGS_H
#define FDG_MAPPINGS_H

#include "../integration/integration_rules.h"
#include "../operations/matrices.h"
#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned ndim;
    integration_spec_t *int_specs;
    // TODO: This is dirty. Clean it up!
    PyObject *dofs;
    PyObject *integration_registry;
    PyObject *basis_registry;
    //
    double values[];
} coordinate_map_object;

FDG_INTERNAL
extern PyType_Spec coordinate_map_type_spec;

FDG_INTERNAL
const double *coordinate_map_values(const coordinate_map_object *map);

FDG_INTERNAL
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

FDG_INTERNAL
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
FDG_INTERNAL
const double *space_map_inverse_at_integration_point(const space_map_object *map, size_t flat_index);

FDG_INTERNAL
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
FDG_INTERNAL
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
FDG_INTERNAL
double space_map_backward_derivative(const space_map_object *map, size_t integration_point_index, unsigned idx_dim,
                                     unsigned idx_coord);

FDG_INTERNAL
extern PyMethodDef transformation_functions[];

FDG_INTERNAL
PyArrayObject *compute_basis_transform_impl(const space_map_object *map, const Py_ssize_t order);

/**
 * Compute the inverse transformation from the Jacobian matrix.
 *
 * Inverts the jacobian matrix using QR decomposition and computes the determinant.
 * The inverse transformation is stored in the output matrix.
 *
 * @param jacobian The Jacobian matrix to invert. Overwritten during the QR decomposition. Must have dimensions (rows,
 * cols).
 * @param q_matrix The Q matrix from the QR decomposition of the Jacobian. Must have dimensions (rows, rows).
 * @param out_matrix The output matrix to store the inverse transformation. Must have dimensions (cols, rows).
 * @returns The determinant of the Jacobian matrix.
 */
FDG_INTERNAL
double compute_inverse_transform(const matrix_t jacobian, const matrix_t q_matrix, const matrix_t out_matrix);

/**
 * Compute the transformation factors for k-form basis from the inverse maps.
 *
 * The inverse maps must be stored point-major, with n_dims * n_maps row-major entries
 * per point (rows corresponding to the reference dimensions and columns to the physical
 * dimensions).
 *
 * @param n_dims Number of dimensions of the reference space.
 * @param n_maps Number of coordinates (physical dimensions).
 * @param order Order of the k-form basis.
 * @param inverse_maps Point-major array of the inverse map entries.
 * @param determinant Array of n_pts determinants.
 * @param n_pts Number of points.
 * @param out Output array of combination_total_count(n_dims, order) *
 *            combination_total_count(n_maps, order) * n_pts values, ordered with the
 *            input component index as the slowest and the point index as the fastest.
 *
 * @return 0 on success, -1 on allocation failure (with a Python exception set).
 */
FDG_INTERNAL
int compute_basis_transform_from_inverse(const unsigned n_dims, const unsigned n_maps, const unsigned order,
                                         const double *inverse_maps, const double *determinant, const size_t n_pts,
                                         double *out);

#endif // FDG_MAPPINGS_H
