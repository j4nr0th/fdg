#ifndef FDG_MAPPINGS_H
#define FDG_MAPPINGS_H

#include "../integration/integration_rules.h"
#include "../operations/matrices.h"
#include "degrees_of_freedom.h"
#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned ndim;
    integration_spec_t *int_specs;
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
 * Construct a space map from already validated coordinate maps.
 *
 * The coordinate maps must share their integration spaces; this is verified.
 * Each map in @p maps gets a new reference stored.
 *
 * @param subtype Space map type to allocate (the registered space map type or
 *                a subtype).
 * @param n_maps Number of coordinate maps; determines the physical dimension.
 * @param maps The coordinate maps, one per physical dimension.
 * @return The new space map, or NULL with a Python exception set.
 */
FDG_INTERNAL
space_map_object *space_map_object_create(PyTypeObject *subtype, unsigned n_maps, coordinate_map_object *const *maps);

/**
 * Restrict a space map to a reference-space boundary `x[idim] == end ? +1 : -1`.
 *
 * The restriction is a values-level pass over the integration-point values of
 * the coordinate maps: the fixed axis is evaluated at the plane and the
 * surviving axes are resampled onto the face grid. Exact whenever the
 * integration order is at least the dof order along every axis.
 *
 * @param state Interpreter module state.
 * @param integration_registry Registry supplying the element and face rules.
 * @param map Space map to restrict; must have `map->ndim >= 1` and
 *            `idim < map->ndim`.
 * @param idim Index of the fixed dimension.
 * @param end Which side of the slab to restrict to; non-zero for the upper.
 * @param provided_face_space Optional integration space for the face; must
 *                            have `map->ndim - 1` dimensions if given.
 * @return The restricted space map, or NULL with a Python exception set.
 */
FDG_INTERNAL
space_map_object *space_map_boundary_impl(const interplib_module_state_t *state,
                                          integration_registry_object *integration_registry,
                                          const space_map_object *map, unsigned idim, int end,
                                          integration_space_object *provided_face_space);

/**
 * Restrict a space map to the boundary given by a full orientation array in a
 * single values-level pass.
 *
 * The first @p bdim entries of @p orientation are the fixed normal
 * axes of the boundary (signed one-based, negative for the start of the axis),
 * the remaining entries the surviving axes in ascending order. The face
 * integration space is the element space with the fixed axes removed, so the
 * face values follow by evaluating the fixed-axis interpolant at the plane on
 * every value block (values and gradients) of the coordinate maps.
 *
 * @param state Interpreter module state.
 * @param integration_registry Registry supplying the element and face rules.
 * @param map Space map to restrict.
 * @param bdim Number of dimensions of the boundary, `1 <= bdim <= map->ndim`.
 * @param orientation Full element-dimension orientation of the boundary.
 * @return The restricted space map, or NULL with a Python exception set.
 */
FDG_INTERNAL
space_map_object *space_map_boundary_oriented_impl(const interplib_module_state_t *state,
                                                   integration_registry_object *integration_registry,
                                                   const space_map_object *map, unsigned bdim,
                                                   const int8_t *orientation);

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

#endif // FDG_MAPPINGS_H
