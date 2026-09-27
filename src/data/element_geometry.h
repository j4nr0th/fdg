#ifndef FDG_ELEMENT_GEOMETRY_H
#define FDG_ELEMENT_GEOMETRY_H

#include "element_data.h"

/**
 * Collection of per-element geometry data: coordinate DoFs of a space map
 * for every element. A type-safe alias of @ref element_data_t restricted
 * to ELEMENT_DATA_KIND_GEOMETRY.
 */
typedef element_data_t element_geometry_t;

/**
 * @brief Create an empty geometry collection.
 *
 * @param out Receives the pointer to the new collection on success.
 * @param allocator Allocator used for all memory of the collection.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the collection and must release it with
 * element_geometry_free.
 */
FDG_INTERNAL
fdg_result_t element_geometry_create(element_geometry_t **out, const cutl_allocator_t *allocator);

/**
 * @brief Release a geometry collection and all memory it owns.
 *
 * @param geometry Collection to release; may be NULL.
 * @param allocator The allocator the collection was created with.
 */
FDG_INTERNAL
void element_geometry_free(element_geometry_t *geometry, const cutl_allocator_t *allocator);

/**
 * @brief Add a geometry option to the options table, or look up an equal one.
 *
 * The first added option fixes ndim and coord_count for the whole
 * collection. An option with a different ndim or coord_count is rejected
 * with FDG_ERROR_NOT_IN_DOMAIN.
 *
 * The arguments are preconditions, not values to validate: a dimension
 * outside [1, 63], a basis spec whose family is invalid, a zero coordinate
 * count or a null spec array aborts through CUTL_ASSERT. A caller that
 * handles untrusted input must check these conditions itself and report
 * them instead of relying on this function.
 *
 * @param geometry Collection to add the option to.
 * @param ndim Number of reference dimensions, in [1, 63].
 * @param coord_count Number of physical coordinates, at least one.
 * @param basis_specs [ndim] specs of the shared geometry function space,
 *        with valid basis families.
 * @param int_specs [ndim] integration specs of the option.
 * @param out_index Receives the index of the (possibly existing) equal
 *        option on success.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if ndim or
 *         coord_count differ from the values fixed by the first option,
 *         FDG_ERROR_FAILED_ALLOCATION if memory allocation fails. On
 *         failure, `*out_index` is left unmodified.
 */
FDG_INTERNAL
fdg_result_t element_geometry_add_option(element_geometry_t *geometry, unsigned ndim, unsigned coord_count,
                                         const basis_spec_t basis_specs[static ndim],
                                         const integration_spec_t int_specs[static ndim], unsigned *out_index);

/**
 * @brief Append one element with the coordinate DoFs of the given option.
 *
 * The arguments are preconditions, not values to validate: an option index
 * outside [0, element_geometry_option_count(geometry)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param geometry Collection to append to.
 * @param option_index Index into the options table, in
 *        [0, element_geometry_option_count(geometry)).
 * @param values Coordinate-major blocks of tensor-order DoFs; must hold
 *        element_geometry_option_value_count(geometry, option_index)
 *        doubles.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails.
 */
FDG_INTERNAL
fdg_result_t element_geometry_add_element(element_geometry_t *geometry, unsigned option_index, const double values[]);

/**
 * @brief Overwrite the value block of one existing element.
 *
 * The arguments are preconditions, not values to validate: an element id
 * outside [0, element_geometry_element_count(geometry)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param geometry Collection to modify.
 * @param element_id Element to overwrite, in
 *        [0, element_geometry_element_count(geometry)).
 * @param values New value block, copied over the element's old block; must
 *        hold as many doubles as the element's block.
 */
FDG_INTERNAL
void element_geometry_set_element_values(element_geometry_t *geometry, uint64_t element_id, const double values[]);

/**
 * @brief Get the number of elements in the collection.
 *
 * @param geometry Collection to query.
 * @return Number of appended elements.
 */
FDG_INTERNAL
uint64_t element_geometry_element_count(const element_geometry_t *geometry);

/**
 * @brief Get the number of options in the options table.
 *
 * @param geometry Collection to query.
 * @return Number of distinct options.
 */
FDG_INTERNAL
unsigned element_geometry_option_count(const element_geometry_t *geometry);

/**
 * @brief Get one option of the options table.
 *
 * @param geometry Collection to query.
 * @param index Option index, in [0, element_geometry_option_count(geometry)).
 * @return Pointer to the option, owned by the collection.
 */
FDG_INTERNAL
const element_data_option_t *element_geometry_option(const element_geometry_t *geometry, unsigned index);

/**
 * @brief Get the number of doubles stored per element of an option.
 *
 * @param geometry Collection to query.
 * @param index Option index, in [0, element_geometry_option_count(geometry)).
 * @return Number of doubles elements referencing this option store.
 */
FDG_INTERNAL
size_t element_geometry_option_value_count(const element_geometry_t *geometry, unsigned index);

/**
 * @brief Get the per-element option indices.
 *
 * @param geometry Collection to query.
 * @return Array of element_geometry_element_count(geometry) indices, owned
 *         by the collection.
 */
FDG_INTERNAL
const uint32_t *element_geometry_element_options(const element_geometry_t *geometry);

/**
 * @brief Get the CSR offsets of the per-element value blocks.
 *
 * @param geometry Collection to query.
 * @return Array of element_geometry_element_count(geometry) + 1 offsets;
 *         block of element `e` is `[offsets[e], offsets[e + 1])`. Owned by
 *         the collection.
 */
FDG_INTERNAL
const uint64_t *element_geometry_offsets(const element_geometry_t *geometry);

/**
 * @brief Get the flat array of all element values.
 *
 * @param geometry Collection to query.
 * @return Array of element_geometry_value_count(geometry) doubles, owned
 *         by the collection.
 */
FDG_INTERNAL
double *element_geometry_values(element_geometry_t *geometry);

/**
 * @brief Get the total number of stored doubles.
 *
 * @param geometry Collection to query.
 * @return Number of doubles in the flat values array.
 */
FDG_INTERNAL
size_t element_geometry_value_count(const element_geometry_t *geometry);

#endif // FDG_ELEMENT_GEOMETRY_H
