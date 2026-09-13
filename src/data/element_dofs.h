#ifndef FDG_ELEMENT_DOFS_H
#define FDG_ELEMENT_DOFS_H

#include "element_data.h"

/**
 * Collection of per-element degrees of freedom. A type-safe alias of
 * @ref element_data_t restricted to ELEMENT_DATA_KIND_DOF.
 */
typedef element_data_t element_dofs_t;

/**
 * @brief Create an empty degrees-of-freedom collection.
 *
 * @param out Receives the pointer to the new collection on success.
 * @param allocator Allocator used for all memory of the collection.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the collection and must release it with
 * element_dofs_free.
 */
FDG_INTERNAL
fdg_result_t element_dofs_create(element_dofs_t **out, const cutl_allocator_t *allocator);

/**
 * @brief Release a degrees-of-freedom collection and all memory it owns.
 *
 * @param dofs Collection to release; may be NULL.
 * @param allocator The allocator the collection was created with.
 */
FDG_INTERNAL
void element_dofs_free(element_dofs_t *dofs, const cutl_allocator_t *allocator);

/**
 * @brief Add a function-space option to the options table, or look up an
 *        equal one.
 *
 * The first added option fixes ndim for the whole collection. Options with
 * a different ndim or invalid basis types are rejected with
 * FDG_ERROR_NOT_IN_DOMAIN.
 *
 * @param dofs Collection to add the option to.
 * @param ndim Number of reference dimensions.
 * @param basis_specs [ndim] specs of the function space.
 * @param out_index Receives the index of the (possibly existing) equal
 *        option on success.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN for an invalid
 *         or incompatible option, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out_index` is left unmodified.
 */
FDG_INTERNAL
fdg_result_t element_dofs_add_option(element_dofs_t *dofs, unsigned ndim, const basis_spec_t basis_specs[static ndim],
                                     unsigned *out_index);

/**
 * @brief Append one element with the DoFs of the given option.
 *
 * @param dofs Collection to append to.
 * @param option_index Index into the options table, in
 *        [0, element_dofs_option_count(dofs)).
 * @param values Flat DoF values; must hold
 *        element_dofs_option_value_count(dofs, option_index) doubles.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if the option
 *         index is invalid, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails.
 */
FDG_INTERNAL
fdg_result_t element_dofs_add_element(element_dofs_t *dofs, unsigned option_index, const double values[]);

/**
 * @brief Overwrite the value block of one existing element.
 *
 * @param dofs Collection to modify.
 * @param element_id Element to overwrite, in
 *        [0, element_dofs_element_count(dofs)).
 * @param values New value block, copied over the element's old block.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if the element
 *         id is invalid.
 */
FDG_INTERNAL
fdg_result_t element_dofs_set_element_values(element_dofs_t *dofs, uint64_t element_id, const double values[]);

/**
 * @brief Get the number of elements in the collection.
 *
 * @param dofs Collection to query.
 * @return Number of appended elements.
 */
FDG_INTERNAL
uint64_t element_dofs_element_count(const element_dofs_t *dofs);

/**
 * @brief Get the number of options in the options table.
 *
 * @param dofs Collection to query.
 * @return Number of distinct options.
 */
FDG_INTERNAL
unsigned element_dofs_option_count(const element_dofs_t *dofs);

/**
 * @brief Get one option of the options table.
 *
 * @param dofs Collection to query.
 * @param index Option index, in [0, element_dofs_option_count(dofs)).
 * @return Pointer to the option, owned by the collection.
 */
FDG_INTERNAL
const element_data_option_t *element_dofs_option(const element_dofs_t *dofs, unsigned index);

/**
 * @brief Get the number of doubles stored per element of an option.
 *
 * @param dofs Collection to query.
 * @param index Option index, in [0, element_dofs_option_count(dofs)).
 * @return Number of doubles elements referencing this option store.
 */
FDG_INTERNAL
size_t element_dofs_option_value_count(const element_dofs_t *dofs, unsigned index);

/**
 * @brief Get the per-element option indices.
 *
 * @param dofs Collection to query.
 * @return Array of element_dofs_element_count(dofs) indices, owned by the
 *         collection.
 */
FDG_INTERNAL
const uint32_t *element_dofs_element_options(const element_dofs_t *dofs);

/**
 * @brief Get the CSR offsets of the per-element value blocks.
 *
 * @param dofs Collection to query.
 * @return Array of element_dofs_element_count(dofs) + 1 offsets; block of
 *         element `e` is `[offsets[e], offsets[e + 1])`. Owned by the
 *         collection.
 */
FDG_INTERNAL
const uint64_t *element_dofs_offsets(const element_dofs_t *dofs);

/**
 * @brief Get the flat array of all element values.
 *
 * @param dofs Collection to query.
 * @return Array of element_dofs_value_count(dofs) doubles, owned by the
 *         collection.
 */
FDG_INTERNAL
double *element_dofs_values(element_dofs_t *dofs);

/**
 * @brief Get the total number of stored doubles.
 *
 * @param dofs Collection to query.
 * @return Number of doubles in the flat values array.
 */
FDG_INTERNAL
size_t element_dofs_value_count(const element_dofs_t *dofs);

#endif // FDG_ELEMENT_DOFS_H
