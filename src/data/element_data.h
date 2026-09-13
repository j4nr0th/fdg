#ifndef FDG_ELEMENT_DATA_H
#define FDG_ELEMENT_DATA_H

#include "../basis/basis_set.h"
#include "../basis/covector_basis.h"
#include "../common/error.h"
#include "../integration/integration_rules.h"
#include <cutl/allocators.h>
#include <cutl/iterators/combination_iterator.h>
#include <stdint.h>

/**
 * @brief Kind of per-element data stored in an element data collection.
 *
 * The kind is fixed by the first added option and shared by every element
 * of the collection.
 */
typedef enum
{
    ELEMENT_DATA_KIND_INVALID = 0, // No option has been added yet.
    ELEMENT_DATA_KIND_DOF,         // Plain DoFs of a single function space.
    ELEMENT_DATA_KIND_KFORM,       // Component DoFs of a k-form.
    ELEMENT_DATA_KIND_GEOMETRY,    // Coordinate DoFs of a geometry map.
} element_data_kind_t;

/**
 * @brief One entry of the options table: a distinct data specification.
 *
 * This is a tagged union: @p kind selects which payload member is active.
 * All variants share the number of reference dimensions and the specs of
 * the underlying function space; only the kind-specific payload differs.
 *
 * The specs are deep copies owned by the collection; @p value_count is
 * filled in by element_data_add_option and must be zero when the caller
 * passes the option in.
 */
typedef struct
{
    element_data_kind_t kind;  // Which payload member is active.
    unsigned ndim;             // Number of reference dimensions.
    basis_spec_t *basis_specs; // [ndim] Specs of the underlying function space.
    union {
        struct
        {
            unsigned order; // K-form order, in [0, ndim].
        } kform;
        struct
        {
            unsigned coord_count;          // Number of physical coordinates.
            integration_spec_t *int_specs; // [ndim] Integration specs.
        } geometry;
    };
    size_t value_count; // Number of doubles stored per element; set by add_option.
} element_data_option_t;

/**
 * Opaque collection of per-element data.
 *
 * Elements reference one option each through an index into a small options
 * table, and their values are stored back to back in one flat array, with
 * a CSR-style offsets array marking where each element's block starts.
 */
typedef struct element_data_t element_data_t;

/**
 * @brief Create an empty element data collection.
 *
 * @param out Receives the pointer to the new collection on success.
 * @param allocator Allocator used for all memory of the collection,
 *        including the collection object itself.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the collection and must release it with
 * element_data_free.
 */
FDG_INTERNAL
fdg_result_t element_data_create(element_data_t **out, const cutl_allocator_t *allocator);

/**
 * @brief Release an element data collection and all memory it owns.
 *
 * @param data Collection to release; may be NULL, in which case nothing
 *        happens.
 * @param allocator The allocator the collection was created with.
 */
FDG_INTERNAL
void element_data_free(element_data_t *data, const cutl_allocator_t *allocator);

/**
 * @brief Add an option to the options table, or look up an equal one.
 *
 * The option is validated and its specs are deep copied. If an option with
 * equal kind, payload and specification values is already present, its
 * index is returned and nothing is added.
 *
 * The first added option fixes the kind, the number of reference
 * dimensions and, for geometry data, the number of coordinates for the
 * whole collection. Later options that disagree return
 * FDG_ERROR_NOT_IN_DOMAIN, as do options with invalid contents (unknown
 * basis type, k-form order exceeding ndim or zero-order basis axes in a
 * nonzero-order k-form, geometry data without integration specs or
 * without coordinates).
 *
 * On success the option's @p value_count field is set to the number of
 * doubles that elements referencing this option must provide.
 *
 * @param data Collection to add the option to.
 * @param option Option to add; only read, never stored.
 * @param out_index Receives the index of the (possibly existing) equal
 *        option on success.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN for an invalid
 *         or incompatible option, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out_index` is left unmodified.
 */
FDG_INTERNAL
fdg_result_t element_data_add_option(element_data_t *data, const element_data_option_t *option, unsigned *out_index);

/**
 * @brief Append one element with the data of the given option.
 *
 * @param data Collection to append to.
 * @param option_index Index into the options table, in
 *        [0, element_data_option_count(data)).
 * @param values Value block of the element; layout is determined by the
 *        kind: flat DoFs for DOF and KFORM kinds, coordinate-major blocks
 *        of flat DoFs for GEOMETRY.
 * @param count Number of doubles in @p values; must equal
 *        element_data_option_value_count of the referenced option.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if the option
 *         index or the value count is invalid, FDG_ERROR_FAILED_ALLOCATION
 *         if memory allocation fails.
 */
FDG_INTERNAL
fdg_result_t element_data_add_element(element_data_t *data, unsigned option_index, const double values[], size_t count);

/**
 * @brief Overwrite the value block of one existing element.
 *
 * @param data Collection to modify.
 * @param element_id Element to overwrite, in
 *        [0, element_data_element_count(data)).
 * @param values New value block, copied over the element's old block.
 * @param count Number of doubles in @p values; must equal the block size
 *        of the element.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if the element
 *         id or the value count is invalid.
 */
FDG_INTERNAL
fdg_result_t element_data_set_element_values(element_data_t *data, uint64_t element_id, const double values[],
                                             size_t count);

/**
 * @brief Get the kind of data stored in the collection.
 *
 * @param data Collection to query.
 * @return The kind fixed by the first added option, or
 *         ELEMENT_DATA_KIND_INVALID while the collection is empty.
 */
FDG_INTERNAL
element_data_kind_t element_data_kind(const element_data_t *data);

/**
 * @brief Get the number of elements in the collection.
 *
 * @param data Collection to query.
 * @return Number of appended elements.
 */
FDG_INTERNAL
uint64_t element_data_element_count(const element_data_t *data);

/**
 * @brief Get the number of options in the options table.
 *
 * @param data Collection to query.
 * @return Number of distinct options.
 */
FDG_INTERNAL
unsigned element_data_option_count(const element_data_t *data);

/**
 * @brief Get one option of the options table.
 *
 * @param data Collection to query.
 * @param index Option index, in [0, element_data_option_count(data)).
 * @return Pointer to the option, owned by the collection.
 */
FDG_INTERNAL
const element_data_option_t *element_data_option(const element_data_t *data, unsigned index);

/**
 * @brief Get the number of doubles stored per element of an option.
 *
 * @param option Option to query.
 * @return Number of doubles elements referencing this option store.
 */
FDG_INTERNAL
size_t element_data_option_value_count(const element_data_option_t *option);

/**
 * @brief Get the per-element option indices.
 *
 * @param data Collection to query.
 * @return Array of element_data_element_count(data) indices, owned by the
 *         collection.
 */
FDG_INTERNAL
const uint32_t *element_data_element_options(const element_data_t *data);

/**
 * @brief Get the CSR offsets of the per-element value blocks.
 *
 * @param data Collection to query.
 * @return Array of element_data_element_count(data) + 1 offsets; block of
 *         element `e` is `[offsets[e], offsets[e + 1])`. Owned by the
 *         collection.
 */
FDG_INTERNAL
const uint64_t *element_data_offsets(const element_data_t *data);

/**
 * @brief Get the flat array of all element values.
 *
 * @param data Collection to query.
 * @return Array of element_data_value_count(data) doubles, owned by the
 *         collection.
 */
FDG_INTERNAL
double *element_data_values(element_data_t *data);

/**
 * @brief Get the total number of stored doubles.
 *
 * @param data Collection to query.
 * @return Number of doubles in the flat values array.
 */
FDG_INTERNAL
size_t element_data_value_count(const element_data_t *data);

#endif // FDG_ELEMENT_DATA_H
