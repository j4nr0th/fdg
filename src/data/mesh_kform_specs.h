#ifndef FDG_MESH_KFORM_SPECS_H
#define FDG_MESH_KFORM_SPECS_H

#include "../basis/basis_set.h"
#include "../common/error.h"
#include <cutl/allocators.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * Opaque structure of a batched k-form collection: a fixed set of labeled
 * k-form fields and the deduplicated table of base function spaces their
 * values are derived from, plus one base space index per element.
 *
 * The structure holds no values; it only describes how much storage each
 * element needs and where it lives. The labels and their k-form orders are
 * fixed when the fields are added, the base spaces when they are added,
 * and every element then references one of the base spaces. The setup
 * order is: all fields first, then all base spaces, then elements.
 *
 * Once a values collection borrows the structure, the structure is frozen:
 * every mutator then aborts through CUTL_ASSERT. A caller that hands the
 * structure to a values collection must stop mutating it; the Python
 * bindings report this as a ValueError instead.
 */
typedef struct mesh_kform_specs_t mesh_kform_specs_t;

/**
 * @brief Create an empty k-form structure.
 *
 * @param out Receives the pointer to the new structure on success.
 * @param allocator Allocator used for all memory of the structure,
 *        including owned label copies.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the structure and must release it with
 * mesh_kform_specs_free.
 */
FDG_INTERNAL
fdg_result_t mesh_kform_specs_create(mesh_kform_specs_t **out, const cutl_allocator_t *allocator);

/**
 * @brief Release a k-form structure and all memory it owns.
 *
 * @param specs Structure to release; may be NULL.
 * @param allocator The allocator the structure was created with.
 */
FDG_INTERNAL
void mesh_kform_specs_free(mesh_kform_specs_t *specs, const cutl_allocator_t *allocator);

/**
 * @brief Add a labeled k-form field.
 *
 * The number of reference dimensions is fixed by the first field and must
 * be shared by all fields. Fields must be added before any base space;
 * labels must be unique within the structure.
 *
 * The arguments are preconditions, not values to validate: a null or empty
 * label, a duplicate label, a dimension below one, a dimension that differs
 * from the one fixed by previous fields, an order above ndim, a field
 * added after a base space or element, or a frozen structure aborts
 * through CUTL_ASSERT. A caller that handles untrusted input must check
 * these conditions itself and report them instead of relying on this
 * function.
 *
 * @param specs Structure to add the field to.
 * @param label Null-terminated label of the field; copied.
 * @param ndim Number of reference dimensions, at least one; must match
 *        previously added fields.
 * @param order Order of the k-form field, in [0, ndim].
 * @param out_field Receives the index of the new field on success; may be
 *        NULL when the index is not needed.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out_field` is left unmodified.
 */
FDG_INTERNAL
fdg_result_t mesh_kform_specs_add_field(mesh_kform_specs_t *specs, const char *label, unsigned ndim, unsigned order,
                                        unsigned *out_field);

/**
 * @brief Find a field by its label.
 *
 * This is a query, not a fallible operation: a label that names no field
 * simply yields false.
 *
 * @param specs Structure to query.
 * @param label Label to look up; a null label finds nothing.
 * @param out_field Receives the index of the field when it is found; left
 *        unmodified otherwise.
 * @return true when a field with this label exists, false otherwise.
 */
FDG_INTERNAL
bool mesh_kform_specs_find_field(const mesh_kform_specs_t *specs, const char *label, unsigned *out_field);

/**
 * @brief Add a base function space, or look up an equal one.
 *
 * The space is deep copied into the structure's space table; a space equal
 * to one already in the table reuses its index, so every distinct base
 * space is stored once no matter how many fields derive from it. At least
 * one field must be added before the first base space; the field count is
 * final from then on.
 *
 * The value count of every field on the new space is derived here, so the
 * per-element storage size of a values collection is fixed by the
 * structure.
 *
 * The arguments are preconditions, not values to validate: a space added
 * before any field, a basis axis whose family is invalid, or a zero-order
 * axis under a field of nonzero order aborts through CUTL_ASSERT. A caller
 * that handles untrusted input must check these conditions itself and
 * report them instead of relying on this function.
 *
 * @param specs Structure to add the space to.
 * @param basis_specs [ndim] specs of the base function space, with valid
 *        basis families; ndim is the structure's number of reference
 *        dimensions.
 * @param out_index Receives the index of the (possibly existing) equal
 *        space on success.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out_index` is left unmodified.
 */
FDG_INTERNAL
fdg_result_t mesh_kform_specs_add_space(mesh_kform_specs_t *specs, const basis_spec_t basis_specs[],
                                        unsigned *out_index);

/**
 * @brief Append one element referencing a base space.
 *
 * The arguments are preconditions, not values to validate: a structure
 * without fields or a space index outside
 * [0, mesh_kform_specs_space_count(specs)) aborts through CUTL_ASSERT. A
 * caller that handles untrusted input must check these conditions itself
 * and report them instead of relying on this function.
 *
 * @param specs Structure to append to.
 * @param space_index Index of the element's base space, in
 *        [0, mesh_kform_specs_space_count(specs)).
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails.
 */
FDG_INTERNAL
fdg_result_t mesh_kform_specs_add_element(mesh_kform_specs_t *specs, unsigned space_index);

/**
 * @brief Freeze the structure against further mutation.
 *
 * Called when a values collection borrows the structure. Freezing is
 * permanent; it cannot be undone.
 *
 * @param specs Structure to freeze.
 */
FDG_INTERNAL
void mesh_kform_specs_freeze(mesh_kform_specs_t *specs);

/**
 * @brief Report whether the structure is frozen.
 *
 * @param specs Structure to query.
 * @return true once mesh_kform_specs_freeze was called, false before.
 */
FDG_INTERNAL
bool mesh_kform_specs_is_frozen(const mesh_kform_specs_t *specs);

/**
 * @brief Get the number of k-form fields.
 *
 * @param specs Structure to query.
 * @return Number of labeled fields.
 */
FDG_INTERNAL
unsigned mesh_kform_specs_field_count(const mesh_kform_specs_t *specs);

/**
 * @brief Get the label of one field.
 *
 * The arguments are preconditions, not values to validate: a field index
 * outside [0, mesh_kform_specs_field_count(specs)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param specs Structure to query.
 * @param field Field index, in [0, mesh_kform_specs_field_count(specs)).
 * @return Null-terminated label, owned by the structure.
 */
FDG_INTERNAL
const char *mesh_kform_specs_field_label(const mesh_kform_specs_t *specs, unsigned field);

/**
 * @brief Get the k-form order of one field.
 *
 * The arguments are preconditions, not values to validate: a field index
 * outside [0, mesh_kform_specs_field_count(specs)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param specs Structure to query.
 * @param field Field index, in [0, mesh_kform_specs_field_count(specs)).
 * @return Order of the k-form field.
 */
FDG_INTERNAL
unsigned mesh_kform_specs_field_order(const mesh_kform_specs_t *specs, unsigned field);

/**
 * @brief Get the number of reference dimensions of the structure.
 *
 * @param specs Structure to query.
 * @return Number of reference dimensions; zero while no field is added.
 */
FDG_INTERNAL
unsigned mesh_kform_specs_ndim(const mesh_kform_specs_t *specs);

/**
 * @brief Get the number of distinct base function spaces.
 *
 * @param specs Structure to query.
 * @return Number of spaces in the space table.
 */
FDG_INTERNAL
unsigned mesh_kform_specs_space_count(const mesh_kform_specs_t *specs);

/**
 * @brief Get the basis specs of one base space.
 *
 * The arguments are preconditions, not values to validate: a space index
 * outside [0, mesh_kform_specs_space_count(specs)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param specs Structure to query.
 * @param index Space index, in [0, mesh_kform_specs_space_count(specs)).
 * @return Array of mesh_kform_specs_ndim(specs) basis specs, owned by the
 *         structure.
 */
FDG_INTERNAL
const basis_spec_t *mesh_kform_specs_space_basis_specs(const mesh_kform_specs_t *specs, unsigned index);

/**
 * @brief Get the number of doubles one field stores for an element on one
 *        base space.
 *
 * The arguments are preconditions, not values to validate: a space index
 * outside [0, mesh_kform_specs_space_count(specs)) or a field index
 * outside [0, mesh_kform_specs_field_count(specs)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check these
 * conditions itself and report them instead of relying on this function.
 *
 * @param specs Structure to query.
 * @param space_index Space index, in [0, mesh_kform_specs_space_count(specs)).
 * @param field Field index, in [0, mesh_kform_specs_field_count(specs)).
 * @return Doubles the field stores for an element on this base space.
 */
FDG_INTERNAL
size_t mesh_kform_specs_space_value_count(const mesh_kform_specs_t *specs, unsigned space_index, unsigned field);

/**
 * @brief Get the base space index of one element.
 *
 * The arguments are preconditions, not values to validate: an element id
 * outside [0, mesh_kform_specs_element_count(specs)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param specs Structure to query.
 * @param element_id Element to query, in
 *        [0, mesh_kform_specs_element_count(specs)).
 * @return Index into the space table.
 */
FDG_INTERNAL
unsigned mesh_kform_specs_element_space(const mesh_kform_specs_t *specs, uint64_t element_id);

/**
 * @brief Get the number of elements in the structure.
 *
 * @param specs Structure to query.
 * @return Number of appended elements.
 */
FDG_INTERNAL
uint64_t mesh_kform_specs_element_count(const mesh_kform_specs_t *specs);

/**
 * @brief Get the number of doubles stored per element for one base space.
 *
 * The arguments are preconditions, not values to validate: a space index
 * outside [0, mesh_kform_specs_space_count(specs)) aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param specs Structure to query.
 * @param space_index Space index, in [0, mesh_kform_specs_space_count(specs)).
 * @return Sum over all fields of the doubles the field stores for an
 *         element on this base space.
 */
FDG_INTERNAL
size_t mesh_kform_specs_element_value_count(const mesh_kform_specs_t *specs, unsigned space_index);

#endif // FDG_MESH_KFORM_SPECS_H
