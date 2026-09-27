#ifndef FDG_ELEMENT_KFORMS_H
#define FDG_ELEMENT_KFORMS_H

#include "mesh_kform_specs.h"

/**
 * Opaque collection of per-element k-form values: the values of a fixed
 * set of labeled k-form fields, grouped per element.
 *
 * The collection borrows a mesh_kform_specs_t that defines its structure:
 * the labeled fields, the base function spaces, and the base space of
 * every element. The specs must outlive the collection and are never
 * freed by it; borrowing freezes them against further mutation. The value
 * storage is allocated once at creation, sized by the specs, and
 * zero-filled; elements are then filled in order with
 * element_kforms_add_element until the cursor reaches the element count
 * of the specs.
 */
typedef struct element_kforms_t element_kforms_t;

/**
 * @brief Create a k-form value collection for a structure.
 *
 * All storage is allocated at its final size and zero-filled; nothing is
 * ever reallocated. The per-field offsets are built by walking the
 * per-element base space indices of the specs.
 *
 * The arguments are preconditions, not values to validate: a null specs
 * pointer or a structure without fields aborts through CUTL_ASSERT. A
 * caller that handles untrusted input must check these conditions itself
 * and report them instead of relying on this function.
 *
 * @param out Receives the pointer to the new collection on success.
 * @param specs Structure defining the collection; borrowed, must outlive
 *        the collection.
 * @param allocator Allocator used for all memory of the collection.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the collection and must release it with
 * element_kforms_free.
 */
FDG_INTERNAL
fdg_result_t element_kforms_create(element_kforms_t **out, const mesh_kform_specs_t *specs,
                                   const cutl_allocator_t *allocator);

/**
 * @brief Release a k-form value collection and all memory it owns.
 *
 * The borrowed specs are not freed.
 *
 * @param kforms Collection to release; may be NULL.
 * @param allocator The allocator the collection was created with.
 */
FDG_INTERNAL
void element_kforms_free(element_kforms_t *kforms, const cutl_allocator_t *allocator);

/**
 * @brief Fill the next unfilled element with the values of all fields.
 *
 * The arguments are preconditions, not values to validate: a cursor at or
 * beyond mesh_kform_specs_element_count of the specs, or a value block
 * smaller than element_kforms_field_values of the element aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check these
 * conditions itself and report them instead of relying on this function.
 *
 * @param kforms Collection to fill.
 * @param values Field-major values of the element, field 0 first; must
 *        hold element_kforms_element_value_count of the specs for the
 *        element's base space, in field order.
 */
FDG_INTERNAL
void element_kforms_add_element(element_kforms_t *kforms, const double values[]);

/**
 * @brief Overwrite the values of one field of one existing element.
 *
 * The arguments are preconditions, not values to validate: a field index
 * outside [0, mesh_kform_specs_field_count) of the specs, an element id
 * outside [0, mesh_kform_specs_element_count) of the specs, or a value
 * block of the wrong size aborts through CUTL_ASSERT. A caller that
 * handles untrusted input must check these conditions itself and report
 * them instead of relying on this function.
 *
 * @param kforms Collection to modify.
 * @param element_id Element to overwrite, in
 *        [0, mesh_kform_specs_element_count) of the specs.
 * @param field Field index, in [0, mesh_kform_specs_field_count) of the
 *        specs.
 * @param values New field values, copied over the old block; must hold as
 *        many doubles as the field's block of this element.
 */
FDG_INTERNAL
void element_kforms_set_field_values(element_kforms_t *kforms, uint64_t element_id, unsigned field,
                                     const double values[]);

/**
 * @brief Get the structure the collection borrows.
 *
 * @param kforms Collection to query.
 * @return The borrowed structure; owned by the caller of
 *         element_kforms_create.
 */
FDG_INTERNAL
const mesh_kform_specs_t *element_kforms_specs(const element_kforms_t *kforms);

/**
 * @brief Get the position of the append cursor.
 *
 * @param kforms Collection to query.
 * @return Number of elements filled so far; at most
 *         mesh_kform_specs_element_count of the specs.
 */
FDG_INTERNAL
uint64_t element_kforms_filled_count(const element_kforms_t *kforms);

/**
 * @brief Get the flat value array of one field.
 *
 * The arguments are preconditions, not values to validate: a field index
 * outside [0, mesh_kform_specs_field_count) of the specs aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param kforms Collection to query.
 * @param field Field index, in [0, mesh_kform_specs_field_count) of the
 *        specs.
 * @return Array with one block of doubles per element, laid out per
 *         element_kforms_field_offsets(kforms, field). Owned by the
 *         collection.
 */
FDG_INTERNAL
double *element_kforms_field_values(element_kforms_t *kforms, unsigned field);

/**
 * @brief Get the CSR offsets of one field's per-element value blocks.
 *
 * The arguments are preconditions, not values to validate: a field index
 * outside [0, mesh_kform_specs_field_count) of the specs aborts through
 * CUTL_ASSERT. A caller that handles untrusted input must check this
 * condition itself and report it instead of relying on this function.
 *
 * @param kforms Collection to query.
 * @param field Field index, in [0, mesh_kform_specs_field_count) of the
 *        specs.
 * @return Array of mesh_kform_specs_element_count(specs) + 1 offsets;
 *         block of element `e` is `[offsets[e], offsets[e + 1])`. Owned by
 *         the collection.
 */
FDG_INTERNAL
const uint64_t *element_kforms_field_offsets(const element_kforms_t *kforms, unsigned field);

#endif // FDG_ELEMENT_KFORMS_H
