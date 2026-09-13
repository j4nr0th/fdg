#ifndef FDG_ELEMENT_KFORMS_H
#define FDG_ELEMENT_KFORMS_H

#include "element_data.h"

/**
 * Opaque collection of per-element k-form data: a fixed set of labeled
 * k-form fields, with the field values of every element grouped together.
 *
 * All fields of one element are derived from one shared base function
 * space. The distinct base spaces form the options of the collection;
 * every element references one of them. The labels and their k-form
 * orders are fixed when the fields are added, the base spaces when they
 * are added; every element then stores the values of all fields, in
 * field order. The setup order is: all fields first, then all base
 * spaces, then elements.
 */
typedef struct element_kforms_t element_kforms_t;

/**
 * @brief Create an empty k-form collection.
 *
 * @param out Receives the pointer to the new collection on success.
 * @param allocator Allocator used for all memory of the collection,
 *        including owned label copies.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the collection and must release it with
 * element_kforms_free.
 */
FDG_INTERNAL
fdg_result_t element_kforms_create(element_kforms_t **out, const cutl_allocator_t *allocator);

/**
 * @brief Release a k-form collection and all memory it owns.
 *
 * @param kforms Collection to release; may be NULL.
 * @param allocator The allocator the collection was created with.
 */
FDG_INTERNAL
void element_kforms_free(element_kforms_t *kforms, const cutl_allocator_t *allocator);

/**
 * @brief Add a labeled k-form field.
 *
 * The number of reference dimensions is fixed by the first field and must
 * be shared by all fields. Fields must be added before any base space;
 * adding a field after a base space fails. Labels must be unique within
 * the collection.
 *
 * @param kforms Collection to add the field to.
 * @param label Null-terminated label of the field; copied.
 * @param ndim Number of reference dimensions; must match previously added
 *        fields.
 * @param order Order of the k-form field; must not exceed ndim.
 * @param out_field Receives the index of the new field on success; may be
 *        NULL when the index is not needed.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN for an empty or
 *         duplicate label, an order exceeding ndim, an ndim mismatch with
 *         previous fields, or a field added after a base space,
 *         FDG_ERROR_FAILED_ALLOCATION if memory allocation fails. On
 *         failure, `*out_field` is left unmodified.
 */
FDG_INTERNAL
fdg_result_t element_kforms_add_field(element_kforms_t *kforms, const char *label, unsigned ndim, unsigned order,
                                      unsigned *out_field);

/**
 * @brief Find a field by its label.
 *
 * @param kforms Collection to query.
 * @param label Label to look up.
 * @param out_field Receives the index of the field on success.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if no field with
 *         this label exists.
 */
FDG_INTERNAL
fdg_result_t element_kforms_find_field(const element_kforms_t *kforms, const char *label, unsigned *out_field);

/**
 * @brief Add a base function space, or look up an equal one.
 *
 * The space is validated and deep copied into every field's store; all
 * stores keep the same options in the same order. A space whose basis has
 * a zero-order axis is rejected when any field has a nonzero order, since
 * no k-form of nonzero order can be derived from such a space. At least
 * one field must be added before the first base space.
 *
 * @param kforms Collection to add the space to.
 * @param basis_specs [ndim] specs of the base function space; ndim is the
 *        collection's number of reference dimensions.
 * @param out_index Receives the index of the (possibly existing) equal
 *        space on success.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN for an invalid
 *         space or a space added before any field, FDG_ERROR_FAILED_ALLOCATION
 *         if memory allocation fails. A failing allocation can leave the
 *         per-field stores with mismatched options; the collection must
 *         then no longer be used. On other failures, `*out_index` is left
 *         unmodified.
 */
FDG_INTERNAL
fdg_result_t element_kforms_add_space(element_kforms_t *kforms, const basis_spec_t *basis_specs, unsigned *out_index);

/**
 * @brief Append one element holding the values of all fields.
 *
 * @param kforms Collection to append to.
 * @param space_index Index of the element's base space, in
 *        [0, element_kforms_space_count(kforms)).
 * @param values Field-major values of the element, field 0 first; must
 *        hold element_kforms_element_value_count(kforms, space_index)
 *        doubles in field order.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if the space
 *         index is invalid, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. A failing allocation can leave the per-field
 *         stores with mismatched element counts; the collection must then
 *         no longer be used.
 */
FDG_INTERNAL
fdg_result_t element_kforms_add_element(element_kforms_t *kforms, unsigned space_index, const double values[]);

/**
 * @brief Overwrite the values of one field of one existing element.
 *
 * @param kforms Collection to modify.
 * @param element_id Element to overwrite, in
 *        [0, element_kforms_element_count(kforms)).
 * @param field Field index, in [0, element_kforms_field_count(kforms)).
 * @param values New field values, copied over the old block; must hold as
 *        many doubles as the field's block of this element.
 * @return FDG_SUCCESS on success, FDG_ERROR_NOT_IN_DOMAIN if the element
 *         id or the field index is invalid.
 */
FDG_INTERNAL
fdg_result_t element_kforms_set_field_values(element_kforms_t *kforms, uint64_t element_id, unsigned field,
                                             const double values[]);

/**
 * @brief Get the number of k-form fields.
 *
 * @param kforms Collection to query.
 * @return Number of labeled fields.
 */
FDG_INTERNAL
unsigned element_kforms_field_count(const element_kforms_t *kforms);

/**
 * @brief Get the label of one field.
 *
 * @param kforms Collection to query.
 * @param field Field index, in [0, element_kforms_field_count(kforms)).
 * @return Null-terminated label, owned by the collection.
 */
FDG_INTERNAL
const char *element_kforms_field_label(const element_kforms_t *kforms, unsigned field);

/**
 * @brief Get the number of reference dimensions of the collection.
 *
 * @param kforms Collection to query.
 * @return Number of reference dimensions; zero while no field is added.
 */
FDG_INTERNAL
unsigned element_kforms_ndim(const element_kforms_t *kforms);

/**
 * @brief Get the k-form order of one field.
 *
 * @param kforms Collection to query.
 * @param field Field index, in [0, element_kforms_field_count(kforms)).
 * @return Order of the k-form field.
 */
FDG_INTERNAL
unsigned element_kforms_field_order(const element_kforms_t *kforms, unsigned field);

/**
 * @brief Get the number of distinct base function spaces.
 *
 * @param kforms Collection to query.
 * @return Number of options in the shared space table.
 */
FDG_INTERNAL
unsigned element_kforms_space_count(const element_kforms_t *kforms);

/**
 * @brief Get one base space option of the shared space table.
 *
 * @param kforms Collection to query.
 * @param index Space index, in [0, element_kforms_space_count(kforms)).
 * @return Pointer to the option; its basis_specs are the base space, its
 *         payload order the order of field 0. Owned by the collection.
 */
FDG_INTERNAL
const element_data_option_t *element_kforms_space_option(const element_kforms_t *kforms, unsigned index);

/**
 * @brief Get the base space index of one element.
 *
 * @param kforms Collection to query.
 * @param element_id Element to query, in
 *        [0, element_kforms_element_count(kforms)).
 * @return Index into the shared space table.
 */
FDG_INTERNAL
unsigned element_kforms_element_space(const element_kforms_t *kforms, uint64_t element_id);

/**
 * @brief Get the number of doubles stored per element for one base space.
 *
 * @param kforms Collection to query.
 * @param space_index Space index, in [0, element_kforms_space_count(kforms)).
 * @return Sum over all fields of the doubles the field stores for an
 *         element on this base space.
 */
FDG_INTERNAL
size_t element_kforms_element_value_count(const element_kforms_t *kforms, unsigned space_index);

/**
 * @brief Get the number of elements in the collection.
 *
 * @param kforms Collection to query.
 * @return Number of appended elements.
 */
FDG_INTERNAL
uint64_t element_kforms_element_count(const element_kforms_t *kforms);

/**
 * @brief Get the flat value array of one field.
 *
 * @param kforms Collection to query.
 * @param field Field index, in [0, element_kforms_field_count(kforms)).
 * @return Array with one block of doubles per element, laid out per
 *         element_kforms_field_offsets(kforms, field). Owned by the
 *         collection.
 */
FDG_INTERNAL
double *element_kforms_field_values(element_kforms_t *kforms, unsigned field);

/**
 * @brief Get the CSR offsets of one field's per-element value blocks.
 *
 * @param kforms Collection to query.
 * @param field Field index, in [0, element_kforms_field_count(kforms)).
 * @return Array of element_kforms_element_count(kforms) + 1 offsets; block
 *         of element `e` is `[offsets[e], offsets[e + 1])`. Owned by the
 *         collection.
 */
FDG_INTERNAL
const uint64_t *element_kforms_field_offsets(const element_kforms_t *kforms, unsigned field);

#endif // FDG_ELEMENT_KFORMS_H
