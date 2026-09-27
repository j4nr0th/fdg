#include "element_kforms.h"

#include <string.h>

typedef struct
{
    char *label;
    unsigned order;        // K-form order of the field.
    element_data_t *store; // Options are the base spaces; per-element index selects one.
} element_kforms_field_t;

struct element_kforms_t
{
    const cutl_allocator_t *allocator;
    unsigned ndim; // 0 until the first field fixes it.
    unsigned field_count;
    unsigned field_capacity;
    element_kforms_field_t *fields;
};

fdg_result_t element_kforms_create(element_kforms_t **out, const cutl_allocator_t *allocator)
{
    element_kforms_t *const kforms = allocator->allocate(allocator->state, sizeof(*kforms));
    if (!kforms)
        return FDG_ERROR_FAILED_ALLOCATION;
    *kforms = (element_kforms_t){.allocator = allocator};
    *out = kforms;
    return FDG_SUCCESS;
}

void element_kforms_free(element_kforms_t *kforms, const cutl_allocator_t *allocator)
{
    if (!kforms)
        return;
    for (unsigned i = 0; i < kforms->field_count; ++i)
    {
        allocator->deallocate(allocator->state, kforms->fields[i].label);
        element_data_free(kforms->fields[i].store, allocator);
    }
    allocator->deallocate(allocator->state, kforms->fields);
    allocator->deallocate(allocator->state, kforms);
}

static unsigned element_kforms_space_count_impl(const element_kforms_t *kforms)
{
    return kforms->field_count > 0 ? element_data_option_count(kforms->fields[0].store) : 0;
}

fdg_result_t element_kforms_add_field(element_kforms_t *kforms, const char *label, const unsigned ndim,
                                      const unsigned order, unsigned *out_field)
{
    const unsigned space_count = element_kforms_space_count_impl(kforms);
    CUTL_ASSERT(element_kforms_element_count(kforms) == 0 && space_count == 0,
                "Cannot add the field %s: fields must be added before any base space or element.", label ? label : "");
    CUTL_ASSERT(label != NULL, "Field label must not be null.");
    CUTL_ASSERT(label == NULL || label[0] != '\0', "Field label must not be empty.");
    CUTL_ASSERT(ndim >= 1, "Field dimension %u must be positive.", ndim);
    CUTL_ASSERT(order <= ndim, "Field order %u is not in [0, %u].", order, ndim);
    CUTL_ASSERT(kforms->ndim == 0 || kforms->ndim == ndim,
                "Field dimension %u does not match the collection dimension %u.", ndim, kforms->ndim);
    for (unsigned i = 0; i < kforms->field_count; ++i)
    {
        CUTL_ASSERT(strcmp(kforms->fields[i].label, label) != 0, "A field labeled %s already exists.", label);
    }

    if (kforms->field_count == kforms->field_capacity)
    {
        const unsigned new_capacity = kforms->field_capacity > 0 ? 2 * kforms->field_capacity : 4;
        element_kforms_field_t *const fields =
            kforms->allocator->reallocate(kforms->allocator->state, kforms->fields, new_capacity * sizeof(*fields));
        if (!fields)
            return FDG_ERROR_FAILED_ALLOCATION;
        kforms->fields = fields;
        kforms->field_capacity = new_capacity;
    }

    const size_t label_size = strlen(label) + 1;
    char *const label_copy = kforms->allocator->allocate(kforms->allocator->state, label_size);
    if (!label_copy)
        return FDG_ERROR_FAILED_ALLOCATION;
    memcpy(label_copy, label, label_size);

    element_data_t *store;
    const fdg_result_t create_res = element_data_create(&store, kforms->allocator);
    if (create_res != FDG_SUCCESS)
    {
        kforms->allocator->deallocate(kforms->allocator->state, label_copy);
        return create_res;
    }

    kforms->fields[kforms->field_count] = (element_kforms_field_t){.label = label_copy, .order = order, .store = store};
    if (out_field)
        *out_field = kforms->field_count;
    kforms->field_count += 1;
    kforms->ndim = ndim;
    return FDG_SUCCESS;
}

bool element_kforms_find_field(const element_kforms_t *kforms, const char *label, unsigned *out_field)
{
    if (label)
    {
        for (unsigned i = 0; i < kforms->field_count; ++i)
        {
            if (strcmp(kforms->fields[i].label, label) == 0)
            {
                *out_field = i;
                return true;
            }
        }
    }
    return false;
}

fdg_result_t element_kforms_add_space(element_kforms_t *kforms, const basis_spec_t basis_specs[], unsigned *out_index)
{
    CUTL_ASSERT(kforms->field_count > 0, "A base space cannot be added before any field.");
    for (unsigned i = 0; i < kforms->ndim; ++i)
    {
        const bool type_valid = basis_set_type_is_valid(basis_specs[i].type);
        CUTL_ASSERT(type_valid, "Basis axis %u of the base space does not use a valid basis family.", i);
    }
    // A nonzero-order k-form needs a strictly positive basis order on every
    // axis. Validate here so every field store accepts or rejects the space
    // together.
    for (unsigned i = 0; i < kforms->field_count; ++i)
    {
        if (kforms->fields[i].order == 0)
            continue;
        for (unsigned axis = 0; axis < kforms->ndim; ++axis)
        {
            CUTL_ASSERT(basis_specs[axis].order != 0,
                        "Basis axis %u has order 0, which cannot carry the order-%u field %s.", axis,
                        kforms->fields[i].order, kforms->fields[i].label);
        }
    }

    // The option is only read by element_data_add_option, never stored. Each
    // field store holds the space as an option with its own k-form order; the
    // stores fill their option tables in the same order, so the indices match.
    unsigned index = 0;
    for (unsigned i = 0; i < kforms->field_count; ++i)
    {
        const element_data_option_t option = {.kind = ELEMENT_DATA_KIND_KFORM,
                                              .ndim = kforms->ndim,
                                              .basis_specs = (basis_spec_t *)basis_specs,
                                              .kform = {.order = kforms->fields[i].order}};
        unsigned option_index;
        const fdg_result_t res = element_data_add_option(kforms->fields[i].store, &option, &option_index);
        if (res != FDG_SUCCESS)
            return res;
        if (i == 0)
            index = option_index;
        CUTL_ASSERT(option_index == index, "Field stores disagree on the option index of a base space (%u vs %u).",
                    option_index, index);
    }
    *out_index = index;
    return FDG_SUCCESS;
}

fdg_result_t element_kforms_add_element(element_kforms_t *kforms, const unsigned space_index, const double values[])
{
    const unsigned space_count = element_kforms_space_count_impl(kforms);
    CUTL_ASSERT(kforms->field_count > 0, "No fields added; cannot add an element.");
    CUTL_ASSERT(space_index < space_count, "Space index %u is not in [0, %u).", space_index, space_count);
    size_t offset = 0;
    for (unsigned i = 0; i < kforms->field_count; ++i)
    {
        const size_t count = element_data_option_value_count(element_data_option(kforms->fields[i].store, space_index));
        const fdg_result_t res = element_data_add_element(kforms->fields[i].store, space_index, values + offset, count);
        if (res != FDG_SUCCESS)
            return res;
        offset += count;
    }
    return FDG_SUCCESS;
}

void element_kforms_set_field_values(element_kforms_t *kforms, const uint64_t element_id, const unsigned field,
                                     const double values[])
{
    CUTL_ASSERT(field < kforms->field_count, "Field index %u is not in [0, %u).", field, kforms->field_count);
    element_data_t *const store = kforms->fields[field].store;
    const uint64_t element_count = element_data_element_count(store);
    CUTL_ASSERT(element_id < element_count, "Element id %llu is not in [0, %llu).", (unsigned long long)element_id,
                (unsigned long long)element_count);
    const uint64_t *const offsets = element_data_offsets(store);
    element_data_set_element_values(store, element_id, values, offsets[element_id + 1] - offsets[element_id]);
}

unsigned element_kforms_field_count(const element_kforms_t *kforms)
{
    return kforms->field_count;
}

const char *element_kforms_field_label(const element_kforms_t *kforms, const unsigned field)
{
    CUTL_ASSERT(field < kforms->field_count, "Field index %u is not in [0, %u).", field, kforms->field_count);
    return kforms->fields[field].label;
}

unsigned element_kforms_ndim(const element_kforms_t *kforms)
{
    return kforms->ndim;
}

unsigned element_kforms_field_order(const element_kforms_t *kforms, const unsigned field)
{
    CUTL_ASSERT(field < kforms->field_count, "Field index %u is not in [0, %u).", field, kforms->field_count);
    return kforms->fields[field].order;
}

unsigned element_kforms_space_count(const element_kforms_t *kforms)
{
    return element_kforms_space_count_impl(kforms);
}

const element_data_option_t *element_kforms_space_option(const element_kforms_t *kforms, const unsigned index)
{
    const unsigned space_count = element_kforms_space_count_impl(kforms);
    CUTL_ASSERT(index < space_count, "Space index %u is not in [0, %u).", index, space_count);
    return element_data_option(kforms->fields[0].store, index);
}

unsigned element_kforms_element_space(const element_kforms_t *kforms, const uint64_t element_id)
{
    CUTL_ASSERT(kforms->field_count > 0, "No fields added.");
    const uint64_t element_count = element_data_element_count(kforms->fields[0].store);
    CUTL_ASSERT(element_id < element_count, "Element id %llu is not in [0, %llu).", (unsigned long long)element_id,
                (unsigned long long)element_count);
    return element_data_element_options(kforms->fields[0].store)[element_id];
}

size_t element_kforms_element_value_count(const element_kforms_t *kforms, const unsigned space_index)
{
    size_t count = 0;
    for (unsigned i = 0; i < kforms->field_count; ++i)
    {
        count += element_data_option_value_count(element_data_option(kforms->fields[i].store, space_index));
    }
    return count;
}

uint64_t element_kforms_element_count(const element_kforms_t *kforms)
{
    return kforms->field_count > 0 ? element_data_element_count(kforms->fields[0].store) : 0;
}

double *element_kforms_field_values(element_kforms_t *kforms, const unsigned field)
{
    CUTL_ASSERT(field < kforms->field_count, "Field index %u is not in [0, %u).", field, kforms->field_count);
    return element_data_values(kforms->fields[field].store);
}

const uint64_t *element_kforms_field_offsets(const element_kforms_t *kforms, const unsigned field)
{
    CUTL_ASSERT(field < kforms->field_count, "Field index %u is not in [0, %u).", field, kforms->field_count);
    return element_data_offsets(kforms->fields[field].store);
}
