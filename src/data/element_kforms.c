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
    if (element_kforms_element_count(kforms) != 0 || element_kforms_space_count_impl(kforms) != 0)
        return FDG_ERROR_NOT_IN_DOMAIN;
    if (!label || label[0] == '\0' || ndim < 1 || order > ndim)
        return FDG_ERROR_NOT_IN_DOMAIN;
    if (kforms->ndim != 0 && kforms->ndim != ndim)
        return FDG_ERROR_NOT_IN_DOMAIN;
    for (unsigned i = 0; i < kforms->field_count; ++i)
    {
        if (strcmp(kforms->fields[i].label, label) == 0)
            return FDG_ERROR_NOT_IN_DOMAIN;
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

fdg_result_t element_kforms_find_field(const element_kforms_t *kforms, const char *label, unsigned *out_field)
{
    if (label)
    {
        for (unsigned i = 0; i < kforms->field_count; ++i)
        {
            if (strcmp(kforms->fields[i].label, label) == 0)
            {
                *out_field = i;
                return FDG_SUCCESS;
            }
        }
    }
    return FDG_ERROR_NOT_IN_DOMAIN;
}

fdg_result_t element_kforms_add_space(element_kforms_t *kforms, const basis_spec_t basis_specs[], unsigned *out_index)
{
    if (kforms->field_count == 0)
        return FDG_ERROR_NOT_IN_DOMAIN;
    for (unsigned i = 0; i < kforms->ndim; ++i)
    {
        if (!basis_set_type_is_valid(basis_specs[i].type))
            return FDG_ERROR_NOT_IN_DOMAIN;
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
            if (basis_specs[axis].order == 0)
                return FDG_ERROR_NOT_IN_DOMAIN;
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
        ASSERT(option_index == index, "Field stores disagree on the option index of a base space.");
    }
    *out_index = index;
    return FDG_SUCCESS;
}

fdg_result_t element_kforms_add_element(element_kforms_t *kforms, const unsigned space_index, const double values[])
{
    if (kforms->field_count == 0 || space_index >= element_kforms_space_count_impl(kforms))
        return FDG_ERROR_NOT_IN_DOMAIN;
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

fdg_result_t element_kforms_set_field_values(element_kforms_t *kforms, const uint64_t element_id, const unsigned field,
                                             const double values[])
{
    if (field >= kforms->field_count)
        return FDG_ERROR_NOT_IN_DOMAIN;
    element_data_t *const store = kforms->fields[field].store;
    if (element_id >= element_data_element_count(store))
        return FDG_ERROR_NOT_IN_DOMAIN;
    const uint64_t *const offsets = element_data_offsets(store);
    return element_data_set_element_values(store, element_id, values, offsets[element_id + 1] - offsets[element_id]);
}

unsigned element_kforms_field_count(const element_kforms_t *kforms)
{
    return kforms->field_count;
}

const char *element_kforms_field_label(const element_kforms_t *kforms, const unsigned field)
{
    ASSERT(field < kforms->field_count, "Field index %u out of bounds.", field);
    return kforms->fields[field].label;
}

unsigned element_kforms_ndim(const element_kforms_t *kforms)
{
    return kforms->ndim;
}

unsigned element_kforms_field_order(const element_kforms_t *kforms, const unsigned field)
{
    ASSERT(field < kforms->field_count, "Field index %u out of bounds.", field);
    return kforms->fields[field].order;
}

unsigned element_kforms_space_count(const element_kforms_t *kforms)
{
    return element_kforms_space_count_impl(kforms);
}

const element_data_option_t *element_kforms_space_option(const element_kforms_t *kforms, const unsigned index)
{
    ASSERT(index < element_kforms_space_count_impl(kforms), "Space index %u out of bounds.", index);
    return element_data_option(kforms->fields[0].store, index);
}

unsigned element_kforms_element_space(const element_kforms_t *kforms, const uint64_t element_id)
{
    ASSERT(kforms->field_count > 0, "No fields added.");
    ASSERT(element_id < element_data_element_count(kforms->fields[0].store), "Element id %llu out of bounds.",
           (unsigned long long)element_id);
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
    ASSERT(field < kforms->field_count, "Field index %u out of bounds.", field);
    return element_data_values(kforms->fields[field].store);
}

const uint64_t *element_kforms_field_offsets(const element_kforms_t *kforms, const unsigned field)
{
    ASSERT(field < kforms->field_count, "Field index %u out of bounds.", field);
    return element_data_offsets(kforms->fields[field].store);
}
