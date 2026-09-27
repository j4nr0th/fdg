#include "element_data.h"

#include <stdbool.h>
#include <string.h>

struct element_data_t
{
    const cutl_allocator_t *allocator;
    unsigned option_count;
    unsigned option_capacity;
    element_data_option_t *options; // options[0] fixes kind and common payload of the collection.
    uint64_t element_count;
    uint32_t *element_options;
    uint64_t *offsets; // element_count + 1 entries; offsets[0] == 0.
    size_t value_count;
    double *values;
};

static void element_data_option_clear(element_data_option_t *option, const cutl_allocator_t *allocator)
{
    allocator->deallocate(allocator->state, option->basis_specs);
    option->basis_specs = NULL;
}

static int element_data_basis_specs_equal(const basis_spec_t *first, const basis_spec_t *second, const unsigned ndim)
{
    for (unsigned i = 0; i < ndim; ++i)
    {
        if (first[i].type != second[i].type || first[i].order != second[i].order)
            return 0;
    }
    return 1;
}

static int element_data_options_equal(const element_data_option_t *first, const element_data_option_t *second)
{
    if (first->kind != second->kind || first->ndim != second->ndim)
        return 0;
    if (!element_data_basis_specs_equal(first->basis_specs, second->basis_specs, first->ndim))
        return 0;
    return 1;
}

static int element_data_option_compatible(const element_data_option_t *first, const element_data_option_t *option)
{
    if (first->kind != option->kind || first->ndim != option->ndim)
        return 0;
    return 1;
}

fdg_result_t element_data_create(element_data_t **out, const cutl_allocator_t *allocator)
{
    element_data_t *const data = allocator->allocate(allocator->state, sizeof(*data));
    if (!data)
        return FDG_ERROR_FAILED_ALLOCATION;
    uint64_t *const offsets = allocator->allocate(allocator->state, sizeof(*offsets));
    if (!offsets)
    {
        allocator->deallocate(allocator->state, data);
        return FDG_ERROR_FAILED_ALLOCATION;
    }
    offsets[0] = 0;
    *data = (element_data_t){.allocator = allocator, .offsets = offsets};
    *out = data;
    return FDG_SUCCESS;
}

void element_data_free(element_data_t *data, const cutl_allocator_t *allocator)
{
    if (!data)
        return;
    for (unsigned i = 0; i < data->option_count; ++i)
        element_data_option_clear(data->options + i, allocator);
    allocator->deallocate(allocator->state, data->options);
    allocator->deallocate(allocator->state, data->element_options);
    allocator->deallocate(allocator->state, data->offsets);
    allocator->deallocate(allocator->state, data->values);
    allocator->deallocate(allocator->state, data);
}

static void element_data_option_value_count_impl(const element_data_option_t *option, size_t *out_count)
{
    CUTL_ASSERT(option->kind != ELEMENT_DATA_KIND_INVALID,
                "Cannot count the values of an option with kind ELEMENT_DATA_KIND_INVALID.");
    size_t count = 1;
    switch (option->kind)
    {
    case ELEMENT_DATA_KIND_DOF:
        for (unsigned i = 0; i < option->ndim; ++i)
            count *= option->basis_specs[i].order + 1;
        break;
    case ELEMENT_DATA_KIND_INVALID:
        break;
    }
    *out_count = count;
}

size_t element_data_option_value_count(const element_data_option_t *option)
{
    return option->value_count;
}

fdg_result_t element_data_add_option(element_data_t *data, const element_data_option_t *option, unsigned *out_index)
{
    CUTL_ASSERT(option->kind != ELEMENT_DATA_KIND_INVALID, "Option kind must not be ELEMENT_DATA_KIND_INVALID.");
    CUTL_ASSERT(option->ndim >= 1 && option->ndim <= 63, "Option dimension %u is not in [1, 63].", option->ndim);
    CUTL_ASSERT(option->basis_specs != NULL, "Basis spec array of the option must not be null.");
    for (unsigned i = 0; i < option->ndim; ++i)
    {
        const bool type_valid = basis_set_type_is_valid(option->basis_specs[i].type);
        CUTL_ASSERT(type_valid, "Basis spec %u of the option does not use a valid basis family.", i);
    }

    // The compatibility with the first option is a recoverable error, not a
    // precondition: the Python bindings can reach it with per-element options
    // of a different kind or dimension, so it must be reported instead of
    // aborting.
    if (data->option_count > 0 && !element_data_option_compatible(data->options, option))
        return FDG_ERROR_NOT_IN_DOMAIN;

    element_data_option_t copy = *option;
    copy.value_count = 0;
    element_data_option_value_count_impl(&copy, &copy.value_count);

    // Dedup: return the index of an existing equal option.
    for (unsigned i = 0; i < data->option_count; ++i)
    {
        if (element_data_options_equal(data->options + i, &copy))
        {
            *out_index = i;
            return FDG_SUCCESS;
        }
    }

    if (data->option_count == data->option_capacity)
    {
        const unsigned new_capacity = data->option_capacity > 0 ? 2 * data->option_capacity : 8;
        element_data_option_t *const options =
            data->allocator->reallocate(data->allocator->state, data->options, new_capacity * sizeof(*options));
        if (!options)
            return FDG_ERROR_FAILED_ALLOCATION;
        data->options = options;
        data->option_capacity = new_capacity;
    }

    copy.basis_specs = data->allocator->allocate(data->allocator->state, copy.ndim * sizeof(*copy.basis_specs));
    if (!copy.basis_specs)
        return FDG_ERROR_FAILED_ALLOCATION;
    memcpy(copy.basis_specs, option->basis_specs, copy.ndim * sizeof(*copy.basis_specs));

    data->options[data->option_count] = copy;
    *out_index = data->option_count;
    data->option_count += 1;
    return FDG_SUCCESS;
}

fdg_result_t element_data_add_element(element_data_t *data, const unsigned option_index, const double values[],
                                      const size_t count)
{
    CUTL_ASSERT(option_index < data->option_count, "Option index %u is not in [0, %u).", option_index,
                data->option_count);
    const element_data_option_t *const option = data->options + option_index;
    CUTL_ASSERT(count == option->value_count, "Element value count %zu does not match the %zu values of option %u.",
                count, option->value_count, option_index);

    const uint64_t element_count = data->element_count;
    uint32_t *const element_options = data->allocator->reallocate(data->allocator->state, data->element_options,
                                                                  (element_count + 1) * sizeof(uint32_t));
    if (!element_options)
        return FDG_ERROR_FAILED_ALLOCATION;
    data->element_options = element_options;
    uint64_t *const offsets =
        data->allocator->reallocate(data->allocator->state, data->offsets, (element_count + 2) * sizeof(uint64_t));
    if (!offsets)
        return FDG_ERROR_FAILED_ALLOCATION;
    data->offsets = offsets;
    double *const new_values =
        data->allocator->reallocate(data->allocator->state, data->values, (data->value_count + count) * sizeof(double));
    if (!new_values)
        return FDG_ERROR_FAILED_ALLOCATION;
    data->values = new_values;

    element_options[element_count] = option_index;
    memcpy(data->values + data->value_count, values, count * sizeof(double));
    data->value_count += count;
    data->element_count = element_count + 1;
    offsets[data->element_count] = data->value_count;
    if (element_count == 0)
        offsets[0] = 0;
    return FDG_SUCCESS;
}

void element_data_set_element_values(element_data_t *data, const uint64_t element_id, const double values[],
                                     const size_t count)
{
    CUTL_ASSERT(element_id < data->element_count, "Element id %llu is not in [0, %llu).",
                (unsigned long long)element_id, (unsigned long long)data->element_count);
    const size_t begin = data->offsets[element_id];
    CUTL_ASSERT(count == data->offsets[element_id + 1] - begin,
                "Overwrite count %zu does not match the %llu values of element %llu.", count,
                (unsigned long long)(data->offsets[element_id + 1] - begin), (unsigned long long)element_id);
    memcpy(data->values + begin, values, count * sizeof(double));
}

element_data_kind_t element_data_kind(const element_data_t *data)
{
    return data->option_count > 0 ? data->options[0].kind : ELEMENT_DATA_KIND_INVALID;
}

uint64_t element_data_element_count(const element_data_t *data)
{
    return data->element_count;
}

unsigned element_data_option_count(const element_data_t *data)
{
    return data->option_count;
}

const element_data_option_t *element_data_option(const element_data_t *data, const unsigned index)
{
    CUTL_ASSERT(index < data->option_count, "Option index %u is not in [0, %u).", index, data->option_count);
    return data->options + index;
}

const uint32_t *element_data_element_options(const element_data_t *data)
{
    return data->element_options;
}

const uint64_t *element_data_offsets(const element_data_t *data)
{
    return data->offsets;
}

double *element_data_values(element_data_t *data)
{
    return data->values;
}

size_t element_data_value_count(const element_data_t *data)
{
    return data->value_count;
}
