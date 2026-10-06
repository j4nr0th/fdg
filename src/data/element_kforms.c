#include "element_kforms.h"

#include <string.h>

struct element_kforms_t
{
    const mesh_kform_specs_t *specs; // Borrowed; must outlive the collection.
    double **values;                 // [field_count] One contiguous array per field.
    uint64_t **offsets;              // [field_count] element_count + 1 entries per field.
    uint64_t filled;                 // Append cursor.
};

fdg_result_t element_kforms_create(element_kforms_t **out, const mesh_kform_specs_t *specs,
                                   const cutl_allocator_t *allocator)
{
    CUTL_ASSERT(specs != NULL, "The k-form specs must not be null.");
    CUTL_ASSERT(specs == NULL || mesh_kform_specs_field_count(specs) > 0,
                "The k-form specs must have at least one field.");

    const unsigned field_count = mesh_kform_specs_field_count(specs);
    const uint64_t element_count = mesh_kform_specs_element_count(specs);

    element_kforms_t *const kforms = allocator->allocate(allocator->state, sizeof(*kforms));
    if (!kforms)
        return FDG_ERROR_FAILED_ALLOCATION;
    *kforms = (element_kforms_t){.specs = specs};

    kforms->values = allocator->allocate(allocator->state, field_count * sizeof(*kforms->values));
    kforms->offsets = allocator->allocate(allocator->state, field_count * sizeof(*kforms->offsets));
    if (!kforms->values || !kforms->offsets)
        goto failure;
    // Zero the slot arrays up front: the allocator does not, and the failure
    // path below deallocates every slot, so a slot the loop below has not
    // reached yet must already be null.
    memset(kforms->values, 0, field_count * sizeof(*kforms->values));
    memset(kforms->offsets, 0, field_count * sizeof(*kforms->offsets));

    for (unsigned field = 0; field < field_count; ++field)
    {
        size_t total = 0;
        for (uint64_t element_id = 0; element_id < element_count; ++element_id)
        {
            total +=
                mesh_kform_specs_space_value_count(specs, mesh_kform_specs_element_space(specs, element_id), field);
        }

        double *const field_values =
            total > 0 ? allocator->allocate(allocator->state, total * sizeof(*field_values)) : NULL;
        if (total > 0 && !field_values)
            goto failure;
        if (total > 0)
            memset(field_values, 0, total * sizeof(*field_values));

        uint64_t *const field_offsets =
            allocator->allocate(allocator->state, (element_count + 1) * sizeof(*field_offsets));
        if (!field_offsets)
        {
            allocator->deallocate(allocator->state, field_values);
            goto failure;
        }

        uint64_t offset = 0;
        for (uint64_t element_id = 0; element_id < element_count; ++element_id)
        {
            field_offsets[element_id] = offset;
            offset += (uint64_t)mesh_kform_specs_space_value_count(
                specs, mesh_kform_specs_element_space(specs, element_id), field);
        }
        field_offsets[element_count] = offset;

        kforms->values[field] = field_values;
        kforms->offsets[field] = field_offsets;
    }

    *out = kforms;
    return FDG_SUCCESS;

failure:
    // The slots start out null and are filled one by one, so deallocating
    // every slot is safe; both pointer arrays are allocated together.
    if (kforms->values && kforms->offsets)
    {
        for (unsigned field = 0; field < field_count; ++field)
        {
            allocator->deallocate(allocator->state, kforms->values[field]);
            allocator->deallocate(allocator->state, kforms->offsets[field]);
        }
    }
    allocator->deallocate(allocator->state, kforms->values);
    allocator->deallocate(allocator->state, kforms->offsets);
    allocator->deallocate(allocator->state, kforms);
    return FDG_ERROR_FAILED_ALLOCATION;
}

void element_kforms_free(element_kforms_t *kforms, const cutl_allocator_t *allocator)
{
    if (!kforms)
        return;
    const unsigned field_count = mesh_kform_specs_field_count(kforms->specs);
    for (unsigned field = 0; field < field_count; ++field)
    {
        allocator->deallocate(allocator->state, kforms->values[field]);
        allocator->deallocate(allocator->state, kforms->offsets[field]);
    }
    allocator->deallocate(allocator->state, kforms->values);
    allocator->deallocate(allocator->state, kforms->offsets);
    allocator->deallocate(allocator->state, kforms);
}

void element_kforms_add_element(element_kforms_t *kforms, const double values[])
{
    const mesh_kform_specs_t *const specs = kforms->specs;
    const unsigned field_count = mesh_kform_specs_field_count(specs);
    const uint64_t filled = kforms->filled;
    CUTL_ASSERT(filled < mesh_kform_specs_element_count(specs), "The collection is full: %llu of %llu elements filled.",
                (unsigned long long)filled, (unsigned long long)mesh_kform_specs_element_count(specs));

    size_t offset = 0;
    for (unsigned field = 0; field < field_count; ++field)
    {
        const uint64_t *const field_offsets = kforms->offsets[field];
        const size_t count = field_offsets[filled + 1] - field_offsets[filled];
        if (count > 0)
            memcpy(kforms->values[field] + field_offsets[filled], values + offset,
                   count * sizeof(*kforms->values[field]));
        offset += count;
    }
    kforms->filled = filled + 1;
}

void element_kforms_set_field_values(element_kforms_t *kforms, const uint64_t element_id, const unsigned field,
                                     const double values[])
{
    const mesh_kform_specs_t *const specs = kforms->specs;
    CUTL_ASSERT(field < mesh_kform_specs_field_count(specs), "Field index %u is not in [0, %u).", field,
                mesh_kform_specs_field_count(specs));
    const uint64_t element_count = mesh_kform_specs_element_count(specs);
    CUTL_ASSERT(element_id < element_count, "Element id %llu is not in [0, %llu).", (unsigned long long)element_id,
                (unsigned long long)element_count);
    const uint64_t *const field_offsets = kforms->offsets[field];
    const size_t count = field_offsets[element_id + 1] - field_offsets[element_id];
    if (count > 0)
        memcpy(kforms->values[field] + field_offsets[element_id], values, count * sizeof(*kforms->values[field]));
}

const mesh_kform_specs_t *element_kforms_specs(const element_kforms_t *kforms)
{
    return kforms->specs;
}

uint64_t element_kforms_filled_count(const element_kforms_t *kforms)
{
    return kforms->filled;
}

double *element_kforms_field_values(element_kforms_t *kforms, const unsigned field)
{
    CUTL_ASSERT(field < mesh_kform_specs_field_count(kforms->specs), "Field index %u is not in [0, %u).", field,
                mesh_kform_specs_field_count(kforms->specs));
    return kforms->values[field];
}

const uint64_t *element_kforms_field_offsets(const element_kforms_t *kforms, const unsigned field)
{
    CUTL_ASSERT(field < mesh_kform_specs_field_count(kforms->specs), "Field index %u is not in [0, %u).", field,
                mesh_kform_specs_field_count(kforms->specs));
    return kforms->offsets[field];
}
