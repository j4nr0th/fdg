#include "mesh_kform_specs.h"

#include "../kforms/kform_types.h"
#include <string.h>

typedef struct
{
    char *label;
    unsigned order; // K-form order of the field.
} mesh_kform_specs_field_t;

typedef struct
{
    basis_spec_t *basis_specs; // [ndim] Deep copy of the base function space.
    size_t *value_counts;      // [field_count] Doubles each field stores on this space.
} mesh_kform_specs_space_t;

struct mesh_kform_specs_t
{
    const cutl_allocator_t *allocator;
    unsigned ndim; // 0 until the first field fixes it.
    unsigned field_count;
    unsigned field_capacity;
    mesh_kform_specs_field_t *fields;
    unsigned space_count;
    unsigned space_capacity;
    mesh_kform_specs_space_t *spaces;
    uint64_t element_count;
    uint64_t element_capacity;
    uint32_t *element_spaces;
    bool frozen; // Set once a values collection borrows the structure.
};

static int mesh_kform_specs_basis_specs_equal(const basis_spec_t *first, const basis_spec_t *second,
                                              const unsigned ndim)
{
    for (unsigned i = 0; i < ndim; ++i)
    {
        if (first[i].type != second[i].type || first[i].order != second[i].order)
            return 0;
    }
    return 1;
}

fdg_result_t mesh_kform_specs_create(mesh_kform_specs_t **out, const cutl_allocator_t *allocator)
{
    mesh_kform_specs_t *const specs = allocator->allocate(allocator->state, sizeof(*specs));
    if (!specs)
        return FDG_ERROR_FAILED_ALLOCATION;
    *specs = (mesh_kform_specs_t){.allocator = allocator};
    *out = specs;
    return FDG_SUCCESS;
}

void mesh_kform_specs_free(mesh_kform_specs_t *specs, const cutl_allocator_t *allocator)
{
    if (!specs)
        return;
    for (unsigned i = 0; i < specs->field_count; ++i)
        allocator->deallocate(allocator->state, specs->fields[i].label);
    allocator->deallocate(allocator->state, specs->fields);
    for (unsigned i = 0; i < specs->space_count; ++i)
    {
        allocator->deallocate(allocator->state, specs->spaces[i].basis_specs);
        allocator->deallocate(allocator->state, specs->spaces[i].value_counts);
    }
    allocator->deallocate(allocator->state, specs->spaces);
    allocator->deallocate(allocator->state, specs->element_spaces);
    allocator->deallocate(allocator->state, specs);
}

fdg_result_t mesh_kform_specs_add_field(mesh_kform_specs_t *specs, const char *label, const unsigned ndim,
                                        const unsigned order, unsigned *out_field)
{
    CUTL_ASSERT(!specs->frozen, "The k-form specs are frozen: they are in use by an ElementKForms.");
    CUTL_ASSERT(specs->space_count == 0 && specs->element_count == 0,
                "Cannot add the field %s: fields must be added before any base space or element.", label ? label : "");
    CUTL_ASSERT(label != NULL, "Field label must not be null.");
    CUTL_ASSERT(label == NULL || label[0] != '\0', "Field label must not be empty.");
    CUTL_ASSERT(ndim >= 1, "Field dimension %u must be positive.", ndim);
    CUTL_ASSERT(order <= ndim, "Field order %u is not in [0, %u].", order, ndim);
    CUTL_ASSERT(specs->ndim == 0 || specs->ndim == ndim,
                "Field dimension %u does not match the collection dimension %u.", ndim, specs->ndim);
    for (unsigned i = 0; i < specs->field_count; ++i)
    {
        CUTL_ASSERT(strcmp(specs->fields[i].label, label) != 0, "A field labeled %s already exists.", label);
    }

    if (specs->field_count == specs->field_capacity)
    {
        const unsigned new_capacity = specs->field_capacity > 0 ? 2 * specs->field_capacity : 4;
        mesh_kform_specs_field_t *const fields =
            specs->allocator->reallocate(specs->allocator->state, specs->fields, new_capacity * sizeof(*fields));
        if (!fields)
            return FDG_ERROR_FAILED_ALLOCATION;
        specs->fields = fields;
        specs->field_capacity = new_capacity;
    }

    const size_t label_size = strlen(label) + 1;
    char *const label_copy = specs->allocator->allocate(specs->allocator->state, label_size);
    if (!label_copy)
        return FDG_ERROR_FAILED_ALLOCATION;
    memcpy(label_copy, label, label_size);

    specs->fields[specs->field_count] = (mesh_kform_specs_field_t){.label = label_copy, .order = order};
    if (out_field)
        *out_field = specs->field_count;
    specs->field_count += 1;
    specs->ndim = ndim;
    return FDG_SUCCESS;
}

bool mesh_kform_specs_find_field(const mesh_kform_specs_t *specs, const char *label, unsigned *out_field)
{
    if (label)
    {
        for (unsigned i = 0; i < specs->field_count; ++i)
        {
            if (strcmp(specs->fields[i].label, label) == 0)
            {
                *out_field = i;
                return true;
            }
        }
    }
    return false;
}

fdg_result_t mesh_kform_specs_add_space(mesh_kform_specs_t *specs, const basis_spec_t basis_specs[],
                                        unsigned *out_index)
{
    CUTL_ASSERT(!specs->frozen, "The k-form specs are frozen: they are in use by an ElementKForms.");
    CUTL_ASSERT(specs->field_count > 0, "A base space cannot be added before any field.");
    for (unsigned i = 0; i < specs->ndim; ++i)
    {
        const bool type_valid = basis_set_type_is_valid(basis_specs[i].type);
        CUTL_ASSERT(type_valid, "Basis axis %u of the base space does not use a valid basis family.", i);
    }
    // A nonzero-order k-form needs a strictly positive basis order on every
    // axis.
    for (unsigned i = 0; i < specs->field_count; ++i)
    {
        if (specs->fields[i].order == 0)
            continue;
        for (unsigned axis = 0; axis < specs->ndim; ++axis)
        {
            CUTL_ASSERT(basis_specs[axis].order != 0,
                        "Basis axis %u has order 0, which cannot carry the order-%u field %s.", axis,
                        specs->fields[i].order, specs->fields[i].label);
        }
    }

    // Dedup: an equal base space reuses its index, so the table holds every
    // distinct space once no matter how many fields derive from it.
    for (unsigned i = 0; i < specs->space_count; ++i)
    {
        if (mesh_kform_specs_basis_specs_equal(specs->spaces[i].basis_specs, basis_specs, specs->ndim))
        {
            *out_index = i;
            return FDG_SUCCESS;
        }
    }

    if (specs->space_count == specs->space_capacity)
    {
        const unsigned new_capacity = specs->space_capacity > 0 ? 2 * specs->space_capacity : 4;
        mesh_kform_specs_space_t *const spaces =
            specs->allocator->reallocate(specs->allocator->state, specs->spaces, new_capacity * sizeof(*spaces));
        if (!spaces)
            return FDG_ERROR_FAILED_ALLOCATION;
        specs->spaces = spaces;
        specs->space_capacity = new_capacity;
    }

    basis_spec_t *const basis_copy =
        specs->allocator->allocate(specs->allocator->state, specs->ndim * sizeof(*basis_copy));
    if (!basis_copy)
        return FDG_ERROR_FAILED_ALLOCATION;
    memcpy(basis_copy, basis_specs, specs->ndim * sizeof(*basis_copy));

    size_t *const value_counts =
        specs->allocator->allocate(specs->allocator->state, specs->field_count * sizeof(*value_counts));
    if (!value_counts)
    {
        specs->allocator->deallocate(specs->allocator->state, basis_copy);
        return FDG_ERROR_FAILED_ALLOCATION;
    }
    for (unsigned i = 0; i < specs->field_count; ++i)
    {
        const kform_spec_t kform = {
            .ndim = specs->ndim,
            .order = specs->fields[i].order,
            .basis = basis_copy,
        };
        value_counts[i] = kform_spec_total_dofs(&kform);
    }

    specs->spaces[specs->space_count] =
        (mesh_kform_specs_space_t){.basis_specs = basis_copy, .value_counts = value_counts};
    *out_index = specs->space_count;
    specs->space_count += 1;
    return FDG_SUCCESS;
}

fdg_result_t mesh_kform_specs_add_element(mesh_kform_specs_t *specs, const unsigned space_index)
{
    CUTL_ASSERT(!specs->frozen, "The k-form specs are frozen: they are in use by an ElementKForms.");
    CUTL_ASSERT(specs->field_count > 0, "No fields added; cannot add an element.");
    CUTL_ASSERT(space_index < specs->space_count, "Space index %u is not in [0, %u).", space_index, specs->space_count);

    if (specs->element_count == specs->element_capacity)
    {
        const uint64_t new_capacity = specs->element_capacity > 0 ? 2 * specs->element_capacity : 8;
        uint32_t *const element_spaces = specs->allocator->reallocate(specs->allocator->state, specs->element_spaces,
                                                                      new_capacity * sizeof(*element_spaces));
        if (!element_spaces)
            return FDG_ERROR_FAILED_ALLOCATION;
        specs->element_spaces = element_spaces;
        specs->element_capacity = new_capacity;
    }

    specs->element_spaces[specs->element_count] = space_index;
    specs->element_count += 1;
    return FDG_SUCCESS;
}

void mesh_kform_specs_freeze(mesh_kform_specs_t *specs)
{
    specs->frozen = true;
}

bool mesh_kform_specs_is_frozen(const mesh_kform_specs_t *specs)
{
    return specs->frozen;
}

unsigned mesh_kform_specs_field_count(const mesh_kform_specs_t *specs)
{
    return specs->field_count;
}

const char *mesh_kform_specs_field_label(const mesh_kform_specs_t *specs, const unsigned field)
{
    CUTL_ASSERT(field < specs->field_count, "Field index %u is not in [0, %u).", field, specs->field_count);
    return specs->fields[field].label;
}

unsigned mesh_kform_specs_field_order(const mesh_kform_specs_t *specs, const unsigned field)
{
    CUTL_ASSERT(field < specs->field_count, "Field index %u is not in [0, %u).", field, specs->field_count);
    return specs->fields[field].order;
}

unsigned mesh_kform_specs_ndim(const mesh_kform_specs_t *specs)
{
    return specs->ndim;
}

unsigned mesh_kform_specs_space_count(const mesh_kform_specs_t *specs)
{
    return specs->space_count;
}

const basis_spec_t *mesh_kform_specs_space_basis_specs(const mesh_kform_specs_t *specs, const unsigned index)
{
    CUTL_ASSERT(index < specs->space_count, "Space index %u is not in [0, %u).", index, specs->space_count);
    return specs->spaces[index].basis_specs;
}

size_t mesh_kform_specs_space_value_count(const mesh_kform_specs_t *specs, const unsigned space_index,
                                          const unsigned field)
{
    CUTL_ASSERT(space_index < specs->space_count, "Space index %u is not in [0, %u).", space_index, specs->space_count);
    CUTL_ASSERT(field < specs->field_count, "Field index %u is not in [0, %u).", field, specs->field_count);
    return specs->spaces[space_index].value_counts[field];
}

unsigned mesh_kform_specs_element_space(const mesh_kform_specs_t *specs, const uint64_t element_id)
{
    CUTL_ASSERT(element_id < specs->element_count, "Element id %llu is not in [0, %llu).",
                (unsigned long long)element_id, (unsigned long long)specs->element_count);
    return specs->element_spaces[element_id];
}

uint64_t mesh_kform_specs_element_count(const mesh_kform_specs_t *specs)
{
    return specs->element_count;
}

size_t mesh_kform_specs_element_value_count(const mesh_kform_specs_t *specs, const unsigned space_index)
{
    size_t count = 0;
    for (unsigned i = 0; i < specs->field_count; ++i)
        count += mesh_kform_specs_space_value_count(specs, space_index, i);
    return count;
}
