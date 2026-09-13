#include "element_dofs.h"

fdg_result_t element_dofs_create(element_dofs_t **out, const cutl_allocator_t *allocator)
{
    return element_data_create(out, allocator);
}

void element_dofs_free(element_dofs_t *dofs, const cutl_allocator_t *allocator)
{
    element_data_free(dofs, allocator);
}

fdg_result_t element_dofs_add_option(element_dofs_t *dofs, const unsigned ndim,
                                     const basis_spec_t basis_specs[static ndim], unsigned *out_index)
{
    // The option is only read by element_data_add_option, never stored.
    const element_data_option_t option = {
        .kind = ELEMENT_DATA_KIND_DOF,
        .ndim = ndim,
        .basis_specs = (basis_spec_t *)basis_specs,
    };
    return element_data_add_option(dofs, &option, out_index);
}

fdg_result_t element_dofs_add_element(element_dofs_t *dofs, const unsigned option_index, const double values[])
{
    if (option_index >= element_data_option_count(dofs))
        return FDG_ERROR_NOT_IN_DOMAIN;
    const size_t count = element_data_option_value_count(element_data_option(dofs, option_index));
    return element_data_add_element(dofs, option_index, values, count);
}

fdg_result_t element_dofs_set_element_values(element_dofs_t *dofs, const uint64_t element_id, const double values[])
{
    if (element_id >= element_data_element_count(dofs))
        return FDG_ERROR_NOT_IN_DOMAIN;
    const uint64_t *const offsets = element_data_offsets(dofs);
    return element_data_set_element_values(dofs, element_id, values, offsets[element_id + 1] - offsets[element_id]);
}

uint64_t element_dofs_element_count(const element_dofs_t *dofs)
{
    return element_data_element_count(dofs);
}

unsigned element_dofs_option_count(const element_dofs_t *dofs)
{
    return element_data_option_count(dofs);
}

const element_data_option_t *element_dofs_option(const element_dofs_t *dofs, const unsigned index)
{
    return element_data_option(dofs, index);
}

size_t element_dofs_option_value_count(const element_dofs_t *dofs, const unsigned index)
{
    return element_data_option_value_count(element_data_option(dofs, index));
}

const uint32_t *element_dofs_element_options(const element_dofs_t *dofs)
{
    return element_data_element_options(dofs);
}

const uint64_t *element_dofs_offsets(const element_dofs_t *dofs)
{
    return element_data_offsets(dofs);
}

double *element_dofs_values(element_dofs_t *dofs)
{
    return element_data_values(dofs);
}

size_t element_dofs_value_count(const element_dofs_t *dofs)
{
    return element_data_value_count(dofs);
}
