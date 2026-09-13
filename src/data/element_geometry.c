#include "element_geometry.h"

#include <string.h>

fdg_result_t element_geometry_create(element_geometry_t **out, const cutl_allocator_t *allocator)
{
    return element_data_create(out, allocator);
}

void element_geometry_free(element_geometry_t *geometry, const cutl_allocator_t *allocator)
{
    element_data_free(geometry, allocator);
}

fdg_result_t element_geometry_add_option(element_geometry_t *geometry, const unsigned ndim, const unsigned coord_count,
                                         const basis_spec_t basis_specs[static ndim],
                                         const integration_spec_t int_specs[static ndim], unsigned *out_index)
{
    // The option is only read by element_data_add_option, never stored.
    const element_data_option_t option = {
        .kind = ELEMENT_DATA_KIND_GEOMETRY,
        .ndim = ndim,
        .basis_specs = (basis_spec_t *)basis_specs,
        .geometry = {.coord_count = coord_count, .int_specs = (integration_spec_t *)int_specs},
    };
    return element_data_add_option(geometry, &option, out_index);
}

fdg_result_t element_geometry_add_element(element_geometry_t *geometry, const unsigned option_index,
                                          const double values[])
{
    if (option_index >= element_data_option_count(geometry))
        return FDG_ERROR_NOT_IN_DOMAIN;
    const size_t count = element_data_option_value_count(element_data_option(geometry, option_index));
    return element_data_add_element(geometry, option_index, values, count);
}

fdg_result_t element_geometry_set_element_values(element_geometry_t *geometry, const uint64_t element_id,
                                                 const double values[])
{
    if (element_id >= element_data_element_count(geometry))
        return FDG_ERROR_NOT_IN_DOMAIN;
    const uint64_t *const offsets = element_data_offsets(geometry);
    return element_data_set_element_values(geometry, element_id, values, offsets[element_id + 1] - offsets[element_id]);
}

uint64_t element_geometry_element_count(const element_geometry_t *geometry)
{
    return element_data_element_count(geometry);
}

unsigned element_geometry_option_count(const element_geometry_t *geometry)
{
    return element_data_option_count(geometry);
}

const element_data_option_t *element_geometry_option(const element_geometry_t *geometry, const unsigned index)
{
    return element_data_option(geometry, index);
}

size_t element_geometry_option_value_count(const element_geometry_t *geometry, const unsigned index)
{
    return element_data_option_value_count(element_data_option(geometry, index));
}

const uint32_t *element_geometry_element_options(const element_geometry_t *geometry)
{
    return element_data_element_options(geometry);
}

const uint64_t *element_geometry_offsets(const element_geometry_t *geometry)
{
    return element_data_offsets(geometry);
}

double *element_geometry_values(element_geometry_t *geometry)
{
    return element_data_values(geometry);
}

size_t element_geometry_value_count(const element_geometry_t *geometry)
{
    return element_data_value_count(geometry);
}
