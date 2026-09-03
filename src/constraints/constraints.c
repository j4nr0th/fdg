#include "constraints.h"
#include "cutl/iterators/combination_iterator.h"
#include <limits.h>
#include <stdbool.h>

#define CONSTRAINT_STATUS_CASE(stat)                                                                                   \
    case stat:                                                                                                         \
        return #stat
const char *constraint_status_to_str(const constraint_status_t status)
{
    switch (status)
    {
        CONSTRAINT_STATUS_CASE(CONSTRAINT_SUCCESS);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INVALID_ARGUMENT);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INVALID_DIMENSION);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INVALID_ORDER);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_SIZE_OVERFLOW);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INSUFFICIENT_STORAGE);
    }
    return "Unknown";
}
#undef CONSTRAINT_STATUS_CASE

#define CONSTRAINT_STATUS_MSG(stat, msg)                                                                               \
    case stat:                                                                                                         \
        return msg
const char *constraint_status_msg(const constraint_status_t status)
{
    switch (status)
    {
        CONSTRAINT_STATUS_MSG(CONSTRAINT_SUCCESS, "Success");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INVALID_ARGUMENT, "Invalid argument");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INVALID_DIMENSION, "Invalid dimension");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INVALID_ORDER, "Invalid form order");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_SIZE_OVERFLOW, "Size calculation overflow");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INSUFFICIENT_STORAGE, "Insufficient output storage");
    }
    return "Unknown";
}
#undef CONSTRAINT_STATUS_MSG

static constraint_status_t validate_kform_spec(const constraint_kform_spec_t *const spec)
{
    if (!spec)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (spec->ndim > UINT8_MAX)
        return CONSTRAINT_INVALID_DIMENSION;
    if (spec->order > spec->ndim)
        return CONSTRAINT_INVALID_ORDER;
    if (spec->ndim != 0 && !spec->basis_specs)
        return CONSTRAINT_INVALID_ARGUMENT;

    for (unsigned idim = 0; idim < spec->ndim; ++idim)
    {
        if ((spec->order != 0 && spec->basis_specs[idim].order == 0) || spec->basis_specs[idim].type <= BASIS_INVALID ||
            spec->basis_specs[idim].type > BASIS_BERNSTEIN)
            return CONSTRAINT_INVALID_ORDER;
    }
    return CONSTRAINT_SUCCESS;
}
static void component_axes(const unsigned ndim, const unsigned order, const unsigned component,
                           uint8_t axes[const static order == 0 ? 1 : order])
{
    combination_set_to_index((uint8_t)ndim, (uint8_t)order, axes, component);
}

static bool component_has_axis(const unsigned order, const uint8_t axes[const static order == 0 ? 1 : order],
                               const unsigned axis)
{
    for (unsigned i = 0; i < order; ++i)
    {
        if (axes[i] == axis)
            return true;
    }
    return false;
}

static constraint_status_t validate_element_side(const constraint_kform_spec_t *const test_spec,
                                                 const constraint_element_side_t *const side)
{
    if (!side || side->ndim <= test_spec->ndim || !side->basis_specs || !side->orientation)
        return CONSTRAINT_INVALID_ARGUMENT;

    bool used_axes[side->ndim];
    for (unsigned i = 0; i < side->ndim; ++i)
    {
        used_axes[i] = false;
        if ((test_spec->order != 0 && side->basis_specs[i].order == 0) || side->basis_specs[i].type <= BASIS_INVALID ||
            side->basis_specs[i].type > BASIS_BERNSTEIN)
            return CONSTRAINT_INVALID_ARGUMENT;
    }

    for (unsigned i = 0; i < side->ndim; ++i)
    {
        const int8_t mapped_axis = side->orientation[i];
        const unsigned axis = (unsigned)(mapped_axis < 0 ? -mapped_axis : mapped_axis);
        if (axis == 0 || axis > side->ndim || used_axes[axis - 1])
            return CONSTRAINT_INVALID_ARGUMENT;
        used_axes[axis - 1] = true;
    }

    const unsigned fixed_count = side->ndim - test_spec->ndim;
    for (unsigned i = 1; i < fixed_count; ++i)
    {
        const unsigned previous =
            (unsigned)(side->orientation[i - 1] < 0 ? -side->orientation[i - 1] : side->orientation[i - 1]);
        const unsigned current = (unsigned)(side->orientation[i] < 0 ? -side->orientation[i] : side->orientation[i]);
        if (current <= previous)
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    return CONSTRAINT_SUCCESS;
}

static constraint_status_t mapped_component(const constraint_element_side_t *const side, const unsigned boundary_dim,
                                            const unsigned order,
                                            const uint8_t test_axes[const static order == 0 ? 1 : order],
                                            unsigned *const out_component, int *const out_sign)
{
    uint8_t mapped_axes[order == 0 ? 1 : order];
    const unsigned fixed_count = side->ndim - boundary_dim;
    int sign = 1;
    for (unsigned i = 0; i < order; ++i)
    {
        const int8_t mapping = side->orientation[fixed_count + test_axes[i]];
        mapped_axes[i] = (uint8_t)(mapping < 0 ? -mapping : mapping) - 1;
        if (mapping < 0)
            sign = -sign;
    }
    for (unsigned i = 0; i < order; ++i)
    {
        for (unsigned j = i + 1; j < order; ++j)
        {
            if (mapped_axes[i] > mapped_axes[j])
            {
                sign = -sign;
                const uint8_t tmp = mapped_axes[i];
                mapped_axes[i] = mapped_axes[j];
                mapped_axes[j] = tmp;
            }
        }
    }

    *out_component = combination_get_index(side->ndim, order, mapped_axes);
    *out_sign = sign;
    return CONSTRAINT_SUCCESS;
}

static unsigned combination_axis_at(const unsigned ndim, const unsigned order, const unsigned index,
                                    const unsigned position)
{
    unsigned remaining = index;
    unsigned minimum = 0;
    for (unsigned current = 0; current <= position; ++current)
    {
        const unsigned maximum = ndim - (order - current);
        unsigned selected = maximum;
        for (unsigned candidate = minimum; candidate <= maximum; ++candidate)
        {
            const unsigned count =
                combination_total_count((uint8_t)(ndim - candidate - 1), (uint8_t)(order - current - 1));
            if (remaining < count)
            {
                selected = candidate;
                break;
            }
            remaining -= count;
        }
        if (current == position)
            return selected;
        minimum = selected + 1;
    }
    return UINT_MAX;
}

static constraint_status_t mapped_component_for_index(const constraint_element_side_t *const side,
                                                      const unsigned boundary_dim, const unsigned order,
                                                      const unsigned component, unsigned *const out_component,
                                                      int *const out_sign)
{
    if (!side || !out_component || !out_sign || boundary_dim > side->ndim || order > boundary_dim)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (component >= combination_total_count((uint8_t)boundary_dim, (uint8_t)order))
        return CONSTRAINT_INVALID_ARGUMENT;

    const unsigned fixed_count = side->ndim - boundary_dim;
    int sign = 1;
    for (unsigned first = 0; first < order; ++first)
    {
        const unsigned first_axis = combination_axis_at(boundary_dim, order, component, first);
        const int8_t first_mapping = side->orientation[fixed_count + first_axis];
        if (first_mapping < 0)
            sign = -sign;
        for (unsigned second = first + 1; second < order; ++second)
        {
            const unsigned second_axis = combination_axis_at(boundary_dim, order, component, second);
            const unsigned first_element_axis = (unsigned)(first_mapping < 0 ? -first_mapping : first_mapping) - 1;
            const int8_t second_mapping = side->orientation[fixed_count + second_axis];
            const unsigned second_element_axis = (unsigned)(second_mapping < 0 ? -second_mapping : second_mapping) - 1;
            if (first_element_axis > second_element_axis)
                sign = -sign;
        }
    }

    unsigned selected_count = 0;
    unsigned minimum = 0;
    unsigned mapped_index = 0;
    for (unsigned element_axis = 0; element_axis < side->ndim; ++element_axis)
    {
        bool selected = false;
        for (unsigned position = 0; position < order; ++position)
        {
            const unsigned face_axis = combination_axis_at(boundary_dim, order, component, position);
            const int8_t mapping = side->orientation[fixed_count + face_axis];
            if ((unsigned)(mapping < 0 ? -mapping : mapping) - 1 == element_axis)
            {
                selected = true;
                break;
            }
        }
        if (selected)
        {
            for (unsigned candidate = minimum; candidate < element_axis; ++candidate)
                mapped_index += combination_total_count((uint8_t)(side->ndim - candidate - 1),
                                                        (uint8_t)(order - selected_count - 1));
            minimum = element_axis + 1;
            selected_count += 1;
        }
    }
    if (selected_count != order)
        return CONSTRAINT_INVALID_ARGUMENT;
    *out_component = mapped_index;
    *out_sign = sign;
    return CONSTRAINT_SUCCESS;
}

static constraint_status_t validate_trace_basis_values(const constraint_kform_spec_t *const spec,
                                                       const constraint_trace_basis_values_t *const values,
                                                       const size_t point_count)
{
    if (!values || values->point_count != point_count || !values->component_offsets || !values->values)
        return CONSTRAINT_INVALID_ARGUMENT;
    size_t component_count;
    constraint_status_t status = constraint_kform_component_count(spec, &component_count);
    if (status != CONSTRAINT_SUCCESS || values->component_count != component_count)
        return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
    if (values->component_offsets[0] != 0)
        return CONSTRAINT_INVALID_ARGUMENT;
    for (size_t component = 0; component < component_count; ++component)
    {
        size_t dof_count;
        status = constraint_kform_component_dof_count(spec, (unsigned)component, &dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        if (values->component_offsets[component + 1] < values->component_offsets[component] ||
            values->component_offsets[component + 1] - values->component_offsets[component] != dof_count)
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    return CONSTRAINT_SUCCESS;
}

static constraint_status_t constraint_reference_validate(const constraint_kform_spec_t *const test_spec,
                                                         const constraint_element_side_t sides[const static 2])
{
    const constraint_status_t test_status = validate_kform_spec(test_spec);
    if (test_status != CONSTRAINT_SUCCESS)
        return test_status;
    if (test_spec->ndim == UINT_MAX)
        return CONSTRAINT_INVALID_DIMENSION;
    if (test_spec->order > test_spec->ndim)
        return CONSTRAINT_INVALID_ORDER;

    for (unsigned side = 0; side < 2; ++side)
    {
        const constraint_status_t side_status = validate_element_side(test_spec, sides + side);
        if (side_status != CONSTRAINT_SUCCESS)
            return side_status;
    }
    return CONSTRAINT_SUCCESS;
}

static constraint_status_t constraint_reference_counts(const constraint_kform_spec_t *const test_spec,
                                                       const constraint_element_side_t sides[const static 2],
                                                       size_t *const out_row_count, size_t *const out_entry_count)
{
    const constraint_status_t status = constraint_reference_validate(test_spec, sides);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t component_count;
    constraint_status_t result = constraint_kform_component_count(test_spec, &component_count);
    if (result != CONSTRAINT_SUCCESS)
        return result;

    size_t row_count = 0;
    size_t entry_count = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        size_t test_dof_count;
        result = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (result != CONSTRAINT_SUCCESS)
            return result;
        if (row_count > SIZE_MAX - test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        row_count += test_dof_count;

        uint8_t test_axes[test_spec->order == 0 ? 1 : test_spec->order];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        size_t entries_per_row = 0;
        for (unsigned side = 0; side < 2; ++side)
        {
            unsigned element_component;
            int orientation_sign;
            mapped_component(sides + side, test_spec->ndim, test_spec->order, test_axes, &element_component,
                             &orientation_sign);
            size_t element_dof_count;
            result = constraint_kform_component_dof_count(
                &(constraint_kform_spec_t){
                    .ndim = sides[side].ndim, .order = test_spec->order, .basis_specs = sides[side].basis_specs},
                element_component, &element_dof_count);
            if (result != CONSTRAINT_SUCCESS)
                return result;
            if (entries_per_row > SIZE_MAX - element_dof_count)
                return CONSTRAINT_SIZE_OVERFLOW;
            entries_per_row += element_dof_count;
        }
        if (test_dof_count > 0 && entries_per_row > SIZE_MAX / test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        const size_t component_entries = test_dof_count * entries_per_row;
        if (entry_count > SIZE_MAX - component_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        entry_count += component_entries;
    }

    *out_row_count = row_count;
    *out_entry_count = entry_count;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_reference_required(const constraint_kform_spec_t *const test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    return constraint_reference_counts(test_spec, sides, out_row_count, out_entry_count);
}

static constraint_status_t constraint_physical_counts(const constraint_kform_spec_t *const test_spec,
                                                      const constraint_element_side_t sides[const static 2],
                                                      size_t *const out_row_count, size_t *const out_entry_count)
{
    const constraint_status_t status = constraint_reference_validate(test_spec, sides);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t test_component_count;
    constraint_status_t result = constraint_kform_component_count(test_spec, &test_component_count);
    if (result != CONSTRAINT_SUCCESS)
        return result;

    size_t row_count = 0;
    size_t entry_count = 0;
    for (unsigned test_component = 0; test_component < test_component_count; ++test_component)
    {
        size_t test_dof_count;
        result = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (result != CONSTRAINT_SUCCESS)
            return result;
        if (row_count > SIZE_MAX - test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        row_count += test_dof_count;

        size_t entries_per_row = 0;
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            const constraint_element_side_t *const side = sides + side_index;
            const unsigned face_component_count =
                combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                uint8_t face_axes[test_spec->order == 0 ? 1 : test_spec->order];
                component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component,
                                 &orientation_sign);
                const constraint_kform_spec_t element_spec = {
                    .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                size_t element_dof_count;
                result = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                if (result != CONSTRAINT_SUCCESS)
                    return result;
                if (entries_per_row > SIZE_MAX - element_dof_count)
                    return CONSTRAINT_SIZE_OVERFLOW;
                entries_per_row += element_dof_count;
            }
        }
        if (test_dof_count > 0 && entries_per_row > SIZE_MAX / test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        const size_t component_entries = test_dof_count * entries_per_row;
        if (entry_count > SIZE_MAX - component_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        entry_count += component_entries;
    }

    *out_row_count = row_count;
    *out_entry_count = entry_count;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_physical_required(const constraint_kform_spec_t *const test_spec,
                                                 const constraint_element_side_t sides[const static 2],
                                                 size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    return constraint_physical_counts(test_spec, sides, out_row_count, out_entry_count);
}

static constraint_status_t constraint_physical_side_counts(const constraint_kform_spec_t *const test_spec,
                                                           const constraint_element_side_t *const side,
                                                           size_t *const out_row_count, size_t *const out_entry_count)
{
    const constraint_status_t status =
        constraint_reference_validate(test_spec, (const constraint_element_side_t[2]){*side, *side});
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t component_count;
    constraint_status_t result = constraint_kform_component_count(test_spec, &component_count);
    if (result != CONSTRAINT_SUCCESS)
        return result;
    size_t row_count = 0;
    size_t entry_count = 0;
    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        size_t test_dof_count;
        result = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (result != CONSTRAINT_SUCCESS)
            return result;
        row_count += test_dof_count;
        size_t entries_per_row = 0;
        for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
        {
            uint8_t face_axes[test_spec->order == 0 ? 1 : test_spec->order];
            component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
            unsigned element_component;
            int orientation_sign;
            mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component, &orientation_sign);
            const constraint_kform_spec_t element_spec = {
                .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
            size_t element_dof_count;
            result = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
            if (result != CONSTRAINT_SUCCESS)
                return result;
            if (entries_per_row > SIZE_MAX - element_dof_count)
                return CONSTRAINT_SIZE_OVERFLOW;
            entries_per_row += element_dof_count;
        }
        if (test_dof_count > 0 && entries_per_row > SIZE_MAX / test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        const size_t component_entries = test_dof_count * entries_per_row;
        if (entry_count > SIZE_MAX - component_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        entry_count += component_entries;
    }
    *out_row_count = row_count;
    *out_entry_count = entry_count;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_physical_side_required(const constraint_kform_spec_t *const test_spec,
                                                      const constraint_element_side_t *const side,
                                                      size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !side)
        return CONSTRAINT_INVALID_ARGUMENT;
    return constraint_physical_side_counts(test_spec, side, out_row_count, out_entry_count);
}

static void decode_component_dof(const unsigned ndim, const unsigned order, const basis_spec_t basis[const static ndim],
                                 const unsigned component, size_t dof, unsigned digits[const static ndim])
{
    uint8_t axes[order == 0 ? 1 : order];
    component_axes(ndim, order, component, axes);
    for (unsigned idim = ndim; idim > 0; --idim)
    {
        const bool active = component_has_axis(order, axes, idim - 1);
        const size_t dimension_size = (size_t)basis[idim - 1].order + (active ? 0 : 1);
        digits[idim - 1] = (unsigned)(dof % dimension_size);
        dof /= dimension_size;
    }
}

static double evaluate_basis_value(const basis_spec_t spec, const unsigned index, const double x)
{
    const unsigned order = index == 0 && spec.order == 0 ? 0 : spec.order;
    if (order == 0)
        return 1.0;

    double values[order + 1];
    double work[order + 1];
    basis_compute_at_point_prepare(spec.type, order, work);
    basis_compute_at_point_values(spec.type, order, 1, &x, values, work);
    return values[index];
}

static constraint_status_t quadrature_total_count(const unsigned ndim, const constraint_quadrature_t *quadrature,
                                                  size_t *const out_count)
{
    if (ndim != 0 && !quadrature)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t count = 1;
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        if (quadrature[idim].count == 0 || !quadrature[idim].nodes || !quadrature[idim].weights)
            return CONSTRAINT_INVALID_ARGUMENT;
        if (count > SIZE_MAX / quadrature[idim].count)
            return CONSTRAINT_SIZE_OVERFLOW;
        count *= quadrature[idim].count;
    }
    *out_count = count;
    return CONSTRAINT_SUCCESS;
}

static double trace_basis_product_at_point(const constraint_kform_spec_t *test_spec,
                                           const constraint_element_side_t *side, unsigned test_component,
                                           const unsigned test_digits[const static test_spec->ndim],
                                           unsigned element_component,
                                           const unsigned element_digits[const static side->ndim],
                                           const double face_nodes[const static test_spec->ndim]);

static double trace_inner_product(const constraint_kform_spec_t *const test_spec,
                                  const constraint_element_side_t *const side,
                                  const constraint_quadrature_t *quadrature, const unsigned test_component,
                                  const unsigned test_digits[const static test_spec->ndim],
                                  const unsigned element_component,
                                  const unsigned element_digits[const static side->ndim])
{
    const unsigned face_dim = test_spec->ndim;
    size_t quadrature_count;
    const constraint_status_t quadrature_status = quadrature_total_count(face_dim, quadrature, &quadrature_count);
    if (quadrature_status != CONSTRAINT_SUCCESS)
        return 0.0;
    double result = 0;
    double face_nodes[face_dim == 0 ? 1 : face_dim];
    for (size_t point = 0; point < quadrature_count; ++point)
    {
        size_t remaining = point;
        double weight = 1.0;
        for (unsigned idim = face_dim; idim > 0; --idim)
        {
            const unsigned face_axis = idim - 1;
            const unsigned index = (unsigned)(remaining % quadrature[face_axis].count);
            remaining /= quadrature[face_axis].count;
            face_nodes[face_axis] = quadrature[face_axis].nodes[index];
            weight *= quadrature[face_axis].weights[index];
        }
        result += weight * trace_basis_product_at_point(test_spec, side, test_component, test_digits, element_component,
                                                        element_digits, face_nodes);
    }
    return result;
}

static double element_trace_basis_value(const unsigned face_dim, const constraint_element_side_t *const side,
                                        const unsigned order, const unsigned element_component,
                                        const unsigned element_digits[const static side->ndim],
                                        const double face_nodes[const static face_dim == 0 ? 1 : face_dim])
{
    const unsigned fixed_count = side->ndim - face_dim;
    uint8_t element_axes[order == 0 ? 1 : order];
    component_axes(side->ndim, order, element_component, element_axes);

    double element_coordinates[side->ndim];
    for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
        element_coordinates[(unsigned)(side->orientation[fixed_axis] < 0 ? -side->orientation[fixed_axis]
                                                                         : side->orientation[fixed_axis]) -
                            1] = side->orientation[fixed_axis] < 0 ? -1.0 : 1.0;

    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = side->orientation[fixed_count + face_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        element_coordinates[element_axis] = mapping < 0 ? -face_nodes[face_axis] : face_nodes[face_axis];
    }

    double value = 1.0;
    for (unsigned element_axis = 0; element_axis < side->ndim; ++element_axis)
    {
        const bool element_active = component_has_axis(order, element_axes, element_axis);
        const unsigned element_order = side->basis_specs[element_axis].order - (element_active ? 1 : 0);
        value *=
            evaluate_basis_value((basis_spec_t){.type = side->basis_specs[element_axis].type, .order = element_order},
                                 element_digits[element_axis], element_coordinates[element_axis]);
    }
    return value;
}

static double trace_basis_product_at_point(const constraint_kform_spec_t *const test_spec,
                                           const constraint_element_side_t *const side, const unsigned test_component,
                                           const unsigned test_digits[const static test_spec->ndim],
                                           const unsigned element_component,
                                           const unsigned element_digits[const static side->ndim],
                                           const double face_nodes[const static test_spec->ndim])
{
    const unsigned face_dim = test_spec->ndim;
    uint8_t test_axes[test_spec->order == 0 ? 1 : test_spec->order];
    component_axes(face_dim, test_spec->order, test_component, test_axes);

    double value =
        element_trace_basis_value(face_dim, side, test_spec->order, element_component, element_digits, face_nodes);
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const bool test_active = component_has_axis(test_spec->order, test_axes, face_axis);
        const unsigned test_order = test_spec->basis_specs[face_axis].order - (test_active ? 1 : 0);
        value *=
            evaluate_basis_value((basis_spec_t){.type = test_spec->basis_specs[face_axis].type, .order = test_order},
                                 test_digits[face_axis], face_nodes[face_axis]);
    }
    return value;
}

static double trace_pullback_dot(const constraint_trace_pullback_t *const pullback, const unsigned first_component,
                                 const unsigned second_component, const unsigned physical_component_count,
                                 const size_t point_count, const size_t point)
{
    const double *const first =
        pullback->values + ((size_t)first_component * physical_component_count * point_count + point);
    const double *const second =
        pullback->values + ((size_t)second_component * physical_component_count * point_count + point);
    double result = 0.0;
    for (unsigned physical_component = 0; physical_component < physical_component_count; ++physical_component)
    {
        result += first[(size_t)physical_component * point_count] * second[(size_t)physical_component * point_count];
    }
    return result;
}

constraint_status_t constraint_physical_assemble(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t sides[const static 2],
    const constraint_face_quadrature_t quadrature[const static 2], const double *const surface_weights[const static 2],
    const constraint_trace_pullback_t pullbacks[const static 2], const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !surface_weights[0] || !surface_weights[1])
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t row_count;
    size_t required_entries;
    constraint_status_t status = constraint_physical_counts(test_spec, sides, &row_count, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t required_offsets;
    status = constraint_rows_required_offset_count(row_count, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    if (test_spec->order != 0)
    {
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            if (!pullbacks[side_index].values || pullbacks[side_index].physical_component_count == 0 ||
                pullbacks[side_index].point_count == 0)
                return CONSTRAINT_INVALID_ARGUMENT;
            size_t side_point_count;
            status =
                quadrature_total_count(quadrature[side_index].ndim, quadrature[side_index].axes, &side_point_count);
            if (status != CONSTRAINT_SUCCESS || side_point_count != quadrature[side_index].point_count ||
                side_point_count != pullbacks[side_index].point_count)
                return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
        }
    }
    else
    {
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            size_t side_point_count;
            status =
                quadrature_total_count(quadrature[side_index].ndim, quadrature[side_index].axes, &side_point_count);
            if (status != CONSTRAINT_SUCCESS || side_point_count != quadrature[side_index].point_count)
                return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
        }
    }

    size_t test_component_count;
    status = constraint_kform_component_count(test_spec, &test_component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t component_offsets[test_component_count + 1];
    status = constraint_kform_component_offsets(test_spec, test_component_count + 1, component_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < test_component_count; ++test_component)
    {
        uint8_t test_axes[test_spec->order == 0 ? 1 : test_spec->order];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        const size_t test_dof_count = component_offsets[test_component + 1] - component_offsets[test_component];
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_digits[test_spec->ndim == 0 ? 1 : test_spec->ndim];
            decode_component_dof(test_spec->ndim, test_spec->order, test_spec->basis_specs, test_component, test_dof,
                                 test_digits);
            for (unsigned side_index = 0; side_index < 2; ++side_index)
            {
                const constraint_element_side_t *const side = sides + side_index;
                const constraint_trace_pullback_t *const pullback = pullbacks + side_index;
                const unsigned face_component_count =
                    combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
                unsigned test_element_component = 0;
                int test_orientation_sign = 1;
                if (test_spec->order != 0)
                    mapped_component(side, test_spec->ndim, test_spec->order, test_axes, &test_element_component,
                                     &test_orientation_sign);

                for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
                {
                    uint8_t face_axes[test_spec->order == 0 ? 1 : test_spec->order];
                    component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
                    unsigned element_component;
                    int orientation_sign;
                    mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component,
                                     &orientation_sign);
                    const constraint_kform_spec_t element_spec = {
                        .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                    size_t element_dof_count;
                    constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                    for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                    {
                        unsigned element_digits[side->ndim];
                        decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component,
                                             element_dof, element_digits);
                        double coefficient = 0.0;
                        size_t point_count;
                        status = quadrature_total_count(quadrature[side_index].ndim, quadrature[side_index].axes,
                                                        &point_count);
                        if (status != CONSTRAINT_SUCCESS)
                            return status;
                        for (size_t point = 0; point < point_count; ++point)
                        {
                            size_t remaining = point;
                            double quadrature_weight = 1.0;
                            double face_nodes[test_spec->ndim == 0 ? 1 : test_spec->ndim];
                            for (unsigned idim = test_spec->ndim; idim > 0; --idim)
                            {
                                const unsigned face_axis = idim - 1;
                                const unsigned node_index =
                                    (unsigned)(remaining % quadrature[side_index].axes[face_axis].count);
                                remaining /= quadrature[side_index].axes[face_axis].count;
                                face_nodes[face_axis] = quadrature[side_index].axes[face_axis].nodes[node_index];
                                quadrature_weight *= quadrature[side_index].axes[face_axis].weights[node_index];
                            }
                            double pullback_factor = 1.0;
                            if (test_spec->order != 0)
                            {
                                if (test_element_component >=
                                        combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order) ||
                                    element_component >=
                                        combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order))
                                    return CONSTRAINT_INVALID_ARGUMENT;
                                pullback_factor =
                                    trace_pullback_dot(pullback, test_element_component, element_component,
                                                       pullback->physical_component_count, point_count, point);
                            }
                            coefficient += quadrature_weight * surface_weights[side_index][point] * pullback_factor *
                                           trace_basis_product_at_point(test_spec, side, test_component, test_digits,
                                                                        element_component, element_digits, face_nodes);
                        }
                        entries[entry++] = (constraint_entry_t){
                            .side = (uint8_t)side_index,
                            .component = element_component,
                            .local_dof = element_dof,
                            .coefficient = (side_index == 0 ? 1.0 : -1.0) * (double)test_orientation_sign *
                                           (double)orientation_sign * coefficient,
                        };
                    }
                }
            }
            row_offsets[row + 1] = entry;
        }
    }

    *out_row_count = row_count;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_physical_batch_required(
    const constraint_kform_spec_t *const test_spec, const size_t item_count,
    const constraint_physical_batch_item_t items[const static item_count], size_t *const out_row_count,
    size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    size_t total_rows = 0;
    size_t total_entries = 0;
    for (size_t item = 0; item < item_count; ++item)
    {
        if (!items[item].sides || !items[item].quadrature || !items[item].surface_weights[0] ||
            !items[item].surface_weights[1])
            return CONSTRAINT_INVALID_ARGUMENT;
        if (test_spec && test_spec->order != 0 && !items[item].pullbacks)
            return CONSTRAINT_INVALID_ARGUMENT;
        size_t item_rows;
        size_t item_entries;
        const constraint_status_t status =
            constraint_physical_required(test_spec, items[item].sides, &item_rows, &item_entries);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        if (total_rows > SIZE_MAX - item_rows || total_entries > SIZE_MAX - item_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        total_rows += item_rows;
        total_entries += item_entries;
    }
    if (item_count == 0)
    {
        size_t ignored;
        const constraint_status_t status = constraint_kform_component_count(test_spec, &ignored);
        if (status != CONSTRAINT_SUCCESS)
            return status;
    }
    *out_row_count = total_rows;
    *out_entry_count = total_entries;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_physical_batch_assemble(
    const constraint_kform_spec_t *const test_spec, const size_t item_count,
    const constraint_physical_batch_item_t items[const static item_count], const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    size_t required_rows;
    size_t required_entries;
    constraint_status_t status =
        constraint_physical_batch_required(test_spec, item_count, items, &required_rows, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t required_offsets;
    status = constraint_rows_required_offset_count(required_rows, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (size_t item = 0; item < item_count; ++item)
    {
        size_t item_rows;
        size_t item_entries;
        status = constraint_physical_assemble(test_spec, items[item].sides, items[item].quadrature,
                                              items[item].surface_weights, items[item].pullbacks,
                                              row_offset_capacity - row, row_offsets + row, entry_capacity - entry,
                                              entries + entry, &item_rows, &item_entries);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        row_offsets[row] = entry;
        for (size_t local_row = 1; local_row <= item_rows; ++local_row)
            row_offsets[row + local_row] += entry;
        row += item_rows;
        entry += item_entries;
    }
    *out_row_count = row;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_physical_side_assemble(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t *const side,
    const constraint_face_quadrature_t *const quadrature, const double *const surface_weights,
    const constraint_trace_pullback_t *const pullback, const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !side || !quadrature || !surface_weights)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t row_count;
    size_t required_entries;
    constraint_status_t status = constraint_physical_side_counts(test_spec, side, &row_count, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t required_offsets;
    status = constraint_rows_required_offset_count(row_count, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t point_count;
    status = quadrature_total_count(quadrature->ndim, quadrature->axes, &point_count);
    if (status != CONSTRAINT_SUCCESS || point_count != quadrature->point_count)
        return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
    if (test_spec->order != 0 && (!pullback || !pullback->values || pullback->physical_component_count == 0 ||
                                  pullback->point_count != point_count))
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t component_count;
    status = constraint_kform_component_count(test_spec, &component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t component_offsets[component_count + 1];
    status = constraint_kform_component_offsets(test_spec, component_count + 1, component_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        uint8_t test_axes[test_spec->order == 0 ? 1 : test_spec->order];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        const size_t test_dof_count = component_offsets[test_component + 1] - component_offsets[test_component];
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_digits[test_spec->ndim == 0 ? 1 : test_spec->ndim];
            decode_component_dof(test_spec->ndim, test_spec->order, test_spec->basis_specs, test_component, test_dof,
                                 test_digits);
            unsigned test_element_component = 0;
            int test_orientation_sign = 1;
            if (test_spec->order != 0)
                mapped_component(side, test_spec->ndim, test_spec->order, test_axes, &test_element_component,
                                 &test_orientation_sign);

            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                uint8_t face_axes[test_spec->order == 0 ? 1 : test_spec->order];
                component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component,
                                 &orientation_sign);
                const constraint_kform_spec_t element_spec = {
                    .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                size_t element_dof_count;
                status = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                if (status != CONSTRAINT_SUCCESS)
                    return status;
                for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                {
                    unsigned element_digits[side->ndim];
                    decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component,
                                         element_dof, element_digits);
                    double coefficient = 0.0;
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        size_t remaining = point;
                        double quadrature_weight = 1.0;
                        double face_nodes[test_spec->ndim == 0 ? 1 : test_spec->ndim];
                        for (unsigned idim = test_spec->ndim; idim > 0; --idim)
                        {
                            const unsigned face_axis = idim - 1;
                            const unsigned node_index = (unsigned)(remaining % quadrature->axes[face_axis].count);
                            remaining /= quadrature->axes[face_axis].count;
                            face_nodes[face_axis] = quadrature->axes[face_axis].nodes[node_index];
                            quadrature_weight *= quadrature->axes[face_axis].weights[node_index];
                        }
                        double pullback_factor = 1.0;
                        if (test_spec->order != 0)
                        {
                            if (test_element_component >=
                                    combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order) ||
                                element_component >=
                                    combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order))
                                return CONSTRAINT_INVALID_ARGUMENT;
                            pullback_factor =
                                trace_pullback_dot(pullback, test_element_component, element_component,
                                                   pullback->physical_component_count, point_count, point);
                        }
                        coefficient += quadrature_weight * surface_weights[point] * pullback_factor *
                                       trace_basis_product_at_point(test_spec, side, test_component, test_digits,
                                                                    element_component, element_digits, face_nodes);
                    }
                    entries[entry++] = (constraint_entry_t){
                        .side = 0,
                        .component = element_component,
                        .local_dof = element_dof,
                        .coefficient = (double)test_orientation_sign * (double)orientation_sign * coefficient,
                    };
                }
            }
            row_offsets[row + 1] = entry;
        }
    }
    *out_row_count = row_count;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_physical_side_assemble_precomputed(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t *const side,
    const constraint_face_quadrature_t *const quadrature, const double *const surface_weights,
    const constraint_trace_pullback_t *const pullback, const constraint_trace_basis_values_t *const test_basis,
    const constraint_trace_basis_values_t *const element_basis, const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !side || !quadrature || !surface_weights)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t required_rows;
    size_t required_entries;
    constraint_status_t status = constraint_physical_side_counts(test_spec, side, &required_rows, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t required_offsets;
    status = constraint_rows_required_offset_count(required_rows, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t point_count;
    status = quadrature_total_count(quadrature->ndim, quadrature->axes, &point_count);
    if (status != CONSTRAINT_SUCCESS || quadrature->point_count != point_count)
        return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
    if (test_spec->order != 0 && (!pullback || !pullback->values || pullback->physical_component_count == 0 ||
                                  pullback->point_count != point_count))
        return CONSTRAINT_INVALID_ARGUMENT;

    const constraint_kform_spec_t element_spec = {
        .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
    status = validate_trace_basis_values(test_spec, test_basis, point_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    status = validate_trace_basis_values(&element_spec, element_basis, point_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t test_component_count;
    status = constraint_kform_component_count(test_spec, &test_component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    const size_t *const test_offsets = test_basis->component_offsets;
    const size_t *const element_offsets = element_basis->component_offsets;
    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < test_component_count; ++test_component)
    {
        const size_t test_dof_count = test_offsets[test_component + 1] - test_offsets[test_component];
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_element_component = 0;
            int test_orientation_sign = 1;
            status = mapped_component_for_index(side, test_spec->ndim, test_spec->order, test_component,
                                                &test_element_component, &test_orientation_sign);
            if (status != CONSTRAINT_SUCCESS)
                return status;
            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                unsigned element_component;
                int orientation_sign;
                status = mapped_component_for_index(side, test_spec->ndim, test_spec->order, face_component,
                                                    &element_component, &orientation_sign);
                if (status != CONSTRAINT_SUCCESS)
                    return status;
                const size_t element_dof_count =
                    element_offsets[element_component + 1] - element_offsets[element_component];
                for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                {
                    double coefficient = 0.0;
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        size_t remaining = point;
                        double quadrature_weight = 1.0;
                        for (unsigned idim = quadrature->ndim; idim > 0; --idim)
                        {
                            const unsigned axis = idim - 1;
                            const unsigned node = (unsigned)(remaining % quadrature->axes[axis].count);
                            remaining /= quadrature->axes[axis].count;
                            quadrature_weight *= quadrature->axes[axis].weights[node];
                        }
                        double pullback_factor = 1.0;
                        if (test_spec->order != 0)
                            pullback_factor =
                                trace_pullback_dot(pullback, test_element_component, element_component,
                                                   pullback->physical_component_count, point_count, point);
                        const double test_value = test_basis->values[test_offsets[test_component] * point_count +
                                                                     point * test_dof_count + test_dof];
                        const double element_value =
                            element_basis->values[element_offsets[element_component] * point_count +
                                                  point * element_dof_count + element_dof];
                        coefficient +=
                            quadrature_weight * surface_weights[point] * pullback_factor * test_value * element_value;
                    }
                    entries[entry++] = (constraint_entry_t){
                        .side = 0,
                        .component = element_component,
                        .local_dof = element_dof,
                        .coefficient = (double)test_orientation_sign * (double)orientation_sign * coefficient,
                    };
                }
            }
            row_offsets[row + 1] = entry;
        }
    }
    if (row_offset_capacity < row + 1)
        return CONSTRAINT_INSUFFICIENT_STORAGE;
    if (entry_capacity < entry)
        return CONSTRAINT_INSUFFICIENT_STORAGE;
    *out_row_count = row;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_physical_side_load(const constraint_kform_spec_t *const test_spec,
                                                  const constraint_element_side_t *const side,
                                                  const constraint_face_quadrature_t *const quadrature,
                                                  const double *const datum_values, const size_t value_count,
                                                  const double *const surface_weights,
                                                  double values[const static value_count])
{
    if (!test_spec || !side || !quadrature || !datum_values)
        return CONSTRAINT_INVALID_ARGUMENT;

    constraint_status_t status =
        constraint_reference_validate(test_spec, (const constraint_element_side_t[2]){*side, *side});
    if (status != CONSTRAINT_SUCCESS)
        return status;
    // The load is defined on codimension-1 faces: exactly one fixed normal axis.
    if (side->ndim != test_spec->ndim + 1)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t point_count;
    status = quadrature_total_count(quadrature->ndim, quadrature->axes, &point_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (quadrature->ndim != test_spec->ndim || point_count != quadrature->point_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    const constraint_kform_spec_t element_spec = {
        .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
    size_t element_component_count;
    status = constraint_kform_component_count(&element_spec, &element_component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t element_offsets[element_component_count + 1];
    status = constraint_kform_component_offsets(&element_spec, element_component_count + 1, element_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (element_offsets[element_component_count] != value_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    // The datum is an element-frame k-form with k = test_spec->order + 1, given
    // as its C(n, k) physical components sampled at the canonical face points:
    // datum_values[component * point_count + point]. For each face (k-1)-form
    // component J with element-frame axes J_e, the paired datum component is
    // I = J_e U {fixed_axis} and the sign carries the count of J_e axes below
    // the fixed normal axis; at k = n this reduces to the previous
    // sigma_out = side * (-1)^a formula.
    const int8_t fixed_mapping = side->orientation[0];
    const unsigned fixed_axis = (unsigned)(fixed_mapping < 0 ? -fixed_mapping : fixed_mapping) - 1;
    const double side_sign = fixed_mapping < 0 ? -1.0 : 1.0;
    for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
    {
        uint8_t face_axes[test_spec->order == 0 ? 1 : test_spec->order];
        component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
        unsigned element_component;
        int orientation_sign;
        status =
            mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component, &orientation_sign);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        size_t element_dof_count;
        status = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        uint8_t element_axes[test_spec->order == 0 ? 1 : test_spec->order];
        component_axes(side->ndim, test_spec->order, element_component, element_axes);
        unsigned exponent_below_fixed = 0;
        for (unsigned i = 0; i < test_spec->order; ++i)
            exponent_below_fixed += element_axes[i] < fixed_axis ? 1u : 0u;
        uint8_t datum_axes[test_spec->order == 0 ? 1 : test_spec->order + 1];
        for (unsigned i = 0; i < test_spec->order; ++i)
            datum_axes[i] = element_axes[i];
        datum_axes[test_spec->order] = (uint8_t)fixed_axis;
        for (unsigned i = test_spec->order; i > 0 && datum_axes[i - 1] > datum_axes[i]; --i)
        {
            const uint8_t tmp = datum_axes[i - 1];
            datum_axes[i - 1] = datum_axes[i];
            datum_axes[i] = tmp;
        }
        const unsigned datum_component = combination_get_index(side->ndim, test_spec->order + 1, datum_axes);
        const double sign = side_sign * (double)orientation_sign * (exponent_below_fixed % 2 == 0 ? 1.0 : -1.0);
        for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
        {
            unsigned element_digits[side->ndim];
            decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component, element_dof,
                                 element_digits);
            double coefficient = 0.0;
            for (size_t point = 0; point < point_count; ++point)
            {
                size_t remaining = point;
                double quadrature_weight = 1.0;
                double face_nodes[test_spec->ndim == 0 ? 1 : test_spec->ndim];
                for (unsigned idim = test_spec->ndim; idim > 0; --idim)
                {
                    const unsigned face_axis = idim - 1;
                    const unsigned node_index = (unsigned)(remaining % quadrature->axes[face_axis].count);
                    remaining /= quadrature->axes[face_axis].count;
                    face_nodes[face_axis] = quadrature->axes[face_axis].nodes[node_index];
                    quadrature_weight *= quadrature->axes[face_axis].weights[node_index];
                }
                const double weight_total =
                    surface_weights ? quadrature_weight * surface_weights[point] : quadrature_weight;
                coefficient += weight_total * datum_values[datum_component * point_count + point] *
                               element_trace_basis_value(test_spec->ndim, side, test_spec->order, element_component,
                                                         element_digits, face_nodes);
            }
            values[element_offsets[element_component] + element_dof] += sign * coefficient;
        }
    }
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_reference_assemble(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t sides[const static 2],
    const constraint_quadrature_t *quadrature, const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t row_count;
    size_t required_entries;
    constraint_status_t status = constraint_reference_counts(test_spec, sides, &row_count, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t required_offsets;
    status = constraint_rows_required_offset_count(row_count, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t quadrature_count;
    status = quadrature_total_count(test_spec->ndim, quadrature, &quadrature_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    (void)quadrature_count;

    size_t component_count;
    status = constraint_kform_component_count(test_spec, &component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t component_offsets[component_count + 1];
    status = constraint_kform_component_offsets(test_spec, component_count + 1, component_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        uint8_t test_axes[test_spec->order == 0 ? 1 : test_spec->order];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        const size_t test_dof_count = component_offsets[test_component + 1] - component_offsets[test_component];
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_digits[test_spec->ndim == 0 ? 1 : test_spec->ndim];
            decode_component_dof(test_spec->ndim, test_spec->order, test_spec->basis_specs, test_component, test_dof,
                                 test_digits);
            for (unsigned side_index = 0; side_index < 2; ++side_index)
            {
                const constraint_element_side_t *const side = sides + side_index;
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, test_spec->ndim, test_spec->order, test_axes, &element_component,
                                 &orientation_sign);
                const constraint_kform_spec_t element_spec = {
                    .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                size_t element_dof_count;
                constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                {
                    unsigned element_digits[side->ndim];
                    decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component,
                                         element_dof, element_digits);
                    entries[entry++] = (constraint_entry_t){
                        .side = (uint8_t)side_index,
                        .component = element_component,
                        .local_dof = element_dof,
                        .coefficient = (side_index == 0 ? 1.0 : -1.0) * (double)orientation_sign *
                                       trace_inner_product(test_spec, side, quadrature, test_component, test_digits,
                                                           element_component, element_digits),
                    };
                }
            }
            row_offsets[row + 1] = entry;
        }
    }

    *out_row_count = row_count;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_kform_component_count(const constraint_kform_spec_t *const spec, size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    const constraint_status_t status = validate_kform_spec(spec);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    *out_count = combination_total_count((uint8_t)spec->ndim, (uint8_t)spec->order);
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_kform_component_dof_count(const constraint_kform_spec_t *const spec,
                                                         const unsigned component, size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    const constraint_status_t status = validate_kform_spec(spec);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    const size_t component_count = combination_total_count((uint8_t)spec->ndim, (uint8_t)spec->order);
    if ((size_t)component >= component_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    uint8_t component_axes[spec->order == 0 ? 1 : spec->order];
    combination_set_to_index((uint8_t)spec->ndim, (uint8_t)spec->order, component_axes, component);

    size_t dof_count = 1;
    for (unsigned idim = 0, iaxis = 0; idim < spec->ndim; ++idim)
    {
        const bool active = iaxis < spec->order && component_axes[iaxis] == idim;
        const size_t dimension_size = (size_t)spec->basis_specs[idim].order + (active ? 0 : 1);
        if (dimension_size != 0 && dof_count > SIZE_MAX / dimension_size)
            return CONSTRAINT_SIZE_OVERFLOW;
        dof_count *= dimension_size;
        if (active)
            ++iaxis;
    }

    *out_count = dof_count;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_kform_component_offsets(const constraint_kform_spec_t *const spec,
                                                       const size_t offset_count,
                                                       size_t offsets[const static offset_count])
{
    size_t component_count;
    constraint_status_t status = constraint_kform_component_count(spec, &component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (offset_count < component_count + 1)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t offset = 0;
    offsets[0] = 0;
    for (size_t component = 0; component < component_count; ++component)
    {
        size_t dof_count;
        status = constraint_kform_component_dof_count(spec, (unsigned)component, &dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        if (offset > SIZE_MAX - dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        offset += dof_count;
        offsets[component + 1] = offset;
    }
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_rows_required_offset_count(const size_t row_count, size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (row_count == SIZE_MAX)
        return CONSTRAINT_SIZE_OVERFLOW;
    *out_count = row_count + 1;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_rows_required_entry_capacity(const size_t row_count, const size_t entries_per_row,
                                                            size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (entries_per_row != 0 && row_count > SIZE_MAX / entries_per_row)
        return CONSTRAINT_SIZE_OVERFLOW;
    *out_count = row_count * entries_per_row;
    return CONSTRAINT_SUCCESS;
}

constraint_status_t constraint_rows_validate(const constraint_rows_view_t view)
{
    if (view.row_count != 0 && !view.row_offsets)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (view.entry_count != 0 && !view.entries)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (view.row_offsets && view.row_offsets[0] != 0)
        return CONSTRAINT_INVALID_ARGUMENT;

    for (size_t row = 0; row < view.row_count; ++row)
    {
        if (view.row_offsets[row] > view.row_offsets[row + 1])
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    if (view.row_offsets && view.row_offsets[view.row_count] != view.entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    for (size_t entry = 0; entry < view.entry_count; ++entry)
    {
        if (view.entries[entry].side > 1)
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    return CONSTRAINT_SUCCESS;
}
