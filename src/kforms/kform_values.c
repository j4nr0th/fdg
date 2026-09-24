#include "kform_values.h"

#include "../common/common_defines.h"
#include "../kforms/kform_types.h"

void kform_component_basis_values(const unsigned ndim, const basis_spec_t basis[static ndim], const unsigned order,
                                  const uint8_t component_axes[], const kform_trace_axis_t axes[],
                                  multidim_iterator_t *const point_iter, const size_t point_count,
                                  double values[restrict])
{
    const size_t dof_count = kform_component_dof_count(ndim, basis, order, component_axes);

    // init_dim takes dimensions in ascending order while free axes carry their stride slots (point tensor, last axis
    // fastest) in arbitrary order, so locate each slot's owning axis first.
    unsigned point_dims = 0;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        point_dims += axes[axis].kind == KFORM_TRACE_AXIS_FREE;
    }
    if (point_dims == 0)
    {
        multidim_iterator_init(point_iter, 0, (const size_t[1]){0});
    }
    for (unsigned slot = 0; slot < point_dims; ++slot)
    {
        unsigned owner = ndim;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (axes[axis].kind == KFORM_TRACE_AXIS_FREE && axes[axis].free.stride_slot == slot)
            {
                owner = axis;
                break;
            }
        }
        ASSERT(owner < ndim, "Free axes must cover stride slots 0..%u exactly once.", point_dims - 1);
        multidim_iterator_init_dim(point_iter, slot, axes[owner].free.rule_size);
    }
    ASSERT(multidim_iterator_total_size(point_iter) == point_count,
           "Point count does not match the stride-slot tensor (%zu vs %zu).", multidim_iterator_total_size(point_iter),
           point_count);
    const size_t *const point_digits = multidim_iterator_offsets(point_iter);

    for (size_t point = 0; point < point_count; ++point)
    {
        double *const point_values = values + point * dof_count;
        point_values[0] = 1.0;
        size_t current_count = 1;
        unsigned component_axis = 0;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            const bool active = order != 0 && component_axis < order && component_axes[component_axis] == axis;
            if (active)
                component_axis += 1;
            const kform_trace_axis_t *const axis_desc = &axes[axis];
            const bool fixed_axis = axis_desc->kind == KFORM_TRACE_AXIS_FIXED;
            size_t integration_index;
            if (fixed_axis)
            {
                integration_index = axis_desc->mirror ? 0u : 1u;
            }
            else
            {
                integration_index = point_digits[axis_desc->free.stride_slot];
                if (axis_desc->mirror)
                    integration_index = (size_t)axis_desc->free.rule_size - 1 - integration_index;
            }
            const basis_endpoint_set_t *const endpoint =
                fixed_axis ? (active ? axis_desc->fixed.endpoint_lower : axis_desc->fixed.endpoint) : NULL;
            const basis_set_t *const nodes =
                fixed_axis ? NULL : (active ? axis_desc->free.nodes_lower : axis_desc->free.nodes);
            const size_t basis_dim = (size_t)(endpoint ? endpoint->spec.order : nodes->spec.order) + 1;
            for (size_t previous = current_count; previous > 0; --previous)
            {
                const double previous_value = point_values[previous - 1];
                for (size_t basis_index = basis_dim; basis_index > 0; --basis_index)
                {
                    const double basis_value =
                        endpoint ? basis_endpoint_values(endpoint, (unsigned)integration_index)[basis_index - 1]
                                 : basis_set_basis_values(nodes, (unsigned)(basis_index - 1))[integration_index];
                    point_values[(previous - 1) * basis_dim + basis_index - 1] = previous_value * basis_value;
                }
            }
            current_count *= basis_dim;
        }
        ASSERT(current_count == dof_count, "Tensor-product basis count mismatch (%zu vs %zu).", current_count,
               dof_count);
        if (point_dims > 0)
        {
            multidim_iterator_advance(point_iter, point_dims - 1, 1);
        }
    }
}

void kform_inner_product_block(const size_t point_count, const size_t dofs_left, const size_t dofs_right,
                               const double basis_values_left[restrict], const double basis_values_right[restrict],
                               const double weights[restrict], const size_t row0, const size_t col0,
                               const size_t row_stride, double matrix[restrict])
{
    for (size_t left = 0; left < dofs_left; ++left)
        for (size_t right = 0; right < dofs_right; ++right)
            matrix[(row0 + left) * row_stride + col0 + right] = 0.0;

    for (size_t point = 0; point < point_count; ++point)
    {
        const double *const values_left = basis_values_left + point * dofs_left;
        const double *const values_right = basis_values_right + point * dofs_right;
        const double weight = weights[point];
        for (size_t left = 0; left < dofs_left; ++left)
        {
            double *const matrix_row = matrix + (row0 + left) * row_stride + col0;
            const double weighted_left = weight * values_left[left];
#pragma omp simd
            for (size_t right = 0; right < dofs_right; ++right)
                matrix_row[right] += weighted_left * values_right[right];
        }
    }
}
