#pragma once
#include <stddef.h>
#include <stdint.h>

#include "../basis/basis_set.h"

/**
 * @brief Table of k-form basis values, component blocks point-major.
 *
 * Component `c` occupies `values[component_offsets[c] * point_count .. +
 * dof_count * point_count)`, where `dof_count` is the component's local DoF
 * count; an entry is indexed as
 * `values[component_offsets[c] * point_count + point * dof_count + dof]`.
 */
typedef struct
{
    size_t component_count;
    size_t point_count;
    const size_t *component_offsets; // component_count + 1 entries
    double *values;
} kform_values_table_t;

/**
 * @brief Per-axis value source for building a component's basis values.
 *
 * @todo Some of these could be clarified and streamlined. For example, this
 * holds data for both fixed and non-fixed axes, which can be separated by
 * turning this into a tagged union or just have a bool to specify if it is fixed or not.
 *
 * Free axes (tangent axes of a trace) read the basis sets evaluated on the
 * axis quadrature rule; fixed axes (normal axes of a trace) read cached
 * endpoint values. A negative orientation reverses the node index, which is
 * exact for the symmetric Gauss rules used by this library because the
 * mirrored node sits at the negated coordinate.
 */
typedef struct
{
    const basis_set_t *nodes;             // Free axis: full-order set (NULL when fixed).
    const basis_set_t *nodes_lower;       // Free axis: order-1 set for active covector axes.
    const basis_endpoint_set_t *endpoint; // Non-NULL marks a fixed axis.
    const basis_endpoint_set_t *endpoint_lower;
    unsigned end;         // Endpoint index for fixed axes: 0 = -1, 1 = +1.
    unsigned rule_size;   // Free axis: node count of the axis rule.
    unsigned stride_slot; // Which point_strides entry decodes this axis's node index.
    int mirror;           // Reverse the node index (negative orientation).
} kform_trace_axis_t;

/**
 * @brief Compute one k-form component's basis values over all points.
 *
 * The DoF enumeration multiplies the axes in order, so axis 0 is the slowest
 * index of the component's local DoF tuple. Active covector axes (those in
 * @p component_axes) read the order-1 sets.
 *
 * @param ndim Number of axes of the element (may exceed the point-space dim).
 * @param basis Basis specification of each axis.
 * @param order Order of the k-form.
 * @param component_axes Sorted covector axes of the component.
 * @param axes Value source of each axis.
 * @param point_strides Strides of the flat point tensor (last axis fastest,
 *                      matching #integration_spec_point_strides); indexed by
 *                      #kform_trace_axis_t::stride_slot.
 * @param point_count Number of tensor points.
 * @param values Output buffer with `kform_component_dof_count(ndim, basis,
 *               order, component_axes) * point_count` entries.
 */
void kform_component_basis_values(unsigned ndim, const basis_spec_t basis[static ndim], unsigned order,
                                  const uint8_t component_axes[], const kform_trace_axis_t axes[],
                                  const size_t point_strides[], size_t point_count, double values[restrict]);

/**
 * @brief Accumulate a dense weighted inner-product block.
 *
 * Computes `out[(row0 + i) * row_stride + col0 + j] += weights[p] *
 * left[p * dofs_left + i] * right[p * dofs_right + j]` over all points,
 * zero-initializing the block first.
 *
 * @param point_count Number of quadrature points.
 * @param dofs_left Number of DoFs of the left (row) factor.
 * @param dofs_right Number of DoFs of the right (column) factor.
 * @param left Left basis values, point-major.
 * @param right Right basis values, point-major.
 * @param weights Quadrature weight per point.
 * @param row0 First output row of the block.
 * @param col0 First output column of the block.
 * @param row_stride Row stride of the output matrix.
 * @param out Output matrix.
 */
void kform_inner_product_block(size_t point_count, size_t dofs_left, size_t dofs_right, const double *restrict left,
                               const double *restrict right, const double *restrict weights, size_t row0, size_t col0,
                               size_t row_stride, double *restrict out);
