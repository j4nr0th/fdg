#include "boundaries.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

size_t boundary_dof_values_work_size(unsigned ndim, const basis_spec_t basis[static ndim])
{
    size_t total = 1, max_order = 0;
    for (unsigned i = 0; i < ndim; ++i)
    {
        total *= basis[i].order + 1;
        max_order = max_order > basis[i].order ? max_order : basis[i].order;
    }
    return 2 * total + 2 * (max_order + 1);
}

void boundary_dof_values(unsigned ndim, const basis_spec_t basis[static ndim], const double values[restrict],
                         double work[restrict], unsigned bdim, const int8_t orientation[static ndim],
                         double boundary_values[restrict])
{
    CUTL_ASSERT(ndim <= UINT8_MAX, "Element dimension exceeds the supported maximum.");
    CUTL_ASSERT(bdim <= ndim, "Boundary dimension exceeds the element dimension.");

    size_t total = 1, max_order = 0;
    size_t sizes[UINT8_MAX];
    for (unsigned i = 0; i < ndim; ++i)
    {
        sizes[i] = basis[i].order + 1;
        total *= sizes[i];
        max_order = max_order > basis[i].order ? max_order : basis[i].order;
    }

    const unsigned ncontract = ndim - bdim;
    if (ncontract == 0)
    {
        memcpy(boundary_values, values, total * sizeof(*boundary_values));
        return;
    }

    // Two ping-pong regions for the intermediate tensors, the endpoint evaluation scratch at
    // the end of the array.
    double *const prep = work + 2 * total;
    double *const coeffs = prep + (max_order + 1);

    const double *restrict src = values;
    for (unsigned contracted = 0; contracted < ncontract; ++contracted)
    {
        // Decode the axis, negative orientation denotes the start of the axis.
        const int8_t axis_code = orientation[contracted];
        const unsigned axis = (unsigned)(axis_code < 0 ? -axis_code : axis_code) - 1;
        const bool at_the_end = axis_code > 0;
        CUTL_ASSERT(axis < ndim, "Orientation references an axis outside the element.");
        const basis_spec_t axis_basis = basis[axis];

        // Counts of the degrees of freedom before and after the contracted axis, with all
        // previously contracted axes collapsed to one. The axis strides follow from them,
        // iterating last axis fastest.
        size_t pre_count = 1, post_count = 1;
        for (unsigned i = 0; i < ndim; ++i)
        {
            if (i < axis)
            {
                pre_count *= sizes[i];
            }
            else if (i > axis)
            {
                post_count *= sizes[i];
            }
        }
        const size_t ndofs = axis_basis.order + 1;

        // The intermediates alternate between the two work regions, the final contraction
        // writes the trace straight to the output. Every region is a compact tensor of the
        // surviving axes, so the contracted axis slot is simply dropped.
        double *restrict dst =
            contracted + 1 == ncontract ? boundary_values : (contracted % 2 == 0 ? work : work + total);

        // Dispatch on the basis type: endpoint-node bases simply select a degree of freedom,
        // Legendre contracts with the endpoint values of its polynomials, and node-interior
        // bases evaluate their basis functions at the endpoint once per contraction.
        switch (axis_basis.type)
        {
        case BASIS_BERNSTEIN:
        case BASIS_LAGRANGE_UNIFORM:
        case BASIS_LAGRANGE_GAUSS_LOBATTO: {
            const size_t pick = at_the_end ? (size_t)(ndofs - 1) * post_count : 0;
            for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
            {
                for (size_t i_post = 0; i_post < post_count; ++i_post)
                {
                    const size_t line = i_pre * (ndofs * post_count) + i_post;
                    dst[i_pre * post_count + i_post] = src[line + pick];
                }
            }
            break;
        }

        case BASIS_LEGENDRE:
            for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
            {
                for (size_t i_post = 0; i_post < post_count; ++i_post)
                {
                    const size_t line = i_pre * (ndofs * post_count) + i_post;
                    double result = 0.0;
                    // The Legendre values are one at the end and alternate in sign at the start.
                    if (at_the_end)
                    {
                        for (unsigned i_dof = 0; i_dof < ndofs; ++i_dof)
                        {
                            result += src[line + (size_t)i_dof * post_count];
                        }
                    }
                    else
                    {
                        for (unsigned i_dof = 0; i_dof < ndofs; i_dof += 2)
                        {
                            result += src[line + (size_t)i_dof * post_count];
                        }
                        for (unsigned i_dof = 1; i_dof < ndofs; i_dof += 2)
                        {
                            result -= src[line + (size_t)i_dof * post_count];
                        }
                    }
                    dst[i_pre * post_count + i_post] = result;
                }
            }
            break;

        case BASIS_LAGRANGE_GAUSS:
        case BASIS_LAGRANGE_CHEBYSHEV_GAUSS: {
            const double endpoint = at_the_end ? 1.0 : -1.0;
            basis_compute_at_point_prepare(axis_basis.type, axis_basis.order, prep);
            basis_compute_at_point_values(axis_basis.type, axis_basis.order, 1, &endpoint, coeffs, prep);
            for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
            {
                for (size_t i_post = 0; i_post < post_count; ++i_post)
                {
                    const size_t line = i_pre * (ndofs * post_count) + i_post;
                    double result = 0.0;
                    for (unsigned i_dof = 0; i_dof < ndofs; ++i_dof)
                    {
                        result += coeffs[i_dof] * src[line + (size_t)i_dof * post_count];
                    }
                    dst[i_pre * post_count + i_post] = result;
                }
            }
            break;
        }

        default:
            CUTL_ASSERT(0, "Unsupported basis type.");
            break;
        }

        sizes[axis] = 1;
        src = dst;
    }
}

size_t boundary_integration_point_values_work_size(unsigned ndim, const integration_spec_t specs[static ndim],
                                                   unsigned n_components)
{
    size_t total = 1, max_nodes = 0;
    for (unsigned i = 0; i < ndim; ++i)
    {
        const size_t n_nodes = specs[i].order + 1;
        total *= n_nodes;
        max_nodes = max_nodes > n_nodes ? max_nodes : n_nodes;
    }
    return 2 * (size_t)n_components * total + 2 * max_nodes;
}

void boundary_integration_point_values(unsigned ndim, const integration_spec_t specs[static ndim],
                                       const double values[restrict], double work[restrict], unsigned bdim,
                                       const int8_t orientation[static ndim], unsigned n_components,
                                       double boundary_values[restrict])
{
    CUTL_ASSERT(ndim <= UINT8_MAX, "Element dimension exceeds the supported maximum.");
    CUTL_ASSERT(bdim <= ndim, "Boundary dimension exceeds the element dimension.");

    size_t total = 1, max_nodes = 0;
    size_t sizes[UINT8_MAX];
    for (unsigned i = 0; i < ndim; ++i)
    {
        sizes[i] = specs[i].order + 1;
        total *= sizes[i];
        max_nodes = max_nodes > sizes[i] ? max_nodes : sizes[i];
    }

    const unsigned ncontract = ndim - bdim;
    if (ncontract == 0)
    {
        memcpy(boundary_values, values, (size_t)n_components * total * sizeof(*boundary_values));
        return;
    }

    // Two ping-pong regions for the intermediate tensors, the endpoint evaluation scratch at
    // the end of the array.
    double *const prep = work + 2 * (size_t)n_components * total;
    double *const coeffs = prep + max_nodes;

    const double *restrict src = values;
    for (unsigned contracted = 0; contracted < ncontract; ++contracted)
    {
        // Decode the axis, negative orientation denotes the start of the axis.
        const int8_t axis_code = orientation[contracted];
        const unsigned axis = (unsigned)(axis_code < 0 ? -axis_code : axis_code) - 1;
        const bool at_the_end = axis_code > 0;
        CUTL_ASSERT(axis < ndim, "Orientation references an axis outside the element.");
        const integration_spec_t axis_spec = specs[axis];

        // Counts of the integration points before and after the fixed axis, with all
        // previously fixed axes collapsed to one. The axis strides follow from them,
        // iterating last axis fastest.
        size_t pre_count = 1, post_count = 1;
        for (unsigned i = 0; i < ndim; ++i)
        {
            if (i < axis)
            {
                pre_count *= sizes[i];
            }
            else if (i > axis)
            {
                post_count *= sizes[i];
            }
        }
        const size_t n_nodes = axis_spec.order + 1;

        // The intermediates alternate between the two work regions, the final extraction
        // writes the boundary values straight to the output. Every region is a compact
        // tensor of the surviving axes, so the fixed axis slot is simply dropped.
        double *restrict dst =
            contracted + 1 == ncontract ? boundary_values : (contracted % 2 == 0 ? work : work + n_components * total);

        // Dispatch on the rule type: Gauss-Lobatto nodes include the endpoints, so the
        // boundary values are a slice of points, while the interior Gauss-Legendre nodes
        // require evaluating the interpolant through them at the endpoint.
        switch (axis_spec.type)
        {
        case INTEGRATION_RULE_TYPE_GAUSS_LOBATTO: {
            const size_t pick = at_the_end ? (n_nodes - 1) * post_count : 0;
            for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
            {
                for (size_t i_post = 0; i_post < post_count; ++i_post)
                {
                    const double *restrict from = src + (i_pre * (n_nodes * post_count) + i_post + pick) * n_components;
                    double *restrict to = dst + (i_pre * post_count + i_post) * n_components;
                    for (unsigned component = 0; component < n_components; ++component)
                    {
                        to[component] = from[component];
                    }
                }
            }
            break;
        }

        case INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE: {
            const double endpoint = at_the_end ? 1.0 : -1.0;
            basis_compute_at_point_prepare(BASIS_LAGRANGE_GAUSS, axis_spec.order, prep);
            basis_compute_at_point_values(BASIS_LAGRANGE_GAUSS, axis_spec.order, 1, &endpoint, coeffs, prep);
            for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
            {
                for (size_t i_post = 0; i_post < post_count; ++i_post)
                {
                    const double *restrict line = src + (i_pre * (n_nodes * post_count) + i_post) * n_components;
                    double *restrict to = dst + (i_pre * post_count + i_post) * n_components;
                    for (unsigned component = 0; component < n_components; ++component)
                    {
                        double result = 0.0;
                        for (size_t i_node = 0; i_node < n_nodes; ++i_node)
                        {
                            result += coeffs[i_node] * line[(size_t)i_node * post_count * n_components + component];
                        }
                        to[component] = result;
                    }
                }
            }
            break;
        }

        default:
            CUTL_ASSERT(0, "Unsupported integration rule type.");
            break;
        }

        sizes[axis] = 1;
        src = dst;
    }
}

void iterate_over_contraction(unsigned ndim, const basis_spec_t basis[static ndim], unsigned axis,
                              void (*callback)(unsigned index, unsigned ndofs, unsigned stride, void *param),
                              void *param)
{
    CUTL_ASSERT(ndim > axis, "Axis out of bounds");
    unsigned pre_count = 1, post_count = 1;
    for (unsigned i = 0; i < axis; ++i)
    {
        pre_count *= basis[i].order + 1;
    }
    for (unsigned i = axis + 1; i < ndim; ++i)
    {
        post_count *= basis[i].order + 1;
    }
    unsigned ndofs = basis[axis].order + 1;
    for (unsigned i_pre = 0; i_pre < pre_count; ++i_pre)
    {
        for (unsigned i_post = 0; i_post < post_count; ++i_post)
        {
            unsigned index = i_pre * ndofs * post_count + i_post;
            callback(index, ndofs, post_count, param);
        }
    }
}
size_t boundary_dof_iterator_data_size(const unsigned bdim)
{
    return sizeof(boundary_dof_iterator_t) +
           (size_t)bdim * (2 * sizeof(ptrdiff_t) + 3 * sizeof(unsigned) + sizeof(int8_t));
}

void boundary_dof_iterator_init(boundary_dof_iterator_t *iter, const unsigned ndim,
                                const basis_spec_t basis[static ndim], const unsigned bdim,
                                const int8_t orientation[static ndim])
{
    CUTL_ASSERT(ndim >= 1, "Element dimension must be positive.");
    CUTL_ASSERT(ndim <= UINT8_MAX, "Element dimension exceeds the supported maximum.");
    CUTL_ASSERT(bdim <= ndim, "Boundary dimension exceeds the element dimension.");

    // The iterator only keeps references to its initialization data, which must stay valid
    // and unchanged while the iterator is in use.
    iter->ndim = ndim;
    iter->n_free = bdim;
    iter->basis = basis;
    iter->orientation = orientation;
    iter->total = 1;
    iter->visited = 0;
    iter->flat_index = 0;

    ptrdiff_t *const step = boundary_dof_iterator_step(iter);
    ptrdiff_t *const wrap = boundary_dof_iterator_wrap(iter);
    unsigned *const start = boundary_dof_iterator_start(iter);
    unsigned *const end = boundary_dof_iterator_end(iter);
    unsigned *const offset = boundary_dof_iterator_offset(iter);
    int8_t *const direction = boundary_dof_iterator_direction(iter);

    // Strides of the element tensor of degrees of freedom, last axis fastest.
    size_t strides[UINT8_MAX];
    size_t stride = 1;
    for (unsigned axis = ndim; axis-- > 0;)
    {
        strides[axis] = stride;
        stride *= basis[axis].order + 1;
    }

    unsigned slot = 0, previous_axis = 0;
    for (unsigned entry = 0; entry < ndim; ++entry)
    {
        // Decode the axis; the sign selects the endpoint of a fixed normal axis and the
        // iteration direction of a varying axis.
        const int8_t axis_code = orientation[entry];
        const unsigned axis = (unsigned)(axis_code < 0 ? -axis_code : axis_code) - 1;
        CUTL_ASSERT(axis < ndim, "Orientation references an axis outside the element.");
        const size_t size = basis[axis].order + 1;

        if (entry < ndim - bdim)
        {
            // Fixed normal axis: the boundary degrees of freedom sit at one endpoint.
            iter->flat_index += (axis_code > 0 ? size - 1 : 0) * strides[axis];
        }
        else
        {
            CUTL_ASSERT(slot == 0 || axis > previous_axis,
                        "The varying axes of the orientation must be in ascending order.");
            previous_axis = axis;
            direction[slot] = axis_code > 0 ? 1 : -1;
            start[slot] = axis_code > 0 ? 0 : (unsigned)(size - 1);
            end[slot] = axis_code > 0 ? (unsigned)(size - 1) : 0;
            offset[slot] = start[slot];
            step[slot] = (ptrdiff_t)direction[slot] * (ptrdiff_t)strides[axis];
            wrap[slot] = ((ptrdiff_t)start[slot] - (ptrdiff_t)end[slot]) * (ptrdiff_t)strides[axis];
            iter->flat_index += (size_t)offset[slot] * strides[axis];
            iter->total *= size;
            ++slot;
        }
    }
}

int boundary_dof_iterator_next(boundary_dof_iterator_t *iter)
{
    if (iter->visited + 1 >= iter->total)
    {
        // The current degree of freedom is the last one; the state stays untouched, so the
        // call can be repeated.
        return 0;
    }

    const ptrdiff_t *const step = boundary_dof_iterator_step(iter);
    const ptrdiff_t *const wrap = boundary_dof_iterator_wrap(iter);
    const unsigned *const start = boundary_dof_iterator_start(iter);
    const unsigned *const end = boundary_dof_iterator_end(iter);
    unsigned *const offset = boundary_dof_iterator_offset(iter);
    const int8_t *const direction = boundary_dof_iterator_direction(iter);

    // Odometer over the varying axes, the last axis fastest.
    unsigned slot = iter->n_free;
    for (;;)
    {
        CUTL_ASSERT(slot > 0, "Iterator advanced past its last degree of freedom.");
        --slot;
        if (offset[slot] != end[slot])
        {
            offset[slot] += direction[slot];
            iter->flat_index += step[slot];
            break;
        }
        // Wrap this axis back to its start and carry into the next one.
        iter->flat_index += wrap[slot];
        offset[slot] = start[slot];
    }
    ++iter->visited;
    return 1;
}

size_t boundary_dof_indices(const unsigned ndim, const basis_spec_t basis[static ndim], const unsigned bdim,
                            const int8_t orientation[static ndim], uint8_t work[restrict],
                            size_t boundary_indices[restrict])
{
    boundary_dof_iterator_t *const iter = (boundary_dof_iterator_t *)work;
    boundary_dof_iterator_init(iter, ndim, basis, bdim, orientation);
    for (size_t index = 0; index < iter->total; ++index)
    {
        boundary_indices[index] = boundary_dof_iterator_index(iter);
        boundary_dof_iterator_next(iter);
    }
    return iter->total;
}
