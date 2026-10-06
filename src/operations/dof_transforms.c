#include "dof_transforms.h"
#include <stdint.h>

size_t dof_plane_values_work_size(unsigned ndim, const basis_spec_t basis[static ndim], unsigned axis)
{
    CUTL_ASSERT(axis < ndim, "Axis references a dimension outside the element.");
    // Half of the array holds the prepared evaluation scratch, the other half the plane values.
    return 2 * (size_t)(basis[axis].order + 1);
}

void dof_plane_values(unsigned ndim, const basis_spec_t basis[static ndim], const double values[restrict],
                      double work[restrict], unsigned axis, double plane, double out[restrict])
{
    CUTL_ASSERT(ndim <= UINT8_MAX, "Element dimension exceeds the supported maximum.");
    CUTL_ASSERT(axis < ndim, "Axis references a dimension outside the element.");

    const basis_spec_t axis_basis = basis[axis];
    const size_t ndofs = axis_basis.order + 1;

    // Counts of the degrees of freedom before and after the contracted axis. The iteration
    // strides follow from them, iterating last axis fastest.
    size_t pre_count = 1, post_count = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        if (i < axis)
        {
            pre_count *= basis[i].order + 1;
        }
        else if (i > axis)
        {
            post_count *= basis[i].order + 1;
        }
    }

    // Every basis type is evaluated at the plane coordinate once, before the loops: the
    // prepared node scratch occupies the first half of the work array, the plane values of
    // the basis functions the second half.
    basis_compute_at_point_prepare(axis_basis.type, axis_basis.order, work);
    basis_compute_at_point_values(axis_basis.type, axis_basis.order, 1, &plane, work + ndofs, work);
    const double *const coeffs = work + ndofs;

    for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
    {
        for (size_t i_post = 0; i_post < post_count; ++i_post)
        {
            const size_t line = i_pre * (ndofs * post_count) + i_post;
            double result = 0.0;
            for (unsigned i_dof = 0; i_dof < ndofs; ++i_dof)
            {
                result += coeffs[i_dof] * values[line + (size_t)i_dof * post_count];
            }
            out[i_pre * post_count + i_post] = result;
        }
    }
}

void dof_reverse_orientation_values(unsigned ndim, const basis_spec_t basis[static ndim], const double values[restrict],
                                    unsigned axis, double out[restrict])
{
    CUTL_ASSERT(ndim <= UINT8_MAX, "Element dimension exceeds the supported maximum.");
    CUTL_ASSERT(axis < ndim, "Axis references a dimension outside the element.");

    const basis_spec_t axis_basis = basis[axis];
    const size_t ndofs = axis_basis.order + 1;

    // Counts of the degrees of freedom before and after the reversed axis. The iteration
    // strides follow from them, iterating last axis fastest.
    size_t pre_count = 1, post_count = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        if (i < axis)
        {
            pre_count *= basis[i].order + 1;
        }
        else if (i > axis)
        {
            post_count *= basis[i].order + 1;
        }
    }

    // Dispatch on the basis type outside of the loops: node based bases swap each degree of
    // freedom with the one at the mirrored node, while the Legendre coefficients keep their
    // position and take on the sign of their polynomial at the mirrored point.
    switch (axis_basis.type)
    {
    case BASIS_BERNSTEIN:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_count; ++i_post)
            {
                const size_t line = i_pre * (ndofs * post_count) + i_post;
                for (unsigned i_dof = 0; i_dof < ndofs; ++i_dof)
                {
                    out[line + (size_t)i_dof * post_count] = values[line + (size_t)(ndofs - 1 - i_dof) * post_count];
                }
            }
        }
        break;

    case BASIS_LEGENDRE:
        for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_count; ++i_post)
            {
                const size_t line = i_pre * (ndofs * post_count) + i_post;
                for (unsigned i_dof = 0; i_dof < ndofs; i_dof += 2)
                {
                    out[line + (size_t)i_dof * post_count] = values[line + (size_t)i_dof * post_count];
                }
                for (unsigned i_dof = 1; i_dof < ndofs; i_dof += 2)
                {
                    out[line + (size_t)i_dof * post_count] = -values[line + (size_t)i_dof * post_count];
                }
            }
        }
        break;

    default:
        CUTL_ASSERT(0, "Unsupported basis type.");
        break;
    }
}
