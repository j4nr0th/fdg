/**
 * @file constraint_common.h
 * @brief Orientation and common-space helpers shared by the constraint formulations.
 *
 * Both formulations read a higher-dimensional side the same way: a signed one-based axis permutation, and a set
 * of incident elements merged into one common space on the object they share. The sign convention and the
 * merge rule live here so they exist once.
 *
 * Header-only: callers pass their own scratch and nothing allocates.
 */

#pragma once

#include "constraints.h"

#include <cutl/iterators/combination_iterator.h>

/**
 * @brief Map a shared object's component axes into element axes and a sign.
 *
 * Each reversed mapped axis contributes one sign, and sorting the mapped axes into canonical element order
 * flips the covector sign once per transposition.
 *
 * @param side Element side; its orientation is a signed one-based permutation with an increasing-absolute-value
 *             fixed-axis prefix.
 * @param boundary_dim Shared object's dimension, at most `side->ndim`.
 * @param order Component's covector count, at most `boundary_dim`.
 * @param test_axes Component's sorted covector axes, in the object's frame.
 * @param mapped_axes [order] Receives the element axes, sorted.
 * @return `true` if the sign flips.
 */
static inline bool constraint_mapped_axes_and_sign(const constraint_element_side_t *const side,
                                                   const unsigned boundary_dim, const unsigned order,
                                                   const uint8_t test_axes[const static order == 0 ? 1 : order],
                                                   uint8_t mapped_axes[const static order == 0 ? 1 : order])
{
    const unsigned fixed_count = side->ndim - boundary_dim;
    unsigned sign = 0;
    for (unsigned i = 0; i < order; ++i)
    {
        const int8_t mapping = side->orientation[fixed_count + test_axes[i]];
        mapped_axes[i] = (uint8_t)(mapping < 0 ? -mapping : mapping) - 1;
        sign += (unsigned)(mapping < 0);
    }
    // One transposition per swap into canonical element order.
    for (unsigned i = 0; i < order; ++i)
    {
        for (unsigned j = i + 1; j < order; ++j)
        {
            if (mapped_axes[i] > mapped_axes[j])
            {
                sign += 1;
                const uint8_t tmp = mapped_axes[i];
                mapped_axes[i] = mapped_axes[j];
                mapped_axes[j] = tmp;
            }
        }
    }
    return sign & 1;
}

/**
 * @brief Map a shared object's component axes into an element component and its sign.
 *
 * @param side Element side; see #constraint_mapped_axes_and_sign.
 * @param boundary_dim Shared object's dimension.
 * @param order Component's covector count.
 * @param test_axes Component's sorted covector axes, in the object's frame.
 * @param mapped_axes [order] Receives the element axes, sorted.
 * @param out_component Receives the combination index of the mapped axes.
 * @return `true` if the sign flips.
 */
static inline bool constraint_mapped_component(const constraint_element_side_t *const side, const unsigned boundary_dim,
                                               const unsigned order,
                                               const uint8_t test_axes[const static order == 0 ? 1 : order],
                                               uint8_t mapped_axes[const static order == 0 ? 1 : order],
                                               unsigned *const out_component)
{
    const bool sign = constraint_mapped_axes_and_sign(side, boundary_dim, order, test_axes, mapped_axes);
    *out_component = combination_get_index(side->ndim, order, mapped_axes);
    return sign;
}

/**
 * @brief Element axis of one signed one-based orientation entry.
 */
static inline unsigned constraint_orientation_axis(const int8_t mapping)
{
    return (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
}

/**
 * @brief Whether an orientation entry reverses its element axis.
 */
static inline bool constraint_orientation_mirrored(const int8_t mapping)
{
    return mapping < 0;
}

/**
 * @brief Merge the per-element views of one shared object into one common space.
 *
 * The first element seeds the canonical axis order. Later elements lower a basis axis to their own order when it
 * is smaller and raise an integration axis when their rule is more accurate: minimum order, most accurate rule.
 *
 * @param ndim Element dimension.
 * @param bdim Shared object's dimension, below `ndim`.
 * @param nelem Incident element count, at least one.
 * @param elements [nelem] Incident elements' views.
 * @param out_basis [bdim] Receives the merged per-axis basis specification.
 * @param out_integration [bdim] Receives the merged per-axis integration specification.
 */
static inline void constraint_common_space_merge(const unsigned ndim, const unsigned bdim, const unsigned nelem,
                                                 const boundary_element_space_t elements[static nelem],
                                                 basis_spec_t out_basis[static bdim],
                                                 integration_spec_t out_integration[static bdim])
{
    const boundary_element_space_t *const first = elements;
    integration_rules_to_boundary(ndim, first->integration, first->orientation, bdim, out_integration);
    for (unsigned idim = 0; idim < bdim; ++idim)
    {
        const unsigned i_axis = constraint_orientation_axis(first->orientation[ndim - bdim + idim]);
        out_basis[idim] = first->basis[i_axis];
    }

    for (unsigned ie = 1; ie < nelem; ++ie)
    {
        const boundary_element_space_t *const element = elements + ie;
        const int8_t *const varying = element->orientation + (ndim - bdim);
        for (unsigned idim = 0; idim < bdim; ++idim)
        {
            const unsigned i_axis = constraint_orientation_axis(varying[idim]);
            if (out_basis[idim].order > element->basis[i_axis].order)
            {
                out_basis[idim] = element->basis[i_axis];
            }
            if (integration_spec_accuracy(out_integration + idim) <
                integration_spec_accuracy(element->integration + i_axis))
            {
                out_integration[idim] = element->integration[i_axis];
            }
        }
    }
}
