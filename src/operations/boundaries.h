#pragma once

#include "../basis/basis_set.h"
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Computes the values of the degrees of freedom at the boundary of an element.
 *
 * The first `ndim - bdim` entries of @p orientation (the fixed normal axes of the boundary,
 * by convention with ascending absolute values) are contracted one after another: along each
 * fixed axis the degrees of freedom are replaced by the value of the polynomial they represent
 * at the endpoint selected by the orientation, negative denoting the start of the axis and
 * positive the end. The surviving axes keep their relative element order and remain last axis
 * fastest.
 *
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element.
 * @param values Values of the degrees of freedom for the element, `order + 1` per axis and
 *               laid out last axis fastest. Must not overlap with @p work or
 *               @p boundary_values.
 * @param work Work array with boundary_dof_values_work_size(ndim, basis) doubles. Must not
 *               overlap with @p values or @p boundary_values. It is left untouched when
 *               `ndim - bdim <= 1`.
 * @param bdim Number of dimensions of the boundary.
 * @param orientation Orientation of the boundary in the element, one entry per element axis,
 *               with the fixed normal axes of the boundary occupying the first entries.
 * @param boundary_values Output array holding the product of `order + 1` over the non-fixed
 *               axes, receiving the values of the degrees of freedom at the boundary. Must
 *               not overlap with @p values or @p work.
 */
void boundary_dof_values(unsigned ndim, const basis_spec_t basis[static ndim], const double values[restrict],
                         double work[restrict], unsigned bdim, const int8_t orientation[static ndim],
                         double boundary_values[restrict]);

/**
 * @brief Amount of work memory @ref boundary_dof_values requires, counted in doubles.
 *
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element.
 * @return Required number of doubles for the work array of @ref boundary_dof_values.
 */
size_t boundary_dof_values_work_size(unsigned ndim, const basis_spec_t basis[static ndim]);

/**
 * @brief Iterate over all degrees of freedom such that a contraction can be performed over an axis.
 *
 * Contraction means all iterations provide an index such that the index related to @ref axis is 0
 * and if one wishes to vary it, the stride is provided to the callback. The purpose of this function
 * is to simplify contraction operations along a axis, such as min, max, or sum.
 *
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element.
 * @param axis Axis along which to perform the contraction.
 * @param callback Function to call for each degree of freedom along the contraction.
 * @param param User-defined parameter to pass to the callback function.
 */
void iterate_over_contraction(unsigned ndim, const basis_spec_t basis[static ndim], unsigned axis,
                              void (*callback)(unsigned index, unsigned ndofs, unsigned stride, void *param),
                              void *param);
