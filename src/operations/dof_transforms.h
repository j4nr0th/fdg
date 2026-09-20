#pragma once

#include "../basis/basis_set.h"
#include <stddef.h>

/**
 * @brief Computes the values of the degrees of freedom on a plane inside an element.
 *
 * The degrees of freedom along @p axis are contracted at the coordinate @p plane: they are
 * replaced by the value of the polynomial they represent at the plane. The surviving axes
 * keep their relative element order and remain last axis fastest, so the output is the
 * compact trace of the input tensor with the @p axis slot removed.
 *
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element.
 * @param values Values of the degrees of freedom for the element, `order + 1` per axis and
 *               laid out last axis fastest. Must not overlap with @p work or @p out.
 * @param work Work array with dof_plane_values_work_size(ndim, basis, axis) doubles. Must
 *               not overlap with @p values or @p out.
 * @param axis Axis along which the degrees of freedom are contracted.
 * @param plane Coordinate along @p axis at which the basis functions are evaluated.
 * @param out Output array holding the product of `order + 1` over the axes other than
 *               @p axis, receiving the values of the degrees of freedom at the plane. Must
 *               not overlap with @p values or @p work.
 */
void dof_plane_values(unsigned ndim, const basis_spec_t basis[static ndim], const double values[restrict],
                      double work[restrict], unsigned axis, double plane, double out[restrict]);

/**
 * @brief Amount of work memory @ref dof_plane_values requires, counted in doubles.
 *
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element.
 * @param axis Axis along which the degrees of freedom are contracted.
 * @return Required number of doubles for the work array of @ref dof_plane_values.
 */
size_t dof_plane_values_work_size(unsigned ndim, const basis_spec_t basis[static ndim], unsigned axis);

/**
 * @brief Reverses the orientation of the degrees of freedom along one axis of an element.
 *
 * The output represents the same tensor polynomial as the input with @p axis mirrored,
 * `p(..., x, ...)` becoming `p(..., -x, ...)` along that axis. Node based bases (Bernstein
 * and all Lagrange variants) swap each coefficient with the one at the mirrored node, while
 * Legendre coefficients keep their position and change sign, following
 * `P_i(-x) = (-1)^i P_i(x)`. All surviving axes keep their element order and remain last
 * axis fastest.
 *
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element.
 * @param values Values of the degrees of freedom for the element, `order + 1` per axis and
 *               laid out last axis fastest. Must not overlap with @p out.
 * @param axis Axis along which the orientation is reversed.
 * @param out Output array with as many entries as @p values, receiving the reoriented
 *               degrees of freedom. Must not overlap with @p values.
 */
void dof_reverse_orientation_values(unsigned ndim, const basis_spec_t basis[static ndim], const double values[restrict],
                                    unsigned axis, double out[restrict]);
