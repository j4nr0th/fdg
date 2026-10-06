#pragma once

#include "../basis/basis_set.h"
#include <stdalign.h>
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
 * @brief Computes the values at the integration points of a boundary of an element.
 *
 * The first `ndim - bdim` entries of @p orientation (the fixed normal axes of the boundary,
 * by convention with ascending absolute values) are dropped one after another: along each
 * fixed axis the integration points are replaced by the boundary endpoint selected by the
 * orientation, negative denoting the start of the axis and positive the end. Rules with
 * nodes at the endpoints (Gauss-Lobatto) reduce this to selecting the endpoint slice of
 * points, while Gauss-Legendre axes evaluate the interpolant through their nodes at the
 * endpoint. Each integration point carries @p n_components values, laid out component
 * fastest, and the surviving axes keep their full point counts, their relative element
 * order, and remain last axis fastest.
 *
 * @param ndim Number of dimensions of the element.
 * @param specs Integration rule specification for each dimension of the element.
 * @param values Values at the integration points of the element,
 *               `n_components * product(specs[i].order + 1)` doubles laid out as
 *               point-major with the components of each point contiguous. Must not
 *               overlap with @p work or @p boundary_values.
 * @param work Work array with
 *               boundary_integration_point_values_work_size(ndim, specs, n_components)
 *               doubles. Must not overlap with @p values or @p boundary_values. It is
 *               left untouched when `ndim - bdim <= 1`.
 * @param bdim Number of dimensions of the boundary.
 * @param orientation Orientation of the boundary in the element, one entry per element
 *               axis, with the fixed normal axes of the boundary occupying the first
 *               entries.
 * @param n_components Number of values stored at each integration point.
 * @param boundary_values Output array holding
 *               `n_components` times the product of `specs[i].order + 1` over the
 *               non-fixed axes, receiving the values at the integration points of the
 *               boundary. Must not overlap with @p values or @p work.
 */
void boundary_integration_point_values(unsigned ndim, const integration_spec_t specs[static ndim],
                                       const double values[restrict], double work[restrict], unsigned bdim,
                                       const int8_t orientation[static ndim], unsigned n_components,
                                       double boundary_values[restrict]);

/**
 * @brief Amount of work memory @ref boundary_integration_point_values requires, in doubles.
 *
 * @param ndim Number of dimensions of the element.
 * @param specs Integration rule specification for each dimension of the element.
 * @param n_components Number of values stored at each integration point.
 * @return Required number of doubles for the work array of
 *         @ref boundary_integration_point_values.
 */
size_t boundary_integration_point_values_work_size(unsigned ndim, const integration_spec_t specs[static ndim],
                                                   unsigned n_components);

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
/**
 * @brief Cached iterator over the degrees of freedom on a boundary of an element.
 *
 * The iterator walks the degrees of freedom of the boundary in the compact boundary order of
 * @ref boundary_dof_values: the varying (tangent) axes keep their relative element order and
 * remain last axis fastest, while a negative orientation entry reverses the direction in
 * which its varying axis is traversed. Every degree of freedom is reported as a flat index
 * into the degree of freedom tensor of the element, so the k-th visited index pairs with the
 * k-th entry of the boundary values. All per-axis data is precomputed on initialization;
 * advancing amounts to a constant amount of index arithmetic.
 *
 * The iterator is dynamically sized and allocated by the caller as a single chunk of
 * boundary_dof_iterator_data_size(bdim) bytes, aligned for max_align_t. It only stores
 * references to its initialization data, so @p basis and @p orientation must remain valid
 * and unchanged for the whole lifetime of the iterator.
 */
typedef struct
{
    unsigned ndim;                       /**< Number of dimensions of the element. */
    unsigned n_free;                     /**< Number of varying (boundary tangent) axes. */
    size_t total;                        /**< Number of degrees of freedom on the boundary. */
    size_t visited;                      /**< Number of degrees of freedom visited so far. */
    size_t flat_index;                   /**< Flat element index of the current degree of freedom. */
    const basis_spec_t *basis;           /**< Referenced initialization basis, must stay valid and unchanged. */
    const int8_t *orientation;           /**< Referenced initialization orientation, must stay valid and unchanged. */
    alignas(max_align_t) uint8_t data[]; /**< Per-varying-axis caches, unpacked by the accessors below. */
} boundary_dof_iterator_t;

/**
 * @brief Amount of memory needed to store a boundary degree of freedom iterator, in bytes.
 *
 * @param bdim Number of dimensions of the boundary; the amount grows with the number of
 *             varying axes.
 * @return The number of bytes to allocate (heap or aligned stack buffer) before calling
 *         boundary_dof_iterator_init.
 */
size_t boundary_dof_iterator_data_size(unsigned bdim);

/** @brief Flat index change when advancing along each varying axis, cached at initialization. */
static inline ptrdiff_t *boundary_dof_iterator_step(boundary_dof_iterator_t *iter)
{
    return (ptrdiff_t *)iter->data;
}

/** @brief Flat index change when a varying axis wraps around, cached at initialization. */
static inline ptrdiff_t *boundary_dof_iterator_wrap(boundary_dof_iterator_t *iter)
{
    return (ptrdiff_t *)iter->data + iter->n_free;
}

/** @brief First index of each varying axis. */
static inline unsigned *boundary_dof_iterator_start(boundary_dof_iterator_t *iter)
{
    return (unsigned *)(iter->data + (size_t)iter->n_free * (2 * sizeof(ptrdiff_t)));
}

/** @brief Last index of each varying axis. */
static inline unsigned *boundary_dof_iterator_end(boundary_dof_iterator_t *iter)
{
    return (unsigned *)(iter->data + (size_t)iter->n_free * (2 * sizeof(ptrdiff_t) + sizeof(unsigned)));
}

/** @brief Current index along each varying axis. */
static inline unsigned *boundary_dof_iterator_offset(boundary_dof_iterator_t *iter)
{
    return (unsigned *)(iter->data + (size_t)iter->n_free * (2 * sizeof(ptrdiff_t) + 2 * sizeof(unsigned)));
}

/** @brief Iteration direction of each varying axis, `+1` or `-1`. */
static inline int8_t *boundary_dof_iterator_direction(boundary_dof_iterator_t *iter)
{
    return (int8_t *)(iter->data + (size_t)iter->n_free * (2 * sizeof(ptrdiff_t) + 3 * sizeof(unsigned)));
}

/**
 * @brief Initializes a boundary degree of freedom iterator.
 *
 * The iterator must have been allocated with boundary_dof_iterator_data_size(bdim) bytes,
 * aligned for max_align_t. The first `ndim - bdim` entries of @p orientation (the fixed
 * normal axes of the boundary, by convention with ascending absolute values) are pinned to
 * the endpoint selected by their sign, negative denoting the start of the axis and positive
 * the end. The remaining `bdim` entries, in ascending axis order, are the varying axes of
 * the boundary: a negative sign traverses the axis from its last to its first degree of
 * freedom, a positive sign the other way around.
 *
 * @param iter Iterator to initialize, with boundary_dof_iterator_data_size(bdim) bytes of
 *               aligned storage behind it.
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element, referenced by the iterator and must
 *               remain valid and unchanged while the iterator is in use.
 * @param bdim Number of dimensions of the boundary.
 * @param orientation Orientation of the boundary in the element, one entry per element axis,
 *               with the fixed normal axes of the boundary occupying the first entries.
 *               Referenced by the iterator and must remain valid and unchanged while the
 *               iterator is in use.
 */
void boundary_dof_iterator_init(boundary_dof_iterator_t *iter, unsigned ndim, const basis_spec_t basis[static ndim],
                                unsigned bdim, const int8_t orientation[static ndim]);

/**
 * @brief Advances a boundary degree of freedom iterator to its next degree of freedom.
 *
 * Running past the last degree of freedom is not an error: the iterator reports exhaustion
 * and keeps its state, so repeated calls keep returning zero.
 *
 * @param iter Iterator to advance.
 * @return One when the iterator moved to the next degree of freedom, zero when the current
 *         one is the last.
 */
int boundary_dof_iterator_next(boundary_dof_iterator_t *iter);

/**
 * @brief Flat index into the degree of freedom tensor of the element for the current degree
 *          of freedom of the iterator.
 */
static inline size_t boundary_dof_iterator_index(const boundary_dof_iterator_t *iter)
{
    return iter->flat_index;
}

/**
 * @brief Number of degrees of freedom the iterator walks over, in total.
 */
static inline size_t boundary_dof_iterator_count(const boundary_dof_iterator_t *iter)
{
    return iter->total;
}

/**
 * @brief Fills the flat element indices of all degrees of freedom on a boundary.
 *
 * The indices are written in the compact boundary order of @ref boundary_dof_values, one
 * entry per boundary degree of freedom, such that `values[boundary_indices[k]]` is the
 * element degree of freedom belonging to the k-th boundary degree of freedom.
 *
 * @param ndim Number of dimensions of the element.
 * @param basis Basis for each dimension of the element, must remain valid and unchanged for
 *               the duration of the call.
 * @param bdim Number of dimensions of the boundary.
 * @param orientation Orientation of the boundary in the element, one entry per element axis,
 *               with the fixed normal axes of the boundary occupying the first entries.
 * @param work Work array of boundary_dof_iterator_data_size(bdim) bytes, aligned for
 *               max_align_t. Must not overlap with @p boundary_indices.
 * @param boundary_indices Output array receiving one flat element index per boundary degree
 *               of freedom. Must not overlap with @p work.
 * @return Number of written indices, the product of `order + 1` over the varying axes.
 */
size_t boundary_dof_indices(unsigned ndim, const basis_spec_t basis[static ndim], unsigned bdim,
                            const int8_t orientation[static ndim], uint8_t work[restrict],
                            size_t boundary_indices[restrict]);
