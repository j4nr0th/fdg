#pragma once

#include "matrices.h"
#include <cutl/allocators.h>
#include <stddef.h>

/**
 * @brief Computes the determinant and the inverse of a Jacobian matrix using a QR decomposition.
 *
 * The Jacobian is decomposed in place into its QR decomposition, with the Jacobian storage
 * becoming the upper-triangular factor and @p q_matrix becoming the orthogonal factor. The
 * determinant is the product of the diagonal of the triangular factor, and the inverse is
 * obtained by back-substituting the top part of @p q_matrix through the triangular factor into
 * @p out_matrix. Both @p q_matrix and @p out_matrix are overwritten; they are passed in so that
 * the caller controls the work memory.
 *
 * @param jacobian Jacobian matrix with `rows * cols` entries in row-major order, destroyed
 *               in the process.
 * @param q_matrix Work matrix with `rows * rows` entries in row-major order, overwritten with
 *               the orthogonal factor of the decomposition.
 * @param out_matrix Output matrix with `cols * rows` entries in row-major order, receiving the
 *               inverse of @p jacobian.
 * @return Determinant of @p jacobian.
 */
double compute_inverse_transform(const matrix_t jacobian, const matrix_t q_matrix, const matrix_t out_matrix);

/**
 * @brief Computes the transformation coefficients of order-@p order k-form bases from the
 *               inverse of a space map.
 * Each coefficient is the signed sum over the permutations of the selected output coordinates of
 * the products of the corresponding entries of the inverse maps, i.e. the minors of the inverse
 * map at every point. The special cases `order == 1` (the coefficients are the entries of the
 * inverse maps themselves) and `order == n_maps` (the only coefficient is the reciprocal of the
 * determinant) are handled directly, while the general case enumerates the combinations of
 * order input dimensions and output coordinates. The iterator memory is allocated from
 * @p allocator and released before returning.
 *
 * @todo Might be worth to take in memory for the iterators as parameter.
 *
 * @param allocator Allocator used to allocate the combination and permutation iterators from.
 * @param n_dims Number of dimensions of the reference space.
 * @param n_maps Number of coordinate maps, i.e. dimensions of the output space.
 * @param order Order of the k-form, with `1 <= order <= n_maps` and `order <= n_dims`.
 * @param inverse_maps Inverse maps at every point, `n_pts * n_dims * n_maps` doubles laid out
 *               point-major, each point holding an (n_dims x n_maps) row-major matrix.
 * @param determinant Determinant of the forward map at every point, `n_pts` doubles. Only read
 *               when `order == n_maps`.
 * @param n_pts Number of points.
 * @param out Output array holding `n_dims * n_maps * n_pts` doubles when `order == 1`, `n_pts`
 *               doubles when `order == n_maps`, and otherwise
 *               `n_dims * combination_total_count(n_maps, order) * n_pts` doubles. In every case
 *               the coefficients are laid out coefficient-block-major with the point index
 *               varying fastest, where the blocks are enumerated as
 *               `i_in * combination_total_count(n_maps, order) + i_out` with `i_in` indexing the
 *               combinations of order input dimensions and `i_out` the combinations of order
 *               output coordinates, both in the lexicographical order of the combination
 *               iterators.
 * @return 0 on success, -1 if the iterator memory could not be allocated.
 */
int compute_basis_transform_from_inverse(const cutl_allocator_t *allocator, unsigned n_dims, unsigned n_maps,
                                         unsigned order, const double *inverse_maps, const double *determinant,
                                         size_t n_pts, double *out);

/**
 * @brief Computes the determinants and inverse maps of a space map at every point.
 *
 * For every point the Jacobian of the space map is assembled from the coordinate gradients and
 * @ref compute_inverse_transform is invoked on it, producing the determinant and the inverse map
 * of the point. The Jacobian at a point is the (n_maps x n_dim) matrix whose entry at
 * (i_map, j_dim) is the derivative of the i_map-th coordinate with respect to the j_dim-th
 * reference dimension.
 *
 * @param n_dim Number of dimensions of the reference space.
 * @param n_maps Number of coordinate maps, i.e. dimensions of the output space.
 * @param gradients Table of `n_maps * n_dim` pointers laid out as [i_map * n_dim + j_dim]. The
 *               pointer at [i_map * n_dim + j_dim] addresses `total_points` doubles holding the
 *               derivative of the i_map-th coordinate with respect to the j_dim-th reference
 *               dimension at every point.
 * @param total_points Number of points.
 * @param determinant Output array of `total_points` doubles receiving the determinant at every
 *               point.
 * @param inverse_maps Output array of `total_points * n_dim * n_maps` doubles laid out
 *               point-major, each point holding an (n_dim x n_maps) row-major matrix receiving
 *               the inverse map at that point.
 * @param jacobian_work Work array of `n_maps * n_dim` doubles. Must not overlap with any of the
 *               other arrays.
 * @param q_work Work array of `n_maps * n_maps` doubles. Must not overlap with any of the other
 *               arrays.
 */
void compute_space_map_determinants(unsigned n_dim, unsigned n_maps, const double *const *gradients,
                                    size_t total_points, double determinant[restrict], double inverse_maps[restrict],
                                    double jacobian_work[restrict], double q_work[restrict]);

/**
 * @brief Interpolates a sampled space map onto the sample points and computes its inverses.
 *
 * The sample points form a tensor-product grid with `orders[d] + 1` nodes along axis `d`, laid
 * out last axis fastest, and are interpolated from the internal nodes of the sampled coordinate
 * maps, of which there are `internal_orders[d] + 1` along axis `d`, also laid out last axis
 * fastest. For every sample point the position is accumulated as the tensor-product weighted sum
 * of the coordinate values, and the forward Jacobian from the coordinate gradients; the
 * determinant and the inverse map of the point are then computed with
 * @ref compute_inverse_transform.
 *
 * @param n_dims Number of dimensions of the reference space.
 * @param n_coords Number of coordinate maps, i.e. dimensions of the output space.
 * @param orders Sample order along each axis; axis `d` carries `orders[d] + 1` sample nodes.
 * @param internal_orders Internal node order along each axis; axis `d` carries
 *               `internal_orders[d] + 1` internal nodes.
 * @param axis_transformations Table of @p n_dims pointers, one interpolation matrix per axis.
 *               The matrix addressed by axis_transformations[d] holds
 *               `(orders[d] + 1) * (internal_orders[d] + 1)` doubles laid out input-node-major:
 *               the entry at `idx_in * (orders[d] + 1) + idx_out` is the value of the idx_out-th
 *               sample basis polynomial at the idx_in-th internal node of the axis.
 * @param coordinate_values Table of @p n_coords pointers; the pointer at [i_coord] addresses
 *               `product(internal_orders[d] + 1)` doubles holding the values of the i_coord-th
 *               coordinate at every internal node, last axis fastest.
 * @param coordinate_gradients Table of `n_coords * n_dims` pointers laid out as
 *               [i_coord * n_dims + j_dim]. The pointer at [i_coord * n_dims + j_dim] addresses
 *               `product(internal_orders[d] + 1)` doubles holding the derivative of the
 *               i_coord-th coordinate with respect to the j_dim-th reference dimension at every
 *               internal node, last axis fastest.
 * @param total_points Number of sample points, the product of `orders[d] + 1` over the axes.
 * @param positions Output array of `total_points * n_coords` doubles laid out point-major,
 *               receiving the interpolated position of every sample point.
 * @param determinant Output array of `total_points` doubles receiving the determinant of the
 *               forward Jacobian at every sample point.
 * @param inverse_maps Output array of `total_points * n_dims * n_coords` doubles laid out
 *               point-major, each point holding an (n_dims x n_coords) row-major matrix
 *               receiving the inverse map at that point.
 * @param jacobian_work Work array of `n_coords * n_dims` doubles. Must not overlap with any of
 *               the other arrays.
 * @param q_work Work array of `n_coords * n_coords` doubles. Must not overlap with any of the
 *               other arrays.
 */
void interpolate_sampled_map(unsigned n_dims, unsigned n_coords, const unsigned orders[static n_dims],
                             const unsigned internal_orders[static n_dims], const double *const *axis_transformations,
                             const double *const *coordinate_values, const double *const *coordinate_gradients,
                             size_t total_points, double positions[restrict], double determinant[restrict],
                             double inverse_maps[restrict], double jacobian_work[restrict], double q_work[restrict]);
