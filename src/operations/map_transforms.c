#include "map_transforms.h"

#include "matrices.h"
#include <cutl/iterators/combination_iterator.h>
#include <cutl/iterators/permutation_iterator.h>
#include <stdint.h>

double compute_inverse_transform(const matrix_t jacobian, const matrix_t q_matrix, const matrix_t out_matrix)
{
    const unsigned rows = jacobian.rows;
    const unsigned cols = jacobian.cols;

    CUTL_ASSERT(q_matrix.rows == rows && q_matrix.cols == rows, "Q matrix dimensions do not match Jacobian.");
    CUTL_ASSERT(out_matrix.rows == cols && out_matrix.cols == rows, "Output matrix dimensions do not match Jacobian.");

    // Decompose Jacobian into QR decomposition
    fdg_result_t res = matrix_qr_decompose(&jacobian, &q_matrix);
    (void)res;
    CUTL_ASSERT(res == FDG_SUCCESS, "QR decomposition failed.");
    // Compute the determinant from the diagonal of the matrix
    double det = 1;
    for (unsigned i = 0; i < cols; ++i)
    {
        det *= jacobian.values[i * cols + i];
    }

    // Copy the top part of q into out
    for (unsigned irow = 0; irow < cols; ++irow)
    {
        for (unsigned icol = 0; icol < rows; ++icol)
        {
            out_matrix.values[irow * rows + icol] = q_matrix.values[irow * rows + icol];
        }
    }

    // Use decomposition to compute "inverse". This is done simply by applying inverse of the
    // upper triangular (rows x rows) part of the jacobian to the matrix q_mat.
    res = matrix_back_substitute(&jacobian, &out_matrix);
    CUTL_ASSERT(res == FDG_SUCCESS, "Back substitution failed.");
    (void)res;
    return det;
}

int compute_basis_transform_from_inverse(const cutl_allocator_t *allocator, unsigned n_dims, unsigned n_maps,
                                         unsigned order, const double *inverse_maps, const double *determinant,
                                         size_t n_pts, double *out)
{
    if (order == 1)
    {
        // Special case: transformation is just the space map
        for (size_t i_in = 0; i_in < n_dims; ++i_in)
            for (size_t i_out = 0; i_out < n_maps; ++i_out)
                for (size_t i_pt = 0; i_pt < n_pts; ++i_pt)
                    out[(i_in * n_maps + i_out) * n_pts + i_pt] =
                        inverse_maps[i_pt * ((size_t)n_dims * n_maps) + i_in * n_maps + i_out];
    }
    else if (order == n_maps)
    {
        for (size_t i_pt = 0; i_pt < n_pts; ++i_pt)
            out[i_pt] = 1 / determinant[i_pt];
    }
    else // (order != 1 && order != n_maps)
    {
        permutation_iterator_t *iter_out_perm;
        combination_iterator_t *iter_out_comb;
        combination_iterator_t *iter_in_comb;
        void *const mem = cutl_alloc_group(
            allocator,
            (const cutl_alloc_info_t[]){
                {.size = permutation_iterator_required_memory(order, order), .p_ptr = (void **)&iter_out_perm},
                {.size = combination_iterator_required_memory(order), .p_ptr = (void **)&iter_out_comb},
                {.size = combination_iterator_required_memory(order), .p_ptr = (void **)&iter_in_comb},
                {},
            });
        if (!mem)
            return -1;

        size_t idx_in = 0;
        // Iterate over bases in the inputs space
        combination_iterator_init(iter_in_comb, n_dims, order);
        while (!combination_iterator_is_done(iter_in_comb))
        {
            // Indices of current input dimensions
            const uint8_t *const current_in = combination_iterator_current(iter_in_comb);
            // Iterate over bases in the output space.
            size_t idx_out = 0;
            combination_iterator_init(iter_out_comb, n_maps, order);
            while (!combination_iterator_is_done(iter_out_comb))
            {
                // Indices of current output dimensions.
                const uint8_t *const current_out = combination_iterator_current(iter_out_comb);

                // Loop over points
                for (size_t idx_pt = 0; idx_pt < n_pts; ++idx_pt)
                {
                    // Total transformation coefficient, which we will accumulate for each possible basis.
                    double val = 0.0;

                    // Loop over all permutations of the current basis indices.
                    permutation_iterator_init(iter_out_perm, order, order);
                    while (!permutation_iterator_is_done(iter_out_perm))
                    {
                        // Indices for the current permutation
                        const uint8_t *const current_perm = permutation_iterator_current(iter_out_perm);
                        double basis_contribution = 1.0;

                        // Loop over the derivative terms and compute their product
                        for (unsigned idim = 0; idim < order; ++idim)
                        {
                            const unsigned idx_coord = current_out[current_perm[idim]];
                            const unsigned idx_dim = current_in[idim];
                            const double contribution =
                                inverse_maps[idx_pt * ((size_t)n_dims * n_maps) + (size_t)idx_dim * n_maps + idx_coord];
                            basis_contribution *= contribution;
                        }

                        // Check if we flip the sign (meaning subtract) for this contribution.
                        if (permutation_iterator_current_sign(iter_out_perm))
                        {
                            val -= basis_contribution;
                        }
                        else
                        {
                            val += basis_contribution;
                        }

                        permutation_iterator_next(iter_out_perm);
                    }
                    out[(idx_in * combination_total_count(n_maps, order) + idx_out) * n_pts + idx_pt] = val;
                }

                idx_out += 1;
                combination_iterator_next(iter_out_comb);
            }

            idx_in += 1;
            combination_iterator_next(iter_in_comb);
        }

        cutl_dealloc(allocator, mem);
    }
    return 0;
}

void compute_space_map_determinants(unsigned n_dim, unsigned n_maps, const double *const *gradients,
                                    size_t total_points, double determinant[restrict], double inverse_maps[restrict],
                                    double jacobian_work[restrict], double q_work[restrict])
{
    // Now we iterate over all the points
    for (size_t i_pt = 0; i_pt < total_points; ++i_pt)
    {
        // Fill in the Jacobian
        const unsigned rows = n_maps;
        const unsigned cols = n_dim;
        for (unsigned idim = 0; idim < rows; ++idim)
        {
            for (unsigned jdim = 0; jdim < cols; ++jdim)
            {
                // The block at (coordinate * n_dim + dimension) contains the derivative of the
                // coordinate with respect to the reference dimension at every point.
                jacobian_work[idim * cols + jdim] = gradients[idim * n_dim + jdim][i_pt];
            }
        }

        double *const p_inv_map = inverse_maps + i_pt * ((size_t)n_dim * n_maps);
        const matrix_t jacobian_mat = (matrix_t){.rows = rows, .cols = cols, .values = jacobian_work};
        const matrix_t q_matrix = (matrix_t){.rows = rows, .cols = rows, .values = q_work};
        const matrix_t out_mat = (matrix_t){.rows = cols, .cols = rows, .values = p_inv_map};

        determinant[i_pt] = compute_inverse_transform(jacobian_mat, q_matrix, out_mat);
    }
}

void interpolate_sampled_map(unsigned n_dims, unsigned n_coords, const unsigned orders[static n_dims],
                             const unsigned internal_orders[static n_dims], const double *const *axis_transformations,
                             const double *const *coordinate_values, const double *const *coordinate_gradients,
                             size_t total_points, double positions[restrict], double determinant[restrict],
                             double inverse_maps[restrict], double jacobian_work[restrict], double q_work[restrict])
{
    // Number of internal nodes of the sampled maps, i.e. the number of points that are interpolated from.
    size_t nodes_in = 1;
    for (unsigned d = 0; d < n_dims; ++d)
    {
        const size_t n_int = (size_t)internal_orders[d] + 1;
        nodes_in *= n_int;
    }

    const size_t trans_size = (size_t)n_dims * n_coords;
    const matrix_t jacobian_mat = {.rows = n_coords, .cols = n_dims, .values = jacobian_work};
    const matrix_t q_mat = {.rows = n_coords, .cols = n_coords, .values = q_work};

    // Interpolate positions and forward transformation matrices together. This
    // avoids allocating the dense tensor-product interpolation matrix.
    for (size_t i_out = 0; i_out < total_points; ++i_out)
    {
        for (unsigned idim_out = 0; idim_out < n_coords; ++idim_out)
        {
            positions[n_coords * i_out + idim_out] = 0.0;
            for (unsigned idim_in = 0; idim_in < n_dims; ++idim_in)
                jacobian_work[idim_out * n_dims + idim_in] = 0.0;
        }

        for (size_t i_in = 0; i_in < nodes_in; ++i_in)
        {
            double weight = 1.0;
            size_t output_stride = total_points;
            size_t input_stride = nodes_in;
            for (unsigned d = 0; d < n_dims; ++d)
            {
                const unsigned n_out = orders[d] + 1;
                const unsigned n_int = internal_orders[d] + 1;
                output_stride /= n_out;
                input_stride /= n_int;
                const unsigned idx_out = (i_out / output_stride) % n_out;
                const unsigned idx_in = (i_in / input_stride) % n_int;
                weight *= axis_transformations[d][(size_t)idx_in * n_out + idx_out];
            }

            for (unsigned idim_out = 0; idim_out < n_coords; ++idim_out)
            {
                positions[n_coords * i_out + idim_out] += weight * coordinate_values[idim_out][i_in];
                for (unsigned idim_in = 0; idim_in < n_dims; ++idim_in)
                    jacobian_work[idim_out * n_dims + idim_in] +=
                        weight * coordinate_gradients[idim_out * n_dims + idim_in][i_in];
            }
        }

        const matrix_t out_mat = {.rows = n_dims, .cols = n_coords, .values = inverse_maps + i_out * trans_size};
        determinant[i_out] = compute_inverse_transform(jacobian_mat, q_mat, out_mat);
    }
}
