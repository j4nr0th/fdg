#ifndef FDG_MATRICES_H
#define FDG_MATRICES_H
#include "../common/error.h"

typedef struct
{
    unsigned rows;
    unsigned cols;
    double *values;
} matrix_t;

/** Perform a QR decomposition using Givens rotations.
 *
 * @param ar[in,out] Input matrix, which becomes the upper-triangular matrix.
 * @param q[in,out] Matrix, which becomes the orthogonal Q matrix, such that A = QR
 *
 * @returns FDG_SUCCESS if successful, FDG_ERROR_MATRIX_DIMS_MISMATCH if matrices
 * do not have correct dimensions.
 */
interp_result_t matrix_qr_decompose(const matrix_t *ar, const matrix_t *q);

/**
 * Perform matrix multiplication of two input matrices.
 *
 * @param a[in] The first input matrix with dimensions (rows x common_dim).
 * @param b[in] The second input matrix with dimensions (common_dim x cols).
 * @param c[out] The output matrix where the result of the multiplication is stored, with dimensions (rows x cols).
 *
 * @returns FDG_SUCCESS if successful, FDG_ERROR_MATRIX_DIMS_MISMATCH if the dimensions of the matrices are
 * incompatible.
 */
interp_result_t matrix_multiply(const matrix_t *a, const matrix_t *b, const matrix_t *c);

/** Solve the system U X = B using back substitution for an upper-triangular U.
 *
 * If U is not square, the bottom part is assumed to be zero, such that the top square part of U is
 * upper triangular.
 *
 * @param upper Upper triangular matrix U to solve using back substitution.
 * @param b Matrix to solve for inplace.
 *
 * @returns FDG_SUCCESS if successful, FDG_ERROR_MATRIX_DIMS_MISMATCH if the dimensions of the matrices are
 * incompatible.
 */
interp_result_t matrix_back_substitute(const matrix_t *upper, const matrix_t *b);

#endif // FDG_MATRICES_H
