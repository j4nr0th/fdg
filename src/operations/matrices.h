#ifndef FDG_MATRICES_H
#define FDG_MATRICES_H
#include "../common/error.h"

/**
 * @brief Dense row-major matrix with caller-owned storage.
 *
 * The matrix has `rows` rows and `cols` columns, with the entries stored
 * row-major in the array `values`, which must have at least
 * `rows * cols` elements. The struct does not own the `values` array; the
 * caller is responsible for its lifetime.
 */
typedef struct
{
    unsigned rows;  // Number of rows.
    unsigned cols;  // Number of columns.
    double *values; // Entries of the matrix in row-major order, with `rows * cols` elements.
} matrix_t;

/** Perform a QR decomposition using Givens rotations.
 *
 * The input matrix `ar` is reduced in place to the upper-triangular matrix
 * R, and the orthogonal matrix Q is written into `q` such that the original
 * input satisfies A = Q^T R. The decomposition always succeeds for any
 * rectangular input; columns whose remaining rows are numerically zero are
 * skipped.
 *
 * @param ar[in,out] Input matrix, which becomes the upper-triangular matrix.
 *        Must have non-null `values` with `rows * cols` entries. The rows of
 *        the input may not exceed the columns for a full decomposition.
 * @param q[in,out] Matrix, which becomes the orthogonal Q matrix, such that A = QR.
 *        Must have `rows` rows and `rows` columns with non-null `values`.
 *        Its previous contents are overwritten with the identity first.
 */
void matrix_qr_decompose(const matrix_t *ar, const matrix_t *q);

/**
 * Perform matrix multiplication of two input matrices.
 *
 * Computes `c = a * b`. The output matrix `c` must not alias the input
 * matrices. All matrices must have non-null `values` arrays of sufficient
 * size.
 *
 * @param a[in] The first input matrix with dimensions (rows x common_dim).
 * @param b[in] The second input matrix with dimensions (common_dim x cols).
 * @param c[out] The output matrix where the result of the multiplication is stored, with dimensions (rows x cols).
 */
void matrix_multiply(const matrix_t *a, const matrix_t *b, const matrix_t *c);

/** Solve the system U X = B using back substitution for an upper-triangular U.
 *
 * If U is not square, the bottom part is assumed to be zero, such that the top square part of U is
 * upper triangular. The solution overwrites `b` in place. The diagonal of
 * `upper` must be non-zero, otherwise the division is undefined. All
 * matrices must have non-null `values` arrays of sufficient size.
 *
 * @param upper Upper triangular matrix U to solve using back substitution.
 *        Must have at most as many rows as `b` has rows.
 * @param b Matrix to solve for inplace.
 */
void matrix_back_substitute(const matrix_t *upper, const matrix_t *b);

#endif // FDG_MATRICES_H
