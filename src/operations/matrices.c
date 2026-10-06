#include "matrices.h"

#include <cutl/common_defs.h>
#include <math.h>

// static void dbg_print_matrix(const matrix_t *m)
// {
//     for (unsigned i = 0; i < m->rows; ++i)
//     {
//         for (unsigned j = 0; j < m->cols; ++j)
//             printf("%g ", m->values[i * m->cols + j]);
//         printf("\n");
//     }
// }

void matrix_qr_decompose(const matrix_t *const ar, const matrix_t *const q)
{
    // printf("matrix_qr_decompose called with A:\n");
    // dbg_print_matrix(ar);

    const unsigned rows = ar->rows;
    CUTL_ASSERT(q->rows == rows && q->cols == rows,
                "Q matrix dimensions (%u x %u) do not match the input matrix rows (%u).", q->rows, q->cols, rows);
    const unsigned cols = ar->cols;

    // Initialize Q to be identity
    for (unsigned i = 0; i < rows; ++i)
    {
        for (unsigned j = 0; j < rows; ++j)
            q->values[i * rows + j] = (double)(i == j);
    }

    // Check if we can even do anything
    if (rows == 1)
        return;

    double *const r = ar->values;
    double *const qt = q->values;

    // Use Givens rotations to reduce input matrix into upper triangular
    for (unsigned row = 0; row < rows; ++row)
    {
        for (unsigned col = 0; col < cols && col < row; ++col)
        {
            CUTL_ASSERT(row < rows && col < cols,
                        "Givens rotation indexes outside the matrix (row %u, col %u of %u x %u).", row, col, rows,
                        cols);
            double givens_c = r[col * cols + col];
            double givens_s = -r[row * cols + col];
            const double givens_mag = hypot(givens_c, givens_s);
            if (givens_mag < 1e-12)
            {
                // We're probably on a zeroed out part, just skip
                continue;
            }
            givens_c /= givens_mag;
            givens_s /= givens_mag;

            // Apply the rotation to the rows involved in the rotation
            CUTL_ASSERT(fabs(givens_s * r[col * cols + col] + givens_c * r[row * cols + col]) < 1e-12,
                        "Givens rotation somehow does not properly eliminate the entries (c: %g, s:%g, with matrix "
                        "entries %g and %g).",
                        givens_c, givens_s, r[col * cols + col], r[row * cols + col]);

            for (unsigned k = col; k < cols; ++k)
            {
                const double v1 = r[col * cols + k];
                const double v2 = r[row * cols + k];
                r[col * cols + k] = +givens_c * v1 - givens_s * v2;
                r[row * cols + k] = +givens_s * v1 + givens_c * v2;
            }

            CUTL_ASSERT(fabs(r[row * cols + col]) < 1e-12,
                        "Givens rotation failed to reduce matrix to upper triangular (entry was %g)",
                        r[row * cols + col]);
            r[row * cols + col] = 0;

            // Apply the of Givens rotation to the matrix qt
            for (unsigned k = 0; k < rows; ++k)
            {
                const double temp = qt[col * rows + k];
                qt[col * rows + k] = +givens_c * temp - givens_s * qt[row * rows + k];
                qt[row * rows + k] = +givens_s * temp + givens_c * qt[row * rows + k];
            }
        }
    }
}

void matrix_multiply(const matrix_t *a, const matrix_t *b, const matrix_t *c)
{
    const unsigned rows = a->rows;
    const unsigned cols = b->cols;
    const unsigned k = a->cols;

    // Does output have correct size?
    CUTL_ASSERT(rows == c->rows && cols == c->cols,
                "Output matrix dimensions (%u x %u) do not match the result (%u x %u).", c->rows, c->cols, rows, cols);

    // Do the inputs match?
    CUTL_ASSERT(k == b->rows, "Inner dimensions do not match (%u vs %u).", k, b->rows);

    for (unsigned i = 0; i < rows; ++i)
    {
#pragma omp simd
        for (unsigned j = 0; j < cols; ++j)
        {
            double sum = 0;
            for (unsigned l = 0; l < k; ++l)
                sum += a->values[i * k + l] * b->values[l * cols + j];
            c->values[i * cols + j] = sum;
        }
    }
}

void matrix_back_substitute(const matrix_t *upper, const matrix_t *b)
{
    const unsigned u_cols = upper->cols;
    CUTL_ASSERT(u_cols <= b->rows, "Upper triangular matrix has more columns (%u) than the system has rows (%u).",
                u_cols, b->rows);
    const unsigned b_cols = b->cols;
    // Do each column of B separately
#pragma omp simd
    for (unsigned bcol = 0; bcol < b_cols; ++bcol)
    {
        double *const b_ptr = b->values + bcol;
        // Back substitution
        for (unsigned row = u_cols; row > 0; --row)
        {
            double value = b_ptr[(row - 1) * b_cols];
            for (unsigned col = row; col < u_cols; ++col)
                value -= upper->values[(row - 1) * u_cols + col] * b_ptr[col * b_cols];
            b_ptr[(row - 1) * b_cols] = value / upper->values[(row - 1) * u_cols + (row - 1)];
        }
    }
}
