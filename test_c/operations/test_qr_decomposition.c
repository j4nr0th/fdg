#include "../../src/operations/matrices.h"
#include "../common/common.h"

static void print_matrix(const matrix_t *m)
{
    for (unsigned i = 0; i < m->rows; ++i)
    {
        for (unsigned j = 0; j < m->cols; ++j)
            printf("%8.3g ", m->values[i * m->cols + j]);
        printf("\n");
    }
}

enum
{
    DIM_1 = 5,
    DIM_2 = 4,
};

int main()
{
    // 5 x 4 matrix
    double matrix_a[DIM_1 * DIM_2] = {
        0.3, 0.2, 0.1, 0.4, // Row 1
        2.1, 1.0, 4.2, 0.1, // Row 2
        0.5, 1.2, 2.2, 1.4, // Row 3
        0.1, 0.0, 2.0, 0.7, // Row 4
        0.6, 0.9, 0.7, 0.6, // Row 5
    };

    const matrix_t a = {.rows = DIM_1, .cols = DIM_2, .values = matrix_a};
    const matrix_t q = {.rows = DIM_1, .cols = DIM_1, .values = (double[DIM_1 * DIM_1]){}};
    const matrix_t r = {.rows = DIM_1, .cols = DIM_2, .values = (double[DIM_1 * DIM_2]){}};
    const matrix_t ra = {.rows = DIM_1, .cols = DIM_2, .values = (double[DIM_1 * DIM_2]){}};

    printf("Initial matrix A:\n");
    print_matrix(&a);

    // Copy A to R
    for (unsigned i = 0; i < DIM_1; ++i)
    {
        for (unsigned j = 0; j < DIM_2; ++j)
            r.values[i * DIM_2 + j] = a.values[i * DIM_2 + j];
    }

    interp_result_t res = matrix_qr_decompose(&r, &q);
    printf("Matrix Q^T:\n");
    print_matrix(&q);
    printf("Matrix R:\n");
    print_matrix(&r);

    TEST_ASSERTION(res == FDG_SUCCESS, "QR decomposition failed");

    // Multiplying A with Q should give R

    res = matrix_multiply(&q, &a, &ra);
    TEST_ASSERTION(res == FDG_SUCCESS, "Multiplication failed");
    printf("Matrix QA:\n");
    print_matrix(&ra);

    for (unsigned i = 0; i < DIM_1; ++i)
    {
        for (unsigned j = 0; j < DIM_2; ++j)
        {
            const double va = r.values[i * DIM_2 + j];
            const double vb = ra.values[i * DIM_2 + j];
            TEST_NUMBERS_CLOSE(va, vb, 1e-12, 1e-10);
        }
    }

    // Transpose matrix Q back
    for (unsigned i = 0; i < DIM_1; ++i)
    {
        for (unsigned j = 0; j < i; ++j)
        {
            const double tmp = q.values[i * DIM_1 + j];
            q.values[i * DIM_1 + j] = q.values[j * DIM_1 + i];
            q.values[j * DIM_1 + i] = tmp;
        }
    }

    printf("Matrix Q:\n");
    print_matrix(&q);

    res = matrix_multiply(&q, &r, &ra);
    TEST_ASSERTION(res == FDG_SUCCESS, "Multiplication failed");

    printf("Re-computed QR:\n");
    print_matrix(&ra);

    // Compare
    for (unsigned i = 0; i < DIM_1; ++i)
    {
        for (unsigned j = 0; j < DIM_2; ++j)
        {
            const double va = a.values[i * DIM_2 + j];
            const double vb = ra.values[i * DIM_2 + j];
            TEST_NUMBERS_CLOSE(va, vb, 1e-12, 1e-10);
        }
    }

    return 0;
}
