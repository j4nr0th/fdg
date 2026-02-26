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

int main()
{
    // 6x5 upper triangular (extra row of zeros)
    double upper_triangular[6 * 5] = {
        2.0, 4.2, 0.1, 0.0, 0.0, // Row 1
        0.0, 1.0, 4.2, 0.1, 0.0, // Row 2
        0.0, 0.0, 1.9, 3.1, 0.5, // Row 3
        0.0, 0.0, 0.0, 7.0, 2.0, // Row 4
        0.0, 0.0, 0.0, 0.0, 4.0, // Row 5
        0.0, 0.0, 0.0, 0.0, 0.0, // Last row 6
    };
    const matrix_t upper = {.rows = 6, .cols = 5, .values = upper_triangular};

    // 5x3 test input
    double test_input[5 * 3] = {
        1.2, 3.1, 4.2, // Row 1
        2.0, 4.1, 0.4, // Row 2
        0.1, 0.0, 4.1, // Row 3
        0.5, 5.2, 0.6, // Row 4
        2.0, 0.0, 4.0, // Row 5
    };
    const matrix_t test_input_matrix = {.rows = 5, .cols = 3, .values = test_input};
    matrix_t result = {.rows = 6, .cols = 3, .values = (double[6 * 3]){}};
    interp_result_t res = matrix_multiply(&upper, &test_input_matrix, &result);
    TEST_ASSERTION(res == FDG_SUCCESS, "Matrix multiplication failed");

    printf("Matrix A:\n");
    print_matrix(&upper);
    printf("LHS matrix X:\n");
    print_matrix(&test_input_matrix);
    printf("Product AX:\n");
    print_matrix(&result);

    // Solve the back substitution
    // result.rows = 5;
    res = matrix_back_substitute(&upper, &result);
    TEST_ASSERTION(res == FDG_SUCCESS, "Back substitution failed");

    printf("Result of back substitution:\n");
    print_matrix(&result);

    // Check that the top part of result and test_input_matrix are the same
    for (unsigned row = 0; row < 5; ++row)
    {
        for (unsigned col = 0; col < 3; ++col)
        {
            const double val_computed = result.values[row * 3 + col];
            const double val_expected = test_input_matrix.values[row * 3 + col];
            TEST_NUMBERS_CLOSE(val_expected, val_computed, 1e-12, 1e-10);
        }
    }

    return 0;
}
