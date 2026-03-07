#include "../common/common.h"

typedef struct
{
    unsigned idim;
    unsigned level;
    unsigned index;
    long int offset;
} loop_state_t;
static void scale_array_boundary_iterative(const unsigned ndim, const long int dims[static ndim],
                                           const long int strides[static ndim], loop_state_t work_stack[ndim - 1],
                                           double *restrict out_array)
{

    unsigned spos = 1;
    work_stack[0] = (loop_state_t){.idim = ndim, .level = 0, .index = 0, .offset = 0};

    while (spos > 0)
    {
        const loop_state_t state = work_stack[spos - 1];
        const unsigned idim = state.idim;
        const unsigned level = state.level;
        const long int offset = state.offset;
        const unsigned index = state.index;

        if (index == dims[idim - 1])
        {
            // Pop off the stack
            spos -= 1;
        }
        else
        {
            work_stack[spos - 1].index += 1;

            const unsigned new_level = level + (index == 0 || index + 1 == dims[idim - 1]);
            const long new_offset = offset + index * strides[idim - 1];
            if (idim > 2)
            {
                // Push on a new frame (level depends on if we are first/last
                work_stack[spos] = (loop_state_t){.idim = idim - 1, .level = new_level, .offset = new_offset};
                spos += 1;
                ASSERT(spos < ndim, "I miscounted loop stack space needed.");
            }
            else if (new_level > 0)
            {
                const long int dim = dims[0];
                const long int stride = strides[0];
                out_array[new_offset] /= (new_level + 1.0);
                for (long int i = 1; i < dim - 1 && new_level > 1; ++i)
                {
                    out_array[new_offset + i * stride] /= new_level;
                }
                out_array[new_offset + (dim - 1) * stride] /= (new_level + 1.0);
            }
        }
    }
}

enum
{
    NDIM = 4,

    DIM_1 = 3,
    DIM_2 = 2,
    DIM_3 = 4,
    DIM_4 = 5,
    TOTAL_ELEMENTS = DIM_1 * DIM_2 * DIM_3 * DIM_4,

    STRIDE_1 = DIM_2 * DIM_3 * DIM_4,
    STRIDE_2 = DIM_3 * DIM_4,
    STRIDE_3 = DIM_4,
    STRIDE_4 = 1,
};

void run_the_test_4d(const long int dims[static 4], const long int strides[static 4])
{
    double *const test_array = malloc(TOTAL_ELEMENTS * sizeof(*test_array));
    TEST_ASSERTION(test_array != NULL, "Failed to allocate test array");
    // Init the array with all 1s
    for (unsigned i = 0; i < TOTAL_ELEMENTS; ++i)
        test_array[i] = 1.0;

    // Run the method
    scale_array_boundary_iterative(4, dims, strides, (loop_state_t[3]){}, test_array);

    // Manually check it the slow way
    for (unsigned i1 = 0; i1 < dims[0]; ++i1)
    {
        const unsigned l1 = i1 == 0 || i1 == dims[0] - 1;
        for (unsigned i2 = 0; i2 < dims[1]; ++i2)
        {
            const unsigned l2 = i2 == 0 || i2 == dims[1] - 1;
            for (unsigned i3 = 0; i3 < dims[2]; ++i3)
            {
                const unsigned l3 = (i3 == 0 || i3 == dims[2] - 1);
                for (unsigned i4 = 0; i4 < dims[3]; ++i4)
                {
                    const unsigned l4 = (i4 == 0 || i4 == dims[3] - 1);
                    const unsigned l = l1 + l2 + l3 + l4;
                    TEST_NUMBERS_CLOSE(
                        test_array[i1 * strides[0] + i2 * strides[1] + i3 * strides[2] + i4 * strides[3]],
                        1 / (double)l, 1e-12, 1e-10);
                }
            }
        }
    }

    free(test_array);
}

void run_the_test_3d(const long int dims[static 3], const long int strides[static 3])
{
    const size_t total_elements = dims[0] * dims[1] * dims[2];
    double *const test_array = malloc(total_elements * sizeof(*test_array));
    TEST_ASSERTION(test_array != NULL, "Failed to allocate test array");
    // Init the array with all 1s
    for (unsigned i = 0; i < total_elements; ++i)
        test_array[i] = 1.0;

    // Run the method
    scale_array_boundary_iterative(3, dims, strides, (loop_state_t[2]){}, test_array);

    // Manually check it the slow way
    for (unsigned i1 = 0; i1 < dims[0]; ++i1)
    {
        const unsigned l1 = i1 == 0 || i1 == dims[0] - 1;
        for (unsigned i2 = 0; i2 < dims[1]; ++i2)
        {
            const unsigned l2 = i2 == 0 || i2 == dims[1] - 1;
            for (unsigned i3 = 0; i3 < dims[2]; ++i3)
            {
                const unsigned l3 = (i3 == 0 || i3 == dims[2] - 1);
                const unsigned l = l1 + l2 + l3;
                TEST_NUMBERS_CLOSE(test_array[i1 * strides[0] + i2 * strides[1] + i3 * strides[2]], 1 / (double)l,
                                   1e-12, 1e-10);
            }
        }
    }

    free(test_array);
}

void run_the_test_2d(const long int dims[static 2], const long int strides[static 2])
{
    const size_t total_elements = dims[0] * dims[1];
    double *const test_array = malloc(total_elements * sizeof(*test_array));
    TEST_ASSERTION(test_array != NULL, "Failed to allocate test array");
    // Init the array with all 1s
    for (unsigned i = 0; i < total_elements; ++i)
        test_array[i] = 1.0;

    // Run the method
    scale_array_boundary_iterative(2, dims, strides, (loop_state_t[1]){}, test_array);

    // Manually check it the slow way
    for (unsigned i1 = 0; i1 < dims[0]; ++i1)
    {
        const unsigned l1 = i1 == 0 || i1 == dims[0] - 1;
        for (unsigned i2 = 0; i2 < dims[1]; ++i2)
        {
            const unsigned l2 = i2 == 0 || i2 == dims[1] - 1;

            const unsigned l = l1 + l2;
            TEST_NUMBERS_CLOSE(test_array[i1 * strides[0] + i2 * strides[1]], 1 / (double)l, 1e-12, 1e-10);
        }
    }

    free(test_array);
}

int main()
{

    const long int dims[NDIM] = {DIM_1, DIM_2, DIM_3, DIM_4};
    const long int strides[NDIM] = {STRIDE_1, STRIDE_2, STRIDE_3, STRIDE_4};

    run_the_test_4d(dims, strides);
    run_the_test_3d(dims + 1, strides + 1);
    run_the_test_2d(dims + 2, strides + 2);
    return 0;
}
