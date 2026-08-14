#include "../../src/operations/topology.h"
#include "../common/common.h"

typedef struct
{
    unsigned count;
    uint64_t boundary_indices[8];
    uint64_t offsets_1[8];
    uint64_t offsets_2[8];
} iteration_capture_t;

static void capture_iteration(const uint64_t boundary_index, const uint64_t offset_1, const uint64_t offset_2,
                              void *user_data)
{
    iteration_capture_t *const capture = user_data;
    TEST_ASSERTION(capture->count < 8, "Too many boundary iterations.");
    capture->boundary_indices[capture->count] = boundary_index;
    capture->offsets_1[capture->count] = offset_1;
    capture->offsets_2[capture->count] = offset_2;
    capture->count += 1;
}

static void test_reorder_orientation(void)
{
    const uint64_t sizes[] = {3};
    const int8_t forward[] = {1};
    const int8_t reverse[] = {-1};
    uint64_t offsets[1];
    uint64_t strides[1];
    const double input[] = {10, 20, 30};
    double output[3];

    topo_reorder_with_orientation(1, 1, (topo_bnd_iter_t){forward, sizes, offsets, strides}, input, output);
    TEST_NUMBERS_CLOSE(output[0], 10, 0, 0);
    TEST_NUMBERS_CLOSE(output[1], 20, 0, 0);
    TEST_NUMBERS_CLOSE(output[2], 30, 0, 0);

    topo_reorder_with_orientation(1, 1, (topo_bnd_iter_t){reverse, sizes, offsets, strides}, input, output);
    TEST_NUMBERS_CLOSE(output[0], 30, 0, 0);
    TEST_NUMBERS_CLOSE(output[1], 20, 0, 0);
    TEST_NUMBERS_CLOSE(output[2], 10, 0, 0);
}

static void test_reorder_fixed_axis(void)
{
    const uint64_t sizes[] = {2, 3};
    const int8_t lower[] = {-1, 2};
    const int8_t upper[] = {1, 2};
    uint64_t offsets[1];
    uint64_t strides[2];
    const double input[] = {10, 20, 30};
    double output[6] = {};

    topo_reorder_with_orientation(2, 1, (topo_bnd_iter_t){lower, sizes, offsets, strides}, input, output);
    TEST_NUMBERS_CLOSE(output[0], 10, 0, 0);
    TEST_NUMBERS_CLOSE(output[2], 20, 0, 0);
    TEST_NUMBERS_CLOSE(output[4], 30, 0, 0);

    output[0] = output[2] = output[4] = 0;
    topo_reorder_with_orientation(2, 1, (topo_bnd_iter_t){upper, sizes, offsets, strides}, input, output);
    TEST_NUMBERS_CLOSE(output[1], 10, 0, 0);
    TEST_NUMBERS_CLOSE(output[3], 20, 0, 0);
    TEST_NUMBERS_CLOSE(output[5], 30, 0, 0);
}

static void test_boundary_iteration(void)
{
    const uint64_t sizes[] = {3};
    const int8_t forward[] = {1};
    const int8_t reverse[] = {-1};
    uint64_t offsets_1[1], offsets_2[1];
    uint64_t strides_1[1], strides_2[1];
    iteration_capture_t capture = {};

    topo_iterate_boundary(1, 1, (topo_bnd_iter_t){forward, sizes, offsets_1, strides_1},
                          (topo_bnd_iter_t){reverse, sizes, offsets_2, strides_2}, false, capture_iteration, &capture);

    TEST_ASSERTION(capture.count == 3, "Expected three boundary points.");
    for (unsigned i = 0; i < capture.count; ++i)
    {
        TEST_ASSERTION(capture.boundary_indices[i] == i, "Boundary indices were not emitted in order.");
        TEST_ASSERTION(capture.offsets_1[i] == i, "Forward offset was incorrect.");
        TEST_ASSERTION(capture.offsets_2[i] == 2 - i, "Reverse offset was incorrect.");
    }
}

static void test_scalar_boundary_iteration(void)
{
    const uint64_t sizes[] = {4};
    const int8_t lower[] = {-1};
    const int8_t upper[] = {1};
    uint64_t offsets_1[1], offsets_2[1];
    uint64_t strides_1[1], strides_2[1];
    iteration_capture_t capture = {};

    topo_iterate_boundary(1, 0, (topo_bnd_iter_t){lower, sizes, offsets_1, strides_1},
                          (topo_bnd_iter_t){upper, sizes, offsets_2, strides_2}, false, capture_iteration, &capture);

    TEST_ASSERTION(capture.count == 1, "Expected one scalar boundary point.");
    TEST_ASSERTION(capture.boundary_indices[0] == 0, "Scalar boundary index was incorrect.");
    TEST_ASSERTION(capture.offsets_1[0] == 0, "Lower boundary offset was incorrect.");
    TEST_ASSERTION(capture.offsets_2[0] == 3, "Upper boundary offset was incorrect.");
}

static void test_skip_edges(void)
{
    const uint64_t sizes[] = {4};
    const int8_t orientation[] = {1};
    uint64_t offsets_1[1], offsets_2[1];
    uint64_t strides_1[1], strides_2[1];
    iteration_capture_t capture = {};

    topo_iterate_boundary(1, 1, (topo_bnd_iter_t){orientation, sizes, offsets_1, strides_1},
                          (topo_bnd_iter_t){orientation, sizes, offsets_2, strides_2}, true, capture_iteration,
                          &capture);

    TEST_ASSERTION(capture.count == 2, "Expected only interior boundary points.");
    TEST_ASSERTION(capture.offsets_1[0] == 1 && capture.offsets_1[1] == 2, "Edge points were not skipped.");
}

int main(void)
{
    test_reorder_orientation();
    test_reorder_fixed_axis();
    test_boundary_iteration();
    test_scalar_boundary_iteration();
    test_skip_edges();
    return 0;
}
