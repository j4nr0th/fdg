#include "../../src/operations/topology.h"
#include "../common/common.h"
#include "topo_common.h"

int main()
{
    /* Create a simple 1D mesh that looks like this:
     *
     * 1---------->3<-------------2------------->4
     *       2             1             3
     */
    enum
    {
        POINT_COUNT = 4,
        LINE_COUNT = 3,
        DIM_COUNT = 1
    };

    const topo_obj_collection_t line_mesh = {
        .ndim = 1,
        .count = 3,
        .boundary_ids =
            (int64_t[LINE_COUNT * (2 * DIM_COUNT)]){
                // First line
                2,
                3,
                // Second line
                1,
                3,
                // Third line
                2,
                4,
            },
    };

    // Immersion output to compute
    topo_obj_immersion_t immersion_points = {0};

    // Exact correct result
    const topo_obj_immersion_t correct_immersion = {.object_count = POINT_COUNT,
                                                    .parent_dims = 1,
                                                    .element_offsets = (uint64_t[POINT_COUNT + 1]){0, 1, 3, 5, 6},
                                                    .element_ids =
                                                        (uint64_t[6 * DIM_COUNT]){
                                                            // Remember: 0-based!
                                                            // Point 1
                                                            1,
                                                            // Point 2
                                                            0,
                                                            2,
                                                            // Point 3
                                                            0,
                                                            1,
                                                            // Point 4
                                                            2,
                                                        },
                                                    .element_orientation = (int8_t[6 * DIM_COUNT]){
                                                        // Point 1
                                                        //  In element 2
                                                        -1,
                                                        // Point 2
                                                        //  In element 1
                                                        -1,
                                                        //  In element 3
                                                        -1,
                                                        // Point 3
                                                        //  In element 1
                                                        +1,
                                                        //  In element 2
                                                        +1,
                                                        // Point 4
                                                        //  In element 3
                                                        +1,
                                                    }};

    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    const int res = topo_obj_create_immersion_info(1, POINT_COUNT, &line_mesh, allocator, &immersion_points);
    TEST_ASSERTION(res == 0, "Failed function call with return value %d", res);

    // Check that computed immersion is correct
    assert_immersion_equal(DIM_COUNT, &immersion_points, &correct_immersion);
    topo_obj_immersions_free(DIM_COUNT, &immersion_points, allocator);

    return 0;
}
