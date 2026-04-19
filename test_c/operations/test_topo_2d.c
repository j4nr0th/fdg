#include "../../src/operations/topology.h"
#include "../common/common.h"
#include "topo_common.h"

int main()
{
    /* Create a messed up 2D mesh that looks like this:
     *
     *
     * Nodes
     * 1---------->3<----------2---------->4
     * |           ^           ^           |
     * |           |           |           |
     * |           |           |           |
     * v           |           |           v
     * 8---------->5<----------6<----------7
     * ^                                   |
     * |                                   |
     * \-----------------------------------/
     *
     * Edges
     * +---------->+<----------+---------->+
     * |    1      ^     2     ^     3     |
     * |           |           |           |
     * | 8         | 9         | 10        | 11
     * v           |           |           v
     * +---------->+<----------+<----------+
     * ^    4            5           6     |
     * |                                   |
     * \-----------------------------------/
     *                   7
     *
     * Surfaces
     * +---------->+<----------+---------->+
     * |    y      ^           ^           |
     * |    1x     |    x2     |    y4     |
     * |           |     y     |     x     |
     * v           |           |           v
     * +---------->+<----------+<----------+
     * ^     x         y3                  |
     * |                                   |
     * \-----------------------------------/
     *
     *
     *
     */

    // Define some global parameters (fuck you Fortran, this is how a good language does it!)
    enum
    {
        POINT_COUNT = 8,
        LINE_COUNT = 11,
        SURFACE_COUNT = 4,
        DIM_COUNT = 2,
    };

    const topo_obj_collection_t line_mesh = {
        .ndim = 1,
        .count = LINE_COUNT,
        .boundary_ids =
            (int64_t[LINE_COUNT * (2 * 1)]){
                // Line 1 (1->3)
                1,
                3,
                // Line 2 (2->3)
                2,
                3,
                // Line 3 (2->4)
                2,
                4,
                // Line 4 (8->5)
                8,
                5,
                // Line 5 (6->5)
                6,
                5,
                // Line 6 (7->6)
                7,
                6,
                // Line 7 (7->8)
                7,
                8,
                // Line 8 (1->8)
                1,
                8,
                // Line 9 (5->3)
                5,
                3,
                // Line 10 (6->2)
                6,
                2,
                // Line 11 (4->7)
                4,
                7,
            },
    };

    const topo_obj_collection_t surface_mesh = {
        .ndim = 2,
        .count = SURFACE_COUNT,
        .boundary_ids =
            (int64_t[SURFACE_COUNT * (2 * 2)]){
                // Surface 1 LINES (9, 1, 8, 4) POINTS (3, 1, 8, 5)
                9,
                1,
                8,
                4,
                // Surface 2 LINES (9, 5, 10, 2) POINTS (5, 6, 2, 3)
                9,
                5,
                10,
                2,
                // Surface 3 LINES (4, 5, 6, 7) POINTS (5, 6, 7, 8)
                4,
                5,
                6,
                7,
                // Surface 4 LINES (6, 10, 3, 11) POINTS (6, 2, 4, 7)
                6,
                10,
                3,
                11,
            },
    };

    // Immersion output to compute
    topo_obj_immersion_t immersions[DIM_COUNT] = {0};

    // Exact correct result
    const topo_obj_immersion_t correct_immersion_points = {
        .object_count = POINT_COUNT,
        .parent_dims = DIM_COUNT,
        .element_offsets = (uint64_t[POINT_COUNT + 1]){0, 1, 3, 5, 6, 9, 12, 14, 16},
        .element_ids =
            (uint64_t[16]){
                // Remember: 0-based!
                // Point 1 is in element 1
                0,
                // Point 2 is in elements 2 and 4
                1,
                3,
                // Point 3 is in elements 1 and 2
                0,
                1,
                // Point 4 is in element 4
                3,
                // Point 5 is in elements 1, 2, and 3
                0,
                1,
                2,
                // Point 6 is in elements 2, 3, and 4
                1,
                2,
                3,
                // Point 7 is in elements 3 and 4
                2,
                3,
                // Point 8 is in elements 1 and 3
                0,
                2,
            },
        .element_orientation = (int8_t[16 * DIM_COUNT]){
            // Point 1
            //  In element 1
            +1,
            -2,
            // Point 2
            //  In element 2
            +1,
            +2,
            //  In element 4
            +1,
            -2,
            // Point 3
            //  In element 1
            -1,
            -2,
            //  In element 2
            -1,
            +2,
            // Point 4
            //  In element 4
            +1,
            +2,
            // Point 5
            //  In element 1
            -1,
            +2,
            //  In element 2
            -1,
            -2,
            //  In element 3
            -1,
            -2,
            // Point 6
            //  In element 2
            +1,
            -2,
            //  In element 3
            +1,
            -2,
            //  In element 4
            -1,
            -2,
            // Point 7
            //  In element 3
            +1,
            +2,
            //  In element 4
            -1,
            +2,
            // Point 8
            //  In element 1
            +1,
            +2,
            //  In element 3
            -1,
            +2,
        }};
    const topo_obj_immersion_t correct_immersion_lines = {
        .object_count = LINE_COUNT,
        .parent_dims = DIM_COUNT,
        .element_offsets = (uint64_t[LINE_COUNT + 1]){0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 15, 16},
        .element_ids =
            (uint64_t[16]){
                // Line 1 is in element 1
                0,
                // Line 2 is in element 2
                1,
                // Line 3 is in element 4
                3,
                // Line 4 is in elements 1 and 3
                0,
                2,
                // Line 5 is in elements 2 and 3
                1,
                2,
                // Line 6 is in elements 3 and 4
                2,
                3,
                // Line 7 is in element 3
                2,
                // Line 8 is in element 1
                0,
                // Line 9 is in elements 1 and 2
                0,
                1,
                // Line 10 is in elements 2 and 4
                1,
                3,
                // Line 11 is in element 4
                3,
            },
        .element_orientation =
            (int8_t[16 * DIM_COUNT]){
                // Line 1
                //  In element 1
                -2,
                -1,
                // Line 2
                //  In element 2
                +2,
                -1,
                // Line 3
                //  In element 4
                +1,
                +2,
                // Line 4
                //  In element 1
                +2,
                -1,
                //  In element 3
                -1,
                -2,
                // Line 5
                //  In element 2
                -2,
                -1,
                //  In element 3
                -2,
                -1,
                // Line 6
                //  In element 3
                +1,
                -2,
                //  In element 4
                -1,
                -2,
                // Line 7
                //  In element 3
                +2,
                -1,
                // Line 8
                //  In element 1
                +1,
                +2,
                // Line 9
                //  In element 1
                -1,
                -2,
                //  In element 2
                -1,
                +2,
                // Line 10
                //  In element 2
                +1,
                +2,
                //  In element 4
                -2,
                +1,
                // Line 11
                //  In element 4
                +2,
                -1,
            },
    };

    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    const int res = topo_obj_create_immersion_info(DIM_COUNT, POINT_COUNT,
                                                   (const topo_obj_collection_t[DIM_COUNT]){line_mesh, surface_mesh},
                                                   allocator, immersions);
    TEST_ASSERTION(res == 0, "Failed function call with return value %d", res);

    // Check that computed immersion is correct
    assert_immersion_equal(DIM_COUNT, immersions + 1, &correct_immersion_lines);
    printf("Lines are correct!\n");
    assert_immersion_equal(DIM_COUNT, immersions + 0, &correct_immersion_points);
    printf("Points are correct!\n");
    topo_obj_immersions_free(DIM_COUNT, immersions, allocator);

    return 0;
}
