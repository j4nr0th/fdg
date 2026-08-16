#include "../../src/topology/topology.h"
#include "../common/common.h"

int main(void)
{
    const topo_obj_collection_t collections[] = {
        {
            .ndim = 1,
            .count = 20,
            .boundary_ids =
                (uint64_t[40]){
                    0, 1, 1, 2, 2, 3, 3, 0,  4,  5,  5,  6, 6, 7, 7, 4, 0, 4,  1, 5,
                    2, 6, 3, 7, 8, 9, 9, 10, 10, 11, 11, 8, 0, 8, 1, 9, 5, 10, 4, 11,
                },
        },
        {
            .ndim = 2,
            .count = 11,
            .boundary_ids =
                (uint64_t[44]){
                    0, 1,  2, 3,  4,  5,  6, 7,  0,  9,  4, 8,  1,  10, 5, 9,  2,  11, 6,  10, 3,  8,
                    7, 11, 0, 17, 12, 16, 9, 18, 13, 17, 4, 18, 14, 19, 8, 19, 15, 16, 12, 13, 14, 15,
                },
        },
        {
            .ndim = 3,
            .count = 2,
            .boundary_ids = (uint64_t[12]){5, 2, 0, 3, 4, 1, 2, 6, 7, 10, 8, 9},
        },
    };
    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    topo_obj_immersion_t immersions[3] = {0};
    const topo_status_t status = topo_obj_create_immersion_info(3, 12, collections, allocator, immersions);
    TEST_ASSERTION(status == TOPO_SUCCESS, "Could not create 3D immersion information: %s (%s)",
                   topo_status_to_str(status), topo_status_msg(status));

    int8_t orientation[3];
    TEST_ASSERTION(topo_obj_boundary_orientation(immersions + 0, 3, 0, 0, orientation) == TOPO_SUCCESS,
                   "Could not find point boundary in first element.");
    TEST_ASSERTION(topo_obj_boundary_orientation(immersions + 1, 3, 0, 0, orientation) == TOPO_SUCCESS,
                   "Could not find line boundary in first element.");
    TEST_ASSERTION(topo_obj_boundary_orientation(immersions + 2, 3, 2, 0, orientation) == TOPO_SUCCESS,
                   "Could not find face boundary in first element.");
    TEST_ASSERTION(topo_obj_boundary_orientation(immersions + 1, 3, 0, 1, orientation) == TOPO_SUCCESS,
                   "Could not find shared line boundary in second element.");
    TEST_ASSERTION(topo_obj_boundary_orientation(immersions + 2, 3, 2, 1, orientation) == TOPO_SUCCESS,
                   "Could not find shared face boundary in second element.");

    topo_obj_immersions_free(3, immersions, allocator);
    return 0;
}
