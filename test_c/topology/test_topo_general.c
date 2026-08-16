#include "../../src/topology/topology.h"
#include "../common/common.h"

int main(void)
{
    const topo_obj_collection_t faces = {
        .ndim = 2,
        .count = 6,
        .boundary_ids =
            (uint64_t[24]){
                8, 4, 10, 6, 8, 0, 9, 2, 4, 0, 5, 1, 5, 9, 7, 11, 10, 1, 11, 3, 6, 2, 7, 3,
            },
    };
    const uint64_t parent_boundaries[] = {0, 1, 2, 3, 4, 5};
    const int8_t parent_orientation[] = {-2, 4, -1, 3};
    int8_t orientation[4];

    TEST_ASSERTION(topo_obj_boundary_immersion_create(4, 2, &faces, 0, 1, parent_orientation, parent_boundaries,
                                                      orientation) == TOPO_SUCCESS,
                   "Could not create a general two-dimensional boundary orientation.");
    TEST_ASSERTION(orientation[0] == -2 && orientation[1] == -4 && orientation[2] == -1 && orientation[3] == 3,
                   "Unexpected general boundary orientation.");
    return 0;
}
