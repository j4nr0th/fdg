#include "../../src/topology/topology.h"
#include "../common/common.h"

static void test_boundary_orientation(void)
{
    const topo_obj_immersion_t immersion = {
        .object_count = 2,
        .parent_dims = 2,
        .element_offsets = (uint64_t[]){0, 1, 2},
        .element_ids = (uint64_t[]){0, 1},
        .element_orientation = (int8_t[]){-1, 2, 1, -2},
    };
    int8_t orientation[2];
    TEST_ASSERTION(topo_obj_boundary_orientation(&immersion, 2, 1, 1, orientation),
                   "Could not find the element boundary orientation.");
    TEST_ASSERTION(orientation[0] == 1 && orientation[1] == -2, "Unexpected element boundary orientation.");
    TEST_ASSERTION(!topo_obj_boundary_orientation(&immersion, 2, 0, 1, orientation),
                   "Missing element boundary was accepted.");
}

int main(void)
{
    test_boundary_orientation();
    return 0;
}
