#include "../../src/operations/topology.h"
#include "../common/common.h"

static void test_unique_lookup(void)
{
    const topo_obj_immersion_t immersion = {
        .object_count = 3,
        .parent_dims = 2,
        .element_offsets = (uint64_t[]){0, 1, 3, 4},
        .element_ids = (uint64_t[]){0, 0, 1, 2},
        .element_orientation = (int8_t[]){-1, 2, -2, 1, 1, -2, 2, -1},
    };
    int8_t orientations[4] = {};
    uint64_t object_id = UINT64_MAX;

    TEST_ASSERTION(topo_obj_find_common_boundary(&immersion, 2, 0, 1, &object_id, orientations) == TOPO_SUCCESS,
                   "Could not find unique common boundary.");
    TEST_ASSERTION(object_id == 1, "Unexpected common boundary object.");
    TEST_ASSERTION(orientations[0] == -2 && orientations[1] == 1, "Unexpected first element orientation.");
    TEST_ASSERTION(orientations[2] == 1 && orientations[3] == -2, "Unexpected second element orientation.");
}

static void test_missing_lookup(void)
{
    const topo_obj_immersion_t immersion = {
        .object_count = 2,
        .parent_dims = 2,
        .element_offsets = (uint64_t[]){0, 1, 2},
        .element_ids = (uint64_t[]){0, 1},
        .element_orientation = (int8_t[]){-1, 2, 1, -2},
    };
    int8_t orientations[4] = {};
    uint64_t object_id;

    TEST_ASSERTION(topo_obj_find_common_boundary(&immersion, 2, 0, 1, &object_id, orientations) ==
                       TOPO_NO_COMMON_BOUNDARY,
                   "Missing common boundary was accepted.");
    TEST_ASSERTION(topo_obj_find_common_boundary(&immersion, 2, 0, 0, &object_id, orientations) ==
                       TOPO_NO_COMMON_BOUNDARY,
                   "Identical elements were accepted as a boundary pair.");
}

static void test_multiple_lookup(void)
{
    const topo_obj_immersion_t immersion = {
        .object_count = 2,
        .parent_dims = 2,
        .element_offsets = (uint64_t[]){0, 2, 4},
        .element_ids = (uint64_t[]){0, 1, 0, 1},
        .element_orientation = (int8_t[]){-1, 2, 1, -2, -2, 1, 2, -1},
    };
    int8_t orientations[4] = {};
    uint64_t object_id;

    TEST_ASSERTION(topo_obj_find_common_boundary(&immersion, 2, 0, 1, &object_id, orientations) ==
                       TOPO_MULTIPLE_COMMON_BOUNDARIES,
                   "Multiple common boundaries were not rejected.");
}

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
    TEST_ASSERTION(topo_obj_boundary_orientation(&immersion, 2, 1, 1, orientation) == TOPO_SUCCESS,
                   "Could not find the element boundary orientation.");
    TEST_ASSERTION(orientation[0] == 1 && orientation[1] == -2, "Unexpected element boundary orientation.");
    TEST_ASSERTION(topo_obj_boundary_orientation(&immersion, 2, 0, 1, orientation) == TOPO_NO_COMMON_BOUNDARY,
                   "Missing element boundary was accepted.");
}

int main(void)
{
    test_unique_lookup();
    test_missing_lookup();
    test_multiple_lookup();
    test_boundary_orientation();
    return 0;
}
