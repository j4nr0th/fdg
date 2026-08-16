#include "../common/common.h"
#include "topo_common.h"
#include <stdbool.h>
#include <string.h>

/**
 * Check boundary orientation matches the expected one.
 *
 * @param ndim Number of dimensions.
 * @param parent_orientation Orientation of the parent in the global reference frame.
 * @param computed_orientation Computed global orientation to check.
 * @param expected_orientation Expected in the local reference frame, which should be transferred to the global
 * reference frame before comparing.
 * @return When the two computed and expected orientations in the global reference frame do not match zero, otherwise
 * nonzero.
 */
static bool check_boundaries_match(const unsigned ndim, const int8_t parent_orientation[static ndim],
                                   const int8_t computed_orientation[static ndim],
                                   const int8_t expected_orientation[static ndim])
{
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        const int8_t computed = computed_orientation[idim];
        const int8_t expected_local = expected_orientation[idim];
        int8_t expected_global = parent_orientation[expected_local > 0 ? expected_local - 1 : -expected_local - 1];
        if (expected_local < 0)
            expected_global *= -1;
        if (computed != expected_global)
            return false;
    }
    return true;
}

int main()
{
    enum
    {
        NDIM = 4
    };
    const topo_obj_collection_t collection_bnd = {.ndim = NDIM - 1,
                                                  .count = 2LLU * NDIM,
                                                  .boundary_ids = (uint64_t[2 * (NDIM - 1) * (2 * NDIM)]){
                                                      // First volume (nothing in common with 5th)
                                                      0,
                                                      1,
                                                      2,
                                                      3,
                                                      4,
                                                      5,
                                                      // Second volume (nothing in common with 6th)
                                                      0,
                                                      6,
                                                      7,
                                                      8,
                                                      9,
                                                      10,
                                                      // Third volume (nothing in common with 7th)
                                                      1,
                                                      6,
                                                      11,
                                                      12,
                                                      13,
                                                      14,
                                                      // Fourth volume (nothing in common with 8th)
                                                      2,
                                                      7,
                                                      11,
                                                      15,
                                                      16,
                                                      17,
                                                      // Fifth volume
                                                      12,
                                                      8,
                                                      15,
                                                      18,
                                                      19,
                                                      20,
                                                      // Sixth volume
                                                      3,
                                                      13,
                                                      16,
                                                      18,
                                                      21,
                                                      22,
                                                      // Seventh volume
                                                      19,
                                                      21,
                                                      23,
                                                      4,
                                                      9,
                                                      17,
                                                      // Eighth volume
                                                      5,
                                                      23,
                                                      10,
                                                      20,
                                                      14,
                                                      22,
                                                  }};
    const uint64_t boundary_arrangement[2 * NDIM] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int8_t parent_orientation[NDIM] = {+3, -4, +1, -2};
    const int8_t proper_orientation_vol1[NDIM] = {-1, +2, +3, +4};
    const int8_t proper_orientation_vol2[NDIM] = {-2, +1, +3, +4};
    const int8_t proper_orientation_vol3[NDIM] = {-3, +1, +2, +4};
    const int8_t proper_orientation_vol4[NDIM] = {-4, +1, +2, +3};
    const int8_t proper_orientation_vol5[NDIM] = {+1, +3, +2, +4};
    const int8_t proper_orientation_vol6[NDIM] = {+2, +1, +3, +4};
    const int8_t proper_orientation_vol7[NDIM] = {+3, -1, -2, -4};
    const int8_t proper_orientation_vol8[NDIM] = {+4, +1, -3, +2};
    int8_t result_orientation[NDIM] = {0};
    // Volume 1
    int stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 0, 0, parent_orientation,
                                                  boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol1),
                   // memcmp(result_orientation, proper_orientation_vol1, sizeof(int8_t) * NDIM) == 0,
                   "Computed immersion did not match expected result.");

    // Volume 2
    stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 1, 0, parent_orientation,
                                              boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol2),
                   "Computed immersion did not match expected result.");

    // Volume 3
    stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 2, 0, parent_orientation,
                                              boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol3),
                   "Computed immersion did not match expected result.");

    // Volume 4
    stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 3, 0, parent_orientation,
                                              boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol4),
                   "Computed immersion did not match expected result.");

    // Volume 5
    stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 4, 0, parent_orientation,
                                              boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol5),
                   "Computed immersion did not match expected result.");

    // Volume 6
    stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 5, 0, parent_orientation,
                                              boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol6),
                   "Computed immersion did not match expected result.");

    // Volume 7
    stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 6, 0, parent_orientation,
                                              boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol7),
                   "Computed immersion did not match expected result.");

    // Volume 8
    stat = topo_obj_boundary_immersion_create(NDIM, collection_bnd.ndim, &collection_bnd, 7, 0, parent_orientation,
                                              boundary_arrangement, result_orientation);
    TEST_ASSERTION(stat == 0, "Failed function call with return value %d", stat);
    TEST_ASSERTION(check_boundaries_match(NDIM, parent_orientation, result_orientation, proper_orientation_vol8),
                   "Computed immersion did not match expected result.");
    return 0;
}
