//
// Created by jan on 2026-09-13.
//
#include "../common/common.h"

#include "../../src/data/element_geometry.h"

static void test_geometry_roundtrip(void)
{
    element_geometry_t *geometry;
    TEST_FDG_RESULT(element_geometry_create(&geometry, &TEST_ALLOCATOR));

    TEST_ASSERTION(element_geometry_element_count(geometry) == 0, "Empty collection should have no elements.");
    TEST_ASSERTION(element_geometry_option_count(geometry) == 0, "Empty collection should have no options.");
    TEST_ASSERTION(element_geometry_value_count(geometry) == 0, "Empty collection should have no values.");

    const basis_spec_t specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 1},
                                   {.type = BASIS_LAGRANGE_UNIFORM, .order = 1}};
    const integration_spec_t int_specs[2] = {{.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 3},
                                             {.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 3}};
    unsigned index;
    TEST_FDG_RESULT(element_geometry_add_option(geometry, 2, 3, specs, int_specs, &index));
    TEST_ASSERTION(index == 0, "First option should get index 0.");
    TEST_ASSERTION(element_geometry_option_value_count(geometry, 0) == 12,
                   "3 coordinates with order-1 bases in 2D should store 12 values.");

    // Dedup: identical specs return the same index.
    unsigned dedup_index;
    TEST_FDG_RESULT(element_geometry_add_option(geometry, 2, 3, specs, int_specs, &dedup_index));
    TEST_ASSERTION(dedup_index == 0, "Duplicate option should dedup to index 0.");
    TEST_ASSERTION(element_geometry_option_count(geometry) == 1, "Duplicate option should not be added.");

    double values[12];
    for (unsigned i = 0; i < 12; ++i)
        values[i] = (double)i;
    TEST_FDG_RESULT(element_geometry_add_element(geometry, 0, values));
    for (unsigned i = 0; i < 12; ++i)
        values[i] = 100.0 + (double)i;
    TEST_FDG_RESULT(element_geometry_add_element(geometry, 0, values));

    TEST_ASSERTION(element_geometry_element_count(geometry) == 2, "Should have two elements.");
    TEST_ASSERTION(element_geometry_value_count(geometry) == 24, "Should have 24 values.");
    const uint64_t *const offsets = element_geometry_offsets(geometry);
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 12 && offsets[2] == 24, "Offsets should be 0, 12, 24.");
    const double *const stored = element_geometry_values(geometry);
    TEST_NUMBERS_CLOSE(stored[0], 0.0, 1e-14, 0);
    TEST_NUMBERS_CLOSE(stored[12], 100.0, 1e-14, 0);

    for (unsigned i = 0; i < 12; ++i)
        values[i] = -1.0;
    TEST_FDG_RESULT(element_geometry_set_element_values(geometry, 1, values));
    TEST_NUMBERS_CLOSE(stored[12], -1.0, 1e-14, 0);

    // Invalid indices are rejected.
    TEST_ASSERTION(element_geometry_add_element(geometry, 3, values) == FDG_ERROR_NOT_IN_DOMAIN,
                   "Out-of-range option index should be rejected.");
    TEST_ASSERTION(element_geometry_set_element_values(geometry, 2, values) == FDG_ERROR_NOT_IN_DOMAIN,
                   "Out-of-range element id should be rejected.");

    // Coordinate count mismatch is rejected.
    unsigned bad_index;
    TEST_ASSERTION(element_geometry_add_option(geometry, 2, 2, specs, int_specs, &bad_index) == FDG_ERROR_NOT_IN_DOMAIN,
                   "Coordinate count mismatch should be rejected.");

    element_geometry_free(geometry, &TEST_ALLOCATOR);
}

int main(void)
{
    test_geometry_roundtrip();
    return 0;
}
