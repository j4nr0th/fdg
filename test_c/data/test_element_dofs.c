//
// Created by jan on 2026-09-13.
//
#include "../common/common.h"

#include "../../src/data/element_dofs.h"

static void test_dofs_roundtrip(void)
{
    element_dofs_t *dofs;
    TEST_FDG_RESULT(element_dofs_create(&dofs, &TEST_ALLOCATOR));

    const basis_spec_t specs2[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                    {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    unsigned index2;
    TEST_FDG_RESULT(element_dofs_add_option(dofs, 2, specs2, &index2));
    TEST_ASSERTION(index2 == 0, "First option should get index 0.");
    TEST_ASSERTION(element_dofs_option_value_count(dofs, 0) == 9, "Order-2 2D DoF option should store 9 values.");

    const basis_spec_t specs3[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 3},
                                    {.type = BASIS_LAGRANGE_UNIFORM, .order = 3}};
    unsigned index3;
    TEST_FDG_RESULT(element_dofs_add_option(dofs, 2, specs3, &index3));
    TEST_ASSERTION(index3 == 1, "Distinct option should get the next index.");
    TEST_ASSERTION(element_dofs_option_value_count(dofs, 1) == 16, "Order-3 2D DoF option should store 16 values.");

    // Dedup: identical specs return the same index.
    unsigned dedup_index;
    TEST_FDG_RESULT(element_dofs_add_option(dofs, 2, specs3, &dedup_index));
    TEST_ASSERTION(dedup_index == 1, "Duplicate option should dedup to index 1.");
    TEST_ASSERTION(element_dofs_option_count(dofs) == 2, "Duplicate option should not be added.");

    double values[16];
    for (unsigned i = 0; i < 16; ++i)
        values[i] = (double)i;
    TEST_FDG_RESULT(element_dofs_add_element(dofs, index2, values));
    TEST_FDG_RESULT(element_dofs_add_element(dofs, index3, values));
    TEST_FDG_RESULT(element_dofs_add_element(dofs, index2, values));

    TEST_ASSERTION(element_dofs_element_count(dofs) == 3, "Should have three elements.");
    TEST_ASSERTION(element_dofs_value_count(dofs) == 34, "Should have 9 + 16 + 9 values.");
    const uint64_t *const offsets = element_dofs_offsets(dofs);
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 9 && offsets[2] == 25 && offsets[3] == 34,
                   "Offsets should reflect the per-option block sizes.");
    const uint32_t *const element_options = element_dofs_element_options(dofs);
    TEST_ASSERTION(element_options[0] == index2 && element_options[1] == index3 && element_options[2] == index2,
                   "Element options should match the added indices.");

    for (unsigned i = 0; i < 9; ++i)
        values[i] = -1.0;
    TEST_FDG_RESULT(element_dofs_set_element_values(dofs, 0, values));
    TEST_NUMBERS_CLOSE(element_dofs_values(dofs)[0], -1.0, 1e-14, 0);

    TEST_ASSERTION(element_dofs_add_element(dofs, 5, values) == FDG_ERROR_NOT_IN_DOMAIN,
                   "Out-of-range option index should be rejected.");
    TEST_ASSERTION(element_dofs_set_element_values(dofs, 3, values) == FDG_ERROR_NOT_IN_DOMAIN,
                   "Out-of-range element id should be rejected.");

    // ndim mismatch is rejected.
    const basis_spec_t specs1[1] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    unsigned bad_index;
    TEST_ASSERTION(element_dofs_add_option(dofs, 1, specs1, &bad_index) == FDG_ERROR_NOT_IN_DOMAIN,
                   "ndim mismatch should be rejected.");

    element_dofs_free(dofs, &TEST_ALLOCATOR);
}

int main(void)
{
    test_dofs_roundtrip();
    return 0;
}
