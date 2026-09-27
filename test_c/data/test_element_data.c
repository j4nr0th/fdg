//
// Created by jan on 2026-09-13.
//
#include "../common/common.h"

#include "../../src/data/element_data.h"

static void test_empty(void)
{
    element_data_t *data;
    TEST_FDG_RESULT(element_data_create(&data, &TEST_ALLOCATOR));

    TEST_ASSERTION(element_data_kind(data) == ELEMENT_DATA_KIND_INVALID, "Empty store should have invalid kind.");
    TEST_ASSERTION(element_data_element_count(data) == 0, "Empty store should have no elements.");
    TEST_ASSERTION(element_data_option_count(data) == 0, "Empty store should have no options.");
    TEST_ASSERTION(element_data_value_count(data) == 0, "Empty store should have no values.");

    element_data_free(data, &TEST_ALLOCATOR);
}

static void test_dof_option_and_elements(void)
{
    element_data_t *data;
    TEST_FDG_RESULT(element_data_create(&data, &TEST_ALLOCATOR));

    const basis_spec_t specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                   {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    element_data_option_t option = {.kind = ELEMENT_DATA_KIND_DOF, .ndim = 2, .basis_specs = (basis_spec_t *)specs};
    unsigned index;
    TEST_FDG_RESULT(element_data_add_option(data, &option, &index));
    TEST_ASSERTION(index == 0, "First option should get index 0.");
    TEST_ASSERTION(element_data_option_value_count(element_data_option(data, 0)) == 9,
                   "Order-2 2D DoF option should store 9 values.");

    // Dedup: same option again returns the same index and adds nothing.
    unsigned dedup_index;
    TEST_FDG_RESULT(element_data_add_option(data, &option, &dedup_index));
    TEST_ASSERTION(dedup_index == 0, "Duplicate option should dedup to index 0.");
    TEST_ASSERTION(element_data_option_count(data) == 1, "Duplicate option should not be added.");

    // Two elements with distinct values.
    double values[9];
    for (unsigned i = 0; i < 9; ++i)
        values[i] = (double)i;
    TEST_FDG_RESULT(element_data_add_element(data, 0, values, 9));
    for (unsigned i = 0; i < 9; ++i)
        values[i] = 100.0 + (double)i;
    TEST_FDG_RESULT(element_data_add_element(data, 0, values, 9));

    TEST_ASSERTION(element_data_element_count(data) == 2, "Should have two elements.");
    TEST_ASSERTION(element_data_value_count(data) == 18, "Should have 18 values.");
    const uint64_t *const offsets = element_data_offsets(data);
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 9 && offsets[2] == 18, "Offsets should be 0, 9, 18.");
    const double *const stored = element_data_values(data);
    for (unsigned i = 0; i < 9; ++i)
    {
        TEST_NUMBERS_CLOSE(stored[i], (double)i, 1e-14, 0);
        TEST_NUMBERS_CLOSE(stored[9 + i], 100.0 + (double)i, 1e-14, 0);
    }

    // Overwrite the first element.
    for (unsigned i = 0; i < 9; ++i)
        values[i] = -1.0;
    element_data_set_element_values(data, 0, values, 9);
    for (unsigned i = 0; i < 9; ++i)
        TEST_NUMBERS_CLOSE(stored[i], -1.0, 1e-14, 0);

    element_data_free(data, &TEST_ALLOCATOR);
}

static void test_kform_option(void)
{
    element_data_t *data;
    TEST_FDG_RESULT(element_data_create(&data, &TEST_ALLOCATOR));

    // 1-forms in 2D over order-2 uniform bases: two components with
    // p * (p + 1) = 6 DoFs each, for 12 values total.
    const basis_spec_t specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                   {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    element_data_option_t option = {
        .kind = ELEMENT_DATA_KIND_KFORM, .ndim = 2, .basis_specs = (basis_spec_t *)specs, .kform = {.order = 1}};
    unsigned index;
    TEST_FDG_RESULT(element_data_add_option(data, &option, &index));
    TEST_ASSERTION(index == 0, "First option should get index 0.");
    TEST_ASSERTION(element_data_option_value_count(element_data_option(data, 0)) == 12,
                   "Order-1 2D k-form on order-2 bases should store 12 values.");

    element_data_free(data, &TEST_ALLOCATOR);
}

static void test_ndim_mismatch(void)
{
    element_data_t *data;
    TEST_FDG_RESULT(element_data_create(&data, &TEST_ALLOCATOR));

    const basis_spec_t specs_2d[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                      {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    const basis_spec_t specs_1d[1] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    element_data_option_t option_2d = {
        .kind = ELEMENT_DATA_KIND_DOF, .ndim = 2, .basis_specs = (basis_spec_t *)specs_2d};
    unsigned index;
    TEST_FDG_RESULT(element_data_add_option(data, &option_2d, &index));

    element_data_option_t option_1d = {
        .kind = ELEMENT_DATA_KIND_DOF, .ndim = 1, .basis_specs = (basis_spec_t *)specs_1d};
    unsigned bad_index;
    TEST_ASSERTION(element_data_add_option(data, &option_1d, &bad_index) == FDG_ERROR_NOT_IN_DOMAIN,
                   "An option with a different ndim than the first option should be rejected.");

    element_data_free(data, &TEST_ALLOCATOR);
}

static void test_mixed_option_counts(void)
{
    element_data_t *data;
    TEST_FDG_RESULT(element_data_create(&data, &TEST_ALLOCATOR));

    const basis_spec_t specs2[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                    {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    element_data_option_t order2 = {.kind = ELEMENT_DATA_KIND_DOF, .ndim = 2, .basis_specs = (basis_spec_t *)specs2};
    unsigned index2;
    TEST_FDG_RESULT(element_data_add_option(data, &order2, &index2));

    const basis_spec_t specs3[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 3},
                                    {.type = BASIS_LAGRANGE_UNIFORM, .order = 3}};
    element_data_option_t order3 = {.kind = ELEMENT_DATA_KIND_DOF, .ndim = 2, .basis_specs = (basis_spec_t *)specs3};
    unsigned index3;
    TEST_FDG_RESULT(element_data_add_option(data, &order3, &index3));
    TEST_ASSERTION(index2 == 0 && index3 == 1, "Distinct options should get consecutive indices.");
    TEST_ASSERTION(element_data_option_count(data) == 2, "Should have two options.");

    double values[16];
    for (unsigned i = 0; i < 16; ++i)
        values[i] = (double)i;
    TEST_FDG_RESULT(element_data_add_element(data, index2, values, 9));
    TEST_FDG_RESULT(element_data_add_element(data, index3, values, 16));
    TEST_ASSERTION(element_data_value_count(data) == 25, "Should have 25 values.");
    const uint64_t *const offsets = element_data_offsets(data);
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 9 && offsets[2] == 25,
                   "Offsets should reflect the per-option block sizes.");

    const uint32_t *const element_options = element_data_element_options(data);
    TEST_ASSERTION(element_options[0] == index2 && element_options[1] == index3,
                   "Element options should match the added indices.");

    element_data_free(data, &TEST_ALLOCATOR);
}

int main(void)
{
    test_empty();
    test_dof_option_and_elements();
    test_kform_option();
    test_ndim_mismatch();
    test_mixed_option_counts();
    return 0;
}
