//
// Created by jan on 2026-09-13.
//
#include "../common/common.h"

#include "../../src/data/element_kforms.h"

static void test_kforms_fields_spaces_and_elements(void)
{
    element_kforms_t *kforms;
    TEST_FDG_RESULT(element_kforms_create(&kforms, &TEST_ALLOCATOR));

    TEST_ASSERTION(element_kforms_field_count(kforms) == 0, "Empty collection should have no fields.");
    TEST_ASSERTION(element_kforms_element_count(kforms) == 0, "Empty collection should have no elements.");
    TEST_ASSERTION(element_kforms_ndim(kforms) == 0, "Empty collection should have no dimension.");
    TEST_ASSERTION(element_kforms_space_count(kforms) == 0, "Empty collection should have no spaces.");

    unsigned u_field;
    TEST_FDG_RESULT(element_kforms_add_field(kforms, "u", 2, 1, &u_field));
    TEST_ASSERTION(u_field == 0, "First field should get index 0.");
    TEST_ASSERTION(element_kforms_ndim(kforms) == 2, "First field should fix the dimension.");

    unsigned q_field;
    TEST_FDG_RESULT(element_kforms_add_field(kforms, "q", 2, 0, &q_field));
    TEST_ASSERTION(q_field == 1, "Second field should get index 1.");
    TEST_ASSERTION(element_kforms_field_order(kforms, q_field) == 0, "Field order should round-trip.");

    // Duplicate labels, dimension mismatches and orders above ndim abort
    // through CUTL_ASSERT, so they are not tested here. A missing label is
    // a query result, not an error.
    unsigned bad_field;
    TEST_ASSERTION(!element_kforms_find_field(kforms, "missing", &bad_field), "Unknown label should not be found.");

    unsigned found;
    TEST_ASSERTION(element_kforms_find_field(kforms, "q", &found), "An added label should be found.");
    TEST_ASSERTION(found == q_field, "Lookup by label should return the field index.");

    // Two base spaces: order-2 and order-3 uniform bases.
    const basis_spec_t space0_specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                          {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    const basis_spec_t space1_specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 3},
                                          {.type = BASIS_LAGRANGE_UNIFORM, .order = 3}};
    unsigned space0;
    TEST_FDG_RESULT(element_kforms_add_space(kforms, space0_specs, &space0));
    TEST_ASSERTION(space0 == 0, "First space should get index 0.");
    TEST_ASSERTION(element_kforms_space_count(kforms) == 1, "Equal spaces should dedup.");
    TEST_FDG_RESULT(element_kforms_add_space(kforms, space0_specs, &space0));
    unsigned space1;
    TEST_FDG_RESULT(element_kforms_add_space(kforms, space1_specs, &space1));
    TEST_ASSERTION(space1 == 1, "Second space should get index 1.");
    // Field 0 (order 1) on order-2 bases: 2 * p * (p + 1) = 12; on order-3: 24.
    // Field 1 (order 0) on order-2 bases: (p + 1)^2 = 9; on order-3: 16.
    TEST_ASSERTION(element_kforms_element_value_count(kforms, 0) == 21, "Space 0 element should store 12 + 9 values.");
    TEST_ASSERTION(element_kforms_element_value_count(kforms, 1) == 40, "Space 1 element should store 24 + 16 values.");

    const element_data_option_t *const option = element_kforms_space_option(kforms, 1);
    TEST_ASSERTION(option->kind == ELEMENT_DATA_KIND_KFORM && option->ndim == 2 && option->kform.order == 1,
                   "Space option should carry the base space and field 0's order.");
    TEST_ASSERTION(option->basis_specs[0].order == 3 && option->basis_specs[1].order == 3,
                   "Space option should carry the order-3 base space.");

    // Three elements on alternating spaces, field-major: u values then q values.
    double values[40];
    for (unsigned i = 0; i < 21; ++i)
        values[i] = (double)i;
    TEST_FDG_RESULT(element_kforms_add_element(kforms, 0, values));
    for (unsigned i = 0; i < 40; ++i)
        values[i] = 100.0 + (double)i;
    TEST_FDG_RESULT(element_kforms_add_element(kforms, 1, values));
    for (unsigned i = 0; i < 21; ++i)
        values[i] = 200.0 + (double)i;
    TEST_FDG_RESULT(element_kforms_add_element(kforms, 0, values));

    TEST_ASSERTION(element_kforms_element_count(kforms) == 3, "Should have three elements.");
    TEST_ASSERTION(element_kforms_element_space(kforms, 0) == 0, "Element 0 should reference space 0.");
    TEST_ASSERTION(element_kforms_element_space(kforms, 1) == 1, "Element 1 should reference space 1.");
    TEST_ASSERTION(element_kforms_element_space(kforms, 2) == 0, "Element 2 should reference space 0.");

    // Ragged offsets: field 0 blocks are 12, 24, 12 values.
    const uint64_t *const u_offsets = element_kforms_field_offsets(kforms, u_field);
    TEST_ASSERTION(u_offsets[0] == 0 && u_offsets[1] == 12 && u_offsets[2] == 36 && u_offsets[3] == 48,
                   "Field offsets should reflect the per-element space.");
    const double *const q_values = element_kforms_field_values(kforms, q_field);
    const uint64_t *const q_offsets = element_kforms_field_offsets(kforms, q_field);
    TEST_ASSERTION(q_offsets[0] == 0 && q_offsets[1] == 9 && q_offsets[2] == 25 && q_offsets[3] == 34,
                   "Field q offsets should reflect the per-element space.");
    TEST_NUMBERS_CLOSE(q_values[0], 12.0, 1e-14, 0);
    TEST_NUMBERS_CLOSE(q_values[9], 124.0, 1e-14, 0);
    TEST_NUMBERS_CLOSE(q_values[25], 212.0, 1e-14, 0);

    // Overwrite one field of one element.
    for (unsigned i = 0; i < 16; ++i)
        values[i] = -1.0;
    element_kforms_set_field_values(kforms, 1, q_field, values);
    TEST_NUMBERS_CLOSE(q_values[9], -1.0, 1e-14, 0);

    // Out-of-range element ids, field indices and space indices abort
    // through CUTL_ASSERT, so they are not tested here. Fields cannot be
    // added once spaces exist, which also aborts.

    // Labels are owned copies.
    TEST_ASSERTION(strcmp(element_kforms_field_label(kforms, u_field), "u") == 0, "Field label should round-trip.");

    element_kforms_free(kforms, &TEST_ALLOCATOR);
}

static void test_kforms_zero_order_space(void)
{
    element_kforms_t *kforms;
    TEST_FDG_RESULT(element_kforms_create(&kforms, &TEST_ALLOCATOR));

    const basis_spec_t zero_axis[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 0},
                                       {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};

    // Spaces before fields and zero-order axes under nonzero-order fields
    // abort through CUTL_ASSERT, so they are not tested here. A zero-order
    // axis is fine for a zero-order-only collection.
    unsigned field;
    TEST_FDG_RESULT(element_kforms_add_field(kforms, "q", 2, 0, &field));
    unsigned space;
    TEST_FDG_RESULT(element_kforms_add_space(kforms, zero_axis, &space));
    TEST_ASSERTION(space == 0, "Zero-order axis should be accepted for a zero-order field.");

    element_kforms_free(kforms, &TEST_ALLOCATOR);
}

int main(void)
{
    test_kforms_fields_spaces_and_elements();
    test_kforms_zero_order_space();
    return 0;
}
