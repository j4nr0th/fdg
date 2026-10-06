//
// Created by jan on 2026-09-27.
//
#include "../common/common.h"

#include "../../src/data/mesh_kform_specs.h"

static void test_specs_fields_spaces_and_elements(void)
{
    mesh_kform_specs_t *specs;
    TEST_FDG_RESULT(mesh_kform_specs_create(&specs, &TEST_ALLOCATOR));

    TEST_ASSERTION(mesh_kform_specs_field_count(specs) == 0, "Empty structure should have no fields.");
    TEST_ASSERTION(mesh_kform_specs_element_count(specs) == 0, "Empty structure should have no elements.");
    TEST_ASSERTION(mesh_kform_specs_ndim(specs) == 0, "Empty structure should have no dimension.");
    TEST_ASSERTION(mesh_kform_specs_space_count(specs) == 0, "Empty structure should have no spaces.");
    TEST_ASSERTION(!mesh_kform_specs_is_frozen(specs), "A new structure should not be frozen.");

    unsigned u_field;
    TEST_FDG_RESULT(mesh_kform_specs_add_field(specs, "u", 2, 1, &u_field));
    TEST_ASSERTION(u_field == 0, "First field should get index 0.");
    TEST_ASSERTION(mesh_kform_specs_ndim(specs) == 2, "First field should fix the dimension.");

    unsigned q_field;
    TEST_FDG_RESULT(mesh_kform_specs_add_field(specs, "q", 2, 0, &q_field));
    TEST_ASSERTION(q_field == 1, "Second field should get index 1.");
    TEST_ASSERTION(mesh_kform_specs_field_order(specs, q_field) == 0, "Field order should round-trip.");

    // Duplicate labels, dimension mismatches and orders above ndim abort
    // through CUTL_ASSERT, so they are not tested here. A missing label is
    // a query result, not an error.
    unsigned bad_field;
    TEST_ASSERTION(!mesh_kform_specs_find_field(specs, "missing", &bad_field), "Unknown label should not be found.");

    unsigned found;
    TEST_ASSERTION(mesh_kform_specs_find_field(specs, "q", &found), "An added label should be found.");
    TEST_ASSERTION(found == q_field, "Lookup by label should return the field index.");

    // Two base spaces: order-2 and order-3 uniform bases.
    const basis_spec_t space0_specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                          {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    const basis_spec_t space1_specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 3},
                                          {.type = BASIS_LAGRANGE_UNIFORM, .order = 3}};
    unsigned space0;
    TEST_FDG_RESULT(mesh_kform_specs_add_space(specs, space0_specs, &space0));
    TEST_ASSERTION(space0 == 0, "First space should get index 0.");
    unsigned dedup;
    TEST_FDG_RESULT(mesh_kform_specs_add_space(specs, space0_specs, &dedup));
    TEST_ASSERTION(dedup == 0, "Equal spaces should dedup to the same index.");
    TEST_ASSERTION(mesh_kform_specs_space_count(specs) == 1, "Duplicate space should not be added.");
    unsigned space1;
    TEST_FDG_RESULT(mesh_kform_specs_add_space(specs, space1_specs, &space1));
    TEST_ASSERTION(space1 == 1, "Second space should get index 1.");
    // Field 0 (order 1) on order-2 bases: 2 * p * (p + 1) = 12; on order-3: 24.
    // Field 1 (order 0) on order-2 bases: (p + 1)^2 = 9; on order-3: 16.
    TEST_ASSERTION(mesh_kform_specs_space_value_count(specs, 0, u_field) == 12,
                   "Field u should store 12 values on space 0.");
    TEST_ASSERTION(mesh_kform_specs_space_value_count(specs, 0, q_field) == 9,
                   "Field q should store 9 values on space 0.");
    TEST_ASSERTION(mesh_kform_specs_space_value_count(specs, 1, u_field) == 24,
                   "Field u should store 24 values on space 1.");
    TEST_ASSERTION(mesh_kform_specs_space_value_count(specs, 1, q_field) == 16,
                   "Field q should store 16 values on space 1.");
    TEST_ASSERTION(mesh_kform_specs_element_value_count(specs, 0) == 21, "Space 0 element should store 12 + 9 values.");
    TEST_ASSERTION(mesh_kform_specs_element_value_count(specs, 1) == 40,
                   "Space 1 element should store 24 + 16 values.");

    // The stored space is a deep copy that round-trips the basis specs.
    const basis_spec_t *const stored_specs = mesh_kform_specs_space_basis_specs(specs, 1);
    TEST_ASSERTION(stored_specs[0].order == 3 && stored_specs[1].order == 3,
                   "Space 1 should round-trip its basis specs.");
    TEST_ASSERTION(stored_specs != space1_specs, "The stored space should be a copy, not the input.");

    TEST_FDG_RESULT(mesh_kform_specs_add_element(specs, 0));
    TEST_FDG_RESULT(mesh_kform_specs_add_element(specs, 1));
    TEST_FDG_RESULT(mesh_kform_specs_add_element(specs, 0));

    TEST_ASSERTION(mesh_kform_specs_element_count(specs) == 3, "Should have three elements.");
    TEST_ASSERTION(mesh_kform_specs_element_space(specs, 0) == 0, "Element 0 should reference space 0.");
    TEST_ASSERTION(mesh_kform_specs_element_space(specs, 1) == 1, "Element 1 should reference space 1.");
    TEST_ASSERTION(mesh_kform_specs_element_space(specs, 2) == 0, "Element 2 should reference space 0.");

    // Out-of-range element ids, field indices and space indices abort
    // through CUTL_ASSERT, so they are not tested here. Fields cannot be
    // added once spaces exist, which also aborts.

    // Labels are owned copies.
    TEST_ASSERTION(strcmp(mesh_kform_specs_field_label(specs, u_field), "u") == 0, "Field label should round-trip.");

    mesh_kform_specs_free(specs, &TEST_ALLOCATOR);
}

static void test_specs_freeze(void)
{
    mesh_kform_specs_t *specs;
    TEST_FDG_RESULT(mesh_kform_specs_create(&specs, &TEST_ALLOCATOR));

    const basis_spec_t space_specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                         {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    TEST_FDG_RESULT(mesh_kform_specs_add_field(specs, "u", 2, 1, NULL));
    unsigned space;
    TEST_FDG_RESULT(mesh_kform_specs_add_space(specs, space_specs, &space));
    TEST_FDG_RESULT(mesh_kform_specs_add_element(specs, 0));
    TEST_ASSERTION(!mesh_kform_specs_is_frozen(specs), "The structure should not be frozen before borrowing.");

    mesh_kform_specs_freeze(specs);
    TEST_ASSERTION(mesh_kform_specs_is_frozen(specs), "Freezing should be observable.");
    // Further mutation now aborts through CUTL_ASSERT, so it is not tested
    // here. Freezing is permanent; there is no unfreeze.

    mesh_kform_specs_free(specs, &TEST_ALLOCATOR);
}

static void test_specs_zero_order_space(void)
{
    mesh_kform_specs_t *specs;
    TEST_FDG_RESULT(mesh_kform_specs_create(&specs, &TEST_ALLOCATOR));

    const basis_spec_t zero_axis[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 0},
                                       {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};

    // Spaces before fields and zero-order axes under nonzero-order fields
    // abort through CUTL_ASSERT, so they are not tested here. A zero-order
    // axis is fine for a zero-order-only collection.
    unsigned field;
    TEST_FDG_RESULT(mesh_kform_specs_add_field(specs, "q", 2, 0, &field));
    unsigned space;
    TEST_FDG_RESULT(mesh_kform_specs_add_space(specs, zero_axis, &space));
    TEST_ASSERTION(space == 0, "Zero-order axis should be accepted for a zero-order field.");
    TEST_ASSERTION(mesh_kform_specs_space_value_count(specs, space, field) == 3,
                   "Zero-order field on one order-2 and one order-0 axis should store 3 values.");

    mesh_kform_specs_free(specs, &TEST_ALLOCATOR);
}

int main(void)
{
    test_specs_fields_spaces_and_elements();
    test_specs_freeze();
    test_specs_zero_order_space();
    return 0;
}
