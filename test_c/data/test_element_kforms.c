//
// Created by jan on 2026-09-13.
//
#include "../common/common.h"

#include "../../src/data/element_kforms.h"
#include "../../src/data/mesh_kform_specs.h"

/** Builds the structure used by the value tests: fields u (order 1) and q
 * (order 0) in 2D, two base spaces, three elements on spaces 0, 1, 0. */
static mesh_kform_specs_t *build_specs(unsigned *u_field, unsigned *q_field)
{
    mesh_kform_specs_t *specs;
    TEST_FDG_RESULT(mesh_kform_specs_create(&specs, &TEST_ALLOCATOR));
    TEST_FDG_RESULT(mesh_kform_specs_add_field(specs, "u", 2, 1, u_field));
    TEST_FDG_RESULT(mesh_kform_specs_add_field(specs, "q", 2, 0, q_field));

    const basis_spec_t space0_specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 2},
                                          {.type = BASIS_LAGRANGE_UNIFORM, .order = 2}};
    const basis_spec_t space1_specs[2] = {{.type = BASIS_LAGRANGE_UNIFORM, .order = 3},
                                          {.type = BASIS_LAGRANGE_UNIFORM, .order = 3}};
    unsigned space0;
    unsigned space1;
    TEST_FDG_RESULT(mesh_kform_specs_add_space(specs, space0_specs, &space0));
    TEST_FDG_RESULT(mesh_kform_specs_add_space(specs, space1_specs, &space1));
    TEST_FDG_RESULT(mesh_kform_specs_add_element(specs, 0));
    TEST_FDG_RESULT(mesh_kform_specs_add_element(specs, 1));
    TEST_FDG_RESULT(mesh_kform_specs_add_element(specs, 0));
    return specs;
}

static void test_kforms_values_and_offsets(void)
{
    unsigned u_field;
    unsigned q_field;
    mesh_kform_specs_t *specs = build_specs(&u_field, &q_field);

    element_kforms_t *kforms;
    TEST_FDG_RESULT(element_kforms_create(&kforms, specs, &TEST_ALLOCATOR));
    TEST_ASSERTION(element_kforms_specs(kforms) == specs, "The collection should expose its borrowed specs.");
    TEST_ASSERTION(element_kforms_filled_count(kforms) == 0, "A new collection should have an empty cursor.");

    // Storage is allocated once and zero-filled; the offsets reflect the
    // per-element base spaces.
    const double *const q_values = element_kforms_field_values(kforms, q_field);
    const uint64_t *const q_offsets = element_kforms_field_offsets(kforms, q_field);
    TEST_ASSERTION(q_offsets[0] == 0 && q_offsets[1] == 9 && q_offsets[2] == 25 && q_offsets[3] == 34,
                   "Field q offsets should reflect the per-element space.");
    for (unsigned i = 0; i < 34; ++i)
        TEST_NUMBERS_CLOSE(q_values[i], 0.0, 1e-14, 0);

    // Three elements, field-major: u values then q values.
    double values[40];
    for (unsigned i = 0; i < 21; ++i)
        values[i] = (double)i;
    element_kforms_add_element(kforms, values);
    TEST_ASSERTION(element_kforms_filled_count(kforms) == 1, "The cursor should advance per element.");
    for (unsigned i = 0; i < 40; ++i)
        values[i] = 100.0 + (double)i;
    element_kforms_add_element(kforms, values);
    for (unsigned i = 0; i < 21; ++i)
        values[i] = 200.0 + (double)i;
    element_kforms_add_element(kforms, values);
    TEST_ASSERTION(element_kforms_filled_count(kforms) == 3, "All three elements should be filled.");

    // Ragged offsets: field u blocks are 12, 24, 12 values.
    const uint64_t *const u_offsets = element_kforms_field_offsets(kforms, u_field);
    TEST_ASSERTION(u_offsets[0] == 0 && u_offsets[1] == 12 && u_offsets[2] == 36 && u_offsets[3] == 48,
                   "Field offsets should reflect the per-element space.");
    TEST_NUMBERS_CLOSE(q_values[0], 12.0, 1e-14, 0);
    TEST_NUMBERS_CLOSE(q_values[9], 124.0, 1e-14, 0);
    TEST_NUMBERS_CLOSE(q_values[25], 212.0, 1e-14, 0);

    // Overwrite one field of one element.
    for (unsigned i = 0; i < 16; ++i)
        values[i] = -1.0;
    element_kforms_set_field_values(kforms, 1, q_field, values);
    TEST_NUMBERS_CLOSE(q_values[9], -1.0, 1e-14, 0);

    // Out-of-range element ids, field indices and over-filling abort
    // through CUTL_ASSERT, so they are not tested here.

    // Freeing the collection leaves the borrowed specs untouched.
    element_kforms_free(kforms, &TEST_ALLOCATOR);
    TEST_ASSERTION(mesh_kform_specs_element_count(specs) == 3, "The specs should outlive the collection.");

    mesh_kform_specs_free(specs, &TEST_ALLOCATOR);
}

static void test_kforms_without_elements(void)
{
    mesh_kform_specs_t *specs;
    TEST_FDG_RESULT(mesh_kform_specs_create(&specs, &TEST_ALLOCATOR));
    TEST_FDG_RESULT(mesh_kform_specs_add_field(specs, "u", 1, 0, NULL));

    // A structure without spaces and elements yields a value collection
    // with empty storage.
    element_kforms_t *kforms;
    TEST_FDG_RESULT(element_kforms_create(&kforms, specs, &TEST_ALLOCATOR));
    TEST_ASSERTION(element_kforms_filled_count(kforms) == 0, "The cursor should start at zero.");
    const uint64_t *const offsets = element_kforms_field_offsets(kforms, 0);
    TEST_ASSERTION(offsets[0] == 0, "Offsets should start at zero for an empty collection.");

    element_kforms_free(kforms, &TEST_ALLOCATOR);
    mesh_kform_specs_free(specs, &TEST_ALLOCATOR);
}

int main(void)
{
    test_kforms_values_and_offsets();
    test_kforms_without_elements();
    return 0;
}
