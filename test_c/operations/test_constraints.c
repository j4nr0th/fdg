#include "../../src/constraints/constraints.h"
#include "../common/common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void *test_allocate(void *ctx, size_t size)
{
    return malloc(size);
}

static void test_free(void *ctx, void *ptr)
{
    free(ptr);
}

static void *test_reallocate(void *ctx, void *ptr, size_t size)
{
    return realloc(ptr, size);
}

static cutl_allocator_t SYSTEM_TEST_ALLOCATOR = {
    .allocate = test_allocate,
    .deallocate = test_free,
    .reallocate = test_reallocate,
};

static basis_spec_t basis_spec(const unsigned order)
{
    return (basis_spec_t){.type = BASIS_LEGENDRE, .order = order};
}

static void test_component_layout(void)
{
    const basis_spec_t basis[] = {basis_spec(2), basis_spec(3)};
    const kform_spec_t spec = {.ndim = 2, .order = 1, .basis = basis};

    TEST_ASSERTION(kform_spec_component_count(&spec) == 2, "Unexpected one-form component count.");

    TEST_ASSERTION(kform_spec_component_dof_count(&spec, 0) == 8, "Unexpected first component DoF count.");
    TEST_ASSERTION(kform_spec_component_dof_count(&spec, 1) == 9, "Unexpected second component DoF count.");

    size_t offsets[3];
    kform_spec_component_offsets(&spec, 3, offsets);
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 8 && offsets[2] == 17, "Unexpected component offsets.");
}

static void test_scalar_component(void)
{
    const basis_spec_t basis[] = {basis_spec(2)};
    const kform_spec_t spec = {.ndim = 1, .order = 0, .basis = basis};

    TEST_ASSERTION(kform_spec_component_count(&spec) == 1, "Unexpected scalar component count.");
    TEST_ASSERTION(kform_spec_component_dof_count(&spec, 0) == 3, "Unexpected scalar DoF count.");
    size_t offsets[2];
    kform_spec_component_offsets(&spec, 2, offsets);
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 3, "Unexpected scalar component offsets.");
}

static void test_zero_order_scalar_constraints(void)
{
    const basis_spec_t zero_basis[] = {basis_spec(0)};
    const kform_spec_t scalar = {.ndim = 1, .order = 0, .basis = zero_basis};
    const kform_spec_t positive_form = {.ndim = 1, .order = 1, .basis = zero_basis};

    TEST_ASSERTION(kform_spec_component_count(&scalar) == 1, "Scalar degree-zero test basis was rejected.");
    TEST_ASSERTION(kform_spec_component_dof_count(&scalar, 0) == 1, "Unexpected scalar degree-zero DoF count.");
    // Order-zero axes are legal for test spaces: the component with the
    // zero-order axis active simply has no DoFs.
    TEST_ASSERTION(kform_spec_component_count(&positive_form) == 1,
                   "Degree-zero basis was rejected for a positive-degree form.");
    TEST_ASSERTION(kform_spec_component_dof_count(&positive_form, 0) == 0,
                   "Active degree-zero axis unexpectedly produced DoFs.");
}

int main(void)
{
    test_component_layout();
    test_scalar_component();
    test_zero_order_scalar_constraints();
}
