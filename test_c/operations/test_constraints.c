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

/**
 * @brief Test-side trace basis table with owned values.
 *
 * Mirrors the binding-layer table construction: fixed axes read endpoint
 * sets, free axes read the canonical rule nodes with mirrored indices for
 * reversed orientations, and active covector axes read the lowered sets.
 */
typedef struct
{
    kform_values_table_t descriptor;
    size_t offsets[64];
    double *values;
} test_table_t;

static void test_table_free(test_table_t *const table)
{
    free(table->values);
    *table = (test_table_t){};
}

static int test_table_build(basis_set_registry_t *registry, const unsigned element_dim, const unsigned face_dim,
                            const unsigned order, const basis_spec_t *basis_specs, const int8_t *orientation,
                            const integration_spec_t *canonical_specs, const integration_rule_t **canonical_rules,
                            const size_t *canonical_strides, const int element_table, const size_t point_count,
                            test_table_t *const out)
{
    const unsigned ndim = element_table ? element_dim : face_dim;
    const unsigned component_count = combination_total_count((uint8_t)ndim, (uint8_t)order);
    const unsigned free_count = face_dim;
    TEST_ASSERTION(component_count < 64, "Test table component capacity exceeded.");

    kform_trace_axis_t axes[UINT8_MAX];
    const basis_set_t *basis_sets_storage[UINT8_MAX];
    const basis_set_t *basis_sets_lower_storage[UINT8_MAX];
    const basis_endpoint_set_t *endpoint_sets_storage[UINT8_MAX];
    const basis_endpoint_set_t *endpoint_sets_lower_storage[UINT8_MAX];
    const basis_set_t **basis_sets = NULL;
    const basis_set_t **basis_sets_lower = NULL;
    const basis_endpoint_set_t **endpoint_sets = NULL;
    const basis_endpoint_set_t **endpoint_sets_lower = NULL;
    basis_spec_t free_specs[UINT8_MAX];
    basis_spec_t free_specs_lower[UINT8_MAX];
    basis_spec_t lower_specs[UINT8_MAX];
    unsigned source_axes[UINT8_MAX];
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        axes[axis] = (kform_trace_axis_t){};
        source_axes[axis] = face_dim;
        if (!element_table)
        {
            source_axes[axis] = axis;
            free_specs[axis] = basis_specs[axis];
        }
    }
    if (face_dim > 0)
    {
        const unsigned fixed_count = element_dim - face_dim;
        for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
        {
            const int8_t mapping = orientation[fixed_count + face_axis];
            const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
            if (element_table)
            {
                free_specs[face_axis] = basis_specs[element_axis];
                source_axes[element_axis] = face_axis;
            }
        }
    }
    if (order > 0)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            lower_specs[axis] = basis_specs[axis];
            if (lower_specs[axis].order > 0)
                lower_specs[axis].order -= 1;
        }
        for (unsigned axis = 0; axis < free_count; ++axis)
        {
            free_specs_lower[axis] = free_specs[axis];
            if (free_specs_lower[axis].order > 0)
                free_specs_lower[axis].order -= 1;
        }
    }

    if (ndim > 0 && element_table)
    {
        endpoint_sets = endpoint_sets_storage;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            TEST_ASSERTION(basis_set_registry_get_basis_endpoints(registry, &endpoint_sets[axis], basis_specs[axis]) ==
                               FDG_SUCCESS,
                           "Could not fetch endpoint sets.");
        }
        if (order > 0)
        {
            endpoint_sets_lower = endpoint_sets_lower_storage;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                TEST_ASSERTION(basis_set_registry_get_basis_endpoints(registry, &endpoint_sets_lower[axis],
                                                                      lower_specs[axis]) == FDG_SUCCESS,
                               "Could not fetch lowered endpoint sets.");
            }
        }
    }
    if (free_count > 0)
    {
        basis_sets = basis_sets_storage;
        TEST_ASSERTION(basis_set_registry_get_basis_sets(registry, free_count, basis_sets, canonical_rules,
                                                         free_specs) == FDG_SUCCESS,
                       "Could not fetch basis sets.");
        if (order > 0)
        {
            basis_sets_lower = basis_sets_lower_storage;
            TEST_ASSERTION(basis_set_registry_get_basis_sets(registry, free_count, basis_sets_lower, canonical_rules,
                                                             free_specs_lower) == FDG_SUCCESS,
                           "Could not fetch lowered basis sets.");
        }
    }

    if (element_table)
    {
        const unsigned fixed_count = element_dim - face_dim;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            const unsigned slot = source_axes[axis];
            if (slot == face_dim)
            {
                unsigned fixed_axis = 0;
                for (; fixed_axis < fixed_count; ++fixed_axis)
                {
                    if ((unsigned)(orientation[fixed_axis] < 0 ? -orientation[fixed_axis] : orientation[fixed_axis]) -
                            1 ==
                        axis)
                        break;
                }
                TEST_ASSERTION(fixed_axis < fixed_count, "Axis is neither fixed nor free.");
                axes[axis] = (kform_trace_axis_t){
                    .endpoint = endpoint_sets[axis],
                    .endpoint_lower = order > 0 ? endpoint_sets_lower[axis] : NULL,
                    .end = orientation[fixed_axis] < 0 ? 0u : 1u,
                };
            }
            else
            {
                const int8_t mapping = orientation[fixed_count + slot];
                axes[axis] = (kform_trace_axis_t){
                    .nodes = basis_sets[slot],
                    .nodes_lower = order > 0 ? basis_sets_lower[slot] : NULL,
                    .rule_size = canonical_specs[slot].order + 1,
                    .stride_slot = slot,
                    .mirror = mapping < 0,
                };
            }
        }
    }
    else
    {
        for (unsigned axis = 0; axis < face_dim; ++axis)
        {
            axes[axis] = (kform_trace_axis_t){
                .nodes = basis_sets[axis],
                .nodes_lower = order > 0 ? basis_sets_lower[axis] : NULL,
                .rule_size = canonical_specs[axis].order + 1,
                .stride_slot = axis,
                .mirror = 0,
            };
        }
    }

    const kform_spec_t descriptor = {.ndim = ndim, .order = order, .basis = basis_specs};
    const size_t total_dofs = kform_spec_total_dofs(&descriptor);
    *out = (test_table_t){};
    out->values = malloc(sizeof(*out->values) * total_dofs * point_count);
    TEST_ASSERTION(out->values != NULL, "Could not allocate test table values.");
    kform_spec_component_offsets(&descriptor, component_count + 1, out->offsets);
    for (unsigned component = 0; component < component_count; ++component)
    {
        const size_t dof_count = out->offsets[component + 1] - out->offsets[component];
        if (dof_count == 0)
            continue;
        uint8_t component_axes[UINT8_MAX];
        kform_component_axes(&descriptor, component, component_axes);
        kform_component_basis_values(ndim, basis_specs, order, component_axes, axes, canonical_strides, point_count,
                                     out->values + out->offsets[component] * point_count);
    }
    out->descriptor = (kform_values_table_t){.component_count = component_count,
                                             .point_count = point_count,
                                             .component_offsets = out->offsets,
                                             .values = out->values};
    if (basis_sets_lower)
    {
        for (unsigned axis = 0; axis < free_count; ++axis)
            basis_set_registry_release_basis_set(registry, basis_sets_lower[axis]);
    }
    if (basis_sets)
    {
        for (unsigned axis = 0; axis < free_count; ++axis)
            basis_set_registry_release_basis_set(registry, basis_sets[axis]);
    }
    if (endpoint_sets_lower)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
            basis_set_registry_release_basis_endpoints(registry, endpoint_sets_lower[axis]);
    }
    if (endpoint_sets)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
            basis_set_registry_release_basis_endpoints(registry, endpoint_sets[axis]);
    }
    return 0;
}

static void test_physical_scalar_measure(void)
{
    basis_set_registry_t *registry;
    TEST_ASSERTION(basis_set_registry_create(&registry, 1, &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the basis registry.");
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1)};
    const int8_t upper[] = {1, 2};
    integration_rule_t *quadrature;
    TEST_ASSERTION(integration_rule_for_order(&quadrature, INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 1,
                                              &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the quadrature rule.");
    const integration_rule_t *rules[1] = {quadrature};
    const integration_spec_t rule_specs[1] = {quadrature->spec};
    size_t strides[1];
    integration_spec_point_strides(1, rule_specs, strides);
    const size_t point_count = integration_specs_total_points(1, rule_specs);
    double point_weights[2];
    integration_rule_tensor_weights(1, rules, point_weights);

    const kform_spec_t test_spec = {.ndim = 1, .order = 0, .basis = test_basis};
    const constraint_element_side_t side = {.ndim = 2, .basis_specs = element_basis, .orientation = upper};

    test_table_t test_table;
    test_table_build(registry, 2, 1, 0, test_basis, upper, rule_specs, rules, strides, 0, point_count, &test_table);
    test_table_t element_table;
    test_table_build(registry, 2, 1, 0, element_basis, upper, rule_specs, rules, strides, 1, point_count,
                     &element_table);
    double point_factors[2];
    uint32_t components_unit[8];
    size_t dofs_unit[8];
    double coefficients_unit[8];
    size_t offsets_unit[3];
    uint32_t components_triple[8];
    size_t dofs_triple[8];
    double coefficients_triple[8];
    size_t offsets_triple[3];
    for (unsigned measure = 0; measure < 2; ++measure)
    {
        point_factors[0] = measure == 0 ? 1.0 : 3.0;
        point_factors[1] = point_factors[0];
        const constraint_assembly_inputs_t inputs = {.point_weights = point_weights,
                                                     .surface_weights = point_factors,
                                                     .test_table = &test_table.descriptor,
                                                     .element_table = &element_table.descriptor,
                                                     .pullback = NULL};
        if (measure == 0)
        {
            constraint_physical_side_assemble(&test_spec, &side, &inputs, components_unit, dofs_unit, coefficients_unit,
                                              offsets_unit);
        }
        else
        {
            constraint_physical_side_assemble(&test_spec, &side, &inputs, components_triple, dofs_triple,
                                              coefficients_triple, offsets_triple);
        }
    }
    TEST_ASSERTION(offsets_unit[0] == 0 && offsets_unit[1] == 4 && offsets_unit[2] == 8,
                   "Unexpected weighted scalar row offsets.");
    TEST_ASSERTION(offsets_triple[0] == 0 && offsets_triple[1] == 4 && offsets_triple[2] == 8,
                   "Unexpected triple-measure row offsets.");
    for (size_t i = 0; i < 8; ++i)
    {
        TEST_ASSERTION(components_unit[i] == components_triple[i] && dofs_unit[i] == dofs_triple[i],
                       "Measure scaling changed the packed metadata at entry %zu.", i);
        TEST_NUMBERS_CLOSE(coefficients_triple[i], 3.0 * coefficients_unit[i], 1e-12, 0);
    }

    test_table_free(&test_table);
    test_table_free(&element_table);
    cutl_dealloc(&SYSTEM_TEST_ALLOCATOR, quadrature);
    basis_set_registry_destroy(registry);
}

static void test_physical_general_boundary_dimensions(void)
{
    basis_set_registry_t *registry;
    TEST_ASSERTION(basis_set_registry_create(&registry, 1, &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the basis registry.");
    const basis_spec_t point_basis[] = {basis_spec(1), basis_spec(1), basis_spec(1)};
    const int8_t point_orientation[] = {-1, 2, 3};
    const kform_spec_t point_test = {.ndim = 0, .order = 0, .basis = NULL};
    const constraint_element_side_t point_side = {
        .ndim = 3, .basis_specs = point_basis, .orientation = point_orientation};
    const double point_weight[1] = {1.0};

    test_table_t point_element_table;
    test_table_build(registry, 3, 0, 0, point_basis, point_orientation, NULL, NULL, NULL, 1, 1, &point_element_table);
    const constraint_assembly_inputs_t point_inputs = {.point_weights = point_weight,
                                                       .surface_weights = NULL,
                                                       .test_table = &point_element_table.descriptor,
                                                       .element_table = &point_element_table.descriptor,
                                                       .pullback = NULL};
    uint32_t out_components[8];
    size_t out_local_dofs[8];
    double out_coefficients[8];
    size_t out_row_offsets[2];
    constraint_physical_side_assemble(&point_test, &point_side, &point_inputs, out_components, out_local_dofs,
                                      out_coefficients, out_row_offsets);
    TEST_ASSERTION(out_row_offsets[0] == 0 && out_row_offsets[1] == 8, "Unexpected point-boundary dimensions.");

    // A three-dimensional line boundary with a permuted, reversed orientation.
    const basis_spec_t line_basis[] = {basis_spec(1), basis_spec(1), basis_spec(1)};
    const int8_t line_orientation[] = {-1, 3, -2};
    const basis_spec_t line_test_basis[] = {basis_spec(1)};
    const kform_spec_t line_test = {.ndim = 1, .order = 0, .basis = line_test_basis};
    const constraint_element_side_t line_side = {.ndim = 3, .basis_specs = line_basis, .orientation = line_orientation};
    integration_rule_t *quad_rule;
    TEST_ASSERTION(integration_rule_for_order(&quad_rule, INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 1,
                                              &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the quadrature rule.");
    const integration_rule_t *line_rules[1] = {quad_rule};
    const integration_spec_t line_rule_specs[1] = {quad_rule->spec};
    size_t line_strides[1];
    integration_spec_point_strides(1, line_rule_specs, line_strides);
    const size_t line_point_count = integration_specs_total_points(1, line_rule_specs);
    double line_point_weights[2];
    integration_rule_tensor_weights(1, line_rules, line_point_weights);
    test_table_t line_test_table;
    test_table_build(registry, 3, 1, 0, line_test_basis, line_orientation, line_rule_specs, line_rules, line_strides, 0,
                     line_point_count, &line_test_table);
    test_table_t line_element_table;
    test_table_build(registry, 3, 1, 0, line_basis, line_orientation, line_rule_specs, line_rules, line_strides, 1,
                     line_point_count, &line_element_table);
    const double line_surface[] = {1.0, 1.0};
    const constraint_assembly_inputs_t line_inputs = {.point_weights = line_point_weights,
                                                      .surface_weights = line_surface,
                                                      .test_table = &line_test_table.descriptor,
                                                      .element_table = &line_element_table.descriptor,
                                                      .pullback = NULL};
    size_t row_count;
    size_t entry_count;
    constraint_physical_side_layout(&line_test, &line_side, &row_count, &entry_count);
    uint32_t *const line_components = malloc(entry_count * sizeof(*line_components));
    size_t *const line_local_dofs = malloc(entry_count * sizeof(*line_local_dofs));
    double *const line_coefficients = malloc(entry_count * sizeof(*line_coefficients));
    size_t *const line_offsets = malloc((row_count + 1) * sizeof(*line_offsets));
    TEST_ASSERTION(line_components && line_local_dofs && line_coefficients && line_offsets,
                   "Could not allocate line-boundary test storage.");
    constraint_physical_side_assemble(&line_test, &line_side, &line_inputs, line_components, line_local_dofs,
                                      line_coefficients, line_offsets);
    free(line_components);
    free(line_local_dofs);
    free(line_coefficients);
    free(line_offsets);

    // A four-dimensional face boundary, scalar and one-form.
    const basis_spec_t face_basis[] = {basis_spec(1), basis_spec(1), basis_spec(1), basis_spec(1)};
    const int8_t face_orientation[] = {-1, 3, -2, 4};
    const basis_spec_t face_test_basis[] = {basis_spec(1), basis_spec(1)};
    const kform_spec_t face_test = {.ndim = 2, .order = 0, .basis = face_test_basis};
    const constraint_element_side_t face_side = {.ndim = 4, .basis_specs = face_basis, .orientation = face_orientation};
    const integration_rule_t *face_rules[2] = {quad_rule, quad_rule};
    const integration_spec_t face_rule_specs[2] = {quad_rule->spec, quad_rule->spec};
    size_t face_strides[2];
    integration_spec_point_strides(2, face_rule_specs, face_strides);
    const size_t face_point_count = integration_specs_total_points(2, face_rule_specs);
    double face_point_weights[4];
    integration_rule_tensor_weights(2, face_rules, face_point_weights);
    test_table_t face_test_table;
    test_table_build(registry, 4, 2, 0, face_test_basis, face_orientation, face_rule_specs, face_rules, face_strides, 0,
                     face_point_count, &face_test_table);
    test_table_t face_element_table;
    test_table_build(registry, 4, 2, 0, face_basis, face_orientation, face_rule_specs, face_rules, face_strides, 1,
                     face_point_count, &face_element_table);
    const double face_surface[] = {1.0, 1.0, 1.0, 1.0};
    const constraint_assembly_inputs_t face_inputs = {.point_weights = face_point_weights,
                                                      .surface_weights = face_surface,
                                                      .test_table = &face_test_table.descriptor,
                                                      .element_table = &face_element_table.descriptor,
                                                      .pullback = NULL};
    constraint_physical_side_layout(&face_test, &face_side, &row_count, &entry_count);
    TEST_ASSERTION(row_count == 4 && entry_count == 64, "Unexpected four-dimensional face dimensions.");

    const basis_spec_t face_one_form_basis[] = {basis_spec(1), basis_spec(1)};
    const kform_spec_t face_one_form_test = {.ndim = 2, .order = 1, .basis = face_one_form_basis};
    double face_pullback_values[4 * 1 * 4];
    for (unsigned i = 0; i < sizeof(face_pullback_values) / sizeof(*face_pullback_values); ++i)
        face_pullback_values[i] = 1.0;
    const constraint_trace_pullback_t face_pullback = {
        .physical_component_count = 1, .point_count = 4, .values = face_pullback_values};
    test_table_t face_one_form_test_table;
    test_table_build(registry, 4, 2, 1, face_one_form_basis, face_orientation, face_rule_specs, face_rules,
                     face_strides, 0, face_point_count, &face_one_form_test_table);
    test_table_t face_one_form_element_table;
    test_table_build(registry, 4, 2, 1, face_basis, face_orientation, face_rule_specs, face_rules, face_strides, 1,
                     face_point_count, &face_one_form_element_table);
    const constraint_assembly_inputs_t face_one_form_inputs = {.point_weights = face_point_weights,
                                                               .surface_weights = face_surface,
                                                               .test_table = &face_one_form_test_table.descriptor,
                                                               .element_table = &face_one_form_element_table.descriptor,
                                                               .pullback = &face_pullback};
    constraint_physical_side_layout(&face_one_form_test, &face_side, &row_count, &entry_count);
    TEST_ASSERTION(row_count == 4 && entry_count == 64, "Unexpected four-dimensional one-form dimensions.");
    uint32_t face_one_form_components[64];
    size_t face_one_form_local_dofs[64];
    double face_one_form_coefficients[64];
    size_t face_one_form_offsets[5];
    constraint_physical_side_assemble(&face_one_form_test, &face_side, &face_one_form_inputs, face_one_form_components,
                                      face_one_form_local_dofs, face_one_form_coefficients, face_one_form_offsets);
    TEST_ASSERTION(face_one_form_components[0] == 1 && face_one_form_components[8] == 3,
                   "Unexpected odd-orientation one-form component mapping.");

    test_table_free(&point_element_table);
    test_table_free(&line_test_table);
    test_table_free(&line_element_table);
    test_table_free(&face_test_table);
    test_table_free(&face_element_table);
    test_table_free(&face_one_form_test_table);
    test_table_free(&face_one_form_element_table);
    cutl_dealloc(&SYSTEM_TEST_ALLOCATOR, quad_rule);
    basis_set_registry_destroy(registry);
}

static void test_physical_one_form_pullback(void)
{
    basis_set_registry_t *registry;
    TEST_ASSERTION(basis_set_registry_create(&registry, 1, &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the basis registry.");
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1)};
    const int8_t lower[] = {-1, 2};
    const int8_t upper[] = {1, 2};
    const double surface_weights[] = {1.0, 1.0};
    const double identity_pullback[] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    };
    integration_rule_t *quad_rule;
    TEST_ASSERTION(integration_rule_for_order(&quad_rule, INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 1,
                                              &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the quadrature rule.");
    const integration_rule_t *rules[1] = {quad_rule};
    const integration_spec_t rule_specs[1] = {quad_rule->spec};
    size_t strides[1];
    integration_spec_point_strides(1, rule_specs, strides);
    const size_t point_count = integration_specs_total_points(1, rule_specs);
    double point_weights[2];
    integration_rule_tensor_weights(1, rules, point_weights);

    const kform_spec_t test_spec = {.ndim = 1, .order = 1, .basis = test_basis};
    const constraint_element_side_t sides[] = {
        {.ndim = 2, .basis_specs = element_basis, .orientation = lower},
        {.ndim = 2, .basis_specs = element_basis, .orientation = upper},
    };
    const constraint_trace_pullback_t pullbacks[2] = {
        {.physical_component_count = 2, .point_count = 2, .values = identity_pullback},
        {.physical_component_count = 2, .point_count = 2, .values = identity_pullback},
    };

    test_table_t test_table;
    test_table_build(registry, 2, 1, 1, test_basis, lower, rule_specs, rules, strides, 0, point_count, &test_table);
    test_table_t element_tables_raw[2];
    constraint_assembly_inputs_t inputs[2];
    for (unsigned side = 0; side < 2; ++side)
    {
        test_table_build(registry, 2, 1, 1, element_basis, sides[side].orientation, rule_specs, rules, strides, 1,
                         point_count, &element_tables_raw[side]);
        inputs[side] = (constraint_assembly_inputs_t){.point_weights = point_weights,
                                                      .surface_weights = surface_weights,
                                                      .test_table = &test_table.descriptor,
                                                      .element_table = &element_tables_raw[side].descriptor,
                                                      .pullback = &pullbacks[side]};
    }

    uint32_t out_components[4];
    size_t out_local_dofs[4];
    double out_coefficients[4];
    size_t out_row_offsets[2];
    constraint_physical_side_assemble(&test_spec, &sides[0], &inputs[0], out_components, out_local_dofs,
                                      out_coefficients, out_row_offsets);
    uint32_t second_components[2];
    size_t second_local_dofs[2];
    double second_coefficients[2];
    size_t second_row_offsets[2];
    constraint_physical_side_assemble(&test_spec, &sides[1], &inputs[1], second_components, second_local_dofs,
                                      second_coefficients, second_row_offsets);
    // The two-sided contract: side 0's row entries then side 1's, side 1
    // carrying the negative side sign.
    for (size_t entry = 0; entry < 2; ++entry)
    {
        out_components[2 + entry] = second_components[entry];
        out_local_dofs[2 + entry] = second_local_dofs[entry];
        out_coefficients[2 + entry] = -second_coefficients[entry];
    }
    TEST_ASSERTION(out_row_offsets[0] == 0 && out_row_offsets[1] == 2 && second_row_offsets[1] == 2,
                   "Unexpected one-form pullback row offsets.");
    TEST_ASSERTION(out_components[0] == 1 && out_components[1] == 1 && out_components[2] == 1 && out_components[3] == 1,
                   "Normal one-form component was included in the physical trace.");
    TEST_NUMBERS_CLOSE(out_coefficients[0], 2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(out_coefficients[1], -2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(out_coefficients[2], -2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(out_coefficients[3], -2.0, 1e-12, 0);

    test_table_free(&test_table);
    test_table_free(&element_tables_raw[0]);
    test_table_free(&element_tables_raw[1]);
    cutl_dealloc(&SYSTEM_TEST_ALLOCATOR, quad_rule);
    basis_set_registry_destroy(registry);
}

static void test_physical_two_form_face_components(void)
{
    basis_set_registry_t *registry;
    TEST_ASSERTION(basis_set_registry_create(&registry, 1, &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the basis registry.");
    const basis_spec_t test_basis[] = {basis_spec(1), basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1), basis_spec(1)};
    const int8_t orientation[] = {-1, 3, 2};
    const double surface_weights[] = {1.0, 1.0, 1.0, 1.0};
    integration_rule_t *quad_rule;
    TEST_ASSERTION(integration_rule_for_order(&quad_rule, INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 1,
                                              &SYSTEM_TEST_ALLOCATOR) == FDG_SUCCESS,
                   "Could not create the quadrature rule.");
    const integration_rule_t *rules[2] = {quad_rule, quad_rule};
    const integration_spec_t rule_specs[2] = {quad_rule->spec, quad_rule->spec};
    size_t strides[2];
    integration_spec_point_strides(2, rule_specs, strides);
    const size_t point_count = integration_specs_total_points(2, rule_specs);
    double point_weights[4];
    integration_rule_tensor_weights(2, rules, point_weights);

    double pullback_values[3 * 3 * 4] = {0};
    for (unsigned component = 0; component < 3; ++component)
        for (unsigned point = 0; point < 4; ++point)
            pullback_values[(component * 3 + component) * 4 + point] = 1.0;
    const constraint_trace_pullback_t pullback = {
        .physical_component_count = 3, .point_count = 4, .values = pullback_values};
    const kform_spec_t test_spec = {.ndim = 2, .order = 2, .basis = test_basis};
    const constraint_element_side_t side = {.ndim = 3, .basis_specs = element_basis, .orientation = orientation};

    test_table_t test_table;
    test_table_build(registry, 3, 2, 2, test_basis, orientation, rule_specs, rules, strides, 0, point_count,
                     &test_table);
    test_table_t element_table;
    test_table_build(registry, 3, 2, 2, element_basis, orientation, rule_specs, rules, strides, 1, point_count,
                     &element_table);
    const constraint_assembly_inputs_t inputs = {.point_weights = point_weights,
                                                 .surface_weights = surface_weights,
                                                 .test_table = &test_table.descriptor,
                                                 .element_table = &element_table.descriptor,
                                                 .pullback = &pullback};

    size_t row_count;
    size_t entry_count;
    constraint_physical_side_layout(&test_spec, &side, &row_count, &entry_count);
    TEST_ASSERTION(row_count == 1 && entry_count == 2, "Unexpected two-form face dimensions.");

    uint32_t out_components[2];
    size_t out_local_dofs[2];
    double out_coefficients[2];
    size_t out_row_offsets[2];
    constraint_physical_side_assemble(&test_spec, &side, &inputs, out_components, out_local_dofs, out_coefficients,
                                      out_row_offsets);
    TEST_ASSERTION(out_row_offsets[0] == 0 && out_row_offsets[1] == 2,
                   "Two-form face assembly did not match its required storage.");
    TEST_ASSERTION(out_components[0] == 2 && out_components[1] == 2, "Unexpected two-form face component mapping.");

    test_table_free(&test_table);
    test_table_free(&element_table);
    cutl_dealloc(&SYSTEM_TEST_ALLOCATOR, quad_rule);
    basis_set_registry_destroy(registry);
}

static void test_boundary_test_specs(void)
{
    // Two quadrilateral faces inside hexahedral elements: the face spans
    // element axes 1 and 2 through the orientation records, so per-axis
    // minima must follow the mapped axes rather than the canonical order.
    const basis_spec_t element_0[] = {basis_spec(3), basis_spec(2), basis_spec(2)};
    const basis_spec_t element_1[] = {basis_spec(2), basis_spec(3), basis_spec(3)};
    const basis_spec_t *const element_bases[] = {element_0, element_1};
    const int8_t orientations[] = {1, 2, 3, -2, 3, 1};

    // Face axis 0 minimum: element 0 axis 1 (order 2, Legendre); face axis 1
    // minimum: a tie at order 2 that keeps element 0's family.
    basis_spec_t specs[4];
    bool present[4];
    constraint_boundary_test_specs(3, 2, 0, 2, element_bases, orientations, BASIS_INVALID, specs, present);
    TEST_ASSERTION(present[0] && specs[0].order == 0 && specs[1].order == 0 && specs[0].type == BASIS_LEGENDRE &&
                       specs[1].type == BASIS_LEGENDRE,
                   "Unexpected scalar boundary test specs.");

    constraint_boundary_test_specs(3, 2, 1, 2, element_bases, orientations, BASIS_INVALID, specs, present);
    TEST_ASSERTION(present[0] && specs[0].order == 2 && specs[1].order == 0 && specs[0].type == BASIS_LEGENDRE &&
                       specs[1].type == BASIS_LEGENDRE,
                   "Inactive axes must reduce the order by two.");
    TEST_ASSERTION(present[1] && specs[2].order == 0 && specs[3].order == 2,
                   "Unexpected second one-form boundary component.");

    constraint_boundary_test_specs(3, 2, 2, 2, element_bases, orientations, BASIS_LAGRANGE_GAUSS, specs, present);
    TEST_ASSERTION(present[0] && specs[0].order == 2 && specs[1].order == 2 && specs[0].type == BASIS_LAGRANGE_GAUSS &&
                       specs[1].type == BASIS_LAGRANGE_GAUSS,
                   "The basis family override was not applied.");

    // An order-one face on order-(1, 1) elements leaves no reduced room on
    // the inactive axis: the component must be reported absent.
    const basis_spec_t low_element[] = {basis_spec(2), basis_spec(1), basis_spec(1)};
    const basis_spec_t *const low_bases[] = {low_element};
    const int8_t low_orientation[] = {1, 2, 3};
    constraint_boundary_test_specs(3, 2, 1, 1, low_bases, low_orientation, BASIS_INVALID, specs, present);
    TEST_ASSERTION(!present[0] && specs[0].order == 1 && specs[1].order == 0,
                   "Absent components must clamp negative orders to zero.");
    TEST_ASSERTION(!present[1], "Both one-form components cannot survive order-one elements.");
}

static unsigned test_matrix_rank(double *values, const size_t rows, const size_t cols)
{
    size_t rank = 0;
    for (size_t col = 0; col < cols && rank < rows; ++col)
    {
        size_t pivot = rank;
        for (size_t row = rank + 1; row < rows; ++row)
        {
            if (fabs(values[row * cols + col]) > fabs(values[pivot * cols + col]))
                pivot = row;
        }
        if (fabs(values[pivot * cols + col]) < 1e-10)
            continue;
        if (pivot != rank)
        {
            for (size_t swap_col = 0; swap_col < cols; ++swap_col)
            {
                const double tmp = values[rank * cols + swap_col];
                values[rank * cols + swap_col] = values[pivot * cols + swap_col];
                values[pivot * cols + swap_col] = tmp;
            }
        }
        for (size_t row = rank + 1; row < rows; ++row)
        {
            const double factor = values[row * cols + col] / values[rank * cols + col];
            for (size_t eliminate = col; eliminate < cols; ++eliminate)
                values[row * cols + eliminate] -= factor * values[rank * cols + eliminate];
        }
        ++rank;
    }
    return (unsigned)rank;
}

/**
 * @brief Shared equivalence proof for one link-eligible scenario.
 *
 * Assembles the dense moment rows and the single-DoF link rows for the same
 * pair, then verifies both constrain the element DoFs identically: the link
 * rows have full row rank, the dense rows have the same rank, and the dense
 * rows vanish on every random assignment satisfying the links.
 */
int main(void)
{
    test_component_layout();
    test_scalar_component();
    test_zero_order_scalar_constraints();
    test_physical_scalar_measure();
    test_physical_general_boundary_dimensions();
    test_physical_one_form_pullback();
    test_boundary_test_specs();
    test_physical_two_form_face_components();
    test_boundary_test_specs();
}
