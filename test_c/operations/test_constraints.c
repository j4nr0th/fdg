#include "../../src/constraints/constraints.h"
#include "../common/common.h"

static basis_spec_t basis_spec(const unsigned order)
{
    return (basis_spec_t){.type = BASIS_LEGENDRE, .order = order};
}

static void test_component_layout(void)
{
    const basis_spec_t basis[] = {basis_spec(2), basis_spec(3)};
    const constraint_kform_spec_t spec = {.ndim = 2, .order = 1, .basis_specs = basis};
    size_t component_count;
    size_t offsets[3];
    size_t dof_count;

    TEST_ASSERTION(constraint_kform_component_count(&spec, &component_count) == CONSTRAINT_SUCCESS,
                   "Could not count k-form components.");
    TEST_ASSERTION(component_count == 2, "Unexpected one-form component count.");

    TEST_ASSERTION(constraint_kform_component_dof_count(&spec, 0, &dof_count) == CONSTRAINT_SUCCESS,
                   "Could not count first component DoFs.");
    TEST_ASSERTION(dof_count == 8, "Unexpected first component DoF count.");
    TEST_ASSERTION(constraint_kform_component_dof_count(&spec, 1, &dof_count) == CONSTRAINT_SUCCESS,
                   "Could not count second component DoFs.");
    TEST_ASSERTION(dof_count == 9, "Unexpected second component DoF count.");

    TEST_ASSERTION(constraint_kform_component_offsets(&spec, 3, offsets) == CONSTRAINT_SUCCESS,
                   "Could not compute component offsets.");
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 8 && offsets[2] == 17, "Unexpected component offsets.");
}

static void test_scalar_component(void)
{
    const basis_spec_t basis[] = {basis_spec(2)};
    const constraint_kform_spec_t spec = {.ndim = 1, .order = 0, .basis_specs = basis};
    size_t component_count;
    size_t dof_count;
    size_t offsets[2];

    TEST_ASSERTION(constraint_kform_component_count(&spec, &component_count) == CONSTRAINT_SUCCESS,
                   "Could not count scalar components.");
    TEST_ASSERTION(component_count == 1, "Unexpected scalar component count.");
    TEST_ASSERTION(constraint_kform_component_dof_count(&spec, 0, &dof_count) == CONSTRAINT_SUCCESS,
                   "Could not count scalar DoFs.");
    TEST_ASSERTION(dof_count == 3, "Unexpected scalar DoF count.");
    TEST_ASSERTION(constraint_kform_component_offsets(&spec, 2, offsets) == CONSTRAINT_SUCCESS,
                   "Could not compute scalar component offsets.");
    TEST_ASSERTION(offsets[0] == 0 && offsets[1] == 3, "Unexpected scalar component offsets.");
}

static void test_invalid_specs(void)
{
    const basis_spec_t basis[] = {basis_spec(1), basis_spec(1)};
    const constraint_kform_spec_t invalid_order = {.ndim = 2, .order = 3, .basis_specs = basis};
    const constraint_kform_spec_t missing_basis = {.ndim = 1, .order = 0, .basis_specs = NULL};
    size_t result;

    TEST_ASSERTION(constraint_kform_component_count(&invalid_order, &result) == CONSTRAINT_INVALID_ORDER,
                   "Invalid form order was accepted.");
    TEST_ASSERTION(constraint_kform_component_count(&missing_basis, &result) == CONSTRAINT_INVALID_ARGUMENT,
                   "Missing basis specifications were accepted.");
    TEST_ASSERTION(constraint_kform_component_dof_count(&invalid_order, 0, &result) == CONSTRAINT_INVALID_ORDER,
                   "Invalid component specification was accepted.");
}

static void test_row_representation(void)
{
    const constraint_entry_t entries[] = {
        {.side = 0, .component = 0, .local_dof = 4, .coefficient = 1.0},
        {.side = 1, .component = 0, .local_dof = 2, .coefficient = -1.0},
        {.side = 0, .component = 1, .local_dof = 7, .coefficient = 0.5},
    };
    const size_t offsets[] = {0, 2, 3};
    const constraint_rows_view_t view = {
        .row_count = 2,
        .entry_count = 3,
        .row_offsets = offsets,
        .entries = entries,
    };
    size_t required;

    TEST_ASSERTION(constraint_rows_required_offset_count(2, &required) == CONSTRAINT_SUCCESS && required == 3,
                   "Unexpected row-offset capacity.");
    TEST_ASSERTION(constraint_rows_required_entry_capacity(2, 3, &required) == CONSTRAINT_SUCCESS && required == 6,
                   "Unexpected entry capacity.");
    TEST_ASSERTION(constraint_rows_validate(view) == CONSTRAINT_SUCCESS, "Valid row representation was rejected.");

    const size_t bad_offsets[] = {0, 4, 3};
    TEST_ASSERTION(constraint_rows_validate((constraint_rows_view_t){2, 3, bad_offsets, entries}) ==
                       CONSTRAINT_INVALID_ARGUMENT,
                   "Non-monotonic row offsets were accepted.");
    const constraint_entry_t bad_entry = {.side = 2};
    TEST_ASSERTION(constraint_rows_validate((constraint_rows_view_t){1, 1, (size_t[]){0, 1}, &bad_entry}) ==
                       CONSTRAINT_INVALID_ARGUMENT,
                   "Invalid side was accepted.");
}

static void test_reference_sizing(void)
{
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis_1[] = {basis_spec(1), basis_spec(2)};
    const basis_spec_t element_basis_2[] = {basis_spec(2), basis_spec(1)};
    const int8_t orientation_1[] = {-1, 2};
    const int8_t orientation_2[] = {1, -2};
    const constraint_kform_spec_t test_spec = {.ndim = 1, .order = 0, .basis_specs = test_basis};
    const constraint_element_side_t sides[] = {
        {.ndim = 2, .basis_specs = element_basis_1, .orientation = orientation_1},
        {.ndim = 2, .basis_specs = element_basis_2, .orientation = orientation_2},
    };
    size_t row_count;
    size_t entry_count;

    TEST_ASSERTION(constraint_reference_required(&test_spec, sides, &row_count, &entry_count) == CONSTRAINT_SUCCESS,
                   "Could not size reference constraints.");
    TEST_ASSERTION(row_count == 2, "Unexpected reference constraint row count.");
    TEST_ASSERTION(entry_count == 24, "Unexpected reference constraint entry count.");
}

static void test_reference_endpoint_assembly(void)
{
    const basis_spec_t element_basis[] = {basis_spec(1)};
    const int8_t lower[] = {-1};
    const int8_t upper[] = {1};
    const constraint_kform_spec_t test_spec = {.ndim = 0, .order = 0, .basis_specs = NULL};
    const constraint_element_side_t sides[] = {
        {.ndim = 1, .basis_specs = element_basis, .orientation = lower},
        {.ndim = 1, .basis_specs = element_basis, .orientation = upper},
    };
    size_t row_offsets[2];
    constraint_entry_t entries[4];
    size_t row_count;
    size_t entry_count;

    TEST_ASSERTION(constraint_reference_assemble(&test_spec, sides, NULL, 2, row_offsets, 4, entries, &row_count,
                                                 &entry_count) == CONSTRAINT_SUCCESS,
                   "Could not assemble endpoint constraint.");
    TEST_ASSERTION(row_count == 1 && entry_count == 4, "Unexpected endpoint constraint dimensions.");
    TEST_ASSERTION(row_offsets[0] == 0 && row_offsets[1] == 4, "Unexpected endpoint row offsets.");
    TEST_ASSERTION(entries[0].side == 0 && entries[0].local_dof == 0, "Unexpected lower endpoint entry.");
    TEST_NUMBERS_CLOSE(entries[0].coefficient, 1.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[1].coefficient, -1.0, 1e-12, 0);
    TEST_ASSERTION(entries[2].side == 1 && entries[2].local_dof == 0, "Unexpected upper endpoint entry.");
    TEST_NUMBERS_CLOSE(entries[2].coefficient, -1.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[3].coefficient, -1.0, 1e-12, 0);
}

static void test_reference_edge_assembly(void)
{
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1)};
    const int8_t lower[] = {-1, 2};
    const int8_t upper_reversed[] = {1, -2};
    const double nodes[] = {-0.5773502691896258, 0.5773502691896258};
    const double weights[] = {1.0, 1.0};
    const constraint_quadrature_t quadrature[] = {{.count = 2, .nodes = nodes, .weights = weights}};
    const constraint_kform_spec_t test_spec = {.ndim = 1, .order = 0, .basis_specs = test_basis};
    const constraint_element_side_t sides[] = {
        {.ndim = 2, .basis_specs = element_basis, .orientation = lower},
        {.ndim = 2, .basis_specs = element_basis, .orientation = upper_reversed},
    };
    size_t row_offsets[3];
    constraint_entry_t entries[16];
    size_t row_count;
    size_t entry_count;
    size_t required_rows;
    size_t required_entries;
    TEST_ASSERTION(constraint_reference_required(&test_spec, sides, &required_rows, &required_entries) ==
                       CONSTRAINT_SUCCESS,
                   "Could not size edge constraint.");
    TEST_ASSERTION(required_rows == 2 && required_entries == 16, "Unexpected edge sizing: %zu rows, %zu entries.",
                   required_rows, required_entries);

    const constraint_status_t status = constraint_reference_assemble(&test_spec, sides, quadrature, 3, row_offsets, 16,
                                                                     entries, &row_count, &entry_count);
    TEST_ASSERTION(status == CONSTRAINT_SUCCESS, "Could not assemble edge constraint: %s (%s)",
                   constraint_status_to_str(status), constraint_status_msg(status));
    TEST_ASSERTION(row_count == 2 && entry_count == 16, "Unexpected edge constraint dimensions.");
    TEST_ASSERTION(row_offsets[0] == 0 && row_offsets[1] == 8 && row_offsets[2] == 16, "Unexpected edge row offsets.");
    TEST_NUMBERS_CLOSE(entries[0].coefficient, 2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[1].coefficient, 0.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[2].coefficient, -2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[3].coefficient, 0.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[8].coefficient, 0.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[9].coefficient, 2.0 / 3.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[10].coefficient, 0.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[11].coefficient, -2.0 / 3.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[12].coefficient, 0.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[13].coefficient, 2.0 / 3.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[14].coefficient, 0.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[15].coefficient, 2.0 / 3.0, 1e-12, 0);
}

static void test_reference_one_form_component(void)
{
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1)};
    const int8_t lower[] = {-1, 2};
    const int8_t upper[] = {1, 2};
    const double nodes[] = {-0.5773502691896258, 0.5773502691896258};
    const double weights[] = {1.0, 1.0};
    const constraint_quadrature_t quadrature[] = {{.count = 2, .nodes = nodes, .weights = weights}};
    const constraint_kform_spec_t test_spec = {.ndim = 1, .order = 1, .basis_specs = test_basis};
    const constraint_element_side_t sides[] = {
        {.ndim = 2, .basis_specs = element_basis, .orientation = lower},
        {.ndim = 2, .basis_specs = element_basis, .orientation = upper},
    };
    size_t row_offsets[2];
    constraint_entry_t entries[4];
    size_t row_count;
    size_t entry_count;

    TEST_ASSERTION(constraint_reference_assemble(&test_spec, sides, quadrature, 2, row_offsets, 4, entries, &row_count,
                                                 &entry_count) == CONSTRAINT_SUCCESS,
                   "Could not assemble tangential one-form constraint.");
    TEST_ASSERTION(row_count == 1 && entry_count == 4, "Unexpected tangential one-form dimensions.");
    TEST_NUMBERS_CLOSE(entries[0].coefficient, 2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[1].coefficient, -2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[2].coefficient, -2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[3].coefficient, -2.0, 1e-12, 0);

    const constraint_kform_spec_t invalid_test = {.ndim = 1, .order = 2, .basis_specs = test_basis};
    size_t required_rows;
    size_t required_entries;
    TEST_ASSERTION(constraint_reference_required(&invalid_test, sides, &required_rows, &required_entries) ==
                       CONSTRAINT_INVALID_ORDER,
                   "A normal one-form test was accepted on a one-dimensional boundary.");
}

static void test_physical_scalar_measure(void)
{
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1)};
    const int8_t lower[] = {-1, 2};
    const int8_t upper[] = {1, 2};
    const double nodes[] = {-0.5773502691896258, 0.5773502691896258};
    const double weights[] = {1.0, 1.0};
    const double surface_weights[] = {3.0, 3.0};
    const constraint_quadrature_t quadrature[] = {{.count = 2, .nodes = nodes, .weights = weights}};
    const constraint_face_quadrature_t face_quadrature[] = {
        {.ndim = 1, .axes = quadrature, .point_count = 2},
        {.ndim = 1, .axes = quadrature, .point_count = 2},
    };
    const double *const side_surface_weights[] = {surface_weights, surface_weights};
    const constraint_kform_spec_t test_spec = {.ndim = 1, .order = 0, .basis_specs = test_basis};
    const constraint_element_side_t sides[] = {
        {.ndim = 2, .basis_specs = element_basis, .orientation = lower},
        {.ndim = 2, .basis_specs = element_basis, .orientation = upper},
    };
    const constraint_trace_pullback_t pullbacks[2] = {};
    size_t row_offsets[3];
    constraint_entry_t entries[16];
    size_t reference_row_offsets[3];
    constraint_entry_t reference_entries[16];
    size_t row_count;
    size_t entry_count;
    size_t reference_row_count;
    size_t reference_entry_count;

    TEST_ASSERTION(constraint_reference_assemble(&test_spec, sides, quadrature, 3, reference_row_offsets, 16,
                                                 reference_entries, &reference_row_count,
                                                 &reference_entry_count) == CONSTRAINT_SUCCESS,
                   "Could not assemble reference scalar constraint.");

    const constraint_status_t status =
        constraint_physical_assemble(&test_spec, sides, face_quadrature, side_surface_weights, pullbacks, 3,
                                     row_offsets, 16, entries, &row_count, &entry_count);
    TEST_ASSERTION(status == CONSTRAINT_SUCCESS, "Could not assemble weighted scalar constraint: %s (%s)",
                   constraint_status_to_str(status), constraint_status_msg(status));
    TEST_ASSERTION(row_count == reference_row_count && entry_count == reference_entry_count && row_count == 2 &&
                       entry_count == 16,
                   "Unexpected weighted scalar dimensions.");
    for (size_t i = 0; i < entry_count; ++i)
        TEST_NUMBERS_CLOSE(entries[i].coefficient, 3.0 * reference_entries[i].coefficient, 1e-12, 0);
}

static void test_physical_single_side(void)
{
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1)};
    const int8_t orientation[] = {-1, 2};
    const double nodes[] = {-0.5773502691896258, 0.5773502691896258};
    const double weights[] = {1.0, 1.0};
    const double surface_weights[] = {1.0, 1.0};
    const constraint_quadrature_t axes[] = {{.count = 2, .nodes = nodes, .weights = weights}};
    const constraint_face_quadrature_t quadrature = {.ndim = 1, .axes = axes, .point_count = 2};
    const constraint_kform_spec_t test_spec = {.ndim = 1, .order = 0, .basis_specs = test_basis};
    const constraint_element_side_t side = {.ndim = 2, .basis_specs = element_basis, .orientation = orientation};
    size_t row_offsets[3];
    constraint_entry_t entries[8];
    size_t row_count;
    size_t entry_count;

    TEST_ASSERTION(constraint_physical_side_assemble(&test_spec, &side, &quadrature, surface_weights, NULL, 3,
                                                     row_offsets, 8, entries, &row_count,
                                                     &entry_count) == CONSTRAINT_SUCCESS,
                   "Could not assemble one-element scalar boundary constraints.");
    TEST_ASSERTION(row_count == 2 && entry_count == 8, "Unexpected one-element constraint dimensions.");
    TEST_ASSERTION(row_offsets[0] == 0 && row_offsets[1] == 4 && row_offsets[2] == 8,
                   "Unexpected one-element row offsets.");
    TEST_ASSERTION(entries[0].side == 0 && entries[7].side == 0, "Unexpected one-element side metadata.");
}

static void test_physical_general_boundary_dimensions(void)
{
    const basis_spec_t point_basis[] = {basis_spec(1), basis_spec(1), basis_spec(1)};
    const int8_t point_orientation[] = {-1, 2, 3};
    const constraint_kform_spec_t point_test = {.ndim = 0, .order = 0, .basis_specs = NULL};
    const constraint_element_side_t point_side = {
        .ndim = 3, .basis_specs = point_basis, .orientation = point_orientation};
    const constraint_face_quadrature_t point_quadrature = {.ndim = 0, .axes = NULL, .point_count = 1};
    const double point_weight[] = {1.0};
    size_t row_offsets[2];
    constraint_entry_t entries[8];
    size_t row_count;
    size_t entry_count;

    TEST_ASSERTION(constraint_physical_side_assemble(&point_test, &point_side, &point_quadrature, point_weight, NULL, 2,
                                                     row_offsets, 8, entries, &row_count,
                                                     &entry_count) == CONSTRAINT_SUCCESS,
                   "Could not assemble a three-dimensional point boundary.");
    TEST_ASSERTION(row_count == 1 && entry_count == 8, "Unexpected point-boundary dimensions.");

    const basis_spec_t line_basis[] = {basis_spec(1), basis_spec(1), basis_spec(1)};
    const int8_t line_orientation[] = {-1, 3, -2};
    const basis_spec_t line_test_basis[] = {basis_spec(1)};
    const constraint_kform_spec_t line_test = {.ndim = 1, .order = 0, .basis_specs = line_test_basis};
    const constraint_element_side_t line_side = {.ndim = 3, .basis_specs = line_basis, .orientation = line_orientation};
    const double line_nodes[] = {-0.5773502691896258, 0.5773502691896258};
    const double line_weights[] = {1.0, 1.0};
    const constraint_quadrature_t line_axes[] = {{.count = 2, .nodes = line_nodes, .weights = line_weights}};
    const constraint_face_quadrature_t line_quadrature = {.ndim = 1, .axes = line_axes, .point_count = 2};
    const double line_surface[] = {1.0, 1.0};
    TEST_ASSERTION(constraint_physical_side_required(&line_test, &line_side, &row_count, &entry_count) ==
                       CONSTRAINT_SUCCESS,
                   "Could not size a three-dimensional line boundary.");
    constraint_entry_t *line_entries = malloc(entry_count * sizeof(*line_entries));
    size_t *line_offsets = malloc((row_count + 1) * sizeof(*line_offsets));
    TEST_ASSERTION(line_entries && line_offsets, "Could not allocate line-boundary test storage.");
    TEST_ASSERTION(constraint_physical_side_assemble(&line_test, &line_side, &line_quadrature, line_surface, NULL,
                                                     row_count + 1, line_offsets, entry_count, line_entries, &row_count,
                                                     &entry_count) == CONSTRAINT_SUCCESS,
                   "Could not assemble a three-dimensional line boundary.");
    free(line_entries);
    free(line_offsets);

    const basis_spec_t face_basis[] = {basis_spec(1), basis_spec(1), basis_spec(1), basis_spec(1)};
    const int8_t face_orientation[] = {-1, 3, -2, 4};
    const basis_spec_t face_test_basis[] = {basis_spec(1), basis_spec(1)};
    const constraint_kform_spec_t face_test = {.ndim = 2, .order = 0, .basis_specs = face_test_basis};
    const constraint_element_side_t face_side = {.ndim = 4, .basis_specs = face_basis, .orientation = face_orientation};
    const constraint_quadrature_t face_axes[] = {
        {.count = 2, .nodes = line_nodes, .weights = line_weights},
        {.count = 2, .nodes = line_nodes, .weights = line_weights},
    };
    const constraint_face_quadrature_t face_quadrature = {.ndim = 2, .axes = face_axes, .point_count = 4};
    const double face_surface[] = {1.0, 1.0, 1.0, 1.0};
    TEST_ASSERTION(constraint_physical_side_required(&face_test, &face_side, &row_count, &entry_count) ==
                       CONSTRAINT_SUCCESS,
                   "Could not size a four-dimensional face boundary.");
    TEST_ASSERTION(row_count == 4 && entry_count == 64, "Unexpected four-dimensional face dimensions.");

    const basis_spec_t face_one_form_basis[] = {basis_spec(1), basis_spec(1)};
    const constraint_kform_spec_t face_one_form_test = {.ndim = 2, .order = 1, .basis_specs = face_one_form_basis};
    double face_pullback_values[4 * 1 * 4];
    for (unsigned i = 0; i < sizeof(face_pullback_values) / sizeof(*face_pullback_values); ++i)
        face_pullback_values[i] = 1.0;
    const constraint_trace_pullback_t face_pullback = {
        .physical_component_count = 1, .point_count = 4, .values = face_pullback_values};
    TEST_ASSERTION(constraint_physical_side_required(&face_one_form_test, &face_side, &row_count, &entry_count) ==
                       CONSTRAINT_SUCCESS,
                   "Could not size a four-dimensional one-form face boundary.");
    TEST_ASSERTION(row_count == 4 && entry_count == 64, "Unexpected four-dimensional one-form dimensions.");
    constraint_entry_t face_one_form_entries[64];
    size_t face_one_form_offsets[5];
    const constraint_status_t face_status = constraint_physical_side_assemble(
        &face_one_form_test, &face_side, &face_quadrature, face_surface, &face_pullback, 5, face_one_form_offsets, 64,
        face_one_form_entries, &row_count, &entry_count);
    TEST_ASSERTION(face_status == CONSTRAINT_SUCCESS,
                   "Could not assemble a four-dimensional one-form face boundary: %s",
                   constraint_status_to_str(face_status));
    TEST_ASSERTION(face_one_form_entries[0].component == 1 && face_one_form_entries[8].component == 3,
                   "Unexpected odd-orientation one-form component mapping.");
}

static void test_physical_one_form_pullback(void)
{
    const basis_spec_t test_basis[] = {basis_spec(1)};
    const basis_spec_t element_basis[] = {basis_spec(1), basis_spec(1)};
    const int8_t lower[] = {-1, 2};
    const int8_t upper[] = {1, 2};
    const double nodes[] = {-0.5773502691896258, 0.5773502691896258};
    const double weights[] = {1.0, 1.0};
    const double surface_weights[] = {1.0, 1.0};
    const double identity_pullback[] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    };
    const constraint_quadrature_t quadrature[] = {{.count = 2, .nodes = nodes, .weights = weights}};
    const constraint_face_quadrature_t face_quadrature[] = {
        {.ndim = 1, .axes = quadrature, .point_count = 2},
        {.ndim = 1, .axes = quadrature, .point_count = 2},
    };
    const double *const side_surface_weights[] = {surface_weights, surface_weights};
    const constraint_kform_spec_t test_spec = {.ndim = 1, .order = 1, .basis_specs = test_basis};
    const constraint_element_side_t sides[] = {
        {.ndim = 2, .basis_specs = element_basis, .orientation = lower},
        {.ndim = 2, .basis_specs = element_basis, .orientation = upper},
    };
    const constraint_trace_pullback_t pullbacks[] = {
        {.physical_component_count = 2, .point_count = 2, .values = identity_pullback},
        {.physical_component_count = 2, .point_count = 2, .values = identity_pullback},
    };
    size_t row_offsets[2];
    constraint_entry_t entries[4];
    size_t row_count;
    size_t entry_count;

    TEST_ASSERTION(constraint_physical_assemble(&test_spec, sides, face_quadrature, side_surface_weights, pullbacks, 2,
                                                row_offsets, 4, entries, &row_count,
                                                &entry_count) == CONSTRAINT_SUCCESS,
                   "Could not assemble pulled-back one-form constraint.");
    TEST_ASSERTION(row_count == 1 && entry_count == 4, "Unexpected pulled-back one-form dimensions.");
    TEST_ASSERTION(entries[0].component == 1 && entries[1].component == 1 && entries[2].component == 1 &&
                       entries[3].component == 1,
                   "Normal one-form component was included in the physical trace.");
    TEST_NUMBERS_CLOSE(entries[0].coefficient, 2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[1].coefficient, -2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[2].coefficient, -2.0, 1e-12, 0);
    TEST_NUMBERS_CLOSE(entries[3].coefficient, -2.0, 1e-12, 0);
}

int main(void)
{
    test_component_layout();
    test_scalar_component();
    test_invalid_specs();
    test_row_representation();
    test_reference_sizing();
    test_reference_endpoint_assembly();
    test_reference_edge_assembly();
    test_reference_one_form_component();
    test_physical_scalar_measure();
    test_physical_single_side();
    test_physical_general_boundary_dimensions();
    test_physical_one_form_pullback();
    return 0;
}
