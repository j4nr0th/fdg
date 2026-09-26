#include "../../src/constraints/constraints.h"
#include "../../src/integration/integration_rules.h"
#include "../common/common.h"

#include <stdlib.h>
#include <string.h>

enum
{
    MAX_TEST_DIM = 4,
    MAX_TEST_COORDS = 4,
};

/**
 * One resample case: source samples on the face grid, request work and outputs.
 *
 * The samples are filled analytically by the test cases at the source rule
 * nodes, then resampled onto the common (target) grid and checked against the
 * analytic factors there.
 */
typedef struct
{
    integration_rule_registry_t *registry;
    const integration_rule_t *source_rules[MAX_TEST_DIM];
    const integration_rule_t *target_rules[MAX_TEST_DIM];
    unsigned bdim;
    unsigned coords;
    size_t source_points;
    size_t target_points;
    size_t source_strides[MAX_TEST_DIM];
    size_t target_strides[MAX_TEST_DIM];
    double *source_values;    // coords * source_points
    double *source_gradients; // coords * bdim * source_points
    double *determinant;      // target_points
    double *inverse_maps;     // target_points * bdim * coords
    const double *value_ptrs[MAX_TEST_COORDS];
    const double *gradient_ptrs[MAX_TEST_COORDS * MAX_TEST_DIM];
    double *axis_matrices;
    double *positions;
    double *jacobian;
    double *q;
    unsigned *target_orders;
    unsigned *source_orders;
    const double **axis_matrix_rows;
    integration_spec_t *target_specs;
} resample_case_t;

/** Fetch one rule per axis of @p specs from @p registry. */
static void rules_get(integration_rule_registry_t *const registry, const unsigned count,
                      const integration_spec_t specs[const static count],
                      const integration_rule_t *rules[const static count])
{
    for (unsigned axis = 0; axis < count; ++axis)
    {
        TEST_FDG_RESULT(integration_rule_registry_get_rule(registry, specs[axis], rules + axis));
    }
}

/**
 * Create the registry, fetch the source and target rules, and allocate the
 * samples, outputs, and request work of one case; the work is sentinel filled
 * to catch reads of untouched scratch.
 */
static void case_create(resample_case_t *const c, const unsigned bdim, const unsigned coords,
                        const integration_spec_t source_specs[], const integration_spec_t target_specs[])
{
    *c = (resample_case_t){.bdim = bdim, .coords = coords};
    TEST_FDG_RESULT(integration_rule_registry_create(&c->registry, 1, &TEST_ALLOCATOR));
    rules_get(c->registry, bdim, source_specs, c->source_rules);
    rules_get(c->registry, bdim, target_specs, c->target_rules);

    size_t source_dims[MAX_TEST_DIM], target_dims[MAX_TEST_DIM];
    c->source_points = 1;
    c->target_points = 1;
    for (unsigned axis = 0; axis < bdim; ++axis)
    {
        source_dims[axis] = c->source_rules[axis]->spec.order + 1;
        target_dims[axis] = c->target_rules[axis]->spec.order + 1;
        c->source_points *= source_dims[axis];
        c->target_points *= target_dims[axis];
    }
    test_tensor_strides(bdim, source_dims, c->source_strides);
    test_tensor_strides(bdim, target_dims, c->target_strides);

    size_t axis_matrices_size, positions_size, jacobian_size, q_size, scratch_bytes;
    boundary_space_map_resample_work_size(bdim, coords, c->source_rules, c->target_rules, &axis_matrices_size,
                                          &positions_size, &jacobian_size, &q_size, &scratch_bytes);

    c->source_values = malloc(coords * c->source_points * sizeof(*c->source_values));
    c->source_gradients = malloc(coords * bdim * c->source_points * sizeof(*c->source_gradients));
    c->determinant = malloc(c->target_points * sizeof(*c->determinant));
    c->inverse_maps = malloc(c->target_points * bdim * coords * sizeof(*c->inverse_maps));
    c->axis_matrices = malloc(axis_matrices_size * sizeof(*c->axis_matrices));
    c->positions = malloc(positions_size * sizeof(*c->positions));
    c->jacobian = malloc(jacobian_size * sizeof(*c->jacobian));
    c->q = malloc(q_size * sizeof(*c->q));
    void *scratch = malloc(scratch_bytes);
    TEST_ASSERTION(c->source_values != NULL && c->source_gradients != NULL && c->determinant != NULL &&
                       c->inverse_maps != NULL && c->axis_matrices != NULL && c->positions != NULL &&
                       c->jacobian != NULL && c->q != NULL && scratch != NULL,
                   "Failed to allocate case buffers.");

    // Sentinel fill to catch any read of untouched work memory.
    for (size_t i = 0; i < axis_matrices_size; ++i)
        c->axis_matrices[i] = 1234.5;
    for (size_t i = 0; i < positions_size; ++i)
        c->positions[i] = 1234.5;
    for (size_t i = 0; i < jacobian_size; ++i)
        c->jacobian[i] = 1234.5;
    for (size_t i = 0; i < q_size; ++i)
        c->q[i] = 1234.5;
    memset(scratch, 0, scratch_bytes);

    // Carve the scratch block into the request's per-axis work arrays.
    c->target_orders = scratch;
    c->source_orders = c->target_orders + bdim;
    c->axis_matrix_rows = (const double **)(const void *)(c->source_orders + bdim);
    c->target_specs = (integration_spec_t *)(void *)(c->axis_matrix_rows + bdim);

    for (unsigned coordinate = 0; coordinate < coords; ++coordinate)
    {
        c->value_ptrs[coordinate] = c->source_values + (size_t)coordinate * c->source_points;
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            c->gradient_ptrs[coordinate * bdim + axis] =
                c->source_gradients + ((size_t)coordinate * bdim + axis) * c->source_points;
        }
    }
}

/** Run one resample; the source samples must survive the call untouched. */
static void case_run(resample_case_t *const c)
{
    const size_t value_count = (size_t)c->coords * c->source_points;
    const size_t gradient_count = (size_t)c->coords * c->bdim * c->source_points;
    double *const golden_values = malloc(value_count * sizeof(*golden_values));
    double *const golden_gradients = malloc(gradient_count * sizeof(*golden_gradients));
    TEST_ASSERTION(golden_values != NULL && golden_gradients != NULL, "Failed to allocate golden buffers.");
    memcpy(golden_values, c->source_values, value_count * sizeof(*golden_values));
    memcpy(golden_gradients, c->source_gradients, gradient_count * sizeof(*golden_gradients));

    const boundary_space_map_resample_request_t request = {
        .bdim = c->bdim,
        .coords = c->coords,
        .source_rules = c->source_rules,
        .target_rules = c->target_rules,
        .coordinate_values = c->value_ptrs,
        .coordinate_gradients = c->gradient_ptrs,
        .out_determinant = c->determinant,
        .out_inverse_maps = c->inverse_maps,
        .axis_matrices = c->axis_matrices,
        .positions = c->positions,
        .jacobian = c->jacobian,
        .q = c->q,
        .target_orders = c->target_orders,
        .source_orders = c->source_orders,
        .axis_matrix_rows = c->axis_matrix_rows,
        .target_specs = c->target_specs,
    };
    boundary_space_map_resample(&request);

    for (size_t i = 0; i < value_count; ++i)
        TEST_NUMBERS_CLOSE(c->source_values[i], golden_values[i], 0.0, 0.0);
    for (size_t i = 0; i < gradient_count; ++i)
        TEST_NUMBERS_CLOSE(c->source_gradients[i], golden_gradients[i], 0.0, 0.0);

    free(golden_values);
    free(golden_gradients);
}

/** Release the rules, destroy the registry, and free every buffer of one case. */
static void case_destroy(resample_case_t *const c)
{
    for (unsigned axis = 0; axis < c->bdim; ++axis)
    {
        TEST_FDG_RESULT(integration_rule_registry_release_rule(c->registry, c->source_rules[axis]));
        TEST_FDG_RESULT(integration_rule_registry_release_rule(c->registry, c->target_rules[axis]));
    }
    integration_rule_registry_destroy(c->registry);
    free(c->source_values);
    free(c->source_gradients);
    free(c->determinant);
    free(c->inverse_maps);
    free(c->axis_matrices);
    free(c->positions);
    free(c->jacobian);
    free(c->q);
}

/** Advance one row-major multi-index in place, the last axis fastest. */
static void advance_index(const unsigned rank, const size_t dims[const static rank], size_t idx[const static rank])
{
    for (unsigned axis = rank; axis-- > 0;)
    {
        if (++idx[axis] < dims[axis])
            return;
        idx[axis] = 0;
    }
}

/**
 * The returned backward derivative must be the left inverse of the analytic
 * forward Jacobian: J is of full column rank on a face, so its left inverse is
 * unique and this pins every entry and layout of the output.
 */
static void check_left_inverse(const unsigned bdim, const unsigned coords, const double *const jacobian,
                               const double *const inverse)
{
    for (unsigned i = 0; i < bdim; ++i)
    {
        for (unsigned j = 0; j < bdim; ++j)
        {
            double sum = 0.0;
            for (unsigned c = 0; c < coords; ++c)
            {
                sum += inverse[(size_t)i * coords + c] * jacobian[(size_t)c * bdim + j];
            }
            TEST_NUMBERS_CLOSE(sum, i == j ? 1.0 : 0.0, 1e-10, 1e-10);
        }
    }
}

/**
 * A curved one-dimensional face: F(t) = (1 + 0.3 t^2, t) resampled from the
 * element rule onto a finer common rule must reproduce the analytic surface
 * measure, positions, and backward derivatives at every common node.
 */
static void test_curved_face(void)
{
    resample_case_t c;
    const integration_spec_t source_specs[] = {{.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 4}};
    const integration_spec_t target_specs[] = {{.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 5}};
    case_create(&c, 1, 2, source_specs, target_specs);

    const double *const t_nodes = integration_rule_nodes_const(c.source_rules[0]);
    for (size_t i = 0; i < c.source_points; ++i)
    {
        const double t = t_nodes[i];
        c.source_values[i] = 1.0 + 0.3 * t * t;
        c.source_values[c.source_points + i] = t;
        c.source_gradients[i] = 0.6 * t;
        c.source_gradients[c.source_points + i] = 1.0;
    }
    case_run(&c);

    const double *const target = integration_rule_nodes_const(c.target_rules[0]);
    for (size_t i = 0; i < c.target_points; ++i)
    {
        const double t = target[i];
        // Surface measure of the plane curve: |F'(t)|.
        TEST_NUMBERS_CLOSE(c.determinant[i], sqrt(1.0 + 0.36 * t * t), 1e-12, 1e-10);
        TEST_NUMBERS_CLOSE(c.positions[2 * i], 1.0 + 0.3 * t * t, 1e-12, 1e-10);
        TEST_NUMBERS_CLOSE(c.positions[2 * i + 1], t, 1e-12, 1e-10);
        const double jacobian[2] = {0.6 * t, 1.0};
        check_left_inverse(1, 2, jacobian, c.inverse_maps + i * 2);
    }

    case_destroy(&c);
}

/**
 * A two-dimensional face in three dimensions: F(xi, eta) = (1, 2 xi + 0.3 eta,
 * 5 eta + 0.3 xi eta^2) resampled from the element rule onto an anisotropic
 * common rule must reproduce the analytic cross-product measure.
 */
static void test_two_dimensional_face(void)
{
    resample_case_t c;
    const integration_spec_t source_specs[] = {{.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 5},
                                               {.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 5}};
    const integration_spec_t target_specs[] = {{.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 4},
                                               {.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 3}};
    case_create(&c, 2, 3, source_specs, target_specs);

    const double *const xi_nodes = integration_rule_nodes_const(c.source_rules[0]);
    const double *const eta_nodes = integration_rule_nodes_const(c.source_rules[1]);
    const size_t source_dims[2] = {c.source_rules[0]->spec.order + 1, c.source_rules[1]->spec.order + 1};
    size_t idx[2] = {0};
    for (size_t point = 0; point < c.source_points; ++point)
    {
        const double xi = xi_nodes[idx[0]];
        const double eta = eta_nodes[idx[1]];
        const size_t offset = idx[0] * c.source_strides[0] + idx[1] * c.source_strides[1];
        c.source_values[0 * c.source_points + offset] = 1.0;
        c.source_values[1 * c.source_points + offset] = 2.0 * xi + 0.3 * eta;
        c.source_values[2 * c.source_points + offset] = 5.0 * eta + 0.3 * xi * eta * eta;
        // dF/dxi.
        c.source_gradients[(0 * 2 + 0) * c.source_points + offset] = 0.0;
        c.source_gradients[(1 * 2 + 0) * c.source_points + offset] = 2.0;
        c.source_gradients[(2 * 2 + 0) * c.source_points + offset] = 0.3 * eta * eta;
        // dF/deta.
        c.source_gradients[(0 * 2 + 1) * c.source_points + offset] = 0.0;
        c.source_gradients[(1 * 2 + 1) * c.source_points + offset] = 0.3;
        c.source_gradients[(2 * 2 + 1) * c.source_points + offset] = 5.0 + 0.6 * xi * eta;
        advance_index(2, source_dims, idx);
    }
    case_run(&c);

    const double *const target_xi = integration_rule_nodes_const(c.target_rules[0]);
    const double *const target_eta = integration_rule_nodes_const(c.target_rules[1]);
    size_t t_idx[2] = {0};
    const size_t target_dims[2] = {c.target_rules[0]->spec.order + 1, c.target_rules[1]->spec.order + 1};
    for (size_t point = 0; point < c.target_points; ++point)
    {
        const double xi = target_xi[t_idx[0]];
        const double eta = target_eta[t_idx[1]];
        // Surface measure: |dF/dxi x dF/deta|.
        TEST_NUMBERS_CLOSE(c.determinant[point], 10.0 + 1.2 * xi * eta - 0.09 * eta * eta, 1e-12, 1e-10);
        TEST_NUMBERS_CLOSE(c.positions[3 * point], 1.0, 1e-12, 1e-10);
        TEST_NUMBERS_CLOSE(c.positions[3 * point + 1], 2.0 * xi + 0.3 * eta, 1e-12, 1e-10);
        TEST_NUMBERS_CLOSE(c.positions[3 * point + 2], 5.0 * eta + 0.3 * xi * eta * eta, 1e-12, 1e-10);
        const double jacobian[6] = {0.0, 0.0, 2.0, 0.3, 0.3 * eta * eta, 5.0 + 0.6 * xi * eta};
        check_left_inverse(2, 3, jacobian, c.inverse_maps + (size_t)point * 2 * 3);
        advance_index(2, target_dims, t_idx);
    }

    case_destroy(&c);
}

/** Value of the tensor polynomial with the given coefficients at (x, y). */
static double poly_value(const double coefficients[3][3], const double x, const double y)
{
    double value = 0.0;
    double x_power = 1.0;
    for (unsigned i = 0; i < 3; ++i)
    {
        double y_power = 1.0;
        for (unsigned j = 0; j < 3; ++j)
        {
            value += coefficients[i][j] * x_power * y_power;
            y_power *= y;
        }
        x_power *= x;
    }
    return value;
}

/** Partial derivatives of the tensor polynomial with the given coefficients at (x, y). */
static void poly_gradient(const double coefficients[3][3], const double x, const double y, double *const dx,
                          double *const dy)
{
    *dx = 0.0;
    *dy = 0.0;
    double x_power = 1.0;
    double x_power_prev = 0.0;
    for (unsigned i = 0; i < 3; ++i)
    {
        double y_power = 1.0;
        double y_power_prev = 0.0;
        for (unsigned j = 0; j < 3; ++j)
        {
            if (i > 0)
                *dx += (double)i * coefficients[i][j] * x_power_prev * y_power;
            if (j > 0)
                *dy += (double)j * coefficients[i][j] * x_power * y_power_prev;
            y_power_prev = y_power;
            y_power *= y;
        }
        x_power_prev = x_power;
        x_power *= x;
    }
}

/**
 * Randomized exactness: a quadratic surface resolved by the source rule must
 * resample to the analytic factors on a coarser common grid. The face
 * immersion is the identity base plus a random quadratic per coordinate, which
 * keeps the Jacobian well conditioned for any coefficients.
 */
static void test_polynomial_exactness(void)
{
    resample_case_t c;
    const integration_spec_t source_specs[] = {{.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 4},
                                               {.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 4}};
    const integration_spec_t target_specs[] = {{.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 3},
                                               {.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = 3}};
    case_create(&c, 2, 3, source_specs, target_specs);

    test_prng_t rng;
    test_prng_seed(&rng, 20260926u);
    double coefficients[3][3][3]; // [coordinate][i][j]
    for (unsigned coordinate = 0; coordinate < 3; ++coordinate)
    {
        test_fill_random(&coefficients[coordinate][0][0], 9, &rng);
    }

    const double *const xi_nodes = integration_rule_nodes_const(c.source_rules[0]);
    const double *const eta_nodes = integration_rule_nodes_const(c.source_rules[1]);
    const size_t source_dims[2] = {c.source_rules[0]->spec.order + 1, c.source_rules[1]->spec.order + 1};
    size_t idx[2] = {0};
    for (size_t point = 0; point < c.source_points; ++point)
    {
        const double xi = xi_nodes[idx[0]];
        const double eta = eta_nodes[idx[1]];
        const size_t offset = idx[0] * c.source_strides[0] + idx[1] * c.source_strides[1];
        const double base[3] = {xi, eta, 0.0};
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate)
        {
            double dx, dy;
            poly_gradient(coefficients[coordinate], xi, eta, &dx, &dy);
            c.source_values[(size_t)coordinate * c.source_points + offset] =
                base[coordinate] + poly_value(coefficients[coordinate], xi, eta);
            c.source_gradients[((size_t)coordinate * 2 + 0) * c.source_points + offset] =
                (coordinate == 0 ? 1.0 : 0.0) + dx;
            c.source_gradients[((size_t)coordinate * 2 + 1) * c.source_points + offset] =
                (coordinate == 1 ? 1.0 : 0.0) + dy;
        }
        advance_index(2, source_dims, idx);
    }
    case_run(&c);

    const double *const target_xi = integration_rule_nodes_const(c.target_rules[0]);
    const double *const target_eta = integration_rule_nodes_const(c.target_rules[1]);
    const size_t target_dims[2] = {c.target_rules[0]->spec.order + 1, c.target_rules[1]->spec.order + 1};
    size_t t_idx[2] = {0};
    for (size_t point = 0; point < c.target_points; ++point)
    {
        const double xi = target_xi[t_idx[0]];
        const double eta = target_eta[t_idx[1]];
        const double base[3] = {xi, eta, 0.0};
        double jacobian[6]; // [coordinate * bdim + axis]
        double gram[4] = {0.0};
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate)
        {
            double dx, dy;
            poly_gradient(coefficients[coordinate], xi, eta, &dx, &dy);
            jacobian[coordinate * 2 + 0] = (coordinate == 0 ? 1.0 : 0.0) + dx;
            jacobian[coordinate * 2 + 1] = (coordinate == 1 ? 1.0 : 0.0) + dy;
            TEST_NUMBERS_CLOSE(c.positions[3 * point + coordinate],
                               base[coordinate] + poly_value(coefficients[coordinate], xi, eta), 1e-12, 1e-10);
        }
        for (unsigned axis = 0; axis < 2; ++axis)
        {
            for (unsigned other = 0; other < 2; ++other)
            {
                for (unsigned coordinate = 0; coordinate < 3; ++coordinate)
                {
                    gram[axis * 2 + other] += jacobian[coordinate * 2 + axis] * jacobian[coordinate * 2 + other];
                }
            }
        }
        // Surface measure: sqrt(det(J^T J)).
        TEST_NUMBERS_CLOSE(c.determinant[point], sqrt(gram[0] * gram[3] - gram[1] * gram[1]), 1e-12, 1e-10);
        check_left_inverse(2, 3, jacobian, c.inverse_maps + (size_t)point * 2 * 3);
        advance_index(2, target_dims, t_idx);
    }

    case_destroy(&c);
}

int main(void)
{
    test_curved_face();
    test_two_dimensional_face();
    test_polynomial_exactness();
    printf("test_boundary_space_map_resample PASSED\n");
    return 0;
}
