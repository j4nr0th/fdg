#include "../../src/integration/gauss_legendre.h"
#include "../../src/integration/gauss_lobatto.h"
#include "../../src/operations/boundaries.h"
#include "../common/common.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

enum
{
    MAX_TEST_DIM = 4,
    MAX_TEST_ORDER = 5,
    NODE_ITERATIONS = 100,
    NODE_TOLERANCE_DIGITS = 12,
};

static double eval_polynomial(const double coeffs[const static MAX_TEST_ORDER + 1], const unsigned order,
                              const double x)
{
    double value = 0.0;
    for (unsigned i = order + 1; i > 0; --i)
    {
        value = value * x + coeffs[i - 1];
    }
    return value;
}

static void fill_random(double values[restrict], const size_t count, test_prng_t *rng)
{
    for (size_t i = 0; i < count; ++i)
    {
        values[i] = 2.0 * test_prng_next_double(rng) - 1.0;
    }
}

static void check_boundary_output(const unsigned ndim, const integration_spec_t specs[const static MAX_TEST_DIM],
                                  const double values[const restrict], const size_t value_count, const unsigned bdim,
                                  const int8_t orientation[const static MAX_TEST_DIM], const unsigned n_components,
                                  const double expected[const restrict], const size_t expected_count)
{
    const size_t work_size = boundary_integration_point_values_work_size(ndim, specs, n_components);
    double *const work = malloc(work_size * sizeof(double));
    double *const out = malloc(expected_count * sizeof(double));
    double *const golden = malloc(value_count * sizeof(double));
    TEST_ASSERTION(work != NULL && out != NULL && golden != NULL, "Failed to allocate test buffers.");

    memcpy(golden, values, value_count * sizeof(double));
    // Sentinel fill to catch any read of untouched work memory.
    for (size_t i = 0; i < work_size; ++i)
        work[i] = 1234.5;

    boundary_integration_point_values(ndim, specs, values, work, bdim, orientation, n_components, out);

    for (size_t i = 0; i < expected_count; ++i)
    {
        TEST_NUMBERS_CLOSE(out[i], expected[i], 1e-12, 1e-10);
    }
    // The input must remain untouched.
    for (size_t i = 0; i < value_count; ++i)
    {
        TEST_NUMBERS_CLOSE(values[i], golden[i], 0.0, 0.0);
    }

    free(work);
    free(out);
    free(golden);
}

/**
 * Reference extraction for Gauss-Lobatto axes: the boundary points are the slice at the
 * endpoint index of every fixed axis, with all components carried over unchanged.
 */
static void run_lobatto_case(const unsigned ndim, const integration_spec_t specs[const static MAX_TEST_DIM],
                             const unsigned bdim, const int8_t orientation[const static MAX_TEST_DIM],
                             const unsigned n_components, test_prng_t *rng)
{
    size_t sizes[MAX_TEST_DIM], in_strides[MAX_TEST_DIM], out_strides[MAX_TEST_DIM];
    bool is_fixed[MAX_TEST_DIM] = {false};
    bool fixed_end[MAX_TEST_DIM] = {false};
    for (unsigned c = 0; c < ndim - bdim; ++c)
    {
        const int8_t code = orientation[c];
        const unsigned axis = (unsigned)(code < 0 ? -code : code) - 1;
        is_fixed[axis] = true;
        fixed_end[axis] = code > 0;
    }

    size_t in_total = 1, out_total = 1;
    for (unsigned axis = ndim; axis-- > 0;)
    {
        sizes[axis] = specs[axis].order + 1;
        in_strides[axis] = in_total;
        in_total *= sizes[axis];
        if (!is_fixed[axis])
        {
            out_strides[axis] = out_total;
            out_total *= sizes[axis];
        }
    }

    const size_t value_count = n_components * in_total;
    const size_t expected_count = n_components * out_total;
    double *const values = malloc(value_count * sizeof(double));
    double *const expected = malloc(expected_count * sizeof(double));
    TEST_ASSERTION(values != NULL && expected != NULL, "Failed to allocate test buffers.");
    fill_random(values, value_count, rng);

    // Enumerate the surviving points directly; the fixed axes contribute their endpoint index.
    unsigned idx[MAX_TEST_DIM] = {0};
    for (;;)
    {
        size_t in_flat = 0, out_flat = 0;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (is_fixed[axis])
            {
                in_flat += (size_t)(fixed_end[axis] ? sizes[axis] - 1 : 0) * in_strides[axis];
            }
            else
            {
                in_flat += (size_t)idx[axis] * in_strides[axis];
                out_flat += (size_t)idx[axis] * out_strides[axis];
            }
        }
        for (unsigned component = 0; component < n_components; ++component)
        {
            expected[out_flat * n_components + component] = values[in_flat * n_components + component];
        }

        // Advance the odometer over the free axes, last axis fastest.
        bool done = true;
        for (unsigned axis = ndim; axis-- > 0;)
        {
            if (is_fixed[axis])
                continue;
            if (++idx[axis] < sizes[axis])
            {
                done = false;
                break;
            }
            idx[axis] = 0;
        }
        if (done)
            break;
    }

    check_boundary_output(ndim, specs, values, value_count, bdim, orientation, n_components, expected, expected_count);

    free(values);
    free(expected);
}

/**
 * Reference extraction with polynomial reproduction: the point values are products of
 * per-axis random polynomials evaluated at the axis nodes, so a fixed Gauss-Legendre axis
 * must evaluate its interpolant at the endpoint exactly, while fixed Gauss-Lobatto axes
 * and free axes keep their node values.
 */
static void run_point_polynomial_case(const unsigned ndim, const integration_spec_t specs[const static MAX_TEST_DIM],
                                      const unsigned bdim, const int8_t orientation[const static MAX_TEST_DIM],
                                      const unsigned n_components, test_prng_t *rng)
{
    double poly[MAX_TEST_DIM][MAX_TEST_ORDER + 1];
    double nodes[MAX_TEST_DIM][MAX_TEST_ORDER + 1];
    size_t sizes[MAX_TEST_DIM], in_strides[MAX_TEST_DIM], out_strides[MAX_TEST_DIM];
    bool is_fixed[MAX_TEST_DIM] = {false};
    bool fixed_end[MAX_TEST_DIM] = {false};
    double fixed_endpoint[MAX_TEST_DIM] = {0.0};
    for (unsigned c = 0; c < ndim - bdim; ++c)
    {
        const int8_t code = orientation[c];
        const unsigned axis = (unsigned)(code < 0 ? -code : code) - 1;
        is_fixed[axis] = true;
        fixed_end[axis] = code > 0;
        fixed_endpoint[axis] = code > 0 ? 1.0 : -1.0;
    }

    size_t in_total = 1, out_total = 1;
    for (unsigned axis = ndim; axis-- > 0;)
    {
        const unsigned order = specs[axis].order;
        TEST_ASSERTION(order <= MAX_TEST_ORDER, "Test order exceeds the supported maximum.");
        for (unsigned i = 0; i < order + 1; ++i)
        {
            poly[axis][i] = 2.0 * test_prng_next_double(rng) - 1.0;
        }
        const int unconverged = specs[axis].type == INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE
                                    ? gauss_legendre_nodes(order + 1, 1e-12, NODE_ITERATIONS, nodes[axis])
                                    : gauss_lobatto_nodes(order + 1, 1e-12, NODE_ITERATIONS, nodes[axis]);
        TEST_ASSERTION(unconverged == 0, "Node iteration did not converge.");
        sizes[axis] = order + 1;
        in_strides[axis] = in_total;
        in_total *= sizes[axis];
        if (!is_fixed[axis])
        {
            out_strides[axis] = out_total;
            out_total *= sizes[axis];
        }
    }

    const size_t value_count = n_components * in_total;
    const size_t expected_count = n_components * out_total;
    double *const values = malloc(value_count * sizeof(double));
    double *const expected = malloc(expected_count * sizeof(double));
    TEST_ASSERTION(values != NULL && expected != NULL, "Failed to allocate test buffers.");

    unsigned idx[MAX_TEST_DIM] = {0};
    for (;;)
    {
        size_t in_flat = 0, out_flat = 0;
        double value = 1.0, boundary_value = 1.0;
        bool at_boundary_slice = true;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            in_flat += (size_t)idx[axis] * in_strides[axis];
            const double node_value = eval_polynomial(poly[axis], specs[axis].order, nodes[axis][idx[axis]]);
            value *= node_value;
            if (is_fixed[axis])
            {
                at_boundary_slice = at_boundary_slice && idx[axis] == (fixed_end[axis] ? sizes[axis] - 1 : 0);
                boundary_value *= eval_polynomial(poly[axis], specs[axis].order, fixed_endpoint[axis]);
            }
            else
            {
                out_flat += (size_t)idx[axis] * out_strides[axis];
                // Free axes survive the extraction at their node values.
                boundary_value *= node_value;
            }
        }
        for (unsigned component = 0; component < n_components; ++component)
        {
            const double scale = (double)(component + 1);
            values[in_flat * n_components + component] = scale * value;
            if (at_boundary_slice)
            {
                expected[out_flat * n_components + component] = scale * boundary_value;
            }
        }

        // Advance the odometer over every axis, last axis fastest.
        bool done = true;
        for (unsigned axis = ndim; axis-- > 0;)
        {
            if (++idx[axis] < sizes[axis])
            {
                done = false;
                break;
            }
            idx[axis] = 0;
        }
        if (done)
            break;
    }

    check_boundary_output(ndim, specs, values, value_count, bdim, orientation, n_components, expected, expected_count);

    free(values);
    free(expected);
}

static void test_lobatto_selection(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 77001u);

    // 2D: contract each axis of a Lobatto element, both endpoints.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 2},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 3}};
        const int8_t cases[][MAX_TEST_DIM] = {{-1, +2}, {+1, +2}, {+2, -1}, {+2, +1}};
        for (unsigned i = 0; i < 4; ++i)
        {
            run_lobatto_case(2, specs, 1, cases[i], 1, &rng);
        }
    }

    // 3D: one and two fixed axes, both prefix orders for the same pair, with components.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 1},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 3},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 2}};
        const int8_t ascending[MAX_TEST_DIM] = {+1, +3, +2};
        const int8_t descending[MAX_TEST_DIM] = {+3, +1, +2};
        run_lobatto_case(3, specs, 1, ascending, 1, &rng);
        run_lobatto_case(3, specs, 1, descending, 3, &rng);

        const int8_t one_axis[MAX_TEST_DIM] = {+2, +1, +3};
        run_lobatto_case(3, specs, 2, one_axis, 2, &rng);

        const int8_t all_axes[MAX_TEST_DIM] = {-1, +2, -3};
        run_lobatto_case(3, specs, 0, all_axes, 1, &rng);
    }

    // Zero-order axes hold a single point, which both endpoints select.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 0},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 2}};
        const int8_t contract_zero[MAX_TEST_DIM] = {-1, +2};
        run_lobatto_case(2, specs, 1, contract_zero, 2, &rng);
    }

    // No fixed axes: the values are carried over unchanged, components included.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 1},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 2}};
        const int8_t none[MAX_TEST_DIM] = {+1, +2};
        run_lobatto_case(2, specs, 2, none, 3, &rng);
    }
}

static void test_gauss_interpolation(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 77002u);

    // 2D: Gauss axis contracted onto both endpoints, Lobatto axis free.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 3},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 2}};
        const int8_t start[MAX_TEST_DIM] = {-1, +2};
        const int8_t end[MAX_TEST_DIM] = {+1, +2};
        run_point_polynomial_case(2, specs, 1, start, 1, &rng);
        run_point_polynomial_case(2, specs, 1, end, 2, &rng);
    }

    // 3D: two fixed axes of mixed rule types, both prefix orders.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 2},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, 1},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 3}};
        const int8_t ascending[MAX_TEST_DIM] = {-1, -3, +2};
        const int8_t descending[MAX_TEST_DIM] = {-3, -1, +2};
        run_point_polynomial_case(3, specs, 1, ascending, 1, &rng);
        run_point_polynomial_case(3, specs, 1, descending, 3, &rng);

        const int8_t one_fixed[MAX_TEST_DIM] = {+3, +1, +2};
        run_point_polynomial_case(3, specs, 2, one_fixed, 2, &rng);
    }

    // Zero-order Gauss axis: its single point is the constant interpolant at both endpoints.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 0},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 2}};
        const int8_t contract_zero[MAX_TEST_DIM] = {+1, -2};
        run_point_polynomial_case(2, specs, 1, contract_zero, 2, &rng);
    }

    // No fixed axes: polynomial values carried over unchanged.
    {
        const integration_spec_t specs[MAX_TEST_DIM] = {{INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 2},
                                                        {INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, 1}};
        const int8_t none[MAX_TEST_DIM] = {+1, +2};
        run_point_polynomial_case(2, specs, 2, none, 1, &rng);
    }
}

int main()
{
    test_lobatto_selection();
    test_gauss_interpolation();
    return 0;
}
