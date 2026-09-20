#include "../../src/integration/gauss_legendre.h"
#include "../../src/integration/gauss_lobatto.h"
#include "../../src/operations/boundaries.h"
#include "../../src/operations/dof_transforms.h"
#include "../common/common.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum
{
    MAX_TEST_DIM = 4,
    MAX_TEST_ORDER = 5,
    NODE_ITERATIONS = 100,
    NODE_TOLERANCE_DIGITS = 12,
};

// Pascal triangle of binomial coefficients for the Bernstein reference evaluation.
static const double BINOMIAL[MAX_TEST_ORDER + 1][MAX_TEST_ORDER + 1] = {
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 2.0, 1.0, 0.0, 0.0, 0.0},
    {1.0, 3.0, 3.0, 1.0, 0.0, 0.0}, {1.0, 4.0, 6.0, 4.0, 1.0, 0.0}, {1.0, 5.0, 10.0, 10.0, 5.0, 1.0},
};

static void fill_random(double values[restrict], const size_t count, test_prng_t *rng)
{
    for (size_t i = 0; i < count; ++i)
    {
        values[i] = 2.0 * test_prng_next_double(rng) - 1.0;
    }
}

static double integer_power(const double base, const unsigned exponent)
{
    double result = 1.0;
    for (unsigned i = 0; i < exponent; ++i)
    {
        result *= base;
    }
    return result;
}

/** Nodes of the Lagrange families in ascending order, matching the basis set node layout. */
static void lagrange_nodes(const basis_set_type_t type, const unsigned order,
                           double nodes[const static MAX_TEST_ORDER + 1])
{
    switch (type)
    {
    case BASIS_LAGRANGE_UNIFORM:
        for (unsigned i = 0; i < order + 1; ++i)
        {
            nodes[i] = (2.0 * i) / (double)(order)-1.0;
        }
        break;

    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        TEST_ASSERTION(gauss_lobatto_nodes(order + 1, 1e-12, NODE_ITERATIONS, nodes) == 0,
                       "Gauss-Lobatto node iteration did not converge.");
        break;

    case BASIS_LAGRANGE_GAUSS:
        TEST_ASSERTION(gauss_legendre_nodes(order + 1, 1e-12, NODE_ITERATIONS, nodes) == 0,
                       "Gauss-Legendre node iteration did not converge.");
        break;

    default:
        TEST_ASSERTION(0, "Not a Lagrange basis type.");
        break;
    }
}

/**
 * Value of the `index`-th basis function of a 1D basis at `x`, evaluated independently of the
 * basis set machinery: node products for the Lagrange families, the three term recurrence for
 * Legendre, and the Bernstein formula with integer binomials.
 */
static double reference_basis_value(const basis_spec_t basis, const unsigned index, const double x)
{
    switch (basis.type)
    {
    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_GAUSS: {
        double nodes[MAX_TEST_ORDER + 1];
        lagrange_nodes(basis.type, basis.order, nodes);
        double value = 1.0;
        for (unsigned j = 0; j <= basis.order; ++j)
        {
            if (j != index)
            {
                value *= (x - nodes[j]) / (nodes[index] - nodes[j]);
            }
        }
        return value;
    }

    case BASIS_LEGENDRE: {
        if (index == 0)
        {
            return 1.0;
        }
        double previous = 1.0, current = x;
        for (unsigned k = 2; k <= index; ++k)
        {
            const double next = ((2 * k - 1) * x * current - (k - 1) * previous) / (double)k;
            previous = current;
            current = next;
        }
        return current;
    }

    case BASIS_BERNSTEIN:
        return BINOMIAL[basis.order][index] * integer_power((1.0 - x) / 2.0, basis.order - index) *
               integer_power((x + 1.0) / 2.0, index);

    default:
        TEST_ASSERTION(0, "Unsupported basis type.");
        return 0.0;
    }
}

/**
 * Reference plane contraction by direct evaluation: for every surviving position the
 * reference basis values at the plane are summed along the contracted axis only, with all
 * other axes held at their position.
 */
static void reference_plane_values(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                                   const double values[const restrict], const unsigned axis, const double plane,
                                   double expected[restrict])
{
    size_t pre_count = 1, post_count = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        if (i < axis)
        {
            pre_count *= basis[i].order + 1;
        }
        else if (i > axis)
        {
            post_count *= basis[i].order + 1;
        }
    }
    const unsigned ndofs = basis[axis].order + 1;

    double coeffs[MAX_TEST_ORDER + 1];
    for (unsigned i = 0; i < ndofs; ++i)
    {
        coeffs[i] = reference_basis_value(basis[axis], i, plane);
    }

    for (size_t i_pre = 0; i_pre < pre_count; ++i_pre)
    {
        for (size_t i_post = 0; i_post < post_count; ++i_post)
        {
            double result = 0.0;
            for (unsigned i_dof = 0; i_dof < ndofs; ++i_dof)
            {
                result += coeffs[i_dof] * values[(i_pre * ndofs + i_dof) * post_count + i_post];
            }
            expected[i_pre * post_count + i_post] = result;
        }
    }
}

static void run_plane_case(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                           const unsigned axis, const double plane, test_prng_t *rng)
{
    size_t total = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        total *= basis[i].order + 1;
    }
    const size_t out_count = total / (basis[axis].order + 1);

    double *const values = malloc(total * sizeof(double));
    double *const golden = malloc(total * sizeof(double));
    double *const expected = malloc(out_count * sizeof(double));
    double *const out = malloc(out_count * sizeof(double));
    TEST_ASSERTION(values != NULL && golden != NULL && expected != NULL && out != NULL,
                   "Failed to allocate test buffers.");

    fill_random(values, total, rng);
    memcpy(golden, values, total * sizeof(double));
    reference_plane_values(ndim, basis, values, axis, plane, expected);

    const size_t work_size = dof_plane_values_work_size(ndim, basis, axis);
    double *const work = malloc(work_size * sizeof(double));
    TEST_ASSERTION(work != NULL, "Failed to allocate test work buffer.");
    // Sentinel fill to catch any read of untouched work memory.
    for (size_t i = 0; i < work_size; ++i)
        work[i] = 1234.5;

    dof_plane_values(ndim, basis, values, work, axis, plane, out);

    for (size_t i = 0; i < out_count; ++i)
    {
        TEST_NUMBERS_CLOSE(out[i], expected[i], 1e-12, 1e-10);
    }
    // The input must remain untouched.
    for (size_t i = 0; i < total; ++i)
    {
        TEST_NUMBERS_CLOSE(values[i], golden[i], 0.0, 0.0);
    }

    // At the interval endpoints the plane contraction must agree with the boundary
    // contraction bit for bit: both evaluate the same coefficients and walk them in the same
    // ascending order.
    if (plane == -1.0 || plane == +1.0)
    {
        int8_t orientation[MAX_TEST_DIM] = {0};
        orientation[0] = (int8_t)((plane > 0.0 ? 1 : -1) * (int)(axis + 1));
        unsigned entry = 1;
        for (unsigned i = 0; i < ndim; ++i)
        {
            if (i != axis)
            {
                orientation[entry] = (int8_t)(i + 1);
                entry += 1;
            }
        }

        const size_t boundary_work_size = boundary_dof_values_work_size(ndim, basis);
        double *const boundary_work = malloc(boundary_work_size * sizeof(double));
        double *const boundary_out = malloc(out_count * sizeof(double));
        TEST_ASSERTION(boundary_work != NULL && boundary_out != NULL, "Failed to allocate boundary buffers.");

        boundary_dof_values(ndim, basis, values, boundary_work, ndim - 1, orientation, boundary_out);

        // Which families must agree bit for bit: Lagrange-Gauss shares the identical
        // coefficient evaluation path on both sides, Bernstein evaluates to exact endpoint
        // indicators, and the Legendre coefficients are exactly one at the end of the axis.
        // The remaining node based bases pick the endpoint degree of freedom on the boundary
        // side, while the plane contraction divides by the node denominators before applying
        // the remaining factors, so its endpoint values sit a rounding step away from the
        // pick. The Legendre start sum additionally groups the even and odd degrees of
        // freedom separately while the plane contraction walks them interleaved; both round
        // equally validly, so those cases compare with the standard tolerance instead.
        const bool exact = !(basis[axis].type == BASIS_LEGENDRE && plane < 0.0) &&
                           basis[axis].type != BASIS_LAGRANGE_UNIFORM &&
                           basis[axis].type != BASIS_LAGRANGE_GAUSS_LOBATTO;
        for (size_t i = 0; i < out_count; ++i)
        {
            if (exact)
            {
                TEST_NUMBERS_CLOSE(out[i], boundary_out[i], 0.0, 0.0);
            }
            else
            {
                TEST_NUMBERS_CLOSE(out[i], boundary_out[i], 1e-12, 1e-10);
            }
        }

        free(boundary_work);
        free(boundary_out);
    }

    free(work);
    free(values);
    free(golden);
    free(expected);
    free(out);
}

static void run_reversal_case(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                              const unsigned axis, test_prng_t *rng)
{
    static const double planes[] = {-0.5, 0.25, 1.0};

    size_t total = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        total *= basis[i].order + 1;
    }

    double *const values = malloc(total * sizeof(double));
    double *const reversed = malloc(total * sizeof(double));
    double *const restored = malloc(total * sizeof(double));
    TEST_ASSERTION(values != NULL && reversed != NULL && restored != NULL, "Failed to allocate test buffers.");
    fill_random(values, total, rng);

    dof_reverse_orientation_values(ndim, basis, values, axis, reversed);

    // (a) A single-axis uniform Lagrange element must mirror its nodes directly.
    if (ndim == 1 && basis[axis].type == BASIS_LAGRANGE_UNIFORM)
    {
        const unsigned ndofs = basis[axis].order + 1;
        for (unsigned i = 0; i < ndofs; ++i)
        {
            TEST_NUMBERS_CLOSE(reversed[i], values[ndofs - 1 - i], 0.0, 0.0);
        }
    }

    // (b) The reversed tensor evaluated at x is the original tensor evaluated at -x.
    const size_t work_size = dof_plane_values_work_size(ndim, basis, axis);
    const size_t out_count = total / (basis[axis].order + 1);
    double *const work = malloc(work_size * sizeof(double));
    double *const trace_reversed = malloc(out_count * sizeof(double));
    double *const trace_original = malloc(out_count * sizeof(double));
    TEST_ASSERTION(work != NULL && trace_reversed != NULL && trace_original != NULL,
                   "Failed to allocate test buffers.");

    for (unsigned c = 0; c < sizeof(planes) / sizeof(*planes); ++c)
    {
        dof_plane_values(ndim, basis, reversed, work, axis, planes[c], trace_reversed);
        dof_plane_values(ndim, basis, values, work, axis, -planes[c], trace_original);
        // Legendre coefficients only pick up an exact sign, so the two contractions walk
        // identical terms in identical order. The mirror families reverse the order of the
        // summed terms (and the Lagrange node sets are not exactly antisymmetric in floating
        // point), so their traces may differ in the last places.
        const bool exact = basis[axis].type == BASIS_LEGENDRE;
        for (size_t i = 0; i < out_count; ++i)
        {
            if (exact)
            {
                TEST_NUMBERS_CLOSE(trace_reversed[i], trace_original[i], 0.0, 0.0);
            }
            else
            {
                TEST_NUMBERS_CLOSE(trace_reversed[i], trace_original[i], 1e-12, 1e-10);
            }
        }
    }

    // (c) Double reversal restores the original tensor exactly.
    dof_reverse_orientation_values(ndim, basis, reversed, axis, restored);
    for (size_t i = 0; i < total; ++i)
    {
        TEST_NUMBERS_CLOSE(restored[i], values[i], 0.0, 0.0);
    }

    free(work);
    free(trace_reversed);
    free(trace_original);
    free(values);
    free(reversed);
    free(restored);
}

static void test_plane_values(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 20260917u);

    const basis_set_type_t types[] = {
        BASIS_LAGRANGE_UNIFORM, BASIS_LAGRANGE_GAUSS_LOBATTO, BASIS_LAGRANGE_GAUSS, BASIS_LEGENDRE, BASIS_BERNSTEIN,
    };
    const double planes[] = {-1.0, -0.5, 0.25, 1.0};

    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            for (unsigned pattern = 0; pattern < sizeof(types) / sizeof(*types); ++pattern)
            {
                for (unsigned shift = 0; shift <= MAX_TEST_ORDER; ++shift)
                {
                    basis_spec_t basis[MAX_TEST_DIM];
                    for (unsigned i = 0; i < ndim; ++i)
                    {
                        basis[i].type = types[(pattern + i) % (sizeof(types) / sizeof(*types))];
                        basis[i].order = (shift + pattern + 2 * i) % (MAX_TEST_ORDER + 1);
                    }
                    for (unsigned p = 0; p < sizeof(planes) / sizeof(*planes); ++p)
                    {
                        run_plane_case(ndim, basis, axis, planes[p], &rng);
                    }
                }
            }
        }
    }
}

static void test_reverse_orientation(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 424242u);

    const basis_set_type_t types[] = {
        BASIS_LAGRANGE_UNIFORM, BASIS_LAGRANGE_GAUSS_LOBATTO, BASIS_LAGRANGE_GAUSS, BASIS_LEGENDRE, BASIS_BERNSTEIN,
    };

    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            for (unsigned pattern = 0; pattern < sizeof(types) / sizeof(*types); ++pattern)
            {
                for (unsigned shift = 0; shift <= MAX_TEST_ORDER; ++shift)
                {
                    basis_spec_t basis[MAX_TEST_DIM];
                    for (unsigned i = 0; i < ndim; ++i)
                    {
                        basis[i].type = types[(pattern + i) % (sizeof(types) / sizeof(*types))];
                        basis[i].order = (shift + pattern + 2 * i) % (MAX_TEST_ORDER + 1);
                    }
                    run_reversal_case(ndim, basis, axis, &rng);
                }
            }
        }
    }
}

int main()
{
    test_plane_values();
    test_reverse_orientation();
    printf("test_dof_transforms PASSED\n");
    return 0;
}
