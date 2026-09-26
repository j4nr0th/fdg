#include "../../src/basis/basis_lagrange.h"
#include "../../src/operations/boundaries.h"
#include "../common/common.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

enum
{
    MAX_TEST_DIM = 4,
    MAX_TEST_ORDER = 5,
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

/**
 * Independent contraction coefficient for the bases whose contraction is trivially known:
 * endpoint-node bases pick the endpoint degree of freedom and Legendre coefficients are the
 * endpoint values of the Legendre polynomials.
 */
static double reference_coefficient(const basis_spec_t basis, const bool at_the_end, const unsigned index)
{
    switch (basis.type)
    {
    case BASIS_BERNSTEIN:
    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        return (at_the_end ? index == basis.order : index == 0) ? 1.0 : 0.0;

    case BASIS_LEGENDRE:
        if (at_the_end)
            return 1.0;
        return index % 2 == 0 ? 1.0 : -1.0;

    default:
        TEST_ASSERTION(0, "Polynomial reproduction scenarios must be used for this basis type.");
        return 0.0;
    }
}

static void check_boundary_output(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                                  const double values[const restrict], const size_t value_count, const unsigned bdim,
                                  const int8_t orientation[const static MAX_TEST_DIM],
                                  const double expected[const restrict], const size_t expected_count)
{
    const size_t work_size = boundary_dof_values_work_size(ndim, basis);
    double *const work = malloc(work_size * sizeof(double));
    double *const out = malloc(expected_count * sizeof(double));
    double *const golden = malloc(value_count * sizeof(double));
    TEST_ASSERTION(work != NULL && out != NULL && golden != NULL, "Failed to allocate test buffers.");

    memcpy(golden, values, value_count * sizeof(double));
    // Sentinel fill to catch any read of untouched work memory.
    for (size_t i = 0; i < work_size; ++i)
        work[i] = 1234.5;

    boundary_dof_values(ndim, basis, values, work, bdim, orientation, out);

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
 * Reference contraction by direct multi-index evaluation: for every tensor position with all
 * fixed axes at index zero, the boundary value is the product over the fixed axes of the
 * coefficient-weighted sums along each of them.
 */
static void reference_contraction(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                                  const double values[const restrict], const unsigned ncontract,
                                  const int8_t orientation[const static MAX_TEST_DIM], double expected[restrict])
{
    unsigned fixed_axis[MAX_TEST_DIM];
    bool fixed_end[MAX_TEST_DIM];
    bool is_fixed[MAX_TEST_DIM] = {false};
    for (unsigned c = 0; c < ncontract; ++c)
    {
        const int8_t code = orientation[c];
        fixed_axis[c] = (unsigned)(code < 0 ? -code : code) - 1;
        fixed_end[c] = code > 0;
        is_fixed[fixed_axis[c]] = true;
    }

    size_t sizes[MAX_TEST_DIM], in_strides[MAX_TEST_DIM], out_strides[MAX_TEST_DIM];
    size_t in_total = 1, out_total = 1;
    for (unsigned axis = ndim; axis-- > 0;)
    {
        sizes[axis] = basis[axis].order + 1;
        in_strides[axis] = in_total;
        in_total *= sizes[axis];
        if (!is_fixed[axis])
        {
            out_strides[axis] = out_total;
            out_total *= sizes[axis];
        }
    }

    if (ncontract == 0)
    {
        memcpy(expected, values, in_total * sizeof(double));
        return;
    }

    unsigned idx[MAX_TEST_DIM] = {0};
    for (;;)
    {
        size_t in_flat = 0, out_flat = 0;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            in_flat += (size_t)idx[axis] * in_strides[axis];
            if (!is_fixed[axis])
                out_flat += (size_t)idx[axis] * out_strides[axis];
        }

        bool at_fixed_slice = true;
        for (unsigned c = 0; c < ncontract; ++c)
        {
            if (idx[fixed_axis[c]] != 0)
            {
                at_fixed_slice = false;
                break;
            }
        }
        if (at_fixed_slice)
        {
            // Sequential contraction over multiple axes is the weighted sum over all
            // combinations of the fixed axes' degrees of freedom.
            double acc = 0.0;
            unsigned fidx[MAX_TEST_DIM] = {0};
            bool done = false;
            while (!done)
            {
                size_t sample = in_flat;
                double weight = 1.0;
                for (unsigned c = 0; c < ncontract; ++c)
                {
                    sample += (size_t)fidx[c] * in_strides[fixed_axis[c]];
                    weight *= reference_coefficient(basis[fixed_axis[c]], fixed_end[c], fidx[c]);
                }
                acc += weight * values[sample];

                // Advance the fixed-axis odometer.
                bool carry = true;
                for (unsigned c = ncontract; c > 0 && carry; --c)
                {
                    if (++fidx[c - 1] < sizes[fixed_axis[c - 1]])
                        carry = false;
                    else
                        fidx[c - 1] = 0;
                }
                done = carry;
            }
            expected[out_flat] = acc;
        }

        // Advance the odometer, last axis fastest.
        unsigned axis = ndim;
        for (;;)
        {
            TEST_ASSERTION(axis > 0, "Odometer ran past the first axis.");
            axis -= 1;
            if (++idx[axis] < sizes[axis])
                break;
            idx[axis] = 0;
            if (axis == 0)
                return;
        }
    }
}

static void run_contraction_case(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                                 const unsigned bdim, const int8_t orientation[const static MAX_TEST_DIM],
                                 test_prng_t *rng)
{
    size_t in_total = 1, out_total = 1;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        in_total *= basis[axis].order + 1;
    }
    // Output count: product over the axes that are not fixed by the orientation prefix.
    {
        bool is_fixed[MAX_TEST_DIM] = {false};
        for (unsigned c = 0; c < ndim - bdim; ++c)
        {
            const unsigned axis = (unsigned)(orientation[c] < 0 ? -orientation[c] : orientation[c]) - 1;
            is_fixed[axis] = true;
        }
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (!is_fixed[axis])
                out_total *= basis[axis].order + 1;
        }
    }

    double *const values = malloc(in_total * sizeof(double));
    double *const expected = malloc(out_total * sizeof(double));
    TEST_ASSERTION(values != NULL && expected != NULL, "Failed to allocate test buffers.");

    test_fill_random(values, in_total, rng);
    reference_contraction(ndim, basis, values, ndim - bdim, orientation, expected);
    check_boundary_output(ndim, basis, values, in_total, bdim, orientation, expected, out_total);

    free(values);
    free(expected);
}

static void test_single_axis(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 20260917u);

    const basis_set_type_t trivial_types[] = {BASIS_BERNSTEIN, BASIS_LAGRANGE_UNIFORM, BASIS_LAGRANGE_GAUSS_LOBATTO};
    for (unsigned itype = 0; itype < sizeof(trivial_types) / sizeof(*trivial_types); ++itype)
    {
        for (unsigned order = 1; order <= 4; ++order)
        {
            const basis_spec_t basis[MAX_TEST_DIM] = {{trivial_types[itype], order}};
            const int8_t start[MAX_TEST_DIM] = {-1};
            const int8_t end[MAX_TEST_DIM] = {+1};
            run_contraction_case(1, basis, 0, start, &rng);
            run_contraction_case(1, basis, 0, end, &rng);
        }
    }

    // Legendre contracts to the alternating sum at the start and the plain sum at the end.
    for (unsigned order = 1; order <= 6; ++order)
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LEGENDRE, order}};
        const int8_t start[MAX_TEST_DIM] = {-1};
        const int8_t end[MAX_TEST_DIM] = {+1};
        run_contraction_case(1, basis, 0, start, &rng);
        run_contraction_case(1, basis, 0, end, &rng);
    }
}

static void test_multi_axis_selection(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 4242u);

    // 2D: contract each axis of a mixed element, both endpoints.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LAGRANGE_GAUSS_LOBATTO, 2}, {BASIS_LEGENDRE, 3}};
        const int8_t cases[][MAX_TEST_DIM] = {{-1, +2}, {+1, +2}, {+2, -1}};
        for (unsigned i = 0; i < 3; ++i)
        {
            run_contraction_case(2, basis, 1, cases[i], &rng);
        }
    }
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_BERNSTEIN, 1}, {BASIS_LAGRANGE_UNIFORM, 3}};
        const int8_t cases[][MAX_TEST_DIM] = {{-1, +2}, {+1, -2}};
        for (unsigned i = 0; i < 2; ++i)
        {
            run_contraction_case(2, basis, 1, cases[i], &rng);
        }
    }

    // 3D: one, two, and three fixed axes, including both prefix orders for the same pair.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {
            {BASIS_LEGENDRE, 2}, {BASIS_BERNSTEIN, 1}, {BASIS_LAGRANGE_GAUSS_LOBATTO, 3}};
        const int8_t one_axis[MAX_TEST_DIM] = {+2, +1, +3};
        run_contraction_case(3, basis, 2, one_axis, &rng);

        const int8_t two_axes_ascending[MAX_TEST_DIM] = {+1, +3, +2};
        const int8_t two_axes_descending[MAX_TEST_DIM] = {+3, +1, +2};
        run_contraction_case(3, basis, 1, two_axes_ascending, &rng);
        run_contraction_case(3, basis, 1, two_axes_descending, &rng);

        const int8_t all_axes[MAX_TEST_DIM] = {-1, +2, -3};
        run_contraction_case(3, basis, 0, all_axes, &rng);
    }

    // Zero-order axes act as the identity under contraction.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LAGRANGE_UNIFORM, 0}, {BASIS_LEGENDRE, 2}};
        const int8_t contract_zero[MAX_TEST_DIM] = {-1, +2};
        const int8_t contract_other[MAX_TEST_DIM] = {+1, -2};
        run_contraction_case(2, basis, 1, contract_zero, &rng);
        run_contraction_case(2, basis, 1, contract_other, &rng);
    }
}

/**
 * For Lagrange bases, degrees of freedom are values at the nodes; a degree `order` polynomial
 * must therefore be reproduced exactly, including at the endpoints for the node-interior
 * Gauss and Chebyshev-Gauss sets.
 */
static void run_polynomial_case(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                                const unsigned bdim, const int8_t orientation[const static MAX_TEST_DIM],
                                test_prng_t *rng)
{
    double poly[MAX_TEST_DIM][MAX_TEST_ORDER + 1];
    double nodes[MAX_TEST_DIM][MAX_TEST_ORDER + 1];
    size_t sizes[MAX_TEST_DIM], in_strides[MAX_TEST_DIM], out_strides[MAX_TEST_DIM];
    bool is_fixed[MAX_TEST_DIM] = {false};
    double fixed_endpoint[MAX_TEST_DIM] = {0.0};

    size_t in_total = 1, out_total = 1;
    for (unsigned c = 0; c < ndim - bdim; ++c)
    {
        const int8_t code = orientation[c];
        const unsigned axis = (unsigned)(code < 0 ? -code : code) - 1;
        is_fixed[axis] = true;
        fixed_endpoint[axis] = code > 0 ? 1.0 : -1.0;
    }
    for (unsigned axis = ndim; axis-- > 0;)
    {
        const unsigned order = basis[axis].order;
        TEST_ASSERTION(order <= MAX_TEST_ORDER, "Test order exceeds the supported maximum.");
        for (unsigned i = 0; i < order + 1; ++i)
        {
            poly[axis][i] = 2.0 * test_prng_next_double(rng) - 1.0;
        }
        TEST_FDG_RESULT(generate_lagrange_roots(order, basis[axis].type, nodes[axis]));
        sizes[axis] = order + 1;
        in_strides[axis] = in_total;
        in_total *= sizes[axis];
        if (!is_fixed[axis])
        {
            out_strides[axis] = out_total;
            out_total *= sizes[axis];
        }
    }

    double *const values = malloc(in_total * sizeof(double));
    double *const expected = malloc(out_total * sizeof(double));
    TEST_ASSERTION(values != NULL && expected != NULL, "Failed to allocate test buffers.");

    unsigned idx[MAX_TEST_DIM] = {0};
    for (;;)
    {
        size_t in_flat = 0, out_flat = 0;
        double value = 1.0, boundary_value = 1.0;
        bool at_fixed_slice = true;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            in_flat += (size_t)idx[axis] * in_strides[axis];
            const double node_value = eval_polynomial(poly[axis], basis[axis].order, nodes[axis][idx[axis]]);
            value *= node_value;
            if (is_fixed[axis])
            {
                at_fixed_slice = at_fixed_slice && idx[axis] == 0;
                boundary_value *= eval_polynomial(poly[axis], basis[axis].order, fixed_endpoint[axis]);
            }
            else
            {
                out_flat += (size_t)idx[axis] * out_strides[axis];
                // Free axes survive the contraction at their node values.
                boundary_value *= node_value;
            }
        }
        values[in_flat] = value;
        if (at_fixed_slice)
            expected[out_flat] = boundary_value;

        unsigned axis = ndim;
        for (;;)
        {
            TEST_ASSERTION(axis > 0, "Odometer ran past the first axis.");
            axis -= 1;
            if (++idx[axis] < sizes[axis])
                break;
            idx[axis] = 0;
            if (axis == 0)
                goto filled;
        }
    }
filled:

    check_boundary_output(ndim, basis, values, in_total, bdim, orientation, expected, out_total);

    free(values);
    free(expected);
}

static void test_polynomial_reproduction(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 90210u);

    // 2D: Gauss along the contracted axis, Chebyshev-Gauss free, both endpoints.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LAGRANGE_GAUSS, 3}, {BASIS_LAGRANGE_CHEBYSHEV_GAUSS, 4}};
        const int8_t start[MAX_TEST_DIM] = {-1, +2};
        const int8_t end[MAX_TEST_DIM] = {+1, +2};
        run_polynomial_case(2, basis, 1, start, &rng);
        run_polynomial_case(2, basis, 1, end, &rng);
    }
    // 3D: mixed node-interior bases with two fixed axes.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {
            {BASIS_LAGRANGE_GAUSS, 2}, {BASIS_LAGRANGE_CHEBYSHEV_GAUSS, 3}, {BASIS_LAGRANGE_GAUSS, 4}};
        const int8_t two_fixed[MAX_TEST_DIM] = {-1, -3, +2};
        run_polynomial_case(3, basis, 1, two_fixed, &rng);
        const int8_t one_fixed[MAX_TEST_DIM] = {+3, +1, +2};
        run_polynomial_case(3, basis, 2, one_fixed, &rng);
    }
    // Node-including Lagrange sets follow the same identity; cross-checks the selection path.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LAGRANGE_UNIFORM, 3}, {BASIS_LAGRANGE_GAUSS_LOBATTO, 2}};
        const int8_t orientation[MAX_TEST_DIM] = {-2, +1};
        run_polynomial_case(2, basis, 1, orientation, &rng);
    }
    // Zero-order Gauss axis.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LAGRANGE_GAUSS, 0}, {BASIS_LAGRANGE_GAUSS, 2}};
        const int8_t contract_zero[MAX_TEST_DIM] = {+1, -2};
        run_polynomial_case(2, basis, 1, contract_zero, &rng);
    }
}

int main()
{
    test_single_axis();
    test_multi_axis_selection();
    test_polynomial_reproduction();
    return 0;
}
