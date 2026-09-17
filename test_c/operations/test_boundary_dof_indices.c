#include "../../src/operations/boundaries.h"
#include "../common/common.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

enum
{
    MAX_TEST_DIM = 4,
    MAX_TEST_ORDER = 5,
};

static void fill_random(double values[restrict], const size_t count, test_prng_t *rng)
{
    for (size_t i = 0; i < count; ++i)
    {
        values[i] = 2.0 * test_prng_next_double(rng) - 1.0;
    }
}

/**
 * Independent reference: walks the varying axes with a plain odometer, ascending axes with
 * the last axis fastest, stepping each axis in its orientation direction.
 */
static void reference_indices(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM],
                              const unsigned bdim, const int8_t orientation[const static MAX_TEST_DIM],
                              size_t expected[restrict])
{
    bool is_fixed[MAX_TEST_DIM] = {false};
    for (unsigned entry = 0; entry < ndim - bdim; ++entry)
    {
        const int8_t code = orientation[entry];
        is_fixed[(unsigned)(code < 0 ? -code : code) - 1] = true;
    }

    size_t strides[MAX_TEST_DIM];
    size_t stride = 1;
    for (unsigned axis = ndim; axis-- > 0;)
    {
        strides[axis] = stride;
        stride *= basis[axis].order + 1;
    }

    size_t base = 0;
    for (unsigned entry = 0; entry < ndim - bdim; ++entry)
    {
        const int8_t code = orientation[entry];
        const unsigned axis = (unsigned)(code < 0 ? -code : code) - 1;
        const unsigned size = (unsigned)(basis[axis].order + 1);
        base += (size_t)(code > 0 ? size - 1 : 0) * strides[axis];
    }
    size_t total = 1;
    unsigned free_axis[MAX_TEST_DIM], start[MAX_TEST_DIM], end[MAX_TEST_DIM], idx[MAX_TEST_DIM];
    int8_t direction[MAX_TEST_DIM];
    unsigned n_free = 0;
    for (unsigned entry = ndim - bdim; entry < ndim; ++entry)
    {
        const int8_t code = orientation[entry];
        const unsigned axis = (unsigned)(code < 0 ? -code : code) - 1;
        const unsigned size = (unsigned)(basis[axis].order + 1);
        free_axis[n_free] = axis;
        direction[n_free] = code > 0 ? 1 : -1;
        start[n_free] = code > 0 ? 0 : size - 1;
        end[n_free] = code > 0 ? size - 1 : 0;
        idx[n_free] = start[n_free];
        total *= size;
        ++n_free;
    }

    for (size_t k = 0; k < total; ++k)
    {
        size_t flat = base;
        for (unsigned slot = 0; slot < n_free; ++slot)
        {
            flat += (size_t)idx[slot] * strides[free_axis[slot]];
        }
        expected[k] = flat;

        // Advance the odometer, the last varying axis fastest.
        for (unsigned slot = n_free; slot-- > 0;)
        {
            if (idx[slot] != end[slot])
            {
                idx[slot] = (unsigned)(idx[slot] + direction[slot]);
                goto advanced;
            }
            idx[slot] = start[slot];
        }
    advanced:;
    }
}

static void run_case(const unsigned ndim, const basis_spec_t basis[const static MAX_TEST_DIM], const unsigned bdim,
                     const int8_t orientation[const static MAX_TEST_DIM], test_prng_t *rng)
{
    bool is_fixed[MAX_TEST_DIM] = {false};
    for (unsigned entry = 0; entry < ndim - bdim; ++entry)
    {
        const int8_t code = orientation[entry];
        is_fixed[(unsigned)(code < 0 ? -code : code) - 1] = true;
    }
    size_t element_count = 1, boundary_count = 1;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        element_count *= basis[axis].order + 1;
        if (!is_fixed[axis])
        {
            boundary_count *= basis[axis].order + 1;
        }
    }

    size_t *const expected = malloc(boundary_count * sizeof(size_t));
    size_t *const indices = malloc(boundary_count * sizeof(size_t));
    double *const values = malloc(element_count * sizeof(double));
    double *const out = malloc(boundary_count * sizeof(double));
    const size_t work_size = boundary_dof_values_work_size(ndim, basis);
    double *const work = malloc(work_size * sizeof(double));
    TEST_ASSERTION(expected != NULL && indices != NULL && values != NULL && out != NULL && work != NULL,
                   "Failed to allocate test buffers.");

    reference_indices(ndim, basis, bdim, orientation, expected);

    // The one-shot fill reports exactly the reference indices.
    _Alignas(max_align_t) uint8_t fill_work[256];
    TEST_ASSERTION(boundary_dof_iterator_data_size(bdim) <= sizeof(fill_work), "Fill work buffer too small.");
    TEST_ASSERTION(boundary_dof_indices(ndim, basis, bdim, orientation, fill_work, indices) == boundary_count,
                   "boundary_dof_indices returned the wrong count.");
    for (size_t k = 0; k < boundary_count; ++k)
    {
        TEST_ASSERTION(indices[k] == expected[k], "Filled boundary index does not match the reference.");
        TEST_ASSERTION(indices[k] < element_count, "Boundary index escapes the element tensor.");
    }

    // The iterator visits the same indices and exhausts idempotently. It only references the
    // caller-side basis and orientation, which stay alive and unchanged for the whole case.
    _Alignas(max_align_t) uint8_t iter_buffer[256];
    TEST_ASSERTION(boundary_dof_iterator_data_size(bdim) <= sizeof(iter_buffer), "Iterator buffer too small.");
    boundary_dof_iterator_t *const iter = (boundary_dof_iterator_t *)iter_buffer;
    boundary_dof_iterator_init(iter, ndim, basis, bdim, orientation);
    TEST_ASSERTION(boundary_dof_iterator_count(iter) == boundary_count, "Iterator reported the wrong count.");
    for (size_t k = 0; k < boundary_count; ++k)
    {
        TEST_ASSERTION(boundary_dof_iterator_index(iter) == expected[k],
                       "Iterator index does not match the reference.");
        if (k + 1 < boundary_count)
        {
            TEST_ASSERTION(boundary_dof_iterator_next(iter) == 1, "Iterator exhausted early.");
        }
    }
    TEST_ASSERTION(boundary_dof_iterator_next(iter) == 0, "Iterator did not exhaust.");
    TEST_ASSERTION(boundary_dof_iterator_next(iter) == 0, "Exhausted iterator must stay exhausted.");

    // For endpoint-node bases the boundary values are a pure selection of degrees of freedom.
    // boundary_dof_values always emits the ascending compact order, so the values check runs
    // against an ascending reference enumeration while the iterated indices are verified to
    // be a permutation of it.
    bool slice_bases = true;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        slice_bases =
            slice_bases && (basis[axis].type == BASIS_BERNSTEIN || basis[axis].type == BASIS_LAGRANGE_UNIFORM ||
                            basis[axis].type == BASIS_LAGRANGE_GAUSS_LOBATTO);
    }
    if (slice_bases)
    {
        int8_t ascending_orientation[MAX_TEST_DIM];
        for (unsigned entry = 0; entry < ndim; ++entry)
        {
            const int8_t code = orientation[entry];
            ascending_orientation[entry] = entry < ndim - bdim ? code : (int8_t)(code < 0 ? -code : code);
        }
        reference_indices(ndim, basis, bdim, ascending_orientation, expected);

        fill_random(values, element_count, rng);
        boundary_dof_values(ndim, basis, values, work, bdim, orientation, out);
        for (size_t k = 0; k < boundary_count; ++k)
        {
            TEST_NUMBERS_CLOSE(out[k], values[expected[k]], 0.0, 0.0);
            bool found = false;
            for (size_t j = 0; j < boundary_count; ++j)
            {
                found = found || indices[j] == expected[k];
            }
            TEST_ASSERTION(found, "Iterated indices are not a permutation of the boundary dofs.");
        }
    }

    free(expected);
    free(indices);
    free(values);
    free(out);
    free(work);
}

int main()
{
    test_prng_t rng;
    test_prng_seed(&rng, 20260917u);

    // 1D element: both vertices and the full line in both directions.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LAGRANGE_GAUSS_LOBATTO, 3}};
        const int8_t start[MAX_TEST_DIM] = {-1};
        const int8_t end[MAX_TEST_DIM] = {+1};
        run_case(1, basis, 0, start, &rng);
        run_case(1, basis, 0, end, &rng);
        run_case(1, basis, 1, start, &rng);
        run_case(1, basis, 1, end, &rng);
    }
    // 2D element: edges with reversed varying axes and the full face.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LAGRANGE_GAUSS_LOBATTO, 2}, {BASIS_LAGRANGE_UNIFORM, 3}};
        const int8_t edge_start[MAX_TEST_DIM] = {-1, +2};
        const int8_t edge_end[MAX_TEST_DIM] = {+1, -2};
        const int8_t ascending[MAX_TEST_DIM] = {+1, +2};
        const int8_t descending[MAX_TEST_DIM] = {-1, -2};
        run_case(2, basis, 1, edge_start, &rng);
        run_case(2, basis, 1, edge_end, &rng);
        run_case(2, basis, 2, ascending, &rng);
        run_case(2, basis, 2, descending, &rng);
    }
    // 3D element: order-zero fixed axis, two fixed axes, reversed varying axes.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {
            {BASIS_BERNSTEIN, 0}, {BASIS_LAGRANGE_GAUSS_LOBATTO, 3}, {BASIS_LAGRANGE_UNIFORM, 2}};
        const int8_t zero_fixed[MAX_TEST_DIM] = {+1, -2, +3};
        run_case(3, basis, 2, zero_fixed, &rng);
        const int8_t two_fixed[MAX_TEST_DIM] = {+1, -2, -3};
        run_case(3, basis, 1, two_fixed, &rng);
    }
    // Basis types are irrelevant to the indices; Legendre must behave identically.
    {
        const basis_spec_t basis[MAX_TEST_DIM] = {{BASIS_LEGENDRE, 3}, {BASIS_LAGRANGE_GAUSS_LOBATTO, 2}};
        const int8_t orientation[MAX_TEST_DIM] = {-1, -2};
        run_case(2, basis, 1, orientation, &rng);
    }

    printf("test_boundary_dof_indices PASSED\n");
    return 0;
}
