/**
 * @file bench_boundary_mass.c
 * @brief Release benchmark: table-based vs outer-product-iterator-based
 *        generation of one element's boundary constraint mass matrix.
 *
 * Both variants compute the identical reference (C1) boundary mass matrix of
 * #constraint_boundary_mass_assemble on an identity-oriented face with
 * `ndim == bdim`, so neither endpoint evaluation nor axis mirroring applies
 * and the only difference is the value-generation strategy:
 *
 * - table: per-axis registry value tables, tensor product assembly and a
 *   single #kform_inner_product_block per component block (the shipped
 *   engine path).
 * - pair: one #outer_product_pair_iterator_t walk per test x element DoF
 *   pair, exactly like the pre-conversion mass matrix code.
 *
 * Component enumeration, offsets, and signs come from the engine's own
 * layout pass, so the two matrices must agree to roundoff; the benchmark
 * fails loudly when they do not.
 */

#include "../../src/constraints/constraints.h"
#include "../common/common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *bench_allocate(void *ctx, size_t size)
{
    (void)ctx;
    return malloc(size);
}

static void bench_free(void *ctx, void *ptr)
{
    (void)ctx;
    free(ptr);
}

static void *bench_reallocate(void *ctx, void *ptr, size_t size)
{
    (void)ctx;
    return realloc(ptr, size);
}

static cutl_allocator_t BENCH_ALLOCATOR = {
    .allocate = bench_allocate,
    .deallocate = bench_free,
    .reallocate = bench_reallocate,
};

static basis_spec_t bench_basis_spec(const unsigned order)
{
    return (basis_spec_t){.type = BASIS_LEGENDRE, .order = order};
}

/** Monotonic clock in seconds. */
static double bench_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

/**
 * @brief Per-axis test counts of one component (mirrors the engine's rule).
 *
 * Active covector axes read the order-1 basis (`order` functions, offset
 * zero); inactive axes read the full basis minus the first `skip` functions.
 */
static void bench_row_axis_counts(const unsigned boundary_order, const unsigned order,
                                  const uint8_t axes[const static order == 0 ? 1 : order], const unsigned bdim,
                                  const uint8_t axis_skip[const static bdim], unsigned counts[const static bdim],
                                  unsigned offsets[const static bdim])
{
    for (unsigned axis = 0; axis < bdim; ++axis)
    {
        bool active = false;
        for (unsigned i = 0; i < order; ++i)
        {
            active = active || axes[i] == axis;
        }
        if (active)
        {
            counts[axis] = boundary_order;
            offsets[axis] = 0;
        }
        else
        {
            const unsigned full = boundary_order + 1u;
            counts[axis] = full > axis_skip[axis] ? full - axis_skip[axis] : 0u;
            offsets[axis] = axis_skip[axis];
        }
    }
}

/** Odometer increment over per-axis counts, last axis fastest. */
static bool bench_advance_digits(const unsigned bdim, const unsigned counts[const static bdim],
                                 unsigned digits[const static bdim])
{
    for (unsigned axis = bdim; axis-- > 0;)
    {
        if (++digits[axis] < counts[axis])
        {
            return true;
        }
        digits[axis] = 0;
    }
    return false;
}

int main(void)
{
    const unsigned element_orders[] = {2, 4, 6, 8};
    const unsigned form_orders[] = {0, 1, 2};
    const unsigned bdim = 2;

    printf("%8s %6s %10s %16s %16s %8s %10s\n", "elem_or", "form", "rows x cols", "table [ms]", "pair [ms]", "speedup",
           "maxdiff");
    for (unsigned config = 0; config < sizeof(element_orders) / sizeof(element_orders[0]); ++config)
    {
        for (unsigned fo = 0; fo < sizeof(form_orders) / sizeof(form_orders[0]); ++fo)
        {
            const unsigned k = element_orders[config];
            const unsigned order = form_orders[fo];
            const uint8_t axis_skip[2] = {k < 2 ? (uint8_t)k : 2u, k < 2 ? (uint8_t)k : 2u};

            // Untimed setup: rules, basis sets, work buffers, shared layout.
            integration_rule_t *rule = NULL;
            if (integration_rule_for_order(&rule, INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, k + 1u, &BENCH_ALLOCATOR) !=
                FDG_SUCCESS)
            {
                fprintf(stderr, "rule creation failed\n");
                return 1;
            }
            const integration_rule_t *rules[2] = {rule, rule};
            const integration_spec_t rule_specs[2] = {rule->spec, rule->spec};

            basis_set_registry_t *registry = NULL;
            if (basis_set_registry_create(&registry, 1, &BENCH_ALLOCATOR) != FDG_SUCCESS)
            {
                fprintf(stderr, "registry creation failed\n");
                return 1;
            }
            const basis_spec_t boundary_specs[2] = {bench_basis_spec(k), bench_basis_spec(k)};
            const basis_spec_t element_specs[2] = {bench_basis_spec(k), bench_basis_spec(k)};
            const basis_spec_t lower_specs[2] = {bench_basis_spec(k > 0 ? k - 1u : 0u),
                                                 bench_basis_spec(k > 0 ? k - 1u : 0u)};
            const basis_set_t *boundary_sets[2] = {NULL, NULL};
            const basis_set_t *boundary_sets_lower[2] = {NULL, NULL};
            const basis_set_t *element_sets[2] = {NULL, NULL};
            if (basis_set_registry_get_basis_sets(registry, bdim, boundary_sets, rules, boundary_specs) !=
                    FDG_SUCCESS ||
                basis_set_registry_get_basis_sets(registry, bdim, boundary_sets_lower, rules, lower_specs) !=
                    FDG_SUCCESS ||
                basis_set_registry_get_basis_sets(registry, bdim, element_sets, rules, element_specs) != FDG_SUCCESS)
            {
                fprintf(stderr, "basis set creation failed\n");
                return 1;
            }

            const kform_spec_t element_form = {.ndim = bdim, .order = order, .basis = element_specs};
            const int8_t orientation[2] = {1, 2};
            const constraint_boundary_mass_spec_t spec = {
                .ndim = bdim,
                .bdim = bdim,
                .order = order,
                .element_spec = &element_form,
                .boundary_basis = boundary_specs,
                .boundary_integration = rule_specs,
                .orientation = orientation,
                .axis_skip = axis_skip,
            };

            constraint_boundary_mass_work_sizes_t sizes;
            constraint_boundary_mass_work_size(&spec, &sizes);
            const size_t component_count = sizes.component_count;
            const size_t point_count = integration_specs_total_points(bdim, rule_specs);

            size_t rows;
            size_t cols;
            size_t entries;
            constraint_boundary_mass_work_t work = {};
            const size_t iter_mem = combination_iterator_required_memory((uint8_t)order);
            work.point_iter = malloc(multidim_iterator_needed_memory(bdim));
            work.row_offsets = malloc((component_count + 1) * sizeof(*work.row_offsets));
            work.col_offsets = malloc((component_count + 1) * sizeof(*work.col_offsets));
            work.element_components = malloc(component_count * sizeof(*work.element_components));
            work.element_signs = malloc(component_count * sizeof(*work.element_signs));
            work.axes = malloc(bdim * sizeof(*work.axes));
            work.counts = malloc(bdim * sizeof(*work.counts));
            work.offsets = malloc(bdim * sizeof(*work.offsets));
            work.axis_sets = malloc(bdim * sizeof(*work.axis_sets));
            work.dof_iter = malloc(multidim_iterator_needed_memory(bdim));
            work.point_digits = malloc(bdim * sizeof(*work.point_digits));
            work.point_prefix = malloc(bdim * sizeof(*work.point_prefix));
            work.axis_tables = malloc(bdim * sizeof(*work.axis_tables));
            work.mapped_axes = malloc((order == 0 ? 1u : order) * sizeof(*work.mapped_axes));
            work.components = malloc(iter_mem);
            work.blocks = malloc(iter_mem);
            work.row_values = malloc(sizes.row_values * sizeof(*work.row_values));
            work.col_values = malloc(sizes.col_values * sizeof(*work.col_values));
            work.point_factors = malloc(sizes.point_factors * sizeof(*work.point_factors));
            if (!work.point_iter || !work.row_offsets || !work.col_offsets || !work.element_components ||
                !work.element_signs || !work.axes || !work.counts || !work.offsets || !work.axis_sets ||
                !work.dof_iter || !work.point_digits || !work.point_prefix || !work.axis_tables || !work.mapped_axes ||
                !work.components || !work.blocks || !work.row_values || !work.col_values || !work.point_factors)
            {
                fprintf(stderr, "work allocation failed\n");
                return 1;
            }
            constraint_boundary_mass_layout(&spec, &work, false, &rows, &cols, &entries);

            double *const weights = malloc(point_count * sizeof(*weights));
            double *const matrix_table = malloc(rows * cols * sizeof(*matrix_table));
            double *const matrix_pair = malloc(rows * cols * sizeof(*matrix_pair));
            if (!weights || !matrix_table || !matrix_pair)
            {
                fprintf(stderr, "matrix allocation failed\n");
                return 1;
            }
            integration_rule_tensor_weights(bdim, rules, weights);

            const constraint_boundary_mass_request_t request = {
                .spec = &spec,
                .boundary_basis_sets = boundary_sets,
                .boundary_basis_sets_lower = boundary_sets_lower,
                .element_basis_sets = element_sets,
                .element_basis_sets_lower = boundary_sets_lower,
                .element_endpoints = NULL,
                .element_endpoints_lower = NULL,
                .point_weights = weights,
                .surface_weights = NULL,
                .test_pullback = NULL,
                .element_pullback = NULL,
                .factor = 1.0,
                .work = &work,
                .out_matrix = matrix_table,
            };

            // Timed: table engine assembly.
            unsigned reps = 1;
            double elapsed = 0.0;
            for (;;)
            {
                const double start = bench_now();
                for (unsigned r = 0; r < reps; ++r)
                {
                    constraint_boundary_mass_assemble(&request);
                }
                elapsed = bench_now() - start;
                if (elapsed > 0.05 || reps >= (1u << 22))
                {
                    break;
                }
                reps *= 2;
            }
            const double table_ms = 1000.0 * elapsed / (double)reps;

            // Timed: outer-product pair iterator assembly. Shares the
            // engine's layout so only value generation differs.
            outer_product_pair_iterator_t *iter = malloc(outer_product_pair_iterator_data_size(bdim));
            if (!iter)
            {
                fprintf(stderr, "iterator allocation failed\n");
                return 1;
            }
            const basis_set_t *left_sets[2] = {NULL, NULL};
            unsigned left_counts[2];
            unsigned left_offsets[2];
            unsigned left_digits[2];
            unsigned right_counts[2];
            unsigned right_digits[2];
            size_t left_indices[2];
            size_t right_indices[2];
            reps = 1;
            for (;;)
            {
                const double start = bench_now();
                for (unsigned r = 0; r < reps; ++r)
                {
                    memset(matrix_pair, 0, rows * cols * sizeof(*matrix_pair));
                    combination_iterator_init(work.components, (uint8_t)bdim, (uint8_t)order);
                    for (size_t component = 0; !combination_iterator_is_done(work.components);
                         combination_iterator_next(work.components), ++component)
                    {
                        const uint8_t *const component_axes = combination_iterator_current(work.components);
                        bench_row_axis_counts(k, order, component_axes, bdim, axis_skip, left_counts, left_offsets);
                        const size_t row_dofs = work.row_offsets[component + 1] - work.row_offsets[component];
                        if (row_dofs == 0)
                        {
                            continue;
                        }
                        for (unsigned axis = 0; axis < bdim; ++axis)
                        {
                            bool active = false;
                            for (unsigned i = 0; i < order; ++i)
                            {
                                active = active || component_axes[i] == axis;
                            }
                            left_sets[axis] = active ? boundary_sets_lower[axis] : boundary_sets[axis];
                            right_digits[axis] = 0;
                            left_digits[axis] = 0;
                        }
                        const size_t col_dofs = work.col_offsets[component + 1] - work.col_offsets[component];
                        for (unsigned axis = 0; axis < bdim; ++axis)
                        {
                            // Element trace side: active axes read `order`
                            // functions, inactive axes the full basis.
                            bool active = false;
                            for (unsigned i = 0; i < order; ++i)
                            {
                                active = active || component_axes[i] == axis;
                            }
                            right_counts[axis] = active ? k : k + 1u;
                        }
                        const int sign = work.element_signs[component];
                        outer_product_pair_iterator_init(iter, bdim, left_sets, element_sets, rules, 0, 0);
                        for (size_t row = 0;; ++row)
                        {
                            for (unsigned axis = 0; axis < bdim; ++axis)
                            {
                                left_indices[axis] = (size_t)(left_offsets[axis] + left_digits[axis]);
                            }
                            for (size_t col = 0;; ++col)
                            {
                                for (unsigned axis = 0; axis < bdim; ++axis)
                                {
                                    right_indices[axis] = right_digits[axis];
                                }
                                outer_product_pair_iterator_set_basis_indices(iter, left_indices, right_indices);
                                double acc = 0.0;
                                for (;;)
                                {
                                    acc += outer_product_pair_iterator_current_value(iter);
                                    if (!outer_product_pair_iterator_next_integration_point(iter))
                                    {
                                        break;
                                    }
                                }
                                matrix_pair[(work.row_offsets[component] + row) * cols + work.col_offsets[component] +
                                            col] = (double)sign * acc;
                                if (!bench_advance_digits(bdim, right_counts, right_digits))
                                {
                                    break;
                                }
                            }
                            if (!bench_advance_digits(bdim, left_counts, left_digits))
                            {
                                break;
                            }
                        }
                    }
                }
                elapsed = bench_now() - start;
                if (elapsed > 0.05 || reps >= (1u << 22))
                {
                    break;
                }
                reps *= 2;
            }
            const double pair_ms = 1000.0 * elapsed / (double)reps;

            // Parity: the two matrices must agree to roundoff.
            double max_diff = 0.0;
            double max_scale = 0.0;
            for (size_t i = 0; i < rows * cols; ++i)
            {
                const double diff = fabs(matrix_table[i] - matrix_pair[i]);
                max_diff = max_diff > diff ? max_diff : diff;
                const double scale = fabs(matrix_table[i]);
                max_scale = max_scale > scale ? max_scale : scale;
            }
            const double rel_diff = max_scale > 0.0 ? max_diff / max_scale : max_diff;
            if (rel_diff > 1.0e-12)
            {
                fprintf(stderr, "PARITY FAILURE: config k=%u order=%u rel diff %.3e\n", k, order, rel_diff);
                return 1;
            }

            printf("%8u %6u %5zux%-5zu %16.4f %16.4f %7.2fx %10.1e\n", k, order, rows, cols, table_ms, pair_ms,
                   pair_ms / table_ms, rel_diff);

            free(iter);
            free(matrix_pair);
            free(matrix_table);
            free(weights);
            free(work.point_iter);
            free(work.row_offsets);
            free(work.col_offsets);
            free(work.element_components);
            free(work.element_signs);
            free(work.axes);
            free(work.counts);
            free(work.offsets);
            free(work.axis_sets);
            free(work.dof_iter);
            free(work.point_digits);
            free(work.point_prefix);
            free(work.axis_tables);
            free(work.mapped_axes);
            free(work.components);
            free(work.blocks);
            free(work.row_values);
            free(work.col_values);
            free(work.point_factors);
            for (unsigned axis = 0; axis < bdim; ++axis)
            {
                basis_set_registry_release_basis_set(registry, element_sets[axis]);
                basis_set_registry_release_basis_set(registry, boundary_sets_lower[axis]);
                basis_set_registry_release_basis_set(registry, boundary_sets[axis]);
            }
            basis_set_registry_destroy(registry);
            cutl_dealloc(&BENCH_ALLOCATOR, rule);
        }
    }
    return 0;
}
