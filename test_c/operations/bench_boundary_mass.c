/**
 * @file bench_boundary_mass.c
 * @brief Release benchmark of the boundary constraint mass matrix hot paths.
 *
 * Per configuration (identity-oriented face with `ndim == bdim`, so neither
 * endpoint evaluation nor axis mirroring applies) the benchmark times the four
 * hot paths the constraints work targets:
 *
 * - wsize:  #constraint_boundary_mass_work_size (the sizing pass),
 * - ref:    #constraint_boundary_mass_assemble, reference (C1) path,
 * - phys:   #constraint_boundary_mass_assemble, physical path with sampled
 *           pullbacks,
 * - inner:  #kform_inner_product_block at the configuration's largest block
 *           shape (rows/cols of the sizing pass).
 *
 * Both assembled matrices are checked against a naive oracle that evaluates
 * the tensor-product pairing directly from the per-axis basis tables; the
 * benchmark fails loudly when they disagree beyond roundoff.
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

/** Keeps the sizing pass observable to the optimizer. */
static volatile size_t bench_wsize_sink;

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

/** One timed section: call until at least 50 ms elapsed. Returns ns per call. */
typedef void (*bench_call_t)(void *ctx);

static double bench_measure(const bench_call_t call, void *ctx)
{
    unsigned reps = 1;
    for (;;)
    {
        const double start = bench_now();
        for (unsigned r = 0; r < reps; ++r)
        {
            call(ctx);
        }
        const double elapsed = bench_now() - start;
        if (elapsed > 0.05 || reps >= (1u << 24))
        {
            return 1.0e9 * elapsed / (double)reps;
        }
        reps *= 2;
    }
}

typedef struct
{
    const constraint_boundary_mass_spec_t *spec;
    constraint_boundary_mass_work_t *work;
} bench_wsize_ctx_t;

static void bench_call_wsize(void *const ctx)
{
    const bench_wsize_ctx_t *const c = ctx;
    constraint_boundary_mass_work_sizes_t sizes;
    constraint_boundary_mass_work_size(c->spec, c->work, &sizes);
    bench_wsize_sink += sizes.row_values + sizes.col_values;
}

typedef struct
{
    const constraint_boundary_mass_request_t *request;
} bench_assemble_ctx_t;

static void bench_call_assemble(void *const ctx)
{
    const bench_assemble_ctx_t *const c = ctx;
    constraint_boundary_mass_assemble(c->request);
}

typedef struct
{
    size_t point_count;
    size_t dofs_left;
    size_t dofs_right;
    const double *left;
    const double *right;
    const double *weights;
    double *out;
} bench_inner_ctx_t;

static void bench_call_inner(void *const ctx)
{
    const bench_inner_ctx_t *const c = ctx;
    kform_inner_product_block(c->point_count, c->dofs_left, c->dofs_right, c->left, c->right, c->weights, 0, 0,
                              c->dofs_right, c->out);
}

/**
 * @brief Per-axis test counts of one component (mirrors the engine's rule).
 *
 * Active covector axes read the order-1 basis (`boundary_order` functions); inactive axes read the leading
 * full-basis functions minus the last `SKIPPED_BASIS` ones (the window keeps the lowest functions).
 */
static void bench_row_axis_counts(const unsigned boundary_order, const unsigned order,
                                  const uint8_t axes[const static order == 0 ? 1 : order], const unsigned bdim,
                                  unsigned counts[const static bdim])
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
        }
        else
        {
            const unsigned full = boundary_order + 1u;
            counts[axis] = full > SKIPPED_BASIS ? full - SKIPPED_BASIS : 0u;
        }
    }
}

/** Decode a flat component-local index into per-axis digits, axis 0 slowest. */
static void bench_decode_digits(const unsigned bdim, const unsigned counts[const static bdim], size_t flat,
                                unsigned digits[const static bdim])
{
    for (unsigned axis = bdim; axis-- > 0;)
    {
        digits[axis] = (unsigned)(flat % counts[axis]);
        flat /= counts[axis];
    }
}

/** Deterministic non-uniform pullback sample so indexing bugs show up. */
static double bench_pullback_value(const bool element, const unsigned component, const unsigned physical,
                                   const size_t point)
{
    const unsigned mixed = element ? component * 11u + physical * 5u : component * 7u + physical * 13u;
    return element ? 0.3 + 0.02 * (double)((mixed + (unsigned)(point % 89u)) % 89u)
                   : 0.5 + 0.01 * (double)((mixed + (unsigned)(point % 97u)) % 97u);
}

/**
 * @brief Inputs of one naive reference-oracle evaluation.
 *
 * Identity orientation: the mapped element component equals the test
 * component and all orientation signs are +1.
 */
typedef struct
{
    unsigned bdim;
    unsigned boundary_order;
    unsigned element_order;
    unsigned order;
    const unsigned *nodes; ///< [bdim] Points per axis.
    size_t point_count;
    const double *weights; ///< [point_count]
    const basis_set_t *const *boundary_sets;
    const basis_set_t *const *boundary_sets_lower;
    const basis_set_t *const *element_sets;
    const basis_set_t *const *element_sets_lower;
    bool coupled; ///< Physical pairing: every test x element component pair.
    unsigned pullback_components;
    const double *test_pullback;    ///< [component_count * pullback_components * point_count].
    const double *element_pullback; ///< Same layout, element component slot.
    size_t rows;
    size_t cols;
    double *out; ///< [rows * cols], zero-filled by the callee.
} bench_oracle_t;

/**
 * @brief Evaluate one boundary mass matrix directly from the axis tables.
 *
 * Reference mode fills the component-diagonal blocks, physical mode (with
 * sampled pullbacks) accumulates every component pair.
 */
static void bench_naive_matrix(const bench_oracle_t *const n)
{
    const unsigned bdim = n->bdim;
    const unsigned order = n->order;
    const size_t point_count = n->point_count;
    const size_t component_count = combination_total_count((uint8_t)bdim, (uint8_t)order);
    TEST_ASSERTION(component_count > 0, "Component count must be positive.");

    // Row-major point strides (last axis fastest) and the point digits per axis.
    size_t point_stride[3];
    point_stride[bdim - 1] = 1;
    for (unsigned axis = bdim; axis-- > 1;)
    {
        point_stride[axis - 1] = point_stride[axis] * n->nodes[axis];
    }
    unsigned *const point_digits = malloc((size_t)bdim * point_count * sizeof(*point_digits));
    TEST_ASSERTION(point_digits != NULL, "Point digit allocation failed.");
    for (size_t point = 0; point < point_count; ++point)
    {
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            point_digits[(size_t)axis * point_count + point] =
                (unsigned)((point / point_stride[axis]) % n->nodes[axis]);
        }
    }

    // Component row/column offsets from the mirrored counting rules.
    size_t *const row_off = malloc((component_count + 1) * sizeof(*row_off));
    size_t *const col_off = malloc((component_count + 1) * sizeof(*col_off));
    uint8_t *const component_axes = malloc((order == 0 ? 1u : order) * sizeof(*component_axes));
    TEST_ASSERTION(row_off && col_off && component_axes, "Offset allocation failed.");
    unsigned row_counts[3];
    unsigned col_counts[3];
    row_off[0] = 0;
    col_off[0] = 0;
    size_t max_row_dofs = 0;
    for (size_t component = 0; component < component_count; ++component)
    {
        combination_set_to_index((uint8_t)bdim, (uint8_t)order, component_axes, (unsigned)component);
        bench_row_axis_counts(n->boundary_order, order, component_axes, bdim, row_counts);
        size_t row_dofs = 1;
        size_t col_dofs = 1;
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            bool active = false;
            for (unsigned i = 0; i < order; ++i)
            {
                active = active || component_axes[i] == axis;
            }
            // Element trace side: active axes read `element_order` functions, inactive axes the full basis.
            col_counts[axis] = active ? n->element_order : n->element_order + 1u;
            row_dofs *= row_counts[axis];
            col_dofs *= col_counts[axis];
        }
        row_off[component + 1] = row_off[component] + row_dofs;
        col_off[component + 1] = col_off[component] + col_dofs;
        max_row_dofs = max_row_dofs > row_dofs ? max_row_dofs : row_dofs;
    }
    TEST_ASSERTION(row_off[component_count] == n->rows, "Oracle row count disagrees with the layout (%zu vs %zu).",
                   row_off[component_count], n->rows);
    TEST_ASSERTION(col_off[component_count] == n->cols, "Oracle column count disagrees with the layout (%zu vs %zu).",
                   col_off[component_count], n->cols);

    // Column values of every component: [col_off[c] * point_count + col * point_count + point].
    double *const col_values = malloc(n->cols * point_count * sizeof(*col_values));
    double *const row_values = malloc(max_row_dofs * point_count * sizeof(*row_values));
    double *const dot = malloc(point_count * sizeof(*dot));
    TEST_ASSERTION(col_values && row_values && dot, "Value allocation failed.");
    unsigned digits[3];
    for (size_t component = 0; component < component_count; ++component)
    {
        combination_set_to_index((uint8_t)bdim, (uint8_t)order, component_axes, (unsigned)component);
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            bool active = false;
            for (unsigned i = 0; i < order; ++i)
            {
                active = active || component_axes[i] == axis;
            }
            col_counts[axis] = active ? n->element_order : n->element_order + 1u;
        }
        const size_t col_dofs = col_off[component + 1] - col_off[component];
        for (size_t col = 0; col < col_dofs; ++col)
        {
            bench_decode_digits(bdim, col_counts, col, digits);
            double *const values = col_values + (col_off[component] + col) * point_count;
            for (size_t point = 0; point < point_count; ++point)
            {
                double value = 1.0;
                for (unsigned axis = 0; axis < bdim; ++axis)
                {
                    const bool active = order != 0 && col_counts[axis] == n->element_order;
                    const basis_set_t *const set = active ? n->element_sets_lower[axis] : n->element_sets[axis];
                    value *=
                        basis_set_basis_values(set, digits[axis])[point_digits[(size_t)axis * point_count + point]];
                }
                values[point] = value;
            }
        }
    }

    // Row values per component, then the dense fill.
    memset(n->out, 0, n->rows * n->cols * sizeof(*n->out));
    for (size_t component = 0; component < component_count; ++component)
    {
        combination_set_to_index((uint8_t)bdim, (uint8_t)order, component_axes, (unsigned)component);
        bench_row_axis_counts(n->boundary_order, order, component_axes, bdim, row_counts);
        const size_t row_dofs = row_off[component + 1] - row_off[component];
        if (row_dofs == 0)
        {
            continue;
        }
        for (size_t row = 0; row < row_dofs; ++row)
        {
            bench_decode_digits(bdim, row_counts, row, digits);
            double *const values = row_values + row * point_count;
            for (size_t point = 0; point < point_count; ++point)
            {
                double value = 1.0;
                for (unsigned axis = 0; axis < bdim; ++axis)
                {
                    bool active = false;
                    for (unsigned i = 0; i < order; ++i)
                    {
                        active = active || component_axes[i] == axis;
                    }
                    const basis_set_t *const set = active ? n->boundary_sets_lower[axis] : n->boundary_sets[axis];
                    value *=
                        basis_set_basis_values(set, digits[axis])[point_digits[(size_t)axis * point_count + point]];
                }
                values[point] = value;
            }
        }

        const size_t block_begin = n->coupled ? 0 : component;
        const size_t block_end = n->coupled ? component_count : component + 1u;
        for (size_t block = block_begin; block < block_end; ++block)
        {
            if (n->coupled)
            {
                // Identity orientation: the mapped element component is the block itself.
                for (size_t point = 0; point < point_count; ++point)
                {
                    double sum = 0.0;
                    for (unsigned pc = 0; pc < n->pullback_components; ++pc)
                    {
                        sum +=
                            n->test_pullback[((size_t)component * n->pullback_components + pc) * point_count + point] *
                            n->element_pullback[((size_t)block * n->pullback_components + pc) * point_count + point];
                    }
                    dot[point] = sum;
                }
            }
            else
            {
                for (size_t point = 0; point < point_count; ++point)
                {
                    dot[point] = 1.0;
                }
            }
            const size_t col_dofs = col_off[block + 1] - col_off[block];
            for (size_t row = 0; row < row_dofs; ++row)
            {
                const double *const row_values_point = row_values + row * point_count;
                double *const out_row = n->out + (row_off[component] + row) * n->cols + col_off[block];
                for (size_t col = 0; col < col_dofs; ++col)
                {
                    const double *const col_values_point = col_values + (col_off[block] + col) * point_count;
                    double acc = 0.0;
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        acc += n->weights[point] * dot[point] * row_values_point[point] * col_values_point[point];
                    }
                    out_row[col] += acc;
                }
            }
        }
    }

    free(dot);
    free(row_values);
    free(col_values);
    free(component_axes);
    free(col_off);
    free(row_off);
    free(point_digits);
}

/** Largest entrywise relative difference of two equal-length matrices. */
static double bench_max_rel_diff(const double *const a, const double *const b, const size_t count)
{
    double max_diff = 0.0;
    double max_scale = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const double diff = fabs(a[i] - b[i]);
        max_diff = max_diff > diff ? max_diff : diff;
        const double scale = fabs(b[i]);
        max_scale = max_scale > scale ? max_scale : scale;
    }
    return max_scale > 0.0 ? max_diff / max_scale : max_diff;
}

typedef struct
{
    unsigned bdim;
    unsigned k;
    unsigned order;
} bench_config_t;

int main(void)
{
    static const bench_config_t configs[] = {
        {2, 2, 0}, {2, 2, 1}, {2, 2, 2}, {2, 4, 0}, {2, 4, 1}, {2, 4, 2}, {2, 6, 0}, {2, 6, 1},
        {2, 6, 2}, {2, 8, 0}, {2, 8, 1}, {2, 8, 2}, {3, 4, 1}, {3, 4, 2}, {3, 6, 1}, {3, 6, 2},
    };
    const unsigned pullback_components = 2;

    printf("%5s %7s %5s %11s %11s %10s %10s %10s %10s %10s\n", "bdim", "elem_or", "form", "rows x cols", "wsize[ns]",
           "ref[ms]", "phys[ms]", "inner[ms]", "ref_rel", "phys_rel");
    for (size_t config = 0; config < sizeof(configs) / sizeof(configs[0]); ++config)
    {
        const unsigned bdim = configs[config].bdim;
        const unsigned k = configs[config].k;
        const unsigned order = configs[config].order;

        // Untimed setup: rules, basis sets, work buffers, shared layout.
        integration_rule_t *rule = NULL;
        if (integration_rule_for_order(&rule, INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, k + 1u, &BENCH_ALLOCATOR) !=
            FDG_SUCCESS)
        {
            fprintf(stderr, "rule creation failed\n");
            return 1;
        }
        const integration_rule_t *rules[3] = {rule, rule, rule};
        const integration_spec_t rule_specs[3] = {rule->spec, rule->spec, rule->spec};
        unsigned nodes[3];
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            nodes[axis] = rule_specs[axis].order + 1u;
        }

        basis_set_registry_t *registry = NULL;
        if (basis_set_registry_create(&registry, 1, &BENCH_ALLOCATOR) != FDG_SUCCESS)
        {
            fprintf(stderr, "registry creation failed\n");
            return 1;
        }
        const basis_spec_t boundary_specs[3] = {bench_basis_spec(k), bench_basis_spec(k), bench_basis_spec(k)};
        const basis_spec_t element_specs[3] = {bench_basis_spec(k), bench_basis_spec(k), bench_basis_spec(k)};
        const basis_spec_t lower_specs[3] = {bench_basis_spec(k > 0 ? k - 1u : 0u),
                                             bench_basis_spec(k > 0 ? k - 1u : 0u),
                                             bench_basis_spec(k > 0 ? k - 1u : 0u)};
        const basis_set_t *boundary_sets[3] = {NULL, NULL, NULL};
        const basis_set_t *boundary_sets_lower[3] = {NULL, NULL, NULL};
        const basis_set_t *element_sets[3] = {NULL, NULL, NULL};
        if (basis_set_registry_get_basis_sets(registry, bdim, boundary_sets, rules, boundary_specs) != FDG_SUCCESS ||
            basis_set_registry_get_basis_sets(registry, bdim, boundary_sets_lower, rules, lower_specs) != FDG_SUCCESS ||
            basis_set_registry_get_basis_sets(registry, bdim, element_sets, rules, element_specs) != FDG_SUCCESS)
        {
            fprintf(stderr, "basis set creation failed\n");
            return 1;
        }

        const kform_spec_t element_form = {.ndim = bdim, .order = order, .basis = element_specs};
        const int8_t orientation[3] = {1, 2, 3};
        const constraint_boundary_mass_spec_t spec = {
            .ndim = bdim,
            .bdim = bdim,
            .order = order,
            .element_spec = &element_form,
            .boundary_basis = boundary_specs,
            .boundary_integration = rule_specs,
            .orientation = orientation,
        };

        // One block for the sizing scratch, one block for the full work buffers (see
        // constraint_boundary_mass_work_memory).
        constraint_boundary_mass_work_t work = {};
        void *const scratch = malloc(constraint_boundary_mass_work_memory(&spec, NULL));
        if (scratch == NULL)
        {
            fprintf(stderr, "sizing scratch allocation failed\n");
            return 1;
        }
        constraint_boundary_mass_work_init(&work, &spec, NULL, scratch);
        constraint_boundary_mass_work_sizes_t sizes;
        constraint_boundary_mass_work_size(&spec, &work, &sizes);
        void *const work_memory = malloc(constraint_boundary_mass_work_memory(&spec, &sizes));
        if (work_memory == NULL)
        {
            fprintf(stderr, "work allocation failed\n");
            return 1;
        }
        constraint_boundary_mass_work_init(&work, &spec, &sizes, work_memory);
        free(scratch);
        const size_t component_count = sizes.component_count;
        const size_t point_count = integration_specs_total_points(bdim, rule_specs);

        size_t rows;
        size_t cols;
        size_t entries;
        constraint_boundary_mass_layout(&spec, &work, false, &rows, &cols, &entries);
        (void)entries;

        double *const weights = malloc(point_count * sizeof(*weights));
        double *const matrix_ref = malloc(rows * cols * sizeof(*matrix_ref));
        double *const matrix_phys = malloc(rows * cols * sizeof(*matrix_phys));
        double *const matrix_naive_ref = malloc(rows * cols * sizeof(*matrix_naive_ref));
        double *const matrix_naive_phys = malloc(rows * cols * sizeof(*matrix_naive_phys));
        if (!weights || !matrix_ref || !matrix_phys || !matrix_naive_ref || !matrix_naive_phys)
        {
            fprintf(stderr, "matrix allocation failed\n");
            return 1;
        }
        integration_rule_tensor_weights(bdim, rules, weights);

        // Sampled pullbacks for the physical path (identity pairing in the oracle).
        const size_t pullback_values = component_count * pullback_components * point_count;
        double *const test_pullback_values = malloc(pullback_values * sizeof(*test_pullback_values));
        double *const element_pullback_values = malloc(pullback_values * sizeof(*element_pullback_values));
        TEST_ASSERTION(test_pullback_values && element_pullback_values, "Pullback allocation failed.");
        for (size_t i = 0; i < pullback_values; ++i)
        {
            const size_t point = i % point_count;
            const unsigned pc = (unsigned)((i / point_count) % pullback_components);
            const unsigned ci = (unsigned)(i / (point_count * pullback_components));
            test_pullback_values[i] = bench_pullback_value(false, ci, pc, point);
            element_pullback_values[i] = bench_pullback_value(true, ci, pc, point);
        }
        const constraint_trace_pullback_t test_pullback = {.physical_component_count = pullback_components,
                                                           .point_count = point_count,
                                                           .values = test_pullback_values};
        const constraint_trace_pullback_t element_pullback = {.physical_component_count = pullback_components,
                                                              .point_count = point_count,
                                                              .values = element_pullback_values};

        constraint_boundary_mass_request_t ref_request = {
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
            .out_matrix = matrix_ref,
        };
        constraint_boundary_mass_request_t phys_request = ref_request;
        phys_request.test_pullback = &test_pullback;
        phys_request.element_pullback = &element_pullback;
        phys_request.out_matrix = matrix_phys;

        // Timed sections (ns per call).
        const bench_wsize_ctx_t wsize_ctx = {.spec = &spec, .work = &work};
        const bench_assemble_ctx_t ref_ctx = {.request = &ref_request};
        const bench_assemble_ctx_t phys_ctx = {.request = &phys_request};
        const double wsize_ns = bench_measure(bench_call_wsize, (void *)&wsize_ctx);
        const double ref_ns = bench_measure(bench_call_assemble, (void *)&ref_ctx);
        const double phys_ns = bench_measure(bench_call_assemble, (void *)&phys_ctx);

        // Inner-product microbenchmark at the configuration's largest block shape.
        const size_t dofs_left = sizes.row_values / point_count;
        const size_t dofs_right = sizes.col_values / point_count;
        double *const left = malloc(point_count * dofs_left * sizeof(*left));
        double *const right = malloc(point_count * dofs_right * sizeof(*right));
        double *const inner_out = malloc(dofs_left * dofs_right * sizeof(*inner_out));
        TEST_ASSERTION(left && right && inner_out, "Inner product allocation failed.");
        for (size_t i = 0; i < point_count * dofs_left; ++i)
        {
            left[i] = 0.25 + 0.01 * (double)(i % 19u);
        }
        for (size_t i = 0; i < point_count * dofs_right; ++i)
        {
            right[i] = 0.5 + 0.02 * (double)(i % 23u);
        }
        const bench_inner_ctx_t inner_ctx = {
            .point_count = point_count,
            .dofs_left = dofs_left,
            .dofs_right = dofs_right,
            .left = left,
            .right = right,
            .weights = weights,
            .out = inner_out,
        };
        const double inner_ns = bench_measure(bench_call_inner, (void *)&inner_ctx);

        // Parity: both engine paths against the naive oracle.
        const bench_oracle_t oracle = {
            .bdim = bdim,
            .boundary_order = k,
            .element_order = k,
            .order = order,
            .nodes = nodes,
            .point_count = point_count,
            .weights = weights,
            .boundary_sets = boundary_sets,
            .boundary_sets_lower = boundary_sets_lower,
            .element_sets = element_sets,
            .element_sets_lower = boundary_sets_lower,
            .coupled = false,
            .pullback_components = pullback_components,
            .test_pullback = test_pullback_values,
            .element_pullback = element_pullback_values,
            .rows = rows,
            .cols = cols,
            .out = matrix_naive_ref,
        };
        bench_naive_matrix(&oracle);
        bench_oracle_t oracle_phys = oracle;
        oracle_phys.coupled = true;
        oracle_phys.out = matrix_naive_phys;
        bench_naive_matrix(&oracle_phys);

        const double ref_rel = bench_max_rel_diff(matrix_ref, matrix_naive_ref, rows * cols);
        const double phys_rel = bench_max_rel_diff(matrix_phys, matrix_naive_phys, rows * cols);
        if (ref_rel > 1.0e-12 || phys_rel > 1.0e-12)
        {
            fprintf(stderr, "PARITY FAILURE: bdim=%u k=%u order=%u ref=%.3e phys=%.3e\n", bdim, k, order, ref_rel,
                    phys_rel);
            return 1;
        }

        printf("%5u %7u %5u %5zux%-5zu %11.1f %10.4f %10.4f %10.4f %10.1e %10.1e\n", bdim, k, order, rows, cols,
               wsize_ns, ref_ns / 1.0e6, phys_ns / 1.0e6, inner_ns / 1.0e6, ref_rel, phys_rel);

        free(inner_out);
        free(right);
        free(left);
        free(element_pullback_values);
        free(test_pullback_values);
        free(matrix_naive_phys);
        free(matrix_naive_ref);
        free(matrix_phys);
        free(matrix_ref);
        free(weights);
        free(work_memory);
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            basis_set_registry_release_basis_set(registry, element_sets[axis]);
            basis_set_registry_release_basis_set(registry, boundary_sets_lower[axis]);
            basis_set_registry_release_basis_set(registry, boundary_sets[axis]);
        }
        basis_set_registry_destroy(registry);
        cutl_dealloc(&BENCH_ALLOCATOR, rule);
    }
    printf("sizing sink: %zu\n", (size_t)bench_wsize_sink);
    return 0;
}
