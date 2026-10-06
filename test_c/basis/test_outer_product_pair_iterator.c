//
// Created by jan on 2026-09-12.
//
#include "../common/common.h"

#include "../../src/basis/basis_set.h"

// ================================================================
// Main Entry
// ================================================================
int main(void)
{
    basis_set_registry_t *registry;
    integration_rule_registry_t *ir_registry;

    // Create the registries with caching enabled
    TEST_FDG_RESULT(integration_rule_registry_create(&ir_registry, 1 /*should_cache*/, &TEST_ALLOCATOR));
    TEST_FDG_RESULT(basis_set_registry_create(&registry, 1 /*should_cache*/, &TEST_ALLOCATOR));

    enum
    {
        MAX_DIMS = 4,
    };
    const basis_set_type_t basis_types[] = {BASIS_LEGENDRE, BASIS_LAGRANGE_GAUSS_LOBATTO, BASIS_BERNSTEIN};

    test_prng_t prng;
    test_prng_seed(&prng, 20260912);

    for (unsigned ndim = 1; ndim <= MAX_DIMS; ++ndim)
    {
        const basis_set_t *basis_left[MAX_DIMS];
        const basis_set_t *basis_right[MAX_DIMS];
        const integration_rule_t *rules[MAX_DIMS];
        size_t basis_indices_left[MAX_DIMS];
        size_t basis_indices_right[MAX_DIMS];
        size_t point_dims[MAX_DIMS];
        size_t total_points = 1;

        for (unsigned dim = 0; dim < ndim; ++dim)
        {
            const basis_spec_t spec_left = {.type = basis_types[test_prng_next_uint(&prng) % 3],
                                            .order = 1 + test_prng_next_uint(&prng) % 4};
            const basis_spec_t spec_right = {.type = basis_types[test_prng_next_uint(&prng) % 3],
                                             .order = 1 + test_prng_next_uint(&prng) % 4};
            const integration_spec_t rule_spec = {.type = (integration_rule_type_t)(1 + test_prng_next_uint(&prng) % 2),
                                                  .order = spec_left.order > spec_right.order ? spec_left.order
                                                                                              : spec_right.order};

            TEST_FDG_RESULT(integration_rule_registry_get_rule(ir_registry, rule_spec, &rules[dim]));
            TEST_FDG_RESULT(basis_set_registry_get_basis_set(registry, &basis_left[dim], rules[dim], spec_left));
            TEST_FDG_RESULT(basis_set_registry_get_basis_set(registry, &basis_right[dim], rules[dim], spec_right));

            basis_indices_left[dim] = test_prng_next_uint(&prng) % (spec_left.order + 1);
            basis_indices_right[dim] = test_prng_next_uint(&prng) % (spec_right.order + 1);

            // Both sides are retrieved against the same rule, so each dimension has rule order + 1 points
            point_dims[dim] = basis_left[dim]->integration_spec.order + 1;
            total_points *= point_dims[dim];
        }

        // Reference integration point sweep, matching the iterator order (last dimension fastest)
        _Alignas(max_align_t) unsigned char iter_buffer[512];
        multidim_iterator_t *const ref_iter = (multidim_iterator_t *)iter_buffer;
        TEST_ASSERTION(multidim_iterator_needed_memory(ndim) <= sizeof(iter_buffer), "Iterator buffer too small.");
        multidim_iterator_init(ref_iter, ndim, point_dims);

        // Mode 0: fold the rule weights, no derivatives; mode 1: fold the rule weights, derivatives on the first
        // left and last right dimension; mode 2: no rule weights (weights left to the caller), no derivatives
        const unsigned derivative_masks[][2] = {{0, 0}, {1u << 0, 1u << (ndim - 1)}, {0, 0}};

        for (unsigned mode = 0; mode < 3; ++mode)
        {
            const integration_rule_t *const *mode_rules = mode == 2 ? NULL : rules;

            _Alignas(max_align_t) unsigned char pair_buffer[256];
            outer_product_pair_iterator_t *const pair = (outer_product_pair_iterator_t *)pair_buffer;
            TEST_ASSERTION(outer_product_pair_iterator_data_size(ndim) <= sizeof(pair_buffer),
                           "Pair iterator buffer too small.");
            outer_product_pair_iterator_init(pair, ndim, basis_left, basis_right, mode_rules, derivative_masks[mode][0],
                                             derivative_masks[mode][1]);
            outer_product_pair_iterator_set_basis_indices(pair, basis_indices_left, basis_indices_right);

            size_t visited_points = 0;
            for (;;)
            {
                // Manually compute the outer product of the pair at the current integration point
                const size_t *const ip = multidim_iterator_offsets(ref_iter);
                double expected = 1;
                for (unsigned dim = 0; dim < ndim; ++dim)
                {
                    if (mode != 2)
                        expected *= integration_rule_weights_const(rules[dim])[ip[dim]];
                    expected *= ((derivative_masks[mode][0] >> dim) & 1u)
                                    ? basis_set_basis_derivatives(basis_left[dim], basis_indices_left[dim])[ip[dim]]
                                    : basis_set_basis_values(basis_left[dim], basis_indices_left[dim])[ip[dim]];
                    expected *= ((derivative_masks[mode][1] >> dim) & 1u)
                                    ? basis_set_basis_derivatives(basis_right[dim], basis_indices_right[dim])[ip[dim]]
                                    : basis_set_basis_values(basis_right[dim], basis_indices_right[dim])[ip[dim]];
                }

                TEST_NUMBERS_CLOSE(outer_product_pair_iterator_current_value(pair), expected, 1e-12, 1e-12);
                TEST_ASSERTION(outer_product_pair_iterator_point_index(pair) ==
                                   multidim_iterator_get_flat_index(ref_iter),
                               "Iterator point index does not match the reference flat index.");

                ++visited_points;
                const int has_next = outer_product_pair_iterator_next_integration_point(pair);
                multidim_iterator_advance(ref_iter, ndim - 1, 1);
                if (!has_next)
                    break;
            }
            TEST_ASSERTION(visited_points == total_points, "Iterator did not visit every integration point.");
            TEST_ASSERTION(multidim_iterator_is_at_end(ref_iter), "Reference iterator is not at its end position.");
            multidim_iterator_set_to_start(ref_iter);
        }

        for (unsigned dim = 0; dim < ndim; ++dim)
        {
            basis_set_registry_release_basis_set(registry, basis_left[dim]);
            basis_set_registry_release_basis_set(registry, basis_right[dim]);
        }
    }

    // Destroy the registries -> should free everything
    basis_set_registry_destroy(registry);
    integration_rule_registry_destroy(ir_registry);

    printf("test_outer_product_pair_iterator PASSED\n");

    return 0;
}
