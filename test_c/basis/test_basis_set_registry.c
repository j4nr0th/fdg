//
// Created by jan on 2025-09-11.
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

    // Create the registry with caching enabled
    TEST_FDG_RESULT(integration_rule_registry_create(&ir_registry, 1 /*should_cache*/, &TEST_ALLOCATOR));
    TEST_FDG_RESULT(basis_set_registry_create(&registry, 1 /*should_cache*/, &TEST_ALLOCATOR));

    enum
    {
        PAIR_COUNT = 1000,
        INTEGRATION_RULE_MAX_ORDER = 10,
        INTEGRATION_RULE_TYPE_COUNT = 2,
        BASIS_TYPE_COUNT = 6,
        BASIS_SET_MAX_ORDER = 10,
    };

    test_prng_t prng;
    test_prng_seed(&prng, 42);

    const basis_set_t *basis_sets[PAIR_COUNT];
    const integration_rule_t *rules[PAIR_COUNT];

    // Generate PAIR_COUNT different basis sets with random parameters
    for (int i = 0; i < PAIR_COUNT; i++)
    {
        const integration_spec_t rule_spec = {
            .type = (integration_rule_type_t)(test_prng_next_uint(&prng) % INTEGRATION_RULE_TYPE_COUNT + 1),
            .order = 1 + (test_prng_next_uint(&prng) % INTEGRATION_RULE_MAX_ORDER)};

        TEST_FDG_RESULT(integration_rule_registry_get_rule(ir_registry, rule_spec, &rules[i]));

        const basis_spec_t spec = {.type = (basis_set_type_t)(test_prng_next_uint(&prng) % BASIS_TYPE_COUNT + 1),
                                   .order = 1 + (test_prng_next_uint(&prng) % BASIS_SET_MAX_ORDER)};

        // Get basis set and store it
        TEST_FDG_RESULT(basis_set_registry_get_basis_set(registry, &basis_sets[i], rules[i], spec));

        // Request the same basis set again to verify caching
        const basis_set_t *cached_basis;
        TEST_FDG_RESULT(basis_set_registry_get_basis_set(registry, &cached_basis, rules[i], spec));

        // Verify caching worked
        TEST_ASSERTION(basis_sets[i] == cached_basis, "Caching failed: Expected same basis_set pointer");

        // Release cached reference
        TEST_FDG_RESULT(basis_set_registry_release_basis_set(registry, cached_basis));
    }

    // Release all basis sets
    for (int i = 0; i < PAIR_COUNT; i++)
    {
        TEST_FDG_RESULT(basis_set_registry_release_basis_set(registry, basis_sets[i]));
    }

    // Force cleanup of unused basis sets
    basis_set_registry_release_unused_basis_sets(registry);

    // Destroy the registry -> should free everything
    basis_set_registry_destroy(registry);
    integration_rule_registry_destroy(ir_registry);

    printf("test_caching_and_memory PASSED\n");

    return 0;
}
