#include "../../src/integration/integration_rules.h"
#include "../common/common.h"

int main()
{
    // Test hyper-parameters
    enum
    {
        TEST_CASES = 300,
        MIN_ORDER = 0,
        MAX_ORDER = 10,
    };

    integration_rule_registry_t *registry;
    TEST_FDG_RESULT(integration_rule_registry_create(&registry, 1, &TEST_ALLOCATOR));

    const integration_rule_t *rules[TEST_CASES];
    for (unsigned i = 0; i < TEST_CASES; ++i)
    {
        const unsigned order = MIN_ORDER + (rand() % (MAX_ORDER - MIN_ORDER + 1));
        integration_rule_type_t type = 1 + (rand() % 2);
        TEST_FDG_RESULT(integration_rule_registry_get_rule(registry, (integration_spec_t){type, order}, rules + i));
    }

    for (unsigned order = MIN_ORDER; order <= MAX_ORDER; ++order)
    {
        for (integration_rule_type_t type = 1; type <= 2; ++type)
        {
            const integration_rule_t *rule = NULL;
            for (unsigned i = 0; i < TEST_CASES; ++i)
            {
                if (rules[i]->spec.order == order && rules[i]->spec.type == type)
                {
                    if (rule == NULL)
                    {
                        rule = rules[i];
                    }
                    else
                    {
                        TEST_ASSERTION(rule == rules[i], "Only one rule for the type and order should exist");
                    }
                }
            }
        }
    }

    for (unsigned i = 0; i < TEST_CASES; ++i)
    {
        TEST_FDG_RESULT(integration_rule_registry_release_rule(registry, rules[i]));
        if (i % (TEST_CASES / 10) == 0)
            integration_rule_registry_release_unused_rules(registry);

        rules[i] = NULL;
    }

    integration_rule_registry_release_unused_rules(registry);

    integration_rule_registry_destroy(registry);
    return 0;
}
