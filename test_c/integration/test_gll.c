
#include "../../src/integration/integration_rules.h"
#include "../common/common.h"

/**
 * Main test function to verify the numerical integration rule using the Gauss-Lobatto method.
 *
 * This function iteratively tests the integration rule for different levels of accuracy.
 * For each accuracy level, it validates the computed integral of monomials against the analytical result.
 * The test ensures that the integration rule produces results close to the expected analytical values
 * within specified tolerances using the provided testing utilities.
 *
 * @return 0 if the program completes successfully.
 */
int main()
{
    integration_rule_t *rule = NULL;
    for (unsigned accuracy = 0; accuracy < 20; ++accuracy)
    {
        TEST_FDG_RESULT(
            integration_rule_for_accuracy(&rule, INTEGRATION_RULE_TYPE_GAUSS_LOBATTO, accuracy, &TEST_ALLOCATOR));
        TEST_ASSERTION(rule->accuracy >= accuracy, "Accuracy of integration rule is not sufficient");

        for (unsigned j = 0; j <= accuracy; ++j)
        {
            const double analytical = 2.0 / (j + 1) * ((j + 1) & 1);
            double computed = 0.0;
            for (unsigned i_node = 0; i_node < rule->n_nodes; ++i_node)
            {
                double x = 1;
                for (unsigned k = 0; k < j; ++k)
                {
                    x *= integration_rule_nodes_const(rule)[i_node];
                }
                computed += integration_rule_weights_const(rule)[i_node] * x;
            }
            TEST_NUMBERS_CLOSE(computed, analytical, 1e-12, 1e-10);
        }

        cutl_dealloc(&TEST_ALLOCATOR, rule);
    }

    return 0;
}
