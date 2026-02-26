
#ifndef INTERPLIB_INTEGRATION_RULES_H
#define INTERPLIB_INTEGRATION_RULES_H
#include "../common/error.h"
#include <cutl/allocators.h>

typedef enum
{
    INTEGRATION_RULE_TYPE_NONE = 0,
    INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE,
    INTEGRATION_RULE_TYPE_GAUSS_LOBATTO,
} integration_rule_type_t;

INTERPLIB_INTERNAL
const char *integration_rule_type_to_str(integration_rule_type_t type);

typedef struct
{
    integration_rule_type_t type; // Type of the integration rule
    unsigned order;               // Order of the integration rule
} integration_spec_t;

typedef struct
{
    integration_spec_t spec;
    unsigned accuracy; // Order of polynomial which is exactly integrated
    unsigned n_nodes;  // Number of nodes and weights
    double _data[];    // Array with nodes, followed by weights
} integration_rule_t;

static inline double *integration_rule_nodes(integration_rule_t *this)
{
    return this->_data + 0;
}

static inline const double *integration_rule_nodes_const(const integration_rule_t *this)
{
    return this->_data + 0;
}

static inline double *integration_rule_weights(integration_rule_t *this)
{
    return this->_data + this->n_nodes;
}

static inline const double *integration_rule_weights_const(const integration_rule_t *this)
{
    return this->_data + this->n_nodes;
}

INTERPLIB_INTERNAL
interp_result_t integration_rule_for_accuracy(integration_rule_t **out, integration_rule_type_t type, unsigned accuracy,
                                              const cutl_allocator_t *allocator);

INTERPLIB_INTERNAL
interp_result_t integration_rule_for_order(integration_rule_t **out, integration_rule_type_t type, unsigned order,
                                           const cutl_allocator_t *allocator);

typedef struct integration_rule_registry_t integration_rule_registry_t;

/**
 * @brief Initializes an integration rule registry.
 *
 * This function allocates and initializes an `integration_rule_registry_t` object
 * to store integration rule data. The registry will be allocated using the specified
 * allocator.
 *
 * @param[out] out Pointer to an `integration_rule_registry_t*` that will be initialized.
 *                 On success, it points to the newly allocated and initialized registry object.
 * @param[in] should_cache If non-zero, then integration rules are cached and not freed when unused until explicitly
 *                         cleared.
 * @param[in] allocator Pointer to an `cutl_allocator_t` structure for custom
 *                      memory allocation, reallocation, and deallocation operations.
 *
 * @return `INTERP_SUCCESS` on successful initialization.
 *         `INTERP_ERROR_FAILED_ALLOCATION` if memory allocation fails.
 *
 * The caller is responsible for properly deallocating the registry using the corresponding
 * cleanup function when it is no longer necessary.
 */
INTERPLIB_INTERNAL
interp_result_t integration_rule_registry_create(integration_rule_registry_t **out, int should_cache,
                                                 const cutl_allocator_t *allocator);

/**
 * @brief Destroys an integration rule registry.
 *
 * This function releases all resources associated with the given
 * integration rule registry, including its buckets and the rules
 * contained within. Proper deallocation is performed using the
 * allocator specified during the creation of the registry.
 *
 * @param[in,out] this Pointer to the `integration_rule_registry_t`
 *                     to be destroyed. After this function is
 *                     called, the registry and its associated
 *                     resources are invalidated.
 *
 * The caller is responsible for ensuring the registry is no longer
 * in use before calling this function to avoid undefined behavior.
 */
INTERPLIB_INTERNAL
void integration_rule_registry_destroy(integration_rule_registry_t *this);

/**
 * @brief Retrieves or creates an integration rule from a registry.
 *
 * This function fetches an integration rule from the specified registry that matches
 * the provided rule specification. If the rule doesn't already exist in the registry,
 * it is created and added to the appropriate bucket within the registry.
 *
 * @param[in] this Pointer to the `integration_rule_registry_t` instance representing
 *                 the integration rule registry.
 * @param[in] spec The specification of the integration rule which includes the type
 *                 and order of the desired rule.
 * @param[out] p_rule Pointer to a location where the retrieved or newly created
 *                    `integration_rule_t` object will be stored.
 *
 * @return `INTERP_SUCCESS` if the rule is successfully retrieved or created.
 *         `INTERP_ERROR_FAILED_ALLOCATION` if memory allocation fails during the
 *         operation.
 *         Other `interp_result_t` error codes indicating issues with initialization
 *         or rule creation may also be returned.
 *
 * The caller is responsible for ensuring that the registry is initialized before calling
 * this function. The fetched or created rule should be treated as owned by the registry
 * and not freed independently.
 *
 * The rule will not be freed and will remain cached until all references to it have been removed and
 * `integration_rule_registry_release_unused_rules` has been called.
 */
INTERPLIB_INTERNAL
interp_result_t integration_rule_registry_get_rule(integration_rule_registry_t *this, integration_spec_t spec,
                                                   const integration_rule_t **p_rule);

/**
 * @brief Batched version of `integration_rule_registry_get_rule`
 *
 * @param[in] this Pointer to the `integration_rule_registry_t` instance representing
 *                 the integration rule registry.
 * @param[in] cnt Number of rules to retrieve.
 * @param[in] specs The specifications of the integration rules.
 * @param[out] p_rules Array which gets filled with pointers to `integration_rule_t` objects.
 *
 * @return `INTERP_SUCCESS` if the rule is successfully retrieved or created.
 *         `INTERP_ERROR_FAILED_ALLOCATION` if memory allocation fails during the
 *         operation.
 *         Other `interp_result_t` error codes indicating issues with initialization
 *         or rule creation may also be returned.
 *
 * The caller is responsible for ensuring that the registry is initialized before calling
 * this function. The fetched or created rule should be treated as owned by the registry
 * and not freed independently.
 *
 * The rule will not be freed and will remain cached until all references to it have been removed and
 * `integration_rule_registry_release_unused_rules` has been called.
 */
INTERPLIB_INTERNAL
interp_result_t integration_rule_registry_get_rules(integration_rule_registry_t *this, unsigned cnt,
                                                    const integration_spec_t INTERPLIB_ARRAY_ARG(specs, static cnt),
                                                    const integration_rule_t *INTERPLIB_ARRAY_ARG(p_rules, cnt));

/**
 * @brief Releases a specific integration rule from the integration rule registry.
 *
 * This function reduces the reference count of an integration rule within the
 * corresponding bucket in the registry. If the reference count of the rule reaches
 * zero, the function deallocates the rule and removes it from the registry.
 *
 * @param[in] this Pointer to the `integration_rule_registry_t` containing the rule.
 * @param[in] rule Pointer to the `integration_rule_t` to be released.
 *
 * @return `INTERP_SUCCESS` if the rule was successfully released and, if applicable, removed.
 *         `INTERP_ERROR_NOT_IN_REGISTRY` if the specified rule was not found in the registry.
 *
 * This operation might modify the internal structure of the registry, specifically the bucket
 * where the rule is located. The caller should ensure thread-safety if the registry is accessed
 * concurrently.
 */
INTERPLIB_INTERNAL
interp_result_t integration_rule_registry_release_rule(integration_rule_registry_t *this,
                                                       const integration_rule_t *rule);
/**
 * @brief Releases unused integration rules from the registry.
 *
 * This function iterates through the integration rule registry and removes
 * any rules that are no longer in use (i.e., rules with a reference count of zero).
 * Memory associated with the unused rules is deallocated using the allocator
 * provided in the integration rule registry.
 *
 * @param[in] this Pointer to an `integration_rule_registry_t` object.
 *                 This must be a valid, initialized registry. The caller
 *                 retains ownership of this object, and it must not be freed
 *                 while this function is running.
 *
 * The function updates the registry such that only the rules still in use
 * remain in the registry. Any unused rules are deallocated and removed.
 * The caller is responsible for ensuring the registry is not accessed
 * concurrently by other threads.
 */
INTERPLIB_INTERNAL
void integration_rule_registry_release_unused_rules(integration_rule_registry_t *this);

/**
 * @brief Releases all integration rules stored within the registry.
 *
 * This function deallocates and clears all integration rules contained in the buckets
 * of the specified `integration_rule_registry_t` object. After the call returns, no
 * rule from this registry is valid anymore.
 *
 * @param[in] this Pointer to a `const integration_rule_registry_t` structure, which
 *                 contains the integration rules to be released.
 *
 * The caller must ensure the validity of the input `this` pointer. This function does not
 * deallocate the registry object itself; it only removes and releases the rules within it.
 */
INTERPLIB_INTERNAL
void integration_rule_registry_release_all_rules(integration_rule_registry_t *this);

INTERPLIB_INTERNAL
unsigned integration_rule_get_rules(integration_rule_registry_t *this, unsigned max_count,
                                    integration_spec_t INTERPLIB_ARRAY_ARG(specs, max_count));

INTERPLIB_INTERNAL
unsigned integration_rule_spec_get_accuracy(integration_spec_t spec);

INTERPLIB_INTERNAL
size_t integration_specs_total_points(unsigned ndim, const integration_spec_t specs[static ndim]);

#endif // INTERPLIB_INTEGRATION_RULES_H
