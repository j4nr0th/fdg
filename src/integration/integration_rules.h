
#ifndef FDG_INTEGRATION_RULES_H
#define FDG_INTEGRATION_RULES_H
#include "../common/error.h"
#include <cutl/allocators.h>

/**
 * @brief Types of 1D quadrature rules supported by the library.
 *
 * A rule of order `order` has `order + 1` nodes. Gauss-Legendre rules
 * integrate polynomials of degree up to `2 * order + 1` exactly, while
 * Gauss-Lobatto rules additionally include the endpoints of the integration
 * interval and integrate polynomials of degree up to `2 * order - 1`
 * exactly (the single-node Gauss-Lobatto rule coincides with the one-point
 * Gauss rule and integrates degree 1).
 */
typedef enum
{
    INTEGRATION_RULE_TYPE_NONE = 0,       // No integration rule type.
    INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, // Gauss-Legendre quadrature.
    INTEGRATION_RULE_TYPE_GAUSS_LOBATTO,  // Gauss-Lobatto quadrature.
} integration_rule_type_t;

/**
 * @brief Get the name of an integration rule type.
 *
 * @param type Type to get the name for.
 * @return Statically allocated string with the name of the type, such as
 *         "gauss" or "gauss-lobatto", or "unknown" for invalid values.
 */
FDG_INTERNAL
const char *integration_rule_type_to_str(integration_rule_type_t type);

/**
 * @brief Specification of a 1D integration rule: its type and order.
 *
 * A rule of order `order` has `order + 1` nodes.
 */
typedef struct
{
    integration_rule_type_t type; // Type of the integration rule
    unsigned order;               // Order of the integration rule
} integration_spec_t;

/**
 * @brief Precomputed 1D quadrature rule: nodes and weights.
 *
 * The nodes and weights are stored in the flexible array `_data`, nodes
 * first followed by weights, each with `n_nodes` entries. Use the inline
 * accessor functions in this header instead of accessing `_data` directly.
 */
typedef struct
{
    integration_spec_t spec;
    unsigned accuracy; // Order of polynomial which is exactly integrated
    unsigned n_nodes;  // Number of nodes and weights
    double _data[];    // Array with nodes, followed by weights
} integration_rule_t;

/**
 * @brief Get a pointer to the nodes of the rule.
 *
 * @param this Rule to get the nodes of.
 * @return Pointer to the `n_nodes` nodes of the rule.
 */
static inline double *integration_rule_nodes(integration_rule_t *this)
{
    return this->_data + 0;
}

/**
 * @brief Get a const pointer to the nodes of the rule.
 *
 * @param this Rule to get the nodes of.
 * @return Pointer to the `n_nodes` nodes of the rule.
 */
static inline const double *integration_rule_nodes_const(const integration_rule_t *this)
{
    return this->_data + 0;
}

/**
 * @brief Get a pointer to the weights of the rule.
 *
 * @param this Rule to get the weights of.
 * @return Pointer to the `n_nodes` weights of the rule.
 */
static inline double *integration_rule_weights(integration_rule_t *this)
{
    return this->_data + this->n_nodes;
}

/**
 * @brief Get a const pointer to the weights of the rule.
 *
 * @param this Rule to get the weights of.
 * @return Pointer to the `n_nodes` weights of the rule.
 */
static inline const double *integration_rule_weights_const(const integration_rule_t *this)
{
    return this->_data + this->n_nodes;
}

/**
 * @brief Create an integration rule that integrates polynomials of the given degree exactly.
 *
 * The rule is created with the smallest order whose accuracy is at least the
 * requested one, according to integration_rule_spec_get_accuracy.
 *
 * @param out Receives the pointer to the newly created rule on success.
 * @param type Type of the rule.
 * @param accuracy Degree of the polynomial the rule must integrate exactly.
 * @param allocator Allocator used to allocate the rule.
 * @return FDG_SUCCESS on success, FDG_ERROR_INVALID_ENUM if the type is not
 *         supported, FDG_ERROR_FAILED_ALLOCATION if memory allocation fails.
 *         On failure, `*out` is left unmodified.
 *
 * The caller owns the created rule and is responsible for deallocating it
 * with the same allocator once it is no longer needed.
 */
FDG_INTERNAL
fdg_result_t integration_rule_for_accuracy(integration_rule_t **out, integration_rule_type_t type, unsigned accuracy,
                                           const cutl_allocator_t *allocator);

/**
 * @brief Create an integration rule of the given order.
 *
 * The rule has `order + 1` nodes and weights, computed with a tolerance of
 * 1e-14 and at most 1000 Newton iterations per node.
 *
 * @param out Receives the pointer to the newly created rule on success.
 * @param type Type of the rule.
 * @param order Order of the rule; the rule has `order + 1` nodes.
 * @param allocator Allocator used to allocate the rule.
 * @return FDG_SUCCESS on success, FDG_ERROR_INVALID_ENUM if the type is not
 *         supported, FDG_ERROR_FAILED_ALLOCATION if memory allocation fails.
 *         On failure, `*out` is left unmodified.
 *
 * The caller owns the created rule and is responsible for deallocating it
 * with the same allocator once it is no longer needed.
 */
FDG_INTERNAL
fdg_result_t integration_rule_for_order(integration_rule_t **out, integration_rule_type_t type, unsigned order,
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
 * @return `FDG_SUCCESS` on successful initialization.
 *         `FDG_ERROR_FAILED_ALLOCATION` if memory allocation fails.
 *
 * The caller is responsible for properly deallocating the registry using the corresponding
 * cleanup function when it is no longer necessary.
 */
FDG_INTERNAL
fdg_result_t integration_rule_registry_create(integration_rule_registry_t **out, int should_cache,
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
FDG_INTERNAL
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
 * @return `FDG_SUCCESS` if the rule is successfully retrieved or created.
 *         `FDG_ERROR_FAILED_ALLOCATION` if memory allocation fails during the
 *         operation.
 *         Other `fdg_result_t` error codes indicating issues with initialization
 *         or rule creation may also be returned.
 *
 * The caller is responsible for ensuring that the registry is initialized before calling
 * this function. The fetched or created rule should be treated as owned by the registry
 * and not freed independently.
 *
 * The rule will not be freed and will remain cached until all references to it have been removed and
 * `integration_rule_registry_release_unused_rules` has been called.
 */
FDG_INTERNAL
fdg_result_t integration_rule_registry_get_rule(integration_rule_registry_t *this, integration_spec_t spec,
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
 * @return `FDG_SUCCESS` if the rule is successfully retrieved or created.
 *         `FDG_ERROR_FAILED_ALLOCATION` if memory allocation fails during the
 *         operation.
 *         Other `fdg_result_t` error codes indicating issues with initialization
 *         or rule creation may also be returned.
 *
 * The caller is responsible for ensuring that the registry is initialized before calling
 * this function. The fetched or created rule should be treated as owned by the registry
 * and not freed independently.
 *
 * The rule will not be freed and will remain cached until all references to it have been removed and
 * `integration_rule_registry_release_unused_rules` has been called.
 */
FDG_INTERNAL
fdg_result_t integration_rule_registry_get_rules(integration_rule_registry_t *this, unsigned cnt,
                                                 const integration_spec_t FDG_ARRAY_ARG(specs, static cnt),
                                                 const integration_rule_t *FDG_ARRAY_ARG(p_rules, cnt));

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
 * @return `FDG_SUCCESS` if the rule was successfully released and, if applicable, removed.
 *         `FDG_ERROR_NOT_IN_REGISTRY` if the specified rule was not found in the registry.
 *
 * This operation might modify the internal structure of the registry, specifically the bucket
 * where the rule is located. The caller should ensure thread-safety if the registry is accessed
 * concurrently.
 */
FDG_INTERNAL
fdg_result_t integration_rule_registry_release_rule(integration_rule_registry_t *this, const integration_rule_t *rule);
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
FDG_INTERNAL
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
FDG_INTERNAL
void integration_rule_registry_release_all_rules(integration_rule_registry_t *this);

/**
 * @brief Get the specifications of all rules in the registry.
 *
 * @param this Registry to query.
 * @param max_count Maximum number of specifications to write.
 * @param specs Array of `max_count` entries which receives the
 *        specifications of the rules.
 * @return The total number of rules in the registry, which may exceed
 *         `max_count`; in that case only the first `max_count` entries are
 *         written.
 */
FDG_INTERNAL
unsigned integration_rule_get_rules(integration_rule_registry_t *this, unsigned max_count,
                                    integration_spec_t FDG_ARRAY_ARG(specs, max_count));

/**
 * @brief Get the polynomial degree that a rule with the given specification integrates exactly.
 *
 * @param spec Specification of the rule.
 * @return The accuracy: `2 * order + 1` for Gauss-Legendre rules, and
 *         `2 * order - 1` for Gauss-Lobatto rules of positive order (the
 *         order-zero Gauss-Lobatto rule coincides with the one-point Gauss
 *         rule and integrates degree 1). Returns 0 for invalid types.
 */
FDG_INTERNAL
unsigned integration_rule_spec_get_accuracy(integration_spec_t spec);

/**
 * @brief Compute the total number of quadrature points of a tensor-product rule.
 *
 * @param ndim Number of dimensions of the tensor product.
 * @param specs Array of `ndim` integration specifications.
 * @return The product of the node counts, i.e.
 *         `prod_i (specs[i].order + 1)`.
 */
FDG_INTERNAL
size_t integration_specs_total_points(unsigned ndim, const integration_spec_t specs[static ndim]);

#endif // FDG_INTEGRATION_RULES_H
