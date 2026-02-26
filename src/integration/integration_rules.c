//
// Created by jan on 2025-09-07.
//

#include "../common/rw_lock.h"

#include "integration_rules.h"

#include "gauss_legendre.h"
#include "gauss_lobatto.h"

#include <string.h>
#include <threads.h>

interp_result_t integration_rule_for_accuracy(integration_rule_t **out, const integration_rule_type_t type,
                                              const unsigned accuracy, const cutl_allocator_t *allocator)
{
    unsigned required_order;
    switch (type)
    {
    case INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE:
        if (accuracy < 2)
        {
            required_order = 1;
            break;
        }
        required_order = accuracy / 2 + 1;
        break;

    case INTEGRATION_RULE_TYPE_GAUSS_LOBATTO:
        required_order = accuracy / 2 + 2;
        break;

    default:
        return INTERP_ERROR_INVALID_ENUM;
    }

    return integration_rule_for_order(out, type, required_order, allocator);
}
interp_result_t integration_rule_for_order(integration_rule_t **out, const integration_rule_type_t type,
                                           const unsigned order, const cutl_allocator_t *allocator)
{
    integration_rule_t *const this = cutl_alloc(allocator, sizeof *this + 2 * (order + 1) * sizeof *this->_data);
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;
    this->n_nodes = order + 1;
    this->spec = (integration_spec_t){.type = type, .order = order};

    const double DEFAULT_TOLERANCE = 1e-14;
    enum
    {
        DEFAULT_MAX_ITERATIONS = 1000
    };

    switch (type)
    {
    case INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE:
        this->accuracy = order > 0 ? 2 * order - 1 : 0;
        gauss_legendre_nodes_weights(order + 1, DEFAULT_TOLERANCE, DEFAULT_MAX_ITERATIONS, integration_rule_nodes(this),
                                     integration_rule_weights(this));
        break;
    case INTEGRATION_RULE_TYPE_GAUSS_LOBATTO:
        this->accuracy = order > 1 ? 2 * order - 3 : order;
        gauss_lobatto_nodes_weights(order + 1, DEFAULT_TOLERANCE, DEFAULT_MAX_ITERATIONS, integration_rule_nodes(this),
                                    integration_rule_weights(this));
        break;
    default:
        return INTERP_ERROR_INVALID_ENUM;
    }

    *out = this;
    return INTERP_SUCCESS;
}

typedef struct
{
    integration_rule_type_t type; // type of integration rules in this bucket
    unsigned count;               // number of rules in the bucket
    unsigned capacity;            // capacity of the bucket
    integration_rule_t **rules;   // rules in the bucket
    unsigned *ref_counts;         // reference counts for each rule
} integration_rule_type_bucket_t;

static inline interp_result_t integration_rule_type_bucket_init(integration_rule_type_bucket_t *this,
                                                                const integration_rule_type_t type,
                                                                const unsigned starting_size,
                                                                const cutl_allocator_t *allocator)
{
    this->type = type;
    this->count = 0;
    this->capacity = starting_size;
    this->rules = cutl_alloc(allocator, starting_size * sizeof *this->rules);
    if (!this->rules)
        return INTERP_ERROR_FAILED_ALLOCATION;

    this->ref_counts = cutl_alloc(allocator, starting_size * sizeof *this->ref_counts);
    if (!this->ref_counts)
    {
        cutl_dealloc(allocator, this->rules);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    memset(this->ref_counts, 0, starting_size * sizeof *this->ref_counts);

    return INTERP_SUCCESS;
}

static inline void integration_rule_type_bucket_destroy(integration_rule_type_bucket_t *this,
                                                        const cutl_allocator_t *allocator)
{
    for (unsigned i = 0; i < this->count; ++i)
    {
        cutl_dealloc(allocator, this->rules[i]);
    }
    cutl_dealloc(allocator, this->rules);
    cutl_dealloc(allocator, this->ref_counts);
    *this = (integration_rule_type_bucket_t){};
}

static inline interp_result_t integration_rule_type_bucket_add_rule(integration_rule_type_bucket_t *this,
                                                                    integration_rule_t *rule,
                                                                    const cutl_allocator_t *allocator)
{
    ASSERT(rule->spec.type == this->type, "Rule type does not match bucket type.");
    if (this->count == this->capacity)
    {
        const unsigned new_capacity = this->capacity * 2;
        integration_rule_t **new_rules = cutl_realloc(allocator, this->rules, new_capacity * sizeof *new_rules);
        if (!new_rules)
            return INTERP_ERROR_FAILED_ALLOCATION;
        this->rules = new_rules;

        unsigned *new_ref_counts = cutl_realloc(allocator, this->ref_counts, new_capacity * sizeof *new_ref_counts);
        if (!new_ref_counts)
            return INTERP_ERROR_FAILED_ALLOCATION;
        this->ref_counts = new_ref_counts;
        this->capacity = new_capacity;
    }

    this->rules[this->count] = rule;
    this->ref_counts[this->count] = 1;
    this->count += 1;

    return INTERP_SUCCESS;
}

struct integration_rule_registry_t
{
    rw_lock_t lock;                          // Lock for the registry
    cutl_allocator_t allocator;              // Allocator to use
    int should_cache;                        // Whether to cache rules
    unsigned n_buckets;                      // Number of buckets
    integration_rule_type_bucket_t *buckets; // Bucket array
};

INTERPLIB_INTERNAL
interp_result_t integration_rule_registry_create(integration_rule_registry_t **out, const int should_cache,
                                                 const cutl_allocator_t *allocator)
{
    integration_rule_registry_t *const this = cutl_alloc(allocator, sizeof *this);
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;
    *this = (integration_rule_registry_t){
        .allocator = *allocator, .should_cache = (should_cache != 0), .n_buckets = 0, .buckets = NULL};
    const interp_result_t res = rw_lock_init(&this->lock);
    if (res != INTERP_SUCCESS)
    {
        cutl_dealloc(allocator, this);
        return res;
    }
    *out = this;
    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
void integration_rule_registry_destroy(integration_rule_registry_t *this)
{
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        integration_rule_type_bucket_destroy(&this->buckets[i], &this->allocator);
    }
    rw_lock_destroy(&this->lock);
    cutl_dealloc(&this->allocator, this->buckets);
    cutl_dealloc(&this->allocator, this);
}

INTERPLIB_INTERNAL
interp_result_t integration_rule_registry_get_rule(integration_rule_registry_t *this, const integration_spec_t spec,
                                                   const integration_rule_t **p_rule)
{
    rw_lock_acquire_read(&this->lock);
    integration_rule_type_bucket_t *bucket = NULL;
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        if (this->buckets[i].type == spec.type)
        {
            bucket = &this->buckets[i];
            break;
        }
    }
    if (!bucket)
    {
        rw_lock_release_read(&this->lock);
        rw_lock_acquire_write(&this->lock);

        integration_rule_type_bucket_t *const new_buckets =
            cutl_realloc(&this->allocator, this->buckets, (this->n_buckets + 1) * sizeof *new_buckets);
        if (!new_buckets)
            return INTERP_ERROR_FAILED_ALLOCATION;
        this->buckets = new_buckets;
        enum
        {
            BUCKET_STARTING_SIZE = 8
        };
        interp_result_t result = integration_rule_type_bucket_init(this->buckets + this->n_buckets, spec.type,
                                                                   BUCKET_STARTING_SIZE, &this->allocator);
        if (result != INTERP_SUCCESS)
            return result;
        bucket = this->buckets + this->n_buckets;
        this->n_buckets += 1;

        integration_rule_t *rule;
        result = integration_rule_for_order(&rule, spec.type, spec.order, &this->allocator);
        if (result != INTERP_SUCCESS)
            return result;
        result = integration_rule_type_bucket_add_rule(bucket, rule, &this->allocator);
        if (result != INTERP_SUCCESS)
            return result;
        *p_rule = rule;

        rw_lock_release_write(&this->lock);
        return INTERP_SUCCESS;
    }

    for (unsigned i = 0; i < bucket->count; ++i)
    {
        if (bucket->rules[i]->spec.order == spec.order)
        {
            bucket->ref_counts[i] += 1;
            *p_rule = bucket->rules[i];
            rw_lock_release_read(&this->lock);
            return INTERP_SUCCESS;
        }
    }

    rw_lock_release_read(&this->lock);
    rw_lock_acquire_write(&this->lock);

    integration_rule_t *rule;
    interp_result_t result = integration_rule_for_order(&rule, spec.type, spec.order, &this->allocator);
    if (result != INTERP_SUCCESS)
    {
        rw_lock_release_write(&this->lock);
        return result;
    }

    result = integration_rule_type_bucket_add_rule(bucket, rule, &this->allocator);
    if (result == INTERP_SUCCESS)
        *p_rule = rule;

    rw_lock_release_write(&this->lock);
    return INTERP_SUCCESS;
}

interp_result_t integration_rule_registry_get_rules(integration_rule_registry_t *this, const unsigned cnt,
                                                    const integration_spec_t INTERPLIB_ARRAY_ARG(specs, static cnt),
                                                    const integration_rule_t *INTERPLIB_ARRAY_ARG(p_rules, cnt))
{
    for (unsigned i = 0; i < cnt; ++i)
    {
        const integration_rule_t *rule;
        const interp_result_t res = integration_rule_registry_get_rule(this, specs[i], &rule);
        if (res != INTERP_SUCCESS)
        {
            // Release all rules that were retrieved thus far
            for (unsigned j = 0; j < i; ++j)
            {
                integration_rule_registry_release_rule(this, p_rules[j]);
            }
            // Now we can return.
            return res;
        }
        p_rules[i] = rule;
    }

    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
interp_result_t integration_rule_registry_release_rule(integration_rule_registry_t *this,
                                                       const integration_rule_t *rule)
{
    rw_lock_acquire_read(&this->lock);
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        integration_rule_type_bucket_t *const bucket = this->buckets + i;

        // It can only be in the bucket of the correct type
        if (bucket->type != rule->spec.type)
            continue;

        for (unsigned j = 0; j < bucket->count; ++j)
        {
            if (bucket->rules[j] == rule)
            {
                rw_lock_release_read(&this->lock);
                rw_lock_acquire_write(&this->lock);
                bucket->ref_counts[j] -= 1;
                if (bucket->ref_counts[j] == 0 && this->should_cache == 0)
                {
                    cutl_dealloc(&this->allocator, bucket->rules[j]);
                    memmove(bucket->rules + j, bucket->rules + j + 1, (bucket->count - j - 1) * sizeof *bucket->rules);
                    bucket->count -= 1;
                }
                rw_lock_release_write(&this->lock);
                return INTERP_SUCCESS;
            }
        }
    }

    rw_lock_release_read(&this->lock);
    return INTERP_ERROR_NOT_IN_REGISTRY;
}

INTERPLIB_INTERNAL
void integration_rule_registry_release_unused_rules(integration_rule_registry_t *this)
{
    rw_lock_acquire_write(&this->lock);
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        integration_rule_type_bucket_t *const bucket = this->buckets + i;
        for (unsigned j = 0; j < bucket->count; ++j)
        {
            if (bucket->ref_counts[j] == 0)
            {
                cutl_dealloc(&this->allocator, bucket->rules[j]);
            }
        }
        unsigned pos, valid;
        for (pos = 0, valid = 0; pos < bucket->count; ++pos)
        {
            if (bucket->ref_counts[pos] > 0)
            {
                bucket->rules[valid] = bucket->rules[pos];
                bucket->ref_counts[valid] = bucket->ref_counts[pos];
                valid += 1;
            }
        }
        bucket->count = valid;
    }
    rw_lock_release_write(&this->lock);
}
void integration_rule_registry_release_all_rules(integration_rule_registry_t *this)
{
    rw_lock_acquire_write(&this->lock);
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        integration_rule_type_bucket_t *const bucket = this->buckets + i;
        for (unsigned j = 0; j < bucket->count; ++j)
        {
            cutl_dealloc(&this->allocator, bucket->rules[j]);
        }
        bucket->count = 0;
    }
    rw_lock_release_write(&this->lock);
}

unsigned integration_rule_get_rules(integration_rule_registry_t *this, const unsigned max_count,
                                    integration_spec_t INTERPLIB_ARRAY_ARG(specs, max_count))
{
    rw_lock_acquire_read(&this->lock);
    unsigned count = 0;
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        const integration_rule_type_bucket_t *const bucket = this->buckets + i;
        for (unsigned j = 0; j < bucket->count; ++j)
        {
            if (count < max_count)
            {
                specs[count] = bucket->rules[j]->spec;
            }
            count += 1;
        }
    }
    rw_lock_release_read(&this->lock);

    return count;
}
unsigned integration_rule_spec_get_accuracy(const integration_spec_t spec)
{
    switch (spec.type)
    {
    case INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE:
        if (spec.order > 0)
            return 2 * spec.order - 1;
        return 1;
    case INTEGRATION_RULE_TYPE_GAUSS_LOBATTO:
        if (spec.order > 2)
            return 2 * spec.order - 3;
        return 1;
    default:
        return 0;
    }
}

size_t integration_specs_total_points(const unsigned ndim, const integration_spec_t specs[static const ndim])
{
    size_t total = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        total *= specs[i].order + 1;
    }
    return total;
}

const char *integration_rule_type_to_str(const integration_rule_type_t type)
{
    switch (type)
    {
    case INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE:
        return "gauss";
    case INTEGRATION_RULE_TYPE_GAUSS_LOBATTO:
        return "gauss-lobatto";
    default:
        return "unknown";
    }
}
