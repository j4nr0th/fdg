//
// Created by jan on 2025-09-09.
//

#include "basis_set.h"
#include "../common/rw_lock.h"

#include "../integration/gauss_legendre.h"
#include "../integration/gauss_lobatto.h"
#include "../polynomials/bernstein.h"
#include "../polynomials/lagrange.h"
#include "../polynomials/legendre.h"
#include "basis_bernstein.h"
#include "basis_lagrange.h"
#include "basis_legendre.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

// Bucket containing all basis sets of the same integration type
typedef struct
{
    integration_rule_type_t type;
    unsigned count;
    unsigned capacity;
    _Atomic unsigned *ref_counts;
    basis_set_t **basis_sets;
} basis_bucket_itype_t;

// Bucket containing all basis sets of the same basis_sets type
typedef struct
{
    basis_set_type_t type;
    unsigned count;
    unsigned capacity;
    basis_bucket_itype_t *buckets;
} basis_bucket_btype_t;

struct basis_set_registry_t
{
    rw_lock_t lock;
    cutl_allocator_t allocator;
    int should_cache;
    unsigned n_buckets;
    basis_bucket_btype_t *buckets;
};

interp_result_t basis_set_registry_create(basis_set_registry_t **out, int should_cache,
                                          const cutl_allocator_t *allocator)
{
    basis_set_registry_t *const this = cutl_alloc(allocator, sizeof *this);
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;
    *this = (basis_set_registry_t){
        .allocator = *allocator, .should_cache = should_cache != 0, .n_buckets = 0, .buckets = NULL};
    const interp_result_t res = rw_lock_init(&this->lock);
    if (res != INTERP_SUCCESS)
    {
        cutl_dealloc(allocator, this);
        return res;
    }

    *out = this;
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_bucket_btype_init(basis_bucket_btype_t *this, basis_set_type_t type,
                                                          unsigned starting_size, const cutl_allocator_t *allocator)
{
    this->type = type;
    this->count = 0;
    this->capacity = starting_size;
    this->buckets = cutl_alloc(allocator, starting_size * sizeof *this->buckets);
    if (!this->buckets)
    {
        this->capacity = 0;
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_bucket_itype_init(basis_bucket_itype_t *this, integration_rule_type_t type,
                                                          unsigned starting_size, const cutl_allocator_t *allocator)
{
    this->type = type;
    this->count = 0;
    this->capacity = starting_size;
    this->basis_sets = cutl_alloc(allocator, starting_size * sizeof *this->basis_sets);
    if (!this->basis_sets)
    {
        this->capacity = 0;
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    this->ref_counts = cutl_alloc(allocator, starting_size * sizeof *this->ref_counts);
    if (!this->ref_counts)
    {
        cutl_dealloc(allocator, this->basis_sets);
        this->capacity = 0;
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_create(basis_set_t **out, const integration_rule_t *integration_rule,
                                               const basis_spec_t spec, const cutl_allocator_t *allocator)
{
    interp_result_t res;
    switch (spec.type)
    {
    case BASIS_LEGENDRE:
        res = legendre_basis_create(out, spec, integration_rule, allocator);
        break;

    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        res = lagrange_basis_create(out, spec, integration_rule, allocator);
        break;

    case BASIS_BERNSTEIN:
        res = bernstein_basis_create(out, spec, integration_rule, allocator);
        break;

    default:
        return INTERP_ERROR_INVALID_ENUM;
    }

    return res;
}

static inline interp_result_t basis_set_registry_add_btype_bucket(basis_set_registry_t *this,
                                                                  const basis_set_type_t type,
                                                                  const unsigned starting_size)
{
    basis_bucket_btype_t *new_buckets =
        cutl_realloc(&this->allocator, this->buckets, (this->n_buckets + 1) * sizeof *new_buckets);
    if (!new_buckets)
    {
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    this->buckets = new_buckets;
    basis_bucket_btype_t *const first_bucket = this->buckets + this->n_buckets;
    this->n_buckets += 1;
    const interp_result_t res = basis_set_bucket_btype_init(first_bucket, type, starting_size, &this->allocator);
    if (res != INTERP_SUCCESS)
    {
        return res;
    }
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_registry_add_itype_bucket(const basis_set_registry_t *this,
                                                                  basis_bucket_btype_t *first_bucket,
                                                                  const integration_rule_type_t type,
                                                                  const unsigned starting_size)
{
    if (first_bucket->count == first_bucket->capacity)
    {
        const unsigned new_capacity = first_bucket->capacity * 2;
        basis_bucket_itype_t *const new_buckets =
            cutl_realloc(&this->allocator, first_bucket->buckets, new_capacity * sizeof *new_buckets);
        if (!new_buckets)
            return INTERP_ERROR_FAILED_ALLOCATION;
        first_bucket->buckets = new_buckets;
        first_bucket->capacity = new_capacity;
    }
    basis_bucket_itype_t *const second_bucket = first_bucket->buckets + first_bucket->count;
    first_bucket->count += 1;
    const interp_result_t res = basis_set_bucket_itype_init(second_bucket, type, starting_size, &this->allocator);
    if (res != INTERP_SUCCESS)
        return res;
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_registry_add_basis_set(const basis_set_registry_t *this,
                                                               basis_bucket_itype_t *second_bucket,
                                                               const basis_set_t **p_basis,
                                                               const integration_rule_t *integration_rule,
                                                               const basis_spec_t spec)
{
    if (second_bucket->count == second_bucket->capacity)
    {
        const unsigned new_capacity = second_bucket->capacity * 2;
        basis_set_t **const new_basis =
            cutl_realloc(&this->allocator, second_bucket->basis_sets, new_capacity * sizeof *new_basis);
        if (!new_basis)
            return INTERP_ERROR_FAILED_ALLOCATION;
        second_bucket->basis_sets = new_basis;
        _Atomic unsigned *const new_ref_counts =
            cutl_realloc(&this->allocator, second_bucket->ref_counts, new_capacity * sizeof *new_ref_counts);
        if (!new_ref_counts)
            return INTERP_ERROR_FAILED_ALLOCATION;
        second_bucket->ref_counts = new_ref_counts;
        second_bucket->capacity = new_capacity;
    }

    basis_set_t *basis;
    const interp_result_t res = basis_set_create(&basis, integration_rule, spec, &this->allocator);
    if (res != INTERP_SUCCESS)
        return res;

    second_bucket->basis_sets[second_bucket->count] = basis;
    second_bucket->ref_counts[second_bucket->count] = 1;
    second_bucket->count += 1;

    *p_basis = basis;
    return INTERP_SUCCESS;
}

interp_result_t basis_set_registry_get_basis_set(basis_set_registry_t *this, const basis_set_t **p_basis,
                                                 const integration_rule_t *integration_rule, const basis_spec_t spec)
{
    rw_lock_acquire_read(&this->lock);
    enum
    {
        BUCKET_STARTING_SIZE = 8
    };
    basis_bucket_btype_t *first_bucket = NULL;
    basis_bucket_itype_t *second_bucket = NULL;

    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        basis_bucket_btype_t *const b_bucket = this->buckets + i;
        if (b_bucket->type == spec.type)
        {
            first_bucket = b_bucket;
            for (unsigned j = 0; j < first_bucket->count; ++j)
            {
                basis_bucket_itype_t *const i_bucket = first_bucket->buckets + j;
                if (i_bucket->type == integration_rule->spec.type)
                {
                    second_bucket = i_bucket;
                    for (unsigned k = 0; k < second_bucket->count; ++k)
                    {
                        const basis_set_t *const basis = second_bucket->basis_sets[k];
                        if (basis->spec.order == spec.order &&
                            basis->integration_spec.order == integration_rule->spec.order)
                        {
                            second_bucket->ref_counts[k] += 1;
                            *p_basis = basis;
                            rw_lock_release_read(&this->lock);
                            return INTERP_SUCCESS;
                        }
                    }
                    break;
                }
            }
            break;
        }
    }

    rw_lock_release_read(&this->lock);
    rw_lock_acquire_write(&this->lock);
    if (!first_bucket)
    {
        const interp_result_t res = basis_set_registry_add_btype_bucket(this, spec.type, BUCKET_STARTING_SIZE);
        if (res != INTERP_SUCCESS)
        {
            rw_lock_release_write(&this->lock);
            return res;
        }
        first_bucket = this->buckets + this->n_buckets - 1;
    }
    if (!second_bucket)
    {
        const interp_result_t res =
            basis_set_registry_add_itype_bucket(this, first_bucket, integration_rule->spec.type, BUCKET_STARTING_SIZE);
        if (res != INTERP_SUCCESS)
        {
            rw_lock_release_write(&this->lock);
            return res;
        }
        second_bucket = first_bucket->buckets + first_bucket->count - 1;
    }

    const basis_set_t *basis;
    const interp_result_t res = basis_set_registry_add_basis_set(this, second_bucket, &basis, integration_rule, spec);
    rw_lock_release_write(&this->lock);
    if (res != INTERP_SUCCESS)
    {
        return res;
    }
    *p_basis = basis;
    return INTERP_SUCCESS;
}

interp_result_t basis_set_registry_get_basis_sets(basis_set_registry_t *this, unsigned cnt,
                                                  const basis_set_t *INTERPLIB_ARRAY_ARG(p_basis, cnt),
                                                  const integration_rule_t *INTERPLIB_ARRAY_ARG(integration_rule,
                                                                                                static cnt),
                                                  const basis_spec_t INTERPLIB_ARRAY_ARG(specs, static cnt))
{
    for (unsigned i = 0; i < cnt; ++i)
    {
        const basis_set_t *basis;
        const interp_result_t res = basis_set_registry_get_basis_set(this, &basis, integration_rule[i], specs[i]);
        if (res != INTERP_SUCCESS)
        {
            // Release all the basis acquired thus far
            for (unsigned j = 0; j < i; ++j)
                basis_set_registry_release_basis_set(this, p_basis[j]);
            return res;
        }
        p_basis[i] = basis;
    }

    return INTERP_SUCCESS;
}

interp_result_t basis_set_registry_release_basis_set(basis_set_registry_t *this, const basis_set_t *basis)
{
    rw_lock_acquire_read(&this->lock);
    basis_bucket_btype_t *first_bucket = NULL;
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        if (this->buckets[i].type == basis->spec.type)
        {
            first_bucket = this->buckets + i;
            break;
        }
    }
    if (!first_bucket)
    {
        rw_lock_release_read(&this->lock);
        return INTERP_ERROR_NOT_IN_REGISTRY;
    }

    basis_bucket_itype_t *second_bucket = NULL;
    for (unsigned i = 0; i < first_bucket->count; ++i)
    {
        if (first_bucket->buckets[i].type == basis->integration_spec.type)
        {
            second_bucket = first_bucket->buckets + i;
            break;
        }
    }
    if (!second_bucket)
    {
        rw_lock_release_read(&this->lock);
        return INTERP_ERROR_NOT_IN_REGISTRY;
    }

    for (unsigned position = 0; position < second_bucket->count; ++position)
    {
        if (second_bucket->basis_sets[position] == basis)
        {
            rw_lock_release_read(&this->lock);
            rw_lock_acquire_write(&this->lock);
            second_bucket->ref_counts[position] -= 1;
            if (second_bucket->ref_counts[position] == 0 && this->should_cache == 0)
            {
                cutl_dealloc(&this->allocator, second_bucket->basis_sets[position]);
                memmove(second_bucket->basis_sets + position, second_bucket->basis_sets + position + 1,
                        sizeof(*second_bucket->basis_sets) * (second_bucket->count - position - 1));
                second_bucket->count -= 1;
            }
            rw_lock_release_write(&this->lock);
            return INTERP_SUCCESS;
        }
    }

    rw_lock_release_read(&this->lock);
    return INTERP_ERROR_NOT_IN_REGISTRY;
}

void basis_set_registry_destroy(basis_set_registry_t *this)
{
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        basis_bucket_btype_t *first_bucket = this->buckets + i;
        for (unsigned j = 0; j < first_bucket->count; ++j)
        {
            basis_bucket_itype_t *second_bucket = first_bucket->buckets + j;
            for (unsigned k = 0; k < second_bucket->count; ++k)
            {
                cutl_dealloc(&this->allocator, second_bucket->basis_sets[k]);
            }
            cutl_dealloc(&this->allocator, second_bucket->basis_sets);
            cutl_dealloc(&this->allocator, second_bucket->ref_counts);
        }
        cutl_dealloc(&this->allocator, first_bucket->buckets);
    }
    cutl_dealloc(&this->allocator, this->buckets);
    cutl_dealloc(&this->allocator, this);
}

void basis_set_registry_release_unused_basis_sets(basis_set_registry_t *const this)
{
    rw_lock_acquire_write(&this->lock);
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        const basis_bucket_btype_t *const first_bucket = this->buckets + i;
        for (unsigned j = 0; j < first_bucket->count; ++j)
        {
            basis_bucket_itype_t *const second_bucket = first_bucket->buckets + j;
            for (unsigned k = 0; k < second_bucket->count; ++k)
            {
                if (second_bucket->ref_counts[k] == 0 && this->should_cache == 0)
                {
                    cutl_dealloc(&this->allocator, second_bucket->basis_sets[k]);
                    second_bucket->basis_sets[k] = NULL;
                }
            }
            unsigned pos, valid;
            for (pos = 0, valid = 0; pos < second_bucket->count; ++pos)
            {
                if (second_bucket->basis_sets[pos] != NULL)
                {
                    second_bucket->basis_sets[valid] = second_bucket->basis_sets[pos];
                    second_bucket->ref_counts[valid] = second_bucket->ref_counts[pos];
                    valid += 1;
                }
            }
            second_bucket->count = valid;
        }
    }
    rw_lock_release_write(&this->lock);
}
unsigned basis_set_registry_get_sets(basis_set_registry_t *this, unsigned max_count,
                                     basis_spec_t INTERPLIB_ARRAY_ARG(basis_spec, const max_count),
                                     integration_spec_t INTERPLIB_ARRAY_ARG(integration_spec, const max_count))
{
    rw_lock_acquire_read(&this->lock);
    unsigned count = 0;
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        const basis_bucket_btype_t *const first_bucket = this->buckets + i;
        for (unsigned j = 0; j < first_bucket->count; ++j)
        {
            const basis_bucket_itype_t *const second_bucket = first_bucket->buckets + j;
            for (unsigned k = 0; k < second_bucket->count; ++k)
            {
                const basis_set_t *const basis = second_bucket->basis_sets[k];
                if (count < max_count)
                {
                    basis_spec[count] = basis->spec;
                    integration_spec[count] = basis->integration_spec;
                }
                count += 1;
            }
        }
    }

    rw_lock_release_read(&this->lock);
    return count;
}

void basis_compute_at_point_prepare(const basis_set_type_t type, const unsigned order,
                                    double INTERPLIB_ARRAY_ARG(work, restrict order + 1))
{
    // Prepare the roots for Lagrange multipliers
    switch (type)
    {
    case BASIS_LAGRANGE_UNIFORM:
        for (unsigned i = 0; i < order + 1; ++i)
        {
            work[i] = (2.0 * i) / (double)(order)-1.0;
        }
        break;

    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        for (unsigned i = 0; i < order + 1; ++i)
        {
            work[i] = -cos(M_PI * (double)(2 * i + 1) / (double)(2 * (order + 1)));
        }
        break;

    case BASIS_LAGRANGE_GAUSS:
        // TODO: replace the hard-coded values by some parameters?
        gauss_legendre_nodes(order + 1, 1e-12, 100, work);
        break;

    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        gauss_lobatto_nodes(order + 1, 1e-12, 100, work);
        break;

    default:
        // Nothing to do here
        break;
    }
}

void basis_compute_at_point_values(const basis_set_type_t type, const unsigned order, const unsigned cnt,
                                   const double INTERPLIB_ARRAY_ARG(x, restrict static cnt),
                                   double INTERPLIB_ARRAY_ARG(out, restrict cnt *(order + 1)),
                                   double INTERPLIB_ARRAY_ARG(work, restrict order + 1))
{
    switch (type)
    {
    case BASIS_LEGENDRE:
        for (unsigned i = 0; i < cnt; ++i)
            legendre_eval_bonnet_all(order, x[i], out + (size_t)(i * (order + 1)));
        break;

    case BASIS_BERNSTEIN:
        for (unsigned i = 0; i < cnt; ++i)
            bernstein_interpolation_vector((x[i] + 1) / 2, order, out + (size_t)(i * (order + 1)));
        break;

    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        lagrange_polynomial_values_2(cnt, x, order + 1, work, out);
        break;

    default:
        break;
    }
}

void basis_compute_at_point_derivatives(const basis_set_type_t type, const unsigned order, const unsigned cnt,
                                        const double INTERPLIB_ARRAY_ARG(x, restrict static cnt),
                                        double INTERPLIB_ARRAY_ARG(out, restrict cnt *(order + 1)),
                                        double INTERPLIB_ARRAY_ARG(work, restrict order + 1))
{
    switch (type)
    {
    case BASIS_LEGENDRE:
        for (unsigned i = 0; i < cnt; ++i)
        {
            const double v = x[i];
            double *const ptr = out + (size_t)(i * (order + 1));
            // Compute values of polynomials one order less
            legendre_eval_bonnet_all(order, v, ptr);
            // Convert Legendre polynomials one order less to derivatives
            // Value of previously handled polynomial
            double val = ptr[0];
            // The first basis is a constant, so no derivative
            ptr[0] = 0;
            for (unsigned n = 0; n < order; ++n)
            {
                // Store value about to be overwritten
                const double tmp = ptr[n + 1];
                // Overwrite with the value of the derivative
                ptr[n + 1] = (n + 1) * val + v * ptr[n];
                // Carry over to the next iteration
                val = tmp;
            }
        }
        break;

    case BASIS_BERNSTEIN:
        for (unsigned i = 0; i < cnt; ++i)
        {
            double *const ptr = out + (size_t)(i * (order + 1));
            // Compute one order less
            bernstein_interpolation_vector((x[i] + 1) / 2, order - 1, ptr);
            // Convert the order n - 1 from the derivative output into the derivative in place
            {
                const unsigned n = order;
                ptr[n] = n * ptr[n - 1];
                for (unsigned j = n - 1; j > 0; --j)
                {
                    ptr[j] = n * (ptr[j - 1] - ptr[j]);
                }
                ptr[0] *= -(double)n;
                // Apply the chain rule and scale the gradient!
                for (unsigned j = 0; j < n + 1; ++j)
                {
                    ptr[j] /= 2.0;
                }
            }
        }
        break;

    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        lagrange_polynomial_first_derivative_2(cnt, x, order + 1, work, out);
        break;

    default:
        break;
    }
}

void basis_compute_outer_product_basis_required_memory(const unsigned n_basis,
                                                       const basis_spec_t INTERPLIB_ARRAY_ARG(basis_specs, n_basis),
                                                       const unsigned cnt, unsigned *out_elements,
                                                       unsigned *work_elements, unsigned *tmp_elements,
                                                       size_t *iterator_size)
{
    unsigned max_order = 1, total_basis = 1;
    for (unsigned i = 0; i < n_basis; ++i)
    {
        const unsigned i_order = basis_specs[i].order;
        max_order = max_order > i_order ? max_order : i_order;
        total_basis *= (i_order + 1);
    }
    *out_elements = cnt * total_basis;
    *work_elements = max_order + 1;
    *tmp_elements = (max_order + 1) * cnt;
    *iterator_size = multidim_iterator_needed_memory(n_basis);
}

void basis_compute_outer_product_basis(const unsigned n_basis_dims,
                                       const basis_spec_t INTERPLIB_ARRAY_ARG(basis_specs, n_basis_dims),
                                       const unsigned cnt, const double *INTERPLIB_ARRAY_ARG(x, restrict n_basis_dims),
                                       double out[restrict], double work[restrict], double tmp[restrict],
                                       multidim_iterator_t *const iter)
{
    size_t size_out = cnt, max_size_basis = 1;
    for (unsigned dim = 0; dim < n_basis_dims; ++dim)
    {
        const unsigned dim_size = basis_specs[dim].order + 1;
        multidim_iterator_init_dim(iter, dim, dim_size);
        size_out *= dim_size;
        max_size_basis = max_size_basis > dim_size ? max_size_basis : dim_size;
    }
    (void)size_out;
    (void)max_size_basis;
    // memset(out, 0, size_out * sizeof(*out));

    const size_t basis_count = multidim_iterator_total_size(iter);
    for (unsigned i_point = 0; i_point < cnt; ++i_point)
    {
        out[i_point * basis_count] = 1;
    }

    for (unsigned i_dim = 0; i_dim < n_basis_dims; ++i_dim)
    {
        const basis_spec_t *const specs = basis_specs + i_dim;
        const unsigned n_basis = specs->order + 1;
        // Prepare for computations
        basis_compute_at_point_prepare(specs->type, specs->order, work);
        // Compute point values and write it to the buffer
        basis_compute_at_point_values(specs->type, specs->order, cnt, (double *)x[i_dim], tmp, work);

        for (unsigned i_point = 0; i_point < cnt; ++i_point)
        {
            double *const ptr = out + i_point * basis_count;
            multidim_iterator_set_to_start(iter);
            while (!multidim_iterator_is_at_end(iter))
            {
                const double prev_v = ptr[multidim_iterator_get_flat_index(iter)];
                for (unsigned i_basis = 0; i_basis < n_basis; ++i_basis)
                {
                    ASSERT(!multidim_iterator_is_at_end(iter), "Iterator should not be at end at this point ()");
                    const size_t idx = multidim_iterator_get_flat_index(iter);
                    ASSERT(idx + i_point * basis_count < size_out,
                           "Iterator index should be within bounds ((%zu + %zu) vs %zu)", idx, i_point * basis_count,
                           size_out);
                    ASSERT(i_basis + n_basis * i_point < max_size_basis * cnt,
                           "Basis buffer index is out of bounds ((%u + %u) vs %zu).", i_basis, n_basis * i_point,
                           max_size_basis * cnt);
                    ptr[idx] = prev_v * tmp[i_basis + n_basis * i_point];
                    multidim_iterator_advance(iter, i_dim, 1);
                }
            }
        }
    }
}
