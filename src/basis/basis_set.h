#pragma once
#include "../integration/integration_rules.h"
#include <cutl/iterators/multidim_iteration.h>
#include <stdalign.h>
#include <string.h>

/**
 * @brief Types of 1D basis functions supported by the library.
 *
 * The Lagrange variants differ in the placement of their nodes (roots):
 * Gauss nodes are the roots of Legendre polynomials, Gauss-Lobatto nodes
 * additionally include the endpoints of the interval, uniform nodes are
 * equally spaced, and Chebyshev-Gauss nodes are the roots of Chebyshev
 * polynomials. All bases are defined on the interval [-1, 1].
 */
typedef enum
{
    BASIS_INVALID = 0,              // Invalid or uninitialized basis type.
    BASIS_LEGENDRE,                 // Legendre polynomials.
    BASIS_LAGRANGE_GAUSS_LOBATTO,   // Lagrange polynomials on Gauss-Lobatto nodes.
    BASIS_LAGRANGE_GAUSS,           // Lagrange polynomials on Gauss nodes.
    BASIS_LAGRANGE_UNIFORM,         // Lagrange polynomials on uniformly spaced nodes.
    BASIS_LAGRANGE_CHEBYSHEV_GAUSS, // Lagrange polynomials on Chebyshev-Gauss nodes.
    BASIS_BERNSTEIN,                // Bernstein polynomials.
} basis_set_type_t;

/**
 * @brief Check if a basis set type is valid.
 *
 * @param type The basis set type to check.
 * @return Non-zero if the type is valid, zero otherwise.
 */
static inline int basis_set_type_is_valid(const basis_set_type_t type)
{
    // For readibility we use an if statement, compiler optimizes it.
    return type == BASIS_LEGENDRE || type == BASIS_LAGRANGE_GAUSS_LOBATTO || type == BASIS_LAGRANGE_GAUSS ||
           type == BASIS_LAGRANGE_UNIFORM || type == BASIS_LAGRANGE_CHEBYSHEV_GAUSS || type == BASIS_BERNSTEIN;
}

/**
 * @brief Specification of a 1D basis: its type and order.
 *
 * The order is the polynomial degree, so the basis has `order + 1` basis
 * functions. The order must be such that `order + 1` is representable and
 * fits into the arrays sized with it.
 */
typedef struct
{
    basis_set_type_t type; // Type of the basis functions.
    unsigned order;        // Polynomial order of the basis; the basis has order + 1 functions.
} basis_spec_t;

/**
 * @brief Precomputed values and derivatives of a 1D basis at integration nodes.
 *
 * The data is laid out in the flexible array `_data` as, for each of the
 * `order + 1` basis functions, its values at the integration nodes followed
 * by its derivatives at the integration nodes. In other words, the first
 * `(order + 1) * (integration_spec.order + 1)` doubles hold the values and
 * the next block of equal size holds the derivatives. Use the inline accessor
 * functions in this header instead of accessing `_data` directly.
 */
typedef struct
{
    basis_spec_t spec;                   // Specifications for the Basis
    integration_spec_t integration_spec; // Specifications for the Integration rule
    double _data[];                      // Values of the basis_sets and their derivatives at integration nodes
} basis_set_t;
/**
 * @brief Extract the boundary basis given its orientation.
 *
 * @param ndim Number of dimensions for the element.
 * @param element_basis Basis for the element.
 * @param orientation Canonical orientation of the boundary.
 * @param bdim Dimension of the boundary.
 * @param boundary_basis Output basis for the boundary.
 */
void basis_set_to_boundary(unsigned ndim, const basis_set_t element_basis[static ndim],
                           const int8_t orientation[static ndim], unsigned bdim,
                           basis_set_t boundary_basis[static restrict bdim]);

/**
 * @brief Cached values of a 1D basis at the interval endpoints.
 *
 * The data is laid out as the values at `-1` followed by the values at
 * `+1`, with `order + 1` entries for each endpoint.
 */
typedef struct
{
    basis_spec_t spec;
    double _data[];
} basis_endpoint_set_t;

/**
 * @brief Get basis values at one interval endpoint.
 *
 * @param this Cached endpoint values.
 * @param end Zero for `-1`, one for `+1`.
 * @return Pointer to `order + 1` basis values.
 */
static inline const double *basis_endpoint_values(const basis_endpoint_set_t *this, const unsigned end)
{
    ASSERT(end <= 1, "Endpoint index was out of bounds.");
    return this->_data + end * (this->spec.order + 1);
}

typedef struct basis_set_registry_t basis_set_registry_t;

/**
 * @brief Get a pointer to all basis values at all integration nodes.
 *
 * @param this Basis set to get the values from.
 * @return Pointer to an array of `(order + 1) * (integration order + 1)`
 *         doubles: for each basis function, its values at the integration
 *         nodes in order.
 */
static inline const double *basis_set_values_all(const basis_set_t *this)
{
    return this->_data;
}

/**
 * @brief Get a pointer to the values of one basis function at all integration nodes.
 *
 * @param this Basis set to get the values from.
 * @param index Index of the basis function, in the range [0, order].
 * @return Pointer to `integration order + 1` doubles with the values of the
 *         basis function at the integration nodes.
 */
static inline const double *basis_set_basis_values(const basis_set_t *this, const unsigned index)
{
    ASSERT(index <= this->spec.order, "Index was out of bounds.");
    return this->_data + index * (this->integration_spec.order + 1);
}

/**
 * @brief Get a pointer to all basis derivatives at all integration nodes.
 *
 * @param this Basis set to get the derivatives from.
 * @return Pointer to an array of `(order + 1) * (integration order + 1)`
 *         doubles: for each basis function, its derivatives at the
 *         integration nodes in order.
 */
static inline const double *basis_set_derivatives_all(const basis_set_t *this)
{
    return this->_data + (this->spec.order + 1) * (this->integration_spec.order + 1);
}

/**
 * @brief Get a pointer to the derivatives of one basis function at all integration nodes.
 *
 * @param this Basis set to get the derivatives from.
 * @param index Index of the basis function, in the range [0, order].
 * @return Pointer to `integration order + 1` doubles with the derivatives of
 *         the basis function at the integration nodes.
 */
static inline const double *basis_set_basis_derivatives(const basis_set_t *this, const unsigned index)
{
    ASSERT(index <= this->spec.order, "Index was out of bounds.");
    return this->_data + (this->spec.order + 1 + index) * (this->integration_spec.order + 1);
}

/**
 * @brief Create a new basis set registry.
 *
 * The registry caches basis sets so that requesting the same basis
 * specification multiple times returns the same object. The registry is
 * thread-safe: it uses an internal reader-writer lock, and retrieved basis
 * sets are reference counted.
 *
 * @param out Receives the pointer to the newly created registry on success.
 * @param should_cache When non-zero, released basis sets are kept in the
 *        registry until explicitly cleared; otherwise they are deallocated
 *        as soon as their reference count drops to zero.
 * @param allocator Allocator used for all memory managed by the registry,
 *        including the registry object itself.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the created registry and must destroy it with
 * basis_set_registry_destroy.
 */
FDG_INTERNAL
fdg_result_t basis_set_registry_create(basis_set_registry_t **out, int should_cache, const cutl_allocator_t *allocator);

/**
 * @brief Get a basis set matching the given specification, creating it if needed.
 *
 * The basis set is stored in the registry and reference counted; every
 * successful call increments the reference count by one, and the caller must
 * call basis_set_registry_release_basis_set exactly once for the returned
 * basis set when it is no longer needed. The returned pointer stays valid as
 * long as its reference count is positive.
 *
 * @param this Registry to get the basis set from.
 * @param p_basis Receives the pointer to the basis set on success.
 * @param integration_rule Integration rule whose nodes the basis set is
 *        evaluated at. Only the rule's type and order are used to identify
 *        the basis set.
 * @param spec Specification (type and order) of the basis set.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails, or FDG_ERROR_INVALID_ENUM if the basis type is
 *         not supported. On failure, `*p_basis` is left unmodified.
 */
FDG_INTERNAL
fdg_result_t basis_set_registry_get_basis_set(basis_set_registry_t *this, const basis_set_t **p_basis,
                                              const integration_rule_t *integration_rule, basis_spec_t spec);

/**
 * @brief Batched version of basis_set_registry_get_basis_set.
 *
 * @param this Registry to get the basis sets from.
 * @param cnt Number of basis sets to retrieve.
 * @param p_basis Array of `cnt` pointers which receives the basis sets, in
 *        the same order as the corresponding specifications.
 * @param integration_rule Array of `cnt` integration rules used for the
 *        basis sets.
 * @param specs Array of `cnt` basis specifications.
 * @return FDG_SUCCESS on success, otherwise an error code as returned by
 *         basis_set_registry_get_basis_set. On failure, all basis sets
 *         acquired before the failing one are released again, and the
 *         contents of `p_basis` are indeterminate.
 */
FDG_INTERNAL
fdg_result_t basis_set_registry_get_basis_sets(basis_set_registry_t *this, unsigned cnt,
                                               const basis_set_t *FDG_ARRAY_ARG(p_basis, cnt),
                                               const integration_rule_t *FDG_ARRAY_ARG(integration_rule, static cnt),
                                               const basis_spec_t FDG_ARRAY_ARG(specs, static cnt));
/**
 * @brief Get cached basis values at the interval endpoints.
 *
 * Endpoint values are keyed only by the basis specification and therefore do
 * not depend on an integration rule.
 *
 * @param this Registry to get the endpoint values from.
 * @param p_endpoints Receives the endpoint values on success.
 * @param spec Specification of the basis.
 * @return FDG_SUCCESS on success, or an error code on failure.
 */
FDG_INTERNAL
fdg_result_t basis_set_registry_get_basis_endpoints(basis_set_registry_t *this,
                                                    const basis_endpoint_set_t **p_endpoints, basis_spec_t spec);

/**
 * @brief Release a previously retrieved endpoint basis set.
 *
 * @param this Registry the endpoint values were retrieved from.
 * @param endpoints Endpoint values to release.
 * @return FDG_SUCCESS if the endpoint values were found and released.
 */
FDG_INTERNAL
fdg_result_t basis_set_registry_release_basis_endpoints(basis_set_registry_t *this,
                                                        const basis_endpoint_set_t *endpoints);

/**
 * @brief Release a previously retrieved basis set.
 *
 * Decrements the reference count of the basis set by one. If the count
 * reaches zero and the registry was created with caching disabled, the basis
 * set is deallocated and removed from the registry.
 *
 * @param this Registry the basis set was retrieved from.
 * @param basis Basis set to release. It must have been obtained from this
 *        registry with basis_set_registry_get_basis_set or
 *        basis_set_registry_get_basis_sets and must not have been released
 *        already.
 * @return FDG_SUCCESS if the basis set was found and released,
 *         FDG_ERROR_NOT_IN_REGISTRY if it is not in the registry.
 */
FDG_INTERNAL
fdg_result_t basis_set_registry_release_basis_set(basis_set_registry_t *this, const basis_set_t *basis);

/**
 * @brief Destroy a basis set registry and free all its basis sets.
 *
 * All memory managed by the registry, including all cached basis sets and
 * the registry object itself, is deallocated with the registry's allocator.
 * The registry must not be used by any other thread during or after this
 * call, and no basis set obtained from it may be used afterwards.
 *
 * @param this Registry to destroy.
 */
FDG_INTERNAL
void basis_set_registry_destroy(basis_set_registry_t *this);

/**
 * @brief Deallocate basis sets whose reference count has reached zero.
 *
 * Basis sets with a zero reference count are removed from the registry and
 * deallocated. This only applies to registries created with caching
 * disabled; registries with caching enabled retain all basis sets.
 *
 * @param this Registry to clean up.
 */
FDG_INTERNAL
void basis_set_registry_release_unused_basis_sets(basis_set_registry_t *this);

/**
 * @brief Release all basis sets from the registry.
 *
 * @param this Registry to release the basis sets of.
 */
FDG_INTERNAL
void basis_set_registry_release_all_basis_sets(const basis_set_registry_t *this);

/**
 * @brief Get the specifications of all basis sets in the registry.
 *
 * @param this Registry to query.
 * @param max_count Maximum number of specifications to write.
 * @param basis_spec Array of `max_count` entries which receives the basis
 *        specifications of the basis sets.
 * @param integration_spec Array of `max_count` entries which receives the
 *        integration specifications of the basis sets.
 * @return The total number of basis sets in the registry, which may exceed
 *         `max_count`; in that case only the first `max_count` entries of the
 *         output arrays are written.
 */
FDG_INTERNAL
unsigned basis_set_registry_get_sets(basis_set_registry_t *this, unsigned max_count,
                                     basis_spec_t FDG_ARRAY_ARG(basis_spec, max_count),
                                     integration_spec_t FDG_ARRAY_ARG(integration_spec, max_count));

/**
 * @brief Prepare the work array with the nodes of the basis.
 *
 * For Lagrange bases the nodes (roots) of the basis are computed into
 * `work`; the roots depend on the basis type and order. For other basis
 * types nothing is written and the contents of `work` are unspecified. The
 * result is used by basis_compute_at_point_values and
 * basis_compute_at_point_derivatives.
 *
 * @param type Type of the basis. Only the BASIS_LAGRANGE_* types have any
 *        effect.
 * @param order Order of the basis.
 * @param work Array of `order + 1` doubles which receives the nodes.
 */
FDG_INTERNAL
void basis_compute_at_point_prepare(basis_set_type_t type, unsigned order,
                                    double FDG_ARRAY_ARG(work, restrict order + 1));
/**
 * @brief Compute the values of all basis functions at the given points.
 *
 * For each point, writes the values of the `order + 1` basis functions of
 * the given type, evaluated at that point. The values are written
 * row-major: the values for point `i` start at `out[i * (order + 1)]`.
 *
 * @param type Type of the basis functions.
 * @param order Order of the basis.
 * @param cnt Number of points to evaluate at.
 * @param x Array of `cnt` points at which the basis is evaluated. The points
 *        must lie in the domain of the basis; for Lagrange bases prepared
 *        nodes in `work` must have been computed first with
 *        basis_compute_at_point_prepare.
 * @param out Array of `cnt * (order + 1)` doubles which receives the basis
 *        values.
 * @param work Array of `order + 1` doubles, prepared by
 *        basis_compute_at_point_prepare, used as scratch space.
 */
FDG_INTERNAL
void basis_compute_at_point_values(basis_set_type_t type, unsigned order, unsigned cnt,
                                   const double FDG_ARRAY_ARG(x, restrict static cnt),
                                   double FDG_ARRAY_ARG(out, restrict cnt *(order + 1)),
                                   double FDG_ARRAY_ARG(work, restrict order + 1));

/**
 * @brief Compute the first derivatives of all basis functions at the given points.
 *
 * For each point, writes the derivatives of the `order + 1` basis functions
 * of the given type, evaluated at that point. The layout of `out` is the
 * same as in basis_compute_at_point_values: the derivatives for point `i`
 * start at `out[i * (order + 1)]`.
 *
 * @param type Type of the basis functions.
 * @param order Order of the basis.
 * @param cnt Number of points to evaluate at.
 * @param x Array of `cnt` points at which the derivatives are evaluated. For
 *        Lagrange bases, `work` must have been prepared with
 *        basis_compute_at_point_prepare first.
 * @param out Array of `cnt * (order + 1)` doubles which receives the basis
 *        derivatives.
 * @param work Array of `order + 1` doubles, prepared by
 *        basis_compute_at_point_prepare, used as scratch space.
 */
FDG_INTERNAL
void basis_compute_at_point_derivatives(basis_set_type_t type, unsigned order, unsigned cnt,
                                        const double FDG_ARRAY_ARG(x, restrict static cnt),
                                        double FDG_ARRAY_ARG(out, restrict cnt *(order + 1)),
                                        double FDG_ARRAY_ARG(work, restrict order + 1));

/**
 * @brief Compute the sizes of the buffers needed for the outer product of 1D bases.
 *
 * @param n_basis Number of 1D bases in the tensor product.
 * @param basis_specs Array of `n_basis` basis specifications.
 * @param cnt Number of points the outer product will be evaluated at.
 * @param out_elements Receives the number of doubles needed in the `out`
 *        buffer of basis_compute_outer_product_basis.
 * @param work_elements Receives the number of doubles needed in the `work`
 *        buffer.
 * @param tmp_elements Receives the number of doubles needed in the `tmp`
 *        buffer.
 * @param iterator_size Receives the size in bytes of the
 *        multidim_iterator_t buffer.
 */
FDG_INTERNAL
void basis_compute_outer_product_basis_required_memory(unsigned n_basis,
                                                       const basis_spec_t FDG_ARRAY_ARG(basis_specs, n_basis),
                                                       unsigned cnt, unsigned *out_elements, unsigned *work_elements,
                                                       unsigned *tmp_elements, size_t *iterator_size);

/**
 * @brief Compute the tensor-product (outer product) of 1D bases at the given points.
 *
 * For each point, computes the product of the 1D basis functions of all
 * dimensions, i.e. for point `i` and multi-index `(j_1, ..., j_n)` the value
 * `out[i * total + flat(j)] = prod_k basis_k(x_k[i])[j_k]`, where `total` is
 * the number of tensor-product basis functions and `flat` enumerates the
 * multi-indices in the order of the multidim_iterator.
 *
 * @param n_basis_dims Number of 1D bases in the tensor product.
 * @param basis_specs Array of `n_basis_dims` basis specifications.
 * @param cnt Number of points to evaluate at.
 * @param x Array of `n_basis_dims` pointers, each to `cnt` points along that
 *        dimension.
 * @param out Buffer of `out_elements` doubles (see
 *        basis_compute_outer_product_basis_required_memory) which receives
 *        the tensor-product values.
 * @param work Buffer of `work_elements` doubles, used as scratch space.
 * @param tmp Buffer of `tmp_elements` doubles, used as scratch space.
 * @param iter Iterator of at least `iterator_size` bytes, which is
 *        initialized by this function.
 */
FDG_INTERNAL
void basis_compute_outer_product_basis(unsigned n_basis_dims,
                                       const basis_spec_t FDG_ARRAY_ARG(basis_specs, n_basis_dims), unsigned cnt,
                                       const double *FDG_ARRAY_ARG(x, restrict n_basis_dims), double out[restrict],
                                       double work[restrict], double tmp[restrict], multidim_iterator_t *iter);

/**
 * @brief Iterator over the outer product of a pair of tensor-product basis sets along
 * the integration points of the shared per-dimension integration rules.
 *
 * The iterator keeps a single allocation (`data`) holding a fused prefix product
 * cache and the odometer state. For each integration point it yields the scalar
 * `prod_k w_k[ip_k] * L_k[b_k^L][ip_k] * R_k[b_k^R][ip_k]`, where `w_k` are the
 * per-dimension quadrature weights (folded in only when `rules` is provided),
 * and `L`/`R` are the left/right 1D basis values (or derivatives, per mask bit).
 * This is exactly the per-point factor needed to accumulate inner products
 * between the two basis sets, e.g. mass matrix entries.
 *
 * The data layout is: `ndim - 1` doubles (prefix product cache, entry `j` holds
 * `prod_{k <= j}` of the per-dimension factors), followed by `ndim` unsigned
 * integers of integration point indices, `ndim` unsigned integers of left basis
 * indices, and `ndim` unsigned integers of right basis indices.
 *
 * The integration point odometer runs in the same order as the multidim
 * iterators (last dimension fastest), so the step counter equals the flat
 * row-major integration point index used for e.g. determinant lookups.
 */
typedef struct
{
    unsigned ndim;                          // Number of tensor-product dimensions, in [1, 32].
    const basis_set_t *const *basis_left;   // Array of `ndim` pointers to the left operand basis sets.
    const basis_set_t *const *basis_right;  // Array of `ndim` pointers to the right operand basis sets.
    const integration_rule_t *const *rules; // Array of `ndim` integration rules whose weights are folded into the
                                            // cached products, or NULL to leave the weights to the caller.
    unsigned derivative_mask_left;          // Bit `d` set: dimension `d` reads derivatives instead of values (left).
    unsigned derivative_mask_right;         // Bit `d` set: dimension `d` reads derivatives instead of values (right).
    size_t point_index;                     // Flat row-major index of the current integration point.
    // data (aligned as max_align_t to be sure of proper alignment)
    alignas(max_align_t) unsigned char data[];
} outer_product_pair_iterator_t;

// TODO: move these static inline functions to the source file instead.
static inline unsigned outer_product_pair_iterator_ndim(const outer_product_pair_iterator_t *iter)
{
    return iter->ndim;
}

/**
 * @brief Returns a pointer to the prefix product cache of the pair iterator.
 *
 * @note For an iterator with `ndim` dimensions, the cache contains `ndim - 1` double values. Entry `j`
 * holds `prod_{k <= j}` of the per-dimension factors `w_k[ip_k] * L_k * R_k` at the current point.
 *
 * @return Pointer to the prefix product cache of the outer product pair iterator.
 */
static inline double *outer_product_pair_iterator_prefix_cache(const outer_product_pair_iterator_t *iter)
{
    return (double *)(iter->data);
}

/**
 * @brief Returns a pointer to the integration point indices of the pair iterator.
 *
 * @note For an iterator with `ndim` dimensions, the integration point indices array contains `ndim` unsigned
 * integers.
 *
 * @return Pointer to the integration point indices of the outer product pair iterator.
 */
static inline unsigned *outer_product_pair_iterator_integration_point_indices(const outer_product_pair_iterator_t *iter)
{
    return (unsigned *)(iter->data + (iter->ndim - 1) * sizeof(double));
}

/**
 * @brief Returns a pointer to the left operand basis indices of the pair iterator.
 *
 * @note For an iterator with `ndim` dimensions, the basis indices array contains `ndim` unsigned integers.
 *
 * @return Pointer to the left basis indices of the outer product pair iterator.
 */
static inline unsigned *outer_product_pair_iterator_basis_indices_left(const outer_product_pair_iterator_t *iter)
{
    return (unsigned *)(iter->data + (iter->ndim - 1) * sizeof(double)) + iter->ndim;
}

/**
 * @brief Returns a pointer to the right operand basis indices of the pair iterator.
 *
 * @note For an iterator with `ndim` dimensions, the basis indices array contains `ndim` unsigned integers.
 *
 * @return Pointer to the right basis indices of the outer product pair iterator.
 */
static inline unsigned *outer_product_pair_iterator_basis_indices_right(const outer_product_pair_iterator_t *iter)
{
    return (unsigned *)(iter->data + (iter->ndim - 1) * sizeof(double)) + 2 * iter->ndim;
}

static inline size_t outer_product_pair_iterator_data_size(unsigned ndim)
{
    // base size
    size_t size = sizeof(outer_product_pair_iterator_t);
    // add space for the prefix product cache (ndim - 1 doubles)
    size += (ndim - 1) * sizeof(double);
    // add space for the integration point indices (ndim unsigned integers)
    size += ndim * sizeof(unsigned);
    // add space for the left basis indices (ndim unsigned integers)
    size += ndim * sizeof(unsigned);
    // add space for the right basis indices (ndim unsigned integers)
    size += ndim * sizeof(unsigned);
    return size;
}

/**
 * @brief Computes the per-dimension factor `w[ip] * L[b_l][ip] * R[b_r][ip]` of the pair iterator.
 *
 * The quadrature weight is included only when the iterator was initialized with integration rules. Each
 * operand reads derivatives instead of values when the corresponding derivative mask bit is set.
 *
 * @param iter Pointer to the outer product pair iterator.
 * @param dim Dimension to compute the factor of.
 * @return The per-dimension factor at the current integration point and basis indices.
 */
static inline double outer_product_pair_iterator_dim_factor(const outer_product_pair_iterator_t *iter, unsigned dim)
{
    const unsigned *const ip_indices = outer_product_pair_iterator_integration_point_indices(iter);
    const unsigned *const basis_indices_left = outer_product_pair_iterator_basis_indices_left(iter);
    const unsigned *const basis_indices_right = outer_product_pair_iterator_basis_indices_right(iter);

    const double left =
        ((iter->derivative_mask_left >> dim) & 1u)
            ? basis_set_basis_derivatives(iter->basis_left[dim], basis_indices_left[dim])[ip_indices[dim]]
            : basis_set_basis_values(iter->basis_left[dim], basis_indices_left[dim])[ip_indices[dim]];
    const double right =
        ((iter->derivative_mask_right >> dim) & 1u)
            ? basis_set_basis_derivatives(iter->basis_right[dim], basis_indices_right[dim])[ip_indices[dim]]
            : basis_set_basis_values(iter->basis_right[dim], basis_indices_right[dim])[ip_indices[dim]];
    const double weight = iter->rules != NULL ? integration_rule_weights_const(iter->rules[dim])[ip_indices[dim]] : 1;
    return weight * left * right;
}

/**
 * @brief Initializes an outer product pair iterator.
 *
 * @note The prefix product cache is left uninitialized; call
 * outer_product_pair_iterator_set_basis_indices before the first use.
 *
 * @param iter Pointer to the outer product pair iterator to initialize.
 * @param ndim Number of dimensions of the iterator, in [1, 32].
 * @param basis_left Array of `ndim` pointers to the left operand basis sets.
 * @param basis_right Array of `ndim` pointers to the right operand basis sets.
 * @param rules Array of `ndim` integration rules whose weights are folded into the cached products, or NULL.
 * @param derivative_mask_left Mask selecting per-dimension derivative reads on the left operand.
 * @param derivative_mask_right Mask selecting per-dimension derivative reads on the right operand.
 */
static inline void outer_product_pair_iterator_init(outer_product_pair_iterator_t *iter, unsigned ndim,
                                                    const basis_set_t *const *basis_left,
                                                    const basis_set_t *const *basis_right,
                                                    const integration_rule_t *const *rules,
                                                    unsigned derivative_mask_left, unsigned derivative_mask_right)
{
    ASSERT(ndim >= 1 && ndim <= 32, "ndim must be in [1, 32]");
    iter->ndim = ndim;
    iter->basis_left = basis_left;
    iter->basis_right = basis_right;
    iter->rules = rules;
    iter->derivative_mask_left = derivative_mask_left;
    iter->derivative_mask_right = derivative_mask_right;
    iter->point_index = 0;
    // Indices should be zeroed
    memset(outer_product_pair_iterator_integration_point_indices(iter), 0, 3 * ndim * sizeof(unsigned));
}

/**
 * @brief Replaces the derivative masks of the pair iterator.
 *
 * @note Takes effect on the next cache rebuild (outer_product_pair_iterator_set_basis_indices or
 * outer_product_pair_iterator_next_integration_point).
 *
 * @param iter Pointer to the outer product pair iterator.
 * @param derivative_mask_left Mask selecting per-dimension derivative reads on the left operand.
 * @param derivative_mask_right Mask selecting per-dimension derivative reads on the right operand.
 */
static inline void outer_product_pair_iterator_set_derivative_masks(outer_product_pair_iterator_t *iter,
                                                                    unsigned derivative_mask_left,
                                                                    unsigned derivative_mask_right)
{
    iter->derivative_mask_left = derivative_mask_left;
    iter->derivative_mask_right = derivative_mask_right;
}

/**
 * @brief Replaces the per-dimension basis set arrays of the pair iterator.

 * @note Takes effect on the next cache rebuild (outer_product_pair_iterator_set_basis_indices or
 * outer_product_pair_iterator_next_integration_point).
 *
 * @param iter Pointer to the outer product pair iterator.
 * @param basis_left Array of `ndim` pointers to the left operand basis sets.
 * @param basis_right Array of `ndim` pointers to the right operand basis sets.
 */
static inline void outer_product_pair_iterator_set_bases(outer_product_pair_iterator_t *iter,
                                                         const basis_set_t *const *basis_left,
                                                         const basis_set_t *const *basis_right)
{
    iter->basis_left = basis_left;
    iter->basis_right = basis_right;
}

/**
 * @brief Sets the basis function indices of both operands and resets the iterator to the first integration point.
 *
 * Usage contract: after this call, outer_product_pair_iterator_current_value is valid at integration point
 * zero (outer_product_pair_iterator_point_index equals 0). Accumulate the current value, then repeatedly call
 * outer_product_pair_iterator_next_integration_point while it returns 1, accumulating
 * outer_product_pair_iterator_current_value at every step. The point index always equals the flat row-major
 * integration point index of the current point.
 *
 * @param iter Pointer to the outer product pair iterator.
 * @param basis_indices_left Array of `ndim` left operand basis function indices.
 * @param basis_indices_right Array of `ndim` right operand basis function indices.
 */
static inline void outer_product_pair_iterator_set_basis_indices(outer_product_pair_iterator_t *iter,
                                                                 const size_t *basis_indices_left,
                                                                 const size_t *basis_indices_right)
{
    const unsigned ndim = iter->ndim;
    unsigned *const left_indices = outer_product_pair_iterator_basis_indices_left(iter);
    unsigned *const right_indices = outer_product_pair_iterator_basis_indices_right(iter);
    for (unsigned i = 0; i < ndim; ++i)
    {
        left_indices[i] = (unsigned)basis_indices_left[i];
        right_indices[i] = (unsigned)basis_indices_right[i];
    }
    // Reset the integration point odometer to the first point
    memset(outer_product_pair_iterator_integration_point_indices(iter), 0, ndim * sizeof(unsigned));
    iter->point_index = 0;
    // Rebuild the full prefix product cache
    double *const prefix_cache = outer_product_pair_iterator_prefix_cache(iter);
    for (unsigned j = 0; j + 1 < ndim; ++j)
    {
        prefix_cache[j] = (j == 0 ? 1 : prefix_cache[j - 1]) * outer_product_pair_iterator_dim_factor(iter, j);
    }
}

/**
 * @brief Returns the flat row-major index of the current integration point.
 *
 * @param iter Pointer to the outer product pair iterator.
 * @return The flat integration point index of the current point.
 */
static inline size_t outer_product_pair_iterator_point_index(const outer_product_pair_iterator_t *iter)
{
    return iter->point_index;
}

/**
 * @brief Returns the fused per-point factor `prod_k w_k[ip_k] * L_k * R_k` at the current integration point.
 *
 * @param iter Pointer to the outer product pair iterator.
 * @return The factor of the current integration point for the current basis indices.
 */
static inline double outer_product_pair_iterator_current_value(const outer_product_pair_iterator_t *iter)
{
    const unsigned ndim = iter->ndim;
    if (ndim == 1)
    {
        return outer_product_pair_iterator_dim_factor(iter, 0);
    }
    return outer_product_pair_iterator_prefix_cache(iter)[ndim - 2] *
           outer_product_pair_iterator_dim_factor(iter, ndim - 1);
}

/**
 * @brief Advances the pair iterator to the next integration point.
 *
 * @note Only the suffix of the prefix product cache belonging to the dimensions that changed is recomputed,
 * making the amortized cost per point constant in the number of dimensions.
 *
 * @param iter Pointer to the outer product pair iterator.
 * @return 1 if the iterator moved to a valid integration point, 0 if the sweep is exhausted.
 */
static inline int outer_product_pair_iterator_next_integration_point(outer_product_pair_iterator_t *iter)
{
    const unsigned ndim = iter->ndim;
    unsigned *const ip_indices = outer_product_pair_iterator_integration_point_indices(iter);
    unsigned i;
    for (i = ndim; i > 0; --i)
    {
        const unsigned current_index = (ip_indices[i - 1] += 1);
        if (current_index <= iter->basis_left[i - 1]->integration_spec.order)
        {
            break;
        }
        // Reached the end of this dimension, reset the index and continue to the next higher dimension
        ip_indices[i - 1] = 0;
    }
    if (i > 0)
    {
        // Reuse the cached values for the unchanged dimensions (indices below i - 1 wrapped to zero); the last
        // dimension is not cached and always recomputed by outer_product_pair_iterator_current_value
        double *const prefix_cache = outer_product_pair_iterator_prefix_cache(iter);
        for (unsigned j = i - 1; j + 1 < ndim; ++j)
        {
            prefix_cache[j] = (j == 0 ? 1 : prefix_cache[j - 1]) * outer_product_pair_iterator_dim_factor(iter, j);
        }
        iter->point_index += 1;
    }
    // Return 1 if we have not exhausted all integration points, 0 otherwise
    return i > 0;
}
