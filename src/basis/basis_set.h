//
// Created by jan on 2025-09-09.
//

#ifndef FDG_BASIS_H
#define FDG_BASIS_H
#include "../common/error.h"
#include "../integration/integration_rules.h"
#include <cutl/iterators/multidim_iteration.h>

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

typedef struct basis_set_registry_t basis_set_registry_t;

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

#endif // FDG_BASIS_H
