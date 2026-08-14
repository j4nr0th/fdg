//
// Created by jan on 2025-09-09.
//

#ifndef FDG_BASIS_LAGRANGE_H
#define FDG_BASIS_LAGRANGE_H
#include "basis_set.h"

/**
 * @brief Get the roots (nodes) of a Lagrange basis.
 *
 * @param this Lagrange basis set whose roots to return. Must be a Lagrange
 *        basis, i.e. the basis type must be one of the BASIS_LAGRANGE_*
 *        types, otherwise an assertion fires.
 * @return Pointer to the `order + 1` roots of the basis, stored inside the
 *         basis set's data.
 */
static inline const double *lagrange_basis_roots(const basis_set_t *this)
{
    ASSERT(this->spec.type == BASIS_LAGRANGE_UNIFORM || this->spec.type == BASIS_LAGRANGE_GAUSS ||
               this->spec.type == BASIS_LAGRANGE_GAUSS_LOBATTO || this->spec.type == BASIS_LAGRANGE_CHEBYSHEV_GAUSS,
           "This function is only valid for Lagrange basis functions.");
    return this->_data + (this->spec.order + 1) * (2 * (this->integration_spec.order + 1));
}

/**
 * @brief Create a Lagrange basis set evaluated at the nodes of an integration rule.
 *
 * The created basis set holds the values and first derivatives of the
 * `spec.order + 1` Lagrange basis functions at the integration nodes, as
 * well as the roots of the basis.
 *
 * @param out Receives the pointer to the newly created basis set on success.
 * @param spec Specification of the basis. The type must be one of the
 *        BASIS_LAGRANGE_* types.
 * @param rule Integration rule whose nodes the basis is evaluated at. Must
 *        be a valid integration rule with `n_nodes` at least 1.
 * @param allocator Allocator used to allocate the basis set.
 * @return FDG_SUCCESS on success, FDG_ERROR_INVALID_ENUM if the basis type
 *         is not a Lagrange type, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the created basis set and is responsible for deallocating
 * it with the same allocator once it is no longer needed.
 */
FDG_INTERNAL
fdg_result_t lagrange_basis_create(basis_set_t **out, basis_spec_t spec, const integration_rule_t *rule,
                                   const cutl_allocator_t *allocator);

/**
 * @brief Compute the roots (nodes) of a Lagrange basis of the given type.
 *
 * The roots are computed into `roots` in increasing order. For Gauss and
 * Gauss-Lobatto types the roots are computed numerically using Newton
 * iterations with a tolerance of 1e-12 and at most 100 iterations.
 *
 * @param order Order of the basis; the roots array holds `order + 1` entries.
 * @param type Type of the Lagrange basis, one of the BASIS_LAGRANGE_* types.
 * @param roots Array of `order + 1` doubles which receives the roots.
 * @return FDG_SUCCESS on success, FDG_ERROR_INVALID_ENUM if the type is not
 *         a Lagrange type.
 */
FDG_INTERNAL
fdg_result_t generate_lagrange_roots(unsigned order, basis_set_type_t type, double roots[const order + 1]);

#endif // FDG_BASIS_LAGRANGE_H
