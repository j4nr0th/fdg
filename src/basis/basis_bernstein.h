//
// Created by jan on 2025-09-09.
//

#ifndef FDG_BASIS_BERNSTEIN_H
#define FDG_BASIS_BERNSTEIN_H
#include "basis_set.h"

/**
 * @brief Create a Bernstein basis set evaluated at the nodes of an integration rule.
 *
 * The created basis set holds the values and first derivatives of the
 * `spec.order + 1` Bernstein basis functions at the integration nodes.
 *
 * @param out Receives the pointer to the newly created basis set on success.
 * @param spec Specification of the basis. The type field is ignored and the
 *        basis is always created as a Bernstein basis.
 * @param rule Integration rule whose nodes the basis is evaluated at. Must
 *        be a valid integration rule with `n_nodes` at least 1.
 * @param allocator Allocator used to allocate the basis set.
 * @return FDG_SUCCESS on success, FDG_ERROR_FAILED_ALLOCATION if memory
 *         allocation fails. On failure, `*out` is left unmodified.
 *
 * The caller owns the created basis set and is responsible for deallocating
 * it with the same allocator once it is no longer needed.
 */
FDG_INTERNAL
fdg_result_t bernstein_basis_create(basis_set_t **out, basis_spec_t spec, const integration_rule_t *rule,
                                    const cutl_allocator_t *allocator);

#endif // FDG_BASIS_BERNSTEIN_H
