//
// Created by jan on 2025-09-09.
//

#ifndef FDG_BASIS_LAGRANGE_H
#define FDG_BASIS_LAGRANGE_H
#include "basis_set.h"

FDG_INTERNAL
interp_result_t lagrange_basis_create(basis_set_t **out, basis_spec_t spec, const integration_rule_t *rule,
                                      const cutl_allocator_t *allocator);

static inline const double *lagrange_basis_roots(const basis_set_t *this)
{
    ASSERT(this->spec.type == BASIS_LAGRANGE_UNIFORM || this->spec.type == BASIS_LAGRANGE_GAUSS ||
               this->spec.type == BASIS_LAGRANGE_GAUSS_LOBATTO || this->spec.type == BASIS_LAGRANGE_CHEBYSHEV_GAUSS,
           "This function is only valid for Lagrange basis functions.");
    return this->_data + (this->spec.order + 1) * (2 * (this->integration_spec.order + 1));
}

FDG_INTERNAL
interp_result_t generate_lagrange_roots(unsigned order, basis_set_type_t type, double roots[const order + 1]);

#endif // FDG_BASIS_LAGRANGE_H
