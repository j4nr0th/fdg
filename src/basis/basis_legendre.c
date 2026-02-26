//
// Created by jan on 2025-09-09.
//

#include "basis_legendre.h"
#include "../polynomials/legendre.h"

FDG_INTERNAL
interp_result_t legendre_basis_create(basis_set_t **out, const basis_spec_t spec, const integration_rule_t *rule,
                                      const cutl_allocator_t *allocator)
{
    basis_set_t *const this =
        cutl_alloc(allocator, sizeof *this + 2 * sizeof(*this->_data) * (spec.order + 1) * (rule->n_nodes));
    if (!this)
        return FDG_ERROR_FAILED_ALLOCATION;

    double *values = this->_data;
    double *derivatives = this->_data + (spec.order + 1) * (rule->n_nodes);

    for (unsigned i_pt = 0; i_pt < (rule->n_nodes); ++i_pt)
    {
        const double node = integration_rule_nodes_const(rule)[i_pt];
        // Compute the different basis values
        legendre_eval_bonnet_all_stride(spec.order, node, (rule->n_nodes), i_pt, values);
        // Compute the different basis derivatives
        derivatives[i_pt] = 0; // The first basis has no derivative
        double deriv = 0;
        for (unsigned i_deriv = 1; i_deriv <= spec.order; ++i_deriv)
        {
            // Use recurrence formula for subsequent derivatives
            deriv = i_deriv * values[i_pt + (i_deriv - 1) * (rule->n_nodes)] + node * deriv;
            derivatives[i_pt + i_deriv * (rule->n_nodes)] = deriv;
        }
    }

    this->spec = spec;
    this->integration_spec = rule->spec;
    *out = this;
    return FDG_SUCCESS;
}
