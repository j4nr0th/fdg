//
// Created by jan on 2025-09-09.
//

#include "basis_bernstein.h"
#include "../polynomials/bernstein.h"

FDG_INTERNAL
interp_result_t bernstein_basis_create(basis_set_t **out, const basis_spec_t spec, const integration_rule_t *rule,
                                       const cutl_allocator_t *allocator)
{
    basis_set_t *const this =
        cutl_alloc(allocator, sizeof *this + 2 * sizeof(*this->_data) * (spec.order + 1) * rule->n_nodes);
    if (!this)
        return FDG_ERROR_FAILED_ALLOCATION;

    bernstein_interpolation_value_derivative_matrix(rule->n_nodes, integration_rule_nodes_const(rule), spec.order,
                                                    this->_data,
                                                    this->_data + (size_t)(spec.order + 1) * rule->n_nodes);

    this->integration_spec = rule->spec;
    this->spec = spec;
    *out = this;

    // printf("Computed Bernstein basis of order %u on rule with %u nodes:\n", spec.order, rule->n_nodes);
    // for (unsigned i_basis = 0; i_basis < spec.order + 1; ++i_basis)
    // {
    //     printf("\tBasis %u:", i_basis);
    //     const double *basis = basis_set_basis_values(this, i_basis);
    //     for (unsigned ipt = 0; ipt < rule->n_nodes; ++ipt)
    //         printf(" %5.2g", basis[ipt]);
    //     printf("\n");
    // }
    // for (unsigned i_basis = 0; i_basis < spec.order + 1; ++i_basis)
    // {
    //     printf("\tDerivative %u:", i_basis);
    //     const double *derivative = basis_set_basis_derivatives(this, i_basis);
    //     for (unsigned ipt = 0; ipt < rule->n_nodes; ++ipt)
    //         printf(" %5.2g", derivative[ipt]);
    //     printf("\n");
    // }
    // printf("\t Raw data:");
    // for (unsigned ipt = 0; ipt < rule->n_nodes * 2 * (spec.order + 1); ++ipt)
    //     printf(" %5.2g", this->_data[ipt]);
    // printf("\n");

    return FDG_SUCCESS;
}
