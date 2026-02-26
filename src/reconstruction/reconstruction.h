#pragma once
#include "../basis/basis_set.h"

#include <cutl/iterators/multidim_iteration.h>

/**
 * State used for reconstruction using basis functions.
 */
typedef struct
{
    multidim_iterator_t *iter_int;
    multidim_iterator_t *iter_basis;
    const basis_set_t **basis_sets;
} reconstruction_state_t;

/**
 * State used for computing products.
 */
typedef struct
{
    multidim_iterator_t *iter_int;
    multidim_iterator_t *iter_basis_1;
    const basis_set_t **basis_sets_1;
    multidim_iterator_t *iter_basis_2;
    const basis_set_t **basis_sets_2;
} product_state_t;

/**
 * Compute the basis value at the specified integration point, based on the outer product of 1D basis and their
 * derivatives.
 *
 * @param ndim Number of dimensions of the input space.
 * @param iter_int Iterator used for integration points.
 * @param iter_basis Iterator used for basis.
 * @param basis_sets Basis sets to use.
 * @param derivatives Specification of the dimensions for which the derivatives should be used rather than basis values.
 * @return Value of the basis function based on the outer product of 1D basis sets.
 */
double compute_basis_value_at_integration_point_d(unsigned ndim, const multidim_iterator_t *iter_int,
                                                  const multidim_iterator_t *iter_basis,
                                                  const basis_set_t *basis_sets[static const ndim],
                                                  const int derivatives[static const ndim]);

/**
 * Computes the reconstructed value at a given integration point using basis functions
 * and their values or derivatives, scaled by the provided degrees of freedom (DOF).
 *
 * @param ndim[in] The number of dimensions in the multidimensional space.
 * @param iter_int[in,out] Iterator for the integration points.
 * @param iter_basis[in,out] Iterator for the basis functions.
 * @param basis_sets[in] An array of basis sets, one for each dimension.
 * @param derivatives[in] An array of integers indicating whether to use derivatives (non-zero) or
 *                        values (zero) of the basis functions for each dimension.
 * @param ndof[in] Number of degrees of freedom. Used for  bounds checks only.
 * @param dof_values[in] An array of degree-of-freedom values associated with the basis functions.
 * @return The computed value at the integration point based on the basis functions and DOFs.
 */
double compute_reconstruction_at_integration_point_d(unsigned ndim, const multidim_iterator_t *iter_int,
                                                     multidim_iterator_t *iter_basis,
                                                     const basis_set_t *basis_sets[static const ndim],
                                                     const int derivatives[static const ndim], unsigned ndof,
                                                     const double dof_values[static ndof]);

/**
 * Reconstruct value based on its basis and associated degrees of freedom at integration points.
 *
 * @param ndim[in] Number of dimensions of the input space.
 * @param iter_int[in,out] Iterator that deals with iterating over integrating points.
 * @param iter_basis[in,out] Iterator that deals with iterating over basis.
 * @param basis_sets[in,out] Array of basis sets that is used for basis values.
 * @param derivatives[in] Array which specifies if the derivative of the basis should be used instead of the basis
 * itself for each dimension.
 * @param nout[in] Size of the output array. Used for bounds checks only.
 * @param ptr[out] Pointer to the output array, the size of which should be based on integration points.
 * @param ndof[in] Number of degrees of freedom. Used for  bounds checks only.
 * @param dof_values[in] Value of degrees of freedom, the size of which should be based on basis.
 */
void compute_integration_point_values_derivatives(unsigned ndim, multidim_iterator_t *iter_int,
                                                  multidim_iterator_t *iter_basis,
                                                  const basis_set_t *basis_sets[static ndim],
                                                  const int derivatives[static ndim], unsigned nout,
                                                  double ptr[restrict nout], unsigned ndof,
                                                  const double dof_values[restrict static ndof]);

/**
 * Compute the basis value at the specified integration point, based on the outer product of 1D basis. Equivalent to
 * calling `compute_basis_value_at_integration_point_d` with all zeros for the `derivatives` argument.
 *
 * @param ndim Number of dimensions of the input space.
 * @param iter_int Iterator used for integration points.
 * @param iter_basis Iterator used for basis.
 * @param basis_sets Basis sets to use.
 * @return Value of the basis function based on the outer product of 1D basis sets.
 */
double compute_basis_value_at_integration_point(unsigned ndim, const multidim_iterator_t *iter_int,
                                                const multidim_iterator_t *iter_basis,
                                                const basis_set_t *basis_sets[static const ndim]);

/**
 * Computes the value of a basis function at a given integration point by summing
 * the contributions of outer product basis values scaled by the degrees of freedom. Equivalent to calling
 * `compute_reconstruction_at_integration_point_d` with all zeros for the `derivatives` argument.
 *
 * @param ndim The number of dimensions in the integration space and basis set.
 * @param iter_int A pointer to the multidimensional iterator representing the
 *                 current position in the integration point space.
 *                 Its offsets are used to access the correct integration point variables.
 * @param iter_basis Iterator for the basis.
 * @param basis_sets An array of basis sets for each dimension.
 * @param ndof[in] Number of degrees of freedom. Used for  bounds checks only.
 * @param dof_values A pointer to an array of degrees of freedom values.
 * @return The computed reconstruction at the integration point.
 */
double compute_reconstruction_at_integration_point(unsigned ndim, const multidim_iterator_t *iter_int,
                                                   multidim_iterator_t *iter_basis,
                                                   const basis_set_t *basis_sets[const static ndim], unsigned ndof,
                                                   const double dof_values[static ndof]);

/**
 * Reconstruct value based on its basis and associated degrees of freedom at integration points. It is equivalent to
 * calling `compute_integration_point_values_derivatives` with all zeros for the `derivatives` argument.
 *
 * @param ndim[in] Number of dimensions of the input space.
 * @param iter_int[in] Iterator that deals with iterating over integrating points.
 * @param iter_basis[in] Iterator that deals with iterating over basis.
 * @param basis_sets[in] Array of basis sets that is used for basis values.
 * @param nout[in] Size of the output array. Used for bounds checks only.
 * @param ptr[out] Pointer to the output array, the size of which should be based on integration points.
 * @param ndof[in] Number of degrees of freedom. Used for  bounds checks only.
 * @param dof_values[in] Value of degrees of freedom, the size of which should be based on basis.
 */
void compute_integration_point_values(unsigned ndim, multidim_iterator_t *iter_int, multidim_iterator_t *iter_basis,
                                      const basis_set_t *basis_sets[static ndim], unsigned nout,
                                      double ptr[restrict nout], unsigned ndof,
                                      const double dof_values[restrict static ndof]);
