#include "reconstruction.h"

double compute_basis_value_at_integration_point_d(const unsigned ndim, const multidim_iterator_t *const iter_int,
                                                  const multidim_iterator_t *const iter_basis,
                                                  const basis_set_t *basis_sets[static const ndim],
                                                  const int derivatives[static const ndim])
{
    double basis_val = 1;
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        const size_t idx_basis_dim = multidim_iterator_get_offset(iter_basis, idim);
        const double *basis_values;
        if (derivatives[idim])
        {
            basis_values = basis_set_basis_derivatives(basis_sets[idim], idx_basis_dim);
        }
        else
        {
            basis_values = basis_set_basis_values(basis_sets[idim], idx_basis_dim);
        }
        basis_val *= basis_values[multidim_iterator_get_offset(iter_int, idim)];
    }
    return basis_val;
}

double compute_reconstruction_at_integration_point_d(const unsigned ndim, const multidim_iterator_t *const iter_int,
                                                     multidim_iterator_t *const iter_basis,
                                                     const basis_set_t *basis_sets[static const ndim],
                                                     const int derivatives[static const ndim], const unsigned ndof,
                                                     const double dof_values[static ndof])
{
    // Basis and integration iterator must be correct.
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_int),
           "Number of dimensions of integration space is not correct.");
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_basis), "Number of dimensions of basis is not correct.");
    ASSERT((size_t)ndof == multidim_iterator_total_size(iter_basis), "Number of DOFs is not correct.");
    double val = 0;
    multidim_iterator_set_to_start(iter_basis);
    while (!multidim_iterator_is_at_end(iter_basis))
    {
        // For each basis compute value at the integration point
        const double basis_val =
            compute_basis_value_at_integration_point_d(ndim, iter_int, iter_basis, basis_sets, derivatives);
        // Scale the basis value by the degree of freedom
        val += basis_val * dof_values[multidim_iterator_get_flat_index(iter_basis)];
        multidim_iterator_advance(iter_basis, ndim - 1, 1);
    }
    return val;
}

void compute_integration_point_values_derivatives(const unsigned ndim, multidim_iterator_t *iter_int,
                                                  multidim_iterator_t *iter_basis,
                                                  const basis_set_t *basis_sets[static ndim],
                                                  const int derivatives[static ndim], const unsigned nout,
                                                  double ptr[restrict nout], const unsigned ndof,
                                                  const double dof_values[restrict static ndof])
{
    // Basis and integration iterator must be correct.
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_int),
           "Number of dimensions of integration space is not correct.");
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_basis), "Number of dimensions of basis is not correct.");
    ASSERT((size_t)ndof == multidim_iterator_total_size(iter_basis), "Number of DOFs is not correct.");
    ASSERT((size_t)nout == multidim_iterator_total_size(iter_int), "Output array is too small.");

    multidim_iterator_set_to_start(iter_int);
    while (!multidim_iterator_is_at_end(iter_int))
    {
        // Compute the point value
        const double val = compute_reconstruction_at_integration_point_d(ndim, iter_int, iter_basis, basis_sets,
                                                                         derivatives, ndof, dof_values);

        // Write output and advance the integration iterator
        ptr[multidim_iterator_get_flat_index(iter_int)] = val;
        multidim_iterator_advance(iter_int, ndim - 1, 1);
    }
}

double compute_basis_value_at_integration_point(const unsigned ndim, const multidim_iterator_t *const iter_int,
                                                const multidim_iterator_t *const iter_basis,
                                                const basis_set_t *basis_sets[static const ndim])
{
    double basis_val = 1;
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        basis_val *= basis_set_basis_values(
            basis_sets[idim],
            multidim_iterator_get_offset(iter_basis, idim))[multidim_iterator_get_offset(iter_int, idim)];
    }
    return basis_val;
}

double compute_reconstruction_at_integration_point(const unsigned ndim, const multidim_iterator_t *const iter_int,
                                                   multidim_iterator_t *const iter_basis,
                                                   const basis_set_t *basis_sets[const static ndim],
                                                   const unsigned ndof, const double dof_values[static ndof])
{
    // Basis and integration iterator must be correct.
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_int),
           "Number of dimensions of integration space is not correct.");
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_basis), "Number of dimensions of basis is not correct.");
    ASSERT((size_t)ndof == multidim_iterator_total_size(iter_basis), "Number of DOFs is not correct.");

    double val = 0;
    multidim_iterator_set_to_start(iter_basis);
    while (!multidim_iterator_is_at_end(iter_basis))
    {
        // For each basis compute value at the integration point
        const double basis_val = compute_basis_value_at_integration_point(ndim, iter_int, iter_basis, basis_sets);
        // Scale the basis value by the degree of freedom
        val += basis_val * dof_values[multidim_iterator_get_flat_index(iter_basis)];
        multidim_iterator_advance(iter_basis, ndim - 1, 1);
    }
    return val;
}

void compute_integration_point_values(const unsigned ndim, multidim_iterator_t *const iter_int,
                                      multidim_iterator_t *const iter_basis, const basis_set_t *basis_sets[static ndim],
                                      const unsigned nout, double ptr[restrict nout], const unsigned ndof,
                                      const double dof_values[restrict static ndof])
{
    // Basis and integration iterator must be correct.
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_int),
           "Number of dimensions of integration space is not correct.");
    ASSERT((size_t)ndim == multidim_iterator_get_ndims(iter_basis), "Number of dimensions of basis is not correct.");
    ASSERT((size_t)ndof == multidim_iterator_total_size(iter_basis), "Number of DOFs is not correct.");

    multidim_iterator_set_to_start(iter_int);
    while (!multidim_iterator_is_at_end(iter_int))
    {
        // Compute the point value
        const double val =
            compute_reconstruction_at_integration_point(ndim, iter_int, iter_basis, basis_sets, ndof, dof_values);
        // Write output and advance the integration iterator
        ptr[multidim_iterator_get_flat_index(iter_int)] = val;
        multidim_iterator_advance(iter_int, ndim - 1, 1);
    }
}
