#ifndef FDG_INCIDENCE_H
#define FDG_INCIDENCE_H

#include "../basis/basis_set.h"
#include "module.h"

FDG_INTERNAL
extern PyMethodDef incidence_methods[];

typedef struct
{
    unsigned repetitions;  // Number of repetitions to perform
    size_t stride_rep_in;  // Stride between the first DoFs of the input for each repetition
    size_t stride_rep_out; // Stride between the first DoFs of the output for each repetition
    size_t stride_dof;     // Stride between adjacent DoFs with their increasing index
} incidence_repeat_t;

typedef struct
{
    unsigned n;         // Order in the dimension the incidence is applied in.
    size_t pre_stride;  // Number of entries in preceding dimensions.
    size_t post_stride; // Number of entries in the following dimensions.
} incidence_base_strides_t;

typedef struct
{
    const double *restrict values_in; // Array of input values
    double *restrict values_out;      // Array of output values
    double *restrict work;            // Array used for intermediate computations
} incidence_array_specifications_t;

FDG_INTERNAL
void bernstein_apply_incidence_operator(const incidence_base_strides_t *base_strides, const incidence_repeat_t *repeats,
                                        const incidence_array_specifications_t *array_specs, int transpose, int negate);

FDG_INTERNAL
void legendre_apply_incidence_operator(const incidence_base_strides_t *base_strides, const incidence_repeat_t *repeats,
                                       const incidence_array_specifications_t *array_specs, int transpose, int negate);

FDG_INTERNAL
void lagrange_apply_incidence_operator(basis_set_type_t type, const incidence_base_strides_t *base_strides,
                                       const incidence_repeat_t *repeats,
                                       const incidence_array_specifications_t *array_specs, int transpose, int negate);

FDG_INTERNAL
void lagrange_prepare_incidence_transformation(basis_set_type_t type, unsigned n,
                                               double FDG_ARRAY_ARG(work, n + (n + 1) + n * (n + 1)));

FDG_INTERNAL
void apply_incidence_operator_single(basis_set_type_t type, const incidence_base_strides_t *base_strides,
                                     size_t stride_dof, int transpose, int negate,
                                     const incidence_array_specifications_t *array_specs);

#endif // FDG_INCIDENCE_H
