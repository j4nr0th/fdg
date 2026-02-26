#ifndef FDG_INCIDENCE_H
#define FDG_INCIDENCE_H

#include "../basis/basis_set.h"
#include "module.h"

FDG_INTERNAL
extern PyMethodDef incidence_methods[];

FDG_INTERNAL
void bernstein_apply_incidence_operator(
    unsigned n, size_t pre_stride, size_t post_stride, unsigned cols,
    const double FDG_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double FDG_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols), int negate);

FDG_INTERNAL
void legendre_apply_incidence_operator(
    unsigned n, size_t pre_stride, size_t post_stride, unsigned cols,
    const double FDG_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double FDG_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols), int negate);

FDG_INTERNAL
void lagrange_apply_incidence_operator(
    basis_set_type_t type, unsigned n, size_t pre_stride, size_t post_stride, unsigned cols,
    const double FDG_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double FDG_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols),
    double FDG_ARRAY_ARG(work, restrict const n + (n + 1) + n * (n + 1)), int negate);

#endif // FDG_INCIDENCE_H
