#ifndef INTERPLIB_INCIDENCE_H
#define INTERPLIB_INCIDENCE_H

#include "module.h"

INTERPLIB_INTERNAL
extern PyMethodDef incidence_methods[];

INTERPLIB_INTERNAL
void bernstein_apply_incidence_operator(
    unsigned n, size_t pre_stride, size_t post_stride, unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols), int negate);

INTERPLIB_INTERNAL
void legendre_apply_incidence_operator(
    unsigned n, size_t pre_stride, size_t post_stride, unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols), int negate);

INTERPLIB_INTERNAL
void lagrange_apply_incidence_matrix(
    basis_set_type_t type, unsigned n, size_t pre_stride, size_t post_stride, unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(work, restrict const n + (n + 1) + n * (n + 1)), int negate);

#endif // INTERPLIB_INCIDENCE_H
