//
// Created by jan on 5.11.2024.
//

#ifndef BERNSTEIN_H
#define BERNSTEIN_H

#include "../common/common_defines.h"

FDG_INTERNAL
void bernstein_from_power_series(unsigned n, double FDG_ARRAY_ARG(coeffs, static n));

FDG_INTERNAL
void bernstein_interpolation_vector(double t, unsigned n, double FDG_ARRAY_ARG(out, restrict n + 1));

FDG_INTERNAL
void bernstein_interpolation_value_derivative_matrix(unsigned n_in, const double FDG_ARRAY_ARG(t, restrict static n_in),
                                                     unsigned n,
                                                     double FDG_ARRAY_ARG(out_value, restrict(n + 1) * n_in),
                                                     double FDG_ARRAY_ARG(out_derivative, restrict(n + 1) * n_in));

#endif // BERNSTEIN_H
