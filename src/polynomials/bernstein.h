//
// Created by jan on 5.11.2024.
//

#ifndef BERNSTEIN_H
#define BERNSTEIN_H

#include "../common/common_defines.h"

/**
 * @brief Convert a polynomial from power series form to Bernstein form in place.
 *
 * On input, `coeffs` holds the coefficients of a polynomial in the power
 * basis, with the coefficient of `x^k` at index `k`. On return, the same
 * array holds the coefficients of the Bernstein basis `B_k^n` for the same
 * polynomial. The degree `n - 1` polynomial is represented with `n`
 * coefficients in both forms.
 *
 * @param n Number of coefficients.
 * @param coeffs Array of `n` doubles; on input the power series
 *        coefficients, on output the Bernstein coefficients.
 */
FDG_INTERNAL
void bernstein_from_power_series(unsigned n, double FDG_ARRAY_ARG(coeffs, static n));

/**
 * @brief Compute the values of all Bernstein polynomials of degree n at a point.
 *
 * Computes `out[k] = B_k^n(t)` for `k = 0..n`, where
 * `B_k^n(t) = C(n, k) t^k (1-t)^(n-k)`.
 *
 * @param t Point at which the polynomials are evaluated, in the range [0, 1].
 * @param n Degree of the Bernstein polynomials; `n + 1` values are written.
 * @param out Array of `n + 1` doubles which receives the polynomial values.
 */
FDG_INTERNAL
void bernstein_interpolation_vector(double t, unsigned n, double FDG_ARRAY_ARG(out, restrict n + 1));

/**
 * @brief Compute the values and first derivatives of Bernstein polynomials at several points.
 *
 * For each point `t[i]` (given on [-1, 1] and internally mapped to [0, 1]),
 * the values and first derivatives of the `n + 1` Bernstein polynomials of
 * degree `n` are computed. The layout is basis-major with a stride of
 * `n_in`: the value of basis `k` at point `i` is stored at
 * `out_value[k * n_in + i]`, and likewise for the derivatives.
 *
 * @param n_in Number of points.
 * @param t Array of `n_in` points in [-1, 1] at which to evaluate.
 * @param n Degree of the Bernstein polynomials.
 * @param out_value Array of `(n + 1) * n_in` doubles which receives the
 *        polynomial values.
 * @param out_derivative Array of `(n + 1) * n_in` doubles which receives
 *        the first derivatives.
 */
FDG_INTERNAL
void bernstein_interpolation_value_derivative_matrix(unsigned n_in, const double FDG_ARRAY_ARG(t, restrict static n_in),
                                                     unsigned n,
                                                     double FDG_ARRAY_ARG(out_value, restrict(n + 1) * n_in),
                                                     double FDG_ARRAY_ARG(out_derivative, restrict(n + 1) * n_in));

#endif // BERNSTEIN_H
