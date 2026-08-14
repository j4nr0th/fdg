//
// Created by jan on 2025-09-07.
//

#ifndef FDG_LEGENDRE_H
#define FDG_LEGENDRE_H

#include "../common/common_defines.h"

/**
 * @brief Evaluate the two highest Legendre polynomials up to degree n.
 *
 * Computes `out[0] = P_{n-1}(x)` and `out[1] = P_n(x)` using Bonnet's
 * recurrence. Intended for computing the derivative of `P_n`, which
 * satisfies `(x^2 - 1) P'_n(x) = n (P_{n-1}(x) - x P_n(x))`.
 *
 * @param n Degree of the highest polynomial; must be at least 2.
 * @param x Point at which the polynomials are evaluated.
 * @param out Array of 2 doubles which receives `P_{n-1}(x)` and `P_n(x)`.
 */
FDG_INTERNAL
void legendre_eval_bonnet_two(unsigned n, double x, double FDG_ARRAY_ARG(out, 2));

/**
 * @brief Evaluate the highest Legendre polynomials up to degree n.
 *
 * Computes `out[k] = P_{n - m + 1 + k}(x)` for `k = 0..m - 1`, i.e. the
 * highest `m` Legendre polynomials of degrees `n - m + 1` through `n`.
 *
 * @param n Degree of the highest polynomial.
 * @param x Point at which the polynomials are evaluated.
 * @param m Number of polynomials to evaluate; must be at least 1 and at most
 *        `n`.
 * @param out Array of `m` doubles which receives the polynomial values.
 */
FDG_INTERNAL
void legendre_eval_bonnet(unsigned n, double x, unsigned m, double FDG_ARRAY_ARG(out, m));

/**
 * @brief Evaluate all Legendre polynomials up to degree n.
 *
 * Computes `out[k] = P_k(x)` for `k = 0..n` using Bonnet's recurrence.
 *
 * @param n Degree of the highest polynomial.
 * @param x Point at which the polynomials are evaluated.
 * @param out Array of `n + 1` doubles which receives the polynomial values.
 */
FDG_INTERNAL
void legendre_eval_bonnet_all(unsigned n, double x, double FDG_ARRAY_ARG(out, n + 1));

/**
 * @brief Evaluate all Legendre polynomials up to degree n with a strided layout.
 *
 * Computes `out[k * stride + offset] = P_k(x)` for `k = 0..n`. This allows
 * writing the values of several points into the same array with one stride.
 *
 * @param n Degree of the highest polynomial.
 * @param x Point at which the polynomials are evaluated.
 * @param stride Distance in doubles between successive polynomial values.
 * @param offset Index of the first written element within the array.
 * @param out Array of at least `(n + 1) * stride` doubles, of which the
 *        elements `offset + k * stride` are written.
 */
FDG_INTERNAL
void legendre_eval_bonnet_all_stride(unsigned n, double x, unsigned stride, unsigned offset,
                                     double FDG_ARRAY_ARG(out, (n + 1) * stride));

#endif // FDG_LEGENDRE_H
