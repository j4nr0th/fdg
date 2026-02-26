//
// Created by jan on 29.9.2024.

#ifndef LAGRANGE_H
#define LAGRANGE_H
#include "../common/common_defines.h"
#include "../common/error.h"

/**
 * @brief Compute common denominators of Lagrange polynomials.
 *
 * @param n Number of nodes.
 * @param nodes Array with nodes where the Lagrange polynomial is zero.
 * @param denominators Array which receives the denominators.
 */
FDG_INTERNAL
void lagrange_polynomial_denominators(unsigned n, const double FDG_ARRAY_ARG(nodes, restrict static n),
                                      double FDG_ARRAY_ARG(denominators, restrict n));
/**
 * @brief Compute values of Lagrange polynomial coefficients without dividing by the common denominator.
 *
 * @param n Number of nodes.
 * @param j Index of the Lagrange polynomial. At that node its value will be non-zero.
 * @param nodes Array with nodes where the Lagrange polynomial is zero.
 * @param coefficients Array which receives the coefficients. The term's index corresponds to it's power of x.
 */
FDG_INTERNAL
void lagrange_polynomial_coefficients(unsigned n, unsigned j, const double FDG_ARRAY_ARG(nodes, restrict static n),
                                      double FDG_ARRAY_ARG(coefficients, restrict n));

/**
 * @brief Compute values of Lagrange polynomials with given nodes at specified
 * locations. The interpolation can be computed for any function on the same
 * mesh by taking the inner product of the weight matrix with the function
 * values.
 *
 * @param n_pos Number of points where polynomials should be evaluated.
 * @param p_pos Points where the Lagrange polynomials should be evaluated at.
 * @param n_roots Number or roots of Lagrange polynomials, which is also the order of the polynomials.
 * @param p_roots Roots of the lagrange polynomials.
 * @param values Array which receives the values of Lagrange polynomials.
 * @param work Array used to store intermediate results.
 *
 */
FDG_INTERNAL
void lagrange_polynomial_values(unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos), unsigned n_roots,
                                const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                double FDG_ARRAY_ARG(values, restrict n_roots *n_pos),
                                double FDG_ARRAY_ARG(work, restrict n_roots));

FDG_INTERNAL
void lagrange_polynomial_values_2(unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos), unsigned n_roots,
                                  const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                  double FDG_ARRAY_ARG(values, restrict n_roots *n_pos));

FDG_INTERNAL
void lagrange_polynomial_values_transposed(unsigned n_in, const double FDG_ARRAY_ARG(pos, static n_in),
                                           unsigned n_nodes, const double FDG_ARRAY_ARG(x, static n_nodes),
                                           double FDG_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                           double FDG_ARRAY_ARG(work, restrict n_nodes));

FDG_INTERNAL
void lagrange_polynomial_values_transposed_2(unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                             unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                             double FDG_ARRAY_ARG(values, restrict n_roots *n_pos));

/**
 * @brief Compute the first derivative of Lagrange polynomials with given nodes at
 * specified locations. The interpolation can be computed for any function on
 * the same mesh by taking the inner product of the weight matrix with the
 * function values.
 *
 * @param n_pos Number of points where polynomials should be evaluated.
 * @param p_pos Points where the Lagrange polynomials should be evaluated at.
 * @param n_roots Number or roots of Lagrange polynomials, which is also the order of the polynomials.
 * @param p_roots Roots of the lagrange polynomials.
 * @param weights Array which receives the values of Lagrange polynomials.
 * @param work1 Array used to store intermediate results.
 * @param work2 Array used to store intermediate results.
 */
FDG_INTERNAL
void lagrange_polynomial_first_derivative(unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                          unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                          double FDG_ARRAY_ARG(weights, restrict n_roots *n_pos),
                                          double FDG_ARRAY_ARG(work1, restrict n_roots),
                                          double FDG_ARRAY_ARG(work2, restrict n_roots));

FDG_INTERNAL
void lagrange_polynomial_first_derivative_2(unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                            unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                            double FDG_ARRAY_ARG(values, restrict n_roots *n_pos));

FDG_INTERNAL
void lagrange_polynomial_first_derivative_transposed(unsigned n_in, const double FDG_ARRAY_ARG(pos, static n_in),
                                                     unsigned n_nodes, const double FDG_ARRAY_ARG(x, static n_nodes),
                                                     double FDG_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                                     /* cache for denominators (once per fn) */
                                                     double FDG_ARRAY_ARG(work1, restrict n_nodes),
                                                     /* cache for differences (once per node) */
                                                     double FDG_ARRAY_ARG(work2, restrict n_nodes));

FDG_INTERNAL
void lagrange_polynomial_first_derivative_transposed_2(unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                                       unsigned n_roots,
                                                       const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                                       double FDG_ARRAY_ARG(values, restrict n_roots *n_pos));

/**
 * @brief Compute second derivative of Lagrange polynomials with given nodes at
 * specified locations. The interpolation can be computed for any function on
 * the same mesh by taking the inner product of the weight matrix with the
 * function values.
 *
 * @param n_in Number of points where the interpolation will be needed.
 * @param pos Array of nodes where interpolation will be computed
 * @param n_nodes Number of nodes where the function is known.
 * @param x Array of x-values of nodes where the function is known which must
 * be monotonically increasing.
 * @param weights Array which receives the weights for the interpolation.
 * @param work1 Array used to store intermediate results.
 * @param work2 Array used to store intermediate results.
 *
 * @return `FDG_SUCCESS` on success, `FDG_ERROR_NOT_INCREASING` if `x[i +
 * 1] > x[i]` does not hold for all `i`.
 */
FDG_INTERNAL
interp_result_t lagrange_polynomial_second_derivative(unsigned n_in, const double FDG_ARRAY_ARG(pos, static n_in),
                                                      unsigned n_nodes, const double FDG_ARRAY_ARG(x, static n_nodes),
                                                      double FDG_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                                      double FDG_ARRAY_ARG(work1, restrict n_nodes),
                                                      double FDG_ARRAY_ARG(work2, restrict n_nodes));

#endif // LAGRANGE_H
