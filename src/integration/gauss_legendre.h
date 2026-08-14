#ifndef FDG_GAUSS_LEGENDRE_H
#define FDG_GAUSS_LEGENDRE_H
#include "../common/common_defines.h"

/**
 * @brief Compute the nodes and weights of the Gauss-Legendre quadrature rule.
 *
 * The nodes are the roots of the Legendre polynomial of degree `n`,
 * computed numerically with Newton iterations on the interval [-1, 1]. The
 * weights are the corresponding quadrature weights, which integrate
 * polynomials of degree up to `2n - 1` exactly.
 *
 * @param n Number of nodes and weights. Must be at least 1.
 * @param tol Convergence tolerance for the Newton iterations.
 * @param max_iter Maximum number of Newton iterations per node.
 * @param x Array of `n` doubles which receives the nodes.
 * @param w Array of `n` doubles which receives the weights.
 * @return The number of nodes for which the iteration did not converge to
 *         the requested tolerance; zero on success. The arrays are filled
 *         regardless.
 */
FDG_INTERNAL
int gauss_legendre_nodes_weights(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n),
                                 double FDG_ARRAY_ARG(w, restrict n));

/**
 * @brief Compute the nodes of the Gauss-Legendre quadrature rule.
 *
 * Same nodes as gauss_legendre_nodes_weights, without the weights.
 *
 * @param n Number of nodes. Must be at least 1.
 * @param tol Convergence tolerance for the Newton iterations.
 * @param max_iter Maximum number of Newton iterations per node.
 * @param x Array of `n` doubles which receives the nodes.
 * @return The number of nodes for which the iteration did not converge to
 *         the requested tolerance; zero on success. The array is filled
 *         regardless.
 */
FDG_INTERNAL
int gauss_legendre_nodes(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n));

#endif // FDG_GAUSS_LEGENDRE_H
