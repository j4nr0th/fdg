//
// Created by jan on 27.1.2025.
//

#ifndef GAUSSLOBATTO_H
#define GAUSSLOBATTO_H
#include "../common/common_defines.h"

/**
 * @brief Compute the nodes and weights of the Gauss-Lobatto quadrature rule.
 *
 * The nodes include the interval endpoints -1 and 1, and the remaining
 * nodes are computed numerically with Newton iterations on the interval
 * [-1, 1]. The weights are the corresponding quadrature weights, which
 * integrate polynomials of degree up to `2n - 3` exactly.
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
int gauss_lobatto_nodes_weights(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n),
                                double FDG_ARRAY_ARG(w, restrict n));

/**
 * @brief Compute the nodes of the Gauss-Lobatto quadrature rule.
 *
 * Same nodes as gauss_lobatto_nodes_weights, without the weights.
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
int gauss_lobatto_nodes(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n));

#endif // GAUSSLOBATTO_H
