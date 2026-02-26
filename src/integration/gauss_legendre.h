#ifndef FDG_GAUSS_LEGENDRE_H
#define FDG_GAUSS_LEGENDRE_H
#include "../common/common_defines.h"

FDG_INTERNAL
int gauss_legendre_nodes_weights(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n),
                                 double FDG_ARRAY_ARG(w, restrict n));

FDG_INTERNAL
int gauss_legendre_nodes(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n));

#endif // FDG_GAUSS_LEGENDRE_H
