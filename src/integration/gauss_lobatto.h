//
// Created by jan on 27.1.2025.
//

#ifndef GAUSSLOBATTO_H
#define GAUSSLOBATTO_H
#include "../common/common_defines.h"

FDG_INTERNAL
int gauss_lobatto_nodes_weights(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n),
                                double FDG_ARRAY_ARG(w, restrict n));

FDG_INTERNAL
int gauss_lobatto_nodes(unsigned n, double tol, unsigned max_iter, double FDG_ARRAY_ARG(x, restrict n));

#endif // GAUSSLOBATTO_H
