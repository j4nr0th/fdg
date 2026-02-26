//
// Created by jan on 2025-09-07.
//

#ifndef FDG_LEGENDRE_H
#define FDG_LEGENDRE_H

#include "../common/common_defines.h"

FDG_INTERNAL
void legendre_eval_bonnet_two(unsigned n, double x, double FDG_ARRAY_ARG(out, 2));

FDG_INTERNAL
void legendre_eval_bonnet(unsigned n, double x, unsigned m, double FDG_ARRAY_ARG(out, m));

FDG_INTERNAL
void legendre_eval_bonnet_all(unsigned n, double x, double FDG_ARRAY_ARG(out, n + 1));

FDG_INTERNAL
void legendre_eval_bonnet_all_stride(unsigned n, double x, unsigned stride, unsigned offset,
                                     double FDG_ARRAY_ARG(out, (n + 1) * stride));

#endif // FDG_LEGENDRE_H
