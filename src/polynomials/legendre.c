//
// Created by jan on 2025-09-07.
//

#include "legendre.h"

FDG_INTERNAL
void legendre_eval_bonnet_two(const unsigned n, const double x, double FDG_ARRAY_ARG(out, 2))
{
    ASSERT(n >= 2, "n must be at least 2, but was %u", n);
    // n >= 2
    double v1 = 1.0;
    double v2 = x;
    for (unsigned i = 2; i < n + 1; ++i)
    {
        const double k1 = (2 * i - 1) * x;
        const double k2 = (i - 1);
        const double new = (k1 * v2 - k2 * v1) / (double)(i);
        v1 = v2;
        v2 = new;
    }
    out[0] = v1;
    out[1] = v2;
}

FDG_INTERNAL
void legendre_eval_bonnet(const unsigned n, const double x, const unsigned m, double FDG_ARRAY_ARG(out, m))
{
    ASSERT(n >= m, "m can not be more than n, but was n=%u while m=%u", n, m);
    ASSERT(m > 0, "m must be greater than zero");
    if (n + 1 == m)
    {
        out[0] = 1.0;
    }

    if (n - 1 > m)
    {
        out[n - m] = x;
    }
    if (n <= 1)
        return;

    double v1 = 1.0;
    double v2 = x;
    for (unsigned i = 2; i < n + 1; ++i)
    {
        const double k1 = (2 * i - 1) * x;
        const double k2 = (i - 1);
        const double new = (k1 * v2 - k2 * v1) / (double)(i);
        // The position of P_k is at k - (n + 1 - m)
        if (i > (n + 1 - m))
        {
            out[i - (n + 1 - m)] = new;
        }
        v1 = v2;
        v2 = new;
    }
}

FDG_INTERNAL
void legendre_eval_bonnet_all(const unsigned n, const double x, double FDG_ARRAY_ARG(out, n + 1))
{
    out[0] = 1.0;
    if (n == 0)
    {
        return;
    }
    out[1] = x;
    if (n == 1)
    {
        return;
    }

    double v1 = 1.0;
    double v2 = x;
    for (unsigned i = 2; i < n + 1; ++i)
    {
        const double k1 = (2 * i - 1) * x;
        const double k2 = (i - 1);
        const double new = (k1 * v2 - k2 * v1) / (double)(i);
        out[i] = new;
        v1 = v2;
        v2 = new;
    }
}

FDG_INTERNAL

FDG_INTERNAL
void legendre_eval_bonnet_all_stride(const unsigned n, const double x, const unsigned stride, const unsigned offset,
                                     double FDG_ARRAY_ARG(out, (n + 1) * stride))
{
    out += offset;
    out[0 * stride] = 1.0;
    if (n == 0)
    {
        return;
    }
    out[1 * stride] = x;
    if (n == 1)
    {
        return;
    }

    double v1 = 1.0;
    double v2 = x;
    for (unsigned i = 2; i < n + 1; ++i)
    {
        const double k1 = (2 * i - 1) * x;
        const double k2 = (i - 1);
        const double new = (k1 * v2 - k2 * v1) / (double)(i);
        out[i * stride] = new;
        v1 = v2;
        v2 = new;
    }
}
