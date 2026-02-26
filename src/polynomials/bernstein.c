//
// Created by jan on 5.11.2024.
//

#include "bernstein.h"

FDG_INTERNAL
void bernstein_from_power_series(const unsigned n, double FDG_ARRAY_ARG(coeffs, static n))
{
    unsigned base_coefficient = 1;
    for (unsigned k = 0; k < n; ++k)
    {
        const double beta = coeffs[k];

        // Update the remaining entries
        const int diff = (int)(n - k - 1);
        int local = diff;
        for (int i = 1; i < diff + 1; ++i)
        {
            coeffs[k + i] += beta * (double)local;
            // Incorporate the (-1)^i into the binomial coefficient
            local = (local * ((int)i - (int)diff)) / (int)(i + 1);
        }

        coeffs[k] = beta / (double)base_coefficient;
        // Update the binomial coefficient of the polynomial
        base_coefficient = (base_coefficient * (diff)) / (k + 1);
    }
}

FDG_INTERNAL
void bernstein_interpolation_vector(const double t, const unsigned n, double FDG_ARRAY_ARG(out, restrict n + 1))
{
    //  Bernstein polynomials follow the following recursion:
    //
    //  B^{N+1}_k(t) = t B^{N}_{k-1}(t) + (1-t) B^{N}_k(t)
    const double a = 1.0 - t;
    const double b = t;
    //  this is to store the value about to be overridden
    out[0] = 1.0;

    for (unsigned i = 0; i < n; ++i)
    {
        out[i + 1] = out[i] * b;
        for (unsigned j = i; j > 0; --j)
        {
            out[j] = b * out[j - 1] + a * out[j];
        }
        out[0] *= a;
    }
}

FDG_INTERNAL
void bernstein_interpolation_value_derivative_matrix(const unsigned n_in,
                                                     const double FDG_ARRAY_ARG(t, restrict static n_in),
                                                     const unsigned n,
                                                     double FDG_ARRAY_ARG(out_value, restrict(n + 1) * n_in),
                                                     double FDG_ARRAY_ARG(out_derivative, restrict(n + 1) * n_in))
{
    // Quick bail if n is 0
    if (n == 0)
    {
        for (unsigned i_pos = 0; i_pos < n_in; ++i_pos)
        {
            out_value[0 * n_in + i_pos] = 1.0;
            out_derivative[0 * n_in + i_pos] = 0.0;
        }
        return;
    }

    //  Bernstein polynomials follow the following recursion:
    //
    //  B^{N+1}_k(t) = t B^{N}_{k-1}(t) + (1-t) B^{N}_k(t)

    for (unsigned i_pos = 0; i_pos < n_in; ++i_pos)
    {
        const double x = (t[i_pos] + 1) / 2;
        const double a = 1.0 - x;
        const double b = x;

        //  this is to store the value about to be overridden
        out_derivative[0 * n_in + i_pos] = 1.0;

        // Get up to order n - 1 in the derivative output
        for (unsigned i = 0; i < n - 1; ++i)
        {
            out_derivative[(i + 1) * n_in + i_pos] = out_derivative[i * n_in + i_pos] * b;
            for (unsigned j = i; j > 0; --j)
            {
                out_derivative[j * n_in + i_pos] =
                    b * out_derivative[(j - 1) * n_in + i_pos] + a * out_derivative[j * n_in + i_pos];
            }
            out_derivative[0 * n_in + i_pos] *= a;
        }

        // Get the last order into the value array
        out_value[n * n_in + i_pos] = out_derivative[(n - 1) * n_in + i_pos] * b;
        for (unsigned j = n - 1; j > 0; --j)
        {
            out_value[j * n_in + i_pos] =
                b * out_derivative[(j - 1) * n_in + i_pos] + a * out_derivative[j * n_in + i_pos];
        }
        out_value[0 * n + i_pos] = out_derivative[0 * n + i_pos] * a;

        // Convert the order n - 1 from the derivative output into the derivative in place
        out_derivative[n * n_in + i_pos] = n * out_derivative[(n - 1) * n_in + i_pos];
        for (unsigned i = n - 1; i > 0; --i)
        {
            out_derivative[i * n_in + i_pos] =
                n * (out_derivative[(i - 1) * n_in + i_pos] - out_derivative[i * n_in + i_pos]);
        }
        out_derivative[0 * n_in + i_pos] *= -(double)n;
        // Apply the chain rule and scale the gradient!
        for (unsigned i = 0; i < n + 1; ++i)
        {
            out_derivative[i * n_in + i_pos] /= 2.0;
        }
    }
}
