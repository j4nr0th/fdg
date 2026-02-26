//
// Created by jan on 29.9.2024.
//

#include "lagrange.h"

FDG_INTERNAL
void lagrange_polynomial_denominators(unsigned n, const double FDG_ARRAY_ARG(nodes, restrict static n),
                                      double FDG_ARRAY_ARG(denominators, restrict n))
{
    denominators[0] = 1.0;
    // Compute the first denominator directly
    for (unsigned j = 1; j < n; ++j)
    {
        const double dif = nodes[0] - nodes[j];
        denominators[0] *= dif;
        denominators[j] = -dif;
    }

    //  Compute the rest as a loop now that all entries are initialized
    for (unsigned i = 1; i < n; ++i)
    {
        for (unsigned j = i + 1; j < n; ++j)
        {
            const double dif = nodes[i] - nodes[j];
            denominators[i] *= +dif;
            denominators[j] *= -dif;
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_coefficients(unsigned n, unsigned j, const double FDG_ARRAY_ARG(nodes, restrict static n),
                                      double FDG_ARRAY_ARG(coefficients, restrict n))
{
    coefficients[0] = 1.0;
    for (unsigned i = 0; i < j; ++i)
    {
        const double coeff = -nodes[i];
        coefficients[i + 1] = 0.0;
        for (unsigned k = i + 1; k > 0; --k)
        {
            coefficients[k] = coefficients[k - 1] + coeff * coefficients[k];
        }
        coefficients[0] *= coeff;
    }
    for (unsigned i = j + 1; i < n; ++i)
    {
        const double coeff = -nodes[i];
        coefficients[i] = 0.0;
        for (unsigned k = i; k > 0; --k)
        {
            coefficients[k] = coefficients[k - 1] + coeff * coefficients[k];
        }
        coefficients[0] *= coeff;
    }
}

FDG_INTERNAL
void lagrange_polynomial_values(const unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                const unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                double FDG_ARRAY_ARG(values, restrict n_roots *n_pos),
                                double FDG_ARRAY_ARG(work, restrict n_roots))
{
    lagrange_polynomial_denominators(n_roots, p_roots, work);

    //  Invert the denominator
    for (unsigned i = 0; i < n_roots; ++i)
    {
        work[i] = 1.0 / work[i];
    }

    //  Compute the numerator now
    for (unsigned k = 0; k < n_pos; ++k)
    {
        double *const row = values + n_roots * k;
        //  First loop can be used to initialize the row
        {
            const double dif = p_pos[k] - p_roots[0];
            row[0] = 1.0;
            for (unsigned j = 1; j < n_roots; ++j)
            {
                row[j] = +dif;
            }
        }
        for (unsigned i = 1; i < n_roots; ++i)
        {
            const double dif = p_pos[k] - p_roots[i];
            for (unsigned j = 0; j < i; ++j)
            {
                row[j] *= +dif;
            }
            for (unsigned j = i + 1; j < n_roots; ++j)
            {
                row[j] *= +dif;
            }
        }
        //  Multiply by 1/denominator
        for (unsigned i = 0; i < n_roots; ++i)
        {
            row[i] *= work[i];
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_values_2(const unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                  const unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                  double FDG_ARRAY_ARG(values, restrict n_roots *n_pos))
{
    // Special case: n_roots == 1, where there is 1 basis, which is 1 everywhere
    if (n_roots == 1)
    {
        // Constant everywhere, then just return
        for (unsigned k = 0; k < n_pos; ++k)
            values[k] = 1.0;

        return;
    }

    double *const denominators = values + n_roots * (n_pos - 1);
    // Store denominators in the last row
    lagrange_polynomial_denominators(n_roots, p_roots, denominators);

    //  Compute the numerators now
    for (unsigned k = 0; k < n_pos - 1; ++k)
    {
        double *const row = values + n_roots * k;
        //  The first loop can be used to initialize the row
        {
            const double dif = p_pos[k] - p_roots[0];
            row[0] = 1.0;
            for (unsigned j = 1; j < n_roots; ++j)
            {
                row[j] = +dif;
            }
        }
        // Deal with the rest of the iterations
        for (unsigned i = 1; i < n_roots; ++i)
        {
            const double dif = p_pos[k] - p_roots[i];
            for (unsigned j = 0; j < i; ++j)
            {
                row[j] *= +dif;
            }
            for (unsigned j = i + 1; j < n_roots; ++j)
            {
                row[j] *= +dif;
            }
        }
        //  Divide by the denominator
        for (unsigned i = 0; i < n_roots; ++i)
        {
            row[i] /= denominators[i];
        }
    }

    const unsigned k = n_pos - 1;
    double *const row = values + n_roots * (k);
    // Here we already have denominators in the array, so we deal with them on the first run
    {
        const double dif = p_pos[k] - p_roots[0];
        row[0] = 1.0 / denominators[0];
        for (unsigned j = 1; j < n_roots; ++j)
        {
            row[j] = +dif / denominators[j];
        }
    }
    // The rest of these are the same
    for (unsigned i = 1; i < n_roots; ++i)
    {
        const double dif = p_pos[k] - p_roots[i];
        for (unsigned j = 0; j < i; ++j)
        {
            row[j] *= +dif;
        }
        for (unsigned j = i + 1; j < n_roots; ++j)
        {
            row[j] *= +dif;
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_values_transposed(const unsigned n_in, const double FDG_ARRAY_ARG(pos, static n_in),
                                           const unsigned n_nodes, const double FDG_ARRAY_ARG(x, static n_nodes),
                                           double FDG_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                           double FDG_ARRAY_ARG(work, restrict n_nodes))
{
    lagrange_polynomial_denominators(n_nodes, x, work);

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work[i] = 1.0 / work[i];
    }

    //  Compute the numerator now
    for (unsigned k = 0; k < n_in; ++k)
    {
        double *const col = weights + k;
        //  First loop can be used to initialize the column
        {
            const double dif = pos[k] - x[0];
            col[0] = 1.0;
            for (unsigned j = 1; j < n_nodes; ++j)
            {
                col[n_in * j] = +dif;
            }
        }
        for (unsigned i = 1; i < n_nodes; ++i)
        {
            const double dif = pos[k] - x[i];
            for (unsigned j = 0; j < i; ++j)
            {
                col[n_in * j] *= +dif;
            }
            for (unsigned j = i + 1; j < n_nodes; ++j)
            {
                col[n_in * j] *= +dif;
            }
        }
        //  Multiply by 1/denominator
        for (unsigned i = 0; i < n_nodes; ++i)
        {
            col[n_in * i] *= work[i];
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_denominators_stride(const unsigned n_roots,
                                             const double FDG_ARRAY_ARG(p_roots, restrict static n_roots),
                                             const unsigned stride,
                                             double FDG_ARRAY_ARG(denominators, restrict(n_roots - 1) * stride + 1))
{
    // Initialize the first denominator manually
    denominators[0] = 1.0;

    // Compute the first denominator directly and initialize the rest
    for (unsigned j = 1; j < n_roots; ++j)
    {
        const double dif = p_roots[0] - p_roots[j];
        denominators[0] *= dif;
        denominators[j * stride] = -dif;
    }

    //  Compute the rest as a loop now that all entries are initialized
    for (unsigned i = 1; i < n_roots; ++i)
    {
        for (unsigned j = i + 1; j < n_roots; ++j)
        {
            const double dif = p_roots[i] - p_roots[j];
            denominators[i * stride] *= +dif;
            denominators[j * stride] *= -dif;
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_denominators_apply_stride(
    const unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, restrict static n_roots), const unsigned stride,
    double FDG_ARRAY_ARG(denominators, restrict(n_roots - 1) * stride + 1))
{
    for (unsigned i = 0; i < n_roots; ++i)
    {
        for (unsigned j = i + 1; j < n_roots; ++j)
        {
            const double dif = p_roots[i] - p_roots[j];
            denominators[i * stride] /= +dif;
            denominators[j * stride] /= -dif;
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_values_transposed_2(const unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                             const unsigned n_roots,
                                             const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                             double FDG_ARRAY_ARG(values, restrict n_roots *n_pos))
{
    // Stores denominators as the last element of each row
    lagrange_polynomial_denominators_stride(n_roots, p_roots, n_pos, values + (n_pos - 1));

    //  Compute the numerator now
    for (unsigned k = 0; k < n_pos; ++k)
    {
        //  The first loop can be used to initialize the column
        {
            const double dif = p_pos[k] - p_roots[0];
            // We avoid extra storage for denominators by placing them as the last
            // element of each row. This allows for them to stay there until the
            // last column is computed, at which point the result of the calculation
            // they're needed for is used.
            values[k + 0 * n_pos] = 1.0 / values[(n_pos - 1) + n_pos * 0];
            for (unsigned j = 1; j < n_roots; ++j)
            {
                values[k + n_pos * j] = +dif / values[(n_pos - 1) + n_pos * j];
            }
        }
        for (unsigned i = 1; i < n_roots; ++i)
        {
            const double dif = p_pos[k] - p_roots[i];
            for (unsigned j = 0; j < i; ++j)
            {
                values[k + n_pos * j] *= +dif;
            }
            for (unsigned j = i + 1; j < n_roots; ++j)
            {
                values[k + n_pos * j] *= +dif;
            }
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_first_derivative(const unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                          const unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                          double FDG_ARRAY_ARG(weights, restrict n_roots *n_pos),
                                          /* cache for denominators (once per fn) */
                                          double FDG_ARRAY_ARG(work1, restrict n_roots),
                                          /* cache for differences (once per node) */
                                          double FDG_ARRAY_ARG(work2, restrict n_roots))
{
    // compute denominators
    lagrange_polynomial_denominators(n_roots, p_roots, work1);

    //  Invert the denominator
    for (unsigned i = 0; i < n_roots; ++i)
    {
        work1[i] = 1.0 / work1[i];
    }

    //  Now loop per node
    for (unsigned ipos = 0; ipos < n_pos; ++ipos)
    {
        const double v = p_pos[ipos];
        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Compute the differences
            work2[j] = v - p_roots[j];
            //  Initialize the row of weights about to be computed
            weights[n_roots * ipos + j] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_roots; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = i + 1; k < n_roots; ++k)
                {
                    dlijdx *= work2[k];
                }
                //  L_i^j and L_j^i have same numerators
                weights[n_roots * ipos + j] += dlijdx;
                weights[n_roots * ipos + i] += dlijdx;
            }
        }

        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Initialize the row of weights about to be computed
            weights[n_roots * ipos + j] *= work1[j];
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_first_derivative_2(const unsigned n_pos, const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                            const unsigned n_roots, const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                            double FDG_ARRAY_ARG(values, restrict n_roots *n_pos))
{
    // Ideally, we would have two extra arrays - one to store denominators and another
    // to use as per-node storage for numerator terms. Since that would require extra buffers,
    // we instead re-use the `values` array up until the last two positions that have to be
    // computed. At that point we compute the numerator on-the-go and apply the denominators
    // to the result.

    // compute denominators (if necessary
    if (n_pos > 2)
        lagrange_polynomial_denominators(n_roots, p_roots, values + (size_t)(n_pos - 1) * n_roots);

    //  Now loop per node (until the last two)
    for (unsigned i_pos = 0; i_pos + 2 < n_pos; ++i_pos)
    {
        const double v = p_pos[i_pos];
        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Compute the differences
            values[(i_pos + 1) * n_roots + j] = v - p_roots[j];
            //  Initialize the row of weights about to be computed
            values[i_pos * n_roots + j] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_roots; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= values[(i_pos + 1) * n_roots + k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= values[(i_pos + 1) * n_roots + k];
                }
                for (unsigned k = i + 1; k < n_roots; ++k)
                {
                    dlijdx *= values[(i_pos + 1) * n_roots + k];
                }
                //  L_i^j and L_j^i have same numerators
                values[i_pos * n_roots + j] += dlijdx;
                values[i_pos * n_roots + i] += dlijdx;
            }
        }

        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Divide by the denominator
            values[i_pos * n_roots + j] /= values[(n_pos - 1) * n_roots + j];
        }
    }

    //  Second to last node
    if (n_pos > 1)
    {
        const unsigned i_pos = n_pos - 2;
        const double v = p_pos[i_pos];
        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Compute the differences
            values[(i_pos + 1) * n_roots + j] = v - p_roots[j];
            //  Initialize the row of weights about to be computed
            values[i_pos * n_roots + j] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_roots; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= values[(i_pos + 1) * n_roots + k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= values[(i_pos + 1) * n_roots + k];
                }
                for (unsigned k = i + 1; k < n_roots; ++k)
                {
                    dlijdx *= values[(i_pos + 1) * n_roots + k];
                }
                //  L_i^j and L_j^i have same numerators
                values[i_pos * n_roots + j] += dlijdx;
                values[i_pos * n_roots + i] += dlijdx;
            }
        }

        // Apply the denominator
        lagrange_polynomial_denominators_apply_stride(n_roots, p_roots, 1, values + (size_t)(n_pos - 2) * n_roots);
    }

    // The last node
    if (n_pos > 0)
    {
        const double v = p_pos[n_pos - 1];
        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Initialize the row of weights about to be computed
            values[(n_pos - 1) * n_roots + j] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_roots; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= v - p_roots[k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= v - p_roots[k];
                }
                for (unsigned k = i + 1; k < n_roots; ++k)
                {
                    dlijdx *= v - p_roots[k];
                }
                //  L_i^j and L_j^i have same numerators
                values[(n_pos - 1) * n_roots + j] += dlijdx;
                values[(n_pos - 1) * n_roots + i] += dlijdx;
            }
        }

        // Apply the denominator
        lagrange_polynomial_denominators_apply_stride(n_roots, p_roots, 1, values + (size_t)(n_pos - 1) * n_roots);
    }
}

FDG_INTERNAL
void lagrange_polynomial_first_derivative_transposed(const unsigned n_in, const double FDG_ARRAY_ARG(pos, static n_in),
                                                     const unsigned n_nodes,
                                                     const double FDG_ARRAY_ARG(x, static n_nodes),
                                                     double FDG_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                                     /* cache for denominators (once per fn) */
                                                     double FDG_ARRAY_ARG(work1, restrict n_nodes),
                                                     /* cache for differences (once per node) */
                                                     double FDG_ARRAY_ARG(work2, restrict n_nodes))
{
    // compute denominators
    lagrange_polynomial_denominators(n_nodes, x, work1);

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work1[i] = 1.0 / work1[i];
    }

    //  Now loop per node
    for (unsigned ipos = 0; ipos < n_in; ++ipos)
    {
        const double v = pos[ipos];
        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Compute the differences
            work2[j] = v - x[j];
            //  Initialize the row of weights about to be computed
            weights[ipos + j * n_in] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_nodes; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= work2[k];
                }
                for (unsigned k = i + 1; k < n_nodes; ++k)
                {
                    dlijdx *= work2[k];
                }
                //  L_i^j and L_j^i have same numerators
                weights[ipos + j * n_in] += dlijdx;
                weights[ipos + i * n_in] += dlijdx;
            }
        }

        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Initialize the row of weights about to be computed
            weights[ipos + j * n_in] *= work1[j];
        }
    }
}

FDG_INTERNAL
void lagrange_polynomial_first_derivative_transposed_2(const unsigned n_pos,
                                                       const double FDG_ARRAY_ARG(p_pos, static n_pos),
                                                       const unsigned n_roots,
                                                       const double FDG_ARRAY_ARG(p_roots, static n_roots),
                                                       double FDG_ARRAY_ARG(values, restrict n_roots *n_pos))
{
    // Ideally, we would have two extra arrays - one to store denominators and another
    // to use as per-node storage for numerator terms. Since that would require extra buffers,
    // we instead re-use the `values` array up until the last two positions that have to be
    // computed. At that point we compute the numerator on-the-go and apply the denominators
    // to the result.

    // compute denominators (if necessary
    if (n_pos > 2)
        lagrange_polynomial_denominators_stride(n_roots, p_roots, n_pos, values + (n_pos - 1));

    //  Now loop per node (until the last two)
    for (unsigned i_pos = 0; i_pos + 2 < n_pos; ++i_pos)
    {
        const double v = p_pos[i_pos];
        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Compute the differences
            values[i_pos + 1 + j * n_pos] = v - p_roots[j];
            //  Initialize the row of weights about to be computed
            values[i_pos + j * n_pos] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_roots; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= values[i_pos + 1 + k * n_pos];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= values[i_pos + 1 + k * n_pos];
                }
                for (unsigned k = i + 1; k < n_roots; ++k)
                {
                    dlijdx *= values[i_pos + 1 + k * n_pos];
                }
                //  L_i^j and L_j^i have same numerators
                values[i_pos + j * n_pos] += dlijdx;
                values[i_pos + i * n_pos] += dlijdx;
            }
        }

        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Divide by the denominator
            values[i_pos + j * n_pos] /= values[n_pos - 1 + j * n_pos];
        }
    }

    //  Second to last node
    if (n_pos > 1)
    {
        const unsigned i_pos = n_pos - 2;
        const double v = p_pos[i_pos];
        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Compute the differences
            values[i_pos + 1 + j * n_pos] = v - p_roots[j];
            //  Initialize the row of weights about to be computed
            values[i_pos + j * n_pos] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_roots; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= values[i_pos + 1 + k * n_pos];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= values[i_pos + 1 + k * n_pos];
                }
                for (unsigned k = i + 1; k < n_roots; ++k)
                {
                    dlijdx *= values[i_pos + 1 + k * n_pos];
                }
                //  L_i^j and L_j^i have same numerators
                values[i_pos + j * n_pos] += dlijdx;
                values[i_pos + i * n_pos] += dlijdx;
            }
        }

        // Apply the denominator
        lagrange_polynomial_denominators_apply_stride(n_roots, p_roots, n_pos, values + (n_pos - 2));
    }

    // The last node
    if (n_pos > 0)
    {
        const double v = p_pos[n_pos - 1];
        for (unsigned j = 0; j < n_roots; ++j)
        {
            //  Initialize the row of weights about to be computed
            values[n_pos - 1 + j * n_pos] = 0.0;
        }

        //  Compute term d (L_i^j) / d x
        for (unsigned i = 0; i < n_roots; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                double dlijdx = 1.0;
                //  Loop split into three parts to enforce k != {i,
                //  j}
                for (unsigned k = 0; k < j; ++k)
                {
                    dlijdx *= v - p_roots[k];
                }
                for (unsigned k = j + 1; k < i; ++k)
                {
                    dlijdx *= v - p_roots[k];
                }
                for (unsigned k = i + 1; k < n_roots; ++k)
                {
                    dlijdx *= v - p_roots[k];
                }
                //  L_i^j and L_j^i have same numerators
                values[n_pos - 1 + j * n_pos] += dlijdx;
                values[n_pos - 1 + i * n_pos] += dlijdx;
            }
        }

        // Apply the denominator
        lagrange_polynomial_denominators_apply_stride(n_roots, p_roots, n_pos, values + (n_pos - 1));
    }
}

FDG_INTERNAL
interp_result_t lagrange_polynomial_second_derivative(unsigned n_in, const double FDG_ARRAY_ARG(pos, static n_in),
                                                      unsigned n_nodes, const double FDG_ARRAY_ARG(x, static n_nodes),
                                                      double FDG_ARRAY_ARG(weights, restrict n_nodes *n_in),
                                                      double FDG_ARRAY_ARG(work1, restrict n_nodes),
                                                      double FDG_ARRAY_ARG(work2, restrict n_nodes))
{
    // compute denominators
    lagrange_polynomial_denominators(n_nodes, x, work1);

    //  Invert the denominator
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        work1[i] = 1.0 / work1[i];
    }

    //  Now loop per node
    for (unsigned ipos = 0; ipos < n_in; ++ipos)
    {
        const double v = pos[ipos];
        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Compute the differences
            work2[j] = v - x[j];
            //  Initialize the row of weights about to be computed
            weights[n_nodes * ipos + j] = 0.0;
        }

        for (unsigned i = 0; i < n_nodes; ++i)
        {
            for (unsigned j = 0; j < i; ++j)
            {
                for (unsigned k = 0; k < j; ++k)
                {
                    double dlijkdx = 1.0;
                    //  Loop split into four parts to enforce l
                    //  != {i, j, k}
                    for (unsigned l = 0; l < k; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    for (unsigned l = k + 1; l < j; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    for (unsigned l = j + 1; l < i; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    for (unsigned l = i + 1; l < n_nodes; ++l)
                    {
                        dlijkdx *= work2[l];
                    }
                    //  L_i^j and L_j^i have same numerators
                    weights[n_nodes * ipos + k] += 2 * dlijkdx;
                    weights[n_nodes * ipos + j] += 2 * dlijkdx;
                    weights[n_nodes * ipos + i] += 2 * dlijkdx;
                }
            }
        }

        for (unsigned j = 0; j < n_nodes; ++j)
        {
            //  Initialize the row of weights about to be computed
            weights[n_nodes * ipos + j] *= work1[j];
        }
    }

    return FDG_SUCCESS;
}
