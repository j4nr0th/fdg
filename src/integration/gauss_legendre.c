
#include "gauss_legendre.h"
#include "../polynomials/legendre.h"
#include <math.h>

FDG_INTERNAL
int gauss_legendre_nodes_weights(const unsigned n, const double tol, const unsigned max_iter,
                                 double FDG_ARRAY_ARG(x, restrict n), double FDG_ARRAY_ARG(w, restrict n))
{
    ASSERT(n > 0, "n can not be zero.");
    if (n == 1)
    {
        x[0] = 0.0;
        w[0] = 2.0;
        return 0;
    }
    if (n == 2)
    {
        x[0] = -1.0;
        x[1] = 1.0;
        w[0] = 1.0;
        w[1] = 1.0;
        return 0;
    }

    int non_converged = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        // Make an initial guess
        double new_x = cos(M_PI * (double)(4 * i + 3) / (double)(2 * (2 * (n) + 1)));
        double error = 1.0;
        double leg_poly[2];
        for (unsigned iter = 0; iter < max_iter && error > tol; ++iter)
        {
            legendre_eval_bonnet_two(n, new_x, leg_poly);
            const double denominator = 1 - new_x * new_x;
            // Denominator is not here to reduce divisions
            const double dy = (n - 1) * (leg_poly[0] - new_x * leg_poly[1]);
            const double dx = leg_poly[1] / dy * denominator;
            new_x -= dx;
            error = fabs(dx);
        }
        non_converged += (error > tol);
        x[n - 1 - i] = new_x;
        legendre_eval_bonnet_two(n - 1, new_x, leg_poly);

        w[n - 1 - i] = 2.0 / (n * n * leg_poly[1] * leg_poly[1]) * (1 - new_x * new_x);
    }
    return non_converged;
}

FDG_INTERNAL
int gauss_legendre_nodes(const unsigned n, const double tol, const unsigned max_iter,
                         double FDG_ARRAY_ARG(x, restrict n))
{
    ASSERT(n > 0, "n can not be zero.");
    if (n == 1)
    {
        x[0] = 0.0;
        return 0;
    }
    if (n == 2)
    {
        x[0] = -1.0;
        x[1] = 1.0;
        return 0;
    }

    int non_converged = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        // Make an initial guess
        double new_x = cos(M_PI * (double)(4 * i + 3) / (double)(2 * (2 * (n) + 1)));
        double error = 1.0;
        double leg_poly[2];
        for (unsigned iter = 0; iter < max_iter && error > tol; ++iter)
        {
            legendre_eval_bonnet_two(n, new_x, leg_poly);
            const double denominator = 1 - new_x * new_x;
            // Denominator is not here to reduce divisions
            const double dy = (n - 1) * (leg_poly[0] - new_x * leg_poly[1]);
            const double dx = leg_poly[1] / dy * denominator;
            new_x -= dx;
            error = fabs(dx);
        }
        non_converged += (error > tol);
        x[n - 1 - i] = new_x;
    }
    return non_converged;
}
