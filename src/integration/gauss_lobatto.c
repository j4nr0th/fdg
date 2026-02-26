/**
 * Implementation based on Jupyter in:
 *
 * Ciardelli, C., Bozdağ, E., Peter, D., and Van der Lee, S., 2022. SphGLLTools: A toolbox for visualization of large
 * seismic model files based on 3D spectral-element meshes. Computer & Geosciences, v. 159, 105007,
 * doi: https://doi.org/10.1016/j.cageo.2021.105007
 */

#include "gauss_lobatto.h"
#include "../polynomials/legendre.h"
#include <math.h>

FDG_INTERNAL
int gauss_lobatto_nodes_weights(const unsigned n, const double tol, const unsigned max_iter,
                                double FDG_ARRAY_ARG(x, restrict n), double FDG_ARRAY_ARG(w, restrict n))
{
    if (n == 1)
    {
        x[0] = 0.0;
        w[0] = 2.0;
        return 0;
    }
    if (n == 2)
    {
        x[0] = -1.0;
        x[1] = +1.0;
        w[0] = 1.0;
        w[1] = 1.0;
        return 0;
    }

    int non_converged = 0;
    // n >= 2
    x[0] = -1.0;
    x[n - 1] = +1.0;
    w[n - 1] = w[0] = 2.0 / (double)(n * (n - 1));
    const double kx_1 = (1.0 - 3.0 * (n - 2) / (double)(8.0 * (n - 1) * (n - 1) * (n - 1)));
    const double kx_2 = M_PI / (4.0 * (n - 1) + 1);
    for (unsigned i = 2; i < n; ++i)
    {
        double new_x = kx_1 * cos(kx_2 * (4 * i - 3));
        double error = 1.0;
        double leg_poly[2];
        for (unsigned iter = 0; iter < max_iter && error > tol; ++iter)
        {
            legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
            const double denominator = 1 - new_x * new_x;
            const double dy = (n - 1) * (leg_poly[0] - new_x * leg_poly[1]) / denominator;
            const double d2y = (2 * new_x * dy - (n - 1) * n * leg_poly[1]) / denominator;
            const double d3y = (4 * new_x * d2y - ((n - 1) * n - 2) * dy) / denominator;
            const double dx = 2 * dy * d2y / (2 * d2y * d2y - dy * d3y);
            new_x -= dx;
            error = fabs(dx);
        }
        non_converged += (error > tol);
        x[n - i] = new_x;
        legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
        w[n - i] = 2.0 / (n * (n - 1) * leg_poly[1] * leg_poly[1]);
    }
    return non_converged;
}

FDG_INTERNAL
int gauss_lobatto_nodes(const unsigned n, const double tol, const unsigned max_iter,
                        double FDG_ARRAY_ARG(x, restrict n))
{
    if (n == 1)
    {
        x[0] = 0.0;
        return 0;
    }
    if (n == 2)
    {
        x[0] = -1.0;
        x[1] = +1.0;
        return 0;
    }

    int non_converged = 0;
    // n >= 2
    x[0] = -1.0;
    x[n - 1] = +1.0;
    const double kx_1 = (1.0 - 3.0 * (n - 2) / (double)(8.0 * (n - 1) * (n - 1) * (n - 1)));
    const double kx_2 = M_PI / (4.0 * (n - 1) + 1);
    for (unsigned i = 2; i < n; ++i)
    {
        double new_x = kx_1 * cos(kx_2 * (4 * i - 3));
        double error = 1.0;
        double leg_poly[2];
        for (unsigned iter = 0; iter < max_iter && error > tol; ++iter)
        {
            legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
            const double denominator = 1 - new_x * new_x;
            const double dy = (n - 1) * (leg_poly[0] - new_x * leg_poly[1]) / denominator;
            const double d2y = (2 * new_x * dy - (n - 1) * n * leg_poly[1]) / denominator;
            const double d3y = (4 * new_x * d2y - ((n - 1) * n - 2) * dy) / denominator;
            const double dx = 2 * dy * d2y / (2 * d2y * d2y - dy * d3y);
            new_x -= dx;
            error = fabs(dx);
        }
        non_converged += (error > tol);
        x[n - i] = new_x;
        legendre_eval_bonnet_two(n - 1, new_x, leg_poly);
    }
    return non_converged;
}
