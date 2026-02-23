//
// Created by jan on 29.9.2024.
//
#define PY_ARRAY_UNIQUE_SYMBOL _interp
#include "module.h"

//  Numpy
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

// Internal C headers
#include "../common/error.h"
#include "../integration/gauss_lobatto.h"
#include "../polynomials/bernstein.h"
#include "../polynomials/lagrange.h"
#include "basis_objects.h"
#include "integration_objects.h"
#include "kform_objects.h"
#include "mappings.h"
#include "mass_matrices.h"

// Topology
#include "covector_basis.h"
#include "cpyutl.h"
#include "degrees_of_freedom.h"
#include "function_space_objects.h"
#include "incidence.h"
#include "topology/geoid_object.h"
#include "topology/line_object.h"
#include "topology/manifold1d_object.h"
#include "topology/manifold2d_object.h"
#include "topology/manifold_object.h"
#include "topology/surface_object.h"
/**
 *
 *  ALLOCATORS
 *
 */

//  Magic numbers meant for checking with allocators that don't need to store
//  state.
enum
{
    SYSTEM_MAGIC = 0xBadBeef,
    PYTHON_MAGIC = 0x600dBeef,
};

static void *allocate_system(void *state, size_t size)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_RawMalloc(size);
}

static void *reallocate_system(void *state, void *ptr, size_t new_size)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_RawRealloc(ptr, new_size);
}

static void free_system(void *state, void *ptr)
{
    ASSERT(state == (void *)SYSTEM_MAGIC, "Pointer value for system allocator did not match.");
    PyMem_RawFree(ptr);
}

INTERPLIB_INTERNAL
cutl_allocator_t SYSTEM_ALLOCATOR = {
    .allocate = allocate_system,
    .deallocate = free_system,
    .reallocate = reallocate_system,
    .state = (void *)SYSTEM_MAGIC,
};

static void *allocate_python(void *state, size_t size)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_Malloc(size);
}

static void *reallocate_python(void *state, void *ptr, size_t new_size)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    return PyMem_Realloc(ptr, new_size);
}

static void free_python(void *state, void *ptr)
{
    ASSERT(state == (void *)PYTHON_MAGIC, "Pointer value for system allocator did not match.");
    PyMem_Free(ptr);
}

INTERPLIB_INTERNAL
cutl_allocator_t PYTHON_ALLOCATOR = {
    .allocate = allocate_python,
    .deallocate = free_python,
    .reallocate = reallocate_python,
    .state = (void *)PYTHON_MAGIC,
};

#define PRINT_EXPRESSION(expr, fmt) printf(#expr ": " fmt "\n", (expr))

static PyObject *interp_lagrange(PyObject *Py_UNUSED(module), PyObject *args)
{
    PyObject *arg1, *arg2;
    PyArrayObject *out = NULL;
    if (!PyArg_ParseTuple(args, "OO|O!", &arg1, &arg2, &PyArray_Type, &out))
    {
        return NULL;
    }
    PyArrayObject *const roots = (PyArrayObject *)PyArray_FromAny(arg1, PyArray_DescrFromType(NPY_DOUBLE), 1, 1,
                                                                  NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!roots)
    {
        return NULL;
    }
    PyArrayObject *const positions = (PyArrayObject *)PyArray_FromAny(arg2, PyArray_DescrFromType(NPY_DOUBLE), 0, 0,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!positions)
    {
        Py_DECREF(roots);
        return NULL;
    }
    const npy_intp n_roots = PyArray_SIZE(roots);
    const npy_intp n_dim = PyArray_NDIM(positions);
    const npy_intp *const p_dim = PyArray_DIMS(positions);
    if (out)
    {
        if (PyArray_TYPE(out) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_ValueError, "Output array must have the correct type (numpy.double/numpy.float64).");
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        int matches = n_dim + 1 == PyArray_NDIM(out) && PyArray_DIM(out, (int)n_dim) == n_roots;
        for (int n = 0; matches && n < n_dim; ++n)
        {
            matches = p_dim[n] == PyArray_DIM(out, n);
        }
        if (!matches)
        {
            PyErr_SetString(PyExc_ValueError, "Output must have same shape as input array, except for one more"
                                              " dimension, which must be the same length as the array of Lagrange"
                                              " nodes.");
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        Py_INCREF(out);
    }
    else
    {
        npy_intp *const buffer = PyMem_Malloc(sizeof *buffer * (n_dim + 1));
        if (!buffer)
        {
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        for (unsigned i = 0; i < n_dim; ++i)
        {
            buffer[i] = p_dim[i];
        }
        buffer[n_dim] = n_roots;
        out = (PyArrayObject *)PyArray_SimpleNew(n_dim + 1, buffer, NPY_DOUBLE);
        PyMem_Free(buffer);
        if (!out)
        {
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
    }

    const npy_intp n_pos = PyArray_SIZE(positions);

    double *work = PyMem_Malloc(n_roots * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        Py_DECREF(roots);
        Py_DECREF(positions);
        return NULL;
    }

    const double *const p_x = (double *)PyArray_DATA(positions);
    const double *restrict nodes = PyArray_DATA(roots);
    double *const p_out = (double *)PyArray_DATA(out);

    lagrange_polynomial_values(n_pos, p_x, n_roots, nodes, p_out, work);

    PyMem_Free(work);
    return (PyObject *)out;
}

PyDoc_STRVAR(interp_lagrange_doc,
             "lagrange1d(roots: array_like, x: array_like, out: array|None = None, /) -> array\n"
             "Evaluate Lagrange polynomials.\n"
             "\n"
             "This function efficiently evaluates Lagrange basis polynomials, defined by\n"
             "\n"
             ".. math::\n"
             "\n"
             "   \\mathcal{L}^n_i (x) = \\prod\\limits_{j=0, j \\neq i}^{n} \\frac{x - x_j}{x_i - x_j},\n"
             "\n"
             "where the ``roots`` specifies the zeros of the Polynomials :math:`\\{x_0, \\dots, x_n\\}`.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "roots : array_like\n"
             "   Roots of Lagrange polynomials.\n"
             "x : array_like\n"
             "   Points where the polynomials should be evaluated.\n"
             "out : array, optional\n"
             "   Array where the results should be written to. If not given, a new one will be\n"
             "   created and returned. It should have the same shape as ``x``, but with an extra\n"
             "   dimension added, the length of which is ``len(roots)``.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "   Array of Lagrange polynomial values at positions specified by ``x``.\n"

             "Examples\n"
             "--------\n"
             "This example here shows the most basic use of the function to evaluate Lagrange\n"
             "polynomials. First, let us define the roots.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> import numpy as np\n"
             "    >>>\n"
             "    >>> order = 7\n"
             "    >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))\n"
             "\n"
             "Next, we can evaluate the polynomials at positions. Here the interval between the\n"
             "roots is chosen.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from interplib import lagrange1d\n"
             "    >>>\n"
             "    >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)\n"
             "    >>> yvals = lagrange1d(roots, xpos)\n"
             "\n"
             "Note that if we were to give an output array to write to, it would also be the\n"
             "return value of the function (as in no copy is made).\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> yvals is lagrange1d(roots, xpos, yvals)\n"
             "    True\n"
             "\n"
             "Now we can plot these polynomials.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from matplotlib import pyplot as plt\n"
             "    >>>\n"
             "    >>> plt.figure()\n"
             "    >>> for i in range(order + 1):\n"
             "    ...     plt.plot(\n"
             "    ...         xpos,\n"
             "    ...         yvals[..., i],\n"
             "    ...         label=f\"$\\\\mathcal{{L}}^{{{order}}}_{{{i}}}$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> plt.legend()\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n"
             "\n"
             "Accuracy is retained even at very high polynomial order. The following\n"
             "snippet shows that even at absurdly high order of 51, the results still\n"
             "have high accuracy and don't suffer from rounding errors. It also performs\n"
             "well (in this case, the 52 polynomials are each evaluated at 1025 points).\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from time import perf_counter\n"
             "    >>> order = 51\n"
             "    >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))\n"
             "    >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)\n"
             "    >>> t0 = perf_counter()\n"
             "    >>> yvals = lagrange1d(roots, xpos)\n"
             "    >>> t1 = perf_counter()\n"
             "    >>> print(f\"Calculations took {t1 - t0: e} seconds.\")\n"
             "    >>> plt.figure()\n"
             "    >>> for i in range(order + 1):\n"
             "    ...     plt.plot(\n"
             "    ...         xpos,\n"
             "    ...         yvals[..., i],\n"
             "    ...         label=f\"$\\\\mathcal{{L}}^{{{order}}}_{{{i}}}$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> # plt.legend() # No, this is too long\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n");

static PyObject *interp_dlagrange(PyObject *Py_UNUSED(module), PyObject *args)
{
    PyObject *arg1, *arg2;
    PyArrayObject *out = NULL;
    if (!PyArg_ParseTuple(args, "OO|O!", &arg1, &arg2, &PyArray_Type, &out))
    {
        return NULL;
    }
    PyArrayObject *const roots = (PyArrayObject *)PyArray_FromAny(arg1, PyArray_DescrFromType(NPY_DOUBLE), 1, 1,
                                                                  NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!roots)
    {
        return NULL;
    }
    PyArrayObject *const positions = (PyArrayObject *)PyArray_FromAny(arg2, PyArray_DescrFromType(NPY_DOUBLE), 0, 0,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!positions)
    {
        Py_DECREF(roots);
        return NULL;
    }
    const npy_intp n_roots = PyArray_SIZE(roots);
    const npy_intp n_dim = PyArray_NDIM(positions);
    const npy_intp *const p_dim = PyArray_DIMS(positions);

    const size_t size_work = sizeof(double) * 2 * n_roots;
    const size_t size_buffet = sizeof(npy_intp) * (n_dim + 1);
    void *const mem_buffer = PyMem_Malloc(size_buffet > size_work ? size_buffet : size_work);
    if (!mem_buffer)
    {
        Py_DECREF(roots);
        Py_DECREF(positions);
        return NULL;
    }
    double *const work1 = (double *)mem_buffer + 0;
    double *const work2 = (double *)mem_buffer + n_roots;
    npy_intp *const dim_buffer = (npy_intp *)mem_buffer;
    if (out)
    {
        if (PyArray_TYPE(out) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_ValueError, "Output array must have the correct type (numpy.double/numpy.float64).");
            PyMem_Free(mem_buffer);
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        int matches = n_dim + 1 == PyArray_NDIM(out) && PyArray_DIM(out, (int)n_dim) == n_roots;
        for (int n = 0; matches && n < n_dim; ++n)
        {
            matches = p_dim[n] == PyArray_DIM(out, n);
        }
        if (!matches)
        {
            PyErr_SetString(PyExc_ValueError, "Output must have same shape as input array, except for one more"
                                              " dimension, which must be the same length as the array of Lagrange"
                                              " nodes.");
            PyMem_Free(mem_buffer);
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
        Py_INCREF(out);
    }
    else
    {

        for (unsigned i = 0; i < n_dim; ++i)
        {
            dim_buffer[i] = p_dim[i];
        }
        dim_buffer[n_dim] = n_roots;
        out = (PyArrayObject *)PyArray_SimpleNew(n_dim + 1, dim_buffer, NPY_DOUBLE);
        if (!out)
        {
            PyMem_Free(mem_buffer);
            Py_DECREF(roots);
            Py_DECREF(positions);
            return NULL;
        }
    }

    const npy_intp n_pos = PyArray_SIZE(positions);

    const double *const p_x = (double *)PyArray_DATA(positions);
    const double *restrict nodes = PyArray_DATA(roots);
    double *const p_out = (double *)PyArray_DATA(out);

    lagrange_polynomial_first_derivative(n_pos, p_x, n_roots, nodes, p_out, work1, work2);
    PyMem_Free(mem_buffer);

    return (PyObject *)out;
}

PyDoc_STRVAR(interp_dlagrange_doc,
             "dlagrange1d(roots: array_like, x: array_like, out: array|None = None, /) -> array\n"
             "Evaluate derivatives of Lagrange polynomials.\n"
             "\n"
             "This function efficiently evaluates Lagrange basis polynomials derivatives, defined by\n"
             "\n"
             ".. math::\n"
             "\n"
             "   \\frac{d \\mathcal{L}^n_i (x)}{d x} =\n"
             "   \\sum\\limits_{j=0,j \\neq i}^n \\prod\\limits_{k=0, k \\neq i, k \\neq j}^{n}\n"
             "   \\frac{1}{x_i - x_j} \\cdot \\frac{x - x_k}{x_i - x_k},\n"
             "\n"
             "where the ``roots`` specifies the zeros of the Polynomials :math:`\\{x_0, \\dots, x_n\\}`.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "roots : array_like\n"
             "   Roots of Lagrange polynomials.\n"
             "x : array_like\n"
             "   Points where the derivatives of polynomials should be evaluated.\n"
             "out : array, optional\n"
             "   Array where the results should be written to. If not given, a new one will be\n"
             "   created and returned. It should have the same shape as ``x``, but with an extra\n"
             "   dimension added, the length of which is ``len(roots)``.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "   Array of Lagrange polynomial derivatives at positions specified by ``x``.\n"

             "Examples\n"
             "--------\n"
             "This example here shows the most basic use of the function to evaluate derivatives of Lagrange\n"
             "polynomials. First, let us define the roots.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> import numpy as np\n"
             "    >>>\n"
             "    >>> order = 7\n"
             "    >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))\n"
             "\n"
             "Next, we can evaluate the polynomials at positions. Here the interval between the\n"
             "roots is chosen.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from interplib import dlagrange1d\n"
             "    >>>\n"
             "    >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)\n"
             "    >>> yvals = dlagrange1d(roots, xpos)\n"
             "\n"
             "Note that if we were to give an output array to write to, it would also be the\n"
             "return value of the function (as in no copy is made).\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> yvals is dlagrange1d(roots, xpos, yvals)\n"
             "    True\n"
             "\n"
             "Now we can plot these polynomials.\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from matplotlib import pyplot as plt\n"
             "    >>>\n"
             "    >>> plt.figure()\n"
             "    >>> for i in range(order + 1):\n"
             "    ...     plt.plot(\n"
             "    ...         xpos,\n"
             "    ...         yvals[..., i],\n"
             "    ...         label=f\"${{\\\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\\\prime$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> plt.legend()\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n"
             "\n"
             "Accuracy is retained even at very high polynomial order. The following\n"
             "snippet shows that even at absurdly high order of 51, the results still\n"
             "have high accuracy and don't suffer from rounding errors. It also performs\n"
             "well (in this case, the 52 polynomials are each evaluated at 1025 points).\n"
             "\n"
             ".. jupyter-execute::\n"
             "\n"
             "    >>> from time import perf_counter\n"
             "    >>> order = 51\n"
             "    >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))\n"
             "    >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)\n"
             "    >>> t0 = perf_counter()\n"
             "    >>> yvals = dlagrange1d(roots, xpos)\n"
             "    >>> t1 = perf_counter()\n"
             "    >>> print(f\"Calculations took {t1 - t0: e} seconds.\")\n"
             "    >>> plt.figure()\n"
             "    >>> for i in range(order + 1):\n"
             "    ...     plt.plot(\n"
             "    ...         xpos,\n"
             "    ...         yvals[..., i],\n"
             "    ...         label=f\"${{\\\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\\\prime$\"\n"
             "    ...     )\n"
             "    >>> plt.gca().set(\n"
             "    ...     xlabel=\"$x$\",\n"
             "    ...     ylabel=\"$y$\",\n"
             "    ...     title=f\"Lagrange polynomials of order {order}\"\n"
             "    ... )\n"
             "    >>> # plt.legend() # No, this is too long\n"
             "    >>> plt.grid()\n"
             "    >>> plt.show()\n");

static PyObject *interp_d2lagrange(PyObject *module, PyObject *args)
{
    (void)module;
    PyArrayObject *x;
    PyArrayObject *xp;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &xp))
    {
        return NULL;
    }

    ASSERT(PyArray_TYPE(x) == NPY_DOUBLE, "Incorrect type for array x");
    ASSERT(PyArray_TYPE(xp) == NPY_DOUBLE, "Incorrect type for array xp");

    ASSERT(PyArray_NDIM(x) == 1, "Incorrect shape for array x");
    ASSERT(PyArray_NDIM(xp) == 1, "Incorrect shape for array xp");

    npy_intp n_pts = PyArray_SIZE(x);
    npy_intp n_nodes = PyArray_SIZE(xp);

    npy_intp dims[2] = {n_pts, n_nodes};

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    double *work = PyMem_Malloc(2 * n_nodes * sizeof(*work));
    if (work == NULL)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    const double *const p_x = (double *)PyArray_DATA(x);
    double *const p_out = (double *)PyArray_DATA(out);

    const interp_result_t interp_res =
        lagrange_polynomial_second_derivative(n_pts, p_x, n_nodes, PyArray_DATA(xp), p_out, work, work + n_nodes);
    ASSERT(interp_res == INTERP_SUCCESS, "Interpolation failed");

    PyMem_Free(work);
    return (PyObject *)PyArray_Return(out);
}

PyDoc_STRVAR(interp_d2lagrange_doc, "d2lagrange1d(x: np.ndarray, xp: np.ndarray) -> np.ndarray");

INTERPLIB_INTERNAL
PyObject *compute_gauss_lobatto_nodes(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    int order, max_iter = 10;
    double tol = 1e-15;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|id", (char *[4]){"", "max_iter", "tol", NULL}, &order, &max_iter,
                                     &tol))
    {
        return NULL;
    }
    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must be positive, but was given as %i.", order);
        return NULL;
    }
    if (max_iter < 0)
    {
        PyErr_Format(PyExc_ValueError, "Number of maximum iterations must be positive, but was given as %i.", max_iter);
        return NULL;
    }
    if (tol < 0)
    {
        char buffer[16];
        snprintf(buffer, sizeof(buffer), "%g", tol);
        PyErr_Format(PyExc_ValueError, "Tolerance must be positive %s", buffer);
        return NULL;
    }

    const npy_intp array_size = order + 1;
    PyArrayObject *const nodes = (PyArrayObject *)PyArray_SimpleNew(1, &array_size, NPY_DOUBLE);
    if (!nodes)
    {
        return NULL;
    }
    PyArrayObject *const weights = (PyArrayObject *)PyArray_SimpleNew(1, &array_size, NPY_DOUBLE);
    if (!weights)
    {
        Py_DECREF(nodes);
        return NULL;
    }
    double *const p_x = PyArray_DATA(nodes);
    double *const p_w = PyArray_DATA(weights);
    if (order != 0)
    {
        const int non_converged = gauss_lobatto_nodes_weights(order + 1, tol, max_iter, p_x, p_w);
        if (non_converged != 0)
        {
            PyErr_Format(PyExc_RuntimeWarning,
                         "A total of %i nodes were non-converged. Consider changing"
                         " the tolerance or increase the number of iterations.",
                         non_converged);
        }
    }
    else
    {
        // Corner case
        p_x[0] = 0.0;
        p_w[0] = 2.0;
    }

    return PyTuple_Pack(2, nodes, weights);
}

PyDoc_STRVAR(compute_gll_docstring,
             "compute_gll(order: int, max_iter: int = 10, tol: float = 1e-15) -> tuple[array, array]\n"
             "Compute Gauss-Legendre-Lobatto integration nodes and weights.\n"
             "\n"
             "If you are often re-using these, consider caching them.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "   Order of the scheme. The number of node-weight pairs is one more.\n"
             "max_iter : int, default: 10\n"
             "   Maximum number of iterations used to further refine the values.\n"
             "tol : float, default: 1e-15\n"
             "   Tolerance for stopping the refinement of the nodes.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "   Array of ``order + 1`` integration nodes on the interval :math:`[-1, +1]`.\n"
             "array\n"
             "   Array of integration weights which correspond to the nodes.\n");

INTERPLIB_INTERNAL
PyObject *bernstein_interpolation_matrix(PyObject *Py_UNUSED(self), PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Function only takes two arguments.");
        return NULL;
    }
    const unsigned long order = PyLong_AsUnsignedLong(args[0]);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    PyArrayObject *array_in = (PyArrayObject *)PyArray_FromAny(args[1], PyArray_DescrFromType(NPY_DOUBLE), 1, 1,
                                                               NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!array_in)
    {
        return NULL;
    }

    const npy_intp len[2] = {PyArray_SIZE(array_in), (npy_intp)order + 1};
    PyArrayObject *array_out = (PyArrayObject *)PyArray_SimpleNew(2, len, NPY_DOUBLE);
    if (!array_out)
    {
        Py_DECREF(array_in);
        return NULL;
    }

    const double *restrict p_in = PyArray_DATA(array_in);
    double *restrict p_out = PyArray_DATA(array_out);

    //  May be made parallel in another version of the function.
    for (npy_intp i = 0; i < len[0]; ++i)
    {
        bernstein_interpolation_vector(p_in[i], order, p_out + i * len[1]);
    }

    Py_DECREF(array_in);

    return (PyObject *)array_out;
}

INTERPLIB_INTERNAL
const char bernstein_interpolation_matrix_doc[] =
    "bernstein1d(n: int, x: npt.ArrayLike) -> npt.NDArray[np.float64]\n"
    "Compute Bernstein polynomials of given order at given locations.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "n : int\n"
    "   Order of polynomials used.\n"
    "x : (M,) array_like\n"
    "   Flat array of locations where the values should be interpolated.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "(M, n) array"
    "   Matrix containing values of Bernstein polynomial :math:`B^M_j(x_i)` as the\n"
    "   element ``array[i, j]``.\n";

INTERPLIB_INTERNAL
PyObject *bernstein_coefficients(PyObject *Py_UNUSED(self), PyObject *arg)
{
    PyArrayObject *const input_coeffs =
        (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_DOUBLE), 1, 1,
                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_ENSURECOPY, NULL);
    if (!input_coeffs)
        return NULL;

    bernstein_from_power_series(PyArray_DIM(input_coeffs, 0), PyArray_DATA(input_coeffs));

    return (PyObject *)input_coeffs;
}

INTERPLIB_INTERNAL
const char bernstein_coefficients_doc[] = "bernstein_coefficients(x: array_like, /) -> array\n"
                                          "\n"
                                          "Compute Bernstein polynomial coefficients from a power series polynomial.\n"
                                          "Parameters\n"
                                          "----------\n"
                                          "x : array_like\n"
                                          "   Coefficients of the polynomial from 0-th to the highest order.\n"
                                          "\n"
                                          "Returns\n"
                                          "-------\n"
                                          "array\n"
                                          "   Array of coefficients of Bernstein polynomial series.\n";

static PyMethodDef module_methods[] = {
    {
        "lagrange1d",
        interp_lagrange,
        METH_VARARGS,
        interp_lagrange_doc,
    },
    {
        "dlagrange1d",
        interp_dlagrange,
        METH_VARARGS,
        interp_dlagrange_doc,
    },
    {
        "d2lagrange1d",
        interp_d2lagrange,
        METH_VARARGS,
        interp_d2lagrange_doc,
    },
    {
        "bernstein1d",
        (void *)bernstein_interpolation_matrix,
        METH_FASTCALL,
        bernstein_interpolation_matrix_doc,
    },
    {
        "bernstein_coefficients",
        (void *)bernstein_coefficients,
        METH_O,
        bernstein_coefficients_doc,
    },
    {
        .ml_name = "compute_gll",
        .ml_meth = (void *)compute_gauss_lobatto_nodes,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = compute_gll_docstring,
    },
    {
        NULL,
        NULL,
        0,
        NULL,
    }, // sentinel
};

static void free_module_state(void *module)
{
    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(module);
    *module_state = (interplib_module_state_t){};
}

static int interplib_add_types(PyObject *mod)
{
    if (PyArray_ImportNumPyAPI() < 0)
        return -1;

    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        return -1;
    }

    if ((module_state->integration_spec_type =
             cpyutl_add_type_from_spec_to_module(mod, &integration_specs_type_spec, NULL)) == NULL ||
        (module_state->integration_registry_type =
             cpyutl_add_type_from_spec_to_module(mod, &integration_registry_type_spec, NULL)) == NULL ||
        (module_state->basis_spec_type = cpyutl_add_type_from_spec_to_module(mod, &basis_specs_type_spec, NULL)) ==
            NULL ||
        (module_state->basis_registry_type =
             cpyutl_add_type_from_spec_to_module(mod, &basis_registry_type_specs, NULL)) == NULL ||
        (module_state->covector_basis_type =
             cpyutl_add_type_from_spec_to_module(mod, &covector_basis_type_spec, NULL)) == NULL ||
        (module_state->function_space_type =
             cpyutl_add_type_from_spec_to_module(mod, &function_space_type_spec, NULL)) == NULL ||
        (module_state->integration_space_type =
             cpyutl_add_type_from_spec_to_module(mod, &integration_space_type_spec, NULL)) == NULL ||
        (module_state->degrees_of_freedom_type =
             cpyutl_add_type_from_spec_to_module(mod, &degrees_of_freedom_type_spec, NULL)) == NULL ||
        (module_state->coordinate_mapping_type =
             cpyutl_add_type_from_spec_to_module(mod, &coordinate_map_type_spec, NULL)) == NULL ||
        (module_state->space_mapping_type = cpyutl_add_type_from_spec_to_module(mod, &space_map_type_spec, NULL)) ==
            NULL ||
        (module_state->kform_specs_type = cpyutl_add_type_from_spec_to_module(mod, &kform_spec_type_spec, NULL)) ==
            NULL ||
        (module_state->kform_type = cpyutl_add_type_from_spec_to_module(mod, &kform_type_spec, NULL)) == NULL ||
        (module_state->geoid_type = cpyutl_add_type_from_spec_to_module(mod, &geo_id_type_spec, NULL)) == NULL ||
        (module_state->line_type = cpyutl_add_type_from_spec_to_module(mod, &line_type_spec, NULL)) == NULL ||
        (module_state->surf_type = cpyutl_add_type_from_spec_to_module(mod, &surface_type_spec, NULL)) == NULL ||
        (module_state->man_type = cpyutl_add_type_from_spec_to_module(mod, &manifold_type_spec, NULL)) == NULL ||
        (module_state->man1d_type = cpyutl_add_type_from_spec_to_module(mod, &manifold1d_type_spec,
                                                                        (PyObject *)module_state->man_type)) == NULL ||
        (module_state->man2d_type = cpyutl_add_type_from_spec_to_module(mod, &manifold2d_type_spec,
                                                                        (PyObject *)module_state->man_type)) == NULL)
    {
        return -1;
    }

    return 0;
}

static int interplib_add_functions(PyObject *mod)
{
    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        return -1;
    }

    if (PyModule_AddFunctions(mod, mass_matrices_methods) < 0 || PyModule_AddFunctions(mod, incidence_methods) < 0 ||
        PyModule_AddFunctions(mod, transformation_functions) < 0)
        return -1;

    return 0;
}

static int module_add_steal(PyObject *mod, const char *name, PyObject *obj)
{
    if (!obj)
        return -1;
    const int res = PyModule_AddObjectRef(mod, name, obj);
    Py_XDECREF(obj);
    return res;
}

static int interplib_add_registries(PyObject *mod)
{
    interplib_module_state_t *const module_state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        return -1;
    }

    // Add integration registry
    if (module_add_steal(mod, "DEFAULT_INTEGRATION_REGISTRY",
                         (module_state->registry_integration = (PyObject *)integration_registry_object_create(
                              module_state->integration_registry_type))) < 0)
        return -1;

    // Add basis registry
    if (module_add_steal(mod, "DEFAULT_BASIS_REGISTRY",
                         (module_state->registry_basis =
                              (PyObject *)basis_registry_object_create(module_state->basis_registry_type))) < 0)
        return -1;

    return 0;
}

PyModuleDef interplib_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "interplib._interp",
    .m_doc = "Internal C-extension implementing interpolation functions",
    .m_size = sizeof(interplib_module_state_t),
    .m_methods = module_methods,
    .m_free = free_module_state,
    .m_slots =
        (PyModuleDef_Slot[]){
            {.slot = Py_mod_exec, .value = interplib_add_types},
            {.slot = Py_mod_exec, .value = interplib_add_functions},
            {.slot = Py_mod_exec, .value = interplib_add_registries},
            {.slot = Py_mod_multiple_interpreters, .value = Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
            {},
        },
};

PyMODINIT_FUNC PyInit__interp(void)
{
    import_array();

    return PyModuleDef_Init(&interplib_module);
}

int heap_type_traverse_type(PyObject *self, const visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}
