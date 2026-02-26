#include "incidence.h"
#include "../basis/basis_lagrange.h"
#include "../polynomials/lagrange.h"
#include "basis_objects.h"
#include "covector_basis.h"
#include "cutl/iterators/combination_iterator.h"
#include "function_space_objects.h"
#include "kform_objects.h"

void bernstein_apply_incidence_operator(
    const unsigned n, const size_t pre_stride, const size_t post_stride, const unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols), const int negate)
{
    npy_double coeff = (double)n / 2.0;
    if (negate)
        coeff = -coeff;

#pragma omp simd
    for (unsigned i_col = 0; i_col < cols; ++i_col)
    {
        double *const vout = values_out + i_col;
        const double *const vin = values_in + i_col;

        for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_stride; ++i_post)
            {
                const size_t i_out = i_pre * n * post_stride + i_post;
                const size_t i_in = i_pre * (n + 1) * post_stride + i_post;

                vout[(i_out + 0) * cols] -= coeff * vin[(i_in + 0) * cols];
                for (unsigned col = 1; col < n; ++col)
                {
                    const npy_double x = coeff * vin[(i_in + col * post_stride) * cols];
                    vout[(i_out + col * post_stride) * cols] -= x;
                    vout[(i_out + (col - 1) * post_stride) * cols] += x;
                }
                vout[(i_out + (n - 1) * post_stride) * cols] += coeff * vin[(i_in + n * post_stride) * cols];
            }
        }
    }
}

void bernstein_matrix_incidence_operator(const unsigned n, const size_t pre_stride, const size_t post_stride,
                                         const size_t row_stride, double INTERPLIB_ARRAY_ARG(mat, restrict const),
                                         const int negate)
{
    npy_double coeff = (double)n / 2.0;
    if (negate)
        coeff = -coeff;
    for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
    {
        for (size_t i_post = 0; i_post < post_stride; ++i_post)
        {
            const size_t i_out = i_pre * n * post_stride + i_post;
            const size_t i_in = i_pre * (n + 1) * post_stride + i_post;

            // i_out[0] -= coeff * i_in[0];
            mat[(i_out + 0) * row_stride + (i_in + 0)] = -coeff;
            for (unsigned col = 1; col < n; ++col)
            {
                // i_out[col * post_stride] -= coeff * i_in[col * post_stride];
                mat[(i_out + col * post_stride) * row_stride + (i_in + col * post_stride)] = -coeff;
                // i_out[(col - 1) * post_stride] += coeff * i_in[col * post_stride];
                mat[(i_out + (col - 1) * post_stride) * row_stride + (i_in + col * post_stride)] = +coeff;
            }
            // i_out[(n - 1) * post_stride] += coeff * i_in[n * post_stride];
            mat[(i_out + (n - 1) * post_stride) * row_stride + (i_in + n * post_stride)] = +coeff;
        }
    }
}

void legendre_apply_incidence_operator(
    const unsigned n, const size_t pre_stride, const size_t post_stride, const unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols), const int negate)
{
#pragma omp simd
    for (unsigned i_col = 0; i_col < cols; ++i_col)
    {
        double *const vout = values_out + i_col;
        const double *const vin = values_in + i_col;

        for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_stride; ++i_post)
            {
                const size_t i_out = i_pre * n * post_stride + i_post;
                const size_t i_in = i_pre * (n + 1) * post_stride + i_post;

                for (unsigned col = n; col > 0; --col)
                {
                    unsigned coeff = 2 * col - 1;
                    for (unsigned c_row = 0; 2 * c_row < col; ++c_row)
                    {
                        const unsigned r = (col - 1 - 2 * c_row);
                        if (negate)
                        {
                            vout[(i_out + r * post_stride) * cols] -= coeff * vin[(i_in + col * post_stride) * cols];
                        }
                        else
                        {
                            vout[(i_out + r * post_stride) * cols] += coeff * vin[(i_in + col * post_stride) * cols];
                        }
                        coeff -= 4;
                    }
                }
            }
        }
    }
}

void legendre_matrix_incidence_operator(const unsigned n, const size_t pre_stride, const size_t post_stride,
                                        const size_t row_stride, double INTERPLIB_ARRAY_ARG(mat, restrict const),
                                        const int negate)
{
    for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
    {
        for (size_t i_post = 0; i_post < post_stride; ++i_post)
        {
            const size_t i_out = i_pre * n * post_stride + i_post;
            const size_t i_in = i_pre * (n + 1) * post_stride + i_post;

            for (unsigned col = n; col > 0; --col)
            {
                unsigned coeff = 2 * col - 1;
                for (unsigned c_row = 0; 2 * c_row < col; ++c_row)
                {
                    const unsigned r = (col - 1 - 2 * c_row);
                    // i_out[r * post_stride] += coeff * i_in[col * post_stride];
                    if (negate)
                    {
                        mat[(i_out + r * post_stride) * row_stride + (i_in + col * post_stride)] -= coeff;
                    }
                    else
                    {
                        mat[(i_out + r * post_stride) * row_stride + (i_in + col * post_stride)] += coeff;
                    }
                    coeff -= 4;
                }
            }
        }
    }
}

void lagrange_apply_incidence_operator(
    const basis_set_type_t type, const unsigned n, const size_t pre_stride, const size_t post_stride,
    const unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(work, restrict const n + (n + 1) + n * (n + 1)), const int negate)
{
    // Divide up the work array
    double *restrict const out_nodes = work + 0;
    double *restrict const in_nodes = work + n;
    double *restrict const trans_matrix = work + n + (n + 1);

    // Compute nodes for the output set
    interp_result_t res = generate_lagrange_roots(n - 1, type, out_nodes);
    CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
    if (res != INTERP_SUCCESS)
        return;

    res = generate_lagrange_roots(n, type, in_nodes);
    CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
    if (res != INTERP_SUCCESS)
        return;

    lagrange_polynomial_first_derivative_2(n, out_nodes, n + 1, in_nodes, trans_matrix);
    if (negate)
    {
        for (unsigned i = 0; i < n * (n + 1); ++i)
        {
            trans_matrix[i] = -trans_matrix[i];
        }
    }

#pragma omp simd
    for (unsigned i_col = 0; i_col < cols; ++i_col)
    {
        double *const vout = values_out + i_col;
        const double *const vin = values_in + i_col;

        for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_stride; ++i_post)
            {
                const size_t i_out = i_pre * n * post_stride + i_post;
                const size_t i_in = i_pre * (n + 1) * post_stride + i_post;

                // Apply the transformation matrix
                for (unsigned row = 0; row < n; ++row)
                {
                    double v = 0;
                    for (unsigned col = 0; col < n + 1; ++col)
                    {
                        v += trans_matrix[row * (n + 1) + col] * vin[(i_in + col * post_stride) * cols];
                    }
                    vout[(i_out + row * post_stride) * cols] += v;
                }
            }
        }
    }
}

void lagrange_matrix_incidence_operator(const basis_set_type_t type, const unsigned n, const size_t pre_stride,
                                        const size_t post_stride, const size_t row_stride,
                                        double INTERPLIB_ARRAY_ARG(mat, restrict const),
                                        double INTERPLIB_ARRAY_ARG(work, restrict const n + (n + 1) + n * (n + 1)),
                                        const int negate)
{

    // Divide up the work array
    double *restrict const out_nodes = work + 0;
    double *restrict const in_nodes = work + n;
    double *restrict const trans_matrix = work + n + (n + 1);

    // Compute nodes for the output set
    interp_result_t res = generate_lagrange_roots(n - 1, type, out_nodes);
    CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
    if (res != INTERP_SUCCESS)
        return;

    res = generate_lagrange_roots(n, type, in_nodes);
    CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
    if (res != INTERP_SUCCESS)
        return;

    lagrange_polynomial_first_derivative_2(n, out_nodes, n + 1, in_nodes, trans_matrix);
    if (negate)
    {
        for (unsigned i = 0; i < n * (n + 1); ++i)
        {
            trans_matrix[i] = -trans_matrix[i];
        }
    }
    for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
    {
        for (size_t i_post = 0; i_post < post_stride; ++i_post)
        {
            const size_t i_out = i_pre * n * post_stride + i_post;
            const size_t i_in = i_pre * (n + 1) * post_stride + i_post;

            // Write out the transformation matrix
            for (unsigned row = 0; row < n; ++row)
            {
                for (unsigned col = 0; col < n + 1; ++col)
                {
                    mat[(i_out + row * post_stride) * row_stride + (i_in + col * post_stride)] =
                        trans_matrix[row * (n + 1) + col];
                }
            }
        }
    }
}

static PyObject *incidence_matrix(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    const basis_specs_object *basis_specs;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &basis_specs,
                 .type_check = state->basis_spec_type,
                 .kwname = "basis_specs"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (basis_specs->spec.order == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot compute the incidence matrix for a zero-dimensional basis.");
        return NULL;
    }

    const unsigned n = basis_specs->spec.order;
    const npy_intp dims[2] = {n, n + 1};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;

    npy_double *const data = PyArray_DATA(out);
    memset(PyArray_DATA(out), 0, (size_t)n * (n + 1) * sizeof(*data));
    switch (basis_specs->spec.type)
    {
    case BASIS_BERNSTEIN:
        // Use recurrence relation
        {
            // Scale by 1/2 because we change variables from [0, 1] to [-1, +1]
            const npy_double coeff = (double)n / 2.0;
            data[0 * (n + 1) + 0] = -coeff;
            for (unsigned col = 1; col < n; ++col)
            {
                data[col * (n + 1) + col] = -coeff;
                data[(col - 1) * (n + 1) + col] = +coeff;
            }
            data[(n - 1) * (n + 1) + n] = +coeff;
        }
        break;

    case BASIS_LEGENDRE:
        // Use recurrence relation
        {
            for (unsigned col = n; col > 0; --col)
            {
                unsigned coeff = 2 * col - 1;
                for (unsigned c_row = 0; 2 * c_row < col; ++c_row)
                {
                    const unsigned r = (col - 1 - 2 * c_row);
                    data[r * (n + 1) + col] = coeff;
                    coeff -= 4;
                }
            }
        }
        break;

    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        // Use direct evaluation of the derivative at nodes
        {
            // Compute nodes for the output set
            double *const out_nodes = PyMem_Malloc(sizeof(*out_nodes) * n);
            if (!out_nodes)
            {
                Py_DECREF(out);
                return NULL;
            }
            interp_result_t res = generate_lagrange_roots(n - 1, basis_specs->spec.type, out_nodes);
            (void)res;
            CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
            double *const in_nodes = PyMem_Malloc(sizeof(*in_nodes) * (n + 1));
            if (!in_nodes)
            {
                PyMem_Free(out_nodes);
                Py_DECREF(out);
                return NULL;
            }
            res = generate_lagrange_roots(n, basis_specs->spec.type, in_nodes);
            (void)res;
            CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");

            lagrange_polynomial_first_derivative_2(n, out_nodes, n + 1, in_nodes, data);

            PyMem_Free(out_nodes);
            PyMem_Free(in_nodes);
        }
        break;

    default:
        CPYUTL_ASSERT(0, "Unsupported basis type.");
        PyErr_SetString(PyExc_NotImplementedError, "Unsupported basis type (should not have happended).");
        Py_DECREF(out);
        return NULL;
    }

    return (PyObject *)out;
}

PyDoc_STRVAR(incidence_matrix_docstring,
             "incidence_matrix(basis_specs : BasisSpecs) -> numpy.typing.NDArray[numpy.double]\n"
             "Return the incidence matrix to transfer derivative degrees of freedom.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "specs : BasisSpecs\n"
             "    Basis specs for which this incidence matrix should be computed.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    One dimensional incidence matrix. It transfers primal degrees of freedom\n"
             "    for a derivative to a function space one order less than the original.\n");

static PyObject *incidence_operator(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                    const PyObject *kwnames)
{
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    PyObject *py_val;
    const basis_specs_object *basis_specs;
    Py_ssize_t axis = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_val, .kwname = "x"},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &basis_specs,
                 .type_check = state->basis_spec_type,
                 .kwname = "specs"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &axis, .kwname = "axis", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyArrayObject *const array =
        (PyArrayObject *)PyArray_FROMANY(py_val, NPY_DOUBLE, 0, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!array)
        return NULL;

    const unsigned order = basis_specs->spec.order;
    if (order == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot compute the incidence matrix for a zero-dimensional basis.");
        return NULL;
    }

    const unsigned ndim = PyArray_NDIM(array);
    if (axis < 0 && axis >= ndim)
    {
        PyErr_Format(PyExc_IndexError, "Axis index %zd out of bounds for array of dimension %zd", axis, ndim);
        Py_DECREF(array);
        return NULL;
    }

    const npy_intp *const dims = PyArray_DIMS(array);
    if (dims[axis] != order + 1)
    {
        PyErr_Format(PyExc_ValueError, "Dimension %zd of array does not match basis order %zd.", axis, order);
        Py_DECREF(array);
        return NULL;
    }

    npy_intp *out_dims;
    double *work;
    void *const mem = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {.size = sizeof(*out_dims) * ndim, .p_ptr = (void **)&out_dims},
            {.size = sizeof(*work) * ((order + 1) * order + order + (order + 1)), .p_ptr = (void **)&work},
            {},
        });
    if (!mem)
    {
        Py_DECREF(array);
        return NULL;
    }
    // Copy the dims array
    size_t pre_stride = 1, post_stride = 1;
    for (unsigned i = 0; i < axis; ++i)
    {
        const Py_ssize_t sz_dim = dims[i];
        out_dims[i] = sz_dim;
        pre_stride *= sz_dim;
    }
    out_dims[axis] = order;
    for (unsigned i = axis + 1; i < ndim; ++i)
    {
        const Py_ssize_t sz_dim = dims[i];
        out_dims[i] = sz_dim;
        post_stride *= sz_dim;
    }
    // Create the new array
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndim, out_dims, NPY_DOUBLE);
    if (!out)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
        Py_DECREF(array);
        return NULL;
    }

    const double *const vals_in = PyArray_DATA(array);
    npy_double *const out_ptr = PyArray_DATA(out);
    memset(out_ptr, 0, pre_stride * post_stride * order * sizeof(*out_ptr));

    // Apply the incidence and write the result to the output array
    switch (basis_specs->spec.type)
    {
    case BASIS_BERNSTEIN:
        bernstein_apply_incidence_operator(order, pre_stride, post_stride, 1, vals_in, out_ptr, 0);
        break;

    case BASIS_LEGENDRE:
        legendre_apply_incidence_operator(order, pre_stride, post_stride, 1, vals_in, out_ptr, 0);
        break;

    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        lagrange_apply_incidence_operator(basis_specs->spec.type, order, pre_stride, post_stride, 1, vals_in, out_ptr,
                                          work, 0);
        break;

    default:
        CPYUTL_ASSERT(0, "Unsupported basis type.");
        PyErr_SetString(PyExc_NotImplementedError, "Unsupported basis type (should not have happended).");
        Py_DECREF(out);
        return NULL;
    }

    // Cleanup
    cutl_dealloc(&PYTHON_ALLOCATOR, mem);
    Py_DECREF(array);

    return (PyObject *)out;
}

PyDoc_STRVAR(incidence_operator_docstring,
             "incidence_operator(val: numpy.typing.ArrayLike, /, specs: BasisSpecs, axis: int = 0) -> "
             "numpy.typing.NDArray[numpy.double]\n"
             "Apply the incidence operator to an array of degrees of freedom along an axis.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "val : array_like\n"
             "    Array of degrees of freedom to apply the incidence operator to.\n"
             "\n"
             "specs : BasisSpecs\n"
             "    Specifications for basis that determine what set of polynomial is used to take\n"
             "    the derivative.\n"
             "\n"
             "axis : int, default: 0\n"
             "    Axis along which to apply the incidence operator along.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of degrees of freedom that is the result of applying the incidence operator,\n"
             "    along the specified axis.\n");

static void incidence_matrix_fill_block(const unsigned ndim, const basis_spec_t basis[static ndim],
                                        const unsigned order, const uint8_t components[static order],
                                        const unsigned derivative_dim, const unsigned offset_row,
                                        const unsigned offset_col, const size_t row_pitch, const int flip_sign,
                                        double *restrict out_array, double *work)
{
    size_t pre_stride = 1, post_stride = 1;
    unsigned idim, i_component;
    for (idim = 0, i_component = 0; idim < derivative_dim; ++idim)
    {
        unsigned dofs_in_dimension;
        if (i_component < order && idim == components[i_component])
        {
            dofs_in_dimension = basis[idim].order;
            i_component += 1;
        }
        else
        {
            dofs_in_dimension = basis[idim].order + 1;
        }
        pre_stride *= dofs_in_dimension;
    }
    ASSERT(components[i_component] == derivative_dim,
           "I miscounted the components somehow (components[i_component] = %u, derivative_dim = %u).",
           (unsigned)components[i_component], (unsigned)derivative_dim);
    i_component += 1;
    idim += 1;
    for (; idim < ndim; ++idim)
    {
        unsigned dofs_in_dimension;
        if (i_component < order && idim == components[i_component])
        {
            dofs_in_dimension = basis[idim].order;
            i_component += 1;
        }
        else
        {
            dofs_in_dimension = basis[idim].order + 1;
        }
        post_stride *= dofs_in_dimension;
    }
    ASSERT(i_component == order, "I miscounted the components somehow (i_component = %u, order = %u).", i_component,
           order);

    const basis_set_type_t btype = basis[derivative_dim].type;
    const unsigned n = basis[derivative_dim].order;
    double *restrict const mat = out_array + offset_row * row_pitch + offset_col;
    switch (btype)
    {
    case BASIS_BERNSTEIN:
        bernstein_matrix_incidence_operator(n, pre_stride, post_stride, row_pitch, mat, flip_sign);
        break;
    case BASIS_LEGENDRE:
        legendre_matrix_incidence_operator(n, pre_stride, post_stride, row_pitch, mat, flip_sign);
        break;
    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        ASSERT(work != NULL, "Work array was not given!");
        lagrange_matrix_incidence_operator(btype, n, pre_stride, post_stride, row_pitch, mat, work, flip_sign);
        break;
    case BASIS_INVALID:
        ASSERT(0, "Invalid basis type.");
        return;
    }
}

static void incidence_operator_apply_block(const unsigned ndim, const basis_spec_t basis[static ndim],
                                           const unsigned order, const uint8_t components[static order],
                                           const unsigned cols, const unsigned derivative_dim, const int flip_sign,
                                           const double *restrict in_array, double *restrict out_array, double *work)
{
    size_t pre_stride = 1, post_stride = 1;
    unsigned idim, i_component;
    for (idim = 0, i_component = 0; idim < derivative_dim; ++idim)
    {
        unsigned dofs_in_dimension;
        if (i_component < order && idim == components[i_component])
        {
            dofs_in_dimension = basis[idim].order;
            i_component += 1;
        }
        else
        {
            dofs_in_dimension = basis[idim].order + 1;
        }
        pre_stride *= dofs_in_dimension;
    }
    ASSERT(components[i_component] == derivative_dim,
           "I miscounted the components somehow (components[i_component] = %u, derivative_dim = %u).",
           (unsigned)components[i_component], (unsigned)derivative_dim);
    i_component += 1;
    idim += 1;
    for (; idim < ndim; ++idim)
    {
        unsigned dofs_in_dimension;
        if (i_component < order && idim == components[i_component])
        {
            dofs_in_dimension = basis[idim].order;
            i_component += 1;
        }
        else
        {
            dofs_in_dimension = basis[idim].order + 1;
        }
        post_stride *= dofs_in_dimension;
    }
    ASSERT(i_component == order, "I miscounted the components somehow (i_component = %u, order = %u).", i_component,
           order);

    const basis_set_type_t btype = basis[derivative_dim].type;
    const unsigned n = basis[derivative_dim].order;
    switch (btype)
    {
    case BASIS_BERNSTEIN:
        bernstein_apply_incidence_operator(n, pre_stride, post_stride, cols, in_array, out_array, flip_sign);
        break;
    case BASIS_LEGENDRE:
        legendre_apply_incidence_operator(n, pre_stride, post_stride, cols, in_array, out_array, flip_sign);
        break;
    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        ASSERT(work != NULL, "Work array was not given!");
        lagrange_apply_incidence_operator(btype, n, pre_stride, post_stride, cols, in_array, out_array, work,
                                          flip_sign);
        break;
    case BASIS_INVALID:
        ASSERT(0, "Invalid basis type.");
        return;
    }
}

static PyObject *compute_kform_incidence_matrix(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                                const PyObject *kwnames)
{
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    const function_space_object *fn_space;
    Py_ssize_t order;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &fn_space,
                 .type_check = state->function_space_type,
                 .kwname = "base_space"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned n = Py_SIZE(fn_space);
    // Check order is within the allowed range
    if (order >= n || order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must be in the range [0, %u), but it was %zd.", n, order);
        return NULL;
    }

    // Check the function space has order at least 1 in each dimension
    for (unsigned i = 0; i < n; ++i)
    {
        if (fn_space->specs[i].order < 1)
        {
            PyErr_Format(PyExc_ValueError, "Function spaces must have order at least 1 in each dimension.");
            return NULL;
        }
    }

    // Find the highest order of the basis for all dimensions
    unsigned max_order = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        if (max_order < fn_space->specs[i].order)
            max_order = fn_space->specs[i].order;
    }

    // Allocate the memory needed
    combination_iterator_t *iter_component_in, *iter_component_out;
    uint8_t *basis_components;
    unsigned *out_component_offsets, *in_component_offsets;
    double *work_buffer;
    void *const mem = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {.size = combination_iterator_required_memory(order), .p_ptr = (void **)&iter_component_in},
            {.size = combination_iterator_required_memory(order + 1), .p_ptr = (void **)&iter_component_out},
            {.size = sizeof(*basis_components) * (order + 1), .p_ptr = (void **)&basis_components},
            {.size = sizeof(*out_component_offsets) * (combination_total_count(n, order + 1) + 1),
             .p_ptr = (void **)&out_component_offsets},
            {.size = sizeof(*in_component_offsets) * (combination_total_count(n, order) + 1),
             .p_ptr = (void **)&in_component_offsets},
            {.size = sizeof(*work_buffer) * ((max_order + 1) * max_order + max_order + (max_order + 1)),
             .p_ptr = (void **)&work_buffer},
            {},
        });

    if (!mem)
        return NULL;

    // Compute the number of input and output degrees of freedom
    size_t in_dofs = 0, idx_in = 0;
    in_component_offsets[0] = 0;
    combination_iterator_init(iter_component_in, n, order);
    for (const uint8_t *p_basis_components = combination_iterator_current(iter_component_in);
         !combination_iterator_is_done(iter_component_in); combination_iterator_next(iter_component_in), ++idx_in)
    {
        in_dofs += kform_basis_get_num_dofs(n, fn_space->specs, order, p_basis_components);
        in_component_offsets[idx_in + 1] = in_dofs;
    }
    size_t out_dofs = 0, idx_out = 0;
    out_component_offsets[0] = 0;
    combination_iterator_init(iter_component_out, n, order + 1);
    for (const uint8_t *p_basis_components = combination_iterator_current(iter_component_out);
         !combination_iterator_is_done(iter_component_out); combination_iterator_next(iter_component_out), ++idx_out)
    {
        out_dofs += kform_basis_get_num_dofs(n, fn_space->specs, order + 1, p_basis_components);
        // Use this chance to initialize the offsets
        out_component_offsets[idx_out + 1] = out_dofs;
    }

    // Create output matrix
    const npy_intp out_dims[2] = {(npy_intp)out_dofs, (npy_intp)in_dofs};
    PyArrayObject *const out_array = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
    if (!out_array)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
        return NULL;
    }

    // Zero the output just in case
    npy_double *restrict const out_data = PyArray_DATA(out_array);
    memset(out_data, 0, out_dofs * in_dofs * sizeof(*out_data));

    // Loop over input k-form components
    size_t idx_comp_in = 0;
    combination_iterator_reset(iter_component_in);
    for (const uint8_t *const components_in = combination_iterator_current(iter_component_in);
         !combination_iterator_is_done(iter_component_in); combination_iterator_next(iter_component_in), ++idx_comp_in)
    {
        // Prepare the loop over all possible outputs
        basis_components[0] = 0;
        for (unsigned i = 0; i < order; ++i)
        {
            basis_components[i + 1] = components_in[i];
        }

        for (unsigned pv = 0; pv < order; pv += 1)
        {
            while (basis_components[pv] < basis_components[pv + 1])
            {
                const unsigned idx_comp_out = combination_get_index(n, order + 1, basis_components);
                incidence_matrix_fill_block(n, fn_space->specs, order + 1, basis_components, basis_components[pv],
                                            out_component_offsets[idx_comp_out], in_component_offsets[idx_comp_in],
                                            in_dofs, pv & 1u, out_data, work_buffer);
                basis_components[pv] += 1;
            }
            basis_components[pv + 1] += 1;
        }

        while (basis_components[order] < n)
        {
            const unsigned idx_comp_out = combination_get_index(n, order + 1, basis_components);
            // printf("Filling block (%u, %u)\n", idx_comp_out, (unsigned)idx_comp_out);
            incidence_matrix_fill_block(n, fn_space->specs, order + 1, basis_components, basis_components[order],
                                        out_component_offsets[idx_comp_out], in_component_offsets[idx_comp_in], in_dofs,
                                        order & 1u, out_data, work_buffer);
            basis_components[order] += 1;
        }
    }

    // Release the memory
    cutl_dealloc(&PYTHON_ALLOCATOR, mem);

    return (PyObject *)out_array;
}

PyDoc_STRVAR(
    compute_kform_incidence_matrix_docstring,
    "compute_kform_incidence_matrix(base_space: FunctionSpace, order: int) -> numpy.typing.NDArray[numpy.double]\n"
    "Compute the incidence matrix which maps a k-form to its (k + 1)-form derivative.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "base_space : FunctionSpace\n"
    "    Base function space, which describes the function space used for 0-forms.\n"
    "\n"
    "order : int\n"
    "    Order of the k-form to get the incidence matrix for.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Matrix, which maps degrees of freedom for the input k-form to the degrees of\n"
    "    freedom of its (k + 1)-form derivative.\n");

static PyObject *incidence_kform_operator(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                          const PyObject *kwnames)
{
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    const kform_spec_object *specs;
    PyArrayObject *py_vals, *py_out = NULL;
    int b_transpose = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &specs, .type_check = state->kform_specs_type, .kwname = "specs"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_vals, .type_check = &PyArray_Type, .kwname = "values"},
                {.type = CPYARG_TYPE_BOOL, .p_val = &b_transpose, .kwname = "transpose", .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_out, .kwname = "out", .optional = 1, .kw_only = 1},
                {},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (py_out != NULL && Py_IsNone((PyObject *)py_out))
    {
        // None for py_out should be treated as unspecified.
        py_out = NULL;
    }

    const function_space_object *fn_space = specs->function_space;
    const unsigned n = Py_SIZE(fn_space);
    const unsigned order = specs->order;

    unsigned order_in, order_out;
    if (b_transpose)
    {
        order_in = order + 1;
        order_out = order;
        PyErr_SetString(PyExc_NotImplementedError, "Transpose is not supported yet.");
        return NULL;
    }
    else
    {
        order_in = order;
        order_out = order + 1;
        (void)order_out;
    }

    const size_t n_components_in = combination_total_count(n, order_in);
    const size_t n_components_out = combination_total_count(n, order_out);

    // Compute the number of input and output degrees of freedom
    const size_t *const in_component_offsets = specs->component_offsets;
    const size_t in_dofs = in_component_offsets[n_components_in];

    const unsigned in_array_ndim = PyArray_NDIM(py_vals);

    // Find the highest order of the basis for all dimensions
    unsigned max_order = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        if (max_order < fn_space->specs[i].order)
            max_order = fn_space->specs[i].order;
    }

    // Allocate the memory needed for all work buffers and iterators
    combination_iterator_t *iter_component_in, *iter_component_out;
    uint8_t *basis_components;
    size_t *out_component_offsets;
    double *work_buffer;
    void *const mem = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {.size = combination_iterator_required_memory(order), .p_ptr = (void **)&iter_component_in},
            {.size = combination_iterator_required_memory(order + 1), .p_ptr = (void **)&iter_component_out},
            {.size = sizeof(*basis_components) * (order + 1), .p_ptr = (void **)&basis_components},
            {.size = sizeof(*out_component_offsets) * (n_components_out + 1), .p_ptr = (void **)&out_component_offsets},
            {.size = sizeof(*work_buffer) * ((max_order + 1) * max_order + max_order + (max_order + 1)),
             .p_ptr = (void **)&work_buffer},
            {},
        });
    if (!mem)
        return NULL;

    // Compute output offsets
    size_t out_dofs = 0, idx_out = 0;
    out_component_offsets[0] = 0;
    combination_iterator_init(iter_component_out, n, order + 1);
    for (const uint8_t *p_basis_components = combination_iterator_current(iter_component_out);
         !combination_iterator_is_done(iter_component_out); combination_iterator_next(iter_component_out), ++idx_out)
    {
        out_dofs += kform_basis_get_num_dofs(n, fn_space->specs, order + 1, p_basis_components);
        // Use this chance to initialize the offsets
        out_component_offsets[idx_out + 1] = out_dofs;
    }
    ASSERT(idx_out == n_components_out, "I miscounted the components somehow (idx_out = %zu, n_components_out = %zu).",
           idx_out, n_components_out);

    // Check that the input array has "in_dofs" for its last axis size
    size_t cols = 1;
    if (in_array_ndim == 1)
    {
        if (check_input_array(py_vals, 1, (const npy_intp[1]){(npy_intp)in_dofs}, NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "values") < 0)
        {
            cutl_dealloc(&PYTHON_ALLOCATOR, mem);
            return NULL;
        }
    }
    else if (in_array_ndim == 2)
    {
        cols = PyArray_DIM(py_vals, 1);
        if (check_input_array(py_vals, 2, (const npy_intp[2]){(npy_intp)in_dofs, (npy_intp)cols}, NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "values") < 0)
        {
            cutl_dealloc(&PYTHON_ALLOCATOR, mem);
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Input array can only be 1D or 2D.");
        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
        return NULL;
    }

    // Fill out output dimensions
    const npy_intp out_dims[2] = {(npy_intp)out_dofs, (npy_intp)cols};
    // Create output if need be
    if (py_out == NULL)
    {
        // We need to create it the same shape as input, with only the last axis being having the "out_dofs" size
        py_out = (PyArrayObject *)PyArray_SimpleNew(in_array_ndim, out_dims, NPY_DOUBLE);
        if (!py_out)
        {
            cutl_dealloc(&PYTHON_ALLOCATOR, mem);
            return NULL;
        }
    }
    else if (check_input_array(py_out, in_array_ndim, out_dims, NPY_DOUBLE,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "out") < 0)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
        return NULL;
    }
    else
    {
        Py_INCREF(py_out);
    }

    // Zero the output just in case
    npy_double *restrict const out_data = PyArray_DATA(py_out);
    const npy_double *restrict const in_data = PyArray_DATA(py_vals);
    memset(out_data, 0, PyArray_SIZE(py_out) * sizeof(*out_data));

    // Loop over input k-form components
    size_t idx_comp_in = 0;
    combination_iterator_init(iter_component_in, n, order_in);
    for (const uint8_t *const components_in = combination_iterator_current(iter_component_in);
         !combination_iterator_is_done(iter_component_in); combination_iterator_next(iter_component_in), ++idx_comp_in)
    {
        // Prepare the loop over all possible outputs
        basis_components[0] = 0;
        for (unsigned i = 0; i < order; ++i)
        {
            basis_components[i + 1] = components_in[i];
        }

        const size_t in_component_offset = in_component_offsets[idx_comp_in];
        const npy_double *restrict const in_array = in_data + in_component_offset * cols;
        for (unsigned pv = 0; pv < order; pv += 1)
        {
            while (basis_components[pv] < basis_components[pv + 1])
            {
                const unsigned idx_comp_out = combination_get_index(n, order + 1, basis_components);
                const size_t out_component_offset = out_component_offsets[idx_comp_out];
                // printf("Working on block with in offset %zu and out offset %zu (%zX)\n", in_component_offset,
                //        out_component_offset, out_component_offset);
                incidence_operator_apply_block(n, fn_space->specs, order + 1, basis_components, cols,
                                               basis_components[pv], pv & 1u, in_array,
                                               out_data + out_component_offset * cols, work_buffer);
                basis_components[pv] += 1;
            }
            basis_components[pv + 1] += 1;
        }
        ASSERT(out_dofs == out_component_offsets[n_components_out], "Value in the array changed.");

        while (basis_components[order] < n)
        {
            const unsigned idx_comp_out = combination_get_index(n, order + 1, basis_components);
            const size_t out_component_offset = out_component_offsets[idx_comp_out];
            // printf("Working on block with in offset %zu and out offset %zu (%zX)\n", in_component_offset,
            //        out_component_offset, out_component_offset);
            incidence_operator_apply_block(n, fn_space->specs, order + 1, basis_components, cols,
                                           basis_components[order], order & 1u, in_array,
                                           out_data + out_component_offset * cols, work_buffer);
            basis_components[order] += 1;
        }
        ASSERT(out_dofs == out_component_offsets[n_components_out], "Value in the array changed.");
    }

    // Release the memory
    cutl_dealloc(&PYTHON_ALLOCATOR, mem);

    return (PyObject *)py_out;
}

PyDoc_STRVAR(
    incidence_kform_operator_docstring,
    "incidence_kform_operator(specs: KFormSpecs, values: numpy.typing.NDArray[np.double], transpose: bool = False, *, "
    "out: numpy.typing.NDArray[numpy.double] | None = None) -> numpy.typing.NDArray[numpy.double]\n"
    "Apply the incidence operator on the k-form.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "specs : KFormSpecs\n"
    "    Specifications of the input k-form on which this operator is to be applied on.\n"
    "\n"
    "values : array\n"
    "    Array which contains the degrees of freedom of all components flattened along the\n"
    "    last axis. Treated as a row-major matrix or a vector, depending if 1D or 2D.\n"
    "\n"
    "transpose : bool, default: False\n"
    "    Apply the transpose of the incidence operator instead.\n"
    "\n"
    "out : array, optional\n"
    "    Array to which the result is written to. The first axis must have the same size\n"
    "    as the number of output degrees of freedom of the resulting k-form. If the input\n"
    "    was 2D, this must be as well, with the last axis matching the input's last axis.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Values of the degrees of freedom of the derivative of the input k-form. When an\n"
    "    output array is specified through the parameters, another reference to it is\n"
    "    returned, otherwise a new array is created to hold the result and returned.\n");

PyMethodDef incidence_methods[] = {
    {
        .ml_name = "incidence_matrix",
        .ml_meth = (void *)incidence_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = incidence_matrix_docstring,
    },
    {
        .ml_name = "incidence_operator",
        .ml_meth = (void *)incidence_operator,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = incidence_operator_docstring,
    },
    {
        .ml_name = "compute_kform_incidence_matrix",
        .ml_meth = (void *)compute_kform_incidence_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = compute_kform_incidence_matrix_docstring,
    },
    {
        .ml_name = "incidence_kform_operator",
        .ml_meth = (void *)incidence_kform_operator,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = incidence_kform_operator_docstring,
    },
    {}, // sentinel
};
