#ifndef FDG_KFORM_TRANSFORM_H
#define FDG_KFORM_TRANSFORM_H

#include "module.h"
#include <cutl/iterators/combination_iterator.h>
#include <string.h>

typedef unsigned (*kform_transform_dimension_fn)(const void *map);
typedef npy_intp (*kform_transform_point_axis_size_fn)(const void *map, unsigned axis);
typedef const double *(*kform_transform_inverse_maps_fn)(const void *map);
typedef PyArrayObject *(*kform_transform_basis_fn)(const void *map, Py_ssize_t order);

// These operations are the compile-time interface for a mapping representation.
// Each mapping type should provide one const instance of this table.
typedef struct
{
    kform_transform_dimension_fn input_dimensions;
    kform_transform_dimension_fn output_dimensions;
    kform_transform_point_axis_size_fn point_axis_size;
    kform_transform_inverse_maps_fn inverse_maps;
    kform_transform_basis_fn basis_transform;
} kform_transform_operations_t;

typedef struct
{
    const void *map;
    const kform_transform_operations_t *operations;
    Py_ssize_t order;
    Py_ssize_t minimum_order;
    PyObject *components;
    PyObject *out;
} kform_transform_request_t;

typedef struct
{
    PyArrayObject *components;
    PyArrayObject *out;
    unsigned ndim_in;
    unsigned ndim_out;
    unsigned n_components_in;
    unsigned n_components_out;
    size_t total_points;
} kform_transform_arrays_t;

static inline void kform_transform_arrays_clear(kform_transform_arrays_t *arrays)
{
    Py_XDECREF(arrays->components);
    Py_XDECREF(arrays->out);
    *arrays = (kform_transform_arrays_t){};
}

static inline int kform_transform_prepare(const kform_transform_request_t *request, kform_transform_arrays_t *arrays)
{
    *arrays = (kform_transform_arrays_t){};

    const unsigned ndim_in = request->operations->input_dimensions(request->map);
    const unsigned ndim_out = request->operations->output_dimensions(request->map);
    if (request->order < request->minimum_order || request->order > (Py_ssize_t)ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected order to be between %zd and %u, but got %zd.", request->minimum_order,
                     ndim_in, request->order);
        return -1;
    }

    arrays->ndim_in = ndim_in;
    arrays->ndim_out = ndim_out;
    arrays->n_components_in = combination_total_count(ndim_in, (unsigned)request->order);
    arrays->n_components_out = combination_total_count(ndim_out, (unsigned)request->order);

    arrays->components =
        (PyArrayObject *)PyArray_FROMANY(request->components, NPY_DOUBLE, 1 + ndim_in, 1 + ndim_in, NPY_ARRAY_IN_ARRAY);
    if (!arrays->components)
        return -1;

    const npy_intp *const dims_in = PyArray_DIMS(arrays->components);
    const unsigned ndim_components = (unsigned)PyArray_NDIM(arrays->components);
    if (ndim_components != 1 + ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected components to have %u dimensions, but got %u.", 1 + ndim_in,
                     ndim_components);
        kform_transform_arrays_clear(arrays);
        return -1;
    }
    if (dims_in[0] != (npy_intp)arrays->n_components_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected components to have shape (%u, ...), but got (%zd, ...).",
                     arrays->n_components_in, dims_in[0]);
        kform_transform_arrays_clear(arrays);
        return -1;
    }

    arrays->total_points = 1;
    for (unsigned axis = 0; axis < ndim_in; ++axis)
    {
        const npy_intp expected_size = request->operations->point_axis_size(request->map, axis);
        if (dims_in[axis + 1] != expected_size)
        {
            PyErr_Format(PyExc_ValueError,
                         "Components dimension %u did not match the mapping grid (expected %zd, got %zd).", axis,
                         expected_size, dims_in[axis + 1]);
            kform_transform_arrays_clear(arrays);
            return -1;
        }
        arrays->total_points *= (size_t)expected_size;
    }

    if (request->out == NULL || Py_IsNone(request->out))
    {
        npy_intp *const dims_out = PyMem_Malloc(sizeof(*dims_out) * ndim_components);
        if (!dims_out)
        {
            kform_transform_arrays_clear(arrays);
            return -1;
        }
        dims_out[0] = arrays->n_components_out;
        for (unsigned axis = 1; axis < ndim_components; ++axis)
            dims_out[axis] = dims_in[axis];
        arrays->out = (PyArrayObject *)PyArray_SimpleNew(ndim_components, dims_out, NPY_DOUBLE);
        PyMem_Free(dims_out);
        if (!arrays->out)
        {
            kform_transform_arrays_clear(arrays);
            return -1;
        }
    }
    else
    {
        if (!PyArray_Check(request->out))
        {
            PyErr_Format(PyExc_TypeError, "Expected out to be an array, but got %s.", Py_TYPE(request->out)->tp_name);
            kform_transform_arrays_clear(arrays);
            return -1;
        }
        PyArrayObject *const out_array = (PyArrayObject *)request->out;
        if ((unsigned)PyArray_NDIM(out_array) != ndim_components)
        {
            PyErr_Format(PyExc_ValueError, "Expected output to have %u dimensions, but got %u.", ndim_components,
                         (unsigned)PyArray_NDIM(out_array));
            kform_transform_arrays_clear(arrays);
            return -1;
        }
        const npy_intp *const dims_out = PyArray_DIMS(out_array);
        if (dims_out[0] != (npy_intp)arrays->n_components_out)
        {
            PyErr_Format(PyExc_ValueError, "Expected output to have shape (%u, ...), but got (%zd, ...).",
                         arrays->n_components_out, dims_out[0]);
            kform_transform_arrays_clear(arrays);
            return -1;
        }
        for (unsigned axis = 1; axis < ndim_components; ++axis)
        {
            if (dims_out[axis] != dims_in[axis])
            {
                PyErr_Format(PyExc_ValueError,
                             "Expected output to have the same shape as the input after the first dimension, but got "
                             "%zd for dimension %u.",
                             dims_out[axis], axis);
                kform_transform_arrays_clear(arrays);
                return -1;
            }
        }
        arrays->out = out_array;
        Py_INCREF(arrays->out);
    }

    return 0;
}

static inline int kform_transform_apply(const kform_transform_request_t *request,
                                        const kform_transform_arrays_t *arrays)
{
    const size_t total_points = arrays->total_points;
    const unsigned ndim_in = arrays->ndim_in;
    const unsigned ndim_out = arrays->ndim_out;
    const double *restrict const ptr_components = PyArray_DATA(arrays->components);
    double *restrict const ptr_out = PyArray_DATA(arrays->out);

    if (request->order == 0)
    {
        memcpy(ptr_out, ptr_components, sizeof(*ptr_out) * total_points);
        return 0;
    }

    if (request->order == 1)
    {
        const double *const inverse_maps = request->operations->inverse_maps(request->map);
        Py_BEGIN_ALLOW_THREADS;
        for (size_t point = 0; point < total_points; ++point)
        {
#pragma omp simd
            for (unsigned output = 0; output < ndim_out; ++output)
            {
                double value = 0.0;
                for (unsigned input = 0; input < ndim_in; ++input)
                {
                    value += ptr_components[input * total_points + point] *
                             inverse_maps[point * ((size_t)ndim_in * ndim_out) + input * ndim_out + output];
                }
                ptr_out[output * total_points + point] = value;
            }
        }
        Py_END_ALLOW_THREADS;
        return 0;
    }

    PyArrayObject *const transformation = request->operations->basis_transform(request->map, request->order);
    if (!transformation)
        return -1;

    const double *restrict const ptr_transformation = PyArray_DATA(transformation);
    Py_BEGIN_ALLOW_THREADS;
#pragma omp simd
    for (size_t point = 0; point < total_points; ++point)
    {
        for (unsigned output = 0; output < arrays->n_components_out; ++output)
        {
            double value = 0.0;
            for (unsigned input = 0; input < arrays->n_components_in; ++input)
            {
                value += ptr_transformation[(size_t)input * arrays->n_components_out * total_points +
                                            output * total_points + point] *
                         ptr_components[input * total_points + point];
            }
            ptr_out[output * total_points + point] = value;
        }
    }
    Py_END_ALLOW_THREADS;
    Py_DECREF(transformation);
    return 0;
}

#endif // FDG_KFORM_TRANSFORM_H
