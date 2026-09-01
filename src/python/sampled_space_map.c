//
// Created by jan on 2025-09-07.
//

#include "sampled_space_map.h"
#include "../integration/integration_rules.h"
#include "../operations/matrices.h"
#include "../polynomials/lagrange.h"
#include "integration_objects.h"
#include <cutl/iterators/combination_iterator.h>

#include <limits.h>
static PyObject *sampled_space_map_get_orders(PyObject *self, void *Py_UNUSED(closure))
{
    const sampled_space_map_object *const this = (sampled_space_map_object *)self;
    PyObject *const res = PyTuple_New(this->ndim);
    if (!res)
        return NULL;

    for (unsigned d = 0; d < this->ndim; ++d)
    {
        PyObject *const order = PyLong_FromSize_t(this->orders[d]);
        if (!order)
        {
            Py_DECREF(res);
            return NULL;
        }
        PyTuple_SET_ITEM(res, d, order);
    }

    return (PyObject *)res;
}

static PyObject *sampled_space_map_get_determinant(PyObject *self, void *Py_UNUSED(closure))
{
    const sampled_space_map_object *const this = (sampled_space_map_object *)self;
    npy_intp *const dims = PyMem_Malloc(sizeof(*dims) * this->ndim);
    if (!dims)
        return NULL;
    for (unsigned d = 0; d < this->ndim; ++d)
        dims[d] = this->orders[d] + 1;
    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim, dims, NPY_DOUBLE, this->determinant);
    PyMem_Free(dims);
    if (!res)
        return NULL;

    if (PyArray_SetBaseObject(res, (PyObject *)this) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);

    return (PyObject *)res;
}

static PyObject *sampled_space_map_get_inverse_map(PyObject *self, void *Py_UNUSED(closure))
{
    const sampled_space_map_object *const this = (sampled_space_map_object *)self;
    npy_intp *const dims = PyMem_Malloc(sizeof(*dims) * (this->ndim + 2));
    if (!dims)
        return NULL;

    for (unsigned d = 0; d < this->ndim; ++d)
        dims[d] = this->orders[d] + 1; // number of points in each reference dimension

    dims[this->ndim] = this->ndim;       // number of reference dimensions (output)
    dims[this->ndim + 1] = this->coords; // number of physical dimensions (input)

    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim + 2, dims, NPY_DOUBLE, this->inverse_maps);
    PyMem_Free(dims);

    if (!res)
        return NULL;

    if (PyArray_SetBaseObject(res, (PyObject *)this) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}
static PyObject *sampled_space_map_get_physical_points(PyObject *self, void *Py_UNUSED(closure))
{
    const sampled_space_map_object *const this = (sampled_space_map_object *)self;
    npy_intp *const dims = PyMem_Malloc(sizeof(*dims) * (this->ndim + 1));
    if (!dims)
        return NULL;
    for (unsigned d = 0; d < this->ndim; ++d)
        dims[d] = this->orders[d] + 1; // number of points in each reference dimension
    dims[this->ndim] = this->coords;   // number of physical dimensions
    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim + 1, dims, NPY_DOUBLE, this->positions);
    PyMem_Free(dims);
    if (!res)
        return NULL;
    if (PyArray_SetBaseObject(res, (PyObject *)this) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

static PyObject *sampled_space_map_get_input_dimensions(PyObject *self, void *Py_UNUSED(closure))
{
    const sampled_space_map_object *const this = (sampled_space_map_object *)self;
    return PyLong_FromSize_t(this->ndim);
}

static PyObject *sampled_space_map_get_output_dimensions(PyObject *self, void *Py_UNUSED(closure))
{
    const sampled_space_map_object *const this = (sampled_space_map_object *)self;
    return PyLong_FromSize_t(this->coords);
}

static PyGetSetDef sampled_space_map_getsetters[] = {
    {
        .name = "input_dimensions",
        .get = sampled_space_map_get_input_dimensions,
        .doc = "int : Number of input dimensions.",
    },
    {
        .name = "output_dimensions",
        .get = sampled_space_map_get_output_dimensions,
        .doc = "int : Number of output dimensions.",
    },
    {
        .name = "orders",
        .get = sampled_space_map_get_orders,
        .doc = "tuple[int, ...] : Orders of the sampling in each dimension.",
    },
    {
        .name = "determinant",
        .get = sampled_space_map_get_determinant,
        .doc = "array : Array with the values of determinant at sampled points.",
    },
    {
        .name = "inverse_map",
        .get = sampled_space_map_get_inverse_map,
        .doc = "array : Local inverse transformation at each sampled point.\n"
               "\n"
               "This array contains inverse mapping matrix, which is used\n"
               "for the contravarying components. When the dimension of the\n"
               "mapping space (as counted by :meth:`SpaceMap.output_dimensions`)\n"
               "is greater than the dimension of the reference space, this is a\n"
               "rectangular matrix, such that it maps the (rectangular) Jacobian\n"
               "to the identity matrix.\n",
    },
    {
        .name = "positions",
        .get = sampled_space_map_get_physical_points,
        .doc = "array : Array with the positions of the sampled points in the physical space.",
    },
    {0},
};

static void sampled_space_map_dealloc(PyObject *self)
{
    sampled_space_map_object *const this = (sampled_space_map_object *)self;
    PyObject_GC_UnTrack(self);
    PyMem_Free(this->determinant);
    PyMem_Free(this->inverse_maps);
    PyMem_Free(this->positions);
    PyMem_Free(this->orders);
    for (unsigned d = 0; d < this->ndim; ++d)
        Py_XDECREF(this->transformations[d]);
    PyMem_Free(this->transformations);
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

PyDoc_STRVAR(sampled_space_map_doc,
             "SampledSpaceMap(space_map: SpaceMap, samples: Sequence[Sequence[float] | array_like], "
             "integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY)\n"
             "Mapping between reference space and target space, sampled from a SpaceMap.\n"
             "\n"
             "A mapping from the reference space to the target space, which maps the\n"
             ":math:`N`-dimensional reference space to an :math:`M`-dimensional\n"
             "physical space. The purpose of this mapping is to provide easier\n"
             "visualization with VTK and other tools that want sampled data.\n"
             "\n"
             "As such, it cannot be used for integration, only mapping k-forms to the\n"
             "target space. It can however be reused for multiple k-forms, as long as\n"
             "they are reconstructed on the same tensor grid.\n"
             "\n"
             "The samples need not be uniformly spaced. If the sample orders are lower\n"
             "than the orders of the actual coordinate map, the resulting sampled map\n"
             "will not be accurate. Otherwise, the accuracy is almost machine precision,\n"
             "since coordinate maps are defined with polynomial basis.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "space_map : SpaceMap\n"
             "    Mapping of the space in which we sample.\n"
             "\n"
             "samples : Sequence[Sequence[float] | array_like]\n"
             "    One-dimensional sample coordinates for each reference dimension. The\n"
             "    number of sample arrays must match the input dimension of the space map.\n"
             "    The arrays define the tensor grid and may have different lengths.\n"
             "\n"
             "integration_registry : IntegrationRegistry, optional\n"
             "    Registry to get the integration rules from. When omitted, the default\n"
             "    registry is used.\n");

PyDoc_STRVAR(sampled_space_map_uniform_doc,
             "on_uniform_grid(space_map: SpaceMap, orders: Sequence[int], "
             "integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY) -> SampledSpaceMap\n"
             "Create a SampledSpaceMap on a uniform grid of points in the reference space.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "space_map : SpaceMap\n"
             "    Mapping of the space in which we sample.\n"
             "\n"
             "orders : Sequence[int]\n"
             "    Orders of the sampling in each dimension. The number of orders must match\n"
             "    the number of input dimensions of the space map. Must not be negative.\n"
             "\n"
             "integration_registry : IntegrationRegistry, optional\n"
             "    Registry to get the integration rules from. When omitted, the default\n"
             "    registry is used.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "SampledSpaceMap\n"
             "    Sampled map evaluated on the requested uniform tensor grid.\n");

static int sampled_space_map_parse_orders(PyObject *orders_obj, const unsigned ndim, unsigned **p_orders)
{
    PyObject *const orders_seq = PySequence_Fast(orders_obj, "orders must be a sequence.");
    if (!orders_seq)
        return -1;
    if (PySequence_Fast_GET_SIZE(orders_seq) != (Py_ssize_t)ndim)
    {
        PyErr_Format(PyExc_ValueError,
                     "orders must have the same length as the dimension of the SpaceMap (got %zd, expected %u).",
                     PySequence_Fast_GET_SIZE(orders_seq), ndim);
        Py_DECREF(orders_seq);
        return -1;
    }

    unsigned *orders = NULL;
    if (ndim != 0)
    {
        orders = PyMem_Malloc(sizeof(*orders) * ndim);
        if (!orders)
        {
            Py_DECREF(orders_seq);
            return -1;
        }
    }
    for (unsigned d = 0; d < ndim; ++d)
    {
        const Py_ssize_t v = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(orders_seq, d), PyExc_ValueError);
        if (PyErr_Occurred())
        {
            PyMem_Free(orders);
            Py_DECREF(orders_seq);
            return -1;
        }
        if (v < 0)
        {
            PyErr_Format(PyExc_ValueError, "orders[%u] must be >= 0 (got %zd).", d, v);
            PyMem_Free(orders);
            Py_DECREF(orders_seq);
            return -1;
        }
        if (v > (Py_ssize_t)UINT_MAX)
        {
            PyErr_Format(PyExc_OverflowError, "orders[%u] is too large (got %zd).", d, v);
            PyMem_Free(orders);
            Py_DECREF(orders_seq);
            return -1;
        }
        orders[d] = (unsigned)v;
    }
    Py_DECREF(orders_seq);
    *p_orders = orders;
    return 0;
}

static PyObject *sampled_space_map_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(type);
    if (!state)
        return NULL;

    space_map_object *smap = NULL;
    PyObject *samples_obj = NULL;
    integration_registry_object *const registry_obj = (integration_registry_object *)state->registry_integration;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O|O!", (char *[]){"space_map", "samples", "integration_registry", NULL},
            state->space_mapping_type, &smap, &samples_obj, state->integration_registry_type, &registry_obj))
        return NULL;

    const unsigned ndim = smap->ndim;
    PyObject *const samples_seq = PySequence_Fast(samples_obj, "samples must be a sequence.");
    if (!samples_seq)
        return NULL;
    if (PySequence_Fast_GET_SIZE(samples_seq) != (Py_ssize_t)ndim)
    {
        PyErr_Format(PyExc_ValueError,
                     "samples must have the same length as the dimension of the SpaceMap (got %zd, expected %u).",
                     PySequence_Fast_GET_SIZE(samples_seq), ndim);
        Py_DECREF(samples_seq);
        return NULL;
    }

    PyArrayObject **const sample_arrays = ndim ? PyMem_Malloc(sizeof(*sample_arrays) * ndim) : NULL;
    if (ndim != 0 && !sample_arrays)
    {
        Py_DECREF(samples_seq);
        return NULL;
    }
    for (unsigned d = 0; d < ndim; ++d)
        sample_arrays[d] = NULL;

    unsigned converted = 0;
    size_t sample_count = 0;
    for (; converted < ndim; ++converted)
    {
        sample_arrays[converted] = (PyArrayObject *)PyArray_FROMANY(PySequence_Fast_GET_ITEM(samples_seq, converted),
                                                                    NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (!sample_arrays[converted])
            break;
        const npy_intp count = PyArray_SIZE(sample_arrays[converted]);
        if (count == 0)
        {
            PyErr_Format(PyExc_ValueError, "samples[%u] must not be empty.", converted);
            break;
        }
        if (count - 1 > (npy_intp)UINT_MAX)
        {
            PyErr_Format(PyExc_OverflowError, "samples[%u] has too many points.", converted);
            break;
        }
        sample_count += (size_t)count;
    }
    if (converted != ndim)
    {
        for (unsigned d = 0; d < ndim; ++d)
            Py_XDECREF(sample_arrays[d]);
        PyMem_Free(sample_arrays);
        Py_DECREF(samples_seq);
        return NULL;
    }

    double *const samples = PyMem_Malloc(sizeof(*samples) * sample_count);
    unsigned *const orders = ndim ? PyMem_Malloc(sizeof(*orders) * ndim) : NULL;
    if ((sample_count != 0 && !samples) || (ndim != 0 && !orders))
    {
        PyMem_Free(samples);
        PyMem_Free(orders);
        for (unsigned d = 0; d < ndim; ++d)
            Py_XDECREF(sample_arrays[d]);
        PyMem_Free(sample_arrays);
        Py_DECREF(samples_seq);
        return NULL;
    }

    size_t offset = 0;
    for (unsigned d = 0; d < ndim; ++d)
    {
        const size_t count = (size_t)PyArray_SIZE(sample_arrays[d]);
        orders[d] = (unsigned)(count - 1);
        memcpy(samples + offset, PyArray_DATA(sample_arrays[d]), sizeof(*samples) * count);
        offset += count;
        Py_DECREF(sample_arrays[d]);
    }
    PyMem_Free(sample_arrays);
    Py_DECREF(samples_seq);

    sampled_space_map_object *const res = sampled_space_map_create(type, smap, orders, samples, registry_obj->registry);
    PyMem_Free(orders);
    PyMem_Free(samples);
    return (PyObject *)res;
}

static PyObject *sampled_space_map_on_uniform_grid(PyObject *cls, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;

    space_map_object *smap = NULL;
    PyObject *orders_obj = NULL;
    integration_registry_object *const registry_obj = (integration_registry_object *)state->registry_integration;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O|O!", (char *[]){"space_map", "orders", "integration_registry", NULL},
            state->space_mapping_type, &smap, &orders_obj, state->integration_registry_type, &registry_obj))
        return NULL;

    unsigned *orders = NULL;
    if (sampled_space_map_parse_orders(orders_obj, smap->ndim, &orders) < 0)
        return NULL;
    sampled_space_map_object *const res =
        sampled_space_map_create((PyTypeObject *)cls, smap, orders, NULL, registry_obj->registry);
    PyMem_Free(orders);
    return (PyObject *)res;
}

static PyMethodDef sampled_space_map_type_methods[] = {
    {
        .ml_name = "on_uniform_grid",
        .ml_meth = (void *)sampled_space_map_on_uniform_grid,
        .ml_flags = METH_CLASS | METH_VARARGS | METH_KEYWORDS,
        .ml_doc = sampled_space_map_uniform_doc,
    },
    {},
};

static PyType_Slot sampled_space_map_type_slots[] = {
    {Py_tp_new, sampled_space_map_new},
    {Py_tp_dealloc, sampled_space_map_dealloc},
    {Py_tp_traverse, heap_type_traverse_type},
    {Py_tp_getset, sampled_space_map_getsetters},
    {Py_tp_methods, sampled_space_map_type_methods},
    {Py_tp_doc, (void *)sampled_space_map_doc},
    {0, NULL},
};

PyType_Spec sampled_space_map_type_spec = {
    .name = "fdg._fdg.SampledSpaceMap",
    .basicsize = sizeof(sampled_space_map_object),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .slots = sampled_space_map_type_slots,
};

static PyArrayObject *sampled_space_map_basis_transform(const sampled_space_map_object *map, const Py_ssize_t order)
{
    const unsigned n_maps = map->coords;
    const unsigned n_dims = map->ndim;
    if (order <= 0 || order > n_dims)
    {
        PyErr_Format(PyExc_ValueError, "Expected order in range (0, %u], but got %zd.", n_dims, order);
        return NULL;
    }

    if (map->transformations[order - 1] != NULL)
    {
        Py_INCREF(map->transformations[order - 1]);
        return map->transformations[order - 1];
    }

    size_t total_points = 1;
    for (unsigned d = 0; d < n_dims; ++d)
        total_points *= map->orders[d] + 1;

    const npy_intp out_dims[3] = {
        combination_total_count(n_dims, order),
        combination_total_count(n_maps, order),
        (npy_intp)total_points,
    };
    PyArrayObject *const res = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_DOUBLE);
    if (!res)
        return NULL;

    if (compute_basis_transform_from_inverse(n_dims, n_maps, (unsigned)order, map->inverse_maps, map->determinant,
                                             total_points, PyArray_DATA(res)) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    map->transformations[order - 1] = res;
    Py_INCREF(res);
    return res;
}

static double sampled_space_map_backward_derivative(const sampled_space_map_object *map, const size_t point_index,
                                                    const unsigned idx_dim, const unsigned idx_coord)
{
    const unsigned n_dims = map->ndim;
    const unsigned jacobian_size = n_dims * map->coords;
    return map->inverse_maps[point_index * jacobian_size + idx_dim * map->coords + idx_coord];
}

PyDoc_STRVAR(
    transform_kform_to_target_sampled_docstring,
    "transform_kform_to_target_sampled(order: int, smap: SampledSpaceMap, components: array_like, *, out: array | "
    "None = None) -> array\n"
    "\n"
    "Transform k-form values based on a sampled space mapping.\n"
    "\n"
    "0-forms do not need a coordinate transformation. This function therefore\n"
    "accepts only orders greater than zero; handle order-zero values directly.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "order : int\n"
    "    Order of the k-form being transformed. Must be at least 1.\n"
    "\n"
    "smap : SampledSpaceMap\n"
    "    Mapping between the reference and target domain to use.\n"
    "\n"
    "components : array_like\n"
    "    Array with values of components of the k-form in the reference domain at\n"
    "    the sampled points associated with the space mapping.\n"
    "\n"
    "out : array, optional\n"
    "    Array to use to store the output in.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Array with values of the components in the physical space.\n");

// TODO: might be possible to factor with the non-sampled version in the future
static PyObject *transform_kform_to_target_sampled(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                                   PyObject *kwnames)
{
    interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    Py_ssize_t order;
    const sampled_space_map_object *map;
    PyObject *py_components, *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {.type = CPYARG_TYPE_PYTHON,
                 .type_check = state->sampled_space_mapping_type,
                 .p_val = &map,
                 .kwname = "smap"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_components, .kwname = "components"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &out, .kwname = "out", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // If NULL is given for "out", it should be the same as it not being given.
    if (out != NULL && Py_IsNone(out))
    {
        out = NULL;
    }

    // Get the shape of the transformation
    const unsigned ndim_in = map->ndim;
    const unsigned ndim_out = map->coords;
    // Check order before using it as an unsigned combination rank.
    if (order < 0 || order > ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected order to be between 1 and %u, but got %zd.", ndim_in, order);
        return NULL;
    }
    const unsigned n_components_in = combination_total_count(ndim_in, order);
    const unsigned n_components_out = combination_total_count(ndim_out, order);

    // Convert components to be an array
    PyArrayObject *const components =
        (PyArrayObject *)PyArray_FROMANY(py_components, NPY_DOUBLE, 1 + ndim_in, 1 + ndim_in, NPY_ARRAY_IN_ARRAY);
    if (!components)
        return NULL;

    // Check the shape is correct
    const npy_intp *const dims_in = PyArray_DIMS(components);
    const unsigned ndim_components = PyArray_NDIM(components);
    if (dims_in[0] != n_components_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected components to have shape (%u, ...), but got (%zd, ...).",
                     n_components_in, dims_in[0]);
        Py_DECREF(components);
        return NULL;
    }
    // The other dimensions must match the integration rule used by the space map
    for (unsigned idim = 0; idim < ndim_in; ++idim)
    {
        const npy_intp size_in = dims_in[idim + 1];
        if (size_in != map->orders[idim] + 1)
        {
            PyErr_Format(PyExc_ValueError,
                         "Components dimension %u did not match the integration rule of order %u as specified by the "
                         "space map (instead it was %zd).",
                         idim, map->orders[idim], size_in);
            Py_DECREF(components);
            return NULL;
        }
    }

    // Create an output array if needed
    PyArrayObject *out_array;
    if (out == NULL)
    {
        // Create the output array
        npy_intp *const dims_out = PyMem_Malloc(sizeof(*dims_out) * ndim_components);
        if (!dims_out)
        {
            Py_DECREF(components);
            return NULL;
        }
        dims_out[0] = n_components_out;
        for (unsigned idim = 1; idim < ndim_components; ++idim)
        {
            dims_out[idim] = dims_in[idim];
        }
        out_array = (PyArrayObject *)PyArray_SimpleNew(ndim_components, dims_out, NPY_DOUBLE);
        PyMem_Free(dims_out);
        if (!out_array)
        {
            Py_DECREF(components);
            return NULL;
        }
    }
    else
    {
        // We were given one
        if (!PyArray_Check(out))
        {
            PyErr_Format(PyExc_TypeError, "Expected out to be an array, but got %s.", Py_TYPE(out)->tp_name);
            Py_DECREF(components);
            return NULL;
        }
        out_array = (PyArrayObject *)out;
        // Check the shape is correct
        const npy_intp *const dims_out = PyArray_DIMS(out_array);
        if (dims_out[0] != n_components_out)
        {
            PyErr_Format(PyExc_ValueError, "Expected output to have shape (%u, ...), but got (%zd, ...).",
                         n_components_out, dims_out[0]);
            Py_DECREF(components);
            return NULL;
        }
        for (unsigned i = 1; i < ndim_components; ++i)
        {
            if (dims_out[i] != dims_in[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "Expected output to have the same shape as the input after the first dimension, but got "
                             "%zd for dimension %u.",
                             dims_out[i], i);
                Py_DECREF(components);
                return NULL;
            }
        }
        Py_INCREF(out_array);
    }

    size_t total_points = 1;
    for (unsigned d = 0; d < ndim_in; ++d)
        total_points *= map->orders[d] + 1;

    const double *restrict const ptr_components = PyArray_DATA(components);
    double *restrict const ptr_out = PyArray_DATA(out_array);

    if (order == 1)
    {
        // transform_covariant_to_target_impl(total_points, PyArray_DATA(components), PyArray_DATA(out_array), ndim_out,
        //                                    ndim_in, map);
        Py_BEGIN_ALLOW_THREADS;
        for (size_t i = 0; i < total_points; ++i)
        {
#pragma omp simd
            for (unsigned i_out = 0; i_out < ndim_out; ++i_out)
            {
                double val = 0.0;
                for (unsigned i_in = 0; i_in < ndim_in; ++i_in)
                {
                    val += ptr_components[i_in * total_points + i] *
                           sampled_space_map_backward_derivative(map, i, i_in, i_out);
                }
                ptr_out[i_out * total_points + i] = val;
            }
        }
        Py_END_ALLOW_THREADS;
        return (PyObject *)out_array;
    }

    PyArrayObject *const transformation_array = sampled_space_map_basis_transform(map, order);
    if (!transformation_array)
    {
        Py_DECREF(components);
        Py_DECREF(out_array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    const double *restrict const ptr_transformation = PyArray_DATA(transformation_array);
#pragma omp simd
    for (size_t i_pt = 0; i_pt < total_points; ++i_pt)
    {
        for (unsigned i_out = 0; i_out < n_components_out; ++i_out)
        {
            double val = 0.0;
            for (unsigned i_in = 0; i_in < n_components_in; ++i_in)
            {
                val +=
                    ptr_transformation[(size_t)i_in * n_components_out * total_points + i_out * total_points + i_pt] *
                    ptr_components[i_in * total_points + i_pt];
            }
            ptr_out[i_out * total_points + i_pt] = val;
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(transformation_array);
    Py_DECREF(components);
    return (PyObject *)out_array;
}

PyMethodDef sampled_space_map_methods[] = {
    {
        .ml_doc = transform_kform_to_target_sampled_docstring,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_name = "transform_kform_to_target_sampled",
        .ml_meth = (void *)transform_kform_to_target_sampled,
    },
    {0},
};

sampled_space_map_object *sampled_space_map_create(PyTypeObject *type, space_map_object *map, const unsigned *orders,
                                                   const double *samples, integration_rule_registry_t *registry)
{
    const unsigned ndim_in = map->ndim;
    const unsigned ndim_out = Py_SIZE(map);

    sampled_space_map_object *const this = (sampled_space_map_object *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    size_t total_points = 1;
    for (unsigned d = 0; d < ndim_in; ++d)
        total_points *= orders[d] + 1;

    this->ndim = ndim_in;
    this->coords = ndim_out;
    this->orders = PyMem_Malloc(sizeof(unsigned) * ndim_in);
    this->determinant = PyMem_Malloc(sizeof(double) * total_points); // One determinant value per sampled point
    this->inverse_maps =
        PyMem_Malloc(sizeof(double) * total_points * ndim_in * ndim_out);     // One inverse map (ndim_out x ndim_in)
    this->positions = PyMem_Malloc(sizeof(double) * total_points * ndim_out); // One position vector per sampled point
    this->transformations =
        PyMem_Malloc(sizeof(PyArrayObject *) * ndim_in); // One transformation matrix per input dimension
    if (this->transformations)
    {
        // This must be set before we can call Py_DECREF(this) as the destructor tries to free non-NULL
        for (unsigned d = 0; d < ndim_in; ++d)
            this->transformations[d] = NULL;
    }
    if (!this->determinant || !this->inverse_maps || !this->positions || !this->orders || !this->transformations)
    {
        Py_DECREF(this);
        return NULL;
    }

    for (unsigned d = 0; d < ndim_in; ++d)
        this->orders[d] = orders[d];

    // Store one interpolation matrix per input dimension. The matrix layout is
    // input-node-major, with the sampled-node index varying fastest.
    size_t transformation_size = 0;
    size_t nodes_in = 1;
    unsigned max_out_order = 0;
    for (unsigned d = 0; d < ndim_in; ++d)
    {
        const size_t n_out = (size_t)orders[d] + 1;
        const size_t n_int = (size_t)map->int_specs[d].order + 1;
        transformation_size += n_out * n_int;
        nodes_in *= n_int;
        if (orders[d] > max_out_order)
            max_out_order = orders[d];
    }
    const size_t root_storage = samples ? 0 : (size_t)max_out_order + 1;
    double *const axis_transformations =
        PyMem_Malloc(sizeof(*axis_transformations) * (transformation_size + root_storage));
    if (!axis_transformations)
    {
        Py_DECREF(this);
        return NULL;
    }

    // Compute the interpolation matrices for each dimension.
    const integration_rule_t **const rules = python_integration_rules_get(ndim_in, map->int_specs, registry);
    if (!rules)
    {
        PyMem_Free(axis_transformations);
        Py_DECREF(this);
        return NULL;
    }
    double *const uniform_nodes = samples ? NULL : axis_transformations + transformation_size;
    size_t axis_offset = 0;
    size_t sample_offset = 0;
    for (unsigned d = 0; d < ndim_in; ++d)
    {
        const unsigned order_out = orders[d];
        const unsigned order_int = map->int_specs[d].order;
        const unsigned n_out = order_out + 1;
        const unsigned n_int = order_int + 1;
        const double *sample_nodes;
        if (samples)
        {
            sample_nodes = samples + sample_offset;
        }
        else
        {
            for (unsigned j = 0; j <= order_out; ++j)
                uniform_nodes[j] = order_out == 0 ? 0.0 : (2.0 * (double)j) / (double)order_out - 1.0;
            sample_nodes = uniform_nodes;
        }
        lagrange_polynomial_values_transposed_2(n_out, sample_nodes, n_int, integration_rule_nodes_const(rules[d]),
                                                axis_transformations + axis_offset);
        axis_offset += (size_t)n_out * n_int;
        sample_offset += n_out;
    }
    python_integration_rules_release(ndim_in, rules, registry);

    const size_t trans_size = (size_t)ndim_in * ndim_out;
    matrix_t jacobian_mat = {.rows = ndim_out, .cols = ndim_in, .values = NULL};
    matrix_t q_mat = {.rows = ndim_out, .cols = ndim_out, .values = NULL};
    void *const work_ptr = cutl_alloc_group(
        &SYSTEM_ALLOCATOR, (const cutl_alloc_info_t[]){
                               {.size = sizeof(double) * ndim_in * ndim_out, .p_ptr = (void **)&jacobian_mat.values},
                               {.size = sizeof(double) * ndim_out * ndim_out, .p_ptr = (void **)&q_mat.values},
                               {},
                           });
    if (!work_ptr)
    {
        PyMem_Free(axis_transformations);
        Py_DECREF(this);
        return NULL;
    }

    // Interpolate positions and forward transformation matrices together. This
    // avoids allocating the dense tensor-product interpolation matrix.
    for (size_t i_out = 0; i_out < total_points; ++i_out)
    {
        for (unsigned idim_out = 0; idim_out < ndim_out; ++idim_out)
        {
            this->positions[ndim_out * i_out + idim_out] = 0.0;
            for (unsigned idim_in = 0; idim_in < ndim_in; ++idim_in)
                jacobian_mat.values[idim_out * ndim_in + idim_in] = 0.0;
        }

        for (size_t i_in = 0; i_in < nodes_in; ++i_in)
        {
            double weight = 1.0;
            size_t output_stride = total_points;
            size_t input_stride = nodes_in;
            size_t matrix_offset = 0;
            for (unsigned d = 0; d < ndim_in; ++d)
            {
                const unsigned n_out = orders[d] + 1;
                const unsigned n_int = map->int_specs[d].order + 1;
                output_stride /= n_out;
                input_stride /= n_int;
                const unsigned idx_out = (i_out / output_stride) % n_out;
                const unsigned idx_in = (i_in / input_stride) % n_int;
                weight *= axis_transformations[matrix_offset + (size_t)idx_in * n_out + idx_out];
                matrix_offset += (size_t)n_out * n_int;
            }

            for (unsigned idim_out = 0; idim_out < ndim_out; ++idim_out)
            {
                const coordinate_map_object *const coordinate = map->maps[idim_out];
                this->positions[ndim_out * i_out + idim_out] += weight * coordinate->values[i_in];
                for (unsigned idim_in = 0; idim_in < ndim_in; ++idim_in)
                    jacobian_mat.values[idim_out * ndim_in + idim_in] +=
                        weight * coordinate_map_gradient(coordinate, idim_in)[i_in];
            }
        }

        const matrix_t out_mat = {.rows = ndim_in, .cols = ndim_out, .values = this->inverse_maps + i_out * trans_size};
        this->determinant[i_out] = compute_inverse_transform(jacobian_mat, q_mat, out_mat);
    }

    cutl_dealloc(&SYSTEM_ALLOCATOR, work_ptr);
    PyMem_Free(axis_transformations);
    return this;
}
