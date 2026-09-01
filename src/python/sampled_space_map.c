//
// Created by jan on 2025-09-07.
//

#include "sampled_space_map.h"
#include "../integration/integration_rules.h"
#include "../operations/matrices.h"
#include "../polynomials/lagrange.h"
#include "integration_objects.h"
#include <cutl/iterators/combination_iterator.h>

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

PyDoc_STRVAR(sampled_space_map_doc, "SampledSpaceMap(space_map: SpaceMap, orders: Sequence[int], integration_registry: "
                                    "IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY)\n"
                                    "Mapping between reference space and target space, sampled from a SpaceMap.\n"
                                    "\n"
                                    "A mapping from the reference space to the target space, which maps the\n"
                                    ":math:`N`-dimensional reference space to an :math:`M`-dimensional\n"
                                    "physical space. The purpose of this mapping is to provide easier\n"
                                    "visualization with VTK and other tools that want uniformly sampled data.\n"
                                    "\n"
                                    "As such, it cannot be used for integration, only mapping k-forms to the\n"
                                    "target space. It can however be reused for multiple k-forms, as long as\n"
                                    "they are reconstructed on the appropriate uniform grid.\n"
                                    "\n"
                                    "Note that due to how the interpolation works, if the orders are lower than\n"
                                    "the orders of the actual coordinate map, the resulting sampled map will\n"
                                    "not be accurate. Otherwise, the accuracy of the sampled map is almost\n"
                                    "machine precision, since coordinate maps are defined with polynomial basis.\n"
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
                                    "    registry is used.\n");

static PyObject *sampled_space_map_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(type);
    if (!state)
        return NULL;

    space_map_object *smap = NULL;
    PyObject *orders_obj = NULL;
    integration_registry_object *const registry_obj = (integration_registry_object *)state->registry_integration;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O|O!", (char *[]){"space_map", "orders", "integration_registry", NULL},
            state->space_mapping_type, &smap, &orders_obj, state->integration_registry_type, &registry_obj))
        return NULL;

    const unsigned ndim = smap->ndim;
    PyObject *const orders_seq = PySequence_Fast(orders_obj, "orders must be a sequence.");
    if (!orders_seq)
        return NULL;
    if (PySequence_Fast_GET_SIZE(orders_seq) != (Py_ssize_t)ndim)
    {
        PyErr_Format(PyExc_ValueError,
                     "orders must have the same length as the dimension of the SpaceMap (got %zd, expected %u).",
                     PySequence_Fast_GET_SIZE(orders_seq), ndim);
        Py_DECREF(orders_seq);
        return NULL;
    }
    unsigned *const orders = PyMem_Malloc(sizeof(unsigned) * ndim);
    if (!orders)
    {
        Py_DECREF(orders_seq);
        return NULL;
    }

    for (unsigned d = 0; d < ndim; ++d)
    {
        const Py_ssize_t v = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(orders_seq, d), PyExc_ValueError);

        if (PyErr_Occurred())
        {
            PyMem_Free(orders);
            Py_DECREF(orders_seq);
            return NULL;
        }
        if (v < 0)
        {
            PyErr_Format(PyExc_ValueError, "orders[%u] must be >= 0 (got %zd).", d, v);
            PyMem_Free(orders);
            Py_DECREF(orders_seq);
            return NULL;
        }
        orders[d] = (unsigned)v;
    }
    Py_DECREF(orders_seq);
    sampled_space_map_object *const res = sampled_space_map_create(type, smap, orders, registry_obj->registry);
    PyMem_Free(orders);
    return (PyObject *)res;
}

static PyType_Slot sampled_space_map_type_slots[] = {
    {Py_tp_new, sampled_space_map_new},         {Py_tp_dealloc, sampled_space_map_dealloc},
    {Py_tp_traverse, heap_type_traverse_type},  {Py_tp_getset, sampled_space_map_getsetters},
    {Py_tp_doc, (void *)sampled_space_map_doc}, {0, NULL},
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
    "Parameters\n"
    "----------\n"
    "order : int\n"
    "    Order of the k-form being transformed.\n"
    "\n"
    "smap : SampledSpaceMap\n"
    "    Mapping between the reference and target domain to use.\n"
    "\n"
    "components : array_like\n"
    "    Array with values of components of the k-form in the reference domain at\n"
    "    integration points associated with the space mapping.\n"
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
    const unsigned n_components_in = combination_total_count(ndim_in, order);
    const unsigned n_components_out = combination_total_count(ndim_out, order);

    // Check order
    if (order < 0 || order > ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected order to be between 0 and %u, but got %zd.", ndim_in, order);
        return NULL;
    }

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
        for (unsigned i = 0; i < ndim_components; ++i)
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
    if (order == 0)
    {
        // Copy from input to output and we are done
        if (py_components != out) // If they are the same, we don't need to copy
        {
            Py_BEGIN_ALLOW_THREADS;
            memcpy(ptr_out, ptr_components, sizeof(double) * total_points);
            Py_END_ALLOW_THREADS;
        }
        Py_DECREF(components);
        return (PyObject *)out_array;
    }

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
                                                   integration_rule_registry_t *registry)
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

    // Get unique orders (out and int) sorted in ascending order
    typedef struct
    {
        unsigned order_out;           // Order of the output grid for this dimension
        unsigned order_int;           // Order of the integration rule for this dimension (input points)
        integration_rule_type_t type; // Type of the integration rule for this dimension
        unsigned source_dim;          // Source dimension providing the integration rule
        unsigned offset;              // Offset in the output grid for this dimension
    } interpolation_info_t;
    interpolation_info_t *const unique_orders = PyMem_Malloc(sizeof(*unique_orders) * ndim_in);
    if (!unique_orders)
    {
        Py_DECREF(this);
        return NULL;
    }
    // Insertion sort will be fastest for such a small array
    unsigned n_unique = 0, transformation_size = 0, max_out_order = 0, nodes_in = 1;
    for (unsigned d = 0; d < ndim_in; ++d)
    {
        const unsigned order_out = orders[d];
        const unsigned order_int = map->int_specs[d].order;
        const integration_rule_type_t type = map->int_specs[d].type;
        nodes_in *= order_int + 1;
        unsigned i = 0;
        while (i < n_unique && (unique_orders[i].order_out < order_out ||
                                (unique_orders[i].order_out == order_out &&
                                 (unique_orders[i].order_int < order_int ||
                                  (unique_orders[i].order_int == order_int && unique_orders[i].type < type)))))
            ++i;
        if (i == n_unique || unique_orders[i].order_out != order_out || unique_orders[i].order_int != order_int ||
            unique_orders[i].type != type)
        {
            // Shift elements to the right to make space for the new order
            for (unsigned j = n_unique; j > i; --j)
                unique_orders[j] = unique_orders[j - 1];
            // Insert the new order
            unique_orders[i].order_out = order_out;
            unique_orders[i].order_int = order_int;
            unique_orders[i].type = type;
            unique_orders[i].source_dim = d;
            unique_orders[i].offset = transformation_size; // Store the offset for this unique order
            transformation_size += (order_out + 1) * (order_int + 1);
            if (order_out > max_out_order)
                max_out_order = order_out;
            ++n_unique;
        }
    }

    // Allocate memory for the roots of the unique orders
    const unsigned extra_nodes =
        max_out_order + 1 > nodes_in * total_points ? max_out_order + 1 : nodes_in * total_points;
    double *const axis_transformations = PyMem_Malloc(sizeof(double) * (transformation_size + extra_nodes));
    if (!axis_transformations)
    {
        PyMem_Free(unique_orders);
        Py_DECREF(this);
        return NULL;
    }

    // We will reuse this for both output roots and full transformation matrix
    double *const global_transformation = axis_transformations + transformation_size;
    // Interpolation matrix computation
    {
        // Get the integration rules for the space map's integration space
        const integration_rule_t **const rules = python_integration_rules_get(ndim_in, map->int_specs, registry);
        if (!rules)
        {
            PyMem_Free(axis_transformations);
            PyMem_Free(unique_orders);
            Py_DECREF(this);
            return NULL;
        }

        // Compute per-axis matrices for each unique combination
        unsigned current_root_order = ~0u; // Initialize to an invalid order
        for (unsigned i = 0; i < n_unique; ++i)
        {
            const unsigned order_out = unique_orders[i].order_out;
            const unsigned order_int = unique_orders[i].order_int;
            const unsigned offset = unique_orders[i].offset;
            // Prepare the uniform roots
            if (current_root_order != order_out)
            {
                // Uniformly spaced roots for the output grid
                for (unsigned j = 0; j <= order_out; ++j)
                    global_transformation[j] = order_out == 0 ? 0.0 : (2.0 * (double)j) / (double)order_out - 1.0;
                current_root_order = order_out;
            }
            const double *restrict int_nodes = integration_rule_nodes_const(rules[unique_orders[i].source_dim]);
            // Compute the transformation matrix for this unique combination
            lagrange_polynomial_values_transposed_2(order_out + 1, global_transformation, order_int + 1, int_nodes,
                                                    axis_transformations + offset);
        }
        // Release the integration rules
        python_integration_rules_release(ndim_in, rules, registry);

        // Now assemble the global interpolation matrix for the entire space map.
        // Both the output and input grids use row-major tensor-product ordering.
        for (size_t i_out = 0; i_out < total_points; ++i_out)
        {
            for (size_t i_in = 0; i_in < nodes_in; ++i_in)
            {
                double val = 1.0;
                size_t stride_out = total_points;
                size_t stride_int = nodes_in;
                for (unsigned idim = 0; idim < ndim_in; ++idim)
                {
                    unsigned idx_order = 0;
                    for (unsigned j = 0; j < n_unique; ++j)
                    {
                        if (unique_orders[j].order_out == orders[idim] &&
                            unique_orders[j].order_int == map->int_specs[idim].order &&
                            unique_orders[j].type == map->int_specs[idim].type)
                        {
                            idx_order = j;
                            break;
                        }
                    }
                    ASSERT(idx_order < n_unique, "Unique order not found for dimension %u.", idim);
                    const interpolation_info_t info = unique_orders[idx_order];
                    stride_out /= info.order_out + 1;
                    stride_int /= info.order_int + 1;
                    const unsigned idx_out = (i_out / stride_out) % (info.order_out + 1);
                    const unsigned idx_in = (i_in / stride_int) % (info.order_int + 1);
                    const double *restrict axis_transform_matrix = axis_transformations + info.offset;
                    val *= axis_transform_matrix[idx_in * (info.order_out + 1) + idx_out];
                }
                global_transformation[i_out * nodes_in + i_in] = val;
            }
        }
    }
    // No longer needed since we computed the full dense matrix
    PyMem_Free(unique_orders);

    // First interpolation: positions
    for (unsigned idim_out = 0; idim_out < ndim_out; ++idim_out)
    {
        // Values of the coordinate map for the current output dimension
        const double *const values = map->maps[idim_out]->values;
        // Interpolate the values to the output grid
        for (unsigned i = 0; i < total_points; ++i)
        {
            double val = 0.0;
            for (unsigned j = 0; j < nodes_in; ++j)
                val += global_transformation[i * nodes_in + j] * values[j];
            this->positions[ndim_out * i + idim_out] = val;
        }
    }

    // Second interpolation: transformation matrices (gradients)
    {
        const unsigned trans_size = ndim_in * ndim_out;
        // Allocate memory for the work matrices
        matrix_t jacobian_mat = {.rows = ndim_out, .cols = ndim_in, .values = NULL};
        matrix_t q_mat = {.rows = ndim_out, .cols = ndim_out, .values = NULL};
        void *const work_ptr = cutl_alloc_group(
            &SYSTEM_ALLOCATOR,
            (const cutl_alloc_info_t[]){
                {.size = sizeof(double) * (ndim_in * ndim_out), .p_ptr = (void **)&jacobian_mat.values},
                {.size = sizeof(double) * (ndim_out * ndim_out), .p_ptr = (void **)&q_mat.values},
                {},
            });
        if (!work_ptr)
        {
            PyMem_Free(axis_transformations);
            Py_DECREF(this);
            return NULL;
        }

        for (unsigned i_out = 0; i_out < total_points; ++i_out)
        {
            const matrix_t out_mat = {
                .rows = ndim_in, .cols = ndim_out, .values = this->inverse_maps + i_out * trans_size};
            // Interpolate the forward transformation matrices (gradients) for each output dimension
            for (unsigned idim_out = 0; idim_out < ndim_out; ++idim_out)
                for (unsigned idim_in = 0; idim_in < ndim_in; ++idim_in)
                {
                    const double *const grad_vals = coordinate_map_gradient(map->maps[idim_out], idim_in);
                    double val = 0.0;
                    for (unsigned j = 0; j < nodes_in; ++j)
                        val += global_transformation[i_out * nodes_in + j] * grad_vals[j];
                    jacobian_mat.values[idim_out * ndim_in + idim_in] = val;
                }

            // Invert the forward transformation matrix
            this->determinant[i_out] = compute_inverse_transform(jacobian_mat, q_mat, out_mat);
        }

        // Release the work memory
        cutl_dealloc(&SYSTEM_ALLOCATOR, work_ptr);
    }
    // Release the transformation matrix
    PyMem_Free(axis_transformations);

    return this;
}
