#include "integration_objects.h"
#include <Python.h>
#include <numpy/ndarrayobject.h>

integration_registry_object *integration_registry_object_create(PyTypeObject *type)
{
    integration_registry_object *const self = (integration_registry_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;
    self->registry = NULL;
    const interp_result_t res = integration_rule_registry_create(&self->registry, 1, &SYSTEM_ALLOCATOR);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not initialize integration rule registry: %s (%s)",
                     interp_error_str(res), interp_error_msg(res));
        Py_DECREF(self);
        return NULL;
    }
    return self;
}
integration_specs_object *integration_specs_object_create(PyTypeObject *type, const integration_spec_t spec)
{
    integration_specs_object *const self = (integration_specs_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;
    self->spec = spec;
    return self;
}

static PyObject *integration_registry_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 0 || (kwds && PyDict_Size(kwds) != 0))
    {
        PyErr_SetString(PyExc_TypeError, "IntegrationRegistry takes no arguments.");
        return NULL;
    }

    integration_registry_object *const self = integration_registry_object_create(type);
    return (PyObject *)self;
}

static int ensure_integration_registry_state(PyObject *self, PyTypeObject *defining_class,
                                             integration_registry_object **p_this,
                                             const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->integration_registry_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_registry_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_this = (integration_registry_object *)self;
    *p_state = state;

    return 0;
}

static PyObject *integration_registry_repr(PyObject *self)
{
    const interplib_module_state_t *state;
    integration_registry_object *this;
    if (ensure_integration_registry_state(self, NULL, &this, &state) < 0)
        return NULL;

    return PyUnicode_FromFormat("<%s at %p>", state->integration_registry_type->tp_name, this->registry);
}

static void integration_registry_dealloc(integration_registry_object *self)
{
    PyObject_GC_UnTrack(self);
    if (self->registry)
    {
        integration_rule_registry_destroy(self->registry);
        self->registry = NULL;
    }
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

PyDoc_STRVAR(integration_registry_doc, "Registry for integration rules.\n"
                                       "\n"
                                       "This registry contains all available integration rules and caches them for\n"
                                       "efficient retrieval.\n");

static PyObject *integration_registry_usage(PyObject *self, PyTypeObject *defining_class,
                                            PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs, PyObject *kwnames)
{
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "usage() takes no arguments.");
        return NULL;
    }

    const interplib_module_state_t *state;
    integration_registry_object *this;
    if (ensure_integration_registry_state(self, defining_class, &this, &state) < 0)
        return NULL;

    unsigned count = integration_rule_get_rules(this->registry, 0, NULL);

    integration_spec_t *const specs = PyMem_Malloc(count * sizeof(*specs));
    if (!specs)
        return NULL;
    {
        const unsigned new_cnt = integration_rule_get_rules(this->registry, count, specs);
        count = new_cnt < count ? new_cnt : count;
    }

    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(count);
    if (!out)
        return NULL;
    for (unsigned i = 0; i < count; ++i)
    {
        integration_specs_object *const val = integration_specs_object_create(state->integration_spec_type, specs[i]);
        if (!val)
        {
            Py_DECREF(out);
            PyMem_Free(specs);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, val);
    }
    PyMem_Free(specs);
    return (PyObject *)out;
}

static PyObject *integration_registry_clear(PyObject *self, PyTypeObject *defining_class,
                                            PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs, PyObject *kwnames)
{
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "clear() takes no arguments.");
        return NULL;
    }

    const interplib_module_state_t *state;
    integration_registry_object *this;
    if (ensure_integration_registry_state(self, defining_class, &this, &state) < 0)
        return NULL;

    integration_rule_registry_release_unused_rules(this->registry);

    Py_RETURN_NONE;
}

PyType_Spec integration_registry_type_spec = {
    .name = FDG_TYPE_NAME("IntegrationRegistry"),
    .basicsize = sizeof(integration_registry_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, integration_registry_new},
            {Py_tp_repr, integration_registry_repr},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, integration_registry_dealloc},
            {Py_tp_doc, (void *)integration_registry_doc},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     "usage",
                     (void *)integration_registry_usage,
                     METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     "usage() -> tuple[IntegrationSpecs, ...]\nReturns a list of currently stored rules.",
                 },
                 {
                     "clear",
                     (void *)integration_registry_clear,
                     METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     "clear() -> None\nClears all stored rules.",
                 },
                 {},
             }},
            {},
        },
};

/**
 * Determines the integration type based on the provided method string.
 *
 * @param method_str A string representing the integration method. Supported values are:
 *                   - "gauss" for Gauss-Legendre integration
 *                   - "gauss-lobatto" for Gauss-Lobatto integration
 * @return The integration rule type as an `integration_rule_type_t` enum value.
 *         Returns:
 *         - `INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE` if the method is "gauss".
 *         - `INTEGRATION_RULE_TYPE_GAUSS_LOBATTO` if the method is "gauss-lobatto".
 *         - `INTEGRATION_RULE_TYPE_NONE` if the method is unknown.
 *         If the method is unknown, it also raises a Python ValueError with a descriptive message.
 */
static integration_rule_type_t determine_integration_type(const char *method_str)
{
    if (strcmp(method_str, "gauss") == 0)
    {
        return INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE;
    }

    if (strcmp(method_str, "gauss-lobatto") == 0)
    {
        return INTEGRATION_RULE_TYPE_GAUSS_LOBATTO;
    }

    PyErr_Format(PyExc_ValueError, "Unknown integration method \"%s\".", method_str);
    return INTEGRATION_RULE_TYPE_NONE;
}

/* __new__ method */
static PyObject *integration_specs_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    int order;
    const char *method_str = "gauss";

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|s", (char *[]){"order", "method", NULL}, &order, &method_str))
    {
        return NULL;
    }

    if (order < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Order must be a non-negative integer.");
        return NULL;
    }

    const integration_rule_type_t integration_type = determine_integration_type(method_str);
    if (integration_type == INTEGRATION_RULE_TYPE_NONE)
        return NULL;

    integration_specs_object *self = (integration_specs_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->spec = (integration_spec_t){.order = order, .type = integration_type};

    return (PyObject *)self;
}

/* Property: order */
static PyObject *integration_spec_get_order(const integration_specs_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->spec.order);
}

/* Property: accuracy */
static PyObject *integration_spec_get_accuracy(const integration_specs_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(integration_rule_spec_get_accuracy(self->spec));
}

/* Property: method */
static PyObject *integration_spec_get_method(const integration_specs_object *self, void *Py_UNUSED(closure))
{
    return PyUnicode_FromString(integration_rule_type_to_str(self->spec.type));
}

static PyObject *integration_rule_repr(PyObject *self)
{
    const integration_specs_object *const this = (integration_specs_object *)self;
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return NULL;
    }
    if (!PyObject_TypeCheck(self, state->integration_spec_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_spec_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return NULL;
    }

    return PyUnicode_FromFormat("IntegrationSpecs(%d, method=%s)", this->spec.order,
                                integration_rule_type_to_str(this->spec.type));
}

/* Get/set definitions */
static PyGetSetDef integration_rule_getset[] = {
    {
        "order",
        (getter)integration_spec_get_order,
        NULL,
        PyDoc_STR("int : Order of the integration rule."),
        NULL,
    },
    {
        "accuracy",
        (getter)integration_spec_get_accuracy,
        NULL,
        PyDoc_STR("int : Highest order of polynomial that is integrated exactly."),
        NULL,
    },
    {
        "method",
        (getter)integration_spec_get_method,
        NULL,
        PyDoc_STR("_IntegrationMethodHint : Method used for integration."),
        NULL,
    },
    {},
};

static int ensure_integration_spec_state(PyObject *self, PyTypeObject *defining_class,
                                         integration_specs_object **p_this, const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->integration_spec_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_spec_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_this = (integration_specs_object *)self;
    *p_state = state;
    return 0;
}

static PyArrayObject *integration_specs_prepare_array(PyObject *self, PyTypeObject *defining_class,
                                                      PyObject *const *args, const Py_ssize_t nargs,
                                                      const PyObject *kwnames, const integration_rule_t **p_rule,
                                                      integration_rule_registry_t **p_registry)
{
    const interplib_module_state_t *state;
    integration_specs_object *this;
    if (ensure_integration_spec_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (nargs > 1 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Only one argument may be passed to the method.");
        return NULL;
    }

    const integration_registry_object *registry_object;
    if (nargs == 1)
    {
        if (!PyObject_TypeCheck(args[0], state->integration_registry_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_registry_type->tp_name,
                         Py_TYPE(args[0]));
            return NULL;
        }
        registry_object = (integration_registry_object *)args[0];
    }
    else
    {
        registry_object = (integration_registry_object *)state->registry_integration;
    }

    integration_rule_registry_t *const registry = registry_object->registry;

    const integration_rule_t *rule;
    const interp_result_t res = integration_rule_registry_get_rule(registry, this->spec, &rule);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to retrieve integration rule: %s (%s).", interp_error_str(res),
                     interp_error_msg(res));
        return NULL;
    }

    *p_rule = rule;
    *p_registry = registry;

    const npy_intp dims[] = {rule->n_nodes};
    return (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
}

PyDoc_STRVAR(
    integration_specs_nodes_docstring,
    "nodes(registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY) -> numpy.typing.NDArray[numpy.double]\n"
    "Get the integration nodes.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
    "    Registry used to retrieve the integration rule.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Array of integration nodes.\n");

static PyObject *integration_specs_nodes(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames)
{
    const integration_rule_t *rule;
    integration_rule_registry_t *registry;
    PyArrayObject *out = integration_specs_prepare_array(self, defining_class, args, nargs, kwnames, &rule, &registry);
    if (out)
    {
        npy_double *const p_out = PyArray_DATA(out);
        const double *const nodes = integration_rule_nodes_const(rule);
        for (unsigned i = 0; i < rule->n_nodes; ++i)
        {
            p_out[i] = nodes[i];
        }
    }
    const interp_result_t res = integration_rule_registry_release_rule(registry, rule);
    (void)res;
    ASSERT(res == FDG_SUCCESS, "Rule from the registry had to be successfully returned.");
    return (PyObject *)out;
}

PyDoc_STRVAR(
    integration_specs_weights_docstring,
    "weights(registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY) -> numpy.typing.NDArray[numpy.double]\n"
    "Get the integration weights.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
    "    Registry used to retrieve the integration rule.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Array of integration weights.\n");

static PyObject *integration_specs_weights(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const integration_rule_t *rule;
    integration_rule_registry_t *registry;
    PyArrayObject *out = integration_specs_prepare_array(self, defining_class, args, nargs, kwnames, &rule, &registry);
    if (out)
    {
        npy_double *const p_out = PyArray_DATA(out);
        const double *const weights = integration_rule_weights_const(rule);
        for (unsigned i = 0; i < rule->n_nodes; ++i)
        {
            p_out[i] = weights[i];
        }
    }
    const interp_result_t res = integration_rule_registry_release_rule(registry, rule);
    (void)res;
    ASSERT(res == FDG_SUCCESS, "Rule from the registry had to be successfully returned.");
    return (PyObject *)out;
}

PyDoc_STRVAR(integration_specs_docstring,
             "IntegrationSpecs(order : int , /, method: typing.Literal[\"gauss\", \"gauss-lobatto\"] = \"gauss\")\n"
             "Type that describes an integration rule.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "    Order of the integration rule.\n"
             "\n"
             "method : typing.Literal[\"gauss\", \"gauss-lobatto\"], default: \"gauss\"\n"
             "    Method used for integration.\n");

PyObject *integration_spec_richcompare(PyObject *self, PyObject *other, const int op)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        PyErr_Clear();
        state = interplib_get_module_state(Py_TYPE(other));
        if (!state)
        {
            return NULL;
        }
    }
    if (!PyObject_TypeCheck(self, state->integration_spec_type) ||
        !PyObject_TypeCheck(other, state->integration_spec_type) || (op != Py_EQ && op != Py_NE))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    const integration_specs_object *const this = (integration_specs_object *)self;
    const integration_specs_object *const other_specs = (integration_specs_object *)other;
    const int equal = this->spec.order == other_specs->spec.order && this->spec.type == other_specs->spec.type;
    return PyBool_FromLong(op == Py_EQ ? equal : !equal);
}

PyType_Spec integration_specs_type_spec = {
    .name = FDG_TYPE_NAME("IntegrationSpecs"),
    .basicsize = sizeof(integration_specs_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){
            {Py_tp_doc, (void *)integration_specs_docstring},
            {Py_tp_getset, integration_rule_getset},
            {Py_tp_new, integration_specs_new},
            {Py_tp_repr, (reprfunc)integration_rule_repr},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_richcompare, integration_spec_richcompare},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "nodes",
                     .ml_meth = (void *)integration_specs_nodes,
                     .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     .ml_doc = integration_specs_nodes_docstring,
                 },
                 {
                     .ml_name = "weights",
                     .ml_meth = (void *)integration_specs_weights,
                     .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     .ml_doc = integration_specs_weights_docstring,
                 },
                 {},
             }},
            {},
        },
};

static PyObject *integration_space_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_SetString(PyExc_TypeError, "Constructor takes no keyword arguments.");
        return NULL;
    }
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Expected at least one argument.");
        return NULL;
    }
    const interplib_module_state_t *const state = interplib_get_module_state(type);
    if (!state)
        return NULL;

    // Check all args are integration specs
    for (unsigned i = 0; i < nargs; ++i)
    {
        PyObject *const arg = PyTuple_GET_ITEM(args, i);
        if (!PyObject_TypeCheck(arg, state->integration_spec_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_spec_type->tp_name,
                         Py_TYPE(arg)->tp_name);
            return NULL;
        }
    }
    integration_space_object *self = (integration_space_object *)type->tp_alloc(type, nargs);
    if (!self)
        return NULL;
    for (unsigned i = 0; i < nargs; ++i)
    {
        const integration_specs_object *const spec = (integration_specs_object *)PyTuple_GET_ITEM(args, i);
        self->specs[i] = spec->spec;
    }

    return (PyObject *)self;
}

static PyObject *integration_space_get_dimension(PyObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t(Py_SIZE(self));
}

static PyObject *integration_space_get_integration_specs(PyObject *self, void *Py_UNUSED(closure))
{
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    PyObject *const out = PyTuple_New(Py_SIZE(self));
    if (!out)
        return NULL;
    const integration_space_object *const this = (integration_space_object *)self;
    for (unsigned i = 0; i < Py_SIZE(self); ++i)
    {
        integration_specs_object *const spec =
            integration_specs_object_create(state->integration_spec_type, this->specs[i]);
        if (!spec)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, (PyObject *)spec);
    }
    return out;
}

static PyObject *integration_space_get_orders(PyObject *self, void *Py_UNUSED(closure))
{
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    PyObject *const out = PyTuple_New(Py_SIZE(self));
    if (!out)
        return NULL;
    const integration_space_object *const this = (integration_space_object *)self;
    for (unsigned i = 0; i < Py_SIZE(self); ++i)
    {
        PyObject *const val = PyLong_FromLong(this->specs[i].order);
        if (!val)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, val);
    }
    return out;
}

static int ensure_integration_space_state(PyObject *self, PyTypeObject *defining_class,
                                          integration_space_object **p_this, const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->integration_space_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_space_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_this = (integration_space_object *)self;
    *p_state = state;
    return 0;
}

PyDoc_STRVAR(integration_space_weights_docstring,
             "weights(registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, /) -> "
             "numpy.typing.NDArray[numpy.double]\n"
             "Get the integration weights of the space.\n"
             "\n"
             "registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rules.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of integration weights.\n");

static PyObject *integration_space_weights(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    integration_space_object *this;
    if (ensure_integration_space_state(self, defining_class, &this, &state) < 0)
        return NULL;
    if (nargs > 1 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected at most one positional argument.");
        return NULL;
    }
    if (nargs && !PyObject_TypeCheck(args[0], state->integration_registry_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_registry_type->tp_name,
                     Py_TYPE(args[0]));
        return NULL;
    }
    const integration_registry_object *const registry_object =
        (integration_registry_object *)(nargs ? args[0] : state->registry_integration);

    const Py_ssize_t ndim = Py_SIZE(self);
    const size_t mem_dims = sizeof(npy_intp) * ndim;
    const size_t mem_iter = multidim_iterator_needed_memory(ndim);
    npy_intp *const dims_out = PyMem_Malloc(mem_dims > mem_iter ? mem_dims : mem_iter);
    if (!dims_out)
        return NULL;

    for (unsigned i = 0; i < ndim; ++i)
    {
        dims_out[i] = this->specs[i].order + 1;
    }

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndim, dims_out, NPY_DOUBLE);
    if (!out)
    {
        PyMem_Free(dims_out);
        return NULL;
    }

    npy_double *const p_out = PyArray_DATA(out);
    multidim_iterator_t *const iter = (multidim_iterator_t *)dims_out;
    for (unsigned i = 0; i < ndim; ++i)
    {
        multidim_iterator_init_dim(iter, i, this->specs[i].order + 1);
    }
    interp_result_t res = FDG_SUCCESS;
    Py_BEGIN_ALLOW_THREADS;

    p_out[0] = 1.0;
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        const integration_rule_t *rule;
        res = integration_rule_registry_get_rule(registry_object->registry, this->specs[idim], &rule);
        if (res != FDG_SUCCESS)
            break;
        multidim_iterator_set_to_start(iter);
        const unsigned npts = this->specs[idim].order + 1;
        while (!multidim_iterator_is_at_end(iter))
        {
            const double prev_v = p_out[multidim_iterator_get_flat_index(iter)];
            for (unsigned ipt = 0; ipt < npts; ++ipt)
            {
                ASSERT(!multidim_iterator_is_at_end(iter), "Iterator should not be at end at this point");
                const size_t idx = multidim_iterator_get_flat_index(iter);
                p_out[idx] = prev_v * integration_rule_weights_const(rule)[ipt];
                multidim_iterator_advance(iter, idim, 1);
            }
        }
    }

    Py_END_ALLOW_THREADS;
    PyMem_Free(dims_out);

    if (res != FDG_SUCCESS)
    {
        Py_DECREF(out);
        PyErr_Format(PyExc_RuntimeError, "Failed to retrieve integration rule: %s (%s).", interp_error_str(res),
                     interp_error_msg(res));
        return NULL;
    }

    return (PyObject *)out;
}

PyDoc_STRVAR(
    integration_space_nodes_docstring,
    "nodes(registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, /) -> numpy.typing.NDArray[numpy.double]\n"
    "Get the integration nodes of the space.\n"
    "\n"
    "registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
    "    Registry used to retrieve the integration rules.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Array of integration nodes.\n");

static PyObject *integration_space_nodes(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames)
{

    const interplib_module_state_t *state;
    integration_space_object *this;
    if (ensure_integration_space_state(self, defining_class, &this, &state) < 0)
        return NULL;
    if (nargs > 1 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected at most one positional argument.");
        return NULL;
    }
    if (nargs && !PyObject_TypeCheck(args[0], state->integration_registry_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_registry_type->tp_name,
                     Py_TYPE(args[0]));
        return NULL;
    }
    const integration_registry_object *const registry_object =
        (integration_registry_object *)(nargs ? args[0] : state->registry_integration);

    const Py_ssize_t ndim = Py_SIZE(self);
    const size_t mem_dims = sizeof(npy_intp) * (ndim + 1);
    const size_t mem_iter = multidim_iterator_needed_memory(ndim);
    npy_intp *const dims_out = PyMem_Malloc(mem_dims > mem_iter ? mem_dims : mem_iter);
    if (!dims_out)
        return NULL;

    dims_out[0] = ndim;
    size_t size_per_dim = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        const unsigned size_dim = this->specs[i].order + 1;
        dims_out[i + 1] = size_dim;
        size_per_dim *= size_dim;
    }

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndim + 1, dims_out, NPY_DOUBLE);
    if (!out)
    {
        PyMem_Free(dims_out);
        return NULL;
    }

    npy_double *const p_out = PyArray_DATA(out);
    multidim_iterator_t *const iter = (multidim_iterator_t *)dims_out;
    for (unsigned i = 0; i < ndim; ++i)
    {
        multidim_iterator_init_dim(iter, i, this->specs[i].order + 1);
    }
    interp_result_t res = FDG_SUCCESS;
    Py_BEGIN_ALLOW_THREADS;

    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        npy_double *const ptr = p_out + size_per_dim * idim;
        const integration_rule_t *rule;
        res = integration_rule_registry_get_rule(registry_object->registry, this->specs[idim], &rule);
        if (res != FDG_SUCCESS)
            break;
        multidim_iterator_set_to_start(iter);

        while (!multidim_iterator_is_at_end(iter))
        {
            ptr[multidim_iterator_get_flat_index(iter)] =
                integration_rule_nodes_const(rule)[multidim_iterator_get_offset(iter, idim)];
            multidim_iterator_advance(iter, ndim - 1, 1);
        }
    }

    Py_END_ALLOW_THREADS;
    PyMem_Free(dims_out);

    if (res != FDG_SUCCESS)
    {
        Py_DECREF(out);
        PyErr_Format(PyExc_RuntimeError, "Failed to retrieve integration rule: %s (%s).", interp_error_str(res),
                     interp_error_msg(res));
        return NULL;
    }

    return (PyObject *)out;
}

PyDoc_STRVAR(integration_space_docstring,
             "IntegrationSpace(*specs : IntegrationSpecs, /)\n"
             "Integration space defined with integration rules.\n"
             "\n"
             "Integration space defined by tensor product of integration rules in each\n"
             "dimension. Integration rule for each dimension are defined by an\n"
             "IntegrationSpecs object.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "*integration_specs : IntegrationSpecs\n"
             "    Integration specifications for each dimension of the integration space.\n");

PyType_Spec integration_space_type_spec = {
    .name = FDG_TYPE_NAME("IntegrationSpace"),
    .basicsize = sizeof(integration_space_object),
    .itemsize = sizeof(integration_spec_t),
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    .slots = (PyType_Slot[]){
        {Py_tp_new, integration_space_new},
        {Py_tp_traverse, heap_type_traverse_type},
        {
            Py_tp_getset,
            (PyGetSetDef[]){
                {
                    .name = "dimension",
                    .get = integration_space_get_dimension,
                    .doc = "int : Number of dimensions in the integration space.",
                },
                {
                    .name = "integration_specs",
                    .get = integration_space_get_integration_specs,
                    .doc =
                        "tuple[IntegrationSpecs, ...] : Integration specifications that define the integration space.",
                },
                {
                    .name = "orders",
                    .get = integration_space_get_orders,
                    .doc = "tuple[int, ...] : Orders of the integration rules.",
                },
                {},
            },
        },
        {Py_tp_methods,
         (PyMethodDef[]){
             {
                 .ml_name = "weights",
                 .ml_meth = (void *)integration_space_weights,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = (void *)integration_space_weights_docstring,
             },
             {
                 .ml_name = "nodes",
                 .ml_meth = (void *)integration_space_nodes,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = (void *)integration_space_nodes_docstring,
             },
             {},
         }},
        {Py_tp_doc, (void *)integration_space_docstring},
        {},
    }};

multidim_iterator_t *integration_specs_iterator(const unsigned n_specs,
                                                const integration_spec_t FDG_ARRAY_ARG(specs, static n_specs))
{
    multidim_iterator_t *const iter = PyMem_Malloc(multidim_iterator_needed_memory(n_specs));
    if (!iter)
        return NULL;

    for (unsigned i = 0; i < n_specs; ++i)
    {
        multidim_iterator_init_dim(iter, i, specs[i].order + 1);
    }

    return iter;
}

multidim_iterator_t *integration_space_iterator(const integration_space_object *space)
{
    return integration_specs_iterator(Py_SIZE(space), space->specs);
}

const integration_rule_t **python_integration_rules_get(const unsigned n_rules,
                                                        const integration_spec_t specs[const static n_rules],
                                                        integration_rule_registry_t *registry)
{
    const integration_rule_t **const array = PyMem_Malloc(n_rules * sizeof(*array));
    if (!array)
        return NULL;
    for (unsigned irule = 0; irule < n_rules; ++irule)
    {
        const interp_result_t res = integration_rule_registry_get_rule(registry, specs[irule], array + irule);
        if (res != FDG_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Failed to retrieve integration rule: %s (%s).", interp_error_str(res),
                         interp_error_msg(res));
            for (unsigned i = 0; i < irule; ++i)
            {
                integration_rule_registry_release_rule(registry, array[i]);
            }
            PyMem_Free(array);
            return NULL;
        }
    }
    return array;
}

void python_integration_rules_release(const unsigned n_rules, const integration_rule_t *rules[static n_rules],
                                      integration_rule_registry_t *registry)
{
    for (unsigned irule = 0; irule < n_rules; ++irule)
    {
        integration_rule_registry_release_rule(registry, rules[irule]);
        rules[irule] = NULL;
    }
    PyMem_Free(rules);
}
