#include "function_space_objects.h"
#include "basis_objects.h"
#include "integration_objects.h"
#include <cutl/iterators/multidim_iteration.h>

function_space_object *function_space_object_create(PyTypeObject *type, const unsigned n_basis,
                                                    const basis_spec_t FDG_ARRAY_ARG(specs, static n_basis))
{
    function_space_object *const this = (function_space_object *)type->tp_alloc(type, n_basis);
    if (!this)
        return NULL;
    memcpy(this->specs, specs, n_basis * sizeof(basis_spec_t));
    return this;
}

static PyObject *function_space_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_SetString(PyExc_TypeError, "Constructor takes no keyword arguments.");
        return NULL;
    }

    const unsigned n = PyTuple_GET_SIZE(args);
    if (n == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Constructor requires at least one argument.");
        return NULL;
    }

    const interplib_module_state_t *state = interplib_get_module_state(type);
    if (!state)
        return NULL;

    // Check we got basis sets
    for (unsigned i = 0; i < n; ++i)
    {
        const PyObject *const val = PyTuple_GET_ITEM(args, i);
        if (!PyObject_TypeCheck(val, state->basis_spec_type))
        {
            PyErr_Format(PyExc_TypeError, "Argument %i was not a BasisSpec, but was instead %R.", i, val);
            return NULL;
        }
    }

    // Allocate the memory
    function_space_object *const this = (function_space_object *)type->tp_alloc(type, n);
    if (!this)
        return NULL;

    // Write the basis
    for (unsigned i = 0; i < n; ++i)
    {
        const basis_specs_object *const val = (basis_specs_object *)PyTuple_GET_ITEM(args, i);
        this->specs[i] = val->spec;
    }

    return (PyObject *)this;
}

static PyObject *function_space_get_dimensions(PyObject *self, void *Py_UNUSED(closure))
{
    const function_space_object *const this = (function_space_object *)self;
    return PyLong_FromSsize_t(Py_SIZE(this));
}

static PyObject *function_space_get_specs(PyObject *self, void *Py_UNUSED(closure))
{
    const function_space_object *const this = (function_space_object *)self;
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(Py_SIZE(this));
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        basis_specs_object *const val = basis_specs_object_create(state->basis_spec_type, this->specs[i]);
        if (!val)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, val);
    }
    return (PyObject *)out;
}

static PyObject *function_space_get_orders(PyObject *self, void *Py_UNUSED(closure))
{
    const function_space_object *const this = (function_space_object *)self;
    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(Py_SIZE(this));
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        PyObject *const dim = PyLong_FromLong(this->specs[i].order);
        if (!dim)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, dim);
    }
    return (PyObject *)out;
}

static int ensure_function_space_state(PyObject *self, PyTypeObject *defining_class, function_space_object **p_this,
                                       const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->function_space_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->function_space_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    *p_this = (function_space_object *)self;
    *p_state = state;
    return 0;
}

PyDoc_STRVAR(function_space_evaluate_docstring,
             "evaluate(*x: numpy.typing.NDArray[numpy.double], out: numpy.typing.NDArray[numpy.double] | None = None) "
             "-> numpy.typing.NDArray[numpy.double]:\n"
             "Evaluate basis functions at given locations.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "*x : array\n"
             "    Coordinates where the basis functions should be evaluated.\n"
             "    Each array corresponds to a dimension in the function space.\n"
             "out : array, optional\n"
             "    Array where the results should be written to. If not given, a new one\n"
             "    will be created and returned. It should have the same shape as ``x``,\n"
             "    but with an extra dimension added, the length of which is the total\n"
             "    number of basis functions in the function space.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of basis function values at the specified locations.\n");

static PyObject *function_space_evaluate(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames)
{
    function_space_object *this;
    const interplib_module_state_t *state;
    if (ensure_function_space_state(self, defining_class, &this, &state) < 0)
        return NULL;

    PyArrayObject *out = NULL;
    if (kwnames)
    {
        if (PyTuple_GET_SIZE(kwnames) != 1)
        {
            PyErr_SetString(PyExc_TypeError, "evaluate() takes exactly one keyword argument.");
            return NULL;
        }
        PyObject *const kwname = PyTuple_GET_ITEM(kwnames, 0);
        const char *kwrd = PyUnicode_AsUTF8(kwname);
        if (!kwrd)
            return NULL;
        if (strcmp(kwrd, "out") != 0)
        {
            PyErr_Format(PyExc_TypeError, "evaluate() got an unexpected keyword argument '%s'.", kwrd);
            return NULL;
        }
        out = (PyArrayObject *)args[nargs];
    }
    const unsigned n_basis_dims = Py_SIZE(this);
    CPYUTL_ASSERT(n_basis_dims > 0, "Function space should have at least one dimension.");

    if (nargs != n_basis_dims)
    {
        PyErr_Format(PyExc_TypeError,
                     "evaluate() takes exactly the same number of positional arguments as the dimension count (%u), "
                     "but %u were given.",
                     n_basis_dims, (unsigned)nargs);
        return NULL;
    }

    // Check all input arrays have the same shape, correct dtype, and the same flags
    npy_intp n_dim_in = 0;
    const npy_intp *p_dim_in = NULL;
    for (unsigned i = 0; i < n_basis_dims; ++i)
    {

        const PyArrayObject *const in = (PyArrayObject *)args[i];
        if (p_dim_in == NULL)
        {
            p_dim_in = PyArray_DIMS(in);
            n_dim_in = PyArray_NDIM(in);
        }
        if (check_input_array(in, n_dim_in, p_dim_in, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "input") <
            0)
        {
            raise_exception_from_current(
                PyExc_ValueError,
                "All input arrays must have the exact same shape and have the correct data type and flags.");
            return NULL;
        }
    }

    const npy_intp n_dim_out = n_dim_in + n_basis_dims;
    npy_intp *p_dim_out = PyMem_Malloc(n_dim_out * sizeof(*p_dim_out));
    if (!p_dim_out)
    {
        return NULL;
    }
    CPYUTL_ASSERT(p_dim_in, "Input dims were not collected.");
    npy_uintp size_in = 1, max_size_basis = 1;
    for (npy_intp i = 0; i < n_dim_in; ++i)
    {
        const npy_intp dim = p_dim_in[i];
        size_in *= dim;
        p_dim_out[i] = dim;
    }
    npy_uintp size_out = size_in;
    for (npy_intp i = n_dim_in; i < n_dim_out; ++i)
    {
        const unsigned dim = this->specs[i - n_dim_in].order + 1;
        p_dim_out[i] = dim;
        size_out *= dim;
        // outer_prod_basis_cnt *= dim;
        max_size_basis = dim > max_size_basis ? dim : max_size_basis;
    }

    if (out)
    {
        // Check the output has the correct dimensions if given
        if (check_input_array(out, n_dim_out, p_dim_out, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                              "out") < 0)
            return NULL;
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(n_dim_out, p_dim_out, NPY_DOUBLE);
        if (!out)
        {
            PyMem_Free(p_dim_out);
            return NULL;
        }
    }
    PyMem_Free(p_dim_out);
    npy_double *const p_out = (npy_double *)PyArray_DATA(out);
    CPYUTL_ASSERT(size_out == (size_t)PyArray_SIZE(out), "Incorrect output size.");
    const double **const input_ptrs = PyMem_Malloc(n_basis_dims * sizeof(*input_ptrs));
    if (!input_ptrs)
    {
        Py_DECREF(out);
        return NULL;
    }
    for (unsigned i = 0; i < n_basis_dims; ++i)
    {
        input_ptrs[i] = (npy_double *)PyArray_DATA((PyArrayObject *)args[i]);
    }

    Py_BEGIN_ALLOW_THREADS;
    unsigned out_elements, work_elements, tmp_elements;
    size_t iter_size;
    basis_compute_outer_product_basis_required_memory(n_basis_dims, this->specs, size_in, &out_elements, &work_elements,
                                                      &tmp_elements, &iter_size);
    double *const basis_buffer = PyMem_RawMalloc((tmp_elements + work_elements) * sizeof(*basis_buffer));
    multidim_iterator_t *const iterator = PyMem_RawMalloc(iter_size);

    if (basis_buffer && iterator)
    {
        basis_compute_outer_product_basis(n_basis_dims, this->specs, size_in, input_ptrs, p_out,
                                          basis_buffer + tmp_elements, basis_buffer, iterator);
    }
    PyMem_RawFree(iterator);
    PyMem_RawFree(basis_buffer);
    Py_END_ALLOW_THREADS;
    PyMem_Free(input_ptrs);
    return (PyObject *)out;
}

PyDoc_STRVAR(
    function_space_values_at_integration_nodes_docstring,
    "values_at_integration_nodes(integration: IntegrationSpace, /, *, integration_registry: IntegrationRegistry = "
    "DEFAULT_INTEGRATION_REGISTRY, basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY) -> "
    "numpy.typing.NDArray[numpy.double]\n"
    "Return values of basis at integration points.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "integration : IntegrationSpace\n"
    "    Integration space, the nodes of which are used to evaluate basis at.\n"
    "\n"
    "integration_registry : IntegrationRegistry, defaul: DEFAULT_INTEGRATION_REGISTRY\n"
    "    Registry used to obtain the integration rules from.\n"
    "\n"
    "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
    "    Registry used to look up basis values.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Array of basis function values at the integration points locations.\n");

static PyObject *function_space_values_at_integration_nodes(PyObject *self, PyTypeObject *defining_class,
                                                            PyObject *const *args, const Py_ssize_t nargs,
                                                            const PyObject *kwnames)
{
    function_space_object *this;
    const interplib_module_state_t *state;
    if (ensure_function_space_state(self, defining_class, &this, &state) < 0)
        return NULL;
    integration_space_object *integration_space;
    integration_registry_object *integration_registry = (integration_registry_object *)state->registry_integration;
    basis_registry_object *basis_registry = (basis_registry_object *)state->registry_basis;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_space,
                    .type_check = state->integration_space_type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_registry,
                    .type_check = state->integration_registry_type,
                    .kwname = "integration_registry",
                    .kw_only = 1,
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &basis_registry,
                    .type_check = state->basis_registry_type,
                    .kwname = "basis_registry",
                    .kw_only = 1,
                    .optional = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned ndim = Py_SIZE(this);
    if (ndim != Py_SIZE(integration_space))
    {
        PyErr_Format(PyExc_ValueError,
                     "Function space and integration space must have the same number of dimensions (function space had "
                     "%u, but integration space had %u).",
                     ndim, (unsigned)Py_SIZE(integration_space));
        return NULL;
    }

    npy_intp *const p_dim_out = PyMem_Malloc(2 * ndim * sizeof(*p_dim_out));
    if (!p_dim_out)
    {
        return NULL;
    }
    // unsigned total_nodes_cnt = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {

        const unsigned n_nodes = integration_space->specs[i].order + 1;
        p_dim_out[i] = n_nodes;
        // total_nodes_cnt *= n_nodes;
    }
    unsigned total_basis_cnt = 1;
    for (unsigned i = 0; i < ndim; ++i)
    {
        const unsigned n_basis = this->specs[i].order + 1;
        p_dim_out[i + ndim] = n_basis;
        total_basis_cnt *= n_basis;
    }

    // Create the output array
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2 * ndim, p_dim_out, NPY_DOUBLE);
    PyMem_Free(p_dim_out);
    if (!out)
    {
        return NULL;
    }

    // Create the iterators for the integration nodes and the basis
    multidim_iterator_t *const iterator_nodes = integration_specs_iterator(ndim, integration_space->specs);
    multidim_iterator_t *const iterator_basis = function_space_iterator(this);
    if (!iterator_nodes || !iterator_basis)
    {
        PyMem_Free(iterator_basis);
        PyMem_Free(iterator_nodes);
        Py_DECREF(out);
        return NULL;
    }

    CPYUTL_ASSERT((size_t)PyArray_SIZE(out) ==
                      multidim_iterator_total_size(iterator_basis) * multidim_iterator_total_size(iterator_nodes),
                  "Incorrect output size.");

    const basis_set_t **const basis_sets = PyMem_Malloc(ndim * sizeof(*basis_sets));
    if (!basis_sets)
    {
        PyMem_Free(iterator_basis);
        PyMem_Free(iterator_nodes);
        Py_DECREF(out);
        return NULL;
    }

    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        const integration_rule_t *int_rule;
        interp_result_t res = integration_rule_registry_get_rule(integration_registry->registry,
                                                                 integration_space->specs[idim], &int_rule);
        if (res == FDG_SUCCESS)
        {
            const basis_set_t *basis;
            res = basis_set_registry_get_basis_set(basis_registry->registry, &basis, int_rule, this->specs[idim]);
            // Release the rule
            (void)integration_rule_registry_release_rule(integration_registry->registry, int_rule);
            basis_sets[idim] = basis;
        }

        if (res != FDG_SUCCESS)
        {
            // Release the basis acquired so far
            for (unsigned jdim = 0; jdim < idim; ++jdim)
            {
                (void)basis_set_registry_release_basis_set(basis_registry->registry, basis_sets[jdim]);
            }
            PyMem_Free(basis_sets);
            PyMem_Free(iterator_basis);
            PyMem_Free(iterator_nodes);
            Py_DECREF(out);
            PyErr_Format(PyExc_ValueError, "Failed to get basis for dimension %u, reason: %s (%s)", idim,
                         interp_error_str(res), interp_error_msg(res));
            return NULL;
        }
    }

    npy_double *const p_out = (npy_double *)PyArray_DATA(out);
    multidim_iterator_set_to_start(iterator_basis);
    while (!multidim_iterator_is_at_end(iterator_basis))
    {
        const size_t basis_idx = multidim_iterator_get_flat_index(iterator_basis);
        npy_double *const ptr = p_out + basis_idx;

        multidim_iterator_set_to_start(iterator_nodes);
        while (!multidim_iterator_is_at_end(iterator_nodes))
        {
            double basis_value = 1;
            for (unsigned idim = 0; idim < ndim; ++idim)
            {
                const double *basis =
                    basis_set_basis_values(basis_sets[idim], multidim_iterator_get_offset(iterator_basis, idim));
                basis_value *= basis[multidim_iterator_get_offset(iterator_nodes, idim)];
            }

            ptr[multidim_iterator_get_flat_index(iterator_nodes) * total_basis_cnt] = basis_value;

            // Advance to the next node
            multidim_iterator_advance(iterator_nodes, ndim - 1, 1);
        }
        // Next basis
        multidim_iterator_advance(iterator_basis, ndim - 1, 1);
    }

    // Release all the basis now
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        (void)basis_set_registry_release_basis_set(basis_registry->registry, basis_sets[idim]);
    }
    PyMem_Free(basis_sets);

    // Release the iterators
    PyMem_Free(iterator_basis);
    PyMem_Free(iterator_nodes);

    // We are done - return from the function
    return (PyObject *)out;
}

PyDoc_STRVAR(function_space_object_lower_order_docstring,
             "lower_order(idim: int) -> FunctionSpace\n"
             "Create a copy of the space with a lowered order in the specified dimension.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idim : int\n"
             "    Index of the dimension to lower the order on.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "FunctionSpace\n"
             "    New function space with a lower order in the specified dimension.\n");

static PyObject *function_space_object_lower_order(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                   const Py_ssize_t nargs, const PyObject *kwnames)
{
    function_space_object *this;
    const interplib_module_state_t *state;
    if (ensure_function_space_state(self, defining_class, &this, &state) < 0)
        return NULL;
    Py_ssize_t idim;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_SSIZE,
                    .p_val = &idim,
                    .kwname = "idim",
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (idim < 0 || (npy_intp)idim >= Py_SIZE(this))
    {
        PyErr_Format(PyExc_ValueError, "Dimension %zd is out of bounds for a function space with %zd dimensions.", idim,
                     Py_SIZE(this));
        return NULL;
    }

    if (this->specs[idim].order == 0)
    {
        PyErr_Format(PyExc_ValueError, "Cannot lower order of dimension %zd as it has order 0.", idim);
        return NULL;
    }

    const unsigned ndim = Py_SIZE(this);
    function_space_object *const new_space =
        function_space_object_create(state->function_space_type, ndim, this->specs);
    if (!new_space)
        return NULL;
    new_space->specs[idim].order -= 1;
    return (PyObject *)new_space;
}

PyDoc_STRVAR(function_space_type_docstring,
             "FunctionSpace(*specs: BasisSpec)\n"
             "Function space defined with basis.\n"
             "\n"
             "Function space defined by tensor product of basis functions in each dimension.\n"
             "Basis for each dimension are defined by a BasisSpecs object.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "*basis_specs : BasisSpecs\n"
             "    Basis specifications for each dimension of the function space.\n");

static int function_space_equal(const function_space_object *const this, const function_space_object *const that)
{
    const unsigned n_basis = Py_SIZE(this);

    if (Py_SIZE(that) != n_basis)
        return 0;

    for (unsigned i = 0; i < n_basis; ++i)
    {
        if (this->specs[i].order != that->specs[i].order || this->specs[i].type != that->specs[i].type)
            return 0;
    }

    return 1;
}

static PyObject *function_space_rich_compare(PyObject *self, PyObject *other, const int op)
{
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        PyErr_Clear();
        if ((state = interplib_get_module_state(Py_TYPE(other))) == NULL)
            return NULL;
    }

    if (!PyObject_TypeCheck(self, state->function_space_type) ||
        !PyObject_TypeCheck(other, state->function_space_type) || (op != Py_EQ && op != Py_NE))
        Py_RETURN_NOTIMPLEMENTED;

    const function_space_object *const this = (function_space_object *)self;
    const function_space_object *const that = (function_space_object *)other;

    const int equal = function_space_equal(this, that);

    return PyBool_FromLong(op == Py_EQ ? equal : !equal);
}

PyType_Spec function_space_type_spec = {
    .name = FDG_TYPE_NAME("FunctionSpace"),
    .basicsize = sizeof(function_space_object),
    .itemsize = sizeof(basis_spec_t),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_new, function_space_new},
        {Py_tp_doc, (void *)function_space_type_docstring},
        {Py_tp_richcompare, function_space_rich_compare},
        {
            Py_tp_getset,
            (PyGetSetDef[]){
                {
                    .name = "dimension",
                    .get = function_space_get_dimensions,
                    .doc = "int:Number of dimensions in the function space.",
                },
                {
                    .name = "basis_specs",
                    .get = function_space_get_specs,
                    .doc = "tuple[BasisSpecs, ...] : Basis specifications that define the function space.",
                },
                {
                    .name = "orders",
                    .get = function_space_get_orders,
                    .doc = "tuple[int, ...] : Orders of the basis functions in the function space.",
                },
                {},
            },
        },
        {Py_tp_methods,
         (PyMethodDef[]){
             {
                 .ml_name = "evaluate",
                 .ml_meth = (void *)function_space_evaluate,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = function_space_evaluate_docstring,
             },
             {
                 .ml_name = "values_at_integration_nodes",
                 .ml_meth = (void *)function_space_values_at_integration_nodes,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = function_space_values_at_integration_nodes_docstring,
             },
             {
                 .ml_name = "lower_order",
                 .ml_meth = (void *)function_space_object_lower_order,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = function_space_object_lower_order_docstring,
             },
             {},
         }},
        {},
    }};

multidim_iterator_t *function_space_iterator(const function_space_object *space)
{
    const unsigned ndims = Py_SIZE(space);
    multidim_iterator_t *const iter = PyMem_Malloc(multidim_iterator_needed_memory(ndims));
    if (!iter)
        return NULL;

    for (unsigned idim = 0; idim < ndims; ++idim)
    {
        multidim_iterator_init_dim(iter, idim, space->specs[idim].order + 1);
    }

    return iter;
}
