//
// Created by jan on 2025-09-11.
//

#include "basis_objects.h"
#include "integration_objects.h"

#include <numpy/ndarrayobject.h>
#include <string.h>

static basis_set_type_t get_basis_type(const char *str)
{
    if (strcmp(str, "lagrange-uniform") == 0)
    {
        return BASIS_LAGRANGE_UNIFORM;
    }

    if (strcmp(str, "lagrange-gauss") == 0)
    {
        return BASIS_LAGRANGE_GAUSS;
    }

    if (strcmp(str, "lagrange-gauss-lobatto") == 0)
    {
        return BASIS_LAGRANGE_GAUSS_LOBATTO;
    }

    if (strcmp(str, "lagrange-chebyshev-gauss") == 0)
    {
        return BASIS_LAGRANGE_CHEBYSHEV_GAUSS;
    }

    if (strcmp(str, "legendre") == 0)
    {
        return BASIS_LEGENDRE;
    }

    if (strcmp(str, "bernstein") == 0)
    {
        return BASIS_BERNSTEIN;
    }

    return BASIS_INVALID;
}

static const char *basis_type_string(const basis_set_type_t type)
{
    switch (type)
    {
    case BASIS_LAGRANGE_UNIFORM:
        return "lagrange-uniform";
    case BASIS_LAGRANGE_GAUSS:
        return "lagrange-gauss";
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        return "lagrange-gauss-lobatto";
    case BASIS_LEGENDRE:
        return "legendre";
    case BASIS_BERNSTEIN:
        return "bernstein";
    default:
        return "invalid";
    }
}

static PyObject *basis_registry_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 0 || (kwds && PyDict_Size(kwds) != 0))
    {
        PyErr_SetString(PyExc_TypeError, "BasisRegistry takes no arguments.");
        return NULL;
    }

    return (PyObject *)basis_registry_object_create(subtype);
}

static void basis_registry_dealloc(basis_registry_object *self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *const type = Py_TYPE(self);
    if (self->registry)
    {
        basis_set_registry_destroy(self->registry);
        self->registry = NULL;
    }
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static int ensure_basis_registry_and_state(PyObject *self, PyTypeObject *defining_class,
                                           const interplib_module_state_t **p_state, basis_registry_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->basis_registry_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->basis_registry_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    *p_state = state;
    *p_this = (basis_registry_object *)self;
    return 0;
}

PyDoc_STRVAR(basis_registry_usage_docstring,
             "usage() -> tuple[tuple[BasisSpecs, IntegrationSpecs], ...]\n"
             "Return the basis-integration pairs that are held by the registry.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "tuple of (BasisSpecs, IntegrationSpecs)\n"
             "    Tuple of basis-integration specifications pair for each of basis set\n"
             "    held in the registry.\n");

static PyObject *basis_registry_usage(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                      const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    basis_registry_object *this;
    if (ensure_basis_registry_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "usage() takes no arguments.");
        return NULL;
    }

    unsigned usage = basis_set_registry_get_sets(this->registry, 0, (basis_spec_t[]){}, (integration_spec_t[]){});

    integration_spec_t *const specs = PyMem_Malloc(usage * sizeof(*specs));
    if (!specs)
        return NULL;
    basis_spec_t *const basis_specs = PyMem_Malloc(usage * sizeof(*basis_specs));
    if (!basis_specs)
    {
        PyMem_Free(specs);
        return NULL;
    }
    unsigned const new_usage = basis_set_registry_get_sets(this->registry, usage, basis_specs, specs);
    usage = new_usage < usage ? new_usage : usage;

    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(usage);
    if (!out)
    {
        PyMem_Free(basis_specs);
        PyMem_Free(specs);
        return NULL;
    }

    for (unsigned i = 0; i < usage; ++i)
    {
        integration_specs_object *ir = NULL;
        basis_specs_object *br = NULL;
        PyObject *packed = NULL;

        if ((ir = integration_specs_object_create(state->integration_spec_type, specs[i])) == NULL ||
            (br = basis_specs_object_create(state->basis_spec_type, basis_specs[i])) == NULL ||
            (packed = PyTuple_New(2)) == NULL)
        {
            Py_XDECREF(ir);
            Py_XDECREF(br);
            PyMem_Free(basis_specs);
            PyMem_Free(specs);
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(packed, 0, (PyObject *)br);
        PyTuple_SET_ITEM(packed, 1, (PyObject *)ir);
        PyTuple_SET_ITEM(out, i, packed);
    }

    PyMem_Free(basis_specs);
    PyMem_Free(specs);
    return (PyObject *)out;
}

PyDoc_STRVAR(basis_registry_clear_docstring,
             "clear() -> None\nRelease all held basis sets to reduce the memory usage.\n");

static PyObject *basis_registry_clear(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                      const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    basis_registry_object *this;
    if (ensure_basis_registry_and_state(self, defining_class, &state, &this) < 0)
        return NULL;
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "clear() takes no arguments.");
        return NULL;
    }

    basis_set_registry_release_unused_basis_sets(this->registry);

    Py_RETURN_NONE;
}

PyDoc_STRVAR(basis_registry_docstring, "Registry for basis sets.\n"
                                       "\n"
                                       "This registry contains all available basis sets and caches them for efficient\n"
                                       "retrieval.\n");

PyType_Spec basis_registry_type_specs = {
    .name = FDG_TYPE_NAME("BasisRegistry"),
    .basicsize = sizeof(basis_registry_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, basis_registry_new},
            {Py_tp_dealloc, basis_registry_dealloc},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_doc, (void *)basis_registry_docstring},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     "usage",
                     (void *)basis_registry_usage,
                     METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     basis_registry_usage_docstring,
                 },
                 {
                     "clear",
                     (void *)basis_registry_clear,
                     METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     basis_registry_clear_docstring,
                 },
                 {},
             }},
            {},
        },

};

/* __new__ */
static PyObject *basis_specs_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const char *basis_type_str;
    int order;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "si", (char *[]){"", "", NULL}, &basis_type_str, &order))
    {
        return NULL;
    }

    basis_set_type_t const basis_type = get_basis_type(basis_type_str);
    if (basis_type == BASIS_INVALID)
    {
        PyErr_Format(PyExc_ValueError, "Unknown basis type string: %s", basis_type_str);
        return NULL;
    }

    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must be positive, but was given as %i.", order);
        return NULL;
    }

    basis_specs_object *const self = (basis_specs_object *)subtype->tp_alloc(subtype, 0);
    if (!self)
        return NULL;
    self->spec = (basis_spec_t){.order = order, .type = basis_type};

    return (PyObject *)self;
}

/* Properties */

static PyObject *basis_specs_get_order(const basis_specs_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->spec.order);
}

static PyObject *basis_specs_get_type(const basis_specs_object *self, void *Py_UNUSED(closure))
{
    return PyUnicode_FromString(basis_type_string(self->spec.type));
}

/* Get-set table */
static PyGetSetDef basis_getset[] = {
    {
        "order",
        (getter)basis_specs_get_order,
        NULL,
        "int : Order of the basis set.",
        NULL,
    },
    {
        "type",
        (getter)basis_specs_get_type,
        NULL,
        "_BasisTypeHint : Type of the basis used for the set.",
        NULL,
    },
    {},
};

PyDoc_STRVAR(basis_specs_docstring, "BasisSpecs(basis_type: fdg.enum_type.BasisType, order: int)\n"
                                    "Type that describes a set of basis functions.\n"
                                    "\n"
                                    "Parameters\n"
                                    "----------\n"
                                    "basis_type : fdg.enum_type.BasisType\n"
                                    "    Type of the basis used for the set.\n"
                                    "\n"
                                    "order : int\n"
                                    "    Order of the basis in the set.\n");

static int ensure_basis_specs_and_state(PyObject *self, PyTypeObject *defining_class,
                                        const interplib_module_state_t **p_state, basis_specs_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->basis_spec_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->basis_spec_type->tp_name, Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_this = (basis_specs_object *)self;
    *p_state = state;
    return 0;
}

PyDoc_STRVAR(basis_specs_values_docstring,
             "values(x: numpy.typing.ArrayLike, /) -> numpy.typing.NDArray[numpy.double]\n"
             "Evaluate basis functions at given locations.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "x : array_like\n"
             "    Locations where the basis functions should be evaluated.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of basis function values at the specified locations.\n"
             "    It has one more dimension than ``x``, with the last dimension\n"
             "    corresponding to the basis function index.\n");

static PyObject *basis_specs_values(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                    const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    basis_specs_object *this;
    if (ensure_basis_specs_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    if (nargs != 1 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "values() takes exactly one positional-only argument.");
        return NULL;
    }

    PyArrayObject *const x = (PyArrayObject *)args[0];
    const npy_intp dummy[] = {0};
    if (check_input_array(x, 0, dummy, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "x") < 0)
    {
        return NULL;
    }

    const unsigned ndims = PyArray_NDIM(x);
    const npy_intp *const dims = PyArray_DIMS(x);
    const npy_double *const x_data = PyArray_DATA(x);

    npy_intp *const out_dims = PyMem_Malloc((ndims + 1) * sizeof(*out_dims));
    if (!out_dims)
    {
        return NULL;
    }

    unsigned cnt = 1;
    for (unsigned i = 0; i < ndims; ++i)
    {
        out_dims[i] = dims[i];
        cnt *= dims[i];
    }

    out_dims[ndims] = this->spec.order + 1;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndims + 1, out_dims, NPY_DOUBLE);
    PyMem_Free(out_dims);

    if (!out)
    {
        return NULL;
    }

    double *const work = PyMem_Malloc((this->spec.order + 1) * sizeof(*work));
    if (!work)
    {
        Py_DECREF(out);
        return NULL;
    }

    basis_compute_at_point_prepare(this->spec.type, this->spec.order, work);
    basis_compute_at_point_values(this->spec.type, this->spec.order, cnt, x_data, PyArray_DATA(out), work);

    PyMem_Free(work);

    return (PyObject *)out;
}

PyDoc_STRVAR(basis_specs_derivatives_docstring,
             "derivatives(x: numpy.typing.ArrayLike, /) -> numpy.typing.NDArray[numpy.double]\n"
             "Evaluate basis function derivatives at given locations.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "x : array_like\n"
             "    Locations where the basis function derivatives should be evaluated.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of basis function derivatives at the specified locations.\n"
             "    It has one more dimension than ``x``, with the last dimension\n"
             "    corresponding to the basis function index.\n");

static PyObject *basis_specs_derivatives(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    basis_specs_object *this;
    if (ensure_basis_specs_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    if (nargs != 1 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "values() takes exactly one positional-only argument.");
        return NULL;
    }

    const PyArrayObject *const x = (PyArrayObject *)args[0];
    const npy_intp dummy[] = {0};
    if (check_input_array(x, 0, dummy, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "x") < 0)
    {
        return NULL;
    }

    const unsigned ndims = PyArray_NDIM(x);
    const npy_intp *const dims = PyArray_DIMS(x);
    const npy_double *const x_data = PyArray_DATA(x);

    npy_intp *const out_dims = PyMem_Malloc((ndims + 1) * sizeof(*out_dims));
    if (!out_dims)
    {
        return NULL;
    }

    unsigned cnt = 1;
    for (unsigned i = 0; i < ndims; ++i)
    {
        out_dims[i] = dims[i];
        cnt *= dims[i];
    }

    out_dims[ndims] = this->spec.order + 1;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndims + 1, out_dims, NPY_DOUBLE);
    PyMem_Free(out_dims);

    if (!out)
    {
        return NULL;
    }

    double *const work = PyMem_Malloc((this->spec.order + 1) * sizeof(*work));
    if (!work)
    {
        Py_DECREF(out);
        return NULL;
    }

    basis_compute_at_point_prepare(this->spec.type, this->spec.order, work);
    CPYUTL_ASSERT(PyArray_SIZE(out) == (npy_intp)(this->spec.order + 1) * cnt,
                  "Output array is not as large as expected!");
    basis_compute_at_point_derivatives(this->spec.type, this->spec.order, cnt, x_data, PyArray_DATA(out), work);

    PyMem_Free(work);

    return (PyObject *)out;
}

static PyObject *basis_specs_richcompare(PyObject *self, PyObject *other, const int op)
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

    if (!PyObject_TypeCheck(other, state->basis_spec_type) || !PyObject_TypeCheck(self, state->basis_spec_type) ||
        (op != Py_EQ && op != Py_NE))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }

    const basis_specs_object *const this = (basis_specs_object *)self;
    const basis_specs_object *const that = (basis_specs_object *)other;
    int equal = this->spec.order == that->spec.order && this->spec.type == that->spec.type;
    if (op == Py_NE)
    {
        equal = !equal;
    }
    return PyBool_FromLong(equal);
}

/* Spec for the heap type */
PyType_Spec basis_specs_type_spec = {
    .name = FDG_TYPE_NAME("BasisSpecs"),
    .basicsize = sizeof(basis_specs_object),
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, (void *)basis_specs_new},
            {Py_tp_getset, (void *)basis_getset},
            {Py_tp_doc, (void *)basis_specs_docstring},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "values",
                     .ml_meth = (void *)basis_specs_values,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = (void *)basis_specs_values_docstring,
                 },
                 {
                     .ml_name = "derivatives",
                     .ml_meth = (void *)basis_specs_derivatives,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = basis_specs_derivatives_docstring,
                 },
                 {},
             }},
            {Py_tp_richcompare, basis_specs_richcompare},
            {},
        },
};

basis_registry_object *basis_registry_object_create(PyTypeObject *type)
{
    basis_registry_object *const this = (basis_registry_object *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;
    this->registry = NULL;
    const interp_result_t res = basis_set_registry_create(&this->registry, 1, &SYSTEM_ALLOCATOR);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not initialize basis set registry: %s (%s)", interp_error_str(res),
                     interp_error_msg(res));
        Py_DECREF(this);
        return NULL;
    }
    return this;
}
basis_specs_object *basis_specs_object_create(PyTypeObject *type, const basis_spec_t spec)
{
    basis_specs_object *const this = (basis_specs_object *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;
    this->spec = spec;
    return this;
}

const basis_set_t **python_basis_sets_get(const unsigned n_basis, const basis_spec_t specs[const static n_basis],
                                          const integration_rule_t *rules[const static n_basis],
                                          basis_set_registry_t *registry)
{
    const basis_set_t **const array = PyMem_Malloc(n_basis * sizeof(*array));
    if (!array)
        return NULL;
    for (unsigned ibasis = 0; ibasis < n_basis; ++ibasis)
    {
        const interp_result_t res =
            basis_set_registry_get_basis_set(registry, array + ibasis, rules[ibasis], specs[ibasis]);
        if (res != FDG_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Failed to retrieve basis set: %s (%s).", interp_error_str(res),
                         interp_error_msg(res));
            for (unsigned i = 0; i < ibasis; ++i)
            {
                basis_set_registry_release_basis_set(registry, array[i]);
            }
            PyMem_Free(array);
            return NULL;
        }
    }
    return array;
}

void python_basis_sets_release(const unsigned n_basis, const basis_set_t *sets[static n_basis],
                               basis_set_registry_t *registry)
{
    for (unsigned ibasis = 0; ibasis < n_basis; ++ibasis)
    {
        basis_set_registry_release_basis_set(registry, sets[ibasis]);
        sets[ibasis] = NULL;
    }
    PyMem_Free(sets);
}
multidim_iterator_t *python_basis_iterator(const unsigned n_basis, const basis_spec_t specs[const static n_basis])
{
    multidim_iterator_t *const iter = PyMem_Malloc(multidim_iterator_needed_memory(n_basis));
    if (!iter)
        return NULL;

    for (unsigned ibasis = 0; ibasis < n_basis; ++ibasis)
    {
        multidim_iterator_init_dim(iter, ibasis, specs[ibasis].order + 1);
    }

    return iter;
}
