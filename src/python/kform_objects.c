#include "kform_objects.h"
#include "covector_basis.h"
#include "cutl/iterators/combination_iterator.h"

static PyObject *kform_spec_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(type);
    if (!state)
        return NULL;

    Py_ssize_t order;
    function_space_object *space;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "nO!", (char *[]){"order", "base_space", NULL}, &order,
                                     state->function_space_type, &space))
        return NULL;

    unsigned const ndim = Py_SIZE(space);
    if (order < 0 || order > ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected order in [0, %u], got %zd.", ndim, order);
        return NULL;
    }

    for (unsigned i = 0; i < ndim; ++i)
    {
        const basis_spec_t *const spec = space->specs + i;
        if (spec->order == 0)
        {
            PyErr_Format(PyExc_ValueError, "Expected order > 0 for dimension %u, got 0.", i);
            return NULL;
        }
    }

    const unsigned component_cnt = combination_total_count(ndim, order);

    kform_spec_object *const self = (kform_spec_object *)type->tp_alloc(type, component_cnt + 1);
    if (!self)
        return NULL;

    combination_iterator_t *const iter = PyMem_Malloc(combination_iterator_required_memory(order));
    if (!iter)
    {
        PyMem_Free(self->component_offsets);
        return NULL;
    }
    self->component_offsets[0] = 0;
    combination_iterator_init(iter, ndim, order);
    unsigned i = 0;
    for (const uint8_t *const basis_components = combination_iterator_current(iter);
         !combination_iterator_is_done(iter); combination_iterator_next(iter), ++i)
    {
        const unsigned ndofs = kform_basis_get_num_dofs(ndim, space->specs, order, basis_components);
        self->component_offsets[i + 1] = self->component_offsets[i] + ndofs;
    }
    PyMem_Free(iter);

    Py_INCREF(space);
    self->order = order;
    self->function_space = space;

    return (PyObject *)self;
}

static void kform_spec_dealloc(kform_spec_object *self)
{
    PyObject_GC_UnTrack(self);
    Py_DECREF(self->function_space);
    self->function_space = NULL;
    self->order = 0;
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyObject *kform_spec_get_order(const kform_spec_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(self->order);
}

static PyObject *kform_spec_get_function_space(const kform_spec_object *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->function_space);
    return (PyObject *)self->function_space;
}

static PyObject *kform_spec_get_ndim(const kform_spec_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(Py_SIZE(self->function_space));
}

static int ensure_kform_specs_and_state(PyObject *self, PyTypeObject *defining_class,
                                        const kform_spec_object **out_spec, const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->kform_specs_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a KFormSpecs object.");
        return -1;
    }
    *p_state = state;
    *out_spec = (const kform_spec_object *)self;
    return 0;
}

static PyObject *kform_spec_get_component_function_space(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         PyObject *kwnames)
{
    const interplib_module_state_t *state;
    const kform_spec_object *this;
    if (ensure_kform_specs_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    Py_ssize_t idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idx, .kwname = "idx"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned n = Py_SIZE(this->function_space);
    const unsigned k = this->order;
    const unsigned component_cnt = combination_total_count(n, k);
    if (idx < 0 || idx >= component_cnt)
    {
        PyErr_Format(PyExc_ValueError, "Expected index in [0, %u), got %zd.", component_cnt, idx);
        return NULL;
    }

    uint8_t *covector_indices;
    basis_spec_t *out_specs;
    void *const mem = cutl_alloc_group(&PYTHON_ALLOCATOR,
                                       (const cutl_alloc_info_t[]){
                                           {.size = k * sizeof(*covector_indices), .p_ptr = (void **)&covector_indices},
                                           {.size = n * sizeof(*out_specs), .p_ptr = (void **)&out_specs},
                                           {},
                                       });
    if (!mem)
        return NULL;

    combination_set_to_index(n, k, covector_indices, idx);
    for (unsigned i = 0, i_covector = 0; i < n; ++i)
    {
        const basis_spec_t *const spec = this->function_space->specs + i;
        if (i_covector < k && covector_indices[i_covector] == k)
        {
            out_specs[i] = (basis_spec_t){.type = spec->type, .order = spec->order - 1};
            i_covector += 1;
        }
        else
        {
            out_specs[i] = *spec;
        }
    }

    function_space_object *const out_space = function_space_object_create(state->function_space_type, n, out_specs);
    cutl_dealloc(&PYTHON_ALLOCATOR, mem);

    return (PyObject *)out_space;
}

static PyObject *kform_spec_get_component_covector_basis(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         PyObject *kwnames)
{
    const interplib_module_state_t *state;
    const kform_spec_object *this;
    if (ensure_kform_specs_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    Py_ssize_t idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idx, .kwname = "idx"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned n = Py_SIZE(this->function_space);
    const unsigned k = this->order;
    const unsigned component_cnt = combination_total_count(n, k);
    if (idx < 0 || idx >= component_cnt)
    {
        PyErr_Format(PyExc_ValueError, "Expected index in [0, %u), got %zd.", component_cnt, idx);
        return NULL;
    }

    uint8_t *covector_indices = PyMem_Malloc(k * sizeof(*covector_indices));
    if (!covector_indices)
        return NULL;

    combination_set_to_index(n, k, covector_indices, idx);
    covector_basis_t basis = {.dimension = n, .sign = 0};
    for (unsigned i_covector = 0; i_covector < k; ++i_covector)
    {
        basis.basis_bits |= (1 << covector_indices[i_covector]);
    }

    covector_basis_object *const covector_basis = covector_basis_object_create(state->covector_basis_type, basis);
    PyMem_Free(covector_indices);

    return (PyObject *)covector_basis;
}

static PyObject *kform_spec_get_component_count(const kform_spec_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(combination_total_count(Py_SIZE(self->function_space), self->order));
}

static PyObject *kform_specs_get_component_dof_counts(const kform_spec_object *self, void *Py_UNUSED(closure))
{

    const unsigned n = Py_SIZE(self->function_space);
    const unsigned k = self->order;
    const npy_intp components = combination_total_count(n, k);
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &components, NPY_INT64);
    if (!out)
        return NULL;
    npy_int64 *const out_counts = PyArray_DATA(out);
    for (unsigned i = 0; i < components; ++i)
    {
        out_counts[i] = (npy_int64)(self->component_offsets[i + 1] - self->component_offsets[i]);
    }
    return (PyObject *)out;
}

static const kform_spec_object *kform_specs_parse_component_index(PyObject *self, PyTypeObject *defining_class,
                                                                  PyObject *const *args, const Py_ssize_t nargs,
                                                                  PyObject *kwnames,
                                                                  const interplib_module_state_t **state,
                                                                  Py_ssize_t *idx, unsigned *n, unsigned *k)
{
    const kform_spec_object *this;
    if (ensure_kform_specs_and_state(self, defining_class, &this, state) < 0)
    {
        return NULL;
    }

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = idx, .kwname = "idx"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    *n = Py_SIZE(this->function_space);
    *k = this->order;
    const unsigned max_components = combination_total_count(*n, *k);

    if (*idx < 0 || *idx >= max_components)
    {
        PyErr_Format(PyExc_ValueError, "Expected index in [0, %u), got %zd.", max_components, *idx);
        return NULL;
    }
    return this;
}

static PyObject *kform_specs_get_component_slice(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                 const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    Py_ssize_t idx;
    unsigned n;
    unsigned k;
    const kform_spec_object *const this =
        kform_specs_parse_component_index(self, defining_class, args, nargs, kwnames, &state, &idx, &n, &k);
    if (!this)
        return NULL;

    PyObject *start = NULL, *stop = NULL, *slice = NULL;
    if ((start = PyLong_FromSize_t(this->component_offsets[idx])) != NULL &&
        (stop = PyLong_FromSize_t(this->component_offsets[idx + 1])) != NULL)
    {
        slice = PySlice_New(start, stop, NULL);
    }
    Py_XDECREF(start);
    Py_XDECREF(stop);

    return slice;
}

PyDoc_STRVAR(kform_spec_get_component_function_space_docstring,
             "get_component_function_space(idx: int) -> FunctionSpace\n"
             "Get the function space for a component.\n"
             "        \n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the component.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "FunctionSpace\n"
             "    Function space corresponding to the k-form component with the specified index.\n");

PyDoc_STRVAR(kform_spec_get_component_covector_basis_docstring,
             "get_component_basis(idx: int) -> CovectorBasis\n"
             "Get covector basis bundle for a component.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the component.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "CovectorBasis\n"
             "    Covector basis bundle corresponding to the k-form component with the specified index.\n");

PyDoc_STRVAR(kform_specs_get_component_slice_docstring,
             "get_component_slice(idx: int) -> slice\n"
             "Get the slice corresponding to degrees of freedom of a k-form component.\n"
             "\n"
             "The resulting slice can be used to index into the flattened array of degrees\n"
             "of freedom to get the DoFs corresponding to a praticular component.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the k-form component.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "slice\n"
             "    Slice of the flattened array of all k-form degrees of freedom that corresponds\n"
             "    to degrees of freedom of the specified component.\n");

PyDoc_STRVAR(kform_specs_docstring, "KFormSpecs(order: int, base_space: FunctionSpace)\n"
                                    "Differential k-form specification.\n"
                                    "\n"
                                    "Parameters\n"
                                    "----------\n"
                                    "order : int\n"
                                    "    Order of the k-form.\n"
                                    "\n"
                                    "base_space : FunctionSpace\n"
                                    "    Base space to use for the k-forms. This is also the space in which 0-forms\n"
                                    "    are defined.\n");

PyType_Spec kform_spec_type_spec = {
    .name = FDG_TYPE_NAME("KFormSpecs"),
    .basicsize = sizeof(kform_spec_object),
    .itemsize = sizeof(*((kform_spec_object *)0)->component_offsets),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_new, kform_spec_new},
        {Py_tp_dealloc, kform_spec_dealloc},
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_getset,
         (PyGetSetDef[]){
             {
                 .name = "order",
                 .get = (getter)kform_spec_get_order,
                 .doc = "int : Order of the k-form.",
             },
             {
                 .name = "base_space",
                 .get = (getter)kform_spec_get_function_space,
                 .doc = "FunctionSpace : Base function space the k-form is based in.",
             },
             {
                 .name = "dimension",
                 .get = (getter)kform_spec_get_ndim,
                 .doc = "int : Dimension of the space the k-form is in.",
             },
             {
                 .name = "component_count",
                 .get = (getter)kform_spec_get_component_count,
                 .doc = "int : Number of components in the k-form.",
             },
             {
                 .name = "component_dof_counts",
                 .get = (getter)kform_specs_get_component_dof_counts,
                 .doc = "numpy.typing.NDArray[numpy.int64] : Number of DoFs in each component.",
             },
             {},
         }},
        {Py_tp_methods,
         (PyMethodDef[]){
             {
                 .ml_name = "get_component_function_space",
                 .ml_meth = (void *)kform_spec_get_component_function_space,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = kform_spec_get_component_function_space_docstring,
             },
             {
                 .ml_name = "get_component_basis",
                 .ml_meth = (void *)kform_spec_get_component_covector_basis,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = kform_spec_get_component_covector_basis_docstring,
             },
             {
                 .ml_name = "get_component_slice",
                 .ml_meth = (void *)kform_specs_get_component_slice,
                 .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
                 .ml_doc = kform_specs_get_component_slice_docstring,
             },
             {},
         }},
        {Py_tp_doc, (char *)kform_specs_docstring},
        {},
    }};

kform_object *kform_object_create(PyTypeObject *type, kform_spec_object *spec, const int zero_init)
{
    // First, count up the number of DoFs per component
    const unsigned n = Py_SIZE(spec->function_space);
    const unsigned k = spec->order;
    const unsigned component_cnt = combination_total_count(n, k);

    const size_t total_component_dofs = spec->component_offsets[component_cnt];
    kform_object *const this = (kform_object *)type->tp_alloc(type, (Py_ssize_t)total_component_dofs);
    if (!this)
    {
        return NULL;
    }
    this->specs = spec;
    Py_INCREF(spec);
    if (zero_init)
    {
        memset(this->values, 0, total_component_dofs * sizeof(*this->values));
    }
    return this;
}

static PyObject *kform_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(subtype);
    if (!state)
        return NULL;
    kform_spec_object *specs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", (char *[]){"specs", NULL}, state->kform_specs_type, &specs))
        return NULL;

    return (PyObject *)kform_object_create(subtype, specs, 1);
}

static void kform_dealloc(kform_object *self)
{
    PyObject_GC_UnTrack(self);
    Py_DECREF(self->specs);
    self->specs = NULL;
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static int ensure_kform_and_state(PyObject *self, PyTypeObject *defining_class, kform_object **out_spec,
                                  const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->kform_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a KForm object.");
        return -1;
    }
    *p_state = state;
    *out_spec = (kform_object *)self;
    return 0;
}

static kform_object *kform_parse_component_index(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                 const Py_ssize_t nargs, PyObject *kwnames,
                                                 const interplib_module_state_t **state, Py_ssize_t *idx, unsigned *n,
                                                 unsigned *k)
{
    kform_object *this;
    if (ensure_kform_and_state(self, defining_class, (kform_object **)&this, state) < 0)
    {
        return NULL;
    }

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = idx, .kwname = "idx"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    *n = Py_SIZE(this->specs->function_space);
    *k = this->specs->order;
    const unsigned max_components = combination_total_count(*n, *k);

    if (*idx < 0 || *idx >= max_components)
    {
        PyErr_Format(PyExc_ValueError, "Expected index in [0, %u), got %zd.", max_components, *idx);
        return NULL;
    }
    return this;
}

static PyObject *kform_get_component_dofs(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                          const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    Py_ssize_t idx;
    unsigned n;
    unsigned k;

    kform_object *const this =
        kform_parse_component_index(self, defining_class, args, nargs, kwnames, &state, &idx, &n, &k);
    if (!this)
        return NULL;

    double *const out_dofs = this->values + this->specs->component_offsets[idx];
    uint8_t *covector_indices;
    npy_intp *out_dims;
    void *const mem = cutl_alloc_group(&PYTHON_ALLOCATOR,
                                       (const cutl_alloc_info_t[]){
                                           {.size = k * sizeof(*covector_indices), .p_ptr = (void **)&covector_indices},
                                           {.size = n * sizeof(*out_dims), .p_ptr = (void **)&out_dims},
                                           {},
                                       });
    if (!mem)
        return NULL;

    combination_set_to_index(n, k, covector_indices, idx);
    size_t total_dofs = 1;
    for (unsigned i = 0, i_covector = 0; i < n; ++i)
    {
        const basis_spec_t *const spec = this->specs->function_space->specs + i;
        unsigned ndof;
        if (i_covector < k && covector_indices[i_covector] == i)
        {
            ndof = spec->order;
            i_covector += 1;
        }
        else
        {
            ndof = spec->order + 1;
        }
        out_dims[i] = ndof;
        total_dofs *= ndof;
    }
    (void)total_dofs;
    ASSERT(total_dofs == this->specs->component_offsets[idx + 1] - this->specs->component_offsets[idx],
           "Total number of DoFs did not match number compute by offsets (%zu when expecting %zu).", total_dofs,
           this->specs->component_offsets[idx + 1] - this->specs->component_offsets[idx]);

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(n, out_dims, NPY_DOUBLE, out_dofs);
    cutl_dealloc(&PYTHON_ALLOCATOR, mem);
    if (!out)
        return NULL;

    if (PyArray_SetBaseObject(out, (PyObject *)this) < 0)
    {
        Py_DECREF(out);
        return NULL;
    }

    Py_INCREF(this);
    return (PyObject *)out;
}

static PyObject *kform_get_specs(PyObject *self, void *Py_UNUSED(closure))
{
    const interplib_module_state_t *state;
    const kform_object *this;
    if (ensure_kform_and_state(self, NULL, (kform_object **)&this, &state) < 0)
        return NULL;
    Py_INCREF(this->specs);
    return (PyObject *)this->specs;
}

static PyObject *kform_get_values(PyObject *self, void *Py_UNUSED(closure))
{
    const interplib_module_state_t *state;
    kform_object *this;
    if (ensure_kform_and_state(self, NULL, (kform_object **)&this, &state) < 0)
        return NULL;
    const unsigned component_count = combination_total_count(Py_SIZE(this->specs->function_space), this->specs->order);
    const npy_intp total_dofs = (npy_intp)this->specs->component_offsets[component_count];
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(1, &total_dofs, NPY_DOUBLE, this->values);
    if (!out)
        return NULL;

    if (PyArray_SetBaseObject(out, (PyObject *)this) < 0)
    {
        Py_DECREF(out);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)out;
}

PyDoc_STRVAR(kform_get_component_dofs_docstring,
             "get_component_dofs(idx: int) -> numpy.tying.NDArray[numpy.double]\n"
             "Get the array containing the degrees of freedom for a k-form component.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the k-form component.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array containing the degrees of freedom. This is not a copy, so changing\n"
             "    values in it will change the values of degrees of freedom.\n");

PyDoc_STRVAR(kform_docstring, "KForm(specs: KFormSpecs)\n"
                              "Type holding the degrees of freedom of a k-form.\n"
                              "\n"
                              "Parameters\n"
                              "----------\n"
                              "specs : KFormSpecs\n"
                              "    Specification of the k-form that is to be created.\n");

PyType_Spec kform_type_spec = {
    .name = FDG_TYPE_NAME("KForm"),
    .basicsize = sizeof(kform_object),
    .itemsize = sizeof(*((kform_object *)0)->values),
    .flags = Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_DEFAULT,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_new, kform_new},
        {Py_tp_dealloc, kform_dealloc},
        {Py_tp_methods,
         (PyMethodDef[]){
             {
                 .ml_name = "get_component_dofs",
                 .ml_meth = (void *)kform_get_component_dofs,
                 .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
                 .ml_doc = kform_get_component_dofs_docstring,
             },
             {},
         }},
        {Py_tp_getset,
         (PyGetSetDef[]){
             {
                 .name = "specs",
                 .get = (getter)kform_get_specs,
                 .doc = "KFormSpecs : Specifications of the k-form.",
             },
             {
                 .name = "values",
                 .get = (getter)kform_get_values,
                 .doc = "numpy.typing.NDArray[numpy.double] : Values of all k-form degrees of freedom.",
             },
             {},
         }},
        {Py_tp_doc, (char *)kform_docstring},
        {},
    }};
