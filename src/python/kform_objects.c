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

    kform_spec_object *const self = (kform_spec_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

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
    for (unsigned i = 0, i_covector = 0; i < n; ++i)
    {
        basis.basis_bits |= 1 << covector_indices[i_covector];
    }

    covector_basis_object *const covector_basis = covector_basis_object_create(state->covector_basis_type, basis);
    PyMem_Free(covector_indices);

    return (PyObject *)covector_basis;
}

PyType_Spec kform_spec_type_spec = {.name = "interplib._interp.KFormSpecs",
                                    .basicsize = sizeof(kform_spec_object),
                                    .itemsize = 0,
                                    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC |
                                             Py_TPFLAGS_IMMUTABLETYPE,
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
                                             {},
                                         }},
                                        {Py_tp_methods,
                                         (PyMethodDef[]){
                                             {
                                                 .ml_name = "get_component_function_space",
                                                 .ml_meth = (void *)kform_spec_get_component_function_space,
                                                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                                                 .ml_doc = NULL, // TODO
                                             },
                                             {
                                                 .ml_name = "get_component_covector_basis",
                                                 .ml_meth = (void *)kform_spec_get_component_covector_basis,
                                                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                                                 .ml_doc = NULL, // TODO
                                             },
                                             {},
                                         }},
                                        {},
                                    }};
