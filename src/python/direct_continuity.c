#include "direct_continuity.h"
#include "../constraints/direct.h"
#include "basis_objects.h"
#include "cpyutl.h"
#include "integration_objects.h"
#include "kform_objects.h"
#include "mesh_objects.h"
#include "module.h"
#include <numpy/ndarrayobject.h>

/**
 * @brief Reject the element specifications the direct core cannot number.
 *
 * The elimination needs a positive order on every axis; a zero order leaves no test functions. The core reports
 * this as a bare status, so it is checked here to name the offending element and axis.
 *
 * @return 0 on success, -1 with a Python exception set otherwise.
 */
static int direct_check_basis(kform_spec_object *const *const specs, const size_t count, const unsigned ndim)
{
    for (size_t element = 0; element < count; ++element)
    {
        const basis_spec_t *const basis = kform_specs_from_python(specs[element]).basis;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (basis[axis].order == 0u)
            {
                PyErr_Format(PyExc_ValueError,
                             "The direct continuity map needs a positive basis order; element %zu axis %u has "
                             "order 0, which leaves no test functions.",
                             element, axis);
                return -1;
            }
        }
    }
    return 0;
}

/**
 * @brief Copy one size_t array into a fresh int64 array.
 *
 * @return The new array, or NULL with a Python exception set.
 */
static PyArrayObject *direct_make_int64(const size_t *const data, const size_t count)
{
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(1, &(npy_intp){(npy_intp)count}, NPY_INT64);
    if (!array)
        return NULL;
    npy_int64 *const out = PyArray_DATA(array);
    for (size_t i = 0; i < count; ++i)
        out[i] = (npy_int64)data[i];
    return array;
}

static PyObject *direct_dof_map_get_global_dof_count(PyObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t(((direct_dof_map_object *)self)->global_dof_count);
}

static PyObject *direct_dof_map_get_element_dof_count(PyObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t(((direct_dof_map_object *)self)->element_dof_count);
}

static PyObject *direct_dof_map_get_entry_count(PyObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromSize_t(((direct_dof_map_object *)self)->entry_count);
}

static PyObject *direct_dof_map_get_element_offsets(PyObject *self, void *Py_UNUSED(closure))
{
    direct_dof_map_object *this = (direct_dof_map_object *)self;
    Py_INCREF(this->element_offsets);
    return (PyObject *)this->element_offsets;
}

static PyObject *direct_dof_map_get_element_interior_offsets(PyObject *self, void *Py_UNUSED(closure))
{
    direct_dof_map_object *this = (direct_dof_map_object *)self;
    Py_INCREF(this->element_interior_offsets);
    return (PyObject *)this->element_interior_offsets;
}

static PyObject *direct_dof_map_get_entry_offsets(PyObject *self, void *Py_UNUSED(closure))
{
    direct_dof_map_object *this = (direct_dof_map_object *)self;
    Py_INCREF(this->entry_offsets);
    return (PyObject *)this->entry_offsets;
}

static PyObject *direct_dof_map_get_entry_index(PyObject *self, void *Py_UNUSED(closure))
{
    direct_dof_map_object *this = (direct_dof_map_object *)self;
    Py_INCREF(this->entry_index);
    return (PyObject *)this->entry_index;
}

static PyObject *direct_dof_map_get_entry_value(PyObject *self, void *Py_UNUSED(closure))
{
    direct_dof_map_object *this = (direct_dof_map_object *)self;
    Py_INCREF(this->entry_value);
    return (PyObject *)this->entry_value;
}

PyDoc_STRVAR(direct_dof_map_docstring, "DirectDofMap()\n"
                                       "\n"
                                       "    Element-to-global transfer of one direct continuity map.\n"
                                       "\n"
                                       "    The map introduces one unknown per function of every shared\n"
                                       "    object's common Legendre test space -- the L2 projection of the\n"
                                       "    element traces onto it must agree across the incident elements --\n"
                                       "    and eliminates each element's degrees of freedom against those\n"
                                       "    unknowns by a QR of its stacked constraint rows. The orthogonal\n"
                                       "    complement stays element-private, so every degree of freedom of\n"
                                       "    a mesh is numbered once.\n"
                                       "\n"
                                       "    The type cannot be instantiated directly; use\n"
                                       "    :meth:`Mesh.compute_kform_direct_dof_map`.\n");

static PyObject *direct_dof_map_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    (void)type;
    (void)args;
    (void)kwds;
    PyErr_SetString(PyExc_TypeError,
                    "DirectDofMap cannot be instantiated directly; use Mesh.compute_kform_direct_dof_map.");
    return NULL;
}

static int direct_dof_map_traverse(PyObject *self, const visitproc visit, void *arg)
{
    direct_dof_map_object *const this = (direct_dof_map_object *)self;
    Py_VISIT(this->element_offsets);
    Py_VISIT(this->element_interior_offsets);
    Py_VISIT(this->entry_offsets);
    Py_VISIT(this->entry_index);
    Py_VISIT(this->entry_value);
    return heap_type_traverse_type(self, visit, arg);
}

static void direct_dof_map_dealloc(direct_dof_map_object *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->element_offsets);
    Py_CLEAR(self->element_interior_offsets);
    Py_CLEAR(self->entry_offsets);
    Py_CLEAR(self->entry_index);
    Py_CLEAR(self->entry_value);
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyGetSetDef direct_dof_map_getset[] = {
    {.name = "global_dof_count",
     .get = direct_dof_map_get_global_dof_count,
     .doc = "int : Size of the global unknown vector the map numbers."},
    {.name = "element_dof_count",
     .get = direct_dof_map_get_element_dof_count,
     .doc = "int : Sum of the elements' local degree-of-freedom counts."},
    {.name = "entry_count", .get = direct_dof_map_get_entry_count, .doc = "int : Nonzeros of the transfer."},
    {.name = "element_offsets",
     .get = direct_dof_map_get_element_offsets,
     .doc = "numpy.typing.NDArray[numpy.int64] : Local degree-of-freedom offsets of every element. "
            "The array has ``element_count + 1`` entries."},
    {.name = "element_interior_offsets",
     .get = direct_dof_map_get_element_interior_offsets,
     .doc = "numpy.typing.NDArray[numpy.int64] : Element-private degree-of-freedom offsets of every element. "
            "The array has ``element_count + 1`` entries. A private degree of freedom lies in the orthogonal "
            "complement the elimination leaves free, not a coordinate range of the element."},
    {.name = "entry_offsets",
     .get = direct_dof_map_get_entry_offsets,
     .doc = "numpy.typing.NDArray[numpy.int64] : Row offsets of the transfer. The array has "
            "``element_dof_count + 1`` entries. A row holds one entry when the element's paired mode maps to a "
            "single object unknown, and several when the elimination mixes element degrees of freedom."},
    {.name = "entry_index",
     .get = direct_dof_map_get_entry_index,
     .doc = "numpy.typing.NDArray[numpy.int64] : Global degree of freedom of every entry."},
    {.name = "entry_value",
     .get = direct_dof_map_get_entry_value,
     .doc = "numpy.typing.NDArray[numpy.double] : Transfer coefficient of every entry."},
    {},
};

PyDoc_STRVAR(direct_dof_map_scatter_triplets_doc,
             "_scatter_triplets(local_matrices, first_element, stop_element, n_threads=0)\n"
             "\n"
             "    Scatter a range of row-major per-element local matrices into COO triplets.\n"
             "\n"
             "    Element ``e`` contributes ``value_r * m_i_j * value_c`` over every\n"
             "    entry pair ``r``, ``c`` of every local degree-of-freedom pair ``i``,\n"
             "    ``j``. ``local_matrices`` is one flat C-contiguous float64 array\n"
             "    holding one row-major ``n_e x n_e`` block per element in element\n"
             "    order, and its length must equal the sum of the blocks over every\n"
             "    element of the map. Elements ``[first_element, stop_element)`` are\n"
             "    scattered, with the range's block base inside ``local_matrices``\n"
             "    tracked internally. ``n_threads`` picks the worker count; ``0``\n"
             "    uses the OpenMP default, and the result is identical for any\n"
             "    thread count. Returns three arrays ``(rows, cols, values)`` of\n"
             "    int64 rows, int64 columns, and float64 values.\n");

/**
 * @brief Scatter one element range's row-major local matrices into COO triplets.
 *
 * @return A ``(rows, cols, values)`` tuple of numpy arrays, or NULL with a Python exception set.
 */
static PyObject *direct_dof_map_scatter_triplets(PyObject *self, PyObject *const *args, const Py_ssize_t nargs,
                                                 PyObject *kwnames)
{
    PyObject *local_object;
    Py_ssize_t first_element = 0;
    Py_ssize_t stop_element = 0;
    Py_ssize_t n_threads = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &local_object},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &first_element},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &stop_element},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = 1},
                {}},
            args, nargs, kwnames) < 0)
        return NULL;
    if (first_element < 0 || first_element > stop_element)
    {
        PyErr_Format(PyExc_ValueError,
                     "the element range must satisfy 0 <= first_element <= stop_element, but got %zd, %zd.",
                     first_element, stop_element);
        return NULL;
    }
    if (n_threads < 0)
    {
        PyErr_Format(PyExc_ValueError, "n_threads must be nonnegative, but got %zd.", n_threads);
        return NULL;
    }
    if (!PyArray_Check(local_object) || PyArray_NDIM((const PyArrayObject *)local_object) != 1 ||
        PyArray_TYPE((const PyArrayObject *)local_object) != NPY_DOUBLE ||
        !PyArray_ISCONTIGUOUS((const PyArrayObject *)local_object))
    {
        PyErr_SetString(PyExc_TypeError, "local_matrices must be a one-dimensional C-contiguous float64 array.");
        return NULL;
    }

    const direct_dof_map_object *const this = (direct_dof_map_object *)self;
    const npy_intp *const element_offsets = PyArray_DATA(this->element_offsets);
    const uint64_t element_count = (uint64_t)PyArray_DIM(this->element_offsets, 0) - 1u;
    const size_t element_dof_count = (size_t)PyArray_DIM(this->entry_offsets, 0) - 1u;
    const size_t entry_count = this->entry_count;
    if ((uint64_t)stop_element > element_count)
    {
        PyErr_Format(PyExc_ValueError, "the map holds %zu elements, but the range ends at %zd.", (size_t)element_count,
                     stop_element);
        return NULL;
    }

    // The flat array holds one row-major n_e x n_e block per element.
    size_t expected = 0;
    for (uint64_t element = 0; element < element_count; ++element)
    {
        const size_t local_count = (size_t)(element_offsets[element + 1] - element_offsets[element]);
        if (local_count != 0 && local_count > SIZE_MAX / local_count)
        {
            PyErr_SetString(PyExc_OverflowError, "The local matrices exceed addressable memory.");
            return NULL;
        }
        const size_t block = local_count * local_count;
        if (block > SIZE_MAX - expected)
        {
            PyErr_SetString(PyExc_OverflowError, "The local matrices exceed addressable memory.");
            return NULL;
        }
        expected += block;
    }
    if ((uint64_t)PyArray_DIM((const PyArrayObject *)local_object, 0) != (uint64_t)expected)
    {
        PyErr_Format(PyExc_ValueError,
                     "local_matrices must hold %zu entries, one row-major block per element, but got %zd.", expected,
                     PyArray_DIM((const PyArrayObject *)local_object, 0));
        return NULL;
    }

    // The core reads size_t offsets and indices, so the int64 arrays are cast into scratch first.
    size_t *element_offsets_scratch = NULL;
    size_t *entry_offsets_scratch = NULL;
    size_t *entry_index_scratch = NULL;
    void *const cast_memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {(element_count + 1u) * sizeof(*element_offsets_scratch), (void **)&element_offsets_scratch},
            {(element_dof_count + 1u) * sizeof(*entry_offsets_scratch), (void **)&entry_offsets_scratch},
            {entry_count * sizeof(*entry_index_scratch), (void **)&entry_index_scratch},
            {}});
    if (!cast_memory || !element_offsets_scratch || !entry_offsets_scratch || !entry_index_scratch)
    {
        PyErr_NoMemory();
        goto cleanup;
    }
    for (size_t i = 0; i <= element_count; ++i)
        element_offsets_scratch[i] = (size_t)element_offsets[i];
    {
        const npy_int64 *const entry_offsets = PyArray_DATA(this->entry_offsets);
        const npy_int64 *const entry_index = PyArray_DATA(this->entry_index);
        for (size_t i = 0; i <= element_dof_count; ++i)
            entry_offsets_scratch[i] = (size_t)entry_offsets[i];
        for (size_t i = 0; i < entry_count; ++i)
            entry_index_scratch[i] = (size_t)entry_index[i];
    }

    {
        // The range's block base skips every earlier element's block in local_matrices.
        size_t block_base = 0;
        size_t triplet_count = 0;
        for (uint64_t element = 0; element < (uint64_t)stop_element; ++element)
        {
            const size_t local_base = element_offsets_scratch[element];
            const size_t local_count = element_offsets_scratch[element + 1] - local_base;
            size_t entries = 0;
            for (size_t dof = 0; dof < local_count; ++dof)
                entries += entry_offsets_scratch[local_base + dof + 1] - entry_offsets_scratch[local_base + dof];
            if (entries != 0 && entries > SIZE_MAX / entries)
            {
                PyErr_SetString(PyExc_OverflowError, "The triplets exceed addressable memory.");
                goto cleanup;
            }
            const size_t block = local_count * local_count;
            if (block > SIZE_MAX - block_base)
            {
                PyErr_SetString(PyExc_OverflowError, "The local matrices exceed addressable memory.");
                goto cleanup;
            }
            if (element >= (uint64_t)first_element)
            {
                if (entries * entries > SIZE_MAX - triplet_count)
                {
                    PyErr_SetString(PyExc_OverflowError, "The triplets exceed addressable memory.");
                    goto cleanup;
                }
                triplet_count += entries * entries;
            }
            else
                block_base += block;
        }

        PyArrayObject *const rows =
            (PyArrayObject *)PyArray_SimpleNew(1, &(npy_intp){(npy_intp)triplet_count}, NPY_INT64);
        PyArrayObject *const cols =
            rows ? (PyArrayObject *)PyArray_SimpleNew(1, &(npy_intp){(npy_intp)triplet_count}, NPY_INT64) : NULL;
        PyArrayObject *const values =
            cols ? (PyArrayObject *)PyArray_SimpleNew(1, &(npy_intp){(npy_intp)triplet_count}, NPY_DOUBLE) : NULL;
        PyObject *const result = values ? Py_BuildValue("(NNN)", rows, cols, values) : NULL;
        if (result)
        {
            // The plan slice keeps the absolute DoF numbering; only the element range shrinks.
            direct_continuity_plan_t plan = {0};
            plan.element_count = (size_t)(stop_element - first_element);
            plan.element_dof_offsets = element_offsets_scratch + first_element;
            const double *const local_data = PyArray_DATA((PyArrayObject *)local_object);
            const double *const entry_value = PyArray_DATA(this->entry_value);
            Py_BEGIN_ALLOW_THREADS;
            // The int64 index buffers double as the size_t outputs; the counts fit by construction.
            direct_continuity_scatter_triplets(&plan, entry_offsets_scratch, entry_index_scratch, entry_value,
                                               local_data + block_base, (unsigned)n_threads, PyArray_DATA(rows),
                                               PyArray_DATA(cols), PyArray_DATA(values));
            Py_END_ALLOW_THREADS;
        }
        if (!result)
        {
            if (!PyErr_Occurred())
                PyErr_NoMemory();
            goto cleanup;
        }
        cutl_dealloc(&PYTHON_ALLOCATOR, cast_memory);
        return result;
    }

cleanup:
    cutl_dealloc(&PYTHON_ALLOCATOR, cast_memory);
    return NULL;
}

static PyMethodDef direct_dof_map_methods[] = {
    {.ml_name = "_scatter_triplets",
     .ml_meth = (PyCFunction)(void (*)(void))direct_dof_map_scatter_triplets,
     .ml_flags = METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = direct_dof_map_scatter_triplets_doc},
    {},
};

PyType_Spec direct_dof_map_type_spec = {.name = FDG_TYPE_NAME("DirectDofMap"),
                                        .basicsize = sizeof(direct_dof_map_object),
                                        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_IMMUTABLETYPE |
                                                 Py_TPFLAGS_HAVE_GC,
                                        .slots = (PyType_Slot[]){
                                            {Py_tp_new, direct_dof_map_new},
                                            {Py_tp_doc, (void *)direct_dof_map_docstring},
                                            {Py_tp_traverse, direct_dof_map_traverse},
                                            {Py_tp_dealloc, direct_dof_map_dealloc},
                                            {Py_tp_getset, direct_dof_map_getset},
                                            {Py_tp_methods, direct_dof_map_methods},
                                            {},
                                        }};

/**
 * @brief Borrow the per-element k-form specs and read off their common degree.
 *
 * Only the k-form degree must agree across elements; per-axis basis orders are free to differ.
 *
 * @return 0 on success, -1 with a Python exception set otherwise.
 */
static int direct_check_element_specs(const interplib_module_state_t *const state, PyObject *const specs_seq,
                                      const unsigned ndim, const uint64_t element_count, unsigned *const out_order,
                                      kform_spec_object **const out_specs)
{
    for (uint64_t element = 0; element < element_count; ++element)
    {
        PyObject *const spec_object = PySequence_Fast_GET_ITEM(specs_seq, (Py_ssize_t)element);
        if (!PyObject_TypeCheck(spec_object, state->kform_specs_type))
        {
            PyErr_SetString(PyExc_TypeError, "element_specs must contain KFormSpecs objects.");
            return -1;
        }
        kform_spec_object *const spec = (kform_spec_object *)spec_object;
        if (spec->function_space == NULL || Py_SIZE(spec->function_space) != (Py_ssize_t)ndim)
        {
            PyErr_SetString(PyExc_ValueError, "Every element spec must describe the mesh dimension.");
            return -1;
        }
        if (element == 0)
            *out_order = spec->order;
        else if (spec->order != *out_order)
        {
            PyErr_SetString(PyExc_ValueError, "All element specs must have the same k-form degree.");
            return -1;
        }
        out_specs[element] = spec;
    }
    return 0;
}

/**
 * @brief Check that every layout size is representable as a numpy length.
 */
static int direct_check_layout_sizes(const direct_continuity_layout_t *const layout)
{
    if (layout->element_count > (size_t)PY_SSIZE_T_MAX || layout->element_dof_count >= (size_t)PY_SSIZE_T_MAX ||
        layout->entry_count > (size_t)PY_SSIZE_T_MAX || layout->global_dof_count > (size_t)PY_SSIZE_T_MAX)
    {
        PyErr_SetString(PyExc_OverflowError, "The direct continuity map exceeds Python sequence limits.");
        return -1;
    }
    return 0;
}

/**
 * @brief Hand the built arrays to a fresh DirectDofMap.
 *
 * @return The new object, or NULL with a Python exception set.
 */
static PyObject *direct_dof_map_create(PyTypeObject *const type, const direct_continuity_layout_t *const layout,
                                       PyArrayObject *element_offsets, PyArrayObject *element_interior_offsets,
                                       PyArrayObject *entry_offsets, PyArrayObject *entry_index,
                                       PyArrayObject *entry_value)
{
    direct_dof_map_object *const self = (direct_dof_map_object *)type->tp_alloc(type, 0);
    if (!self)
    {
        Py_DECREF(element_offsets);
        Py_DECREF(element_interior_offsets);
        Py_DECREF(entry_offsets);
        Py_DECREF(entry_index);
        Py_DECREF(entry_value);
        return NULL;
    }
    self->global_dof_count = layout->global_dof_count;
    self->element_dof_count = layout->element_dof_count;
    self->entry_count = layout->entry_count;
    self->element_offsets = element_offsets;
    self->element_interior_offsets = element_interior_offsets;
    self->entry_offsets = entry_offsets;
    self->entry_index = entry_index;
    self->entry_value = entry_value;
    return (PyObject *)self;
}

PyObject *mesh_compute_kform_direct_dof_map(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                            const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->mesh_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return NULL;
    }

    PyObject *element_specs_object;
    integration_registry_object *integration_registry = (integration_registry_object *)state->registry_integration;
    basis_registry_object *basis_registry = (basis_registry_object *)state->registry_basis;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &element_specs_object},
                                                    {.type = CPYARG_TYPE_PYTHON,
                                                     .p_val = &integration_registry,
                                                     .type_check = state->integration_registry_type,
                                                     .kwname = "integration_registry",
                                                     .optional = 1,
                                                     .kw_only = 1},
                                                    {.type = CPYARG_TYPE_PYTHON,
                                                     .p_val = &basis_registry,
                                                     .type_check = state->basis_registry_type,
                                                     .kwname = "basis_registry",
                                                     .optional = 1,
                                                     .kw_only = 1},
                                                    {}},
                              args, nargs, kwnames) < 0)
        return NULL;

    topo_mesh_t *const mesh = ((mesh_object *)self)->mesh;
    const unsigned ndim = mesh->ndim;
    const uint64_t element_count = mesh->element_count;
    if (element_count > (uint64_t)PY_SSIZE_T_MAX ||
        element_count > (uint64_t)(SIZE_MAX / sizeof(kform_spec_t) / sizeof(const kform_spec_t *)))
    {
        PyErr_SetString(PyExc_OverflowError, "Mesh dimensions exceed Python sequence limits.");
        return NULL;
    }

    PyObject *const specs_seq = PySequence_Fast(element_specs_object, "element_specs must be a sequence.");
    if (!specs_seq)
        return NULL;
    if (PySequence_Fast_GET_SIZE(specs_seq) != (Py_ssize_t)element_count)
    {
        PyErr_Format(PyExc_ValueError, "element_specs must contain %llu entries.", (unsigned long long)element_count);
        Py_DECREF(specs_seq);
        return NULL;
    }

    void *memory = NULL;
    kform_spec_object **spec_objects = NULL;
    kform_spec_t *spec_values = NULL;
    const kform_spec_t **elements = NULL;
    size_t *entry_offsets = NULL;
    size_t *entry_index = NULL;
    void *scratch_memory = NULL;
    direct_continuity_work_t work;
    direct_continuity_plan_t plan;
    bool plan_ready = false;
    unsigned order = 0;
    PyObject *result = NULL;
    PyArrayObject *element_offsets = NULL;
    PyArrayObject *element_interior_offsets = NULL;
    PyArrayObject *entry_offsets_array = NULL;
    PyArrayObject *entry_index_array = NULL;
    PyArrayObject *entry_value_array = NULL;

    // One block holds the spec table, the pointer array the request borrows and the row-compressed transfer
    // scratch.
    memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){{element_count * sizeof(*spec_objects), (void **)&spec_objects},
                                                       {element_count * sizeof(*spec_values), (void **)&spec_values},
                                                       {element_count * sizeof(*elements), (void **)&elements},
                                                       {}});
    if (!memory || !spec_objects || !spec_values || !elements)
    {
        PyErr_NoMemory();
        goto cleanup;
    }
    if (direct_check_element_specs(state, specs_seq, ndim, element_count, &order, spec_objects) < 0)
        goto cleanup;
    if (direct_check_basis(spec_objects, (size_t)element_count, ndim) < 0)
        goto cleanup;
    for (uint64_t element = 0; element < element_count; ++element)
    {
        spec_values[element] = kform_specs_from_python(spec_objects[element]);
        elements[element] = spec_values + element;
    }

    const direct_continuity_request_t request = {
        .ndim = ndim,
        .order = order,
        .mesh = mesh,
        .elements = elements,
        .basis_registry = basis_registry->registry,
        .integration_registry = integration_registry->registry,
    };

    void *const plan_memory = cutl_alloc(&PYTHON_ALLOCATOR, direct_continuity_plan_memory(&request));
    void *const work_memory = cutl_alloc(&PYTHON_ALLOCATOR, direct_continuity_work_memory(&request));
    if (!plan_memory || !work_memory)
    {
        PyErr_NoMemory();
        cutl_dealloc(&PYTHON_ALLOCATOR, plan_memory);
        cutl_dealloc(&PYTHON_ALLOCATOR, work_memory);
        goto cleanup;
    }
    direct_continuity_plan_init(&plan, &request, plan_memory);
    direct_continuity_work_init(&work, &request, work_memory);
    // The plan counts are set by its init, so a failed prepare can still release what it fetched.
    plan_ready = true;

    fdg_result_t prepare_result;
    direct_continuity_layout_t layout;
    {
        Py_BEGIN_ALLOW_THREADS;
        prepare_result = direct_continuity_prepare(&request, &work, &plan);
        if (prepare_result == FDG_SUCCESS)
            direct_continuity_layout(&request, &plan, &work, &layout);
        Py_END_ALLOW_THREADS;
    }
    if (prepare_result != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not prepare the direct continuity map: %s (%s).",
                     fdg_error_str(prepare_result), fdg_error_msg(prepare_result));
        cutl_dealloc(&PYTHON_ALLOCATOR, work_memory);
        goto fail_plan;
    }
    if (direct_check_layout_sizes(&layout) < 0)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, work_memory);
        goto fail_plan;
    }

    {
        // The transfer is written as size_t, so it is built in scratch and copied into the int64 arrays.
        if (layout.element_dof_count + 1u > SIZE_MAX / sizeof(*entry_offsets))
        {
            PyErr_SetString(PyExc_OverflowError, "The direct continuity map exceeds Python sequence limits.");
            cutl_dealloc(&PYTHON_ALLOCATOR, work_memory);
            goto fail_plan;
        }
        const cutl_alloc_info_t scratch[] = {
            {(layout.element_dof_count + 1u) * sizeof(*entry_offsets), (void **)&entry_offsets},
            {layout.entry_count * sizeof(*entry_index), (void **)&entry_index},
            {},
        };
        if (!(scratch_memory = cutl_alloc_group(&PYTHON_ALLOCATOR, scratch)))
        {
            PyErr_NoMemory();
            cutl_dealloc(&PYTHON_ALLOCATOR, work_memory);
            goto fail_plan;
        }
    }

    entry_value_array = (PyArrayObject *)PyArray_SimpleNew(1, &(npy_intp){(npy_intp)layout.entry_count}, NPY_DOUBLE);
    if (!entry_value_array)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, work_memory);
        goto fail_plan;
    }
    {
        Py_BEGIN_ALLOW_THREADS;
        direct_continuity_build(&request, &plan, &work, entry_offsets, entry_index, PyArray_DATA(entry_value_array));
        Py_END_ALLOW_THREADS;
    }

    element_offsets = direct_make_int64(plan.element_dof_offsets, (size_t)element_count + 1u);
    element_interior_offsets = direct_make_int64(plan.element_interior_offsets, (size_t)element_count + 1u);
    entry_offsets_array = direct_make_int64(entry_offsets, layout.element_dof_count + 1u);
    entry_index_array = direct_make_int64(entry_index, layout.entry_count);
    if (!element_offsets || !element_interior_offsets || !entry_offsets_array || !entry_index_array)
        goto fail_plan;

    cutl_dealloc(&PYTHON_ALLOCATOR, work_memory);
    cutl_dealloc(&PYTHON_ALLOCATOR, scratch_memory);

    result = direct_dof_map_create(state->direct_dof_map_type, &layout, element_offsets, element_interior_offsets,
                                   entry_offsets_array, entry_index_array, entry_value_array);
    element_offsets = NULL;
    element_interior_offsets = NULL;
    entry_offsets_array = NULL;
    entry_index_array = NULL;
    entry_value_array = NULL;
    cutl_dealloc(&PYTHON_ALLOCATOR, memory);
    direct_continuity_plan_release(&plan);
    Py_DECREF(specs_seq);
    return result;

fail_plan:
    if (plan_ready)
        direct_continuity_plan_release(&plan);
    cutl_dealloc(&PYTHON_ALLOCATOR, plan_memory);
    cutl_dealloc(&PYTHON_ALLOCATOR, scratch_memory);

cleanup:
    Py_XDECREF(element_offsets);
    Py_XDECREF(element_interior_offsets);
    Py_XDECREF(entry_offsets_array);
    Py_XDECREF(entry_index_array);
    Py_XDECREF(entry_value_array);
    cutl_dealloc(&PYTHON_ALLOCATOR, memory);
    Py_DECREF(specs_seq);
    return result;
}
