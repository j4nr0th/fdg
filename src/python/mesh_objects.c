#include "mesh_objects.h"
#include "../constraints/constraints.h"
#include "../topology/topology.h"
#include "basis_objects.h"
#include "constraints.h"
#include "cpyutl.h"
#include "cutl/iterators/combination_iterator.h"
#include "kform_objects.h"
#include "mappings.h"
#include "module.h"
#include <numpy/ndarrayobject.h>
#include <string.h>

static PyObject *mesh_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    (void)type;
    (void)args;
    (void)kwds;
    PyErr_SetString(PyExc_TypeError,
                    "Mesh cannot be instantiated directly; use Mesh.from_corners or Mesh.from_collections.");
    return NULL;
}

static void mesh_dealloc(mesh_object *self)
{
    if (self->mesh)
    {
        topo_mesh_free(self->mesh, &SYSTEM_ALLOCATOR);
        self->mesh = NULL;
    }
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyObject *mesh_from_corners(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    Py_ssize_t ndim;
    PyObject *corners_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &ndim},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &corners_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (ndim < 1 || ndim > 63)
    {
        PyErr_Format(PyExc_ValueError, "Expected ndim in [1, 63], got %zd.", ndim);
        return NULL;
    }

    PyArrayObject *const corners_array =
        (PyArrayObject *)PyArray_FROMANY(corners_object, NPY_UINT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (!corners_array)
        return NULL;
    const npy_intp n = PyArray_SIZE(corners_array);
    const npy_intp corners_per_element = (npy_intp)1 << ndim;
    if (n <= 0 || n % corners_per_element != 0)
    {
        PyErr_Format(PyExc_ValueError, "Expected a positive number of corners that is a multiple of 2^%zd, got %zd.",
                     ndim, (Py_ssize_t)n);
        Py_DECREF(corners_array);
        return NULL;
    }
    const uint64_t element_count = (uint64_t)n / (uint64_t)corners_per_element;

    uint64_t max_corner = 0;
    const uint64_t *const data = PyArray_DATA(corners_array);
    for (npy_intp i = 0; i < n; ++i)
    {
        if (data[i] > max_corner)
            max_corner = data[i];
    }
    if (max_corner == UINT64_MAX)
    {
        PyErr_SetString(PyExc_ValueError, "Corner point ID out of range.");
        Py_DECREF(corners_array);
        return NULL;
    }
    const uint64_t point_count = max_corner + 1;

    topo_mesh_t *mesh;
    const topo_status_t topo_status =
        topo_mesh_create_from_corners((unsigned)ndim, element_count, point_count, data, &SYSTEM_ALLOCATOR, &mesh);
    Py_DECREF(corners_array);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not create mesh: %s (%s).", topo_status_to_str(topo_status),
                     topo_status_msg(topo_status));
        return NULL;
    }

    mesh_object *const self = (mesh_object *)((PyTypeObject *)cls)->tp_alloc((PyTypeObject *)cls, 0);
    if (!self)
    {
        topo_mesh_free(mesh, &SYSTEM_ALLOCATOR);
        return NULL;
    }
    self->mesh = mesh;
    return (PyObject *)self;
}

static PyObject *mesh_from_collections(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    Py_ssize_t ndim;
    Py_ssize_t point_count;
    PyObject *collections_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &ndim},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &point_count},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &collections_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (ndim < 1 || ndim > 63)
    {
        PyErr_Format(PyExc_ValueError, "Expected ndim in [1, 63], got %zd.", ndim);
        return NULL;
    }
    if (point_count <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Expected a positive point count, got %zd.", point_count);
        return NULL;
    }
    if (!PyTuple_Check(collections_object) || PyTuple_GET_SIZE(collections_object) != ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected %zd mesh collections.", ndim);
        return NULL;
    }

    topo_obj_collection_t *const collections = cutl_alloc(&SYSTEM_ALLOCATOR, (size_t)ndim * sizeof(*collections));
    if (!collections)
    {
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t idim = 0; idim < ndim; ++idim)
        collections[idim] = (topo_obj_collection_t){0};

    unsigned built = 0;
    for (; built < (unsigned)ndim; ++built)
    {
        PyArrayObject *const array = (PyArrayObject *)PyArray_FROMANY(
            PyTuple_GET_ITEM(collections_object, (Py_ssize_t)built), NPY_UINT64, 2, 2, NPY_ARRAY_IN_ARRAY);
        if (!array)
            break;
        const npy_intp count = PyArray_DIM(array, 0);
        if (PyArray_DIM(array, 1) != 2 * (built + 1))
        {
            PyErr_Format(PyExc_ValueError, "Mesh collection %u must have shape (count, %u).", built, 2 * (built + 1));
            Py_DECREF(array);
            break;
        }
        const size_t ids_count = (size_t)count * 2 * (built + 1);
        uint64_t *const ids = cutl_alloc(&SYSTEM_ALLOCATOR, ids_count * sizeof(*ids));
        if (!ids)
        {
            Py_DECREF(array);
            PyErr_NoMemory();
            break;
        }
        memcpy(ids, PyArray_DATA(array), ids_count * sizeof(*ids));
        Py_DECREF(array);
        collections[built] = (topo_obj_collection_t){.ndim = built + 1, .count = (size_t)count, .boundary_ids = ids};
    }
    if (built != (unsigned)ndim)
    {
        for (unsigned i = 0; i < built; ++i)
            cutl_dealloc(&SYSTEM_ALLOCATOR, (void *)collections[i].boundary_ids);
        cutl_dealloc(&SYSTEM_ALLOCATOR, collections);
        return NULL;
    }

    topo_mesh_t *mesh;
    const topo_status_t topo_status =
        topo_mesh_create_from_collections((unsigned)ndim, (uint64_t)point_count, collections, &SYSTEM_ALLOCATOR, &mesh);
    if (topo_status != TOPO_SUCCESS)
    {
        for (unsigned i = 0; i < (unsigned)ndim; ++i)
            cutl_dealloc(&SYSTEM_ALLOCATOR, (void *)collections[i].boundary_ids);
        cutl_dealloc(&SYSTEM_ALLOCATOR, collections);
        PyErr_Format(PyExc_ValueError, "Could not create mesh: %s (%s).", topo_status_to_str(topo_status),
                     topo_status_msg(topo_status));
        return NULL;
    }

    mesh_object *const self = (mesh_object *)((PyTypeObject *)cls)->tp_alloc((PyTypeObject *)cls, 0);
    if (!self)
    {
        topo_mesh_free(mesh, &SYSTEM_ALLOCATOR);
        return NULL;
    }
    self->mesh = mesh;
    return (PyObject *)self;
}

static PyObject *mesh_get_ndim(const mesh_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(self->mesh->ndim);
}

static PyObject *mesh_get_point_count(const mesh_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLongLong(self->mesh->point_count);
}

static PyObject *mesh_get_element_count(const mesh_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLongLong(self->mesh->element_count);
}

static PyObject *mesh_get_collections(const mesh_object *self, void *Py_UNUSED(closure))
{
    const unsigned ndim = self->mesh->ndim;
    PyObject *const result = PyTuple_New((Py_ssize_t)ndim);
    if (!result)
        return NULL;
    for (unsigned d = 0; d < ndim; ++d)
    {
        const topo_obj_collection_t *const collection = self->mesh->collections + d;
        const npy_intp dims[2] = {(npy_intp)collection->count, (npy_intp)(2 * (d + 1))};
        PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT64);
        if (!array)
        {
            Py_DECREF(result);
            return NULL;
        }
        memcpy(PyArray_DATA(array), collection->boundary_ids,
               (size_t)collection->count * 2 * (d + 1) * sizeof(uint64_t));
        PyTuple_SET_ITEM(result, (Py_ssize_t)d, (PyObject *)array);
    }
    return result;
}

static PyObject *mesh_element_object(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                     const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a Mesh object.");
        return NULL;
    }
    mesh_object *const mesh = (mesh_object *)self;

    Py_ssize_t element_id;
    PyObject *axis_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &axis_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const seq = PySequence_Fast(axis_object, "axis must be a sequence of integers.");
    if (!seq)
        return NULL;
    const unsigned ndim = mesh->mesh->ndim;
    if (PySequence_Fast_GET_SIZE(seq) != (Py_ssize_t)ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected %u axis entries, got %zd.", ndim,
                     (Py_ssize_t)PySequence_Fast_GET_SIZE(seq));
        Py_DECREF(seq);
        return NULL;
    }
    int8_t axis[63];
    unsigned fixed = 0;
    for (unsigned a = 0; a < ndim; ++a)
    {
        const long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, (Py_ssize_t)a));
        if (value == -1 && PyErr_Occurred())
        {
            Py_DECREF(seq);
            return NULL;
        }
        if (value != 0 && value != (long)(a + 1) && value != -(long)(a + 1))
        {
            PyErr_Format(PyExc_ValueError, "Invalid axis entry %ld at index %u; expected 0, %d or %d.", value, a,
                         (int)(a + 1), -(int)(a + 1));
            Py_DECREF(seq);
            return NULL;
        }
        axis[a] = (int8_t)value;
        if (value != 0)
            fixed += 1;
    }
    Py_DECREF(seq);
    if (fixed == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Expected at least one fixed axis.");
        return NULL;
    }

    uint64_t out;
    const topo_status_t topo_status = topo_mesh_element_object(mesh->mesh, (uint64_t)element_id, axis, &out);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Invalid element ID or axis entry: %s (%s).", topo_status_to_str(topo_status),
                     topo_status_msg(topo_status));
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(out);
}

typedef struct
{
    PyObject *list;
    unsigned ndim;
    int error;
} mesh_iterate_collector_t;

static void mesh_iterate_callback(const topo_mesh_t *mesh, const topo_mesh_shared_object_t *object, void *user_data)
{
    (void)mesh;
    mesh_iterate_collector_t *const collector = user_data;
    if (collector->error)
        return;

    const npy_intp element_dims[1] = {(npy_intp)object->element_count};
    PyObject *const element_ids = PyArray_SimpleNew(1, element_dims, NPY_UINT64);
    const npy_intp orientation_dims[2] = {(npy_intp)object->element_count, (npy_intp)collector->ndim};
    PyObject *const orientations = PyArray_SimpleNew(2, orientation_dims, NPY_INT8);
    if (!element_ids || !orientations)
    {
        collector->error = 1;
        Py_XDECREF(element_ids);
        Py_XDECREF(orientations);
        return;
    }
    memcpy(PyArray_DATA((PyArrayObject *)element_ids), object->element_ids,
           (size_t)object->element_count * sizeof(uint64_t));
    memcpy(PyArray_DATA((PyArrayObject *)orientations), object->orientations,
           (size_t)object->element_count * collector->ndim * sizeof(int8_t));

    PyObject *const item = PyTuple_New(4);
    if (!item)
    {
        collector->error = 1;
        Py_DECREF(element_ids);
        Py_DECREF(orientations);
        return;
    }
    PyObject *const mdim_object = PyLong_FromSsize_t((Py_ssize_t)object->mdim);
    PyObject *const id_object = PyLong_FromUnsignedLongLong(object->object_id);
    if (!mdim_object || !id_object)
    {
        collector->error = 1;
        Py_XDECREF(mdim_object);
        Py_XDECREF(id_object);
        Py_DECREF(element_ids);
        Py_DECREF(orientations);
        Py_DECREF(item);
        return;
    }
    PyTuple_SET_ITEM(item, 0, mdim_object);
    PyTuple_SET_ITEM(item, 1, id_object);
    PyTuple_SET_ITEM(item, 2, element_ids);
    PyTuple_SET_ITEM(item, 3, orientations);
    if (PyList_Append(collector->list, item) < 0)
    {
        collector->error = 1;
        Py_DECREF(item);
        return;
    }
    Py_DECREF(item);
}

static PyObject *mesh_iterate_shared(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                     const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a Mesh object.");
        return NULL;
    }
    mesh_object *const mesh = (mesh_object *)self;

    Py_ssize_t mdim;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &mdim},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (mdim < 0 || (unsigned)mdim >= mesh->mesh->ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected mdim in [0, %u), got %zd.", mesh->mesh->ndim, mdim);
        return NULL;
    }

    PyObject *const list = PyList_New(0);
    if (!list)
        return NULL;
    mesh_iterate_collector_t collector = {.list = list, .ndim = mesh->mesh->ndim, .error = 0};
    const topo_status_t topo_status =
        topo_mesh_iterate_shared(mesh->mesh, (unsigned)mdim, mesh_iterate_callback, &collector);
    if (topo_status != TOPO_SUCCESS)
    {
        Py_DECREF(list);
        PyErr_Format(PyExc_ValueError, "Could not iterate over shared objects: %s (%s).",
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        return NULL;
    }
    if (collector.error)
    {
        Py_DECREF(list);
        return NULL;
    }
    return list;
}

static PyObject *mesh_iterate_shared_all(PyObject *self, PyTypeObject *defining_class, PyObject *const *Py_UNUSED(args),
                                         const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a Mesh object.");
        return NULL;
    }
    mesh_object *const mesh = (mesh_object *)self;
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "iterate_shared_all() takes no arguments.");
        return NULL;
    }

    PyObject *const list = PyList_New(0);
    if (!list)
        return NULL;
    mesh_iterate_collector_t collector = {.list = list, .ndim = mesh->mesh->ndim, .error = 0};
    const topo_status_t topo_status = topo_mesh_iterate_shared_all(mesh->mesh, mesh_iterate_callback, &collector);
    if (topo_status != TOPO_SUCCESS)
    {
        Py_DECREF(list);
        PyErr_Format(PyExc_ValueError, "Could not iterate over shared objects: %s (%s).",
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        return NULL;
    }
    if (collector.error)
    {
        Py_DECREF(list);
        return NULL;
    }
    return list;
}

static PyObject *mesh_iterate_boundary(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                       const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a Mesh object.");
        return NULL;
    }
    mesh_object *const mesh = (mesh_object *)self;

    Py_ssize_t mdim;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &mdim},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (mdim < 0 || (unsigned)mdim >= mesh->mesh->ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected mdim in [0, %u), got %zd.", mesh->mesh->ndim, mdim);
        return NULL;
    }

    PyObject *const list = PyList_New(0);
    if (!list)
        return NULL;
    mesh_iterate_collector_t collector = {.list = list, .ndim = mesh->mesh->ndim, .error = 0};
    const topo_status_t topo_status =
        topo_mesh_iterate_boundary(mesh->mesh, (unsigned)mdim, mesh_iterate_callback, &collector);
    if (topo_status != TOPO_SUCCESS)
    {
        Py_DECREF(list);
        PyErr_Format(PyExc_ValueError, "Could not iterate over boundary objects: %s (%s).",
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        return NULL;
    }
    if (collector.error)
    {
        Py_DECREF(list);
        return NULL;
    }
    return list;
}

static PyObject *mesh_iterate_boundary_all(PyObject *self, PyTypeObject *defining_class,
                                           PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a Mesh object.");
        return NULL;
    }
    mesh_object *const mesh = (mesh_object *)self;
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "iterate_boundary_all() takes no arguments.");
        return NULL;
    }

    PyObject *const list = PyList_New(0);
    if (!list)
        return NULL;
    mesh_iterate_collector_t collector = {.list = list, .ndim = mesh->mesh->ndim, .error = 0};
    const topo_status_t topo_status = topo_mesh_iterate_boundary_all(mesh->mesh, mesh_iterate_callback, &collector);
    if (topo_status != TOPO_SUCCESS)
    {
        Py_DECREF(list);
        PyErr_Format(PyExc_ValueError, "Could not iterate over boundary objects: %s (%s).",
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        return NULL;
    }
    if (collector.error)
    {
        Py_DECREF(list);
        return NULL;
    }
    return list;
}

typedef struct
{
    size_t row_count;
    size_t entry_count;
    size_t row_capacity;
    size_t element_ids_capacity;
    size_t components_capacity;
    size_t local_dofs_capacity;
    size_t coefficients_capacity;
    size_t *row_offsets;
    uint64_t *element_ids;
    uint32_t *components;
    size_t *local_dofs;
    double *coefficients;
} mesh_continuity_builder_t;

static void mesh_continuity_builder_release(mesh_continuity_builder_t *const builder)
{
    PyMem_Free(builder->row_offsets);
    PyMem_Free(builder->element_ids);
    PyMem_Free(builder->components);
    PyMem_Free(builder->local_dofs);
    PyMem_Free(builder->coefficients);
    *builder = (mesh_continuity_builder_t){};
}

static int mesh_continuity_builder_grow(void **const pointer, size_t *const capacity, const size_t needed,
                                        const size_t element_size)
{
    if (needed <= *capacity)
        return 0;
    size_t new_capacity = *capacity == 0 ? 16 : *capacity;
    while (new_capacity < needed)
    {
        if (new_capacity > SIZE_MAX / 2)
        {
            new_capacity = needed;
            break;
        }
        new_capacity *= 2;
    }
    size_t bytes;
    if (__builtin_mul_overflow(new_capacity, element_size, &bytes))
    {
        PyErr_NoMemory();
        return -1;
    }
    void *const grown = PyMem_Realloc(*pointer, bytes);
    if (!grown)
    {
        PyErr_NoMemory();
        return -1;
    }
    *pointer = grown;
    *capacity = new_capacity;
    return 0;
}

static int mesh_continuity_builder_append_row(mesh_continuity_builder_t *const builder, const uint64_t element_id,
                                              const uint32_t component, const size_t local_dof,
                                              const double coefficient)
{
    if (builder->entry_count == SIZE_MAX)
    {
        PyErr_NoMemory();
        return -1;
    }
    const size_t entry = builder->entry_count;
    if (mesh_continuity_builder_grow((void **)&builder->element_ids, &builder->element_ids_capacity, entry + 1,
                                     sizeof(*builder->element_ids)) < 0)
        return -1;
    if (mesh_continuity_builder_grow((void **)&builder->components, &builder->components_capacity, entry + 1,
                                     sizeof(*builder->components)) < 0)
        return -1;
    if (mesh_continuity_builder_grow((void **)&builder->local_dofs, &builder->local_dofs_capacity, entry + 1,
                                     sizeof(*builder->local_dofs)) < 0)
        return -1;
    if (mesh_continuity_builder_grow((void **)&builder->coefficients, &builder->coefficients_capacity, entry + 1,
                                     sizeof(*builder->coefficients)) < 0)
        return -1;
    builder->element_ids[entry] = element_id;
    builder->components[entry] = component;
    builder->local_dofs[entry] = local_dof;
    builder->coefficients[entry] = coefficient;
    builder->entry_count = entry + 1;
    return 0;
}

static int mesh_continuity_builder_finish_row(mesh_continuity_builder_t *const builder)
{
    if (builder->row_count > SIZE_MAX - 2)
    {
        PyErr_NoMemory();
        return -1;
    }
    if (mesh_continuity_builder_grow((void **)&builder->row_offsets, &builder->row_capacity, builder->row_count + 2,
                                     sizeof(*builder->row_offsets)) < 0)
        return -1;
    builder->row_count += 1;
    builder->row_offsets[builder->row_count] = builder->entry_count;
    return 0;
}

typedef struct
{
    const interplib_module_state_t *state;
    unsigned ndim;
    unsigned order;
    kform_spec_object **element_specs;
    space_map_object **element_maps;
    int c1_continuous;
    int failed;
    mesh_continuity_builder_t builder;
} mesh_continuity_context_t;

static void mesh_continuity_context_release(mesh_continuity_context_t *const context)
{
    PyMem_Free(context->element_specs);
    PyMem_Free(context->element_maps);
    mesh_continuity_builder_release(&context->builder);
    *context = (mesh_continuity_context_t){};
}

/** Per-element physical factors sampled at the element's own face grid. */
typedef struct
{
    boundary_face_setup_t setup;
    PyArrayObject *transform;
    double *pullback_values;
    double *surface_weights;
    constraint_trace_pullback_t pullback;
} mesh_continuity_face_factors_t;

/**
 * @brief Assemble one shared object's constraint rows on the table engine.
 *
 * The incident elements are linked to the lowest-ID anchor element through
 * star rows: one row per non-anchor element and common test DoF carries
 * that element's trace moments with a -1 side and the anchor's with a +1
 * side. The common Legendre boundary space takes the lowest per-axis order
 * among the incident elements (an element boundary cannot be constrained
 * to a higher-order boundary solution), with order-1 test tables on active
 * covector axes and the full tables minus the two lowest functions on
 * inactive axes, so only the object's own block is constrained. Mapped
 * meshes sample each face's surface measure and k-form pullback at the
 * element's own face grid, which the merge reproduces exactly when all
 * faces share one geometry sampling order.
 */
static int mesh_continuity_assemble_object(mesh_continuity_context_t *const context, const unsigned bdim,
                                           const uint64_t element_count, const uint64_t *const element_ids,
                                           const int8_t *const element_orientations)
{
    const interplib_module_state_t *const state = context->state;
    const unsigned ndim = context->ndim;
    const unsigned order = context->order;
    const int physical = !context->c1_continuous;
    const unsigned nelem = (unsigned)element_count;
    const int8_t **orientations = NULL;
    kform_spec_object **element_specs = NULL;
    double *side_signs = NULL;

    mesh_continuity_face_factors_t *factors = NULL;
    PyArrayObject **transforms = NULL;
    double **pullback_values = NULL;
    double **element_pullback_values = NULL;
    double **surface_weights = NULL;
    constraint_trace_pullback_t *pullbacks = NULL;
    constraint_trace_pullback_t *element_pullbacks = NULL;
    integration_spec_t *element_integrations = NULL;
    boundary_element_space_t *views = NULL;
    constrain_elements_on_boundary_plan_t plan;
    constrain_elements_on_boundary_work_t work = {0};
    const double **surface_rows = NULL;
    const constraint_trace_pullback_t **test_pullback_pointers = NULL;
    const constraint_trace_pullback_t **element_pullback_pointers = NULL;
    uint8_t **pack_sides = NULL;
    uint32_t **pack_components = NULL;
    size_t **pack_dofs = NULL;
    double **pack_coefficients = NULL;
    size_t **pack_offsets = NULL;
    double *arena = NULL;
    double *surface_block = NULL;
    double *pullback_block = NULL;
    void *rows_memory = NULL;
    void *build_memory = NULL;
    void *weights_memory = NULL;
    constraint_trace_pullback_build_work_t canonical_work;
    constraint_trace_pullback_build_work_t element_work;
    int plan_live = 0;
    int failed = 1;

    ASSERT(bdim < ndim, "Shared-object dimension must stay below the element dimension.");
    if (order > bdim)
    {
        // A form of order past the object dimension has no trace components
        // and yields no rows.
        return 0;
    }
    // Plain per-element arrays share one allocation group; arrays whose
    // cleanup dereferences possibly-unfilled slots stay separately zeroed.
    CUTL_ASSERT(nelem > 1 && nelem <= UINT8_MAX, "Shared objects need two or more incident elements.");
    factors = PyMem_Calloc(nelem, sizeof(*factors));
    transforms = PyMem_Calloc(nelem, sizeof(*transforms));
    pullback_values = PyMem_Calloc(nelem, sizeof(*pullback_values));
    element_pullback_values = PyMem_Calloc(nelem, sizeof(*element_pullback_values));
    surface_weights = PyMem_Calloc(nelem, sizeof(*surface_weights));
    pullbacks = PyMem_Calloc(nelem, sizeof(*pullbacks));
    element_pullbacks = PyMem_Calloc(nelem, sizeof(*element_pullbacks));
    pack_sides = PyMem_Calloc(nelem, sizeof(*pack_sides));
    pack_components = PyMem_Calloc(nelem, sizeof(*pack_components));
    pack_dofs = PyMem_Calloc(nelem, sizeof(*pack_dofs));
    pack_coefficients = PyMem_Calloc(nelem, sizeof(*pack_coefficients));
    pack_offsets = PyMem_Calloc(nelem, sizeof(*pack_offsets));
    void *const head_memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){{sizeof(*orientations) * nelem, (void **)&orientations},
                                    {sizeof(*element_specs) * nelem, (void **)&element_specs},
                                    {sizeof(*side_signs) * nelem, (void **)&side_signs},
                                    {sizeof(*views) * nelem, (void **)&views},
                                    {sizeof(*surface_rows) * nelem, (void **)&surface_rows},
                                    {sizeof(*test_pullback_pointers) * nelem, (void **)&test_pullback_pointers},
                                    {sizeof(*element_pullback_pointers) * nelem, (void **)&element_pullback_pointers},
                                    {}});
    if (!head_memory || !factors || !transforms || !pullback_values || !element_pullback_values || !surface_weights ||
        !pullbacks || !element_pullbacks || !pack_sides || !pack_components || !pack_dofs || !pack_coefficients ||
        !pack_offsets)
    {
        PyErr_NoMemory();
        goto out;
    }
    for (unsigned e = 0; e < nelem; ++e)
    {
        orientations[e] = element_orientations + (size_t)e * ndim;
        element_specs[e] = context->element_specs[element_ids[e]];
        side_signs[e] = e == 0 ? 1.0 : -1.0;
    }

    if (bdim == 0)
    {
        // Point objects: scalar continuity degenerates to pairing the
        // elements' corner value functionals. The endpoint sets hold the
        // reference basis values at the interval ends, and the orientation
        // record selects the shared vertex's corner per element. No
        // geometry enters: reference-space corner coupling matches the
        // historical row semantics. Star rows link every non-anchor
        // element to the anchor.
        // TODO: should be a parameter, not the global default.
        basis_set_registry_t *const basis_registry = ((basis_registry_object *)state->registry_basis)->registry;
        const basis_endpoint_set_t **endpoints = PyMem_Malloc((size_t)nelem * ndim * sizeof(*endpoints));
        if (!endpoints)
        {
            PyErr_NoMemory();
            goto out;
        }
        for (unsigned e = 0; e < nelem; ++e)
        {
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                const int8_t mapping = orientations[e][axis];
                ASSERT(mapping != 0, "Orientation records must be one-based and nonzero.");
                if (basis_set_registry_get_basis_endpoints(basis_registry, &endpoints[e * ndim + axis],
                                                           element_specs[e]->function_space->specs[axis]) !=
                    FDG_SUCCESS)
                {
                    PyErr_SetString(PyExc_RuntimeError, "Could not fetch basis endpoint values.");
                    goto out;
                }
            }
        }
        for (unsigned side = 1; side < nelem; ++side)
        {
            const unsigned sides[2] = {0, side};
            for (unsigned s = 0; s < 2; ++s)
            {
                const unsigned e = sides[s];
                unsigned digits[UINT8_MAX];
                size_t dof_count = 1;
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    digits[axis] = 0;
                    dof_count *= (size_t)element_specs[e]->function_space->specs[axis].order + 1u;
                }
                for (size_t dof = 0; dof < dof_count; ++dof)
                {
                    double coefficient = side_signs[e];
                    for (unsigned axis = 0; axis < ndim; ++axis)
                    {
                        const unsigned end = orientations[e][axis] < 0 ? 0u : 1u;
                        coefficient *= basis_endpoint_values(endpoints[e * ndim + axis], end)[digits[axis]];
                    }
                    if (mesh_continuity_builder_append_row(&context->builder, element_ids[e], 0, dof, coefficient) < 0)
                        goto out;
                    for (unsigned axis = ndim; axis-- > 0;)
                    {
                        if (++digits[axis] < (unsigned)element_specs[e]->function_space->specs[axis].order + 1u)
                            break;
                        digits[axis] = 0;
                    }
                }
            }
            if (mesh_continuity_builder_finish_row(&context->builder) < 0)
                goto out;
        }
        for (unsigned e = 0; e < nelem; ++e)
        {
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                basis_set_registry_release_basis_endpoints(basis_registry, endpoints[e * ndim + axis]);
            }
        }
        PyMem_Free(endpoints);
        return 0;
    }
    const size_t component_count = combination_total_count((uint8_t)bdim, (uint8_t)order);
    const size_t order_storage = order == 0 ? 1u : order;
    const size_t axis_items = (size_t)nelem * ndim;

    uint8_t *axis_skip;
    basis_spec_t *out_basis;
    integration_spec_t *out_integration;
    void *const core_memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {sizeof(*element_integrations) * axis_items, (void **)&element_integrations},
            {sizeof(*axis_skip) * bdim, (void **)&axis_skip},
            {sizeof(*out_basis) * bdim, (void **)&out_basis},
            {sizeof(*out_integration) * bdim, (void **)&out_integration},
            {sizeof(*work.axis_fixed) * ndim, (void **)&work.axis_fixed},
            {sizeof(*work.axis_slot) * ndim, (void **)&work.axis_slot},
            {sizeof(*work.element_rules) * ndim, (void **)&work.element_rules},
            {sizeof(*work.mass.point_strides) * bdim, (void **)&work.mass.point_strides},
            {sizeof(*work.mass.row_offsets) * (component_count + 1), (void **)&work.mass.row_offsets},
            {sizeof(*work.mass.col_offsets) * (component_count + 1), (void **)&work.mass.col_offsets},
            {sizeof(*work.mass.element_components) * component_count, (void **)&work.mass.element_components},
            {sizeof(*work.mass.element_signs) * component_count, (void **)&work.mass.element_signs},
            {sizeof(*work.mass.axes) * ndim, (void **)&work.mass.axes},
            {sizeof(*work.mass.counts) * bdim, (void **)&work.mass.counts},
            {sizeof(*work.mass.offsets) * bdim, (void **)&work.mass.offsets},
            {sizeof(*work.mass.axis_sets) * bdim, (void **)&work.mass.axis_sets},
            {sizeof(*work.mass.digits) * bdim, (void **)&work.mass.digits},
            {sizeof(*work.mass.axis_tables) * bdim, (void **)&work.mass.axis_tables},
            {sizeof(*work.mass.mapped_axes) * order_storage, (void **)&work.mass.mapped_axes},
            {combination_iterator_required_memory((uint8_t)order), (void **)&work.mass.components},
            {combination_iterator_required_memory((uint8_t)order), (void **)&work.mass.blocks},
            {sizeof(*plan.rules) * bdim, (void **)&plan.rules},
            {sizeof(*plan.boundary_sets) * bdim, (void **)&plan.boundary_sets},
            {sizeof(*plan.boundary_sets_lower) * bdim, (void **)&plan.boundary_sets_lower},
            {sizeof(*plan.boundary_lower_specs) * bdim, (void **)&plan.boundary_lower_specs},
            {sizeof(*plan.element_sets) * axis_items, (void **)&plan.element_sets},
            {sizeof(*plan.element_sets_lower) * axis_items, (void **)&plan.element_sets_lower},
            {sizeof(*plan.element_endpoints) * axis_items, (void **)&plan.element_endpoints},
            {sizeof(*plan.element_endpoints_lower) * axis_items, (void **)&plan.element_endpoints_lower},
            {sizeof(*plan.element_lower_specs) * axis_items, (void **)&plan.element_lower_specs},
            {sizeof(*plan.item_rows) * nelem, (void **)&plan.item_rows},
            {sizeof(*plan.item_cols) * nelem, (void **)&plan.item_cols},
            {sizeof(*plan.item_offsets) * (nelem + 1u), (void **)&plan.item_offsets},
            {}});
    if (!core_memory)
    {
        PyErr_NoMemory();
        goto out;
    }
    // Own-block windows: inactive axes drop their two lowest test
    // functions, the endpoint functionals owned by the lower-dimensional
    // subobjects whose constraints already tie them. An order-one axis
    // therefore empties and the component's row block drops out — at the
    // lowest order only objects with an active axis of their exact form
    // order keep rows. Active axes ignore the skip and read the
    // order-minus-one basis.
    for (unsigned slot = 0; slot < bdim; ++slot)
    {
        axis_skip[slot] = 2;
    }

    // Per-element views: mapped meshes integrate at each face's own grid,
    // C1 meshes at a reference rule exact for the traced products.
    for (unsigned e = 0; e < nelem; ++e)
    {
        if (physical)
        {
            if (make_boundary_face_setup(state, context->element_maps[element_ids[e]], orientations[e], ndim, bdim,
                                         &factors[e].setup) < 0)
                goto out;
            const integration_spec_t *const canonical_specs = factors[e].setup.canonical_specs;
            for (unsigned slot = 0; slot < bdim; ++slot)
            {
                const int8_t mapping = orientations[e][ndim - bdim + slot];
                const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                element_integrations[e * ndim + element_axis] = canonical_specs[slot];
            }
            for (unsigned fixed_axis = 0; fixed_axis < ndim - bdim; ++fixed_axis)
            {
                const int8_t mapping = orientations[e][fixed_axis];
                const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                element_integrations[e * ndim + element_axis] = canonical_specs[0];
            }
        }
        else
        {
            // The C1 reference rule must resolve the traced test products:
            // row test functions reach the merged basis degree and element
            // traces the element degree, so order the rule one above the
            // largest involved basis order. Tying it to the form order
            // under-integrates: a form order 1 rule vanishes the degree-2
            // Legendre test functions at its nodes and emits quadrature null
            // rows.
            unsigned slot_orders[UINT8_MAX];
            for (unsigned slot = 0; slot < bdim; ++slot)
            {
                unsigned max_order = order;
                for (unsigned other = 0; other < nelem; ++other)
                {
                    const int8_t mapping = orientations[other][ndim - bdim + slot];
                    ASSERT(mapping != 0, "Orientation records must be one-based and nonzero.");
                    const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                    ASSERT(element_axis < ndim, "Face slot maps outside the element axes.");
                    const unsigned axis_order = element_specs[other]->function_space->specs[element_axis].order;
                    max_order = max_order > axis_order ? max_order : axis_order;
                }
                slot_orders[slot] = max_order + 1u;
            }
            for (unsigned slot = 0; slot < bdim; ++slot)
            {
                const int8_t mapping = orientations[e][ndim - bdim + slot];
                const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                ASSERT(element_axis < ndim, "Face slot maps outside the element axes.");
                element_integrations[e * ndim + element_axis] =
                    (integration_spec_t){.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = slot_orders[slot]};
            }
            for (unsigned fixed_axis = 0; fixed_axis < ndim - bdim; ++fixed_axis)
            {
                const int8_t mapping = orientations[e][fixed_axis];
                const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                ASSERT(element_axis < ndim, "Fixed axis maps outside the element axes.");
                // Fixed normal axes read endpoint values; the rule order is
                // irrelevant, so keep the first face slot's order.
                element_integrations[e * ndim + element_axis] =
                    (integration_spec_t){.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = slot_orders[0]};
            }
        }
        views[e] = (boundary_element_space_t){.order = order,
                                              .orientation = orientations[e],
                                              .basis = element_specs[e]->function_space->specs,
                                              .integration = element_integrations + e * ndim};
    }

    constrain_elements_on_boundary_request_t request = {
        .ndim = ndim,
        .bdim = bdim,
        .nforms = 1,
        .nelem = nelem,
        .elements = views,
        .axis_skip = axis_skip,
        .c1_continuous = context->c1_continuous,
        .shared_face_guard = true,
        .surface_weights = NULL,
        .test_pullbacks = NULL,
        .element_pullbacks = NULL,
        .basis_registry = ((basis_registry_object *)state->registry_basis)->registry,
        .integration_registry = ((integration_registry_object *)state->registry_integration)->registry};
    fdg_result_t res = constrain_elements_on_boundary_prepare(&request, &work, out_basis, out_integration, &plan);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not prepare continuity constraints: %s (%s)", fdg_error_str(res),
                     fdg_error_msg(res));
        goto out;
    }
    plan_live = 1;

    // Physical factors at the common frame: the surface measure and the
    // k-form pullback of each face. Both faces must share one geometry
    // sampling order so their sampled factors permute onto the merged
    // rules exactly.
    if (physical)
    {
        for (unsigned e = 0; e < nelem; ++e)
        {
            for (unsigned slot = 0; slot < bdim; ++slot)
            {
                if (factors[e].setup.canonical_specs[slot].order != out_integration[slot].order ||
                    factors[e].setup.canonical_specs[slot].type != out_integration[slot].type)
                {
                    PyErr_SetString(PyExc_ValueError,
                                    "Shared faces with differing geometry sampling orders are not supported.");
                    goto out;
                }
            }
        }
        const unsigned physical_component_count =
            (unsigned)combination_total_count((uint8_t)Py_SIZE(context->element_maps[element_ids[0]]), (uint8_t)order);
        const size_t pullback_row =
            (size_t)combination_total_count((uint8_t)ndim, (uint8_t)order) * physical_component_count;
        // One shared work per build variant: the sizes depend only on the
        // dimensions and the component indexing mode, not on the element.
        const constraint_trace_pullback_build_t canonical_template = {
            .element_dim = ndim, .face_dim = bdim, .order = order, .canonical_components = true};
        const constraint_trace_pullback_build_t element_template = {
            .element_dim = ndim, .face_dim = bdim, .order = order, .element_components = true};
        constraint_trace_pullback_build_work_sizes_t canonical_sizes;
        constraint_trace_pullback_build_work_sizes_t element_sizes;
        constraint_trace_pullback_build_work_size(&canonical_template, &canonical_sizes);
        constraint_trace_pullback_build_work_size(&element_template, &element_sizes);
        size_t surface_total = 0;
        size_t pullback_total = 0;
        for (unsigned e = 0; e < nelem; ++e)
        {
            surface_total += factors[e].setup.point_count;
            pullback_total += order > 0 ? 2u * pullback_row * factors[e].setup.point_count : 0u;
        }
        rows_memory = cutl_alloc_group(
            &PYTHON_ALLOCATOR,
            (const cutl_alloc_info_t[]){{sizeof(*surface_block) * surface_total, (void **)&surface_block},
                                        {sizeof(*pullback_block) * pullback_total, (void **)&pullback_block},
                                        {}});
        build_memory = cutl_alloc_group(
            &PYTHON_ALLOCATOR,
            (const cutl_alloc_info_t[]){
                {sizeof(*canonical_work.axis_source_slots) * canonical_sizes.face_axis_count,
                 (void **)&canonical_work.axis_source_slots},
                {sizeof(*canonical_work.axis_orders) * canonical_sizes.face_axis_count,
                 (void **)&canonical_work.axis_orders},
                {sizeof(*canonical_work.axis_source_strides) * canonical_sizes.face_axis_count,
                 (void **)&canonical_work.axis_source_strides},
                {sizeof(*canonical_work.axis_canonical_strides) * canonical_sizes.face_axis_count,
                 (void **)&canonical_work.axis_canonical_strides},
                {sizeof(*canonical_work.axis_mirrored) * canonical_sizes.face_axis_count,
                 (void **)&canonical_work.axis_mirrored},
                {sizeof(*canonical_work.element_axis_free) * canonical_sizes.element_axis_count,
                 (void **)&canonical_work.element_axis_free},
                {sizeof(*canonical_work.element_source_rank) * canonical_sizes.element_axis_count,
                 (void **)&canonical_work.element_source_rank},
                {sizeof(*canonical_work.element_to_face) * canonical_sizes.element_component_map,
                 (void **)&canonical_work.element_to_face},
                {sizeof(*canonical_work.mapped_axes) * canonical_sizes.axes_scratch,
                 (void **)&canonical_work.mapped_axes},
                {sizeof(*canonical_work.source_axes) * canonical_sizes.axes_scratch,
                 (void **)&canonical_work.source_axes},
                {canonical_sizes.iterator_memory, (void **)&canonical_work.components},
                {sizeof(*element_work.axis_source_slots) * element_sizes.face_axis_count,
                 (void **)&element_work.axis_source_slots},
                {sizeof(*element_work.axis_orders) * element_sizes.face_axis_count, (void **)&element_work.axis_orders},
                {sizeof(*element_work.axis_source_strides) * element_sizes.face_axis_count,
                 (void **)&element_work.axis_source_strides},
                {sizeof(*element_work.axis_canonical_strides) * element_sizes.face_axis_count,
                 (void **)&element_work.axis_canonical_strides},
                {sizeof(*element_work.axis_mirrored) * element_sizes.face_axis_count,
                 (void **)&element_work.axis_mirrored},
                {sizeof(*element_work.element_axis_free) * element_sizes.element_axis_count,
                 (void **)&element_work.element_axis_free},
                {sizeof(*element_work.element_source_rank) * element_sizes.element_axis_count,
                 (void **)&element_work.element_source_rank},
                {sizeof(*element_work.element_to_face) * element_sizes.element_component_map,
                 (void **)&element_work.element_to_face},
                {sizeof(*element_work.mapped_axes) * element_sizes.axes_scratch, (void **)&element_work.mapped_axes},
                {sizeof(*element_work.source_axes) * element_sizes.axes_scratch, (void **)&element_work.source_axes},
                {element_sizes.iterator_memory, (void **)&element_work.components},
                {}});
        if (!rows_memory || !build_memory)
        {
            PyErr_NoMemory();
            goto out;
        }
        size_t surface_offset = 0;
        size_t pullback_offset = 0;
        for (unsigned e = 0; e < nelem; ++e)
        {
            const boundary_face_setup_t *const setup = &factors[e].setup;
            space_map_object *const face_map = setup->face_map;
            surface_weights[e] = surface_block + surface_offset;
            surface_offset += setup->point_count;
            for (size_t point = 0; point < setup->point_count; ++point)
            {
                const size_t source_point = constraint_face_point_to_source(
                    ndim, bdim, orientations[e], face_map->int_specs, setup->canonical_specs, setup->canonical_strides,
                    setup->source_strides, point);
                surface_weights[e][point] = fabs(face_map->determinant[source_point]);
            }
            if (order > 0)
            {
                transforms[e] = compute_basis_transform_impl(face_map, (Py_ssize_t)order);
                if (!transforms[e])
                {
                    PyErr_NoMemory();
                    goto out;
                }
                pullback_values[e] = pullback_block + pullback_offset;
                pullback_offset += pullback_row * setup->point_count;
                element_pullback_values[e] = pullback_block + pullback_offset;
                pullback_offset += pullback_row * setup->point_count;
                const constraint_trace_pullback_build_t build = {
                    .element_dim = ndim,
                    .face_dim = bdim,
                    .order = order,
                    .face_component_count = (unsigned)component_count,
                    .physical_component_count = physical_component_count,
                    .source_point_count = integration_specs_total_points(bdim, face_map->int_specs),
                    .canonical_point_count = setup->point_count,
                    .source_strides = setup->source_strides,
                    .canonical_strides = setup->canonical_strides,
                    .orientation = orientations[e],
                    .source_specs = face_map->int_specs,
                    .canonical_specs = setup->canonical_specs,
                    .transform = (const double *)PyArray_DATA(transforms[e]),
                    .out = pullback_values[e],
                    .canonical_components = true,
                    .work = &canonical_work};
                constraint_trace_pullback_build(&build);
                const constraint_trace_pullback_build_t element_build = {
                    .element_dim = build.element_dim,
                    .face_dim = build.face_dim,
                    .order = build.order,
                    .face_component_count = build.face_component_count,
                    .physical_component_count = build.physical_component_count,
                    .source_point_count = build.source_point_count,
                    .canonical_point_count = build.canonical_point_count,
                    .source_strides = build.source_strides,
                    .canonical_strides = build.canonical_strides,
                    .orientation = build.orientation,
                    .source_specs = build.source_specs,
                    .canonical_specs = build.canonical_specs,
                    .transform = build.transform,
                    .out = element_pullback_values[e],
                    .element_components = true,
                    .work = &element_work};
                constraint_trace_pullback_build(&element_build);
                pullbacks[e] = (constraint_trace_pullback_t){.physical_component_count = physical_component_count,
                                                             .point_count = setup->point_count,
                                                             .values = pullback_values[e]};
                element_pullbacks[e] =
                    (constraint_trace_pullback_t){.physical_component_count = physical_component_count,
                                                  .point_count = setup->point_count,
                                                  .values = element_pullback_values[e]};
            }
        }
        for (unsigned e = 0; e < nelem; ++e)
        {
            surface_rows[e] = surface_weights[e];
            test_pullback_pointers[e] = &pullbacks[e];
            element_pullback_pointers[e] = &element_pullbacks[e];
        }
        request.surface_weights = surface_rows;
        request.test_pullbacks = order > 0 ? test_pullback_pointers : NULL;
        request.element_pullbacks = order > 0 ? element_pullback_pointers : NULL;
    }

    size_t weights_size;
    size_t row_values_size;
    size_t col_values_size;
    constrain_elements_on_boundary_work_size(&request, &plan, &weights_size, &row_values_size, &col_values_size);
    weights_memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){
                               {sizeof(*work.weights) * weights_size, (void **)&work.weights},
                               {sizeof(*work.mass.row_values) * row_values_size, (void **)&work.mass.row_values},
                               {sizeof(*work.mass.col_values) * col_values_size, (void **)&work.mass.col_values},
                               {sizeof(*work.mass.point_factors) * weights_size, (void **)&work.mass.point_factors},
                               {}});
    if (!weights_memory)
    {
        PyErr_NoMemory();
        goto out;
    }
    arena = PyMem_Malloc(plan.total_values * sizeof(*arena));
    if (!arena)
    {
        PyErr_NoMemory();
        goto out;
    }

    constrain_elements_on_boundary_assemble(&request, &plan, &work, arena);

    // Pack both element matrices with the alternating side signs, then emit
    // ONE builder row per common test DoF holding side 0's entries followed
    // by side 1's.
    const int coupled = physical && order > 0;
    size_t rows = 0;
    for (unsigned e = 0; e < nelem; ++e)
    {
        const kform_spec_t element_descriptor = {.ndim = ndim, .order = order, .basis = views[e].basis};
        const constraint_boundary_mass_spec_t spec = {.ndim = ndim,
                                                      .bdim = bdim,
                                                      .order = order,
                                                      .element_spec = &element_descriptor,
                                                      .boundary_basis = out_basis,
                                                      .boundary_integration = out_integration,
                                                      .orientation = orientations[e],
                                                      .axis_skip = axis_skip};
        size_t rows_e;
        size_t cols_e;
        size_t entries;
        constraint_boundary_mass_layout(&spec, &work.mass, coupled, &rows_e, &cols_e, &entries);
        pack_sides[e] = PyMem_Malloc(entries * sizeof(*pack_sides[e]));
        pack_components[e] = PyMem_Malloc(entries * sizeof(*pack_components[e]));
        pack_dofs[e] = PyMem_Malloc(entries * sizeof(*pack_dofs[e]));
        pack_coefficients[e] = PyMem_Malloc(entries * sizeof(*pack_coefficients[e]));
        pack_offsets[e] = PyMem_Malloc((rows_e + 1) * sizeof(*pack_offsets[e]));
        if (!pack_sides[e] || !pack_components[e] || !pack_dofs[e] || !pack_coefficients[e] || !pack_offsets[e])
        {
            PyErr_NoMemory();
            goto out;
        }
        constraint_boundary_mass_pack(&spec, &work.mass, coupled, arena + plan.item_offsets[e], plan.item_cols[e],
                                      side_signs[e], (uint8_t)e, pack_sides[e], pack_components[e], pack_dofs[e],
                                      pack_coefficients[e], pack_offsets[e]);
        if (e == 0)
        {
            rows = rows_e;
        }
        else
        {
            CUTL_ASSERT(rows_e == rows, "The sides' test spaces disagree on the row count.");
        }
    }
    CUTL_ASSERT(plan.item_rows[0] == rows, "The prepared row count disagrees with the packed rows.");
    // Star rows: one builder row per (test DoF, non-anchor element) linking
    // that element's trace moments to the anchor's.
    for (size_t row = 0; row < rows; ++row)
    {
        for (unsigned side = 1; side < nelem; ++side)
        {
            const unsigned sides[2] = {0, side};
            for (unsigned s = 0; s < 2; ++s)
            {
                const unsigned e = sides[s];
                for (size_t entry = pack_offsets[e][row]; entry < pack_offsets[e][row + 1]; ++entry)
                {
                    if (mesh_continuity_builder_append_row(&context->builder, element_ids[e], pack_components[e][entry],
                                                           pack_dofs[e][entry], pack_coefficients[e][entry]) < 0)
                        goto out;
                }
            }
            if (mesh_continuity_builder_finish_row(&context->builder) < 0)
                goto out;
        }
    }
    failed = 0;

out:
    if (plan_live)
    {
        constrain_elements_on_boundary_plan_release(&plan);
    }
    for (unsigned e = 0; e < nelem; ++e)
    {
        Py_XDECREF(transforms[e]);
        if (physical && factors[e].setup.face_object != NULL)
        {
            release_boundary_face_setup(state, bdim, &factors[e].setup);
        }
        PyMem_Free(pack_sides[e]);
        PyMem_Free(pack_components[e]);
        PyMem_Free(pack_dofs[e]);
        PyMem_Free(pack_coefficients[e]);
        PyMem_Free(pack_offsets[e]);
    }
    PyMem_Free(arena);
    cutl_dealloc(&PYTHON_ALLOCATOR, weights_memory);
    cutl_dealloc(&PYTHON_ALLOCATOR, build_memory);
    cutl_dealloc(&PYTHON_ALLOCATOR, rows_memory);
    cutl_dealloc(&PYTHON_ALLOCATOR, core_memory);
    cutl_dealloc(&PYTHON_ALLOCATOR, head_memory);
    PyMem_Free(pack_offsets);
    PyMem_Free(pack_coefficients);
    PyMem_Free(pack_dofs);
    PyMem_Free(pack_components);
    PyMem_Free(pack_sides);
    PyMem_Free(element_pullbacks);
    PyMem_Free(pullbacks);
    PyMem_Free(surface_weights);
    PyMem_Free(element_pullback_values);
    PyMem_Free(pullback_values);
    PyMem_Free(transforms);
    PyMem_Free(factors);
    return failed ? -1 : 0;
}

static void mesh_continuity_object_callback(const topo_mesh_t *const mesh,
                                            const topo_mesh_shared_object_t *const object, void *const user_data)
{
    (void)mesh;
    mesh_continuity_context_t *const context = user_data;
    if (context->failed)
        return;
    if (mesh_continuity_assemble_object(context, object->mdim, object->element_count, object->element_ids,
                                        object->orientations) < 0)
    {
        context->failed = 1;
    }
}

static PyObject *mesh_continuity_builder_to_python(mesh_continuity_builder_t *const builder)
{
    size_t row_bytes;
    size_t element_bytes;
    size_t component_bytes;
    size_t local_dof_bytes;
    size_t coefficient_bytes;
    if (builder->row_count > (size_t)PY_SSIZE_T_MAX || builder->entry_count > (size_t)PY_SSIZE_T_MAX ||
        __builtin_add_overflow(builder->row_count, (size_t)1, &row_bytes) ||
        __builtin_mul_overflow(row_bytes, sizeof(*builder->row_offsets), &row_bytes) ||
        __builtin_mul_overflow(builder->entry_count, sizeof(*builder->element_ids), &element_bytes) ||
        __builtin_mul_overflow(builder->entry_count, sizeof(*builder->components), &component_bytes) ||
        __builtin_mul_overflow(builder->entry_count, sizeof(*builder->local_dofs), &local_dof_bytes) ||
        __builtin_mul_overflow(builder->entry_count, sizeof(*builder->coefficients), &coefficient_bytes))
    {
        PyErr_NoMemory();
        return NULL;
    }
    const npy_intp row_size = (npy_intp)(builder->row_count + 1);
    const npy_intp entry_size = (npy_intp)builder->entry_count;
    PyArrayObject *const row_offsets = (PyArrayObject *)PyArray_SimpleNew(1, &row_size, NPY_UINTP);
    PyArrayObject *const element_ids = (PyArrayObject *)PyArray_SimpleNew(1, &entry_size, NPY_UINT64);
    PyArrayObject *const components = (PyArrayObject *)PyArray_SimpleNew(1, &entry_size, NPY_UINT32);
    PyArrayObject *const local_dofs = (PyArrayObject *)PyArray_SimpleNew(1, &entry_size, NPY_UINTP);
    PyArrayObject *const coefficients = (PyArrayObject *)PyArray_SimpleNew(1, &entry_size, NPY_DOUBLE);
    if (!row_offsets || !element_ids || !components || !local_dofs || !coefficients)
    {
        Py_XDECREF(row_offsets);
        Py_XDECREF(element_ids);
        Py_XDECREF(components);
        Py_XDECREF(local_dofs);
        Py_XDECREF(coefficients);
        return NULL;
    }
    memcpy(PyArray_DATA(row_offsets), builder->row_offsets, row_bytes);
    if (builder->entry_count > 0)
    {
        memcpy(PyArray_DATA(element_ids), builder->element_ids, element_bytes);
        memcpy(PyArray_DATA(components), builder->components, component_bytes);
        memcpy(PyArray_DATA(local_dofs), builder->local_dofs, local_dof_bytes);
        memcpy(PyArray_DATA(coefficients), builder->coefficients, coefficient_bytes);
    }
    PyObject *const result = PyTuple_New(5);
    if (!result)
    {
        Py_DECREF(row_offsets);
        Py_DECREF(element_ids);
        Py_DECREF(components);
        Py_DECREF(local_dofs);
        Py_DECREF(coefficients);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, row_offsets);
    PyTuple_SET_ITEM(result, 1, element_ids);
    PyTuple_SET_ITEM(result, 2, components);
    PyTuple_SET_ITEM(result, 3, local_dofs);
    PyTuple_SET_ITEM(result, 4, coefficients);
    return result;
}

static int mesh_check_element_specs(const interplib_module_state_t *const state, PyObject *const element_specs_seq,
                                    const unsigned ndim, const uint64_t element_count, unsigned *const out_order,
                                    kform_spec_object **const element_specs)
{
    for (uint64_t element_id = 0; element_id < element_count; ++element_id)
    {
        PyObject *const spec_object = PySequence_Fast_GET_ITEM(element_specs_seq, (Py_ssize_t)element_id);
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
        if (element_id == 0)
            *out_order = spec->order;
        else if (spec->order != *out_order)
        {
            PyErr_SetString(PyExc_ValueError, "All element specs must have the same k-form degree.");
            return -1;
        }
        element_specs[element_id] = spec;
    }
    return 0;
}

/** Parses the optional basis_type keyword into a basis family override. */
static int mesh_parse_basis_type(PyObject *const basis_type_object, basis_set_type_t *const out_type)
{
    if (basis_type_object == Py_None)
    {
        *out_type = BASIS_INVALID;
        return 0;
    }
    if (!PyUnicode_Check(basis_type_object))
    {
        PyErr_SetString(PyExc_TypeError, "basis_type must be a BasisType value or None.");
        return -1;
    }
    const char *const name = PyUnicode_AsUTF8(basis_type_object);
    if (!name)
        return -1;
    const basis_set_type_t type = basis_type_from_string(name);
    if (!basis_set_type_is_valid(type))
    {
        PyErr_SetString(PyExc_ValueError, "basis_type is not a valid basis family.");
        return -1;
    }
    *out_type = type;
    return 0;
}

static PyObject *mesh_compute_kform_continuity_constraints(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           PyObject *kwnames)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a Mesh object.");
        return NULL;
    }
    PyObject *element_specs_object;
    PyObject *element_maps_object = Py_None;
    PyObject *basis_type_object = Py_None;
    int c1_continuous = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &element_specs_object},
                                  {.type = CPYARG_TYPE_PYTHON, .p_val = &element_maps_object, .optional = 1},
                                  {.type = CPYARG_TYPE_PYTHON,
                                   .p_val = &basis_type_object,
                                   .kwname = "basis_type",
                                   .optional = 1,
                                   .kw_only = 1},
                                  {.type = CPYARG_TYPE_BOOL,
                                   .p_val = &c1_continuous,
                                   .kwname = "c1_continuous",
                                   .optional = 1,
                                   .kw_only = 1},
                                  {}},
            args, nargs, kwnames) < 0)
        return NULL;

    mesh_object *const mesh_object_this = (mesh_object *)self;
    topo_mesh_t *const mesh = mesh_object_this->mesh;
    const unsigned ndim = mesh->ndim;
    if (mesh->element_count > (uint64_t)PY_SSIZE_T_MAX || mesh->point_count > (uint64_t)PY_SSIZE_T_MAX)
    {
        PyErr_SetString(PyExc_OverflowError, "Mesh dimensions exceed Python sequence limits.");
        return NULL;
    }

    const int have_maps = element_maps_object != Py_None;
    if (!have_maps && !c1_continuous)
    {
        PyErr_SetString(PyExc_ValueError, "element_maps are required unless the mesh is declared C1 continuous.");
        return NULL;
    }

    PyObject *const element_specs_seq = PySequence_Fast(element_specs_object, "element_specs must be a sequence.");
    PyObject *const element_maps_seq =
        have_maps ? PySequence_Fast(element_maps_object, "element_maps must be a sequence.") : NULL;
    if (!element_specs_seq || (have_maps && !element_maps_seq))
    {
        Py_XDECREF(element_specs_seq);
        Py_XDECREF(element_maps_seq);
        return NULL;
    }

    mesh_continuity_context_t context = {.state = state, .ndim = ndim, .c1_continuous = c1_continuous};
    if (PySequence_Fast_GET_SIZE(element_specs_seq) != (Py_ssize_t)mesh->element_count ||
        (have_maps && PySequence_Fast_GET_SIZE(element_maps_seq) != (Py_ssize_t)mesh->element_count))
    {
        PyErr_Format(PyExc_ValueError, "element_specs and element_maps must each contain %llu entries.",
                     (unsigned long long)mesh->element_count);
        goto fail;
    }
    if (mesh->element_count > (uint64_t)(SIZE_MAX / sizeof(*context.element_specs)))
    {
        PyErr_SetString(PyExc_OverflowError, "The element count overflows specification storage.");
        goto fail;
    }
    context.element_specs = PyMem_Malloc((size_t)mesh->element_count * sizeof(*context.element_specs));
    context.element_maps = PyMem_Malloc((size_t)mesh->element_count * sizeof(*context.element_maps));
    if (!context.element_specs || !context.element_maps)
    {
        PyErr_NoMemory();
        goto fail;
    }
    {
        unsigned order = 0;
        if (mesh_check_element_specs(state, element_specs_seq, ndim, mesh->element_count, &order,
                                     context.element_specs) < 0)
            goto fail;
        context.order = order;
        for (uint64_t element_id = 0; element_id < mesh->element_count; ++element_id)
        {
            if (!have_maps)
            {
                context.element_maps[element_id] = NULL;
                continue;
            }
            PyObject *const map_object = PySequence_Fast_GET_ITEM(element_maps_seq, (Py_ssize_t)element_id);
            if (!PyObject_TypeCheck(map_object, state->space_mapping_type))
            {
                PyErr_SetString(PyExc_TypeError, "element_maps must contain SpaceMap objects.");
                goto fail;
            }
            context.element_maps[element_id] = (space_map_object *)map_object;
            if (context.element_maps[element_id]->ndim != ndim)
            {
                PyErr_SetString(PyExc_ValueError, "Every element map must describe the mesh dimension.");
                goto fail;
            }
        }
    }

    // The engine derives one common Legendre boundary space per object, so
    // only the Legendre family override is meaningful here.
    {
        basis_set_type_t type_override = BASIS_INVALID;
        if (mesh_parse_basis_type(basis_type_object, &type_override) < 0)
            goto fail;
        if (type_override != BASIS_INVALID && type_override != BASIS_LEGENDRE)
        {
            PyErr_SetString(PyExc_ValueError,
                            "Continuity test spaces always use the Legendre family; pass basis_type=None or "
                            "BasisType.LEGENDRE.");
            goto fail;
        }
    }

    if (mesh_continuity_builder_grow((void **)&context.builder.row_offsets, &context.builder.row_capacity, 1,
                                     sizeof(*context.builder.row_offsets)) < 0)
        goto fail;
    context.builder.row_offsets[0] = 0;
    if (topo_mesh_iterate_shared_all(mesh, mesh_continuity_object_callback, &context) != TOPO_SUCCESS)
    {
        PyErr_SetString(PyExc_ValueError, "Could not iterate over shared mesh objects.");
        goto fail;
    }
    if (context.failed)
        goto fail;
    {
        PyObject *const result = mesh_continuity_builder_to_python(&context.builder);
        mesh_continuity_context_release(&context);
        Py_DECREF(element_specs_seq);
        Py_XDECREF(element_maps_seq);
        return result;
    }

fail:
    mesh_continuity_context_release(&context);
    Py_DECREF(element_specs_seq);
    Py_XDECREF(element_maps_seq);
    return NULL;
}

static PyObject *mesh_compute_kform_global_constraints(PyObject *self, PyTypeObject *defining_class,
                                                       PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    (void)defining_class;
    PyObject *element_specs = Py_None;
    PyObject *element_maps = Py_None;
    PyObject *boundary_conditions = Py_None;
    PyObject *periodic_pairs = Py_None;
    PyObject *basis_type = Py_None;
    int c1_continuous = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &element_specs},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &element_maps, .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &boundary_conditions, .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &periodic_pairs, .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &basis_type, .kwname = "basis_type", .optional = 1, .kw_only = 1},
                {.type = CPYARG_TYPE_BOOL,
                 .p_val = &c1_continuous,
                 .kwname = "c1_continuous",
                 .optional = 1,
                 .kw_only = 1},
                {}},
            args, nargs, kwnames) < 0)
        return NULL;
    PyObject *const module = PyImport_ImportModule("fdg.boundary_conditions");
    if (!module)
        return NULL;
    PyObject *const function = PyObject_GetAttrString(module, "_compute_kform_global_constraints");
    Py_DECREF(module);
    if (!function)
        return NULL;
    PyObject *const call_args = PyTuple_New(5);
    if (!call_args)
    {
        Py_DECREF(function);
        return NULL;
    }
    PyTuple_SET_ITEM(call_args, 0, Py_NewRef(self));
    PyTuple_SET_ITEM(call_args, 1, Py_NewRef(element_specs));
    PyTuple_SET_ITEM(call_args, 2, Py_NewRef(element_maps));
    PyTuple_SET_ITEM(call_args, 3, Py_NewRef(boundary_conditions));
    PyTuple_SET_ITEM(call_args, 4, Py_NewRef(periodic_pairs));
    PyObject *const call_kwargs = Py_BuildValue("{s:O,s:i}", "basis_type", basis_type, "c1_continuous", c1_continuous);
    if (!call_kwargs)
    {
        Py_DECREF(call_args);
        Py_DECREF(function);
        return NULL;
    }
    PyObject *const result = PyObject_Call(function, call_args, call_kwargs);
    Py_DECREF(call_kwargs);
    Py_DECREF(call_args);
    Py_DECREF(function);
    return result;
}
PyDoc_STRVAR(mesh_docstring, "Mesh()\n"
                             "    Topological mesh built from connected hypercube elements.\n"
                             "\n"
                             "    The mesh holds the complete topology of a set of hypercube elements: the\n"
                             "    collections of all topological objects of every dimension and their\n"
                             "    immersion information, but no geometry. Its primary use is the generation\n"
                             "    of continuity constraints between neighboring elements, see\n"
                             "    ``compute_kform_continuity_constraints``.\n"
                             "\n"
                             "    The type cannot be instantiated directly; use ``from_corners`` or\n"
                             "    ``from_collections``.\n");

static PyGetSetDef mesh_getset[] = {
    {.name = "ndim", .get = (getter)mesh_get_ndim, .doc = "Number of dimensions of the space the mesh is in."},
    {.name = "point_count", .get = (getter)mesh_get_point_count, .doc = "Number of points of the mesh."},
    {.name = "element_count", .get = (getter)mesh_get_element_count, .doc = "Number of elements of the mesh."},
    {.name = "collections",
     .get = (getter)mesh_get_collections,
     .doc = "Collections of topological objects of the mesh, one uint64 array per dimension (copies)."},
    {},
};

static PyMethodDef mesh_methods[] = {
    {
        .ml_name = "from_corners",
        .ml_meth = (void *)mesh_from_corners,
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_corners(ndim, corners, /) -> Mesh\n"
                  "Create a mesh from the corner point IDs of every hypercube element.",
    },
    {
        .ml_name = "from_collections",
        .ml_meth = (void *)mesh_from_collections,
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_collections(ndim, point_count, collections, /) -> Mesh\n"
                  "Create a mesh from the collections of topological objects.",
    },
    {
        .ml_name = "element_object",
        .ml_meth = (void *)mesh_element_object,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "element_object(element_id, axis, /) -> int\n"
                  "Look up the global ID of the object at the given position within one element.",
    },
    {
        .ml_name = "iterate_shared",
        .ml_meth = (void *)mesh_iterate_shared,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "iterate_shared(mdim, /) -> list[tuple]\n"
                  "Iterate over all objects of one dimension shared by at least two elements.",
    },
    {
        .ml_name = "iterate_shared_all",
        .ml_meth = (void *)mesh_iterate_shared_all,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "iterate_shared_all() -> list[tuple]\n"
                  "Iterate over all shared objects, from dimension ndim - 1 down to 0.",
    },
    {
        .ml_name = "iterate_boundary",
        .ml_meth = (void *)mesh_iterate_boundary,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "iterate_boundary(mdim, /) -> list[tuple]\n"
                  "Iterate over all objects of one dimension on the outer boundary of the mesh.",
    },
    {
        .ml_name = "iterate_boundary_all",
        .ml_meth = (void *)mesh_iterate_boundary_all,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "iterate_boundary_all() -> list[tuple]\n"
                  "Iterate over all boundary objects, from dimension ndim - 1 down to 0.",
    },
    {
        .ml_name = "compute_kform_continuity_constraints",
        .ml_meth = (void *)mesh_compute_kform_continuity_constraints,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_kform_continuity_constraints(element_specs, element_maps=None, /, *, basis_type=None, "
                  "c1_continuous=False) -> tuple[numpy.ndarray, ...]\n"
                  "Assemble k-form continuity rows between the consecutive elements of every shared object,\n"
                  "from shared faces down to points. Boundary test spaces are derived automatically: each\n"
                  "component takes the lowest incident element order per axis, reduced by two on axes\n"
                  "without its covector. With c1_continuous the rows pair reference spaces and element_maps\n"
                  "may be omitted; otherwise they pair physical traces and maps are required.\n"
                  "Returns five packed one-dimensional arrays; empty output has row_offsets=[0].",
    },
    {
        .ml_name = "compute_kform_global_constraints",
        .ml_meth = (void *)mesh_compute_kform_global_constraints,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_kform_global_constraints(element_specs, element_maps=None, boundary_conditions=None, "
                  "periodic_pairs=None, /, *, basis_type=None, c1_continuous=False) -> "
                  "tuple[tuple[numpy.ndarray, ...], numpy.ndarray]\n"
                  "Assemble shared, prescribed-boundary, and signed-axis periodic k-form trace rows.\n"
                  "Boundary test spaces are derived automatically; boundary data are physical k-form\n"
                  "callables, and pair descriptors define boundary face mappings.\n"
                  "Returns five packed row arrays and one right-hand-side array.",
    },
    {},
};

PyType_Spec mesh_type_spec = {
    .name = FDG_TYPE_NAME("Mesh"),
    .basicsize = sizeof(mesh_object),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, mesh_new},
            {Py_tp_dealloc, mesh_dealloc},
            {Py_tp_getset, mesh_getset},
            {Py_tp_methods, mesh_methods},
            {Py_tp_doc, (char *)mesh_docstring},
            {},
        },
};
