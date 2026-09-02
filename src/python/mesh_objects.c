#include "mesh_objects.h"
#include "../topology/topology.h"
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

static PyObject *mesh_compute_kform_boundary_constraints(PyObject *self, PyTypeObject *defining_class,
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
    mesh_object *const mesh = (mesh_object *)self;

    PyObject *test_object;
    PyObject *spec_object;
    PyObject *map_object;
    Py_ssize_t element_id;
    Py_ssize_t boundary_id;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &test_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &spec_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &map_object, .type_check = state->space_mapping_type},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &boundary_id},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    kform_spec_object *const test_spec = (kform_spec_object *)test_object;
    kform_spec_object *const element_spec = (kform_spec_object *)spec_object;
    space_map_object *const element_map = (space_map_object *)map_object;
    const unsigned face_dim = Py_SIZE(test_spec->function_space);
    const unsigned order = test_spec->order;
    const unsigned element_dim = Py_SIZE(element_spec->function_space);
    if (face_dim >= element_dim || element_spec->order != order || element_map->ndim != element_dim ||
        mesh->mesh->ndim != element_dim)
    {
        PyErr_SetString(PyExc_ValueError, "Incompatible test, element, or topology dimensions.");
        return NULL;
    }
    if (element_id < 0 || (uint64_t)element_id >= mesh->mesh->element_count || boundary_id < 0)
    {
        PyErr_Format(PyExc_ValueError, "Invalid element ID %zd or boundary ID %zd.", element_id, boundary_id);
        return NULL;
    }

    int8_t *const orientation = PyMem_Malloc(element_dim * sizeof(*orientation));
    if (!orientation)
        return NULL;
    const topo_status_t topo_status = topo_obj_boundary_orientation(
        mesh->mesh->immersions + face_dim, element_dim, (uint64_t)boundary_id, (uint64_t)element_id, orientation);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Boundary %zd is not present in element %zd: %s (%s).", boundary_id, element_id,
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        PyMem_Free(orientation);
        return NULL;
    }
    PyObject *const result =
        compute_kform_boundary_constraints_impl(state, test_spec, element_spec, element_map, orientation);
    PyMem_Free(orientation);
    return result;
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

static int mesh_continuity_append_local_row(mesh_continuity_builder_t *const builder,
                                            const kform_spec_object *const test_spec, const unsigned component,
                                            const size_t local_row, const uint64_t element_id, const double sign,
                                            PyObject *const result)
{
    if (!PyTuple_Check(result) || PyTuple_GET_SIZE(result) != 4)
    {
        PyErr_SetString(PyExc_RuntimeError, "The local continuity assembler returned an invalid result.");
        return -1;
    }
    PyArrayObject *const row_offsets = (PyArrayObject *)PyTuple_GET_ITEM(result, 0);
    PyArrayObject *const components = (PyArrayObject *)PyTuple_GET_ITEM(result, 1);
    PyArrayObject *const local_dofs = (PyArrayObject *)PyTuple_GET_ITEM(result, 2);
    PyArrayObject *const coefficients = (PyArrayObject *)PyTuple_GET_ITEM(result, 3);
    if (!PyArray_Check(row_offsets) || !PyArray_Check(components) || !PyArray_Check(local_dofs) ||
        !PyArray_Check(coefficients) || PyArray_NDIM(row_offsets) != 1 || PyArray_NDIM(components) != 1 ||
        PyArray_NDIM(local_dofs) != 1 || PyArray_NDIM(coefficients) != 1 || PyArray_TYPE(row_offsets) != NPY_UINTP ||
        PyArray_TYPE(components) != NPY_UINT32 || PyArray_TYPE(local_dofs) != NPY_UINTP ||
        PyArray_TYPE(coefficients) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_RuntimeError, "The local continuity assembler returned invalid arrays.");
        return -1;
    }

    const unsigned test_component_count =
        combination_total_count((uint8_t)Py_SIZE(test_spec->function_space), (uint8_t)test_spec->order);
    const size_t row_start = test_spec->component_offsets[component];
    const size_t row_end = test_spec->component_offsets[component + 1];
    if (local_row >= row_end - row_start ||
        (size_t)PyArray_SIZE(row_offsets) != (size_t)test_spec->component_offsets[test_component_count] + 1 ||
        (size_t)PyArray_SIZE(components) != (size_t)PyArray_SIZE(local_dofs) ||
        (size_t)PyArray_SIZE(components) != (size_t)PyArray_SIZE(coefficients))
    {
        PyErr_SetString(PyExc_RuntimeError, "The local continuity assembler returned inconsistent row dimensions.");
        return -1;
    }

    const npy_uintp *const local_offsets = PyArray_DATA(row_offsets);
    const npy_uint32 *const local_components = PyArray_DATA(components);
    const npy_uintp *const local_indices = PyArray_DATA(local_dofs);
    const npy_double *const local_coefficients = PyArray_DATA(coefficients);
    const size_t row = row_start + local_row;
    const size_t entry_count = (size_t)PyArray_SIZE(components);
    const size_t start = (size_t)local_offsets[row];
    const size_t end = (size_t)local_offsets[row + 1];
    if (start > end || end > entry_count)
    {
        PyErr_SetString(PyExc_RuntimeError, "The local continuity assembler returned invalid row offsets.");
        return -1;
    }
    for (size_t entry = start; entry < end; ++entry)
    {
        if (mesh_continuity_builder_append_row(builder, element_id, local_components[entry],
                                               (size_t)local_indices[entry], sign * local_coefficients[entry]) < 0)
            return -1;
    }
    return 0;
}

typedef struct
{
    const interplib_module_state_t *state;
    unsigned ndim;
    unsigned order;
    kform_spec_object **element_specs;
    space_map_object **element_maps;
    size_t *test_dimension_offsets;
    PyObject **test_object_specs;
    size_t test_object_count;
    mesh_continuity_builder_t builder;
    int failed;
} mesh_continuity_context_t;

static void mesh_continuity_context_release(mesh_continuity_context_t *const context)
{
    for (size_t i = 0; i < context->test_object_count; ++i)
        Py_XDECREF(context->test_object_specs[i]);
    PyMem_Free(context->test_object_specs);
    PyMem_Free(context->test_dimension_offsets);
    PyMem_Free(context->element_specs);
    PyMem_Free(context->element_maps);
    mesh_continuity_builder_release(&context->builder);
    *context = (mesh_continuity_context_t){};
}

static void mesh_continuity_pair_callback(const topo_mesh_t *const mesh, const unsigned mdim, const uint64_t object_id,
                                          const uint64_t element_id_1, const int8_t *const orientation_1,
                                          const uint64_t element_id_2, const int8_t *const orientation_2,
                                          void *const user_data)
{
    (void)mesh;
    mesh_continuity_context_t *const context = user_data;
    if (context->failed)
        return;
    const size_t object_index = context->test_dimension_offsets[mdim] + (size_t)object_id;
    PyObject *const component_objects = context->test_object_specs[object_index];
    const Py_ssize_t component_count = PySequence_Fast_GET_SIZE(component_objects);
    for (Py_ssize_t component = 0; component < component_count; ++component)
    {
        kform_spec_object *const test_spec =
            (kform_spec_object *)PySequence_Fast_GET_ITEM(component_objects, component);
        PyObject *const first_result =
            compute_kform_boundary_constraints_impl(context->state, test_spec, context->element_specs[element_id_1],
                                                    context->element_maps[element_id_1], orientation_1);
        if (!first_result)
        {
            context->failed = 1;
            return;
        }
        PyObject *const second_result =
            compute_kform_boundary_constraints_impl(context->state, test_spec, context->element_specs[element_id_2],
                                                    context->element_maps[element_id_2], orientation_2);
        if (!second_result)
        {
            Py_DECREF(first_result);
            context->failed = 1;
            return;
        }
        const size_t row_count = test_spec->component_offsets[component + 1] - test_spec->component_offsets[component];
        for (size_t row = 0; row < row_count; ++row)
        {
            if (mesh_continuity_append_local_row(&context->builder, test_spec, (unsigned)component, row, element_id_1,
                                                 +1.0, first_result) < 0 ||
                mesh_continuity_append_local_row(&context->builder, test_spec, (unsigned)component, row, element_id_2,
                                                 -1.0, second_result) < 0 ||
                mesh_continuity_builder_finish_row(&context->builder) < 0)
            {
                Py_DECREF(first_result);
                Py_DECREF(second_result);
                context->failed = 1;
                return;
            }
        }
        Py_DECREF(first_result);
        Py_DECREF(second_result);
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
    PyObject *element_maps_object;
    PyObject *test_specs_object;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &element_specs_object},
                                                    {.type = CPYARG_TYPE_PYTHON, .p_val = &element_maps_object},
                                                    {.type = CPYARG_TYPE_PYTHON, .p_val = &test_specs_object},
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

    PyObject *const element_specs_seq = PySequence_Fast(element_specs_object, "element_specs must be a sequence.");
    PyObject *const element_maps_seq = PySequence_Fast(element_maps_object, "element_maps must be a sequence.");
    PyObject *const test_outer_seq = PySequence_Fast(test_specs_object, "test_specs must be a sequence.");
    if (!element_specs_seq || !element_maps_seq || !test_outer_seq)
    {
        Py_XDECREF(element_specs_seq);
        Py_XDECREF(element_maps_seq);
        Py_XDECREF(test_outer_seq);
        return NULL;
    }

    mesh_continuity_context_t context = {.state = state, .ndim = ndim};
    if (PySequence_Fast_GET_SIZE(element_specs_seq) != (Py_ssize_t)mesh->element_count ||
        PySequence_Fast_GET_SIZE(element_maps_seq) != (Py_ssize_t)mesh->element_count)
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
    for (uint64_t element_id = 0; element_id < mesh->element_count; ++element_id)
    {
        PyObject *const spec_object = PySequence_Fast_GET_ITEM(element_specs_seq, (Py_ssize_t)element_id);
        PyObject *const map_object = PySequence_Fast_GET_ITEM(element_maps_seq, (Py_ssize_t)element_id);
        if (!PyObject_TypeCheck(spec_object, state->kform_specs_type) ||
            !PyObject_TypeCheck(map_object, state->space_mapping_type))
        {
            PyErr_SetString(PyExc_TypeError, "element_specs and element_maps contain an unexpected object type.");
            goto fail;
        }
        context.element_specs[element_id] = (kform_spec_object *)spec_object;
        context.element_maps[element_id] = (space_map_object *)map_object;
        if (context.element_specs[element_id]->function_space == NULL ||
            Py_SIZE(context.element_specs[element_id]->function_space) != (Py_ssize_t)ndim ||
            context.element_maps[element_id]->ndim != ndim)
        {
            PyErr_SetString(PyExc_ValueError, "Every element spec and map must describe the mesh dimension.");
            goto fail;
        }
        if (element_id == 0)
        {
            context.order = context.element_specs[element_id]->order;
        }
        else if (context.element_specs[element_id]->order != context.order)
        {
            PyErr_SetString(PyExc_ValueError, "All element specs must have the same k-form degree.");
            goto fail;
        }
    }

    if (PySequence_Fast_GET_SIZE(test_outer_seq) != (Py_ssize_t)ndim)
    {
        PyErr_Format(PyExc_ValueError, "test_specs must contain exactly %u dimensions.", ndim);
        goto fail;
    }
    context.test_dimension_offsets = PyMem_Malloc((size_t)(ndim + 1) * sizeof(*context.test_dimension_offsets));
    if (!context.test_dimension_offsets)
    {
        PyErr_NoMemory();
        goto fail;
    }
    context.test_dimension_offsets[0] = 0;
    for (unsigned mdim = 0; mdim < ndim; ++mdim)
    {
        const uint64_t object_count = mdim == 0 ? mesh->point_count : mesh->collections[mdim - 1].count;
        if (object_count > (uint64_t)PY_SSIZE_T_MAX ||
            context.test_dimension_offsets[mdim] > SIZE_MAX - (size_t)object_count)
        {
            PyErr_SetString(PyExc_OverflowError, "test_specs object dimensions overflow size limits.");
            goto fail;
        }
        PyObject *const objects = PySequence_Fast(PySequence_Fast_GET_ITEM(test_outer_seq, mdim),
                                                  "Each test_specs dimension must be a sequence.");
        if (!objects)
            goto fail;
        if (PySequence_Fast_GET_SIZE(objects) != (Py_ssize_t)object_count)
        {
            Py_DECREF(objects);
            PyErr_Format(PyExc_ValueError, "test_specs dimension %u must contain %llu objects.", mdim,
                         (unsigned long long)object_count);
            goto fail;
        }
        const size_t old_count = context.test_object_count;
        size_t new_count;
        size_t test_object_bytes;
        if (__builtin_add_overflow(old_count, (size_t)object_count, &new_count) ||
            __builtin_mul_overflow(new_count, sizeof(*context.test_object_specs), &test_object_bytes))
        {
            Py_DECREF(objects);
            PyErr_SetString(PyExc_OverflowError, "test_specs object dimensions overflow size limits.");
            goto fail;
        }
        PyObject **const grown = PyMem_Realloc(context.test_object_specs, test_object_bytes);
        if (!grown)
        {
            Py_DECREF(objects);
            PyErr_NoMemory();
            goto fail;
        }
        memset(grown + old_count, 0, (new_count - old_count) * sizeof(*grown));
        context.test_object_specs = grown;
        context.test_object_count = new_count;
        const unsigned component_count = combination_total_count((uint8_t)mdim, (uint8_t)context.order);
        for (uint64_t object_id = 0; object_id < object_count; ++object_id)
        {
            PyObject *const component_objects = PySequence_Fast(
                PySequence_Fast_GET_ITEM(objects, (Py_ssize_t)object_id), "Each test_specs object must be a sequence.");
            if (!component_objects)
            {
                Py_DECREF(objects);
                goto fail;
            }
            const Py_ssize_t supplied = PySequence_Fast_GET_SIZE(component_objects);
            if ((mdim < context.order && supplied != 0) ||
                (mdim >= context.order && supplied != 0 && supplied != (Py_ssize_t)component_count))
            {
                Py_DECREF(component_objects);
                Py_DECREF(objects);
                PyErr_Format(PyExc_ValueError, "test_specs[%u][%llu] has the wrong component count.", mdim,
                             (unsigned long long)object_id);
                goto fail;
            }
            for (Py_ssize_t component = 0; component < supplied; ++component)
            {
                PyObject *const test_object = PySequence_Fast_GET_ITEM(component_objects, component);
                if (!PyObject_TypeCheck(test_object, state->kform_specs_type))
                {
                    Py_DECREF(component_objects);
                    Py_DECREF(objects);
                    PyErr_SetString(PyExc_TypeError, "test_specs entries must be KFormSpecs objects.");
                    goto fail;
                }
                kform_spec_object *const test_spec = (kform_spec_object *)test_object;
                if (Py_SIZE(test_spec->function_space) != (Py_ssize_t)mdim || test_spec->order != context.order)
                {
                    Py_DECREF(component_objects);
                    Py_DECREF(objects);
                    PyErr_SetString(PyExc_ValueError,
                                    "test_specs entries must match the object dimension and form degree.");
                    goto fail;
                }
            }
            context.test_object_specs[old_count + (size_t)object_id] = component_objects;
        }
        Py_DECREF(objects);
        context.test_dimension_offsets[mdim + 1] = context.test_object_count;
    }

    if (mesh_continuity_builder_grow((void **)&context.builder.row_offsets, &context.builder.row_capacity, 1,
                                     sizeof(*context.builder.row_offsets)) < 0)
        goto fail;
    context.builder.row_offsets[0] = 0;
    if (topo_mesh_iterate_shared_pairs(mesh, mesh_continuity_pair_callback, &context) != TOPO_SUCCESS)
    {
        PyErr_SetString(PyExc_ValueError, "Could not iterate over shared mesh pairs.");
        goto fail;
    }
    if (context.failed)
        goto fail;
    {
        PyObject *const result = mesh_continuity_builder_to_python(&context.builder);
        mesh_continuity_context_release(&context);
        Py_DECREF(element_specs_seq);
        Py_DECREF(element_maps_seq);
        Py_DECREF(test_outer_seq);
        return result;
    }

fail:
    mesh_continuity_context_release(&context);
    Py_DECREF(element_specs_seq);
    Py_DECREF(element_maps_seq);
    Py_DECREF(test_outer_seq);
    return NULL;
}
PyDoc_STRVAR(mesh_docstring, "Mesh()\n"
                             "    Topological mesh built from connected hypercube elements.\n"
                             "\n"
                             "    The mesh holds the complete topology of a set of hypercube elements: the\n"
                             "    collections of all topological objects of every dimension and their\n"
                             "    immersion information, but no geometry. Its primary use is the generation\n"
                             "    of continuity constraints between neighboring elements, see\n"
                             "    ``compute_kform_boundary_constraints``.\n"
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
        .ml_doc = "compute_kform_continuity_constraints(element_specs, element_maps, test_specs, /) -> "
                  "tuple[numpy.ndarray, ...]\\n"
                  "Assemble hierarchical physical k-form continuity rows for all shared mesh strata.",
    },
    {
        .ml_name = "compute_kform_boundary_constraints",
        .ml_meth = (void *)mesh_compute_kform_boundary_constraints,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_kform_boundary_constraints(test_specs, element_spec, element_map, element_id, boundary_id, "
                  "/) -> tuple[numpy.ndarray, ...]\\n"
                  "Compute one element's physical k-form boundary rows for the given boundary object of the mesh.",
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
