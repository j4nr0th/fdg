#include "mesh_objects.h"
#include "../topology/topology.h"
#include "constraints.h"
#include "cpyutl.h"
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
        .ml_name = "compute_kform_boundary_constraints",
        .ml_meth = (void *)mesh_compute_kform_boundary_constraints,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_kform_boundary_constraints(test_specs, element_spec, element_map, element_id, boundary_id, "
                  "/) -> tuple[numpy.ndarray, ...]\n"
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
