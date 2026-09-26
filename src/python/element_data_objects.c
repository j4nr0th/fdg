#include "element_data_objects.h"
#include "cpyutl.h"
#include "degrees_of_freedom.h"
#include "function_space_objects.h"
#include "integration_objects.h"
#include "kform_objects.h"
#include "mappings.h"
#include "mesh_objects.h"
#include <numpy/ndarrayobject.h>
#include <string.h>

// Section 1: shared helpers.
//
// The three collection types share the same memory layout prefix
// (owned C data, an array of cached spec objects, a frozen flag), so view
// construction and cache clearing are implemented once.

static PyObject *element_collection_make_view(PyObject *self, int *frozen, const void *data, const npy_intp length,
                                              const int typenum)
{
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNewFromData(1, &length, typenum, (void *)data);
    if (!array)
        return NULL;
    if (PyArray_SetBaseObject(array, self) < 0)
    {
        Py_DECREF(array);
        return NULL;
    }
    Py_INCREF(self);
    *frozen = 1;
    return (PyObject *)array;
}

// Section 2: MeshGeometry — per-element space map data of a mesh.

PyDoc_STRVAR(mesh_geometry_docstring, "MeshGeometry()\n"
                                      "\n"
                                      "Batched geometry data: the space map of every element of a mesh.\n"
                                      "\n"
                                      "Elements of a mesh often share only a few distinct geometry\n"
                                      "specifications (function spaces and integration spaces). Instead of\n"
                                      "storing one Python object per element, this type stores a small table of\n"
                                      "distinct options and, per element, an index into that table along with\n"
                                      "an offset into one large, flat array of coordinate values.\n"
                                      "\n"
                                      "Data is added with :meth:`add_element` or one of the constructors\n"
                                      ":meth:`from_elements` and :meth:`from_mesh_points`. Accessing the array\n"
                                      "views :attr:`values`, :attr:`offsets` or :attr:`element_options` freezes\n"
                                      "the collection: no further elements can be added, but values of existing\n"
                                      "elements can still be overwritten with :meth:`set_element_values`.\n"
                                      "Per-element geometry is retrieved as a regular :class:`SpaceMap` with\n"
                                      ":meth:`space_map`.\n");

static int mesh_geometry_ensure_state(PyObject *self, PyTypeObject *defining_class,
                                      const interplib_module_state_t **p_state, mesh_geometry_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    *p_state = state;
    *p_this = (mesh_geometry_object *)self;
    return 0;
}

static PyObject *mesh_geometry_option_function_space(mesh_geometry_object *this, const interplib_module_state_t *state,
                                                     const unsigned index)
{
    if (!this->option_objects[2 * index + 0])
    {
        const element_data_option_t *const option = element_geometry_option(this->data, index);
        this->option_objects[2 * index + 0] =
            (PyObject *)function_space_object_create(state->function_space_type, option->ndim, option->basis_specs);
    }
    return this->option_objects[2 * index + 0];
}

static PyObject *mesh_geometry_option_integration_space(mesh_geometry_object *this,
                                                        const interplib_module_state_t *state, const unsigned index)
{
    if (!this->option_objects[2 * index + 1])
    {
        const element_data_option_t *const option = element_geometry_option(this->data, index);
        integration_space_object *const space = (integration_space_object *)state->integration_space_type->tp_alloc(
            state->integration_space_type, option->ndim);
        if (!space)
            return NULL;
        for (unsigned i = 0; i < option->ndim; ++i)
            space->specs[i] = option->geometry.int_specs[i];
        this->option_objects[2 * index + 1] = (PyObject *)space;
    }
    return this->option_objects[2 * index + 1];
}

static int mesh_geometry_grow_option_objects(mesh_geometry_object *this, const unsigned option_count)
{
    const size_t new_size = 2 * (size_t)option_count * sizeof(*this->option_objects);
    PyObject **const objects =
        this->option_objects ? PyMem_Realloc(this->option_objects, new_size) : PyMem_Malloc(new_size);
    if (!objects)
    {
        PyErr_NoMemory();
        return -1;
    }
    objects[2 * option_count - 2] = NULL;
    objects[2 * option_count - 1] = NULL;
    this->option_objects = objects;
    return 0;
}

PyDoc_STRVAR(mesh_geometry_add_element_docstring, "add_element(space_map, *dofs) -> None\n"
                                                  "\n"
                                                  "Add the geometry of one element to the collection.\n"
                                                  "\n"
                                                  "Parameters\n"
                                                  "----------\n"
                                                  "space_map : SpaceMap\n"
                                                  "    Space map of the element.\n"
                                                  "*dofs : DegreesOfFreedom\n"
                                                  "    Geometry degrees of freedom, one per coordinate of the\n"
                                                  "    space map. All of them must share one function space.\n");

static int mesh_geometry_fill_values(const space_map_object *map, dof_object *const *dofs, double *out)
{
    size_t offset = 0;
    for (Py_ssize_t icoordinate = 0; icoordinate < Py_SIZE(map); ++icoordinate)
    {
        const dof_object *const coordinate_dofs = dofs[icoordinate];
        memcpy(out + offset, coordinate_dofs->values, (size_t)Py_SIZE(coordinate_dofs) * sizeof(*out));
        offset += Py_SIZE(coordinate_dofs);
    }
    return 0;
}

static int mesh_geometry_add_element_impl(mesh_geometry_object *this, const interplib_module_state_t *state,
                                          PyObject *obj, PyObject *const *dof_args, const Py_ssize_t n_dofs)
{
    if (this->frozen)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot add elements to a frozen MeshGeometry; array views of the storage were handed out.");
        return -1;
    }
    if (!PyObject_TypeCheck(obj, state->space_mapping_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s.", state->space_mapping_type->tp_name,
                     Py_TYPE(obj)->tp_name);
        return -1;
    }
    const space_map_object *const map = (space_map_object *)obj;

    if (n_dofs != Py_SIZE(map))
    {
        PyErr_Format(PyExc_ValueError, "Expected %zd degrees of freedom, one per coordinate, but got %zd.",
                     Py_SIZE(map), n_dofs);
        return -1;
    }
    for (Py_ssize_t icoordinate = 0; icoordinate < n_dofs; ++icoordinate)
    {
        if (!PyObject_TypeCheck(dof_args[icoordinate], state->degrees_of_freedom_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected a %s, got %s.", state->degrees_of_freedom_type->tp_name,
                         Py_TYPE(dof_args[icoordinate])->tp_name);
            return -1;
        }
        if (((const dof_object *)dof_args[icoordinate])->n_dims != map->ndim)
        {
            PyErr_Format(PyExc_ValueError, "Expected degrees of freedom with %u dimensions, got %u.", map->ndim,
                         ((const dof_object *)dof_args[icoordinate])->n_dims);
            return -1;
        }
    }

    // All coordinates must share one function space.
    const dof_object *const first = (dof_object *)dof_args[0];
    for (Py_ssize_t icoordinate = 1; icoordinate < n_dofs; ++icoordinate)
    {
        const dof_object *const dofs = (dof_object *)dof_args[icoordinate];
        if (dofs->n_dims != first->n_dims ||
            memcmp(dofs->basis_specs, first->basis_specs, first->n_dims * sizeof(*first->basis_specs)) != 0)
        {
            PyErr_SetString(PyExc_ValueError, "Coordinate maps must share one function space.");
            return -1;
        }
    }

    unsigned index;
    const fdg_result_t res = element_geometry_add_option(this->data, map->ndim, (unsigned)Py_SIZE(map),
                                                         first->basis_specs, map->int_specs, &index);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not add the geometry option: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        return -1;
    }
    if (mesh_geometry_grow_option_objects(this, element_geometry_option_count(this->data)) < 0)
        return -1;

    const size_t count = element_geometry_option_value_count(this->data, index);
    double *const values = PyMem_Malloc(count * sizeof(*values));
    if (!values)
    {
        PyErr_NoMemory();
        return -1;
    }
    mesh_geometry_fill_values(map, (dof_object *const *)dof_args, values);
    const fdg_result_t add_res = element_geometry_add_element(this->data, index, values);
    PyMem_Free(values);
    if (add_res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not add the element values: %s (%s).", fdg_error_str(add_res),
                     fdg_error_msg(add_res));
        return -1;
    }
    return 0;
}

static PyObject *mesh_geometry_add_element_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    mesh_geometry_object *this;
    if (mesh_geometry_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    if (kwnames && PyTuple_GET_SIZE(kwnames))
    {
        PyErr_SetString(PyExc_TypeError, "add_element takes no keyword arguments.");
        return NULL;
    }
    if (nargs < 1)
    {
        PyErr_SetString(PyExc_TypeError, "add_element requires a space map and its degrees of freedom.");
        return NULL;
    }
    if (mesh_geometry_add_element_impl(this, state, args[0], args + 1, nargs - 1) < 0)
        return NULL;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(mesh_geometry_from_elements_docstring, "from_elements(elements, /) -> MeshGeometry\n"
                                                    "\n"
                                                    "Create a new collection from space maps with their geometry\n"
                                                    "degrees of freedom.\n"
                                                    "\n"
                                                    "Parameters\n"
                                                    "----------\n"
                                                    "elements : Sequence[tuple[SpaceMap, DegreesOfFreedom, ...]]\n"
                                                    "    Geometry of every element: its space map and one geometry\n"
                                                    "    degree of freedom per coordinate, in element order.\n"
                                                    "\n"
                                                    "Returns\n"
                                                    "-------\n"
                                                    "MeshGeometry\n"
                                                    "    Collection holding the geometry of all elements.\n");

static PyObject *mesh_geometry_from_elements(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs,
                                             PyObject *kwnames)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    PyObject *elements_object;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &elements_object}, {}}, args,
                              nargs, kwnames) < 0)
        return NULL;

    PyObject *const self = PyObject_CallFunctionObjArgs(cls, NULL);
    if (!self)
        return NULL;
    PyObject *const seq = PySequence_Fast(elements_object, "elements must be a sequence of (space_map, *dofs) tuples.");
    if (!seq)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); ++i)
    {
        PyObject *const element = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyTuple_Check(element) || PyTuple_GET_SIZE(element) < 1)
        {
            PyErr_SetString(PyExc_TypeError, "Expected a (space_map, *dofs) tuple for every element.");
            Py_DECREF(seq);
            Py_DECREF(self);
            return NULL;
        }
        const Py_ssize_t n_dofs = PyTuple_GET_SIZE(element) - 1;
        PyObject **const dof_args = PyMem_Malloc(sizeof(*dof_args) * (size_t)n_dofs);
        if (!dof_args)
        {
            Py_DECREF(seq);
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        for (Py_ssize_t j = 0; j < n_dofs; ++j)
        {
            dof_args[j] = PyTuple_GET_ITEM(element, j + 1);
        }
        const int status = mesh_geometry_add_element_impl((mesh_geometry_object *)self, state,
                                                          PyTuple_GET_ITEM(element, 0), dof_args, n_dofs);
        PyMem_Free(dof_args);
        if (status < 0)
        {
            Py_DECREF(seq);
            Py_DECREF(self);
            return NULL;
        }
    }
    Py_DECREF(seq);
    return self;
}

PyDoc_STRVAR(mesh_geometry_from_mesh_points_docstring,
             "from_mesh_points(mesh, points, integration, /) -> MeshGeometry\n"
             "\n"
             "Create geometry data from the physical coordinates of the mesh points.\n"
             "\n"
             "Every element is equipped with a multilinear (order-1 Lagrange on\n"
             "uniform nodes) geometry, matching the convention of\n"
             "``Hypercube.from_corners``: corner ``k`` of an element lies on the\n"
             "positive side of axis ``d`` exactly when bit ``d`` of ``k`` is set.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "mesh : Mesh\n"
             "    Mesh providing the elements and the point connectivity.\n"
             "points : array_like\n"
             "    Array of shape ``(mesh.point_count, C)`` with the physical\n"
             "    coordinates of every mesh point.\n"
             "integration : IntegrationSpace\n"
             "    Integration space with one specification per reference dimension.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "MeshGeometry\n"
             "    Geometry collection with one element per mesh element, in mesh\n"
             "    element order.\n");

static PyObject *mesh_geometry_from_mesh_points(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs,
                                                PyObject *kwnames)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    PyObject *mesh_object_arg;
    PyObject *points_object;
    PyObject *integration_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &mesh_object_arg, .type_check = state->mesh_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &points_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &integration_object, .type_check = state->integration_space_type},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const mesh_object *const mesh = (mesh_object *)mesh_object_arg;
    const integration_space_object *const integration = (integration_space_object *)integration_object;
    if (Py_SIZE(integration) != mesh->mesh->ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected an integration space with %u dimensions, got %zd.", mesh->mesh->ndim,
                     Py_SIZE(integration));
        return NULL;
    }

    PyArrayObject *const points = (PyArrayObject *)PyArray_FROMANY(points_object, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (!points)
        return NULL;
    if ((uint64_t)PyArray_DIM(points, 0) != mesh->mesh->point_count)
    {
        PyErr_Format(PyExc_ValueError, "Expected %llu point rows, got %lld.",
                     (unsigned long long)mesh->mesh->point_count, (long long)PyArray_DIM(points, 0));
        Py_DECREF(points);
        return NULL;
    }
    const unsigned coord_count = (unsigned)PyArray_DIM(points, 1);
    if (coord_count < 1)
    {
        PyErr_SetString(PyExc_ValueError, "Expected at least one coordinate per point.");
        Py_DECREF(points);
        return NULL;
    }

    PyObject *const self = PyObject_CallFunctionObjArgs(cls, NULL);
    if (!self)
    {
        Py_DECREF(points);
        return NULL;
    }
    mesh_geometry_object *const this = (mesh_geometry_object *)self;

    basis_spec_t basis_specs[mesh->mesh->ndim];
    for (unsigned i = 0; i < mesh->mesh->ndim; ++i)
        basis_specs[i] = (basis_spec_t){.type = BASIS_LAGRANGE_UNIFORM, .order = 1};
    unsigned index;
    const fdg_result_t res =
        element_geometry_add_option(this->data, mesh->mesh->ndim, coord_count, basis_specs, integration->specs, &index);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not add the geometry option: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        Py_DECREF(points);
        Py_DECREF(self);
        return NULL;
    }
    if (mesh_geometry_grow_option_objects(this, element_geometry_option_count(this->data)) < 0)
    {
        Py_DECREF(points);
        Py_DECREF(self);
        return NULL;
    }

    const uint64_t corners_per_element = (uint64_t)1 << mesh->mesh->ndim;
    const size_t values_per_element = coord_count * corners_per_element;
    double *const values = PyMem_Malloc(values_per_element * sizeof(*values));
    if (!values)
    {
        Py_DECREF(points);
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    const npy_double *const point_data = PyArray_DATA(points);
    int8_t axis[63];
    for (uint64_t element_id = 0; element_id < mesh->mesh->element_count; ++element_id)
    {
        for (uint64_t corner = 0; corner < corners_per_element; ++corner)
        {
            for (unsigned idim = 0; idim < mesh->mesh->ndim; ++idim)
                axis[idim] = (corner >> idim) & 1 ? (int8_t)(idim + 1) : (int8_t)-(idim + 1);
            uint64_t point_id;
            topo_mesh_element_object(mesh->mesh, element_id, mesh->mesh->ndim, axis, &point_id);
            // Mesh corner ids have axis 0 as the least significant bit, while
            // the dof tensor index is row-major with the last axis fastest.
            size_t dof_index = 0;
            for (unsigned idim = 0; idim < mesh->mesh->ndim; ++idim)
                dof_index = dof_index * 2 + ((corner >> idim) & 1);
            for (unsigned icoordinate = 0; icoordinate < coord_count; ++icoordinate)
                values[icoordinate * corners_per_element + dof_index] =
                    point_data[(npy_intp)point_id * coord_count + icoordinate];
        }
        const fdg_result_t add_res = element_geometry_add_element(this->data, index, values);
        if (add_res != FDG_SUCCESS)
        {
            PyErr_Format(PyExc_ValueError, "Could not add geometry values: %s (%s).", fdg_error_str(add_res),
                         fdg_error_msg(add_res));
            PyMem_Free(values);
            Py_DECREF(points);
            Py_DECREF(self);
            return NULL;
        }
    }
    PyMem_Free(values);
    Py_DECREF(points);
    return self;
}

PyDoc_STRVAR(mesh_geometry_space_map_docstring,
             "space_map(element_id, /) -> SpaceMap\n"
             "\n"
             "Get the geometry of one element as a space map.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "element_id : int\n"
             "    Index of the element.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "SpaceMap\n"
             "    Space map built from the stored coordinate data of the element.\n");

static int mesh_geometry_check_element(mesh_geometry_object *this, const Py_ssize_t element_id)
{
    if (element_id < 0 || (uint64_t)element_id >= element_geometry_element_count(this->data))
    {
        PyErr_Format(PyExc_IndexError, "Element index %zd out of range for %llu elements.", element_id,
                     (unsigned long long)element_geometry_element_count(this->data));
        return -1;
    }
    return 0;
}

static PyObject *mesh_geometry_space_map_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    mesh_geometry_object *this;
    if (mesh_geometry_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_SSIZE, .p_val = &element_id}, {}}, args, nargs,
                              kwnames) < 0)
        return NULL;
    if (mesh_geometry_check_element(this, element_id) < 0)
        return NULL;

    const uint64_t eid = (uint64_t)element_id;
    const unsigned index = element_geometry_element_options(this->data)[eid];
    const element_data_option_t *const option = element_geometry_option(this->data, index);
    PyObject *const integration_space = mesh_geometry_option_integration_space(this, state, index);
    if (!integration_space)
        return NULL;

    unsigned dofs_per_coordinate = 1;
    for (unsigned i = 0; i < option->ndim; ++i)
        dofs_per_coordinate *= option->basis_specs[i].order + 1;

    PyObject *const coordinate_tuple = PyTuple_New(option->geometry.coord_count);
    if (!coordinate_tuple)
        return NULL;
    const double *const values = element_geometry_values(this->data) + element_geometry_offsets(this->data)[eid];
    for (unsigned icoordinate = 0; icoordinate < option->geometry.coord_count; ++icoordinate)
    {
        dof_object *const dofs = dof_object_create(state->degrees_of_freedom_type, option->ndim, option->basis_specs);
        if (!dofs)
        {
            Py_DECREF(coordinate_tuple);
            return NULL;
        }
        memcpy(dofs->values, values + (size_t)icoordinate * dofs_per_coordinate,
               dofs_per_coordinate * sizeof(*dofs->values));
        PyObject *const coordinate =
            PyObject_CallFunction((PyObject *)state->coordinate_mapping_type, "OO", dofs, integration_space);
        Py_DECREF(dofs);
        if (!coordinate)
        {
            Py_DECREF(coordinate_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(coordinate_tuple, icoordinate, coordinate);
    }

    PyObject *const space_map = PyObject_CallObject((PyObject *)state->space_mapping_type, coordinate_tuple);
    Py_DECREF(coordinate_tuple);
    return space_map;
}

PyDoc_STRVAR(mesh_geometry_option_docstring, "option(index, /) -> tuple[FunctionSpace, IntegrationSpace]\n"
                                             "\n"
                                             "Get the geometry specification of one option.\n"
                                             "\n"
                                             "Parameters\n"
                                             "----------\n"
                                             "index : int\n"
                                             "    Index into the options table.\n"
                                             "\n"
                                             "Returns\n"
                                             "-------\n"
                                             "tuple[FunctionSpace, IntegrationSpace]\n"
                                             "    Function and integration space of the option.\n");

static PyObject *mesh_geometry_option_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                             const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    mesh_geometry_object *this;
    if (mesh_geometry_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t index;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_SSIZE, .p_val = &index}, {}}, args, nargs,
                              kwnames) < 0)
        return NULL;
    if (index < 0 || (unsigned)index >= element_geometry_option_count(this->data))
    {
        PyErr_Format(PyExc_IndexError, "Option index %zd out of range for %u options.", index,
                     element_geometry_option_count(this->data));
        return NULL;
    }

    PyObject *const function_space = mesh_geometry_option_function_space(this, state, (unsigned)index);
    if (!function_space)
        return NULL;
    PyObject *const integration_space = mesh_geometry_option_integration_space(this, state, (unsigned)index);
    if (!integration_space)
        return NULL;
    return PyTuple_Pack(2, function_space, integration_space);
}

PyDoc_STRVAR(mesh_geometry_set_element_values_docstring,
             "set_element_values(element_id, values, /) -> None\n"
             "\n"
             "Overwrite the stored coordinate values of one element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "element_id : int\n"
             "    Index of the element.\n"
             "values : array_like\n"
             "    Flat array with as many entries as the element's option stores.\n");

static PyObject *mesh_geometry_set_element_values_method(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         PyObject *kwnames)
{
    const interplib_module_state_t *state;
    mesh_geometry_object *this;
    if (mesh_geometry_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    PyObject *values_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &values_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (mesh_geometry_check_element(this, element_id) < 0)
        return NULL;

    PyArrayObject *const values = (PyArrayObject *)PyArray_FROMANY(values_object, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (!values)
        return NULL;
    const uint64_t eid = (uint64_t)element_id;
    const size_t expected = element_geometry_offsets(this->data)[eid + 1] - element_geometry_offsets(this->data)[eid];
    if (PyArray_SIZE(values) != (npy_intp)expected)
    {
        PyErr_Format(PyExc_ValueError, "Expected %zu values, got %lld.", expected, (long long)PyArray_SIZE(values));
        Py_DECREF(values);
        return NULL;
    }
    memcpy(element_geometry_values(this->data) + element_geometry_offsets(this->data)[eid], PyArray_DATA(values),
           expected * sizeof(double));
    Py_DECREF(values);
    Py_RETURN_NONE;
}

static PyObject *mesh_geometry_get_element_count(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    return PyLong_FromUnsignedLongLong(element_geometry_element_count(this->data));
}

static PyObject *mesh_geometry_get_option_count(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    return PyLong_FromUnsignedLong(element_geometry_option_count(this->data));
}

static PyObject *mesh_geometry_get_values(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    const npy_intp count = (npy_intp)element_geometry_value_count(this->data);
    return element_collection_make_view(self, &this->frozen, element_geometry_values(this->data), count, NPY_DOUBLE);
}

static PyObject *mesh_geometry_get_offsets(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    const npy_intp count = (npy_intp)element_geometry_element_count(this->data) + 1;
    return element_collection_make_view(self, &this->frozen, element_geometry_offsets(this->data), count, NPY_UINT64);
}

static PyObject *mesh_geometry_get_element_options(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    const npy_intp count = (npy_intp)element_geometry_element_count(this->data);
    return element_collection_make_view(self, &this->frozen, element_geometry_element_options(this->data), count,
                                        NPY_UINT32);
}

static PyObject *mesh_geometry_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 0 || (kwds && PyDict_Size(kwds) != 0))
    {
        PyErr_SetString(PyExc_TypeError, "MeshGeometry takes no arguments.");
        return NULL;
    }
    mesh_geometry_object *const self = (mesh_geometry_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;
    self->option_objects = NULL;
    self->frozen = 0;
    const fdg_result_t res = element_geometry_create(&self->data, &SYSTEM_ALLOCATOR);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not create the geometry storage: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static int mesh_geometry_traverse(mesh_geometry_object *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    if (self->option_objects)
    {
        const size_t count = 2 * (size_t)element_geometry_option_count(self->data);
        for (size_t i = 0; i < count; ++i)
            Py_VISIT(self->option_objects[i]);
    }
    return 0;
}

static int mesh_geometry_clear(mesh_geometry_object *self)
{
    if (self->option_objects)
    {
        const size_t count = 2 * (size_t)element_geometry_option_count(self->data);
        for (size_t i = 0; i < count; ++i)
            Py_CLEAR(self->option_objects[i]);
    }
    return 0;
}

static void mesh_geometry_dealloc(mesh_geometry_object *self)
{
    PyObject_GC_UnTrack(self);
    mesh_geometry_clear(self);
    if (self->option_objects)
    {
        PyMem_Free(self->option_objects);
        self->option_objects = NULL;
    }
    if (self->data)
    {
        element_geometry_free(self->data, &SYSTEM_ALLOCATOR);
        self->data = NULL;
    }
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyMethodDef mesh_geometry_methods[] = {
    {.ml_name = "add_element",
     .ml_meth = (void *)mesh_geometry_add_element_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)mesh_geometry_add_element_docstring},
    {.ml_name = "from_elements",
     .ml_meth = (void *)mesh_geometry_from_elements,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)mesh_geometry_from_elements_docstring},
    {.ml_name = "from_mesh_points",
     .ml_meth = (void *)mesh_geometry_from_mesh_points,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)mesh_geometry_from_mesh_points_docstring},
    {.ml_name = "space_map",
     .ml_meth = (void *)mesh_geometry_space_map_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)mesh_geometry_space_map_docstring},
    {.ml_name = "option",
     .ml_meth = (void *)mesh_geometry_option_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)mesh_geometry_option_docstring},
    {.ml_name = "set_element_values",
     .ml_meth = (void *)mesh_geometry_set_element_values_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)mesh_geometry_set_element_values_docstring},
    {},
};

static PyGetSetDef mesh_geometry_getset[] = {
    {.name = "element_count", .get = mesh_geometry_get_element_count, .doc = "int : Number of stored elements."},
    {.name = "option_count",
     .get = mesh_geometry_get_option_count,
     .doc = "int : Number of distinct options in the options table."},
    {.name = "values",
     .get = mesh_geometry_get_values,
     .doc = "numpy.typing.NDArray[numpy.double] : Flat array of all element values. Freezes the collection on access."},
    {.name = "offsets",
     .get = mesh_geometry_get_offsets,
     .doc = "numpy.typing.NDArray[numpy.uint64] : CSR offsets of the per-element value blocks.\n"
            "\n"
            "The array has ``element_count + 1`` entries. Accessing this property freezes the collection."},
    {.name = "element_options",
     .get = mesh_geometry_get_element_options,
     .doc = "numpy.typing.NDArray[numpy.uint32] : Option index of every element. Freezes the collection on access."},
    {},
};

PyType_Spec mesh_geometry_type_spec = {.name = FDG_TYPE_NAME("MeshGeometry"),
                                       .basicsize = sizeof(mesh_geometry_object),
                                       .itemsize = 0,
                                       .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC |
                                                Py_TPFLAGS_IMMUTABLETYPE,
                                       .slots = (PyType_Slot[]){
                                           {Py_tp_new, mesh_geometry_new},
                                           {Py_tp_doc, (void *)mesh_geometry_docstring},
                                           {Py_tp_traverse, mesh_geometry_traverse},
                                           {Py_tp_clear, mesh_geometry_clear},
                                           {Py_tp_dealloc, mesh_geometry_dealloc},
                                           {Py_tp_methods, mesh_geometry_methods},
                                           {Py_tp_getset, mesh_geometry_getset},
                                           {},
                                       }};

// Section 3: ElementKForms — labeled k-form fields, grouped per element.

PyDoc_STRVAR(element_kforms_docstring, "ElementKForms(ndim: int, /, **fields: int)\n"
                                       "\n"
                                       "Batched k-form data: a fixed set of labeled fields, values\n"
                                       "grouped per element.\n"
                                       "\n"
                                       "When setting up a finite element system one computes element matrices,\n"
                                       "so the k-forms of one element are needed together. Each keyword\n"
                                       "argument of the constructor defines one field: the keyword is its\n"
                                       "unique label and the value its k-form order. All fields are derived\n"
                                       "from one base function space per element; the distinct base spaces\n"
                                       "form the options of the collection. Every element added with\n"
                                       ":meth:`add_element` then stores the values of all fields, in field\n"
                                       "order.\n"
                                       "\n"
                                       "Accessing the array views :meth:`values` or :meth:`offsets` freezes\n"
                                       "the collection: no further elements can be added, but the values of\n"
                                       "existing elements can still be overwritten with\n"
                                       ":meth:`set_field_values`.\n"
                                       "\n"
                                       "Parameters\n"
                                       "----------\n"
                                       "ndim : int\n"
                                       "    Number of reference dimensions, shared by all fields; must be positive.\n"
                                       "\n"
                                       "**fields : int\n"
                                       "    One keyword argument per k-form field: the keyword is the unique\n"
                                       "    label of the field, the value its order, ``0 <= order <= ndim``.\n");

static int element_kforms_ensure_state(PyObject *self, PyTypeObject *defining_class,
                                       const interplib_module_state_t **p_state, element_kforms_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    *p_state = state;
    *p_this = (element_kforms_object *)self;
    return 0;
}

/** Grows the per-space caches to hold at least space_count entries per row. */
static int element_kforms_grow_caches(element_kforms_object *this, const unsigned space_count)
{
    if (space_count <= this->space_capacity)
        return 0;
    unsigned new_capacity = this->space_capacity > 0 ? 2 * this->space_capacity : 4;
    while (new_capacity < space_count)
        new_capacity *= 2;

    PyObject **const spaces = PyMem_Realloc(this->space_objects, (size_t)new_capacity * sizeof(*spaces));
    if (!spaces)
    {
        PyErr_NoMemory();
        return -1;
    }
    for (unsigned i = this->space_capacity; i < new_capacity; ++i)
        spaces[i] = NULL;
    this->space_objects = spaces;

    // Fields are finalized before the first base space is added, so the row
    // count never changes after allocation.
    if (!this->field_specs)
    {
        this->field_specs = PyMem_Calloc((size_t)element_kforms_field_count(this->data), sizeof(*this->field_specs));
        if (!this->field_specs)
        {
            PyErr_NoMemory();
            return -1;
        }
    }
    for (unsigned i = 0; i < element_kforms_field_count(this->data); ++i)
    {
        PyObject **const row = PyMem_Realloc(this->field_specs[i], (size_t)new_capacity * sizeof(*row));
        if (!row)
        {
            PyErr_NoMemory();
            return -1;
        }
        for (unsigned j = this->space_capacity; j < new_capacity; ++j)
            row[j] = NULL;
        this->field_specs[i] = row;
    }
    this->space_capacity = new_capacity;
    return 0;
}

static PyObject *element_kforms_space_function_space(element_kforms_object *this, const interplib_module_state_t *state,
                                                     const unsigned space_index)
{
    if (!this->space_objects[space_index])
    {
        const element_data_option_t *const option = element_kforms_space_option(this->data, space_index);
        PyObject *const space =
            (PyObject *)function_space_object_create(state->function_space_type, option->ndim, option->basis_specs);
        if (!space)
            return NULL;
        this->space_objects[space_index] = space;
    }
    return this->space_objects[space_index];
}

/** Returns the cached KFormSpecs of one field on one base space. */
static PyObject *element_kforms_field_spec(element_kforms_object *this, const interplib_module_state_t *state,
                                           const unsigned field, const unsigned space_index)
{
    if (!this->field_specs[field][space_index])
    {
        PyObject *const space = element_kforms_space_function_space(this, state, space_index);
        if (!space)
            return NULL;
        this->field_specs[field][space_index] =
            PyObject_CallFunction((PyObject *)state->kform_specs_type, "nO",
                                  (Py_ssize_t)element_kforms_field_order(this->data, field), space);
    }
    return this->field_specs[field][space_index];
}

static int element_kforms_check_element(element_kforms_object *this, const Py_ssize_t element_id)
{
    if (element_id < 0 || (uint64_t)element_id >= element_kforms_element_count(this->data))
    {
        PyErr_Format(PyExc_IndexError, "Element index %zd out of range for %llu elements.", element_id,
                     (unsigned long long)element_kforms_element_count(this->data));
        return -1;
    }
    return 0;
}

/** Adds one field from a label object and its k-form order object. */
static int element_kforms_add_field_objects(element_kforms_object *this, PyObject *label_object, PyObject *order_object,
                                            const Py_ssize_t ndim)
{
    if (!PyUnicode_Check(label_object))
    {
        PyErr_SetString(PyExc_TypeError, "Field labels must be strings.");
        return -1;
    }
    if (!PyLong_Check(order_object))
    {
        PyErr_Format(PyExc_TypeError, "The order of field %R must be an integer, got %s.", label_object,
                     Py_TYPE(order_object)->tp_name);
        return -1;
    }
    const long order = PyLong_AsLong(order_object);
    if (order == -1 && PyErr_Occurred())
        return -1;
    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "The order of field %R must not be negative.", label_object);
        return -1;
    }
    const char *const label = PyUnicode_AsUTF8(label_object);
    if (!label)
        return -1;
    const fdg_result_t res = element_kforms_add_field(this->data, label, (unsigned)ndim, (unsigned)order, NULL);
    if (res == FDG_ERROR_NOT_IN_DOMAIN)
    {
        PyErr_Format(PyExc_ValueError,
                     "Invalid field %R: the label must be non-empty and unique, the order must not exceed "
                     "the dimension %zd, and fields cannot be added after base spaces.",
                     label_object, ndim);
        return -1;
    }
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not add the k-form field %R: %s (%s).", label_object,
                     fdg_error_str(res), fdg_error_msg(res));
        return -1;
    }
    return 0;
}

/** Adds one (label, order) pair of a fields sequence. */
static int element_kforms_add_field_pair(element_kforms_object *this, PyObject *pair, const Py_ssize_t ndim)
{
    PyObject *const fast = PySequence_Fast(pair, "fields must be a sequence of (label, order) pairs.");
    if (!fast)
        return -1;
    if (PySequence_Fast_GET_SIZE(fast) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "fields must be a sequence of (label, order) pairs.");
        Py_DECREF(fast);
        return -1;
    }
    const int res = element_kforms_add_field_objects(this, PySequence_Fast_GET_ITEM(fast, 0),
                                                     PySequence_Fast_GET_ITEM(fast, 1), ndim);
    Py_DECREF(fast);
    return res;
}

static element_kforms_object *element_kforms_alloc(PyTypeObject *type)
{
    element_kforms_object *const self = (element_kforms_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;
    self->space_objects = NULL;
    self->field_specs = NULL;
    self->space_capacity = 0;
    self->frozen = 0;
    const fdg_result_t res = element_kforms_create(&self->data, &SYSTEM_ALLOCATOR);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not create the k-form storage: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        Py_DECREF(self);
        return NULL;
    }
    return self;
}

static PyObject *element_kforms_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "ElementKForms takes exactly one positional argument: the dimension.");
        return NULL;
    }
    const Py_ssize_t ndim = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, 0));
    if (ndim == -1 && PyErr_Occurred())
        return NULL;
    if (ndim < 1)
    {
        PyErr_SetString(PyExc_ValueError, "The dimension must be positive.");
        return NULL;
    }
    const Py_ssize_t n_fields = kwds ? PyDict_Size(kwds) : 0;
    if (n_fields < 1)
    {
        PyErr_SetString(PyExc_TypeError, "ElementKForms requires at least one k-form field.");
        return NULL;
    }

    element_kforms_object *const self = element_kforms_alloc(type);
    if (!self)
        return NULL;
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;
    while (PyDict_Next(kwds, &pos, &key, &value))
    {
        if (element_kforms_add_field_objects(self, key, value, ndim) < 0)
        {
            Py_DECREF(self);
            return NULL;
        }
    }
    return (PyObject *)self;
}

static int element_kforms_traverse(element_kforms_object *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    if (self->space_objects)
    {
        for (unsigned i = 0; i < self->space_capacity; ++i)
            Py_VISIT(self->space_objects[i]);
    }
    if (self->field_specs)
    {
        for (unsigned field = 0; field < element_kforms_field_count(self->data); ++field)
        {
            if (!self->field_specs[field])
                continue;
            for (unsigned i = 0; i < self->space_capacity; ++i)
                Py_VISIT(self->field_specs[field][i]);
        }
    }
    return 0;
}

static int element_kforms_clear(element_kforms_object *self)
{
    if (self->space_objects)
    {
        for (unsigned i = 0; i < self->space_capacity; ++i)
            Py_CLEAR(self->space_objects[i]);
    }
    if (self->field_specs)
    {
        for (unsigned field = 0; field < element_kforms_field_count(self->data); ++field)
        {
            if (!self->field_specs[field])
                continue;
            for (unsigned i = 0; i < self->space_capacity; ++i)
                Py_CLEAR(self->field_specs[field][i]);
        }
    }
    return 0;
}

static void element_kforms_dealloc(element_kforms_object *self)
{
    PyObject_GC_UnTrack(self);
    element_kforms_clear(self);
    if (self->field_specs)
    {
        for (unsigned field = 0; field < element_kforms_field_count(self->data); ++field)
            PyMem_Free(self->field_specs[field]);
        PyMem_Free(self->field_specs);
        self->field_specs = NULL;
    }
    if (self->space_objects)
    {
        PyMem_Free(self->space_objects);
        self->space_objects = NULL;
    }
    if (self->data)
    {
        element_kforms_free(self->data, &SYSTEM_ALLOCATOR);
        self->data = NULL;
    }
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

PyDoc_STRVAR(element_kforms_add_element_docstring, "add_element(*kforms) -> None\n"
                                                   "\n"
                                                   "Add the k-form values of one element to the collection.\n"
                                                   "\n"
                                                   "Parameters\n"
                                                   "----------\n"
                                                   "*kforms : KForm\n"
                                                   "    One k-form per field, in field order. The order and\n"
                                                   "    dimension of every k-form must match its field, and all\n"
                                                   "    k-forms of the element must share one base function space.\n");

/** Validates one element's k-form group and extracts its shared base space. */
static int element_kforms_check_group(element_kforms_object *this, const interplib_module_state_t *state,
                                      PyObject *const *args, const Py_ssize_t nargs,
                                      const function_space_object **out_space)
{
    const unsigned field_count = element_kforms_field_count(this->data);
    if (nargs != (Py_ssize_t)field_count)
    {
        PyErr_Format(PyExc_TypeError, "Expected %u k-forms (one per field), got %zd.", field_count, nargs);
        return -1;
    }
    const function_space_object *space = NULL;
    for (unsigned field = 0; field < field_count; ++field)
    {
        if (!PyObject_TypeCheck(args[field], state->kform_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected a %s for field %u, got %s.", state->kform_type->tp_name, field,
                         Py_TYPE(args[field])->tp_name);
            return -1;
        }
        const kform_spec_object *const specs = ((kform_object *)args[field])->specs;
        if ((unsigned)Py_SIZE(specs->function_space) != element_kforms_ndim(this->data) ||
            specs->order != element_kforms_field_order(this->data, field))
        {
            PyErr_Format(PyExc_TypeError, "The specifications of the k-form for field %u do not match the field.",
                         field);
            return -1;
        }
        if (!space)
        {
            space = specs->function_space;
        }
        else if (Py_SIZE(specs->function_space) != Py_SIZE(space) ||
                 memcmp(specs->function_space->specs, space->specs, (size_t)Py_SIZE(space) * sizeof(*space->specs)) !=
                     0)
        {
            PyErr_SetString(PyExc_TypeError, "All k-forms of one element must share one base function space.");
            return -1;
        }
    }
    *out_space = space;
    return 0;
}

static PyObject *element_kforms_add_element_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                   const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_kforms_object *this;
    if (element_kforms_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    if (kwnames && PyTuple_GET_SIZE(kwnames))
    {
        PyErr_SetString(PyExc_TypeError, "add_element takes no keyword arguments.");
        return NULL;
    }
    if (this->frozen)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot add elements to a frozen ElementKForms; array views of the storage were handed out.");
        return NULL;
    }
    const function_space_object *space;
    if (element_kforms_check_group(this, state, args, nargs, &space) < 0)
        return NULL;
    if (!space)
    {
        PyErr_SetString(PyExc_ValueError, "ElementKForms requires at least one k-form field.");
        return NULL;
    }

    unsigned space_index;
    const fdg_result_t space_res = element_kforms_add_space(this->data, space->specs, &space_index);
    if (space_res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not add the base function space option: %s (%s).",
                     fdg_error_str(space_res), fdg_error_msg(space_res));
        return NULL;
    }
    if (element_kforms_grow_caches(this, element_kforms_space_count(this->data)) < 0)
        return NULL;

    const size_t count = element_kforms_element_value_count(this->data, space_index);
    double *const values = PyMem_Malloc(count * sizeof(*values));
    if (!values)
        return PyErr_NoMemory();
    size_t offset = 0;
    for (unsigned field = 0; field < element_kforms_field_count(this->data); ++field)
    {
        const kform_object *const kform = (kform_object *)args[field];
        memcpy(values + offset, kform->values, (size_t)Py_SIZE(kform) * sizeof(*values));
        offset += (size_t)Py_SIZE(kform);
    }
    const fdg_result_t res = element_kforms_add_element(this->data, space_index, values);
    PyMem_Free(values);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not add the element values: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(element_kforms_from_elements_docstring,
             "from_elements(ndim, fields, elements, /) -> ElementKForms\n"
             "\n"
             "Create a new collection from labeled fields and per-element k-form groups.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "ndim : int\n"
             "    Number of reference dimensions, shared by all fields.\n"
             "fields : Sequence[tuple[str, int]]\n"
             "    One ``(label, order)`` pair per k-form field, in field order.\n"
             "elements : Sequence[tuple[KForm, ...]]\n"
             "    One tuple of k-forms per element, in field order. All k-forms of\n"
             "    one element must share one base function space; distinct spaces\n"
             "    across elements are stored as separate options.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "ElementKForms\n"
             "    Collection holding the k-form values of all elements.\n");

static PyObject *element_kforms_from_elements(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs,
                                              PyObject *kwnames)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    Py_ssize_t ndim;
    PyObject *fields_object;
    PyObject *elements_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &ndim},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &fields_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &elements_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (ndim < 1)
    {
        PyErr_SetString(PyExc_ValueError, "The dimension must be positive.");
        return NULL;
    }

    PyObject *const fields_seq = PySequence_Fast(fields_object, "fields must be a sequence of (label, order) pairs.");
    if (!fields_seq)
        return NULL;
    if (PySequence_Fast_GET_SIZE(fields_seq) < 1)
    {
        PyErr_SetString(PyExc_ValueError, "ElementKForms requires at least one k-form field");
        Py_DECREF(fields_seq);
        return NULL;
    }
    element_kforms_object *const this = element_kforms_alloc((PyTypeObject *)cls);
    PyObject *const self = (PyObject *)this;
    if (!self)
    {
        Py_DECREF(fields_seq);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fields_seq); ++i)
    {
        if (element_kforms_add_field_pair(this, PySequence_Fast_GET_ITEM(fields_seq, i), ndim) < 0)
        {
            Py_DECREF(fields_seq);
            Py_DECREF(self);
            return NULL;
        }
    }
    Py_DECREF(fields_seq);

    PyObject *const elements_seq = PySequence_Fast(elements_object, "elements must be a sequence of k-form tuples.");
    if (!elements_seq)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(elements_seq); ++i)
    {
        PyObject *const group =
            PySequence_Fast(PySequence_Fast_GET_ITEM(elements_seq, i), "elements must be a sequence of k-form tuples.");
        if (!group)
        {
            Py_DECREF(elements_seq);
            Py_DECREF(self);
            return NULL;
        }
        PyObject *const result = element_kforms_add_element_method(self, NULL, PySequence_Fast_ITEMS(group),
                                                                   PySequence_Fast_GET_SIZE(group), NULL);
        Py_DECREF(group);
        if (!result)
        {
            Py_DECREF(elements_seq);
            Py_DECREF(self);
            return NULL;
        }
        Py_DECREF(result);
    }
    Py_DECREF(elements_seq);
    return self;
}

/** Shared implementation of zeros and zeros_from_options. */
static PyObject *element_kforms_zeros_common(PyObject *cls, const Py_ssize_t ndim, PyObject *fields_object,
                                             PyObject *spaces_object, PyObject *indices_object, const Py_ssize_t count)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    PyObject *const fields_seq = PySequence_Fast(fields_object, "fields must be a sequence of (label, order) pairs.");
    if (!fields_seq)
        return NULL;
    PyObject *const spaces_seq = PySequence_Fast(spaces_object, "spaces must be a sequence of FunctionSpace objects.");
    if (!spaces_seq)
    {
        Py_DECREF(fields_seq);
        return NULL;
    }
    const Py_ssize_t space_count = PySequence_Fast_GET_SIZE(spaces_seq);
    PyObject *const indices_seq =
        indices_object ? PySequence_Fast(indices_object, "indices must be a sequence of integers.") : NULL;
    if (indices_object && !indices_seq)
    {
        Py_DECREF(spaces_seq);
        Py_DECREF(fields_seq);
        return NULL;
    }
    const Py_ssize_t element_count = indices_seq ? PySequence_Fast_GET_SIZE(indices_seq) : count;
    if (element_count < 0)
    {
        PyErr_SetString(PyExc_ValueError, "The element count must not be negative.");
        Py_XDECREF(indices_seq);
        Py_DECREF(spaces_seq);
        Py_DECREF(fields_seq);
        return NULL;
    }

    element_kforms_object *const this = element_kforms_alloc((PyTypeObject *)cls);
    PyObject *const self = (PyObject *)this;
    if (!self)
    {
        Py_XDECREF(indices_seq);
        Py_DECREF(spaces_seq);
        Py_DECREF(fields_seq);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fields_seq); ++i)
    {
        if (element_kforms_add_field_pair(this, PySequence_Fast_GET_ITEM(fields_seq, i), ndim) < 0)
            goto failure;
    }

    // Add the base spaces and find the largest per-element block.
    size_t max_count = 0;
    for (Py_ssize_t i = 0; i < space_count; ++i)
    {
        PyObject *const space = PySequence_Fast_GET_ITEM(spaces_seq, i);
        if (!PyObject_TypeCheck(space, state->function_space_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected a %s, got %s.", state->function_space_type->tp_name,
                         Py_TYPE(space)->tp_name);
            goto failure;
        }
        unsigned space_index;
        const fdg_result_t res =
            element_kforms_add_space(this->data, ((function_space_object *)space)->specs, &space_index);
        if (res != FDG_SUCCESS)
        {
            PyErr_Format(PyExc_ValueError, "Could not add base space %zd: %s (%s).", i, fdg_error_str(res),
                         fdg_error_msg(res));
            goto failure;
        }
        if (element_kforms_grow_caches(this, element_kforms_space_count(this->data)) < 0)
            goto failure;
        const size_t value_count = element_kforms_element_value_count(this->data, space_index);
        if (value_count > max_count)
            max_count = value_count;
    }
    if (indices_seq)
    {
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(indices_seq); ++i)
        {
            const Py_ssize_t index = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(indices_seq, i));
            if (index == -1 && PyErr_Occurred())
                goto failure;
            if (index < 0 || (unsigned)index >= element_kforms_space_count(this->data))
            {
                PyErr_Format(PyExc_ValueError, "Space index %zd out of range for %u base spaces.", index,
                             element_kforms_space_count(this->data));
                goto failure;
            }
        }
    }

    double *const zeros = max_count > 0 ? PyMem_Calloc(max_count, sizeof(*zeros)) : NULL;
    if (max_count > 0 && !zeros)
    {
        PyErr_NoMemory();
        goto failure;
    }
    for (Py_ssize_t i = 0; i < element_count; ++i)
    {
        unsigned space_index = 0;
        if (indices_seq)
        {
            space_index = (unsigned)PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(indices_seq, i));
        }
        else if (space_count > 1)
        {
            PyErr_SetString(PyExc_ValueError,
                            "Multiple base spaces require one space index per element; use zeros_from_options.");
            PyMem_Free(zeros);
            goto failure;
        }
        const fdg_result_t res = element_kforms_add_element(this->data, space_index, zeros);
        if (res != FDG_SUCCESS)
        {
            PyMem_Free(zeros);
            PyErr_Format(PyExc_ValueError, "Could not add the zero element: %s (%s).", fdg_error_str(res),
                         fdg_error_msg(res));
            goto failure;
        }
    }
    PyMem_Free(zeros);
    Py_XDECREF(indices_seq);
    Py_DECREF(spaces_seq);
    Py_DECREF(fields_seq);
    return self;

failure:
    Py_XDECREF(indices_seq);
    Py_DECREF(spaces_seq);
    Py_DECREF(fields_seq);
    Py_DECREF(self);
    return NULL;
}

PyDoc_STRVAR(element_kforms_zeros_docstring,
             "zeros(ndim, fields, space, count, /) -> ElementKForms\n"
             "\n"
             "Create a zero-initialized collection with one shared base function space.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "ndim : int\n"
             "    Number of reference dimensions, shared by all fields.\n"
             "fields : Sequence[tuple[str, int]]\n"
             "    One ``(label, order)`` pair per k-form field, in field order.\n"
             "space : FunctionSpace\n"
             "    Base function space shared by every element; all fields are\n"
             "    derived from it.\n"
             "count : int\n"
             "    Number of zero-initialized elements.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "ElementKForms\n"
             "    Collection holding ``count`` zero elements.\n");

static PyObject *element_kforms_zeros(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    Py_ssize_t ndim;
    PyObject *fields_object;
    PyObject *space_object;
    Py_ssize_t count;
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &ndim},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &fields_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &space_object},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &count},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (ndim < 1)
    {
        PyErr_SetString(PyExc_ValueError, "The dimension must be positive.");
        return NULL;
    }
    if (!PyObject_TypeCheck(space_object, state->function_space_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s.", state->function_space_type->tp_name,
                     Py_TYPE(space_object)->tp_name);
        return NULL;
    }
    PyObject *const spaces_tuple = PyTuple_Pack(1, space_object);
    if (!spaces_tuple)
        return NULL;
    PyObject *const result = element_kforms_zeros_common(cls, ndim, fields_object, spaces_tuple, NULL, count);
    Py_DECREF(spaces_tuple);
    return result;
}

PyDoc_STRVAR(element_kforms_zeros_from_options_docstring,
             "zeros_from_options(ndim, fields, spaces, indices, /) -> ElementKForms\n"
             "\n"
             "Create a zero-initialized collection with per-element base spaces.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "ndim : int\n"
             "    Number of reference dimensions, shared by all fields.\n"
             "fields : Sequence[tuple[str, int]]\n"
             "    One ``(label, order)`` pair per k-form field, in field order.\n"
             "spaces : Sequence[FunctionSpace]\n"
             "    The distinct base function spaces; all fields are derived from\n"
             "    the base space of an element.\n"
             "indices : Sequence[int]\n"
             "    Index into ``spaces`` for every element; also fixes the element\n"
             "    count.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "ElementKForms\n"
             "    Collection holding one zero element per index.\n");

static PyObject *element_kforms_zeros_from_options(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs,
                                                   PyObject *kwnames)
{
    Py_ssize_t ndim;
    PyObject *fields_object;
    PyObject *spaces_object;
    PyObject *indices_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &ndim},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &fields_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &spaces_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &indices_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (ndim < 1)
    {
        PyErr_SetString(PyExc_ValueError, "The dimension must be positive.");
        return NULL;
    }
    return element_kforms_zeros_common(cls, ndim, fields_object, spaces_object, indices_object, 0);
}

/** Resolves a label argument to a field index, raising KeyError when unknown. */
static int element_kforms_parse_label(element_kforms_object *this, PyObject *label_object, unsigned *out_field)
{
    if (!PyUnicode_Check(label_object))
    {
        PyErr_SetString(PyExc_TypeError, "The label must be a string.");
        return -1;
    }
    const char *const label = PyUnicode_AsUTF8(label_object);
    if (!label)
        return -1;
    const fdg_result_t res = element_kforms_find_field(this->data, label, out_field);
    if (res != FDG_SUCCESS)
    {
        PyErr_SetObject(PyExc_KeyError, label_object);
        return -1;
    }
    return 0;
}

PyDoc_STRVAR(element_kforms_kform_docstring, "kform(element_id, label, /) -> KForm\n"
                                             "\n"
                                             "Get the values of one field of one element as a k-form.\n"
                                             "\n"
                                             "Parameters\n"
                                             "----------\n"
                                             "element_id : int\n"
                                             "    Index of the element.\n"
                                             "label : str\n"
                                             "    Label of the field.\n"
                                             "\n"
                                             "Returns\n"
                                             "-------\n"
                                             "KForm\n"
                                             "    K-form holding the stored values of the field.\n");

static PyObject *element_kforms_kform_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                             const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_kforms_object *this;
    if (element_kforms_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    PyObject *label_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &label_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (element_kforms_check_element(this, element_id) < 0)
        return NULL;
    unsigned field;
    if (element_kforms_parse_label(this, label_object, &field) < 0)
        return NULL;

    const uint64_t eid = (uint64_t)element_id;
    PyObject *const specs =
        element_kforms_field_spec(this, state, field, element_kforms_element_space(this->data, eid));
    if (!specs)
        return NULL;
    kform_object *const kform = kform_object_create(state->kform_type, (kform_spec_object *)specs, 0);
    if (!kform)
        return NULL;
    const uint64_t *const offsets = element_kforms_field_offsets(this->data, field);
    const double *const values = element_kforms_field_values(this->data, field) + offsets[eid];
    memcpy(kform->values, values, (size_t)(offsets[eid + 1] - offsets[eid]) * sizeof(*kform->values));
    return (PyObject *)kform;
}

PyDoc_STRVAR(element_kforms_kforms_docstring, "kforms(element_id, /) -> tuple[KForm, ...]\n"
                                              "\n"
                                              "Get the k-forms of all fields of one element, in field order.\n"
                                              "\n"
                                              "Parameters\n"
                                              "----------\n"
                                              "element_id : int\n"
                                              "    Index of the element.\n"
                                              "\n"
                                              "Returns\n"
                                              "-------\n"
                                              "tuple[KForm, ...]\n"
                                              "    One k-form per field, in field order.\n");

static PyObject *element_kforms_kforms_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                              const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_kforms_object *this;
    if (element_kforms_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_SSIZE, .p_val = &element_id}, {}}, args, nargs,
                              kwnames) < 0)
        return NULL;
    if (element_kforms_check_element(this, element_id) < 0)
        return NULL;

    const unsigned field_count = element_kforms_field_count(this->data);
    PyObject *const tuple = PyTuple_New(field_count);
    if (!tuple)
        return NULL;
    for (unsigned field = 0; field < field_count; ++field)
    {
        PyObject *const label_object = PyUnicode_FromString(element_kforms_field_label(this->data, field));
        if (!label_object)
        {
            Py_DECREF(tuple);
            return NULL;
        }
        PyObject *const kform_args[2] = {PyLong_FromSsize_t(element_id), label_object};
        PyObject *const kform = element_kforms_kform_method(self, defining_class, kform_args, 2, NULL);
        Py_DECREF(label_object);
        Py_DECREF(kform_args[0]);
        if (!kform)
        {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, field, kform);
    }
    return tuple;
}

PyDoc_STRVAR(element_kforms_specs_docstring, "specs(element_id, label, /) -> KFormSpecs\n"
                                             "\n"
                                             "Get the specifications of one field on the base space of one\n"
                                             "element.\n"
                                             "\n"
                                             "Parameters\n"
                                             "----------\n"
                                             "element_id : int\n"
                                             "    Index of the element.\n"
                                             "label : str\n"
                                             "    Label of the field.\n"
                                             "\n"
                                             "Returns\n"
                                             "-------\n"
                                             "KFormSpecs\n"
                                             "    Specifications of the field, derived from the element's base\n"
                                             "    function space.\n");

static PyObject *element_kforms_specs_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                             const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_kforms_object *this;
    if (element_kforms_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    PyObject *label_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &label_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (element_kforms_check_element(this, element_id) < 0)
        return NULL;
    unsigned field;
    if (element_kforms_parse_label(this, label_object, &field) < 0)
        return NULL;
    PyObject *const specs =
        element_kforms_field_spec(this, state, field, element_kforms_element_space(this->data, (uint64_t)element_id));
    if (!specs)
        return NULL;
    Py_INCREF(specs);
    return specs;
}

PyDoc_STRVAR(element_kforms_set_field_values_docstring,
             "set_field_values(element_id, label, values, /) -> None\n"
             "\n"
             "Overwrite the stored values of one field of one element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "element_id : int\n"
             "    Index of the element.\n"
             "label : str\n"
             "    Label of the field.\n"
             "values : array_like\n"
             "    Flat array with as many entries as the field stores per element.\n");

static PyObject *element_kforms_set_field_values_method(PyObject *self, PyTypeObject *defining_class,
                                                        PyObject *const *args, const Py_ssize_t nargs,
                                                        PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_kforms_object *this;
    if (element_kforms_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    PyObject *label_object;
    PyObject *values_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &label_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &values_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (element_kforms_check_element(this, element_id) < 0)
        return NULL;
    unsigned field;
    if (element_kforms_parse_label(this, label_object, &field) < 0)
        return NULL;

    PyArrayObject *const values = (PyArrayObject *)PyArray_FROMANY(values_object, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (!values)
        return NULL;
    const uint64_t *const offsets = element_kforms_field_offsets(this->data, field);
    const size_t expected = (size_t)(offsets[element_id + 1] - offsets[element_id]);
    if (PyArray_SIZE(values) != (npy_intp)expected)
    {
        PyErr_Format(PyExc_ValueError, "Expected %zu values, got %lld.", expected, (long long)PyArray_SIZE(values));
        Py_DECREF(values);
        return NULL;
    }
    memcpy(element_kforms_field_values(this->data, field) + offsets[element_id], PyArray_DATA(values),
           expected * sizeof(double));
    Py_DECREF(values);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(element_kforms_values_docstring, "values(label, /) -> numpy.typing.NDArray[numpy.double]\n"
                                              "\n"
                                              "Get the flat value array of one field. Freezes the\n"
                                              "collection on access.\n"
                                              "\n"
                                              "Parameters\n"
                                              "----------\n"
                                              "label : str\n"
                                              "    Label of the field.\n"
                                              "\n"
                                              "Returns\n"
                                              "-------\n"
                                              "array\n"
                                              "    One block of field values per element.\n");

static PyObject *element_kforms_values_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                              const Py_ssize_t nargs, PyObject *kwnames)
{
    element_kforms_object *this;
    const interplib_module_state_t *state;
    if (element_kforms_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    PyObject *label_object;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &label_object}, {}}, args,
                              nargs, kwnames) < 0)
        return NULL;
    unsigned field;
    if (element_kforms_parse_label(this, label_object, &field) < 0)
        return NULL;
    const uint64_t *const offsets = element_kforms_field_offsets(this->data, field);
    const npy_intp total = (npy_intp)offsets[element_kforms_element_count(this->data)];
    return element_collection_make_view(self, &this->frozen, element_kforms_field_values(this->data, field), total,
                                        NPY_DOUBLE);
}

PyDoc_STRVAR(element_kforms_offsets_docstring, "offsets(label, /) -> numpy.typing.NDArray[numpy.uint64]\n"
                                               "\n"
                                               "Get the CSR offsets of one field's per-element value blocks.\n"
                                               "\n"
                                               "Accessing this method freezes the collection.\n"
                                               "\n"
                                               "Parameters\n"
                                               "----------\n"
                                               "label : str\n"
                                               "    Label of the field.\n"
                                               "\n"
                                               "Returns\n"
                                               "-------\n"
                                               "array\n"
                                               "    Array with ``element_count + 1`` offsets.\n");

static PyObject *element_kforms_offsets_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, PyObject *kwnames)
{
    element_kforms_object *this;
    const interplib_module_state_t *state;
    if (element_kforms_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    PyObject *label_object;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &label_object}, {}}, args,
                              nargs, kwnames) < 0)
        return NULL;
    unsigned field;
    if (element_kforms_parse_label(this, label_object, &field) < 0)
        return NULL;
    return element_collection_make_view(self, &this->frozen, element_kforms_field_offsets(this->data, field),
                                        (npy_intp)element_kforms_element_count(this->data) + 1, NPY_UINT64);
}

static PyObject *element_kforms_get_element_count(PyObject *self, void *Py_UNUSED(closure))
{
    element_kforms_object *this = (element_kforms_object *)self;
    return PyLong_FromUnsignedLongLong(element_kforms_element_count(this->data));
}

static PyObject *element_kforms_get_labels(PyObject *self, void *Py_UNUSED(closure))
{
    element_kforms_object *this = (element_kforms_object *)self;
    const unsigned field_count = element_kforms_field_count(this->data);
    PyObject *const tuple = PyTuple_New(field_count);
    if (!tuple)
        return NULL;
    for (unsigned field = 0; field < field_count; ++field)
    {
        PyObject *const label = PyUnicode_FromString(element_kforms_field_label(this->data, field));
        if (!label)
        {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, field, label);
    }
    return tuple;
}

static PyMethodDef element_kforms_methods[] = {
    {.ml_name = "add_element",
     .ml_meth = (void *)element_kforms_add_element_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_add_element_docstring},
    {.ml_name = "from_elements",
     .ml_meth = (void *)element_kforms_from_elements,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_from_elements_docstring},
    {.ml_name = "zeros",
     .ml_meth = (void *)element_kforms_zeros,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_zeros_docstring},
    {.ml_name = "zeros_from_options",
     .ml_meth = (void *)element_kforms_zeros_from_options,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_zeros_from_options_docstring},
    {.ml_name = "kform",
     .ml_meth = (void *)element_kforms_kform_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_kform_docstring},
    {.ml_name = "kforms",
     .ml_meth = (void *)element_kforms_kforms_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_kforms_docstring},
    {.ml_name = "specs",
     .ml_meth = (void *)element_kforms_specs_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_specs_docstring},
    {.ml_name = "set_field_values",
     .ml_meth = (void *)element_kforms_set_field_values_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_set_field_values_docstring},
    {.ml_name = "values",
     .ml_meth = (void *)element_kforms_values_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_values_docstring},
    {.ml_name = "offsets",
     .ml_meth = (void *)element_kforms_offsets_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_kforms_offsets_docstring},
    {},
};

static PyGetSetDef element_kforms_getset[] = {
    {.name = "element_count", .get = element_kforms_get_element_count, .doc = "int : Number of stored elements."},
    {.name = "labels",
     .get = element_kforms_get_labels,
     .doc = "tuple[str, ...] : Labels of the k-form fields, in field order."},
    {},
};

PyType_Spec element_kforms_type_spec = {.name = FDG_TYPE_NAME("ElementKForms"),
                                        .basicsize = sizeof(element_kforms_object),
                                        .itemsize = 0,
                                        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC |
                                                 Py_TPFLAGS_IMMUTABLETYPE,
                                        .slots = (PyType_Slot[]){
                                            {Py_tp_new, element_kforms_new},
                                            {Py_tp_doc, (void *)element_kforms_docstring},
                                            {Py_tp_traverse, element_kforms_traverse},
                                            {Py_tp_clear, element_kforms_clear},
                                            {Py_tp_dealloc, element_kforms_dealloc},
                                            {Py_tp_methods, element_kforms_methods},
                                            {Py_tp_getset, element_kforms_getset},
                                            {},
                                        }};

// Section 4: ElementDoFs — per-element degrees of freedom.

PyDoc_STRVAR(element_dofs_docstring, "ElementDoFs()\n"
                                     "\n"
                                     "Batched degrees of freedom: the DoF vector of every element.\n"
                                     "\n"
                                     "Instead of storing one Python object per element, this type stores a\n"
                                     "small table of distinct function-space options and, per element, an\n"
                                     "index into that table along with an offset into one large, flat array\n"
                                     "of values.\n"
                                     "\n"
                                     "Data is added with :meth:`add_element` or :meth:`from_elements`.\n"
                                     "Accessing the array views :attr:`values`, :attr:`offsets` or\n"
                                     ":attr:`element_options` freezes the collection: no further elements can\n"
                                     "be added, but values of existing elements can still be overwritten with\n"
                                     ":meth:`set_element_values`. Per-element DoFs are retrieved as a regular\n"
                                     ":class:`DegreesOfFreedom` with :meth:`dofs`.\n");

static int element_dofs_ensure_state(PyObject *self, PyTypeObject *defining_class,
                                     const interplib_module_state_t **p_state, element_dofs_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    *p_state = state;
    *p_this = (element_dofs_object *)self;
    return 0;
}

static PyObject *element_dofs_option_function_space(element_dofs_object *this, const interplib_module_state_t *state,
                                                    const unsigned index)
{
    if (!this->option_objects[index])
    {
        const element_data_option_t *const option = element_dofs_option(this->data, index);
        this->option_objects[index] =
            (PyObject *)function_space_object_create(state->function_space_type, option->ndim, option->basis_specs);
    }
    return this->option_objects[index];
}

static int element_dofs_grow_option_objects(element_dofs_object *this, const unsigned option_count)
{
    const size_t new_size = (size_t)option_count * sizeof(*this->option_objects);
    PyObject **const objects =
        this->option_objects ? PyMem_Realloc(this->option_objects, new_size) : PyMem_Malloc(new_size);
    if (!objects)
    {
        PyErr_NoMemory();
        return -1;
    }
    objects[option_count - 1] = NULL;
    this->option_objects = objects;
    return 0;
}

static int element_dofs_check_element(element_dofs_object *this, const Py_ssize_t element_id)
{
    if (element_id < 0 || (uint64_t)element_id >= element_dofs_element_count(this->data))
    {
        PyErr_Format(PyExc_IndexError, "Element index %zd out of range for %llu elements.", element_id,
                     (unsigned long long)element_dofs_element_count(this->data));
        return -1;
    }
    return 0;
}

PyDoc_STRVAR(element_dofs_add_element_docstring, "add_element(dofs, /) -> None\n"
                                                 "\n"
                                                 "Add the degrees of freedom of one element to the collection.\n"
                                                 "\n"
                                                 "Parameters\n"
                                                 "----------\n"
                                                 "dofs : DegreesOfFreedom\n"
                                                 "    Degrees of freedom of the element.\n");

static PyObject *element_dofs_add_element_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                 const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_dofs_object *this;
    if (element_dofs_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    PyObject *obj;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &obj}, {}}, args, nargs,
                              kwnames) < 0)
        return NULL;
    if (this->frozen)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot add elements to a frozen ElementDoFs; array views of the storage were handed out.");
        return NULL;
    }
    if (!PyObject_TypeCheck(obj, state->degrees_of_freedom_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s.", state->degrees_of_freedom_type->tp_name,
                     Py_TYPE(obj)->tp_name);
        return NULL;
    }
    const dof_object *const dofs = (dof_object *)obj;
    unsigned index;
    const fdg_result_t res = element_dofs_add_option(this->data, dofs->n_dims, dofs->basis_specs, &index);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not add the function space option: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        return NULL;
    }
    if (element_dofs_grow_option_objects(this, element_dofs_option_count(this->data)) < 0)
        return NULL;
    const fdg_result_t add_res = element_dofs_add_element(this->data, index, dofs->values);
    if (add_res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not add the element values: %s (%s).", fdg_error_str(add_res),
                     fdg_error_msg(add_res));
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(element_dofs_from_elements_docstring, "from_elements(dofs, /) -> ElementDoFs\n"
                                                   "\n"
                                                   "Create a new collection from a sequence of degrees of freedom.\n"
                                                   "\n"
                                                   "Parameters\n"
                                                   "----------\n"
                                                   "dofs : Sequence[DegreesOfFreedom]\n"
                                                   "    Degrees of freedom of every element, in element order.\n"
                                                   "\n"
                                                   "Returns\n"
                                                   "-------\n"
                                                   "ElementDoFs\n"
                                                   "    Collection holding the degrees of freedom of all elements.\n");

static PyObject *element_dofs_from_elements(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs,
                                            PyObject *kwnames)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    PyObject *elements_object;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &elements_object}, {}}, args,
                              nargs, kwnames) < 0)
        return NULL;

    PyObject *const self = PyObject_CallFunctionObjArgs(cls, NULL);
    if (!self)
        return NULL;
    PyObject *const seq = PySequence_Fast(elements_object, "dofs must be a sequence of DegreesOfFreedom objects.");
    if (!seq)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); ++i)
    {
        PyObject *const result = element_dofs_add_element_method(self, NULL, &PySequence_Fast_ITEMS(seq)[i], 1, NULL);
        if (!result)
        {
            Py_DECREF(seq);
            Py_DECREF(self);
            return NULL;
        }
        Py_DECREF(result);
    }
    Py_DECREF(seq);
    return self;
}

/** Shared implementation of zeros and zeros_from_options. */
static PyObject *element_dofs_zeros_common(PyObject *cls, PyObject *spaces_object, PyObject *indices_object,
                                           const Py_ssize_t count)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    PyObject *const spaces_seq = PySequence_Fast(spaces_object, "spaces must be a sequence of FunctionSpace objects.");
    if (!spaces_seq)
        return NULL;
    PyObject *const indices_seq =
        indices_object ? PySequence_Fast(indices_object, "indices must be a sequence of integers.") : NULL;
    if (indices_object && !indices_seq)
    {
        Py_DECREF(spaces_seq);
        return NULL;
    }
    const Py_ssize_t element_count = indices_seq ? PySequence_Fast_GET_SIZE(indices_seq) : count;
    if (element_count < 0)
    {
        PyErr_SetString(PyExc_ValueError, "The element count must not be negative.");
        Py_XDECREF(indices_seq);
        Py_DECREF(spaces_seq);
        return NULL;
    }

    PyObject *const self = PyObject_CallFunctionObjArgs(cls, NULL);
    if (!self)
    {
        Py_XDECREF(indices_seq);
        Py_DECREF(spaces_seq);
        return NULL;
    }
    element_dofs_object *const this = (element_dofs_object *)self;

    size_t max_count = 0;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(spaces_seq); ++i)
    {
        PyObject *const space = PySequence_Fast_GET_ITEM(spaces_seq, i);
        if (!PyObject_TypeCheck(space, state->function_space_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected a %s, got %s.", state->function_space_type->tp_name,
                         Py_TYPE(space)->tp_name);
            goto failure;
        }
        unsigned index;
        const fdg_result_t res = element_dofs_add_option(this->data, (unsigned)Py_SIZE(space),
                                                         ((function_space_object *)space)->specs, &index);
        if (res != FDG_SUCCESS)
        {
            PyErr_Format(PyExc_ValueError, "Could not add function space option %zd: %s (%s).", i, fdg_error_str(res),
                         fdg_error_msg(res));
            goto failure;
        }
        if (element_dofs_grow_option_objects(this, element_dofs_option_count(this->data)) < 0)
            goto failure;
        const size_t value_count = element_dofs_option_value_count(this->data, index);
        if (value_count > max_count)
            max_count = value_count;
    }
    if (indices_seq)
    {
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(indices_seq); ++i)
        {
            const Py_ssize_t index = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(indices_seq, i));
            if (index == -1 && PyErr_Occurred())
                goto failure;
            if (index < 0 || (unsigned)index >= element_dofs_option_count(this->data))
            {
                PyErr_Format(PyExc_ValueError, "Option index %zd out of range for %u options.", index,
                             element_dofs_option_count(this->data));
                goto failure;
            }
        }
    }

    double *const zeros = max_count > 0 ? PyMem_Calloc(max_count, sizeof(*zeros)) : NULL;
    if (max_count > 0 && !zeros)
    {
        PyErr_NoMemory();
        goto failure;
    }
    for (Py_ssize_t i = 0; i < element_count; ++i)
    {
        unsigned index = 0;
        if (indices_seq)
        {
            index = (unsigned)PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(indices_seq, i));
        }
        else if (PySequence_Fast_GET_SIZE(spaces_seq) > 1)
        {
            PyErr_SetString(PyExc_ValueError, "Multiple function spaces require one option index per element; use "
                                              "zeros_from_options.");
            PyMem_Free(zeros);
            goto failure;
        }
        const fdg_result_t res = element_dofs_add_element(this->data, index, zeros);
        if (res != FDG_SUCCESS)
        {
            PyMem_Free(zeros);
            PyErr_Format(PyExc_ValueError, "Could not add the zero element: %s (%s).", fdg_error_str(res),
                         fdg_error_msg(res));
            goto failure;
        }
    }
    PyMem_Free(zeros);
    Py_XDECREF(indices_seq);
    Py_DECREF(spaces_seq);
    return self;

failure:
    Py_XDECREF(indices_seq);
    Py_DECREF(spaces_seq);
    Py_DECREF(self);
    return NULL;
}

PyDoc_STRVAR(element_dofs_zeros_docstring, "zeros(space, count, /) -> ElementDoFs\n"
                                           "\n"
                                           "Create a zero-initialized collection with one shared function space.\n"
                                           "\n"
                                           "Parameters\n"
                                           "----------\n"
                                           "space : FunctionSpace\n"
                                           "    Function space of every element.\n"
                                           "count : int\n"
                                           "    Number of zero-initialized elements.\n"
                                           "\n"
                                           "Returns\n"
                                           "-------\n"
                                           "ElementDoFs\n"
                                           "    Collection holding ``count`` zero elements.\n");

static PyObject *element_dofs_zeros(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *space_object;
    Py_ssize_t count;
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &space_object},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &count},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (!PyObject_TypeCheck(space_object, state->function_space_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got %s.", state->function_space_type->tp_name,
                     Py_TYPE(space_object)->tp_name);
        return NULL;
    }
    PyObject *const spaces_tuple = PyTuple_Pack(1, space_object);
    if (!spaces_tuple)
        return NULL;
    PyObject *const result = element_dofs_zeros_common(cls, spaces_tuple, NULL, count);
    Py_DECREF(spaces_tuple);
    return result;
}

PyDoc_STRVAR(element_dofs_zeros_from_options_docstring,
             "zeros_from_options(spaces, indices, /) -> ElementDoFs\n"
             "\n"
             "Create a zero-initialized collection with per-element function spaces.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "spaces : Sequence[FunctionSpace]\n"
             "    The distinct function spaces of the options table.\n"
             "indices : Sequence[int]\n"
             "    Option index of every element; also fixes the element count.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "ElementDoFs\n"
             "    Collection holding one zero element per index.\n");

static PyObject *element_dofs_zeros_from_options(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs,
                                                 PyObject *kwnames)
{
    PyObject *spaces_object;
    PyObject *indices_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &spaces_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &indices_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    return element_dofs_zeros_common(cls, spaces_object, indices_object, 0);
}

PyDoc_STRVAR(element_dofs_dofs_docstring, "dofs(element_id, /) -> DegreesOfFreedom\n"
                                          "\n"
                                          "Get the degrees of freedom of one element.\n"
                                          "\n"
                                          "Parameters\n"
                                          "----------\n"
                                          "element_id : int\n"
                                          "    Index of the element.\n"
                                          "\n"
                                          "Returns\n"
                                          "-------\n"
                                          "DegreesOfFreedom\n"
                                          "    Degrees of freedom holding the stored values of the element.\n");

static PyObject *element_dofs_dofs_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                          const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_dofs_object *this;
    if (element_dofs_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_SSIZE, .p_val = &element_id}, {}}, args, nargs,
                              kwnames) < 0)
        return NULL;
    if (element_dofs_check_element(this, element_id) < 0)
        return NULL;

    const uint64_t eid = (uint64_t)element_id;
    const unsigned index = element_dofs_element_options(this->data)[eid];
    const element_data_option_t *const option = element_dofs_option(this->data, index);
    dof_object *const dofs = dof_object_create(state->degrees_of_freedom_type, option->ndim, option->basis_specs);
    if (!dofs)
        return NULL;
    memcpy(dofs->values, element_dofs_values(this->data) + element_dofs_offsets(this->data)[eid],
           (size_t)Py_SIZE(dofs) * sizeof(*dofs->values));
    return (PyObject *)dofs;
}

PyDoc_STRVAR(element_dofs_option_docstring, "option(index, /) -> FunctionSpace\n"
                                            "\n"
                                            "Get the function space of one option.\n"
                                            "\n"
                                            "Parameters\n"
                                            "----------\n"
                                            "index : int\n"
                                            "    Index into the options table.\n"
                                            "\n"
                                            "Returns\n"
                                            "-------\n"
                                            "FunctionSpace\n"
                                            "    Function space of the option.\n");

static PyObject *element_dofs_option_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                            const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_dofs_object *this;
    if (element_dofs_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t index;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_SSIZE, .p_val = &index}, {}}, args, nargs,
                              kwnames) < 0)
        return NULL;
    if (index < 0 || (unsigned)index >= element_dofs_option_count(this->data))
    {
        PyErr_Format(PyExc_IndexError, "Option index %zd out of range for %u options.", index,
                     element_dofs_option_count(this->data));
        return NULL;
    }
    PyObject *const function_space = element_dofs_option_function_space(this, state, (unsigned)index);
    if (!function_space)
        return NULL;
    Py_INCREF(function_space);
    return function_space;
}

PyDoc_STRVAR(element_dofs_set_element_values_docstring,
             "set_element_values(element_id, values, /) -> None\n"
             "\n"
             "Overwrite the stored values of one element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "element_id : int\n"
             "    Index of the element.\n"
             "values : array_like\n"
             "    Flat array with as many entries as the element's option stores.\n");

static PyObject *element_dofs_set_element_values_method(PyObject *self, PyTypeObject *defining_class,
                                                        PyObject *const *args, const Py_ssize_t nargs,
                                                        PyObject *kwnames)
{
    const interplib_module_state_t *state;
    element_dofs_object *this;
    if (element_dofs_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t element_id;
    PyObject *values_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &values_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (element_dofs_check_element(this, element_id) < 0)
        return NULL;

    PyArrayObject *const values = (PyArrayObject *)PyArray_FROMANY(values_object, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (!values)
        return NULL;
    const uint64_t eid = (uint64_t)element_id;
    const size_t expected = element_dofs_offsets(this->data)[eid + 1] - element_dofs_offsets(this->data)[eid];
    if (PyArray_SIZE(values) != (npy_intp)expected)
    {
        PyErr_Format(PyExc_ValueError, "Expected %zu values, got %lld.", expected, (long long)PyArray_SIZE(values));
        Py_DECREF(values);
        return NULL;
    }
    memcpy(element_dofs_values(this->data) + element_dofs_offsets(this->data)[eid], PyArray_DATA(values),
           expected * sizeof(double));
    Py_DECREF(values);
    Py_RETURN_NONE;
}

static PyObject *element_dofs_get_element_count(PyObject *self, void *Py_UNUSED(closure))
{
    element_dofs_object *this = (element_dofs_object *)self;
    return PyLong_FromUnsignedLongLong(element_dofs_element_count(this->data));
}

static PyObject *element_dofs_get_option_count(PyObject *self, void *Py_UNUSED(closure))
{
    element_dofs_object *this = (element_dofs_object *)self;
    return PyLong_FromUnsignedLong(element_dofs_option_count(this->data));
}

static PyObject *element_dofs_get_values(PyObject *self, void *Py_UNUSED(closure))
{
    element_dofs_object *this = (element_dofs_object *)self;
    const npy_intp count = (npy_intp)element_dofs_value_count(this->data);
    return element_collection_make_view(self, &this->frozen, element_dofs_values(this->data), count, NPY_DOUBLE);
}

static PyObject *element_dofs_get_offsets(PyObject *self, void *Py_UNUSED(closure))
{
    element_dofs_object *this = (element_dofs_object *)self;
    const npy_intp count = (npy_intp)element_dofs_element_count(this->data) + 1;
    return element_collection_make_view(self, &this->frozen, element_dofs_offsets(this->data), count, NPY_UINT64);
}

static PyObject *element_dofs_get_element_options(PyObject *self, void *Py_UNUSED(closure))
{
    element_dofs_object *this = (element_dofs_object *)self;
    const npy_intp count = (npy_intp)element_dofs_element_count(this->data);
    return element_collection_make_view(self, &this->frozen, element_dofs_element_options(this->data), count,
                                        NPY_UINT32);
}

static PyObject *element_dofs_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 0 || (kwds && PyDict_Size(kwds) != 0))
    {
        PyErr_SetString(PyExc_TypeError, "ElementDoFs takes no arguments.");
        return NULL;
    }
    element_dofs_object *const self = (element_dofs_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;
    self->option_objects = NULL;
    self->frozen = 0;
    const fdg_result_t res = element_dofs_create(&self->data, &SYSTEM_ALLOCATOR);
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not create the degrees-of-freedom storage: %s (%s).",
                     fdg_error_str(res), fdg_error_msg(res));
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static int element_dofs_traverse(element_dofs_object *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    if (self->option_objects)
    {
        const size_t count = (size_t)element_dofs_option_count(self->data);
        for (size_t i = 0; i < count; ++i)
            Py_VISIT(self->option_objects[i]);
    }
    return 0;
}

static int element_dofs_clear(element_dofs_object *self)
{
    if (self->option_objects)
    {
        const size_t count = (size_t)element_dofs_option_count(self->data);
        for (size_t i = 0; i < count; ++i)
            Py_CLEAR(self->option_objects[i]);
    }
    return 0;
}

static void element_dofs_dealloc(element_dofs_object *self)
{
    PyObject_GC_UnTrack(self);
    element_dofs_clear(self);
    if (self->option_objects)
    {
        PyMem_Free(self->option_objects);
        self->option_objects = NULL;
    }
    if (self->data)
    {
        element_dofs_free(self->data, &SYSTEM_ALLOCATOR);
        self->data = NULL;
    }
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyMethodDef element_dofs_methods[] = {
    {.ml_name = "add_element",
     .ml_meth = (void *)element_dofs_add_element_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_dofs_add_element_docstring},
    {.ml_name = "from_elements",
     .ml_meth = (void *)element_dofs_from_elements,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_dofs_from_elements_docstring},
    {.ml_name = "zeros",
     .ml_meth = (void *)element_dofs_zeros,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_dofs_zeros_docstring},
    {.ml_name = "zeros_from_options",
     .ml_meth = (void *)element_dofs_zeros_from_options,
     .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_dofs_zeros_from_options_docstring},
    {.ml_name = "dofs",
     .ml_meth = (void *)element_dofs_dofs_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_dofs_dofs_docstring},
    {.ml_name = "option",
     .ml_meth = (void *)element_dofs_option_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_dofs_option_docstring},
    {.ml_name = "set_element_values",
     .ml_meth = (void *)element_dofs_set_element_values_method,
     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
     .ml_doc = (void *)element_dofs_set_element_values_docstring},
    {},
};

static PyGetSetDef element_dofs_getset[] = {
    {.name = "element_count", .get = element_dofs_get_element_count, .doc = "int : Number of stored elements."},
    {.name = "option_count",
     .get = element_dofs_get_option_count,
     .doc = "int : Number of distinct options in the options table."},
    {.name = "values",
     .get = element_dofs_get_values,
     .doc = "numpy.typing.NDArray[numpy.double] : Flat array of all element values. Freezes the collection on access."},
    {.name = "offsets",
     .get = element_dofs_get_offsets,
     .doc = "numpy.typing.NDArray[numpy.uint64] : CSR offsets of the per-element value blocks.\n"
            "\n"
            "The array has ``element_count + 1`` entries. Accessing this property freezes the collection."},
    {.name = "element_options",
     .get = element_dofs_get_element_options,
     .doc = "numpy.typing.NDArray[numpy.uint32] : Option index of every element. Freezes the collection on access."},
    {},
};

PyType_Spec element_dofs_type_spec = {.name = FDG_TYPE_NAME("ElementDoFs"),
                                      .basicsize = sizeof(element_dofs_object),
                                      .itemsize = 0,
                                      .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC |
                                               Py_TPFLAGS_IMMUTABLETYPE,
                                      .slots = (PyType_Slot[]){
                                          {Py_tp_new, element_dofs_new},
                                          {Py_tp_doc, (void *)element_dofs_docstring},
                                          {Py_tp_traverse, element_dofs_traverse},
                                          {Py_tp_clear, element_dofs_clear},
                                          {Py_tp_dealloc, element_dofs_dealloc},
                                          {Py_tp_methods, element_dofs_methods},
                                          {Py_tp_getset, element_dofs_getset},
                                          {},
                                      }};
