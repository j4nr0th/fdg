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
// The collection types with flat array storage (ElementKForms, ElementDoFs)
// share the same memory layout prefix (owned C data, an array of cached spec
// objects, a frozen flag), so view construction is implemented once.

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

// Section 2: MeshGeometry — one space map per element of a mesh.

PyDoc_STRVAR(mesh_geometry_docstring, "MeshGeometry(input_dimensions, output_dimensions, /)\n"
                                      "\n"
                                      "Batched geometry data: the space map of every element of a mesh.\n"
                                      "\n"
                                      "Each element is stored as one :class:`SpaceMap`, so the store holds\n"
                                      "the geometry of a whole mesh without rebuilding maps on access.\n"
                                      "\n"
                                      "The dimensions are shared by every stored space map: only maps from\n"
                                      "``input_dimensions`` reference dimensions to ``output_dimensions``\n"
                                      "physical dimensions can be added.\n"
                                      "\n"
                                      "Data is added with :meth:`add_element` or one of the constructors\n"
                                      ":meth:`from_elements` and :meth:`from_mesh_points`. Per-element geometry\n"
                                      "is retrieved as a regular :class:`SpaceMap` with :meth:`space_map`.\n"
                                      "\n"
                                      "Parameters\n"
                                      "----------\n"
                                      "input_dimensions : int\n"
                                      "    Number of reference dimensions of every space map; must be in\n"
                                      "    ``[0, 255]`` and must not exceed ``output_dimensions``.\n"
                                      "output_dimensions : int\n"
                                      "    Number of physical dimensions of every space map; must be at\n"
                                      "    least 1, and both it and its product with ``input_dimensions``\n"
                                      "    must fit in an unsigned 32-bit integer.\n");

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

/** Grows the map array to hold at least `count` map references. */
static int mesh_geometry_grow_maps(mesh_geometry_object *this, const Py_ssize_t count)
{
    if (count <= this->allocated)
        return 0;
    Py_ssize_t capacity = this->allocated > 0 ? this->allocated : 4;
    while (capacity < count)
        capacity *= 2;
    space_map_object **const maps = PyMem_Realloc(this->maps, (size_t)capacity * sizeof(*maps));
    if (!maps)
    {
        PyErr_NoMemory();
        return -1;
    }
    this->maps = maps;
    this->allocated = capacity;
    return 0;
}

PyDoc_STRVAR(mesh_geometry_add_element_docstring, "add_element(space_map, /) -> None\n"
                                                  "\n"
                                                  "Add the space map of one element to the collection.\n"
                                                  "\n"
                                                  "Parameters\n"
                                                  "----------\n"
                                                  "space_map : SpaceMap\n"
                                                  "    Space map of the element. Its input and output dimensions\n"
                                                  "    must match those of the collection.\n"
                                                  "\n"
                                                  "Raises\n"
                                                  "------\n"
                                                  "ValueError\n"
                                                  "    If the dimensions of the space map differ from the\n"
                                                  "    dimensions of the collection.\n");

static int mesh_geometry_add_element_impl(mesh_geometry_object *this, const interplib_module_state_t *state,
                                          PyObject *map)
{
    if (!PyObject_TypeCheck(map, state->space_mapping_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->space_mapping_type->tp_name,
                     Py_TYPE(map)->tp_name);
        return -1;
    }
    // Subclasses of SpaceMap share the layout prefix, so the cast after the
    // type check reads the dimensions of the map itself.
    const space_map_object *const space_map = (const space_map_object *)map;
    if (space_map->ndim != this->input_dimensions || (unsigned)Py_SIZE(map) != this->output_dimensions)
    {
        PyErr_Format(PyExc_ValueError,
                     "Expected a space map from %u input dimensions to %u output dimensions, but got %u input "
                     "dimensions and %u output dimensions.",
                     this->input_dimensions, this->output_dimensions, space_map->ndim, (unsigned)Py_SIZE(map));
        return -1;
    }
    if (mesh_geometry_grow_maps(this, this->count + 1) < 0)
        return -1;
    Py_INCREF(map);
    this->maps[this->count] = (space_map_object *)map;
    this->count += 1;
    return 0;
}

static PyObject *mesh_geometry_add_element_method(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    mesh_geometry_object *this;
    if (mesh_geometry_ensure_state(self, defining_class, &state, &this) < 0)
        return NULL;
    // The arity is checked before parse_arguments_check: the helper asserts
    // (and aborts through CPYUTL_ASSERT) on a wrong argument count instead of
    // raising, so the error is reported here.
    if (kwnames && PyTuple_GET_SIZE(kwnames))
    {
        PyErr_SetString(PyExc_TypeError, "add_element() takes no keyword arguments.");
        return NULL;
    }
    if (nargs != 1)
    {
        PyErr_Format(PyExc_TypeError, "add_element() takes exactly one argument (%zd given).", nargs);
        return NULL;
    }
    PyObject *map;
    if (parse_arguments_check((cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = &map}, {}}, args, nargs,
                              kwnames) < 0)
        return NULL;
    if (mesh_geometry_add_element_impl(this, state, map) < 0)
        return NULL;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(mesh_geometry_from_elements_docstring, "from_elements(*space_maps) -> MeshGeometry\n"
                                                    "\n"
                                                    "Create a new collection from the space maps of the elements.\n"
                                                    "\n"
                                                    "The input and output dimensions of the collection are taken\n"
                                                    "from the first space map; every further map must match them.\n"
                                                    "\n"
                                                    "Parameters\n"
                                                    "----------\n"
                                                    "*space_maps : SpaceMap\n"
                                                    "    Space map of every element, in element order.\n"
                                                    "\n"
                                                    "Returns\n"
                                                    "-------\n"
                                                    "MeshGeometry\n"
                                                    "    Collection holding the space maps of all elements.\n"
                                                    "\n"
                                                    "Raises\n"
                                                    "------\n"
                                                    "ValueError\n"
                                                    "    If a space map does not have the dimensions of the first.\n");

static PyObject *mesh_geometry_from_elements(PyObject *cls, PyObject *const *args, const Py_ssize_t nargs,
                                             PyObject *kwnames)
{
    const interplib_module_state_t *const state = interplib_get_module_state((PyTypeObject *)cls);
    if (!state)
        return NULL;
    if (kwnames && PyTuple_GET_SIZE(kwnames))
    {
        PyErr_SetString(PyExc_TypeError, "from_elements() takes no keyword arguments.");
        return NULL;
    }
    if (nargs < 1)
    {
        PyErr_SetString(PyExc_TypeError, "from_elements() requires at least one space map.");
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], state->space_mapping_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->space_mapping_type->tp_name,
                     Py_TYPE(args[0])->tp_name);
        return NULL;
    }
    // The dimensions of the collection follow from the first map; the remaining
    // maps are checked against them by mesh_geometry_add_element_impl().
    const space_map_object *const first_map = (const space_map_object *)args[0];
    PyObject *const self = PyObject_CallFunction(cls, "nn", (Py_ssize_t)first_map->ndim, (Py_ssize_t)Py_SIZE(args[0]));
    if (!self)
        return NULL;
    for (Py_ssize_t i = 0; i < nargs; ++i)
    {
        if (mesh_geometry_add_element_impl((mesh_geometry_object *)self, state, args[i]) < 0)
        {
            Py_DECREF(self);
            return NULL;
        }
    }
    return self;
}

/**
 * Build a space map from one coordinate-major block of DoF values.
 *
 * @param state Interpreter module state.
 * @param ndim Number of reference dimensions of the map.
 * @param basis_specs [ndim] Specs of the function space shared by the coordinates.
 * @param coord_count Number of physical coordinates.
 * @param integration_space Integration space of the map.
 * @param values [coord_count * dofs_per_coordinate] values, coordinate-major.
 * @return The new space map, or NULL with a Python exception set.
 */
static PyObject *mesh_geometry_build_space_map(const interplib_module_state_t *state, const unsigned ndim,
                                               const basis_spec_t basis_specs[static ndim], const unsigned coord_count,
                                               PyObject *integration_space, const double *values)
{
    unsigned dofs_per_coordinate = 1;
    for (unsigned i = 0; i < ndim; ++i)
        dofs_per_coordinate *= basis_specs[i].order + 1;

    PyObject *const coordinate_tuple = PyTuple_New(coord_count);
    if (!coordinate_tuple)
        return NULL;
    for (unsigned icoordinate = 0; icoordinate < coord_count; ++icoordinate)
    {
        dof_object *const dofs = dof_object_create(state->degrees_of_freedom_type, ndim, basis_specs);
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

    PyObject *const self = PyObject_CallFunction(cls, "nn", (Py_ssize_t)mesh->mesh->ndim, (Py_ssize_t)coord_count);
    if (!self)
    {
        Py_DECREF(points);
        return NULL;
    }
    mesh_geometry_object *const this = (mesh_geometry_object *)self;

    basis_spec_t basis_specs[mesh->mesh->ndim];
    for (unsigned i = 0; i < mesh->mesh->ndim; ++i)
        basis_specs[i] = (basis_spec_t){.type = BASIS_LAGRANGE_UNIFORM, .order = 1};

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
        PyObject *const map = mesh_geometry_build_space_map(state, mesh->mesh->ndim, basis_specs, coord_count,
                                                            integration_object, values);
        if (!map)
        {
            PyMem_Free(values);
            Py_DECREF(points);
            Py_DECREF(self);
            return NULL;
        }
        const int status = mesh_geometry_add_element_impl(this, state, map);
        Py_DECREF(map);
        if (status < 0)
        {
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

PyDoc_STRVAR(mesh_geometry_space_map_docstring, "space_map(element_id, /) -> SpaceMap\n"
                                                "\n"
                                                "Get the space map of one element.\n"
                                                "\n"
                                                "Parameters\n"
                                                "----------\n"
                                                "element_id : int\n"
                                                "    Index of the element.\n"
                                                "\n"
                                                "Returns\n"
                                                "-------\n"
                                                "SpaceMap\n"
                                                "    Space map stored for the element.\n");

static int mesh_geometry_check_element(mesh_geometry_object *this, const Py_ssize_t element_id)
{
    if (element_id < 0 || element_id >= this->count)
    {
        PyErr_Format(PyExc_IndexError, "Element index %zd out of range for %zd elements.", element_id, this->count);
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

    PyObject *const map = (PyObject *)this->maps[element_id];
    Py_INCREF(map);
    return map;
}

static PyObject *mesh_geometry_get_element_count(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    return PyLong_FromSsize_t(this->count);
}

static PyObject *mesh_geometry_get_input_dimensions(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    return PyLong_FromUnsignedLong(this->input_dimensions);
}

static PyObject *mesh_geometry_get_output_dimensions(PyObject *self, void *Py_UNUSED(closure))
{
    mesh_geometry_object *this = (mesh_geometry_object *)self;
    return PyLong_FromUnsignedLong(this->output_dimensions);
}

static PyObject *mesh_geometry_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds && PyDict_Size(kwds) != 0)
    {
        PyErr_SetString(PyExc_TypeError, "MeshGeometry takes no keyword arguments.");
        return NULL;
    }
    if (PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_Format(PyExc_TypeError, "MeshGeometry takes exactly two arguments, got %zd.", PyTuple_GET_SIZE(args));
        return NULL;
    }
    const Py_ssize_t input_dimensions = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, 0));
    if (input_dimensions == -1 && PyErr_Occurred())
        return NULL;
    const Py_ssize_t output_dimensions = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, 1));
    if (output_dimensions == -1 && PyErr_Occurred())
        return NULL;
    // space_map_object_create() takes these as preconditions; report an
    // invalid combination here instead of ending up with a store to which no
    // space map could ever be added.
    if (input_dimensions < 0 || input_dimensions > UINT8_MAX)
    {
        PyErr_Format(PyExc_ValueError, "Expected input_dimensions in [0, %u], got %zd.", (unsigned)UINT8_MAX,
                     input_dimensions);
        return NULL;
    }
    if (output_dimensions < 1)
    {
        PyErr_Format(PyExc_ValueError, "Expected output_dimensions of at least 1, got %zd.", output_dimensions);
        return NULL;
    }
    if (input_dimensions > output_dimensions)
    {
        PyErr_Format(PyExc_ValueError, "Expected input_dimensions at most output_dimensions, got %zd and %zd.",
                     input_dimensions, output_dimensions);
        return NULL;
    }
    if ((uint64_t)output_dimensions > (uint64_t)UINT_MAX)
    {
        // With input_dimensions 0 the product check below is vacuous, but the
        // dimensions are stored as unsigned and a SpaceMap has at most that
        // many coordinate maps.
        PyErr_Format(PyExc_ValueError, "Expected output_dimensions of at most %u, got %zd.", UINT_MAX,
                     output_dimensions);
        return NULL;
    }
    if ((uint64_t)input_dimensions * (uint64_t)output_dimensions > (uint64_t)UINT_MAX)
    {
        PyErr_Format(PyExc_ValueError,
                     "The product of input_dimensions %zd and output_dimensions %zd exceeds the maximum of %u.",
                     input_dimensions, output_dimensions, UINT_MAX);
        return NULL;
    }
    mesh_geometry_object *const self = (mesh_geometry_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;
    self->maps = NULL;
    self->count = 0;
    self->allocated = 0;
    self->input_dimensions = (unsigned)input_dimensions;
    self->output_dimensions = (unsigned)output_dimensions;
    return (PyObject *)self;
}

static int mesh_geometry_traverse(mesh_geometry_object *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    for (Py_ssize_t i = 0; i < self->count; ++i)
        Py_VISIT(self->maps[i]);
    return 0;
}

static int mesh_geometry_clear(mesh_geometry_object *self)
{
    for (Py_ssize_t i = 0; i < self->count; ++i)
        Py_CLEAR(self->maps[i]);
    self->count = 0;
    return 0;
}

static void mesh_geometry_dealloc(mesh_geometry_object *self)
{
    PyObject_GC_UnTrack(self);
    mesh_geometry_clear(self);
    if (self->maps)
    {
        PyMem_Free(self->maps);
        self->maps = NULL;
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
    {},
};

static PyGetSetDef mesh_geometry_getset[] = {
    {.name = "element_count", .get = mesh_geometry_get_element_count, .doc = "int : Number of stored space maps."},
    {.name = "input_dimensions",
     .get = mesh_geometry_get_input_dimensions,
     .doc = "int : Dimension of the input/reference space."},
    {.name = "output_dimensions",
     .get = mesh_geometry_get_output_dimensions,
     .doc = "int : Dimension of the output/physical space."},
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

    // Everything element_kforms_add_field() takes as a precondition is checked
    // above and here, where a violation can be reported instead of the C core
    // turning it into an assert.
    if (order > ndim)
    {
        PyErr_Format(PyExc_ValueError, "The order of field %R must not exceed the dimension %zd, but got %ld.",
                     label_object, ndim, order);
        return -1;
    }
    if (label[0] == '\0')
    {
        PyErr_SetString(PyExc_ValueError, "Field labels must not be empty.");
        return -1;
    }
    unsigned existing_field;
    if (element_kforms_find_field(this->data, label, &existing_field))
    {
        PyErr_Format(PyExc_ValueError, "A field labeled %R already exists.", label_object);
        return -1;
    }
    if (element_kforms_space_count(this->data) != 0)
    {
        PyErr_Format(PyExc_ValueError, "Cannot add the field %R after base spaces were added.", label_object);
        return -1;
    }
    const unsigned collection_ndim = element_kforms_ndim(this->data);
    if (collection_ndim != 0 && collection_ndim != (unsigned)ndim)
    {
        PyErr_Format(PyExc_ValueError, "The field %R has dimension %zd, but the collection has dimension %u.",
                     label_object, ndim, collection_ndim);
        return -1;
    }

    const fdg_result_t res = element_kforms_add_field(this->data, label, (unsigned)ndim, (unsigned)order, NULL);
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

/**
 * Validates a base function space against the collection before
 * element_kforms_add_space() consumes it: every axis must use a valid basis
 * family, and a nonzero-order field needs a strictly positive basis order on
 * every axis.
 */
static int element_kforms_check_space(element_kforms_object *this, const function_space_object *const space)
{
    for (unsigned axis = 0; axis < (unsigned)Py_SIZE(space); ++axis)
    {
        if (!basis_set_type_is_valid(space->specs[axis].type))
        {
            PyErr_Format(PyExc_ValueError, "Base space axis %u does not use a valid basis family.", axis);
            return -1;
        }
    }
    for (unsigned field = 0; field < element_kforms_field_count(this->data); ++field)
    {
        const unsigned order = element_kforms_field_order(this->data, field);
        if (order == 0)
            continue;
        for (unsigned axis = 0; axis < (unsigned)Py_SIZE(space); ++axis)
        {
            if (space->specs[axis].order == 0)
            {
                PyErr_Format(PyExc_ValueError,
                             "Base space axis %u has order 0, which cannot carry the order-%u field %s.", axis, order,
                             element_kforms_field_label(this->data, field));
                return -1;
            }
        }
    }
    return 0;
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
    // element_data_add_option() takes the dimension in [1, 63] as a
    // precondition; report an invalid one here instead of aborting there.
    if (ndim < 1 || ndim > 63)
    {
        PyErr_Format(PyExc_ValueError, "Expected ndim in [1, 63], got %zd.", ndim);
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

    if (element_kforms_check_space(this, space) < 0)
        return NULL;
    // Only a failing allocation reaches this point: the space rules are
    // checked by element_kforms_check_space() above.
    unsigned space_index;
    const fdg_result_t space_res = element_kforms_add_space(this->data, space->specs, &space_index);
    if (space_res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not add the base function space option: %s (%s).",
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
        PyErr_Format(PyExc_RuntimeError, "Could not add the element values: %s (%s).", fdg_error_str(res),
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
    // element_data_add_option() takes the dimension in [1, 63] as a
    // precondition; report an invalid one here instead of aborting there.
    if (ndim < 1 || ndim > 63)
    {
        PyErr_Format(PyExc_ValueError, "Expected ndim in [1, 63], got %zd.", ndim);
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
        if (element_kforms_check_space(this, (function_space_object *)space) < 0)
            goto failure;
        // Only a failing allocation reaches this point: the space rules are
        // checked by element_kforms_check_space() above.
        unsigned space_index;
        const fdg_result_t res =
            element_kforms_add_space(this->data, ((function_space_object *)space)->specs, &space_index);
        if (res != FDG_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not add base space %zd: %s (%s).", i, fdg_error_str(res),
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
            PyErr_Format(PyExc_RuntimeError, "Could not add the zero element: %s (%s).", fdg_error_str(res),
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
    // element_data_add_option() takes the dimension in [1, 63] as a
    // precondition; report an invalid one here instead of aborting there.
    if (ndim < 1 || ndim > 63)
    {
        PyErr_Format(PyExc_ValueError, "Expected ndim in [1, 63], got %zd.", ndim);
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
    // element_data_add_option() takes the dimension in [1, 63] as a
    // precondition; report an invalid one here instead of aborting there.
    if (ndim < 1 || ndim > 63)
    {
        PyErr_Format(PyExc_ValueError, "Expected ndim in [1, 63], got %zd.", ndim);
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
    if (!element_kforms_find_field(this->data, label, out_field))
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
    if (res == FDG_ERROR_NOT_IN_DOMAIN)
    {
        // The option disagrees with the dimension fixed by the first option;
        // this rule stays a recoverable error in the C core.
        PyErr_Format(PyExc_ValueError, "Could not add the function space option: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        return NULL;
    }
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not add the function space option: %s (%s).", fdg_error_str(res),
                     fdg_error_msg(res));
        return NULL;
    }
    if (element_dofs_grow_option_objects(this, element_dofs_option_count(this->data)) < 0)
        return NULL;
    // Only a failing allocation reaches this point: the option index and the
    // derived value count are valid by construction.
    const fdg_result_t add_res = element_dofs_add_element(this->data, index, dofs->values);
    if (add_res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not add the element values: %s (%s).", fdg_error_str(add_res),
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
        if (res == FDG_ERROR_NOT_IN_DOMAIN)
        {
            // The space disagrees with the dimension fixed by the first
            // space; this rule stays a recoverable error in the C core.
            PyErr_Format(PyExc_ValueError, "Could not add function space option %zd: %s (%s).", i, fdg_error_str(res),
                         fdg_error_msg(res));
            goto failure;
        }
        if (res != FDG_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Could not add function space option %zd: %s (%s).", i, fdg_error_str(res),
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
        // Only a failing allocation reaches this point: the option index is
        // valid by construction.
        const fdg_result_t res = element_dofs_add_element(this->data, index, zeros);
        if (res != FDG_SUCCESS)
        {
            PyMem_Free(zeros);
            PyErr_Format(PyExc_RuntimeError, "Could not add the zero element: %s (%s).", fdg_error_str(res),
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
