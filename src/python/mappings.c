#include "mappings.h"
#include "kform_transform.h"

#include "../integration/integration_rules.h"
#include "../operations/map_transforms.h"
#include "../polynomials/lagrange.h"
#include "basis_objects.h"
#include "degrees_of_freedom.h"
#include "integration_objects.h"

#include <stdbool.h>

/**
 * Allocate a coordinate map on the given integration space without computing its values;
 * the value blocks are filled afterwards by the caller.
 */
static coordinate_map_object *coordinate_map_object_alloc(PyTypeObject *type, const unsigned ndim,
                                                          const integration_spec_t *specs)
{
    size_t n_vals = 1;
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        n_vals *= specs[idim].order + 1;
    }
    coordinate_map_object *const self =
        (coordinate_map_object *)type->tp_alloc(type, (Py_ssize_t)(n_vals * (ndim + 1)));
    if (!self)
    {
        return NULL;
    }
    self->ndim = ndim;
    self->int_specs = PyMem_Malloc(ndim * sizeof(*self->int_specs));
    if (!self->int_specs)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        self->int_specs[idim] = specs[idim];
    }
    return self;
}

/**
 * Construct a coordinate map that evaluates the given degrees of freedom on
 * the given integration space, including all derivatives.
 */
static coordinate_map_object *coordinate_map_object_create(PyTypeObject *type, dof_object *dofs,
                                                           const integration_space_object *integration_space,
                                                           const integration_registry_object *integration_registry,
                                                           const basis_registry_object *basis_registry)
{
    coordinate_map_object *const self =
        coordinate_map_object_alloc(type, (unsigned)Py_SIZE(integration_space), integration_space->specs);
    if (!self)
    {
        return NULL;
    }

    // Call the reconstruct function on the DoFs
    reconstruction_state_t recon_state;
    if (dof_reconstruction_state_init(dofs, self->ndim, integration_space->specs, integration_registry, basis_registry,
                                      &recon_state) < 0)
    {
        Py_DECREF(self);
        return NULL;
    }
    const Py_ssize_t n_vals = (Py_ssize_t)multidim_iterator_total_size(recon_state.iter_int);
    const unsigned ndofs = Py_SIZE(dofs);
    const double *const restrict pdofs = dofs->values;
    compute_integration_point_values(self->ndim, recon_state.iter_int, recon_state.iter_basis, recon_state.basis_sets,
                                     n_vals, self->values + 0 * n_vals, ndofs, pdofs);

    int *const derivative_array = PyMem_Malloc(sizeof(int) * self->ndim);
    if (!derivative_array)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (unsigned i = 0; i < self->ndim; ++i)
    {
        derivative_array[i] = 0;
    }
    for (unsigned i = 0; i < self->ndim; ++i)
    {
        derivative_array[i] = 1; // Set the current dimension to use derivative
        // Compute with the specified derivatives
        compute_integration_point_values_derivatives(self->ndim, recon_state.iter_int, recon_state.iter_basis,
                                                     recon_state.basis_sets, derivative_array, n_vals,
                                                     self->values + (i + 1) * n_vals, ndofs, pdofs);
        derivative_array[i] = 0; // Reset the current dimension
    }
    PyMem_Free(derivative_array);

    return self;
}

static PyObject *coordinate_map_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(type);
    if (!state)
        return NULL;
    dof_object *dofs;
    const integration_space_object *integration_space;
    const integration_registry_object *integration_registry =
        (integration_registry_object *)state->registry_integration;
    const basis_registry_object *basis_registry = (basis_registry_object *)state->registry_basis;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O!|O!O!:CoordinateMap",
            (char *[]){"dofs", "integration_space", "integration_registry", "basis_registry", NULL},
            state->degrees_of_freedom_type, &dofs, state->integration_space_type, &integration_space,
            state->integration_registry_type, &integration_registry, state->basis_registry_type, &basis_registry))
        return NULL;

    return (PyObject *)coordinate_map_object_create(type, dofs, integration_space, integration_registry,
                                                    basis_registry);
}

static void coordinate_map_dealloc(coordinate_map_object *self)
{
    PyObject_GC_UnTrack(self);
    PyMem_Free(self->int_specs);
    self->int_specs = NULL;
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static int coordinate_map_traverse(coordinate_map_object *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}

static PyObject *coordinate_map_get_dimension(PyObject *self, void *Py_UNUSED(closure))
{
    const coordinate_map_object *const this = (coordinate_map_object *)self;
    return PyLong_FromLong(this->ndim);
}

const double *coordinate_map_values(const coordinate_map_object *map)
{
    return map->values;
}

const double *coordinate_map_gradient(const coordinate_map_object *map, const unsigned dim)
{
    CPYUTL_ASSERT(dim < map->ndim, "Dimension index out of bounds.");
    return map->values + (dim + 1) * integration_specs_total_points(map->ndim, map->int_specs);
}

static PyObject *coordinate_map_get_values(PyObject *self, void *Py_UNUSED(closure))
{
    const coordinate_map_object *const this = (coordinate_map_object *)self;
    npy_intp *const dims = PyMem_Malloc(this->ndim * sizeof(*dims));
    if (!dims)
        return NULL;

    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        dims[idim] = this->int_specs[idim].order + 1;
    }

    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim, dims, NPY_DOUBLE, (void *)this->values);
    PyMem_Free(dims);
    if (!res)
        return NULL;

    CPYUTL_ASSERT(PyArray_SIZE(res) * (this->ndim + 1) == Py_SIZE(self), "These sizes should match!");
    if (PyArray_SetBaseObject((PyArrayObject *)res, (PyObject *)self) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

static PyObject *coordinate_map_get_integration_space(PyObject *self, void *Py_UNUSED(closure))
{
    const coordinate_map_object *const this = (coordinate_map_object *)self;
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    integration_space_object *const res =
        (integration_space_object *)state->integration_space_type->tp_alloc(state->integration_space_type, this->ndim);
    if (!res)
        return NULL;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        res->specs[idim] = this->int_specs[idim];
    }
    return (PyObject *)res;
}

static int ensure_coordinate_map_and_state(PyObject *self, PyTypeObject *defining_class,
                                           const interplib_module_state_t **p_state, coordinate_map_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;

    if (!PyObject_TypeCheck(self, state->coordinate_mapping_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->coordinate_mapping_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_state = state;
    *p_this = (coordinate_map_object *)self;
    return 0;
}

static PyObject *coordinate_map_object_gradient(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                const Py_ssize_t nargs, const PyObject *const kwnames)
{
    const interplib_module_state_t *state;
    coordinate_map_object *this;
    if (ensure_coordinate_map_and_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idx},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (idx < 0 || idx >= this->ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected dimension index in range [0, %zd), but got %zd.", this->ndim, idx);
        return NULL;
    }

    npy_intp *const dims = PyMem_Malloc(this->ndim * sizeof(*dims));
    if (!dims)
        return NULL;

    size_t total_cnt = 1;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        const unsigned dim_size = this->int_specs[idim].order + 1;
        total_cnt *= dim_size;
        dims[idim] = dim_size;
    }

    PyArrayObject *const res = (PyArrayObject *)PyArray_SimpleNewFromData(
        this->ndim, dims, NPY_DOUBLE, (void *)(this->values + total_cnt * (idx + 1)));
    PyMem_Free(dims);
    if (!res)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)res, (PyObject *)self) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

static_assert(sizeof(*((coordinate_map_object *)0xB00B1E5)->values) == sizeof(double), "Nice");

PyDoc_STRVAR(coordinate_map_docstring,
             "CoordinateMap(dofs: DegreesOfFreedom, integration_space: IntegrationSpace, "
             "integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, "
             "basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY)\n"
             "\n"
             "Mapping between reference and physical coordinates.\n"
             "\n"
             "This type wraps :meth:`DegreesOfFreedom.reconstruct_at_integration_points()`\n"
             "and :meth:`DegreesOfFreedom.reconstruct_derivative_at_integration_points()`;\n"
             "one coordinate map evaluates a single coordinate together with all of its\n"
             "first derivatives at every integration point. In N-dimensional space, N such\n"
             "maps are used to represent the full mapping.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "dofs : DegreesOfFreedom\n"
             "    Degrees of freedom that define the coordinate map.\n"
             "integration_space : IntegrationSpace\n"
             "    Integration space used for the mapping.\n"
             "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rules.\n"
             "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
             "    Registry used to retrieve the basis specifications.\n");

PyType_Spec coordinate_map_type_spec = {
    .name = FDG_TYPE_NAME("CoordinateMap"),
    .basicsize = sizeof(coordinate_map_object),
    .itemsize = sizeof(*((coordinate_map_object *)0xB00B1E5)->values),
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, (void *)coordinate_map_traverse},
        {Py_tp_dealloc, coordinate_map_dealloc},
        {Py_tp_new, coordinate_map_new},
        {Py_tp_doc, (void *)coordinate_map_docstring},
        {Py_tp_getset,
         (PyGetSetDef[]){
             {
                 .name = "dimension",
                 .get = coordinate_map_get_dimension,
                 .doc = "int : Number of dimensions in the coordinate map.",
             },
             {
                 .name = "values",
                 .get = coordinate_map_get_values,
                 .doc = "numpy.typing.NDArray[numpy.double] : Mapped coordinate values at the integration points.\n"
                        "\n"
                        "These are the physical coordinates of the map evaluated at every\n"
                        "integration point of this map's own integration space, not\n"
                        "degree-of-freedom coefficients. Do not confuse them with\n"
                        ":attr:`DegreesOfFreedom.values`, which holds the expansion\n"
                        "coefficients passed at construction.",
             },
             {
                 .name = "integration_space",
                 .get = coordinate_map_get_integration_space,
                 .doc = "IntegrationSpace : Integration space used for the mapping.",
             },
             {},
         }},
        {
            Py_tp_methods,
            (PyMethodDef[]){
                {
                    .ml_name = "gradient",
                    .ml_meth = (void *)coordinate_map_object_gradient,
                    .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                    .ml_doc = "gradient(idim: int, /) -> numpy.typing.NDArray[numpy.double]\n"
                              "\n"
                              "Retrieve the gradient of the coordinate map for the given dimension.\n"
                              "\n"
                              "Parameters\n"
                              "----------\n"
                              "idim : int\n"
                              "    Index of the dimension, in range ``[0, dimension)``.\n"
                              "\n"
                              "Returns\n"
                              "-------\n"
                              "array\n"
                              "    Derivative of the mapped coordinate with respect to that dimension,\n"
                              "    sampled at the integration points of the map.\n",
                },
                {},
            },
        },
        {},
    }};

static void space_map_object_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    space_map_object *const this = (space_map_object *)self;
    PyTypeObject *const type = Py_TYPE(this);

    if (this->transformations)
    {
        for (unsigned i = 0; i < this->ndim; ++i)
        {
            Py_XDECREF(this->transformations[i]);
            this->transformations[i] = NULL;
        }
        PyMem_Free((void *)this->transformations);
        this->transformations = NULL;
    }
    this->ndim = 0;
    PyMem_Free(this->int_specs);
    this->int_specs = NULL;
    PyMem_Free(this->determinant);
    this->determinant = NULL;
    PyMem_Free(this->inverse_maps);
    this->inverse_maps = NULL;
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        coordinate_map_object *const map = this->maps[i];
        this->maps[i] = NULL;
        Py_DECREF(map);
    }
    type->tp_free((PyObject *)this);
    Py_DECREF(type);
}

space_map_object *space_map_object_create(PyTypeObject *subtype, const unsigned n_maps,
                                          coordinate_map_object *const *maps)
{
    space_map_object *const this = (space_map_object *)subtype->tp_alloc(subtype, n_maps);
    if (!this)
        return NULL;
    // Zero initialize
    this->ndim = 0;
    this->int_specs = NULL;
    this->determinant = NULL;
    this->inverse_maps = NULL;
    for (unsigned i = 0; i < n_maps; ++i)
        this->maps[i] = NULL;
    this->transformations = NULL;

    // Copy the integration space from the first space, then check all others comply
    coordinate_map_object *const first_map = maps[0];
    if (first_map->ndim > n_maps)
    {
        PyErr_Format(PyExc_ValueError,
                     "Can not construct a space map from reference domain with %u dimensions to %u physical "
                     "dimensions. The number of physical dimensions must be equal to, or greater than the number of "
                     "dimensions of the reference space.",
                     first_map->ndim, n_maps);
        return NULL;
    }
    this->ndim = first_map->ndim;
    this->int_specs = PyMem_Malloc(this->ndim * sizeof(*this->int_specs));
    if (!this->int_specs)
    {
        Py_DECREF(this);
        return NULL;
    }
    for (unsigned i = 0; i < this->ndim; ++i)
    {
        this->int_specs[i] = first_map->int_specs[i];
    }

    this->transformations = (PyArrayObject **)PyMem_Malloc(sizeof(*this->transformations) * this->ndim);
    if (!this->transformations)
    {
        Py_DECREF(this);
        return NULL;
    }
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        this->transformations[idim] = NULL;
    }

    this->maps[0] = first_map;
    Py_INCREF(first_map);

    for (unsigned i = 1; i < n_maps; ++i)
    {
        coordinate_map_object *const map = maps[i];
        if (map->ndim != this->ndim)
        {
            PyErr_Format(PyExc_ValueError,
                         "Expected all coordinate maps to have the same number of dimensions, but "
                         "got %zd and %zd.",
                         this->ndim, map->ndim);
            Py_DECREF(this);
            return NULL;
        }

        if (map->ndim != first_map->ndim)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Expected all coordinate maps to have the same integration space, but the first and %u space have "
                "different integration spaces.",
                i + 1);
            Py_DECREF(this);
            return NULL;
        }

        for (unsigned idim = 0; idim < this->ndim; ++idim)
        {
            if (map->int_specs[idim].order != first_map->int_specs[idim].order ||
                map->int_specs[idim].type != first_map->int_specs[idim].type)
            {
                PyErr_Format(PyExc_ValueError,
                             "Expected all coordinate maps to have the same integration order and type, but got "
                             "order %d and type %d for dimension %u.",
                             first_map->int_specs[idim].order, first_map->int_specs[idim].type, idim);
                Py_DECREF(this);
                return NULL;
            }
        }
        this->maps[i] = map;
        Py_INCREF(map);
    }

    const size_t total_points = integration_specs_total_points(this->ndim, this->int_specs);
    const size_t jacobian_size = (size_t)this->ndim * n_maps;

    // Allocate the output arrays
    double *const determinant = PyMem_Malloc(sizeof(*determinant) * total_points);
    if (!determinant)
    {
        Py_DECREF(this);
        return NULL;
    }
    double *const inverse_maps = PyMem_Malloc(sizeof(*inverse_maps) * total_points * jacobian_size);
    if (!inverse_maps)
    {
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }

    // Allocate work arrays
    double *const jacobian = PyMem_RawMalloc(sizeof(*jacobian) * jacobian_size);
    if (!jacobian)
    {
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }
    double *const q_mat = PyMem_RawMalloc(sizeof(*q_mat) * n_maps * n_maps);
    if (!q_mat)
    {
        PyMem_Free(jacobian);
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }

    // The gradient pointer table adapts the coordinate maps to the pure kernel.
    const double **const gradients = PyMem_Malloc(sizeof(*gradients) * jacobian_size);
    if (!gradients)
    {
        PyMem_RawFree(q_mat);
        PyMem_RawFree(jacobian);
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }
    for (unsigned icoordinate = 0; icoordinate < n_maps; ++icoordinate)
    {
        for (unsigned idim = 0; idim < this->ndim; ++idim)
        {
            gradients[(size_t)icoordinate * this->ndim + idim] = coordinate_map_gradient(this->maps[icoordinate], idim);
        }
    }

    Py_BEGIN_ALLOW_THREADS;
    compute_space_map_determinants(this->ndim, n_maps, gradients, total_points, determinant, inverse_maps, jacobian,
                                   q_mat);
    Py_END_ALLOW_THREADS;

    // Free work arrays
    PyMem_Free(gradients);
    PyMem_RawFree(q_mat);
    PyMem_RawFree(jacobian);

    // Store the output
    this->determinant = determinant;
    this->inverse_maps = inverse_maps;
    // Return
    return this;
}

static PyObject *space_map_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(subtype);
    if (!state)
        return NULL;
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_SetString(PyExc_TypeError, "SpaceMap takes no keyword arguments.");
        return NULL;
    }
    const unsigned n_maps = PyTuple_GET_SIZE(args);
    if (n_maps == 0)
    {
        PyErr_SetString(PyExc_TypeError, "SpaceMap requires at least one argument.");
        return NULL;
    }

    for (unsigned i = 0; i < n_maps; ++i)
    {
        PyObject *const o = PyTuple_GET_ITEM(args, i);
        if (!PyObject_TypeCheck(o, state->coordinate_mapping_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->coordinate_mapping_type->tp_name,
                         Py_TYPE(o)->tp_name);
            return NULL;
        }
    }

    coordinate_map_object **const maps = PyMem_Malloc(sizeof(*maps) * n_maps);
    if (!maps)
        return NULL;
    for (unsigned i = 0; i < n_maps; ++i)
    {
        maps[i] = (coordinate_map_object *)PyTuple_GET_ITEM(args, i);
    }
    space_map_object *const this = space_map_object_create(subtype, n_maps, maps);
    PyMem_Free(maps);
    return (PyObject *)this;
}

static int ensure_space_map_and_state(PyObject *self, PyTypeObject *defining_class,
                                      const interplib_module_state_t **p_state, space_map_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;

    if (!PyObject_TypeCheck(self, state->space_mapping_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->space_mapping_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_state = state;
    *p_this = (space_map_object *)self;
    return 0;
}

static PyObject *space_map_get_coordinate_map(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                              const Py_ssize_t nargs, const PyObject *const kwnames)
{
    const interplib_module_state_t *state;
    space_map_object *this;
    if (ensure_space_map_and_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idx, .kwname = "idx"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (idx < 0 || idx >= Py_SIZE(this))
    {
        PyErr_Format(PyExc_ValueError, "Expected dimension index in range [0, %zd), but got %zd.", Py_SIZE(this), idx);
        return NULL;
    }

    coordinate_map_object *const res = this->maps[idx];
    Py_INCREF(res);
    return (PyObject *)res;
}

PyDoc_STRVAR(space_map_get_coordinate_map_docstring,
             "coordinate_map(idx: int) -> CoordinateMap\n"
             "\n"
             "Return the coordinate map for the specified dimension.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the dimension for which the map should be returned.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "CoordinateMap\n"
             "    Map used for the specified coordinate.\n");

PyDoc_STRVAR(space_map_docstring, "SpaceMap(*coordinates: CoordinateMap)\n"
                                  "\n"
                                  "Mapping between a reference space and a physical space.\n"
                                  "\n"
                                  "A mapping from a reference space to a physical space, which maps the\n"
                                  ":math:`N`-dimensional reference space to an :math:`M`-dimensional\n"
                                  "physical space. With this mapping, it is possible to integrate a\n"
                                  "quantity on a deformed element.\n"
                                  "\n"
                                  "Parameters\n"
                                  "----------\n"
                                  "*coordinates : CoordinateMap\n"
                                  "    Maps for each coordinate of physical space. All of these must be\n"
                                  "    defined on the same :class:`IntegrationSpace`.\n");

static PyObject *space_map_get_input_dimension(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    return PyLong_FromLong(this->ndim);
}

static PyObject *space_map_get_output_dimension(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    return PyLong_FromLong(Py_SIZE(this));
}

static PyObject *space_map_get_determinant(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    if (!this->determinant)
    {
        PyErr_SetString(PyExc_NotImplementedError, "The determinant of the mapping is not yet implemented.");
        return NULL;
    }

    // Create the dims array
    npy_intp *const dims = PyMem_Malloc(sizeof(*dims) * this->ndim);
    if (!dims)
        return NULL;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        dims[idim] = this->int_specs[idim].order + 1;
    }
    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim, dims, NPY_DOUBLE, this->determinant);
    PyMem_Free(dims);
    if (!res)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject(res, (PyObject *)this) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

static PyObject *space_map_get_integration_space(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    integration_space_object *const res =
        (integration_space_object *)state->integration_space_type->tp_alloc(state->integration_space_type, this->ndim);
    if (!res)
        return NULL;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        res->specs[idim] = this->int_specs[idim];
    }
    return (PyObject *)res;
}

static PyObject *space_map_get_inverse_map(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    npy_intp *const dims = PyMem_Malloc(sizeof(*dims) * (this->ndim + 2));
    if (!dims)
        return NULL;

    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        dims[idim] = this->int_specs[idim].order + 1;
    }
    dims[this->ndim] = this->ndim;
    dims[this->ndim + 1] = Py_SIZE(this);

    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim + 2, dims, NPY_DOUBLE, this->inverse_maps);
    PyMem_Free(dims);
    if (!res)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject(res, (PyObject *)this) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

PyDoc_STRVAR(space_map_get_inverse_map_docstring,
             "numpy.typing.NDArray[numpy.double] : Local inverse transformation at each integration point.\n"
             "\n"
             "This array contains inverse mapping matrix, which is used\n"
             "for the contravarying components. When the dimension of the\n"
             "mapping space (as counted by :attr:`SpaceMap.output_dimensions`)\n"
             "is greater than the dimension of the reference space, this is a\n"
             "rectangular matrix, such that it maps the (rectangular) Jacobian\n"
             "to the identity matrix.\n");

PyArrayObject *compute_basis_transform_impl(const space_map_object *map, const Py_ssize_t order)
{
    const unsigned n_maps = Py_SIZE(map);
    const unsigned n_dims = map->ndim;
    if (order <= 0 || order > n_dims)
    {
        PyErr_Format(PyExc_ValueError, "Expected order in range (0, %u], but got %zd.", n_dims, order);
        return NULL;
    }

    if (map->transformations[order - 1] != NULL)
    {
        Py_INCREF(map->transformations[order - 1]);
        return map->transformations[order - 1];
    }

    const size_t total_points = integration_specs_total_points(n_dims, map->int_specs);
    const npy_intp out_dims[3] = {
        combination_total_count(n_dims, order),
        combination_total_count(n_maps, order),
        (npy_intp)total_points,
    };
    PyArrayObject *const res = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_DOUBLE);
    if (!res)
        return NULL;

    // The transform only touches raw memory buffers, so it runs without the GIL; the
    // system allocator backs the iterator scratch because pymalloc requires the GIL.
    int status;
    Py_BEGIN_ALLOW_THREADS;
    status = compute_basis_transform_from_inverse(&SYSTEM_ALLOCATOR, n_dims, n_maps, (unsigned)order, map->inverse_maps,
                                                  map->determinant, total_points, PyArray_DATA(res));
    Py_END_ALLOW_THREADS;
    if (status < 0)
    {
        if (!PyErr_Occurred())
        {
            PyErr_NoMemory();
        }
        Py_DECREF(res);
        return NULL;
    }
    map->transformations[order - 1] = res;
    Py_INCREF(res);
    return res;
}

static PyObject *space_map_basis_transform(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, PyObject *kwnames)
{
    const interplib_module_state_t *state;
    const space_map_object *map;

    if (ensure_space_map_and_state(self, defining_class, &state, (space_map_object **)&map) < 0)
        return NULL;

    Py_ssize_t order;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    return (PyObject *)compute_basis_transform_impl(map, order);
}

PyDoc_STRVAR(space_map_basis_transform_docstring,
             "basis_transform(order: int) -> numpy.typing.NDArray[numpy.double]\n"
             "\n"
             "Compute the matrix with transformation factors for k-form basis.\n"
             "\n"
             "Basis transform matrix returned by this function specifies how at integration\n"
             "point a basis from the reference domain contributes to the basis in the target\n"
             "domain.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "    Order of the k-form for which this is to be done, in range\n"
             "    ``(0, input_dimensions]``.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array with three axes. The first indexes over the input basis, the second\n"
             "    over output basis, and the last one over integration points.\n");

PyDoc_STRVAR(space_map_boundary_docstring,
             "boundary(idim: int, end: bool = False, integration_space: IntegrationSpace = ..., *,\n"
             "         integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY) -> SpaceMap\n"
             "\n"
             "Extract a space map restricted to a reference-space boundary.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idim : int\n"
             "    Index of the reference dimension that is fixed.\n"
             "\n"
             "end : bool, default: False\n"
             "    Select the upper boundary at ``+1`` when true; otherwise select the lower\n"
             "    boundary at ``-1``.\n"
             "\n"
             "integration_space : IntegrationSpace, default: the element space\n"
             "    Face integration space used to sample the extracted map. When omitted,\n"
             "    the volume integration space with the fixed axis removed is used.\n"
             "\n"
             "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry to get the element and face quadrature rules from.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "SpaceMap\n"
             "    Mapping from the remaining reference dimensions to the same physical\n"
             "    coordinates. This map provides the tangential pullback and positive\n"
             "    surface measure for forms on this element face.\n");

/** Operator applied along one axis of an integration-point value tensor. */
typedef struct
{
    unsigned element_axis; // Element axis the operator acts on.
    unsigned axis;         // Element axis with earlier contracted axes removed.
    unsigned face_axis;    // Face axis, for resampling operators.
    unsigned out_nodes;    // Node count after the operator; contracted axes have one.
    bool contract;         // Fixed axis: the axis is removed after the operator.
    bool slice;            // Contracted Gauss-Lobatto axis: copy the plane of slice_index.
    unsigned slice_index;  // Picked node for slice operators.
    double plane;          // Evaluation plane for dense contractions.
    const double *weights; // Dense operator; entry (k, j) at weights[k + j*out_nodes] is the
                           // value of the j-th element node polynomial at the k-th output node.
} boundary_axis_operator_t;

/**
 * Apply one boundary axis operator to a last-axis-fastest tensor.
 *
 * @param op Operator to apply.
 * @param ndim Tensor dimension before the operator.
 * @param dims Node counts of the tensor axes before the operator.
 * @param in Input tensor.
 * @param out Output tensor sized for the tensor after the operator, written
 *            in full. Must not alias @p in.
 */
static void boundary_axis_apply(const boundary_axis_operator_t *op, const unsigned ndim,
                                const unsigned dims[static ndim], const double *restrict in, double *restrict out)
{
    size_t outer = 1, inner = 1;
    for (unsigned i = 0; i < op->axis; ++i)
    {
        outer *= dims[i];
    }
    for (unsigned i = op->axis + 1; i < ndim; ++i)
    {
        inner *= dims[i];
    }
    const unsigned in_nodes = dims[op->axis];

    if (op->slice)
    {
        for (size_t o = 0; o < outer; ++o)
        {
            memcpy(out + o * inner, in + (o * in_nodes + op->slice_index) * inner, inner * sizeof(*out));
        }
        return;
    }

    // Dense interpolation along the axis, accumulating over the input nodes.
    memset(out, 0, outer * op->out_nodes * inner * sizeof(*out));
    for (size_t o = 0; o < outer; ++o)
    {
        double *restrict out_block = out + o * op->out_nodes * inner;
        for (unsigned k = 0; k < op->out_nodes; ++k)
        {
            double *restrict out_line = out_block + k * inner;
            for (unsigned j = 0; j < in_nodes; ++j)
            {
                const double weight = op->weights[k + (size_t)j * op->out_nodes];
                const double *restrict in_line = in + (o * in_nodes + j) * inner;
#pragma omp simd
                for (size_t i = 0; i < inner; ++i)
                {
                    out_line[i] += weight * in_line[i];
                }
            }
        }
    }
}

/**
 * Restrict a space map to a boundary in a single values-level pass.
 *
 * Every value block (values and gradients along surviving axes) of each
 * coordinate map is sampled by contracting the fixed axes with the
 * interpolant at their planes and resampling the surviving axes onto the
 * face grid. Exact whenever the integration order is at least the dof order
 * along every axis.
 *
 * @param state Interpreter module state.
 * @param integration_registry Registry supplying the element and face rules.
 * @param map Space map to restrict.
 * @param bdim Number of dimensions of the boundary, `1 <= bdim <= map->ndim`.
 * @param orientation Full element-dimension orientation of the boundary; the
 *                    first @p bdim entries are the signed fixed axes.
 * @param provided_face_space Optional face integration space replacing the
 *                            default element space without the fixed axes.
 * @return The restricted space map, or NULL with a Python exception set.
 */
static space_map_object *space_map_boundary_grid_impl(const interplib_module_state_t *state,
                                                      integration_registry_object *const integration_registry,
                                                      const space_map_object *map, const unsigned bdim,
                                                      const int8_t *orientation,
                                                      const integration_space_object *provided_face_space)
{
    const unsigned ndim = map->ndim;
    const unsigned face_ndim = ndim - bdim;
    const Py_ssize_t n_coordinates = Py_SIZE(map);
    CUTL_ASSERT(1 <= bdim && bdim <= ndim, "Boundary dimension out of range.");
    CUTL_ASSERT(!provided_face_space || Py_SIZE(provided_face_space) == (Py_ssize_t)face_ndim,
                "Face space dimension mismatch.");

    // The default face integration space is the element space with the fixed normal axes
    // removed, so the face grid coincides with the element grid on the surviving axes.
    bool is_fixed[UINT8_MAX] = {false};
    for (unsigned entry = 0; entry < bdim; ++entry)
    {
        const int8_t axis_code = orientation[entry];
        is_fixed[(unsigned)(axis_code < 0 ? -axis_code : axis_code) - 1] = true;
    }
    const integration_spec_t *face_specs;
    integration_spec_t derived_specs[UINT8_MAX];
    if (provided_face_space)
    {
        face_specs = provided_face_space->specs;
    }
    else
    {
        unsigned face_dim = 0;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (!is_fixed[axis])
            {
                derived_specs[face_dim++] = map->int_specs[axis];
            }
        }
        CUTL_ASSERT(face_dim == face_ndim, "Face space dimension mismatch.");
        face_specs = derived_specs;
    }

    // Face axis index of every surviving element axis.
    unsigned face_axis_of[UINT8_MAX];
    {
        unsigned face_axis = 0;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (!is_fixed[axis])
            {
                face_axis_of[axis] = face_axis++;
            }
        }
    }

    // Contract the fixed axes first, then resample the surviving axes onto the face grid.
    boundary_axis_operator_t operators[UINT8_MAX];
    unsigned n_operators = 0;
    size_t weights_count = 0;
    for (unsigned entry = 0; entry < bdim; ++entry)
    {
        const int8_t axis_code = orientation[entry];
        const unsigned axis = (unsigned)(axis_code < 0 ? -axis_code : axis_code) - 1;
        const integration_spec_t *const axis_spec = &map->int_specs[axis];
        boundary_axis_operator_t *const op = &operators[n_operators++];
        *op = (boundary_axis_operator_t){
            .element_axis = axis,
            .axis = axis,
            .out_nodes = 1,
            .contract = true,
            .plane = axis_code > 0 ? 1.0 : -1.0,
        };
        if (axis_spec->type == INTEGRATION_RULE_TYPE_GAUSS_LOBATTO)
        {
            // The endpoints are nodes, so evaluating at the plane picks the endpoint slice.
            op->slice = true;
            op->slice_index = axis_code > 0 ? axis_spec->order : 0;
        }
        else
        {
            weights_count += axis_spec->order + 1;
        }
    }
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        if (is_fixed[axis])
        {
            continue;
        }
        const unsigned face_axis = face_axis_of[axis];
        if (map->int_specs[axis].order == face_specs[face_axis].order &&
            map->int_specs[axis].type == face_specs[face_axis].type)
        {
            continue;
        }
        boundary_axis_operator_t *const op = &operators[n_operators++];
        *op = (boundary_axis_operator_t){
            .element_axis = axis,
            .axis = axis,
            .face_axis = face_axis,
            .out_nodes = face_specs[face_axis].order + 1,
        };
        weights_count += (size_t)op->out_nodes * (map->int_specs[axis].order + 1);
    }
    // Contracted axes shift the index of every operator after them.
    for (unsigned i = 0; i < n_operators; ++i)
    {
        unsigned seen = 0;
        for (unsigned axis = 0; axis < operators[i].element_axis; ++axis)
        {
            seen += is_fixed[axis];
        }
        operators[i].axis -= seen;
    }

    // The dense operators are built from the element and face integration rule nodes.
    bool needs_element_nodes = false, needs_face_nodes = false;
    for (unsigned i = 0; i < n_operators; ++i)
    {
        if (!operators[i].slice)
        {
            needs_element_nodes = true;
            if (!operators[i].contract)
            {
                needs_face_nodes = true;
            }
        }
    }
    const integration_rule_t **element_rules = NULL;
    const integration_rule_t **face_rules = NULL;
    space_map_object *result = NULL;
    coordinate_map_object **coordinates = NULL;
    double *weights_block = NULL;
    double *scratch = NULL;
    Py_ssize_t n_created = 0;
    if (needs_element_nodes)
    {
        element_rules = python_integration_rules_get(ndim, map->int_specs, integration_registry->registry);
        if (!element_rules)
        {
            goto fail;
        }
    }
    if (needs_face_nodes)
    {
        face_rules = python_integration_rules_get(face_ndim, face_specs, integration_registry->registry);
        if (!face_rules)
        {
            goto fail;
        }
    }
    if (weights_count)
    {
        weights_block = PyMem_Malloc(weights_count * sizeof(*weights_block));
        if (!weights_block)
        {
            PyErr_NoMemory();
            goto fail;
        }
    }
    double *weights_cursor = weights_block;
    double weights_work[UINT8_MAX];
    for (unsigned i = 0; i < n_operators; ++i)
    {
        boundary_axis_operator_t *const op = &operators[i];
        if (op->slice)
        {
            continue;
        }
        const unsigned in_nodes = map->int_specs[op->element_axis].order + 1;
        double *const op_weights = weights_cursor;
        weights_cursor += (size_t)op->out_nodes * in_nodes;
        op->weights = op_weights;
        if (op->contract)
        {
            lagrange_polynomial_values_transposed(1, &op->plane, in_nodes,
                                                  integration_rule_nodes_const(element_rules[op->element_axis]),
                                                  op_weights, weights_work);
        }
        else
        {
            lagrange_polynomial_values_transposed(
                op->out_nodes, integration_rule_nodes_const(face_rules[op->face_axis]), in_nodes,
                integration_rule_nodes_const(element_rules[op->element_axis]), op_weights, weights_work);
        }
    }

    const size_t element_points = integration_specs_total_points(ndim, map->int_specs);
    const size_t face_points = integration_specs_total_points(face_ndim, face_specs);
    // Intermediates alternate between two scratch buffers. They shrink with every
    // contraction and may grow with every resampling, so every stage is measured.
    size_t scratch_size;
    {
        size_t running = element_points;
        scratch_size = running;
        for (unsigned i = 0; i < n_operators; ++i)
        {
            const boundary_axis_operator_t *const op = &operators[i];
            running = running / (map->int_specs[op->element_axis].order + 1);
            if (!op->contract)
            {
                running = running * op->out_nodes;
            }
            scratch_size = scratch_size > running ? scratch_size : running;
        }
    }
    scratch = PyMem_Malloc(2 * scratch_size * sizeof(*scratch));
    coordinates = PyMem_Malloc(sizeof(*coordinates) * (size_t)n_coordinates);
    if (!scratch || !coordinates)
    {
        goto fail;
    }

    for (Py_ssize_t icoordinate = 0; icoordinate < n_coordinates; ++icoordinate)
    {
        const coordinate_map_object *const source = map->maps[icoordinate];
        // A restricted map is defined by its sampled values: the boundary pass writes
        // every value block, values first and then the gradients along the free axes.
        coordinate_map_object *const face =
            coordinate_map_object_alloc(state->coordinate_mapping_type, face_ndim, face_specs);
        if (!face)
        {
            goto fail;
        }
        coordinates[icoordinate] = face;
        n_created = icoordinate + 1;

        unsigned face_block = 0;
        for (unsigned block = 0; block <= ndim; ++block)
        {
            if (block > 0 && is_fixed[block - 1])
            {
                continue;
            }
            const double *in = source->values + (size_t)block * element_points;
            double *const dst = face->values + (size_t)face_block * face_points;
            ++face_block;

            unsigned cur_dims[UINT8_MAX];
            unsigned cur_ndim = ndim;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                cur_dims[axis] = map->int_specs[axis].order + 1;
            }
            for (unsigned i = 0; i < n_operators; ++i)
            {
                const boundary_axis_operator_t *const op = &operators[i];
                double *const out = (i + 1 == n_operators) ? dst : scratch + (size_t)(i & 1) * scratch_size;
                boundary_axis_apply(op, cur_ndim, cur_dims, in, out);
                if (op->contract)
                {
                    for (unsigned axis = op->axis; axis + 1 < cur_ndim; ++axis)
                    {
                        cur_dims[axis] = cur_dims[axis + 1];
                    }
                    --cur_ndim;
                }
                else
                {
                    cur_dims[op->axis] = op->out_nodes;
                }
                in = out;
            }
        }
    }

    result = space_map_object_create(state->space_mapping_type, (unsigned)n_coordinates, coordinates);

fail:
    if (face_rules)
    {
        python_integration_rules_release(face_ndim, face_rules, integration_registry->registry);
    }
    if (element_rules)
    {
        python_integration_rules_release(ndim, element_rules, integration_registry->registry);
    }
    PyMem_Free(weights_block);
    PyMem_Free(scratch);
    for (Py_ssize_t icoordinate = 0; icoordinate < n_created; ++icoordinate)
    {
        Py_DECREF(coordinates[icoordinate]);
    }
    PyMem_Free(coordinates);
    return result;
}

space_map_object *space_map_boundary_oriented_impl(const interplib_module_state_t *state,
                                                   integration_registry_object *const integration_registry,
                                                   const space_map_object *map, const unsigned bdim,
                                                   const int8_t *orientation)
{
    return space_map_boundary_grid_impl(state, integration_registry, map, bdim, orientation, NULL);
}

space_map_object *space_map_boundary_impl(const interplib_module_state_t *state,
                                          integration_registry_object *const integration_registry,
                                          const space_map_object *map, const unsigned idim, const int end,
                                          integration_space_object *provided_face_space)
{
    // The restricted map follows from sampling the element grid: the fixed axis is
    // evaluated at the plane and the surviving axes are resampled onto the face grid.
    const unsigned ndim = map->ndim;
    int8_t orientation[UINT8_MAX];
    orientation[0] = (int8_t)((idim + 1) * (end ? 1 : -1));
    unsigned slot = 1;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        if (axis != idim)
        {
            orientation[slot] = (int8_t)(axis + 1);
            ++slot;
        }
    }
    return space_map_boundary_grid_impl(state, integration_registry, map, 1, orientation, provided_face_space);
}

static PyObject *space_map_boundary(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                    const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    space_map_object *this;
    if (ensure_space_map_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    Py_ssize_t idim;
    int end = 0;
    integration_space_object *provided_face_space = NULL;
    integration_registry_object *integration_registry = (integration_registry_object *)state->registry_integration;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idim, .kwname = "idim"},
                {.type = CPYARG_TYPE_BOOL, .p_val = &end, .kwname = "end", .optional = 1},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &provided_face_space,
                 .type_check = state->integration_space_type,
                 .kwname = "integration_space",
                 .optional = 1},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &integration_registry,
                 .type_check = state->integration_registry_type,
                 .kwname = "integration_registry",
                 .optional = 1,
                 .kw_only = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (this->ndim < 1)
    {
        PyErr_SetString(PyExc_ValueError, "Boundary space maps require a non-empty input space.");
        return NULL;
    }
    if (idim < 0 || idim >= this->ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected a boundary dimension in range [0, %u), got %zd.", this->ndim, idim);
        return NULL;
    }
    if (provided_face_space && Py_SIZE(provided_face_space) != this->ndim - 1)
    {
        PyErr_Format(PyExc_ValueError, "Expected a face integration space with %u dimensions, got %zd.", this->ndim - 1,
                     Py_SIZE(provided_face_space));
        return NULL;
    }

    return (PyObject *)space_map_boundary_impl(state, integration_registry, this, (unsigned)idim, end,
                                               provided_face_space);
}

PyType_Spec space_map_type_spec = {
    .name = FDG_TYPE_NAME("SpaceMap"),
    .basicsize = sizeof(space_map_object),
    .itemsize = sizeof(coordinate_map_object),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, space_map_object_dealloc},
            {Py_tp_new, space_map_new},
            {Py_tp_doc, (void *)space_map_docstring},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {
                     .name = "input_dimensions",
                     .get = space_map_get_input_dimension,
                     .doc = "int : Dimension of the input/reference space.",
                 },
                 {
                     .name = "output_dimensions",
                     .get = space_map_get_output_dimension,
                     .doc = "int : Dimension of the output/physical space.",
                 },
                 {
                     .name = "determinant",
                     .get = space_map_get_determinant,
                     .doc = "numpy.typing.NDArray[numpy.double] : Array with the values of determinant at integration "
                            "points.",
                 },
                 {
                     .name = "integration_space",
                     .get = space_map_get_integration_space,
                     .doc = "IntegrationSpace : Integration space used by the mapping.",
                 },
                 {
                     .name = "inverse_map",
                     .get = space_map_get_inverse_map,
                     .doc = space_map_get_inverse_map_docstring,
                 },
                 {},
             }},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "coordinate_map",
                     .ml_meth = (void *)space_map_get_coordinate_map,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = space_map_get_coordinate_map_docstring,
                 },
                 {
                     .ml_name = "basis_transform",
                     .ml_meth = (void *)space_map_basis_transform,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = space_map_basis_transform_docstring,
                 },
                 {
                     .ml_name = "boundary",
                     .ml_meth = (void *)space_map_boundary,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = space_map_boundary_docstring,
                 },
                 {},
             }},
            {},
        },
};

size_t space_map_inverse_size_per_integration_point(const space_map_object *map)
{
    return map->ndim * Py_SIZE(map);
}

double space_map_forward_derivative(const space_map_object *map, const size_t integration_point_index,
                                    const unsigned idx_dim, const unsigned idx_coord)
{
    const coordinate_map_object *const map_dim = map->maps[idx_coord];
    return coordinate_map_gradient(map_dim, idx_dim)[integration_point_index];
}

double space_map_backward_derivative(const space_map_object *map, const size_t integration_point_index,
                                     const unsigned idx_dim, const unsigned idx_coord)
{
    const size_t n_coords = Py_SIZE(map);
    const size_t jacobian_size = (size_t)map->ndim * n_coords;
    return map->inverse_maps[integration_point_index * jacobian_size + idx_dim * n_coords + idx_coord];
}

const double *space_map_inverse_at_integration_point(const space_map_object *map, const size_t flat_index)
{
    return map->inverse_maps + flat_index * space_map_inverse_size_per_integration_point(map);
}

static int prepare_component_transform(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames,
                                       PyArrayObject **p_components, PyArrayObject **p_out_array,
                                       size_t *p_total_points, unsigned *p_ndim_out, unsigned *p_ndim_in,
                                       const space_map_object **p_map)
{
    interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return -1;

    const space_map_object *map;
    PyObject *py_components, *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .type_check = state->space_mapping_type, .p_val = &map, .kwname = "smap"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_components, .kwname = "components"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &out, .kwname = "out", .optional = 1, .kw_only = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return -1;

    // If NULL is given for "out", it should be the same as it not being given.
    if (out != NULL && Py_IsNone(out))
    {
        out = NULL;
    }

    // Get the shape of the transformation
    const unsigned ndim_in = map->ndim;
    const unsigned ndim_out = (unsigned)Py_SIZE(map);

    // Convert components to be an array
    PyArrayObject *const components =
        (PyArrayObject *)PyArray_FROMANY(py_components, NPY_DOUBLE, 2, 0, NPY_ARRAY_IN_ARRAY);
    if (!components)
        return -1;

    // Check the shape is correct
    const npy_intp *const dims_in = PyArray_DIMS(components);
    const unsigned ndim_components = PyArray_NDIM(components);
    if (ndim_components != 1 + ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected components to have %u dimensions, but got %u.", 1 + ndim_in,
                     ndim_components);
        Py_DECREF(components);
        return -1;
    }
    if (dims_in[0] != ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected components to have shape (%u, ...), but got (%zd, ...).", ndim_in,
                     dims_in[0]);
        Py_DECREF(components);
        return -1;
    }
    // The other dimensions must match the integration rule used by the space map
    for (unsigned idim = 0; idim < ndim_in; ++idim)
    {
        const npy_intp size_in = dims_in[idim + 1];
        if (size_in != map->int_specs[idim].order + 1)
        {
            PyErr_Format(PyExc_ValueError,
                         "Components dimension %u did not match the integration rule of order %u as specified by the "
                         "space map (instead it was %zd).",
                         idim, map->int_specs[idim].order, size_in);
            Py_DECREF(components);
            return -1;
        }
    }

    // Create an output array if needed
    PyArrayObject *out_array;
    if (out == NULL)
    {
        // Create the output array
        npy_intp *const dims_out = PyMem_Malloc(sizeof(*dims_out) * ndim_components);
        if (!dims_out)
        {
            Py_DECREF(components);
            return -1;
        }
        dims_out[0] = ndim_out;
        for (unsigned idim = 1; idim < ndim_components; ++idim)
        {
            dims_out[idim] = dims_in[idim];
        }
        out_array = (PyArrayObject *)PyArray_SimpleNew(ndim_components, dims_out, NPY_DOUBLE);
        PyMem_Free(dims_out);
        if (!out_array)
        {
            Py_DECREF(components);
            return -1;
        }
    }
    else
    {
        // We were given one
        if (!PyArray_Check(out))
        {
            PyErr_Format(PyExc_TypeError, "Expected out to be an array, but got %s.", Py_TYPE(out)->tp_name);
            Py_DECREF(components);
            return -1;
        }
        out_array = (PyArrayObject *)out;
        // Check the shape is correct
        const npy_intp *const dims_out = PyArray_DIMS(out_array);
        if (dims_out[0] != ndim_out)
        {
            PyErr_Format(PyExc_ValueError, "Expected output to have shape (%u, ...), but got (%zd, ...).", ndim_out,
                         dims_out[0]);
            Py_DECREF(components);
            return -1;
        }
        Py_INCREF(out_array);
    }

    size_t total_points = 1;
    for (unsigned idim = 1; idim < ndim_components; ++idim)
    {
        total_points *= dims_in[idim];
    }

    *p_components = components;
    *p_out_array = out_array;
    *p_total_points = total_points;
    *p_ndim_out = ndim_out;
    *p_ndim_in = ndim_in;
    *p_map = map;

    return 0;
}

static PyObject *transform_contravariant_to_target(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                                   PyObject *kwnames)
{
    PyArrayObject *components, *out_array;
    size_t total_points;
    unsigned ndim_out, ndim_in;
    const space_map_object *map;
    if (prepare_component_transform(mod, args, nargs, kwnames, &components, &out_array, &total_points, &ndim_out,
                                    &ndim_in, &map) < 0)
        return NULL;

    const npy_double *restrict const ptr_components = PyArray_DATA(components);
    npy_double *restrict const ptr_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < total_points; ++i)
    {
#pragma omp simd
        for (unsigned i_out = 0; i_out < ndim_out; ++i_out)
        {
            double val = 0.0;
            for (unsigned i_in = 0; i_in < ndim_in; ++i_in)
            {
                val += ptr_components[i_in * total_points + i] * space_map_forward_derivative(map, i, i_in, i_out);
            }
            ptr_out[i_out * total_points + i] = val;
        }
    }

    return (PyObject *)out_array;
}

PyDoc_STRVAR(transform_contravariant_to_target_docstring,
             "transform_contravariant_to_target(smap: SpaceMap, components: numpy.typing.ArrayLike, *, out: "
             "numpy.typing.NDArray[numpy.double] | None = None) -> numpy.typing.NDArray[numpy.double]\n"
             "\n"
             "Transform contravariant vector components from reference to target domain.\n"
             "\n"
             "Since the basis of 1-forms are covectors, which are as the name implies covarying,\n"
             "the values of components are contravarying. Once transformed to the target domain,\n"
             "the 1-form can be lowered to a tangent vector field trivially.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "smap : SpaceMap\n"
             "    Mapping from the reference space to the physical space to use to transform the\n"
             "    components.\n"
             "\n"
             "components : array_like\n"
             "    Array whose first dimension indexes the components in the reference space and\n"
             "    has length ``input_dimensions``. The remaining dimensions must match the\n"
             "    integration grid of the space map (``order + 1`` nodes per reference dimension).\n"
             "\n"
             "out : array, optional\n"
             "    Array to use to write the resulting transformed components to. If it is not\n"
             "    specified, a new array is created.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of transformed contravariant components. If the ``out`` parameter was given,\n"
             "    a new reference to it is returned, otherwise a reference to the newly created\n"
             "    output array is returned.\n");

static void transform_covariant_to_target_impl(const size_t total_points,
                                               const double ptr_components[static restrict const total_points],
                                               double ptr_out[restrict const total_points], const unsigned ndim_out,
                                               const unsigned ndim_in, const space_map_object *map)
{
    for (size_t i = 0; i < total_points; ++i)
    {
#pragma omp simd
        for (unsigned i_out = 0; i_out < ndim_out; ++i_out)
        {
            double val = 0.0;
            for (unsigned i_in = 0; i_in < ndim_in; ++i_in)
            {
                val += ptr_components[i_in * total_points + i] * space_map_backward_derivative(map, i, i_in, i_out);
            }
            ptr_out[i_out * total_points + i] = val;
        }
    }
}

static PyObject *transform_covariant_to_target(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                               PyObject *kwnames)
{
    PyArrayObject *components, *out_array;
    size_t total_points;
    unsigned ndim_out, ndim_in;
    const space_map_object *map;
    if (prepare_component_transform(mod, args, nargs, kwnames, &components, &out_array, &total_points, &ndim_out,
                                    &ndim_in, &map) < 0)
        return NULL;

    transform_covariant_to_target_impl(total_points, PyArray_DATA(components), PyArray_DATA(out_array), ndim_out,
                                       ndim_in, map);
    return (PyObject *)out_array;
}

static inline unsigned space_map_transform_input_dimensions(const void *object)
{
    return ((const space_map_object *)object)->ndim;
}

static inline unsigned space_map_transform_output_dimensions(const void *object)
{
    return (unsigned)Py_SIZE((const space_map_object *)object);
}

static inline npy_intp space_map_transform_point_axis_size(const void *object, const unsigned axis)
{
    return ((const space_map_object *)object)->int_specs[axis].order + 1;
}

static inline const double *space_map_transform_inverse_maps(const void *object)
{
    return ((const space_map_object *)object)->inverse_maps;
}

static inline PyArrayObject *space_map_transform_basis(const void *object, const Py_ssize_t order)
{
    return compute_basis_transform_impl((const space_map_object *)object, order);
}

static const kform_transform_operations_t space_map_transform_operations = {
    .input_dimensions = space_map_transform_input_dimensions,
    .output_dimensions = space_map_transform_output_dimensions,
    .point_axis_size = space_map_transform_point_axis_size,
    .inverse_maps = space_map_transform_inverse_maps,
    .basis_transform = space_map_transform_basis,
};

static PyObject *transform_kform_to_target(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                           PyObject *kwnames)
{
    interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    Py_ssize_t order;
    const space_map_object *map;
    PyObject *py_components, *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {.type = CPYARG_TYPE_PYTHON, .type_check = state->space_mapping_type, .p_val = &map, .kwname = "smap"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_components, .kwname = "components"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &out, .kwname = "out", .optional = 1, .kw_only = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const kform_transform_request_t request = {
        .map = map,
        .operations = &space_map_transform_operations,
        .order = order,
        .minimum_order = 0,
        .components = py_components,
        .out = out,
    };
    kform_transform_arrays_t arrays;
    if (kform_transform_prepare(&request, &arrays) < 0)
        return NULL;
    if (kform_transform_apply(&request, &arrays) < 0)
    {
        kform_transform_arrays_clear(&arrays);
        return NULL;
    }
    Py_DECREF(arrays.components);
    return (PyObject *)arrays.out;
}

static PyObject *transform_kform_component_to_target(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                                     PyObject *kwnames)
{
    interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    Py_ssize_t order, index;
    const space_map_object *map;
    PyObject *py_component, *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {.type = CPYARG_TYPE_PYTHON, .type_check = state->space_mapping_type, .p_val = &map, .kwname = "smap"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_component, .kwname = "component"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &index, .kwname = "index"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &out, .kwname = "out", .optional = 1, .kw_only = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // If NULL is given for "out", it should be the same as it not being given.
    if (out != NULL && Py_IsNone(out))
    {
        out = NULL;
    }

    // Get the shape of the transformation
    const unsigned ndim_in = map->ndim;
    const unsigned ndim_out = (unsigned)Py_SIZE(map);
    const unsigned n_components_in = combination_total_count(ndim_in, order);
    const unsigned n_components_out = combination_total_count(ndim_out, order);

    // Check order
    if (order < 0 || order > ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected order to be between 0 and %u, but got %zd.", ndim_in, order);
        return NULL;
    }

    // Check index
    if (index < 0 || index >= n_components_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected index to be between 0 and %u, but got %zd.", n_components_in - 1,
                     index);
        return NULL;
    }

    // Convert components to be an array
    PyArrayObject *const component =
        (PyArrayObject *)PyArray_FROMANY(py_component, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
    if (!component)
        return NULL;

    // Check the shape is correct
    const npy_intp *const dims_in = PyArray_DIMS(component);
    const unsigned ndim_component = PyArray_NDIM(component);

    if (ndim_component < ndim_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected component to have at least %u dimensions, but got %u.", ndim_in,
                     ndim_component);
        Py_DECREF(component);
        return NULL;
    }

    const unsigned extra_dims = ndim_component - ndim_in;
    size_t input_arrays = 1;
    for (unsigned i = 0; i < extra_dims; ++i)
        input_arrays *= (size_t)dims_in[i];

    // The other dimensions must match the integration rule used by the space map
    for (unsigned idim = 0; idim < ndim_in; ++idim)
    {
        const npy_intp size_in = dims_in[idim + extra_dims];
        if (size_in != map->int_specs[idim].order + 1)
        {
            PyErr_Format(PyExc_ValueError,
                         "Components dimension %u did not match the integration rule of order %u as specified by the "
                         "space map (instead it was %zd).",
                         idim, map->int_specs[idim].order, size_in);
            Py_DECREF(component);
            return NULL;
        }
    }

    // Create an output array if needed
    PyArrayObject *out_array;
    if (out == NULL)
    {
        const unsigned out_ndim = ndim_component + 1;
        // Create the output array
        npy_intp *const dims_out = PyMem_Malloc(sizeof(*dims_out) * out_ndim);
        if (!dims_out)
        {
            Py_DECREF(component);
            return NULL;
        }
        for (unsigned idim = 0; idim < extra_dims; ++idim)
        {
            dims_out[idim] = dims_in[idim];
        }
        dims_out[extra_dims] = n_components_out;
        for (unsigned idim = extra_dims; idim < ndim_component; ++idim)
        {
            dims_out[idim + 1] = dims_in[idim];
        }
        out_array = (PyArrayObject *)PyArray_SimpleNew(out_ndim, dims_out, NPY_DOUBLE);
        PyMem_Free(dims_out);
        if (!out_array)
        {
            Py_DECREF(component);
            return NULL;
        }
    }
    else
    {
        // We were given one
        if (!PyArray_Check(out))
        {
            PyErr_Format(PyExc_TypeError, "Expected out to be an array, but got %s.", Py_TYPE(out)->tp_name);
            Py_DECREF(component);
            return NULL;
        }
        out_array = (PyArrayObject *)out;
        // Check the shape is correct
        if ((unsigned)PyArray_NDIM(out_array) != ndim_component + 1)
        {
            PyErr_Format(PyExc_ValueError, "Expected output to have %u dimensions, but got %u.", ndim_component + 1,
                         (unsigned)PyArray_NDIM(out_array));
            Py_DECREF(component);
            return NULL;
        }
        const npy_intp *const dims_out = PyArray_DIMS(out_array);

        for (unsigned i = 0; i < extra_dims; ++i)
        {
            if (dims_out[i] != dims_in[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "Expected output to have the same shape as the input after the first dimension, but got "
                             "%zd for dimension %u.",
                             dims_out[i], i);
                Py_DECREF(component);
                return NULL;
            }
        }
        if (dims_out[extra_dims] != n_components_out)
        {
            PyErr_Format(PyExc_ValueError, "Expected output to have %u components, but got %zd.", n_components_out,
                         dims_out[extra_dims]);
            Py_DECREF(component);
            return NULL;
        }
        for (unsigned i = extra_dims; i < ndim_component; ++i)
        {
            if (dims_out[i + 1] != dims_in[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "Expected output to have the same shape as the input after the first two dimensions, but "
                             "got %zd for dimension %u.",
                             dims_out[i + 1], i);
                Py_DECREF(component);
                return NULL;
            }
        }

        Py_INCREF(out_array);
    }
    const double *restrict const ptr_component_arrays = PyArray_DATA(component);
    double *restrict const ptr_out = PyArray_DATA(out_array);

    const size_t int_pnt_cnt = integration_specs_total_points(ndim_in, map->int_specs);

    if (order == 0)
    {
        // Copy from input to output and we are done
        memcpy(ptr_out, ptr_component_arrays, sizeof(double) * PyArray_SIZE(out_array));
        Py_DECREF(component);
        return (PyObject *)out_array;
    }

    PyArrayObject *const transformation_array = compute_basis_transform_impl(map, order);
    if (!transformation_array)
    {
        Py_DECREF(component);
        Py_DECREF(out_array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    const double *restrict const ptr_transformation = PyArray_DATA(transformation_array);
#pragma omp simd
    for (size_t i_array = 0; i_array < input_arrays; ++i_array)
    {
        const double *restrict const ptr_component = ptr_component_arrays + i_array * int_pnt_cnt;
        double *restrict const ptr_out_array = ptr_out + i_array * int_pnt_cnt * n_components_out;
        for (unsigned i_out = 0; i_out < n_components_out; ++i_out)
        {
            for (size_t i_pt = 0; i_pt < int_pnt_cnt; ++i_pt)
            {
                const double val =
                    ptr_transformation[(size_t)index * n_components_out * int_pnt_cnt + i_out * int_pnt_cnt + i_pt] *
                    ptr_component[i_pt];
                ptr_out_array[i_out * int_pnt_cnt + i_pt] = val;
            }
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(transformation_array);
    Py_DECREF(component);
    return (PyObject *)out_array;
}

PyDoc_STRVAR(transform_covariant_to_target_docstring,
             "transform_covariant_to_target(smap: SpaceMap, components: numpy.typing.ArrayLike, *, out: "
             "numpy.typing.NDArray[numpy.double] "
             "| None = None) -> numpy.typing.NDArray[numpy.double]\n"
             "\n"
             "Transform covariant 1-form components from reference to target domain.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "smap : SpaceMap\n"
             "    Mapping from the reference space to the physical space to use to transform the\n"
             "    components.\n"
             "\n"
             "components : array_like\n"
             "    Array whose first dimension indexes the components in the reference space and\n"
             "    has length ``input_dimensions``. The remaining dimensions must match the\n"
             "    integration grid of the space map (``order + 1`` nodes per reference dimension).\n"
             "\n"
             "out : array, optional\n"
             "    Array to use to write the resulting transformed components to. If it is not\n"
             "    specified, a new array is created.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of transformed covariant components. If the ``out`` parameter was given,\n"
             "    a new reference to it is returned, otherwise a reference to the newly created\n"
             "    output array is returned.\n");

PyDoc_STRVAR(transform_kform_to_target_docstring,
             "transform_kform_to_target(order: int, smap: SpaceMap, components: numpy.typing.ArrayLike, *, out: "
             "numpy.typing.NDArray[numpy.double] | None = None) -> numpy.typing.NDArray[numpy.double]\n"
             "\n"
             "Transform k-form values based on a space mapping.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "    Order of the k-form being transformed.\n"
             "\n"
             "smap : SpaceMap\n"
             "    Mapping between the reference and target domain to use.\n"
             "\n"
             "components : array_like\n"
             "    Array with values of components of the k-form in the reference domain at\n"
             "    integration points associated with the space mapping.\n"
             "\n"
             "out : array, optional\n"
             "    Array to use to store the output in.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array with values of the components in the physical space.\n");

PyDoc_STRVAR(
    transform_kform_component_to_target_docstring,
    "transform_kform_component_to_target(order: int, smap: SpaceMap, component: numpy.typing.ArrayLike, index: int, *, "
    "out: numpy.typing.NDArray[numpy.double] | None = None) -> numpy.typing.NDArray[numpy.double]\n"
    "\n"
    "Transform k-form values based on a space mapping.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "order : int\n"
    "    Order of the k-form being transformed.\n"
    "\n"
    "smap : SpaceMap\n"
    "    Mapping between the reference and target domain to use.\n"
    "\n"
    "component : array_like\n"
    "    Values of the component in the reference domain at the integration points\n"
    "    of the space map. Leading dimensions are batch dimensions; the trailing\n"
    "    dimensions must match the integration grid and the batch dimensions are\n"
    "    preserved in the output.\n"
    "\n"
    "index : int\n"
    "    Index of the component that is to be computed.\n"
    "\n"
    "out : array, optional\n"
    "    Array to use to store the output in.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Array with values of the components in the physical space.\n");

PyMethodDef transformation_functions[] = {
    {
        .ml_name = "transform_contravariant_to_target",
        .ml_meth = (void *)transform_contravariant_to_target,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = transform_contravariant_to_target_docstring,
    },
    {
        .ml_name = "transform_covariant_to_target",
        .ml_meth = (void *)transform_covariant_to_target,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = transform_covariant_to_target_docstring,
    },
    {
        .ml_name = "transform_kform_to_target",
        .ml_meth = (void *)transform_kform_to_target,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = transform_kform_to_target_docstring,
    },
    {
        .ml_name = "transform_kform_component_to_target",
        .ml_meth = (void *)transform_kform_component_to_target,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = transform_kform_component_to_target_docstring,
    },
    {}, // sentinel
};
