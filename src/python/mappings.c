#include "mappings.h"

#include "../operations/matrices.h"
#include "basis_objects.h"
#include "cutl/iterators/combination_iterator.h"
#include "cutl/iterators/permutation_iterator.h"
#include "degrees_of_freedom.h"
#include "integration_objects.h"

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

    // Create the reconstruction state
    reconstruction_state_t recon_state;
    if (dof_reconstruction_state_init(dofs, integration_space, integration_registry, basis_registry, &recon_state) < 0)
        return NULL;
    const unsigned ndim = Py_SIZE(integration_space);
    const Py_ssize_t n_vals = (Py_ssize_t)multidim_iterator_total_size(recon_state.iter_int);
    coordinate_map_object *const self = (coordinate_map_object *)type->tp_alloc(type, n_vals * (ndim + 1));
    if (!self)
    {
        return NULL;
    }
    // Copy integration specs first
    self->ndim = ndim;
    self->int_specs = PyMem_Malloc(ndim * sizeof(*self->int_specs));
    if (!self->int_specs)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        self->int_specs[idim] = integration_space->specs[idim];
    }

    // Call the reconstruct function on the DoFs
    const unsigned ndofs = Py_SIZE(dofs);
    const double *const restrict pdofs = dofs->values;
    compute_integration_point_values(ndim, recon_state.iter_int, recon_state.iter_basis, recon_state.basis_sets, n_vals,
                                     self->values + 0 * n_vals, ndofs, pdofs);

    int *const derivative_array = PyMem_Malloc(sizeof(int) * ndim);
    if (!derivative_array)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (unsigned i = 0; i < ndim; ++i)
    {
        derivative_array[i] = 0;
    }
    for (unsigned i = 0; i < ndim; ++i)
    {
        derivative_array[i] = 1; // Set the current dimension to use derivative
        // Compute with the specified derivatives
        compute_integration_point_values_derivatives(ndim, recon_state.iter_int, recon_state.iter_basis,
                                                     recon_state.basis_sets, derivative_array, n_vals,
                                                     self->values + (i + 1) * n_vals, ndofs, pdofs);
        derivative_array[i] = 0; // Reset the current dimension
    }
    PyMem_Free(derivative_array);

    return (PyObject *)self;
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
    size_t total_points = 1;
    for (unsigned i = 0; i < map->ndim; ++i)
        total_points *= map->int_specs[i].order + 1;
    return map->values + (dim + 1) * total_points;
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

PyType_Spec coordinate_map_type_spec = {
    .name = FDG_TYPE_NAME("CoordinateMap"),
    .basicsize = sizeof(coordinate_map_object),
    .itemsize = sizeof(*((coordinate_map_object *)0xB00B1E5)->values),
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_dealloc, coordinate_map_dealloc},
        {Py_tp_new, coordinate_map_new},
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
                 .doc = "numpy.typing.NDArray[numpy.double] : Values of the coordinate map at the integration points.",
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
                    .ml_doc =
                        "gradient(idim: int, /) -> numpy.typing.NDArray[numpy.double]\nRetrieve the gradient of the "
                        "coordinate map in given dimension.",
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

static void calculate_determinants_and_inverse_maps(
    const unsigned n_dim, const unsigned n_maps, const coordinate_map_object *maps[static n_maps],
    const size_t total_points, double determinant[restrict const total_points], const size_t jacobian_size,
    double inverse_maps[restrict const total_points * jacobian_size], double jacobian[restrict const jacobian_size],
    double q_mat[restrict const n_maps * n_maps])
{
    // Now we iterate over all the points
    for (size_t i_pt = 0; i_pt < total_points; ++i_pt)
    {
        // Fill in the Jacobian
        const unsigned rows = n_maps;
        const unsigned cols = n_dim;
        for (unsigned idim = 0; idim < rows; ++idim)
        {
            for (unsigned jdim = 0; jdim < cols; ++jdim)
            {
                jacobian[idim * cols + jdim] = coordinate_map_gradient(maps[idim], jdim)[i_pt];
            }
        }

        const matrix_t jacobian_mat = (matrix_t){.rows = rows, .cols = cols, .values = jacobian};
        const matrix_t q_matrix = (matrix_t){.rows = rows, .cols = rows, .values = q_mat};

        // Decompose Jacobian into QR decomposition
        interp_result_t res = matrix_qr_decompose(&jacobian_mat, &q_matrix);
        (void)res;
        CPYUTL_ASSERT(res == FDG_SUCCESS, "QR decomposition failed.");
        // Compute the determinant from the diagonal of the matrix
        double det = 1;
        for (unsigned i = 0; i < cols; ++i)
        {
            det *= jacobian[i * cols + i];
        }
        determinant[i_pt] = det;

        double *const p_inv_map = inverse_maps + i_pt * jacobian_size;
        const matrix_t out_mat = (matrix_t){.rows = cols, .cols = rows, .values = p_inv_map};
        // Copy the top part of q into out
        for (unsigned irow = 0; irow < cols; ++irow)
        {
            for (unsigned icol = 0; icol < rows; ++icol)
            {
                p_inv_map[irow * rows + icol] = q_mat[irow * rows + icol];
            }
        }

        // Use decomposition to compute "inverse". This is done simply by applying inverse of the
        // upper triangular (rows x rows) part of the jacobian to the matrix q_mat.
        res = matrix_back_substitute(&jacobian_mat, &out_mat);
        CPYUTL_ASSERT(res == FDG_SUCCESS, "Back substitution failed.");
        (void)res;
    }
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
    coordinate_map_object *const first_map = (coordinate_map_object *)PyTuple_GET_ITEM(args, 0);
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
        coordinate_map_object *const map = (coordinate_map_object *)PyTuple_GET_ITEM(args, i);
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

    calculate_determinants_and_inverse_maps(this->ndim, n_maps, (const coordinate_map_object **)this->maps,
                                            total_points, determinant, jacobian_size, inverse_maps, jacobian, q_mat);

    // Free work arrays
    PyMem_RawFree(q_mat);
    PyMem_RawFree(jacobian);

    // Store the output
    this->determinant = determinant;
    this->inverse_maps = inverse_maps;
    // Return
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
             "Return the coordinate map for the specified dimension.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the dimension for which the map shoudl be returned.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "CoordinateMap\n"
             "    Map used for the specified coordinate.\n");

PyDoc_STRVAR(space_map_docstring, "SpaceMap(*coordinates: CoordinateMap)\n"
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
             "mapping space (as counted by :meth:`SpaceMap.output_dimensions`)\n"
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

    if (order == 1)
    {
        Py_BEGIN_ALLOW_THREADS;
        // Special case: transformation is just the space map
        for (size_t i_in = 0; i_in < n_dims; ++i_in)
            for (size_t i_out = 0; i_out < n_maps; ++i_out)
                for (size_t i_pt = 0; i_pt < total_points; ++i_pt)
                    *(double *)PyArray_GETPTR3(res, i_in, i_out, i_pt) =
                        space_map_backward_derivative(map, i_pt, i_in, i_out);
        Py_END_ALLOW_THREADS;
    }
    else if (order == n_maps)
    {
        npy_double *const ptr = PyArray_DATA(res);
        Py_BEGIN_ALLOW_THREADS;
        // Special case: transformation is just the space map
        for (size_t i_pt = 0; i_pt < total_points; ++i_pt)
            ptr[i_pt] = (1 / map->determinant[i_pt]);
        Py_END_ALLOW_THREADS;
    }
    else // (order != 1 && order != n_maps)
    {
        permutation_iterator_t *iter_out_perm;
        combination_iterator_t *iter_out_comb;
        combination_iterator_t *iter_in_comb;
        void *const mem = cutl_alloc_group(
            &PYTHON_ALLOCATOR,
            (const cutl_alloc_info_t[]){
                {.size = permutation_iterator_required_memory(order, order), .p_ptr = (void **)&iter_out_perm},
                {.size = combination_iterator_required_memory(order), .p_ptr = (void **)&iter_out_comb},
                {.size = combination_iterator_required_memory(order), .p_ptr = (void **)&iter_in_comb},
                {},
            });
        if (!mem)
        {
            Py_DECREF(res);
            return NULL;
        }

        Py_BEGIN_ALLOW_THREADS;

        size_t idx_in = 0;
        // Iterate over bases in the inputs space
        combination_iterator_init(iter_in_comb, n_dims, order);
        while (!combination_iterator_is_done(iter_in_comb))
        {
            // Indices of current input dimensions
            const uint8_t *const current_in = combination_iterator_current(iter_in_comb);
            // Iterate over bases in the output space.
            size_t idx_out = 0;
            combination_iterator_init(iter_out_comb, n_maps, order);
            while (!combination_iterator_is_done(iter_out_comb))
            {
                // Indices of current output dimensions.
                const uint8_t *const current_out = combination_iterator_current(iter_out_comb);

                // Loop over integration points
                for (size_t idx_pt = 0; idx_pt < total_points; ++idx_pt)
                {
                    // Total transformation coefficient, which we will accumulate for each possible basis.
                    double val = 0.0;

                    // Loop over all permutations of the current basis indices.
                    permutation_iterator_init(iter_out_perm, order, order);
                    while (!permutation_iterator_is_done(iter_out_perm))
                    {
                        // Indices for the current permutation
                        const uint8_t *const current_perm = permutation_iterator_current(iter_out_perm);
                        double basis_contribution = 1.0;

                        // Loop over the derivative terms and compute their product
                        for (unsigned idim = 0; idim < order; ++idim)
                        {
                            const unsigned idx_coord = current_out[current_perm[idim]];
                            const unsigned idx_dim = current_in[idim];
                            // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                            const double contribution = space_map_backward_derivative(map, idx_pt, idx_dim, idx_coord);
                            basis_contribution *= contribution;
                            // printf("Adding contribution of dxi_%u/dx_%u=%g to element(%zu, %zu, %zu)\n", idx_dim,
                            // idx_coord,
                            //        contribution, idx_in, idx_out, idx_pt);
                        }

                        // Check if we flip the sign (meaning subtract) for this contribution.
                        if (permutation_iterator_current_sign(iter_out_perm))
                        {
                            val -= basis_contribution;
                        }
                        else
                        {
                            val += basis_contribution;
                        }

                        permutation_iterator_next(iter_out_perm);
                    }
                    *(double *)PyArray_GETPTR3(res, idx_in, idx_out, idx_pt) = val;
                    // ptr_out[idx_in * out_dims[1] * out_dims[2] + idx_out * out_dims[2] + idx_pt] = val;
                }

                idx_out += 1;
                combination_iterator_next(iter_out_comb);
            }

            idx_in += 1;
            combination_iterator_next(iter_in_comb);
        }
        Py_END_ALLOW_THREADS;

        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
    }

    Py_INCREF(res);
    map->transformations[order - 1] = res;

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
             "Compute the matrix with transformation factors for k-form basis.\n"
             "\n"
             "Basis transform matrix returned by this function specifies how at integration point a\n"
             "basis from the reference domain contributes to the basis in the target domain.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "    Order of the k-form for which this is to be done.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array with three axis. The first indexes over the input basis, the second\n"
             "    over output basis, and the last one over integration points.\n");

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
                     .doc = "numpy.typing.NDArray[numpy.double] : Array with the values of determinant at integration."
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
                     .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
                     .ml_doc = space_map_get_coordinate_map_docstring,
                 },
                 {
                     .ml_name = "basis_transform",
                     .ml_meth = (void *)space_map_basis_transform,
                     .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
                     .ml_doc = space_map_basis_transform_docstring,
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
        PyErr_Format(PyExc_ValueError, "Expected components to have shape %u dimensions, but got %u.", ndim_in,
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
             "    Array where the first dimension indexes the components in the reference space. All\n"
             "    other dimensions will be treated as if flattened.\n"
             "\n"
             "out : array, optional\n"
             "    Array to used to write the resulting transformed components to. If it is not\n"
             "    specified, a new array is created.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of transformed covariant components. If the ``out`` parameter was given,\n"
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
                {.type = CPYARG_TYPE_PYTHON, .p_val = &out, .kwname = "out", .optional = 1},
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

    // Convert components to be an array
    PyArrayObject *const components =
        (PyArrayObject *)PyArray_FROMANY(py_components, NPY_DOUBLE, 1 + ndim_in, 1 + ndim_in, NPY_ARRAY_IN_ARRAY);
    if (!components)
        return NULL;

    // Check the shape is correct
    const npy_intp *const dims_in = PyArray_DIMS(components);
    const unsigned ndim_components = PyArray_NDIM(components);
    if (dims_in[0] != n_components_in)
    {
        PyErr_Format(PyExc_ValueError, "Expected components to have shape (%u, ...), but got (%zd, ...).",
                     n_components_in, dims_in[0]);
        Py_DECREF(components);
        return NULL;
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
            return NULL;
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
            return NULL;
        }
        dims_out[0] = n_components_out;
        for (unsigned idim = 1; idim < ndim_components; ++idim)
        {
            dims_out[idim] = dims_in[idim];
        }
        out_array = (PyArrayObject *)PyArray_SimpleNew(ndim_components, dims_out, NPY_DOUBLE);
        PyMem_Free(dims_out);
        if (!out_array)
        {
            Py_DECREF(components);
            return NULL;
        }
    }
    else
    {
        // We were given one
        if (!PyArray_Check(out))
        {
            PyErr_Format(PyExc_TypeError, "Expected out to be an array, but got %s.", Py_TYPE(out)->tp_name);
            Py_DECREF(components);
            return NULL;
        }
        out_array = (PyArrayObject *)out;
        // Check the shape is correct
        const npy_intp *const dims_out = PyArray_DIMS(out_array);
        if (dims_out[0] != n_components_out)
        {
            PyErr_Format(PyExc_ValueError, "Expected output to have shape (%u, ...), but got (%zd, ...).",
                         n_components_out, dims_out[0]);
            Py_DECREF(components);
            return NULL;
        }
        for (unsigned i = 0; i < ndim_components; ++i)
        {
            if (dims_out[i] != dims_in[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "Expected output to have the same shape as the input after the first dimension, but got "
                             "%zd for dimension %u.",
                             dims_out[i], i);
                Py_DECREF(components);
                return NULL;
            }
        }
        Py_INCREF(out_array);
    }

    const size_t total_points = integration_specs_total_points(ndim_in, map->int_specs);

    if (order == 0)
    {
        // Copy from input to output and we are done
        memcpy(PyArray_DATA(out_array), PyArray_DATA(components), sizeof(double) * total_points);
        Py_DECREF(components);
        return (PyObject *)out_array;
    }

    if (order == 1)
    {
        transform_covariant_to_target_impl(total_points, PyArray_DATA(components), PyArray_DATA(out_array), ndim_out,
                                           ndim_in, map);
        return (PyObject *)out_array;
    }

    PyArrayObject *const transformation_array = compute_basis_transform_impl(map, order);
    if (!transformation_array)
    {
        Py_DECREF(components);
        Py_DECREF(out_array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    const double *restrict const ptr_components = PyArray_DATA(components);
    const double *restrict const ptr_transformation = PyArray_DATA(transformation_array);
    double *restrict const ptr_out = PyArray_DATA(out_array);
#pragma omp simd
    for (size_t i_pt = 0; i_pt < total_points; ++i_pt)
    {
        for (unsigned i_out = 0; i_out < n_components_out; ++i_out)
        {
            double val = 0.0;
            for (unsigned i_in = 0; i_in < n_components_in; ++i_in)
            {
                val +=
                    ptr_transformation[(size_t)i_in * n_components_out * total_points + i_out * total_points + i_pt] *
                    ptr_components[i_in * total_points + i_pt];
            }
            ptr_out[i_out * total_points + i_pt] = val;
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(transformation_array);
    Py_DECREF(components);
    return (PyObject *)out_array;
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
                {.type = CPYARG_TYPE_PYTHON, .p_val = &out, .kwname = "out", .optional = 1},
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
             "Transform covariant 1-form components from reference to target domain.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "smap : SpaceMap\n"
             "    Mapping from the reference space to the physical space to use to transform the\n"
             "    components.\n"
             "\n"
             "components : array_like\n"
             "    Array where the first dimension indexes the components in the reference space. All\n"
             "    other dimensions will be treated as if flattened.\n"
             "\n"
             "out : array, optional\n"
             "    Array to used to write the resulting transformed components to. If it is not\n"
             "    specified, a new array is created.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of transformed covariant components. If the ``out`` parameter was given,\n"
             "    a new reference to it is returned, otherwise a reference to the newly created\n"
             "    output array is returned.\n"

);

PyDoc_STRVAR(transform_kform_to_target_docstring,
             "transform_kform_to_target(order: int,smap: SpaceMap, components: numpy.typing.ArrayLike, *, out: "
             "numpy.typing.NDArray[numpy.double] | None = None) -> numpy.typing.NDArray[numpy.double]\n"
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
    "    Values of component in the reference domain at integration points associated\n"
    "    with the space mapping.\n"
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
