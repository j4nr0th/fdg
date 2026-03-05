#include "degrees_of_freedom.h"

#include "../basis/basis_lagrange.h"
#include "../polynomials/bernstein.h"
#include "../polynomials/lagrange.h"
#include "../polynomials/legendre.h"
#include "basis_objects.h"
#include "function_space_objects.h"
#include "incidence.h"
#include "integration_objects.h"

/**
 * Checks if a given Numpy array has a specific shape determined by the number
 * of dimensions and a basis specification for each dimension.
 *
 * @param arr A pointer to the Numpy array object to be checked.
 * @param ndim The number of dimensions against which the array is to be verified.
 * @param specs An array of basis specifications representing the expected shape
 *              for each dimension, where each dimension must match `order + 1`.
 * @return 1 if the array matches the specified shape, 0 otherwise.
 */
static int array_has_shape(const PyArrayObject *const arr, const unsigned ndim,
                           const basis_spec_t specs[const static ndim])
{
    if (PyArray_NDIM(arr) != (int)ndim)
        return 0;

    for (unsigned i = 0; i < ndim; ++i)
    {
        if (PyArray_DIM(arr, (int)i) != specs[i].order + 1)
            return 0;
    }
    return 1;
}

static dof_object *dof_object_create(PyTypeObject *subtype, const unsigned ndim,
                                     const basis_spec_t basis_specs_in[static ndim])
{
    Py_ssize_t total_dofs = 1;
    for (unsigned i = 0; i < ndim; ++i)
        total_dofs *= basis_specs_in[i].order + 1;

    dof_object *const self = (dof_object *)subtype->tp_alloc(subtype, total_dofs);
    if (!self)
        return NULL;
    self->n_dims = ndim;
    self->basis_specs = NULL;

    basis_spec_t *const basis_specs = PyMem_Malloc(ndim * sizeof(*basis_specs));
    if (!basis_specs)
    {
        Py_DECREF(self);
        return NULL;
    }
    self->basis_specs = basis_specs;
    for (unsigned i = 0; i < ndim; ++i)
    {
        basis_specs[i] = basis_specs_in[i];
    }

    return self;
}

static PyObject *dof_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *state = interplib_get_module_state(subtype);
    if (!state)
        return NULL;

    const function_space_object *space;
    PyObject *py_vals = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O", (char *[]){"", "", NULL}, state->function_space_type, &space,
                                     &py_vals))
        return NULL;

    const Py_ssize_t ndim = Py_SIZE(space);
    const basis_spec_t *basis_specs_in = space->specs;
    Py_ssize_t total_dofs = 1;
    for (unsigned i = 0; i < ndim; ++i)
        total_dofs *= basis_specs_in[i].order + 1;

    dof_object *const self = dof_object_create(subtype, ndim, basis_specs_in);

    if (py_vals)
    {
        PyArrayObject *const vals =
            (PyArrayObject *)PyArray_FROMANY(py_vals, NPY_DOUBLE, 0, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (!vals)
        {
            Py_DECREF(self);
            return NULL;
        }
        if (array_has_shape(vals, ndim, basis_specs_in) == 0 && PyArray_SIZE(vals) != total_dofs)
        {
            PyErr_Format(PyExc_ValueError,
                         "Values must be given either as a flat array with the correct number of elements (%u) or with "
                         "exact matching %u-dimensional shape.",
                         total_dofs, ndim);
            Py_DECREF(vals);
            Py_DECREF(self);
            return NULL;
        }
        const npy_double *const vals_data = PyArray_DATA(vals);
        for (unsigned i = 0; i < total_dofs; ++i)
            self->values[i] = vals_data[i];
        Py_DECREF(vals);
    }
    else
    {
        memset(self->values, 0, total_dofs * sizeof(*self->values));
    }

    return (PyObject *)self;
}

PyDoc_STRVAR(dof_docstring,
             "DegreesOfFreedom(function_space : FunctionSpace, values : numpy.typing.ArrayLike | None = None)\n"
             "Degrees of freedom associated with a function space.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "function_space : FunctionSpace\n"
             "    Function space the degrees of freedom belong to.\n"
             "values : array_like, optional\n"
             "    Values of the degrees of freedom. When not specified, they are zero initialized.\n");

static int ensure_dof_and_state(PyObject *self, PyTypeObject *defining_class, const interplib_module_state_t **p_state,
                                dof_object **p_object)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->degrees_of_freedom_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->degrees_of_freedom_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_state = state;
    *p_object = (dof_object *)self;
    return 0;
}

static PyObject *dof_get_function_space(PyObject *self, void *Py_UNUSED(closure))
{
    const interplib_module_state_t *state;
    dof_object *this = (dof_object *)self;
    if (ensure_dof_and_state(self, NULL, &state, &this) < 0)
        return NULL;
    return (PyObject *)function_space_object_create(state->function_space_type, this->n_dims, this->basis_specs);
}

static PyObject *dof_get_total_number(PyObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(Py_SIZE(self));
}

static PyObject *dof_get_shape(PyObject *self, void *Py_UNUSED(closure))
{
    const dof_object *this = (dof_object *)self;
    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(this->n_dims);
    if (!out)
        return NULL;
    for (unsigned i = 0; i < this->n_dims; ++i)
    {
        PyObject *const dim = PyLong_FromLong(this->basis_specs[i].order + 1);
        if (!dim)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, dim);
    }
    return (PyObject *)out;
}

static PyObject *dof_get_values(PyObject *self, void *Py_UNUSED(closure))
{
    const dof_object *this = (dof_object *)self;
    npy_intp *const dims = PyMem_Malloc(this->n_dims * sizeof(*dims));
    if (!dims)
        return NULL;
    for (unsigned i = 0; i < this->n_dims; ++i)
        dims[i] = this->basis_specs[i].order + 1;
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->n_dims, dims, NPY_DOUBLE, (void *)this->values);
    PyMem_Free(dims);
    if (!arr)
        return NULL;
    if (PyArray_SetBaseObject(arr, self) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject *)arr;
}

static int dof_set_values(PyObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    dof_object *this = (dof_object *)self;
    PyArrayObject *const arr = (PyArrayObject *)PyArray_FROMANY(value, NPY_DOUBLE, 0, 0, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr)
        return -1;
    if (array_has_shape(arr, this->n_dims, this->basis_specs) == 0 && PyArray_SIZE(arr) != Py_SIZE(self))
    {
        PyErr_Format(PyExc_ValueError,
                     "Values must either be flat with %u elements or have exact correct %u-dimensional shape.",
                     (unsigned)Py_SIZE(arr), this->n_dims);
        Py_DECREF(arr);
        return -1;
    }
    const npy_double *const vals_data = PyArray_DATA(arr);
    for (unsigned i = 0; i < Py_SIZE(self); ++i)
        this->values[i] = vals_data[i];

    Py_DECREF(arr);
    return 0;
}

PyDoc_STRVAR(dof_reconstruct_at_integration_points_docstring,
             "reconstruct_at_integration_points(integration_space: IntegrationSpace, integration_registry: "
             "IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, basis_registry: BasisRegistry = "
             "DEFAULT_BASIS_REGISTRY, *, out: numpy.typing.NDArray[numpy.double] | None = None, ) -> "
             "numpy.typing.NDArray[numpy.double]\n"
             "Reconstruct the function at the integration points of the given space.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "integration_space : IntegrationSpace\n"
             "    Integration space where the function should be reconstructed.\n"
             "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rules.\n"
             "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
             "    Registry used to retrieve the basis specifications.\n"
             "out : array, optional\n"
             "    Array where the results should be written to. If not given, a new one\n"
             "    will be created and returned. It should have the same shape as the\n"
             "    integration points.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of reconstructed function values at the integration points.\n");

int dof_reconstruction_state_init(const dof_object *this, const unsigned ndim,
                                  const integration_spec_t integration_specs[const static ndim],
                                  const integration_registry_object *python_integration_registry,
                                  const basis_registry_object *python_basis_registry,
                                  reconstruction_state_t *recon_state)
{
    multidim_iterator_t *const iter_int = integration_specs_iterator(ndim, integration_specs);
    if (!iter_int)
    {
        return -1;
    }
    ASSERT(ndim == this->n_dims, "Input ndim does not match the DoF ndim");
    multidim_iterator_t *const iter_basis = python_basis_iterator(ndim, this->basis_specs);
    if (!iter_basis)
    {
        PyMem_Free(iter_int);
        return -1;
    }

    basis_set_registry_t *const basis_registry = (basis_set_registry_t *)python_basis_registry->registry;
    const basis_set_t **basis_sets;
    // Get basis (and first the integration rules)
    {
        integration_rule_registry_t *const integration_registry =
            (integration_rule_registry_t *)python_integration_registry->registry;
        // Get integration rules
        const integration_rule_t **const integration_rules =
            python_integration_rules_get(ndim, integration_specs, integration_registry);
        if (!integration_rules)
        {
            PyMem_Free(iter_basis);
            PyMem_Free(iter_int);
            return -1;
        }
        basis_sets = python_basis_sets_get(ndim, this->basis_specs, integration_rules, basis_registry);
        python_integration_rules_release(ndim, integration_rules, integration_registry);
    }
    if (!basis_sets)
    {
        PyMem_Free(iter_basis);
        PyMem_Free(iter_int);
        return -1;
    }
    *recon_state = (reconstruction_state_t){.iter_int = iter_int, .iter_basis = iter_basis, .basis_sets = basis_sets};

    return 0;
}

void dof_reconstruction_state_release(reconstruction_state_t *const recon_state, basis_set_registry_t *basis_registry,
                                      const unsigned ndim, const basis_set_t *basis_sets[static ndim])
{
    python_basis_sets_release(ndim, basis_sets, basis_registry);
    PyMem_Free(recon_state->iter_basis);
    PyMem_Free(recon_state->iter_int);
    *recon_state = (reconstruction_state_t){};
}

static PyArrayObject *ensure_reconstruction_output(const dof_object *this, const unsigned ndim,
                                                   const integration_spec_t integration_specs[const static ndim],
                                                   PyArrayObject *out_array)
{
    ASSERT(ndim == this->n_dims, "Input ndim does not match the DoF ndim");
    // Check or create output
    if (out_array)
    {
        if (check_input_array(out_array, 0, (const npy_intp[]){}, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "out") < 0)
        {
            raise_exception_from_current(PyExc_ValueError,
                                         "Output array must be a contiguous, aligned array of doubles.");
            return NULL;
        }
        if ((unsigned)PyArray_NDIM(out_array) != ndim)
        {
            PyErr_Format(PyExc_ValueError, "Output array must have %u dimensions, but it had %u.", ndim,
                         (unsigned)PyArray_NDIM(out_array));
            return NULL;
        }
        for (unsigned idim = 0; idim < ndim; ++idim)
        {
            if (PyArray_DIM(out_array, (int)idim) != integration_specs[idim].order + 1)
            {
                PyErr_Format(PyExc_ValueError,
                             "Output array must have the exact same shape as the integration space, but dimension %u "
                             "did not match (integration space: %u, array: %u).",
                             idim, integration_specs[idim].order + 1, (unsigned)PyArray_DIM(out_array, (int)idim));
                return NULL;
            }
        }
        // Good, now incref it
        Py_INCREF(out_array);
    }
    else
    {
        // Create it
        npy_intp *const p_dim_out = PyMem_Malloc(sizeof(*p_dim_out) * ndim);
        if (!p_dim_out)
        {
            return NULL;
        }
        for (unsigned idim = 0; idim < ndim; ++idim)
            p_dim_out[idim] = integration_specs[idim].order + 1;
        out_array = (PyArrayObject *)PyArray_SimpleNew(this->n_dims, p_dim_out, NPY_DOUBLE);
        PyMem_Free(p_dim_out);
        if (!out_array)
        {
            return NULL;
        }
    }

    return out_array;
}

static PyArrayObject *reconstruct_at_integration_points_impl(
    const dof_object *this, const unsigned ndim, const integration_spec_t integration_specs[const static ndim],
    const integration_registry_object *python_integration_registry, const basis_registry_object *python_basis_registry,
    PyArrayObject *out_array, const unsigned n_dof)
{
    // Check or create output
    if ((out_array = ensure_reconstruction_output(this, ndim, integration_specs, out_array)) == NULL)
        return NULL;

    reconstruction_state_t recon_state;
    if (dof_reconstruction_state_init(this, ndim, integration_specs, python_integration_registry, python_basis_registry,
                                      &recon_state) < 0)
    {
        Py_DECREF(out_array);
        return NULL;
    }

    npy_double *const ptr = PyArray_DATA(out_array);
    // Compute the values
    CPYUTL_ASSERT(multidim_iterator_total_size(recon_state.iter_basis) == n_dof,
                  "Basis iterator should have the same number of elements as there are DoFs (%zu vs %u)",
                  multidim_iterator_total_size(recon_state.iter_basis), n_dof);

    compute_integration_point_values(ndim, recon_state.iter_int, recon_state.iter_basis, recon_state.basis_sets,
                                     PyArray_SIZE(out_array), ptr, n_dof, this->values);

    // Free the iterator memory and release the basis sets
    dof_reconstruction_state_release(&recon_state, python_basis_registry->registry, ndim, recon_state.basis_sets);

    return out_array;
}

PyObject *dof_reconstruct_at_integration_points(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    dof_object *this;
    if (ensure_dof_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    // Parse the arguments
    const integration_space_object *integration_space;
    const integration_registry_object *python_integration_registry =
        (integration_registry_object *)state->registry_integration;
    const basis_registry_object *python_basis_registry = (basis_registry_object *)state->registry_basis;
    PyArrayObject *out_array = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_space,
                    .type_check = state->integration_space_type,
                    .kwname = "integration_space",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &python_integration_registry,
                    .type_check = state->integration_registry_type,
                    .kwname = "integration_registry",
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &python_basis_registry,
                    .type_check = state->basis_registry_type,
                    .kwname = "basis_registry",
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &out_array,
                    .type_check = &PyArray_Type,
                    .kwname = "out",
                    .optional = 1,
                    .kw_only = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Do the dimensions match?
    const unsigned n_dof = Py_SIZE(self);
    (void)n_dof;
    const unsigned ndim = this->n_dims;
    if (Py_SIZE(integration_space) != ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected integration space with %u dimensions, but it had only %u.", ndim,
                     Py_SIZE(integration_space));
        return NULL;
    }

    return (PyObject *)reconstruct_at_integration_points_impl(
        this, ndim, integration_space->specs, python_integration_registry, python_basis_registry, out_array, n_dof);
}

int *reconstruction_derivative_indices(const unsigned ndim, PyObject *py_indices)
{
    int *const indices = PyMem_Malloc(sizeof(*indices) * ndim);
    if (!indices)
    {
        return NULL;
    }
    // Zero initialize it all
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        indices[idim] = 0;
    }
    if (PyNumber_Check(py_indices))
    {
        // It's a number, so just one index
        const Py_ssize_t idx = PyNumber_AsSsize_t(py_indices, PyExc_OverflowError);
        if (PyErr_Occurred())
        {
            raise_exception_from_current(PyExc_TypeError,
                                         "Expected an integer index, but it could not be converted from %s object.",
                                         Py_TYPE(py_indices)->tp_name);
            PyMem_Free(indices);
            return NULL;
        }
        if (idx < 0 || idx >= ndim)
        {
            PyErr_Format(PyExc_ValueError, "Expected an index between 0 and %u, but got %zd.", ndim - 1, idx);
            PyMem_Free(indices);
            return NULL;
        }
        indices[idx] = 1;
        return indices;
    }

    PyObject *const seq = PySequence_Fast(py_indices, "Expected a sequence of indices.");
    if (!seq)
    {
        PyMem_Free(indices);
        return NULL;
    }

    for (unsigned i = 0; i < PySequence_Fast_GET_SIZE(seq); ++i)
    {
        PyObject *const item = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyNumber_Check(item))
        {
            PyErr_Format(PyExc_TypeError, "Expected a sequence of integers, but got a non-integer at index %zd.", i);
            PyMem_Free(indices);
            Py_DECREF(seq);
            return NULL;
        }
        const Py_ssize_t idx = PyNumber_AsSsize_t(item, PyExc_OverflowError);
        if (PyErr_Occurred())
        {
            raise_exception_from_current(PyExc_TypeError, "Could not convert an index from %s object.",
                                         Py_TYPE(item)->tp_name);
            PyMem_Free(indices);
            Py_DECREF(seq);
            return NULL;
        }
        if (idx < 0 || idx >= ndim)
        {
            PyErr_Format(PyExc_ValueError, "Expected an index between 0 and %u, but got %zd.", ndim - 1, idx);
            PyMem_Free(indices);
            Py_DECREF(seq);
            return NULL;
        }
        if (indices[idx] != 0)
        {
            PyErr_Format(PyExc_ValueError, "Expected each index to appear only once, but got it twice at index %zd.",
                         idx);
            PyMem_Free(indices);
            Py_DECREF(seq);
            return NULL;
        }
        indices[idx] = 1;
    }
    Py_DECREF(seq);

    return indices;
}

PyDoc_STRVAR(dof_reconstruct_derivative_at_integration_points_docstring,
             "reconstruct_derivative_at_integration_points(integration_space: IntegrationSpace, idim: Sequence[int], "
             "integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, basis_registry: BasisRegistry "
             "= DEFAULT_BASIS_REGISTRY, *, out: numpy.typing.NDArray[numpy.double] | None = None) -> "
             "numpy.typing.NDArray[numpy.double]\n"
             "Reconstruct the derivative of the function in given dimension.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "integration_space : IntegrationSpace\n"
             "    Integration space where the function derivative should be reconstructed.\n"
             "idim : Sequence[int]\n"
             "    Dimensions in which the derivative should be computed. All values\n"
             "    should appear at most once.\n"
             "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rules.\n"
             "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
             "    Registry used to retrieve the basis specifications.\n"
             "out : array, optional\n"
             "    Array where the results should be written to. If not given, a new one\n"
             "    will be created and returned. It should have the same shape as the\n"
             "    integration points.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of reconstructed function derivative values at the integration points.\n");

PyObject *dof_reconstruct_derivative_at_integration_points(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    dof_object *this;
    if (ensure_dof_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    // Parse the arguments
    const integration_space_object *integration_space;
    PyObject *derivative_dimensions;
    const integration_registry_object *python_integration_registry =
        (integration_registry_object *)state->registry_integration;
    const basis_registry_object *python_basis_registry = (basis_registry_object *)state->registry_basis;
    PyArrayObject *out_array = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_space,
                    .type_check = state->integration_space_type,
                    .kwname = "integration_space",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &derivative_dimensions,
                    .kwname = "idim",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &python_integration_registry,
                    .type_check = state->integration_registry_type,
                    .kwname = "integration_registry",
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &python_basis_registry,
                    .type_check = state->basis_registry_type,
                    .kwname = "basis_registry",
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &out_array,
                    .type_check = &PyArray_Type,
                    .kwname = "out",
                    .optional = 1,
                    .kw_only = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Do the dimensions match?
    const unsigned n_dof = Py_SIZE(self);
    (void)n_dof;
    const unsigned ndim = this->n_dims;
    if (Py_SIZE(integration_space) != ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected integration space with %u dimensions, but it had only %u.", ndim,
                     Py_SIZE(integration_space));
        return NULL;
    }

    // Check or create output
    if ((out_array = ensure_reconstruction_output(this, ndim, integration_space->specs, out_array)) == NULL)
        return NULL;

    int *const derivative_indices = reconstruction_derivative_indices(ndim, derivative_dimensions);
    if (!derivative_indices)
    {
        Py_DECREF(out_array);
        return NULL;
    }

    reconstruction_state_t recon_state;
    if (dof_reconstruction_state_init(this, ndim, integration_space->specs, python_integration_registry,
                                      python_basis_registry, &recon_state) < 0)
    {
        PyMem_Free(derivative_indices);
        Py_DECREF(out_array);
        return NULL;
    }

    npy_double *const ptr = PyArray_DATA(out_array);
    // Compute the values
    CPYUTL_ASSERT(multidim_iterator_total_size(recon_state.iter_basis) == n_dof,
                  "Basis iterator should have the same number of elements as there are DoFs (%zu vs %u)",
                  multidim_iterator_total_size(recon_state.iter_basis), n_dof);

    compute_integration_point_values_derivatives(ndim, recon_state.iter_int, recon_state.iter_basis,
                                                 recon_state.basis_sets, derivative_indices, PyArray_SIZE(out_array),
                                                 ptr, n_dof, this->values);

    // Free the iterator memory and release the basis sets
    PyMem_Free(derivative_indices);
    dof_reconstruction_state_release(&recon_state, python_basis_registry->registry, ndim, recon_state.basis_sets);
    return (PyObject *)out_array;
}

PyDoc_STRVAR(dof_derivative_docstring,
             "derivative(idim: int) -> DegreesOfFreedom\n"
             "Return degrees of freedom of the derivative along the reference dimension.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idim : int\n"
             "    Index of the reference dimension along which the derivative should be taken.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "DegreesOfFreedom\n"
             "    Degrees of freedom of the computed derivative.\n");

PyObject *dof_derivative(PyObject *self, PyTypeObject *defining_class, PyObject *const *args, const Py_ssize_t nargs,
                         const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    const dof_object *this;
    if (ensure_dof_and_state(self, defining_class, &state, (dof_object **)&this) < 0)
        return NULL;

    Py_ssize_t index;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &index, .kwname = "idim"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Check input parameters
    if (index < 0 || index >= this->n_dims)
    {
        PyErr_Format(PyExc_ValueError, "Expected an index between 0 and %u, but got %zd.", this->n_dims - 1, index);
        return NULL;
    }

    if (this->basis_specs[index].order == 0)
    {
        PyErr_Format(PyExc_ValueError, "Cannot compute the derivative of a function with order 0 in dimension %zd.",
                     index);
        return NULL;
    }

    // Create a new DoF object
    function_space_object *const function_space_object =
        function_space_object_create(state->function_space_type, this->n_dims, this->basis_specs);
    if (!function_space_object)
        return NULL;
    function_space_object->specs[index].order -= 1;

    dof_object *const new_dofs = (dof_object *)PyObject_Vectorcall(
        (PyObject *)state->degrees_of_freedom_type, (PyObject *[]){(PyObject *)function_space_object}, 1, NULL);
    Py_DECREF(function_space_object);
    if (!new_dofs)
    {
        return NULL;
    }

    const basis_set_type_t type = this->basis_specs[index].type;
    const unsigned n = this->basis_specs[index].order;
    size_t pre_stride = 1, post_stride = 1;
    for (unsigned idim = 0; idim < index; ++idim)
    {
        pre_stride *= this->basis_specs[idim].order + 1;
    }
    for (unsigned idim = index + 1; idim < this->n_dims; ++idim)
    {
        post_stride *= this->basis_specs[idim].order + 1;
    }

    const double *const values_in = this->values;
    double *const values_out = new_dofs->values;
    double *const work_buffer = PyMem_Malloc(sizeof(*work_buffer) * (n + (n + 1) + n * (n + 1)));
    const incidence_array_specifications_t array_specs = {
        .values_in = values_in,
        .values_out = values_out,
        .work = work_buffer,
    };
    const incidence_base_strides_t strides = {
        .pre_stride = pre_stride,
        .post_stride = post_stride,
        .n = n,
    };
    if (type == BASIS_LAGRANGE_CHEBYSHEV_GAUSS || type == BASIS_LAGRANGE_GAUSS ||
        type == BASIS_LAGRANGE_GAUSS_LOBATTO || type == BASIS_LAGRANGE_UNIFORM)
        lagrange_prepare_incidence_transformation(type, n, work_buffer);
    apply_incidence_operator_single(type, &strides, 1, 0, 0, &array_specs);

    PyMem_Free(work_buffer);

    return (PyObject *)new_dofs;
}

PyDoc_STRVAR(dof_at_boundary_docstring, "plane_projection(idim: int, x: float) -> DegreesOfFreedom\n"
                                        "Compute the projection of degrees of freedom on a plane.\n"
                                        "\n"
                                        "Parameters\n"
                                        "----------\n"
                                        "idim : int\n"
                                        "    Index of the dimension that is fixed.\n"
                                        "\n"
                                        "x : float\n"
                                        "    Position of the plane in that dimension.\n"
                                        "\n"
                                        "Returns\n"
                                        "-------\n"
                                        "DegreesOfFreedom\n"
                                        "    Degrees of freedom on the specified plane.\n");

PyObject *dof_at_boundary(PyObject *self, PyTypeObject *defining_class, PyObject *const *args, const Py_ssize_t nargs,
                          const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    const dof_object *this;
    Py_ssize_t idim;
    double value;
    if (ensure_dof_and_state(self, defining_class, &state, (dof_object **)&this) < 0)
        return NULL;

    if (this->n_dims < 2)
    {
        PyErr_Format(PyExc_ValueError, "Cannot compute the DoF at the boundary of a %u-dimensional function.",
                     this->n_dims);
        return NULL;
    }

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idim, .kwname = "idim"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &value, .kwname = "x", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (idim < 0 || idim >= this->n_dims)
    {
        PyErr_Format(PyExc_ValueError, "Expected an index between 0 and %u, but got %zd.", this->n_dims - 1, idim);
        return NULL;
    }

    // Find the basis set I will have to evaluate
    const basis_spec_t *const basis = this->basis_specs + idim;

    // Compute basis values at the given point
    double *basis_values, *roots;
    basis_spec_t *out_specs;
    void *const mem = cutl_alloc_group(
        &PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){
                               {.size = sizeof(*basis_values) * (basis->order + 1), .p_ptr = (void **)&basis_values},
                               {.size = sizeof(*roots) * (basis->order + 1), .p_ptr = (void **)&roots},
                               {.size = sizeof(*out_specs) * (this->n_dims - 1), .p_ptr = (void **)&out_specs},
                               {},
                           });
    if (!mem)
        return NULL;

    switch (basis->type)
    {
    case BASIS_BERNSTEIN:
        bernstein_interpolation_vector(value, basis->order, basis_values);
        break;

    case BASIS_LEGENDRE:
        legendre_eval_bonnet_all(basis->order, value, basis_values);
        break;

    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        // Set the roots
        {
            const fdg_result_t res = generate_lagrange_roots(basis->order, basis->type, roots);
            ASSERT(res == FDG_SUCCESS, "Could not generate roots for Lagrange basis.");
            (void)res;
        }
        lagrange_polynomial_values_2(1, &value, basis->order + 1, roots, basis_values);
        break;

    default:
        ASSERT(0, "Invalid basis type enum value %u", (unsigned)basis->type);
        break;
    }

    // Compute pre- and post-strides while filling in the output basis spaces
    size_t pre_stride = 1, post_stride = 1;
    for (unsigned i = 0; i < idim; ++i)
    {
        const unsigned cnt = this->basis_specs[i].order + 1;
        pre_stride *= cnt;
        out_specs[i] = this->basis_specs[i];
    }
    for (unsigned i = idim + 1; i < this->n_dims; ++i)
    {
        const unsigned cnt = this->basis_specs[i].order + 1;
        post_stride *= cnt;
        out_specs[i - 1] = this->basis_specs[i];
    }

    // Create the output DoF object
    dof_object *const new_dofs = dof_object_create(state->degrees_of_freedom_type, this->n_dims - 1, out_specs);
    if (!new_dofs)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
        return NULL;
    }

    double *restrict const out_dofs = new_dofs->values;
    const double *restrict const in_dofs = this->values;
    const unsigned n_dofs = basis->order + 1;

    // Loop over all DoFs that need to be dealt with
    for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
    {
        for (size_t i_post = 0; i_post < post_stride; ++i_post)
        {
            double v = 0.0;
            for (unsigned i = 0; i < n_dofs; ++i)
            {
                v += in_dofs[i_pre * n_dofs * post_stride + i * post_stride + i_post] * basis_values[i];
            }
            out_dofs[i_pre * post_stride + i_post] = v;
        }
    }

    // Release the memory
    cutl_dealloc(&PYTHON_ALLOCATOR, basis_values);
    // Done
    return (PyObject *)new_dofs;
}

PyDoc_STRVAR(dof_reverse_orientation_docstring,
             ""
             "reverse_orientation(idim: int) -> DegreesOfFreedom\n"
             "Reverse the orientation of DoFs.\n"
             "\n"
             "Maps the domain of basis functions for dimension ``idim`` from :math:`[-1, +1]`\n"
             "to :math:`[+1, -1]`.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idim : int\n"
             "    Index of the dimension on which the orientation should be reversed.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "DegreesOfFreedom\n"
             "    Degrees of freedom with reversed orientation on the specified dimension.\n");

PyObject *dof_reverse_orientation(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    const dof_object *this;
    Py_ssize_t idim;
    if (ensure_dof_and_state(self, defining_class, &state, (dof_object **)&this) < 0)
        return NULL;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idim, .kwname = "idim"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (idim < 0 || idim >= this->n_dims)
    {
        PyErr_Format(PyExc_ValueError, "Expected an index between 0 and %u, but got %zd.", this->n_dims - 1, idim);
        return NULL;
    }

    // Create a new DoF object with the same function space and all
    dof_object *const new_dofs = dof_object_create(state->degrees_of_freedom_type, this->n_dims, this->basis_specs);
    if (!new_dofs)
        return NULL;

    // Get the basis specs for the dimension we are reversing
    const basis_spec_t *const basis = this->basis_specs + idim;

    // Compute pre- and post-strides
    size_t pre_stride = 1, post_stride = 1;
    for (unsigned i = 0; i < idim; ++i)
    {
        const unsigned cnt = this->basis_specs[i].order + 1;
        pre_stride *= cnt;
    }
    for (unsigned i = idim + 1; i < this->n_dims; ++i)
    {
        const unsigned cnt = this->basis_specs[i].order + 1;
        post_stride *= cnt;
    }
    const unsigned n_dofs = basis->order + 1;

    // Loop over the unaffected dimensions
    for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
    {
        for (size_t i_post = 0; i_post < post_stride; ++i_post)
        {
            // Check what we do based on the basis type (and pray compiler moves this switch out of the loop)
            switch (basis->type)
            {
            case BASIS_BERNSTEIN:
            case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
            case BASIS_LAGRANGE_UNIFORM:
            case BASIS_LAGRANGE_GAUSS:
            case BASIS_LAGRANGE_GAUSS_LOBATTO:
                // Reverse basis along the dimension
                for (unsigned i = 0; i < n_dofs; ++i)
                {
                    new_dofs->values[i_pre * n_dofs * post_stride + i * post_stride + i_post] =
                        this->values[i_pre * n_dofs * post_stride + (n_dofs - i - 1) * post_stride + i_post];
                }
                break;

            case BASIS_LEGENDRE:
                // Negate every other coefficient along the dimension
                for (unsigned i = 0; i < n_dofs; i += 2)
                {
                    new_dofs->values[i_pre * n_dofs * post_stride + i * post_stride + i_post] =
                        this->values[i_pre * n_dofs * post_stride + i * post_stride + i_post];
                }
                for (unsigned i = 1; i < n_dofs; i += 2)
                {
                    new_dofs->values[i_pre * n_dofs * post_stride + i * post_stride + i_post] =
                        -this->values[i_pre * n_dofs * post_stride + i * post_stride + i_post];
                }
                break;
            default:
                ASSERT(0, "Invalid basis type enum value %u", (unsigned)basis->type);
                break;
            }
        }
    }

    // Done
    return (PyObject *)new_dofs;
}

PyDoc_STRVAR(
    dof_lagrange_projection_docstring,
    "lagrange_projection(orders: npt.ArrayLike | None = None, *, integration_registry: IntegrationRegistry = "
    "DEFAULT_INTEGRATION_REGISTRY, basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY) -> DegreesOfFreedom\n"
    "Compute projection of degrees of freedom with Lagrange basis.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "orders : array_like\n"
    "    Orders in each dimension. If nothing is given, then orders are taken to be\n"
    "    same as needed to exactly represent the degrees of freedom.\n"
    "\n"
    "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
    "    Registry used to retrieve the integration rules.\n"
    "\n"
    "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
    "    Registry used to retrieve the basis specifications.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "DegreesOfFreedom\n"
    "    Degrees of freedom using Lagrange basis of specified orders.\n");

PyObject *dof_lagrange_projection(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    const dof_object *this;
    if (ensure_dof_and_state(self, defining_class, &state, (dof_object **)&this) < 0)
        return NULL;
    // No keyword args
    PyObject *py_orders = NULL;
    integration_registry_object *py_int_registry = (integration_registry_object *)state->registry_integration;
    basis_registry_object *py_basis_registry = (basis_registry_object *)state->registry_basis;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_orders, .kwname = "orders", .optional = 1},
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &py_int_registry,
                    .type_check = state->integration_registry_type,
                    .kwname = "integration_registry",
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &py_basis_registry,
                    .type_check = state->basis_registry_type,
                    .kwname = "basis_registry",
                    .optional = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    // Null is the same as nothing
    if (Py_IsNone(py_orders))
        py_orders = NULL;

    integration_spec_t *target_int_specs;
    basis_spec_t *target_basis_specs;
    npy_intp *target_shape;
    void *const mem = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {.size = sizeof(*target_int_specs) * this->n_dims, .p_ptr = (void **)&target_int_specs},
            {.size = sizeof(*target_basis_specs) * this->n_dims, .p_ptr = (void **)&target_basis_specs},
            {.size = sizeof(*target_shape) * this->n_dims, .p_ptr = (void **)&target_shape},
            {},
        });
    if (!mem)
        return NULL;

    if (py_orders == NULL)
    {
        // Copy orders from these DoFs
        for (unsigned i = 0; i < this->n_dims; ++i)
        {
            target_int_specs[i].order = this->basis_specs[i].order;
        }
    }
    else
    {
        PyObject *py_seq = PySequence_Fast(py_orders, "Expected a sequence of orders.");
        if (!py_seq)
        {
            cutl_dealloc(&PYTHON_ALLOCATOR, mem);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(py_seq) != this->n_dims)
        {
            Py_DECREF(py_seq);
            PyErr_Format(PyExc_ValueError, "Expected a sequence of orders of length %u, but got %zd.", this->n_dims,
                         PySequence_Fast_GET_SIZE(py_seq));
            cutl_dealloc(&PYTHON_ALLOCATOR, mem);
            return NULL;
        }
        // Get orders from the arguments
        for (unsigned i = 0; i < this->n_dims; ++i)
        {
            PyObject *value = PySequence_Fast_GET_ITEM(py_seq, i);
            if (!PyNumber_Check(value))
            {
                Py_DECREF(py_seq);
                PyErr_Format(PyExc_TypeError, "Expected a number, but got %s as argument %u.", Py_TYPE(value)->tp_name,
                             i);
                cutl_dealloc(&PYTHON_ALLOCATOR, mem);
                return NULL;
            }
            const Py_ssize_t order = PyNumber_AsSsize_t(value, PyExc_OverflowError);
            if (order < 0)
            {
                Py_DECREF(py_seq);
                PyErr_Format(PyExc_TypeError, "Expected a positive number, but got %R as argument %u.", value, i);
                cutl_dealloc(&PYTHON_ALLOCATOR, mem);
                return NULL;
            }
            target_int_specs[i].order = order;
        }
        Py_DECREF(py_seq);
    }

    // Set the type as GLG and fill out the target shape
    Py_ssize_t total_dofs_in = 1;
    for (unsigned i = 0; i < this->n_dims; ++i)
    {
        target_int_specs[i].type = INTEGRATION_RULE_TYPE_GAUSS_LOBATTO;
        target_basis_specs[i].type = BASIS_LAGRANGE_GAUSS_LOBATTO;
        target_basis_specs[i].order = target_int_specs[i].order;
        target_shape[i] = target_int_specs[i].order + 1;
        total_dofs_in *= this->basis_specs[i].order + 1;
    }

    // Create output DoFs
    dof_object *const new_dofs = dof_object_create(state->degrees_of_freedom_type, this->n_dims, target_basis_specs);
    if (!new_dofs)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
        return NULL;
    }

    // Create a numpy array wrapper around the new DoFs object
    PyArrayObject *const new_dofs_array =
        (PyArrayObject *)PyArray_SimpleNewFromData(new_dofs->n_dims, target_shape, NPY_DOUBLE, new_dofs->values);
    if (!new_dofs_array)
    {
        Py_DECREF(new_dofs);
        cutl_dealloc(&PYTHON_ALLOCATOR, mem);
        return NULL;
    }

    // Call the method to evaluate the values of current DoFs at integration points
    PyArrayObject *const values = reconstruct_at_integration_points_impl(
        this, this->n_dims, target_int_specs, py_int_registry, py_basis_registry, new_dofs_array, total_dofs_in);

    Py_XDECREF(values);
    Py_DECREF(new_dofs_array);
    cutl_dealloc(&PYTHON_ALLOCATOR, mem);

    if (!values)
    {
        Py_DECREF(new_dofs);
        return NULL;
    }

    return (PyObject *)new_dofs;
}

static void dof_dealloc(dof_object *self)
{
    PyObject_GC_UnTrack(self);
    PyMem_Free(self->basis_specs);
    self->basis_specs = NULL;
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

PyType_Spec degrees_of_freedom_type_spec = {
    .name = FDG_TYPE_NAME("DegreesOfFreedom"),
    .basicsize = sizeof(dof_object),
    .itemsize = sizeof(double),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_new, dof_new},
        {Py_tp_doc, (void *)dof_docstring},
        {Py_tp_getset,
         (PyGetSetDef[]){
             {
                 .name = "function_space",
                 .get = dof_get_function_space,
                 .doc = "FunctionSpace : Function space the degrees of freedom belong to.",
             },
             {
                 .name = "n_dofs",
                 .get = dof_get_total_number,
                 .doc = "int : Total number of degrees of freedom.",
             },
             {
                 .name = "shape",
                 .get = dof_get_shape,
                 .doc = "tuple[int, ...] : Shape of the degrees of freedom.",
             },
             {
                 .name = "values",
                 .get = dof_get_values,
                 .set = dof_set_values,
                 .doc = "numpy.typing.NDArray[numpy.double] : Values of the degrees of freedom.",
             },
             {},
         }},
        {Py_tp_methods,
         (PyMethodDef[]){
             {
                 .ml_name = "reconstruct_at_integration_points",
                 .ml_meth = (void *)dof_reconstruct_at_integration_points,
                 .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
                 .ml_doc = (void *)dof_reconstruct_at_integration_points_docstring,
             },
             {
                 .ml_name = "reconstruct_derivative_at_integration_points",
                 .ml_meth = (void *)dof_reconstruct_derivative_at_integration_points,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = (void *)dof_reconstruct_derivative_at_integration_points_docstring,
             },
             {
                 .ml_name = "derivative",
                 .ml_meth = (void *)dof_derivative,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = dof_derivative_docstring,
             },
             {
                 .ml_name = "plane_projection",
                 .ml_meth = (void *)dof_at_boundary,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = dof_at_boundary_docstring,
             },
             {
                 .ml_name = "reverse_orientation",
                 .ml_meth = (void *)dof_reverse_orientation,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = dof_reverse_orientation_docstring,
             },
             {
                 .ml_name = "lagrange_projection",
                 .ml_meth = (void *)dof_lagrange_projection,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = dof_lagrange_projection_docstring,
             },
             {},
         }},
        {Py_tp_dealloc, dof_dealloc},
        {},
    }};
