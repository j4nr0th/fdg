#include "mass_matrices.h"
#include "basis_objects.h"
#include "covector_basis.h"
#include "cutl/iterators/combination_iterator.h"
#include "function_space_objects.h"
#include "integration_objects.h"
#include "mappings.h"

static double evaluate_basis_at_integration_point(const unsigned n_space_dim, const multidim_iterator_t *iter_int,
                                                  const multidim_iterator_t *iter_basis,
                                                  const basis_set_t *basis_sets[static n_space_dim])
{
    double basis_out = 1;
    for (unsigned idim = 0; idim < n_space_dim; ++idim)
    {
        const size_t integration_point_idx = multidim_iterator_get_offset(iter_int, idim);

        const size_t b_idx_out = multidim_iterator_get_offset(iter_basis, idim);
        const double out_basis_val = basis_set_basis_values(basis_sets[idim], b_idx_out)[integration_point_idx];

        basis_out *= out_basis_val;
    }
    return basis_out;
}

static double evaluate_basis_derivative_at_integration_point(const unsigned n_space_dim, const unsigned i_derivative,
                                                             const multidim_iterator_t *iter_int,
                                                             const multidim_iterator_t *iter_basis,
                                                             const basis_set_t *basis_sets[static n_space_dim])
{
    double basis_out = 1;
    for (unsigned idim = 0; idim < n_space_dim; ++idim)
    {
        const size_t integration_point_idx = multidim_iterator_get_offset(iter_int, idim);

        const size_t b_idx_out = multidim_iterator_get_offset(iter_basis, idim);
        double out_basis_val;
        if (i_derivative == idim)
        {
            out_basis_val = basis_set_basis_derivatives(basis_sets[idim], b_idx_out)[integration_point_idx];
        }
        else
        {
            out_basis_val = basis_set_basis_values(basis_sets[idim], b_idx_out)[integration_point_idx];
        }

        basis_out *= out_basis_val;
    }
    return basis_out;
}

static double evaluate_kform_basis_at_integration_point(const unsigned n_space_dim, const multidim_iterator_t *iter_int,
                                                        const multidim_iterator_t *iter_basis,
                                                        const basis_set_t *basis_sets[static n_space_dim],
                                                        const basis_set_t *basis_sets_lower[static n_space_dim],
                                                        const unsigned order, const uint8_t derivatives[static order])
{
    double basis_out = 1;
    for (unsigned idim = 0, iderivative = 0; idim < n_space_dim; ++idim)
    {
        const size_t integration_point_idx = multidim_iterator_get_offset(iter_int, idim);

        const size_t b_idx_out = multidim_iterator_get_offset(iter_basis, idim);
        double out_basis_val;

        if (iderivative < order && derivatives[iderivative] == idim)
        {
            out_basis_val = basis_set_basis_values(basis_sets_lower[idim], b_idx_out)[integration_point_idx];
            ++iderivative;
        }
        else
        {
            out_basis_val = basis_set_basis_values(basis_sets[idim], b_idx_out)[integration_point_idx];
        }

        basis_out *= out_basis_val;
    }
    return basis_out;
}

PyDoc_STRVAR(
    compute_mass_matrix_docstring,
    "compute_mass_matrix(space_in: FunctionSpace, space_out: FunctionSpace, integration: "
    "IntegrationSpace | SpaceMap, /, *, integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, "
    "basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY) -> numpy.typing.NDArray[numpy.double]\n"
    "Compute the mass matrix between two function spaces.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "space_in : FunctionSpace\n"
    "    Function space for the input functions.\n"
    "space_out : FunctionSpace\n"
    "    Function space for the output functions.\n"
    "integration : IntegrationSpace or SpaceMap\n"
    "    Integration space used to compute the mass matrix or a space mapping.\n"
    "    If the integration space is provided, the integration is done on the\n"
    "    reference domain. If the mapping is defined instead, the integration\n"
    "    space of the mapping is used, along with the integration being done\n"
    "    on the mapped domain instead.\n"
    "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
    "    Registry used to retrieve the integration rules.\n"
    "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
    "    Registry used to retrieve the basis specifications.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Mass matrix as a 2D array, which maps the primal degress of freedom of the input\n"
    "    function space to dual degrees of freedom of the output function space.\n");

typedef struct
{
    multidim_iterator_t *iter_in;
    multidim_iterator_t *iter_out;
    multidim_iterator_t *iter_int;
    unsigned n_rules;
    const integration_rule_t **rules;
    unsigned n_dim_in;
    const basis_set_t **basis_in;
    unsigned n_dim_out;
    const basis_set_t **basis_out;
    const double *determinant;
} mass_matrix_resources_t;

static void mass_matrix_release_resources(mass_matrix_resources_t *resources,
                                          integration_rule_registry_t *integration_registry,
                                          basis_set_registry_t *basis_registry)
{
    if (resources->basis_out)
        python_basis_sets_release(resources->n_dim_out, resources->basis_out, basis_registry);
    if (resources->basis_in)
        python_basis_sets_release(resources->n_dim_in, resources->basis_in, basis_registry);
    if (resources->rules)
        python_integration_rules_release(resources->n_rules, resources->rules, integration_registry);
    if (resources->iter_out)
        PyMem_Free(resources->iter_out);
    if (resources->iter_in)
        PyMem_Free(resources->iter_in);
    if (resources->iter_int)
        PyMem_Free(resources->iter_int);
    *resources = (mass_matrix_resources_t){};
}

static int mass_matrix_create_resources(const function_space_object *space_in, const function_space_object *space_out,
                                        const unsigned n_rules, const integration_spec_t *p_rules, const double *p_det,
                                        integration_rule_registry_t *integration_registry,
                                        basis_set_registry_t *basis_registry, mass_matrix_resources_t *resources)
{
    mass_matrix_resources_t res = {};
    // Create iterators for function spaces and integration rules
    res.iter_in = function_space_iterator(space_in);
    res.iter_out = function_space_iterator(space_out);
    res.iter_int = integration_specs_iterator(n_rules, p_rules);
    // Get integration rules and basis sets
    res.rules = python_integration_rules_get(n_rules, p_rules, integration_registry);
    res.n_rules = n_rules;
    const Py_ssize_t n_basis_in = Py_SIZE(space_in);
    res.basis_in = res.rules ? python_basis_sets_get(n_basis_in, space_in->specs, res.rules, basis_registry) : NULL;
    res.n_dim_in = n_basis_in;
    const Py_ssize_t n_basis_out = Py_SIZE(space_out);
    res.basis_out = res.rules ? python_basis_sets_get(n_basis_out, space_out->specs, res.rules, basis_registry) : NULL;
    res.n_dim_out = n_basis_out;
    if (!res.iter_in || !res.iter_out || !res.iter_int || !res.rules || !res.basis_in || !res.basis_out)
    {
        mass_matrix_release_resources(&res, integration_registry, basis_registry);
        return -1;
    }
    res.determinant = p_det;
    *resources = res;
    return 0;
}

static int function_spaces_match(const function_space_object *space_in, const function_space_object *space_out)
{
    if (space_in == space_out)
        return 1;

    const unsigned n_space_dim = Py_SIZE(space_in);
    if (n_space_dim != Py_SIZE(space_out))
        return 0;

    // Space contents might match instead
    for (unsigned i = 0; i < n_space_dim; ++i)
    {
        if (space_in->specs[i].order != space_out->specs[i].order ||
            space_in->specs[i].type != space_out->specs[i].type)
            return 0;
    }
    return 1;
}

static double calculate_integration_weight(const unsigned n_space_dim,
                                           const multidim_iterator_t *const iterator_integration,
                                           const integration_rule_t *int_rules[static n_space_dim])
{
    double weight = 1.0;
    for (unsigned idim = 0; idim < n_space_dim; ++idim)
    {
        const size_t integration_point_idx = multidim_iterator_get_offset(iterator_integration, idim);
        weight *= integration_rule_weights_const(int_rules[idim])[integration_point_idx];
    }

    return weight;
}
static PyObject *compute_mass_matrix(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                     const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;

    const function_space_object *space_in, *space_out;
    PyObject *py_integration;
    const integration_registry_object *integration_registry =
        (const integration_registry_object *)state->registry_integration;
    const basis_registry_object *basis_registry = (const basis_registry_object *)state->registry_basis;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &space_in,
                    .type_check = state->function_space_type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &space_out,
                    .type_check = state->function_space_type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &py_integration,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_registry,
                    .type_check = state->integration_registry_type,
                    .optional = 1,
                    .kwname = "integration_registry",
                    .kw_only = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &basis_registry,
                    .type_check = state->basis_registry_type,
                    .optional = 1,
                    .kwname = "basis_registry",
                    .kw_only = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    unsigned n_int_specs;
    const integration_spec_t *p_int_specs;
    const double *p_det;
    if (PyObject_TypeCheck(py_integration, state->integration_space_type))
    {
        const integration_space_object *const integration_space = (const integration_space_object *)py_integration;
        n_int_specs = Py_SIZE(integration_space);
        p_int_specs = integration_space->specs;
        p_det = NULL;
    }
    else if (PyObject_TypeCheck(py_integration, state->space_mapping_type))
    {
        const space_map_object *const space_map = (const space_map_object *)py_integration;
        n_int_specs = space_map->ndim;
        p_int_specs = space_map->int_specs;
        p_det = space_map->determinant;
    }
    else
    {
        PyErr_Format(PyExc_TypeError, "Integration space or space map must be passed, instead %s object was passed.",
                     Py_TYPE(py_integration)->tp_name);
        return NULL;
    }

    const unsigned n_space_dim = Py_SIZE(space_in);
    if (Py_SIZE(space_out) != n_space_dim || n_int_specs != n_space_dim)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Function spaces must have the same dimensionality (space in: %u, space out: %u, integration space: %u).",
            (unsigned)Py_SIZE(space_in), (unsigned)Py_SIZE(space_out), n_int_specs);
        return NULL;
    }

    // Create resources
    mass_matrix_resources_t resources = {};
    if (mass_matrix_create_resources(space_in, space_out, n_int_specs, p_int_specs, p_det,
                                     integration_registry->registry, basis_registry->registry, &resources))
        return NULL;

    const npy_intp dims[2] = {(npy_intp)multidim_iterator_total_size(resources.iter_out),
                              (npy_intp)multidim_iterator_total_size(resources.iter_in)};

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        mass_matrix_release_resources(&resources, integration_registry->registry, basis_registry->registry);
        return NULL;
    }
    npy_double *const p_out = PyArray_DATA(out);

    // Matrix is symmetric if spaces match
    const int is_symmetric = function_spaces_match(space_in, space_out);
    multidim_iterator_set_to_start(resources.iter_in);
    multidim_iterator_set_to_start(resources.iter_out);
    while (!multidim_iterator_is_at_end(resources.iter_out))
    {
        const size_t index_out = multidim_iterator_get_flat_index(resources.iter_out);
        CPYUTL_ASSERT(index_out < (size_t)dims[0], "Out index out of bounds.");
        const size_t index_in = multidim_iterator_get_flat_index(resources.iter_in);
        CPYUTL_ASSERT(index_in < (size_t)dims[1], "In index out of bounds.");

        multidim_iterator_t *const iterator_integration = resources.iter_int;
        // integrate the respective basis
        multidim_iterator_set_to_start(iterator_integration);
        double result = 0;
        // Integrate the basis product
        while (!multidim_iterator_is_at_end(iterator_integration))
        {
            // Compute weight and basis values for these outer product basis and integration
            double weight = resources.determinant
                                ? resources.determinant[multidim_iterator_get_flat_index(iterator_integration)]
                                : 1;
            // double basis_in = 1, basis_out = 1;
            // for (unsigned idim = 0; idim < n_space_dim; ++idim)
            // {
            //     const size_t integration_point_idx = multidim_iterator_get_offset(resources.iter_int, idim);
            //     weight *= integration_rule_weights_const(resources.rules[idim])[integration_point_idx];
            //     basis_in *= basis_set_basis_values(
            //         resources.basis_in[idim],
            //         multidim_iterator_get_offset(resources.iter_in, idim))[integration_point_idx];
            //     basis_out *= basis_set_basis_values(
            //         resources.basis_out[idim],
            //         multidim_iterator_get_offset(resources.iter_out, idim))[integration_point_idx];
            // }

            weight *= calculate_integration_weight(n_space_dim, iterator_integration, resources.rules);

            const double basis_in = evaluate_basis_at_integration_point(n_space_dim, iterator_integration,
                                                                        resources.iter_in, resources.basis_in);

            const double basis_out = evaluate_basis_at_integration_point(n_space_dim, iterator_integration,
                                                                         resources.iter_out, resources.basis_out);

            multidim_iterator_advance(iterator_integration, n_space_dim - 1, 1);
            // Add the contributions to the result
            result += weight * basis_in * basis_out;
        }

        // Write the output
        p_out[index_out * dims[1] + index_in] = result;

        // Advance the input basis
        multidim_iterator_advance(resources.iter_in, n_space_dim - 1, 1);
        // If we've done enough input basis, we advance the output basis and reset the input iterator
        if ((is_symmetric && index_in == index_out) || multidim_iterator_is_at_end(resources.iter_in))
        {
            multidim_iterator_advance(resources.iter_out, n_space_dim - 1, 1);
            multidim_iterator_set_to_start(resources.iter_in);
        }
    }

    // If we're symmetric, we have to fill up the upper diagonal part
    if (is_symmetric)
    {
        for (npy_intp i = 0; i < dims[0]; ++i)
        {
            for (npy_intp j = i + 1; j < dims[1]; ++j)
            {
                p_out[i * dims[1] + j] = p_out[j * dims[1] + i];
            }
        }
    }

    mass_matrix_release_resources(&resources, integration_registry->registry, basis_registry->registry);
    return (PyObject *)out;
}

static PyObject *compute_gradient_mass_matrix(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;

    const function_space_object *space_in, *space_out;
    PyObject *py_integration;
    Py_ssize_t idx_in;
    Py_ssize_t idx_out;
    const integration_registry_object *integration_registry =
        (const integration_registry_object *)state->registry_integration;
    const basis_registry_object *basis_registry = (const basis_registry_object *)state->registry_basis;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &space_in,
                    .type_check = state->function_space_type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &space_out,
                    .type_check = state->function_space_type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &py_integration,
                },
                {
                    .type = CPYARG_TYPE_SSIZE,
                    .p_val = &idx_in,
                    .kwname = "idx_in",
                },
                {
                    .type = CPYARG_TYPE_SSIZE,
                    .p_val = &idx_out,
                    .kwname = "idx_out",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_registry,
                    .type_check = state->integration_registry_type,
                    .optional = 1,
                    .kwname = "integration_registry",
                    .kw_only = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &basis_registry,
                    .type_check = state->basis_registry_type,
                    .optional = 1,
                    .kwname = "basis_registry",
                    .kw_only = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    unsigned n_int_specs;
    const integration_spec_t *p_int_specs;
    const double *p_det;
    const double *inverse_map;
    unsigned n_coords;
    size_t inv_map_stride = 0;
    if (PyObject_TypeCheck(py_integration, state->integration_space_type))
    {
        const integration_space_object *const integration_space = (const integration_space_object *)py_integration;
        n_int_specs = Py_SIZE(integration_space);
        p_int_specs = integration_space->specs;
        p_det = NULL;
        inverse_map = NULL;
        n_coords = n_int_specs;
    }
    else if (PyObject_TypeCheck(py_integration, state->space_mapping_type))
    {
        const space_map_object *const space_map = (const space_map_object *)py_integration;
        n_int_specs = space_map->ndim;
        p_int_specs = space_map->int_specs;
        p_det = space_map->determinant;

        n_coords = Py_SIZE(space_map);
        inverse_map = space_map->inverse_maps;
        inv_map_stride = (size_t)n_coords * n_int_specs;
    }
    else
    {
        PyErr_Format(PyExc_TypeError, "Integration space or space map must be passed, instead %s object was passed.",
                     Py_TYPE(py_integration)->tp_name);
        return NULL;
    }

    // Check input index
    if (idx_in < 0 || idx_in >= n_int_specs)
    {
        PyErr_Format(PyExc_ValueError, "Index %zd out of bounds for input space with %u dimensions.", idx_in,
                     n_int_specs);
        return NULL;
    }
    // Check output index
    if (idx_out < 0 || idx_out >= n_coords)
    {
        PyErr_Format(PyExc_ValueError, "Index %zd out of bounds for output space with %u dimensions.", idx_out,
                     n_coords);
        return NULL;
    }

    const unsigned n_space_dim = Py_SIZE(space_in);
    if (Py_SIZE(space_out) != n_space_dim || n_int_specs != n_space_dim)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Function spaces must have the same dimensionality (space in: %u, space out: %u, integration space: %u).",
            (unsigned)Py_SIZE(space_in), (unsigned)Py_SIZE(space_out), n_int_specs);
        return NULL;
    }

    // Quick check. If there's no space map (p_det = NULL) and idx_in != idx_out,
    // then every entry is zero and we do a quick return.
    if (p_det == NULL && idx_in != idx_out)
    {
        // Compute input and output space sizes
        npy_intp dims[2] = {1, 1};
        for (unsigned i = 0; i < n_int_specs; ++i)
        {
            dims[0] *= space_out->specs[i].order + 1;
            dims[1] *= space_in->specs[i].order + 1;
        }
        // Return already
        return PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    }

    // Create resources
    mass_matrix_resources_t resources = {};
    if (mass_matrix_create_resources(space_in, space_out, n_int_specs, p_int_specs, p_det,
                                     integration_registry->registry, basis_registry->registry, &resources))
    {
        return NULL;
    }

    const npy_intp dims[2] = {(npy_intp)multidim_iterator_total_size(resources.iter_out),
                              (npy_intp)multidim_iterator_total_size(resources.iter_in)};

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        mass_matrix_release_resources(&resources, integration_registry->registry, basis_registry->registry);
        return NULL;
    }
    npy_double *const p_out = PyArray_DATA(out);

    // Matrix is symmetric if spaces match
    const int is_symmetric = function_spaces_match(space_in, space_out);
    multidim_iterator_set_to_start(resources.iter_in);
    multidim_iterator_set_to_start(resources.iter_out);
    while (!multidim_iterator_is_at_end(resources.iter_out))
    {
        const size_t index_out = multidim_iterator_get_flat_index(resources.iter_out);
        CPYUTL_ASSERT(index_out < (size_t)dims[0], "Out index out of bounds.");
        const size_t index_in = multidim_iterator_get_flat_index(resources.iter_in);
        CPYUTL_ASSERT(index_in < (size_t)dims[1], "In index out of bounds.");

        // integrate the respective basis
        multidim_iterator_set_to_start(resources.iter_int);
        double result = 0;
        // Integrate the basis product
        while (!multidim_iterator_is_at_end(resources.iter_int))
        {
            const size_t integration_point_flat_idx = multidim_iterator_get_flat_index(resources.iter_int);
            // Compute weight and basis values for these outer product basis and integration
            const double *const local_inverse =
                inverse_map ? inverse_map + inv_map_stride * integration_point_flat_idx : NULL;
            double weight = resources.determinant ? resources.determinant[integration_point_flat_idx] *
                                                        local_inverse[(size_t)idx_in * n_coords + idx_out]
                                                  : 1;

            weight *= calculate_integration_weight(n_space_dim, resources.iter_int, resources.rules);

            // Chain rule for derivatives
            const double basis_in = evaluate_basis_derivative_at_integration_point(
                n_space_dim, idx_in, resources.iter_int, resources.iter_in, resources.basis_in);

            const double basis_out = evaluate_basis_at_integration_point(n_space_dim, resources.iter_int,
                                                                         resources.iter_out, resources.basis_out);

            multidim_iterator_advance(resources.iter_int, n_space_dim - 1, 1);
            // Add the contributions to the result
            result += weight * basis_in * basis_out;
        }

        // Write the output
        p_out[index_out * dims[1] + index_in] = result;

        // Advance the input basis
        multidim_iterator_advance(resources.iter_in, n_space_dim - 1, 1);
        // If we've done enough input basis, we advance the output basis and reset the input iterator
        if ((is_symmetric && index_in == index_out) || multidim_iterator_is_at_end(resources.iter_in))
        {
            multidim_iterator_advance(resources.iter_out, n_space_dim - 1, 1);
            multidim_iterator_set_to_start(resources.iter_in);
        }
    }

    // If we're symmetric, we have to fill up the upper diagonal part
    if (is_symmetric)
    {
        for (npy_intp i = 0; i < dims[0]; ++i)
        {
            for (npy_intp j = i + 1; j < dims[1]; ++j)
            {
                p_out[i * dims[1] + j] = p_out[j * dims[1] + i];
            }
        }
    }

    mass_matrix_release_resources(&resources, integration_registry->registry, basis_registry->registry);
    return (PyObject *)out;
}

PyDoc_STRVAR(compute_gradient_mass_matrix_docstring,
             "compute_gradient_mass_matrix(space_in: FunctionSpace, idims_in: typing.Sequence[int], space_out: "
             "FunctionSpace, idims_out: typing.Sequence[int], integration: IntegrationSpace | SpaceMap, /, *, "
             "integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, basis_registry: BasisRegistry "
             "= DEFAULT_BASIS_REGISTRY) -> numpy.typing.NDArray[numpy.double]\n"
             "Compute the mass matrix between two function spaces.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "space_in : FunctionSpace\n"
             "    Function space for the input functions.\n"
             "\n"
             "idim_in : Sequence of int\n"
             "    Indices of the dimension that input space is to be differentiated along.\n"
             "\n"
             "space_out : FunctionSpace\n"
             "    Function space for the output functions.\n"
             "\n"
             "idim_out : Sequence of int\n"
             "    Indices of the dimension that input space is to be differentiated along.\n"
             "\n"
             "integration : IntegrationSpace or SpaceMap\n"
             "    Integration space used to compute the mass matrix or a space mapping.\n"
             "    If the integration space is provided, the integration is done on the\n"
             "    reference domain. If the mapping is defined instead, the integration\n"
             "    space of the mapping is used, along with the integration being done\n"
             "    on the mapped domain instead.\n"
             "\n"
             "\n"
             "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rules.\n"
             "\n"
             "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
             "    Registry used to retrieve the basis specifications.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Mass matrix as a 2D array, which maps the primal degrees of freedom of the input\n"
             "    function space to dual degrees of freedom of the output function space.\n");

/*
def compute_kfrom_mass_matrix(
    smap: SpaceMap,
    order: int,
    left_bases: FunctionSpace,
    right_bases: FunctionSpace,
    basis_registry: BasisRegistry,
    int_registry: IntegrationRegistry,
) -> npt.NDArray[np.double]:
    """
    """
    ...
 */

PyDoc_STRVAR(
    compute_mass_matrix_component_docstring,
    "compute_kform_mass_matrix(smap: SpaceMap, order: int, left_bases: FunctionSpace, right_bases: FunctionSpace, *, "
    "int_registry: IntegrationRegistry, basis_registry: BasisRegistry) -> numpy.typing.NDArray[numpy.double]\n"
    "Compute the k-form mass matrix.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "smap : SpaceMap\n"
    "    Mapping of the space in which this is to be computed.\n"
    "\n"
    "order : int\n"
    "    Order of the k-form for which this is to be done.\n"
    "\n"
    "left_bases : FunctionSpace\n"
    "    Function space of 0-forms used as test forms.\n"
    "\n"
    "right_bases : FunctionSpace\n"
    "    Function space of 0-forms used as trial forms.\n"
    "\n"
    "basis_registry : BasisRegistry\n"
    "    Registry to get the basis from.\n"
    "\n"
    "int_registry : IntegrationRegistry\n"
    "    Registry to get the integration rules from.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "array\n"
    "    Mass matrix for inner product of two k-forms.\n");

static PyObject *compute_mass_matrix_component(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                               const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;

    const space_map_object *space_map;
    Py_ssize_t order;
    const function_space_object *fn_left, *fn_right;
    const integration_registry_object *integration_registry =
        (const integration_registry_object *)state->registry_integration;
    const basis_registry_object *basis_registry = (const basis_registry_object *)state->registry_basis;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &space_map,
                    .type_check = state->space_mapping_type,
                    .kwname = "smap",
                },
                {
                    .type = CPYARG_TYPE_SSIZE,
                    .p_val = &order,
                    .kwname = "order",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &fn_left,
                    .type_check = state->function_space_type,
                    .kwname = "basis_left",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &fn_right,
                    .type_check = state->function_space_type,
                    .kwname = "basis_right",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_registry,
                    .type_check = state->integration_registry_type,
                    .optional = 1,
                    .kwname = "integration_registry",
                    .kw_only = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &basis_registry,
                    .type_check = state->basis_registry_type,
                    .optional = 1,
                    .kwname = "basis_registry",
                    .kw_only = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned n = space_map->ndim;
    // Check function spaces and space map match.
    if (n != Py_SIZE(fn_left) || n != Py_SIZE(fn_right))
    {
        PyErr_Format(PyExc_ValueError,
                     "Basis dimensions must match the space map, but got %u and %u when expecting %u.",
                     Py_SIZE(fn_left), Py_SIZE(fn_right), n);
        return NULL;
    }
    // Check the order of k-form is within the possible range.
    if (order < 0 || order > n)
    {
        PyErr_Format(PyExc_ValueError, "Order %zd out of bounds for space map with %u dimensions.", order, n);
        return NULL;
    }

    // Function spaces must have order at least 1 in each dimension
    for (unsigned i = 0; i < n; ++i)
    {
        if (fn_left->specs[i].order < 1 || fn_right->specs[i].order < 1)
        {
            PyErr_Format(PyExc_ValueError, "Function spaces must have order at least 1 in each dimension.");
            return NULL;
        }
    }

    // Calculate the required space for storing intermediate integration weights for each k-form component.
    const unsigned int_pts_cnt = integration_specs_total_points(n, space_map->int_specs);

    // NOTE: we could try exploiting the symmetry of the matrix, but first we should check how critical this is
    // (probably quite significant).
    //
    // const int symmetric = function_spaces_match(fn_left, fn_right);

    // Compute needed space
    combination_iterator_t *iter_component_right, *iter_component_left;
    multidim_iterator_t *iter_basis_right, *iter_basis_left, *iter_int_pts;
    const integration_rule_t **integration_rules;
    const basis_set_t **basis_sets_left, **basis_sets_right, **basis_sets_left_lower, **basis_sets_right_lower;
    basis_spec_t *lower_basis_buffer;
    double *restrict integration_weights;
    void *const mem_1 = cutl_alloc_group(
        &PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){
                               {combination_iterator_required_memory(order), (void **)&iter_component_right},
                               {combination_iterator_required_memory(order), (void **)&iter_component_left},
                               {multidim_iterator_needed_memory(n), (void **)&iter_basis_right},
                               {multidim_iterator_needed_memory(n), (void **)&iter_basis_left},
                               {multidim_iterator_needed_memory(n), (void **)&iter_int_pts},
                               {sizeof(integration_rule_t *) * n, (void **)&integration_rules},
                               {sizeof(basis_set_t *) * n, (void **)&basis_sets_left},
                               {sizeof(basis_set_t *) * n, (void **)&basis_sets_left_lower},
                               {sizeof(basis_set_t *) * n, (void **)&basis_sets_right},
                               {sizeof(basis_set_t *) * n, (void **)&basis_sets_right_lower},
                               {sizeof(basis_spec_t) * n, (void **)&lower_basis_buffer},
                               {sizeof(double) * int_pts_cnt, (void **)&integration_weights},
                               {},
                           });
    if (!mem_1)
        return NULL;

    // Might as well prepare the integration point iterator now
    for (unsigned i = 0; i < n; ++i)
        multidim_iterator_init_dim(iter_int_pts, i, space_map->int_specs[i].order + 1);

    // Count up rows and columns based on DoFs of all components combined
    size_t row_cnt = 0, col_cnt = 0;
    // Loop over input and output bases
    combination_iterator_init(iter_component_right, n, order);
    for (const uint8_t *p_in = combination_iterator_current(iter_component_right);
         !combination_iterator_is_done(iter_component_right); combination_iterator_next(iter_component_right))
    {
        col_cnt += kform_basis_get_num_dofs(n, fn_right->specs, order, p_in);
    }

    combination_iterator_init(iter_component_left, n, order);
    for (const uint8_t *p_out = combination_iterator_current(iter_component_left);
         !combination_iterator_is_done(iter_component_left); combination_iterator_next(iter_component_left))
    {
        row_cnt += kform_basis_get_num_dofs(n, fn_left->specs, order, p_out);
    }

    const npy_intp dims[2] = {(npy_intp)row_cnt, (npy_intp)col_cnt};
    PyArrayObject *const array_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!array_out)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
        return NULL;
    }

    PyArrayObject *transform_array = NULL;

    if (order != 0 && order != n)
    {
        transform_array = compute_basis_transform_impl(space_map, order);
        if (!transform_array)
        {
            Py_DECREF(array_out);
            cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
            return NULL;
        }
    }

    // Get integration rules
    interp_result_t res =
        integration_rule_registry_get_rules(integration_registry->registry, n, space_map->int_specs, integration_rules);
    if (res != INTERP_SUCCESS)
    {
        Py_DECREF(array_out);
        Py_XDECREF(transform_array);
        cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
        return NULL;
    }

    // Get left basis sets
    res = basis_set_registry_get_basis_sets(basis_registry->registry, n, basis_sets_left, integration_rules,
                                            fn_left->specs);
    if (res != INTERP_SUCCESS)
    {
        for (unsigned i = 0; i < n; ++i)
            integration_rule_registry_release_rule(integration_registry->registry, integration_rules[i]);
        Py_DECREF(array_out);
        Py_XDECREF(transform_array);
        cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
        return NULL;
    }

    // Get the right basis sets
    res = basis_set_registry_get_basis_sets(basis_registry->registry, n, basis_sets_right, integration_rules,
                                            fn_right->specs);
    if (res != INTERP_SUCCESS)
    {
        for (unsigned i = 0; i < n; ++i)
        {
            integration_rule_registry_release_rule(integration_registry->registry, integration_rules[i]);
            basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_left[i]);
        }
        Py_DECREF(array_out);
        Py_XDECREF(transform_array);
        cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
        return NULL;
    }

    // Prepare lower basis specs for left
    for (unsigned i = 0; i < n; ++i)
    {
        lower_basis_buffer[i] = (basis_spec_t){.type = fn_left->specs[i].type, .order = fn_left->specs[i].order - 1};
    }
    // Get left lower basis sets
    res = basis_set_registry_get_basis_sets(basis_registry->registry, n, basis_sets_left_lower, integration_rules,
                                            lower_basis_buffer);
    if (res != INTERP_SUCCESS)
    {
        for (unsigned i = 0; i < n; ++i)
        {
            integration_rule_registry_release_rule(integration_registry->registry, integration_rules[i]);
            basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_left[i]);
            basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_right[i]);
        }
        Py_DECREF(array_out);
        Py_XDECREF(transform_array);
        cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
        return NULL;
    }

    // Prepare lower basis specs for right
    for (unsigned i = 0; i < n; ++i)
    {
        lower_basis_buffer[i] = (basis_spec_t){.type = fn_right->specs[i].type, .order = fn_right->specs[i].order - 1};
    }
    // Get right lower basis sets
    res = basis_set_registry_get_basis_sets(basis_registry->registry, n, basis_sets_right_lower, integration_rules,
                                            lower_basis_buffer);
    if (res != INTERP_SUCCESS)
    {
        for (unsigned i = 0; i < n; ++i)
        {
            integration_rule_registry_release_rule(integration_registry->registry, integration_rules[i]);
            basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_left[i]);
            basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_right[i]);
            basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_left_lower[i]);
        }
        Py_DECREF(array_out);
        Py_XDECREF(transform_array);
        cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
        return NULL;
    }

    npy_double *restrict const ptr_mat_out = PyArray_DATA(array_out);

    // Now compute numerical integrals
    size_t row_offset = 0;
    size_t idx_left = 0;
    size_t basis_idx_left = 0;

    // Loop over left k-form components
    combination_iterator_init(iter_component_left, n, order);
    for (const uint8_t *p_basis_components_left = combination_iterator_current(iter_component_left);
         !combination_iterator_is_done(iter_component_left);
         combination_iterator_next(iter_component_left), ++basis_idx_left)
    {
        // Set the iterator for basis functions of the left k-form component
        kform_basis_set_iterator(n, fn_left->specs, order, p_basis_components_left, iter_basis_left);

        size_t idx_right = 0;
        size_t col_offset = 0;
        size_t basis_idx_right = 0;
        // Loop over right k-form components
        combination_iterator_init(iter_component_right, n, order);
        for (const uint8_t *p_basis_components_right = combination_iterator_current(iter_component_right);
             !combination_iterator_is_done(iter_component_right);
             combination_iterator_next(iter_component_right), ++basis_idx_right)
        {
            // Compute the integration weights in advance
            for (multidim_iterator_set_to_start(iter_int_pts); !multidim_iterator_is_at_end(iter_int_pts);
                 multidim_iterator_advance(iter_int_pts, n - 1, 1))
            {
                double int_weight = calculate_integration_weight(n, iter_int_pts, integration_rules);
                const size_t integration_pt_flat_idx = multidim_iterator_get_flat_index(iter_int_pts);
                if (order == 0)
                {
                    ASSERT(transform_array == NULL, "Transform array should be NULL for order 0.");
                    // For 0-form it's just the determinant
                    int_weight *= space_map->determinant[integration_pt_flat_idx];
                }
                else if (order == n)
                {
                    ASSERT(transform_array == NULL, "Transform array should be NULL for order n.");
                    // For n-form it is the inverse of determinant
                    int_weight /= space_map->determinant[integration_pt_flat_idx];
                }
                else
                {
                    ASSERT(transform_array != NULL, "Transform array should not be NULL for order > 0.");
                    // For all others we must compute them from transformation matrix, after determinant
                    int_weight *= space_map->determinant[integration_pt_flat_idx];
                    const npy_double *restrict const trans_mat = PyArray_DATA(transform_array);
                    const npy_intp *restrict const trans_dims = PyArray_DIMS(transform_array);
                    // Contraction of 2-nd dimension for the current components and integration point
                    double dp = 0;
                    ASSERT(basis_idx_left < (size_t)trans_dims[0] && basis_idx_right < (size_t)trans_dims[0],
                           "Input basis indices are not correct for the transformation array shape");
                    for (unsigned m = 0; m < trans_dims[1]; ++m)
                    {
                        const double v_left = trans_mat[basis_idx_left * trans_dims[1] * trans_dims[2] +
                                                        m * trans_dims[2] + integration_pt_flat_idx];
                        const double v_right = trans_mat[basis_idx_right * trans_dims[1] * trans_dims[2] +
                                                         m * trans_dims[2] + integration_pt_flat_idx];
                        dp += v_left * v_right;
                    }
                    // Multiply the factor by the weight
                    int_weight *= dp;
                }

                integration_weights[integration_pt_flat_idx] = int_weight;
            }

            // Set the iterator for basis functions of the right k-form component
            kform_basis_set_iterator(n, fn_right->specs, order, p_basis_components_right, iter_basis_right);

            // Loop over basis functions of the left k-form component
            for (multidim_iterator_set_to_start(iter_basis_left), idx_left = 0;
                 !multidim_iterator_is_at_end(iter_basis_left);
                 multidim_iterator_advance(iter_basis_left, n - 1, 1), ++idx_left)
            {
                // Loop over basis functions of the right k-form component
                for (multidim_iterator_set_to_start(iter_basis_right), idx_right = 0;
                     !multidim_iterator_is_at_end(iter_basis_right);
                     multidim_iterator_advance(iter_basis_right, n - 1, 1), ++idx_right)
                {

                    double integral_value = 0;
                    // Loop over all integration points
                    for (multidim_iterator_set_to_start(iter_int_pts); !multidim_iterator_is_at_end(iter_int_pts);
                         multidim_iterator_advance(iter_int_pts, n - 1, 1))
                    {
                        const size_t integration_pt_flat_idx = multidim_iterator_get_flat_index(iter_int_pts);
                        const double int_weight = integration_weights[integration_pt_flat_idx];
                        const double basis_value_left = evaluate_kform_basis_at_integration_point(
                            n, iter_int_pts, iter_basis_left, basis_sets_left, basis_sets_left_lower, order,
                            p_basis_components_left);
                        const double basis_value_right = evaluate_kform_basis_at_integration_point(
                            n, iter_int_pts, iter_basis_right, basis_sets_right, basis_sets_right_lower, order,
                            p_basis_components_right);
                        integral_value += int_weight * basis_value_left * basis_value_right;
                    }
                    ptr_mat_out[(row_offset + idx_left) * col_cnt + (col_offset + idx_right)] = integral_value;
                }
            }

            const unsigned dofs_right = kform_basis_get_num_dofs(n, fn_right->specs, order, p_basis_components_right);
            ASSERT(dofs_right == idx_right, "I miscounted dof counts");
            col_offset += dofs_right;
        }

        const unsigned dofs_left = kform_basis_get_num_dofs(n, fn_left->specs, order, p_basis_components_left);
        ASSERT(dofs_left == idx_left, "I miscounted dof counts");
        row_offset += dofs_left;
    }

    // Release integration rules and basis
    for (unsigned j = 0; j < n; ++j)
    {
        integration_rule_registry_release_rule(integration_registry->registry, integration_rules[j]);
        basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_left[j]);
        basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_right[j]);
        basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_left_lower[j]);
        basis_set_registry_release_basis_set(basis_registry->registry, basis_sets_right_lower[j]);
    }
    cutl_dealloc(&PYTHON_ALLOCATOR, mem_1);
    Py_XDECREF(transform_array);

    return (PyObject *)array_out;
}

PyMethodDef mass_matrices_methods[] = {
    {
        .ml_name = "compute_mass_matrix",
        .ml_meth = (void *)compute_mass_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = compute_mass_matrix_docstring,
    },
    {
        .ml_name = "compute_gradient_mass_matrix",
        .ml_meth = (void *)compute_gradient_mass_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = compute_gradient_mass_matrix_docstring,
    },
    {
        .ml_name = "compute_kform_mass_matrix",
        .ml_meth = (void *)compute_mass_matrix_component,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = compute_mass_matrix_component_docstring,
    },
    {},
};
