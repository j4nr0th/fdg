#include "../constraints/constraints.h"
#include "../topology/topology.h"
#include "basis_objects.h"
#include "constraints.h"
#include "cpyutl.h"
#include "cutl/iterators/combination_iterator.h"
#include "integration_objects.h"
#include "kform_objects.h"
#include "mappings.h"
#include "module.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

static int make_integration_space(const interplib_module_state_t *state, const space_map_object *map,
                                  integration_space_object **out)
{
    integration_space_object *space =
        (integration_space_object *)state->integration_space_type->tp_alloc(state->integration_space_type, map->ndim);
    if (!space)
        return -1;
    for (unsigned idim = 0; idim < map->ndim; ++idim)
        space->specs[idim] = map->int_specs[idim];
    *out = space;
    return 0;
}

static int make_boundary_map(const interplib_module_state_t *state, const space_map_object *volume_map,
                             const int8_t *orientation, const unsigned element_dim, const unsigned face_dim,
                             integration_space_object *face_space, PyObject **out)
{
    PyObject *current = (PyObject *)volume_map;
    Py_INCREF(current);
    const unsigned fixed_count = element_dim - face_dim;
    for (unsigned fixed_index = fixed_count; fixed_index > 0; --fixed_index)
    {
        const int8_t fixed_orientation = orientation[fixed_index - 1];
        const unsigned source_axis = (unsigned)(fixed_orientation < 0 ? -fixed_orientation : fixed_orientation) - 1;
        PyObject *end_object = PyBool_FromLong(fixed_orientation > 0);
        if (!end_object)
        {
            Py_DECREF(current);
            return -1;
        }
        PyObject *result = PyObject_CallMethod(current, "boundary", "iO", (int)source_axis, end_object);
        Py_DECREF(end_object);
        Py_DECREF(current);
        if (!result)
            return -1;
        current = result;
    }
    (void)state;
    (void)face_space;
    *out = current;
    return 0;
}

static void release_collection_arrays(const unsigned count, PyArrayObject *arrays[const static count])
{
    for (unsigned i = 0; i < count; ++i)
        Py_XDECREF(arrays[i]);
}
typedef struct
{
    PyArrayObject **collection_arrays;
    topo_obj_collection_t *collections;
    topo_obj_immersion_t *immersions;
    int8_t *orientation;
    void *memory;
} boundary_topology_t;

static void release_boundary_topology(const unsigned element_dim, boundary_topology_t *const topology)
{
    if (topology->immersions)
        topo_obj_immersions_free(element_dim, topology->immersions, &PYTHON_ALLOCATOR);
    release_collection_arrays(element_dim, topology->collection_arrays);
    cutl_dealloc(&PYTHON_ALLOCATOR, topology->memory);
    *topology = (boundary_topology_t){};
}

static int make_boundary_topology(PyObject *const collections_object, const unsigned element_dim, const unsigned npts,
                                  boundary_topology_t *const topology)
{
    *topology = (boundary_topology_t){};
    topology->memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {sizeof(*topology->collection_arrays) * element_dim, (void **)&topology->collection_arrays},
            {sizeof(*topology->collections) * element_dim, (void **)&topology->collections},
            {sizeof(*topology->immersions) * element_dim, (void **)&topology->immersions},
            {sizeof(*topology->orientation) * element_dim, (void **)&topology->orientation},
            {}});
    if (!topology->memory)
        return -1;
    memset(topology->collection_arrays, 0, sizeof(*topology->collection_arrays) * element_dim);
    memset(topology->immersions, 0, sizeof(*topology->immersions) * element_dim);
    for (unsigned idim = 0; idim < element_dim; ++idim)
    {
        topology->collection_arrays[idim] = (PyArrayObject *)PyArray_FROMANY(PyTuple_GET_ITEM(collections_object, idim),
                                                                             NPY_UINT64, 2, 2, NPY_ARRAY_IN_ARRAY);
        if (!topology->collection_arrays[idim] || PyArray_DIM(topology->collection_arrays[idim], 1) != 2 * (idim + 1))
        {
            PyErr_Format(PyExc_ValueError, "Mesh collection %u must have shape (count, %u).", idim, 2 * (idim + 1));
            release_boundary_topology(element_dim, topology);
            return -1;
        }
        topology->collections[idim] = (topo_obj_collection_t){
            .ndim = idim + 1,
            .count = (size_t)PyArray_DIM(topology->collection_arrays[idim], 0),
            .boundary_ids = PyArray_DATA(topology->collection_arrays[idim]),
        };
    }
    const topo_status_t status = topo_obj_create_immersion_info(element_dim, npts, topology->collections,
                                                                &PYTHON_ALLOCATOR, topology->immersions);
    if (status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not create mesh immersions: %s (%s).", topo_status_to_str(status),
                     topo_status_msg(status));
        release_boundary_topology(element_dim, topology);
        return -1;
    }
    return 0;
}

static size_t total_points(const unsigned ndim, const integration_spec_t specs[const static ndim])
{
    size_t result = 1;
    for (unsigned idim = 0; idim < ndim; ++idim)
        result *= specs[idim].order + 1;
    return result;
}

static void get_digits(const unsigned ndim, const integration_spec_t specs[const static ndim], size_t point,
                       unsigned digits[const static ndim])
{
    for (unsigned idim = ndim; idim > 0; --idim)
    {
        const unsigned axis = idim - 1;
        digits[axis] = (unsigned)(point % (specs[axis].order + 1));
        point /= specs[axis].order + 1;
    }
}

static unsigned get_source_face_axis(const unsigned element_dim, const unsigned face_dim,
                                     const int8_t orientation[const static element_dim], const unsigned source_axis)
{
    const unsigned fixed_count = element_dim - face_dim;
    unsigned face_axis = 0;
    for (unsigned axis = 0; axis < source_axis; ++axis)
    {
        bool fixed = false;
        for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
        {
            fixed |= (unsigned)(orientation[fixed_axis] < 0 ? -orientation[fixed_axis] : orientation[fixed_axis]) - 1 ==
                     axis;
        }
        if (!fixed)
            ++face_axis;
    }
    return face_axis;
}

static size_t canonical_point_to_source(const unsigned element_dim, const unsigned face_dim,
                                        const int8_t orientation[const static element_dim],
                                        const integration_spec_t source_specs[const static face_dim],
                                        const integration_spec_t canonical_specs[const static face_dim],
                                        const unsigned canonical_digits[const static face_dim])
{
    const unsigned fixed_count = element_dim - face_dim;
    size_t source_point = 0;
    size_t stride = 1;
    for (unsigned source_axis = face_dim; source_axis > 0; --source_axis)
    {
        const unsigned source_face_axis = source_axis - 1;
        unsigned source_digit = 0;
        for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
        {
            const int8_t mapping = orientation[fixed_count + face_axis];
            const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
            if (get_source_face_axis(element_dim, face_dim, orientation, element_axis) == source_face_axis)
            {
                source_digit = mapping < 0 ? source_specs[source_face_axis].order - canonical_digits[face_axis]
                                           : canonical_digits[face_axis];
                break;
            }
        }
        source_point += source_digit * stride;
        stride *= source_specs[source_face_axis].order + 1;
    }
    (void)canonical_specs;
    return source_point;
}

static unsigned map_component(const unsigned face_dim, const unsigned element_dim, const unsigned order,
                              const uint8_t face_axes[const static order == 0 ? 1 : order],
                              const int8_t orientation[const static element_dim], int *sign,
                              uint8_t mapped[const static order == 0 ? 1 : order])
{
    *sign = 1;
    for (unsigned i = 0; i < order; ++i)
    {
        const int8_t mapping = orientation[element_dim - face_dim + face_axes[i]];
        mapped[i] = (uint8_t)(mapping < 0 ? -mapping : mapping) - 1;
        if (mapping < 0)
            *sign = -*sign;
    }
    for (unsigned i = 0; i < order; ++i)
        for (unsigned j = i + 1; j < order; ++j)
            if (mapped[i] > mapped[j])
            {
                *sign = -*sign;
                const uint8_t tmp = mapped[i];
                mapped[i] = mapped[j];
                mapped[j] = tmp;
            }
    (void)face_dim;
    return combination_get_index(element_dim, order, mapped);
}

static int create_pullback(const space_map_object *volume_face, const unsigned element_dim, const unsigned face_dim,
                           const int8_t orientation[const static element_dim], const unsigned order,
                           const integration_spec_t face_specs[const static face_dim],
                           const integration_spec_t canonical_specs[const static face_dim], PyArrayObject **out)
{
    if (volume_face->ndim != face_dim)
        return -1;
    const unsigned physical_components = combination_total_count((uint8_t)Py_SIZE(volume_face), (uint8_t)order);
    const unsigned face_components = combination_total_count((uint8_t)face_dim, (uint8_t)order);
    const size_t point_count = total_points(face_dim, canonical_specs);
    const npy_intp dims[3] = {combination_total_count((uint8_t)element_dim, (uint8_t)order), physical_components,
                              (npy_intp)point_count};
    PyArrayObject *pullback = (PyArrayObject *)PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
    if (!pullback)
        return -1;

    if (order == 0)
    {
        *out = pullback;
        return 0;
    }

    PyArrayObject *transform = compute_basis_transform_impl(volume_face, order);
    if (!transform)
    {
        Py_DECREF(pullback);
        return -1;
    }
    uint8_t *face_axes;
    unsigned *canonical_digits;
    uint8_t *mapped;
    void *const memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){
                               {sizeof(*face_axes) * order, (void **)&face_axes},
                               {sizeof(*canonical_digits) * (face_dim > 0 ? face_dim : 1), (void **)&canonical_digits},
                               {sizeof(*mapped) * order, (void **)&mapped},
                               {}});
    if (!memory)
    {
        Py_DECREF(transform);
        Py_DECREF(pullback);
        return -1;
    }
    const size_t source_point_count = total_points(face_dim, face_specs);
    const npy_double *transform_data = PyArray_DATA(transform);
    npy_double *pullback_data = PyArray_DATA(pullback);
    for (unsigned face_component = 0; face_component < face_components; ++face_component)
    {
        combination_set_to_index((uint8_t)face_dim, (uint8_t)order, face_axes, face_component);
        int orientation_sign;
        const unsigned element_component =
            map_component(face_dim, element_dim, order, face_axes, orientation, &orientation_sign, mapped);
        (void)orientation_sign;
        for (size_t canonical_point = 0; canonical_point < point_count; ++canonical_point)
        {
            get_digits(face_dim, canonical_specs, canonical_point, canonical_digits);
            const size_t source_point = canonical_point_to_source(element_dim, face_dim, orientation, face_specs,
                                                                  canonical_specs, canonical_digits);
            for (unsigned physical_component = 0; physical_component < physical_components; ++physical_component)
            {
                const size_t source_index =
                    ((size_t)face_component * physical_components + physical_component) * source_point_count +
                    source_point;
                const size_t target_index =
                    ((size_t)element_component * physical_components + physical_component) * point_count +
                    canonical_point;
                pullback_data[target_index] = transform_data[source_index];
            }
        }
    }
    cutl_dealloc(&PYTHON_ALLOCATOR, memory);
    Py_DECREF(transform);
    *out = pullback;
    return 0;
}
static PyObject *compute_kform_boundary_constraints(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                                    const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;

    PyObject *test_object;
    PyObject *spec_object;
    PyObject *map_object;
    PyObject *collections_object;
    Py_ssize_t npts;
    Py_ssize_t element_id;
    Py_ssize_t boundary_id;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &test_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &spec_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &map_object, .type_check = state->space_mapping_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &collections_object},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &npts},
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
    if (npts < 0 || element_id < 0 || boundary_id < 0 || face_dim >= element_dim || element_spec->order != order ||
        Py_SIZE(element_spec->function_space) != element_dim || element_map->ndim != element_dim)
    {
        PyErr_SetString(PyExc_ValueError, "Incompatible test, element, or topology dimensions.");
        return NULL;
    }
    if (!PyTuple_Check(collections_object) || PyTuple_GET_SIZE(collections_object) != element_dim)
    {
        PyErr_Format(PyExc_ValueError, "Expected %u mesh collections.", element_dim);
        return NULL;
    }

    boundary_topology_t topology;
    if (make_boundary_topology(collections_object, element_dim, (unsigned)npts, &topology) < 0)
        return NULL;
    const unsigned boundary_immersion_index = face_dim;
    const topo_status_t topo_status =
        topo_obj_boundary_orientation(topology.immersions + boundary_immersion_index, element_dim,
                                      (uint64_t)boundary_id, (uint64_t)element_id, topology.orientation);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Boundary %zd is not present in element %zd: %s (%s).", boundary_id, element_id,
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        release_boundary_topology(element_dim, &topology);
        return NULL;
    }

    PyObject *const result =
        compute_kform_boundary_constraints_impl(state, test_spec, element_spec, element_map, topology.orientation);
    release_boundary_topology(element_dim, &topology);
    return result;
}

typedef struct
{
    PyObject *face_object;
    integration_space_object *face_space;
    integration_spec_t *canonical_specs;
    constraint_quadrature_t *quadrature_axes;
    const integration_rule_t **rules;
    size_t point_count;
    void *memory;
} boundary_face_setup_t;

typedef struct
{
    constraint_trace_basis_values_t descriptor;
    size_t *component_offsets;
    double *values;
    void *memory;
} trace_basis_table_t;

static void release_trace_basis_table(trace_basis_table_t *const table)
{
    if (!table)
        return;
    cutl_dealloc(&PYTHON_ALLOCATOR, table->memory);
    *table = (trace_basis_table_t){};
}

static size_t trace_basis_point_index(const unsigned element_dim, const unsigned face_dim, const int8_t *orientation,
                                      const unsigned basis_axis, const size_t point,
                                      const integration_spec_t *canonical_specs, const size_t *point_strides,
                                      const bool element_table)
{
    if (!element_table)
        return (point / point_strides[basis_axis]) % ((size_t)canonical_specs[basis_axis].order + 1);

    const unsigned fixed_count = element_dim - face_dim;
    for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
    {
        const int8_t mapping = orientation[fixed_axis];
        if ((unsigned)(mapping < 0 ? -mapping : mapping) - 1 == basis_axis)
            return mapping < 0 ? 0 : 1;
    }
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = orientation[fixed_count + face_axis];
        if ((unsigned)(mapping < 0 ? -mapping : mapping) - 1 == basis_axis)
        {
            const size_t index = (point / point_strides[face_axis]) % ((size_t)canonical_specs[face_axis].order + 1);
            return mapping < 0 ? (size_t)canonical_specs[face_axis].order - index : index;
        }
    }
    return 0;
}

static int make_trace_basis_table(const unsigned element_dim, const unsigned face_dim, const unsigned order,
                                  const basis_spec_t *basis_specs, const int8_t *orientation,
                                  const integration_spec_t *canonical_specs,
                                  const integration_rule_t *const *source_rules,
                                  integration_registry_object *const integration_registry,
                                  basis_registry_object *const basis_registry, const bool element_table,
                                  const size_t point_count, trace_basis_table_t *const out)
{
    *out = (trace_basis_table_t){};
    const unsigned ndim = element_table ? element_dim : face_dim;
    const unsigned component_count = combination_total_count((uint8_t)ndim, (uint8_t)order);
    const size_t point_stride_count = face_dim > 0 ? face_dim : 1;
    size_t *point_strides;
    const integration_rule_t **canonical_rules;
    const integration_rule_t **axis_rules;
    const basis_set_t **basis_sets = NULL;
    const basis_set_t **basis_sets_lower = NULL;
    basis_spec_t *lower_specs;
    uint8_t *component_axes;
    const integration_rule_t *endpoint_rule = NULL;
    const unsigned axis_count = ndim > 0 ? ndim : 1;
    void *const memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){{sizeof(*point_strides) * point_stride_count, (void **)&point_strides},
                                    {sizeof(*canonical_rules) * face_dim, (void **)&canonical_rules},
                                    {sizeof(*axis_rules) * axis_count, (void **)&axis_rules},
                                    {sizeof(*lower_specs) * ndim, (void **)&lower_specs},
                                    {sizeof(*component_axes) * (order > 0 ? order : 1), (void **)&component_axes},
                                    {}});
    if (!memory)
        return -1;

    size_t stride = 1;
    for (unsigned axis = face_dim; axis > 0; --axis)
    {
        point_strides[axis - 1] = stride;
        stride *= (size_t)canonical_specs[axis - 1].order + 1;
    }
    if (stride != point_count)
    {
        PyErr_SetString(PyExc_ValueError, "Trace basis point count does not match its quadrature.");
        goto fail;
    }
    if (face_dim > 0)
    {
        const unsigned fixed_count = element_dim - face_dim;
        for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
        {
            const int8_t mapping = orientation[fixed_count + face_axis];
            const unsigned source_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
            const unsigned source_face_axis = get_source_face_axis(element_dim, face_dim, orientation, source_axis);
            canonical_rules[face_axis] = source_rules[source_face_axis];
        }
    }

    if (element_table && element_dim > face_dim)
    {
        const integration_spec_t endpoint_spec = {
            .type = INTEGRATION_RULE_TYPE_GAUSS_LOBATTO,
            .order = 1,
        };
        if (integration_rule_registry_get_rule(integration_registry->registry, endpoint_spec, &endpoint_rule) !=
            FDG_SUCCESS)
            goto fail;
    }
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        if (!element_table)
        {
            axis_rules[axis] = canonical_rules[axis];
            continue;
        }
        axis_rules[axis] = endpoint_rule;
        const unsigned fixed_count = element_dim - face_dim;
        for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
        {
            const int8_t mapping = orientation[fixed_count + face_axis];
            if ((unsigned)(mapping < 0 ? -mapping : mapping) - 1 == axis)
            {
                axis_rules[axis] = canonical_rules[face_axis];
                break;
            }
        }
    }

    const constraint_kform_spec_t descriptor = {.ndim = ndim, .order = order, .basis_specs = basis_specs};
    size_t total_dofs = 0;
    for (unsigned component = 0; component < component_count; ++component)
    {
        size_t dof_count;
        if (constraint_kform_component_dof_count(&descriptor, component, &dof_count) != CONSTRAINT_SUCCESS ||
            total_dofs > SIZE_MAX - dof_count)
        {
            PyErr_SetString(PyExc_OverflowError, "Trace basis DoF count exceeds the size limit.");
            goto fail;
        }
        total_dofs += dof_count;
    }
    if (total_dofs > SIZE_MAX / point_count || total_dofs * point_count > SIZE_MAX / sizeof(*out->values))
    {
        PyErr_SetString(PyExc_OverflowError, "Trace basis values exceed the size limit.");
        goto fail;
    }
    out->memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {sizeof(*out->component_offsets) * ((size_t)component_count + 1), (void **)&out->component_offsets},
            {sizeof(*out->values) * total_dofs * point_count, (void **)&out->values},
            {}});
    if (!out->memory)
        goto fail;
    out->component_offsets[0] = 0;
    for (unsigned component = 0; component < component_count; ++component)
    {
        size_t dof_count;
        if (constraint_kform_component_dof_count(&descriptor, component, &dof_count) != CONSTRAINT_SUCCESS)
        {
            PyErr_SetString(PyExc_OverflowError, "Trace basis DoF count exceeds the size limit.");
            goto fail;
        }
        out->component_offsets[component + 1] = out->component_offsets[component] + dof_count;
    }

    if (ndim > 0)
    {
        basis_sets = python_basis_sets_get(ndim, basis_specs, axis_rules, basis_registry->registry);
        if (!basis_sets)
            goto fail;
        if (order > 0)
        {
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                lower_specs[axis] = basis_specs[axis];
                lower_specs[axis].order -= 1;
            }
            basis_sets_lower = python_basis_sets_get(ndim, lower_specs, axis_rules, basis_registry->registry);
            if (!basis_sets_lower)
                goto fail;
        }
    }

    for (unsigned component = 0; component < component_count; ++component)
    {
        const size_t dof_count = out->component_offsets[component + 1] - out->component_offsets[component];
        combination_set_to_index((uint8_t)ndim, (uint8_t)order, component_axes, component);
        double *const component_values = out->values + out->component_offsets[component] * point_count;
        for (size_t point = 0; point < point_count; ++point)
        {
            double *const point_values = component_values + point * dof_count;
            point_values[0] = 1.0;
            size_t current_count = 1;
            unsigned component_axis = 0;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                const bool active = order > 0 && component_axis < order && component_axes[component_axis] == axis;
                if (active)
                    component_axis += 1;
                const basis_set_t *const basis = active ? basis_sets_lower[axis] : basis_sets[axis];
                const size_t basis_dim = (size_t)basis->spec.order + 1;
                const size_t integration_index = trace_basis_point_index(
                    element_dim, face_dim, orientation, axis, point, canonical_specs, point_strides, element_table);
                for (size_t previous = current_count; previous > 0; --previous)
                {
                    const double previous_value = point_values[previous - 1];
                    for (size_t basis_index = basis_dim; basis_index > 0; --basis_index)
                    {
                        point_values[(previous - 1) * basis_dim + basis_index - 1] =
                            previous_value *
                            basis_set_basis_values(basis, (unsigned)(basis_index - 1))[integration_index];
                    }
                }
                current_count *= basis_dim;
            }
            ASSERT(current_count == dof_count, "Trace basis DoF count mismatch (%zu vs %zu).", current_count,
                   dof_count);
        }
    }
    if (basis_sets_lower)
        python_basis_sets_release(ndim, basis_sets_lower, basis_registry->registry);
    if (basis_sets)
        python_basis_sets_release(ndim, basis_sets, basis_registry->registry);
    cutl_dealloc(&PYTHON_ALLOCATOR, memory);
    if (endpoint_rule)
        integration_rule_registry_release_rule(integration_registry->registry, endpoint_rule);
    out->descriptor = (constraint_trace_basis_values_t){
        .component_count = component_count,
        .point_count = point_count,
        .component_offsets = out->component_offsets,
        .values = out->values,
    };
    return 0;

fail:
    if (basis_sets_lower)
        python_basis_sets_release(ndim, basis_sets_lower, basis_registry->registry);
    if (basis_sets)
        python_basis_sets_release(ndim, basis_sets, basis_registry->registry);
    cutl_dealloc(&PYTHON_ALLOCATOR, memory);
    if (endpoint_rule)
        integration_rule_registry_release_rule(integration_registry->registry, endpoint_rule);
    release_trace_basis_table(out);
    return -1;
}

static void release_boundary_face_setup(const interplib_module_state_t *state, unsigned face_dim,
                                        boundary_face_setup_t *setup);
static int make_boundary_face_setup(const interplib_module_state_t *state, const space_map_object *element_map,
                                    const int8_t *orientation, const unsigned element_dim, const unsigned face_dim,
                                    boundary_face_setup_t *setup)
{
    *setup = (boundary_face_setup_t){};
    if (make_boundary_map(state, element_map, orientation, element_dim, face_dim, NULL, &setup->face_object) < 0)
        goto fail;
    space_map_object *const face_map = (space_map_object *)setup->face_object;
    if (make_integration_space(state, face_map, &setup->face_space) < 0)
        goto fail;
    const integration_spec_t *const face_specs = face_map->int_specs;
    if (face_dim > 0)
    {
        setup->memory = cutl_alloc_group(
            &PYTHON_ALLOCATOR,
            (const cutl_alloc_info_t[]){{sizeof(*setup->canonical_specs) * face_dim, (void **)&setup->canonical_specs},
                                        {sizeof(*setup->quadrature_axes) * face_dim, (void **)&setup->quadrature_axes},
                                        {}});
        if (!setup->memory)
            goto fail;
    }
    const unsigned fixed_count = element_dim - face_dim;
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = orientation[fixed_count + face_axis];
        const unsigned source_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        setup->canonical_specs[face_axis] =
            face_specs[get_source_face_axis(element_dim, face_dim, orientation, source_axis)];
    }
    setup->point_count = total_points(face_dim, setup->canonical_specs);
    integration_registry_object *const integration_registry =
        (integration_registry_object *)state->registry_integration;
    setup->rules = python_integration_rules_get(face_dim, face_specs, integration_registry->registry);
    if (!setup->rules)
        goto fail;
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = orientation[fixed_count + face_axis];
        const unsigned source_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        const unsigned source_face_axis = get_source_face_axis(element_dim, face_dim, orientation, source_axis);
        const integration_rule_t *const rule = setup->rules[source_face_axis];
        setup->quadrature_axes[face_axis] = (constraint_quadrature_t){
            .count = rule->n_nodes,
            .nodes = integration_rule_nodes_const(rule),
            .weights = integration_rule_weights_const(rule),
        };
    }
    return 0;
fail:
    release_boundary_face_setup(state, face_dim, setup);
    return -1;
}

static void release_boundary_face_setup(const interplib_module_state_t *state, const unsigned face_dim,
                                        boundary_face_setup_t *setup)
{
    if (setup->rules)
        python_integration_rules_release(face_dim, setup->rules,
                                         ((integration_registry_object *)state->registry_integration)->registry);
    Py_XDECREF(setup->face_space);
    Py_XDECREF(setup->face_object);
    cutl_dealloc(&PYTHON_ALLOCATOR, setup->memory);
    *setup = (boundary_face_setup_t){};
}

PyObject *compute_kform_boundary_constraints_impl(const interplib_module_state_t *state, kform_spec_object *test_spec,
                                                  kform_spec_object *element_spec, space_map_object *element_map,
                                                  const int8_t *orientation)
{
    const unsigned face_dim = Py_SIZE(test_spec->function_space);
    const unsigned order = test_spec->order;
    const unsigned element_dim = Py_SIZE(element_spec->function_space);

    boundary_face_setup_t setup;
    if (make_boundary_face_setup(state, element_map, orientation, element_dim, face_dim, &setup) < 0)
        return NULL;
    space_map_object *const face_map = (space_map_object *)setup.face_object;
    const integration_spec_t *const face_specs = face_map->int_specs;

    PyArrayObject *pullback = NULL;
    double *surface_weights = NULL;
    unsigned *canonical_digits = NULL;
    void *scratch_memory = NULL;
    trace_basis_table_t test_basis_table = {};
    trace_basis_table_t element_basis_table = {};
    PyArrayObject *row_array = NULL;
    PyArrayObject *component_array = NULL;
    PyArrayObject *dof_array = NULL;
    PyArrayObject *coefficient_array = NULL;
    constraint_entry_t *entries = NULL;

    if (create_pullback(face_map, element_dim, face_dim, orientation, order, face_specs, setup.canonical_specs,
                        &pullback) < 0)
        goto fail;

    const constraint_kform_spec_t test_descriptor = {
        .ndim = face_dim, .order = order, .basis_specs = test_spec->function_space->specs};
    const constraint_element_side_t side_descriptor = {
        .ndim = element_dim, .basis_specs = element_spec->function_space->specs, .orientation = orientation};
    const constraint_trace_pullback_t pullback_descriptor = {.physical_component_count = (unsigned)Py_SIZE(element_map),
                                                             .point_count = setup.point_count,
                                                             .values = PyArray_DATA(pullback)};
    const constraint_face_quadrature_t face_quadrature = {
        .ndim = face_dim, .axes = setup.quadrature_axes, .point_count = setup.point_count};
    size_t row_count;
    size_t entry_count;
    constraint_status_t constraint_status =
        constraint_physical_side_required(&test_descriptor, &side_descriptor, &row_count, &entry_count);
    if (constraint_status != CONSTRAINT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not size boundary constraints: %s.",
                     constraint_status_to_str(constraint_status));
        goto fail;
    }
    if (entry_count > SIZE_MAX / sizeof(*entries))
    {
        PyErr_SetString(PyExc_OverflowError, "Boundary constraint entries exceed the size limit.");
        goto fail;
    }
    scratch_memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){
                               {sizeof(*surface_weights) * setup.point_count, (void **)&surface_weights},
                               {sizeof(*canonical_digits) * (face_dim > 0 ? face_dim : 1), (void **)&canonical_digits},
                               {sizeof(*entries) * entry_count, (void **)&entries},
                               {}});
    if (!scratch_memory)
        goto fail;
    for (size_t point = 0; point < setup.point_count; ++point)
    {
        get_digits(face_dim, setup.canonical_specs, point, canonical_digits);
        const size_t source_point = canonical_point_to_source(element_dim, face_dim, orientation, face_specs,
                                                              setup.canonical_specs, canonical_digits);
        surface_weights[point] = fabs(face_map->determinant[source_point]);
    }

    integration_registry_object *const integration_registry =
        (integration_registry_object *)state->registry_integration;
    basis_registry_object *const basis_registry = (basis_registry_object *)state->registry_basis;
    if (make_trace_basis_table(element_dim, face_dim, order, test_spec->function_space->specs, orientation,
                               setup.canonical_specs, setup.rules, integration_registry, basis_registry, false,
                               setup.point_count, &test_basis_table) < 0)
        goto fail;
    if (make_trace_basis_table(element_dim, face_dim, order, element_spec->function_space->specs, orientation,
                               setup.canonical_specs, setup.rules, integration_registry, basis_registry, true,
                               setup.point_count, &element_basis_table) < 0)
        goto fail;

    const npy_intp row_dims[1] = {(npy_intp)(row_count + 1)};
    const npy_intp entry_dims[1] = {(npy_intp)entry_count};
    row_array = (PyArrayObject *)PyArray_SimpleNew(1, row_dims, NPY_UINTP);
    component_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT32);
    dof_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINTP);
    coefficient_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_DOUBLE);
    if (!row_array || !component_array || !dof_array || !coefficient_array)
        goto fail;
    size_t actual_rows;
    size_t actual_entries;
    constraint_status = constraint_physical_side_assemble_precomputed(
        &test_descriptor, &side_descriptor, &face_quadrature, surface_weights, &pullback_descriptor,
        &test_basis_table.descriptor, &element_basis_table.descriptor, row_count + 1, (size_t *)PyArray_DATA(row_array),
        entry_count, entries, &actual_rows, &actual_entries);
    if (constraint_status != CONSTRAINT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not assemble boundary constraints: %s.",
                     constraint_status_to_str(constraint_status));
        goto fail;
    }
    for (size_t i = 0; i < actual_entries; ++i)
    {
        ((uint32_t *)PyArray_DATA(component_array))[i] = entries[i].component;
        ((size_t *)PyArray_DATA(dof_array))[i] = entries[i].local_dof;
        ((double *)PyArray_DATA(coefficient_array))[i] = entries[i].coefficient;
    }
    cutl_dealloc(&PYTHON_ALLOCATOR, scratch_memory);
    Py_DECREF(pullback);
    release_trace_basis_table(&test_basis_table);
    release_trace_basis_table(&element_basis_table);
    release_boundary_face_setup(state, face_dim, &setup);
    {
        PyObject *result = PyTuple_New(4);
        if (!result)
        {
            Py_DECREF(row_array);
            Py_DECREF(component_array);
            Py_DECREF(dof_array);
            Py_DECREF(coefficient_array);
            return NULL;
        }
        PyTuple_SET_ITEM(result, 0, row_array);
        PyTuple_SET_ITEM(result, 1, component_array);
        PyTuple_SET_ITEM(result, 2, dof_array);
        PyTuple_SET_ITEM(result, 3, coefficient_array);
        return result;
    }

fail:
    Py_XDECREF(row_array);
    Py_XDECREF(component_array);
    Py_XDECREF(dof_array);
    Py_XDECREF(coefficient_array);
    cutl_dealloc(&PYTHON_ALLOCATOR, scratch_memory);
    release_trace_basis_table(&test_basis_table);
    release_trace_basis_table(&element_basis_table);
    Py_XDECREF(pullback);
    release_boundary_face_setup(state, face_dim, &setup);
    return NULL;
}
static PyObject *compute_kform_boundary_load(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                             const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;
    PyObject *test_object;
    PyObject *spec_object;
    PyObject *map_object;
    PyObject *collections_object;
    PyObject *data_object;
    Py_ssize_t npts;
    Py_ssize_t element_id;
    Py_ssize_t boundary_id;
    int weighted = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &test_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &spec_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &map_object, .type_check = state->space_mapping_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &collections_object},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &npts},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_id},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &boundary_id},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &data_object},
                {.type = CPYARG_TYPE_BOOL, .p_val = &weighted, .kwname = "surface_measure", .optional = 1},
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
    const unsigned component_count = combination_total_count((uint8_t)element_dim, (uint8_t)(order + 1));
    if (npts < 0 || element_id < 0 || boundary_id < 0 || face_dim != element_dim - 1 || element_spec->order != order ||
        element_map->ndim != element_dim)
    {
        PyErr_SetString(PyExc_ValueError, "Incompatible test, element, or topology dimensions: the load is defined on "
                                          "codimension-1 boundary faces.");
        return NULL;
    }
    if (!PyTuple_Check(collections_object) || PyTuple_GET_SIZE(collections_object) != element_dim)
    {
        PyErr_Format(PyExc_ValueError, "Expected %u mesh collections.", element_dim);
        return NULL;
    }
    PyObject **data_callables;
    void *const callables_memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){{sizeof(*data_callables) * component_count, (void **)&data_callables}, {}});
    if (!callables_memory)
        return NULL;
    PyObject *data_sequence = NULL;
    if (PyCallable_Check(data_object))
    {
        if (component_count != 1)
        {
            PyErr_Format(PyExc_ValueError,
                         "A k-form datum with %u components needs one callable per component; pass a sequence.",
                         component_count);
            cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
            return NULL;
        }
        data_callables[0] = data_object;
    }
    else
    {
        data_sequence = PySequence_Fast(
            data_object, "Boundary load data must be a callable or a sequence of callables, one per k-form component.");
        if (!data_sequence)
        {
            cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
            return NULL;
        }
        const Py_ssize_t sequence_length = PySequence_Fast_GET_SIZE(data_sequence);
        if (sequence_length != (Py_ssize_t)component_count)
        {
            PyErr_Format(PyExc_ValueError,
                         "Boundary load data must contain one callable per k-form component: expected %u, got %zd.",
                         component_count, sequence_length);
            Py_DECREF(data_sequence);
            cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
            return NULL;
        }
        PyObject **const data_items = PySequence_Fast_ITEMS(data_sequence);
        for (unsigned component = 0; component < component_count; ++component)
        {
            if (!PyCallable_Check(data_items[component]))
            {
                PyErr_Format(PyExc_TypeError, "Boundary load data component %u is not callable.", component);
                Py_DECREF(data_sequence);
                cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
                return NULL;
            }
            data_callables[component] = data_items[component];
        }
        Py_DECREF(data_sequence);
    }

    boundary_topology_t topology;
    if (make_boundary_topology(collections_object, element_dim, (unsigned)npts, &topology) < 0)
    {
        cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
        return NULL;
    }
    const unsigned boundary_immersion_index = face_dim;
    const topo_status_t topo_status =
        topo_obj_boundary_orientation(topology.immersions + boundary_immersion_index, element_dim,
                                      (uint64_t)boundary_id, (uint64_t)element_id, topology.orientation);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Boundary %zd is not present in element %zd: %s (%s).", boundary_id, element_id,
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        release_boundary_topology(element_dim, &topology);
        cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
        return NULL;
    }

    const int8_t *const orientation = topology.orientation;
    boundary_face_setup_t setup;
    if (make_boundary_face_setup(state, element_map, topology.orientation, element_dim, face_dim, &setup) < 0)
    {
        release_boundary_topology(element_dim, &topology);
        cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
        return NULL;
    }
    PyArrayObject *data_array = NULL;
    PyObject *result = NULL;
    double *data_owned = NULL;
    double *surface_weights = NULL;
    unsigned *canonical_digits = NULL;
    void *load_memory = NULL;

    space_map_object *const face_map = (space_map_object *)setup.face_object;
    const integration_spec_t *const face_specs = face_map->int_specs;
    if (component_count > SIZE_MAX / setup.point_count ||
        component_count * setup.point_count > SIZE_MAX / sizeof(*data_owned))
    {
        PyErr_SetString(PyExc_OverflowError, "Boundary load data exceeds the size limit.");
        goto load_fail;
    }
    load_memory =
        cutl_alloc_group(&PYTHON_ALLOCATOR,
                         (const cutl_alloc_info_t[]){
                             {sizeof(*surface_weights) * (weighted ? setup.point_count : 1), (void **)&surface_weights},
                             {sizeof(*canonical_digits) * (face_dim > 0 ? face_dim : 1), (void **)&canonical_digits},
                             {sizeof(*data_owned) * component_count * setup.point_count, (void **)&data_owned},
                             {}});
    if (!load_memory)
        goto load_fail;

    if (weighted)
    {
        for (size_t point = 0; point < setup.point_count; ++point)
        {
            get_digits(face_dim, setup.canonical_specs, point, canonical_digits);
            const size_t source_point = canonical_point_to_source(element_dim, face_dim, orientation, face_specs,
                                                                  setup.canonical_specs, canonical_digits);
            surface_weights[point] = fabs(face_map->determinant[source_point]);
        }
    }
    PyObject *coords_tuple = PyTuple_New(element_dim);
    if (!coords_tuple)
        goto load_fail;
    for (unsigned idim = 0; idim < element_dim; ++idim)
    {
        const npy_intp dims[1] = {(npy_intp)setup.point_count};
        PyArrayObject *coord_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!coord_array)
        {
            Py_DECREF(coords_tuple);
            goto load_fail;
        }
        const double *const values = coordinate_map_values(face_map->maps[idim]);
        double *const out = (double *)PyArray_DATA(coord_array);
        for (size_t point = 0; point < setup.point_count; ++point)
        {
            get_digits(face_dim, setup.canonical_specs, point, canonical_digits);
            const size_t source_point = canonical_point_to_source(element_dim, face_dim, orientation, face_specs,
                                                                  setup.canonical_specs, canonical_digits);
            out[point] = values[source_point];
        }
        PyTuple_SET_ITEM(coords_tuple, idim, (PyObject *)coord_array);
    }
    for (unsigned component = 0; component < component_count; ++component)
    {
        PyObject *const data_result = PyObject_CallObject(data_callables[component], coords_tuple);
        if (!data_result)
        {
            Py_DECREF(coords_tuple);
            goto load_fail;
        }
        data_array = (PyArrayObject *)PyArray_FROMANY(data_result, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
        Py_DECREF(data_result);
        if (!data_array)
        {
            Py_DECREF(coords_tuple);
            goto load_fail;
        }
        double *const datum_row = data_owned + (size_t)component * setup.point_count;
        if (PyArray_NDIM(data_array) == 0 && PyArray_SIZE(data_array) == 1)
        {
            const double value = *(const double *)PyArray_DATA(data_array);
            for (size_t i = 0; i < setup.point_count; ++i)
                datum_row[i] = value;
        }
        else if (PyArray_NDIM(data_array) == 1 && PyArray_SIZE(data_array) == (npy_intp)setup.point_count)
        {
            memcpy(datum_row, PyArray_DATA(data_array), setup.point_count * sizeof(*datum_row));
        }
        else
        {
            PyErr_Format(PyExc_ValueError, "Boundary load data component %u must return an array of %zu values.",
                         component, setup.point_count);
            Py_DECREF(coords_tuple);
            goto load_fail;
        }
        Py_CLEAR(data_array);
    }
    Py_DECREF(coords_tuple);

    const constraint_kform_spec_t test_descriptor = {
        .ndim = face_dim, .order = order, .basis_specs = test_spec->function_space->specs};
    const constraint_element_side_t side_descriptor = {
        .ndim = element_dim, .basis_specs = element_spec->function_space->specs, .orientation = orientation};
    const constraint_face_quadrature_t face_quadrature = {
        .ndim = face_dim, .axes = setup.quadrature_axes, .point_count = setup.point_count};
    const constraint_kform_spec_t element_descriptor = {
        .ndim = element_dim, .order = order, .basis_specs = element_spec->function_space->specs};
    size_t element_component_count;
    constraint_status_t constraint_status =
        constraint_kform_component_count(&element_descriptor, &element_component_count);
    if (constraint_status != CONSTRAINT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not size the boundary load: %s.",
                     constraint_status_to_str(constraint_status));
        goto load_fail;
    }
    size_t value_count = 0;
    for (unsigned component = 0; component < element_component_count; ++component)
    {
        size_t component_dofs;
        constraint_status = constraint_kform_component_dof_count(&element_descriptor, component, &component_dofs);
        if (constraint_status != CONSTRAINT_SUCCESS || value_count > SIZE_MAX - component_dofs)
        {
            PyErr_Format(PyExc_ValueError, "Could not size the boundary load: %s.",
                         constraint_status_to_str(constraint_status));
            goto load_fail;
        }
        value_count += component_dofs;
    }
    const npy_intp value_dims[1] = {(npy_intp)value_count};
    result = (PyObject *)PyArray_ZEROS(1, value_dims, NPY_DOUBLE, 0);
    if (!result)
        goto load_fail;
    constraint_status = constraint_physical_side_load(&test_descriptor, &side_descriptor, &face_quadrature, data_owned,
                                                      value_count, weighted ? surface_weights : NULL,
                                                      (double *)PyArray_DATA((PyArrayObject *)result));
    if (constraint_status != CONSTRAINT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not assemble boundary load: %s.",
                     constraint_status_to_str(constraint_status));
        Py_DECREF(result);
        result = NULL;
        goto load_fail;
    }
    cutl_dealloc(&PYTHON_ALLOCATOR, load_memory);
    release_boundary_face_setup(state, face_dim, &setup);
    release_boundary_topology(element_dim, &topology);
    cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
    return result;

load_fail:
    Py_XDECREF(data_array);
    Py_XDECREF(result);
    cutl_dealloc(&PYTHON_ALLOCATOR, load_memory);
    release_boundary_face_setup(state, face_dim, &setup);
    release_boundary_topology(element_dim, &topology);
    cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
    return NULL;
}

PyMethodDef constraint_methods[] = {
    {
        .ml_name = "compute_kform_boundary_constraints",
        .ml_meth = (void *)compute_kform_boundary_constraints,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_kform_boundary_constraints(test_specs, element_spec, element_map, collections, npts, "
                  "element_id, boundary_id) -> tuple[numpy.ndarray, ...]\nCompute one element's physical k-form "
                  "boundary rows.",
    },
    {
        .ml_name = "compute_kform_boundary_load",
        .ml_meth = (void *)compute_kform_boundary_load,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_kform_boundary_load(test_specs, element_spec, element_map, collections, npts, "
                  "element_id, boundary_id, data, surface_measure=False) -> numpy.ndarray\nCompute the boundary "
                  "load of one element face: the pairing of the trace of the element (k-1)-form basis against "
                  "the components of a k-form datum, where k = element_spec.order + 1. Provide one callable "
                  "per element-frame k-form component (each called with the physical coordinates of the "
                  "canonical face points); a bare callable is accepted when k equals the element dimension. "
                  "With surface_measure=True the data is integrated with the mapped face Jacobian (physical "
                  "surface measure); otherwise the metric-free chain integral is assembled.",
    },
    {},
};
