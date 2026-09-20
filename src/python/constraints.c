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

typedef struct
{
    PyObject *face_object;                      // Restricted space map (owned reference).
    space_map_object *face_map;                 // Borrowed alias of face_object.
    const integration_rule_t **source_rules;    // Rules of the source-frame face axes.
    const integration_rule_t **canonical_rules; // Rules permuted to canonical test axes.
    integration_spec_t *canonical_specs;        // Canonical axis specs.
    size_t *canonical_strides;                  // Canonical row-major point strides.
    size_t *source_strides;                     // Source-frame row-major point strides.
    double *point_weights;                      // Canonical tensor quadrature weights.
    size_t point_count;                         // Total canonical face points.
    void *memory;
} boundary_face_setup_t;

static void release_boundary_face_setup(const interplib_module_state_t *state, unsigned face_dim,
                                        boundary_face_setup_t *setup);
static int make_boundary_face_setup(const interplib_module_state_t *state, const space_map_object *element_map,
                                    const int8_t *orientation, const unsigned element_dim, const unsigned face_dim,
                                    boundary_face_setup_t *setup)
{
    *setup = (boundary_face_setup_t){};
    // Restrict the volume map to the face in one values-level pass: the orientation
    // prefix holds the fixed normal axes, the tail the surviving face axes.
    const unsigned fixed_count = element_dim - face_dim;
    space_map_object *const face = space_map_boundary_oriented_impl(state, element_map, fixed_count, orientation);
    if (!face)
    {
        goto fail;
    }
    setup->face_object = (PyObject *)face;
    setup->face_map = face;

    const integration_spec_t *const face_specs = setup->face_map->int_specs;
    // The canonical specs determine the tensor point count, which sizes the
    // weight buffer; derive them into stack scratch before allocating.
    integration_spec_t canonical_specs_scratch[UINT8_MAX];
    constraint_face_canonical_specs(element_dim, face_dim, orientation, face_specs, canonical_specs_scratch);
    const size_t slot_count = face_dim > 0 ? face_dim : 1;
    const size_t weight_count = integration_specs_total_points(face_dim, canonical_specs_scratch);
    setup->memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR, (const cutl_alloc_info_t[]){
                               {sizeof(*setup->source_rules) * slot_count, (void **)&setup->source_rules},
                               {sizeof(*setup->canonical_rules) * slot_count, (void **)&setup->canonical_rules},
                               {sizeof(*setup->canonical_specs) * slot_count, (void **)&setup->canonical_specs},
                               {sizeof(*setup->canonical_strides) * slot_count, (void **)&setup->canonical_strides},
                               {sizeof(*setup->source_strides) * slot_count, (void **)&setup->source_strides},
                               {sizeof(*setup->point_weights) * weight_count, (void **)&setup->point_weights},
                               {}});
    if (!setup->memory)
        goto fail;
    memcpy(setup->canonical_specs, canonical_specs_scratch, sizeof(*setup->canonical_specs) * slot_count);
    memset(setup->source_rules, 0, sizeof(*setup->source_rules) * slot_count);
    memset(setup->canonical_rules, 0, sizeof(*setup->canonical_rules) * slot_count);
    setup->point_count = weight_count;

    integration_registry_object *const integration_registry =
        (integration_registry_object *)state->registry_integration;
    setup->source_rules = python_integration_rules_get(face_dim, face_specs, integration_registry->registry);
    if (!setup->source_rules)
        goto fail;
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = orientation[fixed_count + face_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        const unsigned source_axis = constraint_face_source_axis(element_dim, face_dim, orientation, element_axis);
        setup->canonical_rules[face_axis] = setup->source_rules[source_axis];
    }
    integration_spec_point_strides(face_dim, setup->canonical_specs, setup->canonical_strides);
    integration_spec_point_strides(face_dim, face_specs, setup->source_strides);
    integration_rule_tensor_weights(face_dim, setup->canonical_rules, setup->point_weights);
    return 0;
fail:
    release_boundary_face_setup(state, face_dim, setup);
    return -1;
}

static void release_boundary_face_setup(const interplib_module_state_t *state, const unsigned face_dim,
                                        boundary_face_setup_t *setup)
{
    if (setup->source_rules)
        python_integration_rules_release(face_dim, setup->source_rules,
                                         ((integration_registry_object *)state->registry_integration)->registry);
    Py_XDECREF(setup->face_object);
    cutl_dealloc(&PYTHON_ALLOCATOR, setup->memory);
    *setup = (boundary_face_setup_t){};
}

/**
 * @brief Precomputed tensor-product trace basis values with owned storage.
 */
typedef struct
{
    kform_values_table_t descriptor;
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

/**
 * @brief Build one trace basis table on the canonical face points.
 *
 * The test table evaluates the face test space directly in the canonical
 * frame; the element table evaluates the element trace bases with fixed axes
 * at their signed endpoints and negative orientations reading mirrored node
 * indices (exact for the symmetric Gauss rules used by this library).
 */
static int make_trace_basis_table(const unsigned element_dim, const unsigned face_dim, const unsigned order,
                                  const basis_spec_t *basis_specs, const int8_t *orientation,
                                  const integration_spec_t *canonical_specs, const integration_rule_t **canonical_rules,
                                  const size_t *canonical_strides, basis_registry_object *const basis_registry,
                                  const bool element_table, const size_t point_count, trace_basis_table_t *const out)
{
    *out = (trace_basis_table_t){};
    const unsigned ndim = element_table ? element_dim : face_dim;
    const unsigned component_count = combination_total_count((uint8_t)ndim, (uint8_t)order);
    const unsigned free_count = face_dim;
    const unsigned axis_count = ndim > 0 ? ndim : 1;
    kform_trace_axis_t *axes;
    const basis_set_t **basis_sets = NULL;
    const basis_set_t **basis_sets_lower = NULL;
    const basis_endpoint_set_t **endpoint_sets = NULL;
    const basis_endpoint_set_t **endpoint_sets_lower = NULL;
    basis_spec_t *free_specs;
    basis_spec_t *free_specs_lower;
    basis_spec_t *lower_specs;
    unsigned *source_axes;
    void *const memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {sizeof(*axes) * axis_count, (void **)&axes},
            {sizeof(*free_specs) * (free_count > 0 ? free_count : 1), (void **)&free_specs},
            {sizeof(*free_specs_lower) * (free_count > 0 ? free_count : 1), (void **)&free_specs_lower},
            {sizeof(*lower_specs) * axis_count, (void **)&lower_specs},
            {sizeof(*source_axes) * axis_count, (void **)&source_axes},
            {}});
    if (!memory)
        return -1;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        axes[axis] = (kform_trace_axis_t){};
        source_axes[axis] = face_dim;
        if (!element_table)
        {
            source_axes[axis] = axis;
            free_specs[axis] = basis_specs[axis];
        }
    }
    if (face_dim > 0)
    {
        const unsigned fixed_count = element_dim - face_dim;
        for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
        {
            const int8_t mapping = orientation[fixed_count + face_axis];
            const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
            if (element_table)
            {
                free_specs[face_axis] = basis_specs[element_axis];
                source_axes[element_axis] = face_axis;
            }
        }
    }
    if (order > 0)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            lower_specs[axis] = basis_specs[axis];
            // Order-zero axes cannot lose another degree; no component reads
            // their lowered table because the matching components have no DoFs.
            if (lower_specs[axis].order > 0)
                lower_specs[axis].order -= 1;
        }
        for (unsigned axis = 0; axis < free_count; ++axis)
        {
            free_specs_lower[axis] = free_specs[axis];
            if (free_specs_lower[axis].order > 0)
                free_specs_lower[axis].order -= 1;
        }
    }

    if (ndim > 0 && element_table)
    {
        endpoint_sets = python_basis_endpoints_get(ndim, basis_specs, basis_registry->registry);
        if (!endpoint_sets)
            goto fail;
        if (order > 0)
        {
            endpoint_sets_lower = python_basis_endpoints_get(ndim, lower_specs, basis_registry->registry);
            if (!endpoint_sets_lower)
                goto fail;
        }
    }
    if (free_count > 0)
    {
        basis_sets = python_basis_sets_get(free_count, free_specs, canonical_rules, basis_registry->registry);
        if (!basis_sets)
            goto fail;
        if (order > 0)
        {
            basis_sets_lower =
                python_basis_sets_get(free_count, free_specs_lower, canonical_rules, basis_registry->registry);
            if (!basis_sets_lower)
                goto fail;
        }
    }

    // Describe every axis of the table: fixed normal axes read endpoint
    // values; free axes read the canonical rule nodes, mirrored when the
    // orientation reverses the axis.
    if (element_table)
    {
        const unsigned fixed_count = element_dim - face_dim;
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            const unsigned slot = source_axes[axis];
            if (slot == face_dim)
            {
                unsigned fixed_axis = 0;
                for (; fixed_axis < fixed_count; ++fixed_axis)
                {
                    if ((unsigned)(orientation[fixed_axis] < 0 ? -orientation[fixed_axis] : orientation[fixed_axis]) -
                            1 ==
                        axis)
                        break;
                }
                ASSERT(fixed_axis < fixed_count, "Axis is neither fixed nor free.");
                axes[axis] = (kform_trace_axis_t){
                    .endpoint = endpoint_sets[axis],
                    .endpoint_lower = order > 0 ? endpoint_sets_lower[axis] : NULL,
                    .end = orientation[fixed_axis] < 0 ? 0u : 1u,
                };
            }
            else
            {
                const int8_t mapping = orientation[fixed_count + slot];
                axes[axis] = (kform_trace_axis_t){
                    .nodes = basis_sets[slot],
                    .nodes_lower = order > 0 ? basis_sets_lower[slot] : NULL,
                    .rule_size = canonical_specs[slot].order + 1,
                    .stride_slot = slot,
                    .mirror = mapping < 0,
                };
            }
        }
    }
    else
    {
        for (unsigned axis = 0; axis < face_dim; ++axis)
        {
            axes[axis] = (kform_trace_axis_t){
                .nodes = basis_sets[axis],
                .nodes_lower = order > 0 ? basis_sets_lower[axis] : NULL,
                .rule_size = canonical_specs[axis].order + 1,
                .stride_slot = axis,
                .mirror = 0,
            };
        }
    }

    const kform_spec_t descriptor = {.ndim = ndim, .order = order, .basis = basis_specs};
    const size_t total_dofs = kform_spec_total_dofs(&descriptor);
    out->memory = cutl_alloc_group(
        &PYTHON_ALLOCATOR,
        (const cutl_alloc_info_t[]){
            {sizeof(*out->component_offsets) * ((size_t)component_count + 1), (void **)&out->component_offsets},
            {sizeof(*out->values) * total_dofs * point_count, (void **)&out->values},
            {}});
    if (!out->memory)
        goto fail;
    kform_spec_component_offsets(&descriptor, component_count + 1, out->component_offsets);
    out->descriptor = (kform_values_table_t){.component_count = component_count,
                                             .point_count = point_count,
                                             .component_offsets = out->component_offsets,
                                             .values = out->values};
    for (unsigned component = 0; component < component_count; ++component)
    {
        const size_t dof_count = out->component_offsets[component + 1] - out->component_offsets[component];
        if (dof_count == 0)
            continue;
        uint8_t component_axes[UINT8_MAX];
        kform_component_axes(&descriptor, component, component_axes);
        kform_component_basis_values(ndim, basis_specs, order, component_axes, axes, canonical_strides, point_count,
                                     out->values + out->component_offsets[component] * point_count);
    }
    if (basis_sets_lower)
        python_basis_sets_release(free_count, basis_sets_lower, basis_registry->registry);
    if (basis_sets)
        python_basis_sets_release(free_count, basis_sets, basis_registry->registry);
    if (endpoint_sets_lower)
        python_basis_endpoints_release(ndim, endpoint_sets_lower, basis_registry->registry);
    if (endpoint_sets)
        python_basis_endpoints_release(ndim, endpoint_sets, basis_registry->registry);
    cutl_dealloc(&PYTHON_ALLOCATOR, memory);
    return 0;

fail:
    if (basis_sets_lower)
        python_basis_sets_release(free_count, basis_sets_lower, basis_registry->registry);
    if (basis_sets)
        python_basis_sets_release(free_count, basis_sets, basis_registry->registry);
    if (endpoint_sets_lower)
        python_basis_endpoints_release(ndim, endpoint_sets_lower, basis_registry->registry);
    if (endpoint_sets)
        python_basis_endpoints_release(ndim, endpoint_sets, basis_registry->registry);
    cutl_dealloc(&PYTHON_ALLOCATOR, memory);
    release_trace_basis_table(out);
    return -1;
}

static PyObject *packed_kform_constraints_to_csr(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                                 const PyObject *kwnames)
{
    const interplib_module_state_t *const state = PyModule_GetState(module);
    if (!state)
        return NULL;

    PyObject *packed_object;
    PyObject *spec_object;
    Py_ssize_t element_count;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &packed_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &spec_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &element_count},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (element_count < 0)
    {
        PyErr_SetString(PyExc_ValueError, "element_count must be non-negative.");
        return NULL;
    }
    if (!PyTuple_Check(packed_object) || PyTuple_GET_SIZE(packed_object) != 5)
    {
        PyErr_SetString(PyExc_TypeError, "packed must contain five constraint arrays.");
        return NULL;
    }

    PyArrayObject *arrays[5];
    const int expected_types[] = {NPY_UINTP, NPY_UINT64, NPY_UINT32, NPY_UINTP, NPY_DOUBLE};
    const char *const names[] = {"row_offsets", "element_ids", "components", "local_dofs", "coefficients"};
    for (unsigned index = 0; index < 5; ++index)
    {
        PyObject *const object = PyTuple_GET_ITEM(packed_object, index);
        if (!PyArray_Check(object) || PyArray_NDIM((PyArrayObject *)object) != 1 ||
            PyArray_TYPE((PyArrayObject *)object) != expected_types[index])
        {
            PyErr_Format(PyExc_TypeError, "packed[%u] must be a one-dimensional %s array.", index, names[index]);
            return NULL;
        }
        arrays[index] = (PyArrayObject *)object;
    }

    const size_t offset_count = (size_t)PyArray_SIZE(arrays[0]);
    if (offset_count == 0)
    {
        PyErr_SetString(PyExc_ValueError, "packed row_offsets must contain at least one entry.");
        return NULL;
    }
    const size_t entry_count = (size_t)PyArray_SIZE(arrays[1]);
    for (unsigned index = 2; index < 5; ++index)
    {
        if ((size_t)PyArray_SIZE(arrays[index]) != entry_count)
        {
            PyErr_SetString(PyExc_ValueError, "packed entry arrays must have equal lengths.");
            return NULL;
        }
    }
    const size_t first_offset = *(const npy_uintp *)PyArray_GETPTR1(arrays[0], 0);
    if (first_offset != 0)
    {
        PyErr_SetString(PyExc_ValueError, "packed row_offsets must start at zero.");
        return NULL;
    }
    size_t previous_offset = first_offset;
    for (size_t row = 1; row < offset_count; ++row)
    {
        const size_t offset = *(const npy_uintp *)PyArray_GETPTR1(arrays[0], (npy_intp)row);
        if (previous_offset > offset)
        {
            PyErr_SetString(PyExc_ValueError, "packed row_offsets must be non-decreasing.");
            return NULL;
        }
        previous_offset = offset;
    }
    if (previous_offset != entry_count)
    {
        PyErr_SetString(PyExc_ValueError, "packed row_offsets must end at the entry count.");
        return NULL;
    }

    const kform_spec_object *const specs = (const kform_spec_object *)spec_object;
    const unsigned component_count =
        combination_total_count((uint8_t)Py_SIZE(specs->function_space), (uint8_t)specs->order);
    const size_t dofs_per_element = specs->component_offsets[component_count];
    const size_t element_count_size = (size_t)element_count;

    const npy_intp entry_dims[1] = {(npy_intp)entry_count};
    PyArrayObject *const column_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_INTP);
    if (!column_array)
        return NULL;
    npy_intp *const columns = PyArray_DATA(column_array);
    for (size_t entry = 0; entry < entry_count; ++entry)
    {
        const npy_uint64 element_id = *(const npy_uint64 *)PyArray_GETPTR1(arrays[1], (npy_intp)entry);
        const npy_uint32 component = *(const npy_uint32 *)PyArray_GETPTR1(arrays[2], (npy_intp)entry);
        const npy_uintp local_dof = *(const npy_uintp *)PyArray_GETPTR1(arrays[3], (npy_intp)entry);
        if (element_id >= element_count_size || component >= component_count)
        {
            PyErr_SetString(PyExc_ValueError, "packed constraint entries reference an invalid element or component.");
            Py_DECREF(column_array);
            return NULL;
        }
        const size_t component_start = specs->component_offsets[component];
        const size_t component_end = specs->component_offsets[component + 1];
        if ((size_t)local_dof >= component_end - component_start)
        {
            PyErr_SetString(PyExc_ValueError, "packed constraint entries reference an invalid local DoF.");
            Py_DECREF(column_array);
            return NULL;
        }
        const size_t column = (size_t)element_id * dofs_per_element + component_start + local_dof;
        columns[entry] = (npy_intp)column;
    }

    PyObject *const result = PyTuple_New(3);
    if (!result)
    {
        Py_DECREF(column_array);
        return NULL;
    }
    Py_INCREF(arrays[4]);
    PyTuple_SET_ITEM(result, 0, (PyObject *)arrays[4]);
    PyTuple_SET_ITEM(result, 1, (PyObject *)column_array);
    Py_INCREF(arrays[0]);
    PyTuple_SET_ITEM(result, 2, (PyObject *)arrays[0]);
    return result;
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
    space_map_object *const face_map = setup.face_map;
    const integration_spec_t *const face_specs = face_map->int_specs;

    PyArrayObject *transform = NULL;
    double *pullback_values = NULL;
    double *surface_weights = NULL;
    trace_basis_table_t test_table = {};
    trace_basis_table_t element_table = {};
    PyArrayObject *row_array = NULL;
    PyArrayObject *component_array = NULL;
    PyArrayObject *dof_array = NULL;
    PyArrayObject *coefficient_array = NULL;

    const unsigned physical_components = combination_total_count((uint8_t)Py_SIZE(element_map), (uint8_t)order);
    const unsigned element_components = combination_total_count((uint8_t)element_dim, (uint8_t)order);
    if (order != 0)
    {
        transform = compute_basis_transform_impl(face_map, order);
        if (!transform)
            goto fail;
    }
    pullback_values =
        PyMem_Malloc(sizeof(*pullback_values) * (size_t)element_components * physical_components * setup.point_count);
    if (!pullback_values)
        goto fail;
    const constraint_trace_pullback_build_t build = {
        .element_dim = element_dim,
        .face_dim = face_dim,
        .order = order,
        .face_component_count = combination_total_count((uint8_t)face_dim, (uint8_t)order),
        .physical_component_count = physical_components,
        .source_point_count = integration_specs_total_points(face_dim, face_specs),
        .canonical_point_count = setup.point_count,
        .source_strides = setup.source_strides,
        .canonical_strides = setup.canonical_strides,
        .orientation = orientation,
        .source_specs = face_specs,
        .canonical_specs = setup.canonical_specs,
        .transform = transform ? (const double *)PyArray_DATA(transform) : NULL,
        .out = pullback_values,
    };
    constraint_trace_pullback_build(&build);
    Py_CLEAR(transform);

    // The surface measure sampled at the canonical face points.
    surface_weights = PyMem_Malloc(sizeof(*surface_weights) * setup.point_count);
    if (!surface_weights)
        goto fail;
    for (size_t point = 0; point < setup.point_count; ++point)
    {
        const size_t source_point =
            constraint_face_point_to_source(element_dim, face_dim, orientation, face_specs, setup.canonical_specs,
                                            setup.canonical_strides, setup.source_strides, point);
        surface_weights[point] = fabs(face_map->determinant[source_point]);
    }

    basis_registry_object *const basis_registry = (basis_registry_object *)state->registry_basis;
    if (make_trace_basis_table(element_dim, face_dim, order, test_spec->function_space->specs, orientation,
                               setup.canonical_specs, setup.canonical_rules, setup.canonical_strides, basis_registry,
                               false, setup.point_count, &test_table) < 0)
        goto fail;
    if (make_trace_basis_table(element_dim, face_dim, order, element_spec->function_space->specs, orientation,
                               setup.canonical_specs, setup.canonical_rules, setup.canonical_strides, basis_registry,
                               true, setup.point_count, &element_table) < 0)
        goto fail;

    const kform_spec_t test_descriptor = {.ndim = face_dim, .order = order, .basis = test_spec->function_space->specs};
    const constraint_element_side_t side_descriptor = {
        .ndim = element_dim, .basis_specs = element_spec->function_space->specs, .orientation = orientation};
    const constraint_trace_pullback_t pullback_descriptor = {
        .physical_component_count = physical_components, .point_count = setup.point_count, .values = pullback_values};
    const constraint_assembly_inputs_t inputs = {.point_weights = setup.point_weights,
                                                 .surface_weights = surface_weights,
                                                 .test_table = &test_table.descriptor,
                                                 .element_table = &element_table.descriptor,
                                                 .pullback = &pullback_descriptor};
    size_t row_count;
    size_t entry_count;
    constraint_physical_side_layout(&test_descriptor, &side_descriptor, &row_count, &entry_count);
    const npy_intp row_dims[1] = {(npy_intp)(row_count + 1)};
    const npy_intp entry_dims[1] = {(npy_intp)entry_count};
    row_array = (PyArrayObject *)PyArray_SimpleNew(1, row_dims, NPY_UINTP);
    component_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT32);
    dof_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINTP);
    coefficient_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_DOUBLE);
    if (!row_array || !component_array || !dof_array || !coefficient_array)
        goto fail;
    constraint_physical_side_assemble(&test_descriptor, &side_descriptor, &inputs,
                                      (uint32_t *)PyArray_DATA(component_array), (size_t *)PyArray_DATA(dof_array),
                                      (double *)PyArray_DATA(coefficient_array), (size_t *)PyArray_DATA(row_array));

    PyMem_Free(surface_weights);
    release_trace_basis_table(&test_table);
    release_trace_basis_table(&element_table);
    PyMem_Free(pullback_values);
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
    Py_XDECREF(transform);
    PyMem_Free(surface_weights);
    release_trace_basis_table(&test_table);
    release_trace_basis_table(&element_table);
    PyMem_Free(pullback_values);
    release_boundary_face_setup(state, face_dim, &setup);
    return NULL;
}

/** Packs the five reference constraint arrays into the returned tuple. */
static PyObject *reference_result_tuple(PyArrayObject *const row_array, PyArrayObject *const side_array,
                                        PyArrayObject *const component_array, PyArrayObject *const dof_array,
                                        PyArrayObject *const coefficient_array)
{
    PyObject *result = PyTuple_New(5);
    if (!result)
    {
        Py_DECREF(row_array);
        Py_DECREF(side_array);
        Py_DECREF(component_array);
        Py_DECREF(dof_array);
        Py_DECREF(coefficient_array);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, (PyObject *)row_array);
    PyTuple_SET_ITEM(result, 1, (PyObject *)side_array);
    PyTuple_SET_ITEM(result, 2, (PyObject *)component_array);
    PyTuple_SET_ITEM(result, 3, (PyObject *)dof_array);
    PyTuple_SET_ITEM(result, 4, (PyObject *)coefficient_array);
    return result;
}

PyObject *compute_kform_reference_constraints_impl(const interplib_module_state_t *state, kform_spec_object *test_spec,
                                                   kform_spec_object *element_spec_1, const int8_t *orientation_1,
                                                   kform_spec_object *element_spec_2, const int8_t *orientation_2)
{
    const unsigned face_dim = Py_SIZE(test_spec->function_space);
    const unsigned order = test_spec->order;
    const unsigned element_dim_1 = Py_SIZE(element_spec_1->function_space);
    const unsigned element_dim_2 = Py_SIZE(element_spec_2->function_space);

    PyArrayObject *row_array = NULL;
    PyArrayObject *side_array = NULL;
    PyArrayObject *component_array = NULL;
    PyArrayObject *dof_array = NULL;
    PyArrayObject *coefficient_array = NULL;
    integration_spec_t *rule_specs = NULL;
    const integration_rule_t **rules = NULL;
    double *point_weights = NULL;
    trace_basis_table_t test_table = {};
    trace_basis_table_t element_table_1 = {};
    trace_basis_table_t element_table_2 = {};

    if (element_dim_1 != element_dim_2 || element_dim_1 <= face_dim)
    {
        PyErr_SetString(PyExc_ValueError, "Both element sides must share the boundary's element dimension.");
        goto fail;
    }

    const kform_spec_t test_descriptor = {.ndim = face_dim, .order = order, .basis = test_spec->function_space->specs};
    const constraint_element_side_t sides[2] = {
        {.ndim = element_dim_1, .basis_specs = element_spec_1->function_space->specs, .orientation = orientation_1},
        {.ndim = element_dim_2, .basis_specs = element_spec_2->function_space->specs, .orientation = orientation_2},
    };

    // Per face axis, a Gauss-Legendre rule chosen exact for the pairing
    // integrand: the trace factors reach the test order plus the larger
    // element order on the mapped axis, with one spare degree for the
    // inactive-axis basis shifts.
    const size_t slot_count = face_dim > 0 ? face_dim : 1;
    rule_specs = PyMem_Malloc(slot_count * sizeof(*rule_specs));
    if (!rule_specs)
        goto fail;
    constraint_reference_rule_specs(&test_descriptor, sides, rule_specs);
    const size_t point_count = integration_specs_total_points(face_dim, rule_specs);
    point_weights = PyMem_Malloc(point_count * sizeof(*point_weights));
    if (!point_weights)
        goto fail;
    if (face_dim > 0)
    {
        rules = python_integration_rules_get(face_dim, rule_specs,
                                             ((integration_registry_object *)state->registry_integration)->registry);
        if (!rules)
            goto fail;
        integration_rule_tensor_weights(face_dim, rules, point_weights);
    }
    else
    {
        point_weights[0] = 1.0;
    }
    basis_registry_object *const basis_registry = (basis_registry_object *)state->registry_basis;
    size_t *reference_strides = PyMem_Malloc(slot_count * sizeof(*reference_strides));
    if (!reference_strides)
        goto fail;
    integration_spec_point_strides(face_dim, rule_specs, reference_strides);
    if (make_trace_basis_table(element_dim_1, face_dim, order, test_spec->function_space->specs, orientation_1,
                               rule_specs, rules, reference_strides, basis_registry, false, point_count,
                               &test_table) < 0)
    {
        PyMem_Free(reference_strides);
        goto fail;
    }
    if (make_trace_basis_table(element_dim_1, face_dim, order, element_spec_1->function_space->specs, orientation_1,
                               rule_specs, rules, reference_strides, basis_registry, true, point_count,
                               &element_table_1) < 0)
    {
        PyMem_Free(reference_strides);
        goto fail;
    }
    if (make_trace_basis_table(element_dim_2, face_dim, order, element_spec_2->function_space->specs, orientation_2,
                               rule_specs, rules, reference_strides, basis_registry, true, point_count,
                               &element_table_2) < 0)
    {
        PyMem_Free(reference_strides);
        goto fail;
    }
    PyMem_Free(reference_strides);

    size_t row_count;
    size_t entry_count;
    constraint_reference_layout(&test_descriptor, sides, &row_count, &entry_count);
    const npy_intp row_dims[1] = {(npy_intp)(row_count + 1)};
    const npy_intp entry_dims[1] = {(npy_intp)entry_count};
    row_array = (PyArrayObject *)PyArray_SimpleNew(1, row_dims, NPY_UINTP);
    side_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT8);
    component_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT32);
    dof_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINTP);
    coefficient_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_DOUBLE);
    if (!row_array || !side_array || !component_array || !dof_array || !coefficient_array)
        goto fail;
    const kform_values_table_t *element_tables[2] = {&element_table_1.descriptor, &element_table_2.descriptor};
    constraint_reference_assemble(&test_descriptor, sides, point_weights, &test_table.descriptor, element_tables,
                                  (uint8_t *)PyArray_DATA(side_array), (uint32_t *)PyArray_DATA(component_array),
                                  (size_t *)PyArray_DATA(dof_array), (double *)PyArray_DATA(coefficient_array),
                                  (size_t *)PyArray_DATA(row_array));

    // Matching trace spaces reduce the exact moment rows to per-DoF links:
    // same packed contract, one entry per side per row, signs read from the
    // dense row itself.
    PyArrayObject *link_row_array = NULL;
    PyArrayObject *link_side_array = NULL;
    PyArrayObject *link_component_array = NULL;
    PyArrayObject *link_dof_array = NULL;
    PyArrayObject *link_coefficient_array = NULL;
    if (constraint_reference_links_eligible(&test_descriptor, sides))
    {
        size_t link_rows;
        size_t link_entries;
        constraint_reference_links_layout(&test_descriptor, sides, &link_rows, &link_entries);
        const npy_intp link_row_dims[1] = {(npy_intp)(link_rows + 1)};
        const npy_intp link_entry_dims[1] = {(npy_intp)link_entries};
        link_row_array = (PyArrayObject *)PyArray_SimpleNew(1, link_row_dims, NPY_UINTP);
        link_side_array = (PyArrayObject *)PyArray_SimpleNew(1, link_entry_dims, NPY_UINT8);
        link_component_array = (PyArrayObject *)PyArray_SimpleNew(1, link_entry_dims, NPY_UINT32);
        link_dof_array = (PyArrayObject *)PyArray_SimpleNew(1, link_entry_dims, NPY_UINTP);
        link_coefficient_array = (PyArrayObject *)PyArray_SimpleNew(1, link_entry_dims, NPY_DOUBLE);
        if (!link_row_array || !link_side_array || !link_component_array || !link_dof_array || !link_coefficient_array)
            goto fail;
        constraint_reference_links_reduce(
            &test_descriptor, sides, row_count, (uint8_t *)PyArray_DATA(side_array),
            (uint32_t *)PyArray_DATA(component_array), (size_t *)PyArray_DATA(dof_array),
            (double *)PyArray_DATA(coefficient_array), (size_t *)PyArray_DATA(row_array),
            (uint8_t *)PyArray_DATA(link_side_array), (uint32_t *)PyArray_DATA(link_component_array),
            (size_t *)PyArray_DATA(link_dof_array), (double *)PyArray_DATA(link_coefficient_array),
            (size_t *)PyArray_DATA(link_row_array));
        Py_DECREF(row_array);
        Py_DECREF(side_array);
        Py_DECREF(component_array);
        Py_DECREF(dof_array);
        Py_DECREF(coefficient_array);
        return reference_result_tuple(link_row_array, link_side_array, link_component_array, link_dof_array,
                                      link_coefficient_array);
    }

    PyMem_Free(point_weights);
    if (rules)
        python_integration_rules_release(face_dim, rules,
                                         ((integration_registry_object *)state->registry_integration)->registry);
    PyMem_Free(rule_specs);
    release_trace_basis_table(&test_table);
    release_trace_basis_table(&element_table_1);
    release_trace_basis_table(&element_table_2);
    return reference_result_tuple(row_array, side_array, component_array, dof_array, coefficient_array);

fail:
    Py_XDECREF(row_array);
    Py_XDECREF(side_array);
    Py_XDECREF(component_array);
    Py_XDECREF(dof_array);
    Py_XDECREF(coefficient_array);
    Py_XDECREF(link_row_array);
    Py_XDECREF(link_side_array);
    Py_XDECREF(link_component_array);
    Py_XDECREF(link_dof_array);
    Py_XDECREF(link_coefficient_array);
    PyMem_Free(point_weights);
    if (rules)
        python_integration_rules_release(face_dim, rules,
                                         ((integration_registry_object *)state->registry_integration)->registry);
    PyMem_Free(rule_specs);
    release_trace_basis_table(&test_table);
    release_trace_basis_table(&element_table_1);
    release_trace_basis_table(&element_table_2);
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
    void *load_memory = NULL;
    double *data_owned = NULL;
    double *surface_weights = NULL;
    trace_basis_table_t element_table = {};

    space_map_object *const face_map = setup.face_map;
    const integration_spec_t *const face_specs = face_map->int_specs;
    {
        load_memory = cutl_alloc_group(
            &PYTHON_ALLOCATOR,
            (const cutl_alloc_info_t[]){
                {sizeof(*surface_weights) * (weighted ? setup.point_count : 1), (void **)&surface_weights},
                {sizeof(*data_owned) * component_count * setup.point_count, (void **)&data_owned},
                {}});
        if (!load_memory)
            goto load_fail;
    }

    if (weighted)
    {
        for (size_t point = 0; point < setup.point_count; ++point)
        {
            const size_t source_point =
                constraint_face_point_to_source(element_dim, face_dim, orientation, face_specs, setup.canonical_specs,
                                                setup.canonical_strides, setup.source_strides, point);
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
            const size_t source_point =
                constraint_face_point_to_source(element_dim, face_dim, orientation, face_specs, setup.canonical_specs,
                                                setup.canonical_strides, setup.source_strides, point);
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

    const kform_spec_t test_descriptor = {.ndim = face_dim, .order = order, .basis = test_spec->function_space->specs};
    const constraint_element_side_t side_descriptor = {
        .ndim = element_dim, .basis_specs = element_spec->function_space->specs, .orientation = orientation};
    const kform_spec_t element_descriptor = {
        .ndim = element_dim, .order = order, .basis = element_spec->function_space->specs};
    const size_t value_count = kform_spec_total_dofs(&element_descriptor);
    result = (PyObject *)PyArray_ZEROS(1, &(npy_intp){(npy_intp)value_count}, NPY_DOUBLE, 0);
    if (!result)
        goto load_fail;
    basis_registry_object *const basis_registry = (basis_registry_object *)state->registry_basis;
    if (make_trace_basis_table(element_dim, face_dim, order, element_spec->function_space->specs, orientation,
                               setup.canonical_specs, setup.canonical_rules, setup.canonical_strides, basis_registry,
                               true, setup.point_count, &element_table) < 0)
        goto load_fail;
    constraint_physical_side_load(&test_descriptor, &side_descriptor, setup.point_weights, data_owned,
                                  weighted ? surface_weights : NULL, &element_table.descriptor,
                                  (double *)PyArray_DATA((PyArrayObject *)result));
    cutl_dealloc(&PYTHON_ALLOCATOR, load_memory);
    release_trace_basis_table(&element_table);
    release_boundary_face_setup(state, face_dim, &setup);
    release_boundary_topology(element_dim, &topology);
    cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
    return result;

load_fail:
    Py_XDECREF(data_array);
    Py_XDECREF(result);
    cutl_dealloc(&PYTHON_ALLOCATOR, load_memory);
    release_trace_basis_table(&element_table);
    release_boundary_face_setup(state, face_dim, &setup);
    release_boundary_topology(element_dim, &topology);
    cutl_dealloc(&PYTHON_ALLOCATOR, callables_memory);
    return NULL;
}

/**
 * @brief Parse one element's signed orientation record.
 *
 * Accepts a sequence of `ndim` non-zero integers in `[-ndim, ndim]`.
 */
static int parse_orientation_sequence(PyObject *object, const unsigned ndim, int8_t *const out)
{
    PyObject *const sequence = PySequence_Fast(object, "Orientations must be a sequence of integer axis mappings.");
    if (!sequence)
        return -1;
    if (PySequence_Fast_GET_SIZE(sequence) != (Py_ssize_t)ndim)
    {
        PyErr_Format(PyExc_ValueError, "Orientation records must contain %u entries.", ndim);
        Py_DECREF(sequence);
        return -1;
    }
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        const long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, (Py_ssize_t)axis));
        if (value == -1 && PyErr_Occurred())
        {
            Py_DECREF(sequence);
            return -1;
        }
        if (value == 0 || value < -(long)ndim || value > (long)ndim)
        {
            PyErr_SetString(PyExc_ValueError, "Orientation is not a signed one-based permutation.");
            Py_DECREF(sequence);
            return -1;
        }
        out[axis] = (int8_t)value;
    }
    Py_DECREF(sequence);
    return 0;
}

static PyObject *compute_kform_boundary_mass_matrices(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                                      const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;
    PyObject *specs_object;
    PyObject *orientations_object;
    PyObject *integrations_object;
    PyObject *axis_skip_object = Py_None;
    Py_ssize_t boundary_dim = -1;
    int c1_continuous = 0;
    int packed = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &specs_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &orientations_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &integrations_object},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &axis_skip_object,
                 .kwname = "axis_skip",
                 .optional = 1,
                 .kw_only = 1},
                {.type = CPYARG_TYPE_SSIZE,
                 .p_val = &boundary_dim,
                 .kwname = "boundary_dimension",
                 .optional = 1,
                 .kw_only = 1},
                {.type = CPYARG_TYPE_BOOL,
                 .p_val = &c1_continuous,
                 .kwname = "c1_continuous",
                 .optional = 1,
                 .kw_only = 1},
                {.type = CPYARG_TYPE_BOOL, .p_val = &packed, .kwname = "packed", .optional = 1, .kw_only = 1},
                {}},
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const specs_seq = PySequence_Fast(specs_object, "element_specs must be a sequence of KFormSpecs.");
    PyObject *const orientations_seq =
        PySequence_Fast(orientations_object, "orientations must be a sequence of orientation records.");
    PyObject *const integrations_seq =
        PySequence_Fast(integrations_object, "element_integrations must be a sequence of IntegrationSpaces.");
    if (!specs_seq || !orientations_seq || !integrations_seq)
    {
        Py_XDECREF(specs_seq);
        Py_XDECREF(orientations_seq);
        Py_XDECREF(integrations_seq);
        return NULL;
    }
    const Py_ssize_t nelem_ssize = PySequence_Fast_GET_SIZE(specs_seq);
    if (nelem_ssize < 2 || PySequence_Fast_GET_SIZE(orientations_seq) != nelem_ssize ||
        PySequence_Fast_GET_SIZE(integrations_seq) != nelem_ssize)
    {
        PyErr_SetString(PyExc_ValueError,
                        "element_specs, orientations, and element_integrations must be equal-length sequences of at "
                        "least two elements.");
        Py_DECREF(specs_seq);
        Py_DECREF(orientations_seq);
        Py_DECREF(integrations_seq);
        return NULL;
    }
    kform_spec_object *const first_spec = (kform_spec_object *)PySequence_Fast_GET_ITEM(specs_seq, 0);
    const unsigned ndim = (unsigned)Py_SIZE(first_spec->function_space);
    const unsigned order = first_spec->order;
    const unsigned bdim = boundary_dim >= 0 ? (unsigned)boundary_dim : ndim - 1u;
    if (bdim == 0 || bdim >= ndim)
    {
        PyErr_Format(PyExc_ValueError, "Boundary dimension %u is not in [1, %u).", bdim, ndim);
        Py_DECREF(specs_seq);
        Py_DECREF(orientations_seq);
        Py_DECREF(integrations_seq);
        return NULL;
    }
    const size_t nelem = (size_t)nelem_ssize;
    // Live from the prepare call onwards; declared before any failure jump so
    // the cleanup path never reads an uninitialized flag.
    int plan_live = 0;

    // Plan and work arrays. Every buffer is allocated (or NULL) up front so
    // the single cleanup label can release them on any failure path; the plan
    // itself is live from the prepare call onwards.
    const size_t axis_items = nelem * ndim;
    const size_t component_count = combination_total_count((uint8_t)bdim, (uint8_t)order);
    const size_t order_storage = order == 0 ? 1u : order;
    const size_t iterator_memory = combination_iterator_required_memory((uint8_t)order);
    int8_t *orientations = PyMem_Malloc(nelem * ndim * sizeof(*orientations));
    boundary_element_space_t *views = PyMem_Malloc(nelem * sizeof(*views));
    uint8_t *axis_skip = NULL;
    size_t *item_rows = PyMem_Malloc(nelem * sizeof(*item_rows));
    size_t *item_cols = PyMem_Malloc(nelem * sizeof(*item_cols));
    size_t *item_offsets = PyMem_Malloc((nelem + 1) * sizeof(*item_offsets));
    basis_spec_t *out_basis = PyMem_Malloc(bdim * sizeof(*out_basis));
    integration_spec_t *out_integration = PyMem_Malloc(bdim * sizeof(*out_integration));
    const integration_rule_t **plan_rules = PyMem_Malloc(bdim * sizeof(*plan_rules));
    const basis_set_t **plan_boundary_sets = PyMem_Malloc(bdim * sizeof(*plan_boundary_sets));
    const basis_set_t **plan_boundary_sets_lower = PyMem_Malloc(bdim * sizeof(*plan_boundary_sets_lower));
    basis_spec_t *plan_boundary_lower_specs = PyMem_Malloc(bdim * sizeof(*plan_boundary_lower_specs));
    const basis_set_t **plan_element_sets = PyMem_Malloc(axis_items * sizeof(*plan_element_sets));
    const basis_set_t **plan_element_sets_lower = PyMem_Malloc(axis_items * sizeof(*plan_element_sets_lower));
    const basis_endpoint_set_t **plan_element_endpoints = PyMem_Malloc(axis_items * sizeof(*plan_element_endpoints));
    const basis_endpoint_set_t **plan_element_endpoints_lower =
        PyMem_Malloc(axis_items * sizeof(*plan_element_endpoints_lower));
    basis_spec_t *plan_element_lower_specs = PyMem_Malloc(axis_items * sizeof(*plan_element_lower_specs));
    constrain_elements_on_boundary_work_t work = {0};
    work.axis_fixed = PyMem_Malloc(ndim * sizeof(*work.axis_fixed));
    work.axis_slot = PyMem_Malloc(ndim * sizeof(*work.axis_slot));
    work.element_rules = PyMem_Malloc(ndim * sizeof(*work.element_rules));
    work.mass.point_strides = PyMem_Malloc(bdim * sizeof(*work.mass.point_strides));
    work.mass.row_offsets = PyMem_Malloc((component_count + 1) * sizeof(*work.mass.row_offsets));
    work.mass.col_offsets = PyMem_Malloc((component_count + 1) * sizeof(*work.mass.col_offsets));
    work.mass.element_components = PyMem_Malloc(component_count * sizeof(*work.mass.element_components));
    work.mass.element_signs = PyMem_Malloc(component_count * sizeof(*work.mass.element_signs));
    work.mass.axes = PyMem_Malloc(ndim * sizeof(*work.mass.axes));
    work.mass.counts = PyMem_Malloc(bdim * sizeof(*work.mass.counts));
    work.mass.offsets = PyMem_Malloc(bdim * sizeof(*work.mass.offsets));
    work.mass.axis_sets = PyMem_Malloc(bdim * sizeof(*work.mass.axis_sets));
    work.mass.digits = PyMem_Malloc(bdim * sizeof(*work.mass.digits));
    work.mass.axis_tables = PyMem_Malloc(bdim * sizeof(*work.mass.axis_tables));
    work.mass.mapped_axes = PyMem_Malloc(order_storage * sizeof(*work.mass.mapped_axes));
    work.mass.components = PyMem_Malloc(iterator_memory);
    work.mass.blocks = PyMem_Malloc(iterator_memory);
    if (!orientations || !views || !item_rows || !item_cols || !item_offsets || !out_basis || !out_integration ||
        !plan_rules || !plan_boundary_sets || !plan_boundary_sets_lower || !plan_boundary_lower_specs ||
        !plan_element_sets || !plan_element_sets_lower || !plan_element_endpoints || !plan_element_endpoints_lower ||
        !plan_element_lower_specs || !work.axis_fixed || !work.axis_slot || !work.element_rules ||
        !work.mass.point_strides || !work.mass.row_offsets || !work.mass.col_offsets || !work.mass.element_components ||
        !work.mass.element_signs || !work.mass.axes || !work.mass.counts || !work.mass.offsets ||
        !work.mass.axis_sets || !work.mass.digits || !work.mass.axis_tables || !work.mass.mapped_axes ||
        !work.mass.components || !work.mass.blocks)
    {
        PyErr_NoMemory();
        goto fail_memory;
    }
    for (size_t element = 0; element < nelem; ++element)
    {
        kform_spec_object *const spec = (kform_spec_object *)PySequence_Fast_GET_ITEM(specs_seq, (Py_ssize_t)element);
        PyObject *const integration = PySequence_Fast_GET_ITEM(integrations_seq, (Py_ssize_t)element);
        if (!PyObject_TypeCheck(spec, state->kform_specs_type) || (unsigned)Py_SIZE(spec->function_space) != ndim ||
            spec->order != order)
        {
            PyErr_SetString(PyExc_ValueError, "All element specs must share one dimension and k-form order.");
            goto fail_memory;
        }
        if (!PyObject_TypeCheck(integration, state->integration_space_type) || (unsigned)Py_SIZE(integration) != ndim)
        {
            PyErr_SetString(PyExc_ValueError, "Element integrations must be IntegrationSpaces of the mesh dimension.");
            goto fail_memory;
        }
        if (parse_orientation_sequence(PySequence_Fast_GET_ITEM(orientations_seq, (Py_ssize_t)element), ndim,
                                       orientations + element * ndim) < 0)
        {
            goto fail_memory;
        }
        views[element] = (boundary_element_space_t){.order = order,
                                                    .orientation = orientations + element * ndim,
                                                    .basis = spec->function_space->specs,
                                                    .integration = ((integration_space_object *)integration)->specs};
    }

    if (axis_skip_object != Py_None)
    {
        PyObject *const skip_seq = PySequence_Fast(axis_skip_object, "axis_skip must be a sequence of integers.");
        if (!skip_seq)
            goto fail_memory;
        if (PySequence_Fast_GET_SIZE(skip_seq) != (Py_ssize_t)bdim)
        {
            PyErr_Format(PyExc_ValueError, "axis_skip must contain %u entries.", bdim);
            Py_DECREF(skip_seq);
            goto fail_memory;
        }
        axis_skip = PyMem_Malloc(bdim * sizeof(*axis_skip));
        if (!axis_skip)
        {
            Py_DECREF(skip_seq);
            PyErr_NoMemory();
            goto fail_memory;
        }
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            const long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(skip_seq, (Py_ssize_t)axis));
            if (value == -1 && PyErr_Occurred())
            {
                Py_DECREF(skip_seq);
                goto fail_memory;
            }
            axis_skip[axis] = (uint8_t)value;
        }
        Py_DECREF(skip_seq);
    }

    constrain_elements_on_boundary_request_t request = {
        .ndim = ndim,
        .bdim = bdim,
        .nforms = 1,
        .nelem = (unsigned)nelem,
        .elements = views,
        .axis_skip = axis_skip,
        .c1_continuous = c1_continuous != 0,
        .surface_weights = NULL,
        .test_pullbacks = NULL,
        .element_pullbacks = NULL,
        .basis_registry = ((basis_registry_object *)state->registry_basis)->registry,
        .integration_registry = ((integration_registry_object *)state->registry_integration)->registry};
    constrain_elements_on_boundary_plan_t plan;
    plan.rules = plan_rules;
    plan.boundary_sets = plan_boundary_sets;
    plan.boundary_sets_lower = plan_boundary_sets_lower;
    plan.boundary_lower_specs = plan_boundary_lower_specs;
    plan.element_sets = plan_element_sets;
    plan.element_sets_lower = plan_element_sets_lower;
    plan.element_endpoints = plan_element_endpoints;
    plan.element_endpoints_lower = plan_element_endpoints_lower;
    plan.element_lower_specs = plan_element_lower_specs;
    plan.item_rows = item_rows;
    plan.item_cols = item_cols;
    plan.item_offsets = item_offsets;
    fdg_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = constrain_elements_on_boundary_prepare(&request, &work, out_basis, out_integration, &plan);
    Py_END_ALLOW_THREADS;
    // The prepare call NULL-fills the plan's reference slots before any
    // registry fetch, so the plan is releasable from here on.
    plan_live = 1;
    if (res != FDG_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not prepare boundary constraint batch: %s (%s)", fdg_error_str(res),
                     fdg_error_msg(res));
        goto fail_memory;
    }

    PyArrayObject *const arena =
        (PyArrayObject *)PyArray_SimpleNew(1, &(npy_intp){(npy_intp)plan.total_values}, NPY_DOUBLE);
    if (!arena)
        goto fail_memory;

    // Value tables sized after prepare: their sizes depend on the merged
    // common rules.
    size_t weights_size;
    size_t row_values_size;
    size_t col_values_size;
    constrain_elements_on_boundary_work_size(&request, &plan, &weights_size, &row_values_size, &col_values_size);
    work.weights = PyMem_Malloc(weights_size * sizeof(*work.weights));
    work.mass.row_values = PyMem_Malloc(row_values_size * sizeof(*work.mass.row_values));
    work.mass.col_values = PyMem_Malloc(col_values_size * sizeof(*work.mass.col_values));
    work.mass.point_factors = PyMem_Malloc(weights_size * sizeof(*work.mass.point_factors));
    if (!work.weights || !work.mass.row_values || !work.mass.col_values || !work.mass.point_factors)
    {
        PyErr_NoMemory();
        goto fail_memory;
    }

    Py_BEGIN_ALLOW_THREADS;
    constrain_elements_on_boundary_assemble(&request, &plan, &work, (double *)PyArray_DATA(arena));
    Py_END_ALLOW_THREADS;

    // Common boundary space as Python objects.
    function_space_object *const common_space =
        function_space_object_create(state->function_space_type, bdim, out_basis);
    PyObject *const common_specs = common_space ? PyObject_CallFunction((PyObject *)state->kform_specs_type, "nO",
                                                                        (Py_ssize_t)order, (PyObject *)common_space)
                                                : NULL;
    Py_XDECREF(common_space);
    integration_space_object *const common_integration =
        (integration_space_object *)state->integration_space_type->tp_alloc(state->integration_space_type,
                                                                            (Py_ssize_t)bdim);
    if (!common_integration || !common_specs)
    {
        Py_XDECREF(common_specs);
        Py_XDECREF((PyObject *)common_integration);
        Py_DECREF(arena);
        goto fail_memory;
    }
    memcpy(common_integration->specs, out_integration, bdim * sizeof(*out_integration));

    PyObject *const matrices = PyTuple_New((Py_ssize_t)nelem);
    PyObject *const packed_rows = packed ? PyTuple_New((Py_ssize_t)nelem) : NULL;
    if (!matrices || (packed && !packed_rows))
    {
        Py_XDECREF(packed_rows);
        Py_DECREF(matrices);
        Py_DECREF(common_specs);
        Py_DECREF((PyObject *)common_integration);
        Py_DECREF(arena);
        goto fail_memory;
    }
    for (size_t element = 0; element < nelem; ++element)
    {
        const npy_intp dims[2] = {(npy_intp)item_rows[element], (npy_intp)item_cols[element]};
        PyArrayObject *const matrix = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (!matrix || (packed && !packed_rows))
        {
            Py_XDECREF(matrix);
            Py_DECREF(matrices);
            Py_XDECREF(packed_rows);
            Py_DECREF(common_specs);
            Py_DECREF((PyObject *)common_integration);
            Py_DECREF(arena);
            goto fail_memory;
        }
        memcpy(PyArray_DATA(matrix), (double *)PyArray_DATA(arena) + item_offsets[element],
               (size_t)(PyArray_DIM(matrix, 0) * PyArray_DIM(matrix, 1)) * sizeof(double));
        PyTuple_SET_ITEM(matrices, (Py_ssize_t)element, (PyObject *)matrix);
        if (packed)
        {
            const kform_spec_t element_spec = {.ndim = ndim, .order = order, .basis = views[element].basis};
            const constraint_boundary_mass_spec_t spec = {.ndim = ndim,
                                                          .bdim = bdim,
                                                          .order = order,
                                                          .element_spec = &element_spec,
                                                          .boundary_basis = out_basis,
                                                          .boundary_integration = out_integration,
                                                          .orientation = views[element].orientation,
                                                          .axis_skip = axis_skip};
            size_t rows;
            size_t cols;
            size_t entries;
            constraint_boundary_mass_layout(&spec, &work.mass, false, &rows, &cols, &entries);
            const npy_intp entry_dims[1] = {(npy_intp)entries};
            const npy_intp row_dims[1] = {(npy_intp)(rows + 1)};
            PyArrayObject *const side_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT8);
            PyArrayObject *const component_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT32);
            PyArrayObject *const dof_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINTP);
            PyArrayObject *const coefficient_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_DOUBLE);
            PyArrayObject *const row_array = (PyArrayObject *)PyArray_SimpleNew(1, row_dims, NPY_UINTP);
            if (!side_array || !component_array || !dof_array || !coefficient_array || !row_array)
            {
                Py_XDECREF(side_array);
                Py_XDECREF(component_array);
                Py_XDECREF(dof_array);
                Py_XDECREF(coefficient_array);
                Py_XDECREF(row_array);
                Py_DECREF(matrices);
                Py_DECREF(packed_rows);
                Py_DECREF(common_specs);
                Py_DECREF((PyObject *)common_integration);
                Py_DECREF(arena);
                goto fail_memory;
            }
            constraint_boundary_mass_pack(&spec, &work.mass, false, (const double *)PyArray_DATA(matrix), (size_t)cols,
                                          1.0, (uint8_t)element, (uint8_t *)PyArray_DATA(side_array),
                                          (uint32_t *)PyArray_DATA(component_array), (size_t *)PyArray_DATA(dof_array),
                                          (double *)PyArray_DATA(coefficient_array), (size_t *)PyArray_DATA(row_array));
            PyObject *const item =
                PyTuple_Pack(5, row_array, side_array, component_array, dof_array, coefficient_array);
            Py_DECREF(row_array);
            Py_DECREF(side_array);
            Py_DECREF(component_array);
            Py_DECREF(dof_array);
            Py_DECREF(coefficient_array);
            if (!item)
            {
                Py_DECREF(matrices);
                Py_DECREF(packed_rows);
                Py_DECREF(common_specs);
                Py_DECREF((PyObject *)common_integration);
                Py_DECREF(arena);
                goto fail_memory;
            }
            PyTuple_SET_ITEM(packed_rows, (Py_ssize_t)element, item);
        }
    }

    PyMem_Free(out_integration);
    PyMem_Free(out_basis);
    PyMem_Free(item_offsets);
    PyMem_Free(item_cols);
    PyMem_Free(item_rows);
    PyMem_Free(axis_skip);
    PyMem_Free(views);
    PyMem_Free(orientations);
    Py_DECREF(specs_seq);
    Py_DECREF(orientations_seq);
    Py_DECREF(integrations_seq);
    Py_DECREF(arena);
    PyObject *const result =
        PyTuple_Pack(4, common_specs, common_integration, matrices, packed ? packed_rows : Py_None);
    Py_DECREF(common_specs);
    Py_DECREF((PyObject *)common_integration);
    Py_DECREF(matrices);
    Py_XDECREF(packed_rows);
    Py_BEGIN_ALLOW_THREADS;
    constrain_elements_on_boundary_plan_release(&plan);
    Py_END_ALLOW_THREADS;
    return result;

fail_memory:
    if (plan_live)
    {
        Py_BEGIN_ALLOW_THREADS;
        constrain_elements_on_boundary_plan_release(&plan);
        Py_END_ALLOW_THREADS;
    }
    PyMem_Free(work.mass.point_factors);
    PyMem_Free(work.mass.col_values);
    PyMem_Free(work.mass.row_values);
    PyMem_Free(work.weights);
    PyMem_Free(work.mass.blocks);
    PyMem_Free(work.mass.components);
    PyMem_Free(work.mass.mapped_axes);
    PyMem_Free(work.mass.axis_tables);
    PyMem_Free(work.mass.digits);
    PyMem_Free(work.mass.axis_sets);
    PyMem_Free(work.mass.offsets);
    PyMem_Free(work.mass.counts);
    PyMem_Free(work.mass.axes);
    PyMem_Free(work.mass.element_signs);
    PyMem_Free(work.mass.element_components);
    PyMem_Free(work.mass.col_offsets);
    PyMem_Free(work.mass.row_offsets);
    PyMem_Free(work.mass.point_strides);
    PyMem_Free(work.element_rules);
    PyMem_Free(work.axis_slot);
    PyMem_Free(work.axis_fixed);
    PyMem_Free(plan_element_lower_specs);
    PyMem_Free(plan_element_endpoints_lower);
    PyMem_Free(plan_element_endpoints);
    PyMem_Free(plan_element_sets_lower);
    PyMem_Free(plan_element_sets);
    PyMem_Free(plan_boundary_lower_specs);
    PyMem_Free(plan_boundary_sets_lower);
    PyMem_Free(plan_boundary_sets);
    PyMem_Free(plan_rules);
    PyMem_Free(out_integration);
    PyMem_Free(out_basis);
    PyMem_Free(item_offsets);
    PyMem_Free(item_cols);
    PyMem_Free(item_rows);
    PyMem_Free(axis_skip);
    PyMem_Free(views);
    PyMem_Free(orientations);
    Py_DECREF(specs_seq);
    Py_DECREF(orientations_seq);
    Py_DECREF(integrations_seq);
    return NULL;
}

static PyObject *compute_boundary_space_map_factors(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                                    const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;
    PyObject *map_object;
    PyObject *orientation_object;
    PyObject *common_object;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &map_object, .type_check = state->space_mapping_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &orientation_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &common_object, .type_check = state->integration_space_type},
                {}},
            args, nargs, kwnames) < 0)
        return NULL;

    space_map_object *const map = (space_map_object *)map_object;
    integration_space_object *const common = (integration_space_object *)common_object;
    const unsigned coords = (unsigned)Py_SIZE(map);
    const unsigned ndim = map->ndim;
    const unsigned bdim = (unsigned)Py_SIZE(common);
    const unsigned fixed_count = ndim - bdim;
    if (bdim == 0 || bdim >= ndim)
    {
        PyErr_Format(PyExc_ValueError, "Boundary dimension %zu is not in [1, %u).", Py_SIZE(common), ndim);
        return NULL;
    }
    int8_t *const orientation = PyMem_Malloc(ndim * sizeof(*orientation));
    if (!orientation)
        return PyErr_NoMemory();
    if (parse_orientation_sequence(orientation_object, ndim, orientation) < 0)
    {
        PyMem_Free(orientation);
        return NULL;
    }

    space_map_object *const face_map = space_map_boundary_oriented_impl(state, map, fixed_count, orientation);
    PyMem_Free(orientation);
    if (!face_map)
        return NULL;
    if ((unsigned)face_map->ndim != bdim)
    {
        PyErr_SetString(PyExc_ValueError, "The restricted face map disagrees with the boundary dimension.");
        Py_DECREF((PyObject *)face_map);
        return NULL;
    }

    integration_rule_registry_t *const registry =
        ((integration_registry_object *)state->registry_integration)->registry;
    const integration_rule_t **const source_rules = python_integration_rules_get(bdim, face_map->int_specs, registry);
    const integration_rule_t **const target_rules = python_integration_rules_get(bdim, common->specs, registry);
    if (!source_rules || !target_rules)
    {
        if (source_rules)
            python_integration_rules_release(bdim, source_rules, registry);
        if (target_rules)
            python_integration_rules_release(bdim, target_rules, registry);
        Py_DECREF((PyObject *)face_map);
        return NULL;
    }

    const npy_intp point_count = (npy_intp)integration_specs_total_points(bdim, common->specs);
    PyArrayObject *const determinant = (PyArrayObject *)PyArray_SimpleNew(1, &point_count, NPY_DOUBLE);
    const npy_intp inverse_dims[3] = {point_count, (npy_intp)bdim, (npy_intp)coords};
    PyArrayObject *const inverse_maps = (PyArrayObject *)PyArray_SimpleNew(3, inverse_dims, NPY_DOUBLE);
    size_t axis_matrices_size;
    size_t positions_size;
    size_t jacobian_size;
    size_t q_size;
    boundary_space_map_resample_work_size(bdim, coords, source_rules, target_rules, &axis_matrices_size,
                                          &positions_size, &jacobian_size, &q_size);
    double *const axis_matrices = PyMem_Malloc(axis_matrices_size * sizeof(*axis_matrices));
    double *const positions = PyMem_Malloc(positions_size * sizeof(*positions));
    double *const jacobian = PyMem_Malloc(jacobian_size * sizeof(*jacobian));
    double *const q = PyMem_Malloc(q_size * sizeof(*q));
    const double **const coordinate_values = PyMem_Malloc(coords * sizeof(*coordinate_values));
    const double **const coordinate_gradients = PyMem_Malloc((size_t)coords * bdim * sizeof(*coordinate_gradients));
    if (!determinant || !inverse_maps || !axis_matrices || !positions || !jacobian || !q || !coordinate_values ||
        !coordinate_gradients)
    {
        PyMem_Free(coordinate_gradients);
        PyMem_Free(coordinate_values);
        PyMem_Free(q);
        PyMem_Free(jacobian);
        PyMem_Free(positions);
        PyMem_Free(axis_matrices);
        Py_XDECREF(determinant);
        Py_XDECREF(inverse_maps);
        python_integration_rules_release(bdim, source_rules, registry);
        python_integration_rules_release(bdim, target_rules, registry);
        Py_DECREF((PyObject *)face_map);
        return PyErr_NoMemory();
    }
    for (unsigned coordinate = 0; coordinate < coords; ++coordinate)
    {
        coordinate_values[coordinate] = coordinate_map_values(face_map->maps[coordinate]);
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            coordinate_gradients[(size_t)coordinate * bdim + axis] =
                coordinate_map_gradient(face_map->maps[coordinate], axis);
        }
    }

    const boundary_space_map_resample_request_t request = {.bdim = bdim,
                                                           .coords = coords,
                                                           .source_rules = source_rules,
                                                           .target_rules = target_rules,
                                                           .coordinate_values = coordinate_values,
                                                           .coordinate_gradients = coordinate_gradients,
                                                           .out_determinant = (double *)PyArray_DATA(determinant),
                                                           .out_inverse_maps = (double *)PyArray_DATA(inverse_maps),
                                                           .axis_matrices = axis_matrices,
                                                           .positions = positions,
                                                           .jacobian = jacobian,
                                                           .q = q};
    Py_BEGIN_ALLOW_THREADS;
    boundary_space_map_resample(&request);
    Py_END_ALLOW_THREADS;

    PyMem_Free(coordinate_gradients);
    PyMem_Free(coordinate_values);
    PyMem_Free(q);
    PyMem_Free(jacobian);
    PyMem_Free(positions);
    PyMem_Free(axis_matrices);
    python_integration_rules_release(bdim, source_rules, registry);
    python_integration_rules_release(bdim, target_rules, registry);
    Py_DECREF((PyObject *)face_map);

    PyObject *const result = PyTuple_Pack(2, determinant, inverse_maps);
    Py_DECREF(determinant);
    Py_DECREF(inverse_maps);
    return result;
}

PyMethodDef constraint_methods[] = {
    {
        .ml_name = "packed_kform_constraints_to_csr",
        .ml_meth = (void *)packed_kform_constraints_to_csr,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc =
            "packed_kform_constraints_to_csr(packed, specs, element_count, /) -> "
            "tuple[numpy.ndarray, ...]\nConvert packed global k-form rows to CSR data, indices, and indptr arrays.",
    },
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
    {
        .ml_name = "compute_kform_boundary_mass_matrices",
        .ml_meth = (void *)compute_kform_boundary_mass_matrices,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_kform_boundary_mass_matrices(element_specs, orientations, element_integrations, "
                  "axis_skip=None, *, boundary_dimension=None, c1_continuous=False, packed=False) -> tuple\n"
                  "Assemble every incident element's mass matrix against the common boundary space of one shared "
                  "object. Returns (common KFormSpecs, common IntegrationSpace, per-element dense matrices, "
                  "per-element packed COO tuples or None). Rows are the common Legendre k-form test space with "
                  "axis_skip[axis] lowest functions removed on inactive axes; columns are the mapped element trace "
                  "DoFs. Coefficients carry the orientation signs but no side signs.",
    },
    {
        .ml_name = "compute_boundary_space_map_factors",
        .ml_meth = (void *)compute_boundary_space_map_factors,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "compute_boundary_space_map_factors(space_map, orientation, common_integration) -> tuple\n"
                  "Interpolate a face-restricted space map onto the common boundary integration grid. Returns "
                  "(determinant, inverse_maps) sampled at the common boundary points, where the determinant is the "
                  "surface measure of the face immersion and inverse_maps has shape (points, boundary_dim, coords).",
    },
    {},
};
