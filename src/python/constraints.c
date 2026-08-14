#include "../constraints/constraints.h"
#include "../operations/topology.h"
#include "basis_objects.h"
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

static size_t canonical_point_to_source(const unsigned element_dim, const unsigned face_dim,
                                        const int8_t orientation[const static element_dim],
                                        const integration_spec_t source_specs[const static face_dim],
                                        const integration_spec_t canonical_specs[const static face_dim],
                                        const unsigned canonical_digits[const static face_dim])
{
    const unsigned fixed_count = element_dim - face_dim;
    unsigned source_digits[face_dim == 0 ? 1 : face_dim];
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const unsigned source_axis =
            (unsigned)(orientation[face_axis + 1] < 0 ? -orientation[face_axis + 1] : orientation[face_axis + 1]) - 1;
        unsigned source_face_axis = 0;
        for (unsigned i = 0; i < source_axis; ++i)
        {
            bool fixed = false;
            for (unsigned fixed_index = 0; fixed_index < fixed_count; ++fixed_index)
                fixed |=
                    ((unsigned)(orientation[fixed_index] < 0 ? -orientation[fixed_index] : orientation[fixed_index]) -
                     1) == i;
            if (!fixed)
                ++source_face_axis;
        }
        source_digits[source_face_axis] = orientation[face_axis + 1] < 0
                                              ? source_specs[source_face_axis].order - canonical_digits[face_axis]
                                              : canonical_digits[face_axis];
    }

    size_t source_point = 0;
    size_t stride = 1;
    for (unsigned axis = 0; axis < face_dim; ++axis)
    {
        source_point += source_digits[axis] * stride;
        stride *= source_specs[axis].order + 1;
    }
    (void)canonical_specs;
    return source_point;
}

static unsigned map_component(const unsigned face_dim, const unsigned element_dim, const unsigned order,
                              const uint8_t face_axes[const static order == 0 ? 1 : order],
                              const int8_t orientation[const static element_dim], int *sign)
{
    uint8_t mapped[order == 0 ? 1 : order];
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
                *sign = -*sign;
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
    const size_t source_point_count = total_points(face_dim, face_specs);
    const npy_double *transform_data = PyArray_DATA(transform);
    npy_double *pullback_data = PyArray_DATA(pullback);
    for (unsigned face_component = 0; face_component < face_components; ++face_component)
    {
        uint8_t face_axes[order == 0 ? 1 : order];
        combination_set_to_index((uint8_t)face_dim, (uint8_t)order, face_axes, face_component);
        int orientation_sign;
        const unsigned element_component =
            map_component(face_dim, element_dim, order, face_axes, orientation, &orientation_sign);
        (void)orientation_sign;
        for (size_t canonical_point = 0; canonical_point < point_count; ++canonical_point)
        {
            unsigned canonical_digits[face_dim == 0 ? 1 : face_dim];
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
    Py_DECREF(transform);
    *out = pullback;
    return 0;
}

#if 0
static PyObject *compute_kform_boundary_constraints_pair_legacy(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                                    const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;
    PyObject *test_object;
    PyObject *specs_object;
    PyObject *maps_object;
    PyObject *collections_object;
    PyObject *element_ids_object;
    Py_ssize_t npts;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &test_object, .type_check = state->kform_specs_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &specs_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &maps_object},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &collections_object},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &npts},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &element_ids_object},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    kform_spec_object *test_spec = (kform_spec_object *)test_object;
    PyObject *spec_object_1, *spec_object_2, *map_object_1, *map_object_2;
    if (get_pair(specs_object, "element_specs", &spec_object_1, &spec_object_2) < 0 ||
        get_pair(maps_object, "element_maps", &map_object_1, &map_object_2) < 0)
        return NULL;
    if (!PyObject_TypeCheck(spec_object_1, state->kform_specs_type) ||
        !PyObject_TypeCheck(spec_object_2, state->kform_specs_type) ||
        !PyObject_TypeCheck(map_object_1, state->space_mapping_type) ||
        !PyObject_TypeCheck(map_object_2, state->space_mapping_type))
    {
        PyErr_SetString(PyExc_TypeError, "element_specs and element_maps must contain the expected fdg types.");
        return NULL;
    }
    kform_spec_object *element_specs[2] = {(kform_spec_object *)spec_object_1, (kform_spec_object *)spec_object_2};
    space_map_object *element_maps[2] = {(space_map_object *)map_object_1, (space_map_object *)map_object_2};

    const unsigned face_dim = Py_SIZE(test_spec->function_space);
    const unsigned order = test_spec->order;
    if (face_dim == 0 || order > face_dim)
    {
        PyErr_SetString(PyExc_ValueError,
                        "The test k-form must be defined on a non-empty boundary and have valid order.");
        return NULL;
    }
    const unsigned element_dim = Py_SIZE(element_spec->function_space);
    if (element_specs[0]->order != order || element_specs[1]->order != order ||
        Py_SIZE(element_specs[0]->function_space) != element_dim ||
        Py_SIZE(element_specs[1]->function_space) != element_dim || element_maps[0]->ndim != element_dim ||
        element_maps[1]->ndim != element_dim || Py_SIZE(element_maps[0]) != Py_SIZE(element_maps[1]))
    {
        PyErr_SetString(PyExc_ValueError, "Element specs and maps must describe matching N-dimensional k-forms.");
        return NULL;
    }

    if (npts < 0 || !PyTuple_Check(collections_object) || PyTuple_GET_SIZE(collections_object) != element_dim)
    {
        PyErr_Format(PyExc_ValueError, "Expected %u mesh collections and a non-negative npts.", element_dim);
        return NULL;
    }
    PyArrayObject *collection_arrays[element_dim];
    for (unsigned i = 0; i < element_dim; ++i)
        collection_arrays[i] = NULL;
    topo_obj_collection_t collections[element_dim];
    for (unsigned idim = 0; idim < element_dim; ++idim)
    {
        collection_arrays[idim] = (PyArrayObject *)PyArray_FROMANY(PyTuple_GET_ITEM(collections_object, idim),
                                                                   NPY_UINT64, 2, 2, NPY_ARRAY_IN_ARRAY);
        if (!collection_arrays[idim] || PyArray_DIM(collection_arrays[idim], 1) != 2 * (idim + 1))
        {
            PyErr_Format(PyExc_ValueError, "Mesh collection %u must have shape (count, %u).", idim, 2 * (idim + 1));
            release_collection_arrays(element_dim, collection_arrays);
            return NULL;
        }
        collections[idim] = (topo_obj_collection_t){
            .ndim = idim + 1,
            .count = (size_t)PyArray_DIM(collection_arrays[idim], 0),
            .boundary_ids = PyArray_DATA(collection_arrays[idim]),
        };
    }
    PyArrayObject *element_ids_array =
        (PyArrayObject *)PyArray_FROMANY(element_ids_object, NPY_UINT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (!element_ids_array || PyArray_DIM(element_ids_array, 0) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "element_ids must have shape (2,).");
        Py_XDECREF(element_ids_array);
        release_collection_arrays(element_dim, collection_arrays);
        return NULL;
    }
    topo_obj_immersion_t immersions[element_dim] = {};
    topo_status_t topo_status =
        topo_obj_create_immersion_info(element_dim, (unsigned)npts, collections, &PYTHON_ALLOCATOR, immersions);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not create mesh immersions: %s (%s).", topo_status_to_str(topo_status),
                     topo_status_msg(topo_status));
        Py_DECREF(element_ids_array);
        release_collection_arrays(element_dim, collection_arrays);
        return NULL;
    }
    int8_t orientation_values[2 * element_dim];
    uint64_t common_boundary_id;
    const uint64_t *element_ids = PyArray_DATA(element_ids_array);
    topo_status = topo_obj_find_common_boundary(immersions + element_dim - 1, element_dim, element_ids[0],
                                                element_ids[1], &common_boundary_id, orientation_values);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not find a unique common boundary: %s (%s).",
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        topo_obj_immersions_free(element_dim, immersions, &PYTHON_ALLOCATOR);
        Py_DECREF(element_ids_array);
        release_collection_arrays(element_dim, collection_arrays);
        return NULL;
    }
    const npy_intp orientation_dims[2] = {2, (npy_intp)element_dim};
    PyArrayObject *orientation_array = (PyArrayObject *)PyArray_SimpleNew(2, orientation_dims, NPY_INT8);
    if (!orientation_array)
    {
        topo_obj_immersions_free(element_dim, immersions, &PYTHON_ALLOCATOR);
        Py_DECREF(element_ids_array);
        release_collection_arrays(element_dim, collection_arrays);
        return NULL;
    }
    memcpy(PyArray_DATA(orientation_array), orientation_values, sizeof(orientation_values));
    const int8_t *orientation_data = PyArray_DATA(orientation_array);
    Py_DECREF(element_ids_array);
    topo_obj_immersions_free(element_dim, immersions, &PYTHON_ALLOCATOR);
    for (unsigned idim = 0; idim < element_dim; ++idim)
    {
        Py_DECREF(collection_arrays[idim]);
        collection_arrays[idim] = NULL;
    }

    space_map_object *face_maps[2] = {};
    integration_space_object *face_spaces[2] = {};
    PyArrayObject *pullbacks[2] = {};
    double *surface_weights[2] = {};
    constraint_quadrature_t *quadrature_axes[2] = {};
    const integration_rule_t **rules[2] = {};
    size_t face_point_count[2] = {};
    integration_registry_object *const integration_registry =
        (integration_registry_object *)state->registry_integration;
    for (unsigned side = 0; side < 2; ++side)
    {
        const int8_t *side_orientation = orientation_data + side * element_dim;
        PyObject *face_object = NULL;
        if (make_boundary_map(state, element_maps[side], side_orientation, element_dim, face_dim, NULL, &face_object) < 0)
            goto fail;
        face_maps[side] = (space_map_object *)face_object;
        if (make_integration_space(state, face_maps[side], face_spaces + side) < 0)
            goto fail;
        const integration_spec_t *face_specs = face_maps[side]->int_specs;
        const unsigned normal_axis =
            (unsigned)(side_orientation[0] < 0 ? -side_orientation[0] : side_orientation[0]) - 1;
        integration_spec_t canonical_specs[face_dim];
        for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
        {
            const unsigned source_axis =
                (unsigned)(side_orientation[face_axis + 1] < 0 ? -side_orientation[face_axis + 1]
                                                               : side_orientation[face_axis + 1]) -
                1;
            const unsigned source_face_axis = source_axis < normal_axis ? source_axis : source_axis - 1;
            canonical_specs[face_axis] = face_specs[source_face_axis];
        }
        if (create_pullback(face_maps[side], element_dim, face_dim, side_orientation, order, face_specs,
                            canonical_specs, pullbacks + side) < 0)
            goto fail;
        face_point_count[side] = total_points(face_dim, face_specs);
        surface_weights[side] = PyMem_Malloc(face_point_count[side] * sizeof(*surface_weights[side]));
        if (!surface_weights[side])
            goto fail;
        unsigned canonical_digits[face_dim];
        for (size_t point = 0; point < face_point_count[side]; ++point)
        {
            get_digits(face_dim, canonical_specs, point, canonical_digits);
            const size_t source_point =
                canonical_point_to_source(element_dim, side_orientation, face_specs, canonical_specs, canonical_digits);
            surface_weights[side][point] = fabs(face_maps[side]->determinant[source_point]);
        }
        quadrature_axes[side] = PyMem_Malloc(face_dim * sizeof(*quadrature_axes[side]));
        if (!quadrature_axes[side])
            goto fail;
        rules[side] = python_integration_rules_get(face_dim, face_specs, integration_registry->registry);
        if (!rules[side])
            goto fail;
        for (unsigned i = 0; i < face_dim; ++i)
        {
            const unsigned source_axis =
                (unsigned)(side_orientation[i + 1] < 0 ? -side_orientation[i + 1] : side_orientation[i + 1]) - 1;
            const unsigned source_face_axis = source_axis < normal_axis ? source_axis : source_axis - 1;
            const integration_rule_t *rule = rules[side][source_face_axis];
            quadrature_axes[side][i] = (constraint_quadrature_t){.count = rule->n_nodes,
                                                                 .nodes = integration_rule_nodes_const(rule),
                                                                 .weights = integration_rule_weights_const(rule)};
        }
    }

    constraint_kform_spec_t test_descriptor = {
        .ndim = face_dim, .order = order, .basis_specs = test_spec->function_space->specs};
    constraint_element_side_t side_descriptors[2] = {
        {.ndim = element_dim, .basis_specs = element_specs[0]->function_space->specs, .orientation = orientation_data},
        {.ndim = element_dim,
         .basis_specs = element_specs[1]->function_space->specs,
         .orientation = orientation_data + element_dim},
    };
    constraint_trace_pullback_t pullback_descriptors[2] = {
        {.physical_component_count = (unsigned)Py_SIZE(element_maps[0]),
         .point_count = face_point_count[0],
         .values = PyArray_DATA(pullbacks[0])},
        {.physical_component_count = (unsigned)Py_SIZE(element_maps[1]),
         .point_count = face_point_count[1],
         .values = PyArray_DATA(pullbacks[1])},
    };
    size_t row_count;
    size_t entry_count;
    constraint_status_t status =
        constraint_physical_required(&test_descriptor, side_descriptors, &row_count, &entry_count);
    if (status != CONSTRAINT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not size constraints: %s (%s).", constraint_status_to_str(status),
                     constraint_status_msg(status));
        for (unsigned side = 0; side < 2; ++side)
            if (rules[side])
                python_integration_rules_release(face_dim, rules[side], integration_registry->registry);
        for (unsigned side = 0; side < 2; ++side)
        {
            PyMem_Free(quadrature_axes[side]);
            PyMem_Free(surface_weights[side]);
        }
        goto fail;
    }

    const npy_intp row_dims[1] = {(npy_intp)(row_count + 1)};
    const npy_intp entry_dims[1] = {(npy_intp)entry_count};
    PyArrayObject *row_array = (PyArrayObject *)PyArray_SimpleNew(1, row_dims, NPY_UINTP);
    PyArrayObject *side_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT8);
    PyArrayObject *component_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT32);
    PyArrayObject *dof_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINTP);
    PyArrayObject *coefficient_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_DOUBLE);
    if (!row_array || !side_array || !component_array || !dof_array || !coefficient_array)
    {
        Py_XDECREF(row_array);
        Py_XDECREF(side_array);
        Py_XDECREF(component_array);
        Py_XDECREF(dof_array);
        Py_XDECREF(coefficient_array);
        for (unsigned side = 0; side < 2; ++side)
            if (rules[side])
                python_integration_rules_release(face_dim, rules[side], integration_registry->registry);
        for (unsigned side = 0; side < 2; ++side)
        {
            PyMem_Free(quadrature_axes[side]);
            PyMem_Free(surface_weights[side]);
        }
        goto fail;
    }
    size_t actual_rows;
    size_t actual_entries;
    constraint_entry_t *raw_entries = (constraint_entry_t *)PyMem_Malloc(entry_count * sizeof(*raw_entries));
    if (raw_entries)
    {
        const constraint_face_quadrature_t face_quadrature[2] = {
            {.ndim = face_dim, .axes = quadrature_axes[0], .point_count = face_point_count[0]},
            {.ndim = face_dim, .axes = quadrature_axes[1], .point_count = face_point_count[1]},
        };
        const double *const side_surface_weights[2] = {surface_weights[0], surface_weights[1]};
        status = constraint_physical_assemble(&test_descriptor, side_descriptors, face_quadrature, side_surface_weights,
                                              pullback_descriptors, row_count + 1, (size_t *)PyArray_DATA(row_array),
                                              entry_count, raw_entries, &actual_rows, &actual_entries);
    }
    if (status != CONSTRAINT_SUCCESS || !raw_entries)
    {
        PyMem_Free(raw_entries);
        Py_DECREF(row_array);
        Py_DECREF(side_array);
        Py_DECREF(component_array);
        Py_DECREF(dof_array);
        Py_DECREF(coefficient_array);
        for (unsigned side = 0; side < 2; ++side)
            if (rules[side])
                python_integration_rules_release(face_dim, rules[side], integration_registry->registry);
        for (unsigned side = 0; side < 2; ++side)
        {
            PyMem_Free(quadrature_axes[side]);
            PyMem_Free(surface_weights[side]);
        }
        PyErr_Format(PyExc_ValueError, "Could not assemble constraints: %s (%s).", constraint_status_to_str(status),
                     constraint_status_msg(status));
        goto fail;
    }
    for (size_t i = 0; i < actual_entries; ++i)
    {
        ((uint8_t *)PyArray_DATA(side_array))[i] = raw_entries[i].side;
        ((uint32_t *)PyArray_DATA(component_array))[i] = raw_entries[i].component;
        ((size_t *)PyArray_DATA(dof_array))[i] = raw_entries[i].local_dof;
        ((double *)PyArray_DATA(coefficient_array))[i] = raw_entries[i].coefficient;
    }
    PyMem_Free(raw_entries);
    for (unsigned side = 0; side < 2; ++side)
    {
        if (rules[side])
            python_integration_rules_release(face_dim, rules[side], integration_registry->registry);
        PyMem_Free(quadrature_axes[side]);
        PyMem_Free(surface_weights[side]);
    }

    PyObject *result = PyTuple_New(5);
    if (!result)
    {
        Py_DECREF(row_array);
        Py_DECREF(side_array);
        Py_DECREF(component_array);
        Py_DECREF(dof_array);
        Py_DECREF(coefficient_array);
        goto fail;
    }
    PyTuple_SET_ITEM(result, 0, row_array);
    PyTuple_SET_ITEM(result, 1, side_array);
    PyTuple_SET_ITEM(result, 2, component_array);
    PyTuple_SET_ITEM(result, 3, dof_array);
    PyTuple_SET_ITEM(result, 4, coefficient_array);
    for (unsigned side = 0; side < 2; ++side)
    {
        Py_DECREF(face_maps[side]);
        Py_DECREF(pullbacks[side]);
        Py_DECREF(face_spaces[side]);
    }
    Py_DECREF(orientation_array);
    return result;

fail:
    for (unsigned side = 0; side < 2; ++side)
    {
        if (rules[side])
            python_integration_rules_release(face_dim, rules[side], integration_registry->registry);
        PyMem_Free(quadrature_axes[side]);
        PyMem_Free(surface_weights[side]);
        Py_XDECREF(face_maps[side]);
        Py_XDECREF(pullbacks[side]);
        Py_XDECREF(face_spaces[side]);
    }
    Py_DECREF(orientation_array);
    return NULL;
}

#endif

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

    PyArrayObject *collection_arrays[element_dim];
    topo_obj_collection_t collections[element_dim];
    for (unsigned idim = 0; idim < element_dim; ++idim)
        collection_arrays[idim] = NULL;
    for (unsigned idim = 0; idim < element_dim; ++idim)
    {
        collection_arrays[idim] = (PyArrayObject *)PyArray_FROMANY(PyTuple_GET_ITEM(collections_object, idim),
                                                                   NPY_UINT64, 2, 2, NPY_ARRAY_IN_ARRAY);
        if (!collection_arrays[idim] || PyArray_DIM(collection_arrays[idim], 1) != 2 * (idim + 1))
        {
            PyErr_Format(PyExc_ValueError, "Mesh collection %u must have shape (count, %u).", idim, 2 * (idim + 1));
            release_collection_arrays(element_dim, collection_arrays);
            return NULL;
        }
        collections[idim] = (topo_obj_collection_t){
            .ndim = idim + 1,
            .count = (size_t)PyArray_DIM(collection_arrays[idim], 0),
            .boundary_ids = PyArray_DATA(collection_arrays[idim]),
        };
    }

    topo_obj_immersion_t immersions[element_dim] = {};
    topo_status_t topo_status =
        topo_obj_create_immersion_info(element_dim, (unsigned)npts, collections, &PYTHON_ALLOCATOR, immersions);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not create mesh immersions: %s (%s).", topo_status_to_str(topo_status),
                     topo_status_msg(topo_status));
        release_collection_arrays(element_dim, collection_arrays);
        return NULL;
    }
    int8_t *const orientation = PyMem_Malloc(element_dim * sizeof(*orientation));
    if (!orientation)
    {
        topo_obj_immersions_free(element_dim, immersions, &PYTHON_ALLOCATOR);
        release_collection_arrays(element_dim, collection_arrays);
        return NULL;
    }
    const unsigned boundary_immersion_index = face_dim == 0 ? 0 : face_dim - 1;
    topo_status = topo_obj_boundary_orientation(immersions + boundary_immersion_index, element_dim,
                                                (uint64_t)boundary_id, (uint64_t)element_id, orientation);
    topo_obj_immersions_free(element_dim, immersions, &PYTHON_ALLOCATOR);
    release_collection_arrays(element_dim, collection_arrays);
    if (topo_status != TOPO_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Boundary %zd is not present in element %zd: %s (%s).", boundary_id, element_id,
                     topo_status_to_str(topo_status), topo_status_msg(topo_status));
        PyMem_Free(orientation);
        return NULL;
    }

    PyObject *face_object = NULL;
    if (make_boundary_map(state, element_map, orientation, element_dim, face_dim, NULL, &face_object) < 0)
    {
        PyMem_Free(orientation);
        return NULL;
    }
    space_map_object *const face_map = (space_map_object *)face_object;
    integration_space_object *face_space = NULL;
    integration_spec_t *canonical_specs = NULL;
    unsigned *canonical_digits = NULL;
    PyArrayObject *pullback = NULL;
    double *surface_weights = NULL;
    constraint_quadrature_t *quadrature_axes = NULL;
    const integration_rule_t **rules = NULL;
    if (make_integration_space(state, face_map, &face_space) < 0)
        goto single_fail;
    const integration_spec_t *const face_specs = face_map->int_specs;
    const unsigned normal_axis = (unsigned)(orientation[0] < 0 ? -orientation[0] : orientation[0]) - 1;
    canonical_specs = PyMem_Malloc(face_dim * sizeof(*canonical_specs));
    if (!canonical_specs)
        goto single_fail;
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const unsigned source_axis =
            (unsigned)(orientation[face_axis + 1] < 0 ? -orientation[face_axis + 1] : orientation[face_axis + 1]) - 1;
        const unsigned source_face_axis = source_axis < normal_axis ? source_axis : source_axis - 1;
        canonical_specs[face_axis] = face_specs[source_face_axis];
    }

    if (create_pullback(face_map, element_dim, face_dim, orientation, order, face_specs, canonical_specs, &pullback) <
        0)
        goto single_fail;
    const size_t point_count = total_points(face_dim, canonical_specs);
    surface_weights = PyMem_Malloc(point_count * sizeof(*surface_weights));
    quadrature_axes = PyMem_Malloc(face_dim * sizeof(*quadrature_axes));
    if (!surface_weights || !quadrature_axes)
    {
        PyMem_Free(surface_weights);
        PyMem_Free(quadrature_axes);
        goto single_fail;
    }
    canonical_digits = PyMem_Malloc(face_dim * sizeof(*canonical_digits));
    if (!canonical_digits)
        goto single_fail;
    for (size_t point = 0; point < point_count; ++point)
    {
        get_digits(face_dim, canonical_specs, point, canonical_digits);
        const size_t source_point = canonical_point_to_source(element_dim, face_dim, orientation, face_specs,
                                                              canonical_specs, canonical_digits);
        surface_weights[point] = fabs(face_map->determinant[source_point]);
    }
    integration_registry_object *const integration_registry =
        (integration_registry_object *)state->registry_integration;
    rules = python_integration_rules_get(face_dim, face_specs, integration_registry->registry);
    if (!rules)
    {
        PyMem_Free(surface_weights);
        PyMem_Free(quadrature_axes);
        goto single_fail;
    }
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const unsigned source_axis =
            (unsigned)(orientation[face_axis + 1] < 0 ? -orientation[face_axis + 1] : orientation[face_axis + 1]) - 1;
        const unsigned source_face_axis = source_axis < normal_axis ? source_axis : source_axis - 1;
        const integration_rule_t *const rule = rules[source_face_axis];
        quadrature_axes[face_axis] = (constraint_quadrature_t){.count = rule->n_nodes,
                                                               .nodes = integration_rule_nodes_const(rule),
                                                               .weights = integration_rule_weights_const(rule)};
    }

    const constraint_kform_spec_t test_descriptor = {
        .ndim = face_dim, .order = order, .basis_specs = test_spec->function_space->specs};
    const constraint_element_side_t side_descriptor = {
        .ndim = element_dim, .basis_specs = element_spec->function_space->specs, .orientation = orientation};
    const constraint_trace_pullback_t pullback_descriptor = {.physical_component_count = (unsigned)Py_SIZE(element_map),
                                                             .point_count = point_count,
                                                             .values = PyArray_DATA(pullback)};
    const constraint_face_quadrature_t face_quadrature = {
        .ndim = face_dim, .axes = quadrature_axes, .point_count = point_count};
    size_t row_count;
    size_t entry_count;
    constraint_status_t constraint_status =
        constraint_physical_side_required(&test_descriptor, &side_descriptor, &row_count, &entry_count);
    if (constraint_status != CONSTRAINT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Could not size boundary constraints: %s.",
                     constraint_status_to_str(constraint_status));
        python_integration_rules_release(face_dim, rules, integration_registry->registry);
        rules = NULL;
        PyMem_Free(surface_weights);
        PyMem_Free(quadrature_axes);
        goto single_fail;
    }
    const npy_intp row_dims[1] = {(npy_intp)(row_count + 1)};
    const npy_intp entry_dims[1] = {(npy_intp)entry_count};
    PyArrayObject *row_array = (PyArrayObject *)PyArray_SimpleNew(1, row_dims, NPY_UINTP);
    PyArrayObject *component_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINT32);
    PyArrayObject *dof_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_UINTP);
    PyArrayObject *coefficient_array = (PyArrayObject *)PyArray_SimpleNew(1, entry_dims, NPY_DOUBLE);
    constraint_entry_t *entries = PyMem_Malloc(entry_count * sizeof(*entries));
    if (!row_array || !component_array || !dof_array || !coefficient_array || !entries)
    {
        Py_XDECREF(row_array);
        Py_XDECREF(component_array);
        Py_XDECREF(dof_array);
        Py_XDECREF(coefficient_array);
        PyMem_Free(entries);
        python_integration_rules_release(face_dim, rules, integration_registry->registry);
        rules = NULL;
        PyMem_Free(surface_weights);
        PyMem_Free(quadrature_axes);
        goto single_fail;
    }
    size_t actual_rows;
    size_t actual_entries;
    constraint_status = constraint_physical_side_assemble(
        &test_descriptor, &side_descriptor, &face_quadrature, surface_weights, &pullback_descriptor, row_count + 1,
        (size_t *)PyArray_DATA(row_array), entry_count, entries, &actual_rows, &actual_entries);
    python_integration_rules_release(face_dim, rules, integration_registry->registry);
    rules = NULL;
    PyMem_Free(surface_weights);
    PyMem_Free(quadrature_axes);
    if (constraint_status != CONSTRAINT_SUCCESS)
    {
        Py_XDECREF(row_array);
        Py_XDECREF(component_array);
        Py_XDECREF(dof_array);
        Py_XDECREF(coefficient_array);
        PyMem_Free(entries);
        goto single_fail;
    }
    for (size_t i = 0; i < actual_entries; ++i)
    {
        ((uint32_t *)PyArray_DATA(component_array))[i] = entries[i].component;
        ((size_t *)PyArray_DATA(dof_array))[i] = entries[i].local_dof;
        ((double *)PyArray_DATA(coefficient_array))[i] = entries[i].coefficient;
    }
    PyMem_Free(entries);
    Py_DECREF(pullback);
    Py_DECREF(face_space);
    Py_DECREF(face_object);
    PyMem_Free(canonical_specs);
    PyMem_Free(canonical_digits);
    PyMem_Free(orientation);
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

single_fail:
    if (rules)
        python_integration_rules_release(face_dim, rules, integration_registry->registry);
    Py_XDECREF(pullback);
    Py_XDECREF(face_space);
    Py_XDECREF(face_object);
    PyMem_Free(canonical_specs);
    PyMem_Free(canonical_digits);
    PyMem_Free(surface_weights);
    PyMem_Free(quadrature_axes);
    PyMem_Free(orientation);
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
    {},
};
