/**
 * @file direct.c
 * @brief Implementation of the direct shared-DoF continuity formulation.
 *
 * See direct.h for the formulation. Per element the objects' boundary mass blocks are stacked into one constraint
 * matrix; the QR of its transpose splits the element's DoFs into the object-coefficient part and an orthogonal
 * complement of element-private DoFs. The transfer is emitted once per nonzero, grouped by ascending local and
 * global index, with coefficients below a relative roundoff threshold dropped.
 */

#include "direct.h"

#include <cutl/allocators.h>
#include <cutl/iterators/combination_iterator.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../operations/matrices.h"
#include "constraint_common.h"

/**
 * @brief Coefficients at most 2^-#DIRECT_PRUNE_SHIFT of their part's maximum magnitude count as roundoff and
 *        are dropped.
 */
enum
{
    DIRECT_PRUNE_SHIFT = 40,
};

static void direct_continuity_walk(const direct_continuity_request_t *request, const direct_continuity_plan_t *plan,
                                   direct_continuity_work_t *work, direct_entry_fn emit, void *param);

/**
 * @brief Test function count one object axis contributes to a component.
 *
 * An active covector axis reads the order-one basis; any other axis reads the leading functions of the full
 * basis with the last #SKIPPED_BASIS dropped. This is `boundary_mass_row_axis_counts` as a bare count.
 */
static unsigned direct_axis_test_count(const basis_spec_t spec, const bool active)
{
    if (active)
    {
        return spec.order;
    }
    return spec.order + 1u > SKIPPED_BASIS ? spec.order + 1u - SKIPPED_BASIS : 0u;
}

/**
 * @brief Test DoFs of one object's block for one component of a k-form.
 *
 * @param basis [dim] Common Legendre test basis of the object's axes.
 * @param dim Object dimension.
 * @param order k-form order.
 * @param component_axes Sorted covector axes of the component, in the object's frame.
 */
static size_t direct_object_component_dofs(const basis_spec_t *const basis, const unsigned dim, const unsigned order,
                                           const uint8_t *const component_axes)
{
    size_t dofs = 1;
    unsigned next_axis = 0;
    for (unsigned j = 0; j < dim; ++j)
    {
        const bool active = next_axis < order && component_axes[next_axis] == j;
        dofs *= direct_axis_test_count(basis[j], active);
        next_axis += active ? 1u : 0u;
    }
    return dofs;
}

/**
 * @brief Test DoFs of one object's whole block, over all of its components.
 *
 * An order past the object's dimension has no trace, so such an object owns nothing.
 */
static size_t direct_object_block_size(const basis_spec_t *const basis, const unsigned dim, const unsigned order)
{
    if (order > dim)
    {
        return 0;
    }
    size_t total = 0;
    uint8_t axes[UINT8_MAX];
    const unsigned component_count = combination_total_count((uint8_t)dim, (uint8_t)order);
    for (unsigned component = 0; component < component_count; ++component)
    {
        combination_set_to_index((uint8_t)dim, (uint8_t)order, axes, component);
        total += direct_object_component_dofs(basis, dim, order, axes);
    }
    return total;
}

/**
 * @brief Number of objects of every dimension below `ndim` the mesh has.
 */
static uint64_t direct_entity_total(const topo_mesh_t *const mesh, const unsigned ndim)
{
    uint64_t total = 0;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        total += mesh->immersions[dim].object_count;
    }
    return total;
}

/**
 * @brief Whether one object takes part in the map: incident to at least one element, and of a dimension that
 *        traces the k-form.
 */
static bool direct_object_carries(const direct_continuity_request_t *const request, const unsigned dim,
                                  const uint64_t incident)
{
    return incident > 0 && request->order <= dim;
}

/**
 * @brief Number of (object, incident element) pairs the mesh has.
 */
static uint64_t direct_pair_total(const direct_continuity_request_t *const request)
{
    const topo_mesh_t *const mesh = request->mesh;
    uint64_t total = 0;
    for (unsigned dim = 0; dim < request->ndim; ++dim)
    {
        for (uint64_t object = 0; object < mesh->immersions[dim].object_count; ++object)
        {
            uint64_t incident;
            const uint64_t *ids;
            const int8_t *records;
            topo_obj_immersion_of_object(mesh->immersions + dim, object, &incident, &ids, &records);
            if (direct_object_carries(request, dim, incident))
            {
                total += incident;
            }
        }
    }
    return total;
}

/**
 * @brief Largest number of elements one object's immersion records.
 */
static uint64_t direct_max_incident(const direct_continuity_request_t *const request)
{
    const topo_mesh_t *const mesh = request->mesh;
    uint64_t max_incident = 1;
    for (unsigned dim = 0; dim < request->ndim; ++dim)
    {
        for (uint64_t object = 0; object < mesh->immersions[dim].object_count; ++object)
        {
            uint64_t incident;
            const uint64_t *ids;
            const int8_t *records;
            topo_obj_immersion_of_object(mesh->immersions + dim, object, &incident, &ids, &records);
            if (direct_object_carries(request, dim, incident))
            {
                max_incident = incident > max_incident ? incident : max_incident;
            }
        }
    }
    return max_incident;
}

/**
 * @brief Largest basis order any element axis carries.
 */
static unsigned direct_max_basis_order(const direct_continuity_request_t *const request)
{
    unsigned order = 1;
    for (uint64_t element = 0; element < request->mesh->element_count; ++element)
    {
        for (unsigned axis = 0; axis < request->ndim; ++axis)
        {
            const unsigned axis_order = request->elements[element]->basis[axis].order;
            order = axis_order > order ? axis_order : order;
        }
    }
    return order;
}

/**
 * @brief Bound on one element's stacked constraint rows.
 *
 * Every incident object's common orders are minima over its incident elements, so they never exceed the
 * element's own orders; the element touches at most `3^ndim - 1` objects, each of dimension at most `ndim - 1`.
 */
static size_t direct_row_bound(const direct_continuity_request_t *const request)
{
    const unsigned ndim = request->ndim;
    const unsigned order = request->order;
    if (order >= ndim)
    {
        // No object of dimension below ndim carries a trace of the top order.
        return 1;
    }
    uint64_t objects = 1;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        objects *= 3u;
    }
    uint64_t combinations = 1;
    for (unsigned i = 0; i < order; ++i)
    {
        combinations = combinations * (ndim - 1u - i) / (i + 1u);
    }
    uint64_t window = 1;
    const unsigned basis_order = direct_max_basis_order(request);
    for (unsigned dim = 1; dim < ndim; ++dim)
    {
        window *= basis_order;
    }
    return (size_t)((objects - 1u) * combinations * window);
}

fdg_result_t direct_continuity_prepare(const direct_continuity_request_t *const request,
                                       direct_continuity_work_t *const work, direct_continuity_plan_t *const plan)
{
    const unsigned ndim = request->ndim;
    const unsigned order = request->order;
    const topo_mesh_t *const mesh = request->mesh;
    CUTL_ASSERT(ndim > 0, "The element dimension must be at least one.");
    CUTL_ASSERT(order <= ndim, "The k-form order %u exceeds the element dimension %u.", order, ndim);

    plan->ndim = ndim;
    plan->order = order;
    plan->element_count = mesh->element_count;
    plan->entity_count = direct_entity_total(mesh, ndim);
    plan->pair_count = direct_pair_total(request);
    plan->mesh = mesh;
    plan->elements = request->elements;
    plan->basis_registry = request->basis_registry;
    plan->integration_registry = request->integration_registry;

    uint64_t running = 0;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        plan->entity_dim_offsets[dim] = running;
        running += mesh->immersions[dim].object_count;
    }
    plan->entity_dim_offsets[ndim] = running;
    for (uint64_t element = 0; element <= mesh->element_count; ++element)
    {
        plan->element_rows[element] = 0;
    }

    // Fetched references start NULL so a failed prepare still releases cleanly.
    const size_t entity_slots = (size_t)plan->entity_count * ndim;
    for (size_t slot = 0; slot < entity_slots; ++slot)
    {
        plan->entity_rules[slot] = NULL;
        plan->entity_sets[slot] = NULL;
        plan->entity_sets_lower[slot] = NULL;
    }
    const size_t pair_slots = (size_t)plan->pair_count * ndim;
    for (size_t slot = 0; slot < pair_slots; ++slot)
    {
        plan->pair_element_sets[slot] = NULL;
        plan->pair_element_sets_lower[slot] = NULL;
        plan->pair_element_endpoints[slot] = NULL;
        plan->pair_element_endpoints_lower[slot] = NULL;
    }

    size_t block_running = 0;
    size_t pair_running = 0;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        const unsigned object_count = mesh->immersions[dim].object_count;
        for (unsigned object = 0; object < object_count; ++object)
        {
            const size_t index = direct_entity_index(plan, dim, object);
            basis_spec_t *const basis = plan->entity_basis + index * ndim;
            integration_spec_t *const integration = plan->entity_integration + index * ndim;
            basis_spec_t *const lower = plan->entity_lower_basis + index * ndim;
            uint64_t incident;
            const uint64_t *incident_ids;
            const int8_t *records;
            topo_obj_immersion_of_object(mesh->immersions + dim, object, &incident, &incident_ids, &records);
            // A mesh may declare objects no element touches; such an object owns nothing. An object of
            // dimension under the traced order has no trace either, so it constrains nothing and owns no DoFs.
            plan->entity_block_offsets[index] = block_running;
            if (!direct_object_carries(request, dim, incident))
            {
                continue;
            }
            for (uint64_t i = 0; i < incident; ++i)
            {
                const kform_spec_t *const spec = request->elements[incident_ids[i]];
                work->views[i] = (boundary_element_space_t){.order = order,
                                                            .orientation = records + ndim * i,
                                                            .basis = spec->basis,
                                                            .integration = work->view_integration + i * ndim};
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    // The quadrature is fed per axis at the element's own order, which integrates every product
                    // the pairing forms exactly; the merge raises the object rule to the most accurate view.
                    work->view_integration[i * ndim + axis] = (integration_spec_t){
                        .type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = spec->basis[axis].order};
                }
            }
            constraint_common_space_merge(ndim, dim, incident, work->views, basis, integration);
            for (unsigned j = 0; j < dim; ++j)
            {
                // The test space is hierarchic, which is what the window trim and the Gram conditioning rely on.
                basis[j].type = BASIS_LEGENDRE;
                lower[j] =
                    (basis_spec_t){.type = BASIS_LEGENDRE, .order = basis[j].order > 0 ? basis[j].order - 1u : 0u};
                // Reported, not asserted: a release build drops the check.
                if (basis[j].order == 0)
                {
                    return FDG_ERROR_NOT_IN_DOMAIN;
                }
            }
            const integration_rule_t **const rules = plan->entity_rules + index * ndim;
            const basis_set_t **const sets = plan->entity_sets + index * ndim;
            const basis_set_t **const sets_lower = plan->entity_sets_lower + index * ndim;
            if (dim > 0)
            {
                fdg_result_t res =
                    integration_rule_registry_get_rules(request->integration_registry, dim, integration, rules);
                if (res != FDG_SUCCESS)
                {
                    return res;
                }
                res = basis_set_registry_get_basis_sets(request->basis_registry, dim, sets, rules, basis);
                if (res != FDG_SUCCESS)
                {
                    return res;
                }
                if (order > 0)
                {
                    res = basis_set_registry_get_basis_sets(request->basis_registry, dim, sets_lower, rules, lower);
                    if (res != FDG_SUCCESS)
                    {
                        return res;
                    }
                }
            }
            const size_t rows = direct_object_block_size(basis, dim, order);
            for (uint64_t i = 0; i < incident; ++i)
            {
                const uint64_t element = incident_ids[i];
                const int8_t *const record = records + ndim * i;
                const kform_spec_t *const spec = request->elements[element];
                const size_t pair = pair_running++;
                plan->pair_entities[pair] = index;
                plan->pair_records[pair] = record;
                plan->pair_rows[pair] = (size_t)plan->element_rows[element];
                plan->element_rows[element] += rows;
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    work->axis_fixed[axis] = false;
                    work->axis_slot[axis] = 0;
                }
                for (unsigned face_axis = 0; face_axis < dim; ++face_axis)
                {
                    const int8_t mapping = record[ndim - dim + face_axis];
                    work->axis_slot[constraint_orientation_axis(mapping)] = face_axis;
                }
                for (unsigned fixed_axis = 0; fixed_axis < ndim - dim; ++fixed_axis)
                {
                    const int8_t mapping = record[fixed_axis];
                    work->axis_fixed[constraint_orientation_axis(mapping)] = true;
                }
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    work->element_rules[axis] =
                        dim > 0
                            ? plan->entity_rules[index * ndim + (work->axis_fixed[axis] ? 0 : work->axis_slot[axis])]
                            : NULL;
                }
                basis_spec_t *const element_lower = plan->pair_element_lower_specs + pair * ndim;
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    element_lower[axis] =
                        (basis_spec_t){.type = spec->basis[axis].type,
                                       .order = spec->basis[axis].order > 0 ? spec->basis[axis].order - 1u : 0u};
                }
                const basis_set_t **const element_sets = plan->pair_element_sets + pair * ndim;
                const basis_set_t **const element_sets_lower = plan->pair_element_sets_lower + pair * ndim;
                const basis_endpoint_set_t **const endpoints = plan->pair_element_endpoints + pair * ndim;
                const basis_endpoint_set_t **const endpoints_lower = plan->pair_element_endpoints_lower + pair * ndim;
                if (dim > 0)
                {
                    fdg_result_t res = basis_set_registry_get_basis_sets(request->basis_registry, ndim, element_sets,
                                                                         work->element_rules, spec->basis);
                    if (res != FDG_SUCCESS)
                    {
                        return res;
                    }
                    if (order > 0)
                    {
                        res = basis_set_registry_get_basis_sets(request->basis_registry, ndim, element_sets_lower,
                                                                work->element_rules, element_lower);
                        if (res != FDG_SUCCESS)
                        {
                            return res;
                        }
                    }
                }
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    if (!work->axis_fixed[axis])
                    {
                        continue;
                    }
                    fdg_result_t res = basis_set_registry_get_basis_endpoints(request->basis_registry, endpoints + axis,
                                                                              spec->basis[axis]);
                    if (res != FDG_SUCCESS)
                    {
                        return res;
                    }
                    if (order > 0 && spec->basis[axis].order > 0)
                    {
                        res = basis_set_registry_get_basis_endpoints(request->basis_registry, endpoints_lower + axis,
                                                                     element_lower[axis]);
                        if (res != FDG_SUCCESS)
                        {
                            return res;
                        }
                    }
                }
            }
            block_running += rows;
        }
    }
    plan->entity_block_offsets[plan->entity_count] = block_running;

    // Per-element pair lists: pairs were visited in canonical object order, so counting per element and filling
    // in a second canonical pass keeps every list sorted; element_interior_offsets carries the counts meanwhile.
    for (uint64_t element = 0; element < mesh->element_count; ++element)
    {
        plan->element_interior_offsets[element] = 0;
    }
    pair_running = 0;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        for (uint64_t object = 0; object < mesh->immersions[dim].object_count; ++object)
        {
            uint64_t incident;
            const uint64_t *incident_ids;
            const int8_t *records;
            topo_obj_immersion_of_object(mesh->immersions + dim, object, &incident, &incident_ids, &records);
            if (!direct_object_carries(request, dim, incident))
            {
                continue;
            }
            for (uint64_t i = 0; i < incident; ++i)
            {
                plan->element_interior_offsets[incident_ids[i]] += 1;
            }
        }
    }
    size_t pair_cursor = 0;
    for (uint64_t element = 0; element < mesh->element_count; ++element)
    {
        const size_t count = (size_t)plan->element_interior_offsets[element];
        plan->element_interior_offsets[element] = 0;
        plan->element_object_offsets[element] = pair_cursor;
        pair_cursor += count;
    }
    plan->element_object_offsets[mesh->element_count] = pair_cursor;
    pair_running = 0;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        for (uint64_t object = 0; object < mesh->immersions[dim].object_count; ++object)
        {
            uint64_t incident;
            const uint64_t *incident_ids;
            const int8_t *records;
            topo_obj_immersion_of_object(mesh->immersions + dim, object, &incident, &incident_ids, &records);
            if (!direct_object_carries(request, dim, incident))
            {
                continue;
            }
            for (uint64_t i = 0; i < incident; ++i)
            {
                const uint64_t element = incident_ids[i];
                plan->element_pair_slots[plan->element_object_offsets[element] +
                                         (size_t)plan->element_interior_offsets[element]] = pair_running;
                plan->element_interior_offsets[element] += 1;
                pair_running += 1;
            }
        }
    }

    size_t dof_running = 0;
    // Element-private DoFs come after every object's block.
    size_t interior_running = block_running;
    const size_t row_bound = direct_row_bound(request);
    for (uint64_t element = 0; element < mesh->element_count; ++element)
    {
        const kform_spec_t *const spec = request->elements[element];
        const size_t rows = (size_t)plan->element_rows[element];
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (spec->basis[axis].order == 0)
            {
                return FDG_ERROR_NOT_IN_DOMAIN;
            }
        }
        CUTL_ASSERT(rows <= row_bound, "Element %llu stacks %zu constraint rows, past the %zu bound.",
                    (unsigned long long)element, rows, row_bound);
        plan->element_dof_offsets[element] = dof_running;
        dof_running += kform_spec_total_dofs(spec);
        plan->element_interior_offsets[element] = interior_running;
        interior_running += kform_spec_total_dofs(spec) - rows;
    }
    plan->element_dof_offsets[mesh->element_count] = dof_running;
    plan->element_interior_offsets[mesh->element_count] = interior_running;
    // The row tallies become offsets, which is the form the build reads.
    size_t row_running = 0;
    for (uint64_t element = 0; element < mesh->element_count; ++element)
    {
        const size_t rows = (size_t)plan->element_rows[element];
        plan->element_rows[element] = row_running;
        row_running += rows;
    }
    plan->element_rows[mesh->element_count] = row_running;

    plan->layout.element_count = mesh->element_count;
    plan->layout.entry_count = 0;
    plan->layout.entity_count = plan->entity_count;
    plan->layout.element_dof_count = dof_running;
    plan->layout.global_dof_count = interior_running;
    return FDG_SUCCESS;
}

/**
 * @brief Count one nonzero of the transfer.
 */
static void direct_count_entry(void *const param, const size_t local, const size_t global, const double value)
{
    (void)local;
    (void)global;
    (void)value;
    ++*(size_t *)param;
}

void direct_continuity_layout(const direct_continuity_request_t *const request, direct_continuity_plan_t *const plan,
                              direct_continuity_work_t *const work, direct_continuity_layout_t *const out_layout)
{
    // The nonzero count is the last thing known; the plan keeps it.
    direct_continuity_walk(request, plan, work, direct_count_entry, &plan->layout.entry_count);
    *out_layout = plan->layout;
}

/**
 * @brief State of the transfer's write-back.
 */
typedef struct
{
    size_t *offsets;
    size_t *index;
    double *value;
    size_t entry;
    size_t dof;
    size_t current; ///< Element-local DoF the entries being written belong to.
} direct_build_state_t;

/**
 * @brief Write one nonzero of the transfer.
 */
static void direct_write_entry(void *const param, const size_t local, const size_t global, const double value)
{
    direct_build_state_t *const state = param;
    // A change of local index starts a row block; one DoF may own several entries.
    if (local != state->current)
    {
        CUTL_ASSERT(state->dof == 0 || local > state->current, "The transfer wrote element DoF %zu after %zu.", local,
                    state->current);
        state->current = local;
        state->offsets[local] = state->entry;
        state->dof += 1;
    }
    state->index[state->entry] = global;
    state->value[state->entry] = value;
    state->entry += 1;
}

void direct_continuity_build(const direct_continuity_request_t *const request,
                             const direct_continuity_plan_t *const plan, direct_continuity_work_t *const work,
                             size_t *const entry_offsets, size_t *const entry_index, double *const entry_value)
{
    direct_build_state_t state = {
        .offsets = entry_offsets, .index = entry_index, .value = entry_value, .current = (size_t)-1};
    direct_continuity_walk(request, plan, work, direct_write_entry, &state);
    CUTL_ASSERT(state.dof == plan->layout.element_dof_count, "The transfer wrote %zu of the %zu element DoFs.",
                state.dof, plan->layout.element_dof_count);
    entry_offsets[plan->layout.element_dof_count] = state.entry;
    CUTL_ASSERT(state.entry == plan->layout.entry_count, "The transfer wrote %zu of the %zu expected nonzeros.",
                state.entry, plan->layout.entry_count);
}

void direct_continuity_scatter(const direct_continuity_plan_t *const plan, const size_t *const entry_offsets,
                               const size_t *const entry_index, const double *const entry_value, const size_t element,
                               const double *const matrix, const size_t stride, double *const out,
                               const size_t out_stride, const double factor)
{
    const size_t local_base = plan->element_dof_offsets[element];
    const size_t local_count = plan->element_dof_offsets[element + 1] - local_base;
    for (size_t i = 0; i < local_count; ++i)
    {
        const size_t row_from = entry_offsets[local_base + i];
        const size_t row_to = entry_offsets[local_base + i + 1];
        for (size_t j = 0; j < local_count; ++j)
        {
            const double value = matrix[i * stride + j];
            if (value == 0.0)
            {
                continue;
            }
            const size_t col_from = entry_offsets[local_base + j];
            const size_t col_to = entry_offsets[local_base + j + 1];
            for (size_t row = row_from; row < row_to; ++row)
            {
                const double scaled = factor * value * entry_value[row];
                for (size_t col = col_from; col < col_to; ++col)
                {
                    out[entry_index[row] * out_stride + entry_index[col]] += scaled * entry_value[col];
                }
            }
        }
    }
}

/**
 * @brief Triplets one element's transfer emits, counting its transfer entries.
 */
static size_t direct_element_triplet_count(const direct_continuity_plan_t *const plan,
                                           const size_t *const entry_offsets, const uint64_t element)
{
    const size_t local_base = plan->element_dof_offsets[element];
    const size_t local_count = plan->element_dof_offsets[element + 1] - local_base;
    size_t entries = 0;
    for (size_t dof = 0; dof < local_count; ++dof)
    {
        entries += entry_offsets[local_base + dof + 1] - entry_offsets[local_base + dof];
    }
    return entries * entries;
}

/**
 * @brief Write one element's triplets in local DoF pair, entry pair order.
 *
 * @return The number of triplets written.
 */
static size_t direct_element_scatter_triplets(const size_t *const entry_offsets, const size_t *const entry_index,
                                              const double *const entry_value, const double *const block,
                                              const size_t local_count, size_t *const out_rows, size_t *const out_cols,
                                              double *const out_values)
{
    size_t written = 0;
    for (size_t i = 0; i < local_count; ++i)
    {
        const size_t row_from = entry_offsets[i];
        const size_t row_to = entry_offsets[i + 1];
        for (size_t j = 0; j < local_count; ++j)
        {
            const double value = block[i * local_count + j];
            const size_t col_from = entry_offsets[j];
            const size_t col_to = entry_offsets[j + 1];
            for (size_t row = row_from; row < row_to; ++row)
            {
                const size_t global_row = entry_index[row];
                const double left = entry_value[row] * value;
                for (size_t col = col_from; col < col_to; ++col)
                {
                    out_rows[written] = global_row;
                    out_cols[written] = entry_index[col];
                    out_values[written] = left * entry_value[col];
                    written += 1;
                }
            }
        }
    }
    return written;
}

size_t direct_continuity_triplet_count(const direct_continuity_plan_t *const plan, const size_t *const entry_offsets)
{
    size_t total = 0;
    for (uint64_t element = 0; element < plan->element_count; ++element)
    {
        total += direct_element_triplet_count(plan, entry_offsets, element);
    }
    return total;
}

/**
 * @brief Worker count for one scatter: 0 requests the OpenMP default.
 */
static unsigned direct_scatter_threads(const unsigned n_threads)
{
#ifdef _OPENMP
    if (n_threads == 0u)
    {
        return (unsigned)omp_get_max_threads();
    }
    return n_threads;
#else
    (void)n_threads;
    return 1u;
#endif
}

void direct_continuity_scatter_triplets(const direct_continuity_plan_t *const plan, const size_t *const entry_offsets,
                                        const size_t *const entry_index, const double *const entry_value,
                                        const double *const local_matrices, const unsigned n_threads,
                                        size_t *const out_rows, size_t *const out_cols, double *const out_values)
{
    const uint64_t element_count = plan->element_count;
    // One block holds the per-element triplet starts and the per-element matrix block starts.
    size_t *const scratch = cutl_alloc(&CUTL_STD_ALLOCATOR, 2u * (size_t)(element_count + 1u) * sizeof(*scratch));
    if (scratch == NULL)
    {
        // No scratch, no static partition: one running cursor keeps the output deterministic.
        size_t cursor = 0;
        size_t block_base = 0;
        for (uint64_t element = 0; element < element_count; ++element)
        {
            const size_t local_base = plan->element_dof_offsets[element];
            const size_t local_count = plan->element_dof_offsets[element + 1] - local_base;
            cursor += direct_element_scatter_triplets(entry_offsets + local_base, entry_index, entry_value,
                                                      local_matrices + block_base, local_count, out_rows + cursor,
                                                      out_cols + cursor, out_values + cursor);
            block_base += local_count * local_count;
        }
        return;
    }
    size_t *const starts = scratch;
    size_t *const blocks = scratch + element_count + 1u;

    starts[0] = 0;
    blocks[0] = 0;
#pragma omp parallel for num_threads(direct_scatter_threads(n_threads)) schedule(static)
    for (uint64_t element = 0; element < element_count; ++element)
    {
        starts[element + 1] = direct_element_triplet_count(plan, entry_offsets, element);
        const size_t local_base = plan->element_dof_offsets[element];
        blocks[element + 1] = (plan->element_dof_offsets[element + 1] - local_base) *
                              (plan->element_dof_offsets[element + 1] - local_base);
    }
    for (uint64_t element = 0; element < element_count; ++element)
    {
        starts[element + 1] += starts[element];
        blocks[element + 1] += blocks[element];
    }
#pragma omp parallel for num_threads(direct_scatter_threads(n_threads)) schedule(static)
    for (uint64_t element = 0; element < element_count; ++element)
    {
        const size_t local_base = plan->element_dof_offsets[element];
        direct_element_scatter_triplets(entry_offsets + local_base, entry_index, entry_value,
                                        local_matrices + blocks[element],
                                        plan->element_dof_offsets[element + 1] - local_base, out_rows + starts[element],
                                        out_cols + starts[element], out_values + starts[element]);
    }
    cutl_dealloc(&CUTL_STD_ALLOCATOR, scratch);
}

void direct_continuity_plan_release(const direct_continuity_plan_t *const plan)
{
    for (uint64_t index = 0; index < plan->entity_count; ++index)
    {
        unsigned dim = 0;
        while (dim < plan->ndim && plan->entity_dim_offsets[dim + 1] <= index)
        {
            dim += 1;
        }
        for (unsigned j = 0; j < dim; ++j)
        {
            const size_t slot = index * plan->ndim + j;
            if (plan->entity_rules[slot] != NULL)
            {
                integration_rule_registry_release_rule(plan->integration_registry, plan->entity_rules[slot]);
                plan->entity_rules[slot] = NULL;
            }
            if (plan->entity_sets[slot] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry, plan->entity_sets[slot]);
                plan->entity_sets[slot] = NULL;
            }
            if (plan->entity_sets_lower[slot] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry, plan->entity_sets_lower[slot]);
                plan->entity_sets_lower[slot] = NULL;
            }
        }
    }
    for (uint64_t pair = 0; pair < plan->pair_count; ++pair)
    {
        for (unsigned axis = 0; axis < plan->ndim; ++axis)
        {
            const size_t slot = pair * plan->ndim + axis;
            if (plan->pair_element_endpoints_lower[slot] != NULL)
            {
                basis_set_registry_release_basis_endpoints(plan->basis_registry,
                                                           plan->pair_element_endpoints_lower[slot]);
                plan->pair_element_endpoints_lower[slot] = NULL;
            }
            if (plan->pair_element_endpoints[slot] != NULL)
            {
                basis_set_registry_release_basis_endpoints(plan->basis_registry, plan->pair_element_endpoints[slot]);
                plan->pair_element_endpoints[slot] = NULL;
            }
            if (plan->pair_element_sets_lower[slot] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry, plan->pair_element_sets_lower[slot]);
                plan->pair_element_sets_lower[slot] = NULL;
            }
            if (plan->pair_element_sets[slot] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry, plan->pair_element_sets[slot]);
                plan->pair_element_sets[slot] = NULL;
            }
        }
    }
}

/**
 * @brief Assemble, eliminate and emit one element's transfer.
 */
static void direct_eliminate_element(const direct_continuity_plan_t *const plan, direct_continuity_work_t *const work,
                                     direct_entry_fn emit, void *const param, const uint64_t element)
{
    const unsigned ndim = plan->ndim;
    const unsigned order = plan->order;
    const unsigned component_count = combination_total_count((uint8_t)ndim, (uint8_t)order);
    const kform_spec_t *const spec = plan->elements[element];
    kform_spec_component_offsets(spec, component_count + 1u, work->component_offsets);
    const size_t local_base = plan->element_dof_offsets[element];
    const size_t interior_base = plan->element_interior_offsets[element];
    const size_t dof_count = plan->element_dof_offsets[element + 1] - local_base;
    const size_t row_start = plan->element_rows[element];
    const size_t rows = plan->element_rows[element + 1] - row_start;
    if (rows == 0)
    {
        // A top-order form has no trace on any object, so every DoF stays element-private.
        for (size_t i = 0; i < dof_count; ++i)
        {
            emit(param, local_base + i, interior_base + i, 1.0);
        }
        return;
    }

    for (size_t value = 0; value < rows * dof_count; ++value)
    {
        work->stacked[value] = 0.0;
    }
    for (size_t value = 0; value < rows * rows; ++value)
    {
        work->b_stacked[value] = 0.0;
    }

    // One object block and its Gram per pair, expanded into the element's own column numbering.
    for (size_t entry = plan->element_object_offsets[element]; entry < plan->element_object_offsets[element + 1];
         ++entry)
    {
        const size_t pair = plan->element_pair_slots[entry];
        const size_t index = (size_t)plan->pair_entities[pair];
        const int8_t *const record = plan->pair_records[pair];
        const size_t pair_row = plan->pair_rows[pair];
        unsigned dim = 0;
        while (plan->entity_dim_offsets[dim + 1] <= index)
        {
            dim += 1;
        }
        const size_t object_rows = plan->entity_block_offsets[index + 1] - plan->entity_block_offsets[index];
        const constraint_boundary_mass_spec_t mass_spec = {.ndim = ndim,
                                                           .bdim = dim,
                                                           .order = order,
                                                           .element_spec = spec,
                                                           .boundary_basis = plan->entity_basis + index * ndim,
                                                           .boundary_integration =
                                                               plan->entity_integration + index * ndim,
                                                           .orientation = record};
        const basis_set_t **const boundary_sets = plan->entity_sets + index * ndim;
        const bool with_lower = order > 0;
        constraint_boundary_mass_work_sizes_t sizes;
        constraint_boundary_mass_work_size(&mass_spec, &work->mass, &sizes);
        constraint_boundary_mass_work_init(&work->mass, &mass_spec, &sizes, work->mass_memory);
        size_t assembled_rows;
        size_t assembled_cols;
        size_t assembled_entries;
        constraint_boundary_mass_layout(&mass_spec, &work->mass, false, &assembled_rows, &assembled_cols,
                                        &assembled_entries);
        CUTL_ASSERT(assembled_rows == object_rows, "Object %llu assembles %zu test rows against a block of %zu.",
                    (unsigned long long)index, assembled_rows, object_rows);
        integration_rule_tensor_weights(dim, plan->entity_rules + index * ndim, work->weights);
        const constraint_boundary_mass_request_t mass_request = {
            .spec = &mass_spec,
            .boundary_basis_sets = boundary_sets,
            .boundary_basis_sets_lower = with_lower ? plan->entity_sets_lower + index * ndim : NULL,
            .element_basis_sets = plan->pair_element_sets + pair * ndim,
            .element_basis_sets_lower = with_lower ? plan->pair_element_sets_lower + pair * ndim : NULL,
            .element_endpoints = plan->pair_element_endpoints + pair * ndim,
            .element_endpoints_lower = with_lower ? plan->pair_element_endpoints_lower + pair * ndim : NULL,
            .point_weights = work->weights,
            .surface_weights = NULL,
            .test_pullback = NULL,
            .element_pullback = NULL,
            .factor = 1.0,
            .work = &work->mass,
            .out_matrix = work->block,
        };
        constraint_boundary_mass_assemble(&mass_request);
        for (unsigned component = 0; component < combination_total_count((uint8_t)dim, (uint8_t)order); ++component)
        {
            const size_t col_dofs = work->mass.col_offsets[component + 1] - work->mass.col_offsets[component];
            const size_t element_column = work->component_offsets[work->mass.element_components[component]];
            for (size_t row = 0; row < object_rows; ++row)
            {
                for (size_t dof = 0; dof < col_dofs; ++dof)
                {
                    work->stacked[(pair_row + row) * dof_count + element_column + dof] =
                        work->block[row * assembled_cols + work->mass.col_offsets[component] + dof];
                }
            }
        }
        constraint_boundary_mass_gram(&mass_spec, boundary_sets,
                                      with_lower ? plan->entity_sets_lower + index * ndim : NULL, work->weights,
                                      &work->mass, work->gram);
        for (size_t row = 0; row < object_rows; ++row)
        {
            for (size_t other = 0; other < object_rows; ++other)
            {
                work->b_stacked[(pair_row + row) * rows + pair_row + other] = work->gram[row * object_rows + other];
            }
        }
    }

    // QR of the transpose: with the convention A = Q^T R, the constrained part is Q_1 R^{-T} B and the free part
    // is the rows of Q past the rank.
    matrix_t transposed = {.rows = (unsigned)dof_count, .cols = (unsigned)rows, .values = work->transposed};
    matrix_t orthogonal = {.rows = (unsigned)dof_count, .cols = (unsigned)dof_count, .values = work->q_matrix};
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t dof = 0; dof < dof_count; ++dof)
        {
            work->transposed[dof * rows + row] = work->stacked[row * dof_count + dof];
        }
    }
    matrix_qr_decompose(&transposed, &orthogonal);
    double max_pivot = 0.0;
    for (size_t row = 0; row < rows; ++row)
    {
        const double pivot = work->transposed[row * rows + row];
        const double magnitude = pivot < 0.0 ? -pivot : pivot;
        max_pivot = magnitude > max_pivot ? magnitude : max_pivot;
    }
    CUTL_ASSERT(max_pivot > 0.0, "The stacked constraints of element %llu vanished.", (unsigned long long)element);
    for (size_t row = 0; row < rows; ++row)
    {
        const double pivot = work->transposed[row * rows + row];
        const double magnitude = pivot < 0.0 ? -pivot : pivot;
        CUTL_ASSERT(magnitude > 1e-10 * max_pivot, "Constraint row %zu of element %llu is rank deficient.", row,
                    (unsigned long long)element);
    }

    // Forward substitution R^T Y = B fills y from the block-diagonal Gram b_stacked.
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t column = 0; column < rows; ++column)
        {
            double value = work->b_stacked[row * rows + column];
            for (size_t inner = 0; inner < row; ++inner)
            {
                value -= work->transposed[inner * rows + row] * work->y[inner * rows + column];
            }
            work->y[row * rows + column] = value / work->transposed[row * rows + row];
        }
    }
    double mapped_max = 0.0;
    for (size_t dof = 0; dof < dof_count; ++dof)
    {
        for (size_t row = 0; row < rows; ++row)
        {
            double value = 0.0;
            for (size_t inner = 0; inner < rows; ++inner)
            {
                value += work->q_matrix[inner * dof_count + dof] * work->y[inner * rows + row];
            }
            work->mapped[dof * rows + row] = value;
            const double magnitude = value < 0.0 ? -value : value;
            mapped_max = magnitude > mapped_max ? magnitude : mapped_max;
        }
    }
    double orthogonal_max = 0.0;
    for (size_t value = 0; value < dof_count * dof_count; ++value)
    {
        const double magnitude = work->q_matrix[value] < 0.0 ? -work->q_matrix[value] : work->q_matrix[value];
        orthogonal_max = magnitude > orthogonal_max ? magnitude : orthogonal_max;
    }
    const double mapped_floor = mapped_max * (double)1.0 / (double)(UINT64_C(1) << DIRECT_PRUNE_SHIFT);
    const double orthogonal_floor = orthogonal_max * (double)1.0 / (double)(UINT64_C(1) << DIRECT_PRUNE_SHIFT);

    const size_t cursor_start = plan->element_object_offsets[element];
    const size_t cursor_end = plan->element_object_offsets[element + 1];
    for (size_t dof = 0; dof < dof_count; ++dof)
    {
        const size_t local = local_base + dof;
        // Every DoF's rows restart at the element's first object, so the pair cursor resets with them.
        size_t cursor = cursor_start;
        for (size_t row = 0; row < rows; ++row)
        {
            const double value = work->mapped[dof * rows + row];
            if (value <= mapped_floor && -value <= mapped_floor)
            {
                continue;
            }
            while (cursor + 1 < cursor_end && plan->pair_rows[plan->element_pair_slots[cursor + 1]] <= row)
            {
                cursor += 1;
            }
            const size_t pair = plan->element_pair_slots[cursor];
            const size_t index = (size_t)plan->pair_entities[pair];
            emit(param, local, plan->entity_block_offsets[index] + (row - plan->pair_rows[pair]), value);
        }
        for (size_t mode = rows; mode < dof_count; ++mode)
        {
            const double value = work->q_matrix[mode * dof_count + dof];
            if (value <= orthogonal_floor && -value <= orthogonal_floor)
            {
                continue;
            }
            emit(param, local, interior_base + (mode - rows), value);
        }
    }
}

/**
 * @brief Enumerate every nonzero of the element-to-global transfer.
 *
 * Walks elements; the elimination emits each DoF's entries grouped by ascending local and ascending global index.
 */
static void direct_continuity_walk(const direct_continuity_request_t *request, const direct_continuity_plan_t *plan,
                                   direct_continuity_work_t *work, direct_entry_fn emit, void *param)
{
    (void)request;
    // The boundary-mass scratch starts as the a-priori sizing block, which every per-pair re-init fits into.
    if (plan->element_count > 0)
    {
        const constraint_boundary_mass_spec_t sizing_spec = {.ndim = plan->ndim,
                                                             .bdim = plan->ndim - 1u,
                                                             .order = plan->order,
                                                             .element_spec = plan->elements[0],
                                                             .boundary_basis = NULL,
                                                             .boundary_integration = NULL,
                                                             .orientation = NULL};
        constraint_boundary_mass_work_init(&work->mass, &sizing_spec, NULL, work->mass_memory);
    }
    for (uint64_t element = 0; element < plan->element_count; ++element)
    {
        direct_eliminate_element(plan, work, emit, param, element);
    }
}

/**
 * @brief Largest element-local DoF count the request carries.
 */
static uint64_t direct_max_element_dofs(const direct_continuity_request_t *const request)
{
    uint64_t max_dofs = 1;
    for (uint64_t element = 0; element < request->mesh->element_count; ++element)
    {
        const uint64_t dofs = kform_spec_total_dofs(request->elements[element]);
        max_dofs = dofs > max_dofs ? dofs : max_dofs;
    }
    return max_dofs;
}

/**
 * @brief Largest number of tensor points one object's common rule spans.
 *
 * The merged rule per axis is fed at Gauss-Legendre of the largest basis order, so its node count is bounded by
 * that order plus one per axis.
 */
static uint64_t direct_max_object_points(const direct_continuity_request_t *const request)
{
    uint64_t points = 1;
    for (unsigned dim = 1; dim < request->ndim; ++dim)
    {
        points *= (uint64_t)direct_max_basis_order(request) + 1u;
    }
    return points;
}

/**
 * @brief Largest per-component DoF count any element k-form spec carries.
 */
static uint64_t direct_max_component_dofs(const direct_continuity_request_t *const request)
{
    const unsigned ndim = request->ndim;
    const unsigned order = request->order;
    uint64_t max_dofs = 1;
    const unsigned component_count = combination_total_count((uint8_t)ndim, (uint8_t)order);
    for (uint64_t element = 0; element < request->mesh->element_count; ++element)
    {
        for (unsigned component = 0; component < component_count; ++component)
        {
            const uint64_t dofs = kform_spec_component_dof_count(request->elements[element], component);
            max_dofs = dofs > max_dofs ? dofs : max_dofs;
        }
    }
    return max_dofs;
}

/**
 * @brief Place every work member into one block, optionally assigning the pointers.
 *
 * Single source of truth for #direct_continuity_work_memory and #direct_continuity_work_init, so the two cannot
 * disagree about padding. With @p work NULL only the byte total accumulates.
 *
 * @return Total bytes for one block.
 */
static size_t direct_work_layout(const direct_continuity_request_t *const request, direct_continuity_work_t *const work,
                                 void *const memory)
{
    const unsigned ndim = request->ndim;
    const unsigned order = request->order;
    const size_t align = _Alignof(max_align_t);
    const uint64_t max_element = direct_max_element_dofs(request);
    const uint64_t max_rows = direct_row_bound(request);
    const uint64_t max_pairs = max_rows * max_element;
    const uint64_t max_points = direct_max_object_points(request);
    const uint64_t point_window = max_points > 0 ? max_points : 1;
    uint64_t component_window = 1;
    for (unsigned dim = 1; dim < ndim; ++dim)
    {
        component_window *= direct_max_basis_order(request);
    }
    const uint64_t max_component = direct_max_component_dofs(request);
    const uint64_t row_values = point_window * component_window;
    const uint64_t col_values = point_window * max_component;
    // The embedded boundary-mass scratch: the a-priori members of the widest dimension, then the value tables.
    const constraint_boundary_mass_spec_t sizing_spec = {.ndim = ndim,
                                                         .bdim = ndim - 1u,
                                                         .order = order,
                                                         .element_spec = NULL,
                                                         .boundary_basis = NULL,
                                                         .boundary_integration = NULL,
                                                         .orientation = NULL};
    const uint64_t mass_apriori = constraint_boundary_mass_work_memory(&sizing_spec, NULL);
    const uint64_t mass_values = 8u * (row_values + col_values + point_window);
    const uint64_t mass_block = mass_apriori + mass_values + 4u * align;
    const uint64_t max_incident = direct_max_incident(request);
    const unsigned components = combination_total_count((uint8_t)ndim, (uint8_t)order);
    size_t cursor = 0;
#define DIRECT_TAKE(member, type, count)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        const size_t bytes = (size_t)(count) * sizeof(type);                                                           \
        cursor = (cursor + align - 1u) & ~(align - 1u);                                                                \
        if (work != NULL)                                                                                              \
        {                                                                                                              \
            work->member = (type *)((unsigned char *)memory + cursor);                                                 \
        }                                                                                                              \
        cursor += bytes;                                                                                               \
    } while (false)
    DIRECT_TAKE(mass_memory, unsigned char, mass_block);
    DIRECT_TAKE(weights, double, point_window);
    DIRECT_TAKE(block, double, max_pairs);
    DIRECT_TAKE(gram, double, max_rows *max_rows);
    DIRECT_TAKE(stacked, double, max_pairs);
    DIRECT_TAKE(b_stacked, double, max_rows *max_rows);
    DIRECT_TAKE(transposed, double, max_pairs);
    DIRECT_TAKE(q_matrix, double, max_element *max_element);
    DIRECT_TAKE(y, double, max_rows *max_rows);
    DIRECT_TAKE(mapped, double, max_pairs);
    DIRECT_TAKE(axis_fixed, bool, ndim);
    DIRECT_TAKE(axis_slot, unsigned, ndim);
    DIRECT_TAKE(element_rules, const integration_rule_t *, ndim);
    DIRECT_TAKE(views, boundary_element_space_t, max_incident);
    DIRECT_TAKE(view_integration, integration_spec_t, max_incident * ndim);
    DIRECT_TAKE(component_offsets, size_t, (size_t)components + 1u);
#undef DIRECT_TAKE
    return cursor;
}

size_t direct_continuity_work_memory(const direct_continuity_request_t *const request)
{
    return direct_work_layout(request, NULL, NULL);
}

void direct_continuity_work_init(direct_continuity_work_t *const work, const direct_continuity_request_t *const request,
                                 void *const memory)
{
    (void)direct_work_layout(request, work, memory);
}

/**
 * @brief Place every plan array into one block, optionally assigning the pointers.
 *
 * The scalars are copied here too, so a release after a failed prepare still sees consistent counts.
 *
 * @return Total bytes for one block.
 */
static size_t direct_plan_layout(const direct_continuity_request_t *const request, direct_continuity_plan_t *const plan,
                                 void *const memory)
{
    const unsigned ndim = request->ndim;
    const uint64_t entity_count = direct_entity_total(request->mesh, ndim);
    const uint64_t pair_count = direct_pair_total(request);
    const uint64_t element_count = request->mesh->element_count;
    if (plan != NULL)
    {
        plan->ndim = ndim;
        plan->order = request->order;
        plan->element_count = element_count;
        plan->entity_count = entity_count;
        plan->pair_count = pair_count;
        plan->mesh = request->mesh;
        plan->elements = request->elements;
        plan->basis_registry = request->basis_registry;
        plan->integration_registry = request->integration_registry;
        plan->layout = (direct_continuity_layout_t){0};
    }
    const size_t align = _Alignof(max_align_t);
    size_t cursor = 0;
#define DIRECT_TAKE(member, type, count)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        const size_t bytes = (size_t)(count) * sizeof(type);                                                           \
        cursor = (cursor + align - 1u) & ~(align - 1u);                                                                \
        if (plan != NULL)                                                                                              \
        {                                                                                                              \
            plan->member = (type *)((unsigned char *)memory + cursor);                                                 \
        }                                                                                                              \
        cursor += bytes;                                                                                               \
    } while (false)
    DIRECT_TAKE(entity_dim_offsets, uint64_t, (size_t)ndim + 1u);
    DIRECT_TAKE(entity_basis, basis_spec_t, (size_t)entity_count * ndim);
    DIRECT_TAKE(entity_integration, integration_spec_t, (size_t)entity_count * ndim);
    DIRECT_TAKE(entity_lower_basis, basis_spec_t, (size_t)entity_count * ndim);
    DIRECT_TAKE(entity_block_offsets, size_t, (size_t)entity_count + 1u);
    DIRECT_TAKE(entity_rules, const integration_rule_t *, (size_t)entity_count *ndim);
    DIRECT_TAKE(entity_sets, const basis_set_t *, (size_t)entity_count *ndim);
    DIRECT_TAKE(entity_sets_lower, const basis_set_t *, (size_t)entity_count *ndim);
    DIRECT_TAKE(pair_entities, uint64_t, (size_t)pair_count);
    DIRECT_TAKE(pair_records, const int8_t *, (size_t)pair_count);
    DIRECT_TAKE(pair_rows, size_t, (size_t)pair_count);
    DIRECT_TAKE(pair_element_sets, const basis_set_t *, (size_t)pair_count *ndim);
    DIRECT_TAKE(pair_element_sets_lower, const basis_set_t *, (size_t)pair_count *ndim);
    DIRECT_TAKE(pair_element_endpoints, const basis_endpoint_set_t *, (size_t)pair_count *ndim);
    DIRECT_TAKE(pair_element_endpoints_lower, const basis_endpoint_set_t *, (size_t)pair_count *ndim);
    DIRECT_TAKE(pair_element_lower_specs, basis_spec_t, (size_t)pair_count * ndim);
    DIRECT_TAKE(element_object_offsets, size_t, (size_t)element_count + 1u);
    DIRECT_TAKE(element_pair_slots, size_t, (size_t)pair_count);
    DIRECT_TAKE(element_rows, size_t, (size_t)element_count + 1u);
    DIRECT_TAKE(element_dof_offsets, size_t, (size_t)element_count + 1u);
    DIRECT_TAKE(element_interior_offsets, size_t, (size_t)element_count + 1u);
#undef DIRECT_TAKE
    return cursor;
}

size_t direct_continuity_plan_memory(const direct_continuity_request_t *const request)
{
    return direct_plan_layout(request, NULL, NULL);
}

void direct_continuity_plan_init(direct_continuity_plan_t *const plan, const direct_continuity_request_t *request,
                                 void *const memory)
{
    (void)direct_plan_layout(request, plan, memory);
}
