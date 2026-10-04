/**
 * @file direct.c
 * @brief Implementation of the direct shared-DoF continuity formulation.
 *
 * See direct.h for the formulation. The walk enumerates each element's DoFs in its own numbering, which is what
 * keeps the transfer's row compression a valid partition. The transfer is separable, so the tensor product of
 * one-dimensional L2 projections is the whole thing; matching axes short-circuit to the identity.
 */

#include <math.h>

#include "direct.h"

#include <cutl/iterators/combination_iterator.h>

#include "constraint_common.h"

/**
 * @brief Largest element dimension the mesh layer supports, and so the widest digit tuple a walk carries.
 */
enum
{
    DIRECT_MAX_AXES = 64,
};

/**
 * @brief Which two one-dimensional bases one object axis pairs.
 *
 * The window counts and offsets follow from the specifications and the active flag, so these identify the
 * operator.
 */
typedef struct
{
    basis_spec_t element; ///< Element-axis basis specification.
    basis_spec_t entity;  ///< Object-axis common basis specification.
    bool active;          ///< The component carries a covector along this axis.
} direct_axis_key_t;

/**
 * @brief One axis' run of basis functions a transfer operates on.
 */
typedef struct
{
    basis_spec_t spec; ///< Basis the run belongs to.
    unsigned offset;   ///< First function of the run.
    unsigned count;    ///< Functions in the run.
} direct_axis_window_t;

/**
 * @brief State of one transfer enumeration.
 */
typedef struct
{
    const direct_continuity_request_t *request;
    const direct_continuity_plan_t *plan;
    direct_continuity_work_t *work;
    const kform_spec_t *spec;     ///< Current element's k-form spec.
    uint64_t element;             ///< Current element.
    unsigned component;           ///< Current element component.
    unsigned dim;                 ///< Current object dimension, equal to `ndim` for the private DoFs.
    uint64_t object_id;           ///< Current object.
    const int8_t *record;         ///< Object's orientation record in the element, NULL for the private DoFs.
    double sign;                  ///< Orientation sign of the current component.
    size_t local;                 ///< Element-local flat index of the DoF being emitted.
    size_t object_base;           ///< Global offset of the object's component block.
    size_t interior;              ///< Next element-private DoF of the current element.
    unsigned lo[DIRECT_MAX_AXES]; ///< First element digit of every axis.
    unsigned hi[DIRECT_MAX_AXES]; ///< Last element digit of every axis.
    direct_entry_fn emit;         ///< Nonzero sink.
    void *param;                  ///< Sink's caller data.
} direct_walk_t;

/**
 * @brief Whether a basis family resolves its functions at distinct nodes.
 *
 * Orthogonal and Bernstein bases do not localize, so the direct numbering cannot place a DoF on them.
 */
static bool direct_basis_is_nodal(const basis_set_type_t type)
{
    return type == BASIS_LAGRANGE_GAUSS_LOBATTO || type == BASIS_LAGRANGE_GAUSS || type == BASIS_LAGRANGE_UNIFORM ||
           type == BASIS_LAGRANGE_CHEBYSHEV_GAUSS;
}

/**
 * @brief Run of basis functions one axis contributes, on either side of a transfer.
 *
 * A covector axis reads the order-one basis; any other axis reads the full basis without its endpoint functions,
 * which live on the faces perpendicular to that axis.
 *
 * Preconditions: an active axis has order at least one, an inactive one at least two.
 */
static direct_axis_window_t direct_axis_window(const basis_spec_t spec, const bool active)
{
    if (active)
    {
        return (direct_axis_window_t){
            .spec = {.type = spec.type, .order = spec.order - 1u}, .offset = 0u, .count = spec.order};
    }
    return (direct_axis_window_t){.spec = spec, .offset = 1u, .count = spec.order - 1u};
}

/**
 * @brief Whether two one-dimensional windows read the same functions.
 */
static bool direct_windows_match(const direct_axis_window_t element, const direct_axis_window_t entity)
{
    return element.count == entity.count && element.offset == entity.offset && element.spec.type == entity.spec.type &&
           element.spec.order == entity.spec.order;
}

/**
 * @brief Solve a small dense system with partial pivoting, overwriting the coefficient matrix.
 *
 * @param rows Leading dimension of `matrix` and `vectors`.
 * @param rhs Number of right-hand sides.
 * @param matrix [rows * rows] Coefficient matrix, overwritten with its factors.
 * @param vectors [rows * rhs] Right-hand sides, overwritten with the solution.
 * @param pivot [rows] Pivot row scratch.
 */
static void direct_dense_solve(const unsigned rows, const unsigned rhs, double *const matrix, double *const vectors,
                               unsigned *const pivot)
{
    for (unsigned col = 0; col < rows; ++col)
    {
        unsigned best = col;
        double best_value = matrix[col * rows + col];
        for (unsigned row = col + 1; row < rows; ++row)
        {
            const double candidate = matrix[row * rows + col];
            const double candidate_magnitude = candidate < 0.0 ? -candidate : candidate;
            const double best_magnitude = best_value < 0.0 ? -best_value : best_value;
            if (candidate_magnitude > best_magnitude)
            {
                best = row;
                best_value = candidate;
            }
        }
        pivot[col] = best;
        if (best != col)
        {
            for (unsigned i = 0; i < rows; ++i)
            {
                const double swap = matrix[col * rows + i];
                matrix[col * rows + i] = matrix[best * rows + i];
                matrix[best * rows + i] = swap;
            }
            for (unsigned i = 0; i < rhs; ++i)
            {
                const double swap = vectors[col * rhs + i];
                vectors[col * rhs + i] = vectors[best * rhs + i];
                vectors[best * rhs + i] = swap;
            }
        }
        const double diagonal = matrix[col * rows + col];
        CUTL_ASSERT(diagonal != 0.0, "The one-dimensional transfer operator is singular.");
        for (unsigned row = col + 1; row < rows; ++row)
        {
            const double factor = matrix[row * rows + col] / diagonal;
            matrix[row * rows + col] = 0.0;
            for (unsigned i = col + 1; i < rows; ++i)
            {
                matrix[row * rows + i] -= factor * matrix[col * rows + i];
            }
            for (unsigned i = 0; i < rhs; ++i)
            {
                vectors[row * rhs + i] -= factor * vectors[col * rhs + i];
            }
        }
    }
    for (unsigned col = rows; col-- > 0;)
    {
        const unsigned row = pivot[col];
        const double diagonal = matrix[row * rows + col];
        for (unsigned i = 0; i < rhs; ++i)
        {
            double value = vectors[row * rhs + i];
            for (unsigned j = col + 1; j < rows; ++j)
            {
                value -= matrix[row * rows + j] * vectors[j * rhs + i];
            }
            vectors[col * rhs + i] = value / diagonal;
        }
    }
}

/**
 * @brief Build one one-dimensional transfer operator: the mixed pairing divided by the object's Gram matrix.
 *
 * The quadrature is exact for every product the two windows form. Matching windows short-circuit to the
 * identity, so the common case needs no quadrature at all.
 *
 * @param request Filled request; the registries are borrowed for the duration of the call only.
 * @param work Caller-provided scratch; the operator lands in its one-dimensional block.
 * @param key Specification of the operator.
 */
static void direct_axis_transfer(const direct_continuity_request_t *const request, direct_continuity_work_t *const work,
                                 const direct_axis_key_t *const key, unsigned *const out_rows, unsigned *const out_cols)
{
    const direct_axis_window_t element = direct_axis_window(key->element, key->active);
    const direct_axis_window_t entity = direct_axis_window(key->entity, key->active);
    double *const matrix = work->axis_matrix;
    double *const vectors = work->vectors;
    unsigned *const pivot = work->pivot;
    double *const scale = work->scale;
    *out_rows = entity.count;
    *out_cols = element.count;
    // Zero first, so an empty window still leaves a defined matrix behind.
    for (size_t i = 0; i < (size_t)entity.count * element.count; ++i)
    {
        matrix[i] = 0.0;
    }
    if (entity.count == 0 || element.count == 0)
    {
        return;
    }
    if (direct_windows_match(element, entity))
    {
        for (unsigned r = 0; r < entity.count; ++r)
        {
            matrix[r * element.count + r] = 1.0;
        }
        return;
    }

    const unsigned degree = element.spec.order > entity.spec.order ? element.spec.order : entity.spec.order;
    const integration_spec_t rule_spec = {.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = degree};
    // The batched getter reads one rule per basis, so both entries share the fetched rule.
    const integration_rule_t *rules[2] = {NULL, NULL};
    fdg_result_t result = integration_rule_registry_get_rule(request->integration_registry, rule_spec, &rules[0]);
    CUTL_ASSERT(result == FDG_SUCCESS, "The transfer quadrature could not be fetched (%d).", (int)result);
    rules[1] = rules[0];
    const basis_spec_t specs[2] = {element.spec, entity.spec};
    const basis_set_t *sets[2] = {NULL, NULL};
    result = basis_set_registry_get_basis_sets(request->basis_registry, 2u, sets, rules, specs);
    CUTL_ASSERT(result == FDG_SUCCESS, "The transfer bases could not be fetched (%d).", (int)result);
    const double *const weights = integration_rule_weights_const(rules[0]);
    const unsigned point_count = degree + 1u;
    const unsigned rhs = element.count;

    for (unsigned r = 0; r < entity.count; ++r)
    {
        const double *const entity_row = basis_set_basis_values(sets[1], entity.offset + r);
        for (unsigned s = 0; s < entity.count; ++s)
        {
            const double *const other = basis_set_basis_values(sets[1], entity.offset + s);
            double sum = 0.0;
            for (unsigned point = 0; point < point_count; ++point)
            {
                sum += weights[point] * entity_row[point] * other[point];
            }
            matrix[r * entity.count + s] = sum;
        }
        for (unsigned c = 0; c < element.count; ++c)
        {
            const double *const element_row = basis_set_basis_values(sets[0], element.offset + c);
            double sum = 0.0;
            for (unsigned point = 0; point < point_count; ++point)
            {
                sum += weights[point] * entity_row[point] * element_row[point];
            }
            vectors[r * rhs + c] = sum;
        }
    }
    // Equilibrate before the solve. An equispaced Lagrange Gram matrix spans several orders of magnitude, and
    // solving it as it stands overflows where the exact projection is perfectly finite. This is a fix for the
    // symptom only: a stable formulation that avoids the Gram solve altogether is deferred to a later session.
    for (unsigned r = 0; r < entity.count; ++r)
    {
        scale[r] = matrix[r * entity.count + r] > 0.0 ? 1.0 / sqrt(matrix[r * entity.count + r]) : 1.0;
        for (unsigned i = 0; i < entity.count; ++i)
        {
            matrix[r * entity.count + i] *= scale[r];
        }
        for (unsigned c = 0; c < element.count; ++c)
        {
            vectors[r * rhs + c] *= scale[r];
        }
    }
    direct_dense_solve(entity.count, rhs, matrix, vectors, pivot);
    for (unsigned r = 0; r < entity.count; ++r)
    {
        for (unsigned c = 0; c < element.count; ++c)
        {
            matrix[r * element.count + c] = vectors[r * rhs + c] / scale[r];
        }
    }

    basis_set_registry_release_basis_set(request->basis_registry, sets[0]);
    basis_set_registry_release_basis_set(request->basis_registry, sets[1]);
    integration_rule_registry_release_rule(request->integration_registry, rules[0]);
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
 * @brief Common basis of one object's axes, in the object's canonical axis order.
 *
 * The first incident element fixes the canonical axis order; the others lower an axis to their own specification
 * when it is smaller. #constraint_common_space_merge's basis rule, without its integration half.
 *
 * @param request Filled request.
 * @param dim Object dimension.
 * @param object_id Object ID in its dimension.
 * @param out_basis [dim] Receives the common basis.
 */
static void direct_entity_common_basis(const direct_continuity_request_t *const request, const unsigned dim,
                                       const uint64_t object_id, basis_spec_t *const out_basis)
{
    const topo_mesh_t *const mesh = request->mesh;
    uint64_t count;
    const uint64_t *ids;
    const int8_t *orientations;
    topo_obj_immersion_of_object(mesh->immersions + dim, object_id, &count, &ids, &orientations);
    const unsigned fixed_count = mesh->ndim - dim;
    for (unsigned j = 0; j < dim; ++j)
    {
        // An object's record opens with its `ndim - dim` fixed axes, so the free axes start after them.
        out_basis[j] = request->elements[ids[0]]->basis[constraint_orientation_axis(orientations[fixed_count + j])];
    }
    for (uint64_t i = 1; i < count; ++i)
    {
        const int8_t *const record = orientations + mesh->ndim * i;
        for (unsigned j = 0; j < dim; ++j)
        {
            const unsigned axis = constraint_orientation_axis(record[fixed_count + j]);
            if (out_basis[j].order > request->elements[ids[i]]->basis[axis].order)
            {
                out_basis[j] = request->elements[ids[i]]->basis[axis];
            }
        }
    }
}

/**
 * @brief DoFs of one object's interior for one component of a k-form.
 *
 * @param basis [dim] Common basis of the object's axes.
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
        dofs *= direct_axis_window(basis[j], active).count;
        next_axis += active ? 1u : 0u;
    }
    return dofs;
}

/**
 * @brief DoFs of one object's whole block, over all of its components.
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
 * @brief Offset of one component inside an object's block.
 */
static size_t direct_object_component_offset(const basis_spec_t *const basis, const unsigned dim, const unsigned order,
                                             const unsigned component)
{
    size_t offset = 0;
    uint8_t axes[UINT8_MAX];
    for (unsigned other = 0; other < component; ++other)
    {
        combination_set_to_index((uint8_t)dim, (uint8_t)order, axes, other);
        offset += direct_object_component_dofs(basis, dim, order, axes);
    }
    return offset;
}

/**
 * @brief Whether a component carries a covector along an element axis.
 */
static bool direct_component_has_axis(const direct_continuity_work_t *const work, const unsigned order,
                                      const unsigned axis)
{
    for (unsigned i = 0; i < order; ++i)
    {
        if (work->component_axes[i] == axis)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Emit one object-axis digit tuple of the current element DoF.
 */
static void direct_walk_object(direct_walk_t *const walk, const unsigned axis, const double value)
{
    direct_continuity_work_t *const work = walk->work;
    if (walk->dim == 0)
    {
        // A point carries one coefficient per component, so its digit tuple is empty.
        walk->emit(walk->param, walk->local, walk->object_base, value * walk->sign);
        return;
    }
    for (unsigned entry = 0; entry < work->support_counts[axis]; ++entry)
    {
        const size_t slot = (size_t)axis * work->capacity + entry;
        work->object_digits[axis] = work->support_rows[slot];
        const double scaled = value * work->support_values[slot];
        if (axis + 1 == walk->dim)
        {
            size_t global = walk->object_base;
            for (unsigned j = 0; j < walk->dim; ++j)
            {
                global += (size_t)work->object_digits[j] * work->object_strides[j];
            }
            walk->emit(walk->param, walk->local, global, scaled * walk->sign);
        }
        else
        {
            direct_walk_object(walk, axis + 1, scaled);
        }
    }
}

/**
 * @brief Build one object axis' support: the object rows the current element digit reaches.
 */
static void direct_build_support(direct_walk_t *const walk, const unsigned object_axis)
{
    direct_continuity_work_t *const work = walk->work;
    const unsigned ndim = walk->plan->ndim;
    const int8_t mapping = walk->record[ndim - walk->dim + object_axis];
    const unsigned element_axis = constraint_orientation_axis(mapping);
    const bool active = direct_component_has_axis(work, walk->plan->order, element_axis);
    const direct_axis_key_t key = {
        .element = walk->spec->basis[element_axis],
        .entity =
            walk->plan->entity_basis[direct_entity_index(walk->plan, walk->dim, walk->object_id) * ndim + object_axis],
        .active = active};
    unsigned rows;
    unsigned cols;
    direct_axis_transfer(walk->request, work, &key, &rows, &cols);
    const double *const matrix = work->axis_matrix;
    // A reversed object axis mirrors the element digit inside the element's axis functions.
    const unsigned digit = constraint_orientation_mirrored(mapping)
                               ? work->element_counts[element_axis] - 1u - work->element_digits[element_axis]
                               : work->element_digits[element_axis];
    // The window starts after the endpoint function, so an inactive axis shifts by one.
    const unsigned column = active ? digit : digit - 1u;
    CUTL_ASSERT(column < cols, "Element digit %u is outside the %u functions the axis contributes.", digit, cols);
    work->support_counts[object_axis] = 0;
    for (unsigned row = 0; row < rows; ++row)
    {
        const double coefficient = matrix[row * cols + column];
        if (coefficient != 0.0)
        {
            const size_t slot_index = (size_t)object_axis * work->capacity + work->support_counts[object_axis];
            work->support_rows[slot_index] = row;
            work->support_values[slot_index] = coefficient;
            work->support_counts[object_axis] += 1;
        }
    }
}

fdg_result_t direct_continuity_prepare(const direct_continuity_request_t *const request,
                                       direct_continuity_work_t *const work, direct_continuity_plan_t *const plan)
{
    (void)work;
    const unsigned ndim = request->ndim;
    const unsigned order = request->order;
    const topo_mesh_t *const mesh = request->mesh;
    CUTL_ASSERT(ndim > 0, "The element dimension must be at least one.");
    CUTL_ASSERT(order <= ndim, "The k-form order %u exceeds the element dimension %u.", order, ndim);

    plan->ndim = ndim;
    plan->order = order;
    plan->element_count = mesh->element_count;
    plan->entity_count = direct_entity_total(mesh, ndim);
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

    size_t block_running = 0;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        const unsigned object_count = mesh->immersions[dim].object_count;
        for (unsigned object = 0; object < object_count; ++object)
        {
            const size_t index = direct_entity_index(plan, dim, object);
            basis_spec_t *const basis = plan->entity_basis + index * ndim;
            uint64_t incident;
            const uint64_t *incident_ids;
            const int8_t *records;
            topo_obj_immersion_of_object(mesh->immersions + dim, object, &incident, &incident_ids, &records);
            // A mesh may declare more points than its elements carry; such a point owns nothing.
            plan->entity_block_offsets[index] = block_running;
            if (incident == 0)
            {
                continue;
            }
            direct_entity_common_basis(request, dim, object, basis);
            for (unsigned j = 0; j < dim; ++j)
            {
                // Reported, not asserted: a release build drops the check.
                if (!direct_basis_is_nodal(basis[j].type))
                {
                    return FDG_ERROR_NOT_IN_DOMAIN;
                }
                if (basis[j].order == 0)
                {
                    return FDG_ERROR_NOT_IN_DOMAIN;
                }
            }
            block_running += direct_object_block_size(basis, dim, order);
        }
    }
    plan->entity_block_offsets[plan->entity_count] = block_running;

    size_t dof_running = 0;
    // Element-private DoFs come after every object's block.
    size_t interior_running = block_running;
    for (uint64_t element = 0; element < mesh->element_count; ++element)
    {
        const kform_spec_t *const spec = request->elements[element];
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (!direct_basis_is_nodal(spec->basis[axis].type) || spec->basis[axis].order == 0)
            {
                return FDG_ERROR_NOT_IN_DOMAIN;
            }
        }
        plan->element_dof_offsets[element] = dof_running;
        plan->element_interior_offsets[element] = interior_running;
        dof_running += kform_spec_total_dofs(spec);
        uint8_t axes[UINT8_MAX];
        const unsigned component_count = combination_total_count((uint8_t)ndim, (uint8_t)order);
        for (unsigned component = 0; component < component_count; ++component)
        {
            kform_component_axes(spec, component, axes);
            interior_running += direct_object_component_dofs(spec->basis, ndim, order, axes);
        }
    }
    plan->element_dof_offsets[mesh->element_count] = dof_running;
    plan->element_interior_offsets[mesh->element_count] = interior_running;

    plan->layout.element_count = mesh->element_count;
    plan->layout.entry_count = 0;
    plan->layout.entity_count = plan->entity_count;
    plan->layout.element_dof_count = dof_running;
    plan->layout.global_dof_count = interior_running;
    return FDG_SUCCESS;
}

static void direct_walk(const direct_continuity_request_t *request, const direct_continuity_plan_t *plan,
                        direct_continuity_work_t *work, direct_entry_fn emit, void *param);

/**
 * @brief Count one nonzero of the transfer.
 */
static void direct_count_entry(void *const param, const size_t local, const size_t global, const double value)
{
    (void)local;
    (void)value;
    if (global != DIRECT_NO_ENTRY)
    {
        ++*(size_t *)param;
    }
}

void direct_continuity_layout(const direct_continuity_request_t *const request, direct_continuity_plan_t *const plan,
                              direct_continuity_work_t *const work, direct_continuity_layout_t *const out_layout)
{
    // The nonzero count is the last thing known; the plan keeps it.
    direct_walk(request, plan, work, direct_count_entry, &plan->layout.entry_count);
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
    if (global == DIRECT_NO_ENTRY)
    {
        return;
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
    direct_walk(request, plan, work, direct_write_entry, &state);
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

void direct_continuity_plan_release(const direct_continuity_plan_t *const plan)
{
    (void)plan;
}
/**
 * @brief Place every work member into one block, optionally assigning the pointers.
 *
 * Single source of truth for #direct_continuity_work_memory and #direct_continuity_work_init, so the two cannot
 * disagree about padding. With @p work NULL only the byte total accumulates.
 *
 * @return Total bytes for one block.
 */
/**
 * @brief Largest basis order the request carries, plus room for the one-dimensional operator scratch.
 *
 * The operators are as wide as the basis is, which the k-form order says nothing about: a scalar field of
 * basis order four still needs a four-by-four operator.
 */
static unsigned direct_continuity_basis_capacity(const direct_continuity_request_t *const request)
{
    unsigned basis_order = 1;
    for (uint64_t element = 0; element < request->mesh->element_count; ++element)
    {
        for (unsigned axis = 0; axis < request->ndim; ++axis)
        {
            const unsigned order = request->elements[element]->basis[axis].order;
            basis_order = basis_order > order ? basis_order : order;
        }
    }
    return basis_order + 2u;
}

static size_t direct_work_layout(const direct_continuity_request_t *const request, direct_continuity_work_t *const work,
                                 void *const memory)
{
    const unsigned ndim = request->ndim;
    const unsigned order = request->order;
    const unsigned capacity = direct_continuity_basis_capacity(request);
    const unsigned order_storage = order == 0u ? 1u : order;
    const size_t align = _Alignof(max_align_t);
    const size_t matrix = (size_t)capacity * capacity;
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
    DIRECT_TAKE(axis_matrix, double, matrix);
    DIRECT_TAKE(vectors, double, matrix);
    DIRECT_TAKE(pivot, unsigned, capacity);
    DIRECT_TAKE(scale, double, capacity);
    DIRECT_TAKE(component_axes, uint8_t, order_storage);
    DIRECT_TAKE(object_axes, uint8_t, order_storage);
    DIRECT_TAKE(mapped_axes, uint8_t, order_storage);
    DIRECT_TAKE(fixed_axes, int8_t, ndim);
    DIRECT_TAKE(element_counts, unsigned, ndim);
    DIRECT_TAKE(object_counts, unsigned, ndim);
    DIRECT_TAKE(element_digits, unsigned, ndim);
    DIRECT_TAKE(object_digits, unsigned, ndim);
    DIRECT_TAKE(support_rows, unsigned, (size_t)ndim *capacity);
    DIRECT_TAKE(support_values, double, (size_t)ndim *capacity);
    DIRECT_TAKE(support_counts, unsigned, ndim);
    DIRECT_TAKE(element_strides, size_t, (size_t)ndim + 1u);
    DIRECT_TAKE(object_strides, size_t, (size_t)ndim + 1u);
    DIRECT_TAKE(component_offsets, size_t, (size_t)components + 1u);
#undef DIRECT_TAKE
    if (work != NULL)
    {
        work->capacity = capacity;
    }
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
 * @return Total bytes for one block.
 */
static size_t direct_plan_layout(const direct_continuity_request_t *const request, direct_continuity_plan_t *const plan,
                                 void *const memory)
{
    const uint64_t entity_count = direct_entity_total(request->mesh, request->ndim);
    const uint64_t element_count = request->mesh->element_count;
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
    DIRECT_TAKE(entity_dim_offsets, uint64_t, (size_t)request->ndim + 1u);
    DIRECT_TAKE(entity_basis, basis_spec_t, (size_t)entity_count * request->ndim);
    DIRECT_TAKE(entity_block_offsets, size_t, (size_t)entity_count + 1u);
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
/**
 * @brief Emit one element-local DoF of the current component.
 *
 * Decodes the digit tuple, splits off the axes pinned to a face, and emits either the element-private DoF or the
 * object's entries. Iterating the element's own numbering keeps a DoF's entries consecutive and ascending.
 *
 * @param walk Enumeration state; its element counts are already set.
 * @param local_index Element-local DoF index inside the current component.
 * @param fixed_axes Scratch receiving the signed one-based pinned axes in ascending order.
 */
static void direct_emit_local(direct_walk_t *const walk, const size_t local_index, int8_t *const fixed_axes)
{
    direct_continuity_work_t *const work = walk->work;
    const unsigned ndim = walk->plan->ndim;
    const unsigned order = walk->plan->order;

    size_t rest = local_index;
    unsigned pinned = 0;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        const unsigned digit = (unsigned)(rest / work->element_strides[axis]) % work->element_counts[axis];
        rest -= (size_t)digit * work->element_strides[axis];
        work->element_digits[axis] = digit;
        // Only an axis without a covector of the component reaches a face, and only at an endpoint node.
        if (direct_component_has_axis(work, order, axis))
        {
            continue;
        }
        if (digit == 0u)
        {
            fixed_axes[pinned] = (int8_t)(-(int)axis - 1);
            pinned += 1u;
        }
        else if (digit == walk->spec->basis[axis].order)
        {
            fixed_axes[pinned] = (int8_t)(axis + 1);
            pinned += 1u;
        }
    }

    walk->local =
        walk->plan->element_dof_offsets[walk->element] + work->component_offsets[walk->component] + local_index;
    walk->dim = ndim - pinned;
    if (pinned == 0)
    {
        walk->emit(walk->param, walk->local, walk->plan->element_interior_offsets[walk->element] + walk->interior, 1.0);
        walk->interior += 1;
        return;
    }

    uint64_t object_id;
    topo_mesh_element_object(walk->plan->mesh, walk->element, pinned, fixed_axes, &object_id);
    uint64_t incident;
    const uint64_t *incident_ids;
    const int8_t *records;
    topo_obj_immersion_of_object(walk->plan->mesh->immersions + walk->dim, object_id, &incident, &incident_ids,
                                 &records);
    uint64_t index = 0;
    while (index < incident && incident_ids[index] != walk->element)
    {
        index += 1;
    }
    CUTL_ASSERT(index < incident, "Object %llu does not contain element %llu.", (unsigned long long)object_id,
                (unsigned long long)walk->element);
    walk->object_id = object_id;
    walk->record = records + ndim * index;

    unsigned object_axes = 0;
    for (unsigned j = 0; j < walk->dim; ++j)
    {
        const unsigned element_axis = constraint_orientation_axis(walk->record[ndim - walk->dim + j]);
        if (direct_component_has_axis(work, order, element_axis))
        {
            work->object_axes[object_axes] = (uint8_t)j;
            object_axes += 1u;
        }
    }
    CUTL_ASSERT(object_axes == order, "Only %u of the component's %u covector axes are free on the object.",
                object_axes, order);
    walk->sign =
        constraint_mapped_axes_and_sign(
            &(constraint_element_side_t){.ndim = ndim, .basis_specs = walk->spec->basis, .orientation = walk->record},
            walk->dim, order, work->object_axes, work->mapped_axes)
            ? -1.0
            : 1.0;

    const basis_spec_t *const object_basis =
        walk->plan->entity_basis + direct_entity_index(walk->plan, walk->dim, object_id) * ndim;
    unsigned next = 0;
    for (unsigned j = 0; j < walk->dim; ++j)
    {
        const bool active = next < order && work->object_axes[next] == j;
        work->object_counts[j] = direct_axis_window(object_basis[j], active).count;
        next += active ? 1u : 0u;
    }
    work->object_strides[walk->dim] = 1;
    if (walk->dim > 0)
    {
        work->object_strides[walk->dim - 1u] = 1;
        for (unsigned j = walk->dim - 1u; j-- > 0;)
        {
            work->object_strides[j] = work->object_strides[j + 1u] * work->object_counts[j + 1u];
        }
    }
    walk->object_base =
        walk->plan->entity_block_offsets[direct_entity_index(walk->plan, walk->dim, object_id)] +
        direct_object_component_offset(object_basis, walk->dim, order,
                                       combination_get_index((uint8_t)walk->dim, (uint8_t)order, work->object_axes));
    size_t combinations = 1;
    for (unsigned axis = 0; axis < walk->dim; ++axis)
    {
        direct_build_support(walk, axis);
        combinations *= work->support_counts[axis];
    }
    if (combinations == 0)
    {
        // This element's order exceeds the common space on some object axis, so its trace projects onto an empty
        // space there. The DoF carries no global counterpart and leaves the system, but the row compression must
        // still account for it.
        walk->emit(walk->param, walk->local, DIRECT_NO_ENTRY, 0.0);
        return;
    }
    direct_walk_object(walk, 0u, 1.0);
}

/**
 * @brief Enumerate every nonzero of the element-to-global transfer.
 *
 * Walks elements, components and, per component, the element's own DoF order, so the entries come out grouped by
 * local DoF and ascending.
 */
static void direct_walk(const direct_continuity_request_t *const request, const direct_continuity_plan_t *plan,
                        direct_continuity_work_t *work, direct_entry_fn emit, void *param)
{
    const unsigned ndim = plan->ndim;
    const unsigned order = plan->order;
    const unsigned component_count = combination_total_count((uint8_t)ndim, (uint8_t)order);
    int8_t *const fixed_axes = work->fixed_axes;
    direct_walk_t walk = {.request = request, .plan = plan, .work = work, .emit = emit, .param = param};

    for (uint64_t element = 0; element < plan->element_count; ++element)
    {
        const kform_spec_t *const spec = plan->elements[element];
        walk.spec = spec;
        walk.element = element;
        walk.interior = 0;
        kform_spec_component_offsets(spec, component_count + 1u, work->component_offsets);
        for (unsigned component = 0; component < component_count; ++component)
        {
            kform_component_axes(spec, component, work->component_axes);
            walk.component = component;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                work->element_counts[axis] =
                    spec->basis[axis].order + (direct_component_has_axis(work, order, axis) ? 0u : 1u);
            }
            // Row-major strides: the last axis varies fastest, so its stride is one and every earlier axis
            // multiplies in the count of the axis after it.
            work->element_strides[ndim] = 1;
            if (ndim > 0)
            {
                work->element_strides[ndim - 1u] = 1;
                for (unsigned axis = ndim - 1u; axis-- > 0;)
                {
                    work->element_strides[axis] = work->element_strides[axis + 1u] * work->element_counts[axis + 1u];
                }
            }
            const size_t component_dofs = work->element_strides[0] * work->element_counts[0];
            for (size_t local = 0; local < component_dofs; ++local)
            {
                direct_emit_local(&walk, local, fixed_axes);
            }
        }
    }
}
