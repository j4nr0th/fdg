/**
 * @file constraints.c
 * @brief Implementation of reference and physical trace constraints.
 *
 * Assembly uses one canonical face coordinate system; element-side orientations map it to signed, one-based element
 * axes, and the helpers below keep that mapping and the alternating k-form sign in one place. Routines trust
 * documented preconditions guarded by debug asserts only, size storage with the `*_layout`/`*_work_size` functions,
 * and report registry allocation failures through `fdg_result_t`.
 */

#include "constraints.h"

#include <limits.h>

#include <cutl/iterators/combination_iterator.h>

#include "../operations/map_transforms.h"
#include "../polynomials/lagrange.h"

/**
 * @brief Map a face component's axes into element axes and a sign.
 *
 * The side orientation contributes one sign per reversed mapped axis; sorting the mapped axes into canonical element
 * order flips the alternating covector sign once per transposition.
 *
 * Preconditions: `side->orientation` is a signed one-based permutation with an increasing-absolute-value fixed-axis
 * prefix; `order <= boundary_dim <= side->ndim`; `test_axes` are a valid component's sorted covector axes.
 *
 * @return `true` if the sign must be flipped.
 */
static bool mapped_axes_and_sign(const constraint_element_side_t *const side, const unsigned boundary_dim,
                                 const unsigned order, const uint8_t test_axes[const static order == 0 ? 1 : order],
                                 uint8_t mapped_axes[const static order == 0 ? 1 : order])
{
    const unsigned fixed_count = side->ndim - boundary_dim;
    unsigned sign = 0;
    // Collect the mapped axes from the side's orientation and get the initial sign
    for (unsigned i = 0; i < order; ++i)
    {
        const int8_t mapping = side->orientation[fixed_count + test_axes[i]];
        mapped_axes[i] = (uint8_t)(mapping < 0 ? -mapping : mapping) - 1;
        if (mapping < 0)
            sign += 1;
    }
    // Sort into canonical order; each swap flips the sign (one transposition).
    for (unsigned i = 0; i < order; ++i)
    {
        for (unsigned j = i + 1; j < order; ++j)
        {
            if (mapped_axes[i] > mapped_axes[j])
            {
                sign += 1;
                const uint8_t tmp = mapped_axes[i];
                mapped_axes[i] = mapped_axes[j];
                mapped_axes[j] = tmp;
            }
        }
    }
    // Odd parity indicates a negative sign.
    return sign & 1;
}

/**
 * @brief Map a face component's axes into an element component.
 *
 * Combines the orientation sign and permutation parity into the covector's pullback sign; `out_component` receives the
 * combination index of the mapped axes.
 *
 * Preconditions: as in #mapped_axes_and_sign.
 *
 * @return `true` if the mapped component's sign is flipped.
 */
static bool mapped_component(const constraint_element_side_t *const side, const unsigned boundary_dim,
                             const unsigned order, const uint8_t test_axes[const static order == 0 ? 1 : order],
                             uint8_t mapped_axes[const static order == 0 ? 1 : order], unsigned *const out_component)
{
    const bool sign = mapped_axes_and_sign(side, boundary_dim, order, test_axes, mapped_axes);

    // Get the component index based on the element's mapped axes
    *out_component = combination_get_index(side->ndim, order, mapped_axes);

    return sign;
}

/**
 * @brief Merge the per-element boundary views into one common boundary space.
 *
 * The first element seeds the canonical basis and rules; later elements lower a basis axis to their own order when
 * smaller and raise an integration axis to their own rule when more accurate — per-axis minimum order (the L2 link
 * target) for the basis, most accurate rule for the integration.
 */
static void boundary_common_space_merge(unsigned ndim, unsigned bdim, unsigned nelem,
                                        const boundary_element_space_t elements[static nelem],
                                        basis_spec_t out_basis[static bdim],
                                        integration_spec_t out_integration[static bdim])
{
    const boundary_element_space_t *const first = elements;
    integration_rules_to_boundary(ndim, first->integration, first->orientation, bdim, out_integration);
    for (unsigned idim = 0; idim < bdim; ++idim)
    {
        const int8_t signed_axis = first->orientation[ndim - bdim + idim];
        const unsigned i_axis = signed_axis < 0 ? -signed_axis - 1 : signed_axis - 1;
        out_basis[idim] = first->basis[i_axis];
    }

    for (unsigned ie = 1; ie < nelem; ++ie)
    {
        const boundary_element_space_t *const element = elements + ie;
        const int8_t *const elem_varying = element->orientation + (ndim - bdim);
        for (unsigned idim = 0; idim < bdim; ++idim)
        {
            const int8_t signed_axis = elem_varying[idim];
            const unsigned i_axis = signed_axis < 0 ? -signed_axis - 1 : signed_axis - 1;
            if (out_basis[idim].order > element->basis[i_axis].order)
            {
                out_basis[idim] = element->basis[i_axis];
            }
            if (integration_spec_accuracy(out_integration + idim) <
                integration_spec_accuracy(element->integration + i_axis))
            {
                out_integration[idim] = element->integration[i_axis];
            }
        }
    }
}

/**
 * @brief Per-axis test function counts of one boundary component's row block.
 *
 * Active covector axes read the order-1 basis (`order` functions); inactive axes read the leading
 * `order + 1 - SKIPPED_BASIS` functions of the full basis (floored at zero). The window keeps the low-degree
 * functions of the common space, so pairing the (higher-order) trace against them is its L2 projection onto that
 * space (see #SKIPPED_BASIS and #constraint_boundary_mass_spec_t). An inactive axis windowed to a zero count
 * contributes an empty row block, which callers drop. `counts` is a caller-provided `[bdim]` array.
 */
static void boundary_mass_row_axis_counts(const constraint_boundary_mass_spec_t *const spec,
                                          const uint8_t component_axes[const static spec->order == 0 ? 1 : spec->order],
                                          unsigned counts[spec->bdim])
{
    unsigned i_axis, i_active;
    for (i_axis = 0, i_active = 0; i_active < spec->order; ++i_axis)
    {
        unsigned full_count = spec->boundary_basis[i_axis].order;
        if (component_axes[i_active] == i_axis)
        {
            // The axis is active
            i_active += 1;
        }
        else
        {
            // The axis is inactive: drop the highest functions of its full basis.
            full_count += 1u;
            full_count = full_count > SKIPPED_BASIS ? full_count - SKIPPED_BASIS : 0u;
        }
        counts[i_axis] = full_count;
    }

    // Now everything else is just inactive axes
    for (; i_axis < spec->bdim; ++i_axis)
    {
        const unsigned full_count = spec->boundary_basis[i_axis].order + 1u;
        counts[i_axis] = full_count > SKIPPED_BASIS ? full_count - SKIPPED_BASIS : 0u;
    }
}

/**
 * @brief Count one boundary component's test DoFs from its per-axis counts.
 */
static size_t boundary_mass_row_dofs(const constraint_boundary_mass_spec_t *const spec,
                                     const unsigned counts[static spec->bdim])
{
    size_t dof_count = 1;
    for (unsigned axis = 0; axis < spec->bdim; ++axis)
    {
        dof_count *= counts[axis];
    }
    return dof_count;
}

/**
 * @brief Sample one boundary component's test functions at the common points.
 *
 * Point-major layout, axis 0 the slowest DoF digit; digits index into the per-axis window chosen by the counting
 * pass.
 * DoF digits run through the work buffer's multidim iterator; tensor points through a last-axis-fastest odometer with
 * a prefix cache of intermediate products (#outer_product_pair_iterator_t style).
 *
 * Preconditions: per-axis test function counts are positive (callers drop empty row blocks); each basis set was built
 * at the nodes of `spec->boundary_integration`.
 */
static void boundary_mass_row_values(const constraint_boundary_mass_spec_t *const spec,
                                     const basis_set_t *const *basis_sets, const basis_set_t *const *basis_sets_lower,
                                     const uint8_t component_axes[const static spec->order == 0 ? 1 : spec->order],
                                     const size_t point_count, constraint_boundary_mass_work_t *work)
{
    boundary_mass_row_axis_counts(spec, component_axes, work->counts);
    const unsigned bdim = spec->bdim;
    if (bdim == 0)
    {
        // A zero-dimensional boundary has one scalar DoF at the single empty-tensor point; no iterator axes.
        work->row_values[0] = 1.0;
        return;
    }

    multidim_iterator_t *const dof_iter = work->dof_iter;
    const double **const rows = work->axis_tables;
    size_t tensor_points = 1;
    // Merge walk over the sorted component axes (like mapped_axes_and_sign): each axis consumes its next active
    // entry, so no per-axis rescan of the component is needed.
    unsigned i_active = 0;
    for (unsigned axis = 0; axis < bdim; ++axis)
    {
        const bool active = i_active < spec->order && component_axes[i_active] == axis;
        i_active += active ? 1u : 0u;
        work->axis_sets[axis] = active ? basis_sets_lower[axis] : basis_sets[axis];
        CUTL_ASSERT(work->axis_sets[axis]->integration_spec.order == spec->boundary_integration[axis].order,
                    "Axis %u reads a basis set built at foreign integration nodes.", axis);
        multidim_iterator_init_dim(dof_iter, axis, work->counts[axis]);
        tensor_points *= (size_t)spec->boundary_integration[axis].order + 1u;
    }
    CUTL_ASSERT(tensor_points == point_count, "The odometer bounds do not span the common rule's tensor points.");
    const size_t dof_count = multidim_iterator_total_size(dof_iter);
    unsigned *const point_digits = work->point_digits;
    double *const prefix = work->point_prefix;
    for (multidim_iterator_set_to_start(dof_iter); !multidim_iterator_is_at_end(dof_iter);
         multidim_iterator_advance(dof_iter, bdim - 1, 1))
    {
        const size_t *const digits = multidim_iterator_offsets(dof_iter);
        const size_t flat = multidim_iterator_get_flat_index(dof_iter);
        // Select this DoF's rows, restart the odometer, seed the prefix cache at point 0 (all digits zero).
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            rows[axis] = basis_set_basis_values(work->axis_sets[axis], (unsigned)digits[axis]);
            point_digits[axis] = 0;
            prefix[axis] = (axis == 0 ? 1.0 : prefix[axis - 1]) * rows[axis][0];
        }
        for (size_t point = 0; point < point_count; ++point)
        {
            work->row_values[point * dof_count + flat] = prefix[bdim - 1];
            // Advance from the last axis; a full wrap (carry reaches zero) exhausts the tensor, so no rebuild remains.
            unsigned carry = bdim;
            for (; carry > 0; --carry)
            {
                if (++point_digits[carry - 1] <= spec->boundary_integration[carry - 1].order)
                    break;
                point_digits[carry - 1] = 0;
            }
            if (carry > 0)
            {
                for (unsigned axis = carry - 1; axis < bdim; ++axis)
                {
                    prefix[axis] = (axis == 0 ? 1.0 : prefix[axis - 1]) * rows[axis][point_digits[axis]];
                }
            }
        }
    }
}

/**
 * @brief Build the trace value sources of the element's axes.
 *
 * Free axes read the element basis sets at the mapped common rules, mirrored when the orientation reverses the
 * axis; fixed normal axes read their endpoint values at the signed end.
 */
static void boundary_mass_axis_descriptors(const constraint_boundary_mass_spec_t *const spec,
                                           const constraint_boundary_mass_request_t *const request,
                                           constraint_boundary_mass_work_t *work)
{
    kform_trace_axis_t *const axes = work->axes;
    const unsigned fixed_count = spec->ndim - spec->bdim;
    for (unsigned axis = 0; axis < spec->ndim; ++axis)
    {
        axes[axis] = (kform_trace_axis_t){};
    }
    for (unsigned face_axis = 0; face_axis < spec->bdim; ++face_axis)
    {
        const int8_t mapping = spec->orientation[fixed_count + face_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        axes[element_axis] = (kform_trace_axis_t){
            .kind = KFORM_TRACE_AXIS_FREE,
            .mirror = mapping < 0,
            .free = {.nodes = request->element_basis_sets[element_axis],
                     .nodes_lower = request->element_basis_sets_lower != NULL
                                        ? request->element_basis_sets_lower[element_axis]
                                        : NULL,
                     .rule_size = spec->boundary_integration[face_axis].order + 1u,
                     .stride_slot = face_axis},
        };
    }
    for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
    {
        const int8_t mapping = spec->orientation[fixed_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        axes[element_axis] = (kform_trace_axis_t){
            .kind = KFORM_TRACE_AXIS_FIXED,
            .mirror = mapping < 0,
            .fixed = {.endpoint = request->element_endpoints != NULL ? request->element_endpoints[element_axis] : NULL,
                      .endpoint_lower = request->element_endpoints_lower != NULL
                                            ? request->element_endpoints_lower[element_axis]
                                            : NULL},
        };
    }
}

/**
 * @brief Compute the dense shape of one element's boundary mass matrix.
 *
 * One combination-iterator pass fills the mapped component table and both offset tables: row offsets are
 * per-component starts with the total at `work->row_offsets[component_count]`, column offsets the mapped block
 * starts with the total at `work->col_offsets[component_count]`.
 */
static void boundary_mass_shape(const constraint_boundary_mass_spec_t *const spec,
                                constraint_boundary_mass_work_t *work, size_t *const out_rows, size_t *const out_cols)
{
    const constraint_element_side_t side = {
        .ndim = spec->ndim, .basis_specs = spec->element_spec->basis, .orientation = spec->orientation};
    const unsigned order = spec->order;
    if (order > spec->bdim)
    {
        // A k-form of order past the boundary dimension has no trace components.
        work->row_offsets[0] = 0;
        work->col_offsets[0] = 0;
        *out_rows = 0;
        *out_cols = 0;
        return;
    }
    combination_iterator_init(work->components, (uint8_t)spec->bdim, (uint8_t)order);
    size_t rows = 0;
    size_t cols = 0;
    for (size_t component = 0; !combination_iterator_is_done(work->components);
         combination_iterator_next(work->components), ++component)
    {
        const uint8_t *const component_axes = combination_iterator_current(work->components);
        boundary_mass_row_axis_counts(spec, component_axes, work->counts);
        work->row_offsets[component] = rows;
        rows += boundary_mass_row_dofs(spec, work->counts);
        work->element_signs[component] =
            mapped_axes_and_sign(&side, spec->bdim, order, component_axes, work->mapped_axes) ? -1 : 1;
        work->element_components[component] = combination_get_index(spec->ndim, order, work->mapped_axes);
        work->col_offsets[component] = cols;
        cols += kform_spec_component_dof_count(spec->element_spec, work->element_components[component]);
    }
    const size_t total = combination_iterator_total_count(work->components);
    CUTL_ASSERT(total == combination_total_count((uint8_t)spec->bdim, (uint8_t)order),
                "The component enumeration disagrees with the analytic component count.");
    work->row_offsets[total] = rows;
    work->col_offsets[total] = cols;
    *out_rows = rows;
    *out_cols = cols;
}

void constraint_boundary_mass_layout(const constraint_boundary_mass_spec_t *const spec,
                                     constraint_boundary_mass_work_t *work, const bool physical,
                                     size_t *const out_row_count, size_t *const out_col_count,
                                     size_t *const out_entry_count)
{
    const kform_spec_t boundary_spec = {.ndim = spec->bdim, .order = spec->order, .basis = spec->boundary_basis};
    const size_t component_count = kform_spec_component_count(&boundary_spec);
    size_t rows;
    size_t cols;
    boundary_mass_shape(spec, work, &rows, &cols);

    size_t entries = 0;
    for (unsigned component = 0; component < component_count; ++component)
    {
        const size_t row_dofs = work->row_offsets[component + 1] - work->row_offsets[component];
        if (!physical)
        {
            entries += row_dofs * (work->col_offsets[component + 1] - work->col_offsets[component]);
            continue;
        }
        for (unsigned other = 0; other < component_count; ++other)
        {
            entries += row_dofs * (work->col_offsets[other + 1] - work->col_offsets[other]);
        }
    }
    *out_row_count = rows;
    *out_col_count = cols;
    *out_entry_count = entries;
}

/**
 * @brief Place every boundary mass work member into one block, optionally assigning the pointers.
 *
 * Single source of truth for #constraint_boundary_mass_work_memory and #constraint_boundary_mass_work_init: with
 * @p work NULL only the byte total accumulates (the alignment padding included). @p sizes NULL places the sizing
 * scratch alone — every a-priori-sized member, with the value table members unset.
 *
 * @return Total bytes for one block.
 */
static size_t boundary_mass_work_layout(const constraint_boundary_mass_spec_t *const spec,
                                        const constraint_boundary_mass_work_sizes_t *const sizes,
                                        constraint_boundary_mass_work_t *const work, void *const memory)
{
    size_t cursor = 0;
    const size_t align = _Alignof(max_align_t);
#define BOUNDARY_MASS_TAKE(member, bytes)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        const size_t take_bytes = (bytes);                                                                             \
        cursor = (cursor + align - 1u) & ~(align - 1u);                                                                \
        if (work != NULL)                                                                                              \
        {                                                                                                              \
            work->member = (void *)((unsigned char *)memory + cursor);                                                 \
        }                                                                                                              \
        cursor += take_bytes;                                                                                          \
    } while (false)

    const size_t component_count =
        spec->order > spec->bdim ? 0 : combination_total_count((uint8_t)spec->bdim, (uint8_t)spec->order);
    const size_t iterator_memory = combination_iterator_required_memory((uint8_t)spec->order);
    const size_t order_storage = spec->order == 0 ? 1u : spec->order;
    BOUNDARY_MASS_TAKE(point_iter, multidim_iterator_needed_memory(spec->ndim));
    BOUNDARY_MASS_TAKE(row_offsets, sizeof(size_t) * (component_count + 1u));
    BOUNDARY_MASS_TAKE(col_offsets, sizeof(size_t) * (component_count + 1u));
    BOUNDARY_MASS_TAKE(element_components, sizeof(unsigned) * component_count);
    BOUNDARY_MASS_TAKE(element_signs, sizeof(int) * component_count);
    BOUNDARY_MASS_TAKE(axes, sizeof(kform_trace_axis_t) * spec->ndim);
    BOUNDARY_MASS_TAKE(counts, sizeof(unsigned) * spec->bdim);
    BOUNDARY_MASS_TAKE(axis_sets, sizeof(const basis_set_t *) * spec->bdim);
    BOUNDARY_MASS_TAKE(axis_tables, sizeof(const double *) * spec->bdim);
    BOUNDARY_MASS_TAKE(dof_iter, multidim_iterator_needed_memory(spec->bdim));
    BOUNDARY_MASS_TAKE(point_digits, sizeof(unsigned) * spec->bdim);
    BOUNDARY_MASS_TAKE(point_prefix, sizeof(double) * spec->bdim);
    BOUNDARY_MASS_TAKE(mapped_axes, sizeof(uint8_t) * order_storage);
    BOUNDARY_MASS_TAKE(components, iterator_memory);
    BOUNDARY_MASS_TAKE(blocks, iterator_memory);
    if (sizes != NULL)
    {
        BOUNDARY_MASS_TAKE(row_values, sizeof(double) * sizes->row_values);
        BOUNDARY_MASS_TAKE(col_values, sizeof(double) * sizes->col_values);
        BOUNDARY_MASS_TAKE(point_factors, sizeof(double) * sizes->point_factors);
    }
    else if (work != NULL)
    {
        work->row_values = NULL;
        work->col_values = NULL;
        work->point_factors = NULL;
    }
#undef BOUNDARY_MASS_TAKE
    return cursor;
}

size_t constraint_boundary_mass_work_memory(const constraint_boundary_mass_spec_t *const spec,
                                            const constraint_boundary_mass_work_sizes_t *const sizes)
{
    return boundary_mass_work_layout(spec, sizes, NULL, NULL);
}

void constraint_boundary_mass_work_init(constraint_boundary_mass_work_t *const work,
                                        const constraint_boundary_mass_spec_t *const spec,
                                        const constraint_boundary_mass_work_sizes_t *const sizes, void *const memory)
{
    (void)boundary_mass_work_layout(spec, sizes, work, memory);
}

void constraint_boundary_mass_work_size(const constraint_boundary_mass_spec_t *const spec,
                                        constraint_boundary_mass_work_t *const work,
                                        constraint_boundary_mass_work_sizes_t *const out_sizes)
{
    const size_t point_count = integration_specs_total_points(spec->bdim, spec->boundary_integration);
    if (spec->order > spec->bdim)
    {
        *out_sizes = (constraint_boundary_mass_work_sizes_t){.point_factors = point_count};
        return;
    }
    const constraint_element_side_t side = {
        .ndim = spec->ndim, .basis_specs = spec->element_spec->basis, .orientation = spec->orientation};

    // The sizing pass runs entirely on caller-provided scratch: one iterator plus per-axis and mapped-axis buffers,
    // all sized a priori from the spec (see constraint_boundary_mass_work_init).
    combination_iterator_t *const iter = work->components;
    unsigned *const counts = work->counts;
    uint8_t *const mapped_axes = work->mapped_axes;

    combination_iterator_init(iter, (uint8_t)spec->bdim, (uint8_t)spec->order);
    size_t max_row_dofs = 0;
    size_t max_col_dofs = 0;
    for (const uint8_t *const component_axes = combination_iterator_current(iter); !combination_iterator_is_done(iter);
         combination_iterator_next(iter))
    {
        boundary_mass_row_axis_counts(spec, component_axes, counts);
        const size_t row_dofs = boundary_mass_row_dofs(spec, counts);
        (void)mapped_axes_and_sign(&side, spec->bdim, spec->order, component_axes, mapped_axes);
        const unsigned element_component = combination_get_index(spec->ndim, spec->order, mapped_axes);
        const size_t col_dofs = kform_spec_component_dof_count(spec->element_spec, element_component);
        max_row_dofs = max_row_dofs > row_dofs ? max_row_dofs : row_dofs;
        max_col_dofs = max_col_dofs > col_dofs ? max_col_dofs : col_dofs;
    }

    *out_sizes = (constraint_boundary_mass_work_sizes_t){
        .row_values = max_row_dofs * point_count,
        .col_values = max_col_dofs * point_count,
        .point_factors = point_count,
        .component_count = combination_total_count((uint8_t)spec->bdim, (uint8_t)spec->order)};
}

void constraint_boundary_mass_assemble(const constraint_boundary_mass_request_t *const request)
{
    const constraint_boundary_mass_spec_t *const spec = request->spec;
    constraint_boundary_mass_work_t *const work = request->work;
    const unsigned order = spec->order;
    CUTL_ASSERT(order <= spec->bdim, "Traced order %u exceeds the boundary dimension %u; that order has no trace.",
                order, spec->bdim);
    CUTL_ASSERT((request->test_pullback == NULL) == (request->element_pullback == NULL),
                "Physical pullbacks must be given for both test and element sides.");
    const bool physical = request->test_pullback != NULL;
    CUTL_ASSERT(!physical || request->test_pullback->physical_component_count ==
                                 request->element_pullback->physical_component_count,
                "Pullbacks disagree on the physical component count.");
    const size_t component_count = combination_total_count((uint8_t)spec->bdim, (uint8_t)order);

    const size_t point_count = integration_specs_total_points(spec->bdim, spec->boundary_integration);

    size_t rows;
    size_t cols;
    boundary_mass_shape(spec, work, &rows, &cols);
    for (size_t value = 0; value < rows * cols; ++value)
    {
        request->out_matrix[value] = 0.0;
    }

    boundary_mass_axis_descriptors(spec, request, work);

    const constraint_element_side_t side = {
        .ndim = spec->ndim, .basis_specs = spec->element_spec->basis, .orientation = spec->orientation};
    if (physical)
    {
        // Physical pairing walks every mapped component: the blocks iterator must be a valid (bdim, order) enumeration.
        combination_iterator_init(work->blocks, (uint8_t)spec->bdim, (uint8_t)order);
    }
    combination_iterator_reset(work->components);
    for (size_t component = 0; !combination_iterator_is_done(work->components);
         combination_iterator_next(work->components), ++component)
    {
        const uint8_t *const component_axes = combination_iterator_current(work->components);
        const size_t row_dofs = work->row_offsets[component + 1] - work->row_offsets[component];
        CUTL_ASSERT(component < component_count, "Component index exceeded the boundary component count.");
        CUTL_ASSERT(work->row_offsets[component] + row_dofs <= rows, "Row block leaves the dense matrix.");
        if (row_dofs == 0)
            continue;
        boundary_mass_row_values(spec, request->boundary_basis_sets, request->boundary_basis_sets_lower, component_axes,
                                 point_count, work);

        const unsigned first_block = physical ? 0 : (unsigned)component;
        const unsigned block_end = physical ? (unsigned)component_count : (unsigned)component + 1u;
        for (unsigned block = first_block; block < block_end; ++block)
        {
            const size_t col_dofs = work->col_offsets[block + 1] - work->col_offsets[block];
            CUTL_ASSERT(work->col_offsets[block] + col_dofs <= cols, "Column block leaves the dense matrix.");
            CUTL_ASSERT(block < component_count, "Block index exceeded the boundary component count.");
            if (col_dofs == 0)
                continue;
            // Element axes of the block's mapped component: reference pairing maps the row component itself,
            // physical pairing walks every mapped component through the same orientation.
            if (physical)
            {
                combination_iterator_set_to_index(work->blocks, block);
                (void)mapped_axes_and_sign(&side, spec->bdim, order, combination_iterator_current(work->blocks),
                                           work->mapped_axes);
            }
            else
            {
                (void)mapped_axes_and_sign(&side, spec->bdim, order, component_axes, work->mapped_axes);
            }
            kform_component_basis_values(spec->ndim, spec->element_spec->basis, order, work->mapped_axes, work->axes,
                                         work->point_iter, point_count, work->col_values);

            // The pullback tables hold each side's own covector image, so physical pairing is frame-free; only
            // reference pairing needs the orientation sign to express the row component in the element's covector
            // basis. The factor sweep vectorizes over points: physical pairing accumulates one contiguous row per
            // physical component, then a single contiguous pass folds in the weights (and optional face measure),
            // preserving the original `((sign * weight) * measure) * dot` rounding order.
            const double block_sign = (double)(physical ? 1 : work->element_signs[block]) * request->factor;
            double *const point_factors = work->point_factors;
            if (!physical)
            {
                if (request->surface_weights == NULL)
                {
#pragma omp simd
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        point_factors[point] = block_sign * request->point_weights[point];
                    }
                }
                else
                {
                    const double *const surface = request->surface_weights;
#pragma omp simd
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        point_factors[point] = (block_sign * request->point_weights[point]) * surface[point];
                    }
                }
            }
            else
            {
                const size_t pullback_components = request->test_pullback->physical_component_count;
                const double *const test_values =
                    request->test_pullback->values + (size_t)component * pullback_components * point_count;
                const double *const element_values =
                    request->element_pullback->values +
                    (size_t)work->element_components[block] * pullback_components * point_count;
#pragma omp simd
                for (size_t point = 0; point < point_count; ++point)
                {
                    point_factors[point] = 0.0;
                }
                for (unsigned physical_component = 0; physical_component < pullback_components; ++physical_component)
                {
                    const double *const test_row = test_values + (size_t)physical_component * point_count;
                    const double *const element_row = element_values + (size_t)physical_component * point_count;
#pragma omp simd
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        point_factors[point] += test_row[point] * element_row[point];
                    }
                }
                if (request->surface_weights == NULL)
                {
#pragma omp simd
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        point_factors[point] = (block_sign * request->point_weights[point]) * point_factors[point];
                    }
                }
                else
                {
                    const double *const surface = request->surface_weights;
#pragma omp simd
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        point_factors[point] =
                            ((block_sign * request->point_weights[point]) * surface[point]) * point_factors[point];
                    }
                }
            }
            kform_inner_product_block(point_count, row_dofs, col_dofs, work->row_values, work->col_values,
                                      work->point_factors, work->row_offsets[component], work->col_offsets[block], cols,
                                      request->out_matrix);
        }
    }
}

void constraint_boundary_mass_pack(const constraint_boundary_mass_spec_t *const spec,
                                   constraint_boundary_mass_work_t *work, const bool physical,
                                   const double *const matrix, const size_t row_stride, const double factor,
                                   const uint8_t side, uint8_t out_sides[], uint32_t out_components[],
                                   size_t out_local_dofs[], double out_coefficients[], size_t out_row_offsets[])
{
    const size_t component_count = combination_total_count((uint8_t)spec->bdim, (uint8_t)spec->order);

    size_t rows;
    size_t cols;
    boundary_mass_shape(spec, work, &rows, &cols);
    (void)cols;

    size_t row = 0;
    size_t entry = 0;
    out_row_offsets[0] = 0;
    for (size_t component = 0; component < component_count; ++component)
    {
        const size_t row_dofs = work->row_offsets[component + 1] - work->row_offsets[component];
        if (row_dofs == 0)
            continue;
        const unsigned first_block = physical ? 0 : (unsigned)component;
        const unsigned block_end = physical ? (unsigned)component_count : (unsigned)component + 1u;
        for (size_t test_dof = 0; test_dof < row_dofs; ++test_dof)
        {
            for (unsigned block = first_block; block < block_end; ++block)
            {
                const size_t col_dofs = work->col_offsets[block + 1] - work->col_offsets[block];
                for (size_t element_dof = 0; element_dof < col_dofs; ++element_dof, ++entry)
                {
                    out_sides[entry] = side;
                    out_components[entry] = work->element_components[block];
                    out_local_dofs[entry] = element_dof;
                    out_coefficients[entry] = factor * matrix[(work->row_offsets[component] + test_dof) * row_stride +
                                                              work->col_offsets[block] + element_dof];
                }
            }
            out_row_offsets[row + test_dof + 1] = entry;
        }
        row += row_dofs;
    }
}

fdg_result_t constrain_elements_on_boundary_prepare(const constrain_elements_on_boundary_request_t *const request,
                                                    constrain_elements_on_boundary_work_t *work,
                                                    basis_spec_t *out_basis, integration_spec_t *out_integration,
                                                    constrain_elements_on_boundary_plan_t *const plan)
{
    const unsigned bdim = request->bdim;
    const unsigned ndim = request->ndim;
    const unsigned nelem = request->nelem;
    // Preconditions shared with the common space merge. A zero-dimensional boundary is allowed for scalar (order
    // zero) traces: point rows pair the elements' vertex value functionals through the endpoint tables.
    CUTL_ASSERT(request->nforms > 0, "At least one k-form is required.");
    CUTL_ASSERT(nelem > 0, "At least one element is required for boundary common space.");
    CUTL_ASSERT(ndim > 0, "Space must be at least 1D.");
    CUTL_ASSERT(bdim < ndim, "Boundary dimension must be less than element space dimension.");

    plan->ndim = ndim;
    plan->bdim = bdim;
    plan->nforms = request->nforms;
    plan->nelem = nelem;
    plan->integration_registry = request->integration_registry;
    plan->basis_registry = request->basis_registry;
    plan->boundary_basis = out_basis;
    plan->boundary_integration = out_integration;
    plan->total_values = 0;
    // NULL-fill every reference slot so a failed prepare still releases cleanly.
    for (size_t reference = 0; reference < (size_t)request->nforms * bdim; ++reference)
    {
        plan->rules[reference] = NULL;
        plan->boundary_sets[reference] = NULL;
        plan->boundary_sets_lower[reference] = NULL;
    }
    const size_t item_count = (size_t)request->nforms * nelem;
    for (size_t item = 0; item < item_count; ++item)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            plan->element_sets[item * ndim + axis] = NULL;
            plan->element_sets_lower[item * ndim + axis] = NULL;
            plan->element_endpoints[item * ndim + axis] = NULL;
            plan->element_endpoints_lower[item * ndim + axis] = NULL;
        }
    }

    for (unsigned iform = 0; iform < request->nforms; ++iform)
    {
        const boundary_element_space_t *const views = request->elements + (size_t)iform * nelem;
        for (unsigned ie = 1; ie < nelem; ++ie)
        {
            CUTL_ASSERT(views[ie].order == views[0].order, "All elements must trace the same k-form order.");
        }
        basis_spec_t *const form_basis = out_basis + (size_t)iform * bdim;
        integration_spec_t *const form_integration = out_integration + (size_t)iform * bdim;
        boundary_common_space_merge(ndim, bdim, nelem, views, form_basis, form_integration);
        for (unsigned idim = 0; idim < bdim; ++idim)
        {
            form_basis[idim].type = BASIS_LEGENDRE;
        }

        const unsigned order = views[0].order;
        CUTL_ASSERT(bdim > 0 || order == 0, "A zero-dimensional boundary can only trace scalar (order zero) forms.");
        basis_spec_t *const form_lower = plan->boundary_lower_specs + (size_t)iform * bdim;
        for (unsigned idim = 0; idim < bdim; ++idim)
        {
            // Order-zero axes cannot lose another degree; no component reads their lower table (its components
            // have no rows).
            form_lower[idim] = (basis_spec_t){.type = BASIS_LEGENDRE,
                                              .order = form_basis[idim].order > 0 ? form_basis[idim].order - 1u : 0u};
        }

        const size_t form = (size_t)iform * bdim;
        fdg_result_t res = integration_rule_registry_get_rules(request->integration_registry, bdim, form_integration,
                                                               plan->rules + form);
        if (res != FDG_SUCCESS)
        {
            return res;
        }
        res = basis_set_registry_get_basis_sets(request->basis_registry, bdim, plan->boundary_sets + form,
                                                plan->rules + form, form_basis);
        if (res != FDG_SUCCESS)
        {
            return res;
        }
        if (order > 0)
        {
            res = basis_set_registry_get_basis_sets(request->basis_registry, bdim, plan->boundary_sets_lower + form,
                                                    plan->rules + form, form_lower);
            if (res != FDG_SUCCESS)
            {
                return res;
            }
        }

        for (unsigned ie = 0; ie < nelem; ++ie)
        {
            const boundary_element_space_t *const element = views + ie;
            const size_t item = (size_t)iform * nelem + ie;
            // Classify every element axis as fixed or free, recording free axes' canonical face slots.
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                work->axis_fixed[axis] = false;
                work->axis_slot[axis] = 0;
            }
            for (unsigned face_axis = 0; face_axis < bdim; ++face_axis)
            {
                const int8_t mapping = element->orientation[ndim - bdim + face_axis];
                work->axis_slot[(unsigned)(mapping < 0 ? -mapping : mapping) - 1] = face_axis;
            }
            for (unsigned fixed_axis = 0; fixed_axis < ndim - bdim; ++fixed_axis)
            {
                const int8_t mapping = element->orientation[fixed_axis];
                work->axis_fixed[(unsigned)(mapping < 0 ? -mapping : mapping) - 1] = true;
            }
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                // A zero-dimensional boundary has no boundary rules: every axis is fixed, only endpoints are read.
                work->element_rules[axis] =
                    bdim > 0 ? plan->rules[form + (work->axis_fixed[axis] ? 0 : work->axis_slot[axis])] : NULL;
            }
            basis_spec_t *const item_lower = plan->element_lower_specs + item * ndim;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                item_lower[axis] =
                    (basis_spec_t){.type = element->basis[axis].type,
                                   .order = element->basis[axis].order > 0 ? element->basis[axis].order - 1u : 0u};
            }
            if (bdim > 0)
            {
                res = basis_set_registry_get_basis_sets(request->basis_registry, ndim, plan->element_sets + item * ndim,
                                                        work->element_rules, element->basis);
                if (res != FDG_SUCCESS)
                {
                    return res;
                }
                if (order > 0)
                {
                    res = basis_set_registry_get_basis_sets(request->basis_registry, ndim,
                                                            plan->element_sets_lower + item * ndim, work->element_rules,
                                                            item_lower);
                    if (res != FDG_SUCCESS)
                    {
                        return res;
                    }
                }
            }
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                if (!work->axis_fixed[axis])
                    continue;
                res = basis_set_registry_get_basis_endpoints(
                    request->basis_registry, plan->element_endpoints + item * ndim + axis, element->basis[axis]);
                if (res != FDG_SUCCESS)
                {
                    return res;
                }
                if (order > 0 && element->basis[axis].order > 0)
                {
                    res = basis_set_registry_get_basis_endpoints(
                        request->basis_registry, plan->element_endpoints_lower + item * ndim + axis, item_lower[axis]);
                    if (res != FDG_SUCCESS)
                    {
                        return res;
                    }
                }
            }

            const kform_spec_t element_spec = {.ndim = ndim, .order = order, .basis = element->basis};
            const constraint_boundary_mass_spec_t spec = {.ndim = ndim,
                                                          .bdim = bdim,
                                                          .order = order,
                                                          .element_spec = &element_spec,
                                                          .boundary_basis = form_basis,
                                                          .boundary_integration = form_integration,
                                                          .orientation = element->orientation};
            const bool physical =
                !request->c1_continuous && request->test_pullbacks != NULL && request->test_pullbacks[item] != NULL;
            size_t rows;
            size_t cols;
            size_t entries;
            constraint_boundary_mass_layout(&spec, &work->mass, physical, &rows, &cols, &entries);
            plan->item_rows[item] = rows;
            plan->item_cols[item] = cols;
            plan->item_offsets[item] = plan->total_values;
            plan->total_values += rows * cols;
        }
    }
    plan->item_offsets[item_count] = plan->total_values;
    return FDG_SUCCESS;
}

void constrain_elements_on_boundary_work_size(const constrain_elements_on_boundary_request_t *const request,
                                              const constrain_elements_on_boundary_plan_t *const plan,
                                              constrain_elements_on_boundary_work_t *const work,
                                              size_t *const out_weights, size_t *const out_row_values,
                                              size_t *const out_col_values)
{
    size_t max_weights = 0;
    size_t max_row_values = 0;
    size_t max_col_values = 0;
    for (unsigned iform = 0; iform < request->nforms; ++iform)
    {
        const boundary_element_space_t *const views = request->elements + (size_t)iform * request->nelem;
        const unsigned order = views[0].order;
        const basis_spec_t *const form_basis = plan->boundary_basis + (size_t)iform * request->bdim;
        const integration_spec_t *const form_integration = plan->boundary_integration + (size_t)iform * request->bdim;
        const size_t point_count = integration_specs_total_points(request->bdim, form_integration);
        max_weights = max_weights > point_count ? max_weights : point_count;
        for (unsigned ie = 0; ie < request->nelem; ++ie)
        {
            const kform_spec_t element_spec = {.ndim = request->ndim, .order = order, .basis = views[ie].basis};
            const constraint_boundary_mass_spec_t spec = {.ndim = request->ndim,
                                                          .bdim = request->bdim,
                                                          .order = order,
                                                          .element_spec = &element_spec,
                                                          .boundary_basis = form_basis,
                                                          .boundary_integration = form_integration,
                                                          .orientation = views[ie].orientation};
            constraint_boundary_mass_work_sizes_t sizes;
            constraint_boundary_mass_work_size(&spec, &work->mass, &sizes);
            max_row_values = max_row_values > sizes.row_values ? max_row_values : sizes.row_values;
            max_col_values = max_col_values > sizes.col_values ? max_col_values : sizes.col_values;
        }
    }
    *out_weights = max_weights;
    *out_row_values = max_row_values;
    *out_col_values = max_col_values;
}

void constrain_elements_on_boundary_assemble(const constrain_elements_on_boundary_request_t *const request,
                                             const constrain_elements_on_boundary_plan_t *const plan,
                                             constrain_elements_on_boundary_work_t *work, double *const out_values)
{
    const unsigned bdim = plan->bdim;
    const unsigned ndim = plan->ndim;
    for (unsigned iform = 0; iform < plan->nforms; ++iform)
    {
        const boundary_element_space_t *const views = request->elements + (size_t)iform * plan->nelem;
        const unsigned order = views[0].order;
        if (order > bdim)
        {
            // No trace components past the boundary dimension: prepare recorded empty item slices for this form.
            continue;
        }
        const size_t form = (size_t)iform * bdim;
        const basis_spec_t *const form_basis = plan->boundary_basis + form;
        const integration_spec_t *const form_integration = plan->boundary_integration + form;
        integration_rule_tensor_weights(bdim, plan->rules + form, work->weights);

        for (unsigned ie = 0; ie < plan->nelem; ++ie)
        {
            const boundary_element_space_t *const element = views + ie;
            const size_t item = (size_t)iform * plan->nelem + ie;
            // Classify every element axis as fixed or free, recording free axes' canonical face slots.
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                work->axis_fixed[axis] = false;
                work->axis_slot[axis] = 0;
            }
            for (unsigned face_axis = 0; face_axis < bdim; ++face_axis)
            {
                const int8_t mapping = element->orientation[ndim - bdim + face_axis];
                work->axis_slot[(unsigned)(mapping < 0 ? -mapping : mapping) - 1] = face_axis;
            }
            for (unsigned fixed_axis = 0; fixed_axis < ndim - bdim; ++fixed_axis)
            {
                const int8_t mapping = element->orientation[fixed_axis];
                work->axis_fixed[(unsigned)(mapping < 0 ? -mapping : mapping) - 1] = true;
            }
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                work->element_rules[axis] = plan->rules[form + (work->axis_fixed[axis] ? 0 : work->axis_slot[axis])];
            }

            const kform_spec_t element_spec = {.ndim = ndim, .order = order, .basis = element->basis};
            const constraint_boundary_mass_spec_t spec = {.ndim = ndim,
                                                          .bdim = bdim,
                                                          .order = order,
                                                          .element_spec = &element_spec,
                                                          .boundary_basis = form_basis,
                                                          .boundary_integration = form_integration,
                                                          .orientation = element->orientation};
            const constraint_boundary_mass_request_t mass_request = {
                .spec = &spec,
                .boundary_basis_sets = plan->boundary_sets + form,
                .boundary_basis_sets_lower = order > 0 ? plan->boundary_sets_lower + form : NULL,
                .element_basis_sets = plan->element_sets + item * ndim,
                .element_basis_sets_lower = order > 0 ? plan->element_sets_lower + item * ndim : NULL,
                .element_endpoints = plan->element_endpoints + item * ndim,
                .element_endpoints_lower = order > 0 ? plan->element_endpoints_lower + item * ndim : NULL,
                .point_weights = work->weights,
                .surface_weights = request->surface_weights != NULL ? request->surface_weights[item] : NULL,
                .test_pullback =
                    !request->c1_continuous && request->test_pullbacks != NULL ? request->test_pullbacks[item] : NULL,
                .element_pullback = !request->c1_continuous && request->element_pullbacks != NULL
                                        ? request->element_pullbacks[item]
                                        : NULL,
                .factor = 1.0,
                .work = &work->mass,
                .out_matrix = out_values + plan->item_offsets[item],
            };
            constraint_boundary_mass_assemble(&mass_request);
        }
    }
}

void constrain_elements_on_boundary_plan_release(constrain_elements_on_boundary_plan_t *const plan)
{
    const unsigned bdim = plan->bdim;
    const unsigned ndim = plan->ndim;
    for (unsigned iform = 0; iform < plan->nforms; ++iform)
    {
        const size_t form = (size_t)iform * bdim;
        for (unsigned idim = 0; idim < bdim; ++idim)
        {
            if (plan->rules[form + idim] != NULL)
            {
                integration_rule_registry_release_rule(plan->integration_registry, plan->rules[form + idim]);
                plan->rules[form + idim] = NULL;
            }
            if (plan->boundary_sets[form + idim] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry, plan->boundary_sets[form + idim]);
                plan->boundary_sets[form + idim] = NULL;
            }
            if (plan->boundary_sets_lower[form + idim] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry, plan->boundary_sets_lower[form + idim]);
                plan->boundary_sets_lower[form + idim] = NULL;
            }
        }
    }
    for (size_t item = 0; item < (size_t)plan->nforms * plan->nelem; ++item)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            if (plan->element_endpoints_lower[item * ndim + axis] != NULL)
            {
                basis_set_registry_release_basis_endpoints(plan->basis_registry,
                                                           plan->element_endpoints_lower[item * ndim + axis]);
                plan->element_endpoints_lower[item * ndim + axis] = NULL;
            }
            if (plan->element_endpoints[item * ndim + axis] != NULL)
            {
                basis_set_registry_release_basis_endpoints(plan->basis_registry,
                                                           plan->element_endpoints[item * ndim + axis]);
                plan->element_endpoints[item * ndim + axis] = NULL;
            }
            if (plan->element_sets_lower[item * ndim + axis] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry,
                                                     plan->element_sets_lower[item * ndim + axis]);
                plan->element_sets_lower[item * ndim + axis] = NULL;
            }
            if (plan->element_sets[item * ndim + axis] != NULL)
            {
                basis_set_registry_release_basis_set(plan->basis_registry, plan->element_sets[item * ndim + axis]);
                plan->element_sets[item * ndim + axis] = NULL;
            }
        }
    }
}

void constraint_physical_side_load_work_size(const kform_spec_t *const test_spec, size_t *const out_face_axes,
                                             size_t *const out_datum_axes, size_t *const out_iterator)
{
    const unsigned order = test_spec->order;
    *out_face_axes = order == 0 ? 1u : order;
    *out_datum_axes = *out_face_axes + 1u;
    *out_iterator = combination_iterator_required_memory((uint8_t)order);
}

void constraint_physical_side_load(const kform_spec_t *const test_spec, const constraint_element_side_t *const side,
                                   const double *const point_weights, const double *const datum_values,
                                   const double *const surface_weights, const kform_values_table_t *const element_table,
                                   constraint_physical_side_load_work_t *const work, double values[])
{
    ASSERT(side->ndim == test_spec->ndim + 1, "The load is defined on codimension-one faces.");
    const unsigned face_dim = test_spec->ndim;
    const unsigned order = test_spec->order;
    const kform_spec_t element_spec = {.ndim = side->ndim, .order = order, .basis = side->basis_specs};
    const size_t point_count = element_table->point_count;
    // The datum is an element-frame k-form (k = test_spec->order + 1) given as its C(n, k) physical components at
    // the canonical face points: datum_values[component * point_count + point]. Each face (k-1)-form component J with
    // element-frame axes J_e pairs with datum component I = J_e U {fixed_axis}; the sign counts J_e axes below the
    // fixed normal axis (at k = n this is the sigma_out = side * (-1)^a formula).
    const int8_t fixed_mapping = side->orientation[0];
    const unsigned fixed_axis = (unsigned)(fixed_mapping < 0 ? -fixed_mapping : fixed_mapping) - 1;
    const bool side_sign = fixed_mapping < 0;
    combination_iterator_init(work->face_components, (uint8_t)face_dim, (uint8_t)order);
    for (const uint8_t *face_axes = combination_iterator_current(work->face_components);
         !combination_iterator_is_done(work->face_components); combination_iterator_next(work->face_components))
    {
        unsigned element_component;
        const bool orientation_sign =
            mapped_component(side, face_dim, order, face_axes, work->mapped_axes, &element_component);
        const size_t element_dof_count = kform_spec_component_dof_count(&element_spec, element_component);
        const size_t element_start = element_table->component_offsets[element_component];
        const size_t element_block_dofs = element_table->component_offsets[element_component + 1] -
                                          element_table->component_offsets[element_component];
        const size_t element_block_start = element_start * point_count;
        kform_component_axes(&element_spec, element_component, work->element_axes);
        unsigned exponent_below_fixed = 0;
        for (unsigned i = 0; i < order; ++i)
            exponent_below_fixed += work->element_axes[i] < fixed_axis ? 1u : 0u;
        for (unsigned i = 0; i < order; ++i)
            work->datum_axes[i] = work->element_axes[i];
        work->datum_axes[order] = (uint8_t)fixed_axis;
        for (unsigned i = order; i > 0 && work->datum_axes[i - 1] > work->datum_axes[i]; --i)
        {
            const uint8_t tmp = work->datum_axes[i - 1];
            work->datum_axes[i - 1] = work->datum_axes[i];
            work->datum_axes[i] = tmp;
        }
        const unsigned datum_component = combination_get_index(side->ndim, order + 1, work->datum_axes);
        const double sign = (side_sign ^ orientation_sign ^ (exponent_below_fixed % 2)) ? -1.0 : 1.0;
        for (size_t point = 0; point < point_count; ++point)
        {
            const double weighted_datum = point_weights[point] * (surface_weights ? surface_weights[point] : 1.0) *
                                          datum_values[datum_component * point_count + point] * sign;
            const double *restrict element_point =
                element_table->values + element_block_start + point * element_block_dofs;
            double *restrict out_values = values + element_start;
            for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
            {
                out_values[element_dof] += weighted_datum * element_point[element_dof];
            }
        }
    }
}

unsigned constraint_face_source_axis(const unsigned element_dim, const unsigned face_dim,
                                     const int8_t orientation[const static element_dim], const unsigned element_axis)
{
    const unsigned fixed_count = element_dim - face_dim;
    unsigned face_axis = 0;
    for (unsigned axis = 0; axis < element_axis; ++axis)
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

void constraint_face_canonical_specs(const unsigned element_dim, const unsigned face_dim,
                                     const int8_t orientation[const static element_dim],
                                     const integration_spec_t face_specs[const static face_dim],
                                     integration_spec_t out_specs[const static face_dim])
{
    const unsigned fixed_count = element_dim - face_dim;
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = orientation[fixed_count + face_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        out_specs[face_axis] =
            face_specs[constraint_face_source_axis(element_dim, face_dim, orientation, element_axis)];
    }
}

size_t constraint_face_point_to_source(const unsigned element_dim, const unsigned face_dim,
                                       const int8_t orientation[const static element_dim],
                                       const integration_spec_t source_specs[const static face_dim],
                                       const integration_spec_t canonical_specs[const static face_dim],
                                       const size_t canonical_strides[const static face_dim],
                                       const size_t source_strides[const static face_dim], const size_t canonical_point)
{
    const unsigned fixed_count = element_dim - face_dim;
    size_t source_point = 0;
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = orientation[fixed_count + face_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        const unsigned source_face_axis = constraint_face_source_axis(element_dim, face_dim, orientation, element_axis);
        const size_t digit =
            (canonical_point / canonical_strides[face_axis]) % ((size_t)canonical_specs[face_axis].order + 1);
        source_point += (mapping < 0 ? (size_t)source_specs[source_face_axis].order - digit : digit) *
                        source_strides[source_face_axis];
    }
    return source_point;
}

// Combination index of the sorted copy of @p axes: the component's row in the face map's transform table.
static unsigned sorted_row(const unsigned face_dim, const unsigned order,
                           uint8_t axes[const static order == 0 ? 1 : order])
{
    for (unsigned i = 0; i < order; ++i)
    {
        for (unsigned j = i + 1; j < order; ++j)
        {
            if (axes[i] > axes[j])
            {
                const uint8_t tmp = axes[i];
                axes[i] = axes[j];
                axes[j] = tmp;
            }
        }
    }
    return combination_get_index((uint8_t)face_dim, (uint8_t)order, axes);
}

void constraint_trace_pullback_build_work_size(const constraint_trace_pullback_build_t *const request,
                                               constraint_trace_pullback_build_work_sizes_t *const out_sizes)
{
    const unsigned order = request->order == 0 ? 1u : request->order;
    *out_sizes = (constraint_trace_pullback_build_work_sizes_t){
        .face_axis_count = request->face_dim,
        .element_axis_count = request->element_dim,
        .element_component_map =
            request->element_components
                ? (size_t)combination_total_count((uint8_t)request->element_dim, (uint8_t)request->order) + 1u
                : 1u,
        .axes_scratch = order,
        .iterator_memory = combination_iterator_required_memory((uint8_t)request->order),
    };
}

void constraint_trace_pullback_build(const constraint_trace_pullback_build_t *const request)
{
    const constraint_trace_pullback_build_work_t *const work = request->work;
    const unsigned face_component_count = request->face_component_count;
    const unsigned physical_component_count = request->physical_component_count;
    const size_t point_count = request->canonical_point_count;
    const size_t total = (size_t)request->element_dim > 0
                             ? (size_t)combination_total_count((uint8_t)request->element_dim, (uint8_t)request->order) *
                                   physical_component_count * point_count
                             : (size_t)physical_component_count * point_count;
    if (request->order == 0)
    {
        // Scalar traces carry no tangential pullback; the table stays zero.
        for (size_t i = 0; i < total; ++i)
            request->out[i] = 0.0;
        return;
    }

    const constraint_element_side_t side = {
        .ndim = request->element_dim, .basis_specs = NULL, .orientation = request->orientation};
    // The canonical-to-source point map is component independent: derive the loop-invariant per-axis decode data
    // once, then decode every point with one divide-modulo pair per axis.
    const unsigned fixed_count = request->element_dim - request->face_dim;
    unsigned *const axis_source_slots = work->axis_source_slots;
    unsigned *const axis_orders = work->axis_orders;
    size_t *const axis_source_strides = work->axis_source_strides;
    size_t *const axis_canonical_strides = work->axis_canonical_strides;
    int *const axis_mirrored = work->axis_mirrored;
    // Rank of every element axis among the free axes (the face map's own order), and whether the axis is free.
    bool *const element_axis_free = work->element_axis_free;
    unsigned *const element_source_rank = work->element_source_rank;
    for (unsigned axis = 0; axis < request->element_dim; ++axis)
        element_axis_free[axis] = false;
    for (unsigned face_axis = 0; face_axis < request->face_dim; ++face_axis)
    {
        const int8_t mapping = request->orientation[fixed_count + face_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        axis_source_slots[face_axis] =
            constraint_face_source_axis(request->element_dim, request->face_dim, request->orientation, element_axis);
        axis_mirrored[face_axis] = mapping < 0;
        axis_orders[face_axis] = request->canonical_specs[face_axis].order;
        axis_source_strides[face_axis] = request->source_strides[axis_source_slots[face_axis]];
        axis_canonical_strides[face_axis] = request->canonical_strides[face_axis];
        element_axis_free[element_axis] = true;
        element_source_rank[element_axis] = axis_source_slots[face_axis];
    }
    // Face component indexing enumerates the canonical boundary form's components, writing each mapped element
    // component's block from the face component's transform rows; element component indexing enumerates the element
    // k-form's own components, reading each one's rows from the inverse-mapped face component.
    unsigned *const element_to_face = work->element_to_face;
    const unsigned element_total =
        (unsigned)combination_total_count((uint8_t)request->element_dim, (uint8_t)request->order);
    if (request->element_components)
    {
        for (unsigned index = 0; index < element_total; ++index)
            element_to_face[index] = UINT8_MAX + 1;
        combination_iterator_init(work->components, (uint8_t)request->face_dim, (uint8_t)request->order);
        size_t face_component = 0;
        for (const uint8_t *component_axes = combination_iterator_current(work->components);
             !combination_iterator_is_done(work->components);
             combination_iterator_next(work->components), ++face_component)
        {
            unsigned element_component;
            (void)mapped_component(&side, request->face_dim, request->order, component_axes, work->mapped_axes,
                                   &element_component);
            CUTL_ASSERT(element_component < element_total, "Mapped element component out of range.");
            element_to_face[element_component] = (unsigned)face_component;
        }
    }
    // The mode selects the enumerated space: element components read their face counterpart through the map;
    // canonical and plain modes enumerate the canonical boundary form's components directly.
    const bool element_mode = request->element_components;
    combination_iterator_init(work->components, (uint8_t)(element_mode ? request->element_dim : request->face_dim),
                              (uint8_t)request->order);
    size_t component = 0;
    for (const uint8_t *component_axes = combination_iterator_current(work->components);
         !combination_iterator_is_done(work->components); combination_iterator_next(work->components), ++component)
    {
        const unsigned face_component = element_mode ? element_to_face[component] : (unsigned)component;
        if (element_mode && !request->canonical_components && face_component > face_component_count)
        {
            // A component whose covectors all lie on the fixed normal axes has no tangential face counterpart; the
            // engine never reads its block. Zero it for hygiene.
            for (unsigned physical_component = 0; physical_component < physical_component_count; ++physical_component)
            {
                double *const target =
                    request->out + ((size_t)component * physical_component_count + physical_component) * point_count;
                for (size_t point = 0; point < point_count; ++point)
                    target[point] = 0.0;
            }
            continue;
        }
        CUTL_ASSERT(face_component <= face_component_count, "Element component has no face component mapping.");
        unsigned element_component;
        if (element_mode)
        {
            element_component = (unsigned)component;
        }
        else if (request->canonical_components)
        {
            // The canonical boundary form's components index their own transform rows; no element-side mapping.
            element_component = (unsigned)component;
        }
        else
        {
            mapped_component(&side, request->face_dim, request->order, component_axes, work->mapped_axes,
                             &element_component);
        }
        // Transform rows follow the face map's free-axis order, so a component with a different axis order reads the
        // row of its axes' ranks (canonical axes via the orientation, element axes directly). Canonical rows carry the
        // covector orientation — every mirrored axis flips it and the mapped-axis sort once per transposition — so
        // the table holds the canonical covector's physical image; element rows stay the element's own image.
        // Consumers pair the two physical images without further orientation signs.
        uint8_t *const source_axes = work->source_axes;
        unsigned source_row = face_component;
        int value_sign = 1;
        if (request->canonical_components)
        {
            for (unsigned i = 0; i < request->order; ++i)
            {
                source_axes[i] = (uint8_t)axis_source_slots[component_axes[i]];
                if (axis_mirrored[component_axes[i]])
                    value_sign = -value_sign;
            }
            for (unsigned i = 0; i < request->order; ++i)
            {
                for (unsigned j = i + 1; j < request->order; ++j)
                {
                    if (source_axes[i] > source_axes[j])
                        value_sign = -value_sign;
                }
            }
            source_row = sorted_row(request->face_dim, request->order, source_axes);
        }
        else if (element_mode)
        {
            bool tangential = true;
            for (unsigned i = 0; i < request->order; ++i)
            {
                const unsigned element_axis = component_axes[i];
                tangential &= element_axis_free[element_axis];
                source_axes[i] = (uint8_t)element_source_rank[element_axis];
            }
            if (tangential)
                source_row = sorted_row(request->face_dim, request->order, source_axes);
        }
        for (size_t canonical_point = 0; canonical_point < point_count; ++canonical_point)
        {
            size_t source_point = 0;
            for (unsigned face_axis = 0; face_axis < request->face_dim; ++face_axis)
            {
                const size_t digit =
                    (canonical_point / axis_canonical_strides[face_axis]) % ((size_t)axis_orders[face_axis] + 1);
                source_point += (axis_mirrored[face_axis] ? (size_t)axis_orders[face_axis] - digit : digit) *
                                axis_source_strides[face_axis];
            }
            for (unsigned physical_component = 0; physical_component < physical_component_count; ++physical_component)
            {
                const size_t source_index =
                    ((size_t)source_row * physical_component_count + physical_component) * request->source_point_count +
                    source_point;
                const size_t target_index =
                    ((size_t)element_component * physical_component_count + physical_component) * point_count +
                    canonical_point;
                request->out[target_index] = (double)value_sign * request->transform[source_index];
            }
        }
    }
}

void boundary_space_map_resample_work_size(const unsigned bdim, const unsigned coords,
                                           const integration_rule_t *const *source_rules,
                                           const integration_rule_t *const *target_rules,
                                           size_t *const out_axis_matrices, size_t *const out_positions,
                                           size_t *const out_jacobian, size_t *const out_q,
                                           size_t *const out_scratch_bytes)
{
    size_t matrices = 0;
    size_t points = 1;
    for (unsigned axis = 0; axis < bdim; ++axis)
    {
        matrices += (size_t)(target_rules[axis]->spec.order + 1) * (source_rules[axis]->spec.order + 1);
        points *= (size_t)target_rules[axis]->spec.order + 1;
    }
    *out_axis_matrices = matrices;
    *out_positions = points * coords;
    *out_jacobian = (size_t)bdim * coords;
    *out_q = (size_t)coords * coords;
    *out_scratch_bytes = (size_t)bdim * (2 * sizeof(unsigned) + sizeof(const double *) + sizeof(integration_spec_t));
}

void boundary_space_map_resample(const boundary_space_map_resample_request_t *const request)
{
    const unsigned bdim = request->bdim;
    const unsigned coords = request->coords;
    unsigned *const target_orders = request->target_orders;
    unsigned *const source_orders = request->source_orders;
    const double **const axis_matrices = request->axis_matrix_rows;
    integration_spec_t *const target_specs = request->target_specs;

    size_t offset = 0;
    for (unsigned axis = 0; axis < bdim; ++axis)
    {
        const unsigned n_out = request->target_rules[axis]->spec.order + 1u;
        const unsigned n_in = request->source_rules[axis]->spec.order + 1u;
        // Interpolation matrix from source to target nodes: entry (in, out) holds the target-node value of the
        // source node's Lagrange polynomial.
        lagrange_polynomial_values_transposed_2(n_out, integration_rule_nodes_const(request->target_rules[axis]), n_in,
                                                integration_rule_nodes_const(request->source_rules[axis]),
                                                request->axis_matrices + offset);
        axis_matrices[axis] = request->axis_matrices + offset;
        offset += (size_t)n_out * n_in;
        target_orders[axis] = n_out - 1u;
        source_orders[axis] = n_in - 1u;
        target_specs[axis] = request->target_rules[axis]->spec;
    }

    interpolate_sampled_map(bdim, coords, target_orders, source_orders, axis_matrices, request->coordinate_values,
                            request->coordinate_gradients, integration_specs_total_points(bdim, target_specs),
                            request->positions, request->out_determinant, request->out_inverse_maps, request->jacobian,
                            request->q);
}
