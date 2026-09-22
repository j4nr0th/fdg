/**
 * @file constraints.c
 * @brief Implementation of reference and physical trace constraints.
 *
 * Assembly uses one canonical face coordinate system. Element-side
 * orientations map that system to signed, one-based element axes; the helper
 * functions below keep the mapping and its alternating k-form sign in one
 * place. Routines trust documented preconditions guarded by debug asserts
 * only, storage is sized with the `*_layout`/`*_work_size` functions, and
 * registry-backed routines report allocation failures through
 * `fdg_result_t`.
 */

#include "constraints.h"

#include <limits.h>
#include <math.h>

#include <cutl/allocators.h>
#include <cutl/iterators/combination_iterator.h>

#include "../operations/map_transforms.h"
#include "../polynomials/lagrange.h"

/**
 * @brief Test whether a component contains one active covector axis.
 *
 * The active axes are sorted, but a linear scan keeps this helper independent
 * of the combination representation and is negligible beside quadrature work.
 */
static bool component_has_axis(const unsigned order, const uint8_t axes[const static order == 0 ? 1 : order],
                               const unsigned axis)
{
    for (unsigned i = 0; i < order; ++i)
    {
        if (axes[i] == axis)
            return true;
    }
    return false;
}

/**
 * @brief Map a face component's axes into element axes and a sign.
 *
 * The side orientation contributes one sign for every reversed mapped axis.
 * The mapped axes are sorted into canonical element order, and each sorting
 * swap flips the alternating covector sign by one permutation transposition.
 *
 * Preconditions: `side->orientation` is a signed one-based permutation whose
 * fixed-axis prefix increases in absolute value; `order <= boundary_dim <=
 * side->ndim`; `test_axes` are the sorted covector axes of a valid component.
 */
static void mapped_axes_and_sign(const constraint_element_side_t *const side, const unsigned boundary_dim,
                                 const unsigned order, const uint8_t test_axes[const static order == 0 ? 1 : order],
                                 uint8_t mapped_axes[const static order == 0 ? 1 : order], int *const out_sign)
{
    const unsigned fixed_count = side->ndim - boundary_dim;
    int sign = 1;
    // Collect the mapped axes from the side's orientation and get the initial sign
    for (unsigned i = 0; i < order; ++i)
    {
        const int8_t mapping = side->orientation[fixed_count + test_axes[i]];
        mapped_axes[i] = (uint8_t)(mapping < 0 ? -mapping : mapping) - 1;
        if (mapping < 0)
            sign = -sign;
    }
    // Bubble sort the mapped axes to the canonical order and adjust the sign
    // accordingly: each swap flips the alternating covector sign by one
    // permutation transposition.
    for (unsigned i = 0; i < order; ++i)
    {
        for (unsigned j = i + 1; j < order; ++j)
        {
            if (mapped_axes[i] > mapped_axes[j])
            {
                sign = -sign;
                const uint8_t tmp = mapped_axes[i];
                mapped_axes[i] = mapped_axes[j];
                mapped_axes[j] = tmp;
            }
        }
    }
    *out_sign = sign;
}

/**
 * @brief Map a face component's axes into an element component.
 *
 * The side orientation contributes one sign for every reversed mapped axis.
 * Sorting the mapped axes into canonical element order contributes the
 * permutation parity. Together these signs are the pullback sign of the
 * covector component, while `out_component` is its combination index.
 *
 * Preconditions: `side->orientation` is a signed one-based permutation whose
 * fixed-axis prefix increases in absolute value; `order <= boundary_dim <=
 * side->ndim`; `test_axes` are the sorted covector axes of a valid component.
 */
static void mapped_component(const constraint_element_side_t *const side, const unsigned boundary_dim,
                             const unsigned order, const uint8_t test_axes[const static order == 0 ? 1 : order],
                             unsigned *const out_component, int *const out_sign)
{
    // Fixed-size scratch bounded by the form order; order zero still gets one
    // harmless placeholder.
    uint8_t mapped_axes[UINT8_MAX];
    mapped_axes_and_sign(side, boundary_dim, order, test_axes, mapped_axes, out_sign);

    // Get the component index based on the element's mapped axes
    *out_component = combination_get_index(side->ndim, order, mapped_axes);
}

/**
 * @brief Take the physical-component dot product of two pullback samples.
 *
 * Values are strided by `point_count`, so this selects one point from two
 * component blocks without copying either sampled vector.
 */
static double trace_pullback_dot(const constraint_trace_pullback_t *const pullback, const unsigned first_component,
                                 const unsigned second_component, const unsigned physical_component_count,
                                 const size_t point_count, const size_t point)
{
    const double *const first =
        pullback->values + ((size_t)first_component * physical_component_count * point_count + point);
    const double *const second =
        pullback->values + ((size_t)second_component * physical_component_count * point_count + point);
    double result = 0.0;
    for (unsigned physical_component = 0; physical_component < physical_component_count; ++physical_component)
    {
        result += first[(size_t)physical_component * point_count] * second[(size_t)physical_component * point_count];
    }
    return result;
}

/**
 * @brief Accumulate one test × element trace inner-product block.
 *
 * Computes `out[i * row_stride + column + j] += factor(point) *
 * test_values[point * test_dofs + i] * element_values[point * element_dofs + j]`
 * for every quadrature point, where the point factor is the quadrature weight
 * times the optional surface measure times @p factor and, when a pullback is
 * given, the physical dot product of the two pullback samples. The block is
 * zero-initialized first, so disjoint column ranges accumulate independently.
 *
 * Preconditions: the test and element component blocks hold `test_dofs *
 * point_count` and `element_dofs * point_count` entries point-major; the
 * column range `[column, column + element_dofs)` of every written row
 * belongs to this block alone.
 */
static void trace_component_block(const size_t point_count, const size_t test_dofs, const size_t element_dofs,
                                  const double *restrict test_values, const double *restrict element_values,
                                  const double *restrict point_weights, const double *restrict surface_weights,
                                  const double factor, const constraint_trace_pullback_t *restrict pullback,
                                  const unsigned first_component, const unsigned second_component, const size_t column,
                                  const size_t row_stride, double *restrict out)
{
    for (size_t i = 0; i < test_dofs; ++i)
    {
        double *restrict out_row = out + i * row_stride + column;
        for (size_t j = 0; j < element_dofs; ++j)
        {
            out_row[j] = 0.0;
        }
    }
    const unsigned physical_component_count = pullback ? pullback->physical_component_count : 1;
    for (size_t point = 0; point < point_count; ++point)
    {
        double point_factor = point_weights[point] * (surface_weights ? surface_weights[point] : 1.0) * factor;
        if (pullback)
        {
            point_factor *= trace_pullback_dot(pullback, first_component, second_component, physical_component_count,
                                               point_count, point);
        }
        const double *restrict test_point = test_values + point * test_dofs;
        const double *restrict element_point = element_values + point * element_dofs;
        for (size_t i = 0; i < test_dofs; ++i)
        {
            const double scaled = point_factor * test_point[i];
            double *restrict out_row = out + i * row_stride + column;
            for (size_t j = 0; j < element_dofs; ++j)
            {
                out_row[j] += scaled * element_point[j];
            }
        }
    }
}

/**
 * @brief Compute the mapped size of one test component on one side.
 *
 * Reference pairing contributes the mapped component's DoFs; physical pairing
 * visits every face component because the pullback can couple components.
 */
static size_t side_entries_per_test_component(const constraint_element_side_t *const side, const unsigned boundary_dim,
                                              const unsigned order,
                                              const uint8_t face_axes[const static order == 0 ? 1 : order])
{
    unsigned element_component;
    int orientation_sign;
    mapped_component(side, boundary_dim, order, face_axes, &element_component, &orientation_sign);
    const kform_spec_t element_spec = {.ndim = side->ndim, .order = order, .basis = side->basis_specs};
    return kform_spec_component_dof_count(&element_spec, element_component);
}

/**
 * @brief Merge the per-element boundary views into one common boundary space.
 *
 * The first element seeds the canonical boundary basis and integration rules;
 * every other element then lowers a basis axis to its own order when smaller
 * and raises an integration axis to its own rule when more accurate. The
 * basis merge therefore takes the per-axis minimum order (the L2 link
 * target) while the integration merge takes the most accurate rule.
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
            // The merged test space is the lowest per-axis order among the
            // incident elements: an element boundary cannot be constrained
            // to a higher-order boundary solution, so higher-order traces
            // conform to the lower-order boundary state in the L2 sense.
            // Integration stays at the highest accuracy so the common
            // quadrature resolves every side's traced products.
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

void boundary_common_space(const boundary_common_space_request_t *const request, basis_spec_t *out_basis,
                           integration_spec_t *out_integration)
{
    // Assert preconditions for boundary common space.
    CUTL_ASSERT(request->bdim != 0, "0-D boundary common space is trivial, so do not use this.");
    CUTL_ASSERT(request->nelem > 1, "At least two elements are required for boundary common space.");
    CUTL_ASSERT(request->ndim > 1, "Space must be at least 2D.");
    CUTL_ASSERT(request->bdim < request->ndim, "Boundary dimension must be less than element space dimension.");

    boundary_common_space_merge(request->ndim, request->bdim, request->nelem, request->elements, out_basis,
                                out_integration);
    // Finally, force the basis set to use Legendre basis
    for (unsigned idim = 0; idim < request->bdim; ++idim)
    {
        out_basis[idim].type = BASIS_LEGENDRE;
    }
}

/**
 * @brief Per-axis test function counts of one boundary component's row block.
 *
 * Active covector axes read the order-1 basis (`order` functions); inactive
 * axes read the full basis with the first `axis_skip[axis]` functions
 * removed. The offsets carry the matching start index into each axis's
 * function table. `counts` and `offsets` are caller-provided `[bdim]` arrays.
 */
static void boundary_mass_row_axis_counts(const constraint_boundary_mass_spec_t *const spec,
                                          const uint8_t component_axes[const static spec->order == 0 ? 1 : spec->order],
                                          unsigned counts[spec->bdim], unsigned offsets[spec->bdim])
{
    for (unsigned axis = 0; axis < spec->bdim; ++axis)
    {
        const bool active = component_has_axis(spec->order, component_axes, axis);
        const unsigned full_count = spec->boundary_basis[axis].order + 1u;
        const unsigned skip = !active && spec->axis_skip != NULL ? spec->axis_skip[axis] : 0u;
        // An active axis of order zero yields zero test functions, so the
        // whole component's row block drops out.
        if (active)
        {
            counts[axis] = spec->boundary_basis[axis].order;
            offsets[axis] = 0;
        }
        else
        {
            counts[axis] = full_count > skip ? full_count - skip : 0u;
            offsets[axis] = skip;
        }
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
 * The values are laid out point-major with axis 0 the slowest DoF digit, and
 * the skipped functions shift each axis's read window by its offset. The
 * work buffers carry the point strides and the per-axis scratch; the point
 * strides must be filled before the first call.
 */
static void boundary_mass_row_values(const constraint_boundary_mass_spec_t *const spec,
                                     const basis_set_t *const *basis_sets, const basis_set_t *const *basis_sets_lower,
                                     const uint8_t component_axes[const static spec->order == 0 ? 1 : spec->order],
                                     const size_t point_count, constraint_boundary_mass_work_t *work)
{
    boundary_mass_row_axis_counts(spec, component_axes, work->counts, work->offsets);
    const unsigned bdim = spec->bdim;

    size_t dof_count = 1;
    for (unsigned axis = 0; axis < bdim; ++axis)
    {
        const bool active = component_has_axis(spec->order, component_axes, axis);
        work->axis_sets[axis] = active ? basis_sets_lower[axis] : basis_sets[axis];
        dof_count *= work->counts[axis];
        work->digits[axis] = 0;
    }
    for (size_t dof = 0; dof < dof_count; ++dof)
    {
        for (unsigned axis = 0; axis < bdim; ++axis)
        {
            work->axis_tables[axis] =
                basis_set_basis_values(work->axis_sets[axis], work->offsets[axis] + work->digits[axis]);
        }
        double *const out_column = work->row_values + dof;
        for (size_t point = 0; point < point_count; ++point)
        {
            double value = 1.0;
            for (unsigned axis = 0; axis < bdim; ++axis)
            {
                const size_t node =
                    (point / work->point_strides[axis]) % ((size_t)spec->boundary_integration[axis].order + 1);
                value *= work->axis_tables[axis][node];
            }
            out_column[point * dof_count] = value;
        }
        for (unsigned axis = bdim; axis-- > 0;)
        {
            if (++work->digits[axis] < work->counts[axis])
                break;
            work->digits[axis] = 0;
        }
    }
}

/**
 * @brief Build the trace value sources of the element's axes.
 *
 * Free axes read the element basis sets evaluated at the mapped common rules,
 * mirrored when the orientation reverses the axis; fixed normal axes read
 * their endpoint values at the signed end.
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
            .nodes = request->element_basis_sets[element_axis],
            .nodes_lower =
                request->element_basis_sets_lower != NULL ? request->element_basis_sets_lower[element_axis] : NULL,
            .rule_size = spec->boundary_integration[face_axis].order + 1u,
            .stride_slot = face_axis,
            .mirror = mapping < 0,
        };
    }
    for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
    {
        const int8_t mapping = spec->orientation[fixed_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        axes[element_axis] = (kform_trace_axis_t){
            .endpoint = request->element_endpoints != NULL ? request->element_endpoints[element_axis] : NULL,
            .endpoint_lower =
                request->element_endpoints_lower != NULL ? request->element_endpoints_lower[element_axis] : NULL,
            .end = mapping < 0 ? 0u : 1u,
        };
    }
}

/**
 * @brief Compute the dense shape of one element's boundary mass matrix.
 *
 * One combination-iterator pass over the common boundary components fills the
 * mapped component table and both offset tables: row offsets are per-component
 * starts with the total row count at `work->row_offsets[component_count]`;
 * column offsets are the mapped component block starts with the total column
 * count at `work->col_offsets[component_count]`.
 */
static void boundary_mass_shape(const constraint_boundary_mass_spec_t *const spec,
                                constraint_boundary_mass_work_t *work, size_t *const out_rows, size_t *const out_cols)
{
    const constraint_element_side_t side = {
        .ndim = spec->ndim, .basis_specs = spec->element_spec->basis, .orientation = spec->orientation};
    const unsigned order = spec->order;
    if (order > spec->bdim)
    {
        // A k-form of order past the boundary dimension has no trace
        // components at all.
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
        boundary_mass_row_axis_counts(spec, component_axes, work->counts, work->offsets);
        work->row_offsets[component] = rows;
        rows += boundary_mass_row_dofs(spec, work->counts);
        mapped_axes_and_sign(&side, spec->bdim, order, component_axes, work->mapped_axes,
                             work->element_signs + component);
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

void constraint_boundary_mass_work_size(const constraint_boundary_mass_spec_t *const spec,
                                        constraint_boundary_mass_work_sizes_t *const out_sizes)
{
    const size_t point_count = integration_specs_total_points(spec->bdim, spec->boundary_integration);
    const constraint_element_side_t side = {
        .ndim = spec->ndim, .basis_specs = spec->element_spec->basis, .orientation = spec->orientation};

    // The sizing pass needs one iterator plus per-axis and mapped-axis
    // scratch; all bounded by `bdim` and `order`.
    combination_iterator_t *iter;
    unsigned *counts;
    unsigned *offsets;
    uint8_t *mapped_axes;
    void *const mem = cutl_alloc_group(
        &CUTL_STD_ALLOCATOR, (const cutl_alloc_info_t[]){
                                 {combination_iterator_required_memory((uint8_t)spec->order), (void **)&iter},
                                 {sizeof(*counts) * spec->bdim, (void **)&counts},
                                 {sizeof(*offsets) * spec->bdim, (void **)&offsets},
                                 {(spec->order == 0 ? 1u : spec->order) * sizeof(*mapped_axes), (void **)&mapped_axes},
                                 {}});
    if (mem == NULL)
    {
        // A failed sizing pass reports zero buffers; the caller cannot
        // assemble without them anyway.
        *out_sizes = (constraint_boundary_mass_work_sizes_t){};
        return;
    }

    if (spec->order > spec->bdim)
    {
        cutl_dealloc(&CUTL_STD_ALLOCATOR, mem);
        *out_sizes = (constraint_boundary_mass_work_sizes_t){.point_factors = point_count};
        return;
    }
    combination_iterator_init(iter, (uint8_t)spec->bdim, (uint8_t)spec->order);
    size_t max_row_dofs = 0;
    size_t max_col_dofs = 0;
    int sign;
    for (; !combination_iterator_is_done(iter); combination_iterator_next(iter))
    {
        const uint8_t *const component_axes = combination_iterator_current(iter);
        boundary_mass_row_axis_counts(spec, component_axes, counts, offsets);
        const size_t row_dofs = boundary_mass_row_dofs(spec, counts);
        mapped_axes_and_sign(&side, spec->bdim, spec->order, component_axes, mapped_axes, &sign);
        const unsigned element_component = combination_get_index(spec->ndim, spec->order, mapped_axes);
        const size_t col_dofs = kform_spec_component_dof_count(spec->element_spec, element_component);
        max_row_dofs = max_row_dofs > row_dofs ? max_row_dofs : row_dofs;
        max_col_dofs = max_col_dofs > col_dofs ? max_col_dofs : col_dofs;
    }
    cutl_dealloc(&CUTL_STD_ALLOCATOR, mem);

    *out_sizes = (constraint_boundary_mass_work_sizes_t){
        .row_values = max_row_dofs * point_count,
        .col_values = max_col_dofs * point_count,
        .point_factors = point_count,
        .component_count = combination_total_count((uint8_t)spec->bdim, (uint8_t)spec->order)};
}

/**
 * @brief Physical dot product of one test and one element pullback component.
 */
static double boundary_mass_pullback_dot(const constraint_trace_pullback_t *const test_pullback,
                                         const unsigned test_component,
                                         const constraint_trace_pullback_t *const element_pullback,
                                         const unsigned element_component, const size_t point_count, const size_t point)
{
    CUTL_ASSERT(test_pullback->physical_component_count == element_pullback->physical_component_count,
                "Pullbacks disagree on the physical component count.");
    const double *const test_values =
        test_pullback->values +
        ((size_t)test_component * test_pullback->physical_component_count * point_count + point);
    const double *const element_values =
        element_pullback->values +
        ((size_t)element_component * element_pullback->physical_component_count * point_count + point);
    double result = 0.0;
    for (unsigned physical_component = 0; physical_component < test_pullback->physical_component_count;
         ++physical_component)
    {
        result += test_values[(size_t)physical_component * point_count] *
                  element_values[(size_t)physical_component * point_count];
    }
    return result;
}

void constraint_boundary_mass_assemble(const constraint_boundary_mass_request_t *const request)
{
    const constraint_boundary_mass_spec_t *const spec = request->spec;
    constraint_boundary_mass_work_t *const work = request->work;
    const unsigned order = spec->order;
    const size_t component_count = combination_total_count((uint8_t)spec->bdim, (uint8_t)order);
    CUTL_ASSERT((request->test_pullback == NULL) == (request->element_pullback == NULL),
                "Physical pullbacks must be given for both test and element sides.");

    const size_t point_count = integration_specs_total_points(spec->bdim, spec->boundary_integration);
    integration_spec_point_strides(spec->bdim, spec->boundary_integration, work->point_strides);

    size_t rows;
    size_t cols;
    boundary_mass_shape(spec, work, &rows, &cols);
    for (size_t value = 0; value < rows * cols; ++value)
    {
        request->out_matrix[value] = 0.0;
    }
    if (order > spec->bdim)
    {
        return;
    }

    boundary_mass_axis_descriptors(spec, request, work);

    const bool physical = request->test_pullback != NULL;
    const constraint_element_side_t side = {
        .ndim = spec->ndim, .basis_specs = spec->element_spec->basis, .orientation = spec->orientation};
    if (physical)
    {
        // Physical pairing walks every mapped component: the blocks iterator
        // must be a valid (bdim, order) enumeration before unranking.
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
            // The element axes of the block's mapped component: reference
            // pairing maps the row component itself, physical pairing walks
            // every mapped component through the same orientation.
            int block_orientation_sign;
            if (physical)
            {
                combination_iterator_set_to_index(work->blocks, block);
                mapped_axes_and_sign(&side, spec->bdim, order, combination_iterator_current(work->blocks),
                                     work->mapped_axes, &block_orientation_sign);
            }
            else
            {
                mapped_axes_and_sign(&side, spec->bdim, order, component_axes, work->mapped_axes,
                                     &block_orientation_sign);
            }
            kform_component_basis_values(spec->ndim, spec->element_spec->basis, order, work->mapped_axes, work->axes,
                                         work->point_strides, point_count, work->col_values);

            // The pullback tables hold each side's own covector image, so the
            // physical pairing is frame-free; only reference pairing needs the
            // orientation sign to express the row component in the element's
            // covector basis.
            const double block_sign = (double)(physical ? 1 : work->element_signs[block]) * request->factor;
            for (size_t point = 0; point < point_count; ++point)
            {
                double point_factor = block_sign * request->point_weights[point];
                if (request->surface_weights != NULL)
                {
                    point_factor *= request->surface_weights[point];
                }
                if (physical)
                {
                    point_factor *= boundary_mass_pullback_dot(request->test_pullback, (unsigned)component,
                                                               request->element_pullback,
                                                               work->element_components[block], point_count, point);
                }
                work->point_factors[point] = point_factor;
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
    // Assert preconditions shared with the common space merge.
    CUTL_ASSERT(bdim != 0, "0-D boundary common space is trivial, so do not use this.");
    CUTL_ASSERT(request->nforms > 0, "At least one k-form is required.");
    CUTL_ASSERT(nelem > 1, "At least two elements are required for boundary common space.");
    CUTL_ASSERT(ndim > 1, "Space must be at least 2D.");
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
    // NULL-fill every reference slot so a failed prepare still releases
    // cleanly.
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
        basis_spec_t *const form_lower = plan->boundary_lower_specs + (size_t)iform * bdim;
        for (unsigned idim = 0; idim < bdim; ++idim)
        {
            // Order-zero axes cannot lose another degree; no component reads
            // their lower table because the matching components have no rows.
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
            // Classify every element axis as fixed normal axis or free axis,
            // recording the canonical face slot of free axes.
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
            basis_spec_t *const item_lower = plan->element_lower_specs + item * ndim;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                item_lower[axis] =
                    (basis_spec_t){.type = element->basis[axis].type,
                                   .order = element->basis[axis].order > 0 ? element->basis[axis].order - 1u : 0u};
            }
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
            const constraint_boundary_mass_spec_t spec = {
                .ndim = ndim,
                .bdim = bdim,
                .order = order,
                .element_spec = &element_spec,
                .boundary_basis = form_basis,
                .boundary_integration = form_integration,
                .orientation = element->orientation,
                .axis_skip = request->axis_skip != NULL ? request->axis_skip + (size_t)iform * bdim : NULL};
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
        const uint8_t *const form_skip =
            request->axis_skip != NULL ? request->axis_skip + (size_t)iform * request->bdim : NULL;
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
                                                          .orientation = views[ie].orientation,
                                                          .axis_skip = form_skip};
            constraint_boundary_mass_work_sizes_t sizes;
            constraint_boundary_mass_work_size(&spec, &sizes);
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
        const size_t form = (size_t)iform * bdim;
        const basis_spec_t *const form_basis = plan->boundary_basis + form;
        const integration_spec_t *const form_integration = plan->boundary_integration + form;
        integration_rule_tensor_weights(bdim, plan->rules + form, work->weights);

        for (unsigned ie = 0; ie < plan->nelem; ++ie)
        {
            const boundary_element_space_t *const element = views + ie;
            const size_t item = (size_t)iform * plan->nelem + ie;
            // Classify every element axis as fixed normal axis or free axis,
            // recording the canonical face slot of free axes.
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
            const constraint_boundary_mass_spec_t spec = {
                .ndim = ndim,
                .bdim = bdim,
                .order = order,
                .element_spec = &element_spec,
                .boundary_basis = form_basis,
                .boundary_integration = form_integration,
                .orientation = element->orientation,
                .axis_skip = request->axis_skip != NULL ? request->axis_skip + (size_t)iform * bdim : NULL};
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

        // Debug guard: every incident side samples the same physical face,
        // so the integrated surface measure - and, for k-forms, every
        // pullback moment - must agree across sides regardless of each
        // side's local orientation. A mismatch flags a wrong canonical
        // point mapping (mirrored or permuted axes) before the values are
        // packed into rows.
        if (!request->c1_continuous && request->surface_weights != NULL)
        {
            const size_t face_points = integration_specs_total_points(bdim, form_integration);
            const double *const *const surfaces = request->surface_weights + (size_t)iform * plan->nelem;
            double reference_measure = 0.0;
            for (size_t point = 0; point < face_points; ++point)
            {
                reference_measure += work->weights[point] * surfaces[0][point];
            }
            for (unsigned ie = 1; ie < plan->nelem; ++ie)
            {
                double measure = 0.0;
                for (size_t point = 0; point < face_points; ++point)
                {
                    measure += work->weights[point] * surfaces[ie][point];
                }
                const double deviation = fabs(measure - reference_measure);
                if (deviation > 1e-9 * (1.0 + fabs(reference_measure)))
                {
                    // Debug diagnostics: dump both sides' sampled weights so
                    // the mis-mapped canonical point is visible.
                    fprintf(stderr, "surface measure mismatch on side %u: %g vs %g over %zu points\n", ie, measure,
                            reference_measure, face_points);
                    for (size_t point = 0; point < face_points; ++point)
                    {
                        fprintf(stderr, "  point %zu: anchor %g side %g\n", point, surfaces[0][point],
                                surfaces[ie][point]);
                    }
                }
                CUTL_ASSERT(deviation <= 1e-9 * (1.0 + fabs(reference_measure)),
                            "Incident sides disagree on the shared face's integrated surface measure.");
            }
            if (request->test_pullbacks != NULL)
            {
                const constraint_trace_pullback_t *const reference =
                    request->test_pullbacks[(size_t)iform * plan->nelem];
                const unsigned face_components =
                    (unsigned)combination_total_count((uint8_t)bdim, (uint8_t)views[0].order);
                const size_t component_blocks = (size_t)face_components * reference->physical_component_count;
                for (unsigned ie = 1; ie < plan->nelem; ++ie)
                {
                    const constraint_trace_pullback_t *const other =
                        request->test_pullbacks[(size_t)iform * plan->nelem + ie];
                    CUTL_ASSERT(other->physical_component_count == reference->physical_component_count &&
                                    other->point_count == reference->point_count,
                                "Incident sides disagree on the trace pullback layout.");
                    for (size_t block = 0; block < component_blocks; ++block)
                    {
                        double moment_reference = 0.0;
                        double moment = 0.0;
                        for (size_t point = 0; point < face_points; ++point)
                        {
                            const size_t index = block * reference->point_count + point;
                            moment_reference += work->weights[point] * surfaces[0][point] * reference->values[index];
                            moment += work->weights[point] * surfaces[ie][point] * other->values[index];
                        }
                        // Sides may carry opposite covector signs from mirrored
                        // axes, but the moment's magnitude is side independent.
                        CUTL_ASSERT(fabs(fabs(moment) - fabs(moment_reference)) <=
                                        1e-9 * (1.0 + fabs(moment_reference)),
                                    "Incident sides disagree on a shared face's pullback moment.");
                    }
                }
            }
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

static void physical_side_layout_impl(const kform_spec_t *const test_spec, const constraint_element_side_t *const side,
                                      size_t *const out_row_count, size_t *const out_entry_count)
{
    const size_t component_count = kform_spec_component_count(test_spec);
    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    size_t row_count = 0;
    size_t entry_count = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        const size_t test_dof_count = kform_spec_component_dof_count(test_spec, test_component);
        row_count += test_dof_count;
        uint8_t face_axes[UINT8_MAX];
        size_t entries_per_row = 0;
        for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
        {
            kform_component_axes(test_spec, face_component, face_axes);
            entries_per_row += side_entries_per_test_component(side, test_spec->ndim, test_spec->order, face_axes);
        }
        entry_count += test_dof_count * entries_per_row;
    }
    *out_row_count = row_count;
    *out_entry_count = entry_count;
}

void constraint_physical_side_layout(const kform_spec_t *const test_spec, const constraint_element_side_t *const side,
                                     size_t *const out_row_count, size_t *const out_entry_count)
{
    physical_side_layout_impl(test_spec, side, out_row_count, out_entry_count);
}

static void physical_assemble_impl(const kform_spec_t *const test_spec, const constraint_element_side_t *const sides,
                                   const constraint_assembly_inputs_t *const inputs, const unsigned side_count,
                                   uint8_t *const out_sides, uint32_t out_components[], size_t out_local_dofs[],
                                   double out_coefficients[], size_t out_row_offsets[])
{
    const unsigned face_dim = test_spec->ndim;
    const unsigned order = test_spec->order;
    const size_t component_count = kform_spec_component_count(test_spec);
    const unsigned face_component_count = combination_total_count((uint8_t)face_dim, (uint8_t)order);
    const kform_values_table_t *const test_table = inputs[0].test_table;
    const size_t point_count = test_table->point_count;

    size_t row = 0;
    size_t entry = 0;
    out_row_offsets[0] = 0;
    // The packed-row contract is unchanged. Every mapped face component can
    // couple through the physical pullback, so each row's block spans all
    // face components of every side; the pullback factors depend only on the
    // component pair and the point, never on the DoF indices.
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        uint8_t test_axes[UINT8_MAX];
        kform_component_axes(test_spec, test_component, test_axes);
        const size_t test_dof_count =
            test_table->component_offsets[test_component + 1] - test_table->component_offsets[test_component];
        const size_t test_block_start = test_table->component_offsets[test_component] * point_count;
        unsigned test_element_components[2] = {0, 0};
        int test_orientation_signs[2] = {1, 1};
        size_t side_entries[2] = {0, 0};
        size_t row_entries = 0;
        for (unsigned side_index = 0; side_index < side_count; ++side_index)
        {
            if (order != 0)
            {
                mapped_component(sides + side_index, face_dim, order, test_axes, &test_element_components[side_index],
                                 &test_orientation_signs[side_index]);
            }
            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                uint8_t face_axes[UINT8_MAX];
                kform_component_axes(test_spec, face_component, face_axes);
                side_entries[side_index] +=
                    side_entries_per_test_component(sides + side_index, face_dim, order, face_axes);
            }
            row_entries += side_entries[side_index];
        }
        for (unsigned side_index = 0; side_index < side_count; ++side_index)
        {
            const constraint_element_side_t *const side = sides + side_index;
            const constraint_assembly_inputs_t *const input = inputs + side_index;
            const kform_values_table_t *const element_table = input->element_table;
            const constraint_trace_pullback_t *const pullback = order != 0 ? input->pullback : NULL;
            const double side_sign = side_index == 0 ? 1.0 : -1.0;
            // Pullback rows hold each side's own covector image; the dot is
            // frame-free and carries no covector orientation sign.
            const double side_factor = side_sign;
            size_t column = side_index == 0 ? 0 : side_entries[0];
            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                uint8_t face_axes[UINT8_MAX];
                kform_component_axes(test_spec, face_component, face_axes);
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, face_dim, order, face_axes, &element_component, &orientation_sign);
                const kform_spec_t element_spec = {.ndim = side->ndim, .order = order, .basis = side->basis_specs};
                const size_t element_dof_count = kform_spec_component_dof_count(&element_spec, element_component);
                const size_t element_block_start = element_table->component_offsets[element_component] * point_count;
                trace_component_block(
                    point_count, test_dof_count, element_dof_count, test_table->values + test_block_start,
                    element_table->values + element_block_start, input->point_weights, input->surface_weights,
                    side_factor, pullback, test_element_components[side_index], element_component, column, row_entries,
                    out_coefficients + entry);
                column += element_dof_count;
            }
        }
        for (unsigned side_index = 0; side_index < side_count; ++side_index)
        {
            const constraint_element_side_t *const side = sides + side_index;
            size_t column = side_index == 0 ? 0 : side_entries[0];
            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                uint8_t face_axes[UINT8_MAX];
                kform_component_axes(test_spec, face_component, face_axes);
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, face_dim, order, face_axes, &element_component, &orientation_sign);
                (void)orientation_sign;
                const kform_spec_t element_spec = {.ndim = side->ndim, .order = order, .basis = side->basis_specs};
                const size_t element_dof_count = kform_spec_component_dof_count(&element_spec, element_component);
                for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof)
                {
                    size_t entry_index = entry + test_dof * row_entries + column;
                    for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof, ++entry_index)
                    {
                        if (out_sides)
                            out_sides[entry_index] = (uint8_t)side_index;
                        out_components[entry_index] = (uint32_t)element_component;
                        out_local_dofs[entry_index] = element_dof;
                    }
                }
                column += element_dof_count;
            }
        }
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof)
        {
            out_row_offsets[row + test_dof + 1] = entry + (test_dof + 1) * row_entries;
        }
        row += test_dof_count;
        entry += test_dof_count * row_entries;
    }
}

void constraint_physical_side_assemble(const kform_spec_t *const test_spec, const constraint_element_side_t *const side,
                                       const constraint_assembly_inputs_t *const inputs, uint32_t out_components[],
                                       size_t out_local_dofs[], double out_coefficients[], size_t out_row_offsets[])
{
    physical_assemble_impl(test_spec, side, inputs, 1, NULL, out_components, out_local_dofs, out_coefficients,
                           out_row_offsets);
}

void constraint_physical_side_load(const kform_spec_t *const test_spec, const constraint_element_side_t *const side,
                                   const double *const point_weights, const double *const datum_values,
                                   const double *const surface_weights, const kform_values_table_t *const element_table,
                                   double values[])
{
    ASSERT(side->ndim == test_spec->ndim + 1, "The load is defined on codimension-one faces.");
    const unsigned face_dim = test_spec->ndim;
    const unsigned order = test_spec->order;
    const unsigned face_component_count = combination_total_count((uint8_t)face_dim, (uint8_t)order);
    const kform_spec_t element_spec = {.ndim = side->ndim, .order = order, .basis = side->basis_specs};
    const size_t point_count = element_table->point_count;
    // The datum is an element-frame k-form with k = test_spec->order + 1, given
    // as its C(n, k) physical components sampled at the canonical face points:
    // datum_values[component * point_count + point]. For each face (k-1)-form
    // component J with element-frame axes J_e, the paired datum component is
    // I = J_e U {fixed_axis} and the sign carries the count of J_e axes below
    // the fixed normal axis; at k = n this reduces to the previous
    // sigma_out = side * (-1)^a formula.
    const int8_t fixed_mapping = side->orientation[0];
    const unsigned fixed_axis = (unsigned)(fixed_mapping < 0 ? -fixed_mapping : fixed_mapping) - 1;
    const double side_sign = fixed_mapping < 0 ? -1.0 : 1.0;
    for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
    {
        uint8_t face_axes[UINT8_MAX];
        kform_component_axes(test_spec, face_component, face_axes);
        unsigned element_component;
        int orientation_sign;
        mapped_component(side, face_dim, order, face_axes, &element_component, &orientation_sign);
        const size_t element_dof_count = kform_spec_component_dof_count(&element_spec, element_component);
        const size_t element_start = element_table->component_offsets[element_component];
        const size_t element_block_dofs = element_table->component_offsets[element_component + 1] -
                                          element_table->component_offsets[element_component];
        const size_t element_block_start = element_start * point_count;
        uint8_t element_axes[UINT8_MAX];
        kform_component_axes(&element_spec, element_component, element_axes);
        unsigned exponent_below_fixed = 0;
        for (unsigned i = 0; i < order; ++i)
            exponent_below_fixed += element_axes[i] < fixed_axis ? 1u : 0u;
        uint8_t datum_axes[UINT8_MAX + 1];
        for (unsigned i = 0; i < order; ++i)
            datum_axes[i] = element_axes[i];
        datum_axes[order] = (uint8_t)fixed_axis;
        for (unsigned i = order; i > 0 && datum_axes[i - 1] > datum_axes[i]; --i)
        {
            const uint8_t tmp = datum_axes[i - 1];
            datum_axes[i - 1] = datum_axes[i];
            datum_axes[i] = tmp;
        }
        const unsigned datum_component = combination_get_index(side->ndim, order + 1, datum_axes);
        const double sign = side_sign * (double)orientation_sign * (exponent_below_fixed % 2 == 0 ? 1.0 : -1.0);
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

// Combination index of the sorted copy of @p axes: the row of a component
// within the face map's own ascending-axis transform table.
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

void constraint_trace_pullback_build(const constraint_trace_pullback_build_t *const request)
{
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
    const kform_spec_t face_spec = {.ndim = request->face_dim, .order = request->order, .basis = NULL};
    // The canonical to source point map is component independent: derive the
    // loop-invariant per-axis decode data once, then decode every point with
    // one divide-modulo pair per axis.
    const unsigned fixed_count = request->element_dim - request->face_dim;
    // TODO: all of these should be passed in as work arrays.
    unsigned axis_source_slots[UINT8_MAX];
    unsigned axis_orders[UINT8_MAX];
    size_t axis_source_strides[UINT8_MAX];
    size_t axis_canonical_strides[UINT8_MAX];
    int axis_mirrored[UINT8_MAX];
    // Rank of every element axis among the free axes (the face map's own
    // axis order) and whether the axis is free at all.
    bool element_axis_free[UINT8_MAX];
    unsigned element_source_rank[UINT8_MAX];
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
    // Face component indexing enumerates the canonical boundary form's
    // components and writes each mapped element component's block from the
    // face component's transform rows. Element component indexing instead
    // enumerates the element k-form's own components, reading each one's
    // transform rows from the inverse-mapped face component.
    unsigned element_to_face[UINT8_MAX + 1];
    if (request->element_components)
    {
        const unsigned element_total =
            (unsigned)combination_total_count((uint8_t)request->element_dim, (uint8_t)request->order);
        for (unsigned index = 0; index <= UINT8_MAX && index < element_total; ++index)
            element_to_face[index] = UINT8_MAX + 1;
        for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
        {
            uint8_t face_axes[UINT8_MAX];
            kform_component_axes(&face_spec, face_component, face_axes);
            unsigned element_component;
            int orientation_sign;
            mapped_component(&side, request->face_dim, request->order, face_axes, &element_component,
                             &orientation_sign);
            CUTL_ASSERT(element_component < element_total, "Mapped element component out of range.");
            element_to_face[element_component] = face_component;
        }
    }
    const unsigned written_components =
        request->element_components
            ? (unsigned)combination_total_count((uint8_t)request->element_dim, (uint8_t)request->order)
            : face_component_count;
    // TODO: swap over the iteration to use combination iterator instead
    for (unsigned component = 0; component < written_components; ++component)
    {
        const unsigned face_component = request->element_components ? element_to_face[component] : component;
        if (request->element_components && !request->canonical_components && face_component > face_component_count)
        {
            // A component whose covectors lie entirely on the fixed normal
            // axes has no tangential face counterpart; the engine's block
            // enumeration never reads its block. Zero it for hygiene.
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
        uint8_t face_axes[UINT8_MAX];
        unsigned element_component;
        if (request->element_components)
        {
            kform_component_axes(&(kform_spec_t){.ndim = request->element_dim, .order = request->order, .basis = NULL},
                                 component, face_axes);
            element_component = component;
        }
        else
        {
            kform_component_axes(&face_spec, component, face_axes);
            if (request->canonical_components)
            {
                // The canonical boundary form's components index their own
                // transform rows directly; no element-side mapping applies.
                element_component = component;
            }
            else
            {
                int orientation_sign;
                mapped_component(&side, request->face_dim, request->order, face_axes, &element_component,
                                 &orientation_sign);
            }
        }
        // The transform rows follow the face map's own free-axis order, so a
        // component whose axis order differs from that order must read the
        // row of its axes' ranks: canonical axes map through the orientation,
        // element axes are already the map's own axes. Canonical rows carry
        // the covector orientation: every mirrored axis flips its covector
        // and the sort of the mapped axes flips once per transposition, so
        // the table holds the canonical covector's physical image. Element
        // rows stay the element's own image; consumers pair the two physical
        // images without further orientation signs.
        uint8_t source_axes[UINT8_MAX];
        unsigned source_row = face_component;
        int value_sign = 1;
        if (request->canonical_components)
        {
            kform_component_axes(&face_spec, face_component, source_axes);
            for (unsigned i = 0; i < request->order; ++i)
            {
                source_axes[i] = (uint8_t)axis_source_slots[source_axes[i]];
                if (axis_mirrored[face_axes[i]])
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
        else if (request->element_components)
        {
            kform_component_axes(&(kform_spec_t){.ndim = request->element_dim, .order = request->order, .basis = NULL},
                                 component, source_axes);
            bool tangential = true;
            for (unsigned i = 0; i < request->order; ++i)
            {
                const unsigned element_axis = source_axes[i];
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

void constraint_boundary_test_specs(const unsigned ndim, const unsigned boundary_dim, const unsigned order,
                                    const size_t element_count, const basis_spec_t *const *element_bases,
                                    const int8_t *orientations, const basis_set_type_t type_override,
                                    basis_spec_t out_specs[], bool out_present[])
{
    ASSERT(element_count > 0, "At least one incident element is required.");
    ASSERT(ndim >= 1 && ndim <= UINT8_MAX, "Element dimension out of range.");
    ASSERT(boundary_dim < ndim, "The boundary dimension must be below the element dimension.");
    ASSERT(order <= boundary_dim, "Form order exceeds the boundary dimension.");
    ASSERT(type_override == BASIS_INVALID || basis_set_type_is_valid(type_override), "Invalid basis family override.");

    const unsigned fixed_count = ndim - boundary_dim;
    // Records must be signed permutations whose fixed normal-axis prefix
    // increases in absolute value; checked only in debug builds.
    for (size_t element = 0; element < element_count; ++element)
    {
        ASSERT(element_bases[element] != NULL, "Missing element basis specifications.");
        bool used_axes[UINT8_MAX];
        for (unsigned idim = 0; idim < ndim; ++idim)
        {
            used_axes[idim] = false;
            ASSERT(basis_set_type_is_valid(element_bases[element][idim].type), "Invalid basis family.");
        }
        for (unsigned idim = 0; idim < ndim; ++idim)
        {
            const int mapped_axis = orientations[element * ndim + idim];
            const unsigned axis = (unsigned)(mapped_axis < 0 ? -mapped_axis : mapped_axis);
            ASSERT(axis != 0 && axis <= ndim && !used_axes[axis - 1], "Orientation is not a signed permutation.");
            used_axes[axis - 1] = true;
        }
        for (unsigned idim = 1; idim < fixed_count; ++idim)
        {
            const int previous = orientations[element * ndim + idim - 1];
            const int current = orientations[element * ndim + idim];
            ASSERT((current < 0 ? -current : current) > (previous < 0 ? -previous : previous),
                   "The fixed-axis prefix must increase in absolute value.");
        }
    }

    const uint8_t face_ndim = (uint8_t)boundary_dim;
    const uint8_t form_order = (uint8_t)order;
    const size_t component_count = combination_total_count(face_ndim, form_order);

    // Per canonical axis, the minimum order and the family of the element
    // achieving it; the strict comparison keeps the lowest element on ties.
    unsigned min_order[UINT8_MAX];
    basis_set_type_t min_type[UINT8_MAX];
    for (unsigned axis = 0; axis < boundary_dim; ++axis)
    {
        min_order[axis] = UINT_MAX;
        min_type[axis] = BASIS_INVALID;
    }
    for (size_t element = 0; element < element_count; ++element)
    {
        for (unsigned axis = 0; axis < boundary_dim; ++axis)
        {
            const int mapped_axis = orientations[element * ndim + fixed_count + axis];
            const unsigned element_axis = (unsigned)(mapped_axis < 0 ? -mapped_axis : mapped_axis) - 1;
            ASSERT(element_axis < ndim, "Mapped element axis out of bounds.");
            const basis_spec_t *const spec = &element_bases[element][element_axis];
            if (spec->order < min_order[axis])
            {
                min_order[axis] = spec->order;
                min_type[axis] = spec->type;
            }
        }
    }

    // Reduce inactive axes by two and drop components that would go negative.
    for (size_t component = 0; component < component_count; ++component)
    {
        uint8_t axes[UINT8_MAX];
        kform_component_axes(&(kform_spec_t){.ndim = face_ndim, .order = form_order, .basis = NULL},
                             (unsigned)component, axes);
        bool present = true;
        for (unsigned axis = 0; axis < boundary_dim; ++axis)
        {
            const int reduced = (int)min_order[axis] - (component_has_axis(form_order, axes, axis) ? 0 : 2);
            present = present && reduced >= 0;
            out_specs[component * boundary_dim + axis] =
                (basis_spec_t){.type = type_override != BASIS_INVALID ? type_override : min_type[axis],
                               .order = (unsigned)(reduced < 0 ? 0 : reduced)};
        }
        out_present[component] = present;
    }
}

void boundary_space_map_resample_work_size(const unsigned bdim, const unsigned coords,
                                           const integration_rule_t *const *source_rules,
                                           const integration_rule_t *const *target_rules,
                                           size_t *const out_axis_matrices, size_t *const out_positions,
                                           size_t *const out_jacobian, size_t *const out_q)
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
}

void boundary_space_map_resample(const boundary_space_map_resample_request_t *const request)
{
    const unsigned bdim = request->bdim;
    const unsigned coords = request->coords;
    unsigned target_orders[UINT8_MAX];
    unsigned source_orders[UINT8_MAX];
    const double *axis_matrices[UINT8_MAX];
    integration_spec_t target_specs[UINT8_MAX];

    size_t offset = 0;
    for (unsigned axis = 0; axis < bdim; ++axis)
    {
        const unsigned n_out = request->target_rules[axis]->spec.order + 1u;
        const unsigned n_in = request->source_rules[axis]->spec.order + 1u;
        // Interpolation matrix from the source nodes to the target nodes:
        // entry (in, out) holds the target-node value of the source node's
        // Lagrange polynomial.
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
