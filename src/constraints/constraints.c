/**
 * @file constraints.c
 * @brief Implementation of reference and physical trace constraints.
 *
 * Assembly uses one canonical face coordinate system. Element-side
 * orientations map that system to signed, one-based element axes; the helper
 * functions below keep the mapping and its alternating k-form sign in one
 * place. The public assembly routines first count rows and entries so caller
 * buffers can be checked before the nested quadrature loops begin.
 */

/**
 * The status macros keep the symbolic and human-readable lookup tables in
 * lockstep without maintaining two independent switch structures.
 */
#include "constraints.h"
#include "cutl/iterators/combination_iterator.h"
#include <limits.h>
#include <stdbool.h>

#define CONSTRAINT_STATUS_CASE(stat)                                                                                   \
    case stat:                                                                                                         \
        return #stat
/**
 * @brief Translate a status code to its symbolic name.
 * @param status Status value to translate.
 * @return A static symbolic name, or `"Unknown"` for an out-of-range value.
 */
const char *constraint_status_to_str(const constraint_status_t status)
{
    switch (status)
    {
        CONSTRAINT_STATUS_CASE(CONSTRAINT_SUCCESS);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INVALID_ARGUMENT);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INVALID_DIMENSION);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INVALID_ORDER);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_SIZE_OVERFLOW);
        CONSTRAINT_STATUS_CASE(CONSTRAINT_INSUFFICIENT_STORAGE);
    }
    return "Unknown";
}
#undef CONSTRAINT_STATUS_CASE

#define CONSTRAINT_STATUS_MSG(stat, msg)                                                                               \
    case stat:                                                                                                         \
        return msg
/**
 * @brief Translate a status code to its short diagnostic message.
 * @param status Status value to translate.
 * @return A static message, or `"Unknown"` for an out-of-range value.
 */
const char *constraint_status_msg(const constraint_status_t status)
{
    switch (status)
    {
        CONSTRAINT_STATUS_MSG(CONSTRAINT_SUCCESS, "Success");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INVALID_ARGUMENT, "Invalid argument");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INVALID_DIMENSION, "Invalid dimension");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INVALID_ORDER, "Invalid form order");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_SIZE_OVERFLOW, "Size calculation overflow");
        CONSTRAINT_STATUS_MSG(CONSTRAINT_INSUFFICIENT_STORAGE, "Insufficient output storage");
    }
    return "Unknown";
}
#undef CONSTRAINT_STATUS_MSG

/**
 * @brief Validate the common k-form specification invariants.
 *
 * The combination iterator stores dimensions and form degree in `uint8_t`, so
 * dimensions and basis orders are bounded before any narrowing conversion.
 * Zero-order axes are permitted for test specifications: their one-dimensional
 * factors evaluate to constants, and components whose active covector axes
 * have zero order simply contribute no DoFs. Element-side specifications keep
 * the stricter check in `validate_element_side`.
 *
 * @param spec Specification to inspect; may be null.
 * @return The first applicable validation status.
 */
static constraint_status_t validate_kform_spec(const constraint_kform_spec_t *const spec)
{
    // Validate before narrowing to uint8_t for the combination iterator; this
    // also guarantees later basis-order reductions cannot underflow.
    if (!spec)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (spec->ndim > UINT8_MAX)
        return CONSTRAINT_INVALID_DIMENSION;
    if (spec->order > spec->ndim)
        return CONSTRAINT_INVALID_ORDER;
    if (spec->ndim != 0 && !spec->basis_specs)
        return CONSTRAINT_INVALID_ARGUMENT;

    // Validate every basis family and order. Orders share the uint8_t bound
    // that keeps every stack scratch array in this file at a compile-time
    // size; zero-order axes stay legal for test spaces because inactive axes
    // contribute constant factors and active zero-order axes yield components
    // without DoFs.
    for (unsigned idim = 0; idim < spec->ndim; ++idim)
    {
        if (spec->basis_specs[idim].order > UINT8_MAX || !basis_set_type_is_valid(spec->basis_specs[idim].type))
            return CONSTRAINT_INVALID_ORDER;
    }
    return CONSTRAINT_SUCCESS;
}
/**
 * @brief Decode a component index into its sorted active-axis combination.
 *
 * `axes` receives the strictly increasing axis numbers used by the component's
 * wedge product. Order-zero components have no axes; the one-element array is
 * still present because standard C does not permit a zero-length array.
 *
 * @param ndim Number of element or face dimensions.
 * @param order Form degree.
 * @param component Lexicographic combination index.
 * @param axes Output array with `order` logical entries, or one placeholder
 *        entry when `order == 0`.
 */
static void component_axes(const unsigned ndim, const unsigned order, const unsigned component,
                           uint8_t axes[const static order == 0 ? 1 : order])
{
    combination_set_to_index((uint8_t)ndim, (uint8_t)order, axes, component);
}

/**
 * @brief Test whether a component contains one active covector axis.
 *
 * The active axes are sorted, but a linear scan keeps this helper independent
 * of the combination representation and is negligible beside quadrature work.
 *
 * @param order Number of active axes.
 * @param axes Sorted active-axis combination.
 * @param axis Zero-based axis to search for.
 * @return `true` when `axis` is active, otherwise `false`.
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
 * @brief Validate one element-side mapping against a face test space.
 *
 * Besides checking pointers and basis types, this verifies that `orientation`
 * is a signed one-based permutation. The fixed normal axes occupy the prefix;
 * requiring their absolute values to increase makes the prefix canonical and
 * prevents equivalent side descriptions from producing different signs.
 *
 * @param test_spec Validated face test-space specification.
 * @param side Element-side specification to inspect.
 * @return A validation status.
 */
static constraint_status_t validate_element_side(const constraint_kform_spec_t *const test_spec,
                                                 const constraint_element_side_t *const side)
{
    if (!side || side->ndim <= test_spec->ndim || !side->basis_specs || !side->orientation)
        return CONSTRAINT_INVALID_ARGUMENT;
    // The element dimension bound keeps every stack array in this file at a
    // compile-time size; records are int8_t, so larger dimensions are unusable
    // anyway.
    if (side->ndim > UINT8_MAX)
        return CONSTRAINT_INVALID_DIMENSION;

    bool used_axes[UINT8_MAX];
    // First validate basis metadata and initialize the occupancy map used to
    // detect duplicate absolute axis numbers.
    for (unsigned i = 0; i < side->ndim; ++i)
    {
        used_axes[i] = false;
        if ((test_spec->order != 0 && side->basis_specs[i].order == 0) || side->basis_specs[i].order > UINT8_MAX ||
            side->basis_specs[i].type <= BASIS_INVALID || side->basis_specs[i].type > BASIS_BERNSTEIN)
            return CONSTRAINT_INVALID_ARGUMENT;
    }

    // A signed permutation is checked by absolute value; its signs are
    // meaningful only after the permutation itself is known to be valid.
    for (unsigned i = 0; i < side->ndim; ++i)
    {
        const int8_t mapped_axis = side->orientation[i];
        const unsigned axis = (unsigned)(mapped_axis < 0 ? -mapped_axis : mapped_axis);
        if (axis == 0 || axis > side->ndim || used_axes[axis - 1])
            return CONSTRAINT_INVALID_ARGUMENT;
        used_axes[axis - 1] = true;
    }

    // Fixed axes are canonicalized in increasing absolute order so their
    // endpoint prefix has a deterministic orientation convention.
    const unsigned fixed_count = side->ndim - test_spec->ndim;
    for (unsigned i = 1; i < fixed_count; ++i)
    {
        const unsigned previous =
            (unsigned)(side->orientation[i - 1] < 0 ? -side->orientation[i - 1] : side->orientation[i - 1]);
        const unsigned current = (unsigned)(side->orientation[i] < 0 ? -side->orientation[i] : side->orientation[i]);
        if (current <= previous)
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Map a face component's axes into an element component.
 *
 * The side orientation contributes one sign for every reversed mapped axis.
 * Sorting the mapped axes into canonical element order contributes the
 * permutation parity. Together these signs are the pullback sign of the
 * covector component, while `out_component` is its combination index.
 *
 * @param side Validated element-side mapping.
 * @param boundary_dim Number of canonical face dimensions.
 * @param order Form degree being mapped.
 * @param test_axes Sorted canonical face axes of the component.
 * @param out_component Receives the mapped element component index.
 * @param out_sign Receives `+1` or `-1`.
 * @return `CONSTRAINT_SUCCESS`; callers must satisfy the documented
 *         validation preconditions before calling this internal helper.
 */
static constraint_status_t mapped_component(const constraint_element_side_t *const side, const unsigned boundary_dim,
                                            const unsigned order,
                                            const uint8_t test_axes[const static order == 0 ? 1 : order],
                                            unsigned *const out_component, int *const out_sign)
{
    // Fixed-size scratch bounded by the validated form order; order zero
    // still gets one harmless placeholder.
    uint8_t mapped_axes[UINT8_MAX];
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
    // Bubble sort the mapped axes to the canonical order and adjust the sign accordingly
    // Sorting produces the canonical element-axis combination; each swap
    // flips the alternating covector sign by one permutation transposition.
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

    // Get the component index based on the element's mapped axes
    *out_component = combination_get_index(side->ndim, order, mapped_axes);
    *out_sign = sign;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Unrank one axis of a lexicographically ordered combination.
 *
 * The combination index is partitioned by the first possible axis. The size
 * of each partition is a binomial coefficient, so subtracting partitions
 * locates the requested axis without allocating or materializing earlier
 * combinations. `position` is zero-based. Invalid inputs are outside this
 * helper's contract and produce `UINT_MAX` only as a defensive fallback.
 *
 * @param ndim Dimension of the combination universe.
 * @param order Number of selected axes.
 * @param index Lexicographic combination index.
 * @param position Axis position to retrieve.
 * @return The zero-based axis, or `UINT_MAX` for an invalid request.
 */
static unsigned combination_axis_at(const unsigned ndim, const unsigned order, const unsigned index,
                                    const unsigned position)
{
    unsigned remaining = index;
    // At each position, skip complete combination blocks until the residual
    // index falls inside the block for the selected axis.
    unsigned minimum = 0;
    for (unsigned current = 0; current <= position; ++current)
    {
        const unsigned maximum = ndim - (order - current);
        unsigned selected = maximum;
        for (unsigned candidate = minimum; candidate <= maximum; ++candidate)
        {
            const unsigned count =
                combination_total_count((uint8_t)(ndim - candidate - 1), (uint8_t)(order - current - 1));
            if (remaining < count)
            {
                selected = candidate;
                break;
            }
            remaining -= count;
        }
        if (current == position)
            return selected;
        minimum = selected + 1;
    }
    return UINT_MAX;
}

/**
 * @brief Map a component index through a validated side orientation.
 *
 * This is the index-based counterpart of mapped_component. It computes the
 * same pullback sign from reversed axes and permutation parity, then ranks the
 * selected element axes directly in lexicographic order.
 *
 * @param side Validated element-side mapping.
 * @param boundary_dim Number of canonical face dimensions.
 * @param order Form degree.
 * @param component Canonical face component index.
 * @param out_component Receives the mapped element component index.
 * @param out_sign Receives `+1` or `-1`.
 * @return `CONSTRAINT_SUCCESS`, or `CONSTRAINT_INVALID_ARGUMENT` for an
 *         invalid pointer, dimension, order, or component index.
 */
static constraint_status_t mapped_component_for_index(const constraint_element_side_t *const side,
                                                      const unsigned boundary_dim, const unsigned order,
                                                      const unsigned component, unsigned *const out_component,
                                                      int *const out_sign)
{
    if (!side || !out_component || !out_sign || boundary_dim > side->ndim || order > boundary_dim)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (component >= combination_total_count((uint8_t)boundary_dim, (uint8_t)order))
        return CONSTRAINT_INVALID_ARGUMENT;

    const unsigned fixed_count = side->ndim - boundary_dim;
    int sign = 1;
    // Reconstruct the alternating sign from reversed face axes and the
    // permutation needed to put their mapped element axes in sorted order.
    for (unsigned first = 0; first < order; ++first)
    {
        const unsigned first_axis = combination_axis_at(boundary_dim, order, component, first);
        const int8_t first_mapping = side->orientation[fixed_count + first_axis];
        if (first_mapping < 0)
            sign = -sign;
        for (unsigned second = first + 1; second < order; ++second)
        {
            const unsigned second_axis = combination_axis_at(boundary_dim, order, component, second);
            const unsigned first_element_axis = (unsigned)(first_mapping < 0 ? -first_mapping : first_mapping) - 1;
            const int8_t second_mapping = side->orientation[fixed_count + second_axis];
            const unsigned second_element_axis = (unsigned)(second_mapping < 0 ? -second_mapping : second_mapping) - 1;
            if (first_element_axis > second_element_axis)
                sign = -sign;
        }
    }

    unsigned selected_count = 0;
    // Rank the selected element axes directly. This avoids a temporary axis
    // array and uses the same lexicographic ordering as combination_get_index.
    unsigned minimum = 0;
    unsigned mapped_index = 0;
    for (unsigned element_axis = 0; element_axis < side->ndim; ++element_axis)
    {
        bool selected = false;
        for (unsigned position = 0; position < order; ++position)
        {
            const unsigned face_axis = combination_axis_at(boundary_dim, order, component, position);
            const int8_t mapping = side->orientation[fixed_count + face_axis];
            if ((unsigned)(mapping < 0 ? -mapping : mapping) - 1 == element_axis)
            {
                selected = true;
                break;
            }
        }
        if (selected)
        {
            for (unsigned candidate = minimum; candidate < element_axis; ++candidate)
                mapped_index += combination_total_count((uint8_t)(side->ndim - candidate - 1),
                                                        (uint8_t)(order - selected_count - 1));
            minimum = element_axis + 1;
            selected_count += 1;
        }
    }
    if (selected_count != order)
        return CONSTRAINT_INVALID_ARGUMENT;
    *out_component = mapped_index;
    *out_sign = sign;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Validate a precomputed trace-basis table.
 *
 * The offsets are checked against the DoF count of every component. This
 * prevents the assembly loop from having to rediscover component boundaries
 * and ensures each table uses the same component ordering as the spec.
 *
 * @param spec Specification represented by the table.
 * @param values Table metadata and values to inspect.
 * @param point_count Required number of points.
 * @return A validation status.
 */
static constraint_status_t validate_trace_basis_values(const constraint_kform_spec_t *const spec,
                                                       const constraint_trace_basis_values_t *const values,
                                                       const size_t point_count)
{
    if (!values || values->point_count != point_count || !values->component_offsets || !values->values)
        return CONSTRAINT_INVALID_ARGUMENT;
    size_t component_count;
    constraint_status_t status = constraint_kform_component_count(spec, &component_count);
    if (status != CONSTRAINT_SUCCESS || values->component_count != component_count)
        return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
    if (values->component_offsets[0] != 0)
        return CONSTRAINT_INVALID_ARGUMENT;
    for (size_t component = 0; component < component_count; ++component)
    {
        size_t dof_count;
        status = constraint_kform_component_dof_count(spec, (unsigned)component, &dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        if (values->component_offsets[component + 1] < values->component_offsets[component] ||
            values->component_offsets[component + 1] - values->component_offsets[component] != dof_count)
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Validate the shared inputs of a two-sided trace operation.
 *
 * The count and assembly paths use this one gate so that they agree on side
 * orientation, basis, and dimension invariants before calculating storage.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @return A validation status.
 */
static constraint_status_t constraint_reference_validate(const constraint_kform_spec_t *const test_spec,
                                                         const constraint_element_side_t sides[const static 2])
{
    const constraint_status_t test_status = validate_kform_spec(test_spec);
    if (test_status != CONSTRAINT_SUCCESS)
        return test_status;
    if (test_spec->ndim == UINT_MAX)
        return CONSTRAINT_INVALID_DIMENSION;
    if (test_spec->order > test_spec->ndim)
        return CONSTRAINT_INVALID_ORDER;

    for (unsigned side = 0; side < 2; ++side)
    {
        const constraint_status_t side_status = validate_element_side(test_spec, sides + side);
        if (side_status != CONSTRAINT_SUCCESS)
            return side_status;
    }
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Count rows and entries for a reference trace matrix.
 *
 * Reference pairing couples one mapped element component per side to each test
 * component. This helper deliberately performs the same component/DoF
 * traversal used by assembly, but only accumulates sizes and checks overflow.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two validated element-side specifications.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return A validation, overflow, or success status.
 */
static constraint_status_t constraint_reference_counts(const constraint_kform_spec_t *const test_spec,
                                                       const constraint_element_side_t sides[const static 2],
                                                       size_t *const out_row_count, size_t *const out_entry_count)
{
    const constraint_status_t status = constraint_reference_validate(test_spec, sides);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t component_count;
    constraint_status_t result = constraint_kform_component_count(test_spec, &component_count);
    if (result != CONSTRAINT_SUCCESS)
        return result;

    size_t row_count = 0;
    size_t entry_count = 0;
    // Component order is the row-order contract shared by sizing and
    // assembly; each component contributes all of its local test DoFs.
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        size_t test_dof_count;
        result = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (result != CONSTRAINT_SUCCESS)
            return result;
        if (row_count > SIZE_MAX - test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        row_count += test_dof_count;

        uint8_t test_axes[UINT8_MAX];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        size_t entries_per_row = 0;
        for (unsigned side = 0; side < 2; ++side)
        {
            unsigned element_component;
            int orientation_sign;
            mapped_component(sides + side, test_spec->ndim, test_spec->order, test_axes, &element_component,
                             &orientation_sign);
            size_t element_dof_count;
            result = constraint_kform_component_dof_count(
                &(constraint_kform_spec_t){
                    .ndim = sides[side].ndim, .order = test_spec->order, .basis_specs = sides[side].basis_specs},
                element_component, &element_dof_count);
            if (result != CONSTRAINT_SUCCESS)
                return result;
            if (entries_per_row > SIZE_MAX - element_dof_count)
                return CONSTRAINT_SIZE_OVERFLOW;
            entries_per_row += element_dof_count;
        }
        if (test_dof_count > 0 && entries_per_row > SIZE_MAX / test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        const size_t component_entries = test_dof_count * entries_per_row;
        if (entry_count > SIZE_MAX - component_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        entry_count += component_entries;
    }

    *out_row_count = row_count;
    *out_entry_count = entry_count;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Public storage-query wrapper for reference assembly.
 *
 * This checks the output pointers before delegating to the shared reference
 * counter, allowing callers to size both packed arrays without modifying them.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return The status returned by constraint_reference_counts.
 */
constraint_status_t constraint_reference_required(const constraint_kform_spec_t *const test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    return constraint_reference_counts(test_spec, sides, out_row_count, out_entry_count);
}

/**
 * @brief Count storage for the two-sided physical trace matrix.
 *
 * Unlike reference pairing, physical pairing visits every face component on
 * each side because the pullback dot product may couple components. The
 * traversal mirrors physical assembly and checks every addition and product
 * before accumulating it.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return A validation, overflow, or success status.
 */
static constraint_status_t constraint_physical_counts(const constraint_kform_spec_t *const test_spec,
                                                      const constraint_element_side_t sides[const static 2],
                                                      size_t *const out_row_count, size_t *const out_entry_count)
{
    const constraint_status_t status = constraint_reference_validate(test_spec, sides);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t test_component_count;
    constraint_status_t result = constraint_kform_component_count(test_spec, &test_component_count);
    if (result != CONSTRAINT_SUCCESS)
        return result;

    size_t row_count = 0;
    size_t entry_count = 0;
    // Physical coupling visits every face component on both sides, unlike the
    // reference path which has one mapped component per side.
    for (unsigned test_component = 0; test_component < test_component_count; ++test_component)
    {
        size_t test_dof_count;
        result = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (result != CONSTRAINT_SUCCESS)
            return result;
        if (row_count > SIZE_MAX - test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        row_count += test_dof_count;

        size_t entries_per_row = 0;
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            const constraint_element_side_t *const side = sides + side_index;
            const unsigned face_component_count =
                combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
            // The mapped component determines the column block whose DoFs
            // can couple to this test row through the physical pullback.
            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                uint8_t face_axes[UINT8_MAX];
                component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component,
                                 &orientation_sign);
                const constraint_kform_spec_t element_spec = {
                    .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                size_t element_dof_count;
                result = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                if (result != CONSTRAINT_SUCCESS)
                    return result;
                if (entries_per_row > SIZE_MAX - element_dof_count)
                    return CONSTRAINT_SIZE_OVERFLOW;
                entries_per_row += element_dof_count;
            }
        }
        if (test_dof_count > 0 && entries_per_row > SIZE_MAX / test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        const size_t component_entries = test_dof_count * entries_per_row;
        if (entry_count > SIZE_MAX - component_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        entry_count += component_entries;
    }

    *out_row_count = row_count;
    *out_entry_count = entry_count;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Public storage-query wrapper for two-sided physical assembly.
 *
 * The wrapper validates output pointers before delegating to the shared count
 * implementation so callers can use it safely during buffer sizing.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return The status returned by constraint_physical_counts.
 */
constraint_status_t constraint_physical_required(const constraint_kform_spec_t *const test_spec,
                                                 const constraint_element_side_t sides[const static 2],
                                                 size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    return constraint_physical_counts(test_spec, sides, out_row_count, out_entry_count);
}

/**
 * @brief Count storage for one side of a physical trace matrix.
 *
 * The single-side count is kept separate from the two-sided count because it
 * is also used by precomputed-basis and boundary-load paths.
 *
 * @param test_spec Face test-space specification.
 * @param side Element-side specification.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return A validation, overflow, or success status.
 */
static constraint_status_t constraint_physical_side_counts(const constraint_kform_spec_t *const test_spec,
                                                           const constraint_element_side_t *const side,
                                                           size_t *const out_row_count, size_t *const out_entry_count)
{
    const constraint_status_t status =
        constraint_reference_validate(test_spec, (const constraint_element_side_t[2]){*side, *side});
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t component_count;
    constraint_status_t result = constraint_kform_component_count(test_spec, &component_count);
    if (result != CONSTRAINT_SUCCESS)
        return result;
    size_t row_count = 0;
    size_t entry_count = 0;
    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        size_t test_dof_count;
        result = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (result != CONSTRAINT_SUCCESS)
            return result;
        row_count += test_dof_count;
        size_t entries_per_row = 0;
        for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
        {
            uint8_t face_axes[UINT8_MAX];
            component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
            unsigned element_component;
            int orientation_sign;
            mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component, &orientation_sign);
            const constraint_kform_spec_t element_spec = {
                .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
            size_t element_dof_count;
            result = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
            if (result != CONSTRAINT_SUCCESS)
                return result;
            if (entries_per_row > SIZE_MAX - element_dof_count)
                return CONSTRAINT_SIZE_OVERFLOW;
            entries_per_row += element_dof_count;
        }
        if (test_dof_count > 0 && entries_per_row > SIZE_MAX / test_dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        const size_t component_entries = test_dof_count * entries_per_row;
        if (entry_count > SIZE_MAX - component_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        entry_count += component_entries;
    }
    *out_row_count = row_count;
    *out_entry_count = entry_count;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Public storage-query wrapper for one-sided physical assembly.
 *
 * @param test_spec Face test-space specification.
 * @param side Element-side specification.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return The status returned by constraint_physical_side_counts.
 */
constraint_status_t constraint_physical_side_required(const constraint_kform_spec_t *const test_spec,
                                                      const constraint_element_side_t *const side,
                                                      size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !side)
        return CONSTRAINT_INVALID_ARGUMENT;
    return constraint_physical_side_counts(test_spec, side, out_row_count, out_entry_count);
}

/**
 * @brief Decode one component-local DoF into tensor-product basis digits.
 *
 * Digits are produced by mixed-radix division, with the last element axis as
 * the fastest-changing digit. Active form axes have one fewer basis function,
 * so the radix is selected from the component's wedge combination.
 *
 * @param ndim Number of element axes.
 * @param order Form degree.
 * @param basis Element-axis basis specifications.
 * @param component Component whose local DoF is being decoded.
 * @param dof Local DoF index.
 * @param digits Output array of `ndim` basis indices.
 */
static void decode_component_dof(const unsigned ndim, const unsigned order, const basis_spec_t basis[const static ndim],
                                 const unsigned component, size_t dof, unsigned digits[const static ndim])
{
    uint8_t axes[UINT8_MAX];
    component_axes(ndim, order, component, axes);
    for (unsigned idim = ndim; idim > 0; --idim)
    {
        const bool active = component_has_axis(order, axes, idim - 1);
        const size_t dimension_size = (size_t)basis[idim - 1].order + (active ? 0 : 1);
        digits[idim - 1] = (unsigned)(dof % dimension_size);
        dof /= dimension_size;
    }
}

/**
 * @brief Sum the local DoF counts of all components before one component.
 *
 * Replaces cumulative offset tables so callers never need scratch storage
 * proportional to the component count.
 *
 * @param spec Validated k-form specification.
 * @param component Exclusive component bound.
 * @param out_start Receives the summed DoF count of components `[0, component)`.
 * @return A public constraint status.
 */
static constraint_status_t component_dof_start(const constraint_kform_spec_t *const spec, const unsigned component,
                                               size_t *const out_start)
{
    size_t start = 0;
    for (unsigned index = 0; index < component; ++index)
    {
        size_t count;
        const constraint_status_t status = constraint_kform_component_dof_count(spec, index, &count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        start += count;
    }
    *out_start = start;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Evaluate one one-dimensional basis function at a coordinate.
 *
 * Order-zero factors are constant one. Higher-order factors use the basis
 * implementation's prepare/evaluate pair, keeping this helper independent of
 * the concrete polynomial family.
 *
 * @param spec Basis type and order to evaluate.
 * @param index Basis-function index.
 * @param x Coordinate in the reference interval.
 * @return The basis value at `x`.
 */
static double evaluate_basis_value(const basis_spec_t spec, const unsigned index, const double x)
{
    const unsigned order = index == 0 && spec.order == 0 ? 0 : spec.order;
    if (order == 0)
        return 1.0;

    double values[UINT8_MAX + 1];
    double work[UINT8_MAX + 1];
    basis_compute_at_point_prepare(spec.type, order, work);
    basis_compute_at_point_values(spec.type, order, 1, &x, values, work);
    return values[index];
}

/**
 * @brief Validate quadrature axes and compute their tensor-product size.
 *
 * Point flattening throughout this file treats the final axis as the fastest
 * varying axis. The product is accumulated with a pre-check so malformed
 * rules cannot wrap a `size_t` count.
 *
 * @param ndim Number of axes.
 * @param quadrature Axis rules; may be null only when `ndim == 0`.
 * @param out_count Receives the product of all axis counts.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INVALID_ARGUMENT`, or
 *         `CONSTRAINT_SIZE_OVERFLOW`.
 */
static constraint_status_t quadrature_total_count(const unsigned ndim, const constraint_quadrature_t *quadrature,
                                                  size_t *const out_count)
{
    if (ndim != 0 && !quadrature)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t count = 1;
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        if (quadrature[idim].count == 0 || !quadrature[idim].nodes || !quadrature[idim].weights)
            return CONSTRAINT_INVALID_ARGUMENT;
        if (count > SIZE_MAX / quadrature[idim].count)
            return CONSTRAINT_SIZE_OVERFLOW;
        count *= quadrature[idim].count;
    }
    *out_count = count;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Evaluate the product of test and element trace bases at one point.
 *
 * This forward declaration lets the quadrature integrator appear before the
 * pointwise basis implementation, while keeping pointwise multiplication in
 * one helper.
 */
static double trace_basis_product_at_point(const constraint_kform_spec_t *test_spec,
                                           const constraint_element_side_t *side, unsigned test_component,
                                           const unsigned test_digits[const static test_spec->ndim],
                                           unsigned element_component,
                                           const unsigned element_digits[const static side->ndim],
                                           const double face_nodes[const static test_spec->ndim]);

/**
 * @brief Integrate one test/element trace-basis product over a face.
 *
 * Tensor-product point indices are decoded in the same order as
 * quadrature_total_count. Invalid quadrature metadata is represented by zero
 * here; public callers validate it before relying on the resulting entries.
 *
 * @param test_spec Face test-space specification.
 * @param side Element-side specification.
 * @param quadrature Face-axis quadrature rules.
 * @param test_component Test-space component.
 * @param test_digits Test component-local DoF digits.
 * @param element_component Mapped element component.
 * @param element_digits Element component-local DoF digits.
 * @return The quadrature-weighted inner product.
 */
static double trace_inner_product(const constraint_kform_spec_t *const test_spec,
                                  const constraint_element_side_t *const side,
                                  const constraint_quadrature_t *quadrature, const unsigned test_component,
                                  const unsigned test_digits[const static test_spec->ndim],
                                  const unsigned element_component,
                                  const unsigned element_digits[const static side->ndim])
{
    const unsigned face_dim = test_spec->ndim;
    size_t quadrature_count;
    const constraint_status_t quadrature_status = quadrature_total_count(face_dim, quadrature, &quadrature_count);
    if (quadrature_status != CONSTRAINT_SUCCESS)
        return 0.0;
    double result = 0;
    double face_nodes[UINT8_MAX];
    for (size_t point = 0; point < quadrature_count; ++point)
    {
        size_t remaining = point;
        double weight = 1.0;
        for (unsigned idim = face_dim; idim > 0; --idim)
        {
            const unsigned face_axis = idim - 1;
            const unsigned index = (unsigned)(remaining % quadrature[face_axis].count);
            remaining /= quadrature[face_axis].count;
            face_nodes[face_axis] = quadrature[face_axis].nodes[index];
            weight *= quadrature[face_axis].weights[index];
        }
        result += weight * trace_basis_product_at_point(test_spec, side, test_component, test_digits, element_component,
                                                        element_digits, face_nodes);
    }
    return result;
}

/**
 * @brief Evaluate one element k-form trace basis function on a face point.
 *
 * Fixed axes are placed at signed endpoints and face axes receive mapped
 * canonical coordinates. Each element-axis factor uses the reduced basis order
 * appropriate to whether that axis is active in the traced component.
 *
 * @param face_dim Number of canonical face dimensions.
 * @param side Element-side mapping and bases.
 * @param order Traced form degree.
 * @param element_component Element component being evaluated.
 * @param element_digits Local DoF digits of that component.
 * @param face_nodes Canonical face coordinates.
 * @return The element trace basis value.
 */
static double element_trace_basis_value(const unsigned face_dim, const constraint_element_side_t *const side,
                                        const unsigned order, const unsigned element_component,
                                        const unsigned element_digits[const static side->ndim],
                                        const double face_nodes[const static face_dim == 0 ? 1 : face_dim])
{
    const unsigned fixed_count = side->ndim - face_dim;
    uint8_t element_axes[UINT8_MAX];
    component_axes(side->ndim, order, element_component, element_axes);

    double element_coordinates[UINT8_MAX];
    // Fixed axes are boundary coordinates of the element; their orientation
    // sign selects the endpoint at which the face is embedded.
    for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
        element_coordinates[(unsigned)(side->orientation[fixed_axis] < 0 ? -side->orientation[fixed_axis]
                                                                         : side->orientation[fixed_axis]) -
                            1] = side->orientation[fixed_axis] < 0 ? -1.0 : 1.0;

    // Remaining orientation entries map canonical face coordinates into the
    // element frame and may reverse each coordinate independently.
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping = side->orientation[fixed_count + face_axis];
        const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
        element_coordinates[element_axis] = mapping < 0 ? -face_nodes[face_axis] : face_nodes[face_axis];
    }

    double value = 1.0;
    // Active covector axes use the reduced one-dimensional basis order; all
    // factors are multiplied to form the tensor-product trace basis.
    for (unsigned element_axis = 0; element_axis < side->ndim; ++element_axis)
    {
        const bool element_active = component_has_axis(order, element_axes, element_axis);
        const unsigned element_order = side->basis_specs[element_axis].order - (element_active ? 1 : 0);
        value *=
            evaluate_basis_value((basis_spec_t){.type = side->basis_specs[element_axis].type, .order = element_order},
                                 element_digits[element_axis], element_coordinates[element_axis]);
    }
    return value;
}

/**
 * @brief Evaluate the test trace basis times the element trace basis.
 *
 * The element factor carries the face-to-element coordinate mapping; the test
 * factor is evaluated directly in canonical face coordinates. Component
 * orientation signs are applied by the assembly caller, not hidden here.
 *
 * @param test_spec Face test-space specification.
 * @param side Element-side mapping and bases.
 * @param test_component Test component.
 * @param test_digits Test component-local DoF digits.
 * @param element_component Mapped element component.
 * @param element_digits Element component-local DoF digits.
 * @param face_nodes Canonical face coordinates.
 * @return Product of the two trace basis values.
 */
static double trace_basis_product_at_point(const constraint_kform_spec_t *const test_spec,
                                           const constraint_element_side_t *const side, const unsigned test_component,
                                           const unsigned test_digits[const static test_spec->ndim],
                                           const unsigned element_component,
                                           const unsigned element_digits[const static side->ndim],
                                           const double face_nodes[const static test_spec->ndim])
{
    const unsigned face_dim = test_spec->ndim;
    uint8_t test_axes[UINT8_MAX];
    component_axes(face_dim, test_spec->order, test_component, test_axes);

    double value =
        element_trace_basis_value(face_dim, side, test_spec->order, element_component, element_digits, face_nodes);
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const bool test_active = component_has_axis(test_spec->order, test_axes, face_axis);
        const unsigned test_order = test_spec->basis_specs[face_axis].order - (test_active ? 1 : 0);
        value *=
            evaluate_basis_value((basis_spec_t){.type = test_spec->basis_specs[face_axis].type, .order = test_order},
                                 test_digits[face_axis], face_nodes[face_axis]);
    }
    return value;
}

/**
 * @brief Take the physical-component dot product of two pullback samples.
 *
 * Values are strided by `point_count`, so this selects one point from two
 * component blocks without copying either sampled vector.
 *
 * @param pullback Pullback table containing both element-component samples.
 * @param first_component First element component.
 * @param second_component Second element component.
 * @param physical_component_count Number of physical components per sample.
 * @param point_count Number of points in each sample block.
 * @param point Point to compare.
 * @return The Euclidean dot product at `point`.
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
 * @brief Assemble the two-sided physical trace matrix.
 *
 * All validation and capacity checks occur before the row/entry loops. Each
 * row then visits both sides and every face component; surface measure,
 * pullback coupling, basis values, and the side signs are multiplied at the
 * innermost point loop.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param quadrature Two face quadrature descriptions.
 * @param surface_weights Two unsigned face-measure arrays.
 * @param pullbacks Two sampled physical pullback tables.
 * @param row_offset_capacity Capacity of `row_offsets`.
 * @param row_offsets Output packed row offsets.
 * @param entry_capacity Capacity of `entries`.
 * @param entries Output packed entries.
 * @param out_row_count Receives the row count on success.
 * @param out_entry_count Receives the entry count on success.
 * @return A public constraint status.
 */
constraint_status_t constraint_physical_assemble(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t sides[const static 2],
    const constraint_face_quadrature_t quadrature[const static 2], const double *const surface_weights[const static 2],
    const constraint_trace_pullback_t pullbacks[const static 2], const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !surface_weights[0] || !surface_weights[1])
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t row_count;
    size_t required_entries;
    constraint_status_t status = constraint_physical_counts(test_spec, sides, &row_count, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t required_offsets;
    status = constraint_rows_required_offset_count(row_count, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    // Scalar traces do not have a tangential pullback; non-zero forms require
    // one sampled pullback table per side with the same point count as its rule.
    if (test_spec->order != 0)
    {
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            if (!pullbacks[side_index].values || pullbacks[side_index].physical_component_count == 0 ||
                pullbacks[side_index].point_count == 0)
                return CONSTRAINT_INVALID_ARGUMENT;
            size_t side_point_count;
            status =
                quadrature_total_count(quadrature[side_index].ndim, quadrature[side_index].axes, &side_point_count);
            if (status != CONSTRAINT_SUCCESS || side_point_count != quadrature[side_index].point_count ||
                side_point_count != pullbacks[side_index].point_count)
                return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
        }
    }
    else
    {
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            size_t side_point_count;
            status =
                quadrature_total_count(quadrature[side_index].ndim, quadrature[side_index].axes, &side_point_count);
            if (status != CONSTRAINT_SUCCESS || side_point_count != quadrature[side_index].point_count)
                return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
        }
    }

    size_t test_component_count;
    status = constraint_kform_component_count(test_spec, &test_component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    // This nested order is the packed-row contract: component-major, then
    // component-local DoF order, with one offset written after each row.
    for (unsigned test_component = 0; test_component < test_component_count; ++test_component)
    {
        uint8_t test_axes[UINT8_MAX];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        size_t test_dof_count;
        status = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_digits[UINT8_MAX];
            decode_component_dof(test_spec->ndim, test_spec->order, test_spec->basis_specs, test_component, test_dof,
                                 test_digits);
            for (unsigned side_index = 0; side_index < 2; ++side_index)
            {
                const constraint_element_side_t *const side = sides + side_index;
                const constraint_trace_pullback_t *const pullback = pullbacks + side_index;
                const unsigned face_component_count =
                    combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
                unsigned test_element_component = 0;
                int test_orientation_sign = 1;
                if (test_spec->order != 0)
                    mapped_component(side, test_spec->ndim, test_spec->order, test_axes, &test_element_component,
                                     &test_orientation_sign);

                // Every mapped face component can couple through the physical
                // pullback, so physical assembly visits all component blocks.
                for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
                {
                    uint8_t face_axes[UINT8_MAX];
                    component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
                    unsigned element_component;
                    int orientation_sign;
                    mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component,
                                     &orientation_sign);
                    const constraint_kform_spec_t element_spec = {
                        .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                    size_t element_dof_count;
                    constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                    for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                    {
                        unsigned element_digits[UINT8_MAX];
                        decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component,
                                             element_dof, element_digits);
                        double coefficient = 0.0;
                        size_t point_count;
                        status = quadrature_total_count(quadrature[side_index].ndim, quadrature[side_index].axes,
                                                        &point_count);
                        if (status != CONSTRAINT_SUCCESS)
                            return status;
                        // Decode the flat tensor-product point index with the
                        // final face axis as the fastest-changing coordinate.
                        for (size_t point = 0; point < point_count; ++point)
                        {
                            size_t remaining = point;
                            double quadrature_weight = 1.0;
                            double face_nodes[UINT8_MAX];
                            for (unsigned idim = test_spec->ndim; idim > 0; --idim)
                            {
                                const unsigned face_axis = idim - 1;
                                const unsigned node_index =
                                    (unsigned)(remaining % quadrature[side_index].axes[face_axis].count);
                                remaining /= quadrature[side_index].axes[face_axis].count;
                                face_nodes[face_axis] = quadrature[side_index].axes[face_axis].nodes[node_index];
                                quadrature_weight *= quadrature[side_index].axes[face_axis].weights[node_index];
                            }
                            double pullback_factor = 1.0;
                            if (test_spec->order != 0)
                            {
                                if (test_element_component >=
                                        combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order) ||
                                    element_component >=
                                        combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order))
                                    return CONSTRAINT_INVALID_ARGUMENT;
                                pullback_factor =
                                    trace_pullback_dot(pullback, test_element_component, element_component,
                                                       pullback->physical_component_count, point_count, point);
                            }
                            coefficient += quadrature_weight * surface_weights[side_index][point] * pullback_factor *
                                           trace_basis_product_at_point(test_spec, side, test_component, test_digits,
                                                                        element_component, element_digits, face_nodes);
                        }
                        entries[entry++] = (constraint_entry_t){
                            .side = (uint8_t)side_index,
                            .component = element_component,
                            .local_dof = element_dof,
                            .coefficient = (side_index == 0 ? 1.0 : -1.0) * (double)test_orientation_sign *
                                           (double)orientation_sign * coefficient,
                        };
                    }
                }
            }
            row_offsets[row + 1] = entry;
        }
    }

    *out_row_count = row_count;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Compute combined storage requirements for a physical batch.
 *
 * Each item is counted independently and totals are accumulated in input
 * order with overflow checks. The count depends only on the spaces and sides,
 * so quadrature contents are intentionally not inspected here.
 *
 * @param test_spec Shared face test-space specification.
 * @param item_count Number of batch items.
 * @param items Batch descriptors.
 * @param out_row_count Receives the total row count.
 * @param out_entry_count Receives the total entry count.
 * @return A validation, overflow, or success status.
 */
constraint_status_t constraint_physical_batch_required(
    const constraint_kform_spec_t *const test_spec, const size_t item_count,
    const constraint_physical_batch_item_t items[const static item_count], size_t *const out_row_count,
    size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    size_t total_rows = 0;
    size_t total_entries = 0;
    for (size_t item = 0; item < item_count; ++item)
    {
        if (!items[item].sides || !items[item].quadrature || !items[item].surface_weights[0] ||
            !items[item].surface_weights[1])
            return CONSTRAINT_INVALID_ARGUMENT;
        if (test_spec && test_spec->order != 0 && !items[item].pullbacks)
            return CONSTRAINT_INVALID_ARGUMENT;
        size_t item_rows;
        size_t item_entries;
        const constraint_status_t status =
            constraint_physical_required(test_spec, items[item].sides, &item_rows, &item_entries);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        if (total_rows > SIZE_MAX - item_rows || total_entries > SIZE_MAX - item_entries)
            return CONSTRAINT_SIZE_OVERFLOW;
        total_rows += item_rows;
        total_entries += item_entries;
    }
    if (item_count == 0)
    {
        size_t ignored;
        const constraint_status_t status = constraint_kform_component_count(test_spec, &ignored);
        if (status != CONSTRAINT_SUCCESS)
            return status;
    }
    *out_row_count = total_rows;
    *out_entry_count = total_entries;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Assemble a concatenated batch of physical trace matrices.
 *
 * The per-item assembler writes local rows first; this wrapper shifts each
 * item's row offsets by the entry prefix accumulated from earlier items. The
 * initial requirement query checks aggregate capacity before dispatch.
 *
 * @param test_spec Shared face test-space specification.
 * @param item_count Number of batch items.
 * @param items Batch descriptors in output order.
 * @param row_offset_capacity Capacity of combined row offsets.
 * @param row_offsets Output combined row offsets.
 * @param entry_capacity Capacity of combined entries.
 * @param entries Output combined entries.
 * @param out_row_count Receives the total row count.
 * @param out_entry_count Receives the total entry count.
 * @return A public constraint status.
 */
constraint_status_t constraint_physical_batch_assemble(
    const constraint_kform_spec_t *const test_spec, const size_t item_count,
    const constraint_physical_batch_item_t items[const static item_count], const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    size_t required_rows;
    size_t required_entries;
    constraint_status_t status =
        constraint_physical_batch_required(test_spec, item_count, items, &required_rows, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t required_offsets;
    status = constraint_rows_required_offset_count(required_rows, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    // Each item is assembled into its own slice first; the entry prefix is
    // then added to that item's local row offsets before advancing the slices.
    for (size_t item = 0; item < item_count; ++item)
    {
        size_t item_rows;
        size_t item_entries;
        status = constraint_physical_assemble(test_spec, items[item].sides, items[item].quadrature,
                                              items[item].surface_weights, items[item].pullbacks,
                                              row_offset_capacity - row, row_offsets + row, entry_capacity - entry,
                                              entries + entry, &item_rows, &item_entries);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        row_offsets[row] = entry;
        // The nested assembler starts its offsets at zero, so rebase every
        // local row boundary by the global entry prefix.
        for (size_t local_row = 1; local_row <= item_rows; ++local_row)
            row_offsets[row + local_row] += entry;
        row += item_rows;
        entry += item_entries;
    }
    *out_row_count = row;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Assemble a one-sided physical trace matrix.
 *
 * This follows the same row/component traversal as the two-sided assembler but
 * emits only side zero. The test-side orientation sign and candidate
 * element-component orientation sign remain separate until each entry is
 * written, making both parity contributions explicit.
 *
 * @param test_spec Face test-space specification.
 * @param side Element-side specification.
 * @param quadrature Face quadrature.
 * @param surface_weights Unsigned face-measure values.
 * @param pullback Sampled physical pullback for non-zero form order.
 * @param row_offset_capacity Capacity of `row_offsets`.
 * @param row_offsets Output packed row offsets.
 * @param entry_capacity Capacity of `entries`.
 * @param entries Output packed entries.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return A public constraint status.
 */
constraint_status_t constraint_physical_side_assemble(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t *const side,
    const constraint_face_quadrature_t *const quadrature, const double *const surface_weights,
    const constraint_trace_pullback_t *const pullback, const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !side || !quadrature || !surface_weights)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t row_count;
    size_t required_entries;
    constraint_status_t status = constraint_physical_side_counts(test_spec, side, &row_count, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t required_offsets;
    status = constraint_rows_required_offset_count(row_count, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t point_count;
    status = quadrature_total_count(quadrature->ndim, quadrature->axes, &point_count);
    if (status != CONSTRAINT_SUCCESS || point_count != quadrature->point_count)
        return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
    if (test_spec->order != 0 && (!pullback || !pullback->values || pullback->physical_component_count == 0 ||
                                  pullback->point_count != point_count))
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t component_count;
    status = constraint_kform_component_count(test_spec, &component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        uint8_t test_axes[UINT8_MAX];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        size_t test_dof_count;
        status = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_digits[UINT8_MAX];
            decode_component_dof(test_spec->ndim, test_spec->order, test_spec->basis_specs, test_component, test_dof,
                                 test_digits);
            unsigned test_element_component = 0;
            int test_orientation_sign = 1;
            if (test_spec->order != 0)
                mapped_component(side, test_spec->ndim, test_spec->order, test_axes, &test_element_component,
                                 &test_orientation_sign);

            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                uint8_t face_axes[UINT8_MAX];
                component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component,
                                 &orientation_sign);
                const constraint_kform_spec_t element_spec = {
                    .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                size_t element_dof_count;
                status = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                if (status != CONSTRAINT_SUCCESS)
                    return status;
                for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                {
                    unsigned element_digits[UINT8_MAX];
                    decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component,
                                         element_dof, element_digits);
                    double coefficient = 0.0;
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        size_t remaining = point;
                        double quadrature_weight = 1.0;
                        double face_nodes[UINT8_MAX];
                        for (unsigned idim = test_spec->ndim; idim > 0; --idim)
                        {
                            const unsigned face_axis = idim - 1;
                            const unsigned node_index = (unsigned)(remaining % quadrature->axes[face_axis].count);
                            remaining /= quadrature->axes[face_axis].count;
                            face_nodes[face_axis] = quadrature->axes[face_axis].nodes[node_index];
                            quadrature_weight *= quadrature->axes[face_axis].weights[node_index];
                        }
                        double pullback_factor = 1.0;
                        if (test_spec->order != 0)
                        {
                            if (test_element_component >=
                                    combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order) ||
                                element_component >=
                                    combination_total_count((uint8_t)side->ndim, (uint8_t)test_spec->order))
                                return CONSTRAINT_INVALID_ARGUMENT;
                            pullback_factor =
                                trace_pullback_dot(pullback, test_element_component, element_component,
                                                   pullback->physical_component_count, point_count, point);
                        }
                        coefficient += quadrature_weight * surface_weights[point] * pullback_factor *
                                       trace_basis_product_at_point(test_spec, side, test_component, test_digits,
                                                                    element_component, element_digits, face_nodes);
                    }
                    entries[entry++] = (constraint_entry_t){
                        .side = 0,
                        .component = element_component,
                        .local_dof = element_dof,
                        .coefficient = (double)test_orientation_sign * (double)orientation_sign * coefficient,
                    };
                }
            }
            row_offsets[row + 1] = entry;
        }
    }
    *out_row_count = row_count;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Assemble a one-sided physical trace from precomputed basis tables.
 *
 * This table path preserves the direct assembler's row order, orientation
 * signs, pullback dot products, and face-measure factors while replacing
 * repeated basis evaluation with indexed table reads. Metadata is validated
 * before output rows are written.
 *
 * @param test_spec Face test-space specification.
 * @param side Element-side specification.
 * @param quadrature Face quadrature matching both tables.
 * @param surface_weights Unsigned face-measure values.
 * @param pullback Sampled physical pullback for non-zero form order.
 * @param test_basis Precomputed canonical test trace values.
 * @param element_basis Precomputed element trace values.
 * @param row_offset_capacity Capacity of `row_offsets`.
 * @param row_offsets Output packed row offsets.
 * @param entry_capacity Capacity of `entries`.
 * @param entries Output packed entries.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return A public constraint status.
 */
constraint_status_t constraint_physical_side_assemble_precomputed(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t *const side,
    const constraint_face_quadrature_t *const quadrature, const double *const surface_weights,
    const constraint_trace_pullback_t *const pullback, const constraint_trace_basis_values_t *const test_basis,
    const constraint_trace_basis_values_t *const element_basis, const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count || !side || !quadrature || !surface_weights)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t required_rows;
    size_t required_entries;
    constraint_status_t status = constraint_physical_side_counts(test_spec, side, &required_rows, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t required_offsets;
    status = constraint_rows_required_offset_count(required_rows, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t point_count;
    status = quadrature_total_count(quadrature->ndim, quadrature->axes, &point_count);
    if (status != CONSTRAINT_SUCCESS || quadrature->point_count != point_count)
        return status == CONSTRAINT_SUCCESS ? CONSTRAINT_INVALID_ARGUMENT : status;
    if (test_spec->order != 0 && (!pullback || !pullback->values || pullback->physical_component_count == 0 ||
                                  pullback->point_count != point_count))
        return CONSTRAINT_INVALID_ARGUMENT;

    const constraint_kform_spec_t element_spec = {
        .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
    status = validate_trace_basis_values(test_spec, test_basis, point_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    status = validate_trace_basis_values(&element_spec, element_basis, point_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t test_component_count;
    status = constraint_kform_component_count(test_spec, &test_component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
    const size_t *const test_offsets = test_basis->component_offsets;
    const size_t *const element_offsets = element_basis->component_offsets;
    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < test_component_count; ++test_component)
    {
        const size_t test_dof_count = test_offsets[test_component + 1] - test_offsets[test_component];
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_element_component = 0;
            int test_orientation_sign = 1;
            status = mapped_component_for_index(side, test_spec->ndim, test_spec->order, test_component,
                                                &test_element_component, &test_orientation_sign);
            if (status != CONSTRAINT_SUCCESS)
                return status;
            for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
            {
                unsigned element_component;
                int orientation_sign;
                status = mapped_component_for_index(side, test_spec->ndim, test_spec->order, face_component,
                                                    &element_component, &orientation_sign);
                if (status != CONSTRAINT_SUCCESS)
                    return status;
                const size_t element_dof_count =
                    element_offsets[element_component + 1] - element_offsets[element_component];
                for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                {
                    double coefficient = 0.0;
                    for (size_t point = 0; point < point_count; ++point)
                    {
                        size_t remaining = point;
                        double quadrature_weight = 1.0;
                        for (unsigned idim = quadrature->ndim; idim > 0; --idim)
                        {
                            const unsigned axis = idim - 1;
                            const unsigned node = (unsigned)(remaining % quadrature->axes[axis].count);
                            remaining /= quadrature->axes[axis].count;
                            quadrature_weight *= quadrature->axes[axis].weights[node];
                        }
                        double pullback_factor = 1.0;
                        if (test_spec->order != 0)
                            pullback_factor =
                                trace_pullback_dot(pullback, test_element_component, element_component,
                                                   pullback->physical_component_count, point_count, point);
                        const double test_value = test_basis->values[test_offsets[test_component] * point_count +
                                                                     point * test_dof_count + test_dof];
                        const double element_value =
                            element_basis->values[element_offsets[element_component] * point_count +
                                                  point * element_dof_count + element_dof];
                        coefficient +=
                            quadrature_weight * surface_weights[point] * pullback_factor * test_value * element_value;
                    }
                    entries[entry++] = (constraint_entry_t){
                        .side = 0,
                        .component = element_component,
                        .local_dof = element_dof,
                        .coefficient = (double)test_orientation_sign * (double)orientation_sign * coefficient,
                    };
                }
            }
            row_offsets[row + 1] = entry;
        }
    }
    if (row_offset_capacity < row + 1)
        return CONSTRAINT_INSUFFICIENT_STORAGE;
    if (entry_capacity < entry)
        return CONSTRAINT_INSUFFICIENT_STORAGE;
    *out_row_count = row;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Assemble a boundary load from sampled element-frame k-form data.
 *
 * For each traced component, this selects the datum component containing the
 * fixed normal axis, applies the wedge insertion sign, and accumulates its
 * quadrature pairing with every element trace basis function. The accumulator
 * is intentionally not cleared so multiple faces can contribute to one load.
 *
 * @param test_spec Face test-space specification of degree one below the datum.
 * @param side Codimension-one element-side specification.
 * @param quadrature Face quadrature.
 * @param datum_values Sampled element-frame datum components.
 * @param value_count Length of `values`.
 * @param surface_weights Optional unsigned face measures.
 * @param values Output accumulator.
 * @return A public constraint status.
 */
constraint_status_t constraint_physical_side_load(const constraint_kform_spec_t *const test_spec,
                                                  const constraint_element_side_t *const side,
                                                  const constraint_face_quadrature_t *const quadrature,
                                                  const double *const datum_values, const size_t value_count,
                                                  const double *const surface_weights,
                                                  double values[const static value_count])
{
    if (!test_spec || !side || !quadrature || !datum_values)
        return CONSTRAINT_INVALID_ARGUMENT;

    constraint_status_t status =
        constraint_reference_validate(test_spec, (const constraint_element_side_t[2]){*side, *side});
    if (status != CONSTRAINT_SUCCESS)
        return status;
    // The load is defined on codimension-1 faces: exactly one fixed normal axis.
    if (side->ndim != test_spec->ndim + 1)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t point_count;
    status = quadrature_total_count(quadrature->ndim, quadrature->axes, &point_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (quadrature->ndim != test_spec->ndim || point_count != quadrature->point_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    const constraint_kform_spec_t element_spec = {
        .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
    size_t element_component_count;
    status = constraint_kform_component_count(&element_spec, &element_component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    size_t value_total;
    status = component_dof_start(&element_spec, (unsigned)element_component_count, &value_total);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (value_total != value_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    const unsigned face_component_count = combination_total_count((uint8_t)test_spec->ndim, (uint8_t)test_spec->order);
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
        component_axes(test_spec->ndim, test_spec->order, face_component, face_axes);
        unsigned element_component;
        int orientation_sign;
        status =
            mapped_component(side, test_spec->ndim, test_spec->order, face_axes, &element_component, &orientation_sign);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        size_t element_dof_count;
        status = constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        size_t element_start;
        status = component_dof_start(&element_spec, element_component, &element_start);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        uint8_t element_axes[UINT8_MAX];
        component_axes(side->ndim, test_spec->order, element_component, element_axes);
        unsigned exponent_below_fixed = 0;
        for (unsigned i = 0; i < test_spec->order; ++i)
            exponent_below_fixed += element_axes[i] < fixed_axis ? 1u : 0u;
        uint8_t datum_axes[UINT8_MAX + 1];
        for (unsigned i = 0; i < test_spec->order; ++i)
            datum_axes[i] = element_axes[i];
        datum_axes[test_spec->order] = (uint8_t)fixed_axis;
        for (unsigned i = test_spec->order; i > 0 && datum_axes[i - 1] > datum_axes[i]; --i)
        {
            const uint8_t tmp = datum_axes[i - 1];
            datum_axes[i - 1] = datum_axes[i];
            datum_axes[i] = tmp;
        }
        const unsigned datum_component = combination_get_index(side->ndim, test_spec->order + 1, datum_axes);
        const double sign = side_sign * (double)orientation_sign * (exponent_below_fixed % 2 == 0 ? 1.0 : -1.0);
        for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
        {
            unsigned element_digits[UINT8_MAX];
            decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component, element_dof,
                                 element_digits);
            double coefficient = 0.0;
            for (size_t point = 0; point < point_count; ++point)
            {
                size_t remaining = point;
                double quadrature_weight = 1.0;
                double face_nodes[UINT8_MAX];
                for (unsigned idim = test_spec->ndim; idim > 0; --idim)
                {
                    const unsigned face_axis = idim - 1;
                    const unsigned node_index = (unsigned)(remaining % quadrature->axes[face_axis].count);
                    remaining /= quadrature->axes[face_axis].count;
                    face_nodes[face_axis] = quadrature->axes[face_axis].nodes[node_index];
                    quadrature_weight *= quadrature->axes[face_axis].weights[node_index];
                }
                const double weight_total =
                    surface_weights ? quadrature_weight * surface_weights[point] : quadrature_weight;
                coefficient += weight_total * datum_values[datum_component * point_count + point] *
                               element_trace_basis_value(test_spec->ndim, side, test_spec->order, element_component,
                                                         element_digits, face_nodes);
            }
            values[element_start + element_dof] += sign * coefficient;
        }
    }
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Assemble the two-sided reference-space trace matrix.
 *
 * After sizing and quadrature validation, each test DoF is paired with the
 * mapped component on each side. The scalar face integral is shared by all
 * entries of that component row, while side and orientation signs are applied
 * when the packed entry is emitted.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param quadrature Face-axis quadrature rules.
 * @param row_offset_capacity Capacity of `row_offsets`.
 * @param row_offsets Output packed row offsets.
 * @param entry_capacity Capacity of `entries`.
 * @param entries Output packed entries.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return A public constraint status.
 */
constraint_status_t constraint_reference_assemble(
    const constraint_kform_spec_t *const test_spec, const constraint_element_side_t sides[const static 2],
    const constraint_quadrature_t *quadrature, const size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], const size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *const out_row_count, size_t *const out_entry_count)
{
    if (!out_row_count || !out_entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    size_t row_count;
    size_t required_entries;
    constraint_status_t status = constraint_reference_counts(test_spec, sides, &row_count, &required_entries);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t required_offsets;
    status = constraint_rows_required_offset_count(row_count, &required_offsets);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (row_offset_capacity < required_offsets || entry_capacity < required_entries)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t quadrature_count;
    status = quadrature_total_count(test_spec->ndim, quadrature, &quadrature_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    (void)quadrature_count;

    size_t component_count;
    status = constraint_kform_component_count(test_spec, &component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    size_t row = 0;
    size_t entry = 0;
    row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        uint8_t test_axes[UINT8_MAX];
        component_axes(test_spec->ndim, test_spec->order, test_component, test_axes);
        size_t test_dof_count;
        status = constraint_kform_component_dof_count(test_spec, test_component, &test_dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            unsigned test_digits[UINT8_MAX];
            decode_component_dof(test_spec->ndim, test_spec->order, test_spec->basis_specs, test_component, test_dof,
                                 test_digits);
            for (unsigned side_index = 0; side_index < 2; ++side_index)
            {
                const constraint_element_side_t *const side = sides + side_index;
                unsigned element_component;
                int orientation_sign;
                mapped_component(side, test_spec->ndim, test_spec->order, test_axes, &element_component,
                                 &orientation_sign);
                const constraint_kform_spec_t element_spec = {
                    .ndim = side->ndim, .order = test_spec->order, .basis_specs = side->basis_specs};
                size_t element_dof_count;
                constraint_kform_component_dof_count(&element_spec, element_component, &element_dof_count);
                for (size_t element_dof = 0; element_dof < element_dof_count; ++element_dof)
                {
                    unsigned element_digits[UINT8_MAX];
                    decode_component_dof(side->ndim, test_spec->order, side->basis_specs, element_component,
                                         element_dof, element_digits);
                    entries[entry++] = (constraint_entry_t){
                        .side = (uint8_t)side_index,
                        .component = element_component,
                        .local_dof = element_dof,
                        .coefficient = (side_index == 0 ? 1.0 : -1.0) * (double)orientation_sign *
                                       trace_inner_product(test_spec, side, quadrature, test_component, test_digits,
                                                           element_component, element_digits),
                    };
                }
            }
            row_offsets[row + 1] = entry;
        }
    }

    *out_row_count = row_count;
    *out_entry_count = entry;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Count the validated specification's k-form components.
 *
 * The combination iterator supplies the binomial count and the public helper
 * preserves its validation status rather than narrowing invalid dimensions.
 *
 * @param spec Test-space specification.
 * @param out_count Receives `C(spec->ndim, spec->order)`.
 * @return A public constraint status.
 */
constraint_status_t constraint_kform_component_count(const constraint_kform_spec_t *const spec, size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    const constraint_status_t status = validate_kform_spec(spec);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    *out_count = combination_total_count((uint8_t)spec->ndim, (uint8_t)spec->order);
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Count the local DoFs of one validated k-form component.
 *
 * The component combination determines which axes use reduced covector basis
 * orders. The product is accumulated with an overflow check before output is
 * committed.
 *
 * @param spec Test-space specification.
 * @param component Component combination index.
 * @param out_count Receives the local DoF count.
 * @return A public constraint status.
 */
constraint_status_t constraint_kform_component_dof_count(const constraint_kform_spec_t *const spec,
                                                         const unsigned component, size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    const constraint_status_t status = validate_kform_spec(spec);
    if (status != CONSTRAINT_SUCCESS)
        return status;

    const size_t component_count = combination_total_count((uint8_t)spec->ndim, (uint8_t)spec->order);
    if ((size_t)component >= component_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    uint8_t component_axes[UINT8_MAX];
    combination_set_to_index((uint8_t)spec->ndim, (uint8_t)spec->order, component_axes, component);

    size_t dof_count = 1;
    for (unsigned idim = 0, iaxis = 0; idim < spec->ndim; ++idim)
    {
        const bool active = iaxis < spec->order && component_axes[iaxis] == idim;
        const size_t dimension_size = (size_t)spec->basis_specs[idim].order + (active ? 0 : 1);
        if (dimension_size != 0 && dof_count > SIZE_MAX / dimension_size)
            return CONSTRAINT_SIZE_OVERFLOW;
        dof_count *= dimension_size;
        if (active)
            ++iaxis;
    }

    *out_count = dof_count;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Build cumulative offsets for all component-local DoFs.
 *
 * The offset array is the shared indexing contract for packed values and
 * precomputed basis tables. Each component count is checked before the running
 * total advances, so a failure cannot publish an overflowing final offset.
 *
 * @param spec Test-space specification.
 * @param offset_count Number of available offset entries.
 * @param offsets Output cumulative offsets.
 * @return A public constraint status.
 */
constraint_status_t constraint_kform_component_offsets(const constraint_kform_spec_t *const spec,
                                                       const size_t offset_count,
                                                       size_t offsets[const static offset_count])
{
    size_t component_count;
    constraint_status_t status = constraint_kform_component_count(spec, &component_count);
    if (status != CONSTRAINT_SUCCESS)
        return status;
    if (offset_count < component_count + 1)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

    size_t offset = 0;
    offsets[0] = 0;
    for (size_t component = 0; component < component_count; ++component)
    {
        size_t dof_count;
        status = constraint_kform_component_dof_count(spec, (unsigned)component, &dof_count);
        if (status != CONSTRAINT_SUCCESS)
            return status;
        if (offset > SIZE_MAX - dof_count)
            return CONSTRAINT_SIZE_OVERFLOW;
        offset += dof_count;
        offsets[component + 1] = offset;
    }
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Derive per-component test-space basis specifications on a boundary.
 *
 * For every canonical boundary axis the returned order is the lowest order
 * found among the incident elements (mapped through their orientation
 * records), so shared objects are never overconstrained by higher-order
 * neighbours. Each component then reduces the order by two on every axis that
 * does not carry one of its covector axes; a component is reported absent
 * when any reduced order would become negative. The basis family of an axis
 * is taken from the element achieving the per-axis minimum (ties keep the
 * lowest element index), unless `type_override` selects a single family.
 *
 * @param ndim Element dimension.
 * @param boundary_dim Boundary-object dimension, strictly below `ndim`.
 * @param order Form degree, at most `boundary_dim`.
 * @param element_count Number of incident elements, at least one.
 * @param element_bases Borrowed per-element array of axis specifications.
 * @param orientations Signed one-based orientation records; the fixed-axis
 *        prefix must increase in absolute value.
 * @param type_override Family forced onto every output axis, or
 *        `BASIS_INVALID` to derive families from the incident elements.
 * @param out_capacity Available component-major output slots.
 * @param out_specs Component-major axis specifications; absent components
 *        clamp negative orders to zero.
 * @param out_present Component availability flags.
 * @return A public constraint status.
 */
constraint_status_t constraint_boundary_test_specs(const unsigned ndim, const unsigned boundary_dim,
                                                   const unsigned order, const size_t element_count,
                                                   const basis_spec_t *const *const element_bases,
                                                   const int8_t *const orientations,
                                                   const basis_set_type_t type_override, const size_t out_capacity,
                                                   basis_spec_t out_specs[const static out_capacity],
                                                   bool out_present[const static out_capacity])
{
    if (!element_bases || !orientations || element_count == 0)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (ndim == 0 || ndim > UINT8_MAX || boundary_dim >= ndim)
        return CONSTRAINT_INVALID_DIMENSION;
    if (order > boundary_dim)
        return CONSTRAINT_INVALID_ORDER;
    if (type_override != BASIS_INVALID && !basis_set_type_is_valid(type_override))
        return CONSTRAINT_INVALID_ORDER;

    const unsigned fixed_count = ndim - boundary_dim;
    // Validate every incident basis family and orientation record before
    // deriving anything; records must be signed permutations whose fixed
    // normal-axis prefix increases in absolute value.
    for (size_t element = 0; element < element_count; ++element)
    {
        if (!element_bases[element])
            return CONSTRAINT_INVALID_ARGUMENT;
        bool used_axes[UINT8_MAX];
        for (unsigned idim = 0; idim < ndim; ++idim)
        {
            used_axes[idim] = false;
            if (!basis_set_type_is_valid(element_bases[element][idim].type))
                return CONSTRAINT_INVALID_ORDER;
        }
        for (unsigned idim = 0; idim < ndim; ++idim)
        {
            const int mapped_axis = orientations[element * ndim + idim];
            const unsigned axis = (unsigned)(mapped_axis < 0 ? -mapped_axis : mapped_axis);
            if (axis == 0 || axis > ndim || used_axes[axis - 1])
                return CONSTRAINT_INVALID_ARGUMENT;
            used_axes[axis - 1] = true;
        }
        for (unsigned idim = 1; idim < fixed_count; ++idim)
        {
            const int previous = orientations[element * ndim + idim - 1];
            const int current = orientations[element * ndim + idim];
            if ((current < 0 ? -current : current) <= (previous < 0 ? -previous : previous))
                return CONSTRAINT_INVALID_ARGUMENT;
        }
    }

    const uint8_t face_ndim = (uint8_t)boundary_dim;
    const uint8_t form_order = (uint8_t)order;
    const size_t component_count = combination_total_count(face_ndim, form_order);
    if (out_capacity < component_count)
        return CONSTRAINT_INSUFFICIENT_STORAGE;

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
        component_axes(face_ndim, form_order, (unsigned)component, axes);
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
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Compute the packed-row offset-array length.
 *
 * A CSR-like row representation stores one sentinel after the final row, hence
 * the required length is `row_count + 1`.
 *
 * @param row_count Number of rows.
 * @param out_count Receives the required length.
 * @return `CONSTRAINT_SUCCESS`, invalid output, or overflow.
 */
constraint_status_t constraint_rows_required_offset_count(const size_t row_count, size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (row_count == SIZE_MAX)
        return CONSTRAINT_SIZE_OVERFLOW;
    *out_count = row_count + 1;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Compute a rectangular packed-entry capacity.
 *
 * The multiplication is checked before it is performed so callers can size a
 * flat entry buffer without relying on wrapped arithmetic.
 *
 * @param row_count Number of rows.
 * @param entries_per_row Entries in each row.
 * @param out_count Receives the product.
 * @return `CONSTRAINT_SUCCESS`, invalid output, or overflow.
 */
constraint_status_t constraint_rows_required_entry_capacity(const size_t row_count, const size_t entries_per_row,
                                                            size_t *const out_count)
{
    if (!out_count)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (entries_per_row != 0 && row_count > SIZE_MAX / entries_per_row)
        return CONSTRAINT_SIZE_OVERFLOW;
    *out_count = row_count * entries_per_row;
    return CONSTRAINT_SUCCESS;
}

/**
 * @brief Validate a packed row-offset and entry view.
 *
 * Validation is intentionally structural: offsets must describe contiguous,
 * non-decreasing row intervals and entries may refer only to the two supported
 * sides. It does not inspect component or local-DoF bounds because those
 * require element specifications, which are not part of the view.
 *
 * @param view Packed row view to inspect.
 * @return `CONSTRAINT_SUCCESS` when structurally valid, otherwise
 *         `CONSTRAINT_INVALID_ARGUMENT`.
 */
constraint_status_t constraint_rows_validate(const constraint_rows_view_t view)
{
    if (view.row_count != 0 && !view.row_offsets)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (view.entry_count != 0 && !view.entries)
        return CONSTRAINT_INVALID_ARGUMENT;
    if (view.row_offsets && view.row_offsets[0] != 0)
        return CONSTRAINT_INVALID_ARGUMENT;

    for (size_t row = 0; row < view.row_count; ++row)
    {
        if (view.row_offsets[row] > view.row_offsets[row + 1])
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    if (view.row_offsets && view.row_offsets[view.row_count] != view.entry_count)
        return CONSTRAINT_INVALID_ARGUMENT;

    for (size_t entry = 0; entry < view.entry_count; ++entry)
    {
        if (view.entries[entry].side > 1)
            return CONSTRAINT_INVALID_ARGUMENT;
    }
    return CONSTRAINT_SUCCESS;
}
