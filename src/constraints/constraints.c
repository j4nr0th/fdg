/**
 * @file constraints.c
 * @brief Implementation of reference and physical trace constraints.
 *
 * Assembly uses one canonical face coordinate system. Element-side
 * orientations map that system to signed, one-based element axes; the helper
 * functions below keep the mapping and its alternating k-form sign in one
 * place. Routines are `void`: documented preconditions are guarded by debug
 * asserts only, and storage is sized with the `*_layout` functions.
 */

#include "constraints.h"

#include <limits.h>
#include <math.h>

#include "cutl/iterators/combination_iterator.h"

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

    // Get the component index based on the element's mapped axes
    *out_component = combination_get_index(side->ndim, order, mapped_axes);
    *out_sign = sign;
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

void boundary_common_space(unsigned ndim, unsigned nelem, unsigned bdim, const int8_t *orientation[static ndim],
                           const basis_spec_t *element_basis[static ndim],
                           const integration_spec_t *element_integration[static ndim],
                           basis_spec_t boundary_basis[bdim], integration_spec_t boundary_integration[bdim])
{
    // Assert preconditions for boundary common space.
    CUTL_ASSERT(bdim != 0, "0-D boundary common space is trivial, so do not use this.");
    CUTL_ASSERT(nelem > 1, "At least two elements are required for boundary common space.");
    CUTL_ASSERT(ndim > 1, "Space must be at least 2D.");
    CUTL_ASSERT(bdim < ndim, "Boundary dimension must be less than element space dimension.");
    // For the first element, we just transform the element basis and integration rules to the boundary space.
    integration_rules_to_boundary(ndim, element_integration[0], orientation[0], bdim, boundary_integration);
    basis_spec_to_boundary(ndim, element_basis[0], orientation[0], bdim, boundary_basis);
    // For future elements, we update the particular dimension, if the boundary order is lower or integration rule is
    // more accurate.
    for (unsigned ie = 1; ie < nelem; ++ie)
    {
        const basis_spec_t *elem_basis = element_basis[ie];
        const integration_spec_t *elem_integration = element_integration[ie];
        const int8_t *elem_varying = orientation[ie] + (ndim - bdim);
        for (unsigned idim = 0; idim < bdim; ++idim)
        {
            const int8_t signed_axis = elem_varying[idim];
            const unsigned i_axis = signed_axis < 0 ? -signed_axis - 1 : signed_axis - 1;
            if (boundary_basis[idim].order > elem_basis[i_axis].order)
            {
                boundary_basis[idim] = elem_basis[i_axis];
            }
            if (integration_spec_accuracy(boundary_integration + idim) <
                integration_spec_accuracy(elem_integration + i_axis))
            {
                boundary_integration[idim] = elem_integration[i_axis];
            }
        }
    }

    // Finally, force the basis set to use Legendre basis
    for (unsigned idim = 0; idim < bdim; ++idim)
    {
        boundary_basis[idim].type = BASIS_LEGENDRE;
    }
}

void constrain_kform_components(unsigned ndim, const int8_t orientation[static ndim],
                                const basis_set_t *element_basis[static ndim], unsigned bdim,
                                const basis_set_t *boundary_basis[static bdim],
                                const integration_rule_t boundary_integration[static bdim], unsigned k,
                                combination_iterator_t *iter)
{
    // TODO: factor these VLAs out as user-supplied work arrays.
    const basis_set_t *current_basis[bdim];
    uint8_t element_axes[bdim];

    // Loop over local k-form components
    combination_iterator_init(iter, bdim, k);
    for (const uint8_t *comb = combination_iterator_current(iter); !combination_iterator_is_done(iter);
         combination_iterator_next(iter))
    {
        // Translate the boundary component basis into the element's basis
        bool sign = true;
        for (unsigned i = 0; i < bdim; ++i)
        {
            const int8_t signed_axis = comb[i];
            if (signed_axis < 0)
            {
                sign = !sign;
                element_axes[i] = -signed_axis - 1;
            }
            else
            {
                element_axes[i] = signed_axis - 1;
            }
        }
        // Bubble sort the axes and correct sign (TODO: factor, I think we re-use this other places as well)
        for (unsigned i = 0; i < bdim - 1; ++i)
        {
            for (unsigned j = 0; j < bdim - i - 1; ++j)
            {
                if (element_axes[j] > element_axes[j + 1])
                {
                    uint8_t temp = element_axes[j];
                    element_axes[j] = element_axes[j + 1];
                    element_axes[j + 1] = temp;
                    sign = !sign;
                }
            }
        }

        // Get the component index
        const size_t element_component = combination_get_index(bdim, k, element_axes);
        (void)element_component;
        // Get the element's boundary space
        basis_set_to_boundary(ndim, element_basis, orientation, bdim, current_basis);
        // TODO: compute the boundary mass matrix based on these parameters + boundary transform.
        //
        // (ndim, k, boundary_basis, boundary_integration, current_basis, element_axes)
        (void)boundary_basis;
        (void)boundary_integration;
    }
}

void constraint_reference_layout(const kform_spec_t *const test_spec, const constraint_element_side_t sides[static 2],
                                 size_t *const out_row_count, size_t *const out_entry_count)
{
    ASSERT(test_spec->order <= test_spec->ndim, "Test k-form order exceeds the face dimension.");
    const size_t component_count = kform_spec_component_count(test_spec);
    size_t row_count = 0;
    size_t entry_count = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        const size_t test_dof_count = kform_spec_component_dof_count(test_spec, test_component);
        row_count += test_dof_count;
        uint8_t test_axes[UINT8_MAX];
        kform_component_axes(test_spec, test_component, test_axes);
        size_t entries_per_row = 0;
        for (unsigned side = 0; side < 2; ++side)
        {
            entries_per_row +=
                side_entries_per_test_component(sides + side, test_spec->ndim, test_spec->order, test_axes);
        }
        entry_count += test_dof_count * entries_per_row;
    }
    *out_row_count = row_count;
    *out_entry_count = entry_count;
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

static void physical_two_sided_layout_impl(const kform_spec_t *const test_spec,
                                           const constraint_element_side_t sides[static 2], size_t *const out_row_count,
                                           size_t *const out_entry_count)
{
    // Rows are shared by both sides: side 0's blocks then side 1's close each
    // row, so the row count equals the one-sided count.
    size_t first_rows;
    size_t first_entries;
    physical_side_layout_impl(test_spec, sides, &first_rows, &first_entries);
    size_t second_rows;
    size_t second_entries;
    physical_side_layout_impl(test_spec, sides + 1, &second_rows, &second_entries);
    ASSERT(first_rows == second_rows, "The two sides have different test-space row counts.");
    *out_row_count = first_rows;
    *out_entry_count = first_entries + second_entries;
}

void constraint_physical_batch_layout(const kform_spec_t *const test_spec, const size_t item_count,
                                      const constraint_physical_batch_item_t items[const static item_count],
                                      size_t *const out_row_count, size_t *const out_entry_count)
{
    size_t row_count = 0;
    size_t entry_count = 0;
    for (size_t item = 0; item < item_count; ++item)
    {
        size_t item_rows;
        size_t item_entries;
        physical_two_sided_layout_impl(test_spec, items[item].sides, &item_rows, &item_entries);
        row_count += item_rows;
        entry_count += item_entries;
    }
    *out_row_count = row_count;
    *out_entry_count = entry_count;
}

void constraint_reference_rule_specs(const kform_spec_t *const test_spec,
                                     const constraint_element_side_t sides[const static 2],
                                     integration_spec_t out_specs[const static test_spec->ndim])
{
    const unsigned face_dim = test_spec->ndim;
    const unsigned fixed_count = sides[0].ndim - face_dim;
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const unsigned test_order = test_spec->basis[face_axis].order;
        unsigned max_element_order = 0;
        for (unsigned side = 0; side < 2; ++side)
        {
            const int8_t mapping = sides[side].orientation[fixed_count + face_axis];
            const unsigned axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
            const unsigned element_order = sides[side].basis_specs[axis].order;
            max_element_order = element_order > max_element_order ? element_order : max_element_order;
        }
        const unsigned accuracy = test_order + max_element_order + 1;
        out_specs[face_axis] =
            (integration_spec_t){.type = INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE, .order = accuracy / 2 + 1};
    }
}

void constraint_reference_assemble(const kform_spec_t *const test_spec,
                                   const constraint_element_side_t sides[const static 2],
                                   const double *const point_weights, const kform_values_table_t *const test_table,
                                   const kform_values_table_t *element_tables[const static 2], uint8_t out_sides[],
                                   uint32_t out_components[], size_t out_local_dofs[], double out_coefficients[],
                                   size_t out_row_offsets[])
{
    const unsigned face_dim = test_spec->ndim;
    const unsigned order = test_spec->order;
    const size_t component_count = kform_spec_component_count(test_spec);
    const size_t point_count = test_table->point_count;

    size_t row = 0;
    size_t entry = 0;
    out_row_offsets[0] = 0;
    // The packed-row contract is unchanged: component-major rows, side-ordered
    // entry blocks, one offset per row. Each component's coefficients form one
    // dense test-dofs × row-entries block, accumulated side block by side
    // block; the row entries never mix between components.
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        uint8_t test_axes[UINT8_MAX];
        kform_component_axes(test_spec, test_component, test_axes);
        const size_t test_dof_count =
            test_table->component_offsets[test_component + 1] - test_table->component_offsets[test_component];
        const size_t test_block_start = test_table->component_offsets[test_component] * point_count;
        unsigned element_components[2] = {0, 0};
        int orientation_signs[2] = {1, 1};
        size_t side_entries[2] = {0, 0};
        size_t row_entries = 0;
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            const constraint_element_side_t *const side = sides + side_index;
            mapped_component(side, face_dim, order, test_axes, element_components + side_index,
                             orientation_signs + side_index);
            const kform_spec_t element_spec = {.ndim = side->ndim, .order = order, .basis = side->basis_specs};
            side_entries[side_index] = kform_spec_component_dof_count(&element_spec, element_components[side_index]);
            row_entries += side_entries[side_index];
        }
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            const kform_values_table_t *const element_table = element_tables[side_index];
            const size_t element_block_start =
                element_table->component_offsets[element_components[side_index]] * point_count;
            const double side_sign = side_index == 0 ? 1.0 : -1.0;
            trace_component_block(point_count, test_dof_count, side_entries[side_index],
                                  test_table->values + test_block_start, element_table->values + element_block_start,
                                  point_weights, NULL, side_sign * (double)orientation_signs[side_index], NULL, 0, 0,
                                  side_index == 0 ? 0 : side_entries[0], row_entries, out_coefficients + entry);
        }
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof)
        {
            out_row_offsets[row + test_dof + 1] = entry + (test_dof + 1) * row_entries;
            size_t entry_index = entry + test_dof * row_entries;
            for (unsigned side_index = 0; side_index < 2; ++side_index)
            {
                for (size_t element_dof = 0; element_dof < side_entries[side_index]; ++element_dof, ++entry_index)
                {
                    out_sides[entry_index] = (uint8_t)side_index;
                    out_components[entry_index] = (uint32_t)element_components[side_index];
                    out_local_dofs[entry_index] = element_dof;
                }
            }
        }
        row += test_dof_count;
        entry += test_dof_count * row_entries;
    }
}

bool constraint_reference_links_eligible(const kform_spec_t *const test_spec,
                                         const constraint_element_side_t sides[const static 2])
{
    const unsigned face_dim = test_spec->ndim;
    const unsigned fixed_count = sides[0].ndim - face_dim;
    // Free slots: both sides must map to the same family and order, and the
    // test order must match so each component's Gram block is square. A slot
    // whose orientation sign differs between the sides is mirrored, which
    // flips its node index; only Lagrange families on symmetric node sets
    // turn that flip into a pure DoF permutation.
    for (unsigned face_axis = 0; face_axis < face_dim; ++face_axis)
    {
        const int8_t mapping_0 = sides[0].orientation[fixed_count + face_axis];
        const int8_t mapping_1 = sides[1].orientation[fixed_count + face_axis];
        const basis_spec_t *const spec_0 = &sides[0].basis_specs[(mapping_0 < 0 ? -mapping_0 : mapping_0) - 1];
        const basis_spec_t *const spec_1 = &sides[1].basis_specs[(mapping_1 < 0 ? -mapping_1 : mapping_1) - 1];
        if (spec_0->type != spec_1->type || spec_0->order != spec_1->order ||
            test_spec->basis[face_axis].order != spec_0->order)
        {
            return false;
        }
        const bool mirrored = (mapping_0 < 0) != (mapping_1 < 0);
        if (mirrored)
        {
            const basis_set_type_t test_type = test_spec->basis[face_axis].type;
            const bool test_nodal = test_type == BASIS_LAGRANGE_GAUSS || test_type == BASIS_LAGRANGE_GAUSS_LOBATTO ||
                                    test_type == BASIS_LAGRANGE_UNIFORM || test_type == BASIS_LAGRANGE_CHEBYSHEV_GAUSS;
            const bool element_nodal =
                spec_0->type == BASIS_LAGRANGE_GAUSS || spec_0->type == BASIS_LAGRANGE_GAUSS_LOBATTO ||
                spec_0->type == BASIS_LAGRANGE_UNIFORM || spec_0->type == BASIS_LAGRANGE_CHEBYSHEV_GAUSS;
            if (!test_nodal || !element_nodal)
                return false;
        }
    }
    // Fixed slots: a single DoF may carry the endpoint value on both sides.
    for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
    {
        const int8_t mapping_0 = sides[0].orientation[fixed_axis];
        const int8_t mapping_1 = sides[1].orientation[fixed_axis];
        const basis_set_type_t type_0 = sides[0].basis_specs[(mapping_0 < 0 ? -mapping_0 : mapping_0) - 1].type;
        const basis_set_type_t type_1 = sides[1].basis_specs[(mapping_1 < 0 ? -mapping_1 : mapping_1) - 1].type;
        const bool supported_0 = type_0 == BASIS_LAGRANGE_GAUSS_LOBATTO || type_0 == BASIS_BERNSTEIN;
        const bool supported_1 = type_1 == BASIS_LAGRANGE_GAUSS_LOBATTO || type_1 == BASIS_BERNSTEIN;
        if (!supported_0 || !supported_1)
        {
            return false;
        }
    }
    return true;
}

void constraint_reference_links_layout(const kform_spec_t *const test_spec,
                                       const constraint_element_side_t sides[const static 2],
                                       size_t *const out_row_count, size_t *const out_entry_count)
{
    (void)sides;
    const size_t row_count = kform_spec_total_dofs(test_spec);
    *out_row_count = row_count;
    *out_entry_count = 2 * row_count;
}

void constraint_reference_links_reduce(const kform_spec_t *const test_spec,
                                       const constraint_element_side_t sides[const static 2],
                                       const size_t dense_row_count, const uint8_t dense_sides[const static 1],
                                       const uint32_t dense_components[const static 1],
                                       const size_t dense_local_dofs[const static 1],
                                       const double dense_coefficients[const static 1],
                                       const size_t dense_row_offsets[const static 1], uint8_t out_sides[],
                                       uint32_t out_components[], size_t out_local_dofs[], double out_coefficients[],
                                       size_t out_row_offsets[])
{
    const unsigned face_dim = test_spec->ndim;
    const unsigned order = test_spec->order;
    const size_t component_count = kform_spec_component_count(test_spec);
    const unsigned fixed_count = sides[0].ndim - face_dim;
    size_t test_offsets[UINT8_MAX + 1];
    kform_spec_component_offsets(test_spec, component_count + 1, test_offsets);

    size_t row = 0;
    size_t entry = 0;
    out_row_offsets[0] = 0;
    for (unsigned test_component = 0; test_component < component_count; ++test_component)
    {
        uint8_t test_axes[UINT8_MAX];
        kform_component_axes(test_spec, test_component, test_axes);
        // Per side, the element DoF counts and strides of THIS component:
        // active mapped axes carry the lowered count, every other axis the
        // full count.
        size_t counts[2][UINT8_MAX];
        size_t side_strides[2][UINT8_MAX];
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            const constraint_element_side_t *const side = sides + side_index;
            for (unsigned axis = 0; axis < side->ndim; ++axis)
                counts[side_index][axis] = side->basis_specs[axis].order + 1u;
            for (unsigned pos = 0; pos < order; ++pos)
            {
                const int8_t mapping = side->orientation[fixed_count + test_axes[pos]];
                const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                counts[side_index][element_axis] = side->basis_specs[element_axis].order;
            }
            size_t stride = 1;
            for (unsigned stride_axis = side->ndim; stride_axis-- > 0;)
            {
                side_strides[side_index][stride_axis] = stride;
                stride *= counts[side_index][stride_axis];
            }
        }
        unsigned element_components[2] = {0, 0};
        int orientation_signs[2] = {1, 1};
        for (unsigned side_index = 0; side_index < 2; ++side_index)
        {
            mapped_component(sides + side_index, face_dim, order, test_axes, &element_components[side_index],
                             &orientation_signs[side_index]);
        }
        const size_t test_dof_count = test_offsets[test_component + 1] - test_offsets[test_component];
        for (size_t test_dof = 0; test_dof < test_dof_count; ++test_dof, ++row)
        {
            ASSERT(row < dense_row_count, "Dense rows do not cover the test space.");
            // Decode the canonical digits along every face axis: axis 0 is
            // the slowest digit, the last axis the fastest. Active covector
            // axes carry one fewer DoF than inactive axes.
            uint8_t digits[UINT8_MAX];
            size_t remainder = test_dof;
            for (unsigned axis = face_dim; axis-- > 0;)
            {
                const unsigned axis_order = test_spec->basis[axis].order;
                const unsigned axis_dofs = axis_order + (component_has_axis(order, test_axes, axis) ? 0u : 1u);
                digits[axis] = (uint8_t)(remainder % axis_dofs);
                remainder /= axis_dofs;
            }
            // The element DoF carrying the canonical trace function on each
            // side: the table mirror flags and the mapped-component signs
            // already canonicalize every side, so the digits carry over
            // verbatim, and the normal digits sit at the endpoint-supported
            // DoF.
            size_t labels[2] = {0, 0};
            for (unsigned side_index = 0; side_index < 2; ++side_index)
            {
                const constraint_element_side_t *const side = sides + side_index;
                for (unsigned axis = 0; axis < face_dim; ++axis)
                {
                    const int8_t mapping = side->orientation[fixed_count + axis];
                    const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                    const unsigned axis_count = counts[side_index][element_axis];
                    const unsigned digit =
                        mapping < 0 ? axis_count - 1u - (unsigned)digits[axis] : (unsigned)digits[axis];
                    labels[side_index] += digit * side_strides[side_index][element_axis];
                }
                for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
                {
                    const int8_t mapping = side->orientation[fixed_axis];
                    const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                    const unsigned support = mapping < 0 ? 0u : side->basis_specs[element_axis].order;
                    labels[side_index] += support * side_strides[side_index][element_axis];
                }
            }
            // The ratio of the dense moment rows is read at the LARGEST
            // side-0 entry: individual matched columns can vanish structurally
            // (parity of the paired basis functions), so they must never
            // define it. Its canonical digits are recovered from the element-0
            // layout and re-encoded in the element-1 frame to locate the
            // matched side-1 column.
            double largest = 0.0;
            double largest_magnitude = 0.0;
            size_t largest_dof = 0;
            for (size_t dense_entry = dense_row_offsets[row]; dense_entry < dense_row_offsets[row + 1]; ++dense_entry)
            {
                if (dense_sides[dense_entry] != 0)
                    continue;
                const double magnitude = fabs(dense_coefficients[dense_entry]);
                if (magnitude > largest_magnitude)
                {
                    largest_magnitude = magnitude;
                    largest = dense_coefficients[dense_entry];
                    largest_dof = dense_local_dofs[dense_entry];
                }
            }
            size_t canonical_digits[UINT8_MAX];
            remainder = largest_dof;
            for (unsigned axis = face_dim; axis-- > 0;)
            {
                const int8_t mapping = sides[0].orientation[fixed_count + axis];
                const unsigned source_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                const unsigned count = counts[0][source_axis];
                const unsigned digit = (unsigned)(remainder % count);
                canonical_digits[axis] = mapping < 0 ? count - 1u - digit : digit;
                remainder /= count;
            }
            size_t partner = 0;
            for (unsigned axis = 0; axis < face_dim; ++axis)
            {
                const int8_t mapping = sides[1].orientation[fixed_count + axis];
                const unsigned other_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                const unsigned count = counts[1][other_axis];
                const unsigned digit = mapping < 0 ? count - 1u - canonical_digits[axis] : canonical_digits[axis];
                partner += digit * side_strides[1][other_axis];
            }
            for (unsigned fixed_axis = 0; fixed_axis < fixed_count; ++fixed_axis)
            {
                const int8_t mapping = sides[1].orientation[fixed_axis];
                const unsigned element_axis = (unsigned)(mapping < 0 ? -mapping : mapping) - 1;
                const unsigned support = mapping < 0 ? 0u : sides[1].basis_specs[element_axis].order;
                partner += support * side_strides[1][element_axis];
            }
            double row_ratio = 0.0;
            size_t seen = 0;
            for (size_t dense_entry = dense_row_offsets[row]; dense_entry < dense_row_offsets[row + 1]; ++dense_entry)
            {
                if (dense_sides[dense_entry] != 1 || dense_components[dense_entry] != element_components[1] ||
                    dense_local_dofs[dense_entry] != partner)
                {
                    continue;
                }
                row_ratio = dense_coefficients[dense_entry] / largest;
                ++seen;
            }
            ASSERT(seen == 1, "A dense trace row is missing a matched element DoF.");
            out_sides[entry] = 0;
            out_components[entry] = (uint32_t)element_components[0];
            out_local_dofs[entry] = labels[0];
            out_coefficients[entry] = 1.0;
            ++entry;
            out_sides[entry] = 1;
            out_components[entry] = (uint32_t)element_components[1];
            out_local_dofs[entry] = labels[1];
            out_coefficients[entry] = row_ratio;
            ++entry;
            out_row_offsets[row + 1] = entry;
        }
    }
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
            const double side_factor = side_sign * (double)test_orientation_signs[side_index];
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
                    side_factor * (double)orientation_sign, pullback, test_element_components[side_index],
                    element_component, column, row_entries, out_coefficients + entry);
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

void constraint_physical_assemble(const kform_spec_t *const test_spec,
                                  const constraint_element_side_t sides[const static 2],
                                  const constraint_assembly_inputs_t inputs[const static 2], uint8_t out_sides[],
                                  uint32_t out_components[], size_t out_local_dofs[], double out_coefficients[],
                                  size_t out_row_offsets[])
{
    physical_assemble_impl(test_spec, sides, inputs, 2, out_sides, out_components, out_local_dofs, out_coefficients,
                           out_row_offsets);
}

void constraint_physical_batch_assemble(const kform_spec_t *const test_spec, const size_t item_count,
                                        const constraint_physical_batch_item_t items[const static item_count],
                                        uint8_t out_sides[], uint32_t out_components[], size_t out_local_dofs[],
                                        double out_coefficients[], size_t out_row_offsets[])
{
    size_t row = 0;
    size_t entry = 0;
    out_row_offsets[0] = 0;
    // Each item is assembled into its own slice first; the entry prefix is
    // then added to that item's local row offsets before advancing the slices.
    for (size_t item = 0; item < item_count; ++item)
    {
        size_t item_rows;
        size_t item_entries;
        physical_two_sided_layout_impl(test_spec, items[item].sides, &item_rows, &item_entries);
        constraint_physical_assemble(test_spec, items[item].sides, items[item].inputs, out_sides + entry,
                                     out_components + entry, out_local_dofs + entry, out_coefficients + entry,
                                     out_row_offsets + row);
        out_row_offsets[row] = entry;
        for (size_t local_row = 1; local_row <= item_rows; ++local_row)
        {
            out_row_offsets[row + local_row] += entry;
        }
        row += item_rows;
        entry += item_entries;
    }
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
    unsigned axis_source_slots[UINT8_MAX];
    unsigned axis_orders[UINT8_MAX];
    size_t axis_source_strides[UINT8_MAX];
    size_t axis_canonical_strides[UINT8_MAX];
    int axis_mirrored[UINT8_MAX];
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
    }
    for (unsigned face_component = 0; face_component < face_component_count; ++face_component)
    {
        uint8_t face_axes[UINT8_MAX];
        kform_component_axes(&face_spec, face_component, face_axes);
        unsigned element_component;
        int orientation_sign;
        mapped_component(&side, request->face_dim, request->order, face_axes, &element_component, &orientation_sign);
        (void)orientation_sign;
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
                const size_t source_index = ((size_t)face_component * physical_component_count + physical_component) *
                                                request->source_point_count +
                                            source_point;
                const size_t target_index =
                    ((size_t)element_component * physical_component_count + physical_component) * point_count +
                    canonical_point;
                request->out[target_index] = request->transform[source_index];
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
