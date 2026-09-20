/**
 * @file constraints.h
 * @brief Reference and physical trace constraints for continuous k-forms.
 *
 * Assembly uses one canonical face coordinate system. Element-side
 * orientations map that system to signed, one-based element axes; the face
 * helpers in this header keep that mapping and its alternating k-form sign
 * in one place. All routines are plain C: they return void, their
 * preconditions are documented here and checked only by debug asserts, and
 * every sizing question is answered by the layout functions. Assemblers
 * write flat parallel output arrays in the packed-row contract described on
 * each function.
 *
 * Dimensions and component indices use the canonical axis order expected by
 * the combination iterator. Basis functions are evaluated on the canonical
 * face coordinates, and every routine trusts the documented preconditions
 * outside debug builds.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../basis/basis_set.h"
#include "../kforms/kform_types.h"
#include "../kforms/kform_values.h"

/**
 * @brief Specification of one higher-dimensional element side.
 *
 * `orientation` is a signed one-based permutation of the element axes: entry
 * `i` maps canonical axis `i` of the side description to element axis
 * `|orientation[i]| - 1`, with a negative sign reversing that axis. The fixed
 * normal axes occupy the first entries and their absolute values increase, so
 * the endpoint prefix has a deterministic orientation convention.
 */
typedef struct
{
    unsigned ndim;                   ///< Element dimension.
    const basis_spec_t *basis_specs; ///< Element-axis basis specifications.
    const int8_t *orientation;       ///< Signed one-based axis mapping.
} constraint_element_side_t;

/**
 * @brief One incident element's view of a shared boundary object.
 *
 * `basis` holds one specification per element axis. For k-form continuity it
 * is the k-form spec's per-axis basis array; the boundary merge only reads
 * per-axis orders and families, so both scalar and k-form trace spaces pass
 * through the same request.
 */
typedef struct
{
    unsigned order;                        ///< Traced k-form order; unused by the space merge.
    const int8_t *orientation;             ///< [ndim] Signed one-based axis mapping.
    const basis_spec_t *basis;             ///< [ndim] Element-axis basis specifications.
    const integration_spec_t *integration; ///< [ndim] Element-axis integration rules.
} boundary_element_space_t;

/**
 * @brief Inputs describing the common boundary space of one shared object.
 */
typedef struct
{
    unsigned ndim;                            ///< Element dimension, at least 2.
    unsigned bdim;                            ///< Boundary dimension, in `[1, ndim)`.
    unsigned nelem;                           ///< Incident element count, at least 2.
    const boundary_element_space_t *elements; ///< [nelem] Per-element views.
} boundary_common_space_request_t;

/**
 * @brief Determine the common boundary space for a set of elements.
 *
 * Based on information about the boundary in all the elements it is contained in
 * a common basis and integration spaces are determined. These specify the basis
 * orders such that all the solutions can be fully resolved. The integration rules
 * are instead of the highest accuracy, to allow integration without any precision
 * loss compared to all elements.
 * For the sake of consistency, the resulting boundary space is always set to use
 * Legendre basis for all dimensions. This is for the reason, that when constraints
 * are assembled with C1 continuous space maps, the resulting constraints are
 * very sparse.
 *
 * Every element view must satisfy the #boundary_element_space_t array lengths
 * against the request's `ndim`, and its orientation record must be a signed
 * one-based permutation whose fixed-axis prefix increases in absolute value.
 *
 * @param request Filled request; read-only.
 * @param out_basis [bdim] Boundary-axis basis specifications.
 * @param out_integration [bdim] Boundary-axis integration rules.
 */
void boundary_common_space(const boundary_common_space_request_t *request, basis_spec_t *out_basis,
                           integration_spec_t *out_integration);

/**
 * @brief Sampled tangential pullback of a physical k-form on a face.
 *
 * `values` holds `element_component_count * physical_component_count *
 * point_count` entries, element component slowest and point fastest.
 */
typedef struct
{
    unsigned physical_component_count; ///< Physical k-form components per sample.
    size_t point_count;                ///< Sampled canonical face points.
    const double *values;              ///< Sampled pullback values.
} constraint_trace_pullback_t;

/**
 * @brief Precomputed inputs for one side of a physical trace assembly.
 *
 * The test and element tables are built on the same canonical face points,
 * and `point_weights` carries the canonical tensor-product quadrature weight
 * of each point.
 */
typedef struct
{
    const double *point_weights;                 ///< [point_count] canonical quadrature weights.
    const double *surface_weights;               ///< [point_count] optional face measure, NULL = unweighted.
    const kform_values_table_t *test_table;      ///< Test trace basis values.
    const kform_values_table_t *element_table;   ///< Element trace basis values.
    const constraint_trace_pullback_t *pullback; ///< Required when order > 0, else NULL.
} constraint_assembly_inputs_t;

/**
 * @brief One two-sided physical trace assembly item in a batch.
 */
typedef struct
{
    const constraint_element_side_t *sides;       ///< Two element-side specifications.
    const constraint_assembly_inputs_t inputs[2]; ///< Per-side assembly inputs.
} constraint_physical_batch_item_t;

/**
 * @brief Parameters for building a sampled trace pullback.
 *
 * The transform is the `compute_basis_transform_impl` output of the mapped
 * face: `face_component_count * physical_component_count * source_point_count`
 * values, element component slowest and source point fastest.
 */
typedef struct
{
    unsigned element_dim;                      ///< Element dimension.
    unsigned face_dim;                         ///< Canonical face dimension.
    unsigned order;                            ///< Traced k-form order.
    unsigned face_component_count;             ///< C(face_dim, order).
    unsigned physical_component_count;         ///< C(coordinates, order) of the mapped face.
    size_t source_point_count;                 ///< Points of the source-frame face tensor.
    size_t canonical_point_count;              ///< Points of the canonical face tensor.
    const size_t *source_strides;              ///< [face_dim] row-major source-frame strides.
    const size_t *canonical_strides;           ///< [face_dim] row-major canonical strides.
    const int8_t *orientation;                 ///< Signed one-based element-axis mapping.
    const integration_spec_t *source_specs;    ///< [face_dim] source-frame axis specs.
    const integration_spec_t *canonical_specs; ///< [face_dim] canonical axis specs.
    const double *transform;                   ///< Sampled face transform (see above).
    double *out;                               ///< [element_comp * physical_comp * canonical_point_count].
} constraint_trace_pullback_build_t;

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
 * @param ndim Element dimension, in `[1, UINT8_MAX]`.
 * @param boundary_dim Boundary-object dimension, strictly below `ndim`.
 * @param order Form degree, at most `boundary_dim`.
 * @param element_count Number of incident elements, at least one.
 * @param element_bases Per-element array of axis specifications.
 * @param orientations Signed one-based orientation records; the fixed-axis
 *        prefix must increase in absolute value.
 * @param type_override Family forced onto every output axis, or
 *        `BASIS_INVALID` to derive families from the incident elements.
 * @param out_specs Component-major axis specifications with
 *        `C(boundary_dim, order) * boundary_dim` entries; absent components
 *        clamp negative orders to zero.
 * @param out_present Component availability flags with
 *        `C(boundary_dim, order)` entries.
 */
void constraint_boundary_test_specs(unsigned ndim, unsigned boundary_dim, unsigned order, size_t element_count,
                                    const basis_spec_t *const *element_bases, const int8_t *orientations,
                                    basis_set_type_t type_override, basis_spec_t out_specs[], bool out_present[]);

/**
 * @brief Shape of one element's boundary mass matrix.
 *
 * The matrix maps the element's face trace DoFs to the common boundary test
 * space: row blocks follow the common Legendre k-form components, column
 * blocks the mapped element components in canonical boundary order. Rows are
 * component-local tensors whose per-axis function counts are
 *
 * - active covector axis: the order-1 basis (`order` functions),
 * - inactive axis: the full basis with the first `axis_skip[axis]` functions
 *   removed (`order + 1 - axis_skip[axis]` functions).
 *
 * The skip replaces the old order-minus-two test-space reduction: lower
 * dimensional boundary objects already enforce continuity there, and skipping
 * the lowest Legendre degrees keeps the enforced test functions high order.
 */
typedef struct
{
    unsigned ndim;                                  ///< Element dimension.
    unsigned bdim;                                  ///< Boundary dimension, in `[1, ndim)`.
    unsigned order;                                 ///< Traced k-form order, at most `bdim`.
    const kform_spec_t *element_spec;               ///< Element k-form spec; `order` must match.
    const basis_spec_t *boundary_basis;             ///< [bdim] Common boundary Legendre basis.
    const integration_spec_t *boundary_integration; ///< [bdim] Common boundary rules.
    const int8_t *orientation;                      ///< [ndim] This element's signed axis mapping.
    const uint8_t *axis_skip;                       ///< [bdim] Skipped test functions on inactive axes, NULL = none.
} constraint_boundary_mass_spec_t;

/**
 * @brief Allocation sizes of one element's boundary mass work buffers.
 *
 * Every field counts doubles or pointers; the caller allocates each array and
 * records the pointers in #constraint_boundary_mass_work_t.
 */
typedef struct
{
    size_t row_values;      ///< Test component value tables, in doubles.
    size_t col_values;      ///< Element component value tables, in doubles.
    size_t point_factors;   ///< Per-point factor buffer, in doubles.
    size_t component_count; ///< Common boundary components, `C(bdim, order)`.
} constraint_boundary_mass_work_sizes_t;

/**
 * @brief Compute the work buffer sizes of one element's boundary mass matrix.
 *
 * @param spec Filled matrix specification.
 * @param out_sizes Receives the allocation sizes.
 */
void constraint_boundary_mass_work_size(const constraint_boundary_mass_spec_t *spec,
                                        constraint_boundary_mass_work_sizes_t *out_sizes);

/**
 * @brief Caller-provided scratch memory of one element's boundary mass matrix.
 *
 * Every member is a caller-allocated array whose length is fixed by the
 * specification: `bdim` entries for the per-axis arrays, `ndim` for the axis
 * descriptors, and `component_count + 1` (from
 * #constraint_boundary_mass_work_sizes_t) for the component tables. The
 * assemble, layout, and pack routines use the arrays as scratch and may
 * overwrite their contents.
 */
typedef struct
{
    size_t *point_strides;              ///< [bdim] Point strides of the boundary rules.
    size_t *row_offsets;                ///< [component_count + 1] Test component offsets.
    size_t *col_offsets;                ///< [component_count + 1] Mapped element offsets.
    unsigned *element_components;       ///< [component_count] Mapped element components.
    int *element_signs;                 ///< [component_count] Mapped covector signs.
    kform_trace_axis_t *axes;           ///< [ndim] Element axis trace descriptors.
    unsigned *counts;                   ///< [bdim] Per-axis test function counts.
    unsigned *offsets;                  ///< [bdim] Per-axis test function offsets.
    const basis_set_t **axis_sets;      ///< [bdim] Per-axis selected basis tables.
    unsigned *digits;                   ///< [bdim] Component-local DoF counters.
    const double **axis_tables;         ///< [bdim] Per-axis basis value tables.
    uint8_t *mapped_axes;               ///< [order] Mapped element axes of the current component.
    combination_iterator_t *components; ///< `combination_iterator_required_memory(order)`.
    combination_iterator_t *blocks;     ///< `combination_iterator_required_memory(order)`; physical pairing only.
    double *row_values;                 ///< [row_values] Test component value tables.
    double *col_values;                 ///< [col_values] Element component value tables.
    double *point_factors;              ///< [point_factors] Per-point factor buffer.
} constraint_boundary_mass_work_t;

/**
 * @brief Inputs and outputs of one element's boundary mass matrix assembly.
 */
typedef struct
{
    const constraint_boundary_mass_spec_t *spec;
    const basis_set_t *const *boundary_basis_sets;        ///< [bdim] Boundary basis at the common rules.
    const basis_set_t *const *boundary_basis_sets_lower;  ///< [bdim] Order-1 sets, NULL when order == 0.
    const basis_set_t *const *element_basis_sets;         ///< [ndim] Element sets at the common rules; NULL at
                                                          ///< fixed normal axes.
    const basis_set_t *const *element_basis_sets_lower;   ///< [ndim] Order-1 sets, NULL when order == 0.
    const basis_endpoint_set_t *const *element_endpoints; ///< [ndim] Endpoint values at fixed normal axes; NULL at
                                                          ///< free axes.
    const basis_endpoint_set_t *const *element_endpoints_lower; ///< [ndim] Order-1 endpoints or NULL.
    const double *point_weights;                                ///< [point_count] Common tensor quadrature weights.
    const double *surface_weights;                       ///< [point_count] Optional face measure, NULL = reference.
    const constraint_trace_pullback_t *test_pullback;    ///< Optional test physical pullback; both or none.
    const constraint_trace_pullback_t *element_pullback; ///< Optional element physical pullback; both or none.
    double factor;                                       ///< Extra scalar, e.g. the constraint side sign.
    constraint_boundary_mass_work_t *work;               ///< Caller-provided scratch buffers.
    double *out_matrix;                                  ///< [row_count * col_count] Dense row-major matrix.
} constraint_boundary_mass_request_t;

/**
 * @brief Compute the dense shape of one element's boundary mass matrix.
 *
 * @param spec Filled matrix specification.
 * @param work Caller-provided scratch; the per-axis, component-table, and
 *        iterator members are used and overwritten.
 * @param physical Non-zero when physical pullback factors will couple every
 *        mapped component pair; reference pairing is component-diagonal.
 * @param out_row_count Receives the common test DoF count.
 * @param out_col_count Receives the mapped element trace DoF count.
 * @param out_entry_count Receives the packed entry count of the COO form.
 */
void constraint_boundary_mass_layout(const constraint_boundary_mass_spec_t *spec, constraint_boundary_mass_work_t *work,
                                     bool physical, size_t *out_row_count, size_t *out_col_count,
                                     size_t *out_entry_count);

/**
 * @brief Assemble one element's boundary mass matrix.
 *
 * Coefficients carry the orientation sign of the mapped components and
 * #constraint_boundary_mass_request_t::factor, the quadrature weights, the
 * optional surface measure, and the optional physical pullback dot product.
 * With reference pairing (no pullbacks) each test component couples only its
 * mapped element component; with pullbacks every mapped component pair
 * couples. The matrix is zero-initialized first.
 *
 * @param request Filled request; `out_matrix` written on return.
 */
void constraint_boundary_mass_assemble(const constraint_boundary_mass_request_t *request);

/**
 * @brief Pack a dense boundary mass matrix into the COO contract.
 *
 * Rows follow the packed-row contract: components with rows in canonical
 * order, then component-local test DoF. Reference pairing emits each row's
 * mapped component block; physical pairing emits every mapped component
 * block. Coefficients are read verbatim from the matrix and multiplied by
 * @p factor (e.g. the constraint side sign); `out_row_offsets` has
 * `row_count + 1` entries starting at zero.
 *
 * @param spec Specification the matrix was assembled with.
 * @param work Caller-provided scratch; the component-table and iterator
 *        members are used and overwritten.
 * @param physical Must match the assembly's pairing mode.
 * @param matrix Dense row-major matrix of `row_count * row_stride` entries.
 * @param row_stride Column stride of the matrix, at least `col_count`.
 * @param factor Scalar multiplying every coefficient, e.g. the side sign.
 * @param side Side label written to every entry of `out_sides`.
 * @param out_sides [entry_count] Side label of each entry.
 * @param out_components [entry_count] Element component of each entry.
 * @param out_local_dofs [entry_count] Component-local DoF of each entry.
 * @param out_coefficients [entry_count] Entry coefficients.
 * @param out_row_offsets [row_count + 1] Packed row offsets.
 */
void constraint_boundary_mass_pack(const constraint_boundary_mass_spec_t *spec, constraint_boundary_mass_work_t *work,
                                   bool physical, const double *matrix, size_t row_stride, double factor, uint8_t side,
                                   uint8_t out_sides[], uint32_t out_components[], size_t out_local_dofs[],
                                   double out_coefficients[], size_t out_row_offsets[]);

/**
 * @brief Inputs of the per-element boundary constraint mass matrices of one
 *        shared object.
 */
typedef struct
{
    unsigned ndim;                            ///< Element dimension, at least 2.
    unsigned bdim;                            ///< Boundary dimension, in `[1, ndim)`.
    unsigned nforms;                          ///< Traced k-form count, at least 1.
    unsigned nelem;                           ///< Incident element count, at least 2.
    const boundary_element_space_t *elements; ///< [nforms * nelem] Form-major element views.
    const uint8_t *axis_skip;                 ///< [nforms * bdim] Per-form skipped test functions, NULL = none.
    bool c1_continuous;                       ///< Reference-space pairing; pullback inputs may be NULL.
    const double *const *surface_weights;     ///< [nforms * nelem] Optional per-item face measure rows.
    const constraint_trace_pullback_t *const *test_pullbacks;    ///< [nforms * nelem] Optional per-item pullbacks.
    const constraint_trace_pullback_t *const *element_pullbacks; ///< [nforms * nelem] Optional per-item pullbacks.
    basis_set_registry_t *basis_registry;                        ///< Registry for basis and endpoint tables.
    integration_rule_registry_t *integration_registry;           ///< Registry for quadrature rules.
} constrain_elements_on_boundary_request_t;

/**
 * @brief Intermediates of one boundary constraint batch.
 *
 * Every member is a caller-allocated array whose length is fixed by the
 * request: `nforms * bdim` for the per-form boundary arrays, `nforms * nelem *
 * ndim` for the per-item element arrays, and `nforms * nelem` (`+ 1` for the
 * offsets) for the matrix layout. #constrain_elements_on_boundary_prepare
 * fills the arrays with the merged common spaces, registry references, and
 * matrix layout; #constrain_elements_on_boundary_assemble reads them; and
 * #constrain_elements_on_boundary_plan_release returns the registry
 * references. The borrowed request and output pointers must outlive the plan.
 */
typedef struct
{
    unsigned ndim;                                        ///< Element dimension, copied from the request.
    unsigned bdim;                                        ///< Boundary dimension, copied from the request.
    unsigned nforms;                                      ///< Traced k-form count, copied from the request.
    unsigned nelem;                                       ///< Incident element count, copied from the request.
    integration_rule_registry_t *integration_registry;    ///< Borrowed quadrature registry.
    basis_set_registry_t *basis_registry;                 ///< Borrowed basis registry.
    const integration_rule_t **rules;                     ///< [nforms * bdim] Common rules per form.
    const basis_set_t **boundary_sets;                    ///< [nforms * bdim] Boundary basis tables per form.
    const basis_set_t **boundary_sets_lower;              ///< [nforms * bdim] Order-1 tables, NULL when order == 0.
    basis_spec_t *boundary_lower_specs;                   ///< [nforms * bdim] Order-1 boundary specs.
    const basis_set_t **element_sets;                     ///< [nforms * nelem * ndim] Element tables per item.
    const basis_set_t **element_sets_lower;               ///< [nforms * nelem * ndim] Order-1 tables or NULL.
    const basis_endpoint_set_t **element_endpoints;       ///< [nforms * nelem * ndim] Fixed-axis endpoints or NULL.
    const basis_endpoint_set_t **element_endpoints_lower; ///< [nforms * nelem * ndim] Order-1 endpoints or NULL.
    basis_spec_t *element_lower_specs;                    ///< [nforms * nelem * ndim] Order-1 element specs.
    const basis_spec_t *boundary_basis;                   ///< Borrowed caller array of per-form common basis.
    const integration_spec_t *boundary_integration;       ///< Borrowed caller array of per-form common rules.
    size_t *item_rows;                                    ///< [nforms * nelem] Row count of each item's matrix.
    size_t *item_cols;                                    ///< [nforms * nelem] Column count of each item's matrix.
    size_t *item_offsets;                                 ///< [nforms * nelem + 1] Start of each item's matrix.
    size_t total_values;                                  ///< Total doubles the assembled values need.
} constrain_elements_on_boundary_plan_t;

/**
 * @brief Caller-provided scratch memory of one boundary constraint batch.
 *
 * Array lengths are fixed by the request (`ndim`, `bdim`), the maximum common
 * rule point count, and the maximum per-item value table sizes; the latter
 * two are reported by #constrain_elements_on_boundary_work_size. The
 * assembler overwrites the contents freely.
 */
typedef struct
{
    double *weights;                          ///< [max point count] Common tensor quadrature weights.
    bool *axis_fixed;                         ///< [ndim] Fixed normal axis classification.
    unsigned *axis_slot;                      ///< [ndim] Canonical face slot of free axes.
    const integration_rule_t **element_rules; ///< [ndim] Element rules at the common face slots.
    constraint_boundary_mass_work_t mass;     ///< Nested per-item mass assembly scratch.
} constrain_elements_on_boundary_work_t;

/**
 * @brief Compute the per-item intermediates of one boundary constraint batch.
 *
 * Per form, the common boundary space is merged from all incident elements
 * into @p out_basis and @p out_integration, the boundary and element basis
 * tables are pulled from the registries into @p plan, and every item's matrix
 * layout is recorded. All plan arrays are sized a priori from the request;
 * the value table sizes follow from #constrain_elements_on_boundary_work_size.
 *
 * @param request Filled request; read-only.
 * @param work Caller-provided scratch; the layout step overwrites it.
 * @param out_basis [nforms * bdim] Per-form common boundary basis.
 * @param out_integration [nforms * bdim] Per-form common boundary rules.
 * @param plan Caller-provided plan; filled on return.
 * @return FDG_SUCCESS on success, or a registry allocation error. On failure
 *         the plan must still be released.
 */
fdg_result_t constrain_elements_on_boundary_prepare(const constrain_elements_on_boundary_request_t *request,
                                                    constrain_elements_on_boundary_work_t *work,
                                                    basis_spec_t *out_basis, integration_spec_t *out_integration,
                                                    constrain_elements_on_boundary_plan_t *plan);

/**
 * @brief Compute the value table sizes of one prepared boundary constraint
 *        batch.
 *
 * @param request Filled request; read-only.
 * @param plan Prepared plan; read-only.
 * @param out_weights Receives the doubles of the largest form's tensor
 *        quadrature weights.
 * @param out_row_values Receives the doubles of the largest per-item test
 *        component tables.
 * @param out_col_values Receives the doubles of the largest per-item element
 *        component tables.
 */
void constrain_elements_on_boundary_work_size(const constrain_elements_on_boundary_request_t *request,
                                              const constrain_elements_on_boundary_plan_t *plan, size_t *out_weights,
                                              size_t *out_row_values, size_t *out_col_values);

/**
 * @brief Assemble the prepared boundary constraint mass matrices.
 *
 * Every element's mass matrix against its form's common space is assembled
 * into its slice of @p out_values (`plan->item_offsets[item]` bytes of offset,
 * `item_rows[item] * item_cols[item]` entries). Coefficients carry the
 * orientation signs but no side signs; the store combining the element
 * matrices applies those. The routine performs no allocation and touches no
 * registry.
 *
 * @param request Filled request; read-only.
 * @param plan Prepared plan; read-only.
 * @param work Caller-provided scratch; sized for every item's form.
 * @param out_values [plan->total_values] Dense row-major item matrices.
 */
void constrain_elements_on_boundary_assemble(const constrain_elements_on_boundary_request_t *request,
                                             const constrain_elements_on_boundary_plan_t *plan,
                                             constrain_elements_on_boundary_work_t *work, double *out_values);

/**
 * @brief Release the registry references of one prepared plan.
 *
 * @param plan Prepared plan; the registry references are invalid on return.
 */
void constrain_elements_on_boundary_plan_release(constrain_elements_on_boundary_plan_t *plan);

/**
 * @brief Resampled coordinate derivatives of one element's face on the common
 *        boundary grid.
 *
 * This is the minimal common SpaceMap payload the boundary constraints
 * consume: the face immersion's determinant (surface measure) and backward
 * derivatives at the common boundary integration points, interpolated from
 * the face-restricted map of any one incident element. A C1 continuous space
 * mapping needs no geometry at all, since reference-space continuity then
 * implies physical continuity.
 */
typedef struct
{
    unsigned bdim;                                 ///< Boundary dimension.
    unsigned coords;                               ///< Physical coordinate count of the face immersion.
    const integration_rule_t *const *source_rules; ///< [bdim] Face map's per-axis rules.
    const integration_rule_t *const *target_rules; ///< [bdim] Common boundary rules.
    const double *const *coordinate_values;        ///< [coords] Face map values, `source_points` each.
    const double *const *coordinate_gradients;     ///< [coords * bdim] Face map gradients, `source_points` each.
    double *out_determinant;                       ///< [target_points] Surface measure at the common points.
    double *out_inverse_maps;                      ///< [target_points * bdim * coords] Backward derivatives.
    double *axis_matrices;                         ///< Work sized by #boundary_space_map_resample_work_size.
    double *positions;                             ///< Work sized by #boundary_space_map_resample_work_size.
    double *jacobian;                              ///< Work sized by #boundary_space_map_resample_work_size.
    double *q;                                     ///< Work sized by #boundary_space_map_resample_work_size.
} boundary_space_map_resample_request_t;

/**
 * @brief Interpolate a face-restricted space map onto the common boundary grid.
 *
 * Per axis, the face map's sampled values and gradients are interpolated to
 * the common rule nodes with the Lagrange interpolant through the source
 * nodes, then inverted per point. Exact whenever the source sampling resolves
 * the map's polynomial degree along every axis.
 *
 * @param request Filled request; outputs written on return.
 */
void boundary_space_map_resample(const boundary_space_map_resample_request_t *request);

/**
 * @brief Size the work buffers of #boundary_space_map_resample.
 *
 * @param bdim Boundary dimension.
 * @param coords Physical coordinate count.
 * @param source_rules [bdim] Face map's per-axis rules.
 * @param target_rules [bdim] Common boundary rules.
 * @param out_axis_matrices Receives the doubles for the interpolation matrices.
 * @param out_positions Receives the doubles for the interpolated positions.
 * @param out_jacobian Receives the doubles for the Jacobian scratch.
 * @param out_q Receives the doubles for the inversion scratch.
 */
void boundary_space_map_resample_work_size(unsigned bdim, unsigned coords,
                                           const integration_rule_t *const *source_rules,
                                           const integration_rule_t *const *target_rules, size_t *out_axis_matrices,
                                           size_t *out_positions, size_t *out_jacobian, size_t *out_q);

/**
 * @brief Compute the packed size of a two-sided reference trace matrix.
 *
 * @param test_spec Face test-space specification with
 *        `order <= ndim`; both sides must satisfy the constraints of
 *        #constraint_element_side_t against `test_spec`.
 * @param sides Two element-side specifications.
 * @param out_row_count Receives the total test DoF count.
 * @param out_entry_count Receives the total packed entry count.
 */
void constraint_reference_layout(const kform_spec_t *test_spec, const constraint_element_side_t sides[static 2],
                                 size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Choose the per-face-axis Gauss rule for a two-sided reference trace.
 *
 * The rule on face axis `a` integrates the trace pairing exactly: its
 * accuracy is the test-axis order plus the larger of the two mapped element
 * axis orders, plus one spare degree for the inactive-axis basis shifts.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param out_specs Receives `test_spec->ndim` integration specifications.
 */
void constraint_reference_rule_specs(const kform_spec_t *test_spec, const constraint_element_side_t sides[static 2],
                                     integration_spec_t out_specs[static test_spec->ndim]);

/**
 * @brief Assemble a two-sided reference-space trace constraint matrix.
 *
 * Rows follow the packed-row contract: component-major, then component-local
 * test DoF. Each row holds the mapped component block of side 0 followed by
 * side 1, with DoFs in table order; `out_row_offsets` has
 * `row_count + 1` entries starting at zero. Coefficients are
 * `side_sign * orientation_sign * integral` with `side_sign` +1 on side 0
 * and -1 on side 1.
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param point_weights Tensor quadrature weights of the shared face rules.
 * @param test_table Test trace basis values on the shared rules.
 * @param element_tables Per-side element trace basis values on the shared
 *        rules; negative orientations read mirrored node indices.
 * @param out_sides [entry_count] Side index of each entry.
 * @param out_components [entry_count] Element component of each entry.
 * @param out_local_dofs [entry_count] Component-local DoF of each entry.
 * @param out_coefficients [entry_count] Entry coefficients.
 * @param out_row_offsets [row_count + 1] Packed row offsets.
 */
void constraint_reference_assemble(const kform_spec_t *test_spec, const constraint_element_side_t sides[static 2],
                                   const double *point_weights, const kform_values_table_t *test_table,
                                   const kform_values_table_t *element_tables[static 2], uint8_t out_sides[],
                                   uint32_t out_components[], size_t out_local_dofs[], double out_coefficients[],
                                   size_t out_row_offsets[]);

/**
 * @brief Test whether a two-sided reference trace reduces to single DoF links.
 *
 * The link form replaces the assembled moment rows with one equality link per
 * test DoF and side. It is exact when both sides sample the same trace space:
 * along every canonical face axis the two sides must map to the same basis
 * family and order, matching the test order, and every fixed normal axis must
 * carry a basis with a single DoF supported at each endpoint (Gauss-Lobatto
 * Lagrange or Bernstein). Under these conditions each component's exact trace
 * Gram block is square and invertible and the two sides' blocks agree up to a
 * node permutation, so the dense rows and the links span the same row space.
 */
bool constraint_reference_links_eligible(const kform_spec_t *test_spec,
                                         const constraint_element_side_t sides[static 2]);

/**
 * @brief Compute the packed size of a two-sided reference link constraint.
 *
 * Rows match @ref constraint_reference_layout; every row holds exactly one
 * entry per side.
 */
void constraint_reference_links_layout(const kform_spec_t *test_spec, const constraint_element_side_t sides[static 2],
                                       size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Reduce assembled dense trace rows to single-DoF links.
 *
 * Requires @ref constraint_reference_links_eligible and the dense rows of
 * @ref constraint_reference_assemble. Every row is replaced by one equality
 * per side linking the endpoint-supported, canonicalized element DoFs that
 * carry the row's test DoF; the coefficient ratio is read from the dense row
 * itself, so orientation signs need no re-derivation. Same packed-row
 * contract, with exactly one entry per side per row.
 */
void constraint_reference_links_reduce(const kform_spec_t *test_spec, const constraint_element_side_t sides[static 2],
                                       size_t dense_row_count, const uint8_t dense_sides[static 1],
                                       const uint32_t dense_components[static 1],
                                       const size_t dense_local_dofs[static 1],
                                       const double dense_coefficients[static 1],
                                       const size_t dense_row_offsets[static 1], uint8_t out_sides[],
                                       uint32_t out_components[], size_t out_local_dofs[], double out_coefficients[],
                                       size_t out_row_offsets[]);

/**
 * @brief Compute the packed size of one side of a physical trace matrix.
 *
 * @param test_spec Face test-space specification with `order <= ndim`.
 * @param side Element-side specification.
 * @param out_row_count Receives the total test DoF count.
 * @param out_entry_count Receives the total packed entry count.
 */
void constraint_physical_side_layout(const kform_spec_t *test_spec, const constraint_element_side_t *side,
                                     size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble one side of a physical trace constraint matrix.
 *
 * Row order as in #constraint_reference_assemble, but every face component's
 * mapped element component block is emitted in face-component order because
 * the physical pullback can couple components. Coefficients are
 * `test_orientation_sign * orientation_sign * integral`.
 *
 * @param test_spec Face test-space specification.
 * @param side Element-side specification.
 * @param inputs Precomputed weights, tables, and pullback of the side.
 * @param out_components [entry_count] Element component of each entry.
 * @param out_local_dofs [entry_count] Component-local DoF of each entry.
 * @param out_coefficients [entry_count] Entry coefficients.
 * @param out_row_offsets [row_count + 1] Packed row offsets.
 */
void constraint_physical_side_assemble(const kform_spec_t *test_spec, const constraint_element_side_t *side,
                                       const constraint_assembly_inputs_t *inputs, uint32_t out_components[],
                                       size_t out_local_dofs[], double out_coefficients[], size_t out_row_offsets[]);

/**
 * @brief Assemble a two-sided physical trace constraint matrix.
 *
 * Same contract as #constraint_physical_side_assemble with side 0's entries
 * before side 1's in each row; coefficients additionally carry the side sign
 * (+1 on side 0, -1 on side 1).
 *
 * @param test_spec Face test-space specification.
 * @param sides Two element-side specifications.
 * @param inputs Per-side assembly inputs.
 * @param out_sides [entry_count] Side index of each entry.
 * @param out_components [entry_count] Element component of each entry.
 * @param out_local_dofs [entry_count] Component-local DoF of each entry.
 * @param out_coefficients [entry_count] Entry coefficients.
 * @param out_row_offsets [row_count + 1] Packed row offsets.
 */
void constraint_physical_assemble(const kform_spec_t *test_spec, const constraint_element_side_t sides[static 2],
                                  const constraint_assembly_inputs_t inputs[static 2], uint8_t out_sides[],
                                  uint32_t out_components[], size_t out_local_dofs[], double out_coefficients[],
                                  size_t out_row_offsets[]);

/**
 * @brief Compute the packed size of a concatenated physical trace batch.
 *
 * @param test_spec Shared face test-space specification.
 * @param item_count Number of batch items.
 * @param items Batch descriptors.
 * @param out_row_count Receives the total row count.
 * @param out_entry_count Receives the total entry count.
 */
void constraint_physical_batch_layout(const kform_spec_t *test_spec, size_t item_count,
                                      const constraint_physical_batch_item_t items[static item_count],
                                      size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble a concatenated batch of two-sided physical trace matrices.
 *
 * Items are assembled in input order; each item's local row offsets are
 * rebased by the accumulated entry count of the earlier items.
 *
 * @param test_spec Shared face test-space specification.
 * @param item_count Number of batch items.
 * @param items Batch descriptors in output order.
 * @param out_sides [entry_count] Side index of each entry.
 * @param out_components [entry_count] Element component of each entry.
 * @param out_local_dofs [entry_count] Component-local DoF of each entry.
 * @param out_coefficients [entry_count] Entry coefficients.
 * @param out_row_offsets [row_count + 1] Combined packed row offsets.
 */
void constraint_physical_batch_assemble(const kform_spec_t *test_spec, size_t item_count,
                                        const constraint_physical_batch_item_t items[static item_count],
                                        uint8_t out_sides[], uint32_t out_components[], size_t out_local_dofs[],
                                        double out_coefficients[], size_t out_row_offsets[]);

/**
 * @brief Assemble a boundary load from sampled element-frame k-form data.
 *
 * For each traced component, this selects the datum component containing the
 * fixed normal axis, applies the wedge insertion sign, and accumulates its
 * quadrature pairing with every element trace basis function. The accumulator
 * is intentionally not cleared so multiple faces can contribute to one load.
 * The side must describe a codimension-one face (`side->ndim ==
 * test_spec->ndim + 1`).
 *
 * @param test_spec Face test-space specification of degree one below the datum.
 * @param side Codimension-one element-side specification.
 * @param point_weights Canonical tensor quadrature weights.
 * @param datum_values Sampled element-frame datum components, point-major:
 *        `datum_values[component * point_count + point]`.
 * @param surface_weights Optional unsigned face measures, NULL = unweighted.
 * @param element_table Element trace basis values on the same points.
 * @param values Output accumulator with one slot per element DoF.
 */
void constraint_physical_side_load(const kform_spec_t *test_spec, const constraint_element_side_t *side,
                                   const double *point_weights, const double *datum_values,
                                   const double *surface_weights, const kform_values_table_t *element_table,
                                   double values[]);

/**
 * @brief Map an element axis to its canonical face position.
 *
 * Counts the non-fixed element axes below `element_axis`; the fixed normal
 * axes are given by the signed prefix of the orientation record.
 *
 * @param element_dim Element dimension.
 * @param face_dim Canonical face dimension.
 * @param orientation Signed one-based element-axis mapping.
 * @param element_axis Element axis to locate.
 * @return The canonical face position in `[0, face_dim)`.
 */
unsigned constraint_face_source_axis(unsigned element_dim, unsigned face_dim,
                                     const int8_t orientation[static element_dim], unsigned element_axis);

/**
 * @brief Choose the canonical-face spec of every test-space face axis.
 *
 * Test-space face axis `a` is mapped by the orientation to an element axis;
 * the output spec is the source-frame spec at that element axis's canonical
 * face position.
 *
 * @param element_dim Element dimension.
 * @param face_dim Canonical face dimension.
 * @param orientation Signed one-based element-axis mapping.
 * @param face_specs Source-frame face-axis specs (the mapped face's specs).
 * @param out_specs Receives `face_dim` canonical specs.
 */
void constraint_face_canonical_specs(unsigned element_dim, unsigned face_dim,
                                     const int8_t orientation[static element_dim],
                                     const integration_spec_t face_specs[static face_dim],
                                     integration_spec_t out_specs[static face_dim]);

/**
 * @brief Map a canonical face point to its source-frame tensor index.
 *
 * The canonical digit of face axis `a` is decoded with `canonical_strides`;
 * negative orientations mirror the digit in the source frame. The mapped
 * digits are re-encoded with the row-major `source_strides`.
 *
 * @param element_dim Element dimension.
 * @param face_dim Canonical face dimension.
 * @param orientation Signed one-based element-axis mapping.
 * @param source_specs Source-frame face-axis specs.
 * @param canonical_specs Canonical face-axis specs.
 * @param canonical_strides [face_dim] row-major canonical strides.
 * @param source_strides [face_dim] row-major source-frame strides.
 * @param canonical_point Flat canonical tensor point.
 * @return The flat source-frame tensor point.
 */
size_t constraint_face_point_to_source(unsigned element_dim, unsigned face_dim,
                                       const int8_t orientation[static element_dim],
                                       const integration_spec_t source_specs[static face_dim],
                                       const integration_spec_t canonical_specs[static face_dim],
                                       const size_t canonical_strides[static face_dim],
                                       const size_t source_strides[static face_dim], size_t canonical_point);

/**
 * @brief Build a sampled trace pullback from a mapped face transform.
 *
 * Fills `request->out` by permuting the source-frame transform samples into
 * the canonical frame: entry
 * `out[(element_component * physical_component_count + physical_component) *
 * canonical_point_count + canonical_point]` receives the transform value at
 * the mapped source point. For order zero the output is zero-filled.
 *
 * @param request Fully populated pullback build parameters.
 */
void constraint_trace_pullback_build(const constraint_trace_pullback_build_t *request);
