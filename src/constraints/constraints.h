/**
 * @file constraints.h
 * @brief Reference and physical trace constraints for continuous k-forms.
 *
 * Assembly uses one canonical face coordinate system; element-side orientations map it to signed, one-based element
 * axes, and the face helpers here keep that mapping and the alternating k-form sign in one place. Routines return
 * void, document preconditions checked only by debug asserts, size storage through the `*_layout`/`*_work_size`
 * functions, and write flat parallel arrays in each function's packed-row contract. Dimensions and component indices
 * follow the combination iterator's canonical axis order; basis functions are evaluated on canonical face
 * coordinates.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cutl/iterators/multidim_iteration.h>

#include "../basis/basis_set.h"
#include "../kforms/kform_types.h"
#include "../kforms/kform_values.h"

/**
 * @brief Specification of one higher-dimensional element side.
 *
 * `orientation` is a signed one-based permutation: entry `i` maps the side's canonical axis `i` to element axis
 * `|orientation[i]| - 1`, a negative sign reversing that axis. Fixed normal axes come first with increasing
 * absolute values, giving the endpoint prefix a deterministic convention.
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
 * `basis` holds one spec per element axis (the k-form spec's per-axis array for k-form continuity). The boundary
 * merge reads only per-axis orders and families, so scalar and k-form trace spaces share one request type.
 */
typedef struct
{
    unsigned order;                        ///< Traced k-form order; unused by the space merge.
    const int8_t *orientation;             ///< [ndim] Signed one-based axis mapping.
    const basis_spec_t *basis;             ///< [ndim] Element-axis basis specifications.
    const integration_spec_t *integration; ///< [ndim] Element-axis integration rules.
} boundary_element_space_t;

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
 * @brief Work buffers of one sampled trace pullback build.
 *
 * Every pointer is caller-provided; sizes come from
 * #constraint_trace_pullback_build_work_size.
 */
typedef struct constraint_trace_pullback_build_work_t_ constraint_trace_pullback_build_work_t;

/**
 * @brief Parameters for building a sampled trace pullback.
 *
 * The transform is the `compute_basis_transform_impl` output of the mapped
 * face: `face_component_count * physical_component_count * source_point_count`
 * values, element component slowest and source point fastest.
 */
typedef struct
{
    unsigned element_dim;                         ///< Element dimension.
    unsigned face_dim;                            ///< Canonical face dimension.
    unsigned order;                               ///< Traced k-form order.
    unsigned face_component_count;                ///< C(face_dim, order).
    unsigned physical_component_count;            ///< C(coordinates, order) of the mapped face.
    size_t source_point_count;                    ///< Points of the source-frame face tensor.
    size_t canonical_point_count;                 ///< Points of the canonical face tensor.
    const size_t *source_strides;                 ///< [face_dim] row-major source-frame strides.
    const size_t *canonical_strides;              ///< [face_dim] row-major canonical strides.
    const int8_t *orientation;                    ///< Signed one-based element-axis mapping.
    const integration_spec_t *source_specs;       ///< [face_dim] source-frame axis specs.
    const integration_spec_t *canonical_specs;    ///< [face_dim] canonical axis specs.
    const double *transform;                      ///< Sampled face transform (see above).
    double *out;                                  ///< [element_comp * physical_comp * canonical_point_count].
    bool canonical_components;                    ///< Index `out` by canonical boundary component.
    bool element_components;                      ///< Index `out` by element component (C(element_dim, order)
                                                  ///< blocks) instead of face component.
    constraint_trace_pullback_build_work_t *work; ///< Caller-provided scratch buffers.
} constraint_trace_pullback_build_t;

/**
 * @brief Allocation sizes of one trace pullback build's work buffers.
 */
typedef struct
{
    unsigned face_axis_count;     ///< Per-face-axis work arrays, `face_dim`.
    unsigned element_axis_count;  ///< Per-element-axis work arrays, `element_dim`.
    size_t element_component_map; ///< `element_to_face` slots: `C(element_dim, order) + 1`.
    unsigned axes_scratch;        ///< Component axes scratch slots, `max(order, 1)`.
    size_t iterator_memory;       ///< Combination iterator memory, `required_memory(order)`.
} constraint_trace_pullback_build_work_sizes_t;

/**
 * @brief Compute the work buffer sizes of one sampled trace pullback build.
 *
 * @param request Build request; only its dimensions and `element_components`
 *                flag are read.
 * @param out_sizes Output allocation sizes.
 */
void constraint_trace_pullback_build_work_size(const constraint_trace_pullback_build_t *request,
                                               constraint_trace_pullback_build_work_sizes_t *out_sizes);

/**
 * @brief Work buffers of one sampled trace pullback build.
 *
 * Every pointer is caller-provided; sizes come from
 * #constraint_trace_pullback_build_work_size.
 */
typedef struct constraint_trace_pullback_build_work_t_
{
    unsigned *axis_source_slots;        ///< [face_axis_count] Free-axis rank of each face axis.
    unsigned *axis_orders;              ///< [face_axis_count] Canonical rule order per face axis.
    size_t *axis_source_strides;        ///< [face_axis_count] Source-frame stride per face axis.
    size_t *axis_canonical_strides;     ///< [face_axis_count] Canonical stride per face axis.
    int *axis_mirrored;                 ///< [face_axis_count] Mirror flag per face axis.
    bool *element_axis_free;            ///< [element_axis_count] Free-axis classification.
    unsigned *element_source_rank;      ///< [element_axis_count] Free-axis rank per element axis.
    unsigned *element_to_face;          ///< [element_component_map] Element to face component map.
    uint8_t *mapped_axes;               ///< [axes_scratch] Mapped axes scratch.
    uint8_t *source_axes;               ///< [axes_scratch] Rank scratch of the current component.
    combination_iterator_t *components; ///< `iterator_memory` bytes.
} constraint_trace_pullback_build_work_t;

/**
 * @brief Highest Legendre functions dropped from every inactive axis of a common boundary test space.
 *
 * The window keeps the leading functions of the full basis on each axis that carries no covector of a k-form
 * component: pairing the (possibly higher-order) trace against the retained low degrees is exactly the L2
 * projection of the boundary solution onto the lower-order space they span. An axis whose full basis has at most
 * #SKIPPED_BASIS functions contributes an empty row block.
 */
enum
{
    SKIPPED_BASIS = 2, ///< Functions dropped from the high end of every inactive axis.
};

/**
 * @brief Shape of one element's boundary mass matrix.
 *
 * Maps the element's face trace DoFs to the common boundary test space: row blocks follow the common Legendre
 * k-form components, column blocks the mapped element components in canonical boundary order. Rows are
 * component-local tensors with per-axis counts
 *
 * - active covector axis: the order-1 basis (`order` functions, offset zero),
 * - inactive axis: the full basis (`order + 1` functions) windowed to its first `order + 1 - SKIPPED_BASIS`
 *   functions — the last #SKIPPED_BASIS (highest-degree) functions are dropped, floored at a zero count.
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
 * @brief Caller-provided scratch memory of one element's boundary mass matrix.
 *
 * All members are caller-allocated arrays with lengths fixed by the spec: `bdim` per-axis entries, `ndim` axis
 * descriptors, `component_count + 1` (#constraint_boundary_mass_work_sizes_t) component tables. The assemble,
 * layout, and pack routines use them as scratch and may overwrite the contents. #constraint_boundary_mass_work_init
 * assigns every member from one block of #constraint_boundary_mass_work_memory bytes.
 */
typedef struct
{
    multidim_iterator_t
        *point_iter;     ///< `multidim_iterator_needed_memory(ndim)`; tensor points of the column value tables.
    size_t *row_offsets; ///< [component_count + 1] Test component offsets.
    size_t *col_offsets; ///< [component_count + 1] Mapped element offsets.
    unsigned *element_components;       ///< [component_count] Mapped element components.
    int *element_signs;                 ///< [component_count] Mapped covector signs.
    kform_trace_axis_t *axes;           ///< [ndim] Element axis trace descriptors.
    unsigned *counts;                   ///< [bdim] Per-axis test function counts.
    const basis_set_t **axis_sets;      ///< [bdim] Per-axis selected basis tables.
    const double **axis_tables;         ///< [bdim] Per-axis basis value tables.
    multidim_iterator_t *dof_iter;      ///< `multidim_iterator_needed_memory(bdim)`; component-local DoF digits.
    unsigned *point_digits;             ///< [bdim] Point odometer digits of the row value sweep.
    double *point_prefix;               ///< [bdim] Prefix products of the axis row tables.
    uint8_t *mapped_axes;               ///< [order] Mapped element axes of the current component.
    combination_iterator_t *components; ///< `combination_iterator_required_memory(order)`.
    combination_iterator_t *blocks;     ///< `combination_iterator_required_memory(order)`; physical pairing only.
    double *row_values;                 ///< [row_values] Test component value tables.
    double *col_values;                 ///< [col_values] Element component value tables.
    double *point_factors;              ///< [point_factors] Per-point factor buffer.
} constraint_boundary_mass_work_t;

/**
 * @brief Compute the work buffer sizes of one element's boundary mass matrix.
 *
 * Does not allocate: the sizing pass runs on the caller-provided work scratch.
 *
 * @param spec Filled matrix specification.
 * @param work Caller-provided scratch; only `counts` (`bdim`), `mapped_axes`
 *             (`max(order, 1)`), and `components` (`combination_iterator_required_memory(order)` bytes) must be
 *             valid — all sized a priori from the spec, e.g. by #constraint_boundary_mass_work_init. The contents
 *             are overwritten; the value table members are not read.
 * @param out_sizes Receives the allocation sizes.
 */
void constraint_boundary_mass_work_size(const constraint_boundary_mass_spec_t *spec,
                                        constraint_boundary_mass_work_t *work,
                                        constraint_boundary_mass_work_sizes_t *out_sizes);

/**
 * @brief Total bytes of one element's boundary mass work buffers.
 *
 * Single-block convenience around #constraint_boundary_mass_work_size: this tells the allocation size,
 * #constraint_boundary_mass_work_init wires every member of #constraint_boundary_mass_work_t into that one block,
 * and the sizing scratch runs from a first small block:
 *
 *     constraint_boundary_mass_work_init(&work, &spec, NULL,
 *                                        malloc(constraint_boundary_mass_work_memory(&spec, NULL)));
 *     constraint_boundary_mass_work_size(&spec, &work, &sizes);
 *     constraint_boundary_mass_work_init(&work, &spec, &sizes,
 *                                        malloc(constraint_boundary_mass_work_memory(&spec, &sizes)));
 *
 * (Free the first block after the second init.)
 *
 * @param spec Filled matrix specification.
 * @param sizes Work sizes from #constraint_boundary_mass_work_size, or NULL for the sizing scratch alone (the
 *              `counts`, `mapped_axes`, and `components` members the sizing pass reads, plus every
 *              other a-priori-sized member; the value table members stay NULL).
 * @return Total bytes for one block holding every member the call initializes.
 */
size_t constraint_boundary_mass_work_memory(const constraint_boundary_mass_spec_t *spec,
                                            const constraint_boundary_mass_work_sizes_t *sizes);

/**
 * @brief Point every member of one boundary mass work struct into one memory block.
 *
 * @param work Work struct filled on return; previously held pointers are not freed.
 * @param spec Filled matrix specification.
 * @param sizes Work sizes as in #constraint_boundary_mass_work_memory.
 * @param memory Block of #constraint_boundary_mass_work_memory bytes.
 */
void constraint_boundary_mass_work_init(constraint_boundary_mass_work_t *work,
                                        const constraint_boundary_mass_spec_t *spec,
                                        const constraint_boundary_mass_work_sizes_t *sizes, void *memory);

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
 * @param work Caller-provided scratch; per-axis, component-table, and iterator members are used and overwritten.
 * @param physical Non-zero: physical pullbacks couple every mapped component pair; reference pairing is
 *        component-diagonal.
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
 * Coefficients carry the mapped components' orientation sign and #constraint_boundary_mass_request_t::factor, the
 * quadrature weights, the optional surface measure, and the optional physical pullback dot product. Reference
 * pairing (no pullbacks) couples each test component only with its mapped element component; pullbacks couple every
 * mapped pair. The matrix is zero-initialized first.
 *
 * @param request Filled request; `out_matrix` written on return.
 */
void constraint_boundary_mass_assemble(const constraint_boundary_mass_request_t *request);

/**
 * @brief Pack a dense boundary mass matrix into the COO contract.
 *
 * Rows follow the packed-row contract: components in canonical order, then component-local test DoF. Reference
 * pairing emits each row's mapped component block; physical pairing every mapped block. Coefficients are read
 * verbatim and multiplied by @p factor (e.g. the constraint side sign); `out_row_offsets` has `row_count + 1`
 * entries starting at zero.
 *
 * @param spec Specification the matrix was assembled with.
 * @param work Caller-provided scratch; component-table and iterator members are used and overwritten.
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
                                   bool physical, const double *matrix, size_t row_stride, double factor, uint64_t side,
                                   uint64_t out_sides[], uint32_t out_components[], size_t out_local_dofs[],
                                   double out_coefficients[], size_t out_row_offsets[]);

/**
 * @brief Inputs of the per-element boundary constraint mass matrices of one
 *        shared object.
 */
typedef struct
{
    unsigned ndim;                                            ///< Element dimension, at least 2.
    unsigned bdim;                                            ///< Boundary dimension, in `[1, ndim)`.
    unsigned nforms;                                          ///< Traced k-form count, at least 1.
    unsigned nelem;                                           ///< Incident element count, at least 1.
    const boundary_element_space_t *elements;                 ///< [nforms * nelem] Form-major element views.
    bool c1_continuous;                                       ///< Reference-space pairing; pullback inputs may be NULL.
    const double *const *surface_weights;                     ///< [nforms * nelem] Optional per-item face measure rows.
    const constraint_trace_pullback_t *const *test_pullbacks; ///< [nforms * nelem] Optional per-item pullbacks.
    const constraint_trace_pullback_t *const *element_pullbacks; ///< [nforms * nelem] Optional per-item pullbacks.
    basis_set_registry_t *basis_registry;                        ///< Registry for basis and endpoint tables.
    integration_rule_registry_t *integration_registry;           ///< Registry for quadrature rules.
} constrain_elements_on_boundary_request_t;

/**
 * @brief Intermediates of one boundary constraint batch.
 *
 * Caller-allocated arrays sized from the request: `nforms * bdim` per-form boundary, `nforms * nelem * ndim`
 * per-item element, `nforms * nelem` (`+ 1` offsets) matrix layout.
 * #constrain_elements_on_boundary_prepare fills them, #constrain_elements_on_boundary_assemble reads them,
 * #constrain_elements_on_boundary_plan_release returns the registry references. Borrowed request and output pointers
 * must outlive the plan.
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
 * Array lengths follow from the request (`ndim`, `bdim`), the maximum common rule point count, and the maximum
 * per-item value table sizes (the latter two from #constrain_elements_on_boundary_work_size). The assembler
 * overwrites the contents freely.
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
 * Per form, merge the common boundary space from all incident elements into @p out_basis and @p out_integration,
 * pull the boundary and element basis tables from the registries into @p plan, and record every item's matrix
 * layout. All plan arrays are sized a priori from the request; value table sizes follow from
 * #constrain_elements_on_boundary_work_size.
 *
 * @param request Filled request; read-only.
 * @param work Caller-provided scratch; the layout step overwrites it.
 * @param out_basis [nforms * bdim] Per-form common boundary basis.
 * @param out_integration [nforms * bdim] Per-form common boundary rules.
 * @param plan Caller-provided plan; filled on return.
 * @return FDG_SUCCESS or a registry allocation error; the plan must be released on failure too.
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
 * @param work Caller-provided scratch; only the nested `mass` members #constraint_boundary_mass_work_size reads are
 *             required (sized a priori from the spec, e.g. by #constrain_elements_on_boundary_work_init).
 * @param out_weights Receives the doubles of the largest form's tensor quadrature weights.
 * @param out_row_values Receives the doubles of the largest per-item test component tables.
 * @param out_col_values Receives the doubles of the largest per-item element component tables.
 */
void constrain_elements_on_boundary_work_size(const constrain_elements_on_boundary_request_t *request,
                                              const constrain_elements_on_boundary_plan_t *plan,
                                              constrain_elements_on_boundary_work_t *work, size_t *out_weights,
                                              size_t *out_row_values, size_t *out_col_values);

/**
 * @brief Assemble the prepared boundary constraint mass matrices.
 *
 * Assembles every element's mass matrix against its form's common space into its slice of @p out_values (offset
 * `plan->item_offsets[item]`, `item_rows[item] * item_cols[item]` entries). Coefficients carry orientation signs
 * but no side signs; the combining store applies those. No allocation, no registry access.
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
 * @brief Resampled coordinate derivatives of one element's face on the common boundary grid.
 *
 * The minimal SpaceMap payload the boundary constraints consume: the face immersion's determinant (surface measure)
 * and backward derivatives at the common boundary integration points, interpolated from any one incident element's
 * face-restricted map. A C1 continuous space mapping needs no geometry — reference-space continuity then implies
 * physical continuity.
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
    unsigned *target_orders;                       ///< [bdim] Work: per-axis target interpolation degrees.
    unsigned *source_orders;                       ///< [bdim] Work: per-axis source interpolation degrees.
    const double **axis_matrix_rows;               ///< [bdim] Work: per-axis interpolation matrices.
    integration_spec_t *target_specs;              ///< [bdim] Work: target specs of the common grid.
} boundary_space_map_resample_request_t;

/**
 * @brief Interpolate a face-restricted space map onto the common boundary grid.
 *
 * Per axis, interpolate the face map's sampled values and gradients to the common rule nodes with the Lagrange
 * interpolant through the source nodes, then invert per point. Exact when the source sampling resolves the map's
 * polynomial degree along every axis.
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
 * @param out_scratch_bytes Receives the bytes for the request's per-axis work arrays (`target_orders`,
 *        `source_orders`, `axis_matrix_rows`, `target_specs`).
 */
void boundary_space_map_resample_work_size(unsigned bdim, unsigned coords,
                                           const integration_rule_t *const *source_rules,
                                           const integration_rule_t *const *target_rules, size_t *out_axis_matrices,
                                           size_t *out_positions, size_t *out_jacobian, size_t *out_q,
                                           size_t *out_scratch_bytes);

/**
 * @brief Work buffers of #constraint_physical_side_load.
 */
typedef struct
{
    uint8_t *face_axes;                      ///< [max(order, 1)] Current face component's covector axes.
    uint8_t *element_axes;                   ///< [max(order, 1)] Mapped element component's covector axes.
    uint8_t *datum_axes;                     ///< [max(order, 1) + 1] Paired datum component's axes.
    uint8_t *mapped_axes;                    ///< [max(order, 1)] Mapped axes scratch.
    combination_iterator_t *face_components; ///< `combination_iterator_required_memory(order)`.
} constraint_physical_side_load_work_t;

/**
 * @brief Size the work buffers of #constraint_physical_side_load.
 *
 * @param test_spec Test (k-1)-form specification of the face.
 * @param out_face_axes Receives the face and scratch axis slots, `max(order, 1)`.
 * @param out_datum_axes Receives the datum axis slots, `max(order, 1) + 1`.
 * @param out_iterator Receives the combination iterator memory bytes.
 */
void constraint_physical_side_load_work_size(const kform_spec_t *test_spec, size_t *out_face_axes,
                                             size_t *out_datum_axes, size_t *out_iterator);

/**
 * @brief Assemble a boundary load from sampled element-frame k-form data.
 *
 * For each traced component, select the datum component containing the fixed normal axis, apply the wedge
 * insertion sign, and accumulate its quadrature pairing with every element trace basis function. The accumulator is
 * not cleared, so multiple faces can contribute to one load. The side must be codimension one (`side->ndim ==
 * test_spec->ndim + 1`).
 *
 * @param test_spec Face test-space specification of degree one below the datum.
 * @param side Codimension-one element-side specification.
 * @param point_weights Canonical tensor quadrature weights.
 * @param datum_values Sampled element-frame datum components, point-major:
 *        `datum_values[component * point_count + point]`.
 * @param surface_weights Optional unsigned face measures, NULL = unweighted.
 * @param element_table Element trace basis values on the same points.
 * @param work Caller-provided work buffers.
 * @param values Output accumulator with one slot per element DoF.
 */
void constraint_physical_side_load(const kform_spec_t *test_spec, const constraint_element_side_t *side,
                                   const double *point_weights, const double *datum_values,
                                   const double *surface_weights, const kform_values_table_t *element_table,
                                   constraint_physical_side_load_work_t *work, double values[]);

/**
 * @brief Map an element axis to its canonical face position.
 *
 * Counts the non-fixed element axes below `element_axis`; fixed normal axes are the orientation's signed prefix.
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
 * Test-space face axis `a` maps through the orientation to an element axis; the output spec is the source-frame
 * spec at that element axis's canonical face position.
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
 * Decodes face axis `a`'s canonical digit with `canonical_strides` (negative orientations mirror it in the source
 * frame) and re-encodes the mapped digits with the row-major `source_strides`.
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
 * Fills `request->out` by permuting source-frame transform samples into the canonical frame: entry
 * `out[(element_component * physical_component_count + physical_component) * canonical_point_count +
 * canonical_point]` receives the transform value at the mapped source point. Order zero zero-fills.
 *
 * @param request Fully populated pullback build parameters.
 */
void constraint_trace_pullback_build(const constraint_trace_pullback_build_t *request);
