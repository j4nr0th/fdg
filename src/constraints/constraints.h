/**
 * @file constraints.h
 * @brief Trace-constraint assembly for tensor-product differential-form spaces.
 *
 * The API describes a k-form test space on a canonical interface and one or
 * two higher-dimensional element sides. It assembles reference-space trace
 * pairings, physical-space pairings, and boundary loads into packed sparse
 * rows. All pointers supplied by the caller are borrowed; this module never
 * allocates or frees caller-owned storage.
 *
 * Dimensions and component indices use the canonical axis order expected by
 * the combination iterator. Basis functions are evaluated on [-1, 1]. Sizing
 * outputs and output counts are committed only after their checks succeed;
 * assembly buffers may be partially populated only where a function documents
 * that behavior.
 */
#pragma once

#include "../basis/basis_set.h"
#include "../kforms/kform_types.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Status and error codes returned by constraint functions.
 *
 * Every public operation returns one of these values. A successful operation
 * returns @ref CONSTRAINT_SUCCESS; no exception or global error state is used.
 * Values outside this enumeration are reported as "Unknown" by the
 * status-string helpers.
 */
typedef enum
{
    CONSTRAINT_SUCCESS = 0,          /**< Operation completed successfully. */
    CONSTRAINT_INVALID_ARGUMENT,     /**< Null, inconsistent, or out-of-range argument. */
    CONSTRAINT_INVALID_DIMENSION,    /**< Dimension exceeds the supported range. */
    CONSTRAINT_INVALID_ORDER,        /**< Form degree or basis order is invalid. */
    CONSTRAINT_SIZE_OVERFLOW,        /**< A required size does not fit in size_t. */
    CONSTRAINT_INSUFFICIENT_STORAGE, /**< A caller-provided output buffer is too small. */
} constraint_status_t;

/**
 * @brief Specification of one higher-dimensional element side.
 *
 * A side has strictly more dimensions than the test face. The first
 * `ndim - test_ndim` entries of `orientation` identify the fixed element axes
 * normal to the face; the remaining entries map canonical face axes to element
 * axes. Entries are signed one-based axis numbers: the sign reverses the
 * corresponding coordinate, and absolute values form a permutation of
 * `1..ndim`. The fixed-axis absolute values must be in increasing order.
 *
 * `basis_specs` and `orientation` each have `ndim` entries and are borrowed.
 */
typedef struct
{
    unsigned ndim;                   /**< Number of dimensions of the element. */
    const basis_spec_t *basis_specs; /**< `ndim` tensor-product element bases. */
    const int8_t *orientation;       /**< Signed one-based fixed-axis/face-axis mapping. */
} constraint_element_side_t;

/**
 * @brief Tensor-product quadrature rule over a canonical face.
 *
 * `axes` has `ndim` entries. The flattened point index uses the last face axis
 * as the fastest-changing index. `point_count` is caller-supplied and must
 * equal the product of all axis counts when the rule is passed to assembly.
 */
typedef struct
{
    unsigned ndim;                   /**< Number of quadrature axes. */
    const integration_rule_t **axes; /**< `ndim` one-dimensional rules. */
    size_t point_count;              /**< Cached product of the axis node counts. */
} constraint_face_quadrature_t;

/**
 * @brief Sampled tangential pullback of a physical k-form on a face.
 *
 * Storage is `[element_component][physical_component][point]`; component `c`,
 * physical component `p`, and point `i` are stored at
 * `values[c * physical_component_count * point_count + p * point_count + i]`.
 * For a k-form trace, `physical_component_count` normally equals
 * `C(element_ndim, k)`. Assembly requires the table to contain every component
 * selected by the side orientation.
 */
typedef struct
{
    unsigned physical_component_count; /**< Number of physical components per element component. */
    size_t point_count;                /**< Number of sampled face points. */
    const double *values;              /**< Borrowed `[element component][physical component][point]` data. */
} constraint_trace_pullback_t;

/**
 * @brief One two-sided physical trace assembly item in a batch.
 *
 * All items in one batch share the test specification passed to the batch
 * function. The two sides, quadrature rules, face measures, and pullbacks may
 * differ between items. `sides`, `quadrature`, and `surface_weights` point to
 * arrays of exactly two records/pointers; `pullbacks` likewise points to two
 * records when the test form order is non-zero.
 */
typedef struct
{
    const constraint_element_side_t *sides;         /**< Two element sides, weighted +1 and -1. */
    const constraint_face_quadrature_t *quadrature; /**< Two canonical-face quadrature rules. */
    const double *surface_weights[2];               /**< Two unsigned face-measure arrays. */
    const constraint_trace_pullback_t *pullbacks;   /**< Two sampled physical k-form pullbacks. */
} constraint_physical_batch_item_t;

/**
 * @brief Precomputed tensor-product trace basis values.
 *
 * `component_offsets` has `component_count + 1` cumulative DoF offsets. If
 * `dofs = offsets[c + 1] - offsets[c]`, the value of local DoF `dof` of
 * component `c` at point `point` is stored at
 * `values[offsets[c] * point_count + point * dofs + dof]`.
 * Both arrays are borrowed; their shape must match the k-form specification
 * and quadrature point count supplied to the consuming function.
 */
typedef struct
{
    size_t component_count;          /**< Number of components represented by the table. */
    size_t point_count;              /**< Number of points represented by the table. */
    const size_t *component_offsets; /**< `component_count + 1` cumulative DoF offsets. */
    const double *values;            /**< Component/point/DoF-major values. */
} constraint_trace_basis_values_t;

/**
 * @brief One non-zero entry of an assembled constraint-matrix row.
 *
 * The row is implicit: the entry belongs to the row whose interval in
 * `constraint_rows_view_t::row_offsets` contains its packed index.
 */
typedef struct
{
    uint8_t side;       /**< Element side: 0 for the first side, 1 for the second. */
    unsigned component; /**< Lexicographic k-form component on that side. */
    size_t local_dof;   /**< Flattened DoF index within `component`. */
    double coefficient; /**< Matrix coefficient for this row/column pair. */
} constraint_entry_t;

/**
 * @brief Read-only view of a packed sparse-row constraint matrix.
 *
 * Row `i` contains `entries[row_offsets[i] .. row_offsets[i + 1])`.
 * `row_offsets` therefore has `row_count + 1` entries, starts at zero, and
 * ends at `entry_count`. The arrays are borrowed and are not modified.
 */
typedef struct
{
    size_t row_count;                  /**< Number of packed rows. */
    size_t entry_count;                /**< Number of packed entries. */
    const size_t *row_offsets;         /**< `row_count + 1` non-decreasing offsets. */
    const constraint_entry_t *entries; /**< `entry_count` packed entries. */
} constraint_rows_view_t;

/**
 * @brief Return the symbolic name of a constraint status.
 * @param status Status value to translate.
 * @return Pointer to a static string such as `"CONSTRAINT_SUCCESS"`, or
 *         `"Unknown"` for an out-of-range value.
 */
const char *constraint_status_to_str(constraint_status_t status);

/**
 * @brief Return the short human-readable message for a constraint status.
 * @param status Status value to translate.
 * @return Pointer to a static string such as `"Invalid dimension"`, or
 *         `"Unknown"` for an out-of-range value.
 */
const char *constraint_status_msg(constraint_status_t status);

/**
 * @brief Derive per-component test-space basis specifications on a boundary.
 *
 * For every canonical boundary axis the derived order is the lowest order
 * among the incident elements (mapped through their orientation records);
 * each component then reduces the order by two on axes without one of its
 * covector axes and is reported absent when any reduced order would become
 * negative. Basis families follow the element achieving the per-axis minimum
 * unless overridden.
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
 * @param out_capacity Available component-major output slots; at least the
 *        number of components of the boundary space.
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
                                                   bool out_present[const static out_capacity]);

/**
 * @brief Compute storage requirements for a two-sided reference trace matrix.
 *
 * There is one row per test-space DoF. Each row contains the trace pairing with
 * the corresponding mapped component on each side.
 *
 * @param test_spec Test-space specification on the canonical face.
 * @param sides Two element-side descriptions.
 * @param out_row_count Receives the number of rows.
 * @param out_entry_count Receives the packed-entry count.
 * @return `CONSTRAINT_SUCCESS`, a specification validation error, or
 *         `CONSTRAINT_SIZE_OVERFLOW`. Outputs are unchanged on failure.
 */
constraint_status_t constraint_reference_required(const kform_spec_t *test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble a two-sided reference-space trace constraint matrix.
 *
 * Each test DoF produces a face L2 pairing with the mapped trace basis on both
 * element sides. Side 0 is positive and side 1 negative; orientation
 * reversals and component permutation parity are included in each coefficient.
 *
 * @param test_spec Test-space specification.
 * @param sides Two element-side descriptions accepted by
 *        constraint_reference_required.
 * @param quadrature One-dimensional rules for the face axes. The product of
 *        their node counts determines the quadrature points; this pointer may
 *        be null for a zero-dimensional face.
 * @param row_offset_capacity Number of available `row_offsets` entries.
 * @param row_offsets Output packed-row offsets with room for `row_count + 1`.
 * @param entry_capacity Number of available `entries` records.
 * @param entries Output packed entries.
 * @param out_row_count Receives the number of rows written.
 * @param out_entry_count Receives the number of entries written.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INSUFFICIENT_STORAGE`, or an
 *         input/quadrature/overflow error. Counts are written only on success.
 */
constraint_status_t constraint_reference_assemble(const kform_spec_t *test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  const integration_rule_t **quadrature, size_t row_offset_capacity,
                                                  size_t row_offsets[const static row_offset_capacity],
                                                  size_t entry_capacity,
                                                  constraint_entry_t entries[const static entry_capacity],
                                                  size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Compute storage requirements for a two-sided physical trace matrix.
 *
 * The row structure matches the reference count, but every side contributes
 * all mapped face components because the physical pullback can couple them.
 * Quadrature, surface measures, and pullback tables are not needed here.
 *
 * @param test_spec Test-space specification.
 * @param sides Two element-side descriptions.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the packed-entry count.
 * @return `CONSTRAINT_SUCCESS`, a validation error, or
 *         `CONSTRAINT_SIZE_OVERFLOW`. Outputs are unchanged on failure.
 */
constraint_status_t constraint_physical_required(const kform_spec_t *test_spec,
                                                 const constraint_element_side_t sides[const static 2],
                                                 size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Compute storage requirements for one side of a physical trace.
 *
 * This is the single-side counterpart of constraint_physical_required.
 *
 * @param test_spec Test-space specification.
 * @param side Element-side description.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the packed-entry count.
 * @return `CONSTRAINT_SUCCESS`, a validation error, or
 *         `CONSTRAINT_SIZE_OVERFLOW`. Outputs are unchanged on failure.
 */
constraint_status_t constraint_physical_side_required(const kform_spec_t *test_spec,
                                                      const constraint_element_side_t *side, size_t *out_row_count,
                                                      size_t *out_entry_count);

/**
 * @brief Assemble one side of a physical trace constraint matrix.
 *
 * A coefficient is the tensor-product quadrature sum of test and element
 * trace basis values, multiplied by the unsigned face measure and, for
 * non-zero form order, the dot product of the two sampled tangential pullback
 * components. All entries have `side == 0`.
 *
 * @param test_spec Test-space specification.
 * @param side Element-side specification.
 * @param quadrature Face quadrature whose `point_count` equals the product of
 *        its axis counts.
 * @param surface_weights Unsigned face-measure values, one per point.
 * @param pullback Sampled physical pullback; required for non-zero order and
 *        ignored for order zero.
 * @param row_offset_capacity Capacity of `row_offsets`.
 * @param row_offsets Output packed row offsets with room for `row_count + 1`.
 * @param entry_capacity Capacity of `entries`.
 * @param entries Output packed entries.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INSUFFICIENT_STORAGE`, or an
 *         input/quadrature/overflow error. Counts are written only on success.
 */
constraint_status_t constraint_physical_side_assemble(
    const kform_spec_t *test_spec, const constraint_element_side_t *side,
    const constraint_face_quadrature_t *quadrature, const double *surface_weights,
    const constraint_trace_pullback_t *pullback, size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble one side of a physical trace using precomputed basis values.
 *
 * This has the same signs, row order, and coefficient definition as
 * constraint_physical_side_assemble, but reads the test and element trace
 * basis values from caller-precomputed tables. Tables must use the layout
 * documented by constraint_trace_basis_values_t and have offsets consistent
 * with the supplied specifications.
 *
 * @param test_spec Test-space specification.
 * @param side Element-side specification.
 * @param quadrature Face quadrature matching both tables' point counts.
 * @param surface_weights Unsigned face-measure values, one per point.
 * @param pullback Sampled physical pullback, required for non-zero order.
 * @param test_basis Precomputed canonical test trace values.
 * @param element_basis Precomputed element trace values for all components.
 * @param row_offset_capacity Capacity of `row_offsets`.
 * @param row_offsets Output packed row offsets.
 * @param entry_capacity Capacity of `entries`.
 * @param entries Output packed entries.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INSUFFICIENT_STORAGE`, or an
 *         input/quadrature/table/overflow error. Counts are written only on
 *         success.
 */
constraint_status_t constraint_physical_side_assemble_precomputed(
    const kform_spec_t *test_spec, const constraint_element_side_t *side,
    const constraint_face_quadrature_t *quadrature, const double *surface_weights,
    const constraint_trace_pullback_t *pullback, const constraint_trace_basis_values_t *test_basis,
    const constraint_trace_basis_values_t *element_basis, size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble a physical boundary load for a general k-form datum.
 *
 * The test space has degree `k - 1` on a codimension-one face and
 * `datum_values` contains element-frame k-form components at face quadrature
 * points. For each face component, the routine selects the datum component
 * obtained by adjoining the fixed normal axis and integrates it with the
 * traced element basis. The result is added to `values`, not assigned, so the
 * caller must zero that array before the first accumulation.
 *
 * With `surface_weights == NULL` this is the metric-free chain integral. If
 * supplied, each quadrature contribution is also multiplied by the unsigned
 * mapped face measure.
 *
 * @param test_spec Test-space specification of degree `k - 1`.
 * @param side Element side with exactly one fixed normal axis.
 * @param quadrature Face quadrature with a consistent `point_count`.
 * @param datum_values Values laid out as
 *        `[datum_component * point_count + point]` for all `C(side->ndim, k)`
 *        components.
 * @param value_count Length of `values`; it must equal the total element trace
 *        DoF count.
 * @param surface_weights Optional unsigned face measures, one per point.
 * @param values Output/accumulator for element trace DoFs.
 * @return `CONSTRAINT_SUCCESS` or an input/quadrature/overflow error. The
 *         accumulator may be partially updated if a later validation fails.
 */
constraint_status_t constraint_physical_side_load(const kform_spec_t *test_spec, const constraint_element_side_t *side,
                                                  const constraint_face_quadrature_t *quadrature,
                                                  const double *datum_values, size_t value_count,
                                                  const double *surface_weights,
                                                  double values[const static value_count]);

/**
 * @brief Assemble a two-sided physical trace constraint matrix.
 *
 * Side 0 entries carry a positive sign and side 1 entries a negative sign, so
 * each row expresses equality of the two physical traces. The two
 * quadratures, surface-measure arrays, and pullback tables may differ.
 *
 * @param test_spec Test-space specification.
 * @param sides Two element-side specifications.
 * @param quadrature Two face quadrature descriptions.
 * @param surface_weights Two arrays of unsigned face measures.
 * @param pullbacks Two sampled pullbacks; required for non-zero order.
 * @param row_offset_capacity Capacity of `row_offsets`.
 * @param row_offsets Output offsets with room for `row_count + 1` entries.
 * @param entry_capacity Capacity of `entries`.
 * @param entries Output packed entries.
 * @param out_row_count Receives the row count.
 * @param out_entry_count Receives the entry count.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INSUFFICIENT_STORAGE`, or an
 *         input/quadrature/pullback/overflow error. Counts are written only on
 *         success.
 */
constraint_status_t constraint_physical_assemble(
    const kform_spec_t *test_spec, const constraint_element_side_t sides[const static 2],
    const constraint_face_quadrature_t quadrature[const static 2], const double *const surface_weights[const static 2],
    const constraint_trace_pullback_t pullbacks[const static 2], size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Compute storage requirements for a batch of physical trace matrices.
 *
 * Every item uses the same test specification. Returned counts are sums in
 * input order, with overflow checked before either output is written. A
 * zero-item batch is valid when the test specification is valid.
 *
 * @param test_spec Shared test-space specification.
 * @param item_count Number of items in `items`.
 * @param items Batch items; not dereferenced when the count is zero.
 * @param out_row_count Receives the summed row count.
 * @param out_entry_count Receives the summed entry count.
 * @return `CONSTRAINT_SUCCESS`, an item/specification validation error, or
 *         `CONSTRAINT_SIZE_OVERFLOW`. Outputs are unchanged on failure.
 */
constraint_status_t constraint_physical_batch_required(
    const kform_spec_t *test_spec, size_t item_count,
    const constraint_physical_batch_item_t items[const static item_count], size_t *out_row_count,
    size_t *out_entry_count);

/**
 * @brief Assemble a batch of two-sided physical trace matrices.
 *
 * Items are concatenated in input order. Row offsets are rebased to the
 * combined entry array; each entry retains its item's side, component, local
 * DoF, and usual +1/-1 side sign.
 *
 * @param test_spec Shared test-space specification.
 * @param item_count Number of items in `items`.
 * @param items Batch items in output order.
 * @param row_offset_capacity Capacity of the combined row-offset array.
 * @param row_offsets Output combined packed-row offsets.
 * @param entry_capacity Capacity of the combined entry array.
 * @param entries Output combined packed entries.
 * @param out_row_count Receives the total row count.
 * @param out_entry_count Receives the total entry count.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INSUFFICIENT_STORAGE`, or an
 *         input/quadrature/pullback/overflow error. Earlier items may have
 *         written output before a later item reports an error.
 */
constraint_status_t constraint_physical_batch_assemble(
    const kform_spec_t *test_spec, size_t item_count,
    const constraint_physical_batch_item_t items[const static item_count], size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Compute the number of offsets required for packed rows.
 * @param row_count Number of rows.
 * @param out_count Receives `row_count + 1`.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INVALID_ARGUMENT` for a null
 *         output, or `CONSTRAINT_SIZE_OVERFLOW` when `row_count == SIZE_MAX`.
 */
constraint_status_t constraint_rows_required_offset_count(size_t row_count, size_t *out_count);

/**
 * @brief Compute packed-entry capacity for a rectangular row structure.
 * @param row_count Number of rows.
 * @param entries_per_row Number of entries in each row.
 * @param out_count Receives `row_count * entries_per_row`.
 * @return `CONSTRAINT_SUCCESS`, `CONSTRAINT_INVALID_ARGUMENT` for a null
 *         output, or `CONSTRAINT_SIZE_OVERFLOW` when the product overflows
 *         `size_t`.
 */
constraint_status_t constraint_rows_required_entry_capacity(size_t row_count, size_t entries_per_row,
                                                            size_t *out_count);

/**
 * @brief Validate a packed sparse-row constraint representation.
 *
 * For a non-empty row set, `row_offsets` must be non-null, start at zero, be
 * non-decreasing, and end at `entry_count`. If entries exist, `entries` must
 * be non-null and every entry must name side 0 or side 1. A zero-row view may
 * omit `row_offsets`; callers should use an entry count of zero for such a
 * view.
 *
 * @param view Borrowed packed-row representation to inspect.
 * @return `CONSTRAINT_SUCCESS` when consistent, or
 *         `CONSTRAINT_INVALID_ARGUMENT` otherwise.
 */
constraint_status_t constraint_rows_validate(constraint_rows_view_t view);
