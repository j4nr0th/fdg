#ifndef FDG_CONSTRAINTS_H
#define FDG_CONSTRAINTS_H

#include "../basis/basis_set.h"
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Status and error codes returned by the constraint functions.
 *
 * A value of CONSTRAINT_SUCCESS means the operation completed successfully;
 * any other value is an error and describes why the operation failed.
 */
typedef enum
{
    CONSTRAINT_SUCCESS = 0,       // The operation completed successfully.
    CONSTRAINT_INVALID_ARGUMENT,  // An argument was invalid, such as a null pointer or an inconsistent specification.
    CONSTRAINT_INVALID_DIMENSION, // A dimension was out of the supported range.
    CONSTRAINT_INVALID_ORDER,     // A basis order or form degree was invalid.
    CONSTRAINT_SIZE_OVERFLOW,     // A required size calculation overflowed the size_t range.
    CONSTRAINT_INSUFFICIENT_STORAGE, // An output buffer was too small for the requested result.
} constraint_status_t;

/**
 * @brief Specification of the test k-form space on the boundary face.
 *
 * The basis specifications are given for the canonical face axes, in the
 * order the axes appear in the face.
 */
typedef struct
{
    // Number of dimensions of the boundary test space.
    unsigned ndim;
    // Differential form degree. This must not exceed ndim.
    unsigned order;
    // Basis specifications for the boundary axes, in canonical face order.
    const basis_spec_t *basis_specs;
} constraint_kform_spec_t;

/**
 * @brief Specification of one side (element) of the interface.
 *
 * The side has more dimensions than the test space: the first
 * `ndim - test ndim` fixed axes are normal to the face, and the remaining
 * axes span the face itself.
 */
typedef struct
{
    // Number of dimensions of the element.
    unsigned ndim;
    // Tensor-product basis specifications for the element axes.
    const basis_spec_t *basis_specs;
    // Signed one-based orientation: entry 0 is the fixed normal axis; entries 1..ndim-1
    // map canonical face axes to element axes. Negative entries reverse orientation.
    const int8_t *orientation;
} constraint_element_side_t;

/**
 * @brief One-dimensional quadrature rule along a single canonical face axis.
 */
typedef struct
{
    // Number of quadrature points along one canonical face axis.
    unsigned count;
    const double *nodes;
    const double *weights;
} constraint_quadrature_t;

/**
 * @brief Tensor-product quadrature rule over a face.
 *
 * The total number of points is the product of the node counts of the
 * individual axes and is cached in `point_count`.
 */
typedef struct
{
    unsigned ndim;
    const constraint_quadrature_t *axes;
    size_t point_count;
} constraint_face_quadrature_t;

/**
 * @brief Sampled pullback of the physical k-form onto the face.
 *
 * The values are laid out so that the component for physical component
 * `p` at point `i` of component `c` is stored at
 * `values[c * physical_component_count * point_count + p * point_count + i]`.
 */
typedef struct
{
    // Number of physical k-form components in each sampled pullback.
    unsigned physical_component_count;
    // Number of face quadrature points in the sampled pullback.
    size_t point_count;
    // Values are indexed as [element component][physical component][point].
    const double *values;
} constraint_trace_pullback_t;

/**
 * @brief Precomputed tensor-product basis values for trace assembly.
 *
 * Values are laid out by component, then point, then local degree of freedom:
 * ``values[offsets[c] * point_count + point * dofs[c] + dof]``. The offsets
 * array has ``component_count + 1`` entries and stores cumulative DoF counts.
 */
typedef struct
{
    size_t component_count;
    size_t point_count;
    const size_t *component_offsets;
    const double *values;
} constraint_trace_basis_values_t;

/**
 * @brief Single non-zero entry of an assembled constraint matrix row.
 *
 * The entry couples one degree of freedom of the test space (the row) with
 * one degree of freedom of an element side (the column).
 */
typedef struct
{
    // 0 for the first element side, 1 for the second.
    uint8_t side;
    // Lexicographic k-form component index on the element side.
    unsigned component;
    // Flattened DoF index within that component.
    size_t local_dof;
    double coefficient;
} constraint_entry_t;

/**
 * @brief Read-only view of an assembled constraint matrix in packed row form.
 *
 * The rows are indexed by the degrees of freedom of the test space, in the
 * same order as the offsets computed by constraint_kform_component_offsets.
 * The entries of row `i` are `entries[row_offsets[i] .. row_offsets[i + 1])`.
 */
typedef struct
{
    // Number of rows and entries in the packed representation.
    size_t row_count;
    size_t entry_count;
    const size_t *row_offsets;
    const constraint_entry_t *entries;
} constraint_rows_view_t;

/**
 * @brief Get the name of a constraint status value.
 *
 * @param status Status value to get the name for.
 * @return Statically allocated string with the name of the status value,
 *         such as "CONSTRAINT_SUCCESS", or "Unknown" for values outside of
 *         the enum.
 */
const char *constraint_status_to_str(constraint_status_t status);

/**
 * @brief Get the description of a constraint status value.
 *
 * @param status Status value to get the message for.
 * @return Statically allocated string with a short description of what the
 *         status value means, or "Unknown" for values outside of the enum.
 */
const char *constraint_status_msg(constraint_status_t status);

/**
 * @brief Compute the number of k-form components of the test space.
 *
 * @param spec Specification of the test space. The specification is
 *        validated; on failure the outputs are left unmodified.
 * @param out_count Receives the number of components, which is the binomial
 *        coefficient C(ndim, order).
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INVALID_ARGUMENT if
 *         `spec` or `out_count` is null, CONSTRAINT_INVALID_DIMENSION if
 *         `ndim` exceeds 255, or CONSTRAINT_INVALID_ORDER if the order
 *         exceeds the dimension or a basis order is zero or the basis type
 *         is invalid.
 */
constraint_status_t constraint_kform_component_count(const constraint_kform_spec_t *spec, size_t *out_count);

/**
 * @brief Compute the number of degrees of freedom of one k-form component.
 *
 * For a component with active axes (those in the wedge product) the basis
 * order along the active axes is reduced by one, so a component of order
 * `order` has `prod_i (order_i + (active_i ? 0 : 1))` degrees of freedom.
 *
 * @param spec Specification of the test space, validated as in
 *        constraint_kform_component_count.
 * @param component Index of the component, in the range
 *        [0, C(ndim, order)).
 * @param out_count Receives the number of degrees of freedom.
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INVALID_ARGUMENT if
 *         `out_count` is null or `component` is out of range,
 *         CONSTRAINT_INVALID_DIMENSION, CONSTRAINT_INVALID_ORDER, or
 *         CONSTRAINT_SIZE_OVERFLOW if the count does not fit in a size_t.
 */
constraint_status_t constraint_kform_component_dof_count(const constraint_kform_spec_t *spec, unsigned component,
                                                         size_t *out_count);

/**
 * @brief Compute the cumulative offsets of the components into a flattened DoF array.
 *
 * Fills `offsets[0..component_count]` with the start index of each
 * component in the flattened array of all test degrees of freedom; the last
 * entry holds the total number of degrees of freedom.
 *
 * @param spec Specification of the test space, validated as in
 *        constraint_kform_component_count.
 * @param offset_count Number of entries available in `offsets`. Must be at
 *        least the number of components plus one.
 * @param offsets Array of `offset_count` entries which receives the
 *        offsets.
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INSUFFICIENT_STORAGE if
 *         `offset_count` is too small, otherwise the validation or overflow
 *         errors of the functions above.
 */
constraint_status_t constraint_kform_component_offsets(const constraint_kform_spec_t *spec, size_t offset_count,
                                                       size_t offsets[const static offset_count]);

/**
 * @brief Compute the number of rows and entries of a reference-space L2 trace constraint matrix.
 *
 * The reference-space constraint equates, for each test degree of freedom,
 * the trace basis coefficients on both sides of the interface. This
 * function reports the sizes needed by constraint_reference_assemble
 * without assembling the matrix.
 *
 * @param test_spec Specification of the test space.
 * @param sides Array of 2 element side specifications. Each side must have
 *        more dimensions than the test space, with valid basis
 *        specifications and a valid orientation permutation.
 * @param out_row_count Receives the number of rows, one per test degree of
 *        freedom.
 * @param out_entry_count Receives the number of entries.
 * @return CONSTRAINT_SUCCESS on success, otherwise the validation error of
 *         the specifications or CONSTRAINT_SIZE_OVERFLOW if a required size
 *         does not fit in a size_t. On failure the outputs are unmodified.
 */
constraint_status_t constraint_reference_required(const constraint_kform_spec_t *test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble a reference-space L2 trace constraint matrix into caller-owned packed rows.
 *
 * The assembled matrix has one row per test degree of freedom. Each row
 * contains entries coupling to the degrees of freedom of the two element
 * sides, with the coefficient of side 0 being the basis inner product over
 * the face quadrature and the coefficient of side 1 its negative.
 *
 * @param test_spec Specification of the test space.
 * @param sides Array of 2 element side specifications, as validated by
 *        constraint_reference_required.
 * @param quadrature Tensor-product quadrature rule over the canonical face,
 *        with one `constraint_quadrature_t` per test dimension. Each axis
 *        must have a positive node count with valid nodes and weights.
 * @param row_offset_capacity Number of entries available in `row_offsets`.
 * @param row_offsets Array which receives the packed row offsets. Must have
 *        room for `row_count + 1` entries, where `row_count` is as reported
 *        by constraint_reference_required. `row_offsets[0]` is set to 0.
 * @param entry_capacity Number of entries available in `entries`.
 * @param entries Array which receives the packed entries.
 * @param out_row_count Receives the number of rows written.
 * @param out_entry_count Receives the number of entries written.
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INSUFFICIENT_STORAGE if
 *         the provided capacities are smaller than the required sizes,
 *         otherwise the validation errors of the inputs.
 */
constraint_status_t constraint_reference_assemble(const constraint_kform_spec_t *test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  const constraint_quadrature_t *quadrature, size_t row_offset_capacity,
                                                  size_t row_offsets[const static row_offset_capacity],
                                                  size_t entry_capacity,
                                                  constraint_entry_t entries[const static entry_capacity],
                                                  size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Compute the number of rows and entries of a physical-space trace constraint matrix.
 *
 * The physical-space constraint pairs the test degrees of freedom with the
 * element degrees of freedom through the pullback of the physical k-form
 * and the unsigned face measure, for both sides of the interface. This
 * function reports the sizes needed by constraint_physical_assemble without
 * assembling the matrix.
 *
 * @param test_spec Specification of the test space.
 * @param sides Array of 2 element side specifications.
 * @param out_row_count Receives the number of rows, one per test degree of
 *        freedom.
 * @param out_entry_count Receives the number of entries.
 * @return CONSTRAINT_SUCCESS on success, otherwise the validation error of
 *         the specifications or CONSTRAINT_SIZE_OVERFLOW. On failure the
 *         outputs are unmodified.
 */
constraint_status_t constraint_physical_required(const constraint_kform_spec_t *test_spec,
                                                 const constraint_element_side_t sides[const static 2],
                                                 size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Compute the number of rows and entries of a single-side physical-space trace constraint.
 *
 * Same as constraint_physical_required, but for one side only; the
 * resulting rows couple only to the degrees of freedom of that single side.
 *
 * @param test_spec Specification of the test space.
 * @param side Element side specification.
 * @param out_row_count Receives the number of rows, one per test degree of
 *        freedom.
 * @param out_entry_count Receives the number of entries.
 * @return CONSTRAINT_SUCCESS on success, otherwise the validation error of
 *         the specifications or CONSTRAINT_SIZE_OVERFLOW. On failure the
 *         outputs are unmodified.
 */
constraint_status_t constraint_physical_side_required(const constraint_kform_spec_t *test_spec,
                                                      const constraint_element_side_t *side, size_t *out_row_count,
                                                      size_t *out_entry_count);

/**
 * @brief Assemble a single-side physical-space trace constraint matrix.
 *
 * The assembled matrix has one row per test degree of freedom, with entries
 * coupling to the degrees of freedom of the given element side, weighted by
 * the face quadrature, the unsigned face measure (`surface_weights`) and
 * the sampled pullback of the physical k-form.
 *
 * @param test_spec Specification of the test space.
 * @param side Element side specification.
 * @param quadrature Tensor-product quadrature rule over the face. Its
 *        `point_count` must match the product of the axis node counts.
 * @param surface_weights Array of `point_count` unsigned face measures, one
 *        per quadrature point.
 * @param pullback Sampled pullback of the physical k-form; only used when
 *        the form order is non-zero. Must have `point_count` points. May be
 *        null for order zero.
 * @param row_offset_capacity Number of entries available in `row_offsets`.
 * @param row_offsets Array which receives the packed row offsets. Must have
 *        room for `row_count + 1` entries. `row_offsets[0]` is set to 0.
 * @param entry_capacity Number of entries available in `entries`.
 * @param entries Array which receives the packed entries, all with
 *        `side` set to 0.
 * @param out_row_count Receives the number of rows written.
 * @param out_entry_count Receives the number of entries written.
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INSUFFICIENT_STORAGE if
 *         the provided capacities are smaller than the required sizes,
 *         otherwise the validation errors of the inputs.
 */
constraint_status_t constraint_physical_side_assemble(
    const constraint_kform_spec_t *test_spec, const constraint_element_side_t *side,
    const constraint_face_quadrature_t *quadrature, const double *surface_weights,
    const constraint_trace_pullback_t *pullback, size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble a physical trace using caller-precomputed basis values.
 *
 * This has the same output and sign conventions as
 * @ref constraint_physical_side_assemble, but obtains both the test and
 * element trace basis values from registry-independent precomputed tables.
 *
 * @param test_basis Precomputed values for the canonical test components.
 * @param element_basis Precomputed values for all element components of the
 *        traced form.
 */
constraint_status_t constraint_physical_side_assemble_precomputed(
    const constraint_kform_spec_t *test_spec, const constraint_element_side_t *side,
    const constraint_face_quadrature_t *quadrature, const double *surface_weights,
    const constraint_trace_pullback_t *pullback, const constraint_trace_basis_values_t *test_basis,
    const constraint_trace_basis_values_t *element_basis, size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Assemble the physical-space boundary load of one element face for a
 *        general k-form datum.
 *
 * Computes the chain integral of the components of a k-form datum (element
 * frame, k = test_spec->order + 1) against the trace of the element
 * (k-1)-form basis on a codimension-1 boundary face. For each face component
 * J (element-frame axes J_e) the only contributing datum component is
 * I = J_e U {a} with a the fixed normal axis:
 *
 * ``values[j] = s * o * (-1)^{|{i in J_e : i < a}|} * sum_p w_p u_I(g_p) B_j(g_p)``
 * (chain integral), or, with ``surface_weights != NULL``, the same sum with
 * each weight multiplied by ``surface_weights[p]`` (surface-measure pairing).
 *
 * Here ``s`` and ``a`` are the side and index of the fixed normal axis, ``o``
 * is the orientation sign of the mapped component, ``w_p`` are the reference
 * face quadrature weights. When provided, ``surface_weights[p]`` is the
 * absolute mapped face Jacobian determinant at the same point. ``u_I(g_p)`` is
 * the sampled element-frame component of the datum at the canonical face points
 * and ``B_j`` is the element basis of the traced component. At k equal to the element dimension
 * (single datum component) the sign reduces to the outward orientation
 * ``s * (-1)^a`` of the previous scalar-chain-integral behavior.
 *
 * @param test_spec Specification of the (k-1)-form test space on the face.
 * @param side Element side specification with exactly one fixed normal axis.
 * @param quadrature Tensor-product quadrature over the canonical face.
 * @param datum_values Element-frame k-form components sampled at the canonical
 *        face quadrature points, laid out row-major as
 *        ``[component * quadrature->point_count + point]`` with
 *        ``combination_total_count(side->ndim, test_spec->order + 1)``
 *        component rows.
 * @param value_count Length of the output array: the total number of element
 *        (k-1)-form degrees of freedom.
 * @param surface_weights Mapped face Jacobian at the canonical points, or
 *        NULL to assemble the metric-free chain integral.
 * @param values Output array, accumulated over the mapped components.
 * @return `CONSTRAINT_SUCCESS` on success, or an error status.
 */
constraint_status_t constraint_physical_side_load(const constraint_kform_spec_t *test_spec,
                                                  const constraint_element_side_t *side,
                                                  const constraint_face_quadrature_t *quadrature,
                                                  const double *datum_values, size_t value_count,
                                                  const double *surface_weights,
                                                  double values[const static value_count]);

/**
 * @brief Assemble a physical-space trace constraint matrix for both sides of the interface.
 *
 * Same as constraint_physical_side_assemble, but assembles the rows for
 * both sides at once, with entries of side 0 weighted by +1 and entries of
 * side 1 weighted by -1, so that the assembled equations express the
 * equality of the traces of the two sides.
 *
 * @param test_spec Specification of the test space.
 * @param sides Array of 2 element side specifications.
 * @param quadrature Array of 2 face quadrature rules, one per side.
 * @param surface_weights Array of 2 pointers to the unsigned face measures,
 *        one array per side.
 * @param pullbacks Array of 2 sampled pullbacks of the physical k-form, one
 *        per side. Only used when the form order is non-zero.
 * @param row_offset_capacity Number of entries available in `row_offsets`.
 * @param row_offsets Array which receives the packed row offsets. Must have
 *        room for `row_count + 1` entries. `row_offsets[0]` is set to 0.
 * @param entry_capacity Number of entries available in `entries`.
 * @param entries Array which receives the packed entries.
 * @param out_row_count Receives the number of rows written.
 * @param out_entry_count Receives the number of entries written.
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INSUFFICIENT_STORAGE if
 *         the provided capacities are smaller than the required sizes,
 *         otherwise the validation errors of the inputs.
 */
constraint_status_t constraint_physical_assemble(
    const constraint_kform_spec_t *test_spec, const constraint_element_side_t sides[const static 2],
    const constraint_face_quadrature_t quadrature[const static 2], const double *const surface_weights[const static 2],
    const constraint_trace_pullback_t pullbacks[const static 2], size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

/**
 * @brief Compute the number of row offsets needed for a packed row representation.
 *
 * @param row_count Number of rows.
 * @param out_count Receives `row_count + 1`.
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INVALID_ARGUMENT if
 *         `out_count` is null, CONSTRAINT_SIZE_OVERFLOW if `row_count` is
 *         SIZE_MAX.
 */
constraint_status_t constraint_rows_required_offset_count(size_t row_count, size_t *out_count);

/**
 * @brief Compute the entry capacity needed for a packed row representation.
 *
 * @param row_count Number of rows.
 * @param entries_per_row Number of entries per row.
 * @param out_count Receives `row_count * entries_per_row`.
 * @return CONSTRAINT_SUCCESS on success, CONSTRAINT_INVALID_ARGUMENT if
 *         `out_count` is null, CONSTRAINT_SIZE_OVERFLOW if the product does
 *         not fit in a size_t.
 */
constraint_status_t constraint_rows_required_entry_capacity(size_t row_count, size_t entries_per_row,
                                                            size_t *out_count);

/**
 * @brief Validate a packed row representation of a constraint matrix.
 *
 * Checks that the offsets are non-decreasing, start at zero and end at the
 * entry count, that the entries are non-null when the counts are non-zero,
 * and that every entry references a valid side.
 *
 * @param view View of the packed rows to validate.
 * @return CONSTRAINT_SUCCESS if the representation is consistent,
 *         CONSTRAINT_INVALID_ARGUMENT otherwise.
 */
constraint_status_t constraint_rows_validate(constraint_rows_view_t view);

#endif // FDG_CONSTRAINTS_H
