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

#include "../basis/basis_set.h"
#include "../kforms/kform_types.h"
#include "../kforms/kform_values.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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
 * @param ndim Dimensionality of the elements' space.
 * @param nelem Number of elements the boundary is contained in.
 * @param bdim Dimensionality of the boundary (face) space.
 * @param orientation Array of signed one-based axis mappings for each element.
 * @param element_basis Array of element-axis basis specifications for each element.
 * @param element_integration Array of element-axis integration rules for each element.
 * @param boundary_basis Output array of boundary-axis basis specifications.
 * @param boundary_integration Output array of boundary-axis integration rules.
 */
void boundary_common_space(unsigned ndim, unsigned nelem, unsigned bdim, const int8_t *orientation[static ndim],
                           const basis_spec_t *element_basis[static ndim],
                           const integration_spec_t *element_integration[static ndim],
                           basis_spec_t boundary_basis[bdim], integration_spec_t boundary_integration[bdim]);

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
