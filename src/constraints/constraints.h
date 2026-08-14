#ifndef FDG_CONSTRAINTS_H
#define FDG_CONSTRAINTS_H

#include "../basis/basis_set.h"
#include <stddef.h>
#include <stdint.h>

typedef enum
{
    CONSTRAINT_SUCCESS = 0,
    CONSTRAINT_INVALID_ARGUMENT,
    CONSTRAINT_INVALID_DIMENSION,
    CONSTRAINT_INVALID_ORDER,
    CONSTRAINT_SIZE_OVERFLOW,
    CONSTRAINT_INSUFFICIENT_STORAGE,
} constraint_status_t;

typedef struct
{
    // Number of dimensions of the boundary test space.
    unsigned ndim;
    // Differential form degree. This must not exceed ndim.
    unsigned order;
    // Basis specifications for the boundary axes, in canonical face order.
    const basis_spec_t *basis_specs;
} constraint_kform_spec_t;

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

typedef struct
{
    // Number of quadrature points along one canonical face axis.
    unsigned count;
    const double *nodes;
    const double *weights;
} constraint_quadrature_t;

typedef struct
{
    unsigned ndim;
    const constraint_quadrature_t *axes;
    size_t point_count;
} constraint_face_quadrature_t;

typedef struct
{
    // Number of physical k-form components in each sampled pullback.
    unsigned physical_component_count;
    // Number of face quadrature points in the sampled pullback.
    size_t point_count;
    // Values are indexed as [element component][physical component][point].
    const double *values;
} constraint_trace_pullback_t;

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

typedef struct
{
    // Number of rows and entries in the packed representation.
    size_t row_count;
    size_t entry_count;
    const size_t *row_offsets;
    const constraint_entry_t *entries;
} constraint_rows_view_t;

const char *constraint_status_to_str(constraint_status_t status);
const char *constraint_status_msg(constraint_status_t status);

constraint_status_t constraint_kform_component_count(const constraint_kform_spec_t *spec, size_t *out_count);
constraint_status_t constraint_kform_component_dof_count(const constraint_kform_spec_t *spec, unsigned component,
                                                         size_t *out_count);
constraint_status_t constraint_kform_component_offsets(const constraint_kform_spec_t *spec, size_t offset_count,
                                                       size_t offsets[const static offset_count]);

constraint_status_t constraint_reference_required(const constraint_kform_spec_t *test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  size_t *out_row_count, size_t *out_entry_count);

// Assemble reference-space L2 trace constraints into caller-owned packed rows.
constraint_status_t constraint_reference_assemble(const constraint_kform_spec_t *test_spec,
                                                  const constraint_element_side_t sides[const static 2],
                                                  const constraint_quadrature_t *quadrature, size_t row_offset_capacity,
                                                  size_t row_offsets[const static row_offset_capacity],
                                                  size_t entry_capacity,
                                                  constraint_entry_t entries[const static entry_capacity],
                                                  size_t *out_row_count, size_t *out_entry_count);

constraint_status_t constraint_physical_required(const constraint_kform_spec_t *test_spec,
                                                 const constraint_element_side_t sides[const static 2],
                                                 size_t *out_row_count, size_t *out_entry_count);

constraint_status_t constraint_physical_side_required(const constraint_kform_spec_t *test_spec,
                                                      const constraint_element_side_t *side, size_t *out_row_count,
                                                      size_t *out_entry_count);

constraint_status_t constraint_physical_side_assemble(
    const constraint_kform_spec_t *test_spec, const constraint_element_side_t *side,
    const constraint_face_quadrature_t *quadrature, const double *surface_weights,
    const constraint_trace_pullback_t *pullback, size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

// Assemble physical trace pairings using unsigned face measures and side-specific pullbacks.
constraint_status_t constraint_physical_assemble(
    const constraint_kform_spec_t *test_spec, const constraint_element_side_t sides[const static 2],
    const constraint_face_quadrature_t quadrature[const static 2], const double *const surface_weights[const static 2],
    const constraint_trace_pullback_t pullbacks[const static 2], size_t row_offset_capacity,
    size_t row_offsets[const static row_offset_capacity], size_t entry_capacity,
    constraint_entry_t entries[const static entry_capacity], size_t *out_row_count, size_t *out_entry_count);

constraint_status_t constraint_rows_required_offset_count(size_t row_count, size_t *out_count);
constraint_status_t constraint_rows_required_entry_capacity(size_t row_count, size_t entries_per_row,
                                                            size_t *out_count);
constraint_status_t constraint_rows_validate(constraint_rows_view_t view);

#endif // FDG_CONSTRAINTS_H
