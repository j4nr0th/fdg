#ifndef FDG_PYTHON_CONSTRAINTS_H
#define FDG_PYTHON_CONSTRAINTS_H

#include "kform_objects.h"
#include "mappings.h"
#include "module.h"

/**
 * @brief Restricted face map plus the canonical/source quadrature frames of
 *        one element boundary.
 *
 * The face map is the volume map restricted to the face in one values-level
 * pass; its own integration specs are the source frame, and the canonical
 * frame is that frame permuted to canonical test-axis order.
 */
typedef struct
{
    PyObject *face_object;                      // Restricted space map (owned reference).
    space_map_object *face_map;                 // Borrowed alias of face_object.
    const integration_rule_t **source_rules;    // Rules of the source-frame face axes.
    const integration_rule_t **canonical_rules; // Rules permuted to canonical test axes.
    integration_spec_t *canonical_specs;        // Canonical axis specs.
    size_t *canonical_strides;                  // Canonical row-major point strides.
    size_t *source_strides;                     // Source-frame row-major point strides.
    double *point_weights;                      // Canonical tensor quadrature weights.
    size_t point_count;                         // Total canonical face points.
    void *memory;
} boundary_face_setup_t;

FDG_INTERNAL
int make_boundary_face_setup(const interplib_module_state_t *state, const space_map_object *element_map,
                             const int8_t *orientation, const unsigned element_dim, const unsigned face_dim,
                             boundary_face_setup_t *setup);
FDG_INTERNAL
void release_boundary_face_setup(const interplib_module_state_t *state, const unsigned face_dim,
                                 boundary_face_setup_t *setup);

#endif // FDG_PYTHON_CONSTRAINTS_H
