/**
 * @file direct.h
 * @brief Direct shared-DoF continuity: boundary DoFs introduced, element DoFs eliminated by L2 projection.
 *
 * Every mesh object of dimension below the element dimension carries one global unknown per function of its
 * common test space: the windowed Legendre space the hybridized trace constraints pair against, of per-axis
 * order the minimum over the incident elements. Each element's constraint rows are its objects' boundary mass
 * blocks stacked, and a QR of the transpose eliminates the element's DoFs against the object unknowns: the
 * constrained part maps the object coefficients, the orthogonal complement stays element-private. Any basis
 * family works; a basis order of zero is rejected as degenerate.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../common/error.h"
#include "../kforms/kform_types.h"
#include "../topology/mesh.h"
#include "constraints.h"

/**
 * @brief Inputs of one direct continuity map.
 */
typedef struct
{
    unsigned ndim;                                     ///< Element dimension, at least one.
    unsigned order;                                    ///< Traced k-form order, at most `ndim`.
    const topo_mesh_t *mesh;                           ///< Hypercubic mesh; borrowed, must outlive the map.
    const kform_spec_t *const *elements;               ///< [element_count] Per-element k-form spec; borrowed.
    basis_set_registry_t *basis_registry;              ///< Registry the object and element tables come from.
    integration_rule_registry_t *integration_registry; ///< Registry the object quadrature comes from.
} direct_continuity_request_t;

/**
 * @brief Sizes of the arrays a direct continuity map fills.
 */
typedef struct
{
    size_t element_count;     ///< Number of elements.
    size_t entity_count;      ///< Number of shared objects of every dimension below `ndim`.
    size_t element_dof_count; ///< Sum of the elements' local DoF counts.
    size_t global_dof_count;  ///< Total size of the global unknown vector.
    size_t entry_count;       ///< Nonzeros of the element-to-global transfer.
} direct_continuity_layout_t;

/**
 * @brief Scratch of one direct continuity map.
 *
 * Caller-provided throughout; #direct_continuity_work_memory sizes one block and
 * #direct_continuity_work_init points every member into it. Buffers size off the worst element: `q_matrix` by
 * the largest local DoF count, `gram`, `b_stacked` and `y` by the largest stacked row count, the row-by-column
 * buffers by their product.
 */
typedef struct
{
    constraint_boundary_mass_work_t mass;     ///< Boundary-mass scratch, re-initialized per object and element.
    unsigned char *mass_memory;               ///< Backing block of #mass, sized for the worst pair.
    double *weights;                          ///< [max object points] Common tensor quadrature weights.
    double *block;                            ///< [pairs] One pair's dense assembly, rows by element trace DoFs.
    double *gram;                             ///< [rows * rows] One object's test Gram.
    double *stacked;                          ///< [pairs] The element's stacked constraints, rows by element DoFs.
    double *b_stacked;                        ///< [rows * rows] The objects' Grams block-diagonal.
    double *transposed;                       ///< [pairs] The stacked transpose the QR reduces.
    double *q_matrix;                         ///< [element * element] Orthogonal factor of the transpose.
    double *y;                                ///< [rows * rows] Solved lower-triangular system.
    double *mapped;                           ///< [pairs] The constrained part of the transfer.
    bool *axis_fixed;                         ///< [ndim] Fixed normal axis classification of the current pair.
    unsigned *axis_slot;                      ///< [ndim] Canonical object slot of every free axis.
    const integration_rule_t **element_rules; ///< [ndim] The pair's common rules in element axis order.
    boundary_element_space_t *views;          ///< [max incident] Merge views of one object's elements.
    integration_spec_t *view_integration;     ///< [max incident * ndim] Per-view rules for the merge.
    size_t *component_offsets;                ///< [C(ndim, order) + 1] Element component offsets.
} direct_continuity_work_t;

/**
 * @brief Intermediates of one direct continuity map.
 *
 * The object arrays live per shared object, indexed by #direct_entity_index and strided by `ndim` with only the
 * object's dimension entries valid. The pair arrays live per (object, incident element) pair, packed by object in
 * canonical order; every element's pairs are listed in canonical order through #element_object_offsets.
 */
typedef struct
{
    unsigned ndim;                                       ///< Element dimension.
    unsigned order;                                      ///< Traced k-form order.
    uint64_t element_count;                              ///< Number of elements.
    uint64_t entity_count;                               ///< Shared objects of every dimension below `ndim`.
    const topo_mesh_t *mesh;                             ///< Borrowed mesh.
    const kform_spec_t *const *elements;                 ///< Borrowed per-element spec.
    basis_set_registry_t *basis_registry;                ///< Borrowed basis registry.
    integration_rule_registry_t *integration_registry;   ///< Borrowed quadrature registry.
    uint64_t *entity_dim_offsets;                        ///< [ndim + 1] Flat object numbering offsets.
    basis_spec_t *entity_basis;                          ///< [entity_count * ndim] Common Legendre test basis.
    integration_spec_t *entity_integration;              ///< [entity_count * ndim] Common rules.
    basis_spec_t *entity_lower_basis;                    ///< [entity_count * ndim] Order-1 test basis.
    size_t *entity_block_offsets;                        ///< [entity_count + 1] Object block offsets.
    const integration_rule_t **entity_rules;             ///< [entity_count * ndim] Fetched common rules.
    const basis_set_t **entity_sets;                     ///< [entity_count * ndim] Fetched test tables.
    const basis_set_t **entity_sets_lower;               ///< [entity_count * ndim] Order-one tables or NULL.
    uint64_t pair_count;                                 ///< Number of (object, incident element) pairs.
    uint64_t *pair_entities;                             ///< [pair_count] Flat object index of every pair.
    const int8_t **pair_records;                         ///< [pair_count] The element's orientation record.
    size_t *pair_rows;                                   ///< [pair_count] Pair row offset in its element's stack.
    const basis_set_t **pair_element_sets;               ///< [pair_count * ndim] Element tables on the common rules.
    const basis_set_t **pair_element_sets_lower;         ///< [pair_count * ndim] Order-1 tables, NULL for order 0.
    const basis_endpoint_set_t **pair_element_endpoints; ///< [pair_count * ndim] Fixed-axis endpoints or NULL.
    const basis_endpoint_set_t **pair_element_endpoints_lower; ///< [pair_count * ndim] Order-1 endpoints, NULL.
    basis_spec_t *pair_element_lower_specs;                    ///< [pair_count * ndim] Order-1 element specs.
    size_t *element_object_offsets;                            ///< [element_count + 1] Element's pair list start.
    size_t *element_pair_slots;                                ///< [pair_count] Pair slots of each element's list.
    size_t *element_rows;                                      ///< [element_count + 1] Stacked constraint row offsets.
    size_t *element_dof_offsets;                               ///< [element_count + 1] Local DoF offsets.
    size_t
        *element_interior_offsets; ///< [element_count + 1] Global first-private-DoF index; last entry: private total.
    direct_continuity_layout_t layout; ///< Sizes from #direct_continuity_layout.
} direct_continuity_plan_t;

/**
 * @brief Emit one nonzero of the element-to-global transfer.
 *
 * Called once per nonzero of every element-local DoF, grouped by ascending local index and ascending global
 * index. Every DoF owns at least one entry.
 *
 * @param param Caller data of the enumeration.
 * @param local Element-local flat DoF index.
 * @param global Global DoF index.
 * @param value Transfer coefficient.
 */
typedef void (*direct_entry_fn)(void *param, size_t local, size_t global, double value);

/**
 * @brief Prepare the intermediates of a direct continuity map.
 *
 * Merges every incident element's space into one common Legendre test space per object, fetches the object and
 * element tables from the registries, and sizes each object's block and the elements' local and private DoF
 * ranges. The plan owns the fetched registry references until #direct_continuity_plan_release.
 *
 * @param request Filled request; read-only.
 * @param work Caller-provided scratch; sized by #direct_continuity_work_memory.
 * @param plan Caller-allocated plan; its arrays are sized by #direct_continuity_plan_memory.
 * @return FDG_SUCCESS, or #FDG_ERROR_NOT_IN_DOMAIN for a basis order of zero, which leaves no test functions.
 *         Reported rather than asserted so the precondition survives a release build.
 */
fdg_result_t direct_continuity_prepare(const direct_continuity_request_t *request, direct_continuity_work_t *work,
                                       direct_continuity_plan_t *plan);

/**
 * @brief Count the arrays of a direct continuity map.
 *
 * Runs the enumeration of #direct_continuity_build and counts its nonzeros.
 *
 * @param request Filled request; read-only.
 * @param plan Prepared plan; its `layout` is completed in place and `out_layout` mirrors it.
 * @param work Caller-provided scratch.
 * @param out_layout Receives the array sizes.
 */
void direct_continuity_layout(const direct_continuity_request_t *request, direct_continuity_plan_t *plan,
                              direct_continuity_work_t *work, direct_continuity_layout_t *out_layout);

/**
 * @brief Build the element-to-global transfer of a direct continuity map.
 *
 * Writes it row-compressed: `entry_offsets` delimits one element-local DoF's global DoFs, `entry_index` the
 * global DoF of each and `entry_value` its coefficient. Coefficients below a relative roundoff threshold are
 * dropped. Only top-order forms come out as plain +/-1 identities; lower orders mix object and private modes.
 *
 * @param request Filled request; read-only.
 * @param plan Prepared plan.
 * @param work Caller-provided scratch.
 * @param entry_offsets [element_dof_count + 1] Row-block starts; the last entry is `entry_count`.
 * @param entry_index [entry_count] Global DoF of every nonzero.
 * @param entry_value [entry_count] Coefficient of every nonzero.
 */
void direct_continuity_build(const direct_continuity_request_t *request, const direct_continuity_plan_t *plan,
                             direct_continuity_work_t *work, size_t *entry_offsets, size_t *entry_index,
                             double *entry_value);

/**
 * @brief Transfer one element matrix onto the global numbering.
 *
 * Applies `out[g_i][g_j] += factor * m_i_j * v_i * v_j` over the element's transfer entries.
 *
 * @param plan Prepared plan.
 * @param entry_offsets Row-compressed starts from #direct_continuity_build.
 * @param entry_index Global DoF of every nonzero.
 * @param entry_value Coefficient of every nonzero.
 * @param element Element to scatter.
 * @param matrix [local_count * stride] Element matrix, row-major.
 * @param stride Column stride of `matrix`.
 * @param out Global matrix, row-major.
 * @param out_stride Column stride of `out`.
 * @param factor Extra scalar.
 */
void direct_continuity_scatter(const direct_continuity_plan_t *plan, const size_t *entry_offsets,
                               const size_t *entry_index, const double *entry_value, size_t element,
                               const double *matrix, size_t stride, double *out, size_t out_stride, double factor);

/**
 * @brief Release a prepared plan.
 *
 * Returns every registry reference #direct_continuity_prepare fetched. The plan arrays themselves belong to the
 * caller.
 */
void direct_continuity_plan_release(const direct_continuity_plan_t *plan);

/**
 * @brief Bytes of the scratch #direct_continuity_prepare and #direct_continuity_build need.
 *
 * Reads the request's dimensions, every element's basis orders, and the mesh's object and incidence counts.
 */
size_t direct_continuity_work_memory(const direct_continuity_request_t *request);

/**
 * @brief Point every member of one work struct into one memory block.
 *
 * @param work Work struct filled on return.
 * @param request Filled request.
 * @param memory Block of #direct_continuity_work_memory bytes.
 */
void direct_continuity_work_init(direct_continuity_work_t *work, const direct_continuity_request_t *request,
                                 void *memory);

/**
 * @brief Bytes of the arrays #direct_continuity_prepare fills.
 *
 * @param request Filled request; reads its dimensions and its mesh's object and incidence counts.
 */
size_t direct_continuity_plan_memory(const direct_continuity_request_t *request);

/**
 * @brief Point every array of one plan into one memory block.
 *
 * @param plan Plan struct filled on return.
 * @param request Filled request.
 * @param memory Block of #direct_continuity_plan_memory bytes.
 */
void direct_continuity_plan_init(direct_continuity_plan_t *plan, const direct_continuity_request_t *request,
                                 void *memory);

/**
 * @brief Flat index of one shared object in the plan's numbering.
 */
static inline size_t direct_entity_index(const direct_continuity_plan_t *plan, unsigned dim, uint64_t object_id)
{
    return (size_t)plan->entity_dim_offsets[dim] + (size_t)object_id;
}
