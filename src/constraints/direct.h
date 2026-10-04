/**
 * @file direct.h
 * @brief Direct continuity: shared objects' DoFs are explicit unknowns rather than eliminated multipliers.
 *
 * The hybridized formulation (constraints.h) keeps every element's local DoFs and adds a multiplier per
 * interface equation. This one replaces each element-local DoF that a shared object carries by the object's own
 * DoF and transfers the element matrices straight onto that numbering. What no object reaches stays element
 * private, so the global space is element-interior DoFs plus, for every object of dimension `ndim - 1` down to
 * `0`, the DoFs interior to that object: faces, edges, nodes. That is the fewest unknowns that still enforce
 * continuity exactly.
 *
 * @section direct_ownership Which object owns a DoF
 *
 * Component `I` of a `k`-form is `dxi_I` tensored with the element's function space: a covector axis `a` of `I`
 * reads the order-one basis (`order` functions), any other axis reads the full basis (`order + 1`). Only an
 * endpoint function lives on an object, so component `I` reaches the face perpendicular to axis `a` exactly when
 * `a` is not a covector axis and its digit is `0` (start) or `order` (end). Those faces name the smallest object
 * carrying the DoF, and it is numbered there. Because the endpoints are nodal, this needs a Lagrange family;
 * orthogonal and Bernstein bases do not localize and are rejected. Lifting that is deferred: a non-localizing
 * family needs its own decomposition in place of node support.
 *
 * @section direct_common An object's common space
 *
 * Per axis the order is the minimum over the incident elements, read from the object's free axes, so the block
 * is the largest space every incident element can represent. An axis carrying a covector keeps its whole
 * order-one space; an axis without one keeps only the functions between the endpoints. An element above the
 * minimum enters by L2 projection, and where the common space is empty on an axis the element-local DoF reaches
 * no global DoF and leaves the system.
 *
 * That projection is the weakest path in the module: it is a small Gram solve, and an equispaced Lagrange Gram
 * matrix is badly scaled enough to overflow unscaled. The solve is equilibrated for that reason; a stable
 * formulation that avoids it is deferred to a later session.
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
    basis_set_registry_t *basis_registry;              ///< Registry the transfer's one-dimensional bases come from.
    integration_rule_registry_t *integration_registry; ///< Registry the transfer's quadrature comes from.
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
 * #direct_continuity_work_init points every member into it.
 */
typedef struct
{
    double *axis_matrix;       ///< Backing store of the one-dimensional transfer operator.
    unsigned capacity;         ///< Longest side of a one-dimensional operator; sized by the largest basis order.
    double *vectors;           ///< [capacity * capacity] Right-hand sides of the dense solve.
    unsigned *pivot;           ///< [capacity] Pivot scratch of the dense solve.
    double *scale;             ///< [capacity] Row scaling of the dense solve.
    uint8_t *component_axes;   ///< [max(order, 1)] Covector axes of the current element component.
    uint8_t *object_axes;      ///< [max(order, 1)] The same axes in the object's frame.
    uint8_t *mapped_axes;      ///< [max(order, 1)] Scratch for the orientation sign helper.
    int8_t *fixed_axes;        ///< [ndim] Signed one-based axes the object is pinned to in the element.
    unsigned *element_counts;  ///< [ndim] Function count of every element axis.
    unsigned *object_counts;   ///< [ndim] Function count of every object axis.
    unsigned *element_digits;  ///< [ndim] Current element digit tuple.
    unsigned *object_digits;   ///< [ndim] Current object digit tuple.
    unsigned *support_rows;    ///< [ndim * capacity] Object rows one element digit reaches.
    double *support_values;    ///< [ndim * capacity] Their coefficients.
    unsigned *support_counts;  ///< [ndim] Entries in each support.
    size_t *element_strides;   ///< [ndim + 1] Row-major strides of the element's digit tuple.
    size_t *object_strides;    ///< [ndim + 1] Row-major strides of the object's digit tuple.
    size_t *component_offsets; ///< [C(ndim, order) + 1] Element component offsets.
} direct_continuity_work_t;

/**
 * @brief Intermediates of one direct continuity map.
 */
typedef struct
{
    unsigned ndim;                                     ///< Element dimension.
    unsigned order;                                    ///< Traced k-form order.
    uint64_t element_count;                            ///< Number of elements.
    uint64_t entity_count;                             ///< Number of shared objects of every dimension.
    const topo_mesh_t *mesh;                           ///< Borrowed mesh.
    const kform_spec_t *const *elements;               ///< Borrowed per-element spec.
    basis_set_registry_t *basis_registry;              ///< Borrowed basis registry.
    integration_rule_registry_t *integration_registry; ///< Borrowed quadrature registry.
    uint64_t *entity_dim_offsets;                      ///< [ndim + 1] Flat object numbering offsets.
    basis_spec_t *entity_basis;                        ///< [entity_count * ndim] Common per-axis basis.
    size_t *entity_block_offsets;                      ///< [entity_count + 1] Object block offsets.
    size_t *element_dof_offsets;                       ///< [element_count + 1] Local DoF offsets.
    size_t *element_interior_offsets;                  ///< [element_count + 1] Element-private DoF offsets.
    direct_continuity_layout_t layout;                 ///< Sizes from #direct_continuity_layout.
} direct_continuity_plan_t;

/**
 * @brief Marks an element-local DoF with no transfer entry.
 *
 * An element above its neighbours' common order can project onto an empty common space, and such a DoF leaves
 * the system. The walk still has to account for it so that the row compression stays a valid partition.
 */
#define DIRECT_NO_ENTRY ((size_t)-1)

/**
 * @brief Emit one nonzero of the element-to-global transfer.
 *
 * Called once per nonzero, and once with #DIRECT_NO_ENTRY for a DoF that reaches no global DoF.
 *
 * @param param Caller data of the enumeration.
 * @param local Element-local flat DoF index.
 * @param global Global DoF index, or #DIRECT_NO_ENTRY.
 * @param value Transfer coefficient.
 */
typedef void (*direct_entry_fn)(void *param, size_t local, size_t global, double value);

/**
 * @brief Prepare the intermediates of a direct continuity map.
 *
 * Records every object's canonical axis order and common basis, sizes each object's block and the elements'
 * local and element-private DoF ranges. Takes no registry reference.
 *
 * @param request Filled request; read-only.
 * @param work Caller-provided scratch; sized by #direct_continuity_work_memory.
 * @param plan Caller-allocated plan; its arrays are sized by #direct_continuity_plan_memory.
 * @return FDG_SUCCESS, or #FDG_ERROR_NOT_IN_DOMAIN for a basis that is not a nodal family of positive order.
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
 * global DoF of each and `entry_value` its coefficient. A space-matching object gives exactly one entry, of
 * coefficient `+1` or `-1`; a richer element gives a weighted combination.
 *
 * @param request Filled request; read-only.
 * @param plan Prepared plan; read-only.
 * @param work Caller-provided scratch.
 * @param entry_offsets [element_dof_count + 1] Entry offset of each element-local DoF.
 * @param entry_index [entry_count] Global DoF of each entry.
 * @param entry_value [entry_count] Coefficient of each entry.
 */
void direct_continuity_build(const direct_continuity_request_t *request, const direct_continuity_plan_t *plan,
                             direct_continuity_work_t *work, size_t *entry_offsets, size_t *entry_index,
                             double *entry_value);

/**
 * @brief Transfer one element matrix onto the global numbering.
 *
 * Accumulates `out[g_i][g_j] += factor * value_i * matrix[i][j] * value_j` through the transfer of `element`,
 * so a DoF with several entries adds the weighted combination.
 *
 * @param plan Prepared plan; read-only.
 * @param entry_offsets [element_dof_count + 1] Entry offsets from #direct_continuity_build.
 * @param entry_index [entry_count] Global DoF of each entry.
 * @param entry_value [entry_count] Coefficient of each entry.
 * @param element Element whose local numbering `matrix` uses.
 * @param matrix Element matrix.
 * @param stride Row stride of `matrix`, at least that element's local DoF count.
 * @param out Global matrix.
 * @param out_stride Row stride of `out`, at least `plan->layout.global_dof_count`.
 * @param factor Scalar multiplying every contribution.
 */
void direct_continuity_scatter(const direct_continuity_plan_t *plan, const size_t *entry_offsets,
                               const size_t *entry_index, const double *entry_value, size_t element,
                               const double *matrix, size_t stride, double *out, size_t out_stride, double factor);

/**
 * @brief Release a prepared plan.
 *
 * A plan takes no registry reference, so this only clears it for symmetry with the other formulations.
 *
 * @param plan Prepared plan; invalid on return.
 */
void direct_continuity_plan_release(const direct_continuity_plan_t *plan);

/**
 * @brief Bytes of the scratch #direct_continuity_prepare and #direct_continuity_build need.
 *
 * @param request Filled request; reads its dimensions, the mesh's element count and every element's per-axis
 *        basis order, because the transfer operators are as wide as the basis is.
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
 * @param request Filled request; reads its dimensions and its mesh's object counts.
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
 *
 * @param plan Prepared plan; read-only.
 * @param dim Object dimension, in `[0, plan->ndim)`.
 * @param object_id Object ID in its dimension's collection.
 */
static inline size_t direct_entity_index(const direct_continuity_plan_t *plan, unsigned dim, uint64_t object_id)
{
    return (size_t)plan->entity_dim_offsets[dim] + (size_t)object_id;
}
