#pragma once
#include <cutl/allocators.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Enum with success and error codes for topological functions.
 */
typedef enum
{
    TOPO_SUCCESS = 0,                // Success.
    TOPO_FAILED_ALLOC,               // Failed memory allocation.
    TOPO_NO_COMMON_BOUNDARY,         // Two non-opposite boundaries in an object have no common boundary.
    TOPO_INVALID_PARENT_BOUNDARIES,  // Parent object had invalid orientation with repeating indices.
    TOPO_INVALID_ELEMENT,            // Objects in an element did not appear as often as expected.
    TOPO_MULTIPLE_COMMON_BOUNDARIES, // Two elements share more than one boundary object.
} topo_status_t;

/**
 * Get the name of the status value (such as "TOPO_SUCCESS", for example).
 *
 * @param status Status value to get the string for.
 * @return Statically allocated string with the name of the status value.
 */
const char *topo_status_to_str(topo_status_t status);

/**
 * Get the description of what the status value is describing.
 *
 * @param status Status value to get the message for.
 * @return Statically allocated string with the message explaining the meaning of the status value.
 */
const char *topo_status_msg(topo_status_t status);

/**
 * @brief Collection of n-dimensional topological objects.
 *
 * The elements themselves are defined by their boundaries. Additional
 * information that is also needed, such as IDs of the elements they are in,
 * is also provided.
 */
typedef struct
{
    // Dimensionality of objects in the collection.
    unsigned ndim;
    // Number of objects in the collection.
    size_t count;
    // IDs of boundaries for all objects (count * bnd_cnt). These boundaries are specified
    // such that the boundary at index ``i`` is perpendicular to the axis ``i`` at the start
    // and the boundary at index ``i + ndim`` is perpendicular to it at the end.
    uint64_t *boundary_ids;
} topo_obj_collection_t;

/**
 * @brief Specification of how m-dimensional objects are immersed in n-dimensional space.
 *
 * This information is what n-dimensional elements they are contained within, what are the fixed element axis
 * of these objects, and how their local axes map to those of the element.
 */
typedef struct
{
    // Number of objects the immersion information is for.
    unsigned object_count;
    // Dimensionality of the space objects are immersed in.
    unsigned parent_dims;
    // Number of element indices before element_ids of the element begin
    uint64_t *element_offsets;
    // IDs of the parents of these elements.
    uint64_t *element_ids;
    // First ndim entries identify the object within the element, with the next (parent_dims - ndim)
    // specifying the mapping from its local axis to those of the element containing them.
    int8_t *element_orientation;
} topo_obj_immersion_t;

/**
 * Compute the number of boundaries the topological object has (hint: it is two times more than dimensions).
 *
 * This cool function multiplies by 2, but it is here because the code is a lot clearer when there's a proper name
 * for this.
 *
 * @param ndim[in] Dimensionality of the topological object.
 * @return Number of boundaries the object has.
 */
unsigned topo_obj_boundary_count(unsigned ndim);

/**
 * Find the common boundary of two objects from the same collection. The orientation of the boundary returned is as
 * it is within the first object. If the first object has reversed orientation, then the returned boundary has
 * its orientation reversed as well.
 *
 * @param collection[in] Collection the two objects belong to.
 * @param id_1[in] ID of the first object.
 * @param id_2[in] ID of the second object.
 * @return UINT64_MAX if the two share no boundary, otherwise an ID of the shared boundary, with the orientation
 *         as in the object with ID of ``id_1``.
 */
uint64_t topo_obj_common_boundary(const topo_obj_collection_t *collection, uint64_t id_1, uint64_t id_2);

/**
 * Find the index of the common boundary of two objects from the same collection.
 *
 * @param collection[in] Collection the two objects belong to.
 * @param id_1[in] ID of the first object.
 * @param id_2[in] ID of the second object.
 * @return Index of the common boundary in the element with ID ``id_1``, UINT64_MAX if there is no common boundary.
 */
uint64_t topo_obj_common_boundary_index(const topo_obj_collection_t *collection, uint64_t id_1, uint64_t id_2);

/**
 * Determine immersion information from topological description of elements in terms of boundaries. These
 * boundaries are in turn again described by their boundaries, and so on until 0-D objects (points).
 *
 * @param ndim[in] Number of dimensions of the space all objects are immersed in.
 * @param npts[in] Number of points in the mesh which do not have their own collection.
 * @param collections[in] Collections of objects going from 1-D (lines) to ndim-D (elements themselves).
 * @param allocator[in] Allocator to use to create the immersions in.
 * @param immersions[out] Array, which receives computed immersion information for objects from 0-D to (ndim-1)-D
 * @return TOPO_SUCCESS if successful, otherwise an error code.
 */
topo_status_t topo_obj_create_immersion_info(unsigned ndim, unsigned npts,
                                             const topo_obj_collection_t collections[static ndim],
                                             const cutl_allocator_t *allocator, topo_obj_immersion_t immersions[ndim]);

/**
 * Release all memory for immersions and clear them.
 *
 * The immersions must have been created by topo_obj_create_immersion_info
 * using the same allocator. After the call, the immersion structures are
 * zeroed out and all pointers into them are invalid.
 *
 * @param ndim Number of dimensions (and immersions).
 * @param immersions Immersions to release.
 * @param allocator Allocator with which the memory for immersions was allocated.
 */
void topo_obj_immersions_free(unsigned ndim, topo_obj_immersion_t immersions[const ndim],
                              const cutl_allocator_t *allocator);

/**
 * Create immersion information (position in the element and its relative orientation) for a boundary of an object from
 * a collection.
 *
 * @param ndim[in] Number of dimensions of the space everything is immersed in.
 * @param idim[in] Dimension of the boundary objects.
 * @param collection[in] Collection the object is from.
 * @param bdim[in] Index/dimension of the boundary in question.
 * @param fixed_axes[in] Number of axes indices used for identifying the boundary.
 * @param parent_orientation[in] The first (ndim-idim) entries identify the parent within the element, with the
 * remaining idim specifying how its axes map to those of the parent.
 * @param boundaries[in] Array of 1-based indices of other boundaries in the same topological object.
 * @param orient_arr[out] Array that receives the specification of the boundary in the element as the first
 * (ndim-idim) entries and the mapping of its local axes to those of the element as the final idim entries.
 * @return TOPO_SUCCESS if successful, TOPO_NO_COMMON_BOUNDARY if there are boundaries that do not share a boundary
 * among each other.
 */
topo_status_t topo_obj_boundary_immersion_create(unsigned ndim, unsigned idim, const topo_obj_collection_t *collection,
                                                 unsigned bdim, unsigned fixed_axes,
                                                 const int8_t parent_orientation[const static ndim],
                                                 const uint64_t boundaries[const static 2 * ndim],
                                                 int8_t orient_arr[const ndim]);

/**
 * Immersion information with IDs of elements an object is contained in.
 *
 * Information for each element consists of two parts:
 * - element IDs,
 * - orientation.
 *
 *  Orientation data consists of ``n`` 1-based, signed indices. If the object has ``m`` dimensions, then
 *  the first ``n-m`` entries specify the position of the object within the element. A negative number
 *  means it is at the start of the boundary, while a positive number means it is at the end of it.
 *  The remaining ``m`` entries describe the mapping between its own local coordinates and those of the
 *  element.
 *
 *  As such, when one iterates over different elements the object is in, the element ID array should
 *  be iterated one integer at a time, while the orientation array should instead advance ``n`` entries
 *  at a time.
 *
 * @param[in] immersion Immersion info of objects.
 * @param[in] object_id ID of the object to get the immersion for.
 * @param[out] p_cnt Pointer to the location where the size of the output array is stored.
 * @param[out] p_ids Pointer to the location where the element array pointer is stored.
 * @param[out] p_orientations Pointer to the location where the orientation array pointer is stored.
 */
void topo_obj_immersion_of_object(const topo_obj_immersion_t *immersion, uint64_t object_id, uint64_t *p_cnt,
                                  const uint64_t **p_ids, const int8_t **p_orientations);

/**
 * Get the orientation of one boundary object within an element.
 *
 * @param immersion Immersion information for codimension-one objects.
 * @param parent_dims Number of dimensions in the parent elements.
 * @param object_id Boundary object ID.
 * @param element_id Parent element ID.
 * @param orientation Receives the parent-dimension orientation record.
 * @return TOPO_SUCCESS if the object is in the element, otherwise TOPO_NO_COMMON_BOUNDARY.
 */
topo_status_t topo_obj_boundary_orientation(const topo_obj_immersion_t *immersion, unsigned parent_dims,
                                            uint64_t object_id, uint64_t element_id,
                                            int8_t orientation[const static parent_dims]);

/**
 * Find the unique immersed boundary object shared by two elements.
 *
 * The output orientations contain one parent-dimension orientation record for each element, in element ID order.
 *
 * @param immersion Immersion information for codimension-one objects.
 * @param parent_dims Number of dimensions in the parent elements.
 * @param element_id_1 ID of the first element.
 * @param element_id_2 ID of the second element.
 * @param p_object_id Receives the shared boundary object ID.
 * @param orientations Receives the two orientation records, with the first record at offset zero.
 * @return TOPO_SUCCESS, TOPO_NO_COMMON_BOUNDARY, or TOPO_MULTIPLE_COMMON_BOUNDARIES.
 */
topo_status_t topo_obj_find_common_boundary(const topo_obj_immersion_t *immersion, unsigned parent_dims,
                                            uint64_t element_id_1, uint64_t element_id_2, uint64_t *p_object_id,
                                            int8_t orientations[const static 2 * parent_dims]);

/**
 * Type with info and work memory needed to iterate over a boundary.
 * The boundary is in an element with ndim dimensions and has mdim free axes.
 */
typedef struct
{
    const int8_t *restrict orientation; // Orientation of the boundary (size: ndim).
    const uint64_t *restrict sizes;     // Sizes of each axis in the element's frame of reference (size: ndim)
    uint64_t *restrict offsets;         // Work array in which offsets for each axis will be held (size: mdim)
    uint64_t *restrict strides;         // Work array in which strides for each axis will be held (size: ndim)
} topo_bnd_iter_t;

/**
 * Reorder the input array associated with an object based on the specified orientation of the axes. Work arrays must be
 * provided.
 *
 * The input array is laid out with the axes in the order of the non-fixed
 * axes of the boundary; the output array is laid out in the element's frame
 * of reference, with the reordered and possibly reversed axes.
 *
 * @param ndim Total number of dimensions of the topological space.
 * @param mdim Number of non-fixed axes of the section the array represents.
 * @param bnd_iter Arrays with input data and working memory. Must contain an
 *        `orientation` array of `ndim` entries, `sizes` of `ndim` entries,
 *        and work arrays `offsets` of `mdim` and `strides` of `ndim`
 *        entries. The work arrays are overwritten.
 * @param in Input array that is to be reordered.
 * @param out Destination array, which receives the reordering. Must not
 *        alias `in`.
 */
void topo_reorder_with_orientation(unsigned ndim, unsigned mdim, topo_bnd_iter_t bnd_iter, const double in[restrict],
                                   double out[restrict]);

/**
 * Call a specific function with degree of freedom indices for a boundary with two different orientations.
 *
 * The callback is invoked once per position of the boundary, with the index
 * of the boundary DoF itself as well as the corresponding DoF indices in
 * both elements, followed by the user-provided pointer. The iteration stops
 * as soon as either side runs out of positions, so both elements must have
 * boundary sections of equal size.
 *
 * @param ndim Total number of dimensions of the topological space.
 * @param mdim Number of non-fixed axes of the section the array represents.
 * @param bnd_iter_1 Arrays with input data and working memory for element 1.
 *        Must satisfy the same preconditions as in topo_reorder_with_orientation.
 * @param bnd_iter_2 Arrays with input data and working memory for element 2.
 *        Must satisfy the same preconditions as in topo_reorder_with_orientation.
 * @param skip_edges When non-zero, edges (any axis being at min or max value) are skipped.
 * @param callback Callback function that is called on for iterations. It accepts the index of DoFs for the boundary
 *                 itself, then for the first and second element, as well as user-provided pointer.
 * @param user_data Pointer passed to the callback function for each iteration.
 */
void topo_iterate_boundary(unsigned ndim, unsigned mdim, topo_bnd_iter_t bnd_iter_1, topo_bnd_iter_t bnd_iter_2,
                           bool skip_edges,
                           void (*callback)(uint64_t idx_bnd, uint64_t idx_1, uint64_t idx_2, void *user_data),
                           void *user_data);

/**
 * Connect elements together based on their boundaries.
 *
 * Iterates over all immersed boundary objects of every dimension and, for
 * each pair of elements that share a boundary object, computes the local
 * orders of the shared boundary as the per-axis minimum of the two element
 * orders. The connectivity itself is implicit in the immersion
 * information; this function currently only performs the shared-boundary
 * order bookkeeping.
 *
 * @param n_elements Number of elements with ndim each.
 * @param ndim Number of dimensions for the space elements are in.
 * @param element_orders Orders for each element for each of the dimensions.
 *        Must have `ndim * n_elements` entries, ordered by element ID, with
 *        the ID of each element within the range of the immersion data.
 * @param immersions Array with boundary immersions for each order of boundaries.
 */
void topo_connect_boundaries(uint64_t n_elements, unsigned ndim,
                             const unsigned element_orders[static ndim * n_elements],
                             const topo_obj_immersion_t immersions[static const ndim]);
