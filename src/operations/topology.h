#pragma once
#include <cutl/allocators.h>
#include <stdint.h>

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
    int64_t *boundary_ids;
} topo_obj_collection_t;

/**
 * @brief Specification of how m-dimensional objects are immersed in n-dimensional space.
 *
 * This information is what n-dimensional elements they are contained within, what are the fixed element axis
 * of these objects, and how their local axes map to those of the element.
 */
typedef struct
{
    // The collection the immersion information is aboutn.
    const topo_obj_collection_t *collection;
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
 * This cool function multiplies by 2, but it is here, because code is a lot clearer when there's a proper name
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
 * @return Zero if the two share no boundary, otherwise an ID of the shared boundary, with the orientation
 *         as in the object with ID of ``id_1``.
 */
int64_t topo_obj_common_boundary(const topo_obj_collection_t *collection, int64_t id_1, int64_t id_2);

/**
 * Find the index of the common boundary of two objects from the same collection.
 *
 * @param collection[in] Collection the two objects belong to.
 * @param id_1[in] ID of the first object.
 * @param id_2[in] ID of the second object.
 * @return Index of the common boundary in the element with ID ``id_1``, negative value if there is no common boundary.
 */
int64_t topo_obj_common_boundary_index(const topo_obj_collection_t *collection, int64_t id_1, int64_t id_2);

/**
 * Return the array with IDs of elements an object is contained in.
 *
 * @param[in] immersion Immersion info of objects.
 * @param[in] id ID of the object to ge the array for.
 * @param[out] p_cnt Pointer to the location where the size of the output array is stored.
 * @param[out] p_ids Pointer to the location where the array pointer is stored.
 */
void topo_obj_elements(const topo_obj_immersion_t *immersion, uint64_t id, uint64_t *p_cnt, const uint64_t **p_ids);

/**
 * Determine immersion information from topological description of elements in terms of boundaries. These
 * boundaries are in turn again described by their boundaries, and so on until 0-D objects (points).
 *
 * @param ndim[in] Number of dimensions of the space all objects are immersed in.
 * @param collections[in] Collections of objects going from 0-D (points) to ndim-D (elements themselves).
 * @param allocator[in] Allocator to use to create the immersions in.
 * @param immersions[out] Array, which receives computed immersion information for objects from 0-D to (ndim-1)-D
 * @return Zero if successful, otherwise an error code.
 */
int topo_obj_create_immersion_info(unsigned ndim, const topo_obj_collection_t collections[static ndim],
                                   const cutl_allocator_t *allocator, topo_obj_immersion_t immersions[ndim]);
