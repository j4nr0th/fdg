#include "topology.h"
#include <math.h>
#include <stdbool.h>

unsigned topo_obj_boundary_count(const unsigned ndim)
{
    return 2 * ndim;
}

void topo_obj_elements(const topo_obj_immersion_t *immersion, const uint64_t id, uint64_t *p_cnt,
                       const uint64_t **p_ids)
{
    const uint64_t offset = immersion->element_offsets[id];
    const uint64_t cnt = immersion->element_offsets[id + 1] - offset;
    *p_cnt = cnt;
    *p_ids = immersion->element_ids + offset;
}

/**
 * Convert (signed) one-based index to a zero based index, discarding the sign.
 *
 * Basically just ``abs(idx) - 1``.
 *
 * @param idx Index using one-based indexing.
 * @return Zero-based index.
 */
static uint64_t indexing_to_zero_based(const int64_t idx)
{
    return (idx < 0 ? -idx : idx) - 1;
}

int64_t topo_obj_common_boundary_index(const topo_obj_collection_t *collection, const int64_t id_1, const int64_t id_2)
{
    const unsigned bnds_per_object = topo_obj_boundary_count(collection->ndim);

    const uint64_t id_1_idx = indexing_to_zero_based(id_1);
    const uint64_t id_2_idx = indexing_to_zero_based(id_2);

    const int64_t *const bnds_1 = collection->boundary_ids + id_1_idx * bnds_per_object;
    const int64_t *const bnds_2 = collection->boundary_ids + id_2_idx * bnds_per_object;

    for (unsigned ib1 = 0; ib1 < bnds_per_object; ++ib1)
    {
        const uint64_t id_1_bnd = indexing_to_zero_based(bnds_1[ib1]);
        for (unsigned ib2 = 0; ib2 < bnds_per_object; ++ib2)
        {
            const uint64_t id_2_bnd = indexing_to_zero_based(bnds_2[ib2]);
            if (id_1_bnd == id_2_bnd)
            {
                return ib1;
            }
        }
    }

    // There was no common boundary!
    return -1;
}

int64_t topo_obj_common_boundary(const topo_obj_collection_t *collection, const int64_t id_1, const int64_t id_2)
{
    const int64_t idx = topo_obj_common_boundary_index(collection, id_1, id_2);
    if (idx < 0) // No common boundary
        return 0;

    // Get the boundary from the first object and adjust the id.
    const unsigned bnds_per_object = topo_obj_boundary_count(collection->ndim);
    const uint64_t id_1_idx = indexing_to_zero_based(id_1);
    const int64_t *const bnds_1 = collection->boundary_ids + id_1_idx * bnds_per_object;
    const int64_t bnd_id = bnds_1[idx];
    if (id_1 < 0)
        return -bnd_id;
    return bnd_id;
}

/**
 * Release all memory for immersions and clear them.
 *
 * @param ndim Number of dimensions (and immersions).
 * @param immersions Immersions to release.
 * @param allocator Allocator with which the memory for immersions was allocated.
 */
static void topo_obj_immersions_clean(const unsigned ndim, topo_obj_immersion_t immersions[const ndim],
                                      const cutl_allocator_t *const allocator)
{
    for (unsigned i = 0; i < ndim; ++i)
    {
        // Free memory if needed
        if (immersions[i].element_ids != NULL)
            cutl_dealloc(allocator, immersions[i].element_ids);
        if (immersions[i].element_offsets != NULL)
            cutl_dealloc(allocator, immersions[i].element_offsets);
        if (immersions[i].element_orientation != NULL)
            cutl_dealloc(allocator, immersions[i].element_orientation);
        // Clear the memory
        immersions[i] = (topo_obj_immersion_t){0};
    }
}

/**
 * Recursively count up the number of times a boundary object occurs in all elements. The number of times the count
 * is increased for each boundary object with (n - k) dimensions is k!
 *
 * @param parent ID of the parent object.
 * @param ndim Dimension of the parent object.
 * @param collections Collections the objects belong to.
 * @param immersions Immersions for objects.
 */
static void topo_obj_recursively_count_elements(const uint64_t parent, const unsigned ndim,
                                                const topo_obj_collection_t collections[static ndim],
                                                topo_obj_immersion_t immersions[const ndim])
{
    const uint64_t bnds_per_object = topo_obj_boundary_count(ndim);
    const int64_t *const bnds = collections[ndim - 1].boundary_ids + parent * bnds_per_object;

    for (unsigned ib = 0; ib < ndim; ++ib)
    {
        const uint64_t idx_start = indexing_to_zero_based(bnds[ib]);
        const uint64_t idx_end = indexing_to_zero_based(bnds[ib + ndim]);
        immersions[ndim - 1].element_offsets[idx_start] += 1;
        immersions[ndim - 1].element_offsets[idx_end] += 1;

        if (ndim != 0)
        {
            topo_obj_recursively_count_elements(idx_start, ndim - 1, collections, immersions);
            topo_obj_recursively_count_elements(idx_end, ndim - 1, collections, immersions);
        }
    }
}

/**
 * Value used for invalid element index.
 */
static const uint64_t INVALID_ELEMENT_ID = UINT64_MAX;

static int8_t *immersion_orientation_for_object(const uint64_t ie, const uint64_t obj_idx,
                                                const topo_obj_immersion_t *immersion)
{
    // Get the element indices array
    const uint64_t offset = immersion->element_offsets[obj_idx];
    const uint64_t capacity = immersion->element_offsets[obj_idx + 1] - offset;
    uint64_t *const element_indices = immersion->element_ids + offset;
    int8_t *const element_orientation = immersion->element_orientation + immersion->parent_dims * offset;
    // Count up the used entries
    uint64_t used_entries = 0;
    for (unsigned i = 0; i < capacity; ++i)
    {
        if (element_indices[i] != INVALID_ELEMENT_ID)
            break;
        used_entries += 1;
    }

    if (used_entries == capacity)
        return NULL; // Error: all entries have been used up!

    // Find the position to insert the element index at (first index that is not less than)
    uint64_t insertion_idx;
    for (insertion_idx = 0; insertion_idx < used_entries; ++insertion_idx)
    {
        if (element_indices[insertion_idx] == ie)
            return NULL; // Error: was already inserted!
        if (element_indices[insertion_idx] > ie)
            break;
    }

    // Move previous indices out of the way if needed
    if (insertion_idx != used_entries)
    {
        // We do a memmove
        for (unsigned i = used_entries; i > insertion_idx; --i)
            element_indices[i] = element_indices[i - 1];
        // We do a memmove for orientation array as well
        for (unsigned i = immersion->parent_dims * used_entries; i > immersion->parent_dims * insertion_idx; --i)
            element_orientation[i] = element_orientation[i - immersion->parent_dims];
    }
    // Now we can insert the new entry
    element_indices[insertion_idx] = ie;
    // Return the orientation array so that we can write to it
    return element_orientation + immersion->parent_dims * insertion_idx;
}

/**
 * @brief Recursively determine the orientation of boundaries making up an object relative to an element they are in.
 *
 * Recursively iterate over boundaries of a topological object to determine its orientation with
 * respect to the element it is contained in. The orientation is stored in the immersion corresponding
 * to objects with dimension from 0 to (n - 1). The information about orientation is stored in two parts -
 * the first m entries specify the fixed axis with 1-based indexing, with the sign indicating if it is at
 * the start or at the end of the axis. The other (n - m) entries map the axis of the boundary to those of the
 * parent, again using 1-based indexing. The sign shows if the direction of that axis is the same in the
 * topological object as it is in the element.
 *
 * This function works recursively, so that after orientation of the boundary is finished, it can be called on its
 * own boundaries. The recursion should never go very deep, as the total number of dimensions for any element is
 * low (I would be surprised if we ever got to more than 6 or 7).
 *
 * @param ie[in] Index of the element that the objects are processed under.
 * @param parent_idx[in] Index of the parent element within its collection.
 * @param idim[in] Index of the dimension we are currently in on this level of recursion.
 * @param ndim[in] Total number of dimensions in the topological space.
 * @param collections[in] Collections with all topological objects for dimensions from 1 to n.
 * @param parent_orientation[in] Array specifying the parent's orientation.
 * @param immersions[out] Immersions for all topological objects for dimensions from 0 to (n - 1).
 */
static void topo_obj_recursively_orient(const uint64_t ie, const uint64_t parent_idx, const unsigned idim,
                                        const unsigned ndim, const topo_obj_collection_t collections[static ndim],
                                        const int8_t parent_orientation[idim],
                                        topo_obj_immersion_t immersions[const ndim])
{
    // Get the boundaries of the parent from the collection
    const uint64_t boundaries_per_elem = topo_obj_boundary_count(idim);
    const int64_t *const boundaries = collections[idim].boundary_ids + parent_idx * boundaries_per_elem;

    const uint64_t fixed_axis = ndim - idim;

    // Loop over all dimensions
    for (unsigned bdim = 0; bdim < idim; ++bdim)
    {
        // We may not need to do this boundary if we have more than 1 fixed axis, as we
        // always do only the cases where the fixed axis indices will be ascending.
        if (fixed_axis > 1)
        {
            // Check what the index of the boundary is in the parent's reference frame
            const uint8_t parent_boundary_idx = indexing_to_zero_based(parent_orientation[fixed_axis + bdim]);
            const uint8_t prev_axis_idx = indexing_to_zero_based(parent_orientation[fixed_axis - 1]);
            // We can skip this boundary, it would not have sorted indices.
            if (parent_boundary_idx > prev_axis_idx)
                continue;

            CUTL_ASSERT(parent_boundary_idx != prev_axis_idx,
                        "Parent boundary index is the same as the previous axis index for the object ID");
        }

        // Get the boundary perpendicular to axis bdim (in parent reference frame)
        const int64_t boundary = boundaries[bdim];
        const uint64_t bnd_idx = indexing_to_zero_based(boundary);

        // Get the orientation array of the boundary, which we will write into
        int8_t *const orient_arr = immersion_orientation_for_object(ie, bnd_idx, immersions + idim);
        CUTL_ASSERT(orient_arr != NULL, "Could not orientation array was already used up for this object!");

        // Start of the ID is the same as the parent
        for (unsigned i = 0; i < fixed_axis; ++i)
            orient_arr[i] = parent_orientation[i];

        // Set the index of the boundary object (what dimension it is perpendicular to)
        orient_arr[fixed_axis] = (int8_t)-(bdim + 1);

        // Identify the axis of the boundary with respect to the parent. No worries if the boundary is a point, this
        // loop just won't run then
        for (unsigned axis_index = 0, jb = 1; axis_index < idim; ++axis_index)
        {
            // We are on this boundary, skip
            if (axis_index == bdim)
                continue;

            // Find what the index of the common boundary is
            int64_t common_index = topo_obj_common_boundary_index(collections + idim, boundary, boundaries[axis_index]);

            CUTL_ASSERT(common_index >= 0, "There must be at least one common boundary between the two non-opposite "
                                           "boundaries of the same object");
            // Adjust for the sign of the index
            common_index += 1; // Convert to 1-based indexing
            // If we are in the second half of the array, we flip the sign
            if (common_index > idim)
                common_index *= -1;

            // Record the orientation for the child
            orient_arr[fixed_axis + jb] = (int8_t)common_index;
            jb += 1;
        }

        // The entries of orient_arr[fixed_axis:ndim] are now relative to the parent.
        // To adjust, we need to use the orientation of the parent.
        for (unsigned i = fixed_axis; i < ndim; ++i)
        {
            // Convert from 1-based indexing to index and orientation
            const int8_t child_axis_idx = orient_arr[i];
            const unsigned axis_index = indexing_to_zero_based(child_axis_idx);
            // Get the parent axis for that
            int8_t parent_axis = parent_orientation[axis_index];
            if (child_axis_idx < 0) // Flip if needed
                parent_axis *= -1;

            // Write it back
            orient_arr[i] = parent_axis;
        }

        // If we are dealing with points only, we are done.
        if (idim == 0)
            continue;

        // With the object's orientation fully determined, we can now recursively do this for its boundaries.
        topo_obj_recursively_orient(ie, bnd_idx, idim - 1, ndim, collections, orient_arr, immersions);
    }
}

int topo_obj_create_immersion_info(const unsigned ndim, const topo_obj_collection_t collections[const static ndim],
                                   const cutl_allocator_t *const allocator, topo_obj_immersion_t immersions[const ndim])
{
    // If no dimensions, we have nothing to do
    if (ndim == 0)
        return 0;

    // Clear the output
    for (unsigned i = 0; i < ndim; ++i)
        immersions[i] = (topo_obj_immersion_t){0};

    // Iterate over the objects and their boundaries
    for (unsigned idim = 0; idim + 1 < ndim; ++idim)
    {
        const topo_obj_collection_t *const boundaries = collections + idim;

        // Specify the parent's dimension
        immersions[idim].parent_dims = ndim;

        // Allocate the element counters (+1 element for cumulative sum later)
        uint64_t *const counters = cutl_alloc(allocator, sizeof(*counters) * (boundaries->count + 1));
        if (!counters)
        {
            topo_obj_immersions_clean(ndim, immersions, allocator);
            return -1;
        }

        // Zero the counters (NOTE: maybe ignore the last entry)
        for (unsigned i = 0; i < boundaries->count + 1; ++i)
            counters[i] = 0;

        immersions[idim].element_offsets = counters;
    }

    // For every element, iterate over all boundaries (and their boundaries) to count how many elements they appear in
    for (uint64_t ie = 0; ie < collections[ndim - 1].count; ++ie)
    {
        topo_obj_recursively_count_elements(ie, ndim, collections, immersions);
    }

    // Scale the counts by their multiplicity (factorial of the dimension)
    unsigned multiplicity = ndim;
    for (unsigned idim = 1; idim < ndim; ++idim)
    {
        // Scale the count array by the multiplicity
        uint64_t *const counts = immersions[ndim - 1 - idim].element_offsets;
        for (unsigned j = 0; j < collections[ndim - 1 - idim].count; ++j)
        {
            CUTL_ASSERT(counts[j] % multiplicity == 0,
                        "Each object count should be exactly multiple of the multiplicity");
            counts[j] /= multiplicity;
        }

        // Adjust multiplicity for the next iteration
        multiplicity *= ndim - idim;
    }

    // We now have to convert count arrays into cumulative sums (makes offset array)
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        // For now this is a count array, as we have not yet cum-summed it into an offset array
        uint64_t *const counts = immersions[idim].element_offsets;

        // Now we can cum-sum the count array so it becomes an offset array.
        uint64_t total_elements = 0;
        for (uint64_t i = 0; i < collections[idim].count; ++i)
        {
            const uint64_t count = counts[i];
            counts[i] = total_elements;
            total_elements += count;
        }
        // Write in the last entry
        counts[collections[idim].count] = total_elements;
    }

    // Allocate the memory for the arrays that specify what element each object belongs to and the array,
    // which identifies its axis.
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        topo_obj_immersion_t *const immersion = immersions + idim;
        const uint64_t total_entries = immersions->element_offsets[immersion->collection->count];

        immersion->element_ids = cutl_alloc(allocator, sizeof(*immersion->element_ids) * total_entries);
        if (!immersion->element_ids)
        {
            topo_obj_immersions_clean(ndim, immersions, allocator);
            return -1;
        }
        // Clear by setting it to invalid IDs
        for (uint64_t i = 0; i < total_entries; ++i)
            immersion->element_ids[i] = INVALID_ELEMENT_ID;

        // We leave these alone, since we fill them in at the same time as element_ids array.
        immersion->element_orientation =
            cutl_alloc(allocator, sizeof(*immersion->element_orientation) * total_entries * ndim);
        if (!immersion->element_orientation)
        {
            topo_obj_immersions_clean(ndim, immersions, allocator);
            return -1;
        }
    }

    // Prepare working buffer
    int8_t *const parent_orientation = cutl_alloc(allocator, sizeof(*parent_orientation) * ndim);
    if (!parent_orientation)
    {
        topo_obj_immersions_clean(ndim, immersions, allocator);
        return -1;
    }
    // Parent orientation is for now just identity
    for (int8_t i = 0; i < (int8_t)ndim; ++i)
        parent_orientation[i] = (int8_t)(i + 1);

    // Similar process now as with the element counting, but we must now also deal with the orientation
    for (unsigned ie = 0; ie < collections[ndim - 1].count; ++ie)
    {
        topo_obj_recursively_orient(ie, ie, ndim - 1, ndim, collections, parent_orientation, immersions);
    }
    cutl_dealloc(allocator, parent_orientation);

    // Finally done!
    return 0;
}
