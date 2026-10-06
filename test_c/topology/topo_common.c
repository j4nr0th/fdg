#include "topo_common.h"
#include "../common/common.h"

void assert_immersion_equal(const unsigned ndim, const topo_obj_immersion_t *a, const topo_obj_immersion_t *b)
{
    // Check the parent collection is the exact same
    TEST_ASSERTION(a->object_count == b->object_count, "Object count missmatch (%u vs %u)", a->object_count,
                   b->object_count);
    // Parent dims should also be the same
    TEST_ASSERTION(a->parent_dims == b->parent_dims, "Parent dimensions mismatch (%u vs %u)", a->parent_dims,
                   b->parent_dims);
    // Check offsets and counts for all elements
    const uint64_t object_cnt = a->object_count;
    for (uint64_t i_object = 0; i_object < object_cnt; ++i_object)
    {
        // First, check the offsets are correct
        const uint64_t offset = a->element_offsets[i_object];
        TEST_ASSERTION(offset == b->element_offsets[i_object], "Element offsets mismatch at index %zu (%zu vs %zu)",
                       (size_t)i_object, (size_t)offset, (size_t)b->element_offsets[i_object]);

        // Loop over elements for each object
        const uint64_t obj_element_cnt = a->element_offsets[i_object + 1] - offset;

        // Check element indices match
        {
            const uint64_t *element_indices_a = a->element_ids + offset;
            const uint64_t *element_indices_b = b->element_ids + offset;
            for (uint64_t i_element = 0; i_element < obj_element_cnt; ++i_element)
            {
                TEST_ASSERTION(element_indices_a[i_element] == element_indices_b[i_element],
                               "Element IDs mismatch at index %zu for object %zu (%zu vs %zu)", (size_t)(i_element),
                               (size_t)i_object, (size_t)element_indices_a[i_element],
                               (size_t)element_indices_b[i_element]);
            }
        }

        // Check element orientations are correct
        {
            const unsigned obj_ndim = ndim;
            const int8_t *element_orientations_a = a->element_orientation + offset * obj_ndim;
            const int8_t *element_orientations_b = b->element_orientation + offset * obj_ndim;
            for (uint64_t i_element = 0; i_element < obj_element_cnt; ++i_element)
            {
                for (uint64_t i_axis = 0; i_axis < obj_ndim; ++i_axis)
                {
                    TEST_ASSERTION(element_orientations_a[i_element * obj_ndim + i_axis] ==
                                       element_orientations_b[i_element * obj_ndim + i_axis],
                                   "Element orientation mismatch for element %zu (entry %zu) for object %zu (axis %u "
                                   "was %d instead of %d)",
                                   (size_t)a->element_ids[offset + i_element] + 1, (size_t)i_element,
                                   (size_t)i_object + 1, (unsigned)i_axis + 1,
                                   (int)element_orientations_a[i_element * obj_ndim + i_axis],
                                   (int)element_orientations_b[i_element * obj_ndim + i_axis]);
                }
            }
        }
    }
}
