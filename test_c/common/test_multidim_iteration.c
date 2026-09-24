//
// Created by jan on 2026-09-23.
//
#include "../common/common.h"

#include <cutl/iterators/multidim_iteration.h>

// ================================================================
// Main Entry
// ================================================================
int main(void)
{
    enum
    {
        MAX_DIMS = 3,
        POISON = 0xAB,
    };
    const size_t dims[MAX_DIMS] = {2, 3, 4};
    const size_t block_memory = multidim_iterator_needed_memory(MAX_DIMS);

    _Alignas(max_align_t) unsigned char iter_buffer[128];
    _Alignas(max_align_t) unsigned char ref_buffer[128];
    TEST_ASSERTION(block_memory <= sizeof(iter_buffer), "Iterator buffer too small.");
    TEST_ASSERTION(block_memory <= sizeof(ref_buffer), "Reference buffer too small.");
    multidim_iterator_t *const iter = (multidim_iterator_t *)iter_buffer;
    multidim_iterator_t *const ref = (multidim_iterator_t *)ref_buffer;

    for (unsigned ndim = 1; ndim <= MAX_DIMS; ++ndim)
    {
        size_t total_points = 1;
        for (unsigned dim = 0; dim < ndim; ++dim)
        {
            total_points *= dims[dim];
        }

        // Poison the raw block first: an offset the sequence forgets keeps a deterministic nonzero value instead of
        // fresh-page zeros masking the bug.
        for (size_t byte = 0; byte < block_memory; ++byte)
        {
            iter_buffer[byte] = POISON;
        }
        for (unsigned dim = 0; dim < ndim; ++dim)
        {
            multidim_iterator_init_dim(iter, dim, dims[dim]);
        }
        TEST_ASSERTION(multidim_iterator_get_ndims(iter) == ndim, "init_dim did not set the dimension count.");
        for (unsigned dim = 0; dim < ndim; ++dim)
        {
            TEST_ASSERTION(multidim_iterator_get_dim(iter, dim) == dims[dim], "init_dim stored the wrong size.");
            TEST_ASSERTION(multidim_iterator_get_offset(iter, dim) == 0,
                           "The init_dim sequence left a stale offset; it must start at the origin.");
        }

        // Walk both in lockstep: same points, same flat indices, same end.
        multidim_iterator_init(ref, ndim, dims);
        for (size_t point = 0; point < total_points; ++point)
        {
            TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == multidim_iterator_get_flat_index(ref),
                           "The init_dim iterator diverged from the init reference.");
            multidim_iterator_advance(iter, ndim - 1, 1);
            multidim_iterator_advance(ref, ndim - 1, 1);
        }
        TEST_ASSERTION(multidim_iterator_is_at_end(iter), "Sweep did not finish at the end position.");

        // Callers reuse one work block and re-run only the init sequence between sweeps; stale offsets there would
        // shift every subsequent sweep.
        for (unsigned dim = 0; dim < ndim; ++dim)
        {
            multidim_iterator_init_dim(iter, dim, dims[dim]);
        }
        TEST_ASSERTION(multidim_iterator_is_at_start(iter) && multidim_iterator_get_flat_index(iter) == 0,
                       "Re-initialization after a sweep did not return to the origin.");
    }

    printf("test_multidim_iteration PASSED\n");

    return 0;
}
