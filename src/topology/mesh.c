#include "mesh.h"

#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Cube enumeration
 *
 * The objects inside one hypercube element are enumerated by their axis mask
 * (bit set = the axis the object spans) and their planes: an object of
 * dimension ``m`` has ``C(ndim, m)`` masks with ``2^(ndim - m)`` positions
 * each. The masks are taken in ascending numeric order, and the positions are
 * the mixed-radix values of the fixed-axis plane indices in {0, 1}, with the
 * low axes least significant, so the local ID of an object with mask rank ``r``
 * is ``r * 2^(ndim - m)`` plus the mixed-radix value of its planes. A corner
 * of the element is the object of dimension zero at the plane combination
 * given by its corner coordinate, so its local ID is the mixed-radix corner
 * index itself.
 * ========================================================================== */

/**
 * Number of combinations of k elements out of n, or UINT64_MAX on overflow.
 */
static uint64_t nck(const uint64_t n, const uint64_t k)
{
    uint64_t res = 1;
    for (uint64_t i = 0; i < k; ++i)
    {
        if (res > UINT64_MAX / (n - i))
            return UINT64_MAX;
        const uint64_t next = res * (n - i) / (i + 1);
        if (next == 0)
            return UINT64_MAX;
        res = next;
    }
    return res;
}

/**
 * Enumerate the axis masks of a given size in ascending order.
 *
 * @param ndim Number of dimensions of the space.
 * @param mdim Dimension of the objects the masks describe.
 * @param out Receives the masks; its capacity must be at least C(ndim, mdim).
 * @return Number of masks written.
 */
static unsigned mask_enum(const unsigned ndim, const unsigned mdim, uint64_t *const out)
{
    unsigned count = 0;
    for (uint64_t mask = ((uint64_t)1 << mdim) - 1; mask < ((uint64_t)1 << ndim);)
    {
        out[count] = mask;
        count += 1;
        const uint64_t c = mask & -mask;
        const uint64_t r = mask + c;
        mask = ((r ^ mask) >> 2) / c | r;
    }
    return count;
}

/**
 * Number of objects of dimension ``mdim`` inside one element, the total of
 * C(ndim, mdim) * 2^(ndim - mdim).
 */
static bool subobj_count(const unsigned ndim, const unsigned mdim, uint64_t *out)
{
    const uint64_t combos = nck(ndim, mdim);
    if (combos == UINT64_MAX)
        return false;
    return !__builtin_mul_overflow(combos, (uint64_t)1 << (ndim - mdim), out);
}

/**
 * Corner IDs of one sub-object of an element, sorted in ascending order.
 *
 * @param ndim Number of dimensions of the space.
 * @param mdim Dimension of the sub-object.
 * @param element_corners Corner IDs of the element, one entry per corner index.
 * @param mask Axis mask of the sub-object (its spanning axes).
 * @param planes Plane of each element axis, with the fixed axes at 0 or 1.
 * @param out Receives the sorted corner IDs, 2^mdim entries.
 */
static void subobject_corners(const unsigned ndim, const unsigned mdim, const uint64_t element_corners[],
                              const uint64_t mask, const uint8_t planes[const ndim], uint64_t *const out)
{
    uint64_t base = 0;
    for (unsigned a = 0; a < ndim; ++a)
    {
        if (!((mask >> a) & 1U))
            base |= (uint64_t)planes[a] << a;
    }
    const uint64_t corner_count = (uint64_t)1 << mdim;
    for (uint64_t free_bits = 0; free_bits < corner_count; ++free_bits)
    {
        uint64_t corner_index = base;
        unsigned free_axis = 0;
        for (unsigned a = 0; a < ndim; ++a)
        {
            if (!((mask >> a) & 1U))
                continue;
            if ((free_bits >> free_axis) & 1U)
                corner_index |= (uint64_t)1 << a;
            free_axis += 1;
        }
        out[free_bits] = element_corners[corner_index];
    }
    // Insertion sort, the arrays are small.
    for (uint64_t i = 1; i < corner_count; ++i)
    {
        const uint64_t value = out[i];
        uint64_t j = i;
        while (j > 0 && out[j - 1] > value)
        {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = value;
    }
}

/* ==========================================================================
 * Corner maps
 *
 * A map from sorted corner-point sets to object IDs, used to merge the
 * objects implied by the corner data of the elements.
 * ========================================================================== */

typedef struct
{
    // Number of corners per object, the length of every key.
    uint64_t key_len;
    // Power of two size of the table; hash_mask is capacity - 1.
    uint64_t capacity;
    uint64_t hash_mask;
    // Keys, laid out as capacity blocks of key_len entries.
    uint64_t *keys;
    // Object IDs per slot, UINT64_MAX when the slot is empty.
    uint64_t *ids;
    // Number of inserted objects.
    uint64_t count;
} corner_map_t;

static void corner_map_release(corner_map_t *map, const cutl_allocator_t *allocator)
{
    cutl_dealloc(allocator, map->keys);
    cutl_dealloc(allocator, map->ids);
    *map = (corner_map_t){0};
}

static bool corner_map_init(corner_map_t *map, const uint64_t capacity, const uint64_t key_len,
                            const cutl_allocator_t *allocator)
{
    *map = (corner_map_t){.key_len = key_len, .capacity = capacity, .hash_mask = capacity - 1};
    size_t key_bytes;
    if (__builtin_mul_overflow((size_t)capacity, (size_t)key_len, &key_bytes) ||
        __builtin_mul_overflow(key_bytes, sizeof(uint64_t), &key_bytes))
        return false;
    map->keys = cutl_alloc(allocator, key_bytes);
    if (!map->keys)
        return false;
    size_t id_bytes;
    if (__builtin_mul_overflow((size_t)capacity, sizeof(uint64_t), &id_bytes))
    {
        cutl_dealloc(allocator, map->keys);
        *map = (corner_map_t){0};
        return false;
    }
    map->ids = cutl_alloc(allocator, id_bytes);
    if (!map->ids)
    {
        cutl_dealloc(allocator, map->keys);
        *map = (corner_map_t){0};
        return false;
    }
    for (uint64_t i = 0; i < capacity; ++i)
        map->ids[i] = UINT64_MAX;
    return true;
}

static uint64_t corner_map_hash(const uint64_t *const key, const uint64_t key_len)
{
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t i = 0; i < key_len; ++i)
    {
        h ^= key[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/**
 * Find the ID of the object with the given sorted corner set, creating a new
 * object when it is not in the map yet.
 *
 * @param map Map to search in.
 * @param key Sorted corner point IDs, map->key_len entries.
 * @param out Receives the object ID.
 * @param created Receives whether a new object was created.
 */
static void corner_map_find(corner_map_t *map, const uint64_t *const key, uint64_t *const out, bool *const created)
{
    const uint64_t hash = corner_map_hash(key, map->key_len);
    uint64_t probe = 0;
    for (;;)
    {
        const uint64_t slot = (hash + probe * (probe + 1) / 2) & map->hash_mask;
        if (map->ids[slot] == UINT64_MAX)
        {
            memcpy(map->keys + slot * map->key_len, key, sizeof(uint64_t) * (size_t)map->key_len);
            map->ids[slot] = map->count;
            *out = map->count;
            map->count += 1;
            *created = true;
            return;
        }
        const uint64_t *const slot_key = map->keys + slot * map->key_len;
        uint64_t i;
        for (i = 0; i < map->key_len; ++i)
        {
            if (slot_key[i] != key[i])
                break;
        }
        if (i == map->key_len)
        {
            *out = map->ids[slot];
            *created = false;
            return;
        }
        probe += 1;
    }
}

/**
 * Next power of two greater than or equal to ``value``, or false on overflow.
 */
static bool next_pow2(const uint64_t value, uint64_t *const out)
{
    if (value > (uint64_t)1 << 62)
        return false;
    uint64_t p = 1;
    while (p < value)
        p <<= 1;
    *out = p;
    return true;
}

/* ==========================================================================
 * Construction from corner points
 * ========================================================================== */

typedef struct
{
    // Collections of the objects of dimensions 1 to ndim, [ndim].
    topo_obj_collection_t *collections;
} corner_build_t;

static void corner_build_release(corner_build_t *build, const unsigned ndim, const cutl_allocator_t *allocator)
{
    if (build->collections)
    {
        for (unsigned mdim = 1; mdim <= ndim; ++mdim)
            cutl_dealloc(allocator, build->collections[mdim - 1].boundary_ids);
        cutl_dealloc(allocator, build->collections);
    }
    *build = (corner_build_t){0};
}

/**
 * Resolve the boundary objects of one sub-object into their global IDs.
 *
 * The boundaries of an object of dimension ``mdim`` are its ``2 * mdim``
 * objects of dimension ``mdim - 1``: slot ``i`` is the boundary perpendicular
 * to the object's axis ``i`` at its start and slot ``i + mdim`` is the one at
 * its end. The start side of an axis is the plane zero side, the one that
 * contains the smallest corner point of the object.
 *
 * @param ndim Number of dimensions of the space.
 * @param mdim Dimension of the object. Must be at least one.
 * @param element_corners Corner IDs of the element, one per corner index.
 * @param mask Axis mask of the object.
 * @param planes Plane of each element axis, with the fixed axes at 0 or 1.
 * @param boundary_map Map of the boundary objects for the MERGE, NULL when
 *        mdim == 1 (the boundaries of a line are its two corner points).
 * @param scratch Scratch array with 2^(mdim - 1) entries.
 * @param boundaries Receives the global IDs of the boundaries, 2 * mdim entries.
 */
static void subobject_boundaries(const unsigned ndim, const unsigned mdim, const uint64_t element_corners[],
                                 const uint64_t mask, const uint8_t planes[const ndim],
                                 corner_map_t *const boundary_map, uint64_t *const scratch, uint64_t *const boundaries)
{
    // The object's axes are its spanning axes in ascending order.
    unsigned n_axes = 0;
    uint64_t axes[64];
    for (unsigned a = 0; a < ndim; ++a)
    {
        if ((mask >> a) & 1U)
            axes[n_axes++] = a;
    }

    for (unsigned i = 0; i < n_axes; ++i)
    {
        for (unsigned side = 0; side < 2; ++side)
        {
            uint8_t boundary_planes[64];
            memcpy(boundary_planes, planes, sizeof(boundary_planes));
            boundary_planes[axes[i]] = (uint8_t)side;
            const uint64_t boundary_mask = mask & ~((uint64_t)1 << axes[i]);
            subobject_corners(ndim, mdim - 1, element_corners, boundary_mask, boundary_planes, scratch);
            if (mdim == 1)
            {
                boundaries[i + (unsigned)side * mdim] = scratch[0];
            }
            else
            {
                uint64_t id;
                bool created;
                corner_map_find(boundary_map, scratch, &id, &created);
                boundaries[i + (unsigned)side * mdim] = id;
            }
        }
    }
}

/**
 * Create the collections of a mesh from the corner points of its elements.
 *
 * @param ndim Number of dimensions of the space.
 * @param element_count Number of elements.
 * @param point_count Number of points of the mesh.
 * @param corners Corner point IDs, element_count * 2^ndim entries.
 * @param allocator Allocator for the collections.
 * @param build Receives the built collections. On failure its memory is
 *        released before returning.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT, TOPO_SIZE_OVERFLOW
 *         or TOPO_FAILED_ALLOC on failure.
 */
static topo_status_t corner_build(const unsigned ndim, const uint64_t element_count, const uint64_t point_count,
                                  const uint64_t *const corners, const cutl_allocator_t *const allocator,
                                  corner_build_t *build)
{
    if (!corners)
        return TOPO_INVALID_ARGUMENT;
    *build = (corner_build_t){0};

    const uint64_t corners_per_element = (uint64_t)1 << ndim;

    // Validate the corner IDs: they must all name existing points, and the
    // corners of one element must be distinct.
    for (uint64_t e = 0; e < element_count; ++e)
    {
        const uint64_t *const element_corners = corners + e * corners_per_element;
        for (uint64_t c = 0; c < corners_per_element; ++c)
        {
            if (element_corners[c] >= point_count)
                return TOPO_INVALID_ARGUMENT;
        }
        for (uint64_t a = 0; a + 1 < corners_per_element; ++a)
        {
            for (uint64_t b = a + 1; b < corners_per_element; ++b)
            {
                if (element_corners[a] == element_corners[b])
                    return TOPO_INVALID_ARGUMENT;
            }
        }
    }

    topo_status_t status = TOPO_SUCCESS;
    corner_map_t maps[64] = {0};
    uint64_t *combos = NULL;
    uint64_t *masks_arr = NULL;
    uint64_t *bscratch = NULL;
    uint64_t *boundaries = NULL;
    uint8_t *written = NULL;

    // Estimate the object counts for the maps of every dimension.
    uint64_t count;
    uint64_t estimate;
    for (unsigned mdim = 1; mdim < ndim && status == TOPO_SUCCESS; ++mdim)
    {
        if (!subobj_count(ndim, mdim, &count) || __builtin_mul_overflow(element_count, count, &estimate))
            status = TOPO_SIZE_OVERFLOW;
    }

    // Allocate the maps. The capacity is a power of two above two times the
    // estimate, so the table is never fuller than half.
    for (unsigned mdim = 1; mdim < ndim && status == TOPO_SUCCESS; ++mdim)
    {
        if (!subobj_count(ndim, mdim, &count) || __builtin_mul_overflow(element_count, count, &estimate))
        {
            status = TOPO_SIZE_OVERFLOW;
            break;
        }
        uint64_t capacity;
        if (!next_pow2(estimate + 1, &capacity) || capacity > (uint64_t)1 << 62)
        {
            status = TOPO_SIZE_OVERFLOW;
            break;
        }
        capacity <<= 1;
        const uint64_t key_len = (uint64_t)1 << mdim;
        if (!corner_map_init(maps + mdim, capacity, key_len, allocator))
        {
            status = TOPO_FAILED_ALLOC;
            break;
        }
    }
    if (status != TOPO_SUCCESS)
        goto done;

    build->collections = cutl_alloc(allocator, (size_t)ndim * sizeof(topo_obj_collection_t));
    if (!build->collections)
    {
        status = TOPO_FAILED_ALLOC;
        goto done;
    }
    for (unsigned mdim = 1; mdim <= ndim; ++mdim)
        build->collections[mdim - 1] = (topo_obj_collection_t){.ndim = mdim};

    size_t scratch_bytes;
    if (__builtin_mul_overflow((size_t)1 << ndim, sizeof(uint64_t), &scratch_bytes))
    {
        status = TOPO_SIZE_OVERFLOW;
        goto done;
    }
    combos = cutl_alloc(allocator, scratch_bytes);
    masks_arr = cutl_alloc(allocator, scratch_bytes);
    if (!combos || !masks_arr)
    {
        status = TOPO_FAILED_ALLOC;
        goto done;
    }

    uint8_t planes[64] = {0};
    for (unsigned mdim = 1; mdim <= ndim; ++mdim)
    {
        const unsigned mask_count = mask_enum(ndim, mdim, masks_arr);
        const uint64_t plane_combos = (uint64_t)1 << (ndim - mdim);

        // Insert every object of this dimension into its map.
        if (mdim < ndim)
        {
            for (uint64_t e = 0; e < element_count; ++e)
            {
                const uint64_t *const ecorners = corners + e * corners_per_element;
                for (unsigned mi = 0; mi < mask_count; ++mi)
                {
                    for (uint64_t pc = 0; pc < plane_combos; ++pc)
                    {
                        uint64_t pi = 0;
                        for (unsigned a = 0; a < ndim; ++a)
                        {
                            if ((masks_arr[mi] >> a) & 1U)
                                continue;
                            planes[a] = (uint8_t)((pc >> pi) & 1U);
                            pi += 1;
                        }
                        subobject_corners(ndim, mdim, ecorners, masks_arr[mi], planes, combos);
                        uint64_t id;
                        bool created;
                        corner_map_find(maps + mdim, combos, &id, &created);
                    }
                }
            }
        }

        // Allocate the boundary IDs of this collection.
        const uint64_t object_count = mdim < ndim ? maps[mdim].count : element_count;
        uint64_t entries;
        if (__builtin_mul_overflow(object_count, (uint64_t)2 * mdim, &entries))
        {
            status = TOPO_SIZE_OVERFLOW;
            goto done;
        }
        build->collections[mdim - 1].count = object_count;
        build->collections[mdim - 1].boundary_ids = cutl_alloc(allocator, (size_t)entries * sizeof(uint64_t));
        if (!build->collections[mdim - 1].boundary_ids)
        {
            status = TOPO_FAILED_ALLOC;
            goto done;
        }

        // The boundaries of an object are written once, from the first element
        // that contains it, so its frame is canonical. The frame of an object
        // of dimension mdim is its set of mdim spanning axes in the ascending
        // element axis order; writing it once keeps the frame stable, which the
        // immersion computation relies on.
        if (mdim < ndim)
        {
            written = cutl_alloc(allocator, object_count ? (size_t)object_count : 1);
            if (!written)
            {
                status = TOPO_FAILED_ALLOC;
                goto done;
            }
            memset(written, 0, (size_t)object_count);
        }

        corner_map_t *const boundary_map = mdim > 1 ? maps + mdim - 1 : NULL;
        const uint64_t boundary_corner_count = (uint64_t)1 << (mdim - 1);
        bscratch = cutl_alloc(allocator, (size_t)boundary_corner_count * sizeof(uint64_t));
        boundaries = cutl_alloc(allocator, (size_t)(2 * mdim) * sizeof(uint64_t));
        if (!bscratch || !boundaries)
        {
            status = TOPO_FAILED_ALLOC;
            goto done;
        }

        for (uint64_t e = 0; e < element_count; ++e)
        {
            const uint64_t *const ecorners = corners + e * corners_per_element;
            for (unsigned mi = 0; mi < mask_count; ++mi)
            {
                for (uint64_t pc = 0; pc < plane_combos; ++pc)
                {
                    uint64_t pi = 0;
                    for (unsigned a = 0; a < ndim; ++a)
                    {
                        if ((masks_arr[mi] >> a) & 1U)
                            continue;
                        planes[a] = (uint8_t)((pc >> pi) & 1U);
                        pi += 1;
                    }
                    subobject_corners(ndim, mdim, ecorners, masks_arr[mi], planes, combos);
                    uint64_t object_id;
                    if (mdim == ndim)
                    {
                        object_id = e;
                    }
                    else
                    {
                        bool created;
                        corner_map_find(maps + mdim, combos, &object_id, &created);
                        if (written[object_id])
                            continue;
                    }
                    subobject_boundaries(ndim, mdim, ecorners, masks_arr[mi], planes, boundary_map, bscratch,
                                         boundaries);
                    uint64_t *const row = build->collections[mdim - 1].boundary_ids + object_id * (uint64_t)(2 * mdim);
                    for (unsigned slot = 0; slot < 2 * mdim; ++slot)
                        row[slot] = boundaries[slot];
                    if (mdim < ndim)
                        written[object_id] = 1;
                }
            }
        }
        cutl_dealloc(allocator, bscratch);
        cutl_dealloc(allocator, boundaries);
        bscratch = NULL;
        boundaries = NULL;
        cutl_dealloc(allocator, written);
        written = NULL;
    }

done:
    cutl_dealloc(allocator, written);
    cutl_dealloc(allocator, bscratch);
    cutl_dealloc(allocator, boundaries);
    cutl_dealloc(allocator, combos);
    cutl_dealloc(allocator, masks_arr);
    for (unsigned mdim = 1; mdim < ndim; ++mdim)
        corner_map_release(maps + mdim, allocator);
    if (status != TOPO_SUCCESS)
        corner_build_release(build, ndim, allocator);
    return status;
}

/* ==========================================================================
 * Validation
 * ========================================================================== */

/**
 * Validate the immersion records of a mesh against its element count.
 *
 * @param mesh Mesh with the immersions already stored.
 * @return false when an immersion record names an element outside the mesh.
 */
static bool mesh_validate_immersions(const topo_mesh_t *const mesh)
{
    const unsigned ndim = mesh->ndim;
    for (unsigned mdim = 0; mdim < ndim; ++mdim)
    {
        const topo_obj_immersion_t *const immersion = mesh->immersions + mdim;
        for (uint64_t g = 0; g < immersion->object_count; ++g)
        {
            uint64_t element_count;
            const uint64_t *ids;
            const int8_t *orients;
            topo_obj_immersion_of_object(immersion, g, &element_count, &ids, &orients);
            for (uint64_t i = 0; i < element_count; ++i)
            {
                if (ids[i] >= mesh->element_count)
                    return false;
            }
        }
    }
    return true;
}

/* ==========================================================================
 * Public API
 * ========================================================================== */

topo_status_t topo_mesh_create(const unsigned ndim, const uint64_t point_count,
                               topo_obj_collection_t *const collections, topo_obj_immersion_t *const immersions,
                               const cutl_allocator_t *const allocator, topo_mesh_t **const out)
{
    if (!allocator || !out || !collections || !immersions)
        return TOPO_INVALID_ARGUMENT;
    if (ndim == 0 || ndim > 63)
        return TOPO_INVALID_ARGUMENT;
    if (point_count == 0)
        return TOPO_INVALID_ARGUMENT;
    const uint64_t element_count = collections[ndim - 1].count;
    if (element_count == 0)
        return TOPO_INVALID_ARGUMENT;

    topo_mesh_t *mesh = cutl_alloc(allocator, sizeof(*mesh));
    if (!mesh)
        return TOPO_FAILED_ALLOC;
    *mesh = (topo_mesh_t){0};

    mesh->ndim = ndim;
    mesh->point_count = point_count;
    mesh->collections = collections;
    mesh->immersions = immersions;
    mesh->element_count = element_count;

    if (!mesh_validate_immersions(mesh))
    {
        cutl_dealloc(allocator, mesh);
        return TOPO_INVALID_ARGUMENT;
    }

    *out = mesh;
    return TOPO_SUCCESS;
}

topo_status_t topo_mesh_create_from_collections(const unsigned ndim, const uint64_t point_count,
                                                topo_obj_collection_t *const collections,
                                                const cutl_allocator_t *const allocator, topo_mesh_t **const out)
{
    if (!allocator || !out || !collections)
        return TOPO_INVALID_ARGUMENT;
    if (ndim == 0 || ndim > 63)
        return TOPO_INVALID_ARGUMENT;
    if (point_count == 0)
        return TOPO_INVALID_ARGUMENT;

    topo_obj_immersion_t *const immersions = cutl_alloc(allocator, (size_t)ndim * sizeof(topo_obj_immersion_t));
    if (!immersions)
        return TOPO_FAILED_ALLOC;
    for (unsigned i = 0; i < ndim; ++i)
        immersions[i] = (topo_obj_immersion_t){0};

    const topo_status_t immersion_status =
        topo_obj_create_immersion_info(ndim, (unsigned)point_count, collections, allocator, immersions);
    if (immersion_status != TOPO_SUCCESS)
    {
        topo_obj_immersions_free(ndim, immersions, allocator);
        cutl_dealloc(allocator, immersions);
        return immersion_status;
    }

    const topo_status_t status = topo_mesh_create(ndim, point_count, collections, immersions, allocator, out);
    if (status != TOPO_SUCCESS)
    {
        topo_obj_immersions_free(ndim, immersions, allocator);
        cutl_dealloc(allocator, immersions);
    }
    return status;
}

topo_status_t topo_mesh_create_from_corners(const unsigned ndim, const uint64_t element_count,
                                            const uint64_t point_count, const uint64_t *const corners,
                                            const cutl_allocator_t *const allocator, topo_mesh_t **const out)
{
    if (!allocator || !out)
        return TOPO_INVALID_ARGUMENT;
    if (ndim == 0 || ndim > 63)
        return TOPO_INVALID_ARGUMENT;
    if (element_count == 0 || point_count == 0)
        return TOPO_INVALID_ARGUMENT;

    corner_build_t build;
    topo_status_t status = corner_build(ndim, element_count, point_count, corners, allocator, &build);
    if (status != TOPO_SUCCESS)
        return status;

    topo_obj_immersion_t *const immersions = cutl_alloc(allocator, (size_t)ndim * sizeof(topo_obj_immersion_t));
    if (!immersions)
    {
        corner_build_release(&build, ndim, allocator);
        return TOPO_FAILED_ALLOC;
    }
    for (unsigned i = 0; i < ndim; ++i)
        immersions[i] = (topo_obj_immersion_t){0};

    status = topo_obj_create_immersion_info(ndim, (unsigned)point_count, build.collections, allocator, immersions);
    if (status != TOPO_SUCCESS)
    {
        topo_obj_immersions_free(ndim, immersions, allocator);
        cutl_dealloc(allocator, immersions);
        corner_build_release(&build, ndim, allocator);
        return status;
    }

    status = topo_mesh_create(ndim, point_count, build.collections, immersions, allocator, out);
    if (status != TOPO_SUCCESS)
    {
        topo_obj_immersions_free(ndim, immersions, allocator);
        cutl_dealloc(allocator, immersions);
        corner_build_release(&build, ndim, allocator);
    }
    return status;
}

void topo_mesh_free(topo_mesh_t *const mesh, const cutl_allocator_t *const allocator)
{
    if (!mesh)
        return;
    if (mesh->collections)
    {
        for (unsigned mdim = 1; mdim <= mesh->ndim; ++mdim)
            cutl_dealloc(allocator, mesh->collections[mdim - 1].boundary_ids);
        cutl_dealloc(allocator, mesh->collections);
    }
    if (mesh->immersions)
    {
        topo_obj_immersions_free(mesh->ndim, mesh->immersions, allocator);
        cutl_dealloc(allocator, mesh->immersions);
    }
    cutl_dealloc(allocator, mesh);
}

unsigned topo_mesh_ndim(const topo_mesh_t *const mesh)
{
    return mesh->ndim;
}

uint64_t topo_mesh_point_count(const topo_mesh_t *const mesh)
{
    return mesh->point_count;
}

uint64_t topo_mesh_element_count(const topo_mesh_t *const mesh)
{
    return mesh->element_count;
}

const topo_obj_collection_t *topo_mesh_collections(const topo_mesh_t *const mesh)
{
    return mesh->collections;
}

const topo_obj_immersion_t *topo_mesh_immersions(const topo_mesh_t *const mesh)
{
    return mesh->immersions;
}

topo_status_t topo_mesh_element_object(const topo_mesh_t *const mesh, const uint64_t element_id,
                                       const int8_t axis[const], uint64_t *const out)
{
    if (!mesh || !out)
        return TOPO_INVALID_ARGUMENT;
    if (element_id >= mesh->element_count)
        return TOPO_INVALID_ARGUMENT;

    // Decode the axis specification: fixed axes are the entries with a nonzero
    // value, and their side is the sign. The element itself spans every axis.
    uint64_t object_id = element_id;
    unsigned mdim = mesh->ndim;
    uint64_t mask = ((uint64_t)1 << mesh->ndim) - 1;
    unsigned fixed = 0;
    for (unsigned a = 0; a < mesh->ndim; ++a)
    {
        const int8_t value = axis[a];
        if (value == (int8_t)(a + 1) || value == -(int8_t)(a + 1))
        {
            fixed += 1;
        }
        else if (value != 0)
        {
            return TOPO_INVALID_ARGUMENT;
        }
    }
    if (fixed == 0)
        return TOPO_INVALID_ARGUMENT;

    // Descend the boundary chains: for every fixed axis, in ascending order,
    // cross from the current object into the boundary perpendicular to the
    // axis at the requested side. The slot of the boundary within the current
    // object is the rank of the axis among the object's spanning axes, plus
    // mdim when the boundary is the one at the end of the axis.
    for (unsigned a = 0; a < mesh->ndim; ++a)
    {
        const int8_t value = axis[a];
        if (value == 0)
            continue;
        uint64_t rank = 0;
        uint64_t lower = mask & (((uint64_t)1 << a) - 1);
        while (lower)
        {
            rank += lower & 1;
            lower >>= 1;
        }
        const uint64_t slot = value > 0 ? rank + mdim : rank;
        const topo_obj_collection_t *const collection = mesh->collections + (mdim - 1);
        object_id = collection->boundary_ids[object_id * (uint64_t)(2 * mdim) + slot];
        mask &= ~((uint64_t)1 << a);
        mdim -= 1;
    }
    *out = object_id;
    return TOPO_SUCCESS;
}

/**
 * Check whether an object of one dimension lies on the outer boundary of the
 * mesh.
 *
 * An object is on the boundary of the mesh when it is contained in a boundary
 * face: an object of dimension ``ndim - 1`` that is in exactly one element.
 * The orientation record of the object in one element gives, for every fixed
 * axis, the element's boundary face perpendicular to that axis at the same
 * side; the object is on the boundary when one of those faces is in a single
 * element.
 *
 * @param mesh Mesh to query.
 * @param g Object ID of dimension ``mdim``.
 * @param mdim Dimension of the object.
 * @param element_count[out] Receives the number of elements the object is in.
 * @param element_ids[out] Receives the element IDs.
 * @param orientations[out] Receives the orientation records.
 * @return true when the object lies on the outer boundary of the mesh.
 */
static bool mesh_object_on_boundary(const topo_mesh_t *const mesh, const uint64_t g, const unsigned mdim,
                                    uint64_t *const element_count, const uint64_t **const element_ids,
                                    const int8_t **const orientations)
{
    const unsigned ndim = mesh->ndim;
    const unsigned fixed = ndim - mdim;
    const topo_obj_immersion_t *const immersion = mesh->immersions + mdim;
    const topo_obj_immersion_t *const face_immersion = mesh->immersions + (ndim - 1);
    const topo_obj_collection_t *const element_collection = mesh->collections + (ndim - 1);

    topo_obj_immersion_of_object(immersion, g, element_count, element_ids, orientations);
    for (uint64_t i = 0; i < *element_count; ++i)
    {
        for (unsigned j = 0; j < fixed; ++j)
        {
            const int8_t value = (*orientations)[i * ndim + j];
            const unsigned axis = value < 0 ? (unsigned)(-(int)value) : (unsigned)value;
            const uint64_t slot = (uint64_t)(axis - 1) + (value > 0 ? ndim : 0);
            const uint64_t face_id = element_collection->boundary_ids[(*element_ids)[i] * (uint64_t)(2 * ndim) + slot];
            uint64_t face_count;
            const uint64_t *face_ids;
            const int8_t *face_orients;
            topo_obj_immersion_of_object(face_immersion, face_id, &face_count, &face_ids, &face_orients);
            if (face_count == 1)
                return true;
        }
    }
    return false;
}

/**
 * Iterate over the objects of one dimension that are contained in at least
 * ``min_count`` elements.
 *
 * @param mesh Mesh to iterate over.
 * @param mdim Dimension of the objects to iterate over.
 * @param min_count Objects with fewer elements are skipped.
 * @param callback Callback invoked for each visited object.
 * @param user_data Pointer passed to the callback.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if the arguments are
 *         invalid.
 */
static topo_status_t mesh_iterate(const topo_mesh_t *const mesh, const unsigned mdim, const uint64_t min_count,
                                  topo_mesh_callback_t callback, void *const user_data)
{
    if (!mesh || !callback)
        return TOPO_INVALID_ARGUMENT;
    if (mdim >= mesh->ndim)
        return TOPO_INVALID_ARGUMENT;

    const topo_obj_immersion_t *const immersion = mesh->immersions + mdim;
    for (uint64_t g = 0; g < immersion->object_count; ++g)
    {
        uint64_t element_count;
        const uint64_t *ids;
        const int8_t *orients;
        topo_obj_immersion_of_object(immersion, g, &element_count, &ids, &orients);
        if (element_count < min_count)
            continue;
        const topo_mesh_shared_object_t object = {
            .mdim = mdim, .object_id = g, .element_count = element_count, .element_ids = ids, .orientations = orients};
        callback(mesh, &object, user_data);
    }
    return TOPO_SUCCESS;
}

topo_status_t topo_mesh_iterate_shared(const topo_mesh_t *const mesh, const unsigned mdim,
                                       topo_mesh_callback_t callback, void *const user_data)
{
    return mesh_iterate(mesh, mdim, 2, callback, user_data);
}

topo_status_t topo_mesh_iterate_shared_all(const topo_mesh_t *const mesh, topo_mesh_callback_t callback,
                                           void *const user_data)
{
    if (!mesh || !callback)
        return TOPO_INVALID_ARGUMENT;
    for (unsigned mdim = mesh->ndim; mdim-- > 0;)
    {
        const topo_status_t status = topo_mesh_iterate_shared(mesh, mdim, callback, user_data);
        if (status != TOPO_SUCCESS)
            return status;
    }
    return TOPO_SUCCESS;
}

topo_status_t topo_mesh_iterate_boundary(const topo_mesh_t *const mesh, const unsigned mdim,
                                         topo_mesh_callback_t callback, void *const user_data)
{
    if (!mesh || !callback)
        return TOPO_INVALID_ARGUMENT;
    if (mdim >= mesh->ndim)
        return TOPO_INVALID_ARGUMENT;

    const topo_obj_immersion_t *const immersion = mesh->immersions + mdim;
    for (uint64_t g = 0; g < immersion->object_count; ++g)
    {
        uint64_t element_count;
        const uint64_t *ids;
        const int8_t *orients;
        if (!mesh_object_on_boundary(mesh, g, mdim, &element_count, &ids, &orients))
            continue;
        const topo_mesh_shared_object_t object = {
            .mdim = mdim, .object_id = g, .element_count = element_count, .element_ids = ids, .orientations = orients};
        callback(mesh, &object, user_data);
    }
    return TOPO_SUCCESS;
}

topo_status_t topo_mesh_iterate_boundary_all(const topo_mesh_t *const mesh, topo_mesh_callback_t callback,
                                             void *const user_data)
{
    if (!mesh || !callback)
        return TOPO_INVALID_ARGUMENT;
    for (unsigned mdim = mesh->ndim; mdim-- > 0;)
    {
        const topo_status_t status = topo_mesh_iterate_boundary(mesh, mdim, callback, user_data);
        if (status != TOPO_SUCCESS)
            return status;
    }
    return TOPO_SUCCESS;
}
