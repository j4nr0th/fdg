#pragma once
#include <cutl/allocators.h>
#include <stdint.h>

#include "topology.h"

/**
 * @brief Topological mesh built from connected hypercube elements.
 *
 * The mesh holds the complete topology of a set of hypercube elements: the
 * collections of all topological objects of every dimension, from the lines up
 * to the elements themselves, and the immersion information of all objects of
 * every dimension, from the points up to the element boundaries. The mesh has
 * no geometry; the elements are identified solely through their topology.
 *
 * The primary use is the generation of continuity constraints between
 * neighboring elements: iterate over boundary objects shared by multiple
 * elements (see @ref topo_mesh_iterate_shared_all) and assemble trace
 * constraints on each of them.
 *
 * The mesh can be built in three ways:
 *
 * - @ref topo_mesh_create_from_corners builds the collections from the corner
 *   points of every element (the caller names the shared points by giving the
 *   same point ID as a corner of every element it is in);
 * - @ref topo_mesh_create_from_collections builds the immersions from the given
 *   collections using @ref topo_obj_create_immersion_info;
 * - @ref topo_mesh_create accepts fully built collections and immersions and
 *   only stores them in the mesh.
 *
 * All three constructors take ownership of the passed arrays on success, so
 * the caller must not free them afterwards; every array must have been
 * allocated with the same allocator, and @ref topo_mesh_free releases all of
 * the memory it owns with it.
 */
typedef struct topo_mesh
{
    // Number of dimensions of the space the mesh is in.
    unsigned ndim;
    // Number of points in the mesh.
    uint64_t point_count;
    // Collections of objects of every dimension, entry d holding the objects of
    // dimension d + 1; the last entry holds the elements. [ndim]
    topo_obj_collection_t *collections;
    // Immersion information for objects of every dimension, entry d holding the
    // objects of dimension d, from the points up to the element boundaries. [ndim]
    topo_obj_immersion_t *immersions;
    // Number of elements of the mesh, equal to collections[ndim - 1].count.
    uint64_t element_count;
} topo_mesh_t;

/**
 * @brief Information about one topological object of a mesh.
 *
 * Passed to the callback of @ref topo_mesh_iterate_shared,
 * @ref topo_mesh_iterate_shared_all, @ref topo_mesh_iterate_boundary and
 * @ref topo_mesh_iterate_boundary_all. The shared objects are contained in at
 * least two elements; the boundary objects lie on the outer boundary of the
 * mesh (and may be contained in one or several elements, for example a point
 * in the middle of a boundary edge of a two-dimensional mesh).
 */
typedef struct
{
    // Dimension of the objects.
    unsigned mdim;
    // ID of the object. For ``mdim == 0`` this is a point ID into the implicit
    // point set; otherwise it is an index into the collection of objects with
    // that dimension.
    uint64_t object_id;
    // Number of elements the object is contained in.
    uint64_t element_count;
    // IDs of the elements the object is contained in, sorted in ascending
    // order. There are ``element_count`` entries.
    const uint64_t *element_ids;
    // Orientation records for all elements the object is in, ``ndim`` entries
    // per element. The record of element ``i`` has ``ndim`` entries and is at
    // offset ``i * ndim``.
    const int8_t *orientations;
} topo_mesh_shared_object_t;

/**
 * Callback invoked for each object during iteration.
 *
 * @param mesh The mesh that is being iterated over.
 * @param object Information about the object.
 * @param user_data Opaque pointer passed through from the iteration function.
 */
typedef void (*topo_mesh_callback_t)(const topo_mesh_t *mesh, const topo_mesh_shared_object_t *object, void *user_data);

/**
 * Callback invoked for each consecutive pair of elements containing a shared object.
 *
 * @param mesh The mesh being iterated.
 * @param mdim Dimension of the shared object.
 * @param object_id ID of the shared object.
 * @param element_id_1 First element ID in the pair.
 * @param orientation_1 Orientation of the object in the first element.
 * @param element_id_2 Second element ID in the pair.
 * @param orientation_2 Orientation of the object in the second element.
 * @param user_data Opaque pointer passed through from the iteration function.
 */
typedef void (*topo_mesh_pair_callback_t)(const topo_mesh_t *mesh, unsigned mdim, uint64_t object_id,
                                          uint64_t element_id_1, const int8_t *orientation_1, uint64_t element_id_2,
                                          const int8_t *orientation_2, void *user_data);

/**
 * Create a mesh from already computed collections and immersions.
 *
 * The mesh takes ownership of all passed arrays on success; each of them must
 * have been allocated with ``allocator`` and must not be freed by the caller
 * afterwards. On failure all the arrays stay with the caller.
 *
 * The collections describe the objects of every dimension from the lines up
 * to the elements; the immersions describe the objects of every dimension
 * from the points up to the element boundaries, in the layout produced by
 * @ref topo_obj_create_immersion_info.
 *
 * @param ndim[in] Number of dimensions of the space. Must be between 1 and 63.
 * @param point_count[in] Number of points in the mesh. Must be positive.
 * @param collections[in] Collections of the objects of dimensions 1 to ndim,
 *        transferred on success. Must have ``ndim`` entries.
 * @param immersions[in] Immersions of the objects of dimensions 0 to ndim - 1,
 *        transferred on success. Must have ``ndim`` entries.
 * @param allocator[in] Allocator used for all passed memory and the mesh.
 * @param out[out] Receives the created mesh. On failure it is left unmodified.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if an argument is
 *         invalid, TOPO_FAILED_ALLOC if a memory allocation fails.
 */
topo_status_t topo_mesh_create(unsigned ndim, uint64_t point_count, topo_obj_collection_t *collections,
                               topo_obj_immersion_t *immersions, const cutl_allocator_t *allocator, topo_mesh_t **out);

/**
 * Create a mesh from the collections of the topological objects.
 *
 * The mesh computes the immersion information from the collections using
 * @ref topo_obj_create_immersion_info, then stores both. On success it takes
 * ownership of the passed collections; on failure the collections stay with
 * the caller.
 *
 * @param ndim[in] Number of dimensions of the space. Must be between 1 and 63.
 * @param point_count[in] Number of points in the mesh. Must be positive.
 * @param collections[in] Collections of the objects of dimensions 1 to ndim,
 *        transferred on success. Must have ``ndim`` entries and be consistently
 *        oriented; the last entry holds the elements.
 * @param allocator[in] Allocator used for all passed memory and the mesh.
 * @param out[out] Receives the created mesh. On failure it is left unmodified.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if an argument is
 *         invalid, TOPO_SIZE_OVERFLOW, TOPO_FAILED_ALLOC or a topology error
 *         code if the collections are inconsistent.
 */
topo_status_t topo_mesh_create_from_collections(unsigned ndim, uint64_t point_count, topo_obj_collection_t *collections,
                                                const cutl_allocator_t *allocator, topo_mesh_t **out);

/**
 * Create a mesh from the corner points of every hypercube element.
 *
 * The caller provides, for every element, the IDs of its ``2^ndim`` corner
 * points, in the element's own frame: the corner with local mixed-radix
 * index ``k`` (bit ``a`` of ``k`` set = the corner lies on the positive side of
 * element axis ``a``) has its point ID at ``corners[element * 2^ndim + k]``.
 * A point that belongs to several elements is shared by giving its ID as a
 * corner of each of them.
 *
 * The mesh merges the topological objects implied by the corner points: two
 * objects are merged when they are made of the same set of corner points. The
 * point IDs must come from a consistent axis-aligned gluing: for every object
 * the corners must embed it identically in all of the elements it is in. The
 * input is validated as far as the merging machinery allows; inconsistent
 * inputs surface as errors from the immersion computation.
 *
 * @param ndim[in] Number of dimensions of the space. Must be between 1 and 63.
 * @param element_count[in] Number of elements of the mesh. Must be positive.
 * @param point_count[in] Number of points in the mesh. Must be greater than the
 *        largest corner point ID.
 * @param corners[in] Corner point IDs, ``element_count * 2^ndim`` entries.
 * @param allocator[in] Allocator used for all memory and the mesh.
 * @param out[out] Receives the created mesh. On failure it is left unmodified.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if an argument is
 *         invalid, TOPO_SIZE_OVERFLOW or TOPO_FAILED_ALLOC on failure, or a
 *         topology error code if the corner data is inconsistent.
 */
topo_status_t topo_mesh_create_from_corners(unsigned ndim, uint64_t element_count, uint64_t point_count,
                                            const uint64_t *corners, const cutl_allocator_t *allocator,
                                            topo_mesh_t **out);

/**
 * Release all memory owned by the mesh and free the mesh itself.
 *
 * @param mesh Mesh to free. May be NULL, in which case this function does nothing.
 * @param allocator Allocator with which the mesh and all its memory was allocated.
 */
void topo_mesh_free(topo_mesh_t *mesh, const cutl_allocator_t *allocator);

/**
 * Get the number of dimensions of the mesh.
 *
 * @param mesh Mesh to query.
 * @return Number of dimensions.
 */
unsigned topo_mesh_ndim(const topo_mesh_t *mesh);

/**
 * Get the number of points in the mesh.
 *
 * @param mesh Mesh to query.
 * @return Number of points in the mesh.
 */
uint64_t topo_mesh_point_count(const topo_mesh_t *mesh);

/**
 * Get the number of elements of the mesh.
 *
 * @param mesh Mesh to query.
 * @return Number of elements (objects of the highest dimension).
 */
uint64_t topo_mesh_element_count(const topo_mesh_t *mesh);

/**
 * Get the collections of topological objects of the mesh.
 *
 * @param mesh Mesh to query.
 * @return Array with ``topo_mesh_ndim(mesh)`` collections, owned by the mesh.
 *         Entry ``collections[d]`` holds the objects of dimension ``d + 1``,
 *         so the last entry holds the elements themselves.
 */
const topo_obj_collection_t *topo_mesh_collections(const topo_mesh_t *mesh);

/**
 * Get the immersion information of the mesh.
 *
 * @param mesh Mesh to query.
 * @return Array with ``topo_mesh_ndim(mesh)`` immersions, owned by the mesh.
 *         Entry ``immersions[d]`` holds the immersion of the objects of
 *         dimension ``d``, where the objects of dimension zero are the points.
 */
const topo_obj_immersion_t *topo_mesh_immersions(const topo_mesh_t *mesh);

/**
 * Look up the global ID of the object at a given position within one element.
 *
 * The position is specified with one signed, 1-based axis index per element
 * axis, following the immersion orientation convention (@ref topo_obj_immersion_t):
 * entry ``j`` is zero when the object spans element axis ``j``, ``-(j + 1)``
 * when the object is perpendicular to the axis at its start and ``+(j + 1)``
 * when it is perpendicular to the axis at its end. The dimension of the found
 * object is the number of free (zero) axes; a fully specified position
 * (no zero entries) identifies a point.
 *
 * For example, ``(-1, +2, -3)`` in three dimensions identifies the point at
 * the corner reached by following element axis 0 to its start, axis 1 to its
 * end and axis 2 to its start.
 *
 * The object is found by descending the boundary chains of the collections:
 * starting at the element, each fixed axis crosses from the current object
 * into the boundary perpendicular to it at the requested side. The lookup
 * takes time proportional to the number of fixed axes and needs no storage
 * beyond the collections themselves.
 *
 * @param mesh Mesh to query.
 * @param element_id[in] ID of the element. Must be in
 *        [0, topo_mesh_element_count(mesh)).
 * @param axis[in] Position of the object within the element, one entry per
 *        element axis. Entry ``j`` must be zero, ``-(j + 1)`` or ``+(j + 1)``;
 *        at least one entry must be nonzero.
 * @param out[out] Receives the global ID of the object. For points this is a
 *        point ID; otherwise it is an index into the collection of the
 *        respective object dimension.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if the element ID or
 *         an axis entry is invalid.
 */
topo_status_t topo_mesh_element_object(const topo_mesh_t *mesh, uint64_t element_id, const int8_t axis[],
                                       uint64_t *out);

/**
 * Iterate over all objects of one dimension that are shared by at least two
 * elements.
 *
 * The callback is invoked once per shared object of the given dimension, with
 * the element IDs sorted in ascending order. To build continuity constraints
 * without over-constraining, pair the consecutive elements of each shared
 * object and constrain only the interior of the object (skipping its own
 * boundaries).
 *
 * @param mesh Mesh to iterate over.
 * @param mdim[in] Dimension of the objects to iterate over. Must be in
 *        [0, ndim).
 * @param callback[in] Callback invoked for each shared object.
 * @param user_data Pointer passed to the callback.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if the arguments are
 *         invalid.
 */
topo_status_t topo_mesh_iterate_shared(const topo_mesh_t *mesh, unsigned mdim, topo_mesh_callback_t callback,
                                       void *user_data);

/**
 * Iterate over all shared objects of the mesh, from dimension ``ndim - 1``
 * down to dimension zero.
 *
 * This is a convenience wrapper around @ref topo_mesh_iterate_shared in the
 * order required for continuity constraints: constraining every boundary
 * object dimension by dimension, from the highest to the lowest, ensures that
 * no degree of freedom is constrained more than once.
 *
 * @param mesh Mesh to iterate over.
 * @param callback[in] Callback invoked for each shared object.
 * @param user_data Pointer passed to the callback.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if the arguments are
 *         invalid.
 */
topo_status_t topo_mesh_iterate_shared_all(const topo_mesh_t *mesh, topo_mesh_callback_t callback, void *user_data);

/**
 * Iterate over consecutive element pairs of all shared objects, from dimension
 * ``ndim - 1`` down to dimension zero.
 *
 * Each object contained in elements ``[e0, ..., eN]`` invokes the callback for
 * ``(e0, e1)``, ..., ``(eN-1, eN)``. This produces an acyclic spanning path
 * through every shared object's element occurrences.
 *
 * @param mesh Mesh to iterate over.
 * @param callback Callback invoked for every consecutive pair.
 * @param user_data Pointer passed to the callback.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if an argument is invalid.
 */
topo_status_t topo_mesh_iterate_shared_pairs(const topo_mesh_t *mesh, topo_mesh_pair_callback_t callback,
                                             void *user_data);

/**
 * Iterate over all objects of one dimension that lie on the outer boundary of
 * the mesh.
 *
 * An object lies on the outer boundary of the mesh when it is contained in a
 * boundary face: an object of dimension ``ndim - 1`` that is in exactly one
 * element. Objects of dimension ``ndim - 1`` on the boundary are therefore in
 * exactly one element; lower-dimensional objects can be in several elements
 * and still lie on the boundary, for example a point in the middle of a
 * boundary edge of a two-dimensional mesh.
 *
 * @param mesh Mesh to iterate over.
 * @param mdim[in] Dimension of the objects to iterate over. Must be in
 *        [0, ndim).
 * @param callback[in] Callback invoked for each boundary object.
 * @param user_data Pointer passed to the callback.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if the arguments are
 *         invalid.
 */
topo_status_t topo_mesh_iterate_boundary(const topo_mesh_t *mesh, unsigned mdim, topo_mesh_callback_t callback,
                                         void *user_data);

/**
 * Iterate over all boundary objects of the mesh, from dimension ``ndim - 1``
 * down to dimension zero.
 *
 * This is a convenience wrapper around @ref topo_mesh_iterate_boundary that
 * iterates every object dimension in descending order.
 *
 * @param mesh Mesh to iterate over.
 * @param callback[in] Callback invoked for each boundary object.
 * @param user_data Pointer passed to the callback.
 * @return TOPO_SUCCESS on success, TOPO_INVALID_ARGUMENT if the arguments are
 *         invalid.
 */
topo_status_t topo_mesh_iterate_boundary_all(const topo_mesh_t *mesh, topo_mesh_callback_t callback, void *user_data);
