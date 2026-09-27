#ifndef FDG_MESH_OBJECTS_H
#define FDG_MESH_OBJECTS_H

#include "../topology/mesh.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    topo_mesh_t *mesh;
} mesh_object;

FDG_INTERNAL
extern PyType_Spec mesh_type_spec;

/**
 * @brief Check the collection preconditions of the topology layer.
 *
 * Re-checks what topo_mesh_create_from_collections() and
 * topo_obj_create_immersion_info() take on faith: boundary IDs inside the
 * range of the collection they index, a non-empty element collection, counts
 * that survive the casts to unsigned, and pairwise distinct boundary slots
 * per object. Every collection must already be a contiguous uint64 array of
 * shape (count, 2 * (idim + 1)).
 *
 * @param ndim[in] Number of collections, at least one.
 * @param point_count[in] Number of points named by IDs in the first collection.
 * @param collections[in] Collections of dimensions 1 through ndim.
 * @return 0 on success, -1 with a Python exception set otherwise.
 */
FDG_INTERNAL
int mesh_check_collections(const unsigned ndim, const uint64_t point_count,
                           const topo_obj_collection_t collections[static ndim]);

/**
 * @brief Report a topology status as the Python exception it maps to.
 *
 * The C core can only report resource failures any more - everything else
 * aborts through CUTL_ASSERT - so the status surfaces as MemoryError with the
 * context as message prefix.
 *
 * @param context[in] Message prefix, such as "Could not create mesh".
 * @param status[in] Resource status to report, not TOPO_SUCCESS.
 */
FDG_INTERNAL
void raise_topology_status(const char *context, topo_status_t status);

#endif // FDG_MESH_OBJECTS_H
