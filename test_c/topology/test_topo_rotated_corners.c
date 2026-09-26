#include "../../src/topology/mesh.h"
#include "../common/common.h"

#include <string.h>

#define POINT_COUNT 27
#define ELEMENT_COUNT 8

// Point id of grid node (ix, iy, iz) on a 3x3x3 node grid, last axis fastest.
static uint64_t grid_point(const unsigned ix, const unsigned iy, const unsigned iz)
{
    return (uint64_t)(9 * (unsigned)ix + 3 * (unsigned)iy + (unsigned)iz);
}

// Standard corner lists: element flat id = 4 ex + 2 ey + ez, local corner bit
// a = axis a.
static void build_corners(uint64_t corners[ELEMENT_COUNT * 8])
{
    for (unsigned ez = 0; ez < 2; ++ez)
        for (unsigned ey = 0; ey < 2; ++ey)
            for (unsigned ex = 0; ex < 2; ++ex)
            {
                const uint64_t flat = 4 * ex + 2 * ey + ez;
                for (unsigned local = 0; local < 8; ++local)
                {
                    const unsigned cx = local & 1;
                    const unsigned cy = (local >> 1) & 1;
                    const unsigned cz = (local >> 2) & 1;
                    corners[flat * 8 + local] = grid_point(ex + cx, ey + cy, ez + cz);
                }
            }
}

// Rename local bit axis a to physical axis permutation[a].
static unsigned permute_bits(const unsigned entry, const unsigned permutation[3])
{
    unsigned out = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
        out += ((entry >> axis) & 1) << permutation[axis];
    return out;
}

typedef struct
{
    uint64_t face_04;
    uint64_t face_02;
    uint64_t face_45;
    uint64_t face_46;
    int8_t orientation_04[2][3];
    int8_t orientation_45[2][3];
    int8_t orientation_46[2][3];
    uint64_t element_04[2];
    uint64_t element_45[2];
    uint64_t element_46[2];
} shared_faces_t;

static void collect_face_pair(const topo_mesh_t *mesh, const unsigned mdim, const uint64_t object_id,
                              const uint64_t element_id_1, const int8_t *orientation_1, const uint64_t element_id_2,
                              const int8_t *orientation_2, void *user_data)
{
    (void)mesh;
    if (mdim != 2)
        return;
    shared_faces_t *const faces = user_data;
    const int8_t *const orientations[2] = {orientation_1, orientation_2};
    if ((element_id_1 == 0 && element_id_2 == 4) || (element_id_1 == 4 && element_id_2 == 0))
    {
        memcpy(faces->orientation_04[0], orientations[0], 3 * sizeof(int8_t));
        memcpy(faces->orientation_04[1], orientations[1], 3 * sizeof(int8_t));
        faces->element_04[0] = element_id_1;
        faces->element_04[1] = element_id_2;
        faces->face_04 = object_id;
    }
    else if ((element_id_1 == 0 && element_id_2 == 2) || (element_id_1 == 2 && element_id_2 == 0))
    {
        faces->face_02 = object_id;
    }
    else if ((element_id_1 == 4 && element_id_2 == 5) || (element_id_1 == 5 && element_id_2 == 4))
    {
        memcpy(faces->orientation_45[0], orientations[0], 3 * sizeof(int8_t));
        memcpy(faces->orientation_45[1], orientations[1], 3 * sizeof(int8_t));
        faces->element_45[0] = element_id_1;
        faces->element_45[1] = element_id_2;
        faces->face_45 = object_id;
    }
    else if ((element_id_1 == 4 && element_id_2 == 6) || (element_id_1 == 6 && element_id_2 == 4))
    {
        memcpy(faces->orientation_46[0], orientations[0], 3 * sizeof(int8_t));
        memcpy(faces->orientation_46[1], orientations[1], 3 * sizeof(int8_t));
        faces->element_46[0] = element_id_1;
        faces->element_46[1] = element_id_2;
        faces->face_46 = object_id;
    }
}

// One-based signed axis code for the face at (axis, side): positive at the end
// side of the axis, negative at its start side.
static int8_t face_axis_code(const unsigned axis, const int end)
{
    return (int8_t)(end ? (int)(axis + 1) : -(int)(axis + 1));
}

static uint64_t element_face(const topo_mesh_t *mesh, const uint64_t element_id, const unsigned axis, const int end)
{
    // A face of an element is fixed by its single perpendicular axis.
    const int8_t code = face_axis_code(axis, end);
    uint64_t out = UINT64_MAX;
    topo_mesh_element_object(mesh, element_id, 1, &code, &out);
    return out;
}

// The face boundary row of one element must contain six distinct objects.
static void assert_face_row_distinct(const topo_mesh_t *mesh, const uint64_t element_id)
{
    const topo_obj_collection_t *const faces = topo_mesh_collections(mesh) + 2;
    for (unsigned slot_a = 0; slot_a < 6; ++slot_a)
        for (unsigned slot_b = slot_a + 1; slot_b < 6; ++slot_b)
            TEST_ASSERTION(faces->boundary_ids[element_id * 6 + slot_a] != faces->boundary_ids[element_id * 6 + slot_b],
                           "Face row of element %llu has duplicate boundary ids at slots %u and %u.",
                           (unsigned long long)element_id, slot_a, slot_b);
}

static void assert_orientation(const int8_t *actual, const int8_t expected[3], const char *context)
{
    TEST_ASSERTION(actual[0] == expected[0] && actual[1] == expected[1] && actual[2] == expected[2],
                   "%s: orientation is {%d, %d, %d}, expected {%d, %d, %d}.", context, actual[0], actual[1], actual[2],
                   expected[0], expected[1], expected[2]);
}

// The unpermuted control mesh: every face lookup of element 0 is distinct and
// agrees with the shared face objects.
static void test_canonical_mesh(void)
{
    uint64_t corners[ELEMENT_COUNT * 8];
    build_corners(corners);
    topo_mesh_t *mesh = NULL;
    TEST_ASSERTION(topo_mesh_create_from_corners(3, ELEMENT_COUNT, POINT_COUNT, corners, &TEST_ALLOCATOR, &mesh) ==
                       TOPO_SUCCESS,
                   "Could not create the canonical mesh.");

    shared_faces_t faces = {0};
    TEST_ASSERTION(topo_mesh_iterate_shared_pairs(mesh, collect_face_pair, &faces) == TOPO_SUCCESS,
                   "Could not iterate shared pairs of the canonical mesh.");

    TEST_ASSERTION(element_face(mesh, 0, 0, 1) == faces.face_04, "Element 0 x-end is not the shared (0, 4) face.");
    TEST_ASSERTION(element_face(mesh, 0, 1, 1) == faces.face_02, "Element 0 y-end is not the shared (0, 2) face.");
    assert_face_row_distinct(mesh, 0);

    topo_mesh_free(mesh, &TEST_ALLOCATOR);
}

// Element 4 with its corners renamed by the cyclic axis permutation (1, 2, 0):
// local axes (0, 1, 2) map to physical (y, z, x).
static void test_rotated_element(void)
{
    uint64_t corners[ELEMENT_COUNT * 8];
    build_corners(corners);
    const unsigned rotation[3] = {1, 2, 0};
    uint64_t original[8];
    memcpy(original, corners + 4 * 8, sizeof(original));
    for (unsigned local = 0; local < 8; ++local)
        corners[4 * 8 + local] = original[permute_bits(local, rotation)];

    topo_mesh_t *mesh = NULL;
    TEST_ASSERTION(topo_mesh_create_from_corners(3, ELEMENT_COUNT, POINT_COUNT, corners, &TEST_ALLOCATOR, &mesh) ==
                       TOPO_SUCCESS,
                   "Could not create the rotated mesh.");

    shared_faces_t faces = {0};
    TEST_ASSERTION(topo_mesh_iterate_shared_pairs(mesh, collect_face_pair, &faces) == TOPO_SUCCESS,
                   "Could not iterate shared pairs of the rotated mesh.");
    TEST_ASSERTION(faces.face_04 != 0 && faces.face_45 != 0 && faces.face_46 != 0,
                   "Missing expected face pairs around element 4.");

    // Local frame of element 4: axis 0 = physical y, axis 1 = physical z,
    // axis 2 = physical x.
    TEST_ASSERTION(element_face(mesh, 4, 0, 1) == faces.face_46,
                   "Element 4 local y-end is not the shared (4, 6) face: got %llu, expected %llu.",
                   (unsigned long long)element_face(mesh, 4, 0, 1), (unsigned long long)faces.face_46);
    TEST_ASSERTION(element_face(mesh, 4, 1, 1) == faces.face_45,
                   "Element 4 local z-end is not the shared (4, 5) face.");
    TEST_ASSERTION(element_face(mesh, 4, 2, 0) == faces.face_04,
                   "Element 4 local x-start is not the shared (0, 4) face.");
    assert_face_row_distinct(mesh, 4);

    // Immersion records. Position entry: element-local 1-based signed axis.
    // Mapping entries are indexed by the shared face object's own axes, which
    // follow the declaring element's local frame with the fixed bit removed,
    // and give the element-local 1-based axis each face axis corresponds to.
    // Shared (4, 6) face, declared by element 4 (local (y, z, x) minus y ->
    // face axes (z, x)): element 4 = {+1, z -> local 1, x -> local 2} =
    // {1, 2, 3}; element 6 = {-2, z -> 3, x -> 1} = {-2, 3, 1}.
    // Shared (0, 4) face, declared by element 0 (canonical minus x -> face
    // axes (y, z)): element 4 = {-3, y -> local 0, z -> local 1} = {-3, 1, 2};
    // element 0 = {1, y -> 2, z -> 3} = {1, 2, 3}.
    const int8_t record_46_4[3] = {1, 2, 3};
    const int8_t record_46_6[3] = {-2, 3, 1};
    const int8_t record_04_4[3] = {-3, 1, 2};
    const int8_t record_04_0[3] = {1, 2, 3};
    for (unsigned side = 0; side < 2; ++side)
    {
        assert_orientation(faces.orientation_46[side], faces.element_46[side] == 4 ? record_46_4 : record_46_6,
                           "Record for (4, 6)");
        assert_orientation(faces.orientation_04[side], faces.element_04[side] == 4 ? record_04_4 : record_04_0,
                           "Record for (0, 4)");
    }
    topo_mesh_free(mesh, &TEST_ALLOCATOR);
}

int main(void)
{
    test_canonical_mesh();
    test_rotated_element();
    return 0;
}
