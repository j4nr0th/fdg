#include "../../src/topology/mesh.h"
#include "../common/common.h"

#include <limits.h>
#include <stdbool.h>
#include <string.h>

typedef struct
{
    uint64_t count;
    uint64_t max_elements;
} shared_count_t;

static void count_shared(const topo_mesh_t *mesh, const topo_mesh_shared_object_t *object, void *user_data)
{
    (void)mesh;
    shared_count_t *const stats = (shared_count_t *)user_data;
    stats->count += 1;
    if (object->element_count > stats->max_elements)
        stats->max_elements = object->element_count;
}

typedef struct
{
    unsigned last_mdim;
    uint64_t counts[3];
} descending_order_t;

static void count_descending(const topo_mesh_t *mesh, const topo_mesh_shared_object_t *object, void *user_data)
{
    (void)mesh;
    descending_order_t *const order = (descending_order_t *)user_data;
    TEST_ASSERTION(object->mdim <= order->last_mdim, "Iteration over all dimensions is not descending.");
    order->last_mdim = object->mdim;
    order->counts[object->mdim] += 1;
}

typedef struct
{
    unsigned last_mdim;
    bool seen_dimension[3];
    uint64_t last_object[3];
    uint64_t pair_counts[3];
    uint64_t target_object;
    uint64_t target_pairs;
} pair_capture_t;

static void capture_pair(const topo_mesh_t *const mesh, const unsigned mdim, const uint64_t object_id,
                         const uint64_t element_id_1, const int8_t *const orientation_1, const uint64_t element_id_2,
                         const int8_t *const orientation_2, void *const user_data)
{
    pair_capture_t *const capture = user_data;
    TEST_ASSERTION(mdim < mesh->ndim && mdim < 3, "Pair iterator returned an invalid dimension.");
    TEST_ASSERTION(mdim <= capture->last_mdim, "Pair iteration is not dimension-descending.");
    capture->last_mdim = mdim;
    if (capture->seen_dimension[mdim])
        TEST_ASSERTION(object_id >= capture->last_object[mdim], "Pair iteration is not object-ID ascending.");
    capture->seen_dimension[mdim] = true;
    capture->last_object[mdim] = object_id;
    TEST_ASSERTION(element_id_1 < element_id_2, "Pair element IDs are not sorted.");

    uint64_t element_count;
    const uint64_t *element_ids;
    const int8_t *orientations;
    topo_obj_immersion_of_object(mesh->immersions + mdim, object_id, &element_count, &element_ids, &orientations);
    uint64_t pair_index = element_count;
    for (uint64_t index = 1; index < element_count; ++index)
    {
        if (element_ids[index - 1] == element_id_1 && element_ids[index] == element_id_2)
        {
            pair_index = index;
            break;
        }
    }
    TEST_ASSERTION(pair_index < element_count, "Pair callback did not match an immersion occurrence.");
    TEST_ASSERTION(
        memcmp(orientation_1, orientations + (pair_index - 1) * mesh->ndim, mesh->ndim * sizeof(*orientation_1)) == 0 &&
            memcmp(orientation_2, orientations + pair_index * mesh->ndim, mesh->ndim * sizeof(*orientation_2)) == 0,
        "Pair callback returned incorrect orientations.");
    capture->pair_counts[mdim] += 1;
    if (mdim == 0 && object_id == capture->target_object)
        capture->target_pairs += 1;
}

static void test_1d(void)
{
    // A chain of four elements, glued end to start: the topology of a single
    // line split into four elements. Element i spans points i and i + 1.
    const uint64_t corners[] = {0, 1, 1, 2, 2, 3, 3, 4};
    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    topo_mesh_t *mesh = NULL;
    TEST_ASSERTION(topo_mesh_create_from_corners(1, 4, 5, corners, allocator, &mesh) == TOPO_SUCCESS,
                   "Could not create 1D mesh.");

    TEST_ASSERTION(topo_mesh_ndim(mesh) == 1, "Unexpected mesh dimension.");
    TEST_ASSERTION(topo_mesh_point_count(mesh) == 5, "Unexpected point count.");
    TEST_ASSERTION(topo_mesh_element_count(mesh) == 4, "Unexpected element count.");

    // In 1D the elements are the lines themselves.
    const topo_obj_collection_t *const collections = topo_mesh_collections(mesh);
    TEST_ASSERTION(collections[0].ndim == 1 && collections[0].count == 4, "Unexpected line collection.");
    for (uint64_t line = 0; line < 4; ++line)
    {
        TEST_ASSERTION(collections[0].boundary_ids[2 * line] == line, "Element %llu has wrong start point.",
                       (unsigned long long)line);
        TEST_ASSERTION(collections[0].boundary_ids[2 * line + 1] == line + 1, "Element %llu has wrong end point.",
                       (unsigned long long)line);
    }

    // The interior points are shared by two elements, the end points by one.
    const topo_obj_immersion_t *const immersions = topo_mesh_immersions(mesh);
    TEST_ASSERTION(immersions[0].object_count == 5, "Unexpected point immersion count.");
    for (uint64_t point = 0; point < 5; ++point)
    {
        uint64_t cnt;
        const uint64_t *ids;
        const int8_t *orientations;
        topo_obj_immersion_of_object(immersions + 0, point, &cnt, &ids, &orientations);
        if (point == 0 || point == 4)
        {
            TEST_ASSERTION(cnt == 1, "End point %llu is shared.", (unsigned long long)point);
            TEST_ASSERTION(ids[0] == (point == 0 ? 0 : 3), "End point %llu in wrong element.",
                           (unsigned long long)point);
            TEST_ASSERTION(orientations[0] == (point == 0 ? -1 : +1), "End point %llu has wrong orientation.",
                           (unsigned long long)point);
        }
        else
        {
            TEST_ASSERTION(cnt == 2, "Interior point %llu is not shared.", (unsigned long long)point);
            TEST_ASSERTION(ids[0] == point - 1 && ids[1] == point, "Interior point %llu in wrong elements.",
                           (unsigned long long)point);
            TEST_ASSERTION(orientations[0] == +1 && orientations[1] == -1, "Interior point %llu has wrong orientation.",
                           (unsigned long long)point);
        }
    }

    // Only the three interior points are shared.
    shared_count_t points = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 0, count_shared, &points) == TOPO_SUCCESS,
                   "Could not iterate over shared points.");
    TEST_ASSERTION(points.count == 3, "Unexpected number of shared points.");
    TEST_ASSERTION(points.max_elements == 2, "Unexpected maximum element count of shared points.");

    // The eight boundary slots are the element endpoints: points 0 and 4.
    shared_count_t boundary = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_boundary(mesh, 0, count_shared, &boundary) == TOPO_SUCCESS,
                   "Could not iterate over boundary points.");
    TEST_ASSERTION(boundary.count == 2, "Unexpected number of boundary points.");

    // In 1D the lines are the elements; iterating over them is invalid.
    shared_count_t lines = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 1, count_shared, &lines) == TOPO_INVALID_ARGUMENT,
                   "Iteration over element-dimension objects was accepted.");

    // Lookup of an object position within an element.
    uint64_t object;
    const int8_t point_spec[] = {-1};
    TEST_ASSERTION(topo_mesh_element_object(mesh, 2, point_spec, &object) == TOPO_SUCCESS,
                   "Could not look up element point.");
    TEST_ASSERTION(object == 2, "Element 2 has wrong start point.");

    topo_mesh_free(mesh, allocator);
    topo_mesh_free(NULL, allocator);
}

// Point ID of the grid point (x, y) of a three-by-three grid.
static uint64_t grid2(const uint64_t x, const uint64_t y)
{
    return x + 3 * y;
}

static void test_2d(void)
{
    // A two by two block of unit squares. Element (ix, iy) has the points
    // (ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1).
    const uint64_t corners[16] = {
        grid2(0, 0), grid2(1, 0), grid2(0, 1), grid2(1, 1), // element 0
        grid2(1, 0), grid2(2, 0), grid2(1, 1), grid2(2, 1), // element 1
        grid2(0, 1), grid2(1, 1), grid2(0, 2), grid2(1, 2), // element 2
        grid2(1, 1), grid2(2, 1), grid2(1, 2), grid2(2, 2), // element 3
    };
    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    topo_mesh_t *mesh = NULL;
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 4, 9, corners, allocator, &mesh) == TOPO_SUCCESS,
                   "Could not create 2D mesh.");

    TEST_ASSERTION(topo_mesh_point_count(mesh) == 9, "Unexpected point count.");
    TEST_ASSERTION(topo_mesh_element_count(mesh) == 4, "Unexpected element count.");

    const topo_obj_collection_t *const collections = topo_mesh_collections(mesh);
    TEST_ASSERTION(collections[0].count == 12, "Unexpected line count.");
    TEST_ASSERTION(collections[1].count == 4, "Unexpected quad count.");

    // Lines in creation order: per element the lines parallel to the x axis
    // first, then those parallel to the y axis. The start point is at index
    // 2 * l. The grid point (x, y) has the ID x + 3*y. Repeated corner sets
    // from the element boundaries merge into the same line.
    const uint64_t expected_lines[12][2] = {
        {0, 1}, {3, 4}, {0, 3}, {1, 4}, // element 0
        {1, 2}, {4, 5}, {2, 5},         // element 1
        {6, 7}, {3, 6}, {4, 7},         // element 2
        {7, 8}, {5, 8},                 // element 3
    };
    for (uint64_t line = 0; line < 12; ++line)
    {
        TEST_ASSERTION(collections[0].boundary_ids[2 * line] == expected_lines[line][0],
                       "Line %llu has wrong start point.", (unsigned long long)line);
        TEST_ASSERTION(collections[0].boundary_ids[2 * line + 1] == expected_lines[line][1],
                       "Line %llu has wrong end point.", (unsigned long long)line);
    }

    // Quads: {x-start, y-start, x-end, y-end} boundary layout.
    const uint64_t expected_quads[4][4] = {
        {2, 0, 3, 1},
        {3, 4, 6, 5},
        {8, 1, 9, 7},
        {9, 5, 11, 10},
    };
    for (uint64_t quad = 0; quad < 4; ++quad)
    {
        for (unsigned i = 0; i < 4; ++i)
        {
            TEST_ASSERTION(collections[1].boundary_ids[4 * quad + i] == expected_quads[quad][i],
                           "Quad %llu has wrong boundary %u.", (unsigned long long)quad, i);
        }
    }

    // The vertical line at the shared x-interface (line 3) is shared by quads
    // 0 and 1.
    const topo_obj_immersion_t *const immersions = topo_mesh_immersions(mesh);
    {
        uint64_t cnt;
        const uint64_t *ids;
        const int8_t *orientations;
        topo_obj_immersion_of_object(immersions + 1, 3, &cnt, &ids, &orientations);
        TEST_ASSERTION(cnt == 2 && ids[0] == 0 && ids[1] == 1, "Line 3 is shared by the wrong quads.");
        TEST_ASSERTION(orientations[0] == +1 && orientations[1] == +2, "Line 3 has wrong orientation in quad 0.");
        TEST_ASSERTION(orientations[2] == -1 && orientations[3] == +2, "Line 3 has wrong orientation in quad 1.");
    }

    // The center point is in all four quads.
    {
        uint64_t cnt;
        const uint64_t *ids;
        const int8_t *orientations;
        topo_obj_immersion_of_object(immersions + 0, 4, &cnt, &ids, &orientations);
        TEST_ASSERTION(cnt == 4, "Center point is not in all four elements.");
        const int8_t expected[4][2] = {{+1, +2}, {-1, +2}, {+1, -2}, {-1, -2}};
        for (uint64_t i = 0; i < 4; ++i)
        {
            TEST_ASSERTION(ids[i] == i, "Center point in wrong element order.");
            TEST_ASSERTION(orientations[2 * i] == expected[i][0] && orientations[2 * i + 1] == expected[i][1],
                           "Center point has wrong orientation in element %llu.", (unsigned long long)i);
        }
    }

    // Shared objects: 4 shared lines (each in 2 elements) and 5 shared points
    // (4 on the interior grid lines in 2 elements, the center in all 4).
    shared_count_t lines = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 1, count_shared, &lines) == TOPO_SUCCESS,
                   "Could not iterate over shared lines.");
    TEST_ASSERTION(lines.count == 4, "Unexpected number of shared lines.");
    TEST_ASSERTION(lines.max_elements == 2, "Unexpected maximum element count of shared lines.");

    shared_count_t points = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 0, count_shared, &points) == TOPO_SUCCESS,
                   "Could not iterate over shared points.");
    TEST_ASSERTION(points.count == 5, "Unexpected number of shared points.");
    TEST_ASSERTION(points.max_elements == 4, "Unexpected maximum element count of shared points.");

    // The boundary is the outer ring: 8 lines and 8 points. The four edge
    // mid-points (in two elements each) lie on the boundary, too.
    shared_count_t boundary_lines = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_boundary(mesh, 1, count_shared, &boundary_lines) == TOPO_SUCCESS,
                   "Could not iterate over boundary lines.");
    TEST_ASSERTION(boundary_lines.count == 8, "Unexpected number of boundary lines.");
    shared_count_t boundary_points = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_boundary(mesh, 0, count_shared, &boundary_points) == TOPO_SUCCESS,
                   "Could not iterate over boundary points.");
    TEST_ASSERTION(boundary_points.count == 8, "Unexpected number of boundary points.");

    pair_capture_t pair_capture = {.last_mdim = UINT_MAX, .target_object = grid2(1, 1)};
    TEST_ASSERTION(topo_mesh_iterate_shared_pairs(mesh, capture_pair, &pair_capture) == TOPO_SUCCESS,
                   "Could not iterate over shared element pairs.");
    TEST_ASSERTION(pair_capture.pair_counts[1] == 4 && pair_capture.pair_counts[0] == 7 &&
                       pair_capture.target_pairs == 3,
                   "Shared pair iterator returned incorrect two-dimensional pair counts.");

    // Lookup of object positions within one element.
    uint64_t object;
    const int8_t point_spec[] = {-1, +2};
    TEST_ASSERTION(topo_mesh_element_object(mesh, 0, point_spec, &object) == TOPO_SUCCESS,
                   "Could not look up corner point.");
    TEST_ASSERTION(object == grid2(0, 1), "Corner point has wrong ID.");
    const int8_t edge_spec[] = {0, -2};
    TEST_ASSERTION(topo_mesh_element_object(mesh, 0, edge_spec, &object) == TOPO_SUCCESS,
                   "Could not look up boundary edge.");
    TEST_ASSERTION(object == 0, "Boundary edge has wrong ID.");

    // Invalid arguments are rejected.
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 2, count_shared, &points) == TOPO_INVALID_ARGUMENT,
                   "Iteration over an invalid dimension was accepted.");
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 1, NULL, &points) == TOPO_INVALID_ARGUMENT,
                   "Iteration with a null callback was accepted.");
    TEST_ASSERTION(topo_mesh_iterate_shared(NULL, 1, count_shared, &points) == TOPO_INVALID_ARGUMENT,
                   "Iteration over a null mesh was accepted.");
    const int8_t invalid_spec[] = {0, 0};
    TEST_ASSERTION(topo_mesh_element_object(mesh, 0, invalid_spec, &object) == TOPO_INVALID_ARGUMENT,
                   "Lookup of the element itself was accepted.");
    const int8_t bad_axis[] = {0, -1};
    TEST_ASSERTION(topo_mesh_element_object(mesh, 0, bad_axis, &object) == TOPO_INVALID_ARGUMENT,
                   "Lookup with an invalid axis entry was accepted.");

    topo_mesh_free(mesh, allocator);
}

// Point ID of the grid point (x, y, z) of a three-by-two-by-two grid.
static uint64_t grid3(const uint64_t x, const uint64_t y, const uint64_t z)
{
    return x + 3 * (y + 2 * z);
}

static void test_3d_two_cubes(void)
{
    // Two cubes glued together along their x-interface.
    uint64_t corners[16];
    for (uint64_t z = 0; z < 2; ++z)
    {
        for (uint64_t y = 0; y < 2; ++y)
        {
            for (uint64_t x = 0; x < 2; ++x)
            {
                const uint64_t k = x + 2 * (y + 2 * z);
                corners[0 * 8 + k] = grid3(x, y, z);
                corners[1 * 8 + k] = grid3(x + 1, y, z);
            }
        }
    }
    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    topo_mesh_t *mesh = NULL;
    TEST_ASSERTION(topo_mesh_create_from_corners(3, 2, 12, corners, allocator, &mesh) == TOPO_SUCCESS,
                   "Could not create 3D mesh.");

    TEST_ASSERTION(topo_mesh_point_count(mesh) == 12, "Unexpected point count.");
    TEST_ASSERTION(topo_mesh_element_count(mesh) == 2, "Unexpected element count.");
    const topo_obj_collection_t *const collections = topo_mesh_collections(mesh);
    TEST_ASSERTION(collections[0].count == 20, "Unexpected line count.");
    TEST_ASSERTION(collections[1].count == 11, "Unexpected face count.");

    // The two elements: {x-start, y-start, z-start, x-end, y-end, z-end} boundaries.
    const uint64_t expected_elements[2][6] = {
        {4, 2, 0, 5, 3, 1},
        {5, 8, 6, 10, 9, 7},
    };
    for (uint64_t element = 0; element < 2; ++element)
    {
        for (unsigned i = 0; i < 6; ++i)
        {
            TEST_ASSERTION(collections[2].boundary_ids[6 * element + i] == expected_elements[element][i],
                           "Element %llu has wrong boundary %u.", (unsigned long long)element, i);
        }
    }

    // The shared face (at the x-interface, x-plane 1 of element 0) is face 5,
    // shared by both elements.
    {
        uint64_t cnt;
        const uint64_t *ids;
        const int8_t *orientations;
        topo_obj_immersion_of_object(topo_mesh_immersions(mesh) + 2, 5, &cnt, &ids, &orientations);
        TEST_ASSERTION(cnt == 2 && ids[0] == 0 && ids[1] == 1, "Face 5 is shared by the wrong elements.");
        TEST_ASSERTION(orientations[0] == +1 && orientations[1] == +2 && orientations[2] == +3,
                       "Face 5 has wrong orientation in element 0.");
        TEST_ASSERTION(orientations[3] == -1 && orientations[4] == +2 && orientations[5] == +3,
                       "Face 5 has wrong orientation in element 1.");
    }

    // Shared objects: 1 face, 4 lines, and 4 points, all shared by both elements.
    shared_count_t faces = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 2, count_shared, &faces) == TOPO_SUCCESS,
                   "Could not iterate over shared faces.");
    TEST_ASSERTION(faces.count == 1, "Unexpected number of shared faces.");
    TEST_ASSERTION(faces.max_elements == 2, "Unexpected maximum element count of shared faces.");

    shared_count_t lines = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 1, count_shared, &lines) == TOPO_SUCCESS,
                   "Could not iterate over shared lines.");
    TEST_ASSERTION(lines.count == 4, "Unexpected number of shared lines.");
    TEST_ASSERTION(lines.max_elements == 2, "Unexpected maximum element count of shared lines.");

    shared_count_t points = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 0, count_shared, &points) == TOPO_SUCCESS,
                   "Could not iterate over shared points.");
    TEST_ASSERTION(points.count == 4, "Unexpected number of shared points.");
    TEST_ASSERTION(points.max_elements == 2, "Unexpected maximum element count of shared points.");

    // Iteration over all dimensions descends from the highest to the lowest,
    // with one face, four lines, and four points in total.
    descending_order_t order = {UINT_MAX, {0, 0, 0}};
    TEST_ASSERTION(topo_mesh_iterate_shared_all(mesh, NULL, NULL) == TOPO_INVALID_ARGUMENT,
                   "Null callback was accepted by iteration over all dims.");
    TEST_ASSERTION(topo_mesh_iterate_shared_all(mesh, count_descending, &order) == TOPO_SUCCESS,
                   "Could not iterate over all shared objects.");
    TEST_ASSERTION(order.counts[2] == 1 && order.counts[1] == 4 && order.counts[0] == 4,
                   "Iteration over all dims visited the wrong objects.");

    // Boundary objects: 10 faces, 16 lines, and 8 points.
    descending_order_t boundary_order = {UINT_MAX, {0, 0, 0}};
    TEST_ASSERTION(topo_mesh_iterate_boundary_all(mesh, count_descending, &boundary_order) == TOPO_SUCCESS,
                   "Could not iterate over all boundary objects.");
    // 10 boundary faces. All 20 lines and all 12 points lie on the outer
    // boundary of the box: the four lines and eight points of the internal
    // face are contained in boundary faces perpendicular to them.
    TEST_ASSERTION(boundary_order.counts[2] == 10 && boundary_order.counts[1] == 20 && boundary_order.counts[0] == 12,
                   "Boundary iteration visited the wrong objects.");

    topo_mesh_free(mesh, allocator);
}

static void test_3d_eight_cubes(void)
{
    // A two by two by two block of unit cubes on a three-by-three-by-three
    // grid; the point (x, y, z) has the ID x + 3*y + 9*z. Cube (cx, cy, cz)
    // is element cx + 2 * (cy + 2 * cz), corner with local offset (bx, by,
    // bz) has the mixed-radix index bx + 2 * by + 4 * bz.
    uint64_t corners[8 * 8];
    for (uint64_t cz = 0; cz < 2; ++cz)
    {
        for (uint64_t cy = 0; cy < 2; ++cy)
        {
            for (uint64_t cx = 0; cx < 2; ++cx)
            {
                const uint64_t element = cx + 2 * (cy + 2 * cz);
                for (uint64_t bz = 0; bz < 2; ++bz)
                {
                    for (uint64_t by = 0; by < 2; ++by)
                    {
                        for (uint64_t bx = 0; bx < 2; ++bx)
                        {
                            const uint64_t k = bx + 2 * by + 4 * bz;
                            corners[element * 8 + k] = (cx + bx) + 3 * ((cy + by) + 3 * (cz + bz));
                        }
                    }
                }
            }
        }
    }
    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    topo_mesh_t *mesh = NULL;
    TEST_ASSERTION(topo_mesh_create_from_corners(3, 8, 27, corners, allocator, &mesh) == TOPO_SUCCESS,
                   "Could not create 3D mesh.");

    TEST_ASSERTION(topo_mesh_point_count(mesh) == 27, "Unexpected point count.");
    TEST_ASSERTION(topo_mesh_element_count(mesh) == 8, "Unexpected element count.");
    const topo_obj_collection_t *const collections = topo_mesh_collections(mesh);
    TEST_ASSERTION(collections[0].count == 54, "Unexpected line count.");
    TEST_ASSERTION(collections[1].count == 36, "Unexpected face count.");

    // Shared objects: 12 interior faces in 2 elements, 30 shared lines (up to 4
    // elements), 19 shared points (up to 8 elements).
    shared_count_t faces = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 2, count_shared, &faces) == TOPO_SUCCESS,
                   "Could not iterate over shared faces.");
    TEST_ASSERTION(faces.count == 12, "Unexpected number of shared faces.");
    TEST_ASSERTION(faces.max_elements == 2, "Unexpected maximum element count of shared faces.");

    shared_count_t lines = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 1, count_shared, &lines) == TOPO_SUCCESS,
                   "Could not iterate over shared lines.");
    TEST_ASSERTION(lines.count == 30, "Unexpected number of shared lines.");
    TEST_ASSERTION(lines.max_elements == 4, "Unexpected maximum element count of shared lines.");

    shared_count_t points = {0, 0};
    TEST_ASSERTION(topo_mesh_iterate_shared(mesh, 0, count_shared, &points) == TOPO_SUCCESS,
                   "Could not iterate over shared points.");
    TEST_ASSERTION(points.count == 19, "Unexpected number of shared points.");
    TEST_ASSERTION(points.max_elements == 8, "Unexpected maximum element count of shared points.");

    // The center point is in all 8 elements, with orientations matching the
    // cube layout.
    {
        uint64_t cnt;
        const uint64_t *ids;
        const int8_t *orientations;
        topo_obj_immersion_of_object(topo_mesh_immersions(mesh) + 0, 13, &cnt, &ids, &orientations);
        TEST_ASSERTION(cnt == 8, "Center point is not in all eight elements.");
        for (uint64_t i = 0; i < 8; ++i)
        {
            TEST_ASSERTION(ids[i] == i, "Center point in wrong element order.");
            const int8_t ex = (i & 1U) ? -1 : +1;
            const int8_t ey = (i & 2U) ? -2 : +2;
            const int8_t ez = (i & 4U) ? -3 : +3;
            TEST_ASSERTION(orientations[3 * i] == ex && orientations[3 * i + 1] == ey && orientations[3 * i + 2] == ez,
                           "Center point has wrong orientation in element %llu.", (unsigned long long)i);
        }
    }

    // The center point shared by all 8 elements is the most connected point.
    {
        uint64_t max_count = 0;
        const topo_obj_immersion_t *const immersion = topo_mesh_immersions(mesh) + 0;
        for (uint64_t point = 0; point < immersion->object_count; ++point)
        {
            uint64_t cnt;
            const uint64_t *ids;
            const int8_t *orientations;
            topo_obj_immersion_of_object(immersion, point, &cnt, &ids, &orientations);
            if (cnt > max_count)
                max_count = cnt;
        }
        TEST_ASSERTION(max_count == 8, "Unexpected maximum point connectivity.");
    }

    pair_capture_t pair_capture = {.last_mdim = UINT_MAX, .target_object = 13};
    TEST_ASSERTION(topo_mesh_iterate_shared_pairs(mesh, capture_pair, &pair_capture) == TOPO_SUCCESS,
                   "Could not iterate over shared element pairs.");
    uint64_t expected_pairs[3] = {0, 0, 0};
    const topo_obj_immersion_t *const pair_immersions = topo_mesh_immersions(mesh);
    for (unsigned mdim = 0; mdim < 3; ++mdim)
    {
        for (uint64_t object_id = 0; object_id < pair_immersions[mdim].object_count; ++object_id)
        {
            uint64_t element_count;
            const uint64_t *unused_ids;
            const int8_t *unused_orientations;
            topo_obj_immersion_of_object(pair_immersions + mdim, object_id, &element_count, &unused_ids,
                                         &unused_orientations);
            if (element_count > 1)
                expected_pairs[mdim] += element_count - 1;
        }
    }
    TEST_ASSERTION(pair_capture.pair_counts[2] == expected_pairs[2] &&
                       pair_capture.pair_counts[1] == expected_pairs[1] && pair_capture.pair_counts[0] == 37 &&
                       pair_capture.target_pairs == 7,
                   "Shared pair iterator returned incorrect three-dimensional pair counts.");

    topo_mesh_free(mesh, allocator);
}

static void *failing_allocate(void *state, size_t size)
{
    (void)state;
    (void)size;
    return NULL;
}

static void *failing_reallocate(void *state, void *ptr, size_t new_size)
{
    (void)state;
    (void)ptr;
    (void)new_size;
    return NULL;
}

static void failing_deallocate(void *state, void *ptr)
{
    (void)state;
    (void)ptr;
}

static void test_errors(void)
{
    const cutl_allocator_t *allocator = cutl_allocator_get_default();
    topo_mesh_t *mesh = (topo_mesh_t *)(void *)0x1;

    TEST_ASSERTION(topo_mesh_create_from_corners(0, 1, 2, NULL, allocator, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Zero-dimensional mesh was accepted.");
    TEST_ASSERTION(topo_mesh_create_from_corners(64, 1, 2, NULL, allocator, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Overly high-dimensional mesh was accepted.");
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 0, 1, NULL, allocator, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Mesh without elements was accepted.");
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 1, 0, NULL, allocator, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Mesh without points was accepted.");
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 1, 1, NULL, allocator, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Null corners were accepted.");
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 1, 4, NULL, allocator, NULL) == TOPO_INVALID_ARGUMENT,
                   "Null output pointer was accepted.");
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 1, 4, NULL, NULL, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Null allocator was accepted.");

    // Corner IDs must name existing points.
    const uint64_t bad_point[] = {0, 1, 2, 4};
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 1, 4, bad_point, allocator, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Out-of-range corner point was accepted.");

    // The corners of one element must be distinct.
    const uint64_t bad_corners[] = {0, 1, 2, 1};
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 1, 4, bad_corners, allocator, &mesh) == TOPO_INVALID_ARGUMENT,
                   "Repeated corner point was accepted.");

    // An allocator that always fails must be reported as such, and the output
    // must be left unmodified.
    static const cutl_allocator_t failing_allocator = {
        .state = NULL,
        .allocate = failing_allocate,
        .deallocate = failing_deallocate,
        .reallocate = failing_reallocate,
    };
    const uint64_t good_corners[] = {0, 1, 2, 3};
    TEST_ASSERTION(topo_mesh_create_from_corners(2, 1, 4, good_corners, &failing_allocator, &mesh) == TOPO_FAILED_ALLOC,
                   "Failing allocator was not reported.");
    TEST_ASSERTION(mesh == (topo_mesh_t *)(void *)0x1, "Output pointer was modified on failure.");
}

int main(void)
{
    test_1d();
    test_2d();
    test_3d_two_cubes();
    test_3d_eight_cubes();
    test_errors();
    return 0;
}
