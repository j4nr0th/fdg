/**
 * @file test_direct_continuity.c
 * @brief Tests of the direct shared-DoF continuity map.
 *
 * The map's contract is checked against counts a conforming nodal space must have, and against continuity
 * itself: two elements sharing an object must land on the very same global DoFs. That is the property the whole
 * formulation exists for, so it is asserted directly rather than inferred from a solve.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/basis/basis_set.h"
#include "../../src/constraints/direct.h"
#include "../../src/kforms/kform_types.h"
#include "../../src/topology/mesh.h"
#include "../common/common.h"

/**
 * @brief Bounds of the tests' per-element scratch.
 *
 * Only has to cover what the tested meshes produce: three dimensions, and the functions one object's trace
 * reaches per axis.
 */
enum
{
    DIRECT_TEST_MAX_AXES = 8,
    DIRECT_TEST_MAX_ENTRIES = 4096,
    DIRECT_TEST_MAX_CORNERS = 8,
    DIRECT_TEST_MAX_ELEMENTS = 64,
};

/**
 * @brief One adversarial case: a mesh, a k-form order and a basis order per element axis.
 *
 * The older helpers only ever put two uniform orders on a mesh, which does not reach the cases that break the
 * map: anisotropy inside one element, one rich element among poor ones, and every family crossed with every
 * k-form order. One case carries all of it, so an adversarial test only has to describe the mesh it wants.
 */
typedef struct
{
    unsigned ndim;           ///< Element dimension.
    unsigned cells;          ///< Cells per axis of the structured grid; zero for a pinwheel.
    unsigned kform_order;    ///< Traced k-form order.
    basis_set_type_t family; ///< Lagrange family every axis of every element uses.
    uint64_t element_count;  ///< Number of elements.
    uint64_t point_count;    ///< Number of mesh points.
    uint64_t corners[DIRECT_TEST_MAX_ELEMENTS * DIRECT_TEST_MAX_CORNERS]; ///< Corner point IDs, per element.
    unsigned orders[DIRECT_TEST_MAX_ELEMENTS * DIRECT_TEST_MAX_AXES];     ///< Basis orders, per element axis.
} direct_case_t;

/**
 * @brief One mesh, k-form and order, prepared with its direct map and its transfer.
 */
typedef struct
{
    topo_mesh_t *mesh;
    direct_continuity_request_t request;
    direct_continuity_plan_t plan;
    direct_continuity_work_t work;
    kform_spec_t *spec_storage;
    basis_spec_t *basis_storage;
    const kform_spec_t **specs;
    direct_continuity_layout_t layout;
    size_t *entry_offsets;
    size_t *entry_index;
    double *entry_value;
    void *plan_memory;
    void *work_memory;
    basis_set_registry_t *basis_registry;
    integration_rule_registry_t *integration_registry;
    uint64_t element_count;
} direct_test_t;

/**
 * @brief Row-major index of one element axis's basis order inside a #direct_case_t.
 */
#define DIRECT_CASE_ORDER(case, element, axis) ((case)->orders[(element) * DIRECT_TEST_MAX_AXES + (axis)])

/**
 * @brief Structured grid corners of an axis-aligned mesh of `cells` cells per axis.
 *
 * Corner IDs run over the tensor lattice with the first axis slowest, matching #topo_mesh_create_from_corners.
 *
 * @param ndim Dimension.
 * @param cells Per-axis cell count.
 * @param out_corners [2^ndim * cells^ndim] Receives the corner IDs.
 * @param out_points Receives the lattice point count.
 */
static void structured_corners(const unsigned ndim, const unsigned cells, uint64_t *const out_corners,
                               uint64_t *const out_points)
{
    const unsigned corners = 1u << ndim;
    uint64_t lattice[DIRECT_TEST_MAX_AXES];
    uint64_t points = 1;
    uint64_t element_count = 1;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        lattice[axis] = (uint64_t)cells + 1u;
        points *= lattice[axis];
        element_count *= cells;
    }
    *out_points = points;

    for (uint64_t element = 0; element < element_count; ++element)
    {
        uint64_t rest = element;
        uint64_t lower[DIRECT_TEST_MAX_AXES];
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            lower[axis] = rest % cells;
            rest /= cells;
        }
        for (unsigned corner = 0; corner < corners; ++corner)
        {
            uint64_t id = 0;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                id = id * lattice[axis] + lower[axis] + ((corner >> axis) & 1u);
            }
            out_corners[element * corners + corner] = id;
        }
    }
}

/**
 * @brief Fill a case with a structured grid mesh, leaving the basis orders at zero.
 */
static void direct_case_grid(direct_case_t *const out, const unsigned ndim, const unsigned cells,
                             const unsigned kform_order)
{
    memset(out, 0, sizeof(*out));
    out->cells = cells;
    out->ndim = ndim;
    out->kform_order = kform_order;
    out->family = BASIS_LAGRANGE_GAUSS_LOBATTO;
    out->element_count = 1;
    for (unsigned axis = 0; axis < ndim; ++axis)
    {
        out->element_count *= cells;
    }
    TEST_ASSERTION(out->element_count <= DIRECT_TEST_MAX_ELEMENTS, "The case holds more elements than the tests do.");
    structured_corners(ndim, cells, out->corners, &out->point_count);
}

/**
 * @brief Fill a case with a structured grid mesh whose every element carries one order on every axis.
 */
static void direct_case_uniform(direct_case_t *const out, const unsigned ndim, const unsigned cells,
                                const unsigned kform_order, const unsigned order)
{
    direct_case_grid(out, ndim, cells, kform_order);
    for (uint64_t element = 0; element < out->element_count; ++element)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            DIRECT_CASE_ORDER(out, element, axis) = order;
        }
    }
}

/**
 * @brief Fill a case with elements that meet in one common corner point and share nothing else.
 *
 * The elements occupy the orthants of the even-parity code, so any two of them differ on at least two axes and
 * therefore have the origin as their only common object. A structured grid cannot produce this: its elements
 * meeting at an interior point always share a face through it as well. The origin is point zero and the points
 * are numbered densely in first-use order, so the mesh declares no point that no element carries.
 */
static void direct_case_corner_fan(direct_case_t *const out, const unsigned ndim, const unsigned kform_order,
                                   const unsigned element_count)
{
    const unsigned corners = 1u << ndim;
    memset(out, 0, sizeof(*out));
    out->ndim = ndim;
    out->kform_order = kform_order;
    out->family = BASIS_LAGRANGE_GAUSS_LOBATTO;
    out->element_count = element_count;
    int8_t coordinates[DIRECT_TEST_MAX_ELEMENTS * DIRECT_TEST_MAX_CORNERS][DIRECT_TEST_MAX_AXES];
    unsigned next = 0;
    for (unsigned element = 0; element < element_count; ++element)
    {
        unsigned sides[DIRECT_TEST_MAX_AXES] = {0u};
        if (ndim == 1u)
        {
            // A one-dimensional mesh cannot differ on two axes, so two intervals sharing one point is the
            // degenerate shape there.
            sides[0] = element & 1u;
        }
        else
        {
            unsigned rest = element;
            unsigned parity = 0u;
            for (unsigned axis = 1; axis < ndim; ++axis)
            {
                sides[axis] = rest & 1u;
                rest >>= 1;
                parity ^= sides[axis];
            }
            sides[0] = parity;
        }
        for (unsigned corner = 0; corner < corners; ++corner)
        {
            unsigned id = next;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                coordinates[next][axis] = (int8_t)(sides[axis] + ((corner >> axis) & 1u));
            }
            for (unsigned seen = 0; seen < next; ++seen)
            {
                bool equal = true;
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    equal = equal && coordinates[seen][axis] == coordinates[next][axis];
                }
                if (equal)
                {
                    id = seen;
                    break;
                }
            }
            if (id == next)
            {
                next += 1u;
            }
            out->corners[element * corners + corner] = id;
        }
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            DIRECT_CASE_ORDER(out, element, axis) = 2u;
        }
    }
    out->point_count = next;
}

/**
 * @brief Overwrite one element's basis orders.
 */
static void direct_case_set_element(direct_case_t *const c, const uint64_t element, const unsigned *const axis_orders)
{
    for (unsigned axis = 0; axis < c->ndim; ++axis)
    {
        DIRECT_CASE_ORDER(c, element, axis) = axis_orders[axis];
    }
}

/**
 * @brief Copy a case with its elements renumbered, so a physical element carries a different element ID.
 *
 * The mesh, the orders and the element IDs all move together, which is exactly what relabelling a mesh means.
 */
static void direct_case_relabel(const direct_case_t *const c, const uint64_t *const permutation,
                                direct_case_t *const out)
{
    const unsigned corners = 1u << c->ndim;
    *out = *c;
    for (uint64_t element = 0; element < c->element_count; ++element)
    {
        memcpy(out->corners + element * corners, c->corners + permutation[element] * corners,
               sizeof(uint64_t) * corners);
        for (unsigned axis = 0; axis < c->ndim; ++axis)
        {
            DIRECT_CASE_ORDER(out, element, axis) = DIRECT_CASE_ORDER(c, permutation[element], axis);
        }
    }
}

/**
 * @brief Copy a case with its axes renumbered, both in the mesh and in the per-axis orders.
 *
 * A structured grid of equally many cells per axis maps onto itself under an axis permutation, so the copy is
 * the same mesh with its axis roles exchanged. It is what says the map depends on which axis is rich only
 * through the permutation the caller asked for.
 */
static void direct_case_permute_axes(const direct_case_t *const c, const unsigned *const permutation,
                                     direct_case_t *const out)
{
    const unsigned corners = 1u << c->ndim;
    *out = *c;
    TEST_ASSERTION(c->cells > 0u, "Only a structured grid can have its axes permuted.");
    for (uint64_t element = 0; element < c->element_count; ++element)
    {
        for (unsigned corner = 0; corner < corners; ++corner)
        {
            uint64_t id = 0;
            for (unsigned axis = 0; axis < c->ndim; ++axis)
            {
                // The element index runs with the first axis slowest, so the old coordinate of an axis is the
                // digit that far in, and every lattice extent is the same, which keeps the packing a bijection.
                uint64_t rest = element;
                for (unsigned skip = 0; skip < permutation[axis]; ++skip)
                {
                    rest /= c->cells;
                }
                const uint64_t coordinate = rest % c->cells + ((corner >> axis) & 1u);
                id = id * (c->cells + 1u) + coordinate;
            }
            out->corners[element * corners + corner] = id;
        }
        for (unsigned axis = 0; axis < c->ndim; ++axis)
        {
            DIRECT_CASE_ORDER(out, element, axis) = DIRECT_CASE_ORDER(c, element, permutation[axis]);
        }
    }
}

/**
 * @brief Build the mesh, the specifications and the registries of one case and prepare its map.
 *
 * The result is returned instead of asserted, so a case the map has to reject is prepared exactly like one it
 * has to accept; the rejection test only has to check the value.
 */
static fdg_result_t direct_test_prepare(const direct_case_t *const c, direct_test_t *const out)
{
    const unsigned ndim = c->ndim;
    const uint64_t element_count = c->element_count;
    out->element_count = element_count;
    TEST_ASSERTION(topo_mesh_create_from_corners(ndim, element_count, c->point_count, c->corners, &TEST_ALLOCATOR,
                                                 &out->mesh) == TOPO_SUCCESS,
                   "The mesh of the case could not be created.");
    out->spec_storage = malloc(sizeof(*out->spec_storage) * element_count);
    out->basis_storage = malloc(sizeof(*out->basis_storage) * element_count * ndim);
    out->specs = malloc(sizeof(*out->specs) * element_count);
    for (uint64_t element = 0; element < element_count; ++element)
    {
        for (unsigned axis = 0; axis < ndim; ++axis)
        {
            out->basis_storage[element * ndim + axis] =
                (basis_spec_t){.type = c->family, .order = c->orders[element * DIRECT_TEST_MAX_AXES + axis]};
        }
        out->spec_storage[element] =
            (kform_spec_t){.ndim = ndim, .order = c->kform_order, .basis = out->basis_storage + element * ndim};
        out->specs[element] = &out->spec_storage[element];
    }
    // Elements of differing orders reach the transfer's projection path, which needs real bases.
    TEST_ASSERTION(basis_set_registry_create(&out->basis_registry, 1, &TEST_ALLOCATOR) == FDG_SUCCESS,
                   "The basis registry could not be created.");
    TEST_ASSERTION(integration_rule_registry_create(&out->integration_registry, 1, &TEST_ALLOCATOR) == FDG_SUCCESS,
                   "The integration registry could not be created.");
    out->request = (direct_continuity_request_t){.ndim = ndim,
                                                 .order = c->kform_order,
                                                 .mesh = out->mesh,
                                                 .elements = out->specs,
                                                 .basis_registry = out->basis_registry,
                                                 .integration_registry = out->integration_registry};
    out->plan_memory = malloc(direct_continuity_plan_memory(&out->request));
    out->work_memory = malloc(direct_continuity_work_memory(&out->request));
    direct_continuity_plan_init(&out->plan, &out->request, out->plan_memory);
    direct_continuity_work_init(&out->work, &out->request, out->work_memory);
    return direct_continuity_prepare(&out->request, &out->work, &out->plan);
}

/**
 * @brief Prepare one case and materialise its transfer, requiring the map to be built.
 */
static void direct_test_build(const direct_case_t *const c, direct_test_t *const out)
{
    TEST_FDG_RESULT(direct_test_prepare(c, out));
    direct_continuity_layout(&out->request, &out->plan, &out->work, &out->layout);
    out->entry_offsets = malloc(sizeof(size_t) * (out->layout.element_dof_count + 1u));
    out->entry_index = malloc(sizeof(size_t) * out->layout.entry_count);
    out->entry_value = malloc(sizeof(double) * out->layout.entry_count);
    direct_continuity_build(&out->request, &out->plan, &out->work, out->entry_offsets, out->entry_index,
                            out->entry_value);
}

/**
 * @brief Release a prepared map and its mesh.
 */
static void direct_test_teardown(direct_test_t *const test)
{
    free(test->entry_offsets);
    free(test->entry_index);
    free(test->entry_value);
    free(test->spec_storage);
    free(test->basis_storage);
    free(test->specs);
    integration_rule_registry_destroy(test->integration_registry);
    basis_set_registry_destroy(test->basis_registry);
    direct_continuity_plan_release(&test->plan);
    free(test->plan_memory);
    free(test->work_memory);
    topo_mesh_free(test->mesh, &TEST_ALLOCATOR);
}

static void direct_test_check_transfer(const direct_test_t *const test);
static void direct_test_check_continuity(const direct_test_t *const test);

/**
 * @brief Assert the layout invariants every conforming transfer must satisfy.
 *
 * Every case runs this beside #direct_test_check_transfer and #direct_test_check_continuity, so a case that
 * breaks any of them names the invariant rather than whichever check happened to run first.
 */
static void direct_test_check_conforming(const direct_test_t *const test)
{
    const direct_continuity_layout_t *const layout = &test->layout;
    TEST_ASSERTION(layout->global_dof_count <= layout->element_dof_count,
                   "The map numbers %zu unknowns out of only %zu element degrees of freedom.", layout->global_dof_count,
                   layout->element_dof_count);
    TEST_ASSERTION(test->entry_offsets[0] == 0u, "The row offsets do not start at zero.");
    TEST_ASSERTION(test->entry_offsets[layout->element_dof_count] == layout->entry_count,
                   "The row offsets end at %zu, but the map counts %zu entries.",
                   test->entry_offsets[layout->element_dof_count], layout->entry_count);
    TEST_ASSERTION(test->plan.element_dof_offsets[test->element_count] == layout->element_dof_count,
                   "The element offsets do not end at the element degree-of-freedom count.");
    TEST_ASSERTION(test->plan.element_interior_offsets[test->element_count] == layout->global_dof_count,
                   "The interior offsets do not end at the global degree-of-freedom count.");
    for (uint64_t element = 0; element < test->element_count; ++element)
    {
        TEST_ASSERTION(test->plan.element_dof_offsets[element + 1u] > test->plan.element_dof_offsets[element],
                       "Element %llu carries no degree of freedom at all.", (unsigned long long)element);
    }
    // Every global unknown has to be reached, or the assembled system has an empty row or column.
    char *const unreached = malloc(layout->global_dof_count);
    memset(unreached, 1, layout->global_dof_count);
    for (size_t entry = 0; entry < layout->entry_count; ++entry)
    {
        unreached[test->entry_index[entry]] = 0;
    }
    for (size_t dof = 0; dof < layout->global_dof_count; ++dof)
    {
        TEST_ASSERTION(!unreached[dof], "Global degree of freedom %zu is reached by no entry.", dof);
    }
    free(unreached);
    direct_test_check_transfer(test);
    direct_test_check_continuity(test);
}

static void direct_test_setup_mixed(const unsigned ndim, const unsigned cells, const unsigned order,
                                    const unsigned even_order, const unsigned odd_order, direct_test_t *const out);
static void direct_test_setup_pattern(const unsigned ndim, const unsigned cells, const unsigned order,
                                      const unsigned even_order, const unsigned odd_order, bool alternating,
                                      basis_set_type_t family, direct_test_t *const out);

/**
 * @brief Build the map with every axis on one chosen basis family.
 */
static void direct_test_setup_family(const unsigned ndim, const unsigned cells, const unsigned order,
                                     const unsigned basis_order, const basis_set_type_t family,
                                     direct_test_t *const out)
{
    direct_test_setup_pattern(ndim, cells, order, basis_order, basis_order, false, family, out);
}

static void direct_test_setup(const unsigned ndim, const unsigned cells, const unsigned order,
                              const unsigned basis_order, direct_test_t *const out)
{
    direct_test_setup_mixed(ndim, cells, order, basis_order, basis_order, out);
}

/**
 * @brief Build a mesh whose even elements use the first basis order and its odd elements the second.
 *
 * @param ndim Dimension.
 * @param cells Per-axis cell count.
 * @param order k-form order.
 * @param even_order Basis order of the even elements.
 * @param odd_order Basis order of the odd elements.
 * @param out Receives the prepared map.
 */
static void direct_test_setup_mixed(const unsigned ndim, const unsigned cells, const unsigned order,
                                    const unsigned even_order, const unsigned odd_order, direct_test_t *const out)
{
    direct_test_setup_pattern(ndim, cells, order, even_order, odd_order, true, BASIS_LAGRANGE_GAUSS_LOBATTO, out);
}

/**
 * @brief Build a mesh whose elements differ on the basis order in a chosen pattern.
 *
 * @param alternating Non-zero: alternate the two orders, so both of them appear on every interior object and
 *                    every shared object has to take a minimum. Zero: put the second order on every element,
 *                    which leaves the first one to say nothing more than what a uniform mesh of that order is.
 */
static void direct_test_setup_pattern(const unsigned ndim, const unsigned cells, const unsigned order,
                                      const unsigned even_order, const unsigned odd_order, const bool alternating,
                                      const basis_set_type_t family, direct_test_t *const out)
{
    direct_case_t c;
    direct_case_uniform(&c, ndim, cells, order, odd_order);
    if (alternating && even_order != odd_order)
    {
        for (uint64_t element = 0; element < c.element_count; element += 2u)
        {
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                DIRECT_CASE_ORDER(&c, element, axis) = even_order;
            }
        }
    }
    c.family = family;
    direct_test_build(&c, out);
}

/**
 * @brief Global DoFs one element's transfer reaches inside a given object's block.
 */
static size_t direct_test_touching_dofs(const direct_test_t *const test, const uint64_t element, const unsigned dim,
                                        const uint64_t object, size_t *const out)
{
    const size_t index = direct_entity_index(&test->plan, dim, object);
    const size_t block = test->plan.entity_block_offsets[index];
    const size_t block_size = test->plan.entity_block_offsets[index + 1u] - block;
    const size_t base = test->plan.element_dof_offsets[element];
    const size_t count = test->plan.element_dof_offsets[element + 1u] - base;
    size_t found = 0;
    for (size_t local = 0; local < count; ++local)
    {
        bool touches = false;
        for (size_t entry = test->entry_offsets[base + local]; entry < test->entry_offsets[base + local + 1u]; ++entry)
        {
            const size_t global = test->entry_index[entry];
            touches |= global >= block && global < block + block_size;
        }
        if (touches)
        {
            TEST_ASSERTION(found < DIRECT_TEST_MAX_ENTRIES, "An object reached more DoFs than expected.");
            for (size_t entry = test->entry_offsets[base + local]; entry < test->entry_offsets[base + local + 1u];
                 ++entry)
            {
                const size_t global = test->entry_index[entry];
                if (global >= block && global < block + block_size)
                {
                    out[found++] = global;
                }
            }
        }
    }
    return found;
}

/**
 * @brief Assert the invariants every transfer must satisfy.
 */
static void direct_test_check_transfer(const direct_test_t *const test)
{
    for (size_t local = 0; local < test->layout.element_dof_count; ++local)
    {
        const size_t from = test->entry_offsets[local];
        const size_t to = test->entry_offsets[local + 1u];
        // An element of a higher order than its neighbours' common axis can project onto an empty common
        // space, so an empty range is legal; what is not legal is an inconsistent one.
        TEST_ASSERTION(from <= to && to <= test->layout.entry_count, "Element DoF %zu has an invalid entry range.",
                       local);
        for (size_t entry = from; entry < to; ++entry)
        {
            TEST_ASSERTION(test->entry_index[entry] < test->layout.global_dof_count,
                           "Entry %zu points outside the global space.", entry);
            for (size_t other = from; other < entry; ++other)
            {
                TEST_ASSERTION(test->entry_index[other] != test->entry_index[entry],
                               "Element DoF %zu reaches global DoF %zu twice.", local, test->entry_index[entry]);
            }
        }
    }
}

/**
 * @brief Assert that every element of a shared object reaches the same set of local DoFs.
 *
 * That is continuity stated purely about the numbering, and it is the property that distinguishes the direct
 * formulation from the hybridized one.
 */
static void direct_test_check_continuity(const direct_test_t *const test)
{
    const topo_mesh_t *const mesh = test->plan.mesh;
    for (unsigned dim = 0; dim < mesh->ndim; ++dim)
    {
        const topo_obj_immersion_t *const immersion = mesh->immersions + dim;
        for (unsigned object = 0; object < immersion->object_count; ++object)
        {
            uint64_t incident;
            const uint64_t *ids;
            const int8_t *orientations;
            topo_obj_immersion_of_object(immersion, object, &incident, &ids, &orientations);
            if (incident < 2)
            {
                continue;
            }
            const size_t block_index = direct_entity_index(&test->plan, dim, object);
            const size_t block = test->plan.entity_block_offsets[block_index];
            const size_t block_size = test->plan.entity_block_offsets[block_index + 1u] - block;
            // A shared object's DoFs must appear once from every incident element, which is continuity stated
            // purely about the numbering.
            for (size_t dof = 0; dof < block_size; ++dof)
            {
                unsigned sides = 0;
                for (uint64_t side = 0; side < incident; ++side)
                {
                    size_t reached[DIRECT_TEST_MAX_ENTRIES];
                    const size_t count = direct_test_touching_dofs(test, ids[side], dim, object, reached);
                    for (size_t i = 0; i < count; ++i)
                    {
                        if (reached[i] == block + dof)
                        {
                            sides += 1;
                        }
                    }
                }
                // Every incident element must reach every DoF of a shared object; how many of its own DoFs
                // contribute is one under a space-matching object and more under a projection, so only the
                // lower bound is an invariant.
                TEST_ASSERTION(sides >= incident,
                               "Object %u of dimension %u has a DoF %zu that only %u of its %llu elements reach.",
                               object, dim, dof, sides, (unsigned long long)incident);
            }
        }
    }
}

/**
 * @brief A conforming nodal scalar space on an axis-aligned mesh has one DoF per lattice point.
 */
static void test_scalar_counts_match_the_lattice(void)
{
    static const unsigned basis_orders[] = {1u, 2u, 3u};
    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        for (unsigned i = 0; i < sizeof(basis_orders) / sizeof(basis_orders[0]); ++i)
        {
            direct_test_t test;
            direct_test_setup(ndim, 2u, 0u, basis_orders[i], &test);
            uint64_t expected = 1;
            for (unsigned axis = 0; axis < ndim; ++axis)
            {
                expected *= 2u * basis_orders[i] + 1u;
            }
            TEST_ASSERTION(test.layout.global_dof_count == (size_t)expected,
                           "A %uD scalar mesh of two cells at basis order %u has %zu global DoFs, expected %llu.", ndim,
                           basis_orders[i], test.layout.global_dof_count, (unsigned long long)expected);
            direct_test_check_transfer(&test);
            direct_test_check_continuity(&test);
            direct_test_teardown(&test);
        }
    }
}

/**
 * @brief The direct formulation never uses more unknowns than the hybridized element-local ones.
 */
static void test_direct_is_never_larger(void)
{
    static const unsigned kform_orders[] = {0u, 1u, 2u};
    static const unsigned basis_orders[] = {1u, 2u, 3u};
    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        for (unsigned k = 0; k < sizeof(kform_orders) / sizeof(kform_orders[0]); ++k)
        {
            if (kform_orders[k] > ndim)
            {
                continue;
            }
            for (unsigned i = 0; i < sizeof(basis_orders) / sizeof(basis_orders[0]); ++i)
            {
                direct_test_t test;
                direct_test_setup(ndim, 2u, kform_orders[k], basis_orders[i], &test);
                TEST_ASSERTION(test.layout.global_dof_count <= test.layout.element_dof_count,
                               "A %uD order-%u map of basis order %u uses %zu of %zu element DoFs.", ndim,
                               kform_orders[k], basis_orders[i], test.layout.global_dof_count,
                               test.layout.element_dof_count);
                direct_test_check_transfer(&test);
                direct_test_check_continuity(&test);
                direct_test_teardown(&test);
            }
        }
    }
}

/**
 * @brief A shared DoF may only be reached by one coefficient sign.
 *
 * A sign error on one of the two sides would double the coupling instead of matching the fields, so every side
 * must transfer a shared DoF with the same `+1` or `-1`.
 */
static void test_shared_coefficients_agree(void)
{
    const unsigned ndim = 3;
    direct_test_t test;
    direct_test_setup(ndim, 2u, 1u, 2u, &test);
    const topo_mesh_t *const mesh = test.plan.mesh;
    for (unsigned dim = 0; dim < ndim; ++dim)
    {
        const topo_obj_immersion_t *const immersion = mesh->immersions + dim;
        for (unsigned object = 0; object < immersion->object_count; ++object)
        {
            uint64_t incident;
            const uint64_t *ids;
            const int8_t *orientations;
            topo_obj_immersion_of_object(immersion, object, &incident, &ids, &orientations);
            if (incident < 2)
            {
                continue;
            }
            const size_t index = direct_entity_index(&test.plan, dim, object);
            const size_t block = test.plan.entity_block_offsets[index];
            const size_t block_size = test.plan.entity_block_offsets[index + 1u] - block;
            for (size_t global = block; global < block + block_size; ++global)
            {
                int8_t seen = 0;
                for (uint64_t side = 0; side < incident; ++side)
                {
                    const size_t base = test.plan.element_dof_offsets[ids[side]];
                    const size_t count = test.plan.element_dof_offsets[ids[side] + 1u] - base;
                    for (size_t local = 0; local < count; ++local)
                    {
                        for (size_t entry = test.entry_offsets[base + local];
                             entry < test.entry_offsets[base + local + 1u]; ++entry)
                        {
                            if (test.entry_index[entry] != global)
                            {
                                continue;
                            }
                            const double value = test.entry_value[entry];
                            TEST_ASSERTION(value == 1.0 || value == -1.0,
                                           "Object %u of dimension %u transfers the coefficient %g.", object, dim,
                                           value);
                            const int8_t sign = value < 0.0 ? -1 : 1;
                            seen = seen == 0 ? sign : seen;
                            TEST_ASSERTION(seen == sign,
                                           "Element %llu reaches object %u's DoF %zu with a different sign.",
                                           (unsigned long long)ids[side], object, global - block);
                        }
                    }
                }
            }
        }
    }
    direct_test_teardown(&test);
}

/**
 * @brief Scattering a symmetric element matrix keeps the global matrix symmetric.
 *
 * The orientation sign reaches the system only through the scatter, so this is where a sign error on one side
 * would show up.
 */
static void test_scatter_preserves_symmetry(void)
{
    const unsigned ndim = 3;
    direct_test_t test;
    direct_test_setup(ndim, 2u, 1u, 2u, &test);
    const size_t dofs = test.layout.element_dof_count / test.layout.element_count;
    const size_t global = test.layout.global_dof_count;
    double *const matrix = malloc(sizeof(double) * dofs * dofs);
    double *const assembled = calloc(global * global, sizeof(double));
    for (size_t i = 0; i < dofs; ++i)
    {
        for (size_t j = 0; j < dofs; ++j)
        {
            // Symmetric in its indices, so any asymmetry after the scatter comes from the transfer.
            matrix[i * dofs + j] = (double)((i * i + j * j + 3u * (i * j) % 5u) % 7u) / 8.0 - 0.25;
        }
        matrix[i * dofs + i] += (double)dofs;
    }
    for (size_t element = 0; element < test.layout.element_count; ++element)
    {
        direct_continuity_scatter(&test.plan, test.entry_offsets, test.entry_index, test.entry_value, element, matrix,
                                  dofs, assembled, global, 1.0);
    }
    for (size_t i = 0; i < global; ++i)
    {
        TEST_ASSERTION(assembled[i * global + i] > 0.0, "Global DoF %zu carries no diagonal.", i);
        for (size_t j = 0; j < global; ++j)
        {
            TEST_ASSERTION(assembled[i * global + j] == assembled[j * global + i],
                           "The assembled matrix is not symmetric at (%zu, %zu): %g against %g.", i, j,
                           assembled[i * global + j], assembled[j * global + i]);
        }
    }
    free(matrix);
    free(assembled);
    direct_test_teardown(&test);
}

/**
 * @brief Every global DoF must be reachable, otherwise the assembled system has a zero row or column.
 */
static void test_every_global_dof_is_reachable(void)
{
    static const unsigned kform_orders[] = {0u, 1u, 2u};
    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        for (unsigned k = 0; k < sizeof(kform_orders) / sizeof(kform_orders[0]); ++k)
        {
            if (kform_orders[k] > ndim)
            {
                continue;
            }
            direct_test_t test;
            direct_test_setup(ndim, 2u, kform_orders[k], 2u, &test);
            char *const seen = calloc(test.layout.global_dof_count, sizeof(char));
            for (size_t entry = 0; entry < test.layout.entry_count; ++entry)
            {
                seen[test.entry_index[entry]] = 1;
            }
            for (size_t dof = 0; dof < test.layout.global_dof_count; ++dof)
            {
                TEST_ASSERTION(seen[dof], "Global DoF %zu of the %uD order-%u map is unreachable.", dof, ndim,
                               kform_orders[k]);
            }
            free(seen);
            direct_test_teardown(&test);
        }
    }
}

/**
 * @brief A mesh whose elements disagree on an axis order still maps, dropping what cannot be represented.
 *
 * The common space of a shared object is the per-axis minimum over its incident elements, so a higher-order
 * element's extra trace freedom is projected away. Where the common space is empty on an axis the element's DoF
 * simply has no global counterpart, and the walk must still produce a valid row compression.
 */
static void test_mixed_element_orders(void)
{
    static const unsigned orders[][2] = {{2u, 1u}, {1u, 2u}, {3u, 2u}, {2u, 3u}};
    for (unsigned ndim = 2; ndim <= 3; ++ndim)
    {
        for (unsigned variant = 0; variant < sizeof(orders) / sizeof(orders[0]); ++variant)
        {
            direct_test_t test;
            direct_test_setup_mixed(ndim, 2u, 0u, orders[variant][0], orders[variant][1], &test);
            direct_test_check_transfer(&test);
            direct_test_check_continuity(&test);
            direct_test_teardown(&test);
            direct_test_setup_pattern(ndim, 2u, 0u, orders[variant][1], orders[variant][0], false,
                                      BASIS_LAGRANGE_GAUSS_LOBATTO, &test);
            direct_test_check_transfer(&test);
            direct_test_check_continuity(&test);
            TEST_ASSERTION(test.layout.global_dof_count <= test.layout.element_dof_count,
                           "The %uD mixed-order map uses %zu of %zu element DoFs.", ndim, test.layout.global_dof_count,
                           test.layout.element_dof_count);
            direct_test_teardown(&test);
        }
    }
}

/**
 * @brief Every k-form order maps for the uniform Lagrange family as well as for Gauss-Lobatto.
 */
static void test_uniform_family(void)
{
    static const basis_set_type_t families[] = {BASIS_LAGRANGE_UNIFORM, BASIS_LAGRANGE_GAUSS,
                                                BASIS_LAGRANGE_GAUSS_LOBATTO, BASIS_LAGRANGE_CHEBYSHEV_GAUSS};
    for (unsigned family = 0; family < sizeof(families) / sizeof(families[0]); ++family)
    {
        for (unsigned k = 0; k <= 2u; ++k)
        {
            direct_test_t test;
            direct_test_setup_family(2u, 2u, k, 2u, families[family], &test);
            direct_test_check_transfer(&test);
            direct_test_check_continuity(&test);
            direct_test_teardown(&test);
        }
    }
}

/**
 * @brief Every Lagrange family crossed with every k-form order, in one, two and three dimensions.
 *
 * A family or a k-form order the scratch is sized for wrongly shows up here and nowhere else: the walk's operator
 * block is as wide as the basis order, so a family that reports a different function count overruns it.
 */
static void test_adversarial_every_family_and_kform_order(void)
{
    static const basis_set_type_t families[] = {BASIS_LAGRANGE_UNIFORM, BASIS_LAGRANGE_GAUSS,
                                                BASIS_LAGRANGE_GAUSS_LOBATTO, BASIS_LAGRANGE_CHEBYSHEV_GAUSS};
    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        // One cell per axis is the degenerate grid; two is the smallest mesh with an interior object.
        for (unsigned cells = 1; cells <= 2; ++cells)
        {
            for (unsigned family = 0; family < sizeof(families) / sizeof(families[0]); ++family)
            {
                for (unsigned k = 0; k <= ndim; ++k)
                {
                    direct_case_t c;
                    direct_case_uniform(&c, ndim, cells, k, 2u);
                    c.family = families[family];
                    direct_test_t test;
                    direct_test_build(&c, &test);
                    direct_test_check_conforming(&test);
                    direct_test_teardown(&test);
                }
            }
        }
    }
}

/**
 * @brief One element whose axes disagree on the order, among neighbours that all agree.
 *
 * The common basis of a shared object is read from the axes its record leaves free, so an element that is rich
 * on one axis and poor on another is what separates reading the right axis from reading the fixed ones. The rich
 * element is put at a corner, at a corner of the opposite parity and in the middle of the mesh in turn, because
 * the objects it meets on the way there are not the same ones.
 */
static void test_adversarial_anisotropy_inside_one_element(void)
{
    static const unsigned orders_2d[][2] = {{4u, 2u}, {2u, 4u}, {5u, 2u}, {2u, 5u}, {6u, 3u}, {3u, 6u}};
    static const unsigned orders_3d[][3] = {{4u, 2u, 2u}, {2u, 4u, 2u}, {2u, 2u, 4u}, {5u, 2u, 3u}, {3u, 5u, 2u}};
    // Three cells per axis in two dimensions put an element in the middle of the mesh; in three dimensions the
    // same role falls on element 13 of a 3x3x3 grid.
    for (unsigned variant = 0; variant < sizeof(orders_2d) / sizeof(orders_2d[0]); ++variant)
    {
        static const uint64_t positions[] = {0u, 4u, 8u};
        for (unsigned position = 0; position < sizeof(positions) / sizeof(positions[0]); ++position)
        {
            direct_case_t c;
            direct_case_uniform(&c, 2u, 3u, 0u, 2u);
            direct_case_set_element(&c, positions[position], orders_2d[variant]);
            direct_test_t test;
            direct_test_build(&c, &test);
            direct_test_check_conforming(&test);
            direct_test_teardown(&test);
        }
    }
    for (unsigned variant = 0; variant < sizeof(orders_3d) / sizeof(orders_3d[0]); ++variant)
    {
        for (unsigned position = 0; position < 8u; ++position)
        {
            for (unsigned k = 0; k <= 3u; ++k)
            {
                direct_case_t c;
                direct_case_uniform(&c, 3u, 2u, k, 2u);
                direct_case_set_element(&c, position, orders_3d[variant]);
                direct_test_t test;
                direct_test_build(&c, &test);
                direct_test_check_conforming(&test);
                direct_test_teardown(&test);
            }
        }
    }
    for (unsigned variant = 0; variant < sizeof(orders_3d) / sizeof(orders_3d[0]); ++variant)
    {
        direct_case_t c;
        direct_case_uniform(&c, 3u, 3u, 0u, 2u);
        direct_case_set_element(&c, 13u, orders_3d[variant]);
        direct_test_t test;
        direct_test_build(&c, &test);
        direct_test_check_conforming(&test);
        direct_test_teardown(&test);
    }
}

/**
 * @brief Every element on a different order, so no two incident elements agree on any axis.
 */
static void test_adversarial_distinct_order_per_element(void)
{
    for (unsigned k = 0; k <= 2u; ++k)
    {
        direct_case_t c;
        direct_case_uniform(&c, 2u, 2u, k, 2u);
        for (uint64_t element = 0; element < c.element_count; ++element)
        {
            const unsigned orders[2] = {2u + (unsigned)element, 5u - (unsigned)element};
            direct_case_set_element(&c, element, orders);
        }
        direct_test_t test;
        direct_test_build(&c, &test);
        direct_test_check_conforming(&test);
        direct_test_teardown(&test);
    }
    for (unsigned k = 0; k <= 3u; ++k)
    {
        direct_case_t c;
        direct_case_uniform(&c, 3u, 2u, k, 1u);
        // The first element stays at order one, which has no face-interior function at all, so its trace
        // projects onto an empty common space and its face degrees of freedom leave the system.
        for (uint64_t element = 0; element < c.element_count; ++element)
        {
            const unsigned orders[3] = {2u + (unsigned)element, 2u, 1u + (unsigned)(c.element_count - element)};
            direct_case_set_element(&c, element, orders);
        }
        direct_test_t test;
        direct_test_build(&c, &test);
        direct_test_check_conforming(&test);
        direct_test_teardown(&test);
    }
}

/**
 * @brief A checkerboard of two orders, which puts both of them on every interior object.
 */
static void test_adversarial_checkerboard_orders(void)
{
    static const unsigned pair[][2] = {{2u, 5u}, {1u, 4u}, {3u, 6u}};
    for (unsigned variant = 0; variant < sizeof(pair) / sizeof(pair[0]); ++variant)
    {
        for (unsigned k = 0; k <= 2u; ++k)
        {
            direct_case_t c;
            direct_case_uniform(&c, 2u, 3u, k, pair[variant][0]);
            for (uint64_t element = 0; element < c.element_count; ++element)
            {
                uint64_t rest = element;
                const unsigned ix = (unsigned)(rest % 3u);
                const unsigned iy = (unsigned)(rest / 3u);
                const unsigned orders[2] = {(ix + iy) % 2u == 0u ? pair[variant][0] : pair[variant][1],
                                            (ix + iy) % 2u == 0u ? pair[variant][0] : pair[variant][1]};
                direct_case_set_element(&c, element, orders);
            }
            direct_test_t test;
            direct_test_build(&c, &test);
            direct_test_check_conforming(&test);
            direct_test_teardown(&test);
        }
        for (unsigned k = 0; k <= 3u; ++k)
        {
            direct_case_t c;
            direct_case_uniform(&c, 3u, 2u, k, pair[variant][0]);
            for (uint64_t element = 0; element < c.element_count; ++element)
            {
                uint64_t rest = element;
                const unsigned parity = (unsigned)((rest % 2u) + (rest / 2u) % 2u + rest / 4u % 2u);
                const unsigned order = parity % 2u == 0u ? pair[variant][0] : pair[variant][1];
                const unsigned orders[3] = {order, order, order};
                direct_case_set_element(&c, element, orders);
            }
            direct_test_t test;
            direct_test_build(&c, &test);
            direct_test_check_conforming(&test);
            direct_test_teardown(&test);
        }
    }
}

/**
 * @brief Two adjacent rich elements against poor ones, so one poor element sees a rich one on both of its sides.
 */
static void test_adversarial_rich_pair_next_to_poor(void)
{
    for (unsigned k = 0; k <= 2u; ++k)
    {
        // Elements 0 and 1 share an edge in a two-by-two grid; element 2 sits between the two of them.
        static const uint64_t rich[][2] = {{0u, 1u}, {1u, 3u}, {0u, 3u}};
        for (unsigned variant = 0; variant < sizeof(rich) / sizeof(rich[0]); ++variant)
        {
            direct_case_t c;
            direct_case_uniform(&c, 2u, 2u, k, 2u);
            const unsigned high[2] = {5u, 4u};
            direct_case_set_element(&c, rich[variant][0], high);
            direct_case_set_element(&c, rich[variant][1], high);
            direct_test_t test;
            direct_test_build(&c, &test);
            direct_test_check_conforming(&test);
            direct_test_teardown(&test);
        }
    }
    for (unsigned k = 0; k <= 3u; ++k)
    {
        direct_case_t c;
        direct_case_uniform(&c, 3u, 2u, k, 2u);
        // Elements 0 and 1 share a face, and element 6 touches both of them through the mesh centre.
        const unsigned high[3] = {4u, 3u, 2u};
        direct_case_set_element(&c, 0u, high);
        direct_case_set_element(&c, 1u, high);
        direct_test_t test;
        direct_test_build(&c, &test);
        direct_test_check_conforming(&test);
        direct_test_teardown(&test);
    }
}

/**
 * @brief The two extreme order ratios, which drive the projection path harder than anything else.
 *
 * A poor element among rich ones gives the object a common axis of order one, whose window is empty, so the
 * rich elements' face degrees of freedom drop out of the system. A rich element among poor ones gives the
 * object a tiny common space the rich element has to project onto. Both have to leave a valid row compression.
 */
static void test_adversarial_extreme_order_ratios(void)
{
    static const basis_set_type_t families[] = {BASIS_LAGRANGE_UNIFORM, BASIS_LAGRANGE_GAUSS,
                                                BASIS_LAGRANGE_GAUSS_LOBATTO, BASIS_LAGRANGE_CHEBYSHEV_GAUSS};
    for (unsigned family = 0; family < sizeof(families) / sizeof(families[0]); ++family)
    {
        for (unsigned ndim = 1; ndim <= 3; ++ndim)
        {
            for (unsigned k = 0; k <= ndim; ++k)
            {
                for (unsigned rich = 0; rich < 2u; ++rich)
                {
                    direct_case_t c;
                    direct_case_uniform(&c, ndim, 2u, k, rich == 0u ? 2u : 6u);
                    const unsigned high = rich == 0u ? 6u : 2u;
                    const unsigned orders[3] = {high, high, high};
                    direct_case_set_element(&c, 0u, orders);
                    c.family = families[family];
                    direct_test_t test;
                    direct_test_build(&c, &test);
                    direct_test_check_conforming(&test);
                    direct_test_teardown(&test);
                }
            }
        }
    }
}

/**
 * @brief The degenerate meshes: a single element, one cell per axis and elements meeting in one point.
 */
static void test_adversarial_degenerate_shapes(void)
{
    static const basis_set_type_t families[] = {BASIS_LAGRANGE_UNIFORM, BASIS_LAGRANGE_GAUSS,
                                                BASIS_LAGRANGE_GAUSS_LOBATTO, BASIS_LAGRANGE_CHEBYSHEV_GAUSS};
    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        // Two intervals in one dimension, and every code word of the even-parity orthants above it: all the
        // elements carry the origin and no pair of them shares anything else.
        const unsigned fan = ndim == 1u ? 2u : 1u << (ndim - 1u);
        for (unsigned k = 0; k <= ndim; ++k)
        {
            // A single element shares every one of its objects with nothing, so every block is its own trace and
            // the global numbering has to reproduce the element's degrees of freedom one for one.
            direct_case_t single;
            direct_case_uniform(&single, ndim, 1u, k, 3u);
            direct_test_t test;
            direct_test_build(&single, &test);
            direct_test_check_conforming(&test);
            TEST_ASSERTION(test.layout.global_dof_count == test.layout.element_dof_count,
                           "A single %uD element of order %u numbers %zu of its %zu degrees of freedom.", ndim, k,
                           test.layout.global_dof_count, test.layout.element_dof_count);
            direct_test_teardown(&test);

            for (unsigned family = 0; family < sizeof(families) / sizeof(families[0]); ++family)
            {
                direct_case_t corners;
                direct_case_corner_fan(&corners, ndim, k, fan);
                corners.family = families[family];
                direct_test_build(&corners, &test);
                direct_test_check_conforming(&test);
                direct_test_teardown(&test);
            }
        }
    }
}

/**
 * @brief Renumbering the elements, and moving the rich one, must not change the size of the map.
 *
 * The object numbering a mesh carries is not the element numbering, so a map that counted elements instead of
 * objects would pass every check above and still come out a different size here.
 */
static void test_adversarial_element_relabelling_is_invariant(void)
{
    direct_case_t base;
    direct_case_uniform(&base, 3u, 2u, 1u, 2u);
    const unsigned anisotropic[3] = {4u, 2u, 3u};
    direct_case_set_element(&base, 0u, anisotropic);

    // Every rotation of the element order of a cube, plus the reversal and the two transpositions of it.
    static const uint64_t permutations[][8] = {{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}, {7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u},
                                               {1u, 0u, 3u, 2u, 5u, 4u, 7u, 6u}, {2u, 3u, 0u, 1u, 6u, 7u, 4u, 5u},
                                               {4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u}, {7u, 4u, 1u, 6u, 3u, 0u, 5u, 2u},
                                               {3u, 2u, 1u, 0u, 7u, 6u, 5u, 4u}, {5u, 7u, 4u, 6u, 1u, 3u, 0u, 2u}};
    for (unsigned variant = 0; variant < sizeof(permutations) / sizeof(permutations[0]); ++variant)
    {
        direct_case_t relabelled;
        direct_case_relabel(&base, permutations[variant], &relabelled);
        direct_test_t plain;
        direct_test_t moved;
        direct_test_build(&base, &plain);
        direct_test_build(&relabelled, &moved);
        TEST_ASSERTION(moved.layout.global_dof_count == plain.layout.global_dof_count,
                       "Renumbering the elements changes the global count from %zu to %zu.",
                       plain.layout.global_dof_count, moved.layout.global_dof_count);
        TEST_ASSERTION(moved.layout.entry_count == plain.layout.entry_count,
                       "Renumbering the elements changes the entry count from %zu to %zu.", plain.layout.entry_count,
                       moved.layout.entry_count);
        TEST_ASSERTION(moved.layout.element_dof_count == plain.layout.element_dof_count,
                       "Renumbering the elements changes the element degree-of-freedom count from %zu to %zu.",
                       plain.layout.element_dof_count, moved.layout.element_dof_count);
        direct_test_check_conforming(&plain);
        direct_test_check_conforming(&moved);
        direct_test_teardown(&plain);
        direct_test_teardown(&moved);
    }

    // Which element carries the rich orders must not matter either; only which objects they meet does.
    for (uint64_t rich = 0; rich < base.element_count; ++rich)
    {
        direct_case_t moved = base;
        for (uint64_t element = 0; element < base.element_count; ++element)
        {
            const unsigned orders[3] = {2u, 2u, 2u};
            direct_case_set_element(&moved, element, orders);
        }
        direct_case_set_element(&moved, rich, anisotropic);
        direct_test_t test;
        direct_test_t reference;
        direct_test_build(&base, &reference);
        direct_test_build(&moved, &test);
        TEST_ASSERTION(test.layout.global_dof_count == reference.layout.global_dof_count,
                       "Moving the rich element to %llu changes the global count from %zu to %zu.",
                       (unsigned long long)rich, reference.layout.global_dof_count, test.layout.global_dof_count);
        TEST_ASSERTION(test.layout.entry_count == reference.layout.entry_count,
                       "Moving the rich element to %llu changes the entry count from %zu to %zu.",
                       (unsigned long long)rich, reference.layout.entry_count, test.layout.entry_count);
        direct_test_check_conforming(&test);
        direct_test_teardown(&test);
        direct_test_teardown(&reference);
    }
}

/**
 * @brief Exchanging the axes of the mesh exchanges the roles of the orders and changes nothing else.
 */
static void test_adversarial_axis_permutation_is_invariant(void)
{
    direct_case_t base;
    direct_case_uniform(&base, 2u, 3u, 0u, 2u);
    const unsigned anisotropic[2] = {5u, 2u};
    direct_case_set_element(&base, 0u, anisotropic);
    const unsigned swapped[2] = {1u, 0u};
    direct_case_t transposed;
    direct_case_permute_axes(&base, swapped, &transposed);

    direct_test_t reference;
    direct_test_t test;
    direct_test_build(&base, &reference);
    direct_test_build(&transposed, &test);
    TEST_ASSERTION(test.layout.global_dof_count == reference.layout.global_dof_count,
                   "Exchanging the mesh axes changes the global count from %zu to %zu.",
                   reference.layout.global_dof_count, test.layout.global_dof_count);
    TEST_ASSERTION(test.layout.entry_count == reference.layout.entry_count,
                   "Exchanging the mesh axes changes the entry count from %zu to %zu.", reference.layout.entry_count,
                   test.layout.entry_count);
    direct_test_check_conforming(&reference);
    direct_test_check_conforming(&test);
    direct_test_teardown(&reference);
    direct_test_teardown(&test);

    // The same exchange in three dimensions, with a rich axis that is neither first nor last.
    direct_case_uniform(&base, 3u, 2u, 2u, 2u);
    const unsigned triad[3] = {3u, 5u, 2u};
    direct_case_set_element(&base, 0u, triad);
    const unsigned rotation[3] = {1u, 2u, 0u};
    direct_case_permute_axes(&base, rotation, &transposed);
    direct_test_build(&base, &reference);
    direct_test_build(&transposed, &test);
    TEST_ASSERTION(test.layout.global_dof_count == reference.layout.global_dof_count,
                   "Rotating the mesh axes changes the global count from %zu to %zu.",
                   reference.layout.global_dof_count, test.layout.global_dof_count);
    TEST_ASSERTION(test.layout.entry_count == reference.layout.entry_count,
                   "Rotating the mesh axes changes the entry count from %zu to %zu.", reference.layout.entry_count,
                   test.layout.entry_count);
    direct_test_check_conforming(&reference);
    direct_test_check_conforming(&test);
    direct_test_teardown(&reference);
    direct_test_teardown(&test);
}

/**
 * @brief A degree of freedom above the object's common order owns a weighted combination, not one signed entry.
 */
static void test_adversarial_projected_dof_owns_several_entries(void)
{
    direct_case_t c;
    // The neighbours hold the common axis at order three, whose window has two functions, so an element at
    // order five on that axis has to reach both of them from each of its own four.
    direct_case_uniform(&c, 2u, 2u, 0u, 3u);
    const unsigned rich[2] = {3u, 5u};
    direct_case_set_element(&c, 0u, rich);
    direct_test_t test;
    direct_test_build(&c, &test);
    size_t projected = 0;
    for (size_t local = 0; local < test.layout.element_dof_count; ++local)
    {
        const size_t from = test.entry_offsets[local];
        const size_t to = test.entry_offsets[local + 1u];
        if (to - from < 2u)
        {
            continue;
        }
        projected += 1u;
        for (size_t entry = from; entry < to; ++entry)
        {
            TEST_ASSERTION(isfinite(test.entry_value[entry]), "Element DoF %zu carries the coefficient %g.", local,
                           test.entry_value[entry]);
            // A stored zero would make the row a different map with the same shape.
            TEST_ASSERTION(test.entry_value[entry] != 0.0, "Element DoF %zu stores a zero coefficient.", local);
        }
    }
    TEST_ASSERTION(projected > 0u, "No element degree of freedom of the richer element owns more than one entry.");
    direct_test_teardown(&test);
}

/**
 * @brief A degree of freedom on a space-matching object owns exactly one entry, with the orientation's sign.
 */
static void test_adversarial_matching_dof_owns_one_signed_entry(void)
{
    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        for (unsigned k = 0; k <= ndim; ++k)
        {
            direct_case_t c;
            direct_case_uniform(&c, ndim, 2u, k, 3u);
            direct_test_t test;
            direct_test_build(&c, &test);
            TEST_ASSERTION(test.layout.entry_count == test.layout.element_dof_count,
                           "The %uD order-%u uniform map has %zu entries for %zu element degrees of freedom.", ndim, k,
                           test.layout.entry_count, test.layout.element_dof_count);
            for (size_t local = 0; local < test.layout.element_dof_count; ++local)
            {
                const size_t to = test.entry_offsets[local + 1u];
                TEST_ASSERTION(to - test.entry_offsets[local] == 1u,
                               "Element DoF %zu of the uniform map owns %zu entries instead of one.", local,
                               to - test.entry_offsets[local]);
                const double value = test.entry_value[test.entry_offsets[local]];
                TEST_ASSERTION(value == 1.0 || value == -1.0,
                               "Element DoF %zu of the uniform map carries the coefficient %g.", local, value);
            }
            direct_test_check_conforming(&test);
            direct_test_teardown(&test);
        }
    }
}

/**
 * @brief Scattering a symmetric element matrix stays symmetric for every mixed-order pattern.
 *
 * Elements of differing orders have differing local sizes, so the element matrices cannot be one block: a
 * transfer that indexed the wrong element's degrees of freedom would scatter a block of the wrong size into
 * the right rows and break the symmetry without any visible count changing.
 */
static void test_adversarial_scatter_stays_symmetric_under_mixed_orders(void)
{
    direct_case_t patterns[4];
    direct_case_uniform(&patterns[0], 2u, 2u, 1u, 2u);
    {
        const unsigned rich[2] = {5u, 3u};
        direct_case_set_element(&patterns[0], 0u, rich);
    }
    direct_case_uniform(&patterns[1], 2u, 2u, 1u, 2u);
    {
        const unsigned rich[2] = {4u, 4u};
        direct_case_set_element(&patterns[1], 1u, rich);
        direct_case_set_element(&patterns[1], 2u, rich);
    }
    direct_case_uniform(&patterns[2], 2u, 3u, 1u, 3u);
    for (uint64_t element = 0; element < patterns[2].element_count; ++element)
    {
        uint64_t rest = element;
        const unsigned order = ((rest % 3u) + (rest / 3u)) % 2u == 0u ? 3u : 5u;
        const unsigned orders[2] = {order, order};
        direct_case_set_element(&patterns[2], element, orders);
    }
    direct_case_uniform(&patterns[3], 3u, 2u, 2u, 2u);
    {
        const unsigned rich[3] = {4u, 2u, 3u};
        direct_case_set_element(&patterns[3], 3u, rich);
    }

    for (unsigned variant = 0; variant < sizeof(patterns) / sizeof(patterns[0]); ++variant)
    {
        direct_test_t test;
        direct_test_build(&patterns[variant], &test);
        size_t widest = 0;
        for (uint64_t element = 0; element < test.element_count; ++element)
        {
            const size_t size = test.plan.element_dof_offsets[element + 1u] - test.plan.element_dof_offsets[element];
            widest = size > widest ? size : widest;
        }
        const size_t global = test.layout.global_dof_count;
        double *const matrix = malloc(sizeof(double) * widest * widest);
        double *const assembled = calloc(global * global, sizeof(double));
        for (uint64_t element = 0; element < test.element_count; ++element)
        {
            const size_t size = test.plan.element_dof_offsets[element + 1u] - test.plan.element_dof_offsets[element];
            for (size_t i = 0; i < size; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                {
                    // Symmetric in its indices, so any asymmetry after the scatter comes from the transfer.
                    matrix[i * size + j] = (double)((i * i + j * j + 3u * (i * j) % 5u) % 7u) / 8.0 - 0.25;
                }
                matrix[i * size + i] += (double)size;
            }
            direct_continuity_scatter(&test.plan, test.entry_offsets, test.entry_index, test.entry_value, element,
                                      matrix, size, assembled, global, 1.0);
        }
        for (size_t i = 0; i < global; ++i)
        {
            TEST_ASSERTION(assembled[i * global + i] > 0.0,
                           "Pattern %u leaves global degree of freedom %zu without a diagonal.", variant, i);
            for (size_t j = 0; j < global; ++j)
            {
                // A projected degree of freedom owns several entries, so the two triangles of the global matrix
                // accumulate the same terms in a different order and only agree to rounding. An orientation sign
                // that disagreed between two elements would put the two a whole factor of two apart, which no
                // rounding tolerance can hide.
                TEST_ASSERTION(fabs(assembled[i * global + j] - assembled[j * global + i]) <=
                                   1e-9 * (1.0 + fabs(assembled[i * global + j]) + fabs(assembled[j * global + i])),
                               "Pattern %u assembles an asymmetric matrix at (%zu, %zu): %.17g against %.17g.", variant,
                               i, j, assembled[i * global + j], assembled[j * global + i]);
            }
        }
        free(matrix);
        free(assembled);
        direct_test_check_conforming(&test);
        direct_test_teardown(&test);
    }
}

/**
 * @brief A specification the map cannot honour is reported, not walked into.
 *
 * Both rejections have to happen before any work is done, so the case is prepared and only the result is read:
 * a basis with no nodes or no functions at all would otherwise size the scratch with an order of zero and walk
 * an element that has no degrees of freedom.
 */
static void test_adversarial_invalid_specs_are_rejected(void)
{
    direct_case_t c;
    direct_case_uniform(&c, 2u, 2u, 0u, 2u);
    c.family = BASIS_LEGENDRE;
    direct_test_t test;
    TEST_ASSERTION(direct_test_prepare(&c, &test) == FDG_ERROR_NOT_IN_DOMAIN,
                   "A Legendre basis was accepted by the direct map.");
    direct_test_teardown(&test);

    // One offending element among sound ones is enough; the map has no valid global space for the object it
    // shares with its neighbours either.
    direct_case_uniform(&c, 2u, 2u, 0u, 2u);
    c.family = BASIS_LAGRANGE_GAUSS_LOBATTO;
    TEST_ASSERTION(direct_test_prepare(&c, &test) == FDG_SUCCESS, "The sound case was rejected.");
    direct_test_teardown(&test);

    direct_case_uniform(&c, 2u, 2u, 0u, 2u);
    const unsigned degenerate[2] = {0u, 2u};
    direct_case_set_element(&c, 3u, degenerate);
    TEST_ASSERTION(direct_test_prepare(&c, &test) == FDG_ERROR_NOT_IN_DOMAIN,
                   "An axis of order zero was accepted by the direct map.");
    direct_test_teardown(&test);

    direct_case_uniform(&c, 2u, 2u, 0u, 2u);
    c.family = BASIS_BERNSTEIN;
    TEST_ASSERTION(direct_test_prepare(&c, &test) == FDG_ERROR_NOT_IN_DOMAIN,
                   "A Bernstein basis was accepted by the direct map.");
    direct_test_teardown(&test);
}

/**
 * @brief A deterministic sweep of random per-element per-axis orders has to leave every invariant intact.
 *
 * The fixed cases each attack one shape; this attacks the combinations none of them reaches, and the seed makes
 * a failure the same on every run.
 */
static void test_adversarial_random_orders_hold_every_invariant(void)
{
    test_prng_t rng;
    test_prng_seed(&rng, 20261004u);
    for (unsigned ndim = 1; ndim <= 3; ++ndim)
    {
        for (unsigned round = 0; round < 12u; ++round)
        {
            direct_case_t c;
            direct_case_uniform(&c, ndim, 2u, round % (ndim + 1u), 2u);
            for (uint64_t element = 0; element < c.element_count; ++element)
            {
                for (unsigned axis = 0; axis < ndim; ++axis)
                {
                    DIRECT_CASE_ORDER(&c, element, axis) = 1u + test_prng_next_uint(&rng) % 5u;
                }
            }
            direct_test_t test;
            direct_test_build(&c, &test);
            direct_test_check_conforming(&test);
            direct_test_teardown(&test);
        }
    }
}

int main(void)
{
    test_uniform_family();
    test_scalar_counts_match_the_lattice();
    test_direct_is_never_larger();
    test_shared_coefficients_agree();
    test_scatter_preserves_symmetry();
    test_every_global_dof_is_reachable();
    test_mixed_element_orders();
    test_adversarial_every_family_and_kform_order();
    test_adversarial_anisotropy_inside_one_element();
    test_adversarial_distinct_order_per_element();
    test_adversarial_checkerboard_orders();
    test_adversarial_rich_pair_next_to_poor();
    test_adversarial_extreme_order_ratios();
    test_adversarial_degenerate_shapes();
    test_adversarial_element_relabelling_is_invariant();
    test_adversarial_axis_permutation_is_invariant();
    test_adversarial_projected_dof_owns_several_entries();
    test_adversarial_matching_dof_owns_one_signed_entry();
    test_adversarial_scatter_stays_symmetric_under_mixed_orders();
    test_adversarial_invalid_specs_are_rejected();
    test_adversarial_random_orders_hold_every_invariant();
    return 0;
}
