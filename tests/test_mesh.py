"""Check the topological Mesh wrapper and its boundary-constraint method."""

import numpy as np
import pytest
from fdg import Mesh, compute_kform_boundary_constraints
from fdg._fdg import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    SpaceMap,
)
from fdg.enum_type import BasisType


def grid2(x: int, y: int) -> int:
    """Point ID of the (x, y) node of the 2x2 grid."""
    return x + 3 * y


CORNERS_2X2 = np.array(
    [
        grid2(0, 0),
        grid2(1, 0),
        grid2(0, 1),
        grid2(1, 1),  # element 0 (ix,iy)=(0,0)
        grid2(1, 0),
        grid2(2, 0),
        grid2(1, 1),
        grid2(2, 1),  # element 1 (1,0)
        grid2(0, 1),
        grid2(1, 1),
        grid2(0, 2),
        grid2(1, 2),  # element 2 (0,1)
        grid2(1, 1),
        grid2(2, 1),
        grid2(1, 2),
        grid2(2, 2),  # element 3 (1,1)
    ],
    dtype=np.uint64,
)

# Lines of the 2x2 grid: per element, x-parallel lines first, then y-parallel.
LINES_2X2 = np.array(
    [
        [0, 1],
        [3, 4],
        [0, 3],
        [1, 4],
        [1, 2],
        [4, 5],
        [2, 5],
        [6, 7],
        [3, 6],
        [4, 7],
        [7, 8],
        [5, 8],
    ],
    dtype=np.uint64,
)

# Element boundaries in {x-start, y-start, x-end, y-end} line-ID layout.
QUADS_2X2 = np.array(
    [[2, 0, 3, 1], [3, 4, 6, 5], [8, 1, 9, 7], [9, 5, 11, 10]],
    dtype=np.uint64,
)

GEOM_BASIS = FunctionSpace(
    BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
)


def _affine_map(element_id: int, integration: IntegrationSpace) -> SpaceMap:
    """Affine map of a 2x2-grid element onto its physical unit square."""
    corner = int(CORNERS_2X2[element_id * 4])
    ix, iy = corner % 3, corner // 3
    xi, eta = np.meshgrid(
        np.linspace(-1.0, 1.0, 2), np.linspace(-1.0, 1.0, 2), indexing="ij"
    )
    return SpaceMap(
        CoordinateMap(
            DegreesOfFreedom(GEOM_BASIS, (0.5 * xi + ix - 0.5).ravel()), integration
        ),
        CoordinateMap(
            DegreesOfFreedom(GEOM_BASIS, (0.5 * eta + iy - 0.5).ravel()), integration
        ),
    )


def _apply_boundary_rows(
    result: tuple[np.ndarray, ...], values: np.ndarray
) -> np.ndarray:
    """Apply packed boundary rows to one element's flattened DoFs."""
    row_offsets, _, local_dofs, coefficients = result
    return np.array(
        [
            np.dot(
                coefficients[row_offsets[row] : row_offsets[row + 1]],
                values[local_dofs[row_offsets[row] : row_offsets[row + 1]]],
            )
            for row in range(row_offsets.size - 1)
        ]
    )


def _constant_kform_values(specs: KFormSpecs) -> np.ndarray:
    """Return DoFs for a constant value in every k-form component."""
    return np.ones(int(np.sum(specs.component_dof_counts)))


def test_mesh_from_corners_2d_properties() -> None:
    """Corner-built meshes expose the correct counts and collections."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)
    assert mesh.ndim == 2
    assert mesh.point_count == 9
    assert mesh.element_count == 4

    collections = mesh.collections
    assert len(collections) == 2
    np.testing.assert_array_equal(collections[0], LINES_2X2)
    np.testing.assert_array_equal(collections[1], QUADS_2X2)


def test_mesh_iteration_counts() -> None:
    """Shared and boundary iterations report the expected objects."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)

    shared_lines = mesh.iterate_shared(1)
    assert len(shared_lines) == 4
    for mdim, _, element_ids, orientations in shared_lines:
        assert mdim == 1
        assert element_ids.size == 2
        assert orientations.shape == (2, 2)

    shared_points = mesh.iterate_shared(0)
    assert len(shared_points) == 5
    center = next(item for item in shared_points if item[2].size == 4)
    assert center[0] == 0
    assert int(center[1]) == 4
    np.testing.assert_array_equal(center[2], np.array([0, 1, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(
        center[3],
        np.array([[1, 2], [-1, 2], [1, -2], [-1, -2]], dtype=np.int8),
    )

    assert len(mesh.iterate_boundary(1)) == 8
    assert len(mesh.iterate_boundary(0)) == 8

    shared_all = mesh.iterate_shared_all()
    assert len(shared_all) == 9
    assert all(item[0] == 1 for item in shared_all[:4])
    assert all(item[0] == 0 for item in shared_all[4:])

    boundary_all = mesh.iterate_boundary_all()
    assert len(boundary_all) == 16
    assert all(item[0] == 1 for item in boundary_all[:8])
    assert all(item[0] == 0 for item in boundary_all[8:])


def test_mesh_element_object() -> None:
    """Element-local axis specifications resolve to global object IDs."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)

    # Corner point of element 0 at axis-0 start and axis-1 end: grid point (0, 1).
    assert mesh.element_object(0, [-1, 2]) == 3
    # y-start edge of element 0 is line 0.
    assert mesh.element_object(0, [0, -2]) == 0
    # The shared x-interface of elements 0 and 1 is the same line from both sides.
    assert mesh.element_object(0, [1, 0]) == mesh.element_object(1, [-1, 0])

    with pytest.raises(ValueError):
        mesh.element_object(0, [0, 0])
    with pytest.raises(ValueError):
        mesh.element_object(0, [0, -1])
    with pytest.raises(ValueError):
        mesh.element_object(4, [-1, -2])
    with pytest.raises(ValueError):
        mesh.element_object(0, [-1])


def test_mesh_from_corners_rejects_bad_input() -> None:
    """Invalid constructor arguments raise clear errors."""
    with pytest.raises(ValueError):
        Mesh.from_corners(0, CORNERS_2X2)
    with pytest.raises(ValueError):
        Mesh.from_corners(64, CORNERS_2X2)
    with pytest.raises(ValueError):
        Mesh.from_corners(2, CORNERS_2X2[:7])
    with pytest.raises(ValueError):
        Mesh.from_corners(2, np.array([], dtype=np.uint64))
    with pytest.raises(TypeError):
        Mesh()


def test_mesh_from_collections() -> None:
    """Collection-built meshes match the source collections and topology."""
    lines = np.array(
        [[0, 1], [0, 2], [2, 3], [1, 3], [2, 4], [4, 5], [3, 5]],
        dtype=np.uint64,
    )
    elements = np.array([[0, 1, 2, 3], [2, 4, 5, 6]], dtype=np.uint64)

    mesh = Mesh.from_collections(2, 6, (lines, elements))
    assert mesh.point_count == 6
    assert mesh.element_count == 2
    np.testing.assert_array_equal(mesh.collections[0], lines)
    np.testing.assert_array_equal(mesh.collections[1], elements)

    shared = mesh.iterate_shared(1)
    assert len(shared) == 1
    mdim, object_id, element_ids, orientations = shared[0]
    assert mdim == 1
    assert int(object_id) == 2
    np.testing.assert_array_equal(element_ids, np.array([0, 1], dtype=np.uint64))
    assert orientations.shape == (2, 2)


def test_mesh_boundary_constraints_matches_free_function() -> None:
    """The bound method returns bit-identical results to the free function."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)
    integration = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))
    element_specs = KFormSpecs(1, GEOM_BASIS)
    test_specs = KFormSpecs(1, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))

    shared = mesh.iterate_shared(1)[0]
    object_id = int(shared[1])
    element_id = int(shared[2][0])
    element_map = _affine_map(element_id, integration)

    bound = mesh.compute_kform_boundary_constraints(
        test_specs, element_specs, element_map, element_id, object_id
    )
    free = compute_kform_boundary_constraints(
        test_specs,
        element_specs,
        element_map,
        mesh.collections,
        mesh.point_count,
        element_id,
        object_id,
    )
    for expected, actual in zip(free, bound):
        np.testing.assert_array_equal(actual, expected)


def test_mesh_boundary_constraints_continuity_2d() -> None:
    """Constant k-form traces match across a shared face."""
    mesh = Mesh.from_corners(2, CORNERS_2X2)
    integration = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))
    element_specs = KFormSpecs(1, GEOM_BASIS)
    test_specs = KFormSpecs(1, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))
    values = _constant_kform_values(element_specs)

    mdim, object_id, element_ids, _ = mesh.iterate_shared(1)[0]
    assert mdim == 1
    traces = [
        _apply_boundary_rows(
            mesh.compute_kform_boundary_constraints(
                test_specs,
                element_specs,
                _affine_map(int(element_id), integration),
                int(element_id),
                int(object_id),
            ),
            values,
        )
        for element_id in element_ids
    ]
    np.testing.assert_allclose(traces[0], traces[1])
