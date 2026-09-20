"""Check the batched per-element geometry collection."""

import numpy as np
import pytest
from fdg import Mesh
from fdg._fdg import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    MeshGeometry,
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

# Physical coordinates of the 9 grid points of the 2x2 grid.
POINTS_2X2 = np.array(
    [[px - 1.0, py - 1.0] for py in range(3) for px in range(3)], dtype=np.double
)

GEOM_BASIS = FunctionSpace(
    BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
)

INTEGRATION = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))


def _affine_dofs(element_id: int) -> tuple[DegreesOfFreedom, DegreesOfFreedom]:
    """Geometry degrees of freedom of the affine map of one 2x2-grid element."""
    corner = int(CORNERS_2X2[element_id * 4])
    ix, iy = corner % 3, corner // 3
    xi, eta = np.meshgrid(
        np.linspace(-1.0, 1.0, 2), np.linspace(-1.0, 1.0, 2), indexing="ij"
    )
    return (
        DegreesOfFreedom(GEOM_BASIS, (0.5 * xi + ix - 0.5).ravel()),
        DegreesOfFreedom(GEOM_BASIS, (0.5 * eta + iy - 0.5).ravel()),
    )


def _affine_map(element_id: int, integration: IntegrationSpace) -> SpaceMap:
    """Affine map of a 2x2-grid element onto its physical unit square."""
    x_dofs, y_dofs = _affine_dofs(element_id)
    return SpaceMap(
        CoordinateMap(x_dofs, integration),
        CoordinateMap(y_dofs, integration),
    )


def _add_affine_element(
    store: MeshGeometry, element_id: int, integration: IntegrationSpace
) -> None:
    """Add the affine map of one element with its degrees of freedom."""
    store.add_element(_affine_map(element_id, integration), *_affine_dofs(element_id))


@pytest.fixture
def mesh() -> Mesh:
    """Return the 2x2 element grid."""
    return Mesh.from_corners(2, CORNERS_2X2)


@pytest.fixture
def geometry(mesh: Mesh) -> MeshGeometry:
    """Return geometry data of the 2x2 grid built from the point coordinates."""
    return MeshGeometry.from_mesh_points(mesh, POINTS_2X2, INTEGRATION)


def test_from_mesh_points_matches_manual_maps(geometry: MeshGeometry) -> None:
    """Store-built space maps equal manually built affine maps bit for bit."""
    assert geometry.element_count == 4
    assert geometry.option_count == 1
    np.testing.assert_array_equal(geometry.element_options, np.zeros(4, dtype=np.uint32))
    np.testing.assert_array_equal(
        geometry.offsets, np.array([0, 8, 16, 24, 32], dtype=np.uint64)
    )

    for element_id in range(4):
        stored = geometry.space_map(element_id)
        expected = _affine_map(element_id, INTEGRATION)
        for icoordinate in range(2):
            np.testing.assert_array_equal(
                stored.coordinate_map(icoordinate).values,
                expected.coordinate_map(icoordinate).values,
            )
        np.testing.assert_array_equal(stored.determinant, expected.determinant)
        np.testing.assert_array_equal(stored.inverse_map, expected.inverse_map)


def test_option_dedup() -> None:
    """Identical geometry specifications collapse into one option."""
    store = MeshGeometry()
    _add_affine_element(store, 0, INTEGRATION)
    _add_affine_element(store, 1, INTEGRATION)
    other_integration = IntegrationSpace(IntegrationSpecs(4), IntegrationSpecs(4))
    _add_affine_element(store, 2, other_integration)

    assert store.option_count == 2
    assert store.element_count == 3
    np.testing.assert_array_equal(
        store.element_options, np.array([0, 0, 1], dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        store.offsets, np.array([0, 8, 16, 24], dtype=np.uint64)
    )

    function_space, integration_space = store.option(0)
    assert isinstance(function_space, FunctionSpace)
    assert isinstance(integration_space, IntegrationSpace)
    assert store.option(1)[1].orders == other_integration.orders


def test_views_and_freeze() -> None:
    """Array views expose the storage and freeze the collection."""
    store = MeshGeometry()
    _add_affine_element(store, 0, INTEGRATION)
    _add_affine_element(store, 1, INTEGRATION)

    values = store.values
    offsets = store.offsets
    options = store.element_options
    assert values.dtype == np.double
    assert offsets.dtype == np.uint64
    assert options.dtype == np.uint32
    assert values.size == 16
    values[0] = 42.0
    # The mutation is visible through the space_map getter.
    mutated = store.space_map(0).coordinate_map(0).values
    fresh = _affine_map(0, INTEGRATION).coordinate_map(0).values
    assert not np.array_equal(mutated, fresh)
    np.testing.assert_array_equal(
        store.space_map(1).coordinate_map(0).values,
        _affine_map(1, INTEGRATION).coordinate_map(0).values,
    )

    with pytest.raises(ValueError):
        _add_affine_element(store, 2, INTEGRATION)

    # Overwriting existing values stays allowed after freezing.
    store.set_element_values(1, np.full(8, 7.0))
    np.testing.assert_array_equal(values[8:16], np.full(8, 7.0))


def test_errors(mesh: Mesh) -> None:
    """Invalid usage raises the expected exceptions."""
    empty = MeshGeometry()
    with pytest.raises(IndexError):
        empty.space_map(0)
    with pytest.raises(IndexError):
        empty.option(0)
    with pytest.raises(TypeError):
        empty.add_element("not a space map")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        empty.add_element(_affine_map(0, INTEGRATION))

    with pytest.raises(ValueError):
        MeshGeometry.from_mesh_points(mesh, POINTS_2X2[:, 0], INTEGRATION)
    with pytest.raises(ValueError):
        MeshGeometry.from_mesh_points(mesh, POINTS_2X2[:5], INTEGRATION)
    one_d_integration = IntegrationSpace(IntegrationSpecs(3))
    with pytest.raises(ValueError):
        MeshGeometry.from_mesh_points(mesh, POINTS_2X2, one_d_integration)

    store = MeshGeometry()
    _add_affine_element(store, 0, INTEGRATION)
    with pytest.raises(IndexError):
        store.set_element_values(1, np.zeros(8))
    with pytest.raises(ValueError):
        store.set_element_values(0, np.zeros(7))

    # Geometry with mismatched per-coordinate function spaces is rejected.
    x_dofs = DegreesOfFreedom(GEOM_BASIS, [0.0, 1.0, 0.0, 1.0])
    y_dofs = DegreesOfFreedom(
        FunctionSpace(
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
        ),
        np.zeros(9),
    )
    mixed = SpaceMap(
        CoordinateMap(x_dofs, INTEGRATION),
        CoordinateMap(y_dofs, INTEGRATION),
    )
    with pytest.raises(ValueError):
        MeshGeometry.from_elements([(mixed, x_dofs, y_dofs)])


def test_geometry_maps_feed_boundary_constraints(
    mesh: Mesh, geometry: MeshGeometry
) -> None:
    """Store-built maps are interchangeable with manually built maps."""
    element_specs = KFormSpecs(1, GEOM_BASIS)
    test_specs = KFormSpecs(1, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))

    shared = mesh.iterate_shared(1)[0]
    object_id = int(shared[1])
    element_id = int(shared[2][0])

    stored_map = geometry.space_map(element_id)
    manual_map = _affine_map(element_id, INTEGRATION)

    stored_result = mesh.compute_kform_boundary_constraints(
        test_specs, element_specs, stored_map, element_id, object_id
    )
    manual_result = mesh.compute_kform_boundary_constraints(
        test_specs, element_specs, manual_map, element_id, object_id
    )
    for expected, actual in zip(manual_result, stored_result):
        np.testing.assert_array_equal(actual, expected)
