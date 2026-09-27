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
    compute_kform_boundary_mass_matrices,
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
    """Add the affine map of one element to the store."""
    store.add_element(_affine_map(element_id, integration))


def _map_1_in_2_out() -> SpaceMap:
    """Space map from the 1D reference space onto a 2D physical space."""
    basis = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1))
    integration = IntegrationSpace(IntegrationSpecs(3))
    return SpaceMap(
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0]), integration),
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0]), integration),
    )


def _map_2_in_3_out() -> SpaceMap:
    """Space map from the 2D reference space onto a 3D physical space."""
    return SpaceMap(
        *(
            CoordinateMap(DegreesOfFreedom(GEOM_BASIS, np.zeros(4)), INTEGRATION)
            for _ in range(3)
        )
    )


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


def test_errors(mesh: Mesh) -> None:
    """Invalid usage raises the expected exceptions."""
    empty = MeshGeometry(2, 2)
    with pytest.raises(IndexError):
        empty.space_map(0)
    with pytest.raises(TypeError):
        empty.add_element()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        empty.add_element("not a space map")  # type: ignore[arg-type]

    # Missing arguments raise instead of aborting: the positional-only specs
    # have a NULL keyword name, which used to crash the error formatting.
    # from_elements() no longer goes through that helper and reports its own
    # missing space map instead.
    with pytest.raises(TypeError):
        empty.space_map()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MeshGeometry.from_elements()
    with pytest.raises(TypeError):
        MeshGeometry.from_mesh_points()  # type: ignore[call-arg]

    # A bare space map is all add_element takes now.
    _add_affine_element(empty, 0, INTEGRATION)
    assert empty.element_count == 1

    with pytest.raises(ValueError):
        MeshGeometry.from_mesh_points(mesh, POINTS_2X2[:, 0], INTEGRATION)
    with pytest.raises(ValueError):
        MeshGeometry.from_mesh_points(mesh, POINTS_2X2[:5], INTEGRATION)
    one_d_integration = IntegrationSpace(IntegrationSpecs(3))
    with pytest.raises(ValueError):
        MeshGeometry.from_mesh_points(mesh, POINTS_2X2, one_d_integration)

    # Geometry with mismatched per-coordinate function spaces is stored as
    # given, together with everything else that is a space map.
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
    store = MeshGeometry.from_elements(mixed)
    assert store.element_count == 1
    assert store.space_map(0) is mixed

    with pytest.raises(TypeError):
        MeshGeometry.from_elements("not a map")  # type: ignore[arg-type]


def test_constructor_dimensions() -> None:
    """The constructor takes two validated dimensions and exposes them."""
    with pytest.raises(TypeError):
        MeshGeometry()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MeshGeometry(2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MeshGeometry(2, 2, 2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MeshGeometry(input_dimensions=2, output_dimensions=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MeshGeometry("2", 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MeshGeometry(2, 2.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        MeshGeometry(-1, 2)
    with pytest.raises(ValueError):
        MeshGeometry(0, 0)
    with pytest.raises(ValueError):
        MeshGeometry(3, 2)
    with pytest.raises(ValueError):
        MeshGeometry(2, 2**31)

    empty = MeshGeometry(2, 3)
    assert empty.input_dimensions == 2
    assert empty.output_dimensions == 3
    assert empty.element_count == 0

    store = MeshGeometry(2, 2)
    _add_affine_element(store, 0, INTEGRATION)
    assert store.input_dimensions == 2
    assert store.output_dimensions == 2
    assert store.element_count == 1


def test_add_element_dimension_mismatch() -> None:
    """add_element rejects space maps of different dimensions."""
    store = MeshGeometry(2, 2)
    # One input dimension instead of two.
    with pytest.raises(ValueError):
        store.add_element(_map_1_in_2_out())
    # Two input dimensions, but three output dimensions.
    with pytest.raises(ValueError):
        store.add_element(_map_2_in_3_out())
    assert store.element_count == 0

    _add_affine_element(store, 0, INTEGRATION)
    assert store.element_count == 1


def test_from_elements_dimensions() -> None:
    """from_elements infers the dimensions from its first space map."""
    first = _map_2_in_3_out()
    store = MeshGeometry.from_elements(first, _map_2_in_3_out())
    assert store.input_dimensions == 2
    assert store.output_dimensions == 3
    assert store.element_count == 2
    assert store.space_map(0) is first

    with pytest.raises(ValueError):
        MeshGeometry.from_elements(first, _affine_map(0, INTEGRATION))


def test_geometry_maps_feed_boundary_constraints(
    mesh: Mesh, geometry: MeshGeometry
) -> None:
    """Store-built maps are interchangeable with manually built maps."""
    element_specs = KFormSpecs(1, GEOM_BASIS)

    shared = mesh.iterate_shared(1)[0]
    element_ids = [int(element_id) for element_id in shared[2]]
    orientations = [[int(v) for v in record] for record in shared[3]]

    stored_maps = [geometry.space_map(element_id) for element_id in element_ids]
    manual_maps = [_affine_map(element_id, INTEGRATION) for element_id in element_ids]

    stored_result = compute_kform_boundary_mass_matrices(
        [element_specs for _ in element_ids],
        orientations,
        [space_map.integration_space for space_map in stored_maps],
        element_maps=stored_maps,
        boundary_dimension=1,
    )
    manual_result = compute_kform_boundary_mass_matrices(
        [element_specs for _ in element_ids],
        orientations,
        [space_map.integration_space for space_map in manual_maps],
        element_maps=manual_maps,
        boundary_dimension=1,
    )
    for expected, actual in zip(manual_result[2], stored_result[2]):
        np.testing.assert_array_equal(actual, expected)
