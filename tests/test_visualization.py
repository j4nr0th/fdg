"""Tests for finite-element visualization helpers."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    Mesh,
    MeshGeometry,
    SpaceMap,
)
from fdg.visualization import (
    lagrange_geometry_cells,
    lagrange_hexahedral_grid,
    lagrange_quadrilateral_grid,
)


def _interval_map(integration_order: int) -> SpaceMap:
    """Map the reference interval affinely onto the unit interval."""
    basis = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1))
    integration = IntegrationSpace(IntegrationSpecs(integration_order))
    return SpaceMap(CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0]), integration))


def _plane_line_map(integration_order: int) -> SpaceMap:
    """Map the reference interval diagonally onto the unit square."""
    basis = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1))
    integration = IntegrationSpace(IntegrationSpecs(integration_order))
    return SpaceMap(
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0]), integration),
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0]), integration),
    )


def _square_map(integration_orders: tuple[int, int] = (3, 3)) -> SpaceMap:
    """Map the reference square affinely onto the unit square."""
    integration = IntegrationSpace(
        IntegrationSpecs(integration_orders[0]),
        IntegrationSpecs(integration_orders[1]),
    )
    basis = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    return SpaceMap(
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 0.0, 1.0, 1.0]), integration),
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0, 0.0, 1.0]), integration),
    )


def _flat_square_map(integration_order: int = 3) -> SpaceMap:
    """Map the reference square onto the unit square in the ``z = 0`` plane."""
    integration = IntegrationSpace(
        IntegrationSpecs(integration_order), IntegrationSpecs(integration_order)
    )
    basis = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    return SpaceMap(
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 0.0, 1.0, 1.0]), integration),
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0, 0.0, 1.0]), integration),
        CoordinateMap(DegreesOfFreedom(basis, np.zeros(4)), integration),
    )


def _cube_map(integration_order: int = 3) -> SpaceMap:
    """Map the reference cube affinely onto the unit cube."""
    integration = IntegrationSpace(
        *(IntegrationSpecs(integration_order) for _ in range(3))
    )
    basis = FunctionSpace(*(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(3)))
    nodes = np.linspace(-1.0, 1.0, 2)
    coordinates = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    return SpaceMap(
        *(
            CoordinateMap(
                DegreesOfFreedom(basis, (0.5 * (coordinate + 1.0)).ravel()),
                integration,
            )
            for coordinate in coordinates
        )
    )


def _hypercube_map(input_dimensions: int, output_dimensions: int) -> SpaceMap:
    """Map a reference hypercube onto the origin of a larger physical space."""
    integration = IntegrationSpace(
        *(IntegrationSpecs(3) for _ in range(input_dimensions))
    )
    basis = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(input_dimensions))
    )
    return SpaceMap(
        *(
            CoordinateMap(
                DegreesOfFreedom(basis, np.zeros(2**input_dimensions)), integration
            )
            for _ in range(output_dimensions)
        )
    )


def test_lagrange_quadrilateral_grid_interpolates_nonlinear_data() -> None:
    """High-order quadrilateral ordering preserves a polynomial field."""
    integration = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))
    basis = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    space_map = SpaceMap(
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 0.0, 1.0, 1.0]), integration),
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0, 0.0, 1.0]), integration),
    )
    nodes = np.linspace(0.0, 1.0, 7)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    field = x**2 + 2.0 * y**2 + 3.0 * x * y
    grid = lagrange_quadrilateral_grid([space_map], 6, {"field": [field]})
    assert "HigherOrderDegrees" not in grid.cell_data

    query_nodes = np.linspace(0.05, 0.95, 11)
    query_x, query_y = np.meshgrid(query_nodes, query_nodes, indexing="ij")
    query = pv.PolyData(
        np.column_stack(
            [
                query_x.ravel(),
                query_y.ravel(),
                np.zeros(query_x.size),
            ]
        )
    )
    sampled = query.sample(grid)
    expected = (query_x**2 + 2.0 * query_y**2 + 3.0 * query_x * query_y).ravel()
    np.testing.assert_allclose(sampled["field"], expected, atol=1.0e-12)


def test_lagrange_hexahedral_grid_accepts_anisotropic_orders(tmp_path) -> None:
    """Anisotropic orders survive high-order VTK serialization."""
    integration = IntegrationSpace(
        IntegrationSpecs(3), IntegrationSpecs(3), IntegrationSpecs(3)
    )
    basis = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    coordinates = np.meshgrid(
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, 1.0)),
        indexing="ij",
    )
    space_map = SpaceMap(
        *(
            CoordinateMap(DegreesOfFreedom(basis, coordinate.ravel()), integration)
            for coordinate in coordinates
        )
    )
    orders = (1, 2, 3)
    shape = tuple(order + 1 for order in orders)
    field = np.arange(np.prod(shape), dtype=float).reshape(shape)

    grid = lagrange_hexahedral_grid([space_map], orders, {"field": [field]})
    assert grid.n_cells == 1
    assert grid.n_points == np.prod(shape)
    assert grid.celltypes[0] == pv.CellType.LAGRANGE_HEXAHEDRON
    np.testing.assert_array_equal(grid.point_data["field"], field.ravel())
    output = tmp_path / "anisotropic.vtu"
    grid.save(output)
    loaded = pv.read(output)
    degrees = loaded.GetCellData().GetHigherOrderDegrees()
    assert degrees is not None
    assert degrees.GetTuple(0) == (1.0, 2.0, 3.0)
    assert [loaded.GetCell(0).GetOrder(axis) for axis in range(3)] == [1, 2, 3]


def test_lagrange_geometry_cells_builds_curve_cells() -> None:
    """One reference dimension yields a VTK Lagrange curve in VTK point order."""
    mesh = Mesh.from_corners(1, np.array([0, 1], dtype=np.uint64))
    geometry = MeshGeometry.from_mesh_points(
        mesh, np.array([[0.0], [1.0]]), IntegrationSpace(IntegrationSpecs(3))
    )
    cells, celltypes, points, degrees = lagrange_geometry_cells(geometry)

    assert celltypes.dtype == np.uint8
    assert celltypes.tolist() == [pv.CellType.LAGRANGE_CURVE]
    np.testing.assert_array_equal(degrees, [[3]])
    assert points.shape == (4, 3)
    assert points.dtype == np.double
    # The points stay in C-order layout: increasing parametric coordinate.
    np.testing.assert_allclose(points[:, 0], np.linspace(0.0, 1.0, 4), atol=1.0e-15)
    np.testing.assert_allclose(points[:, 1:], 0.0)
    # VTK stores the vertices (parametric coordinates 0 and 1) first, so the
    # connectivity reorders the natural points 0, 3, 1, 2 into VTK order.
    np.testing.assert_array_equal(cells, [4, 0, 3, 1, 2])


def test_lagrange_geometry_cells_curve_interpolates_polynomial() -> None:
    """A curve built from raw cell arrays interpolates a quadratic field."""
    geometry = MeshGeometry.from_elements(_interval_map(5))
    cells, celltypes, points, _ = lagrange_geometry_cells(geometry, order=3)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    nodes = np.linspace(0.0, 1.0, 4)
    grid.point_data["field"] = nodes**2 + 2.0 * nodes

    query_nodes = np.linspace(0.05, 0.95, 11)
    query = pv.PolyData(
        np.column_stack(
            [query_nodes, np.zeros_like(query_nodes), np.zeros_like(query_nodes)]
        )
    )
    sampled = query.sample(grid)
    expected = query_nodes**2 + 2.0 * query_nodes
    np.testing.assert_allclose(sampled["field"], expected, atol=1.0e-12)


def test_lagrange_geometry_cells_dispatches_on_reference_dimensions() -> None:
    """Two and three reference dimensions select the cell family."""
    _, quad_types, _, _ = lagrange_geometry_cells(
        MeshGeometry.from_elements(_square_map()), order=1
    )
    assert quad_types.tolist() == [pv.CellType.LAGRANGE_QUADRILATERAL]

    _, hex_types, _, _ = lagrange_geometry_cells(
        MeshGeometry.from_elements(_cube_map()), order=1
    )
    assert hex_types.tolist() == [pv.CellType.LAGRANGE_HEXAHEDRON]


def test_lagrange_geometry_cells_default_orders_follow_integration_rules() -> None:
    """Without an explicit order every cell uses its own integration orders."""
    geometry = MeshGeometry.from_elements(_square_map((2, 4)), _square_map((3, 3)))

    cells, celltypes, points, degrees = lagrange_geometry_cells(geometry)
    np.testing.assert_array_equal(degrees, [[2, 4], [3, 3]])
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    assert grid.n_cells == 2
    assert grid.n_points == 3 * 5 + 4 * 4


def test_lagrange_geometry_cells_explicit_orders_override_defaults() -> None:
    """An explicit order applies uniformly over the integration-rule orders."""
    geometry = MeshGeometry.from_elements(_square_map((5, 1)), _square_map((1, 5)))

    _, _, points, degrees = lagrange_geometry_cells(geometry, order=2)
    np.testing.assert_array_equal(degrees, [[2, 2], [2, 2]])
    assert points.shape == (18, 3)

    _, _, points, degrees = lagrange_geometry_cells(geometry, order=(1, 2))
    np.testing.assert_array_equal(degrees, [[1, 2], [1, 2]])
    assert points.shape == (12, 3)

    _, _, points, degrees = lagrange_geometry_cells(
        MeshGeometry.from_elements(_cube_map()), order=(1, 2, 3)
    )
    np.testing.assert_array_equal(degrees, [[1, 2, 3]])
    assert points.shape == (24, 3)


def test_lagrange_geometry_cells_pads_to_three_coordinates() -> None:
    """Physical coordinates are zero padded up to VTK's three columns."""
    _, _, points, _ = lagrange_geometry_cells(
        MeshGeometry.from_elements(_interval_map(3)), order=2
    )
    assert points.shape == (3, 3)
    np.testing.assert_allclose(points[:, 1:], 0.0)

    _, _, points, _ = lagrange_geometry_cells(
        MeshGeometry.from_elements(_plane_line_map(3)), order=2
    )
    assert points.shape == (3, 3)
    np.testing.assert_allclose(points[:, 0], points[:, 1])
    np.testing.assert_allclose(points[:, 2], 0.0)

    _, _, points, _ = lagrange_geometry_cells(
        MeshGeometry.from_elements(_flat_square_map(3)), order=2
    )
    assert points.shape == (9, 3)
    np.testing.assert_allclose(points[:, 2], 0.0)


def test_lagrange_geometry_cells_empty_geometry() -> None:
    """An empty geometry yields well-formed empty arrays instead of raising."""
    cells, celltypes, points, degrees = lagrange_geometry_cells(
        MeshGeometry(2, 3), order=2
    )

    assert cells.shape == (0,)
    assert cells.dtype == np.intp
    assert celltypes.shape == (0,)
    assert celltypes.dtype == np.uint8
    assert points.shape == (0, 3)
    assert points.dtype == np.double
    assert degrees.shape == (0, 2)
    assert degrees.dtype == np.int32


def test_lagrange_geometry_cells_rejects_invalid_dimensions() -> None:
    """Reference and physical dimensions outside the supported range raise."""
    with pytest.raises(ValueError):
        lagrange_geometry_cells(MeshGeometry(0, 1))
    with pytest.raises(ValueError):
        lagrange_geometry_cells(MeshGeometry(4, 4))
    with pytest.raises(ValueError):
        lagrange_geometry_cells(MeshGeometry(2, 4))


def test_lagrange_geometry_cells_rejects_non_positive_orders() -> None:
    """Explicit and default sampling orders must all be positive."""
    geometry = MeshGeometry.from_elements(_square_map((0, 0)), _square_map((3, 3)))
    with pytest.raises(ValueError):
        lagrange_geometry_cells(geometry, order=0)
    with pytest.raises(ValueError):
        lagrange_geometry_cells(geometry, order=(2, 0))
    with pytest.raises(ValueError):
        lagrange_geometry_cells(geometry, order=-1)
    with pytest.raises(ValueError):
        lagrange_geometry_cells(geometry)


def test_lagrange_geometry_cells_hexahedron_interpolates_polynomial() -> None:
    """Hexahedral cells built from raw cell arrays interpolate a cubic field."""
    geometry = MeshGeometry.from_elements(_cube_map())
    cells, celltypes, points, _ = lagrange_geometry_cells(geometry, order=3)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    nodes = np.linspace(0.0, 1.0, 4)
    x, y, z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    field = x**2 + 2.0 * y * z + 3.0 * x * y * z
    grid.point_data["field"] = field.ravel()

    query_nodes = np.linspace(0.1, 0.9, 4)
    query_x, query_y, query_z = np.meshgrid(
        query_nodes, query_nodes, query_nodes, indexing="ij"
    )
    query = pv.PolyData(
        np.column_stack([query_x.ravel(), query_y.ravel(), query_z.ravel()])
    )
    sampled = query.sample(grid)
    expected = (
        query_x**2 + 2.0 * query_y * query_z + 3.0 * query_x * query_y * query_z
    ).ravel()
    np.testing.assert_allclose(sampled["field"], expected, atol=1.0e-12)


def test_lagrange_grids_reject_empty_sequence() -> None:
    """An empty element sequence still raises a ValueError."""
    with pytest.raises(ValueError):
        lagrange_quadrilateral_grid([], 2)
    with pytest.raises(ValueError):
        lagrange_hexahedral_grid([], 2)


def test_lagrange_quadrilateral_grid_rejects_mixed_dimensions() -> None:
    """Maps that disagree on the physical dimension are rejected as a whole."""
    with pytest.raises(ValueError):
        lagrange_quadrilateral_grid([_square_map(), _flat_square_map()], 2)


def test_lagrange_hexahedral_grid_rejects_zero_order() -> None:
    """A zero order is rejected instead of building a one-point cell."""
    with pytest.raises(ValueError):
        lagrange_hexahedral_grid([_cube_map()], 0)


def test_lagrange_grids_reject_wrong_dimensions() -> None:
    """The wrappers validate reference and physical dimensions up front."""
    with pytest.raises(ValueError):
        lagrange_quadrilateral_grid([_cube_map()], 2)
    with pytest.raises(ValueError):
        lagrange_hexahedral_grid([_square_map()], 3)
    with pytest.raises(ValueError):
        lagrange_quadrilateral_grid([_hypercube_map(2, 4)], 2)
    with pytest.raises(ValueError):
        lagrange_hexahedral_grid([_hypercube_map(3, 4)], 3)


def test_lagrange_quadrilateral_grid_validates_point_data() -> None:
    """Point-data sequence length and array shape are validated."""
    field = np.zeros((3, 3))
    with pytest.raises(ValueError):
        lagrange_quadrilateral_grid([_square_map()], 2, {"f": [field, field]})
    with pytest.raises(ValueError):
        lagrange_quadrilateral_grid([_square_map()], 2, {"f": [np.zeros((2, 2))]})
