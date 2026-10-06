"""Utilities for visualizing mapped finite-element fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv

from fdg._fdg import (
    DegreesOfFreedom,
    IntegrationSpace,
    IntegrationSpecs,
    KForm,
    KFormSpecs,
    MeshGeometry,
    SampledSpaceMap,
    SpaceMap,
    transform_kform_to_target_sampled,
)
from fdg.degrees_of_freedom import reconstruct
from fdg.domains import (
    HypercubeDomain,
    _vtk_1d_indices,
    _vtk_2d_indices,
    _vtk_3d_indices,
)
from fdg.enum_type import IntegrationMethod


def _sample_orders(sample_order: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Normalize a scalar or per-axis sampling order."""
    if isinstance(sample_order, int):
        orders = (sample_order,) * ndim
    else:
        orders = tuple(sample_order)
        if len(orders) != ndim:
            raise ValueError(f"Expected {ndim} sample orders, got {len(orders)}.")
    if any(order < 0 for order in orders):
        raise ValueError("Sample orders must be non-negative.")
    return orders


def _lagrange_orders(order: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Normalize a sampling order and require a positive Lagrange order.

    Parameters
    ----------
    order : int or sequence of int
        Uniform sampling order, either for every reference axis or per axis.
    ndim : int
        Number of reference axes the order must cover.

    Returns
    -------
    tuple of int
        One positive sampling order per reference axis.

    Raises
    ------
    ValueError
        If ``order`` does not supply one entry per reference axis, if an
        order is negative, or if an order is zero: a Lagrange cell needs at
        least one interval per axis, so its order cannot be zero.
    """
    orders = _sample_orders(order, ndim)
    if any(component < 1 for component in orders):
        raise ValueError("The Lagrange order must be positive.")
    return orders


def sample_kform_on_uniform_grid(
    specs: KFormSpecs,
    values: npt.ArrayLike,
    space_map: SpaceMap,
    sample_order: int | Sequence[int],
) -> tuple[SampledSpaceMap, tuple[npt.NDArray[np.double], ...]]:
    """Sample and transform a k-form on a uniform reference-space grid.

    Parameters
    ----------
    specs : KFormSpecs
        Specification of the k-form represented by ``values``.
    values : array_like
        Flattened k-form degrees of freedom.
    space_map : SpaceMap
        Element map used to transform the sampled k-form.
    sample_order : int or sequence of int
        Uniform sampling order, either for every reference axis or per axis.

    Returns
    -------
    sampled_map : SampledSpaceMap
        Sampled element map. Its ``positions`` array contains physical points.
    components : tuple of array
        K-form components in the physical basis at the sampled points.
    """
    orders = _sample_orders(sample_order, space_map.input_dimensions)
    sampled_map = SampledSpaceMap.on_uniform_grid(space_map, orders=orders)
    nodes = tuple(np.linspace(-1.0, 1.0, order + 1) for order in orders)
    reference_grid = np.meshgrid(*nodes, indexing="ij")

    kform = KForm(specs)
    kform.values[:] = np.asarray(values)
    reference_components = tuple(
        np.asarray(
            reconstruct(
                DegreesOfFreedom(
                    specs.get_component_function_space(component),
                    kform.get_component_dofs(component),
                ),
                *reference_grid,
            )
        )
        for component in range(specs.component_count)
    )
    if specs.order == 0:
        return sampled_map, reference_components
    transformed = transform_kform_to_target_sampled(
        specs.order, sampled_map, reference_components
    )
    return sampled_map, tuple(np.asarray(component) for component in transformed)


def sample_domain(
    domain: HypercubeDomain, sample_order: int | Sequence[int]
) -> npt.NDArray[np.double]:
    """Sample a hypercube domain on a uniform reference-space grid.

    Parameters
    ----------
    domain : HypercubeDomain
        Domain whose coordinate map is sampled.
    sample_order : int or sequence of int
        Uniform sampling order, either for every reference axis or per axis.

    Returns
    -------
    array
        Physical positions with shape ``(n_0 + 1, ..., n_N + 1, ndim_physical)``.
    """
    orders = _sample_orders(sample_order, domain.ndim_reference)
    integration = IntegrationSpace(
        *(IntegrationSpecs(order, IntegrationMethod.GAUSS) for order in orders)
    )
    sampled_map = SampledSpaceMap.on_uniform_grid(domain(integration), orders=orders)
    return np.asarray(sampled_map.positions)


_LAGRANGE_CELL_TYPES: dict[int, pv.CellType] = {
    1: pv.CellType.LAGRANGE_CURVE,
    2: pv.CellType.LAGRANGE_QUADRILATERAL,
    3: pv.CellType.LAGRANGE_HEXAHEDRON,
}


def lagrange_geometry_cells(
    geometry: MeshGeometry,
    order: int | Sequence[int] | None = None,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.uint8],
    npt.NDArray[np.double],
    npt.NDArray[np.int32],
]:
    """Assemble raw VTK cell arrays from a batched element geometry.

    The cell family follows the reference dimensions of ``geometry``: one
    gives Lagrange curves, two Lagrange quadrilaterals, and three Lagrange
    hexahedra.

    Parameters
    ----------
    geometry : MeshGeometry
        Geometry of every element of a mesh. All elements share one pair of
        ``input_dimensions`` and ``output_dimensions``. The reference
        dimensions must be 1, 2, or 3 and the physical dimensions must not
        exceed 3.
    order : int or sequence of int, optional
        Uniform sampling order of every cell: a scalar applies to all
        reference axes, a sequence supplies one order per reference axis.
        If omitted, each cell is sampled at the orders of its own
        integration rule, ``geometry.space_map(i).integration_space.orders``,
        which may be anisotropic and may differ between cells. Orders must
        be positive.

    Returns
    -------
    cells : (N,) intp ndarray
        VTK legacy connectivity: per cell its point count followed by its
        local point indices, concatenated into one array.
    celltypes : (n_cells,) uint8 ndarray
        One cell type per element, chosen from the reference dimensions:
        ``LAGRANGE_CURVE``, ``LAGRANGE_QUADRILATERAL``, or
        ``LAGRANGE_HEXAHEDRON``.
    points : (total_points, 3) double ndarray
        Physical cell points, stored in C-order tensor-product layout per
        cell and padded with zeros to three coordinates.
    degrees : (n_cells, input_dimensions) int32 ndarray
        Per-cell sampling orders, shaped for the VTK ``HigherOrderDegrees``
        cell data.

    Raises
    ------
    ValueError
        If ``geometry`` has reference dimensions other than 1, 2, and 3,
        more than three physical dimensions, an explicit ``order`` with the
        wrong number of entries or a non-positive order, or a cell whose
        integration-rule orders are not positive.

    Notes
    -----
    Points stay in C-order tensor-product layout; the connectivity maps
    VTK's vertices-edges-faces-body point order onto them with the cached
    permutations of :mod:`fdg.domains`. For a curve that permutation is not
    the identity: VTK puts the vertices at parametric coordinates 0 and 1
    before the points of the curve interior, so C order along the single
    axis is not the Lagrange curve order. An empty ``geometry`` returns
    empty arrays with the shapes described above instead of raising.
    """
    cell_type = _LAGRANGE_CELL_TYPES.get(geometry.input_dimensions)
    if cell_type is None:
        raise ValueError(
            "Lagrange cells require one, two, or three reference dimensions, "
            f"got {geometry.input_dimensions}."
        )
    output_dimensions = geometry.output_dimensions
    if output_dimensions > 3:
        raise ValueError(
            "Lagrange cells require at most three physical coordinates, "
            f"got {output_dimensions}."
        )
    uniform_orders: tuple[int, ...] | None = None
    if order is not None:
        uniform_orders = _lagrange_orders(order, geometry.input_dimensions)

    cells_parts: list[npt.NDArray[np.intp]] = []
    points_parts: list[npt.NDArray[np.double]] = []
    degrees_rows: list[tuple[int, ...]] = []
    point_offset = 0
    for element in range(geometry.element_count):
        space_map = geometry.space_map(element)
        if uniform_orders is None:
            orders = _lagrange_orders(
                space_map.integration_space.orders, geometry.input_dimensions
            )
        else:
            orders = uniform_orders
        sampled_map = SampledSpaceMap.on_uniform_grid(space_map, orders=orders)
        positions = np.asarray(sampled_map.positions)
        point_count = int(np.prod(np.asarray(orders, dtype=np.intp) + 1))
        if output_dimensions == 3:
            points_parts.append(positions.reshape(point_count, 3))
        else:
            padded = np.zeros((point_count, 3), dtype=np.double)
            padded[:, :output_dimensions] = positions.reshape(
                point_count, output_dimensions
            )
            points_parts.append(padded)
        local_indices = np.empty(point_count, dtype=np.intp)
        if geometry.input_dimensions == 1:
            vtk_indices = _vtk_1d_indices(orders[0])
        elif geometry.input_dimensions == 2:
            vtk_indices = _vtk_2d_indices(*orders)
        else:
            vtk_indices = _vtk_3d_indices(*orders)
        local_indices[vtk_indices.astype(np.intp, copy=False)] = np.arange(
            point_offset, point_offset + point_count, dtype=np.intp
        )
        cell = np.empty(point_count + 1, dtype=np.intp)
        cell[0] = point_count
        cell[1:] = local_indices
        cells_parts.append(cell)
        degrees_rows.append(orders)
        point_offset += point_count

    cells = np.concatenate(cells_parts) if cells_parts else np.empty(0, dtype=np.intp)
    celltypes = np.full(len(cells_parts), cell_type, dtype=np.uint8)
    if points_parts:
        points = np.concatenate(points_parts, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.double)
    degrees = np.asarray(degrees_rows, dtype=np.int32).reshape(
        geometry.element_count, geometry.input_dimensions
    )
    return cells, celltypes, points, degrees


def lagrange_quadrilateral_grid(
    space_maps: Sequence[SpaceMap],
    order: int,
    point_data: Mapping[str, Sequence[npt.ArrayLike]] | None = None,
) -> pv.UnstructuredGrid:
    """Build one VTK Lagrange quadrilateral per sampled element map.

    Parameters
    ----------
    space_maps : sequence of SpaceMap
        Non-empty sequence of element maps with exactly two reference
        dimensions. Every map must share them and must produce either two or
        three physical coordinates.
    order : int
        Positive polynomial order used for uniform tensor-product sampling in
        both reference directions.
    point_data : mapping[str, sequence of array-like], optional
        Named scalar arrays to attach as point data. The sequence for each
        name must have one item per map, and each item must have shape
        ``(order + 1, order + 1)`` in C-order reference-grid layout.

    Returns
    -------
    pyvista.UnstructuredGrid
        An unstructured grid containing one
        ``LAGRANGE_QUADRILATERAL`` cell per map. Points and point data are
        reordered from C-order tensor-product layout into VTK's
        vertices-edges-interior layout. Two-coordinate maps are embedded in
        the ``z = 0`` plane.

    Raises
    ------
    ValueError
        If ``order`` is not positive, ``space_maps`` is empty, the maps do
        not all have the same dimensions, a map does not have two reference
        dimensions, a map does not produce two or three physical coordinates,
        a point-data sequence has the wrong length, or an array has the wrong
        tensor-grid shape.

    Notes
    -----
    Sampling is performed directly on each map. This is preferable to
    slicing a high-order three-dimensional VTK cell when a visualization
    plane is known in advance, because the resulting cell topology remains
    an explicit high-order quadrilateral. Because the dimensions are shared
    by all maps, a list mixing two- and three-coordinate maps is rejected;
    only maps that agree on both dimensions are accepted. Unlike
    :func:`lagrange_hexahedral_grid`, no ``HigherOrderDegrees`` cell data is
    stored.
    """
    if order < 1:
        raise ValueError("The Lagrange order must be positive.")
    if point_data is None:
        point_data = {}
    if any(len(values) != len(space_maps) for values in point_data.values()):
        raise ValueError("Every point-data sequence must match the number of elements.")
    if len(space_maps) == 0:
        raise ValueError("At least one space map is required.")
    geometry = MeshGeometry.from_elements(*space_maps)
    if geometry.input_dimensions != 2:
        raise ValueError("Lagrange quadrilaterals require two reference dimensions.")
    if geometry.output_dimensions not in (2, 3):
        raise ValueError("Lagrange quadrilaterals require 2D or 3D coordinates.")
    cells, celltypes, points, _ = lagrange_geometry_cells(geometry, order)
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    expected_shape = (order + 1, order + 1)
    for name, values in point_data.items():
        sampled_values = []
        for value in values:
            data = np.asarray(value)
            if data.shape != expected_shape:
                raise ValueError(
                    f"Point data {name!r} has shape {data.shape}, expected "
                    f"{expected_shape}."
                )
            sampled_values.append(data.ravel())
        grid.point_data[name] = np.concatenate(sampled_values)
    return grid


def lagrange_hexahedral_grid(
    space_maps: Sequence[SpaceMap],
    order: int | Sequence[int],
    point_data: Mapping[str, Sequence[npt.ArrayLike]] | None = None,
) -> pv.UnstructuredGrid:
    """Build VTK Lagrange hexahedra from sampled element maps.

    Parameters
    ----------
    space_maps : sequence of SpaceMap
        Non-empty sequence of element maps used to generate the cell points.
        Every map must have three reference and three physical dimensions.
    order : int or sequence of int
        Lagrange sampling order. A scalar applies to all three reference axes;
        a sequence supplies ``(order_x, order_y, order_z)`` independently.
        Orders must be positive.
    point_data : mapping of str to sequence of array_like, optional
        Per-element arrays sampled on the same tensor grids as the cells. For
        orders ``(o0, o1, o2)``, each array must have shape
        ``(o0 + 1, o1 + 1, o2 + 1)``.

    Returns
    -------
    pyvista.UnstructuredGrid
        High-order Lagrange-hexahedron cells with optional point data.

    Raises
    ------
    ValueError
        If an order is not positive, ``space_maps`` is empty, the maps do not
        all have the same dimensions, a map does not have three reference
        dimensions, a map does not produce three physical coordinates, a
        point-data sequence has the wrong length, or an array has the wrong
        tensor-grid shape.

    Notes
    -----
    Each element owns its points. Keeping elements separate avoids assuming
    that independently sampled curved maps have bitwise-identical shared
    points. Tensor-product points and point data use C order before VTK
    vertices-edges-faces-body reordering. The cell-data attribute
    ``HigherOrderDegrees`` is populated so direction-dependent orders survive
    VTK serialization. A zero order used to be accepted here and produced a
    one-point cell that VTK cannot meaningfully interpolate; it is rejected
    now.
    """
    orders = _lagrange_orders(order, 3)
    if point_data is None:
        point_data = {}
    if any(len(values) != len(space_maps) for values in point_data.values()):
        raise ValueError("Every point-data sequence must match the number of elements.")
    if len(space_maps) == 0:
        raise ValueError("At least one space map is required.")
    geometry = MeshGeometry.from_elements(*space_maps)
    if geometry.input_dimensions != 3:
        raise ValueError("Lagrange hexahedra require three reference dimensions.")
    if geometry.output_dimensions != 3:
        raise ValueError("Lagrange hexahedra require three physical coordinates.")
    cells, celltypes, points, degrees = lagrange_geometry_cells(geometry, order)
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    expected_shape = tuple(component + 1 for component in orders)
    for name, values in point_data.items():
        sampled_values = []
        for value in values:
            data = np.asarray(value)
            if data.shape != expected_shape:
                raise ValueError(
                    f"Point data {name!r} has shape {data.shape}, expected "
                    f"{expected_shape}."
                )
            sampled_values.append(data.ravel())
        grid.point_data[name] = np.concatenate(sampled_values)
    grid.cell_data["HigherOrderDegrees"] = degrees
    grid.GetCellData().SetHigherOrderDegrees(
        grid.GetCellData().GetArray("HigherOrderDegrees")
    )
    return grid
