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
    SampledSpaceMap,
    SpaceMap,
    transform_kform_to_target_sampled,
)
from fdg.degrees_of_freedom import reconstruct
from fdg.domains import HypercubeDomain, _vtk_2d_indices, _vtk_3d_indices
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


def lagrange_quadrilateral_grid(
    space_maps: Sequence[SpaceMap],
    order: int,
    point_data: Mapping[str, Sequence[npt.ArrayLike]] | None = None,
) -> pv.UnstructuredGrid:
    """Build one VTK Lagrange quadrilateral per sampled element map.

    Parameters
    ----------
    space_maps : sequence of SpaceMap
        Element maps with exactly two reference dimensions. Their physical
        output may have two or three coordinates.
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
        If ``order`` is not positive, a map does not have two reference
        dimensions, a map does not produce two or three physical coordinates,
        a point-data sequence has the wrong length, or an array has the wrong
        tensor-grid shape.

    Notes
    -----
    Sampling is performed directly on each map. This is preferable to
    slicing a high-order three-dimensional VTK cell when a visualization
    plane is known in advance, because the resulting cell topology remains
    an explicit high-order quadrilateral.
    """
    if order < 1:
        raise ValueError("The Lagrange order must be positive.")
    if point_data is None:
        point_data = {}
    if any(len(values) != len(space_maps) for values in point_data.values()):
        raise ValueError("Every point-data sequence must match the number of elements.")

    points: list[npt.NDArray[np.double]] = []
    cells: list[npt.NDArray[np.intp]] = []
    sampled_data: dict[str, list[npt.NDArray[np.double]]] = {
        name: [] for name in point_data
    }
    point_offset = 0
    point_count = (order + 1) ** 2
    vtk_indices = _vtk_2d_indices(order, order).astype(np.intp, copy=False)

    for element, space_map in enumerate(space_maps):
        if space_map.input_dimensions != 2:
            raise ValueError("Lagrange quadrilaterals require two reference dimensions.")
        sampled_map = SampledSpaceMap.on_uniform_grid(space_map, orders=(order, order))
        positions = np.asarray(sampled_map.positions)
        if positions.shape[-1] not in (2, 3):
            raise ValueError("Lagrange quadrilaterals require 2D or 3D coordinates.")
        if positions.shape[-1] == 2:
            points.append(
                np.column_stack(
                    [
                        positions[..., 0].ravel(),
                        positions[..., 1].ravel(),
                        np.zeros(point_count),
                    ]
                )
            )
        else:
            points.append(
                np.stack([positions[..., axis].ravel() for axis in range(3)], axis=1)
            )
        for name, values in point_data.items():
            data = np.asarray(values[element])
            if data.shape != positions.shape[:-1]:
                raise ValueError(
                    f"Point data {name!r} has shape {data.shape}, expected "
                    f"{positions.shape[:-1]}."
                )
            sampled_data[name].append(data.ravel())
        vtk_order = np.empty(point_count, dtype=np.intp)
        vtk_order[vtk_indices] = np.arange(
            point_offset, point_offset + point_count, dtype=np.intp
        )
        cells.append(np.concatenate((np.array([point_count], dtype=np.intp), vtk_order)))
        point_offset += point_count

    grid = pv.UnstructuredGrid(
        np.concatenate(cells),
        np.full(len(cells), pv.CellType.LAGRANGE_QUADRILATERAL, dtype=np.uint8),
        np.concatenate(points, axis=0),
    )
    for name, values in sampled_data.items():
        grid.point_data[name] = np.concatenate(values)
    return grid


def lagrange_hexahedral_grid(
    space_maps: Sequence[SpaceMap],
    order: int,
    point_data: Mapping[str, Sequence[npt.ArrayLike]] | None = None,
) -> pv.UnstructuredGrid:
    """Build VTK Lagrange cells from sampled element maps.

    Parameters
    ----------
    space_maps : sequence of SpaceMap
        Element maps used to generate the cell points.
    order : int
        Lagrange sampling order along every reference axis.
    point_data : mapping of str to sequence of array_like, optional
        Per-element arrays sampled on the same tensor grids as the cells. Each
        array must have shape ``(order + 1, order + 1, order + 1)``.

    Returns
    -------
    pyvista.UnstructuredGrid
        High-order Lagrange-hexahedron cells with optional point data.

    Notes
    -----
    Each element owns its points. Keeping elements separate avoids assuming that
    independently sampled curved maps have bitwise-identical shared points.
    Tensor-product points and point data use C order throughout.
    """
    if order < 0:
        raise ValueError("The Lagrange order must be non-negative.")
    if point_data is None:
        point_data = {}
    if any(len(values) != len(space_maps) for values in point_data.values()):
        raise ValueError("Every point-data sequence must match the number of elements.")

    points: list[npt.NDArray[np.double]] = []
    cells: list[npt.NDArray[np.intp]] = []
    sampled_data: dict[str, list[npt.NDArray[np.double]]] = {
        name: [] for name in point_data
    }
    point_offset = 0
    point_count = (order + 1) ** 3
    vtk_indices = _vtk_3d_indices(order, order, order).astype(np.intp, copy=False)

    for element, space_map in enumerate(space_maps):
        sampled_map = SampledSpaceMap.on_uniform_grid(
            space_map, orders=(order, order, order)
        )
        positions = np.asarray(sampled_map.positions)
        if positions.shape[-1] != 3:
            raise ValueError("Lagrange hexahedra require three physical coordinates.")
        points.append(
            np.stack([positions[..., axis].ravel() for axis in range(3)], axis=1)
        )
        for name, values in point_data.items():
            data = np.asarray(values[element])
            if data.shape != positions.shape[:-1]:
                raise ValueError(
                    f"Point data {name!r} has shape {data.shape}, expected "
                    f"{positions.shape[:-1]}."
                )
            sampled_data[name].append(data.ravel())
        vtk_order = np.empty(point_count, dtype=np.intp)
        vtk_order[vtk_indices] = np.arange(
            point_offset, point_offset + point_count, dtype=np.intp
        )
        cells.append(np.concatenate((np.array([point_count], dtype=np.intp), vtk_order)))
        point_offset += point_count

    grid = pv.UnstructuredGrid(
        np.concatenate(cells),
        np.full(len(cells), pv.CellType.LAGRANGE_HEXAHEDRON, dtype=np.uint8),
        np.concatenate(points, axis=0),
    )
    for name, values in sampled_data.items():
        grid.point_data[name] = np.concatenate(values)
    return grid
