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
from fdg.domains import HypercubeDomain, _vtk_3d_indices
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
