"""Tests for finite-element visualization helpers."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    SpaceMap,
)
from fdg.visualization import lagrange_quadrilateral_grid


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
