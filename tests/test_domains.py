"""Tests for geometric domain helpers."""

import itertools

import numpy as np
import pytest
from fdg import Hypercube, Quad


def _curved_boundary_points() -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate compatible curved faces of a three-dimensional domain."""
    nodes = np.linspace(-1.0, +1.0, 7)

    def mapping(a: np.ndarray, b: np.ndarray, c: np.ndarray):
        return (
            a + 0.15 * (1 - a**2) * np.sin(np.pi * b / 2) * np.sin(np.pi * c / 2),
            b + 0.12 * (1 - b**2) * np.sin(np.pi * a / 2) * np.sin(np.pi * c / 2),
            c + 0.10 * (1 - c**2) * np.sin(np.pi * a / 2) * np.sin(np.pi * b / 2),
        )

    boundary_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for boundary_axis in range(3):
        pair: list[np.ndarray] = []
        for side in (-1.0, +1.0):
            grid = np.meshgrid(
                *(nodes for axis in range(3) if axis != boundary_axis),
                indexing="ij",
            )
            reference: list[np.ndarray] = []
            boundary_axis_index = 0
            for axis in range(3):
                if axis == boundary_axis:
                    reference.append(np.full_like(grid[0], side))
                else:
                    reference.append(grid[boundary_axis_index])
                    boundary_axis_index += 1
            pair.append(np.stack(mapping(*reference), axis=-1))
        boundary_pairs.append((pair[0], pair[1]))
    return boundary_pairs


def test_hypercube_from_corners_uses_mesh_corner_order() -> None:
    """Corner IDs use axis zero as the least-significant bit."""
    corners = [
        (float(x), float(y), float(z), float(w))
        for w, z, y, x in itertools.product((0, 1), repeat=4)
    ]
    domain = Hypercube.from_corners(*corners)

    points = np.meshgrid(*([np.array([-1.0, 0.0, 1.0])] * 4), indexing="ij")
    sampled = domain.sample(*points)
    for coordinate, reference in zip(sampled, points, strict=True):
        assert coordinate == pytest.approx((reference + 1) / 2)
    assert domain.compute_size() == pytest.approx(1.0)


def test_hypercube_from_boundary_pairs() -> None:
    """Opposite quadrilateral faces assemble a three-dimensional cube."""
    x_start = Quad.from_corners((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))
    x_end = Quad.from_corners((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))
    y_start = Quad.from_corners((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))
    y_end = Quad.from_corners((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))
    z_start = Quad.from_corners((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    z_end = Quad.from_corners((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))

    domain = Hypercube((x_start, x_end), (y_start, y_end), (z_start, z_end))
    points = np.meshgrid(*([np.array([-1.0, 0.0, 1.0])] * 3), indexing="ij")
    sampled = domain.sample(*points)
    for coordinate, reference in zip(sampled, points, strict=True):
        assert coordinate == pytest.approx((reference + 1) / 2)


def test_hypercube_from_curved_boundary_points() -> None:
    """Fitted maps preserve compatible curved boundary traces."""
    boundary_points = _curved_boundary_points()
    domain = Hypercube.from_boundary_points(*boundary_points)
    nodes = np.linspace(-1.0, +1.0, 7)
    grid = np.meshgrid(nodes, nodes, indexing="ij")

    for boundary_axis, pair in enumerate(boundary_points):
        for end, expected in enumerate(pair):
            actual = domain.boundary(boundary_axis, bool(end)).sample(*grid)
            for coordinate_index, coordinate in enumerate(actual):
                np.testing.assert_allclose(
                    coordinate, expected[..., coordinate_index], atol=1e-12
                )

    face = domain.boundary(0)
    face_y, face_z = np.meshgrid(nodes, nodes, indexing="ij")
    curved_y = face.sample(face_y, face_z)[1]
    assert np.max(np.abs(curved_y - face_y)) > 0.05


def test_hypercube_from_boundary_points_supports_four_dimensions() -> None:
    """Tensor-product boundary points generalize beyond three dimensions."""
    nodes = np.linspace(-1.0, +1.0, 3)
    boundary_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for boundary_axis in range(4):
        pair: list[np.ndarray] = []
        for side in (-1.0, +1.0):
            grid = np.meshgrid(
                *(nodes for axis in range(4) if axis != boundary_axis),
                indexing="ij",
            )
            reference: list[np.ndarray] = []
            boundary_axis_index = 0
            for axis in range(4):
                if axis == boundary_axis:
                    reference.append(np.full_like(grid[0], side))
                else:
                    reference.append(grid[boundary_axis_index])
                    boundary_axis_index += 1
            pair.append(np.stack(reference, axis=-1))
        boundary_pairs.append((pair[0], pair[1]))

    domain = Hypercube.from_boundary_points(*boundary_pairs)
    points = tuple(np.meshgrid(*([nodes] * 4), indexing="ij"))
    sampled = domain.sample(*points)
    for index in range(4):
        np.testing.assert_allclose(sampled[index], points[index], atol=1e-12)


def test_hypercube_rejects_inconsistent_curved_boundaries() -> None:
    """Neighboring fitted boundaries must agree on their shared edge."""
    boundary_points = _curved_boundary_points()
    broken = [(start.copy(), end.copy()) for start, end in boundary_points]
    broken[1][0][0, :, 0] += 0.1

    with pytest.raises(ValueError, match="intersections"):
        Hypercube.from_boundary_points(*broken)


def test_hypercube_rejects_non_hypercube_corner_count() -> None:
    """Corner construction requires exactly a power-of-two number of corners."""
    with pytest.raises(ValueError, match="power of two"):
        Hypercube.from_corners((0, 0), (1, 0), (0, 1))
