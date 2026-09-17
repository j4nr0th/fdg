"""Tests for the image-sequence mixed Poisson example."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from examples.image_sequence_poisson import (
    image_domain_lengths,
    load_image_stack,
    make_manufactured_functions,
)


def test_load_image_stack_normalizes_and_flips_images(tmp_path) -> None:
    """Image order, grayscale conversion, normalization, and y orientation are stable."""
    rgb = np.asarray(
        [
            [[255, 0, 0], [0, 0, 255]],
            [[0, 255, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(rgb, mode="RGB").save(tmp_path / "01.png")
    Image.fromarray(np.full((2, 2), 64, dtype=np.uint8), mode="L").save(
        tmp_path / "02.png"
    )
    (tmp_path / "ignored.txt").write_text("not an image")

    stack = load_image_stack(tmp_path)

    expected_first = np.asarray([[150.0, 255.0], [76.0, 29.0]], dtype=np.double) / 255.0
    np.testing.assert_allclose(stack, [expected_first, np.full((2, 2), 64 / 255)])
    assert stack.dtype == np.double
    assert float(stack.min()) >= 0.0
    assert float(stack.max()) <= 1.0


def test_cell_average_reconstruction_preserves_pixel_centers() -> None:
    """Cell centers retain square-pixel spacing and their average values."""
    samples = np.arange(3 * 2 * 4, dtype=np.double).reshape((3, 2, 4))
    lengths = image_domain_lengths(samples)
    assert lengths == (1.0, 0.5, 2.0)
    solution, _ = make_manufactured_functions(samples)
    x = (np.arange(4, dtype=np.double) + 0.5) / 4.0
    y = (np.arange(2, dtype=np.double) + 0.5) / 4.0
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    for z_index in range(3):
        values = solution(x_grid, y_grid, np.full_like(x_grid, z_index))
        np.testing.assert_allclose(values.T, samples[z_index])


def test_load_image_stack_rejects_missing_or_mismatched_images(tmp_path) -> None:
    """Missing files and inconsistent raster dimensions fail explicitly."""
    with pytest.raises(ValueError, match="No supported images"):
        load_image_stack(tmp_path)

    Image.fromarray(np.zeros((3, 4), dtype=np.uint8), mode="L").save(tmp_path / "01.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(tmp_path / "02.png")
    with pytest.raises(ValueError, match="expected"):
        load_image_stack(tmp_path)


# def test_image_sequence_cli_exports_high_order_solution(tmp_path, capsys) -> None:
#     """The CLI solves a constant stack and exports unambiguous VTK cells."""
#     input_directory = tmp_path / "images"
#     input_directory.mkdir()
#     for index in range(3):
#         Image.fromarray(np.full((2, 4), 128, dtype=np.uint8), mode="L").save(
#             input_directory / f"{index:02d}.png"
#         )
#     output = tmp_path / "solution.vtu"

#     main(["1", "1", "1", "2", "3", str(input_directory), "--output", str(output)])

#     progress_output = capsys.readouterr().out
#     for label in (
#         "Load images",
#         "Source derivatives",
#         "Element maps",
#         "Face test spaces",
#         "Element matrices",
#         "Source projections",
#         "Weak Dirichlet loads",
#         "Continuity constraints",
#         "Constraint condensation",
#         "Constraint solve",
#         "Back substitution",
#         "Solution postprocess",
#         "VTK sampling",
#         "VTK export",
#     ):
#         assert label in progress_output
#     assert "\r" in progress_output

#     grid = pv.read(output)
#     assert grid.n_cells == 2
#     assert all(
#         cell_type == pv.CellType.LAGRANGE_HEXAHEDRON for cell_type in grid.celltypes
#     )
#     assert [grid.GetCell(0).GetOrder(axis) for axis in range(3)] == [3, 3, 3]
#     np.testing.assert_allclose(grid.bounds, (0.0, 1.0, 0.0, 0.5, 0.0, 2.0),
#     atol=1.0e-12)
#     for name in ("u", "manufactured", "abs_error"):
#         assert name in grid.point_data
#         assert np.isfinite(grid.point_data[name]).all()
#     np.testing.assert_allclose(grid.point_data["manufactured"], 128.0 / 255.0)
#     assert float(np.min(grid.point_data["u"])) >= -1.0e-10
#     assert float(np.max(grid.point_data["u"])) <= 1.0 + 1.0e-10
