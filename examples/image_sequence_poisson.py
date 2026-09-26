"""Solve a mixed Poisson problem manufactured from an image sequence.

The input images are ordered z planes. Each pixel is treated as a square
physical cell whose grayscale value is its cell average; the plane dimensions
therefore retain the image aspect ratio. The cell averages are reconstructed
on a tensor grid, and the resulting three-dimensional field defines both the
manufactured solution and a finite-difference Laplacian source.

Usage
-----

``python examples/image_sequence_poisson.py Nx Ny px py pz INPUT_DIR``

The computed solution is exported to ``image_sequence_poisson.vtu`` unless an
alternative path is supplied with ``--output``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    Mesh,
    SampledSpaceMap,
    SpaceMap,
    compute_kform_boundary_load,
    compute_kform_mass_matrix,
    incidence_kform_operator,
    packed_kform_constraints_to_csr,
    projection_kform_l2_dual,
    reconstruct,
    transform_kform_to_target,
    transform_kform_to_target_sampled,
)
from fdg.visualization import lagrange_hexahedral_grid
from PIL import Image
from scipy.ndimage import map_coordinates, spline_filter

Cells = tuple[int, int, int]
Orders = tuple[int, int, int]
FieldFunction = Callable[
    [npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]],
    npt.NDArray[np.double],
]
SUPPORTED_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
)
REFERENCE_MIN = -1.0
REFERENCE_MAX = 1.0
INTEGRATION_EXTRA_ORDER = 4


class Progress:
    """Render a live spinner and progress bar for one computation stage."""

    _bar_width = 24
    _spinner = ("|", "/", "-", "\\")

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.current = 0
        self._spinner_index = 0
        self._stop = Event()
        self._lock = Lock()
        self._thread = Thread(target=self._animate, daemon=True)
        self._render()
        self._thread.start()

    def _render(self) -> None:
        """Render the current stage state in place."""
        with self._lock:
            fraction = min(max(self.current / self.total, 0.0), 1.0)
            filled = int(round(self._bar_width * fraction))
            bar = "=" * filled + "." * (self._bar_width - filled)
            spinner = self._spinner[self._spinner_index % len(self._spinner)]
            print(
                f"{self.label:<24} [{bar}] {self.current:>4}/{self.total:<4} {spinner}",
                end="\r",
                flush=True,
            )
            self._spinner_index += 1

    def _animate(self) -> None:
        """Advance the spinner while a stage is running."""
        while not self._stop.wait(0.1):
            self._render()

    def update(self, current: int | None = None) -> None:
        """Set or increment completed work and redraw the stage."""
        if current is None:
            self.current += 1
        else:
            self.current = int(current)
        self._render()

    def finish(self) -> None:
        """Stop animation, render a completed bar, and terminate its line."""
        self._stop.set()
        self._thread.join()
        self.current = self.total
        self._render()
        print(flush=True)


def _positive_int(text: str) -> int:
    """Parse a strictly positive command-line integer."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def load_image_stack(directory: Path) -> npt.NDArray[np.double]:
    """Load ordered grayscale image planes as normalized cell averages.

    Parameters
    ----------
    directory : pathlib.Path
        Input directory. Regular files with common raster-image suffixes are
        sorted lexicographically and interpreted as increasing z planes.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_images, n_y_pixels, n_x_pixels)`` with ``float64``
        values in ``[0, 1]``. Each value is the average brightness of a
        square pixel cell. The first array row is the physical lower-y row;
        image columns retain their left-to-right x orientation.

    Raises
    ------
    ValueError
        If the directory is invalid, no supported images are found, an image
        cannot be decoded, or image dimensions differ.
    """
    if not directory.is_dir():
        raise ValueError(f"Input path is not a directory: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found in {directory}")

    progress = Progress("Load images", len(paths))

    frames: list[npt.NDArray[np.double]] = []
    image_shape: tuple[int, int] | None = None
    for path in paths:
        try:
            with Image.open(path) as image:
                grayscale = np.asarray(image.convert("L"), dtype=np.double) / 255.0
        except OSError as error:
            progress.finish()
            raise ValueError(f"Could not read image {path}: {error}") from error
        if grayscale.ndim != 2:
            progress.finish()
            raise ValueError(f"Image {path} did not produce a two-dimensional array.")
        if image_shape is None:
            image_shape = grayscale.shape
        elif grayscale.shape != image_shape:
            progress.finish()
            raise ValueError(
                f"Image {path} has shape {grayscale.shape}; expected {image_shape}."
            )
        # Image row zero is the visual top; physical y increases upward.
        frames.append(np.flipud(grayscale))
        progress.update()

    progress.finish()
    return np.clip(np.stack(frames, axis=0), 0.0, 1.0)


def image_domain_lengths(
    samples: npt.NDArray[np.double],
) -> tuple[float, float, float]:
    """Return physical ``(x, y, z)`` lengths for an image stack.

    The pixel cells are square, so the longest image side has length one.
    Consecutive image planes are separated by one unit in z.
    """
    n_images, n_y, n_x = samples.shape
    longest_side = float(max(n_x, n_y))
    return n_x / longest_side, n_y / longest_side, float(n_images - 1)


def _cell_average_axes(
    samples: npt.NDArray[np.double],
) -> tuple[tuple[npt.NDArray[np.double], ...], tuple[float, float, float]]:
    """Return reconstruction axes and domain lengths for cell-average data."""
    n_images, n_y, n_x = samples.shape
    lengths = image_domain_lengths(samples)
    x_edges = np.linspace(0.0, lengths[0], n_x + 1)
    y_edges = np.linspace(0.0, lengths[1], n_y + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    axes = (
        np.arange(n_images, dtype=np.double),
        np.concatenate(([0.0], y_centers, [lengths[1]]), dtype=np.double),
        np.concatenate(([0.0], x_centers, [lengths[0]]), dtype=np.double),
    )
    return axes, lengths


def _interpolator(
    samples: npt.NDArray[np.double], lengths: tuple[float, float, float]
) -> FieldFunction:
    """Create a memory-bounded B-spline callable for cell-average samples.

    The image values are stored at pixel centers. ``map_coordinates`` performs
    separable local interpolation and therefore avoids the global sparse spline
    coefficient solve used by a cubic ``RegularGridInterpolator`` on a large
    three-dimensional image stack.
    """
    values = np.asarray(samples, dtype=np.double)
    spline_order = min(3, *(size - 1 for size in values.shape))
    coefficients = (
        spline_filter(values, order=spline_order, mode="nearest")
        if spline_order > 1
        else values
    )
    n_images, n_y, n_x = values.shape
    x_length, y_length, _ = lengths

    def evaluate(
        x: npt.NDArray[np.double],
        y: npt.NDArray[np.double],
        z: npt.NDArray[np.double],
    ) -> npt.NDArray[np.double]:
        """Evaluate the cell-center spline on broadcast physical coordinates."""
        x_array, y_array, z_array = np.broadcast_arrays(
            np.asarray(x, dtype=np.double),
            np.asarray(y, dtype=np.double),
            np.asarray(z, dtype=np.double),
        )
        coordinates = np.stack(
            (
                np.clip(z_array.ravel(), 0.0, n_images - 1.0),
                np.clip(y_array.ravel() / y_length * n_y - 0.5, 0.0, n_y - 1.0),
                np.clip(x_array.ravel() / x_length * n_x - 0.5, 0.0, n_x - 1.0),
            ),
            axis=0,
        )
        return np.asarray(
            map_coordinates(
                coefficients,
                coordinates,
                order=spline_order,
                mode="nearest",
                prefilter=False,
            ),
            dtype=np.double,
        ).reshape(x_array.shape)

    return evaluate


def _second_difference(
    values: npt.NDArray[np.double], axis: int, coordinates: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
    """Approximate one coordinate's second derivative on cell-center samples."""
    if values.shape[axis] < 2:
        return np.zeros_like(values)
    edge_order = min(2, values.shape[axis] - 1)
    first = np.gradient(values, coordinates, axis=axis, edge_order=edge_order)  # type: ignore
    return np.gradient(first, coordinates, axis=axis, edge_order=edge_order)  # type: ignore


def make_manufactured_functions(
    samples: npt.NDArray[np.double],
) -> tuple[FieldFunction, FieldFunction]:
    """Build local-spline image-field and point-sampled Laplacian callables.

    Parameters
    ----------
    samples : numpy.ndarray
        Cell-average image stack with shape ``(n_images, n_y_pixels, n_x_pixels)``.

    Returns
    -------
    tuple of callable
        The manufactured solution and its image-derived Laplacian. Both
        callables accept ``(x, y, z)`` arrays and return broadcast-shaped
        values. Pixel values are reconstructed at their cell centers with a
        separable cubic B-spline when each axis has enough support.
    """
    axes, lengths = _cell_average_axes(samples)
    z_coordinates = axes[0]
    y_coordinates = axes[1][1:-1]
    x_coordinates = axes[2][1:-1]
    progress = Progress("Source derivatives", 3)
    source_samples = np.zeros_like(samples)
    for axis, coordinates in enumerate((z_coordinates, y_coordinates, x_coordinates)):
        source_samples += _second_difference(samples, axis, coordinates)
        progress.update()
    progress.finish()
    return _interpolator(samples, lengths), _interpolator(source_samples, lengths)


def element_indices(cells: Cells) -> list[tuple[int, int, int]]:
    """Return structured element indices in tensor-product order."""
    return list(product(*(range(count) for count in cells)))  # type: ignore


def grid_point(index: tuple[int, int, int], cells: Cells) -> int:
    """Return a point ID in the structured tensor-product vertex grid."""
    nx, ny, _ = cells
    strides = (1, nx + 1, (nx + 1) * (ny + 1))
    return sum(axis_index * stride for axis_index, stride in zip(index, strides))


def mesh_corners(cells: Cells) -> npt.NDArray[np.uint64]:
    """Return eight corner IDs for every structured hexahedral element."""
    corners: list[int] = []
    for index in element_indices(cells):
        for local_corner in range(2**3):
            corners.append(
                grid_point(  # type: ignore
                    tuple(  # type: ignore
                        index[axis] + ((local_corner >> axis) & 1)  # type: ignore
                        for axis in range(3)  # type: ignore
                    ),  # type: ignore
                    cells,
                )  # type: ignore
            )
    return np.asarray(corners, dtype=np.uint64)


def make_element_maps(
    cells: Cells, lengths: tuple[float, float, float], integration: IntegrationSpace
) -> list[SpaceMap]:
    """Create affine maps covering the physical image-aspect domain."""
    geometry_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(3))
    )
    local_nodes = np.meshgrid(*([np.asarray((-1.0, 1.0))] * 3), indexing="ij")
    cell_array = np.asarray(cells, dtype=np.double)
    lower_domain = np.zeros(3)
    domain_width = np.asarray(lengths, dtype=np.double)
    maps: list[SpaceMap] = []
    indices = element_indices(cells)
    progress = Progress("Element maps", len(indices))
    for index in indices:
        lower = lower_domain + domain_width * np.asarray(index) / cell_array
        widths = domain_width / cell_array
        coordinates = [
            lower[axis] + 0.5 * widths[axis] * (local_nodes[axis] + 1.0)
            for axis in range(3)
        ]
        # Optimization point: these constant-affine maps could bypass the
        # general per-quadrature-point SpaceMap Jacobian work.
        maps.append(
            SpaceMap(
                *(
                    CoordinateMap(
                        DegreesOfFreedom(geometry_space, coordinate.ravel()),
                        integration,
                    )
                    for coordinate in coordinates
                )
            )
        )
        progress.update()
    progress.finish()
    return maps


def face_test_specs(
    mesh: Mesh, orders: Orders, cells: Cells
) -> tuple[list[KFormSpecs], list[list[list[KFormSpecs]]]]:
    """Build axis-specific 2-form tests for every mesh face."""
    face_tests: list[KFormSpecs] = []
    progress = Progress("Face test spaces", 3)
    for normal_axis in range(3):
        tangent_axes = tuple(axis for axis in range(3) if axis != normal_axis)
        face_space = FunctionSpace(
            *(BasisSpecs(BasisType.BERNSTEIN, orders[axis]) for axis in tangent_axes)
        )
        face_tests.append(KFormSpecs(2, face_space))
        progress.update()

    progress.finish()
    object_counts = (
        mesh.point_count,
        int(mesh.collections[0].shape[0]),
        int(mesh.collections[1].shape[0]),
    )
    test_specs: list[list[list[KFormSpecs]]] = [
        [[] for _ in range(count)] for count in object_counts
    ]
    for element_id in range(mesh.element_count):
        for normal_axis in range(3):
            for side in (-1, 1):
                face_id = mesh.element_object(element_id, side * (normal_axis + 1))
                test_specs[2][face_id] = [face_tests[normal_axis]]
    expected_faces = (
        (cells[0] + 1) * cells[1] * cells[2]
        + cells[0] * (cells[1] + 1) * cells[2]
        + cells[0] * cells[1] * (cells[2] + 1)
    )
    if len(mesh.iterate_boundary(2)) + len(mesh.iterate_shared(2)) != expected_faces:
        raise RuntimeError("Structured mesh face enumeration is inconsistent.")
    return face_tests, test_specs


def solve(
    mesh: Mesh,
    maps: list[SpaceMap],
    specs_q: KFormSpecs,
    specs_u: KFormSpecs,
    face_tests: list[KFormSpecs],
    source: FieldFunction,
    solution: FieldFunction,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float]:
    """Solve the anisotropic mixed Poisson system by local condensation.

    All elements have the same axis-aligned affine Jacobian, so the local
    mixed matrix is assembled and factorized once. Continuity constraints are
    condensed to a sparse multiplier system instead of forming one global
    mixed matrix and retaining a dense matrix for every element.
    """
    element_count = mesh.element_count
    nq = int(np.sum(specs_q.component_dof_counts))
    nu = int(np.sum(specs_u.component_dof_counts))
    q_total = element_count * nq
    u_total = element_count * nu
    local_size = nq + nu

    matrix_progress = Progress("Element matrices", 1)
    q_mass = np.asarray(
        compute_kform_mass_matrix(
            maps[0], specs_q.order, specs_q.base_space, specs_q.base_space
        )
    )
    u_mass = np.asarray(
        compute_kform_mass_matrix(
            maps[0], specs_u.order, specs_u.base_space, specs_u.base_space
        )
    )
    derivative = np.asarray(incidence_kform_operator(specs_q, u_mass, right=True))
    derivative_transpose = np.asarray(
        incidence_kform_operator(specs_q, u_mass, transpose=True)
    )
    local_mixed = np.block(
        [
            [q_mass, derivative_transpose],
            [derivative, np.zeros((nu, nu), dtype=np.double)],
        ]
    )
    local_factor = scipy.linalg.lu_factor(local_mixed, check_finite=False)
    matrix_progress.update()
    matrix_progress.finish()
    del q_mass, u_mass, derivative, derivative_transpose, local_mixed

    rhs_q = np.zeros(q_total)
    rhs_u = np.zeros(u_total)
    source_progress = Progress("Source projections", element_count)
    for element_id, element_map in enumerate(maps):
        rhs_u[element_id * nu : (element_id + 1) * nu] = np.asarray(
            projection_kform_l2_dual([source], specs_u, element_map)[0]
        ).reshape(-1)
        source_progress.update()
    source_progress.finish()

    boundary_faces = mesh.iterate_boundary(2)
    boundary_progress = Progress("Weak Dirichlet loads", len(boundary_faces))
    for _, boundary_id, element_ids, orientations in boundary_faces:
        element_id = int(element_ids[0])
        normal_axis = abs(int(orientations[0, 0])) - 1
        rhs_q[element_id * nq : (element_id + 1) * nq] += compute_kform_boundary_load(
            face_tests[normal_axis],
            specs_q,
            maps[element_id],
            mesh.collections,
            mesh.point_count,
            element_id,
            int(boundary_id),
            [solution],
            surface_measure=False,
        )
        boundary_progress.update()
    boundary_progress.finish()

    continuity_progress = Progress("Continuity constraints", 1)
    packed, constraint_rhs = mesh.compute_kform_global_constraints(
        [specs_q] * element_count,
        maps,
    )
    constraint_data, constraint_columns, constraint_offsets = (
        packed_kform_constraints_to_csr(packed, specs_q, element_count)
    )
    constraints = scipy.sparse.csr_matrix(
        (constraint_data, constraint_columns, constraint_offsets),
        shape=(constraint_offsets.size - 1, element_count * nq),
    )
    constraints.sum_duplicates()
    continuity_progress.update()
    continuity_progress.finish()

    coo = constraints.tocoo(copy=False)
    element_numbers = coo.col // nq
    sorted_indices = np.argsort(element_numbers, kind="stable")
    sorted_elements = element_numbers[sorted_indices]

    def constraint_block(
        element_id: int,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.double]] | None:
        """Build the local constraint block for one element on demand."""
        start = int(np.searchsorted(sorted_elements, element_id, side="left"))
        end = int(np.searchsorted(sorted_elements, element_id, side="right"))
        if start == end:
            return None
        selected = sorted_indices[start:end]
        rows, row_inverse = np.unique(coo.row[selected], return_inverse=True)
        block = np.zeros((rows.size, nq), dtype=np.double)
        local_columns = coo.col[selected] - element_id * nq
        np.add.at(block, (row_inverse, local_columns), coo.data[selected])
        return rows.astype(np.intp, copy=False), block

    constraint_count = constraints.shape[0]
    condensed_rhs = -np.asarray(constraint_rhs, dtype=np.double).copy()
    schur_rows: list[np.ndarray] = []
    schur_columns: list[np.ndarray] = []
    schur_values: list[np.ndarray] = []
    condensation_progress = Progress("Constraint condensation", element_count)
    for element_id in range(element_count):
        local_rhs = np.concatenate(
            (
                rhs_q[element_id * nq : (element_id + 1) * nq],
                rhs_u[element_id * nu : (element_id + 1) * nu],
            )
        )
        local_solution = scipy.linalg.lu_solve(
            local_factor, local_rhs, check_finite=False
        )
        block_data = constraint_block(element_id)
        if block_data is not None:
            rows, block = block_data
            condensed_rhs[rows] += block @ local_solution[:nq]
            lifted = np.zeros((local_size, rows.size), dtype=np.double)
            lifted[:nq] = block.T
            response = scipy.linalg.lu_solve(local_factor, lifted, check_finite=False)
            local_schur = block @ response[:nq]
            schur_rows.append(np.repeat(rows, rows.size))
            schur_columns.append(np.tile(rows, rows.size))
            schur_values.append(local_schur.ravel())
        condensation_progress.update()
    condensation_progress.finish()

    multipliers = np.zeros(constraint_count, dtype=np.double)
    if constraint_count:
        schur = scipy.sparse.coo_matrix(
            (
                np.concatenate(schur_values),
                (np.concatenate(schur_rows), np.concatenate(schur_columns)),
            ),
            shape=(constraint_count, constraint_count),
        ).tocsr()
        schur.sum_duplicates()
        constraint_solve_progress = Progress("Constraint solve", 1)
        multipliers = scipy.sparse.linalg.splu(schur.tocsc()).solve(condensed_rhs)
        constraint_solve_progress.update()
        constraint_solve_progress.finish()

    q_dofs: list[np.ndarray] = []
    u_dofs: list[np.ndarray] = []
    back_substitution_progress = Progress("Back substitution", element_count)
    for element_id in range(element_count):
        local_rhs = np.concatenate(
            (
                rhs_q[element_id * nq : (element_id + 1) * nq],
                rhs_u[element_id * nu : (element_id + 1) * nu],
            )
        )
        block_data = constraint_block(element_id)
        if block_data is not None:
            rows, block = block_data
            local_rhs[:nq] -= block.T @ multipliers[rows]
        local_solution = scipy.linalg.lu_solve(
            local_factor, local_rhs, check_finite=False
        )
        q_dofs.append(local_solution[:nq])
        u_dofs.append(local_solution[nq:])
        back_substitution_progress.update()
    back_substitution_progress.finish()

    q_values = np.concatenate(q_dofs)
    continuity_residual = float(
        np.max(np.abs(constraints @ q_values - constraint_rhs), initial=0.0)
    )
    error_squared = 0.0
    postprocess_progress = Progress("Solution postprocess", element_count)
    for element_id, element_map in enumerate(maps):
        u_component = DegreesOfFreedom(
            specs_u.get_component_function_space(0), u_dofs[element_id]
        )
        reference_values = u_component.reconstruct_at_integration_points(
            element_map.integration_space
        )
        physical_values = transform_kform_to_target(
            specs_u.order, element_map, [reference_values]
        )[0]
        coordinates = tuple(
            np.asarray(element_map.coordinate_map(axis).values) for axis in range(3)
        )
        error_squared += np.sum(
            (physical_values - solution(*coordinates)) ** 2
            * np.abs(element_map.determinant)
            * element_map.integration_space.weights()
        )
        postprocess_progress.update()
    postprocess_progress.finish()
    return q_dofs, u_dofs, continuity_residual, float(np.sqrt(error_squared))


def sample_solution(
    maps: list[SpaceMap],
    specs_u: KFormSpecs,
    u_dofs: list[np.ndarray],
    orders: Orders,
    solution: FieldFunction,
) -> pv.UnstructuredGrid:
    """Sample solved and manufactured n-forms and build VTK cells.

    ParaView does not preserve direction-dependent hexahedral orders when it
    extracts a surface. Use one isotropic visualization order so the exported
    cell remains unambiguous through that pipeline, while the finite-element
    solve itself retains the requested per-axis orders.
    """
    solution_samples: list[npt.NDArray[np.double]] = []
    manufactured_samples: list[npt.NDArray[np.double]] = []
    error_samples: list[npt.NDArray[np.double]] = []
    export_order = max(orders)
    export_orders = (export_order,) * 3
    vtk_progress = Progress("VTK sampling", len(maps))
    for element_map, element_dofs in zip(maps, u_dofs, strict=True):
        nodes = np.meshgrid(
            *(
                np.linspace(REFERENCE_MIN, REFERENCE_MAX, export_order + 1)
                for _ in range(3)
            ),
            indexing="ij",
        )
        sampled_map = SampledSpaceMap.on_uniform_grid(element_map, orders=export_orders)
        u_component = DegreesOfFreedom(
            specs_u.get_component_function_space(0), element_dofs
        )
        reference_values = np.asarray(reconstruct(u_component, *nodes))
        physical_values = np.asarray(
            transform_kform_to_target_sampled(
                specs_u.order, sampled_map, [reference_values]
            )[0]
        )
        coordinates = tuple(
            np.asarray(sampled_map.positions[..., axis], dtype=np.double)
            for axis in range(3)
        )
        manufactured = np.asarray(solution(*coordinates))
        solution_samples.append(physical_values)
        manufactured_samples.append(manufactured)
        error_samples.append(np.abs(physical_values - manufactured))
        vtk_progress.update()
    vtk_progress.finish()
    return lagrange_hexahedral_grid(
        maps,
        export_orders,
        {
            "u": solution_samples,
            "manufactured": manufactured_samples,
            "abs_error": error_samples,
        },
    )


def run(
    nx: int, ny: int, px: int, py: int, pz: int, input_directory: Path, output: Path
) -> None:
    """Load images, solve the mixed problem, and export its VTK solution."""
    samples = load_image_stack(input_directory)
    image_count = samples.shape[0]
    cells = (nx, ny, image_count - 1)
    lengths = image_domain_lengths(samples)
    orders = (px, py, pz)
    integration_order = max(orders) + INTEGRATION_EXTRA_ORDER
    integration = IntegrationSpace(
        *(IntegrationSpecs(integration_order, IntegrationMethod.GAUSS) for _ in range(3))
    )
    mesh = Mesh.from_corners(3, mesh_corners(cells))
    maps = make_element_maps(cells, lengths, integration)
    solution, source = make_manufactured_functions(samples)
    del samples
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.BERNSTEIN, order) for order in orders)
    )
    specs_q = KFormSpecs(2, base_space)
    specs_u = KFormSpecs(3, base_space)
    face_tests, _ = face_test_specs(mesh, orders, cells)
    _, u_dofs, continuity_residual, error = solve(
        mesh,
        maps,
        specs_q,
        specs_u,
        face_tests,
        source,
        solution,
    )
    grid = sample_solution(maps, specs_u, u_dofs, orders, solution)
    print(f"images: {image_count}")
    vtk_export_progress = Progress("VTK export", 1)
    grid.save(str(output))
    vtk_export_progress.finish()
    print(f"orders: {orders[0]}x{orders[1]}x{orders[2]}")
    print(f"continuity residual: {continuity_residual:.6e}")
    print(f"L2 error: {error:.6e}")
    print(f"output: {output}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the image-sequence solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nx", type=_positive_int, help="number of x elements")
    parser.add_argument("ny", type=_positive_int, help="number of y elements")
    parser.add_argument("px", type=_positive_int, help="x polynomial order")
    parser.add_argument("py", type=_positive_int, help="y polynomial order")
    parser.add_argument("pz", type=_positive_int, help="z polynomial order")
    parser.add_argument("input_directory", type=Path, help="ordered image directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image_sequence_poisson.vtu"),
        help="output VTK file (default: image_sequence_poisson.vtu)",
    )
    arguments = parser.parse_args(argv)
    if not arguments.input_directory.is_dir():
        parser.error(f"Input path is not a directory: {arguments.input_directory}")
    if not arguments.output.parent.is_dir():
        parser.error(f"Output parent is not a directory: {arguments.output.parent}")
    try:
        run(
            arguments.nx,
            arguments.ny,
            arguments.px,
            arguments.py,
            arguments.pz,
            arguments.input_directory,
            arguments.output,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
