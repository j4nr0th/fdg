r"""
.. currentmodule:: fdg

Direct continuity for a multi-element Laplace solve.
=====================================================

The previous example made a field continuous by *equations*: it compared the
physical traces of neighboring elements and let ``solve_hybridized`` carry the
multipliers. :meth:`Mesh.compute_kform_direct_dof_map` offers the complementary
route. It walks the same shared objects, but instead of contrasting two
element-local fields it solves for one *global* field. Every local degree of
freedom is mapped onto the global unknowns of the object it belongs to, so
continuity holds by construction and no continuity rows are needed at all.

The map is purely topological and never looks at coordinates, so it works for
curved elements as well; the geometry enters only through the stiffness and
mass matrices scattered through the transfer. Global unknowns are numbered per
shared object, from the highest-dimensional faces down to points, followed by
the element-private degrees of freedom.

Both formulations run below on the same mesh, maps, and polynomial order. The
hybridized path spends one unknown per element-local degree of freedom plus one
multiplier per constraint row, the direct path one unknown per global degree of
freedom. The printed tables compare those counts and the two physical
:math:`L^2` errors under p- and h-refinement, and the closing plot shows the
p-refinement rate of both formulations. The two systems are not compared entry
by entry: the direct path keeps the physical trace equations implicit in the
numbering, the hybridized path states them.
"""  # noqa: D205 D400

from __future__ import annotations

from itertools import product
from time import perf_counter

import numpy as np
import numpy.typing as npt
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    DirectDofMap,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    Mesh,
    MeshGeometry,
    MeshKFormSpecs,
    SpaceMap,
    laplace_stiffness,
    solve_hybridized,
)
from fdg.integration import projection_l2_dual
from matplotlib import pyplot as plt

PackedRows = tuple[
    npt.NDArray[np.uintp],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uintp],
    npt.NDArray[np.double],
]

DEFORMATION = 0.2
GEO_ORDER = 2


# %%
# The shared problem
# ------------------
#
# A structured partition with ``cells`` cells per axis. The corner IDs come from
# one global ``(cells + 1) x ...`` point lattice, so neighboring elements share
# point IDs and the mesh derives every face, edge, and vertex from them.


def point_id(index: tuple[int, ...], cells: int) -> int:
    """Return the point ID of one index tuple in a structured lattice."""
    return sum(value * (cells + 1) ** axis for axis, value in enumerate(index))


def mesh_corners(ndim: int, cells: int) -> npt.NDArray[np.uint64]:
    """Return the corner IDs of a structured mesh with ``cells`` cells per axis."""
    corners: list[int] = []
    for element_index in product(range(cells), repeat=ndim):
        for local_corner in range(2**ndim):
            lattice_index = tuple(
                element_index[axis] + ((local_corner >> axis) & 1) for axis in range(ndim)
            )
            corners.append(point_id(lattice_index, cells))
    return np.asarray(corners, dtype=np.uint64)


def make_mesh(ndim: int, cells: int) -> Mesh:
    """Build the structured mesh."""
    return Mesh.from_corners(ndim, mesh_corners(ndim, cells))


def _deformed_coordinates(
    *coordinates: npt.NDArray[np.double],
) -> tuple[npt.NDArray[np.double], ...]:
    """Deform global coordinates while preserving the outer boundary."""
    bump = np.ones_like(coordinates[0])
    for coordinate in coordinates:
        bump *= 1.0 - coordinate**2
    return tuple(coordinate + DEFORMATION * bump for coordinate in coordinates)


def deformed_element_coordinates(
    ndim: int,
    cells: int,
) -> list[tuple[npt.NDArray[np.double], ...]]:
    """Return the geometry nodes of every element of the deformed grid.

    The reference cell of element ``e`` is the interval of width ``2 / cells``
    centered on ``2 * (e - (cells - 1) / 2) / cells``, so the elements together
    cover ``[-1, 1] ** ndim`` whatever the cell count is.
    """
    nodes = np.linspace(-1.0, 1.0, GEO_ORDER + 1)
    grid = np.meshgrid(*([nodes] * ndim), indexing="ij")
    return [
        _deformed_coordinates(
            *(
                (grid[axis] + 2.0 * element_index[axis] - (cells - 1)) / cells
                for axis in range(ndim)
            )
        )
        for element_index in product(range(cells), repeat=ndim)
    ]


def geometry_space(ndim: int) -> FunctionSpace:
    """Return the polynomial space the element geometry is represented in."""
    return FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER) for _ in range(ndim))
    )


def make_element_maps(ndim: int, integration_order: int, cells: int) -> list[SpaceMap]:
    """Build matching curved maps from reference cells to the physical grid."""
    space = geometry_space(ndim)
    integration = IntegrationSpace(
        *(
            IntegrationSpecs(integration_order, IntegrationMethod.GAUSS)
            for _ in range(ndim)
        )
    )
    return [
        SpaceMap(
            *(
                CoordinateMap(DegreesOfFreedom(space, coordinate.ravel()), integration)
                for coordinate in element_coordinates
            )
        )
        for element_coordinates in deformed_element_coordinates(ndim, cells)
    ]


def manufactured_solution(*coordinates: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Return a smooth mixed-parity solution with zero normal derivative."""
    return (
        1.0
        + 0.75 * len(coordinates)
        + np.sum(
            [
                0.5 * np.cos(np.pi * coordinate) + 0.25 * np.sin(0.5 * np.pi * coordinate)
                for coordinate in coordinates
            ],
            axis=0,
        )
    )


def manufactured_source(*coordinates: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Return the manufactured source ``-Delta(u)``."""
    return np.sum(
        [
            0.5 * np.pi**2 * np.cos(np.pi * coordinate)
            + (np.pi**2 / 16.0) * np.sin(0.5 * np.pi * coordinate)
            for coordinate in coordinates
        ],
        axis=0,
    )


def physical_error(
    solution: npt.NDArray[np.double],
    maps: list[SpaceMap],
    element_specs: list[KFormSpecs],
) -> float:
    """Return the :math:`L^2` error of an element-major solution vector."""
    error_squared = 0.0
    base_space = element_specs[0].base_space
    dofs_per_element = int(np.sum(element_specs[0].component_dof_counts))
    for element_id, element_map in enumerate(maps):
        offset = element_id * dofs_per_element
        reference_values = DegreesOfFreedom(
            base_space, solution[offset : offset + dofs_per_element]
        ).reconstruct_at_integration_points(element_map.integration_space)
        ndim = element_map.input_dimensions
        coordinates = [element_map.coordinate_map(axis).values for axis in range(ndim)]
        exact = manufactured_solution(*coordinates)
        error_squared += np.sum(
            (reference_values - exact) ** 2
            * np.abs(element_map.determinant)
            * element_map.integration_space.weights()
        )
    return float(np.sqrt(error_squared))


# %%
# Hybridized solve
# ----------------
#
# The path of the previous example, kept so both run on the same mesh.


def build_continuity_rows(
    mesh: Mesh,
    maps: list[SpaceMap],
    element_specs: list[KFormSpecs],
) -> PackedRows:
    """Assemble hierarchical continuity rows through the public mesh method."""
    return mesh.compute_kform_continuity_constraints(element_specs, maps)


def solve_hybridized_continuity(
    ndim: int,
    order: int,
    cells: int,
) -> tuple[float, int, int, int]:
    """Solve the problem with continuity and Dirichlet rows.

    Returns the physical :math:`L^2` error, the unknowns of the augmented
    system, the continuity rows, and the constraint rows.
    """
    mesh = make_mesh(ndim, cells)
    maps = make_element_maps(ndim, order + 4, cells)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    continuity = build_continuity_rows(mesh, maps, element_specs)

    dofs_per_element = int(np.sum(element_specs[0].component_dof_counts))
    total_dofs = len(maps) * dofs_per_element
    rhs = np.zeros(total_dofs)
    for element_id, element_map in enumerate(maps):
        offset = element_id * dofs_per_element
        rhs[offset : offset + dofs_per_element] = projection_l2_dual(
            manufactured_source, base_space, element_map
        ).values.flatten()

    boundary_conditions = {
        int(object_id): manufactured_solution
        for _, object_id, _, _ in mesh.iterate_boundary(ndim - 1)
    }
    constraints, constraint_rhs = mesh.compute_kform_global_constraints(
        element_specs, maps, boundary_conditions
    )

    geometry = MeshGeometry.from_elements(*maps)
    structure = MeshKFormSpecs.from_space(ndim, [("u", 0)], base_space, len(maps))
    result = solve_hybridized(
        geometry, structure, rhs, constraints, constraint_rhs, laplace_stiffness
    )
    if result.constraint_residual > 1.0e-10:
        raise RuntimeError(
            f"hybridized trace residual is too large: {result.constraint_residual:.3e}"
        )

    error = physical_error(np.concatenate(result.element_dofs), maps, element_specs)
    return (
        error,
        total_dofs + constraints[0].size - 1,
        continuity[0].size - 1,
        constraints[0].size - 1,
    )


# %%
# Direct transfer
# ---------------
#
# The second path starts from ``Mesh.compute_kform_direct_dof_map``, which
# describes a sparse element-to-global transfer: ``entry_offsets`` slices it into
# one block per element-local degree of freedom, ``entry_index`` names the global
# unknown of every entry, ``entry_value`` is its coefficient. One local degree of
# freedom may reach several global unknowns, which is what lets a higher-order
# trace project onto the common space. ``element_offsets`` splits the
# element-local numbering into one block per element; ``element_interior_offsets``
# counts the element-private degrees of freedom, which the map places behind
# every shared object rather than inside ``element_offsets``.


def element_entries(
    transfer: DirectDofMap,
    element_id: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.double]]:
    """Return the local DoF, global DoF, and coefficient of one element."""
    first = int(transfer.element_offsets[element_id])
    last = int(transfer.element_offsets[element_id + 1])
    counts = np.diff(transfer.entry_offsets[first : last + 1])
    local_dofs = np.repeat(np.arange(last - first, dtype=np.intp), counts)
    start = int(transfer.entry_offsets[first])
    stop = int(transfer.entry_offsets[last])
    return (
        local_dofs,
        np.asarray(transfer.entry_index[start:stop], dtype=np.intp),
        np.asarray(transfer.entry_value[start:stop], dtype=np.double),
    )


def physical_node_coordinates(
    ndim: int,
    order: int,
    cells: int,
) -> list[tuple[npt.NDArray[np.double], ...]]:
    """Return the physical coordinates of every element's field basis nodes.

    The field basis is nodal on the Gauss--Lobatto points, so the quadrature of
    the same rule evaluates the element geometry exactly where the field degrees
    of freedom live.
    """
    space = geometry_space(ndim)
    nodes = IntegrationSpace(
        *(IntegrationSpecs(order, IntegrationMethod.GAUSS_LOBATTO) for _ in range(ndim))
    )
    return [
        tuple(
            DegreesOfFreedom(space, coordinate.ravel()).reconstruct_at_integration_points(
                nodes
            )
            for coordinate in element_coordinates
        )
        for element_coordinates in deformed_element_coordinates(ndim, cells)
    ]


def boundary_global_dofs(
    transfer: DirectDofMap,
    coordinates: list[tuple[npt.NDArray[np.double], ...]],
) -> npt.NDArray[np.intp]:
    """Return the global DoFs that stand for a point of the outer boundary.

    The transfer is topological and does not say which unknowns lie on the
    outer boundary; the geometry does, because the deformation vanishes there.
    """
    boundary = np.zeros(transfer.global_dof_count, dtype=bool)
    for element_id, element_coordinates in enumerate(coordinates):
        local_dofs, global_dofs, _ = element_entries(transfer, element_id)
        points = np.stack(
            [coordinate.ravel() for coordinate in element_coordinates], axis=-1
        )[local_dofs]
        boundary[global_dofs] = np.any(np.abs(np.abs(points) - 1.0) < 1.0e-12, axis=-1)
    return np.flatnonzero(boundary)


def prescribed_values(
    transfer: DirectDofMap,
    coordinates: list[tuple[npt.NDArray[np.double], ...]],
    global_dofs: npt.NDArray[np.intp],
) -> npt.NDArray[np.double]:
    """Return the manufactured solution at the given global DoFs.

    Every element reaching a global DoF evaluates the solution at the same
    physical point, since the element maps agree on the shared objects. The
    values are averaged away the last floating-point difference between two
    elements describing one node.
    """
    totals = np.zeros(transfer.global_dof_count)
    counts = np.zeros(transfer.global_dof_count)
    for element_id, element_coordinates in enumerate(coordinates):
        local_dofs, global_indices, _ = element_entries(transfer, element_id)
        exact = manufactured_solution(*element_coordinates).ravel()[local_dofs]
        np.add.at(totals, global_indices, exact)
        np.add.at(counts, global_indices, 1.0)
    return totals[global_dofs] / counts[global_dofs]


def assemble_global_laplace(
    transfer: DirectDofMap,
    maps: list[SpaceMap],
    element_specs: list[KFormSpecs],
    base_space: FunctionSpace,
) -> tuple[np.ndarray, npt.NDArray[np.double]]:
    """Scatter every element operator and load vector onto the global unknowns."""
    size = transfer.global_dof_count
    matrix = np.zeros((size, size))
    rhs = np.zeros(size)
    for element_id, element_map in enumerate(maps):
        local_dofs, global_dofs, values = element_entries(transfer, element_id)
        stiffness = laplace_stiffness(
            element_id, [element_specs[element_id]], element_map
        )
        load = projection_l2_dual(
            manufactured_source, base_space, element_map
        ).values.flatten()
        # One entry of the scatter is ``T_i * M[i, j] * T_j``.
        np.add.at(
            matrix,
            (global_dofs[:, None], global_dofs[None, :]),
            values[:, None]
            * stiffness[local_dofs[:, None], local_dofs[None, :]]
            * values[None, :],
        )
        np.add.at(rhs, global_dofs, values * load[local_dofs])
    return matrix, rhs


def element_solution(
    transfer: DirectDofMap,
    solution: npt.NDArray[np.double],
    element_count: int,
) -> npt.NDArray[np.double]:
    """Pull one global solution back into the element-major layout.

    The inverse transfer is the transpose of the forward one.
    """
    local = np.zeros(transfer.element_dof_count)
    for element_id in range(element_count):
        local_dofs, global_dofs, coefficients = element_entries(transfer, element_id)
        block = slice(
            int(transfer.element_offsets[element_id]),
            int(transfer.element_offsets[element_id + 1]),
        )
        np.add.at(local[block], local_dofs, coefficients * solution[global_dofs])
    return local


def solve_direct_continuity(
    ndim: int,
    order: int,
    cells: int,
) -> tuple[float, int, np.ndarray]:
    """Assemble and solve the global system of the direct formulation.

    Returns the physical :math:`L^2` error, the number of global unknowns, and
    the assembled system matrix.
    """
    mesh = make_mesh(ndim, cells)
    maps = make_element_maps(ndim, order + 4, cells)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]

    transfer = mesh.compute_kform_direct_dof_map(element_specs)
    matrix, rhs = assemble_global_laplace(transfer, maps, element_specs, base_space)

    coordinates = physical_node_coordinates(ndim, order, cells)
    boundary = boundary_global_dofs(transfer, coordinates)
    solution = np.zeros(transfer.global_dof_count)
    solution[boundary] = prescribed_values(transfer, coordinates, boundary)

    # Eliminating the Dirichlet rows and columns leaves a system on the free unknowns.
    free = np.setdiff1d(np.arange(transfer.global_dof_count), boundary)
    reduced = matrix[np.ix_(free, free)]
    reduced_rhs = rhs[free] - matrix[np.ix_(free, boundary)] @ solution[boundary]
    solution[free] = np.linalg.solve(reduced, reduced_rhs)

    local = element_solution(transfer, solution, len(maps))
    return physical_error(local, maps, element_specs), transfer.global_dof_count, matrix


# %%
# Comparison
# ----------
#
# The direct system is symmetric by construction, since a symmetric element
# matrix stays symmetric under a transfer applied in both of its indices.


def compare(ndim: int, order: int, cells: int) -> tuple[float, float, int, int]:
    """Solve both formulations once, validate the direct system, and report.

    Returns the direct and the hybridized :math:`L^2` error and the number of
    unknowns of each system.
    """
    started = perf_counter()
    hybrid_error, hybrid_unknowns, continuity_rows, rows = solve_hybridized_continuity(
        ndim, order, cells
    )
    hybrid_elapsed = perf_counter() - started

    started = perf_counter()
    direct_error, direct_unknowns, matrix = solve_direct_continuity(ndim, order, cells)
    direct_elapsed = perf_counter() - started

    asymmetry = np.max(np.abs(matrix - matrix.T))
    if asymmetry > 1.0e-10 * np.max(np.abs(matrix)):
        raise RuntimeError(f"the direct system is not symmetric: {asymmetry:.3e}")
    if not np.isfinite(direct_error):
        raise RuntimeError("the direct error is not finite")
    print(
        f"{ndim}D, {cells} cell{'s' if cells > 1 else ''} per axis, p={order}: "
        f"direct unknowns={direct_unknowns}, hybridized unknowns={hybrid_unknowns} "
        f"({hybrid_unknowns / direct_unknowns:.2f}x), rows={continuity_rows} continuity "
        f"+ {rows - continuity_rows} boundary, "
        f"direct L2={direct_error:.6e}, hybridized L2={hybrid_error:.6e}, "
        f"direct={direct_elapsed:.2f}s, hybridized={hybrid_elapsed:.2f}s",
        flush=True,
    )
    return direct_error, hybrid_error, direct_unknowns, hybrid_unknowns


def _decreasing(errors: list[float]) -> bool:
    """Return whether every error is strictly smaller than the one before it."""
    return all(later < earlier for earlier, later in zip(errors, errors[1:]))


def main() -> None:
    """Report p- and h-refinement of both continuity formulations."""
    order_sweeps = {2: (1, 2, 3, 4, 5), 3: (1, 2, 3)}
    convergence: dict[int, tuple[tuple[int, ...], list[float], list[float]]] = {}
    for ndim, orders in order_sweeps.items():
        print(f"p-refinement, {ndim}D curved mesh, {2**ndim} elements:", flush=True)
        direct_errors: list[float] = []
        hybrid_errors: list[float] = []
        for order in orders:
            direct_error, hybrid_error, _, _ = compare(ndim, order, 2)
            direct_errors.append(direct_error)
            hybrid_errors.append(hybrid_error)
        if not _decreasing(direct_errors):
            raise RuntimeError("the direct L2 error did not decrease under p-refinement")
        print(
            "  direct L2 errors: " + ", ".join(f"{error:.6e}" for error in direct_errors),
            flush=True,
        )
        convergence[ndim] = (orders, direct_errors, hybrid_errors)

    print("h-refinement, 2D curved mesh, p=2:", flush=True)
    refinement_errors: list[float] = []
    for cells in (1, 2, 3, 4, 5):
        direct_error, _, _, _ = compare(2, 2, cells)
        refinement_errors.append(direct_error)
    if not _decreasing(refinement_errors):
        raise RuntimeError("the direct L2 error did not decrease under h-refinement")

    fig, axis = plt.subplots()
    for ndim, (orders, direct_errors, hybrid_errors) in convergence.items():
        axis.semilogy(orders, direct_errors, marker="o", label=f"{ndim}D direct")
        axis.semilogy(
            orders, hybrid_errors, marker="x", linestyle="--", label=f"{ndim}D hybridized"
        )
    axis.set(
        xlabel="polynomial order p",
        ylabel=r"$\|u_h - u\|_{L^2}$",
        title="Direct and hybridized continuity, same mesh",
    )
    axis.grid(True, which="both")
    axis.legend()
    fig.tight_layout()
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
