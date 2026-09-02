r"""
.. currentmodule:: fdg

Hierarchical continuity for a direct multi-element Laplace solve.
==================================================================

This example assembles the scalar primal Laplace operator on conforming
quadrilateral, hexahedral, and four-dimensional hypercube meshes. Shared
strata are processed from faces to edges to points. Only consecutive element
pairs are compared for each shared object, so the continuity rows form a
forest rather than a cycle.

Every element map is a matching curved restriction of one globally deformed
coordinate map. The deformation vanishes on the outer boundary and agrees on
all internal interfaces, so geometry is not used to identify topology.

The prototype deliberately keeps the test spaces explicit. On a shared
object, the default scalar trace test degree is the minimum element degree
minus two in every tangential direction. Thus degree-one traces have no
interior rows; their edge and face boundary values are connected when the
lower-dimensional strata are processed. The reported zero-form residual is
the maximum residual of these physical trace equations after the solve.
"""  # noqa: D205 D400

from __future__ import annotations

from itertools import combinations, product
from time import perf_counter

import numpy as np
import numpy.typing as npt
import scipy.linalg
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
    SpaceMap,
    compute_kform_boundary_constraints,
    compute_kform_incidence_matrix,
    compute_kform_mass_matrix,
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


def grid_point(*index: int) -> int:
    """Return the point ID in a tensor grid with three points per axis."""
    return sum(value * 3**axis for axis, value in enumerate(index))


def mesh_corners(ndim: int) -> npt.NDArray[np.uint64]:
    """Return corners of a two-by-two tensor-product mesh."""
    corners: list[int] = []
    for element_index in product(range(2), repeat=ndim):
        for local_corner in range(2**ndim):
            point_index = tuple(
                element_index[axis] + ((local_corner >> axis) & 1) for axis in range(ndim)
            )
            corners.append(grid_point(*point_index))
    return np.asarray(corners, dtype=np.uint64)


def make_mesh(ndim: int) -> Mesh:
    """Build the two-by-two (or two-by-two-by-two) mesh."""
    return Mesh.from_corners(ndim, mesh_corners(ndim))


def _deformed_coordinates(
    *coordinates: npt.NDArray[np.double],
) -> tuple[npt.NDArray[np.double], ...]:
    """Deform global coordinates while preserving the outer boundary."""
    bump = np.ones_like(coordinates[0])
    for coordinate in coordinates:
        bump *= 1.0 - coordinate**2
    return tuple(coordinate + DEFORMATION * bump for coordinate in coordinates)


def make_element_maps(ndim: int, integration_order: int) -> list[SpaceMap]:
    """Build matching curved maps from reference cells to the physical grid."""
    geometry_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, GEO_ORDER) for _ in range(ndim))
    )
    geometry_nodes = np.linspace(-1.0, 1.0, GEO_ORDER + 1)
    geometry_grid = np.meshgrid(*([geometry_nodes] * ndim), indexing="ij")
    integration = IntegrationSpace(
        *(
            IntegrationSpecs(integration_order, IntegrationMethod.GAUSS)
            for _ in range(ndim)
        )
    )
    maps: list[SpaceMap] = []
    for element_index in product(range(2), repeat=ndim):
        reference_coordinates = tuple(
            0.5 * geometry_grid[axis] + element_index[axis] - 0.5 for axis in range(ndim)
        )
        coordinates = _deformed_coordinates(*reference_coordinates)
        maps.append(
            SpaceMap(
                *(
                    CoordinateMap(
                        DegreesOfFreedom(geometry_space, coordinate.ravel()), integration
                    )
                    for coordinate in coordinates
                )
            )
        )
    return maps


def _object_count(mesh: Mesh, mdim: int) -> int:
    """Return the number of mesh objects of one dimension."""
    if mdim == 0:
        return mesh.point_count
    return int(mesh.collections[mdim - 1].shape[0])


def _mapped_orders(
    element_specs: list[KFormSpecs],
    mdim: int,
    element_ids: npt.NDArray[np.uint64],
    orientations: npt.NDArray[np.int8],
) -> tuple[int, ...]:
    """Return minimum element orders in the object's canonical axes."""
    ndim = element_specs[0].dimension
    fixed_count = ndim - mdim
    result: list[int] = []
    for canonical_axis in range(mdim):
        orders = [
            element_specs[int(element_id)].base_space.orders[
                abs(int(orientations[row, fixed_count + canonical_axis])) - 1
            ]
            for row, element_id in enumerate(element_ids)
        ]
        result.append(min(orders))
    return tuple(result)


def make_test_specs(
    mesh: Mesh,
    element_specs: list[KFormSpecs],
    form_order: int,
    basis_type: BasisType,
) -> list[list[list[KFormSpecs]]]:
    """Build explicit per-object trace tests for the scalar prototype."""
    ndim = mesh.ndim
    result: list[list[list[KFormSpecs]]] = []
    shared_by_dimension: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    for mdim in range(ndim):
        shared_by_dimension.append(
            {
                int(object_id): (element_ids, orientations)
                for _, object_id, element_ids, orientations in mesh.iterate_shared(mdim)
            }
        )

    for mdim in range(ndim):
        objects: list[list[KFormSpecs]] = []
        for object_id in range(_object_count(mesh, mdim)):
            incident = shared_by_dimension[mdim].get(object_id)
            if incident is None or mdim < form_order:
                objects.append([])
                continue
            element_ids, orientations = incident
            mapped_orders = _mapped_orders(element_specs, mdim, element_ids, orientations)
            component_specs: list[KFormSpecs] = []
            for active_axes in combinations(range(mdim), form_order):
                test_orders = tuple(
                    order if axis in active_axes else order - 2
                    for axis, order in enumerate(mapped_orders)
                )
                if min(test_orders, default=0) < 0:
                    continue
                test_space = FunctionSpace(
                    *(BasisSpecs(basis_type, order) for order in test_orders)
                )
                component_specs.append(KFormSpecs(form_order, test_space))
            objects.append(component_specs)
        result.append(objects)
    return result


def _local_component_rows(
    local_result: tuple[np.ndarray, ...],
    test_spec: KFormSpecs,
    component: int,
    element_id: int,
    sign: float,
) -> list[list[tuple[int, int, int, float]]]:
    """Return one packed entry list per canonical test row."""
    row_offsets, local_components, local_dofs, coefficients = local_result
    component_counts = np.asarray(test_spec.component_dof_counts)
    row_start = int(np.sum(component_counts[:component]))
    row_count = int(component_counts[component])
    return [
        [
            (
                element_id,
                int(local_components[index]),
                int(local_dofs[index]),
                sign * float(coefficients[index]),
            )
            for index in range(int(row_offsets[row]), int(row_offsets[row + 1]))
        ]
        for row in range(row_start, row_start + row_count)
    ]


def build_continuity_rows_reference(
    mesh: Mesh,
    maps: list[SpaceMap],
    element_specs: list[KFormSpecs],
    test_specs: list[list[list[KFormSpecs]]],
) -> PackedRows:
    """Assemble cycle-free rows using the local boundary API reference."""
    rows: list[list[tuple[int, int, int, float]]] = []
    for mdim, object_id, shared_element_ids, _ in mesh.iterate_shared_all():
        object_tests = test_specs[mdim][int(object_id)]
        if not object_tests:
            continue
        for first, second in zip(
            shared_element_ids[:-1], shared_element_ids[1:], strict=True
        ):
            first_id, second_id = int(first), int(second)
            for component, test_spec in enumerate(object_tests):
                first_result = compute_kform_boundary_constraints(
                    test_spec,
                    element_specs[first_id],
                    maps[first_id],
                    mesh.collections,
                    mesh.point_count,
                    first_id,
                    int(object_id),
                )
                second_result = compute_kform_boundary_constraints(
                    test_spec,
                    element_specs[second_id],
                    maps[second_id],
                    mesh.collections,
                    mesh.point_count,
                    second_id,
                    int(object_id),
                )
                first_rows = _local_component_rows(
                    first_result, test_spec, component, first_id, +1.0
                )
                second_rows = _local_component_rows(
                    second_result, test_spec, component, second_id, -1.0
                )
                if len(first_rows) != len(second_rows):
                    raise ValueError(
                        "Paired elements produced different trace row counts."
                    )
                rows.extend(
                    first_row + second_row
                    for first_row, second_row in zip(first_rows, second_rows, strict=True)
                )
    row_offsets = np.zeros(len(rows) + 1, dtype=np.uintp)
    element_ids: list[int] = []
    components: list[int] = []
    local_dofs: list[int] = []
    coefficients: list[float] = []
    for row, entries in enumerate(rows):
        element_ids.extend(entry[0] for entry in entries)
        components.extend(entry[1] for entry in entries)
        local_dofs.extend(entry[2] for entry in entries)
        coefficients.extend(entry[3] for entry in entries)
        row_offsets[row + 1] = len(element_ids)
    return (
        row_offsets,
        np.asarray(element_ids, dtype=np.uint64),
        np.asarray(components, dtype=np.uint32),
        np.asarray(local_dofs, dtype=np.uintp),
        np.asarray(coefficients, dtype=np.double),
    )


def build_continuity_rows(
    mesh: Mesh,
    maps: list[SpaceMap],
    element_specs: list[KFormSpecs],
    test_specs: list[list[list[KFormSpecs]]],
) -> PackedRows:
    """Assemble hierarchical rows through the public C-backed mesh method."""
    return mesh.compute_kform_continuity_constraints(element_specs, maps, test_specs)


def packed_to_dense(packed: PackedRows, element_specs: list[KFormSpecs]) -> np.ndarray:
    """Materialize global packed rows as a dense matrix."""
    row_offsets, element_ids, components, local_dofs, coefficients = packed
    dofs_per_element = [int(np.sum(spec.component_dof_counts)) for spec in element_specs]
    element_offsets = np.cumsum([0, *dofs_per_element[:-1]], dtype=np.intp)
    matrix = np.zeros((row_offsets.size - 1, int(sum(dofs_per_element))))
    for row in range(matrix.shape[0]):
        for index in range(int(row_offsets[row]), int(row_offsets[row + 1])):
            element = int(element_ids[index])
            component = int(components[index])
            column = (
                int(element_offsets[element])
                + int(element_specs[element].get_component_slice(component).start)
                + int(local_dofs[index])
            )
            matrix[row, column] += coefficients[index]
    return matrix


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
    r"""Return :math:`- \nabla` of :func:`manufactured_solution`."""
    return np.sum(
        [
            0.5 * np.pi**2 * np.cos(np.pi * coordinate)
            + (np.pi**2 / 16.0) * np.sin(0.5 * np.pi * coordinate)
            for coordinate in coordinates
        ],
        axis=0,
    )


def solve_direct_laplace(
    ndim: int, order: int
) -> tuple[np.ndarray, np.ndarray, float, int, int, float]:
    """Solve one direct primal problem and return diagnostics."""
    mesh = make_mesh(ndim)
    maps = make_element_maps(ndim, order + 4)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    tests = make_test_specs(mesh, element_specs, 0, BasisType.LEGENDRE)
    packed = build_continuity_rows(mesh, maps, element_specs, tests)
    continuity = packed_to_dense(packed, element_specs)

    n0 = int(np.sum(element_specs[0].component_dof_counts))
    total_dofs = len(maps) * n0
    stiffness = np.zeros((total_dofs, total_dofs))
    rhs = np.zeros(total_dofs)
    incidence = compute_kform_incidence_matrix(base_space, 0)
    for element_id, element_map in enumerate(maps):
        mass_one = np.asarray(
            compute_kform_mass_matrix(element_map, 1, base_space, base_space)
        )
        local_stiffness = incidence.T @ mass_one @ incidence
        offset = element_id * n0
        stiffness[offset : offset + n0, offset : offset + n0] = local_stiffness
        rhs[offset : offset + n0] = projection_l2_dual(
            manufactured_source, base_space, element_map
        ).values.flatten()

    gauge = np.zeros((1, total_dofs))
    gauge[0, 0] = 1.0
    constraints = np.vstack((continuity, gauge))
    saddle = np.block(
        [
            [stiffness, constraints.T],
            [constraints, np.zeros((constraints.shape[0], constraints.shape[0]))],
        ]
    )
    constraint_rhs = np.zeros(constraints.shape[0])
    constraint_rhs[-1] = 1.0
    solution = scipy.linalg.solve(
        saddle,
        np.concatenate((rhs, constraint_rhs)),
        assume_a="gen",
    )[:total_dofs]
    continuity_residual = float(np.max(np.abs(continuity @ solution), initial=0.0))
    if continuity_residual > 1.0e-10:
        raise RuntimeError(
            f"0-form continuity residual is too large: {continuity_residual:.3e}"
        )

    error_squared = 0.0
    for element_id, element_map in enumerate(maps):
        values = solution[element_id * n0 : (element_id + 1) * n0]
        dofs = DegreesOfFreedom(base_space, values)
        reference_values = dofs.reconstruct_at_integration_points(
            element_map.integration_space
        )
        coordinates = [element_map.coordinate_map(axis).values for axis in range(ndim)]
        exact = manufactured_solution(*coordinates)
        error_squared += np.sum(
            (reference_values - exact) ** 2
            * np.abs(element_map.determinant)
            * element_map.integration_space.weights()
        )
    rank = int(np.linalg.matrix_rank(saddle))
    expected_rank = saddle.shape[0]
    if rank != expected_rank:
        raise RuntimeError(f"Saddle system is rank deficient: {rank}/{expected_rank}")
    return (
        solution,
        continuity,
        float(np.sqrt(error_squared)),
        rank,
        constraints.shape[0],
        continuity_residual,
    )


def main() -> None:
    """Report p-refinement for curved 2D, 3D, and 4D meshes."""
    order_sweeps = {
        2: (1, 2, 3, 4, 5, 6, 7, 8),
        3: (1, 2, 3, 4, 5),
        # Dense saddle solves make p >= 3 unnecessarily expensive in 4D.
        4: (1, 2),
    }
    convergence: dict[int, tuple[tuple[int, ...], list[float]]] = {}
    for ndim, orders in order_sweeps.items():
        print(
            f"{ndim}D curved mesh (deformation={DEFORMATION:g}):",
            flush=True,
        )
        errors: list[float] = []
        for order in orders:
            started = perf_counter()
            solution, continuity, error, rank, constraint_count, residual = (
                solve_direct_laplace(ndim, order)
            )
            elapsed = perf_counter() - started
            del solution
            errors.append(error)
            system_size = continuity.shape[1] + constraint_count
            print(
                f"  p={order}: 0-form continuity residual={residual:.3e}, "
                f"rows={continuity.shape[0]}, rank={rank}/{system_size}, "
                f"L2={error:.6e}, solve={elapsed:.2f}s",
                flush=True,
            )
        convergence[ndim] = (orders, errors)
        ratios = [previous / current for previous, current in zip(errors, errors[1:])]
        print(
            "  L2 errors: " + ", ".join(f"{error:.6e}" for error in errors),
            flush=True,
        )
        if ratios:
            print(
                "  successive improvement: "
                + ", ".join(f"{ratio:.3f}x" for ratio in ratios),
                flush=True,
            )

    _, axis = plt.subplots()
    for ndim, (orders, errors) in convergence.items():
        axis.semilogy(orders, errors, marker="o", label=f"{ndim}D")
    axis.set(
        xlabel="polynomial order p",
        ylabel="L2 error",
        title="Curved multi-element 0-form Laplace convergence",
    )
    axis.grid(True, which="both")
    axis.legend()
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
