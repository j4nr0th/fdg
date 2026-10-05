"""Sparse assembly and solver agreement for the direct continuity system.

The transfer's scatter becomes a standard sparse assembly, and the reduced
system goes to SciPy and to hybsol; the tests pin sparse against the dense
example path and hybsol against SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sparse_la
from fdg import (
    BasisSpecs,
    BasisType,
    DirectDofMap,
    FunctionSpace,
    KFormSpecs,
    Mesh,
    SpaceMap,
    laplace_stiffness,
    scatter_csc,
)

from examples.plot_direct_continuity_sparse import (
    assemble_sparse_global_laplace,
    block_owners,
    prescribe_dirichlet,
    solve_hybsol_reduced,
)
from examples.plot_multi_element_laplace_direct_continuity import (
    assemble_global_laplace,
    boundary_object_globals,
    make_element_maps,
    make_mesh,
)


@dataclass
class _Case:
    """One assembled 2D case shared by the tests."""

    transfer: DirectDofMap
    mesh: Mesh
    maps: list[SpaceMap]
    order: int
    cells: int
    sparse_matrix: sp.csc_matrix
    rhs: np.ndarray
    dense_matrix: np.ndarray
    dense_rhs: np.ndarray
    boundary: np.ndarray
    data: np.ndarray


def _assembled_case() -> _Case:
    """Assemble the shared case with both the sparse and the dense scatter."""
    ndim, order, cells = 2, 2, 3
    mesh = make_mesh(ndim, cells)
    maps = make_element_maps(ndim, order + 4, cells)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    transfer = mesh.compute_kform_direct_dof_map(element_specs)
    sparse_matrix, rhs = assemble_sparse_global_laplace(
        transfer, maps, element_specs, base_space
    )
    dense_matrix, dense_rhs = assemble_global_laplace(
        transfer, maps, element_specs, base_space
    )
    boundary, data = boundary_object_globals(transfer, mesh, maps, order, cells)
    return _Case(
        transfer,
        mesh,
        maps,
        order,
        cells,
        sparse_matrix,
        rhs,
        dense_matrix,
        dense_rhs,
        boundary,
        data,
    )


def test_sparse_assembly_equals_dense() -> None:
    """The COO scatter reproduces the dense np.add.at assembly."""
    case = _assembled_case()
    scale = float(np.max(np.abs(case.dense_matrix)))
    assert case.sparse_matrix.shape == case.dense_matrix.shape
    difference = float(np.max(np.abs(case.sparse_matrix.toarray() - case.dense_matrix)))
    assert difference <= 1.0e-12 * scale
    rhs_difference = float(np.max(np.abs(case.rhs - case.dense_rhs)))
    assert rhs_difference <= 1.0e-12 * float(np.max(np.abs(case.dense_rhs)))


def test_sparse_solution_agrees_with_dense() -> None:
    """The sparse Dirichlet solve reproduces the dense example solution."""
    case = _assembled_case()
    free, reduced, reduced_rhs = prescribe_dirichlet(
        case.sparse_matrix, case.rhs, case.boundary, case.data
    )
    dense = np.zeros(case.transfer.global_dof_count)
    dense[case.boundary] = case.data
    dense[free] = np.linalg.solve(reduced.toarray(), reduced_rhs)
    sparse = np.zeros(case.transfer.global_dof_count)
    sparse[case.boundary] = case.data
    sparse[free] = sparse_la.splu(reduced).solve(reduced_rhs)
    scale = float(np.max(np.abs(dense)))
    assert float(np.max(np.abs(sparse - dense))) <= 1.0e-12 * scale


def test_scipy_and_hybsol_agree() -> None:
    """The hybsol block solve reproduces the SciPy solve."""
    case = _assembled_case()
    free, reduced, reduced_rhs = prescribe_dirichlet(
        case.sparse_matrix, case.rhs, case.boundary, case.data
    )
    owners, _ = block_owners(case.transfer, case.mesh, case.order, case.cells)
    hybsol_free, blocking, _ = solve_hybsol_reduced(reduced, reduced_rhs, owners, free)
    assert blocking == "element/object"
    scipy_free = sparse_la.splu(reduced).solve(reduced_rhs)
    scale = float(np.max(np.abs(scipy_free)))
    assert float(np.max(np.abs(hybsol_free - scipy_free))) <= 1.0e-12 * scale


def test_dirichlet_rows_are_prescribed() -> None:
    """Free rows are satisfied and boundary unknowns carry the projection."""
    case = _assembled_case()
    free, reduced, reduced_rhs = prescribe_dirichlet(
        case.sparse_matrix, case.rhs, case.boundary, case.data
    )
    owners, _ = block_owners(case.transfer, case.mesh, case.order, case.cells)
    hybsol_free, _, _ = solve_hybsol_reduced(reduced, reduced_rhs, owners, free)
    solution = np.zeros(case.transfer.global_dof_count)
    solution[case.boundary] = case.data
    solution[free] = hybsol_free
    residual = case.sparse_matrix[free] @ solution - case.rhs[free]
    assert float(np.max(np.abs(residual))) <= 1.0e-10 * float(np.max(np.abs(case.rhs)))
    assert np.array_equal(solution[case.boundary], case.data)


def test_solve_is_deterministic() -> None:
    """Repeated hybsol solves return identical solutions."""
    case = _assembled_case()
    free, reduced, reduced_rhs = prescribe_dirichlet(
        case.sparse_matrix, case.rhs, case.boundary, case.data
    )
    owners, _ = block_owners(case.transfer, case.mesh, case.order, case.cells)
    first, _, _ = solve_hybsol_reduced(reduced, reduced_rhs, owners, free)
    second, _, _ = solve_hybsol_reduced(reduced, reduced_rhs, owners, free)
    assert np.array_equal(first, second)


def _flat_stiffness(
    transfer: DirectDofMap,
    maps: list[SpaceMap],
    element_specs: list[KFormSpecs],
) -> npt.NDArray[np.double]:
    """Pack every element's raw stiffness into the flat array scatter_csc consumes."""
    sizes = np.diff(transfer.element_offsets) ** 2
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    local = np.empty(int(sizes.sum()))
    for element_id, element_map in enumerate(maps):
        stiffness = laplace_stiffness(
            element_id, [element_specs[element_id]], element_map
        )
        local[
            int(starts[element_id]) : int(starts[element_id]) + int(sizes[element_id])
        ] = stiffness.reshape(-1)
    return local


def _transfer_for(
    ndim: int, order: int, cells: int
) -> tuple[DirectDofMap, list[SpaceMap], list[KFormSpecs], FunctionSpace]:
    """Build the mesh, maps, and specs of one case and return its direct transfer."""
    mesh = make_mesh(ndim, cells)
    maps = make_element_maps(ndim, order + 4, cells)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    element_specs = [KFormSpecs(0, base_space) for _ in maps]
    return (
        mesh.compute_kform_direct_dof_map(element_specs),
        maps,
        element_specs,
        base_space,
    )


def test_scatter_csc_matches_dense_2d() -> None:
    """The batched C scatter reproduces the dense np.add.at assembly in 2D."""
    transfer, maps, element_specs, base_space = _transfer_for(2, 2, 3)
    local = _flat_stiffness(transfer, maps, element_specs)
    matrix = scatter_csc(transfer, local).toarray()
    dense, _ = assemble_global_laplace(transfer, maps, element_specs, base_space)
    scale = float(np.max(np.abs(dense)))
    assert float(np.max(np.abs(matrix - dense))) <= 1.0e-12 * scale


def test_scatter_csc_matches_dense_3d() -> None:
    """The batched C scatter reproduces the dense np.add.at assembly in 3D."""
    transfer, maps, element_specs, base_space = _transfer_for(3, 2, 2)
    local = _flat_stiffness(transfer, maps, element_specs)
    matrix = scatter_csc(transfer, local).toarray()
    dense, _ = assemble_global_laplace(transfer, maps, element_specs, base_space)
    scale = float(np.max(np.abs(dense)))
    assert float(np.max(np.abs(matrix - dense))) <= 1.0e-12 * scale


def test_scatter_csc_is_deterministic_over_threads() -> None:
    """One and four threads emit byte-identical CSC matrices."""
    transfer, maps, element_specs, _ = _transfer_for(2, 4, 4)
    local = _flat_stiffness(transfer, maps, element_specs)
    first = scatter_csc(transfer, local, n_threads=1)
    fourth = scatter_csc(transfer, local, n_threads=4)
    assert np.array_equal(first.indptr, fourth.indptr)
    assert np.array_equal(first.indices, fourth.indices)
    assert np.array_equal(first.data, fourth.data)


def test_scatter_csc_rejects_bad_input() -> None:
    """Wrong dtypes, shapes, lengths, and thread counts raise."""
    transfer, maps, element_specs, _ = _transfer_for(2, 2, 2)
    local = _flat_stiffness(transfer, maps, element_specs)
    with pytest.raises(TypeError):
        scatter_csc(transfer, local.astype(np.float32))
    with pytest.raises(TypeError):
        scatter_csc(transfer, local.reshape(1, -1))
    with pytest.raises(ValueError):
        scatter_csc(transfer, np.concatenate([local, [1.0]]))
    with pytest.raises(ValueError):
        scatter_csc(transfer, local[:-1])
    with pytest.raises(ValueError):
        scatter_csc(transfer, local, n_threads=-1)


def test_scatter_csc_matches_dense_over_batches() -> None:
    """The batched C scatter keeps every batch's block base when several batches run."""
    transfer, maps, element_specs, base_space = _transfer_for(2, 10, 4)
    local = _flat_stiffness(transfer, maps, element_specs)
    matrix = scatter_csc(transfer, local).toarray()
    dense, _ = assemble_global_laplace(transfer, maps, element_specs, base_space)
    scale = float(np.max(np.abs(dense)))
    assert float(np.max(np.abs(matrix - dense))) <= 1.0e-12 * scale


def test_scatter_csc_batches_match_single_batch(monkeypatch) -> None:
    """A one-triplet batch target yields the same CSC as one large batch."""
    transfer, maps, element_specs, _ = _transfer_for(2, 10, 4)
    local = _flat_stiffness(transfer, maps, element_specs)
    single = scatter_csc(transfer, local)
    # One triplet per batch: every element forms its own batch.
    monkeypatch.setattr("fdg.sparse.BATCH_TRIPLETS", 1)
    batched = scatter_csc(transfer, local)
    assert np.array_equal(single.indptr, batched.indptr)
    assert np.array_equal(single.indices, batched.indices)
    # The batches sum duplicates in a different order, so only rounding differs.
    assert float(np.max(np.abs(single.data - batched.data))) <= 1.0e-12 * float(
        np.max(np.abs(single.data))
    )


def test_scatter_triplets_empty_range_is_empty() -> None:
    """An empty element range yields three empty typed arrays."""
    transfer, maps, element_specs, _ = _transfer_for(2, 2, 3)
    local = _flat_stiffness(transfer, maps, element_specs)
    rows, cols, values = transfer._scatter_triplets(local, 0, 0)
    assert rows.size == cols.size == values.size == 0
    assert rows.dtype == np.int64
    assert cols.dtype == np.int64
    assert values.dtype == np.float64
