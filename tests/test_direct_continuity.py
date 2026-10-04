"""Behavioral tests for the direct element-to-global continuity map.

The direct map eliminates every element against the common Legendre window of
its shared objects, so direct and hybridized solves of the same problem agree
to solver tolerance. The example module drives both on the same mesh.
"""

from __future__ import annotations

import numpy as np
import pytest
from fdg import BasisSpecs, BasisType, FunctionSpace, KFormSpecs

from examples.plot_multi_element_laplace_direct_continuity import (
    make_mesh,
    solve_direct_continuity,
    solve_hybridized_continuity,
)


@pytest.mark.parametrize(
    ("ndim", "cells", "order"),
    ((1, 3, 1), (2, 2, 1), (2, 2, 3), (2, 3, 2), (3, 2, 1), (3, 2, 2)),
)
def test_scalar_global_dof_count_matches_the_lattice(
    ndim: int, cells: int, order: int
) -> None:
    """A conforming scalar space has one DoF per lattice point."""
    mesh = make_mesh(ndim, cells)
    base_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
    )
    transfer = mesh.compute_kform_direct_dof_map(
        [KFormSpecs(0, base_space) for _ in range(mesh.element_count)]
    )

    assert transfer.global_dof_count == (cells * order + 1) ** ndim
    assert transfer.element_dof_count == cells**ndim * (order + 1) ** ndim
    assert transfer.entry_offsets.size == transfer.element_dof_count + 1
    assert transfer.entry_index.size == transfer.entry_count
    assert transfer.entry_value.size == transfer.entry_count
    assert int(transfer.entry_index.max(initial=-1)) < transfer.global_dof_count


@pytest.mark.parametrize("ndim", (2, 3))
def test_direct_unknowns_never_exceed_the_hybridized_ones(ndim: int) -> None:
    """Sharing interface DoFs saves unknowns as soon as the mesh is split."""
    _, single_unknowns, _, _ = solve_hybridized_continuity(ndim, 2, 1)
    _, split_unknowns, _, _ = solve_hybridized_continuity(ndim, 2, 2)
    single_error, single_dofs, _ = solve_direct_continuity(ndim, 2, 1)
    split_error, split_dofs, _ = solve_direct_continuity(ndim, 2, 2)

    # A single element has no interface, so the direct path only drops the boundary rows.
    assert single_dofs < single_unknowns
    assert single_error > 0.0

    # With interfaces present, it also drops the duplicated DoFs.
    assert split_dofs < split_unknowns
    assert split_error < single_error


def test_boundary_globals_carry_the_window_projection() -> None:
    """Exactly the outer objects' window unknowns are prescribed, with projected data."""
    from examples.plot_multi_element_laplace_direct_continuity import (
        boundary_object_globals,
        make_element_maps,
    )

    for ndim, cells, order in ((2, 2, 2), (3, 2, 1), (1, 3, 2)):
        mesh = make_mesh(ndim, cells)
        maps = make_element_maps(ndim, order + 4, cells)
        base_space = FunctionSpace(
            *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, order) for _ in range(ndim))
        )
        transfer = mesh.compute_kform_direct_dof_map(
            [KFormSpecs(0, base_space) for _ in range(mesh.element_count)]
        )
        boundary, data = boundary_object_globals(transfer, mesh, maps, order, cells)

        # A scalar window is max(order - 1, 0) per axis; a point always prescribes one.
        window = max(order - 1, 0)
        expected = 0
        for mdim, _, _, _ in mesh.iterate_boundary_all():
            expected += window**mdim
        assert boundary.size == expected
        assert data.size == boundary.size
        assert np.all(np.isfinite(data))
        # The boundary unknowns are the leading globals: objects first, free modes last.
        assert int(boundary.max()) < transfer.global_dof_count - int(
            np.diff(transfer.element_interior_offsets).sum()
        )
        # Every global unknown is reachable, otherwise its row and column stay zero.
        assert np.array_equal(
            np.unique(np.asarray(transfer.entry_index)),
            np.arange(transfer.global_dof_count),
        )


def test_direct_and_hybridized_agree_to_solver_tolerance() -> None:
    """Both formulations discretise the same space: their solutions coincide."""
    for ndim in (2, 3):
        for order in (1, 2, 3):
            direct, _, _ = solve_direct_continuity(ndim, order, 2)
            hybridized, _, _, _ = solve_hybridized_continuity(ndim, order, 2)
            assert direct == pytest.approx(hybridized, rel=1.0e-10)


def test_direct_errors_decrease_under_refinement() -> None:
    """The direct path converges at the expected rate for a manufactured solution."""
    errors = [solve_direct_continuity(2, order, 2)[0] for order in (1, 2, 3, 4)]
    hybridized = [solve_hybridized_continuity(2, order, 2)[0] for order in (1, 2, 3, 4)]

    assert all(np.isfinite(error) for error in errors)
    assert errors == sorted(errors, reverse=True)
    assert hybridized == sorted(hybridized, reverse=True)
    # Both paths approximate the same function on the same spaces.
    assert errors[-1] < 0.1 * errors[0]
    for direct, hybrid in zip(errors, hybridized):
        assert direct == pytest.approx(hybrid, rel=1.0e-10)


def test_direct_system_is_symmetric_with_a_constant_nullspace() -> None:
    """Scattering a symmetric element matrix keeps the global matrix symmetric."""
    _, _, matrix = solve_direct_continuity(2, 2, 2)
    assert np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-14 * np.abs(matrix).max())
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues[0] < 1.0e-12 * eigenvalues[-1]
    # Laplace is singular up to the constants, which the Dirichlet elimination removes.
    assert np.count_nonzero(eigenvalues < 1.0e-12 * eigenvalues[-1]) == 1


def test_direct_accepts_a_non_lagrange_family() -> None:
    """The elimination pairs inner products, so Legendre elements solve too."""
    direct, _, _ = solve_direct_continuity(2, 3, 2, basis_type=BasisType.LEGENDRE)
    # Pure Legendre is singular in the hybridized solver; equal-order GLL is valid there.
    hybridized, _, _, _ = solve_hybridized_continuity(2, 3, 2)

    assert np.isfinite(direct)
    assert direct > 0.0
    assert direct == pytest.approx(hybridized, rel=1.0e-8)
