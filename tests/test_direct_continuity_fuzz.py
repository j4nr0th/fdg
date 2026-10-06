"""Randomised hunt for crashes and invariant violations in the direct transfer.

Every case is drawn from one seed and printed, so a failure reproduces without
the fuzzer. Only structural invariants are asserted, never expected sizes; a
violation is a wrong answer unless the process died.
"""

import random

import numpy as np
import pytest
from fdg import BasisSpecs, DirectDofMap, FunctionSpace, KFormSpecs, Mesh
from fdg.enum_type import BasisType

FAMILIES = (
    BasisType.LAGRANGE_UNIFORM,
    BasisType.LAGRANGE_GAUSS,
    BasisType.LAGRANGE_GAUSS_LOBATTO,
    BasisType.LAGRANGE_CHEBYSHEV_GAUSS,
    BasisType.LEGENDRE,
    BasisType.BERNSTEIN,
)

SEED_COUNT = 240
WIDE_ORDER_SEED_COUNT = 64
UNUSED_POINT_SEED_COUNT = 24


def grid_corners(ndim: int, cells: int) -> np.ndarray:
    """Corner point IDs of a structured grid, ``2**ndim`` per element."""
    lattice = cells + 1
    corners = np.zeros(cells**ndim * 2**ndim, dtype=np.uint64)
    for element in range(cells**ndim):
        rest = element
        lower = []
        for _ in range(ndim):
            lower.append(rest % cells)
            rest //= cells
        for corner in range(2**ndim):
            point = 0
            for axis in range(ndim):
                point = point * lattice + lower[axis] + ((corner >> axis) & 1)
            corners[element * 2**ndim + corner] = point
    return corners


def vertex_fan_corners(ndim: int, element_count: int) -> np.ndarray:
    """Corners of elements that meet in the origin and share no face.

    One orthant of the even-parity code per element, so any two elements differ
    on at least two axes.
    """
    corners: list[int] = []
    seen: dict[tuple[int, ...], int] = {}
    for element in range(element_count):
        if ndim == 1:
            sides = [element & 1]
        else:
            rest = element
            sides = [0] * ndim
            for axis in range(1, ndim):
                sides[axis] = rest & 1
                rest >>= 1
            sides[0] = sum(sides[1:]) % 2
        for corner in range(2**ndim):
            key = tuple(sides[axis] + ((corner >> axis) & 1) for axis in range(ndim))
            seen.setdefault(key, len(seen))
            corners.append(seen[key])
    return np.array(corners, dtype=np.uint64)


def draw_case(
    rng: random.Random, max_order: int, max_cells: int
) -> tuple[int, np.ndarray, list[tuple[int, ...]], int, BasisType]:
    """Draw one mesh, one order per element and axis, and one k-form order."""
    ndim = rng.choice((1, 2, 3))
    kform_order = rng.randrange(ndim + 1)
    family = rng.choice(FAMILIES)
    if rng.random() < 0.4:
        fans = 2 if ndim == 1 else 2 ** (ndim - 1)
        corners = vertex_fan_corners(ndim, rng.randint(1, fans))
    else:
        corners = grid_corners(ndim, rng.choice(tuple(range(1, max_cells + 1))))
    orders = [
        tuple(rng.randint(1, max_order) for _ in range(ndim))
        for _ in range(corners.size // 2**ndim)
    ]
    return ndim, corners, orders, kform_order, family


def specs_of(
    orders: list[tuple[int, ...]], kform_order: int, family: BasisType
) -> list[KFormSpecs]:
    """One specification per element, every axis on the given basis family."""
    return [
        KFormSpecs(kform_order, FunctionSpace(*(BasisSpecs(family, o) for o in axes)))
        for axes in orders
    ]


def relabelled(corners: np.ndarray, ndim: int, permutation: list[int]) -> np.ndarray:
    """Build the same physical mesh under a different element numbering."""
    block = corners.reshape(-1, 2**ndim)
    return block[np.array(permutation, dtype=np.intp)].reshape(-1)


def describe(
    label: int | str,
    ndim: int,
    corners: np.ndarray,
    orders: list[tuple[int, ...]],
    kform_order: int,
    family: BasisType,
) -> str:
    """Describe the full case, so a failure can be rebuilt without the fuzzer."""
    return (
        f"{label}: ndim={ndim} kform_order={kform_order} family={family.name}\n"
        f"corners={np.array2string(corners, threshold=corners.size + 1)}\n"
        f"orders={orders}"
    )


def build(
    corners: np.ndarray,
    ndim: int,
    orders: list[tuple[int, ...]],
    kform_order: int,
    family: BasisType,
    context: str,
) -> DirectDofMap:
    """Build the map, turning any exception into a failure that names the case."""
    mesh = Mesh.from_corners(ndim, corners)
    try:
        return mesh.compute_kform_direct_dof_map(specs_of(orders, kform_order, family))
    except Exception as error:
        pytest.fail(f"{context}\nraised {type(error).__name__}: {error}")


def check_invariants(dof_map: DirectDofMap, context: str) -> None:
    """Assert the structural invariants every conforming transfer must satisfy."""
    element_dofs = dof_map.element_dof_count
    globals_ = dof_map.global_dof_count
    offsets = dof_map.entry_offsets
    index = dof_map.entry_index
    values = dof_map.entry_value

    assert offsets.shape == (element_dofs + 1,), context
    assert int(offsets[0]) == 0, context
    assert bool(np.all(np.diff(offsets) >= 0)), context
    assert int(offsets[-1]) == dof_map.entry_count, context

    assert bool(np.all(index >= 0)), context
    assert bool(np.all(index < globals_)), context
    assert np.unique(index).size == globals_, context
    assert globals_ <= element_dofs, context
    assert bool(np.all(np.isfinite(values))), context

    # A row is one element-local degree of freedom; it may not name a global twice.
    rows = np.repeat(np.arange(element_dofs), np.diff(offsets))
    order = np.lexsort((index, rows))
    ordered_rows = rows[order]
    ordered_index = index[order]
    same_row = ordered_rows[1:] == ordered_rows[:-1]
    duplicated = same_row & (ordered_index[1:] == ordered_index[:-1])
    assert not bool(np.any(duplicated)), context


@pytest.mark.parametrize("seed", range(SEED_COUNT))
def test_random_mesh_holds_every_invariant(seed: int) -> None:
    """No random mesh may crash the map or break one of its structural invariants."""
    rng = random.Random(seed)
    ndim, corners, orders, kform, family = draw_case(rng, 4, 3)
    context = describe(seed, ndim, corners, orders, kform, family)

    dof_map = build(corners, ndim, orders, kform, family, context)
    check_invariants(dof_map, context)

    elements = corners.size // 2**ndim
    permutation = list(rng.sample(range(elements), elements))
    renamed_corners = relabelled(corners, ndim, permutation)
    assert bool(np.array_equal(np.sort(renamed_corners), np.sort(corners))), context
    # The specification travels with the element; only the element ID changes.
    renamed_orders = [orders[old] for old in permutation]
    renamed = build(renamed_corners, ndim, renamed_orders, kform, family, context)
    check_invariants(renamed, context)
    assert renamed.global_dof_count == dof_map.global_dof_count, context


@pytest.mark.parametrize("seed", range(WIDE_ORDER_SEED_COUNT))
def test_wide_order_gap_holds_every_invariant(seed: int) -> None:
    """A wide gap between the element and the common order stays well scaled."""
    rng = random.Random(seed + 50_000)
    ndim, corners, orders, kform, family = draw_case(rng, 8, 2)
    context = describe(seed, ndim, corners, orders, kform, family)

    check_invariants(build(corners, ndim, orders, kform, family, context), context)


@pytest.mark.parametrize("seed", range(UNUSED_POINT_SEED_COUNT))
def test_declared_but_unused_points_add_no_unknowns(seed: int) -> None:
    """Points no element carries may not enlarge the global space."""
    rng = random.Random(seed + 100_000)
    ndim, corners, orders, kform, family = draw_case(rng, 4, 3)
    context = describe(seed, ndim, corners, orders, kform, family)

    base = Mesh.from_corners(ndim, corners)
    spare = rng.randint(1, 7)
    padded = Mesh.from_collections(ndim, base.point_count + spare, base.collections)
    assert padded.point_count == base.point_count + spare, context

    reference = base.compute_kform_direct_dof_map(specs_of(orders, kform, family))
    dof_map = padded.compute_kform_direct_dof_map(specs_of(orders, kform, family))
    check_invariants(dof_map, context)
    assert dof_map.global_dof_count <= reference.global_dof_count, context


def test_two_quads_project_onto_a_finite_common_space() -> None:
    """A wide order gap must still give finite transfer weights.

    The elimination pairs against a Legendre object Gram, so the equispaced-Gram
    overflow this case once exposed is structurally impossible; finiteness
    stays pinned.
    """
    corners = np.array([0, 1, 3, 2, 2, 3, 5, 4], dtype=np.uint64)
    orders: list[tuple[int, ...]] = [(5, 1), (6, 1)]
    family = BasisType.LAGRANGE_UNIFORM
    context = describe("two quads", 2, corners, orders, 1, family)

    dof_map = build(corners, 2, orders, 1, family, context)
    finite = np.isfinite(dof_map.entry_value)
    bad = dof_map.entry_index[~finite]
    assert bool(np.all(finite)), f"{context}\nglobals {bad} carry a non-finite weight"
