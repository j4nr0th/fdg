"""Check the direct element-to-global transfer of Mesh.compute_kform_direct_dof_map.

Shared objects carry the coefficients of their common windowed Legendre test
space, each element's transfer combines them with its private free modes, and
any basis family is accepted while a zero basis order is not.
"""

import numpy as np
import pytest
from fdg import (
    BasisRegistry,
    BasisSpecs,
    DirectDofMap,
    FunctionSpace,
    IntegrationRegistry,
    KFormSpecs,
    Mesh,
)
from fdg.enum_type import BasisType


def grid2(x: int, y: int) -> int:
    """Point ID of the (x, y) node of the 2x2 grid."""
    return x + 3 * y


CORNERS_2X2 = np.array(
    [
        grid2(0, 0),
        grid2(1, 0),
        grid2(0, 1),
        grid2(1, 1),  # element 0 (ix,iy)=(0,0)
        grid2(1, 0),
        grid2(2, 0),
        grid2(1, 1),
        grid2(2, 1),  # element 1 (1,0)
        grid2(0, 1),
        grid2(1, 1),
        grid2(0, 2),
        grid2(1, 2),  # element 2 (0,1)
        grid2(1, 1),
        grid2(2, 1),
        grid2(1, 2),
        grid2(2, 2),  # element 3 (1,1)
    ],
    dtype=np.uint64,
)

MESH_2X2 = Mesh.from_corners(2, CORNERS_2X2)

# Three intervals of the unit interval sharing their end points.
MESH_LINE = Mesh.from_corners(1, np.array([0, 1, 1, 2, 2, 3], dtype=np.uint64))

# Two intervals [0, 1] and [1, 2] sharing their middle point.
MESH_LINE_TWO = Mesh.from_corners(1, np.array([0, 1, 1, 2], dtype=np.uint64))


def uniform_space(
    ndim: int, order: int, family: BasisType = BasisType.LAGRANGE_UNIFORM
) -> FunctionSpace:
    """Build a tensor space of the given order on every axis."""
    return FunctionSpace(*(BasisSpecs(family, order) for _ in range(ndim)))


def specs_of(
    mesh: Mesh,
    order: int,
    kform_order: int = 0,
    family: BasisType = BasisType.LAGRANGE_UNIFORM,
) -> list[KFormSpecs]:
    """One specification of the given k-form order per element of the mesh."""
    space = uniform_space(mesh.ndim, order, family)
    return [KFormSpecs(kform_order, space) for _ in range(mesh.element_count)]


def test_line_hand_case_is_exact() -> None:
    """Two Legendre line elements pin the whole elimination by hand.

    The shared point's window is one constant, so every element degree of
    freedom maps onto its endpoint unknowns with coefficients 1/2 and +-1/2.
    """
    dof_map = MESH_LINE_TWO.compute_kform_direct_dof_map(
        specs_of(MESH_LINE_TWO, 1, family=BasisType.LEGENDRE)
    )

    assert dof_map.global_dof_count == 3
    assert dof_map.element_dof_count == 4
    # One shared object block per element, nothing element-private.
    assert dof_map.element_interior_offsets.tolist() == [3, 3, 3]
    assert dof_map.entry_offsets.tolist() == [0, 2, 4, 6, 8]
    assert dof_map.entry_index.tolist() == [0, 1, 0, 1, 1, 2, 1, 2]
    # Element 0 evaluates x=0 as (u0-u1)/2 and x=1 as (u0+u1)/2, matching element 1.
    assert dof_map.entry_value.tolist() == pytest.approx(
        [0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5, 0.5], abs=1.0e-12
    )

    assert dof_map.entry_offsets[-1] == dof_map.entry_count == 8


@pytest.mark.parametrize("order", [1, 2, 3])
def test_scalar_map_counts_the_lattice(order: int) -> None:
    """The shared windows of a uniform grid save exactly the duplicated modes."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(specs_of(MESH_2X2, order))

    assert isinstance(dof_map, DirectDofMap)
    assert dof_map.global_dof_count == (2 * order + 1) ** 2
    assert dof_map.element_dof_count == 4 * (order + 1) ** 2
    assert dof_map.global_dof_count <= dof_map.element_dof_count
    # Every global unknown is reached, so no row of the system stays empty.
    assert np.unique(dof_map.entry_index).size == dof_map.global_dof_count


@pytest.mark.parametrize(
    "family", [BasisType.LAGRANGE_UNIFORM, BasisType.LAGRANGE_GAUSS_LOBATTO]
)
@pytest.mark.parametrize("kform_order", [0, 1])
def test_first_order_nodal_map_is_a_signed_identity(
    family: BasisType, kform_order: int
) -> None:
    """Order-one Lagrange elements share their nodes, so the map is +-1."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(
        specs_of(MESH_2X2, 1, kform_order=kform_order, family=family)
    )

    expected_globals = {0: 9, 1: 12}[kform_order]
    assert dof_map.global_dof_count == expected_globals
    assert dof_map.element_dof_count == 16
    # Every degree of freedom keeps exactly one entry of magnitude one.
    assert dof_map.entry_count == dof_map.element_dof_count
    assert np.all(np.diff(dof_map.entry_offsets) == 1)
    assert np.all(np.abs(dof_map.entry_value) == 1.0)
    # Nothing is element-private: the shared windows span the whole trace.
    assert np.all(dof_map.element_interior_offsets == expected_globals)


@pytest.mark.parametrize("family", list(BasisType))
@pytest.mark.parametrize("order", [1, 2, 3])
def test_top_form_map_is_an_exact_identity(family: BasisType, order: int) -> None:
    """A form of full degree shares nothing: every DoF is its own unknown."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(
        specs_of(MESH_2X2, order, kform_order=2, family=family)
    )

    assert dof_map.global_dof_count == dof_map.element_dof_count
    assert dof_map.entry_count == dof_map.element_dof_count
    assert np.all(np.diff(dof_map.entry_offsets) == 1)
    assert np.all(np.abs(dof_map.entry_value) == 1.0)
    assert dof_map.element_interior_offsets.tolist() == list(
        range(0, dof_map.global_dof_count + 1, order**2)
    )


def test_legendre_and_lagrange_count_the_same_globals() -> None:
    """The family changes the coefficients, not the size of the common space."""
    legendre = MESH_2X2.compute_kform_direct_dof_map(
        specs_of(MESH_2X2, 2, family=BasisType.LEGENDRE)
    )
    lagrange = MESH_2X2.compute_kform_direct_dof_map(specs_of(MESH_2X2, 2))

    assert legendre.global_dof_count == lagrange.global_dof_count == 25
    # Non-nodal families mix the shared windows densely instead of relabelling.
    assert legendre.entry_count > legendre.element_dof_count
    assert np.all(np.isfinite(legendre.entry_value))


@pytest.mark.parametrize("order", [1, 2])
def test_map_arrays_have_the_documented_layout(order: int) -> None:
    """The offsets partition the local DoFs and the entries of the transfer."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(specs_of(MESH_2X2, order))

    assert dof_map.element_offsets.dtype == np.int64
    assert dof_map.element_interior_offsets.dtype == np.int64
    assert dof_map.entry_offsets.dtype == np.int64
    assert dof_map.entry_index.dtype == np.int64
    assert dof_map.entry_value.dtype == np.double

    assert dof_map.element_offsets.shape == (MESH_2X2.element_count + 1,)
    assert dof_map.element_interior_offsets.shape == (MESH_2X2.element_count + 1,)
    assert dof_map.entry_offsets.shape == (dof_map.element_dof_count + 1,)
    assert dof_map.entry_index.shape == (dof_map.entry_count,)
    assert dof_map.entry_value.shape == (dof_map.entry_count,)

    assert dof_map.element_offsets[0] == 0
    assert dof_map.element_offsets[-1] == dof_map.element_dof_count
    assert np.all(np.diff(dof_map.element_offsets) > 0)

    # Every element-local DoF owns at least one entry of the row-compressed transfer.
    assert dof_map.entry_offsets[0] == 0
    assert dof_map.entry_offsets[-1] == dof_map.entry_count
    assert np.all(np.diff(dof_map.entry_offsets) >= 1)
    assert np.all(dof_map.entry_index >= 0)
    assert np.all(dof_map.entry_index < dof_map.global_dof_count)

    # The element-private ranges start past every shared object's block and never overlap.
    assert dof_map.element_interior_offsets[0] > 0
    assert np.all(np.diff(dof_map.element_interior_offsets) >= 0)
    assert dof_map.element_interior_offsets[-1] == dof_map.global_dof_count


def test_line_map_counts_its_intervals() -> None:
    """Three order-two intervals share four points and keep one free mode each."""
    dof_map = MESH_LINE.compute_kform_direct_dof_map(specs_of(MESH_LINE, 2))

    assert dof_map.global_dof_count == 3 * 2 + 1
    assert dof_map.element_dof_count == 3 * 3
    # The interior mode is orthogonal to the point windows and survives privately.
    assert dof_map.element_interior_offsets.tolist() == [4, 5, 6, 7]
    assert dof_map.entry_count == dof_map.element_dof_count


@pytest.mark.parametrize("kform_order", [0, 1, 2])
def test_kform_orders_are_numbered(kform_order: int) -> None:
    """Every k-form order of the mesh dimension produces a complete transfer."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(
        specs_of(MESH_2X2, 2, kform_order=kform_order)
    )

    assert dof_map.global_dof_count > 0
    assert dof_map.element_offsets[-1] == dof_map.element_dof_count
    assert dof_map.entry_offsets.shape == (dof_map.element_dof_count + 1,)
    assert dof_map.entry_offsets[-1] == dof_map.entry_count


def test_custom_registries_are_accepted() -> None:
    """The registries can be replaced, as with the other mesh methods."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(
        specs_of(MESH_2X2, 2),
        integration_registry=IntegrationRegistry(),
        basis_registry=BasisRegistry(),
    )

    assert dof_map.global_dof_count == 25


def test_element_specs_must_match_the_mesh() -> None:
    """The sequence must hold one specification per element."""
    with pytest.raises(ValueError, match="element_specs must contain 4 entries"):
        MESH_2X2.compute_kform_direct_dof_map([])

    with pytest.raises(TypeError, match="element_specs must contain KFormSpecs objects"):
        MESH_2X2.compute_kform_direct_dof_map([object()] * 4)  # type: ignore[arg-type]


def test_element_specs_must_share_the_k_form_degree() -> None:
    """One k-form order spans the whole mesh."""
    specs = specs_of(MESH_2X2, 1)
    specs[-1] = KFormSpecs(1, uniform_space(2, 1))

    with pytest.raises(
        ValueError, match="All element specs must have the same k-form degree"
    ):
        MESH_2X2.compute_kform_direct_dof_map(specs)


@pytest.mark.parametrize("family", [BasisType.LEGENDRE, BasisType.BERNSTEIN])
def test_any_basis_family_is_accepted(family: BasisType) -> None:
    """The elimination pairs inner products, so it needs no nodes at all."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(specs_of(MESH_2X2, 2, family=family))

    assert dof_map.global_dof_count == 25
    assert np.unique(dof_map.entry_index).size == dof_map.global_dof_count
    assert np.all(np.isfinite(dof_map.entry_value))


def test_a_mixed_family_mesh_is_accepted() -> None:
    """Different families on different elements share the same Legendre windows."""
    specs = specs_of(MESH_2X2, 2)
    specs[0] = KFormSpecs(
        0,
        FunctionSpace(
            BasisSpecs(BasisType.LEGENDRE, 2),
            BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 2),
        ),
    )
    specs[3] = KFormSpecs(
        0,
        FunctionSpace(
            BasisSpecs(BasisType.LAGRANGE_GAUSS, 2),
            BasisSpecs(BasisType.BERNSTEIN, 2),
        ),
    )

    dof_map = MESH_2X2.compute_kform_direct_dof_map(specs)

    assert np.unique(dof_map.entry_index).size == dof_map.global_dof_count
    assert np.all(np.isfinite(dof_map.entry_value))
    assert dof_map.global_dof_count <= dof_map.element_dof_count


def test_order_zero_basis_is_rejected() -> None:
    """An axis without functions has no test space: the object is degenerate."""
    space = uniform_space(2, 0)
    specs = [KFormSpecs(0, space) for _ in range(MESH_2X2.element_count)]

    with pytest.raises(ValueError, match="needs a positive basis order"):
        MESH_2X2.compute_kform_direct_dof_map(specs)


def test_elements_may_disagree_on_the_basis_order() -> None:
    """Differing element orders are accepted: the object takes the minimum."""
    specs = specs_of(MESH_2X2, 2)
    specs[0] = KFormSpecs(
        0,
        FunctionSpace(
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 3),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
        ),
    )

    direct = MESH_2X2.compute_kform_direct_dof_map(specs)

    assert direct.element_dof_count == 39
    assert direct.global_dof_count <= direct.element_dof_count
    assert direct.entry_offsets.shape == (direct.element_dof_count + 1,)
    assert int(direct.entry_offsets[-1]) == direct.entry_count
    assert np.all(direct.entry_index >= 0)
    assert np.all(direct.entry_index < direct.global_dof_count)
    # Every global DoF is reached, so no row or column of the system is left empty.
    assert len(np.unique(direct.entry_index)) == direct.global_dof_count


def test_registries_are_type_checked() -> None:
    """A registry of the wrong type is rejected before any work is done."""
    with pytest.raises(TypeError):
        MESH_2X2.compute_kform_direct_dof_map(
            specs_of(MESH_2X2, 1),
            basis_registry=object(),  # type: ignore[arg-type]
        )


def test_map_cannot_be_instantiated_directly() -> None:
    """The map only comes out of the mesh method."""
    with pytest.raises(TypeError, match="DirectDofMap cannot be instantiated directly"):
        DirectDofMap()
