"""Check the direct element-to-global transfer of Mesh.compute_kform_direct_dof_map."""

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


def uniform_space(ndim: int, order: int) -> FunctionSpace:
    """Build a tensor space of the given order on every axis."""
    return FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, order) for _ in range(ndim))
    )


def specs_of(mesh: Mesh, order: int, kform_order: int = 0) -> list[KFormSpecs]:
    """One specification of the given k-form order per element of the mesh."""
    space = uniform_space(mesh.ndim, order)
    return [KFormSpecs(kform_order, space) for _ in range(mesh.element_count)]


def global_of_node(dof_map: DirectDofMap, side: int) -> dict[tuple[float, float], int]:
    """Map every node of the 2x2 grid to the global DoF the elements agree on."""
    numbered: dict[tuple[float, float], int] = {}
    for element in range(MESH_2X2.element_count):
        element_x, element_y = element % 2, element // 2
        for digit_x in range(side):
            for digit_y in range(side):
                local = int(dof_map.element_offsets[element]) + digit_x * side + digit_y
                entries = list(
                    range(
                        int(dof_map.entry_offsets[local]),
                        int(dof_map.entry_offsets[local + 1]),
                    )
                )
                assert len(entries) == 1
                entry = entries[0]
                # Equal basis orders make the transfer the identity: unit coefficients.
                assert dof_map.entry_value[entry] == pytest.approx(1.0)
                node = (
                    element_x + digit_x / (side - 1),
                    element_y + digit_y / (side - 1),
                )
                numbered[node] = int(dof_map.entry_index[entry])
    return numbered


@pytest.mark.parametrize("order", [1, 2, 3])
def test_scalar_map_numbers_every_node_once(order: int) -> None:
    """A scalar map is the nodal numbering of the grid, shared by the elements."""
    dof_map = MESH_2X2.compute_kform_direct_dof_map(specs_of(MESH_2X2, order))

    assert isinstance(dof_map, DirectDofMap)
    assert dof_map.global_dof_count == (2 * order + 1) ** 2
    assert dof_map.element_dof_count == 4 * (order + 1) ** 2
    assert dof_map.global_dof_count <= dof_map.element_dof_count

    numbered = global_of_node(dof_map, order + 1)
    assert len(numbered) == dof_map.global_dof_count
    assert len(set(numbered.values())) == dof_map.global_dof_count


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
    """A one-dimensional mesh of three intervals has seven nodes at order two."""
    dof_map = MESH_LINE.compute_kform_direct_dof_map(specs_of(MESH_LINE, 2))

    assert dof_map.global_dof_count == 3 * 2 + 1
    assert dof_map.element_dof_count == 3 * 3
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


def test_non_nodal_basis_is_rejected() -> None:
    """A Legendre or Bernstein basis has no nodes to transfer to."""
    space = FunctionSpace(*(BasisSpecs(BasisType.LEGENDRE, 2) for _ in range(2)))
    specs = [KFormSpecs(0, space) for _ in range(MESH_2X2.element_count)]

    with pytest.raises(ValueError, match="needs a nodal basis family"):
        MESH_2X2.compute_kform_direct_dof_map(specs)


def test_order_zero_basis_is_rejected() -> None:
    """An axis without functions cannot carry a degree of freedom."""
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
