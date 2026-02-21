"""Check coordinate mappings work as expected."""

import numpy as np
import pytest
from interplib._interp import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    SpaceMap,
    transform_contravariant_to_target,
)
from interplib.enum_type import BasisType

_TEST_ORDERS = (1, 2, 5, 10)


@pytest.mark.parametrize("int_order", _TEST_ORDERS)
@pytest.mark.parametrize("basis_order", _TEST_ORDERS)
@pytest.mark.parametrize("basis_type", BasisType)
def test_coord_1d(int_order: int, basis_order: int, basis_type: BasisType) -> None:
    """Check that coordinate as a function of 1 variable works."""
    rng = np.random.default_rng(2198)

    int_spec = IntegrationSpecs(int_order, method="gauss-lobatto")
    int_space = IntegrationSpace(int_spec)

    b_spec = BasisSpecs(basis_type, basis_order)
    b_space = FunctionSpace(b_spec)

    dofs = DegreesOfFreedom(b_space)
    dofs.values = rng.random(dofs.values.shape)

    coord_map = CoordinateMap(dofs, int_space)

    assert np.all(coord_map.values == dofs.reconstruct_at_integration_points(int_space))
    assert np.all(
        coord_map.gradient(0)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[0])
    )


_TEST_ORDERS_2D = ((1, 1), (2, 3), (10, 3), (10, 10))
_TEST_BASIS_2D = (
    (BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (BasisType.LAGRANGE_UNIFORM, BasisType.LAGRNAGE_GAUSS),
    (BasisType.LEGENDRE, BasisType.LAGRNAGE_GAUSS_LOBATTO),
)


@pytest.mark.parametrize(("int_order_1", "int_order_2"), _TEST_ORDERS_2D)
@pytest.mark.parametrize(("basis_order_1", "basis_order_2"), _TEST_ORDERS_2D)
@pytest.mark.parametrize(("basis_type_1", "basis_type_2"), _TEST_BASIS_2D)
def test_coord_2d(
    int_order_1: int,
    basis_order_1: int,
    basis_type_1: BasisType,
    int_order_2: int,
    basis_order_2: int,
    basis_type_2: BasisType,
) -> None:
    """Check that coordinate as a function of 1 variable works."""
    rng = np.random.default_rng(2198)

    int_space = IntegrationSpace(
        IntegrationSpecs(int_order_1, method="gauss-lobatto"),
        IntegrationSpecs(int_order_2, method="gauss-lobatto"),
    )

    b_space = FunctionSpace(
        BasisSpecs(basis_type_1, basis_order_1), BasisSpecs(basis_type_2, basis_order_2)
    )

    dofs = DegreesOfFreedom(b_space)
    dofs.values = rng.random(dofs.values.shape)

    coord_map = CoordinateMap(dofs, int_space)

    assert np.all(coord_map.values == dofs.reconstruct_at_integration_points(int_space))
    assert np.all(
        coord_map.gradient(0)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[0])
    )
    assert np.all(
        coord_map.gradient(1)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[1])
    )


_TEST_ORDERS_3D = (
    (1, 1, 2),
    (2, 3, 1),
    (10, 3, 4),
)
_TEST_BASIS_3D = (
    (BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.LAGRNAGE_GAUSS_LOBATTO),
    (BasisType.LAGRANGE_UNIFORM, BasisType.LAGRNAGE_GAUSS, BasisType.LEGENDRE),
    (BasisType.LEGENDRE, BasisType.LAGRNAGE_GAUSS_LOBATTO, BasisType.LAGRANGE_UNIFORM),
)


@pytest.mark.parametrize(("int_order_1", "int_order_2", "int_order_3"), _TEST_ORDERS_3D)
@pytest.mark.parametrize(
    ("basis_order_1", "basis_order_2", "basis_order_3"), _TEST_ORDERS_3D
)
@pytest.mark.parametrize(("basis_type_1", "basis_type_2", "basis_type_3"), _TEST_BASIS_3D)
def test_coord_3d(
    int_order_1: int,
    basis_order_1: int,
    basis_type_1: BasisType,
    int_order_2: int,
    basis_order_2: int,
    basis_type_2: BasisType,
    int_order_3: int,
    basis_order_3: int,
    basis_type_3: BasisType,
) -> None:
    """Check that coordinate as a function of 3 variable works."""
    rng = np.random.default_rng(2198)

    int_space = IntegrationSpace(
        IntegrationSpecs(int_order_1, method="gauss-lobatto"),
        IntegrationSpecs(int_order_2, method="gauss-lobatto"),
        IntegrationSpecs(int_order_3, method="gauss-lobatto"),
    )

    b_space = FunctionSpace(
        BasisSpecs(basis_type_1, basis_order_1),
        BasisSpecs(basis_type_2, basis_order_2),
        BasisSpecs(basis_type_3, basis_order_3),
    )

    dofs = DegreesOfFreedom(b_space)
    dofs.values = rng.random(dofs.values.shape)

    coord_map = CoordinateMap(dofs, int_space)

    assert np.all(coord_map.values == dofs.reconstruct_at_integration_points(int_space))
    assert np.all(
        coord_map.gradient(0)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[0])
    )
    assert np.all(
        coord_map.gradient(1)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[1])
    )
    assert np.all(
        coord_map.gradient(2)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[2])
    )


@pytest.mark.parametrize("n_int", (1, 2, 4))
@pytest.mark.parametrize("n_b", (2, 4))
@pytest.mark.parametrize("btype", BasisType)
def test_space_map_2_to_2(n_int: int, n_b: int, btype: BasisType) -> None:
    """Test that a 2D -> 2D space map works."""
    # Create the integration space
    int_space = IntegrationSpace(IntegrationSpecs(n_int + 1), IntegrationSpecs(n_int - 1))
    # Create the function space
    func_space = FunctionSpace(BasisSpecs(btype, n_b - 1), BasisSpecs(btype, n_b + 1))
    # Create the DoFs for coordinates
    dofs_1 = DegreesOfFreedom(func_space)
    dofs_2 = DegreesOfFreedom(func_space)
    # Set DoFs to random values for fun!
    rng = np.random.default_rng(1 + n_int**2 + n_b * 3)
    dofs_1.values = rng.random(dofs_1.values.shape)
    dofs_2.values = rng.random(dofs_2.values.shape)
    # Create the coordinate maps
    map_1 = CoordinateMap(dofs_1, int_space)
    map_2 = CoordinateMap(dofs_2, int_space)
    det_real = map_1.gradient(0) * map_2.gradient(1) - map_1.gradient(1) * map_2.gradient(
        0
    )
    # Make the space map
    print("Crashing this code...", end="")
    smap = SpaceMap(map_1, map_2)
    print(" with no survivors!")
    # Check that the determinant checks out
    det_smap = smap.determinant
    assert pytest.approx(det_smap) == det_real


@pytest.mark.parametrize("n_int", (1, 2, 4))
@pytest.mark.parametrize("n_b", (2, 4))
@pytest.mark.parametrize("btype", BasisType)
def test_space_map_1_to_3(n_int: int, n_b: int, btype: BasisType) -> None:
    """Test that a 1D -> 3D space map works.

    This is equivalent to having a curve in 1D space.
    """
    # Create the integration space
    int_space = IntegrationSpace(IntegrationSpecs(n_int))
    # Create the function space
    func_space = FunctionSpace(BasisSpecs(btype, n_b))
    # Create the DoFs for coordinates
    dofs_1 = DegreesOfFreedom(func_space)
    dofs_2 = DegreesOfFreedom(func_space)
    dofs_3 = DegreesOfFreedom(func_space)
    # Set DoFs to random values for fun!
    rng = np.random.default_rng(1 + n_int**2 + n_b * 3)
    dofs_1.values = rng.random(dofs_1.values.shape)
    dofs_2.values = rng.random(dofs_2.values.shape)
    dofs_3.values = rng.random(dofs_3.values.shape)
    # Create the coordinate maps
    map_1 = CoordinateMap(dofs_1, int_space)
    map_2 = CoordinateMap(dofs_2, int_space)
    map_3 = CoordinateMap(dofs_3, int_space)
    det_real = np.sqrt(
        map_1.gradient(0) ** 2 + map_2.gradient(0) ** 2 + map_3.gradient(0) ** 2
    )
    # Make the space map
    print("Crashing this code...", end="")
    smap = SpaceMap(map_1, map_2, map_3)
    print(" with no survivors!")
    # Check that the determinant checks out
    det_smap = smap.determinant
    assert pytest.approx(det_smap) == det_real


@pytest.mark.parametrize(("int_order_1", "int_order_2", "int_order_3"), _TEST_ORDERS_3D)
@pytest.mark.parametrize(
    ("basis_order_1", "basis_order_2", "basis_order_3"), _TEST_ORDERS_3D
)
@pytest.mark.parametrize(("basis_type_1", "basis_type_2", "basis_type_3"), _TEST_BASIS_3D)
def test_contravariant_3d(
    int_order_1: int,
    basis_order_1: int,
    basis_type_1: BasisType,
    int_order_2: int,
    basis_order_2: int,
    basis_type_2: BasisType,
    int_order_3: int,
    basis_order_3: int,
    basis_type_3: BasisType,
) -> None:
    """Check that contravariant components are correctly transformed."""
    rng = np.random.default_rng(2198)

    int_space = IntegrationSpace(
        IntegrationSpecs(int_order_1, method="gauss-lobatto"),
        IntegrationSpecs(int_order_2, method="gauss-lobatto"),
        IntegrationSpecs(int_order_3, method="gauss-lobatto"),
    )

    b_space = FunctionSpace(
        BasisSpecs(basis_type_1, basis_order_1),
        BasisSpecs(basis_type_2, basis_order_2),
        BasisSpecs(basis_type_3, basis_order_3),
    )

    dofs_x = DegreesOfFreedom(b_space)
    dofs_x.values = rng.random(dofs_x.values.shape)
    dofs_y = DegreesOfFreedom(b_space)
    dofs_y.values = rng.random(dofs_y.values.shape)
    dofs_z = DegreesOfFreedom(b_space)
    dofs_z.values = rng.random(dofs_z.values.shape)

    space_map = SpaceMap(
        CoordinateMap(dofs_x, int_space),
        CoordinateMap(dofs_y, int_space),
        CoordinateMap(dofs_z, int_space),
    )

    dofs_vx = DegreesOfFreedom(b_space)
    dofs_vx.values = rng.random(dofs_vx.values.shape)
    dofs_vy = DegreesOfFreedom(b_space)
    dofs_vy.values = rng.random(dofs_vy.values.shape)
    dofs_vz = DegreesOfFreedom(b_space)
    dofs_vz.values = rng.random(dofs_vz.values.shape)

    vx = dofs_vx.reconstruct_at_integration_points(space_map.integration_space)
    vy = dofs_vy.reconstruct_at_integration_points(space_map.integration_space)
    vz = dofs_vz.reconstruct_at_integration_points(space_map.integration_space)
    components = np.array((vx, vy, vz))

    contravariant = transform_contravariant_to_target(space_map, components)

    manual_contravariant = np.zeros_like(components)
    flat_contravariant_mat = np.reshape(
        space_map.inverse_map, (-1, *space_map.inverse_map.shape[-2:])
    )
    flat_components = np.reshape(components, (3, -1))
    flat_contravariant = np.reshape(manual_contravariant, (3, -1))
    for i_pt in range(vx.size):
        mat = flat_contravariant_mat[i_pt, :, :]
        vec_in = flat_components[:, i_pt]
        vec_out = mat.T @ vec_in
        flat_contravariant[:, i_pt] = vec_out
    manual_contravariant = np.reshape(flat_contravariant, contravariant.shape)

    assert pytest.approx(manual_contravariant) == contravariant


if __name__ == "__main__":
    test_contravariant_3d(
        3, 4, BasisType.BERNSTEIN, 3, 4, BasisType.BERNSTEIN, 3, 4, BasisType.BERNSTEIN
    )
