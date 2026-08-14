"""Check coordinate mappings work as expected."""

import numpy as np
import pytest
from fdg import compute_kform_boundary_constraints
from fdg._fdg import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    SpaceMap,
    transform_contravariant_to_target,
)
from fdg.enum_type import BasisType

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


def test_space_map_boundary_extracts_from_source_dofs() -> None:
    """Boundary maps are reconstructed at the face, even when the volume grid omits it."""
    rng = np.random.default_rng(107)
    function_space = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, 3), BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2)
    )
    dofs = [DegreesOfFreedom(function_space) for _ in range(2)]
    for values in dofs:
        values.values = rng.random(values.values.shape)

    volume_integration = IntegrationSpace(
        IntegrationSpecs(4, method="gauss"), IntegrationSpecs(3, method="gauss-lobatto")
    )
    volume_map = SpaceMap(*(CoordinateMap(values, volume_integration) for values in dofs))

    lower = volume_map.boundary(0)
    upper = volume_map.boundary(0, True)
    assert lower.input_dimensions == 1
    assert lower.output_dimensions == 2
    assert lower.integration_space.orders == (3,)
    assert upper.integration_space.orders == (3,)

    face_integration = lower.integration_space
    for index, values in enumerate(dofs):
        expected_lower = values.plane_projection(
            0, -1.0
        ).reconstruct_at_integration_points(face_integration)
        expected_upper = values.plane_projection(
            0, +1.0
        ).reconstruct_at_integration_points(face_integration)
        np.testing.assert_allclose(lower.coordinate_map(index).values, expected_lower)
        np.testing.assert_allclose(upper.coordinate_map(index).values, expected_upper)

    custom_face_space = IntegrationSpace(IntegrationSpecs(2, method="gauss"))
    custom_lower = volume_map.boundary(0, False, custom_face_space)
    assert custom_lower.integration_space.orders == (2,)
    for index, values in enumerate(dofs):
        expected = values.plane_projection(0, -1.0).reconstruct_at_integration_points(
            custom_face_space
        )
        np.testing.assert_allclose(custom_lower.coordinate_map(index).values, expected)


def test_space_map_boundary_supports_zero_dimensional_map() -> None:
    """A 1D map can be restricted to a zero-dimensional point map."""
    function_space = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1))
    dofs = DegreesOfFreedom(function_space)
    int_space = IntegrationSpace(IntegrationSpecs(2, method="gauss"))
    space_map = SpaceMap(CoordinateMap(dofs, int_space))

    point_map = space_map.boundary(0)
    assert point_map.input_dimensions == 0
    assert point_map.integration_space.orders == ()


def test_space_map_boundary_provides_tangential_pullback() -> None:
    """A volume map supplies the pullback for its own restricted face map."""
    function_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    x_dofs = DegreesOfFreedom(function_space, [0.0, 0.0, 1.0, 1.0])
    y_dofs = DegreesOfFreedom(function_space, [0.0, 1.0, 0.0, 1.0])
    volume_space = IntegrationSpace(
        IntegrationSpecs(2, method="gauss"), IntegrationSpecs(2, method="gauss")
    )
    volume_map = SpaceMap(
        CoordinateMap(x_dofs, volume_space), CoordinateMap(y_dofs, volume_space)
    )
    face_space = IntegrationSpace(IntegrationSpecs(3, method="gauss"))

    face_map = volume_map.boundary(0, False, face_space)
    pullback = face_map.basis_transform(1)
    assert pullback.shape == (1, 2, 4)
    np.testing.assert_allclose(pullback[0, 0], 0.0)
    np.testing.assert_allclose(pullback[0, 1], 2.0)


def test_kform_boundary_constraints_python_wrapper() -> None:
    """The Python wrapper returns packed physical boundary constraint arrays."""
    volume_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    x_dofs = DegreesOfFreedom(volume_space, [0.0, 0.0, 1.0, 1.0])
    y_dofs = DegreesOfFreedom(volume_space, [0.0, 1.0, 0.0, 1.0])
    integration = IntegrationSpace(
        IntegrationSpecs(2, method="gauss"), IntegrationSpecs(2, method="gauss")
    )
    first_map = SpaceMap(
        CoordinateMap(x_dofs, integration), CoordinateMap(y_dofs, integration)
    )
    test_specs = KFormSpecs(0, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))
    element_specs = (KFormSpecs(0, volume_space), KFormSpecs(0, volume_space))
    mesh_collections = (
        np.array(
            [[0, 1], [1, 4], [3, 4], [0, 3], [1, 2], [2, 5], [4, 5]],
            dtype=np.uint64,
        ),
        np.array([[0, 3, 2, 1], [4, 1, 6, 5]], dtype=np.uint64),
    )

    row_offsets, components, local_dofs, coefficients = (
        compute_kform_boundary_constraints(
            test_specs,
            element_specs[0],
            first_map,
            mesh_collections,
            6,
            0,
            1,
        )
    )
    assert row_offsets.shape == (3,)
    assert components.shape == local_dofs.shape == coefficients.shape
    assert row_offsets[-1] == coefficients.size
    assert np.any(coefficients > 0)


def test_kform_boundary_constraints_python_one_form() -> None:
    """The wrapper handles tangential one-forms and mesh-derived orientations."""
    volume_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    x_dofs = DegreesOfFreedom(volume_space, [0.0, 0.0, 1.0, 1.0])
    y_dofs = DegreesOfFreedom(volume_space, [0.0, 1.0, 0.0, 1.0])
    integration = IntegrationSpace(
        IntegrationSpecs(2, method="gauss"), IntegrationSpecs(2, method="gauss")
    )
    first_map = SpaceMap(
        CoordinateMap(x_dofs, integration), CoordinateMap(y_dofs, integration)
    )
    test_specs = KFormSpecs(1, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))
    element_specs = (KFormSpecs(1, volume_space), KFormSpecs(1, volume_space))
    mesh_collections = (
        np.array(
            [[0, 1], [1, 4], [3, 4], [0, 3], [1, 2], [2, 5], [4, 5]],
            dtype=np.uint64,
        ),
        np.array([[0, 3, 2, 1], [4, 1, 6, 5]], dtype=np.uint64),
    )

    result = compute_kform_boundary_constraints(
        test_specs,
        element_specs[0],
        first_map,
        mesh_collections,
        6,
        0,
        1,
    )
    row_offsets, components, local_dofs, coefficients = result
    assert row_offsets.shape == (2,)
    assert row_offsets[-1] == 2
    assert np.all(components == 0)
    assert local_dofs.shape == coefficients.shape == (2,)


def test_kform_boundary_constraints_rejects_bad_mesh_collections() -> None:
    """Check that the wrapper rejects mesh collections that don't match the space map."""
    volume_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    dofs = DegreesOfFreedom(volume_space)
    integration = IntegrationSpace(IntegrationSpecs(2), IntegrationSpecs(2))
    space_map = SpaceMap(
        CoordinateMap(dofs, integration), CoordinateMap(dofs, integration)
    )
    test_specs = KFormSpecs(0, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))
    element_specs = (KFormSpecs(0, volume_space), KFormSpecs(0, volume_space))

    with pytest.raises(ValueError, match="mesh collections"):
        compute_kform_boundary_constraints(
            test_specs,
            element_specs[0],
            space_map,
            (np.zeros((1, 2), dtype=np.uint64),),
            4,
            0,
            0,
        )


def test_kform_boundary_constraints_python_three_dimensional_line() -> None:
    """A 3D element can generate constraints on a 1D mesh line."""
    volume_space = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    coordinates = (
        DegreesOfFreedom(volume_space, [0, 1, 0, 1, 0, 1, 0, 1]),
        DegreesOfFreedom(volume_space, [0, 0, 1, 1, 0, 0, 1, 1]),
        DegreesOfFreedom(volume_space, [0, 0, 0, 0, 1, 1, 1, 1]),
    )
    integration = IntegrationSpace(
        IntegrationSpecs(2, method="gauss"),
        IntegrationSpecs(2, method="gauss"),
        IntegrationSpecs(2, method="gauss"),
    )
    space_map = SpaceMap(*(CoordinateMap(dofs, integration) for dofs in coordinates))
    lines = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.uint64,
    )
    surfaces = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 9, 4, 8],
            [1, 10, 5, 9],
            [2, 11, 6, 10],
            [3, 8, 7, 11],
        ],
        dtype=np.uint64,
    )
    volumes = np.array([[5, 2, 0, 3, 4, 1]], dtype=np.uint64)
    test_specs = KFormSpecs(0, FunctionSpace(BasisSpecs(BasisType.LEGENDRE, 1)))
    element_spec = KFormSpecs(0, volume_space)

    row_offsets, components, local_dofs, coefficients = (
        compute_kform_boundary_constraints(
            test_specs,
            element_spec,
            space_map,
            (lines, surfaces, volumes),
            8,
            0,
            0,
        )
    )
    assert row_offsets.shape == (3,)
    assert row_offsets[-1] == 16
    assert components.shape == local_dofs.shape == coefficients.shape == (16,)


def _apply_boundary_rows(
    result: tuple[np.ndarray, ...], values: np.ndarray
) -> np.ndarray:
    """Apply packed boundary rows to one element's flattened DoFs."""
    row_offsets, _, local_dofs, coefficients = result
    return np.array(
        [
            np.dot(
                coefficients[row_offsets[row] : row_offsets[row + 1]],
                values[local_dofs[row_offsets[row] : row_offsets[row + 1]]],
            )
            for row in range(row_offsets.size - 1)
        ]
    )


def _constant_kform_values(specs: KFormSpecs) -> np.ndarray:
    """Return DoFs for a constant value in every k-form component."""
    return np.ones(int(np.sum(specs.component_dof_counts)))


def test_kform_boundary_constraints_continuity_2d_and_3d() -> None:
    """Check polynomial traces on 2D and 3D adjacent-element boundaries."""
    basis_2d = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    integration_2d = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))
    maps_2d = [
        SpaceMap(
            CoordinateMap(DegreesOfFreedom(basis_2d, x_values), integration_2d),
            CoordinateMap(DegreesOfFreedom(basis_2d, [-1, 1, -1, 1]), integration_2d),
        )
        for x_values in ([-1, -1, 0, 0], [0, 0, 1, 1])
    ]
    lines_2d = np.array(
        [[0, 1], [0, 2], [2, 3], [1, 3], [2, 4], [4, 5], [3, 5]],
        dtype=np.uint64,
    )
    elements_2d = np.array([[0, 1, 2, 3], [2, 4, 5, 6]], dtype=np.uint64)

    for face_dim, boundary_id in ((0, 2), (1, 2)):
        for order in range(face_dim + 1):
            test_specs = KFormSpecs(
                order,
                FunctionSpace(
                    *(BasisSpecs(BasisType.LEGENDRE, 1) for _ in range(face_dim))
                ),
            )
            element_specs = KFormSpecs(order, basis_2d)
            values = _constant_kform_values(element_specs)
            traces = [
                _apply_boundary_rows(
                    compute_kform_boundary_constraints(
                        test_specs,
                        element_specs,
                        element_map,
                        (lines_2d, elements_2d),
                        6,
                        element_id,
                        boundary_id,
                    ),
                    values,
                )
                for element_id, element_map in enumerate(maps_2d)
            ]
            np.testing.assert_allclose(traces[0], traces[1])

    basis_3d = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(3))
    )
    integration_3d = IntegrationSpace(*(IntegrationSpecs(3) for _ in range(3)))
    x_values_0 = [-1, -1, 1, 1, -1, -1, 1, 1]
    y_values_0 = [-1, 1, -1, 1, -1, 1, -1, 1]
    z_values = [-1, -1, -1, -1, 1, 1, 1, 1]
    maps_3d = [
        SpaceMap(
            *(
                CoordinateMap(DegreesOfFreedom(basis_3d, values), integration_3d)
                for values in values_set
            )
        )
        for values_set in (
            ([value + 1 for value in x_values_0], y_values_0, z_values),
            ([value + 1 for value in y_values_0], x_values_0, z_values),
        )
    ]
    lines_3d = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 8],
            [0, 8],
            [1, 9],
            [5, 10],
            [4, 11],
        ],
        dtype=np.uint64,
    )
    faces_3d = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 9, 4, 8],
            [1, 10, 5, 9],
            [2, 11, 6, 10],
            [3, 8, 7, 11],
            [0, 17, 12, 16],
            [9, 18, 13, 17],
            [4, 18, 14, 19],
            [8, 19, 15, 16],
            [12, 13, 14, 15],
        ],
        dtype=np.uint64,
    )
    elements_3d = np.array([[5, 2, 0, 3, 4, 1], [2, 6, 7, 10, 8, 9]], dtype=np.uint64)
    collections_3d = (lines_3d, faces_3d, elements_3d)

    for face_dim, boundary_id in ((0, 0), (1, 0), (2, 2)):
        for order in range(face_dim + 1):
            test_specs = KFormSpecs(
                order,
                FunctionSpace(
                    *(BasisSpecs(BasisType.LEGENDRE, 1) for _ in range(face_dim))
                ),
            )
            element_specs = KFormSpecs(order, basis_3d)
            values = _constant_kform_values(element_specs)
            results = [
                compute_kform_boundary_constraints(
                    test_specs,
                    element_specs,
                    element_map,
                    collections_3d,
                    12,
                    element_id,
                    boundary_id,
                )
                for element_id, element_map in enumerate(maps_3d)
            ]
            for result in results:
                row_offsets, components, local_dofs, coefficients = result
                assert row_offsets[-1] == coefficients.size
                assert components.shape == local_dofs.shape == coefficients.shape
                assert np.all(np.isfinite(coefficients))

            traces = [_apply_boundary_rows(result, values) for result in results]
            if face_dim < 2:
                np.testing.assert_allclose(traces[0], traces[1])


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
    flat_contravariant_mat = np.array(
        [
            [
                space_map.coordinate_map(imap).gradient(idim)
                for idim in range(space_map.input_dimensions)
            ]
            for imap in range(space_map.output_dimensions)
        ]
    ).reshape((space_map.output_dimensions, space_map.input_dimensions, -1))
    flat_components = np.reshape(components, (3, -1))
    flat_contravariant = np.reshape(manual_contravariant, (3, -1))
    for i_pt in range(vx.size):
        mat = flat_contravariant_mat[:, :, i_pt]
        vec_in = flat_components[:, i_pt]
        vec_out = mat @ vec_in
        flat_contravariant[:, i_pt] = vec_out
    manual_contravariant = np.reshape(flat_contravariant, contravariant.shape)

    assert pytest.approx(manual_contravariant) == contravariant
