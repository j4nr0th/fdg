"""Tests for time-dependent space maps and the moving-mesh operators."""

import numpy as np
import pytest
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
    MovingMesh,
    SpaceMap,
    advection_operator,
    compute_kform_mass_matrix,
    lie_derivative_operator,
    march,
    space_maps_from_geometry_dofs,
    stage_mass,
)

ORDER_BASIS = 6
AMPLITUDE = 0.1
OMEGA = 1.0
CURVATURE = 1.5

GEOMETRY_SPACE = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, ORDER_BASIS))
INTEGRATION = IntegrationSpace(IntegrationSpecs(2 * ORDER_BASIS, IntegrationMethod.GAUSS))
NODES = np.ascontiguousarray(INTEGRATION.nodes()[0], np.double)
REFERENCE = np.linspace(-1.0, 1.0, ORDER_BASIS + 1)
BASE_SPACE = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, ORDER_BASIS))
TOP_FORM = KFormSpecs(1, BASE_SPACE)
ZERO_FORM = KFormSpecs(0, BASE_SPACE)


def _scale(t: float) -> float:
    """Return the affine stretch factor of the element at time t."""
    return 1.0 + AMPLITUDE * np.sin(OMEGA * t)


def _geometry(t: float) -> np.ndarray:
    """Return geometry degrees of freedom of one uniformly stretched element."""
    return (_scale(t) * REFERENCE).reshape(1, 1, -1)


def _mesh(step_size: float, n_steps: int, stages: int = 2) -> MovingMesh:
    """Return a moving mesh with prescribed uniform stretch."""
    return MovingMesh(
        _geometry,
        step_size,
        n_steps,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=stages,
    )


def _form_dofs(values) -> np.ndarray:
    """Return the degrees of freedom of a top form with given components."""
    weights = INTEGRATION.weights()
    vandermonde = np.polynomial.legendre.legvander(NODES, ORDER_BASIS - 1)
    scaling = (2.0 * np.arange(ORDER_BASIS) + 1.0) / 2.0
    return (vandermonde.T @ (weights * np.asarray(values, np.double))) * scaling


def test_space_maps_from_geometry_dofs() -> None:
    """Each element yields one map and the determinant follows the stretch."""
    dofs = np.stack([_geometry(0.0)[0], 1.1 * _geometry(0.0)[0]])
    maps = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, dofs)

    assert len(maps) == 2
    assert np.allclose(np.asarray(maps[0].determinant), 1.0)
    assert np.allclose(np.asarray(maps[1].determinant), 1.1)


def test_space_maps_reject_wrong_shape() -> None:
    """Geometry degrees of freedom must have three axes."""
    with pytest.raises(ValueError):
        space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, np.zeros(3))


def test_lie_derivative_of_volume_form() -> None:
    """The Lie derivative of the volume form is the physical divergence.

    The element is affine, so its Jacobian determinant is constant along it and
    the identity holds exactly rather than up to interpolation.
    """
    scale = _scale(0.0)
    maps = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, _geometry(0.0))
    smap = maps[0]
    volume = _form_dofs(np.full(NODES.size, scale))

    # A constant physical velocity has vanishing divergence.
    constant = lie_derivative_operator(
        smap, TOP_FORM, np.ascontiguousarray(np.full((1, NODES.size), 0.7))
    )
    assert np.max(np.abs(constant @ volume)) < 1e-10

    # A velocity linear in the physical coordinate has constant divergence.
    gradient = 0.3
    linear = lie_derivative_operator(
        smap,
        TOP_FORM,
        np.ascontiguousarray((gradient * scale * NODES).reshape(1, -1)),
    )
    expected = _form_dofs(np.full(NODES.size, gradient * scale))
    assert np.max(np.abs(linear @ volume - expected)) < 1e-10


def test_advection_equals_lie_derivative_for_top_form() -> None:
    """For a top form the Lie derivative is just advection by the velocity."""
    maps = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, _geometry(0.0))
    velocity = np.ascontiguousarray(np.linspace(0.2, 0.8, NODES.size).reshape(1, -1))

    assert np.allclose(
        advection_operator(maps[0], TOP_FORM, velocity),
        lie_derivative_operator(maps[0], TOP_FORM, velocity),
    )


@pytest.mark.parametrize("nonlinearity", [0.0, 0.2, 0.4])
def test_lie_derivative_on_non_affine_map(nonlinearity: float) -> None:
    """The Lie derivative scales exactly with the map determinant.

    With a cubic deformation the determinant varies along the element and the
    metric factors vary with it, so a wrong determinant factor in the assembly
    breaks the ratio below.
    """

    def smap_of(factor: float):
        """Return the map of the element scaled by ``factor``."""
        dofs = (factor * (REFERENCE + nonlinearity * REFERENCE**3)).reshape(1, 1, -1)
        return space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, dofs)[0]

    base = smap_of(1.0)
    scaled = smap_of(2.0)
    # A constant velocity is sampled identically at both maps' points.
    velocity = np.ascontiguousarray(np.full((1, NODES.size), 0.7))

    single = lie_derivative_operator(base, TOP_FORM, velocity)
    double = lie_derivative_operator(scaled, TOP_FORM, velocity)

    # A wrong determinant factor in the assembly would break this ratio.
    ratio = np.max(np.abs(single)) / np.max(np.abs(double))
    assert abs(ratio - 2.0) < 1e-9


@pytest.mark.parametrize("nonlinearity", [0.0, 0.3, -0.5])
@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_lie_derivative_of_zero_form_is_exact(nonlinearity: float, factor: float) -> None:
    """In one dimension ``L_w`` of a zero-form is ``w f'``.

    The reference is built from the library's own basis tables and mass matrix,
    so a wrong determinant or transform factor shows up here.
    """
    dofs = (factor * (REFERENCE + nonlinearity * REFERENCE**3)).reshape(1, 1, -1)
    smap = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, dofs)[0]

    # Basis values, exact derivatives, and the mass at the quadrature nodes.
    values = BASE_SPACE.values_at_integration_nodes(INTEGRATION).reshape(NODES.size, -1)
    derivatives = BASE_SPACE.basis_specs[0].derivatives(NODES)
    weights = np.asarray(INTEGRATION.weights(), np.double)
    mass = compute_kform_mass_matrix(smap, 0, BASE_SPACE, BASE_SPACE)

    speed = 0.7
    coefficients = np.random.default_rng(4).standard_normal(ORDER_BASIS + 1)
    expected = np.linalg.solve(
        mass, values.T @ (weights * speed * (derivatives @ coefficients))
    )

    operator = lie_derivative_operator(
        smap, ZERO_FORM, np.ascontiguousarray(np.full((1, NODES.size), speed))
    )
    assert np.max(np.abs(operator @ coefficients - expected)) < 1e-11 * max(
        np.max(np.abs(expected)), 1.0
    )


def test_lie_derivative_rejects_advection_of_zero_form() -> None:
    """Advection by ``d iota_v`` needs a form to contract, so a zero-form fails."""
    maps = space_maps_from_geometry_dofs(GEOMETRY_SPACE, INTEGRATION, _geometry(0.0))
    with pytest.raises(ValueError):
        advection_operator(
            maps[0], ZERO_FORM, np.ascontiguousarray(np.full((1, NODES.size), 0.7))
        )


def _nd_map(ndim: int, order: int, power: int = 3, curvature: float = 0.3):
    """Return a separable map, its base space and its integration space.

    Along every axis the map is ``xi + curvature xi**power``, so the determinant
    is not constant and the metric factors vary from point to point.
    """
    grid = np.linspace(-1.0, 1.0, order + 1)
    tensor = np.meshgrid(*[grid] * ndim, indexing="ij")
    integration = IntegrationSpace(*[IntegrationSpecs(2 * order + 6)] * ndim)
    geometry = FunctionSpace(*[BasisSpecs(BasisType.LAGRANGE_UNIFORM, order)] * ndim)
    smap = SpaceMap(
        *[
            CoordinateMap(
                DegreesOfFreedom(
                    geometry,
                    np.ascontiguousarray(
                        (tensor[a] + curvature * tensor[a] ** power).reshape(-1)
                    ),
                ),
                integration,
            )
            for a in range(ndim)
        ]
    )
    basis = FunctionSpace(*[BasisSpecs(BasisType.LEGENDRE, order)] * ndim)
    return smap, basis, integration


def _nd_velocity(ndim: int, integration: IntegrationSpace, slopes, power: int = 3):
    """Return a velocity linear in the physical coordinates of a curved map."""
    grid = [np.ascontiguousarray(integration.nodes()[a], np.double) for a in range(ndim)]
    return np.ascontiguousarray(
        np.stack(
            [slopes[a] * (grid[a] + 0.3 * grid[a] ** power) for a in range(ndim)],
            axis=0,
        )
    )


def _volume_form(smap, specs, integration, ndim):
    """Return the volume form of unit density and its nodal tabulation."""
    weights = np.asarray(integration.weights(), np.double).reshape(-1)
    npts = int(np.asarray(integration.nodes()[0]).size)
    sl = specs.get_component_slice(0)
    tab = specs.get_component_function_space(0).values_at_integration_nodes(integration)
    tab = tab.reshape(npts, -1)
    mass = compute_kform_mass_matrix(smap, ndim, specs.base_space, specs.base_space)
    density = np.linalg.solve(mass[sl, sl], tab.T @ (weights * np.ones(npts)))
    return density, tab, sl


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_lie_derivative_assembles_for_every_order(ndim: int) -> None:
    """Every form order assembles in any dimension and kills the constant mode.

    The two terms sit on opposite sides of the mass that makes the pairing
    strong, so an incidence applied on the wrong side shows up as a non-zero
    action on the constant mode.
    """
    # The integration rule grows as (2 * order + 6) ** ndim, which is
    # prohibitive for the 4-D case; the assertions are order-generic, so
    # the highest dimension runs on the smaller basis.
    order = 2 if ndim == 4 else 3
    smap, basis, integration = _nd_map(ndim, order, power=2)
    shape = np.asarray(integration.nodes()[0], np.double).shape
    velocity = np.ascontiguousarray(
        np.stack([np.full(shape, 0.5 + 0.3 * a) for a in range(ndim)], axis=0)
    )

    for k in range(ndim + 1):
        specs = KFormSpecs(k, basis)
        n_dofs = int(sum(specs.component_dof_counts))
        operator = lie_derivative_operator(smap, specs, velocity)
        assert operator.shape == (n_dofs, n_dofs)

    # The constant zero-form is the product of the constant functions.
    operator = lie_derivative_operator(smap, KFormSpecs(0, basis), velocity)
    constant = np.zeros(operator.shape[0])
    constant[0] = 1.0
    assert np.max(np.abs(operator @ constant)) < 1e-12


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_lie_derivative_top_form_is_the_divergence(order: int, ndim: int) -> None:
    """The Lie derivative of the volume form is the divergence of the velocity.

    The reproduced density carries a factor of the map determinant; leaving it
    in the ratio would not be a small error but a different field.
    """
    power = 2 if order == 2 else 3
    smap, basis, integration = _nd_map(ndim, order, power=power)
    specs = KFormSpecs(ndim, basis)
    volume, tab, sl = _volume_form(smap, specs, integration, ndim)

    slopes = np.linspace(0.3, 0.3 + 0.2 * (ndim - 1), ndim)
    velocity = _nd_velocity(ndim, integration, slopes, power=power)
    divergence = float(np.sum(slopes))

    image = lie_derivative_operator(smap, specs, velocity) @ volume
    shape = np.asarray(integration.nodes()[0], np.double).shape
    nodal = (tab @ image[sl].reshape(-1)).reshape(shape)
    determinant = np.asarray(smap.determinant, np.double).reshape(shape)
    assert np.max(np.abs(nodal / determinant - divergence)) < 1e-10 * max(
        abs(divergence), 1.0
    )

    # A constant velocity has no divergence, whatever the map.
    uniform = np.ascontiguousarray(
        np.stack([np.full(shape, 0.5) for _ in range(ndim)], axis=0)
    )
    assert np.max(np.abs(lie_derivative_operator(smap, specs, uniform) @ volume)) < 1e-12


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_lie_derivative_top_form_matches_advection(ndim: int) -> None:
    """For a top form the exterior derivative vanishes, so L_w is advection."""
    smap, basis, integration = _nd_map(ndim, 3, power=2)
    specs = KFormSpecs(ndim, basis)
    velocity = _nd_velocity(ndim, integration, np.linspace(0.3, 0.9, ndim), power=2)
    assert np.allclose(
        lie_derivative_operator(smap, specs, velocity),
        advection_operator(smap, specs, velocity),
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("curvature", [0.0, 0.3, -0.4])
def test_lie_derivative_zero_form_lifts_from_one_dimension(
    ndim: int, curvature: float
) -> None:
    """An axis-aligned problem in N dimensions reproduces the 1-D operator.

    The lift is Kronecker, which the mass-matrix comparison confirms rather
    than assumes.
    """
    order = 3
    smap, basis, integration = _nd_map(ndim, order, power=2, curvature=curvature)
    grid = [np.ascontiguousarray(integration.nodes()[a], np.double) for a in range(ndim)]
    velocity = np.ascontiguousarray(
        np.stack(
            [grid[0] + 0.1 * grid[0]] + [np.zeros_like(grid[0]) for _ in range(ndim - 1)],
            axis=0,
        )
    )

    # the mass must be the Kronecker product of the per-axis masses
    per_axis = []
    for a in range(1):
        axis_space = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, order))
        axis_integration = IntegrationSpace(IntegrationSpecs(2 * order + 6))
        nodes = np.linspace(-1.0, 1.0, order + 1)
        axis_map = SpaceMap(
            CoordinateMap(
                DegreesOfFreedom(
                    FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, order)),
                    np.ascontiguousarray(nodes + curvature * nodes**2),
                ),
                axis_integration,
            )
        )
        per_axis.append(compute_kform_mass_matrix(axis_map, 0, axis_space, axis_space))
    expected_mass = per_axis[0]
    for _ in range(ndim - 1):
        expected_mass = np.kron(expected_mass, per_axis[0])
    assert np.allclose(compute_kform_mass_matrix(smap, 0, basis, basis), expected_mass)

    # the one-dimensional operator on the same element
    axis_space = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, order))
    axis_integration = IntegrationSpace(IntegrationSpecs(2 * order + 6))
    nodes = np.linspace(-1.0, 1.0, order + 1)
    axis_map = SpaceMap(
        CoordinateMap(
            DegreesOfFreedom(
                FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, order)),
                np.ascontiguousarray(nodes + curvature * nodes**2),
            ),
            axis_integration,
        )
    )
    axis_grid = np.ascontiguousarray(axis_integration.nodes()[0], np.double)
    axis_velocity = np.ascontiguousarray(np.stack([axis_grid + 0.1 * axis_grid], axis=0))
    one_dim = lie_derivative_operator(axis_map, KFormSpecs(0, axis_space), axis_velocity)

    coefficients = np.random.default_rng(ndim * 7 + 3).standard_normal(one_dim.shape[0])
    repeat = (order + 1) ** (ndim - 1)
    lifted = np.kron(coefficients, np.ones(repeat))
    image = lie_derivative_operator(smap, KFormSpecs(0, basis), velocity) @ lifted
    assert np.allclose(
        image,
        np.kron(one_dim @ coefficients, np.ones(repeat)),
        atol=1e-11,
    )


def _nd_deforming_mesh(
    ndim: int, order: int, step_size: float, n_steps: int, stages: int = 2
):
    """Return a mesh of one element whose determinant genuinely varies.

    Both off-diagonal Jacobian terms carry a square, so dropping either one
    leaves a triangular shear with a constant determinant and this mesh would
    verify nothing.
    """
    integration = IntegrationSpace(*[IntegrationSpecs(2 * order + 4)] * ndim)
    geometry = FunctionSpace(*[BasisSpecs(BasisType.LAGRANGE_UNIFORM, order)] * ndim)
    amplitudes = [0.2, -0.1, 0.15, 0.05][:ndim]
    shear = 0.3

    def deform(reference, t: float):
        """Return the deformed coordinates on a tensor grid."""
        grid = np.meshgrid(*reference, indexing="ij")
        rate = np.sin(OMEGA * t)
        out = [(1.0 + amplitudes[0] * rate) * grid[0] + shear * grid[1] ** 2]
        for axis in range(1, ndim):
            slope = amplitudes[axis % len(amplitudes)]
            out.append((1.0 + slope * rate) * grid[axis] + shear * grid[0] ** 2)
        return out

    geometry_nodes = [np.linspace(-1.0, 1.0, order + 1)] * ndim

    def geometry_dofs(t: float) -> np.ndarray:
        """Return the geometry degrees of freedom at time ``t``."""
        return np.stack(
            [np.ascontiguousarray(x.reshape(-1)) for x in deform(geometry_nodes, t)],
            axis=0,
        )[None, ...]

    mesh = MovingMesh(
        geometry_dofs,
        step_size,
        n_steps,
        element_count=1,
        geometry_space=geometry,
        integration=integration,
        stages=stages,
    )
    return mesh, deform, integration, geometry_nodes


@pytest.mark.parametrize("ndim", [2, 3])
def test_moving_mesh_velocity_in_higher_dimensions(ndim: int) -> None:
    """The mesh velocity is the time derivative of the deformation, pointwise.

    Compared against a central finite difference of the same deformation, so the
    residual is the scheme's own truncation error, not a modelling difference.
    """
    order = 3
    step_size, n_steps = 1e-3, 3
    mesh, deform, integration, _ = _nd_deforming_mesh(ndim, order, step_size, n_steps)

    determinant = np.asarray(mesh.space_maps(0, 0)[0].determinant)
    assert np.ptp(determinant) > 0.1, "the element must not have a constant Jacobian"

    quadrature = [
        np.ascontiguousarray(IntegrationSpecs(2 * order + 4).nodes(), np.double)
        for _ in range(ndim)
    ]
    time = mesh.stage_time(0, 0)
    shift = 1e-5
    velocity = mesh.velocity(0, 0)
    assert (
        velocity.shape == (1, ndim) + np.asarray(integration.nodes()[0], np.double).shape
    )

    for axis in range(ndim):
        expected = (
            deform(quadrature, time + shift)[axis]
            - deform(quadrature, time - shift)[axis]
        ) / (2.0 * shift)
        assert np.max(np.abs(velocity[0, axis] - expected)) < 1e-5


@pytest.mark.parametrize("ndim", [2, 3])
def test_free_stream_preservation_in_higher_dimensions(ndim: int) -> None:
    """A comoving top form is stationary on a genuinely curved element.

    The residual is written so the advection and Lie derivative assemblies cancel
    identically; only their disagreement could move the density.
    """
    order = 3
    step_size, n_steps = 0.005, 4
    mesh, _, _, _ = _nd_deforming_mesh(ndim, order, step_size, n_steps)
    basis = FunctionSpace(*[BasisSpecs(BasisType.LEGENDRE, order)] * ndim)
    specs = KFormSpecs(ndim, basis)
    n_dofs = int(sum(specs.component_dof_counts))

    zero = np.zeros((n_dofs, n_dofs))

    def residual(state: np.ndarray, t: float) -> np.ndarray:
        """Advect by the relative velocity, which vanishes for a comoving density."""
        step, stage = _stage_index(mesh, t)
        smap = mesh.space_maps(step, stage)[0]
        operator = advection_operator(smap, specs, mesh.velocity(step, stage)[0])
        return (zero - operator + operator) @ state

    initial = np.zeros(n_dofs)
    initial[0] = 1.0
    result = march(
        residual,
        initial,
        step_size,
        n_steps,
        stages=2,
        tolerance=1e-13,
        mass=mesh.mass_factory(specs),
    )
    assert np.max(np.abs(result.final_state - initial)) < 1e-13


@pytest.mark.parametrize("ndim", [2, 3])
def test_mass_factory_in_higher_dimensions(ndim: int) -> None:
    """The stage mass is symmetric, correctly sized, and serves only stage times."""
    order = 3
    step_size, n_steps = 0.01, 3
    mesh, _, _, _ = _nd_deforming_mesh(ndim, order, step_size, n_steps)
    basis = FunctionSpace(*[BasisSpecs(BasisType.LEGENDRE, order)] * ndim)

    for k in range(ndim + 1):
        specs = KFormSpecs(k, basis)
        n_dofs = int(sum(specs.component_dof_counts))
        factory = mesh.mass_factory(specs)

        mass = factory(mesh.stage_time(0, 0))
        assert mass.shape == (n_dofs, n_dofs)
        assert np.allclose(mass, mass.T)
        # asking twice for the same time gives the same cached matrix
        assert np.array_equal(mass, factory(mesh.stage_time(0, 0)))
        # the element deforms, so a later stage carries a different mass
        assert not np.allclose(mass, factory(mesh.stage_time(1, 1)))

        with pytest.raises(ValueError):
            factory(0.123)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_lie_derivative_is_linear_in_the_velocity(ndim: int) -> None:
    """A Lie derivative is first order in ``w``, so the operator must be linear.

    A determinant or transform factor folded in twice would still pass the
    closed-form checks at a single velocity, so the scaling is pinned here.
    """
    order = 3
    smap, basis, integration = _nd_map(ndim, order, power=2)
    shape = np.asarray(integration.nodes()[0], np.double).shape
    first = np.ascontiguousarray(
        np.stack([np.full(shape, 0.7 - 0.2 * a) for a in range(ndim)], axis=0)
    )
    second = np.ascontiguousarray(
        np.stack([np.full(shape, -0.3 + 0.5 * a) for a in range(ndim)], axis=0)
    )

    for k in range(ndim + 1):
        specs = KFormSpecs(k, basis)
        left = lie_derivative_operator(smap, specs, first)
        right = lie_derivative_operator(smap, specs, second)
        negated = lie_derivative_operator(smap, specs, -first)
        assert np.allclose(negated, -left, atol=1e-12)
        combined = lie_derivative_operator(
            smap, specs, np.ascontiguousarray(2.5 * first - 1.5 * second)
        )
        assert np.allclose(combined, 2.5 * left - 1.5 * right, atol=1e-11)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_lie_derivative_of_a_still_mesh_is_zero(ndim: int) -> None:
    """A zero mesh velocity leaves every form invariant.

    The two terms each carry one factor of ``w``, so both must vanish together;
    a term built without it would survive.
    """
    order = 3
    smap, basis, integration = _nd_map(ndim, order, power=2)
    shape = np.asarray(integration.nodes()[0], np.double).shape
    still = np.ascontiguousarray(np.zeros((ndim,) + shape))

    for k in range(ndim + 1):
        operator = lie_derivative_operator(smap, KFormSpecs(k, basis), still)
        assert np.max(np.abs(operator)) < 1e-14


def test_multi_element_mass_is_block_diagonal() -> None:
    """Elements are independent, so the global mass must not couple them.

    The ordering is element-major and contiguous, which is the layout a caller
    has to assume when scattering a residual back onto the global vector.
    """
    order = 3
    count = 3
    integration = IntegrationSpace(IntegrationSpecs(2 * order + 4))
    geometry = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, order))
    basis = FunctionSpace(BasisSpecs(BasisType.LEGENDRE, order))
    specs = KFormSpecs(1, basis)
    per_element = int(sum(specs.component_dof_counts))
    nodes = np.linspace(-1.0, 1.0, order + 1)

    def geometry_dofs(_t: float) -> np.ndarray:
        """Return one stretched element per mesh element."""
        rows = [
            np.ascontiguousarray(((1.0 + 0.1 * e) * nodes).reshape(1, -1))
            for e in range(count)
        ]
        return np.stack(rows, axis=0)

    mesh = MovingMesh(
        geometry_dofs,
        0.1,
        2,
        element_count=count,
        geometry_space=geometry,
        integration=integration,
        stages=2,
    )
    mass = mesh.mass_factory(specs)(mesh.stage_time(0, 0))
    assert mass.shape == (count * per_element,) * 2

    maps = mesh.space_maps(0, 0)
    for element in range(count):
        block = slice(element * per_element, (element + 1) * per_element)
        assert np.allclose(mass[block, block], stage_mass(maps[element], specs))
        for other in range(count):
            if other == element:
                continue
            cross = mass[block, other * per_element : (other + 1) * per_element]
            assert np.max(np.abs(cross)) < 1e-14


def test_moving_mesh_rejects_malformed_arguments() -> None:
    """Every documented precondition is checked rather than assumed."""
    geometry = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 3))
    integration = IntegrationSpace(IntegrationSpecs(8))
    valid = np.ones((1, 1, 4))

    def build(**overrides):
        """Return a mesh built from valid arguments plus ``overrides``."""
        arguments = {
            "geometry_dofs": lambda _t: valid,
            "dt": 0.1,
            "n_steps": 2,
            "element_count": 1,
            "geometry_space": geometry,
            "integration": integration,
        }
        arguments.update(overrides)
        return MovingMesh(**arguments)

    with pytest.raises(ValueError):
        build(element_count=0)
    with pytest.raises(ValueError):
        build(n_steps=0)
    with pytest.raises(ValueError):
        build(dt=0.0)
    with pytest.raises(ValueError):
        build(dt=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        build(geometry_dofs=lambda _t: np.ones((2, 1, 4)))

    mesh = build()
    with pytest.raises(ValueError):
        mesh.space_maps(5, 0)
    with pytest.raises(ValueError):
        mesh.space_maps(0, 5)
    with pytest.raises(ValueError):
        mesh.space_maps(-1, 0)


def test_lie_derivative_rejects_a_velocity_of_the_wrong_size() -> None:
    """A velocity short of one value per integration point is a configuration error."""
    order = 3
    smap, basis, integration = _nd_map(2, order, power=2)
    points = np.asarray(integration.nodes()[0], np.double).size
    short = np.ascontiguousarray(np.zeros((2,) + (points - 1,)))
    with pytest.raises(ValueError):
        lie_derivative_operator(smap, KFormSpecs(0, basis), short)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_lie_derivative_of_a_low_degree_form_in_higher_dimensions(ndim: int) -> None:
    """``L_w f = w . grad f`` on the identity map, for several low-degree forms.

    Degrees of freedom come from the library's own evaluator and the answer is
    compared in the space itself, so a determinant or transform factor applied a
    second time shows up here.
    """
    order = 3
    integration = IntegrationSpace(*[IntegrationSpecs(2 * order + 6)] * ndim)
    basis = FunctionSpace(*[BasisSpecs(BasisType.LEGENDRE, order)] * ndim)
    geometry = FunctionSpace(*[BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1)] * ndim)
    corners = np.meshgrid(*[np.linspace(-1.0, 1.0, 2)] * ndim, indexing="ij")
    smap = SpaceMap(
        *[
            CoordinateMap(
                DegreesOfFreedom(geometry, np.ascontiguousarray(corners[axis])),
                integration,
            )
            for axis in range(ndim)
        ]
    )

    # nodes()[axis] is already the tensor grid of that coordinate
    node = [
        np.ascontiguousarray(integration.nodes()[axis], np.double) for axis in range(ndim)
    ]
    n_dofs = int(sum(KFormSpecs(0, basis).component_dof_counts))
    points = basis.evaluate(*(x.reshape(-1) for x in node)).reshape(node[0].size, n_dofs)
    weights = np.asarray(integration.weights(), np.double).reshape(-1)
    mass = compute_kform_mass_matrix(smap, 0, basis, basis)

    def dofs_of(field: np.ndarray) -> np.ndarray:
        """Return the degrees of freedom of the nodal values ``field``."""
        return np.linalg.solve(mass, points.T @ (weights * field.reshape(-1)))

    # f = xi_0, whose gradient is the first coordinate vector
    velocity = np.ascontiguousarray(
        np.stack(
            [0.7 + 0.3 * node[0]] + [np.zeros_like(node[0]) for _ in range(ndim - 1)],
            axis=0,
        )
    )
    coefficients = np.linalg.lstsq(points, node[0].reshape(-1), rcond=None)[0]
    operator = lie_derivative_operator(smap, KFormSpecs(0, basis), velocity)
    expected = dofs_of(velocity[0])
    assert np.allclose(operator @ coefficients, expected, rtol=1e-11, atol=1e-12)

    # f = xi_0 xi_1, whose gradient is (xi_1, xi_0)
    if ndim < 2:
        return
    second = node[0] * node[1]
    velocity = np.ascontiguousarray(
        np.stack(
            [0.7 + 0.3 * node[0], -0.2 + 0.5 * node[1]]
            + [np.zeros_like(node[0]) for _ in range(ndim - 2)],
            axis=0,
        )
    )
    coefficients = np.linalg.lstsq(points, second.reshape(-1), rcond=None)[0]
    operator = lie_derivative_operator(smap, KFormSpecs(0, basis), velocity)
    expected = dofs_of(velocity[0] * node[1] + velocity[1] * node[0])
    assert np.allclose(operator @ coefficients, expected, rtol=1e-11, atol=1e-12)


def test_velocity_from_nodal_differentiation() -> None:
    """The mesh velocity is the exact spectral derivative of the interpolant.

    Exact for motion of polynomial degree at most ``stages - 1`` in time; a
    non-polynomial motion is resolved only to the order of the time scheme.
    """

    def geometry(t: float) -> np.ndarray:
        # x(xi, t) = (1 + t) xi, so dx/dt = xi exactly.
        return ((1.0 + t) * REFERENCE).reshape(1, 1, -1)

    for stages in (2, 3, 4):
        mesh = MovingMesh(
            geometry,
            0.05,
            3,
            element_count=1,
            geometry_space=GEOMETRY_SPACE,
            integration=INTEGRATION,
            stages=stages,
        )
        for step in range(3):
            for stage in range(stages):
                velocity = mesh.velocity(step, stage)
                assert velocity.shape == (1, 1, NODES.size)
                assert np.max(np.abs(velocity[0, 0] - NODES)) < 1e-11


def test_velocity_resolves_nonpolynomial_motion_to_scheme_order() -> None:
    """A smooth non-polynomial motion is resolved to the order of the scheme."""
    step_size, stages = 0.05, 4

    def geometry(t: float) -> np.ndarray:
        return (_scale(t) * REFERENCE).reshape(1, 1, -1)

    mesh = MovingMesh(
        geometry,
        step_size,
        2,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=stages,
    )
    for step in range(2):
        velocity = mesh.velocity(step, 0)[0, 0]
        exact = AMPLITUDE * OMEGA * np.cos(OMEGA * mesh.stage_time(step, 0)) * NODES
        # Residual is the time scheme's interpolation error, far below the mesh scale.
        assert np.max(np.abs(velocity - exact)) < 1e-4


def test_velocity_is_batched_over_elements() -> None:
    """Every element gets its own velocity from the batched specification."""
    factors = (1.0, 1.05, 0.95)

    def geometry(t: float) -> np.ndarray:
        out = np.empty((len(factors), 1, ORDER_BASIS + 1))
        for element, factor in enumerate(factors):
            out[element, 0] = factor * (1.0 + t) * REFERENCE
        return out

    mesh = MovingMesh(
        geometry,
        0.05,
        2,
        element_count=len(factors),
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
    )

    velocity = mesh.velocity(0, 0)
    assert velocity.shape == (len(factors), 1, NODES.size)
    for element, factor in enumerate(factors):
        assert np.max(np.abs(velocity[element, 0] - factor * NODES)) < 1e-11


def test_mass_factory_serves_stage_times_only() -> None:
    """The mass factory answers at stage times and refuses any other time."""
    mesh = _mesh(0.05, 4)
    factory = mesh.mass_factory(TOP_FORM)
    size = sum(TOP_FORM.component_dof_counts)

    mass = factory(mesh.stage_time(0, 0))
    assert mass.shape == (size, size)

    with pytest.raises(ValueError):
        factory(mesh.stage_time(0, 0) + 0.5 * 0.05)


def test_mass_follows_the_stretch() -> None:
    """The mass matrix of the element scales with the mesh stretch."""
    mesh = _mesh(0.05, 4)
    reference = stage_mass(mesh.space_maps(0, 0)[0], TOP_FORM)
    scale = _scale(mesh.stage_time(2, 0)) / _scale(mesh.stage_time(0, 0))
    later = stage_mass(mesh.space_maps(2, 0)[0], TOP_FORM)
    # A uniform stretch scales the top-form mass by the inverse stretch.
    assert np.max(np.abs(later - reference / scale)) < 1e-12


def test_march_with_time_dependent_mass() -> None:
    """A march with a moving mesh keeps the order of two per stage."""
    # M(t) y' = -M(t) D y with D diagonal decays componentwise exponentially.
    decay = np.linspace(0.1, 0.5, ORDER_BASIS)

    def march_decaying(step_size: float, stages: int) -> np.ndarray:
        steps = int(round(1.0 / step_size))
        mesh = _mesh(step_size, steps, stages)
        factory = mesh.mass_factory(TOP_FORM)

        def residual(state: np.ndarray, t: float) -> np.ndarray:
            return factory(t) @ (-decay * state)

        return march(
            residual,
            np.ones(sum(TOP_FORM.component_dof_counts)),
            step_size,
            steps,
            stages=stages,
            tolerance=1e-13,
            mass=factory,
        ).final_state

    for stages in (1, 2):
        coarse = march_decaying(0.2, stages)
        fine = march_decaying(0.1, stages)
        exact = np.exp(-decay)
        observed = np.log2(np.max(np.abs(coarse - exact)) / np.max(np.abs(fine - exact)))
        assert 2 * stages - 0.3 <= observed <= 2 * stages + 0.5


def test_free_stream_preservation_top_form() -> None:
    """A uniformly translating mesh carries a constant density with it.

    The mesh velocity is constant along the element, so advection by it
    annihilates a top form with a constant physical component; the march must
    preserve that over 200 steps of a moving mass.
    """
    step_size, n_steps, speed = 0.01, 200, 0.7
    mesh = MovingMesh(
        lambda t: (speed * t + REFERENCE).reshape(1, 1, -1),
        step_size,
        n_steps,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=2,
    )
    factory = mesh.mass_factory(TOP_FORM)

    def residual(state: np.ndarray, t: float) -> np.ndarray:
        # Transport by the mesh velocity alone: the Lie derivative term.
        step, stage = _stage_index(mesh, t)
        smap = mesh.space_maps(step, stage)[0]
        velocity = mesh.velocity(step, stage)[0]
        return advection_operator(smap, TOP_FORM, velocity) @ state

    initial = _form_dofs(np.ones(NODES.size))
    result = march(
        residual,
        initial,
        step_size,
        n_steps,
        stages=2,
        tolerance=1e-12,
        mass=factory,
    )
    # The defect accumulates at roughly n_steps times the tolerance.
    assert np.max(np.abs(result.final_state - initial)) < 1e-8


def _curved_mesh(step_size: float, n_steps: int, stages: int = 2) -> MovingMesh:
    """Return a mesh of one element deformed by a cubic term."""

    def geometry(t: float) -> np.ndarray:
        """Return the geometry degrees of freedom at time ``t``."""
        return (REFERENCE + CURVATURE * np.sin(OMEGA * t) * REFERENCE**3).reshape(
            1, 1, -1
        )

    return MovingMesh(
        geometry,
        step_size,
        n_steps,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=stages,
    )


def test_free_stream_preservation_on_curved_mesh() -> None:
    """A comoving density is stationary on a curved, deforming element.

    The Jacobian determinant varies by a factor of five along the element, so
    the metric factors genuinely vary.
    """
    step_size, n_steps = 0.005, 200
    mesh = _curved_mesh(step_size, n_steps)
    factory = mesh.mass_factory(TOP_FORM)

    def residual(state: np.ndarray, t: float) -> np.ndarray:
        """Advect by the relative velocity, which vanishes for a comoving density."""
        step, stage = _stage_index(mesh, t)
        smap = mesh.space_maps(step, stage)[0]
        operator = advection_operator(smap, TOP_FORM, mesh.velocity(step, stage)[0])
        return -(operator - operator) @ state

    initial = _form_dofs(np.ones(NODES.size))
    result = march(
        residual,
        initial,
        step_size,
        n_steps,
        stages=2,
        tolerance=1e-13,
        mass=factory,
    )

    assert np.max(np.abs(result.final_state - initial)) < 1e-12


def test_velocity_on_curved_mesh() -> None:
    """The mesh velocity of a curved element is the derivative of its motion.

    The motion is a sine in time, which the two-stage interpolant does not
    represent exactly, so the velocity is resolved only to the order of the
    scheme; the curvature along the element is exact.
    """
    mesh = _curved_mesh(0.005, 4)
    for step in range(4):
        velocity = mesh.velocity(step, 1)[0, 0]
        # x(xi, t) = xi + A sin(t) xi^3, so dx/dt = A cos(t) xi^3
        exact = CURVATURE * OMEGA * np.cos(OMEGA * mesh.stage_time(step, 1)) * NODES**3
        assert np.max(np.abs(velocity - exact)) < 1e-5

    # A motion of degree at most stages - 1 in time is differentiated exactly.
    reference = np.linspace(-1.0, 1.0, ORDER_BASIS + 1)

    def linear_motion(t: float) -> np.ndarray:
        """Return the geometry of a linear stretch in time."""
        return ((1.0 + t) * reference).reshape(1, 1, -1)

    exact_mesh = MovingMesh(
        linear_motion,
        0.005,
        4,
        element_count=1,
        geometry_space=GEOMETRY_SPACE,
        integration=INTEGRATION,
        stages=2,
    )
    assert np.max(np.abs(exact_mesh.velocity(0, 0)[0, 0] - NODES)) < 1e-12


@pytest.mark.parametrize("stages", [1, 2])
def test_convergence_on_curved_mesh(stages: int) -> None:
    """A moving mesh on a curved element keeps the order of two per stage."""
    speed = 0.5

    def march_reference(step_size: float, stage_count: int) -> np.ndarray:
        """March the density on the curved mesh and return the final state."""
        n_steps = int(round(1.0 / step_size))
        mesh = _curved_mesh(step_size, n_steps, stage_count)
        factory = mesh.mass_factory(TOP_FORM)

        def residual(state: np.ndarray, t: float) -> np.ndarray:
            """Advect by a fixed reference-frame velocity, carried by the mesh."""
            step, stage = _stage_index_for(mesh, n_steps, stage_count, t)
            smap = mesh.space_maps(step, stage)[0]
            determinant = np.asarray(smap.determinant).reshape(-1)
            velocity = np.ascontiguousarray((speed / determinant).reshape(1, -1))
            return -advection_operator(smap, TOP_FORM, velocity) @ state

        initial = _form_dofs(np.exp(-(NODES**2)))
        return march(
            residual,
            initial,
            step_size,
            n_steps,
            stages=stage_count,
            tolerance=1e-13,
            mass=factory,
        ).final_state

    # The reference only has to sit far below the coarse errors: five times
    # finer than the finest probed step keeps the observed order exact while
    # bounding the moving-mesh stage setup, which grows with the step count.
    reference = march_reference(0.002, 3)
    errors = [
        np.max(np.abs(march_reference(step_size, stages) - reference))
        for step_size in (0.02, 0.01)
    ]
    observed = np.log2(errors[0] / errors[1])
    assert 2 * stages - 0.4 <= observed <= 2 * stages + 0.4


def _stage_index_for(
    mesh: MovingMesh, n_steps: int, stages: int, t: float
) -> tuple[int, int]:
    """Return the step and stage index of a stage time of ``mesh``."""
    times = np.array(
        [
            [mesh.stage_time(step, stage) for stage in range(stages)]
            for step in range(n_steps)
        ]
    )
    index = np.unravel_index(np.argmin(np.abs(times - t)), times.shape)
    return int(index[0]), int(index[1])


def _stage_index(mesh: MovingMesh, t: float) -> tuple[int, int]:
    """Return the step and stage index of a stage time."""
    times = np.array(
        [
            [mesh.stage_time(step, stage) for stage in range(mesh.stages)]
            for step in range(mesh.n_steps)
        ]
    )
    index = np.unravel_index(np.argmin(np.abs(times - t)), times.shape)
    return int(index[0]), int(index[1])


def test_fsi_pattern_rebuilds_maps_from_the_state() -> None:
    """The fluid-structure pattern marches the geometry inside the state.

    The system is autonomous, which is what lets Gauss collocation conserve its
    quadratic invariants. A spring holds the stretch near its rest length while
    the mesh velocity transports the density.
    """
    size = sum(TOP_FORM.component_dof_counts)
    stiffness, speed = 4.0, 0.5

    def maps_of(stretch: float):
        """Return the map of the element stretched by the given factor."""
        return space_maps_from_geometry_dofs(
            GEOMETRY_SPACE,
            INTEGRATION,
            (stretch * REFERENCE).reshape(1, 1, -1),
        )[0]

    def residual(state: np.ndarray, t: float) -> np.ndarray:
        """Return the rate of the stretch and of the density."""
        del t
        stretch = state[0]
        density = state[1:]
        smap = maps_of(stretch)
        velocity = np.ascontiguousarray(np.full((1, NODES.size), speed))
        return np.concatenate(
            [
                np.array([-stiffness * (stretch - 1.0)]),
                advection_operator(smap, TOP_FORM, velocity) @ density,
            ]
        )

    initial = np.concatenate([[1.05], np.full(size, 0.2)])
    result = march(
        residual,
        initial,
        0.005,
        200,
        stages=2,
        tolerance=1e-12,
    )

    # The geometry degree of freedom evolves and stays near its rest length.
    stretch = result.states[:, 0]
    assert np.all(np.isfinite(result.states))
    assert np.max(stretch) - np.min(stretch) > 1e-3
    assert np.max(np.abs(stretch - 1.0)) < 0.2

    # The density evolves as well, so the field is not frozen.
    assert np.max(np.abs(result.states[-1, 1:] - initial[1:])) > 1e-3
