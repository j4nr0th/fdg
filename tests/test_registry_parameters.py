"""Check that assembly entry points honour caller-supplied registries.

The module-state registries are only defaults: every parameterized entry
point must fetch quadrature rules and basis tables from the registries the
caller passes. A fetched rule stays visible in ``registry.usage()``, so a
fresh registry that is really consulted stops being empty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
from fdg import (
    DEFAULT_BASIS_REGISTRY,
    DEFAULT_INTEGRATION_REGISTRY,
    BasisRegistry,
    BasisSpecs,
    BoundaryCondition,
    BoundaryPair,
    FunctionSpace,
    IntegrationRegistry,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    SpaceMap,
    compute_kform_boundary_mass_matrices,
    compute_kform_boundary_trace_moments,
)
from fdg._fdg import (
    compute_boundary_space_map_factors,
    compute_kform_boundary_load,
)
from fdg.boundary_conditions import (
    _append_boundary_rows,
    _append_periodic_rows,
    _compute_kform_global_constraints,
    _restrict_map,
    _windowed_component_basis,
    _windowed_dual_values,
    compute_kform_global_constraints,
)
from fdg.enum_type import BasisType

from examples.plot_multi_element_laplace_continuity import (
    make_element_maps,
    make_mesh,
)

Order = int
Call = Callable[..., Any]


def _ones(*coordinates: np.ndarray) -> np.ndarray:
    """Constant unit datum used by the boundary loads and conditions."""
    return np.ones_like(coordinates[0])


def _setup() -> tuple[Any, list[Any], list[KFormSpecs]]:
    """Build the shared two-by-two mapped mesh with uniform-order tests."""
    mesh = make_mesh(2)
    maps = make_element_maps(2, 6)
    space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 2) for _ in range(2))
    )
    return mesh, maps, [KFormSpecs(0, space) for _ in maps]


def _first_boundary_face(mesh: Any) -> int:
    """ID of the first outer face that touches element zero."""
    for _, object_id, element_ids, _ in mesh.iterate_boundary(mesh.ndim - 1):
        if 0 in element_ids:
            return int(object_id)
    raise AssertionError("mesh has no boundary face on element zero")


def _paired_faces(mesh: Any) -> tuple[int, int]:
    """One lower/upper outer-face pair perpendicular to axis zero."""
    faces = [
        (int(object_id), orientations)
        for _, object_id, _, orientations in mesh.iterate_boundary(mesh.ndim - 1)
    ]
    lower = next(o for o, rec in faces if int(rec[0, 0]) == -1)
    upper = next(o for o, rec in faces if int(rec[0, 0]) == 1)
    return lower, upper


def _assert_same(expected: Any, actual: Any) -> None:
    """Compare nested results whose leaves are arrays or value-like specs."""
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(expected) == len(actual)
        for want, got in zip(expected, actual):
            _assert_same(want, got)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(expected) == len(actual)
        for want, got in zip(expected, actual):
            _assert_same(want, got)
        return
    if expected is None or actual is None:
        assert expected is actual
        return
    if isinstance(expected, KFormSpecs):
        assert isinstance(actual, KFormSpecs)
        assert (expected.order, tuple(expected.base_space.orders)) == (
            actual.order,
            tuple(actual.base_space.orders),
        )
        return
    if isinstance(expected, IntegrationSpace):
        assert isinstance(actual, IntegrationSpace)
        assert tuple(expected.orders) == tuple(actual.orders)
        return
    if isinstance(expected, SpaceMap):
        assert isinstance(actual, SpaceMap)
        np.testing.assert_array_equal(
            np.asarray(expected.determinant), np.asarray(actual.determinant)
        )
        return
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def _fresh() -> tuple[IntegrationRegistry, BasisRegistry]:
    """Create a brand-new registry pair that starts out empty."""
    return IntegrationRegistry(), BasisRegistry()


def test_space_map_boundary_uses_given_registry() -> None:
    """SpaceMap.boundary samples its face grid from the given registry."""
    _, maps, _ = _setup()
    integration_registry, _ = _fresh()

    explicit = maps[0].boundary(0, integration_registry=integration_registry)
    assert integration_registry.usage()

    default = maps[0].boundary(0)
    _assert_same(default, explicit)
    _assert_same(
        maps[0].boundary(0, integration_registry=DEFAULT_INTEGRATION_REGISTRY),
        explicit,
    )


def test_boundary_space_map_factors_use_given_registry() -> None:
    """The factor pull fetches its face and canonical rules from the argument."""
    _, maps, _ = _setup()
    common = IntegrationSpace(IntegrationSpecs(5))
    integration_registry, _ = _fresh()

    explicit = compute_boundary_space_map_factors(
        maps[0], [1, 2], common, integration_registry=integration_registry
    )
    assert integration_registry.usage()

    default = compute_boundary_space_map_factors(maps[0], [1, 2], common)
    _assert_same(default, explicit)


def test_boundary_mass_matrices_use_given_registries() -> None:
    """The batch assembly fetches rules and trace tables from its arguments."""
    _, maps, specs = _setup()
    integration_registry, basis_registry = _fresh()

    explicit = compute_kform_boundary_mass_matrices(
        [specs[0], specs[1]],
        [[1, 2], [-1, 2]],
        [maps[0].integration_space, maps[1].integration_space],
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )
    assert integration_registry.usage()
    assert basis_registry.usage()

    default = compute_kform_boundary_mass_matrices(
        [specs[0], specs[1]],
        [[1, 2], [-1, 2]],
        [maps[0].integration_space, maps[1].integration_space],
    )
    _assert_same(default, explicit)


def test_boundary_trace_moments_use_given_registries() -> None:
    """The single-element trace moments accept and consult both registries."""
    _, maps, specs = _setup()
    integration_registry, basis_registry = _fresh()

    def call(**kwargs: Any) -> Any:
        return compute_kform_boundary_trace_moments(
            [specs[0]],
            [[1, 2]],
            [maps[0].integration_space],
            element_maps=[maps[0]],
            boundary_dimension=1,
            axis_skip=(2,),
            packed=True,
            **kwargs,
        )

    explicit = call(
        integration_registry=integration_registry, basis_registry=basis_registry
    )
    assert integration_registry.usage()
    assert basis_registry.usage()
    _assert_same(call(), explicit)
    _assert_same(
        call(
            integration_registry=DEFAULT_INTEGRATION_REGISTRY,
            basis_registry=DEFAULT_BASIS_REGISTRY,
        ),
        explicit,
    )


def test_boundary_load_uses_given_registries() -> None:
    """The physical load traces against the argument registries."""
    mesh, maps, specs = _setup()
    face_space = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 1))
    face = _first_boundary_face(mesh)
    integration_registry, basis_registry = _fresh()

    def call(**kwargs: Any) -> np.ndarray:
        return compute_kform_boundary_load(
            KFormSpecs(0, face_space),
            specs[0],
            maps[0],
            mesh.collections,
            mesh.point_count,
            0,
            face,
            [_ones, _ones],
            **kwargs,
        )

    explicit = call(
        integration_registry=integration_registry, basis_registry=basis_registry
    )
    assert integration_registry.usage()
    assert basis_registry.usage()
    _assert_same(call(), explicit)


def test_continuity_constraints_use_given_registries() -> None:
    """The C continuity path reads the registries off its context."""
    mesh, maps, specs = _setup()
    integration_registry, basis_registry = _fresh()

    explicit = mesh.compute_kform_continuity_constraints(
        specs,
        maps,
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )
    assert integration_registry.usage()
    assert basis_registry.usage()

    default = mesh.compute_kform_continuity_constraints(specs, maps)
    _assert_same(default, explicit)
    _assert_same(
        mesh.compute_kform_continuity_constraints(
            specs,
            maps,
            integration_registry=DEFAULT_INTEGRATION_REGISTRY,
            basis_registry=DEFAULT_BASIS_REGISTRY,
        ),
        explicit,
    )


def test_mesh_global_constraints_use_given_registries() -> None:
    """The C forwarder passes both registries down to the Python assembler."""
    mesh, maps, specs = _setup()
    integration_registry, basis_registry = _fresh()

    explicit = mesh.compute_kform_global_constraints(
        specs,
        maps,
        None,
        None,
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )
    assert integration_registry.usage()
    assert basis_registry.usage()

    default = mesh.compute_kform_global_constraints(specs, maps)
    _assert_same(default, explicit)


def test_python_global_constraints_thread_registries() -> None:
    """Boundary data and periodic pairs reach every private helper."""
    mesh, maps, specs = _setup()
    lower, upper = _paired_faces(mesh)
    face = next(
        int(object_id)
        for _, object_id, _, orientations in mesh.iterate_boundary(mesh.ndim - 1)
        if abs(int(orientations[0, 0])) == 2
    )
    conditions: list[BoundaryCondition] = [BoundaryCondition((face,), _ones)]
    pairs: list[BoundaryPair] = [BoundaryPair(lower, upper, (1,))]

    def run(
        integration_registry: IntegrationRegistry,
        basis_registry: BasisRegistry,
        *,
        conditions: Sequence[BoundaryCondition] | None = None,
        pairs: Sequence[BoundaryPair] | None = None,
    ) -> Any:
        return compute_kform_global_constraints(
            mesh,
            specs,
            maps,
            boundary_conditions=conditions,
            periodic_pairs=pairs,
            integration_registry=integration_registry,
            basis_registry=basis_registry,
        )

    # Prescribed data drive the dual-moment helpers; periodic pairs drive the
    # shared mass call. They never share a boundary object in one run.
    integration_registry, basis_registry = _fresh()
    with_data = run(integration_registry, basis_registry, conditions=conditions)
    assert integration_registry.usage()
    assert basis_registry.usage()
    _assert_same(
        run(
            DEFAULT_INTEGRATION_REGISTRY,
            DEFAULT_BASIS_REGISTRY,
            conditions=conditions,
        ),
        with_data,
    )

    integration_registry, basis_registry = _fresh()
    periodic = run(integration_registry, basis_registry, pairs=pairs)
    assert integration_registry.usage()
    assert basis_registry.usage()
    _assert_same(
        run(DEFAULT_INTEGRATION_REGISTRY, DEFAULT_BASIS_REGISTRY, pairs=pairs),
        periodic,
    )

    # Prescribed data give the right-hand side something to carry.
    assert np.any(with_data[1])


def _entry_points() -> Sequence[tuple[str, Call]]:
    """Every call that gained an ``integration_registry`` keyword."""
    mesh, maps, specs = _setup()
    face = _first_boundary_face(mesh)
    common = IntegrationSpace(IntegrationSpecs(3))
    face_space = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, 1))
    return (
        ("boundary", lambda **kw: maps[0].boundary(0, **kw)),
        (
            "space_map_factors",
            lambda **kw: compute_boundary_space_map_factors(
                maps[0], [1, 2], common, **kw
            ),
        ),
        (
            "mass_matrices",
            lambda **kw: compute_kform_boundary_mass_matrices(
                [specs[0], specs[1]],
                [[1, 2], [-1, 2]],
                [maps[0].integration_space, maps[1].integration_space],
                **kw,
            ),
        ),
        (
            "trace_moments",
            lambda **kw: compute_kform_boundary_trace_moments(
                [specs[0]],
                [[1, 2]],
                [maps[0].integration_space],
                element_maps=[maps[0]],
                boundary_dimension=1,
                axis_skip=(2,),
                packed=True,
                **kw,
            ),
        ),
        (
            "boundary_load",
            lambda **kw: compute_kform_boundary_load(
                KFormSpecs(0, face_space),
                specs[0],
                maps[0],
                mesh.collections,
                mesh.point_count,
                0,
                face,
                [_ones, _ones],
                **kw,
            ),
        ),
        (
            "continuity",
            lambda **kw: mesh.compute_kform_continuity_constraints(specs, maps, **kw),
        ),
        (
            "mesh_global",
            lambda **kw: mesh.compute_kform_global_constraints(specs, maps, **kw),
        ),
        (
            "python_global",
            lambda **kw: compute_kform_global_constraints(mesh, specs, maps, **kw),
        ),
    )


@pytest.mark.parametrize(
    "call", [pytest.param(fn, id=name) for name, fn in _entry_points()]
)
def test_registry_kwargs_reject_non_registries(call: Call) -> None:
    """Every new keyword is type-checked against the registry classes."""
    with pytest.raises(TypeError):
        call(integration_registry=object())
    with pytest.raises(TypeError):
        call(basis_registry=object())


def test_public_python_registry_params_are_keyword_only() -> None:
    """Registries never shift an existing positional argument."""
    from inspect import Parameter, signature

    public = signature(compute_kform_global_constraints)
    for keyword in ("integration_registry", "basis_registry"):
        assert public.parameters[keyword].kind is Parameter.KEYWORD_ONLY


def test_private_helpers_require_their_registries() -> None:
    """A helper that is not handed registries fails instead of defaulting."""
    from inspect import Parameter, signature

    required = (
        (_restrict_map, ("integration_registry",)),
        (_windowed_component_basis, ("integration_registry",)),
        (_windowed_dual_values, ("integration_registry",)),
        (_append_boundary_rows, ("integration_registry", "basis_registry")),
        (_append_periodic_rows, ("integration_registry", "basis_registry")),
    )
    for helper, keywords in required:
        for keyword in keywords:
            parameter = signature(helper).parameters[keyword]
            assert parameter.kind is Parameter.KEYWORD_ONLY, helper.__name__
            assert parameter.default is Parameter.empty, helper.__name__

    # The assembler reachable from the C forwarder keeps the public defaults.
    assembler = signature(_compute_kform_global_constraints)
    assert assembler.parameters["integration_registry"].default is not Parameter.empty
    assert assembler.parameters["basis_registry"].default is not Parameter.empty
