"""Functions to allow integration of callables."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from fdg._fdg import (
    DEFAULT_BASIS_REGISTRY,
    DEFAULT_INTEGRATION_REGISTRY,
    BasisRegistry,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationRegistry,
    IntegrationSpace,
    KFormSpecs,
    SpaceMap,
    compute_kform_mass_matrix,
    compute_mass_matrix,
    transform_kform_component_to_target,
)


class Integrable(Protocol):
    """Protocol for integrable objects."""

    def __call__(
        self,
        *args: npt.NDArray[np.double],
    ) -> npt.ArrayLike:
        """Evaluate the integrable object at given points.

        Parameters
        ----------
        args : npt.NDArray[np.double]
            Coordinates at which to evaluate the integrable object. Each argument
            corresponds to one dimension.

        Returns
        -------
        npt.ArrayLike
            The evaluated values.
        """
        ...


def _prepare_integration(
    integration: IntegrationSpace | SpaceMap, registry: IntegrationRegistry
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Prepare nodes and weights.

    Parameters
    ----------
    integration : IntegrationSpace or SpaceMap
        Either an ordinary integration space of a space map.

    registry : IntegrationRegistry
        Registry to get the integration nodes and weights from.

    Returns
    -------
    array
        Array of integration nodes where a callable should be evaluated at.

    array
        Array of integration weights to use for integrating the function.
    """
    match integration:
        case IntegrationSpace() as int_space:
            nodes = int_space.nodes(registry)
            weights = int_space.weights(registry)
        case SpaceMap() as smap:
            int_space = smap.integration_space
            weights = int_space.weights(registry) * smap.determinant
            nodes = np.array(
                [
                    smap.coordinate_map(idim).values
                    for idim in range(smap.output_dimensions)
                ]
            )
        case _:
            raise TypeError(
                f"Only {IntegrationSpace} or {SpaceMap} can be passed, but instead "
                f"{type(integration)} was passed."
            )

    return nodes, weights


def integrate_callable(
    func: Integrable,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
    registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
) -> float:
    """Integrate a callable over a specified integration space with given specs.

    Parameters
    ----------
    func : Callable
        The function to integrate. The function should be defined on the space it will
        be integrated on.

    integration_space : IntegrationSpace or SpaceMap
        The space over which to integrate the function or the mapping between the
        integration domain, which is an :math:`N`-dimensional :math:`[-1, +1]` hypercube,
        and the physical domain.

    registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        The registry to use for obtaining the integrator.

    Returns
    -------
    float
        The result of the integration.
    """
    nodes, weights = _prepare_integration(integration=integration, registry=registry)
    return float(
        np.sum(
            np.asarray(func(*[nodes[i, ...] for i in range(nodes.shape[0])])) * weights
        )
    )


def projection_l2_dual(
    func: Integrable,
    function_space: FunctionSpace,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> DegreesOfFreedom:
    """Compute the dual L2 projection of the function on the function space.

    Parameters
    ----------
    func : Integratable
        Function to project. It has to be possible to integrate it.

    function_space : FunctionSpace
        Function space on which to project the function.

    integration : IntegrationSpace or SpaceMap
        Specification of the integration domain.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        The registry to use for obtaining the integrator.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        The registry to use for obtaining the basis values.

    Returns
    -------
    DegreesOfFreedom
        Dual degrees of freedom of the projection.
    """
    nodes, weights = _prepare_integration(
        integration=integration, registry=integration_registry
    )

    func_vals = (
        np.asarray(func(*[nodes[i, ...] for i in range(nodes.shape[0])])) * weights
    )
    del nodes, weights, func

    int_space: IntegrationSpace
    match integration:
        case IntegrationSpace():
            int_space = integration
        case SpaceMap():
            int_space = integration.integration_space
        case _:
            assert False

    basis_values = function_space.values_at_integration_nodes(
        int_space,
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )
    del integration_registry, basis_registry

    func_vals = func_vals.flatten()
    basis_values = basis_values.reshape((func_vals.size, -1))
    dual_dofs = np.sum(func_vals[:, None] * basis_values, axis=0)

    return DegreesOfFreedom(function_space, dual_dofs)


def projection_l2_primal(
    func: Integrable,
    function_space: FunctionSpace,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> DegreesOfFreedom:
    """Compute the L2 projection of the function on the function space.

    Parameters
    ----------
    func : Integratable
        Function to project. It has to be possible to integrate it.

    function_space : FunctionSpace
        Function space on which to project the function.

    integration : IntegrationSpace or SpaceMap
        Specification of the integration domain.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        The registry to use for obtaining the integrator.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        The registry to use for obtaining the basis values.

    Returns
    -------
    DegreesOfFreedom
        Primal degrees of freedom of the projection.
    """
    dual_dofs = projection_l2_dual(
        func,
        function_space,
        integration,
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )

    mass_matrix = compute_mass_matrix(
        function_space,
        function_space,
        integration,
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )
    del integration, integration_registry, basis_registry

    primal_dofs = np.linalg.solve(mass_matrix, dual_dofs.values.flatten())
    del dual_dofs, mass_matrix
    return DegreesOfFreedom(function_space, primal_dofs)


def projection_kform_l2_dual(
    funcs: Sequence[Integrable],
    specs: KFormSpecs,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> tuple[npt.NDArray[np.double], ...]:
    """Compute the dual L2 projection of the :math:`k`-form.

    Parameters
    ----------
    funcs : Sequence of Integratable
        Functions to project. There must be the same number of them as the number
        of the :math:`k`-form components.

    specs : KFormSpecs
        Specifications of the :math:`k`-form to project.

    integration : IntegrationSpace or SpaceMap
        Specification of the integration domain.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        The registry to use for obtaining the integrator.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        The registry to use for obtaining the basis values.

    Returns
    -------
    tuple of array
        Dual degrees of freedom of the projection for each of the components.
    """
    nodes, weights = _prepare_integration(
        integration=integration, registry=integration_registry
    )

    if len(funcs) != specs.component_count:
        raise ValueError(
            f"A {specs.order}-form in {specs.dimension}D has {specs.component_count}"
            f" components, but {len(funcs)} were provided."
        )

    func_vals = (
        np.asarray(
            [func(*[nodes[i, ...] for i in range(nodes.shape[0])]) for func in funcs]
        )
        * weights[None, ...]
    )
    del nodes, funcs, weights

    int_space: IntegrationSpace
    match integration:
        case IntegrationSpace():
            int_space = integration
        case SpaceMap():
            int_space = integration.integration_space
        case _:
            assert False

    for idim in range(int_space.dimension):
        func_vals = func_vals[..., None]

    basis_functions: list[npt.NDArray] = list()
    for idx in range(specs.component_count):
        fn_space = specs.get_component_function_space(idx)
        basis_functions.append(
            fn_space.values_at_integration_nodes(
                int_space,
                integration_registry=integration_registry,
                basis_registry=basis_registry,
            )
        )

    del integration_registry, basis_registry

    if type(integration) is SpaceMap:
        transformed_basis = [
            transform_kform_component_to_target(
                specs.order, integration, basis_functions[i], i
            )
            for i in range(specs.component_count)
        ]
        dual_dofs = tuple(
            np.sum(func_vals * b, axis=tuple(range(int_space.dimension + 1)))
            for b in transformed_basis
        )
    else:
        dual_dofs = tuple(
            np.sum(f * b, axis=tuple(range(int_space.dimension)))
            for f, b in zip(func_vals, basis_functions)
        )

    return dual_dofs


def projection_kform_l2_primal(
    funcs: Sequence[Integrable],
    specs: KFormSpecs,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> tuple[npt.NDArray[np.double], ...]:
    """Compute the primal L2 projection of the :math:`k`-form.

    Parameters
    ----------
    funcs : Sequence of Integratable
        Functions to project. There must be the same number of them as the number
        of the :math:`k`-form components.

    specs : KFormSpecs
        Specifications of the :math:`k`-form to project.

    integration : IntegrationSpace or SpaceMap
        Specification of the integration domain.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        The registry to use for obtaining the integrator.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        The registry to use for obtaining the basis values.

    Returns
    -------
    tuple of array
        Primal degrees of freedom of the projection for each of the components.
    """
    dual_dofs = projection_kform_l2_dual(
        funcs,
        specs,
        integration,
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )
    flat_dual = np.concatenate([dd.flatten() for dd in dual_dofs])
    match integration:
        case SpaceMap() as smap:
            mass_matrix = compute_kform_mass_matrix(
                smap,
                specs.order,
                specs.base_space,
                specs.base_space,
                int_registry=integration_registry,
                basis_registry=basis_registry,
            )
            flat_primal = np.linalg.solve(mass_matrix, flat_dual)
        case IntegrationSpace() as ispace:
            primal_vals: list[npt.NDArray[np.double]] = list()
            for idx, dd in enumerate(dual_dofs):
                fn_space = specs.get_component_function_space(idx)
                mass_matrix = compute_mass_matrix(
                    fn_space,
                    fn_space,
                    ispace,
                    integration_registry=integration_registry,
                    basis_registry=basis_registry,
                )
                primal_vals.append(np.linalg.solve(mass_matrix, dd.flatten()))
            flat_primal = np.concatenate(primal_vals)

        case _:
            raise TypeError(f"Invalid integration type {type(integration)}")

    off = 0
    primal: list[npt.NDArray[np.double]] = list()
    for dd in dual_dofs:
        sz = dd.size
        pd = flat_primal[off : off + sz]
        off += sz
        primal.append(np.reshape(pd, dd.shape))

    return tuple(primal)
