"""Supporting code for degrees of freedom."""

import numpy as np
import numpy.typing as npt

from fdg._fdg import DegreesOfFreedom, FunctionSpace, IntegrationSpace
from fdg.integration import Integrable


def reconstruct(
    dof: DegreesOfFreedom,
    *x: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """Reconstruct function values at given locations.

    Parameters
    ----------
    *x : array
        Coordinates where the function should be reconstructed.
        Each array corresponds to a dimension.

    Returns
    -------
    array
        Array of reconstructed function values at the specified locations.
    """
    for v in x[1:]:
        if v.shape != x[0].shape:
            raise ValueError("All input coordinate arrays must have the same shape.")

    output = (
        dof.function_space.evaluate(
            *(np.ascontiguousarray(v, np.double) for v in x)
        ).reshape((-1, dof.n_dofs))
        * dof.values.flatten()
    )
    return np.sum(output, axis=-1).reshape(x[0].shape)


def compute_dual_degrees_of_freedom(
    fn: Integrable, integration_space: IntegrationSpace, function_space: FunctionSpace, /
) -> DegreesOfFreedom:
    """Compute the dual degrees of freedom.

    Parameters
    ----------
    fn : Integrable
        The function for which to compute the dual degrees of freedom.
    integration_space : IntegrationSpace
        The integration space to use.
    function_space : FunctionSpace
        The function space of the degrees of freedom.

    Returns
    -------
    DegreesOfFreedom
        The dual degrees of freedom.
    """
    dofs = DegreesOfFreedom(function_space)

    nodes = integration_space.nodes()
    weights = integration_space.weights()
    integration_values = (
        np.asarray(fn(*[nodes[i, ...] for i in range(nodes.shape[0])])) * weights
    ).flatten()
    basis_values = function_space.evaluate(
        *[nodes[i, ...] for i in range(nodes.shape[0])]
    ).reshape((integration_values.size, dofs.n_dofs))
    dofs.values = np.sum(basis_values * integration_values[:, None], axis=0)

    return dofs
