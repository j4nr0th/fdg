"""Supporting code for degrees of freedom."""

import numpy as np
import numpy.typing as npt

from fdg._fdg import DegreesOfFreedom


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
