"""Computations of mass matrices."""

from itertools import combinations

import numpy as np
import numpy.typing as npt

from fdg._fdg import CovectorBasis, FunctionSpace, SpaceMap


def component_inner_prod_mass_matrix_block(
    basis_left: CovectorBasis,
    basis_right: CovectorBasis,
    fn_space_left: FunctionSpace,
    fn_space_right: FunctionSpace,
    smap: SpaceMap,
) -> npt.NDArray[np.double]:
    """Compute an inner product mass matrix."""
    n = basis_left.ndim
    if n != basis_right.ndim:
        raise ValueError("Dimension basis must be equal.")
    if fn_space_right.dimension != n:
        raise ValueError("Input function space must have the same dimension basis.")
    if fn_space_left.dimension != n:
        raise ValueError("Outpu function space must have the same dimension basis.")
    if smap.input_dimensions != n:
        raise ValueError("Space map must have the same dimension basis.")

    k = basis_left.rank
    if k != basis_right.rank:
        raise ValueError("Rank of the basis must be equal.")

    # Prepare integration weights
    weights = smap.integration_space.weights()
    if k == 0:
        weights *= smap.determinant
    elif k == n:
        weights /= smap.determinant
    else:
        transformation_matrix = smap.basis_transform(k)

        weights_1 = transformation_matrix[basis_left.index, :, :]
        weights_2 = transformation_matrix[basis_right.index, :, :]
        tw = np.reshape(np.sum(weights_1 * weights_2, axis=0), weights.shape)
        weights *= smap.determinant * tw

    # Babe it's 4 p.m., time for your array flattening
    weights = weights.flatten()

    # Prepare function spaces
    fn_left = fn_space_left
    for i in range(n):
        if i in basis_left:
            fn_left = fn_left.lower_order(i)

    fn_right = fn_space_right
    for i in range(n):
        if i in basis_right:
            fn_right = fn_right.lower_order(i)

    # Get the basis
    bv_left = np.reshape(
        fn_left.values_at_integration_nodes(smap.integration_space), (weights.size, -1)
    )
    bv_right = np.reshape(
        fn_right.values_at_integration_nodes(smap.integration_space), (weights.size, -1)
    )

    # Compute the mass matrix in a big fat multiplication and sum
    return np.sum(
        bv_left[:, :, None] * bv_right[:, None, :] * weights[:, None, None], axis=0
    )


def iterate_kform_components(ndim: int, order: int):
    """Iterate over all components of a k-form.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    order : int
        Order of the k-form.

    Return
    ------
    Generator
        Generator which produces the basis of all k-form components.
    """
    for comb in combinations(range(ndim), order):
        yield CovectorBasis(ndim, *comb)


def compute_inner_prod_mass_matrix(
    smap: SpaceMap,
    order: int,
    fn_space_left: FunctionSpace,
    fn_space_right: FunctionSpace,
):
    """Compute the full inner-product mass matrix for a k-form."""
    blocks: list[list[npt.NDArray[np.double]]] = list()
    ndim = smap.input_dimensions

    for basis_left in iterate_kform_components(ndim, order):
        row: list[npt.NDArray[np.double]] = list()
        for basis_right in iterate_kform_components(ndim, order):
            row.append(
                component_inner_prod_mass_matrix_block(
                    basis_left, basis_right, fn_space_left, fn_space_right, smap
                )
            )
        blocks.append(row)
        del row

    return np.block(blocks)
