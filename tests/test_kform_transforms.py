"""Check that k-forms are propertly transformed from the reference domain."""

from functools import cache
from itertools import combinations

import numpy as np
import pytest
from fdg._fdg import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    SampledSpaceMap,
    SpaceMap,
    transform_kform_component_to_target,
    transform_kform_to_target,
    transform_kform_to_target_sampled,
)
from fdg.enum_type import BasisType


def prepare_random_space_map(
    n: int, m: int
) -> tuple[np.random.Generator, IntegrationSpace, SpaceMap]:
    """Prepare a random space mapping and all other neded objects."""
    rng = np.random.default_rng(124 + n**2 + m**2)
    max_order = 10 - n
    assert max_order > 0

    int_space = IntegrationSpace(
        *(IntegrationSpecs(i) for i in rng.integers(3, max_order, n))
    )
    coord_dofs: tuple[DegreesOfFreedom, ...] = tuple(
        DegreesOfFreedom(
            FunctionSpace(
                *(
                    BasisSpecs(BasisType.BERNSTEIN, bo)
                    for bo in rng.integers(1, max_order, n)
                )
            ),
        )
        for _ in range(m)
    )

    for coord_dof in coord_dofs:
        coord_dof.values = rng.random(coord_dof.shape)

    space_map = SpaceMap(
        *(
            CoordinateMap(
                coord_dof,
                int_space,
            )
            for coord_dof in coord_dofs
        )
    )
    return rng, int_space, space_map


@cache
def binomial_coefficient(n: int, k: int) -> int:
    """Compute the binomial coefficient."""
    if 2 * k > n:
        high = k
        low = n - k
    else:
        high = n - k
        low = k

    return int(np.prod(np.arange(high + 1, n + 1)) // np.prod(np.arange(2, low + 1)))


@pytest.mark.parametrize("n", (1, 2, 3, 4))
@pytest.mark.parametrize("dm", (0, 1, 2))
def test_1forms(n: int, dm: int) -> None:
    """Check if 1-forms are properly transformed."""
    assert n > 0
    m = n + dm
    rng, int_space, space_map = prepare_random_space_map(n, m)

    values_1form = rng.random((n, *(i + 1 for i in int_space.orders)))

    transformed = transform_kform_to_target(1, space_map, values_1form)
    manually_transformed = np.zeros((m, *(i + 1 for i in int_space.orders)))
    for i in range(m):
        v = manually_transformed[i, ...]
        for j in range(n):
            v[:] += space_map.inverse_map[..., j, i] * values_1form[j, ...]

    assert pytest.approx(transformed) == manually_transformed


@pytest.mark.parametrize("n", (2, 3, 4))
@pytest.mark.parametrize("dm", (0, 1, 2))
def test_2forms(n: int, dm: int) -> None:
    """Check if 2-forms are properly transformed."""
    assert n > 1
    m = n + dm
    rng, int_space, space_map = prepare_random_space_map(n, m)

    int_space_shape = tuple(i + 1 for i in int_space.orders)
    comp_cnt_in = binomial_coefficient(n, 2)  # n * (n - 1) // 2
    comp_cnt_out = binomial_coefficient(m, 2)  # m * (m - 1) // 2
    values_2form = rng.random((comp_cnt_in, *int_space_shape))

    transformed = transform_kform_to_target(2, space_map, values_2form)
    trans_array = space_map.basis_transform(2)
    ta = np.reshape(trans_array, (comp_cnt_in, comp_cnt_out, *int_space_shape))
    manually_transformed = np.zeros((comp_cnt_out, *int_space_shape))
    for i in range(comp_cnt_out):
        manually_transformed[i, ...] = np.sum(
            ta[:, i, ...] * values_2form[:, ...], axis=0
        )

    assert pytest.approx(transformed) == manually_transformed


@pytest.mark.parametrize("n", (2, 3, 4, 5))
@pytest.mark.parametrize("dm", (0, 1, 2))
def test_kforms(n: int, dm: int) -> None:
    """Check if k-forms are properly transformed."""
    assert n > 1
    m = n + dm
    rng, int_space, space_map = prepare_random_space_map(n, m)

    int_space_shape = tuple(i + 1 for i in int_space.orders)

    for k in range(1, n + 1):
        comp_cnt_in = binomial_coefficient(n, k)
        comp_cnt_out = binomial_coefficient(m, k)
        values_kform = rng.random((comp_cnt_in, *int_space_shape))

        transformed = transform_kform_to_target(k, space_map, values_kform)
        output = np.empty_like(transformed)
        returned = transform_kform_to_target(k, space_map, values_kform, out=output)
        assert returned is output
        np.testing.assert_allclose(output, transformed)
        per_component = np.zeros_like(transformed)
        for i in range(comp_cnt_in):
            cv = transform_kform_component_to_target(
                k, space_map, values_kform[i, ...], i
            )
            o = np.empty_like(cv)
            rv = transform_kform_component_to_target(
                k, space_map, values_kform[i, ...], i, out=o
            )
            assert rv is o
            assert np.all(rv == o)
            tv_in = rng.random(np.concatenate([rng.integers(2, 4, 3), rv.shape]))
            tv_out_v = transform_kform_component_to_target(k, space_map, tv_in, i)
            tv_out_l = np.reshape(
                np.array(
                    [
                        transform_kform_component_to_target(k, space_map, tv, i)
                        for tv in np.reshape(tv_in, (-1, *rv.shape))
                    ]
                ),
                tv_out_v.shape,
            )
            assert np.all(tv_out_v == tv_out_l)

            per_component += cv
        assert pytest.approx(per_component) == transformed

        trans_array = space_map.basis_transform(k)
        ta = np.reshape(trans_array, (comp_cnt_in, comp_cnt_out, *int_space_shape))
        manually_transformed = np.zeros((comp_cnt_out, *int_space_shape))
        for i in range(comp_cnt_out):
            manually_transformed[i, ...] = np.sum(
                ta[:, i, ...] * values_kform[:, ...], axis=0
            )

        assert pytest.approx(transformed) == manually_transformed


@pytest.mark.parametrize(("ndim", "mdim"), ((1, 1), (2, 2), (2, 3), (3, 5)))
def test_sampled_kform_transforms(ndim: int, mdim: int) -> None:
    """Check sampled k-form transforms against explicit inverse-map minors."""
    function_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(ndim))
    )
    reference_nodes = np.meshgrid(*([np.array((-1.0, 1.0))] * ndim), indexing="ij")
    integration_space = IntegrationSpace(*(IntegrationSpecs(3) for _ in range(ndim)))

    coordinates = []
    for icoordinate in range(mdim):
        values = np.zeros_like(reference_nodes[0])
        for idim in range(ndim):
            coefficient = (icoordinate + 2) if idim == icoordinate % ndim else 0.0
            values += coefficient * reference_nodes[idim]
        coordinates.append(
            CoordinateMap(DegreesOfFreedom(function_space, values), integration_space)
        )
    space_map = SpaceMap(*coordinates)

    sample_orders = tuple(2 + idim % 2 for idim in range(ndim))
    samples = tuple(np.linspace(-1.0, 1.0, order + 1) ** 3 for order in sample_orders)
    sampled_map = SampledSpaceMap(space_map, samples)
    sample_shape = tuple(order + 1 for order in sample_orders)
    inverse_map = sampled_map.inverse_map
    rng = np.random.default_rng(914 + ndim * 17 + mdim)

    for order in range(1, ndim + 1):
        input_bases = tuple(combinations(range(ndim), order))
        output_bases = tuple(combinations(range(mdim), order))
        components = rng.random((len(input_bases), *sample_shape))

        factors = np.empty((len(input_bases), len(output_bases), *sample_shape))
        for input_index, input_basis in enumerate(input_bases):
            for output_index, output_basis in enumerate(output_bases):
                minor = np.take(
                    np.take(inverse_map, input_basis, axis=-2),
                    output_basis,
                    axis=-1,
                )
                factors[input_index, output_index] = np.linalg.det(minor)
        expected = np.einsum("ij...,i...->j...", factors, components)

        transformed = transform_kform_to_target_sampled(order, sampled_map, components)
        np.testing.assert_allclose(transformed, expected, rtol=1e-12, atol=1e-12)

        output = np.empty_like(expected)
        returned = transform_kform_to_target_sampled(
            order, sampled_map, components, out=output
        )
        assert returned is output
        np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)

    invalid_components = np.zeros((1, *sample_shape))
    with pytest.raises(ValueError):
        transform_kform_to_target_sampled(0, sampled_map, invalid_components)
    with pytest.raises(ValueError):
        transform_kform_to_target_sampled(-1, sampled_map, invalid_components)
    with pytest.raises(ValueError):
        transform_kform_to_target_sampled(ndim + 1, sampled_map, invalid_components)


if __name__ == "__main__":
    for n in (2, 3, 4, 5):
        for dm in (0, 1, 2):
            test_kforms(n, dm)
