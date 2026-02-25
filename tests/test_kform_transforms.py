"""Check that k-forms are propertly transformed from the reference domain."""

from functools import cache

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
    transform_kform_component_to_target,
    transform_kform_to_target,
)
from interplib.enum_type import BasisType


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


@pytest.mark.parametrize("n", (1, 2, 3, 4, 5))
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


@pytest.mark.parametrize("n", (2, 3, 4, 5))
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


if __name__ == "__main__":
    for n in (2, 3, 4, 5):
        for dm in (0, 1, 2):
            test_kforms(n, dm)
