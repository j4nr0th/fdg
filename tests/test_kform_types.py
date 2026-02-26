"""Check that k-form related types work as expected."""

from itertools import combinations

import numpy as np
import pytest
from interplib._interp import BasisSpecs, CovectorBasis, FunctionSpace, KForm, KFormSpecs
from interplib.enum_type import BasisType


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_component_offsets(n: int) -> None:
    """Check that k-forms in n-dims have correct offset arrays."""
    rng = np.random.default_rng(67 + n**2)
    for k in range(0, n + 1):
        orders = rng.integers(1, 9, n)
        counts: list[int] = list()
        for components in combinations(range(n), k):
            dof_cnt = orders + 1
            if k:
                dof_cnt[np.array(components)] -= 1
            counts.append(int(np.prod(dof_cnt)))
        specs = KFormSpecs(
            k, FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, o) for o in orders))
        )
        assert np.all(specs.component_dof_counts == counts)


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_component_bases(n: int) -> None:
    """Check that k-forms in n-dims have correct bases."""
    rng = np.random.default_rng(67 + n**2)
    for k in range(0, n + 1):
        orders = rng.integers(1, 9, n)
        specs = KFormSpecs(
            k, FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, o) for o in orders))
        )
        for i, cmb in enumerate(combinations(range(n), k)):
            expected = CovectorBasis(n, *cmb)
            computed = specs.get_component_basis(i)
            assert expected == computed


@pytest.mark.parametrize("n", (1, 2, 3, 4, 6, 7))
def test_component_slices(n: int) -> None:
    """Check that k-forms in n-dims have correct offset arrays."""
    rng = np.random.default_rng(67 + n**2)
    for k in range(0, n + 1):
        orders = rng.integers(1, 9, n)
        specs = KFormSpecs(
            k, FunctionSpace(*(BasisSpecs(BasisType.BERNSTEIN, o) for o in orders))
        )
        kform = KForm(specs)
        v = rng.random(kform.values.shape)
        kform.values[:] = v
        assert np.all(kform.values == v)
        for i in range(specs.component_count):
            s = kform.specs.get_component_slice(i)
            flattened_values = kform.get_component_dofs(i).flatten()
            assert np.all(flattened_values == kform.values[s])


if __name__ == "__main__":
    test_component_slices(7)
    for n in range(1, 8):
        test_component_slices(n)
