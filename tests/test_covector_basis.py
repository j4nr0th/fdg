"""Check that covector basis work as expected."""

from itertools import combinations

import pytest
from fdg._fdg import CovectorBasis


def test_1d():
    """Testing in 1D is very simple."""
    b00 = CovectorBasis(1)
    b01 = CovectorBasis(1, 0)

    assert ~b01 == b00
    assert b01 == ~b00
    assert b00 ^ b01 == b01
    assert b01 ^ b00 == b01
    assert not b01 ^ b01
    assert b00 ^ b00 == b00


@pytest.mark.parametrize("n", range(2, 5))
def test_nd(n: int):
    """Testing in n-D is a bit more involved."""
    b_full = CovectorBasis(n, *range(n))
    b_empty = CovectorBasis(n)
    for k1 in range(1, n):
        for c1 in combinations(range(n), k1):
            b1 = CovectorBasis(n, *c1)
            assert b1 ^ ~b1 == b_full
            assert (~(~b1)).sign == (-1) ** (k1 * (n - k1))
            assert ~b1 ^ b1 == (-b_full if ((k1 * (n - k1)) % 2) else b_full)
            assert b1 ^ b_empty == b1
            assert not (b1 ^ b1)
            for k2 in range(1, n):
                for c2 in combinations(range(n), k2):
                    b2 = CovectorBasis(n, *c2)

                    if k1 != k2:
                        with pytest.raises(ValueError):
                            _ = b1 == b2
                    else:
                        if len(c1) <= n // 2:
                            v1, v2 = c1, c2
                        else:
                            v1 = tuple(i for i in range(n) if i not in c1)
                            v2 = tuple(i for i in range(n) if i not in c2)
                        assert (b1 < b2) == (v1 < v2)
                        assert (b1 <= b2) == (v1 <= v2)
                        assert (b1 >= b2) == (v1 >= v2)
                        assert (b1 > b2) == (v1 > v2)
                        assert (b1 == b2) == (v1 == v2)
                        assert (b1 != b2) == (v1 != v2)


if __name__ == "__main__":
    test_1d()
    test_nd(3)
