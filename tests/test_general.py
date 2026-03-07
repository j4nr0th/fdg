"""Test some general functions that do not really fit in any other category."""

import numpy as np
import pytest
from fdg._fdg import _scale_array_boundary


@pytest.mark.parametrize(("s1", "s2", "s3"), ((3, 2, 4), (2, 5, 5), (11, 23, 44)))
def test_scaling_3d(s1: int, s2: int, s3: int) -> None:
    """Check that scaling works in 3D."""
    count_array = np.zeros((s1, s2, s3), np.uint)
    count_array[:, :, 0] += 1
    count_array[:, :, -1] += 1
    count_array[:, 0, :] += 1
    count_array[:, -1, :] += 1
    count_array[0, :, :] += 1
    count_array[-1, :, :] += 1
    count_array += count_array == 0
    scaled = _scale_array_boundary(count_array)
    assert pytest.approx(scaled) == np.ones_like(scaled)
