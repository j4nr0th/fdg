"""Check that the integration rules work."""

import numpy as np
import pytest
from fdg import IntegrationMethod, IntegrationSpecs


def _exact_integral_monomial(k: int) -> float:
    """Analytic value of integral of x^k on the interval [-1, +1]."""
    if k & 1:
        return 0.0
    else:
        return 2.0 / (k + 1)


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("method", IntegrationMethod)
def test_monomials_integrated_exactly(order: int, method: IntegrationMethod):
    """Check monomials are integrated exactly."""
    rule = IntegrationSpecs(order, method=method)
    acc = rule.accuracy

    for k in range(acc + 1):
        approx = np.sum(rule.weights() * rule.nodes() ** k)
        exact = _exact_integral_monomial(k)
        assert pytest.approx(approx) == exact, (
            f"Failed for k={k}, method={method}, order={order}"
        )
