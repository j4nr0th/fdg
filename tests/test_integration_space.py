"""Check that the integration rules work."""

import numpy as np
import pytest
from fdg import IntegrationMethod, IntegrationSpace, IntegrationSpecs


def _exact_integral_monomial(k: int) -> float:
    """Analytic value of integral of x^k on the interval [-1, +1]."""
    if k & 1:
        return 0.0
    else:
        return 2.0 / (k + 1)


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("method", IntegrationMethod)
def test_monomials_integrated_exactly_1d(order: int, method: IntegrationMethod):
    """Check monomials are integrated exactly."""
    specs = IntegrationSpecs(order, method=method)
    space = IntegrationSpace(specs)
    acc = specs.accuracy

    assert space.dimension == 1
    assert space.orders == (order,)
    assert space.integration_specs == (specs,)

    nodes = space.nodes()
    weights = space.weights()
    for k in range(acc + 1):
        approx = np.sum(weights * np.prod(nodes**k, axis=0))
        exact = _exact_integral_monomial(k)
        assert pytest.approx(approx) == exact, (
            f"Failed for k={k}, method={method}, order={order}"
        )


@pytest.mark.parametrize("order_1", [2, 4, 5])
@pytest.mark.parametrize("order_2", [2, 4, 5])
@pytest.mark.parametrize("method_1", IntegrationMethod)
@pytest.mark.parametrize("method_2", IntegrationMethod)
def test_monomials_integrated_exactly_2d(
    order_1: int, order_2: int, method_1: IntegrationMethod, method_2: IntegrationMethod
):
    """Check monomials are integrated exactly."""
    specs_1 = IntegrationSpecs(order_1, method=method_1)
    specs_2 = IntegrationSpecs(order_2, method=method_2)
    space = IntegrationSpace(specs_1, specs_2)

    assert space.dimension == 2
    assert space.orders == (order_1, order_2)
    assert space.integration_specs == (specs_1, specs_2)

    acc_1 = specs_1.accuracy
    acc_2 = specs_2.accuracy

    nodes = space.nodes()
    weights = space.weights()
    for k1 in range(acc_1 + 1):
        for k2 in range(acc_2 + 1):
            approx = np.sum(weights * nodes[0, ...] ** k1 * nodes[1, ...] ** k2)
            exact = _exact_integral_monomial(k1) * _exact_integral_monomial(k2)
            assert pytest.approx(approx) == exact, (
                f"Failed for {k1=}, {k2=}, {order_1=}, {method_1=}, {order_2=}, "
                f"{method_2=}"
            )


@pytest.mark.parametrize("order_1", [2, 4, 5])
@pytest.mark.parametrize("order_2", [2, 4, 5])
@pytest.mark.parametrize("order_3", [2, 4, 5])
@pytest.mark.parametrize("method_1", IntegrationMethod)
@pytest.mark.parametrize("method_2", IntegrationMethod)
@pytest.mark.parametrize("method_3", IntegrationMethod)
def test_monomials_integrated_exactly_3d(
    order_1: int,
    order_2: int,
    order_3: int,
    method_1: IntegrationMethod,
    method_2: IntegrationMethod,
    method_3: IntegrationMethod,
):
    """Check monomials are integrated exactly."""
    specs_1 = IntegrationSpecs(order_1, method=method_1)
    specs_2 = IntegrationSpecs(order_2, method=method_2)
    specs_3 = IntegrationSpecs(order_3, method=method_3)
    space = IntegrationSpace(specs_1, specs_2, specs_3)

    assert space.dimension == 3
    assert space.orders == (order_1, order_2, order_3)
    assert space.integration_specs == (specs_1, specs_2, specs_3)

    acc_1 = specs_1.accuracy
    acc_2 = specs_2.accuracy
    acc_3 = specs_3.accuracy

    nodes = space.nodes()
    weights = space.weights()
    for k1 in range(acc_1 + 1):
        for k2 in range(acc_2 + 1):
            for k3 in range(acc_3 + 1):
                approx = np.sum(
                    weights
                    * nodes[0, ...] ** k1
                    * nodes[1, ...] ** k2
                    * nodes[2, ...] ** k3
                )
                exact = (
                    _exact_integral_monomial(k1)
                    * _exact_integral_monomial(k2)
                    * _exact_integral_monomial(k3)
                )
                assert pytest.approx(approx) == exact, (
                    f"Failed for {k1=}, {k2=}, {k3=}, {order_1=}, {method_1=}, "
                    f"{order_2=}, {method_2=}, {order_3=}, {method_3=}"
                )
