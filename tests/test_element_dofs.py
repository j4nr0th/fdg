"""Check the batched per-element degrees-of-freedom collection."""

import numpy as np
import pytest
from fdg._fdg import BasisSpecs, DegreesOfFreedom, ElementDoFs, FunctionSpace
from fdg.enum_type import BasisType


def _space(order_x: int, order_y: int) -> FunctionSpace:
    """Return a 2D function space with the given per-axis orders."""
    return FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, order_x),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, order_y),
    )


@pytest.fixture
def store() -> ElementDoFs:
    """Return a collection with two elements at different orders."""
    dofs = ElementDoFs()
    first = DegreesOfFreedom(_space(2, 2), np.arange(9, dtype=np.double))
    second = DegreesOfFreedom(_space(3, 3), np.linspace(0.5, 15.5, 16))
    dofs.add_element(first)
    dofs.add_element(second)
    return dofs


def test_add_and_get_roundtrip(store: ElementDoFs) -> None:
    """DoF values round-trip through the collection."""
    assert store.element_count == 2
    assert store.option_count == 2
    np.testing.assert_array_equal(
        store.element_options, np.array([0, 1], dtype=np.uint32)
    )
    np.testing.assert_array_equal(store.offsets, np.array([0, 9, 25], dtype=np.uint64))
    np.testing.assert_array_equal(
        store.dofs(0).values.ravel(), np.arange(9, dtype=np.double)
    )
    np.testing.assert_array_equal(
        store.dofs(1).values.ravel(), np.linspace(0.5, 15.5, 16)
    )
    assert store.dofs(0).n_dofs == 9
    assert isinstance(store.option(0), FunctionSpace)


def test_option_dedup() -> None:
    """Identical function spaces collapse into one option."""
    store = ElementDoFs()
    space = _space(2, 1)
    store.add_element(DegreesOfFreedom(space, np.arange(6, dtype=np.double)))
    store.add_element(DegreesOfFreedom(space, np.linspace(0.0, 1.0, 6)))
    assert store.option_count == 1
    assert store.element_count == 2
    np.testing.assert_array_equal(store.element_options, np.zeros(2, dtype=np.uint32))
    np.testing.assert_array_equal(store.dofs(1).values.ravel(), np.linspace(0.0, 1.0, 6))


def test_views_and_freeze() -> None:
    """Array views expose the storage and freeze the collection."""
    store = ElementDoFs()
    store.add_element(DegreesOfFreedom(_space(1, 1), [1.0, 2.0, 3.0, 4.0]))
    store.add_element(DegreesOfFreedom(_space(1, 1), [5.0, 6.0, 7.0, 8.0]))

    values = store.values
    offsets = store.offsets
    options = store.element_options
    assert values.dtype == np.double
    assert offsets.dtype == np.uint64
    assert options.dtype == np.uint32
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    np.testing.assert_array_equal(offsets, [0, 4, 8])
    np.testing.assert_array_equal(options, [0, 0])

    values[0] = 10.0
    np.testing.assert_array_equal(store.dofs(0).values.ravel(), [10.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        store.add_element(DegreesOfFreedom(_space(1, 1), [9.0, 10.0, 11.0, 12.0]))

    # Overwriting existing values stays allowed after freezing.
    store.set_element_values(1, [30.0, 40.0, 50.0, 60.0])
    np.testing.assert_array_equal(values, [10.0, 2.0, 3.0, 4.0, 30.0, 40.0, 50.0, 60.0])


def test_errors() -> None:
    """Invalid usage raises the expected exceptions."""
    empty = ElementDoFs()
    with pytest.raises(IndexError):
        empty.dofs(0)
    with pytest.raises(TypeError):
        empty.add_element("not degrees of freedom")  # type: ignore[arg-type]

    store = ElementDoFs()
    store.add_element(DegreesOfFreedom(_space(1, 1), [1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError):
        store.set_element_values(0, [1.0])
    with pytest.raises(IndexError):
        store.set_element_values(1, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(IndexError):
        store.dofs(5)
    with pytest.raises(IndexError):
        store.option(1)


def test_from_elements() -> None:
    """The classmethod builds a collection from a DoF sequence."""
    dofs = [
        DegreesOfFreedom(_space(2, 1), np.zeros(6)),
        DegreesOfFreedom(_space(2, 2), np.ones(9)),
    ]
    store = ElementDoFs.from_elements(dofs)
    assert store.element_count == 2
    np.testing.assert_array_equal(store.dofs(0).values.ravel(), np.zeros(6))
    np.testing.assert_array_equal(store.dofs(1).values.ravel(), np.ones(9))


def test_zeros() -> None:
    """zeros() builds a uniformly zero-initialized collection."""
    store = ElementDoFs.zeros(_space(2, 2), 3)
    assert store.element_count == 3
    assert store.option_count == 1
    np.testing.assert_array_equal(store.values, np.zeros(27))
    np.testing.assert_array_equal(
        store.offsets, np.array([0, 9, 18, 27], dtype=np.uint64)
    )
    for element in range(3):
        np.testing.assert_array_equal(store.dofs(element).values.ravel(), np.zeros(9))
    store.set_element_values(1, np.full(9, 2.0))
    np.testing.assert_array_equal(store.dofs(1).values.ravel(), np.full(9, 2.0))
    assert isinstance(store.option(0), FunctionSpace)
    with pytest.raises(ValueError):
        store.add_element(DegreesOfFreedom(_space(2, 2), np.zeros(9)))


def test_zeros_from_options() -> None:
    """zeros_from_options() builds a zero store with per-element spaces."""
    store = ElementDoFs.zeros_from_options([_space(2, 2), _space(3, 3)], [0, 1, 1, 0])
    assert store.element_count == 4
    assert store.option_count == 2
    np.testing.assert_array_equal(
        store.element_options, np.array([0, 1, 1, 0], dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        store.offsets, np.array([0, 9, 25, 41, 50], dtype=np.uint64)
    )
    np.testing.assert_array_equal(store.values, np.zeros(50))
    store.set_element_values(2, np.linspace(0.0, 15.0, 16))
    np.testing.assert_array_equal(
        store.dofs(2).values.ravel(), np.linspace(0.0, 15.0, 16)
    )
    with pytest.raises(ValueError):
        ElementDoFs.zeros_from_options([_space(2, 2)], [0, 1])
    with pytest.raises(TypeError):
        ElementDoFs.zeros("not a space", 1)  # type: ignore[arg-type]
