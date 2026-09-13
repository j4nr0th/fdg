"""Check the batched per-element k-form collection with labeled fields."""

import numpy as np
import pytest
from fdg._fdg import (
    BasisSpecs,
    ElementKForms,
    FunctionSpace,
    KForm,
    KFormSpecs,
)
from fdg.enum_type import BasisType

FIELDS = [("u", 1), ("q", 0)]


@pytest.fixture
def spaces() -> tuple[FunctionSpace, FunctionSpace]:
    """Return the order-2 and order-3 uniform 2D base spaces."""
    space2 = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2),
    )
    space3 = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 3),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 3),
    )
    return space2, space3


@pytest.fixture
def specs(spaces: tuple[FunctionSpace, FunctionSpace]) -> tuple[KFormSpecs, KFormSpecs]:
    """Return the u (1-form) and q (0-form) field specs on the order-2 space."""
    space2 = spaces[0]
    return KFormSpecs(1, space2), KFormSpecs(0, space2)


@pytest.fixture
def store(specs: tuple[KFormSpecs, KFormSpecs]) -> ElementKForms:
    """Return a collection with fields u and q and two elements."""
    u_specs, q_specs = specs
    kforms = ElementKForms(2, u=1, q=0)
    for value in range(2):
        u = KForm(u_specs)
        u.values[:] = float(value)
        q = KForm(q_specs)
        q.values[:] = 10.0 + float(value)
        kforms.add_element(u, q)
    return kforms


def test_labels_and_construction() -> None:
    """Constructor keywords define the labeled fields."""
    kforms = ElementKForms(2, u=1, q=0)
    assert kforms.labels == ("u", "q")
    assert kforms.element_count == 0

    with pytest.raises(TypeError):
        ElementKForms()
    with pytest.raises(ValueError):
        ElementKForms.from_elements(2, [("x", 1), ("x", 0)], [])
    with pytest.raises(ValueError):
        ElementKForms(2, u=3)
    with pytest.raises(TypeError):
        ElementKForms(2, u="1")  # type: ignore[arg-type]


def test_element_grouping_and_getters(store: ElementKForms) -> None:
    """Element values are grouped per element and round-trip per field."""
    assert store.element_count == 2
    np.testing.assert_array_equal(store.offsets("u"), [0, 12, 24])
    np.testing.assert_array_equal(store.offsets("q"), [0, 9, 18])

    for element in range(2):
        u, q = store.kforms(element)
        np.testing.assert_array_equal(u.values, np.full(12, float(element)))
        np.testing.assert_array_equal(q.values, np.full(9, 10.0 + float(element)))
        assert u.specs.order == 1
        assert q.specs.order == 0

        u_specs = store.specs(element, "u")
        q_specs = store.specs(element, "q")
        assert u_specs.order == 1
        assert q_specs.order == 0
        assert u_specs.dimension == 2
        np.testing.assert_array_equal(
            u.specs.base_space.dimension, u_specs.base_space.dimension
        )


def test_per_element_spaces(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """Each element carries its own base space; all fields derive from it."""
    space2, space3 = spaces
    u2, q2 = KFormSpecs(1, space2), KFormSpecs(0, space2)
    u3, q3 = KFormSpecs(1, space3), KFormSpecs(0, space3)
    kforms = ElementKForms(2, u=1, q=0)
    kforms.add_element(KForm(u2), KForm(q2))
    kforms.add_element(KForm(u3), KForm(q3))

    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 36])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 25])
    assert kforms.kform(1, "u").specs.order == 1
    assert kforms.kform(1, "u").specs.dimension == 2
    # The base space of element 1 is the order-3 space.
    assert kforms.specs(1, "u").base_space == space3
    assert kforms.specs(0, "u").base_space == space2


def test_set_field_values(store: ElementKForms) -> None:
    """Overwriting one field of one element is visible through the getters."""
    store.set_field_values(1, "u", np.full(12, 1.0))
    np.testing.assert_array_equal(store.kform(1, "u").values, np.full(12, 1.0))


def test_views_and_freeze(store: ElementKForms) -> None:
    """Array views expose the storage and freeze the collection."""
    u_values = store.values("u")
    assert u_values.dtype == np.double
    assert u_values.shape == (24,)
    q_offsets = store.offsets("q")
    np.testing.assert_array_equal(q_offsets, [0, 9, 18])

    store.set_field_values(1, "q", np.full(9, -1.0))
    np.testing.assert_array_equal(store.values("q")[9:18], np.full(9, -1.0))
    with pytest.raises(ValueError):
        u = KForm(KFormSpecs(1, store.specs(0, "u").base_space))
        store.add_element(u, KForm(store.specs(0, "q")))
    # Overwriting stays allowed after freezing.
    store.set_field_values(0, "u", np.full(12, 5.0))
    np.testing.assert_array_equal(store.kform(0, "u").values, np.full(12, 5.0))


def test_errors(
    specs: tuple[KFormSpecs, KFormSpecs], spaces: tuple[FunctionSpace, FunctionSpace]
) -> None:
    """Invalid usage raises the expected exceptions."""
    u_specs, q_specs = specs
    space2, space3 = spaces
    kforms = ElementKForms(2, u=1, q=0)
    assert kforms.element_count == 0
    with pytest.raises(IndexError):
        kforms.kform(0, "u")
    with pytest.raises(IndexError):
        kforms.specs(0, "u")

    u = KForm(u_specs)
    q = KForm(q_specs)
    with pytest.raises(TypeError):
        kforms.add_element(u)
    with pytest.raises(TypeError):
        kforms.add_element(u, q, q)
    with pytest.raises(TypeError):
        kforms.add_element("not a k-form", q)  # type: ignore[arg-type]

    # A k-form whose order does not match the field is rejected.
    with pytest.raises(TypeError):
        kforms.add_element(KForm(q_specs), q)
    # A k-form on a foreign dimension is rejected.
    other_specs = KFormSpecs(
        1,
        FunctionSpace(
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 3),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, 3),
        ),
    )
    with pytest.raises(TypeError):
        kforms.add_element(KForm(other_specs), KForm(q_specs))
    # The k-forms of one element must share one base space.
    with pytest.raises(TypeError):
        kforms.add_element(KForm(u_specs), KForm(KFormSpecs(0, space3)))

    kforms.add_element(u, q)
    with pytest.raises(KeyError):
        kforms.kform(0, "missing")
    with pytest.raises(KeyError):
        kforms.specs(0, "missing")
    with pytest.raises(ValueError):
        kforms.set_field_values(0, "u", np.full(11, 0.0))
    with pytest.raises(IndexError):
        kforms.set_field_values(5, "u", np.full(12, 0.0))


def test_from_elements(
    specs: tuple[KFormSpecs, KFormSpecs], spaces: tuple[FunctionSpace, FunctionSpace]
) -> None:
    """The classmethod accepts field pairs and per-element groups."""
    u_specs, q_specs = specs
    space2, space3 = spaces
    u3, q3 = KFormSpecs(1, space3), KFormSpecs(0, space3)
    groups = []
    for value, (u_field_specs, q_field_specs) in enumerate(
        [(u_specs, q_specs), (u3, q3), (u_specs, q_specs)]
    ):
        u = KForm(u_field_specs)
        u.values[:] = float(value)
        q = KForm(q_field_specs)
        q.values[:] = -float(value)
        groups.append((u, q))

    kforms = ElementKForms.from_elements(2, FIELDS, groups)
    assert kforms.labels == ("u", "q")
    assert kforms.element_count == 3
    np.testing.assert_array_equal(kforms.kform(2, "u").values, np.full(12, 2.0))
    np.testing.assert_array_equal(kforms.kform(2, "q").values, np.full(9, -2.0))
    np.testing.assert_array_equal(kforms.kform(1, "u").values, np.full(24, 1.0))
    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 36, 48])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 25, 34])
    with pytest.raises(ValueError):
        ElementKForms.from_elements(2, [("x", 1), ("x", 0)], [])


def test_zeros(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """zeros() builds a uniformly zero-initialized collection."""
    space2 = spaces[0]
    kforms = ElementKForms.zeros(2, FIELDS, space2, 3)
    assert kforms.labels == ("u", "q")
    assert kforms.element_count == 3
    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 24, 36])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 18, 27])
    np.testing.assert_array_equal(kforms.values("u"), np.zeros(36))
    for element in range(3):
        np.testing.assert_array_equal(kforms.kform(element, "u").values, np.zeros(12))
        assert kforms.specs(element, "u").order == 1
    kforms.set_field_values(1, "q", np.full(9, 4.0))
    np.testing.assert_array_equal(kforms.kform(1, "q").values, np.full(9, 4.0))
    with pytest.raises(ValueError):
        kforms.add_element(KForm(KFormSpecs(1, space2)), KForm(KFormSpecs(0, space2)))


def test_zeros_from_options(
    spaces: tuple[FunctionSpace, FunctionSpace],
) -> None:
    """zeros_from_options() builds a zero store with per-element spaces."""
    space2, space3 = spaces
    kforms = ElementKForms.zeros_from_options(2, FIELDS, [space2, space3], [0, 1, 1])
    assert kforms.element_count == 3
    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 36, 60])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 25, 41])
    np.testing.assert_array_equal(kforms.values("u"), np.zeros(60))
    assert kforms.specs(2, "u").base_space == space3
    assert kforms.specs(0, "q").base_space == space2
    kforms.set_field_values(2, "u", np.full(24, 2.0))
    np.testing.assert_array_equal(kforms.kform(2, "u").values, np.full(24, 2.0))
    with pytest.raises(ValueError):
        ElementKForms.zeros_from_options(2, FIELDS, [space2], [0, 1])
