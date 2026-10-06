"""Check the batched per-element k-form values with labeled fields."""

import numpy as np
import pytest
from fdg._fdg import (
    BasisSpecs,
    ElementKForms,
    FunctionSpace,
    KForm,
    KFormSpecs,
    MeshKFormSpecs,
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
def structure(spaces: tuple[FunctionSpace, FunctionSpace]) -> MeshKFormSpecs:
    """Return a structure with fields u and q and two elements on the order-2 space."""
    return MeshKFormSpecs.from_space(2, FIELDS, spaces[0], 2)


@pytest.fixture
def store(
    structure: MeshKFormSpecs, specs: tuple[KFormSpecs, KFormSpecs]
) -> ElementKForms:
    """Return a collection with fields u and q and two filled elements."""
    u_specs, q_specs = specs
    kforms = ElementKForms(structure)
    for value in range(2):
        u = KForm(u_specs)
        u.values[:] = float(value)
        q = KForm(q_specs)
        q.values[:] = 10.0 + float(value)
        kforms.add_element(u, q)
    return kforms


def test_labels_and_construction(
    spaces: tuple[FunctionSpace, FunctionSpace],
) -> None:
    """The borrowed structure defines the labeled fields."""
    structure = MeshKFormSpecs.from_space(2, FIELDS, spaces[0], 0)
    kforms = ElementKForms(structure)
    assert kforms.labels == ("u", "q")
    assert kforms.element_count == 0
    assert kforms.filled_count == 0
    assert kforms.specs is structure

    with pytest.raises(TypeError):
        ElementKForms()
    with pytest.raises(TypeError):
        ElementKForms(structure, structure)
    with pytest.raises(TypeError):
        ElementKForms(2)  # type: ignore[arg-type]


def test_element_grouping_and_getters(store: ElementKForms) -> None:
    """Element values are grouped per element and round-trip per field."""
    assert store.element_count == 2
    assert store.filled_count == 2
    np.testing.assert_array_equal(store.offsets("u"), [0, 12, 24])
    np.testing.assert_array_equal(store.offsets("q"), [0, 9, 18])

    for element in range(2):
        u, q = store.kforms(element)
        np.testing.assert_array_equal(u.values, np.full(12, float(element)))
        np.testing.assert_array_equal(q.values, np.full(9, 10.0 + float(element)))
        assert u.specs.order == 1
        assert q.specs.order == 0

        u_specs = store.field_specs(element, "u")
        q_specs = store.field_specs(element, "q")
        assert u_specs.order == 1
        assert q_specs.order == 0
        assert u_specs.dimension == 2
        np.testing.assert_array_equal(
            u.specs.base_space.dimension, u_specs.base_space.dimension
        )


def test_per_element_spaces(
    spaces: tuple[FunctionSpace, FunctionSpace],
) -> None:
    """Each element carries its own base space; all fields derive from it."""
    space2, space3 = spaces
    u2, q2 = KFormSpecs(1, space2), KFormSpecs(0, space2)
    u3, q3 = KFormSpecs(1, space3), KFormSpecs(0, space3)
    structure = MeshKFormSpecs.from_options(2, FIELDS, [space2, space3], [0, 1])
    kforms = ElementKForms(structure)
    kforms.add_element(KForm(u2), KForm(q2))
    kforms.add_element(KForm(u3), KForm(q3))

    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 36])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 25])
    assert kforms.kform(1, "u").specs.order == 1
    assert kforms.kform(1, "u").specs.dimension == 2
    # The base space of element 1 is the order-3 space.
    assert kforms.field_specs(1, "u").base_space == space3
    assert kforms.field_specs(0, "u").base_space == space2


def test_set_field_values(store: ElementKForms) -> None:
    """Overwriting one field of one element is visible through the getters."""
    store.set_field_values(1, "u", np.full(12, 1.0))
    np.testing.assert_array_equal(store.kform(1, "u").values, np.full(12, 1.0))


def test_views_and_full_store(store: ElementKForms) -> None:
    """Array views expose the storage; a full store rejects new elements."""
    u_values = store.values("u")
    assert u_values.dtype == np.double
    assert u_values.shape == (24,)
    q_offsets = store.offsets("q")
    np.testing.assert_array_equal(q_offsets, [0, 9, 18])

    store.set_field_values(1, "q", np.full(9, -1.0))
    np.testing.assert_array_equal(store.values("q")[9:18], np.full(9, -1.0))

    # The store is full: appending raises IndexError, overwriting stays
    # allowed.
    space = store.specs.space(0)
    with pytest.raises(IndexError):
        store.add_element(KForm(KFormSpecs(1, space)), KForm(KFormSpecs(0, space)))
    store.set_field_values(0, "u", np.full(12, 5.0))
    np.testing.assert_array_equal(store.kform(0, "u").values, np.full(12, 5.0))


def test_filled_count(
    structure: MeshKFormSpecs, specs: tuple[KFormSpecs, KFormSpecs]
) -> None:
    """The cursor starts at zero, advances per element, and caps at the count."""
    u_specs, q_specs = specs
    kforms = ElementKForms(structure)
    assert kforms.filled_count == 0
    for element in range(2):
        kforms.add_element(KForm(u_specs), KForm(q_specs))
        assert kforms.filled_count == element + 1
    assert kforms.filled_count == kforms.element_count
    with pytest.raises(IndexError):
        kforms.add_element(KForm(u_specs), KForm(q_specs))
    assert kforms.filled_count == kforms.element_count


def test_errors(
    specs: tuple[KFormSpecs, KFormSpecs], spaces: tuple[FunctionSpace, FunctionSpace]
) -> None:
    """Invalid usage raises the expected exceptions."""
    u_specs, q_specs = specs
    space2, space3 = spaces
    structure = MeshKFormSpecs.from_space(2, FIELDS, space2, 3)
    kforms = ElementKForms(structure)
    assert kforms.element_count == 3
    assert kforms.filled_count == 0
    # Elements are addressable as soon as the structure defines them, even
    # before they are filled.
    np.testing.assert_array_equal(kforms.kform(0, "u").values, np.zeros(12))
    with pytest.raises(IndexError):
        kforms.kform(5, "u")
    with pytest.raises(IndexError):
        kforms.field_specs(5, "u")

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
    with pytest.raises(TypeError, match="share one base function space"):
        kforms.add_element(KForm(u_specs), KForm(KFormSpecs(0, space3)))
    # The k-forms must use the base space stored for their element.
    with pytest.raises(TypeError, match="stored for element 0"):
        kforms.add_element(KForm(KFormSpecs(1, space3)), KForm(KFormSpecs(0, space3)))

    kforms.add_element(u, q)
    assert kforms.filled_count == 1
    with pytest.raises(KeyError):
        kforms.kform(0, "missing")
    with pytest.raises(KeyError):
        kforms.field_specs(0, "missing")
    with pytest.raises(ValueError):
        kforms.set_field_values(0, "u", np.full(11, 0.0))
    with pytest.raises(IndexError):
        kforms.set_field_values(5, "u", np.full(12, 0.0))


def test_from_elements(
    specs: tuple[KFormSpecs, KFormSpecs], spaces: tuple[FunctionSpace, FunctionSpace]
) -> None:
    """The classmethod accepts a structure and per-element groups."""
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

    structure = MeshKFormSpecs.from_options(2, FIELDS, [space2, space3], [0, 1, 0])
    kforms = ElementKForms.from_elements(structure, groups)
    assert kforms.labels == ("u", "q")
    assert kforms.element_count == 3
    assert kforms.filled_count == 3
    np.testing.assert_array_equal(kforms.kform(2, "u").values, np.full(12, 2.0))
    np.testing.assert_array_equal(kforms.kform(2, "q").values, np.full(9, -2.0))
    np.testing.assert_array_equal(kforms.kform(1, "u").values, np.full(24, 1.0))
    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 36, 48])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 25, 34])
    with pytest.raises(ValueError):
        ElementKForms.from_elements(structure, groups[:-1])


def test_zero_filled(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """A collection on a from_space() structure starts zero-filled."""
    space2 = spaces[0]
    kforms = ElementKForms(MeshKFormSpecs.from_space(2, FIELDS, space2, 3))
    assert kforms.labels == ("u", "q")
    assert kforms.element_count == 3
    assert kforms.filled_count == 0
    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 24, 36])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 18, 27])
    np.testing.assert_array_equal(kforms.values("u"), np.zeros(36))
    for element in range(3):
        np.testing.assert_array_equal(kforms.kform(element, "u").values, np.zeros(12))
        assert kforms.field_specs(element, "u").order == 1
    kforms.set_field_values(1, "q", np.full(9, 4.0))
    np.testing.assert_array_equal(kforms.kform(1, "q").values, np.full(9, 4.0))

    # Appending to a zero-filled store is allowed until the cursor is full;
    # there is no freezing anymore.
    u_specs, q_specs = KFormSpecs(1, space2), KFormSpecs(0, space2)
    u = KForm(u_specs)
    u.values[:] = 7.0
    q = KForm(q_specs)
    q.values[:] = 8.0
    for _ in range(3):
        kforms.add_element(u, q)
    assert kforms.filled_count == 3
    np.testing.assert_array_equal(kforms.kform(0, "u").values, np.full(12, 7.0))
    with pytest.raises(IndexError):
        kforms.add_element(u, q)


def test_per_element_zero_filled(
    spaces: tuple[FunctionSpace, FunctionSpace],
) -> None:
    """A collection on a from_options() structure starts zero-filled."""
    space2, space3 = spaces
    structure = MeshKFormSpecs.from_options(2, FIELDS, [space2, space3], [0, 1, 1])
    kforms = ElementKForms(structure)
    assert kforms.element_count == 3
    np.testing.assert_array_equal(kforms.offsets("u"), [0, 12, 36, 60])
    np.testing.assert_array_equal(kforms.offsets("q"), [0, 9, 25, 41])
    np.testing.assert_array_equal(kforms.values("u"), np.zeros(60))
    assert kforms.field_specs(2, "u").base_space == space3
    assert kforms.field_specs(0, "q").base_space == space2
    kforms.set_field_values(2, "u", np.full(24, 2.0))
    np.testing.assert_array_equal(kforms.kform(2, "u").values, np.full(24, 2.0))
