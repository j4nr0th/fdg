"""Check the k-form structure type behind the batched per-element values."""

import pytest
from fdg._fdg import (
    BasisSpecs,
    ElementKForms,
    FunctionSpace,
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


def test_fields_and_labels() -> None:
    """Constructor keywords define the labeled fields."""
    structure = MeshKFormSpecs(2, u=1, q=0)
    assert structure.ndim == 2
    assert structure.labels == ("u", "q")
    assert structure.element_count == 0
    assert structure.space_count == 0

    with pytest.raises(TypeError):
        MeshKFormSpecs()
    with pytest.raises(TypeError):
        MeshKFormSpecs(2, u="1")  # type: ignore[arg-type]


def test_dimension_bounds(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """The dimension must be in [1, 63] (element storage precondition)."""
    space2 = spaces[0]
    for ndim in (0, 64):
        with pytest.raises(ValueError, match="ndim in"):
            MeshKFormSpecs(ndim, u=1)
        with pytest.raises(ValueError, match="ndim in"):
            MeshKFormSpecs.from_elements(ndim, FIELDS, [])
        with pytest.raises(ValueError, match="ndim in"):
            MeshKFormSpecs.from_space(ndim, FIELDS, space2, 1)
        with pytest.raises(ValueError, match="ndim in"):
            MeshKFormSpecs.from_options(ndim, FIELDS, [space2], [0])


def test_add_field_rules() -> None:
    """Each field rule violation raises its own error before the C core runs."""
    with pytest.raises(ValueError, match="exceed the dimension"):
        MeshKFormSpecs(2, u=3)
    with pytest.raises(ValueError, match="must not be empty"):
        MeshKFormSpecs(2, **{"": 1})
    with pytest.raises(ValueError, match="already exists"):
        MeshKFormSpecs.from_elements(2, [("u", 1), ("u", 0)], [])
    with pytest.raises(ValueError, match="at least one k-form field"):
        MeshKFormSpecs.from_elements(2, [], [])


def test_add_space_dedup(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """Equal base spaces dedup to one index, distinct ones get new indices."""
    space2, space3 = spaces
    structure = MeshKFormSpecs(2, u=1, q=0)
    assert structure.add_space(space2) == 0
    assert structure.add_space(space2) == 0
    assert structure.space_count == 1
    assert structure.add_space(space3) == 1
    assert structure.space_count == 2

    assert structure.space(0) == space2
    assert structure.space(1) == space3
    with pytest.raises(IndexError):
        structure.space(2)
    with pytest.raises(IndexError):
        structure.space(-1)


def test_add_element(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """Elements reference base spaces by index and report their space."""
    space2, space3 = spaces
    structure = MeshKFormSpecs(2, u=1, q=0)
    structure.add_space(space2)
    structure.add_space(space3)
    structure.add_element(0)
    structure.add_element(1)
    structure.add_element(0)

    assert structure.element_count == 3
    assert structure.element_space(0) == 0
    assert structure.element_space(1) == 1
    assert structure.element_space(2) == 0
    with pytest.raises(IndexError):
        structure.element_space(3)

    with pytest.raises(ValueError):
        structure.add_element(2)
    with pytest.raises(ValueError):
        structure.add_element(-1)


def test_from_space(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """from_space() builds a structure with one shared base space."""
    space2 = spaces[0]
    structure = MeshKFormSpecs.from_space(2, FIELDS, space2, 3)
    assert structure.labels == ("u", "q")
    assert structure.element_count == 3
    assert structure.space_count == 1
    assert structure.element_space(2) == 0

    with pytest.raises(ValueError):
        MeshKFormSpecs.from_space(2, FIELDS, space2, -1)


def test_from_options(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """from_options() builds a structure with per-element base spaces."""
    space2, space3 = spaces
    structure = MeshKFormSpecs.from_options(2, FIELDS, [space2, space3], [0, 1, 1])
    assert structure.element_count == 3
    assert structure.space_count == 2
    assert structure.element_space(0) == 0
    assert structure.element_space(2) == 1

    with pytest.raises(ValueError, match="Space index 1 out of range"):
        MeshKFormSpecs.from_options(2, FIELDS, [space2], [0, 1])
    with pytest.raises(TypeError):
        MeshKFormSpecs.from_options(2, FIELDS, ["not a space"], [0])  # type: ignore[list-item]


def test_from_elements(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """from_elements() derives and dedups the spaces of the element groups."""
    space2, space3 = spaces
    groups = [
        [KFormSpecs(1, space2), KFormSpecs(0, space2)],
        [KFormSpecs(1, space3), KFormSpecs(0, space3)],
        [KFormSpecs(1, space2), KFormSpecs(0, space2)],
    ]
    structure = MeshKFormSpecs.from_elements(2, FIELDS, groups)
    assert structure.labels == ("u", "q")
    assert structure.element_count == 3
    assert structure.space_count == 2
    assert structure.element_space(0) == 0
    assert structure.element_space(1) == 1

    # The specs of one group must share one base function space.
    with pytest.raises(TypeError, match="share one base function space"):
        MeshKFormSpecs.from_elements(
            2, FIELDS, [[KFormSpecs(1, space2), KFormSpecs(0, space3)]]
        )
    # The order of every spec must match its field.
    with pytest.raises(TypeError, match="do not match the field"):
        MeshKFormSpecs.from_elements(
            2, FIELDS, [[KFormSpecs(2, space2), KFormSpecs(0, space2)]]
        )
    # The dimension of every spec must match the collection.
    one_dim = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2))
    with pytest.raises(TypeError, match="do not match the field"):
        MeshKFormSpecs.from_elements(
            2, FIELDS, [[KFormSpecs(1, one_dim), KFormSpecs(0, one_dim)]]
        )


def test_field_specs(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """Field specs are derived from the base space of an element."""
    space2, space3 = spaces
    structure = MeshKFormSpecs.from_options(2, FIELDS, [space2, space3], [0, 1])
    assert structure.field_specs(0, "u").base_space == space2
    assert structure.field_specs(1, "u").base_space == space3
    assert structure.field_specs(1, "u").order == 1
    assert structure.field_specs(0, "q").order == 0

    with pytest.raises(KeyError):
        structure.field_specs(0, "missing")
    with pytest.raises(IndexError):
        structure.field_specs(2, "u")


def test_zero_order_axis_cannot_carry_ordered_field() -> None:
    """A base space with an order-0 axis cannot support a nonzero-order field."""
    zero_space = FunctionSpace(
        BasisSpecs(BasisType.LEGENDRE, 0), BasisSpecs(BasisType.LEGENDRE, 0)
    )
    with pytest.raises(ValueError, match="order 0"):
        MeshKFormSpecs.from_space(2, [("u", 1)], zero_space, 1)

    structure = MeshKFormSpecs(2, u=1, q=0)
    with pytest.raises(ValueError, match="order 0"):
        structure.add_space(zero_space)


def test_space_dimension_must_match() -> None:
    """A base space must have the dimension of the structure."""
    one_dim = FunctionSpace(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 2))
    structure = MeshKFormSpecs(2, u=1, q=0)
    with pytest.raises(ValueError, match="dimensions"):
        structure.add_space(one_dim)


def test_freeze(spaces: tuple[FunctionSpace, FunctionSpace]) -> None:
    """Borrowing the structure freezes it against further mutation."""
    space2, space3 = spaces
    structure = MeshKFormSpecs.from_space(2, FIELDS, space2, 1)
    store = ElementKForms(structure)
    assert store.element_count == 1

    with pytest.raises(ValueError, match="in use by an ElementKForms"):
        structure.add_space(space3)
    with pytest.raises(ValueError, match="in use by an ElementKForms"):
        structure.add_element(0)
