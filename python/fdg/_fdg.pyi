# TODO: "default: None" should be replaced with "optional"

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Self, final

import numpy as np
import numpy.typing as npt

from fdg.boundary_conditions import (
    BoundaryCondition,
    BoundaryData,
    BoundaryPair,
    BoundaryPairGroup,
)
from fdg.enum_type import _BasisTypeHint, _IntegrationMethodHint

PackedRows = tuple[
    npt.NDArray[np.uintp],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uintp],
    npt.NDArray[np.double],
]

@final
class IntegrationRegistry:
    """Registry for integration rules.

    This registry contains all available integration rules and caches them for
    efficient retrieval.
    """

    def __new__(cls) -> Self: ...
    def usage(self) -> tuple[IntegrationSpecs, ...]:
        """Return the integration rules currently held by the registry.

        Returns
        -------
        tuple of IntegrationSpecs
            Integration specifications of every rule stored in the registry.
        """
        ...
    def clear(self) -> None:
        """Release all held integration rules that are not currently in use."""
        ...

DEFAULT_INTEGRATION_REGISTRY: IntegrationRegistry = ...

@final
class IntegrationSpecs:
    """Type that describes an integration rule.

    Parameters
    ----------
    order : int
        Order of the integration rule.

    method : fdg.IntegrationMethod, default: "gauss"
        Method used for integration.
    """

    def __new__(cls, order: int, method: _IntegrationMethodHint = "gauss") -> Self: ...
    @property
    def order(self) -> int:
        """Order of the integration rule."""
        ...

    @property
    def accuracy(self) -> int:
        """Highest order of polynomial that is integrated exactly."""
        ...

    @property
    def method(self) -> _IntegrationMethodHint:
        """Method used for integration."""
        ...

    def nodes(
        self, registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, /
    ) -> npt.NDArray[np.double]:
        """Get the integration nodes.

        Parameters
        ----------
        registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to retrieve the integration rule.

        Returns
        -------
        array
            Array of integration nodes.
        """
        ...

    def weights(
        self, registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, /
    ) -> npt.NDArray[np.double]:
        """Get the integration weights.

        Parameters
        ----------
        registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to retrieve the integration rule.

        Returns
        -------
        array
            Array of integration weights.
        """
        ...

@final
class BasisRegistry:
    """Registry for basis sets.

    This registry contains all available basis sets and caches them for efficient
    retrieval.
    """

    def __new__(cls) -> Self: ...
    def usage(self) -> tuple[tuple[BasisSpecs, IntegrationSpecs], ...]:
        """Return the basis-integration pairs that are held by the registry.

        Returns
        -------
        tuple of (BasisSpecs, IntegrationSpecs)
            One ``(BasisSpecs, IntegrationSpecs)`` pair for each basis set held in the
            registry.
        """
        ...
    def clear(self) -> None:
        """Release all held basis sets to reduce the memory usage."""
        ...

DEFAULT_BASIS_REGISTRY: BasisRegistry = ...

@final
class CovectorBasis:
    """Type used to specify covector basis bundle.

    Parameters
    ----------
    n : int
        Dimension of the space basis bundle is in.

    *idx : int
        Indices of basis present in the bundle. Should be sorted and non-repeating.
    """

    def __new__(cls, n: int, /, *idx: int) -> Self: ...
    @property
    def ndim(self) -> int:
        """Number of dimensions of the space the basis are in."""
        ...

    @property
    def rank(self) -> int:
        """Number of basis contained."""
        ...

    @property
    def sign(self) -> int:
        """The sign of the basis."""
        ...

    @property
    def index(self) -> int:
        """Index of the basis for the k-form."""
        ...

    def __xor__(self, other: CovectorBasis, /) -> CovectorBasis:
        """Wedge product of the two CovectorBasis."""
        ...

    def __neg__(self) -> CovectorBasis:
        """Negate the CovectorBasis."""
        ...

    def __invert__(self) -> CovectorBasis:
        """Hodge of the CovectorBasis."""
        ...

    def __eq__(self, other) -> bool:
        """Compare two CovectorBasis."""
        ...

    def __gt__(self, other: CovectorBasis) -> bool:
        """Comparison to sort basis."""
        ...

    def __ge__(self, other: CovectorBasis) -> bool:
        """Comparison to sort basis."""
        ...

    def __lt__(self, other: CovectorBasis) -> bool:
        """Comparison to sort basis."""
        ...

    def __le__(self, other: CovectorBasis) -> bool:
        """Comparison to sort basis."""
        ...

    def __bool__(self) -> bool:
        """Check for non-zero basis."""
        ...

    def __str__(self) -> str:
        """Representation of the object."""
        ...

    def __hash__(self) -> int:
        """Hash the object."""
        ...

    def __repr__(self) -> str:
        """Representation of the object."""
        ...

    def __contains__(self, other: int | CovectorBasis) -> bool:
        """Check if the component is contained in the basis."""
        ...

    def normalize(self) -> tuple[int, CovectorBasis]:
        """Normalize the basis by splitting the sign.

        Returns
        -------
        tuple of (int, CovectorBasis)
            Sign of the original basis (``-1`` or ``1``) and the same basis with a
            positive sign, so that their product reproduces the original basis.
        """
        ...

class BasisSpecs:
    """Type that describes specifications for a basis set.

    Parameters
    ----------
    basis_type : fdg.enum_type.BasisType
        Type of the basis used for the set.

    order : int
        Order of the basis in the set.
    """

    def __new__(cls, basis_type: _BasisTypeHint, order: int, /) -> Self: ...
    @property
    def type(self) -> _BasisTypeHint:
        """Type of the basis used for the set."""
        ...

    @property
    def order(self) -> int:
        """Order of the basis in the set."""
        ...

    def values(self, x: npt.NDArray[np.double], /) -> npt.NDArray[np.double]:
        """Evaluate basis functions at given locations.

        Parameters
        ----------
        x : array
            Locations where the basis functions should be evaluated. Must be a
            C-contiguous ``float64`` array.

        Returns
        -------
        array
            Array of basis function values at the specified locations.
            It has one more dimension than ``x``, with the last dimension
            corresponding to the basis function index.
        """
        ...

    def derivatives(self, x: npt.NDArray[np.double], /) -> npt.NDArray[np.double]:
        """Evaluate basis function derivatives at given locations.

        Parameters
        ----------
        x : array
            Locations where the basis function derivatives should be evaluated. Must be
            a C-contiguous ``float64`` array.

        Returns
        -------
        array
            Array of basis function derivatives at the specified locations.
            It has one more dimension than ``x``, with the last dimension
            corresponding to the basis function index.
        """
        ...

@final
class FunctionSpace:
    """Function space defined with basis.

    Function space defined by tensor product of basis functions in each dimension.
    Basis for each dimension are defined by a BasisSpecs object.

    Parameters
    ----------
    *basis_specs : BasisSpecs
        Basis specifications for each dimension of the function space.
    """

    def __new__(cls, *basis_specs: BasisSpecs) -> Self: ...
    @property
    def dimension(self) -> int:
        """Number of dimensions in the function space."""
        ...
    @property
    def basis_specs(self) -> tuple[BasisSpecs, ...]:
        """Basis specifications that define the function space."""
        ...
    @property
    def orders(self) -> tuple[int, ...]:
        """Orders of the basis in each dimension."""
        ...

    def evaluate(
        self, *x: npt.NDArray[np.double], out: npt.NDArray[np.double] | None = None
    ) -> npt.NDArray[np.double]:
        """Evaluate basis functions at given locations.

        Parameters
        ----------
        *x : array
            Coordinates where the basis functions should be evaluated.
            Each array corresponds to a dimension in the function space.
        out : array, optional
            Array where the results should be written to. If not given, a new one
            will be created and returned. It must have the shape of the input arrays
            extended by one dimension per function space dimension, of size the order
            of that dimension plus one.

        Returns
        -------
        array
            Array of basis function values at the specified locations, with one extra
            dimension per dimension of the function space.
        """
        ...

    def values_at_integration_nodes(
        self,
        integration: IntegrationSpace,
        /,
        transpose: bool = False,
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> npt.NDArray[np.double]:
        """Return values of basis at integration points.

        Parameters
        ----------
        integration : IntegrationSpace
            Integration space, the nodes of which are used to evaluate basis at.

        transpose : bool, default: False
            Order the array so that axes indexing the integration points come before
            the ones indexing the bases.

        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to obtain the integration rules from.

        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Registry used to look up basis values.

        Returns
        -------
        array
            Array of basis function values at the integration points locations.
        """
        ...

    def lower_order(self, idim: int) -> FunctionSpace:
        """Create a copy of the space with a lowered order in the specified dimension.

        Parameters
        ----------
        idim : int
            Index of the dimension to lower the order on.

        Returns
        -------
        FunctionSpace
            New function space with a lower order in the specified dimension.
        """
        ...

    def boundary(self, idim: int) -> FunctionSpace:
        """Return the function space on a boundary perpendicular to one dimension.

        The lower and upper boundaries perpendicular to the same dimension have
        the same function space.

        Parameters
        ----------
        idim : int
            Index of the dimension fixed by the boundary.

        Returns
        -------
        FunctionSpace
            New function space containing the basis specifications of the
            remaining dimensions.
        """
        ...

@final
class IntegrationSpace:
    """Integration space defined with integration rules.

    Integration space defined by tensor product of integration rules in each
    dimension. Integration rule for each dimension are defined by an
    IntegrationSpecs object.

    Parameters
    ----------
    *integration_specs : IntegrationSpecs
        Integration specifications for each dimension of the integration space.
    """

    def __new__(cls, *integration_specs: IntegrationSpecs) -> Self: ...
    @property
    def dimension(self) -> int:
        """Number of dimensions in the integration space."""
        ...
    @property
    def integration_specs(self) -> tuple[IntegrationSpecs, ...]:
        """Integration specifications that define the integration space."""
        ...
    @property
    def orders(self) -> tuple[int, ...]:
        """Orders of the integration rules in each dimension."""
        ...

    def nodes(
        self, registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, /
    ) -> npt.NDArray[np.double]:
        """Get the integration nodes of the space.

        Parameters
        ----------
        registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to retrieve the integration rules.

        Returns
        -------
        array
            Array of integration nodes.
        """
        ...

    def weights(
        self, registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, /
    ) -> npt.NDArray[np.double]:
        """Get the integration weights of the space.

        Parameters
        ----------
        registry : fdg.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to retrieve the integration rules.

        Returns
        -------
        array
            Array of integration weights.
        """
        ...

@final
class DegreesOfFreedom:
    """Degrees of freedom associated with a function space.

    Parameters
    ----------
    function_space : FunctionSpace
        Function space the degrees of freedom belong to.
    values : array_like, optional
        Values of the degrees of freedom. When not specified, they are zero initialized.
    """

    def __new__(
        cls, function_space: FunctionSpace, values: npt.ArrayLike | None = None, /
    ) -> Self: ...
    @property
    def function_space(self) -> FunctionSpace:
        """Function space the degrees of freedom belong to."""
        ...
    @property
    def n_dofs(self) -> int:
        """Total number of degrees of freedom."""
        ...
    @property
    def values(self) -> npt.NDArray[np.double]:
        """Coefficient values of the degrees of freedom.

        These are the expansion coefficients of the discrete function, not
        sampled function values. Do not confuse them with
        :attr:`CoordinateMap.values`, which holds the mapped coordinates
        evaluated at the integration points of the map's own integration
        space.
        """
        ...
    @values.setter
    def values(self, value: npt.ArrayLike) -> None:
        """Assign new values to the degrees of freedom."""
        ...
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the degrees of freedom."""
        ...

    def reconstruct_at_integration_points(
        self,
        integration_space: IntegrationSpace,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
        *,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Reconstruct the function at the integration points of the given space.

        Parameters
        ----------
        integration_space : IntegrationSpace
            Integration space where the function should be reconstructed.
        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to retrieve the integration rules.
        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Registry used to retrieve the basis specifications.
        out : array, optional
            Array where the results should be written to. If not given, a new one
            will be created and returned. It should have the same shape as the
            integration points.

        Returns
        -------
        array
            Array of reconstructed function values at the integration points.
        """
        ...

    def reconstruct_derivative_at_integration_points(
        self,
        integration_space: IntegrationSpace,
        idim: int | Sequence[int],
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
        *,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Reconstruct the derivative of the function in given dimension.

        Parameters
        ----------
        integration_space : IntegrationSpace
            Integration space where the function derivative should be reconstructed.
        idim : int or Sequence[int]
            Dimension in which the derivative should be computed, or the sequence of
            dimensions in which it should be computed. All values in a sequence should
            appear at most once.
        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to retrieve the integration rules.
        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Registry used to retrieve the basis specifications.
        out : array, optional
            Array where the results should be written to. If not given, a new one
            will be created and returned. It should have the same shape as the
            integration points.

        Returns
        -------
        array
            Array of reconstructed function derivative values at the integration points.
        """
        ...

    def derivative(self, idim: int) -> DegreesOfFreedom:
        """Return degrees of freedom of the derivative along the reference dimension.

        Parameters
        ----------
        idim : int
            Index of the reference dimension along which the derivative should be taken.

        Returns
        -------
        DegreesOfFreedom
            Degrees of freedom of the computed derivative.
        """
        ...

    def plane_projection(self, idim: int, x: float) -> DegreesOfFreedom:
        """Compute the projection of degrees of freedom on a plane.

        Parameters
        ----------
        idim : int
            Index of the dimension that is fixed.

        x : float
            Position of the plane in that dimension.

        Returns
        -------
        DegreesOfFreedom
            Degrees of freedom on the specified plane.
        """
        ...

    def reverse_orientation(self, idim: int) -> DegreesOfFreedom:
        """Reverse the orientation of DoFs.

        Maps the domain of basis functions for dimension ``idim`` from :math:`[-1, +1]`
        to :math:`[+1, -1]`.

        Parameters
        ----------
        idim : int
            Index of the dimension on which the orientation should be reversed.

        Returns
        -------
        DegreesOfFreedom
            Degrees of freedom with reversed orientation on the specified dimension.
        """
        ...

    def lagrange_projection(
        self,
        orders: npt.ArrayLike | None = None,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> DegreesOfFreedom:
        """Compute projection of degrees of freedom with Lagrange basis.

        Parameters
        ----------
        orders : array_like
            Orders in each dimension. If nothing is given, then orders are taken to be
            same as needed to exactly represent the degrees of freedom.

        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry used to obtain the integration rules from.

        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Registry used to look up the basis specifications.

        Returns
        -------
        DegreesOfFreedom
            Degrees of freedom using Lagrange basis of specified orders.
        """
        ...

@final
class ElementDoFs:
    """Batched degrees of freedom: the DoF vector of every element.

    Instead of storing one Python object per element, this type stores a
    small table of distinct function-space options and, per element, an
    index into that table along with an offset into one large, flat array
    of values.

    Data is added with :meth:`add_element` or :meth:`from_elements`.
    Accessing the array views :attr:`values`, :attr:`offsets` or
    :attr:`element_options` freezes the collection: no further elements can
    be added, but values of existing elements can still be overwritten with
    :meth:`set_element_values`. Per-element DoFs are retrieved as a regular
    :class:`DegreesOfFreedom` with :meth:`dofs`.
    """

    def __new__(cls) -> Self: ...
    @classmethod
    def from_elements(cls, dofs: Sequence[DegreesOfFreedom], /) -> ElementDoFs:
        """Create a new collection from a sequence of degrees of freedom.

        Parameters
        ----------
        dofs : Sequence[DegreesOfFreedom]
            Degrees of freedom of every element, in element order.

        Returns
        -------
        ElementDoFs
            Collection holding the degrees of freedom of all elements.
        """
        ...
    @classmethod
    def zeros(cls, space: FunctionSpace, count: int, /) -> ElementDoFs:
        """Create a zero-initialized collection with one shared function space.

        Parameters
        ----------
        space : FunctionSpace
            Function space of every element.
        count : int
            Number of zero-initialized elements.

        Returns
        -------
        ElementDoFs
            Collection holding ``count`` zero elements.
        """
        ...
    @classmethod
    def zeros_from_options(
        cls, spaces: Sequence[FunctionSpace], indices: Sequence[int], /
    ) -> ElementDoFs:
        """Create a zero-initialized collection with per-element function spaces.

        Parameters
        ----------
        spaces : Sequence[FunctionSpace]
            The distinct function spaces of the options table.
        indices : Sequence[int]
            Option index of every element; also fixes the element count.

        Returns
        -------
        ElementDoFs
            Collection holding one zero element per index.
        """
        ...
    def add_element(self, dofs: DegreesOfFreedom, /) -> None:
        """Add the degrees of freedom of one element to the collection.

        Parameters
        ----------
        dofs : DegreesOfFreedom
            Degrees of freedom of the element.
        """
        ...
    def dofs(self, element_id: int, /) -> DegreesOfFreedom:
        """Get the degrees of freedom of one element.

        Parameters
        ----------
        element_id : int
            Index of the element.

        Returns
        -------
        DegreesOfFreedom
            Degrees of freedom holding the stored values of the element.
        """
        ...
    def option(self, index: int, /) -> FunctionSpace:
        """Get the function space of one option.

        Parameters
        ----------
        index : int
            Index into the options table.

        Returns
        -------
        FunctionSpace
            Function space of the option.
        """
        ...
    def set_element_values(self, element_id: int, values: npt.ArrayLike, /) -> None:
        """Overwrite the stored values of one element.

        Parameters
        ----------
        element_id : int
            Index of the element.
        values : array_like
            Flat array with as many entries as the element's option stores.
        """
        ...
    @property
    def element_count(self) -> int:
        """Number of stored elements."""
        ...
    @property
    def option_count(self) -> int:
        """Number of distinct options in the options table."""
        ...
    @property
    def values(self) -> npt.NDArray[np.double]:
        """Flat array of all element values. Freezes the collection on access."""
        ...
    @property
    def offsets(self) -> npt.NDArray[np.uint64]:
        """CSR offsets of the per-element value blocks.

        The array has ``element_count + 1`` entries. Accessing this property
        freezes the collection.
        """
        ...
    @property
    def element_options(self) -> npt.NDArray[np.uint32]:
        """Option index of every element. Freezes the collection on access."""
        ...

@final
class ElementKForms:
    """Batched k-form data: a fixed set of labeled fields, values grouped per element.

    When setting up a finite element system one computes element matrices,
    so the k-forms of one element are needed together. Each keyword
    argument of the constructor defines one field: the keyword is its
    unique label and the value its k-form order. All fields are derived
    from one base function space per element; the distinct base spaces
    form the options of the collection. Every element added with
    :meth:`add_element` then stores the values of all fields, in field
    order.

    Accessing the array views :meth:`values` or :meth:`offsets` freezes
    the collection: no further elements can be added, but the values of
    existing elements can still be overwritten with
    :meth:`set_field_values`.

    Parameters
    ----------
    ndim : int
        Number of reference dimensions, shared by all fields; must be positive.

    **fields : int
        One keyword argument per k-form field: the keyword is the unique
        label of the field, the value its order, ``0 <= order <= ndim``.
    """

    def __new__(cls, ndim: int, /, **fields: int) -> Self: ...
    @classmethod
    def from_elements(
        cls,
        ndim: int,
        fields: Sequence[tuple[str, int]],
        elements: Sequence[tuple[KForm, ...]],
        /,
    ) -> ElementKForms:
        """Create a new collection from labeled fields and per-element k-form groups.

        Parameters
        ----------
        ndim : int
            Number of reference dimensions, shared by all fields.
        fields : Sequence[tuple[str, int]]
            One ``(label, order)`` pair per k-form field, in field order.
        elements : Sequence[tuple[KForm, ...]]
            One tuple of k-forms per element, in field order. All k-forms
            of one element must share one base function space; distinct
            spaces across elements are stored as separate options.

        Returns
        -------
        ElementKForms
            Collection holding the k-form values of all elements.
        """
        ...
    @classmethod
    def zeros(
        cls,
        ndim: int,
        fields: Sequence[tuple[str, int]],
        space: FunctionSpace,
        count: int,
        /,
    ) -> ElementKForms:
        """Create a zero-initialized collection with one shared base function space.

        Parameters
        ----------
        ndim : int
            Number of reference dimensions, shared by all fields.
        fields : Sequence[tuple[str, int]]
            One ``(label, order)`` pair per k-form field, in field order.
        space : FunctionSpace
            Base function space shared by every element; all fields are
            derived from it.
        count : int
            Number of zero-initialized elements.

        Returns
        -------
        ElementKForms
            Collection holding ``count`` zero elements.
        """
        ...
    @classmethod
    def zeros_from_options(
        cls,
        ndim: int,
        fields: Sequence[tuple[str, int]],
        spaces: Sequence[FunctionSpace],
        indices: Sequence[int],
        /,
    ) -> ElementKForms:
        """Create a zero-initialized collection with per-element base spaces.

        Parameters
        ----------
        ndim : int
            Number of reference dimensions, shared by all fields.
        fields : Sequence[tuple[str, int]]
            One ``(label, order)`` pair per k-form field, in field order.
        spaces : Sequence[FunctionSpace]
            The distinct base function spaces; all fields are derived from
            the base space of an element.
        indices : Sequence[int]
            Index into ``spaces`` for every element; also fixes the
            element count.

        Returns
        -------
        ElementKForms
            Collection holding one zero element per index.
        """
        ...
    def add_element(self, *kforms: KForm) -> None:
        """Add the k-form values of one element to the collection.

        Parameters
        ----------
        *kforms : KForm
            One k-form per field, in field order. The order and dimension
            of every k-form must match its field, and all k-forms of the
            element must share one base function space.
        """
        ...
    def kform(self, element_id: int, label: str, /) -> KForm:
        """Get the values of one field of one element as a k-form.

        Parameters
        ----------
        element_id : int
            Index of the element.
        label : str
            Label of the field.

        Returns
        -------
        KForm
            K-form holding the stored values of the field.
        """
        ...
    def kforms(self, element_id: int, /) -> tuple[KForm, ...]:
        """Get the k-forms of all fields of one element, in field order.

        Parameters
        ----------
        element_id : int
            Index of the element.

        Returns
        -------
        tuple[KForm, ...]
            One k-form per field, in field order.
        """
        ...
    def specs(self, element_id: int, label: str, /) -> KFormSpecs:
        """Get the specifications of one field on the base space of one element.

        Parameters
        ----------
        element_id : int
            Index of the element.
        label : str
            Label of the field.

        Returns
        -------
        KFormSpecs
            Specifications of the field, derived from the element's base
            function space.
        """
        ...
    def set_field_values(
        self, element_id: int, label: str, values: npt.ArrayLike, /
    ) -> None:
        """Overwrite the stored values of one field of one element.

        Parameters
        ----------
        element_id : int
            Index of the element.
        label : str
            Label of the field.
        values : array_like
            Flat array with as many entries as the field stores per element.
        """
        ...
    def values(self, label: str, /) -> npt.NDArray[np.double]:
        """Get the flat value array of one field. Freezes the collection on access.

        Parameters
        ----------
        label : str
            Label of the field.

        Returns
        -------
        array
            One block of field values per element.
        """
        ...
    def offsets(self, label: str, /) -> npt.NDArray[np.uint64]:
        """Get the CSR offsets of one field's per-element value blocks.

        Accessing this method freezes the collection.

        Parameters
        ----------
        label : str
            Label of the field.

        Returns
        -------
        array
            Array with ``element_count + 1`` offsets.
        """
        ...
    @property
    def element_count(self) -> int:
        """Number of stored elements."""
        ...
    @property
    def labels(self) -> tuple[str, ...]:
        """Labels of the k-form fields, in field order."""
        ...

@final
class MeshGeometry:
    """Batched geometry data: the space map of every element of a mesh.

    Elements of a mesh often share only a few distinct geometry
    specifications (function spaces and integration spaces). Instead of
    storing one Python object per element, this type stores a small table of
    distinct options and, per element, an index into that table along with
    an offset into one large, flat array of coordinate values.

    Data is added with :meth:`add_element` or one of the constructors
    :meth:`from_elements` and :meth:`from_mesh_points`. Accessing the array
    views :attr:`values`, :attr:`offsets` or :attr:`element_options` freezes
    the collection: no further elements can be added, but values of existing
    elements can still be overwritten with :meth:`set_element_values`.
    Per-element geometry is retrieved as a regular :class:`SpaceMap` with
    :meth:`space_map`.
    """

    def __new__(cls) -> Self: ...
    @classmethod
    def from_elements(
        cls, elements: Sequence[tuple[SpaceMap, *tuple[DegreesOfFreedom, ...]]], /
    ) -> MeshGeometry:
        """Create a new collection from space maps with their geometry degrees of freedom.

        Parameters
        ----------
        elements : Sequence[tuple[SpaceMap, DegreesOfFreedom, ...]]
            Geometry of every element: its space map and one geometry degree of
            freedom per coordinate, in element order.

        Returns
        -------
        MeshGeometry
            Collection holding the geometry of all elements.
        """
        ...
    @classmethod
    def from_mesh_points(
        cls, mesh: Mesh, points: npt.ArrayLike, integration: IntegrationSpace, /
    ) -> MeshGeometry:
        """Create geometry data from the physical coordinates of the mesh points.

        Every element is equipped with a multilinear (order-1 Lagrange on
        uniform nodes) geometry, matching the convention of
        ``Hypercube.from_corners``: corner ``k`` of an element lies on the
        positive side of axis ``d`` exactly when bit ``d`` of ``k`` is set.

        Parameters
        ----------
        mesh : Mesh
            Mesh providing the elements and the point connectivity.
        points : array_like
            Array of shape ``(mesh.point_count, C)`` with the physical
            coordinates of every mesh point.
        integration : IntegrationSpace
            Integration space with one specification per reference dimension.

        Returns
        -------
        MeshGeometry
            Geometry collection with one element per mesh element, in mesh
            element order.
        """
        ...
    def add_element(self, space_map: SpaceMap, *dofs: DegreesOfFreedom) -> None:
        """Add the geometry of one element to the collection.

        Parameters
        ----------
        space_map : SpaceMap
            Space map of the element.
        *dofs : DegreesOfFreedom
            Geometry degrees of freedom, one per coordinate of the space map.
            All of them must share one function space.
        """
        ...
    def space_map(self, element_id: int, /) -> SpaceMap:
        """Get the geometry of one element as a space map.

        Parameters
        ----------
        element_id : int
            Index of the element.

        Returns
        -------
        SpaceMap
            Space map built from the stored coordinate data of the element.
        """
        ...
    def option(self, index: int, /) -> tuple[FunctionSpace, IntegrationSpace]:
        """Get the geometry specification of one option.

        Parameters
        ----------
        index : int
            Index into the options table.

        Returns
        -------
        tuple[FunctionSpace, IntegrationSpace]
            Function and integration space of the option.
        """
        ...
    def set_element_values(self, element_id: int, values: npt.ArrayLike, /) -> None:
        """Overwrite the stored coordinate values of one element.

        Parameters
        ----------
        element_id : int
            Index of the element.
        values : array_like
            Flat array with as many entries as the element's option stores.
        """
        ...
    @property
    def element_count(self) -> int:
        """Number of stored elements."""
        ...
    @property
    def option_count(self) -> int:
        """Number of distinct options in the options table."""
        ...
    @property
    def values(self) -> npt.NDArray[np.double]:
        """Flat array of all element values. Freezes the collection on access."""
        ...
    @property
    def offsets(self) -> npt.NDArray[np.uint64]:
        """CSR offsets of the per-element value blocks.

        The array has ``element_count + 1`` entries. Accessing this property
        freezes the collection.
        """
        ...
    @property
    def element_options(self) -> npt.NDArray[np.uint32]:
        """Option index of every element. Freezes the collection on access."""
        ...

@final
class KFormSpecs:
    """Differential k-form specification.

    Parameters
    ----------
    order : int
        Order of the k-form.

    base_space : FunctionSpace
        Base space to use for the k-forms. This is also the space in which 0-forms
        are defined.
    """

    def __new__(cls, order: int, base_space: FunctionSpace) -> Self: ...
    @property
    def order(self) -> int:
        """Order of the k-form."""
        ...
    @property
    def base_space(self) -> FunctionSpace:
        """Base function space the k-form is based in."""
        ...

    @property
    def dimension(self) -> int:
        """Dimension of the space the k-form is in."""
        ...

    @property
    def component_count(self) -> int:
        """Number of components in the k-form."""
        ...

    def get_component_function_space(self, idx: int) -> FunctionSpace:
        """Get the function space for a component.

        Parameters
        ----------
        idx : int
            Index of the component.

        Returns
        -------
        FunctionSpace
            Function space corresponding to the k-form component with the specified index.
        """
        ...

    def get_component_basis(self, idx: int) -> CovectorBasis:
        """Get covector basis bundle for a component.

        Parameters
        ----------
        idx : int
            Index of the component.

        Returns
        -------
        CovectorBasis
            Covector basis bundle corresponding to the k-form component with the specified
            index.
        """
        ...

    def get_component_slice(self, idx: int) -> slice:
        """Get the slice corresponding to degrees of freedom of a k-form component.

        The resulting slice can be used to index into the flattened array of degrees
        of freedom to get the DoFs corresponding to a particular component.

        Parameters
        ----------
        idx : int
            Index of the k-form component.

        Returns
        -------
        slice
            Slice of the flattened array of all k-form degrees of freedom that corresponds
            to degrees of freedom of the specified component.
        """
        ...

    @property
    def component_dof_counts(self) -> npt.NDArray[np.int64]:
        """Number of DoFs in each component."""
        ...

@final
class KForm:
    """Type holding the degrees of freedom of a k-form.

    Parameters
    ----------
    specs : KFormSpecs
        Specification of the k-form that is to be created.
    """

    def __new__(cls, specs: KFormSpecs) -> Self: ...
    @property
    def specs(self) -> KFormSpecs:
        """Specifications of the k-form."""
        ...

    @property
    def values(self) -> npt.NDArray[np.double]:
        """Values of all k-form degrees of freedom."""
        ...

    def get_component_dofs(self, idx: int) -> npt.NDArray[np.double]:
        """Get the array containing the degrees of freedom for a k-form component.

        Parameters
        ----------
        idx : int
            Index of the k-form component.

        Returns
        -------
        array
            Array containing the degrees of freedom. This is not a copy, so changing
            values in it will change the values of degrees of freedom.
        """
        ...

    def get_component(self, idx: int) -> DegreesOfFreedom:
        """Get the DegreesOfFreedom object corresponding to a k-form component.

        Note that this object contains a copy of the degrees of freedom for
        the component, so changing values in it will not change the values of
        the k-form. If you wish to change them, consider using the
        ``get_component_dofs`` method instead.

        Parameters
        ----------
        idx : int
            Index of the k-form component.

        Returns
        -------
        DegreesOfFreedom
            DegreesOfFreedom object containing the degrees of freedom for the
            specified k-form component.
        """
        ...

# Fields of a mesh iteration tuple: (mdim, object_id, element_ids, orientations).
# ``orientations`` has shape (element_count, ndim); row ``i`` is the orientation
# record of ``element_ids[i]``.
MeshSharedObject = tuple[int, int, npt.NDArray[np.uint64], npt.NDArray[np.int8]]

@final
class Mesh:
    """Topological mesh built from connected hypercube elements.

    The mesh holds the full topology of a set of hypercube elements — object
    collections per dimension plus immersion information, but no geometry. Its
    main use is generating continuity constraints between neighboring elements,
    see ``compute_kform_continuity_constraints``.

    The type cannot be instantiated directly; use ``from_corners`` or
    ``from_collections``.
    """

    @classmethod
    def from_corners(cls, ndim: int, corners: npt.ArrayLike, /) -> Self:
        """Create a mesh from the corner point IDs of every hypercube element.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the mesh.

        corners : array_like
            Corner point IDs of every hypercube element, ``2**ndim`` entries per
            element; the same point IDs name shared points.

        Returns
        -------
        Mesh
            Mesh built from the given corners.
        """
        ...

    @classmethod
    def from_collections(
        cls,
        ndim: int,
        point_count: int,
        collections: tuple[npt.ArrayLike, ...],
        /,
    ) -> Self:
        """Create a mesh from the collections of topological objects.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the mesh.

        point_count : int
            Number of mesh points represented implicitly by point IDs.

        collections : tuple of array_like
            Boundary-ID arrays for mesh objects of dimensions 1 through N. The
            last collection contains the N-dimensional elements.

        Returns
        -------
        Mesh
            Mesh built from the given collections.
        """
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions of the space the mesh is in."""
        ...

    @property
    def point_count(self) -> int:
        """Number of points of the mesh."""
        ...

    @property
    def element_count(self) -> int:
        """Number of elements of the mesh."""
        ...

    @property
    def collections(self) -> tuple[npt.NDArray[np.uint64], ...]:
        """Boundary-ID arrays of the mesh objects of every dimension (uint64 copies)."""
        ...

    # TODO: rework this signature to only take fixed axes and not need the varying ones.
    def element_object(self, element_id: int, axis: Sequence[int], /) -> int:
        """Look up the global ID of the object at a position within one element.

        Parameters
        ----------
        element_id : int
            ID of the element.

        axis : sequence of int
            Axis specification of length ``ndim``; entry ``i`` is 0 for a free
            axis, or ``i + 1`` / ``-(i + 1)`` to fix the axis at its end / start
            side. At least one axis must be fixed.

        Returns
        -------
        int
            Global object ID: a point ID for objects of dimension 0, otherwise
            an index into the corresponding collection.
        """
        ...

    def iterate_shared(self, mdim: int, /) -> list[MeshSharedObject]:
        """Iterate over all objects of one dimension shared by at least two elements.

        Parameters
        ----------
        mdim : int
            Dimension of the objects, ``0 <= mdim < ndim``.

        Returns
        -------
        list of tuple
            One ``(mdim, object_id, element_ids, orientations)`` tuple per
            shared object.
        """
        ...

    def iterate_shared_all(self) -> list[MeshSharedObject]:
        """Iterate over all shared objects, from dimension ``ndim - 1`` down to 0.

        Returns
        -------
        list of tuple
            One ``(mdim, object_id, element_ids, orientations)`` tuple per
            shared object.
        """
        ...

    def iterate_boundary(self, mdim: int, /) -> list[MeshSharedObject]:
        """Iterate over all objects of one dimension on the outer boundary of the mesh.

        Parameters
        ----------
        mdim : int
            Dimension of the objects, ``0 <= mdim < ndim``.

        Returns
        -------
        list of tuple
            One ``(mdim, object_id, element_ids, orientations)`` tuple per
            boundary object.
        """
        ...

    def iterate_boundary_all(self) -> list[MeshSharedObject]:
        """Iterate over all boundary objects, from dimension ``ndim - 1`` down to 0.

        Returns
        -------
        list of tuple
            One ``(mdim, object_id, element_ids, orientations)`` tuple per
            boundary object.
        """
        ...

    # TODO: remove the basis_type parameter
    def compute_kform_continuity_constraints(
        self,
        element_specs: Sequence[KFormSpecs],
        element_maps: Sequence[SpaceMap] | None = None,
        /,
        *,
        basis_type: _BasisTypeHint | None = None,
        c1_continuous: bool = False,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> tuple[
        npt.NDArray[np.uintp],
        npt.NDArray[np.uint64],
        npt.NDArray[np.uint32],
        npt.NDArray[np.uintp],
        npt.NDArray[np.double],
    ]:
        """Assemble k-form continuity rows between neighboring elements.

        Shared objects are visited from the highest dimension down to points.
        Every shared object contributes one row per test function, pairing the
        first element of its ascending incident-element list (the anchor) with
        each of its remaining elements, so the anchor links all of them
        without introducing a cycle.

        The trace test spaces are derived automatically. A component exists
        only when all of its covector axes lie in the shared object (there are
        ``mdim`` choose ``k`` of them). Each component reads ``order`` functions
        of the common space — the per-axis minimum order of the incident
        elements — on its covector axes and the leading ``order - 1``
        functions on the remaining axes (floored at zero). A component with a
        zero-function axis contributes no rows.

        Parameters
        ----------
        element_specs : Sequence[KFormSpecs]
            One volume k-form specification per mesh element. The sequence
            must contain exactly ``element_count`` entries. All specifications
            must have the mesh dimension and the same k-form degree; their
            basis orders may differ.

        element_maps : Sequence[SpaceMap], default: None
            One reference-to-physical map per mesh element supplying the
            physical trace geometry. Required unless ``c1_continuous`` is set.

        basis_type : fdg.BasisType or str, default: None
            Accepted as ``None`` or ``"legendre"`` only: the derived test
            spaces always use the Legendre family, any other family raises
            ``ValueError``.

        c1_continuous : bool, default: False
            Pair reference-space traces without geometry factors. With this
            flag set, reference-domain continuity is imposed and
            ``element_maps`` may be omitted.

        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry to get the quadrature rules from.

        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Registry to get the basis tables and endpoint values from.

        Returns
        -------
        row_offsets : array
            ``uintp`` CSR-like row boundaries of length
            ``number_of_rows + 1``. Entry ``i`` belongs to
            ``[row_offsets[i], row_offsets[i + 1])``. Empty output is
            represented by ``[0]``.

        element_ids : array
            ``uint64`` global element ID for each packed entry.

        components : array
            ``uint32`` element-frame k-form component for each packed entry.

        local_dofs : array
            ``uintp`` local DoF index within the component named by
            ``components``.

        coefficients : array
            ``double`` trace coefficient of each packed entry: the side sign
            (+1 for the anchor element, -1 for the paired one) times the basis
            value of that element's own space at the shared end; with
            ``c1_continuous`` only the side sign applies.
        """
        ...

    def compute_kform_global_constraints(
        self,
        element_specs: Sequence[KFormSpecs],
        element_maps: Sequence[SpaceMap] | None = None,
        boundary_conditions: Mapping[int, BoundaryData]
        | Sequence[BoundaryCondition]
        | None = None,
        periodic_pairs: Sequence[BoundaryPair | BoundaryPairGroup] | None = None,
        /,
        *,
        basis_type: _BasisTypeHint | None = None,
        c1_continuous: bool = False,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> tuple[
        tuple[
            npt.NDArray[np.uintp],
            npt.NDArray[np.uint64],
            npt.NDArray[np.uint32],
            npt.NDArray[np.uintp],
            npt.NDArray[np.double],
        ],
        npt.NDArray[np.double],
    ]:
        """Assemble global k-form trace constraints and their right-hand side.

        Automatically derived shared-object continuity rows are augmented by
        optional physical boundary data and explicit periodic or transformed
        boundary pairs. Boundary face data are propagated to all
        lower-dimensional descendants and imposed once on a deterministic
        owner element, so adjacent prescribed faces do not duplicate edge or
        point equations.

        Parameters
        ----------
        element_specs : Sequence[KFormSpecs]
            One volume k-form specification per mesh element, exactly as for
            :meth:`compute_kform_continuity_constraints`.

        element_maps : Sequence[SpaceMap], default: None
            One reference-to-physical map per mesh element. Required whenever
            boundary data or periodic pairs are given, and whenever
            ``c1_continuous`` is not set.

        boundary_conditions : mapping or sequence, default: None
            Prescribed boundary data; see the :mod:`fdg.boundary_conditions`
            documentation for the accepted forms.

        periodic_pairs : sequence of BoundaryPair or BoundaryPairGroup, default: None
            Explicit pairs of outer faces, or ordered groups of equal-length
            face collections. Each group is expanded to corresponding lower
            strata; ``axis_map`` is a signed permutation of canonical boundary
            axes, allowing reversals and axis permutations. Duplicate
            lower-stratum relations are reduced to an acyclic forest.

        basis_type : fdg.BasisType or str, default: None
            Accepted as ``None`` or ``"legendre"`` only: the derived test
            spaces always use the Legendre family, any other family raises
            ``ValueError``.

        c1_continuous : bool, default: False
            Impose continuity in reference space without geometry factors;
            ``element_maps`` may be omitted in that case unless boundary data
            or periodic pairs require them.

        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry to get the quadrature rules from.

        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Registry to get the trace basis tables from.

        Returns
        -------
        rows : tuple of arrays
            ``(row_offsets, element_ids, components, local_dofs, coefficients)``
            in the global packed-row format.

        rhs : array
            ``double`` prescribed value per packed constraint row. Shared and
            periodic rows have zero right-hand side.
        """
        ...

@final
class CoordinateMap:
    """Mapping between reference and physical coordinates.

    This type wraps :meth:`DegreesOfFreedom.reconstruct_at_integration_points()`
    and :meth:`DegreesOfFreedom.reconstruct_derivative_at_integration_points()`;
    one coordinate map evaluates a single coordinate together with all of its
    first derivatives at every integration point. In N-dimensional space, N such
    maps are used to represent the full mapping.

    Parameters
    ----------
    dofs : DegreesOfFreedom
        Degrees of freedom that define the coordinate map.
    integration_space : IntegrationSpace
        Integration space used for the mapping.
    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry used to retrieve the integration rules.
    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry used to retrieve the basis specifications.
    """

    def __new__(
        cls,
        dofs: DegreesOfFreedom,
        integration_space: IntegrationSpace,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> Self: ...
    @property
    def dimension(self) -> int:
        """Number of dimensions in the coordinate map."""
        ...

    @property
    def integration_space(self) -> IntegrationSpace:
        """Integration space used for the mapping."""
        ...
    @property
    def values(self) -> npt.NDArray[np.double]:
        """Mapped coordinate values at the integration points.

        These are the physical coordinates of the map evaluated at every
        integration point of this map's own integration space, not
        degree-of-freedom coefficients. Do not confuse them with
        :attr:`DegreesOfFreedom.values`, which holds the expansion
        coefficients passed at construction.
        """
        ...

    def gradient(self, idim: int, /) -> npt.NDArray[np.double]:
        """Retrieve the gradient of the coordinate map for the given dimension.

        Parameters
        ----------
        idim : int
            Index of the dimension, in range ``[0, dimension)``.

        Returns
        -------
        array
            Derivative of the mapped coordinate with respect to that dimension,
            sampled at the integration points of the map.
        """
        ...

@final
class SpaceMap:
    """Mapping between a reference space and a physical space.

    A mapping from a reference space to a physical space, which maps the
    :math:`N`-dimensional reference space to an :math:`M`-dimensional
    physical space. With this mapping, it is possible to integrate a
    quantity on a deformed element.

    Parameters
    ----------
    *coordinates : CoordinateMap
        Maps for each coordinate of physical space. All of these must be
        defined on the same :class:`IntegrationSpace`.
    """

    def __new__(cls, *coordinates: CoordinateMap) -> Self: ...
    def coordinate_map(self, idx: int) -> CoordinateMap:
        """Return the coordinate map for the specified dimension.

        Parameters
        ----------
        idx : int
            Index of the dimension for which the map should be returned.

        Returns
        -------
        CoordinateMap
            Map used for the specified coordinate.
        """
        ...

    @property
    def integration_space(self) -> IntegrationSpace:
        """Integration space used by the mapping."""
        ...

    @property
    def input_dimensions(self) -> int:
        """Dimension of the input/reference space."""
        ...

    @property
    def output_dimensions(self) -> int:
        """Dimension of the output/physical space."""
        ...

    @property
    def determinant(self) -> npt.NDArray[np.double]:
        """Array with the values of determinant at integration points."""
        ...

    @property
    def inverse_map(self) -> npt.NDArray[np.double]:
        """Local inverse transformation at each integration point.

        This array contains inverse mapping matrix, which is used
        for the contravarying components. When the dimension of the
        mapping space (as counted by :attr:`SpaceMap.output_dimensions`)
        is greater than the dimension of the reference space, this is a
        rectangular matrix, such that it maps the (rectangular) Jacobian
        to the identity matrix.
        """
        ...

    def basis_transform(self: SpaceMap, order: int) -> npt.NDArray[np.double]:
        """Compute the matrix with transformation factors for k-form basis.

        Basis transform matrix returned by this function specifies how at integration
        point a basis from the reference domain contributes to the basis in the target
        domain.

        Parameters
        ----------
        order : int
            Order of the k-form for which this is to be done, in range
            ``(0, input_dimensions]``.

        Returns
        -------
        array
            Array with three axes. The first indexes over the input basis, the second
            over output basis, and the last one over integration points.
        """
        ...

    def boundary(
        self,
        idim: int,
        end: bool = False,
        integration_space: IntegrationSpace = ...,
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    ) -> SpaceMap:
        """Extract a space map restricted to a reference-space boundary.

        Parameters
        ----------
        idim : int
            Index of the reference dimension that is fixed.

        end : bool, default: False
            Select the upper boundary at ``+1`` when true; otherwise select the lower
            boundary at ``-1``.

        integration_space : IntegrationSpace, default: the element space
            Face integration space used to sample the extracted map. When omitted,
            the volume integration space with the fixed axis removed is used.

        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry to get the element and face quadrature rules from.

        Returns
        -------
        SpaceMap
            Mapping from the remaining reference dimensions to the same physical
            coordinates. This map provides the tangential pullback and positive
            surface measure for forms on this element face.
        """
        ...

class SampledSpaceMap:
    """Mapping between reference space and target space, sampled from a SpaceMap.

    A mapping from the reference space to the target space, which maps the
    :math:`N`-dimensional reference space to an :math:`M`-dimensional
    physical space. The purpose of this mapping is to provide easier
    visualization with VTK and other tools that want sampled data.

    As such, it cannot be used for integration, only mapping k-forms to the
    target space. It can however be reused for multiple k-forms, as long as
    they are reconstructed on the same tensor grid.

    The samples need not be uniformly spaced. If the sample orders are lower
    than the orders of the actual coordinate map, the resulting sampled map
    will not be accurate. Otherwise, the accuracy is almost machine precision,
    since coordinate maps are defined with polynomial basis.

    Parameters
    ----------
    space_map : SpaceMap
        Mapping of the space in which we sample.

    samples : Sequence[Sequence[float] | array_like]
        One-dimensional sample coordinates for each reference dimension. The
        number of sample arrays must match the input dimension of the space map.
        The arrays define the tensor grid, may have different lengths, and must
        not be empty.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry to get the integration rules from.
    """

    def __new__(
        cls,
        space_map: SpaceMap,
        samples: Sequence[Sequence[float] | npt.ArrayLike],
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    ) -> Self: ...
    @classmethod
    def on_uniform_grid(
        cls,
        space_map: SpaceMap,
        orders: Sequence[int],
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    ) -> Self:
        """Create a SampledSpaceMap on a uniform grid of points in the reference space.

        Parameters
        ----------
        space_map : SpaceMap
            Mapping of the space in which we sample.

        orders : Sequence[int]
            Orders of the sampling in each dimension. The number of orders must match
            the number of input dimensions of the space map. Must not be negative.

        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Registry to get the integration rules from.

        Returns
        -------
        SampledSpaceMap
            Sampled map evaluated on the requested uniform tensor grid.
        """
        ...
    @property
    def orders(self) -> tuple[int, ...]:
        """Orders of the sampling in each dimension."""
        ...
    @property
    def input_dimensions(self) -> int:
        """Dimension of the input/reference space."""
        ...

    @property
    def output_dimensions(self) -> int:
        """Dimension of the output/physical space."""
        ...

    @property
    def determinant(self) -> npt.NDArray[np.double]:
        """Array with the values of determinant at sampled points."""
        ...

    @property
    def positions(self) -> npt.NDArray[np.double]:
        """Array with the positions of the sampled points in the physical space."""
        ...

    @property
    def inverse_map(self) -> npt.NDArray[np.double]:
        """Local inverse transformation at each sampled point.

        This array contains inverse mapping matrix, which is used
        for the contravarying components. When the dimension of the
        mapping space (as counted by :attr:`SpaceMap.output_dimensions`)
        is greater than the dimension of the reference space, this is a
        rectangular matrix, such that it maps the (rectangular) Jacobian
        to the identity matrix.
        """
        ...

def compute_kform_mass_matrix(
    smap: SpaceMap,
    order: int,
    basis_left: FunctionSpace,
    basis_right: FunctionSpace,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> npt.NDArray[np.double]:
    """Compute the k-form mass matrix.

    Parameters
    ----------
    smap : SpaceMap
        Mapping of the space in which this is to be computed.

    order : int
        Order of the k-form for which this is to be done.

    basis_left : FunctionSpace
        Function space of 0-forms used as test forms.

    basis_right : FunctionSpace
        Function space of 0-forms used as trial forms.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry to get the integration rules from.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry to get the basis from.

    Returns
    -------
    array
        Mass matrix for the inner product of two k-forms; rows span the degrees of
        freedom of ``basis_left`` and columns those of ``basis_right``.
    """
    ...

def compute_kform_incidence_matrix(
    base_space: FunctionSpace, order: int
) -> npt.NDArray[np.double]:
    """Compute the incidence matrix which maps a k-form to its (k + 1)-form derivative.

    Parameters
    ----------
    base_space : FunctionSpace
        Base function space, which describes the function space used for 0-forms.

    order : int
        Order of the k-form to get the incidence matrix for.

    Returns
    -------
    array
        Matrix, which maps degrees of freedom for the input k-form to the degrees of
        freedom of its (k + 1)-form derivative.
    """
    ...

def compute_kform_interior_product_matrix(
    smap: SpaceMap,
    order: int,
    basis_left: FunctionSpace,
    basis_right: FunctionSpace,
    vector_field_components: npt.NDArray[np.double],
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> npt.NDArray[np.double]:
    """Compute the mass matrix that is the result of interior product in an inner product.

    Parameters
    ----------
    smap : SpaceMap
        Mapping of the space in which this is to be computed.

    order : int
        Order of the k-form for which this is to be done; the left test form has
        order ``order - 1``.

    basis_left : FunctionSpace
        Function space of 0-forms used as test forms.

    basis_right : FunctionSpace
        Function space of 0-forms used as trial forms.

    vector_field_components : array
        Vector field components involved in the interior product, sampled at the
        integration points of the map: shape ``(space_map.output_dimensions,
        npts_0, ..., npts_k)`` where ``npts_i`` is the number of integration
        points along axis ``i``.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry to get the integration rules from.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry to get the basis from.

    Returns
    -------
    array
        Mass matrix mapping the degrees of freedom of the k-form built from
        ``basis_right`` to those of the (k - 1)-form built from ``basis_left``,
        pairing each test form with the interior product of the trial form and
        the vector field.
    """
    ...

def compute_kform_boundary_mass_matrices(
    element_specs: Sequence[KFormSpecs],
    orientations: Sequence[Sequence[int]],
    element_integrations: Sequence[IntegrationSpace],
    /,
    *,
    element_maps: Sequence[SpaceMap] | None = None,
    boundary_dimension: int | None = None,
    c1_continuous: bool = False,
    packed: bool = False,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> tuple[
    KFormSpecs | None,
    IntegrationSpace,
    tuple[npt.NDArray[np.double], ...],
    tuple[PackedRows, ...] | None,
]:
    """Assemble incident elements' mass matrices against one common boundary space.

    Requires two or more incident elements of one shared object. Each element
    provides one orientation record: a signed one-based permutation of the
    element axes whose first ``ndim - boundary_dimension`` entries name the fixed
    normal axes and whose tail maps the free canonical boundary axes. Rows are the
    windowed common Legendre test space of the shared object (the two highest
    functions removed on axes inactive in a component); columns are the mapped
    element trace DoFs, the element k-form DoF counts of the boundary components.
    With ``element_maps`` (one SpaceMap per element) the assembly samples each
    face's surface measure and k-form pullback on its own canonical grid;
    C1-continuous requests ignore the maps. A form order past the boundary
    dimension has no trace: ``common_specs`` is ``None`` and the matrices are
    empty.

    Parameters
    ----------
    element_specs : Sequence[KFormSpecs]
        Element k-form specification per incident element.

    orientations : Sequence[Sequence[int]]
        Signed one-based permutation of the element axes per element.

    element_integrations : Sequence[IntegrationSpace]
        Element integration space per element.

    element_maps : Sequence[SpaceMap], optional
        One volume map per element; ignored when ``c1_continuous`` is set.

    boundary_dimension : int, optional
        Boundary dimension, defaults to ``ndim - 1``. Zero is allowed for scalar
        (order zero) traces: point rows pair vertex value functionals through the
        endpoint tables.

    c1_continuous : bool, default: False
        Reference-frame pairing; ``element_maps`` is ignored.

    packed : bool, default: False
        Also return one packed row tuple per element.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry to get the quadrature rules from.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry to get the traced basis tables from.

    Returns
    -------
    tuple
        ``(common_specs, common_integration, matrices, packed)``: the merged
        common k-form specification (``None`` when the form order exceeds the
        boundary dimension), the common integration space, one dense matrix per
        element, and — with ``packed=True`` — one packed row tuple per element
        ``(row_offsets, sides, components, local_dofs, coefficients)``. ``sides``
        holds the element's index in ``element_specs``; the coefficients are the
        dense matrix entries in row-major order.
    """
    ...

def compute_kform_boundary_trace_moments(
    element_specs: Sequence[KFormSpecs],
    orientations: Sequence[Sequence[int]],
    element_integrations: Sequence[IntegrationSpace],
    /,
    *,
    element_maps: Sequence[SpaceMap] | None = None,
    boundary_dimension: int | None = None,
    c1_continuous: bool = False,
    packed: bool = False,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> tuple[
    KFormSpecs | None,
    IntegrationSpace,
    tuple[npt.NDArray[np.double], ...],
    tuple[PackedRows, ...] | None,
]:
    """Assemble trace mass rows of one or more elements against the common boundary space.

    The prescribed-data interface behind strong boundary conditions: each
    element's trace pairing with the same row and column conventions as
    :func:`compute_kform_boundary_mass_matrices`. Requires at least one element
    and returns one dense matrix and one packed tuple per element. Bind a
    right-hand side by multiplying the rows with the mapped data degrees of
    freedom.

    Parameters
    ----------
    element_specs : Sequence[KFormSpecs]
        Element k-form specification per element; at least one.

    orientations : Sequence[Sequence[int]]
        Signed one-based permutation of the element axes per element.

    element_integrations : Sequence[IntegrationSpace]
        Element integration space per element.

    element_maps : Sequence[SpaceMap], optional
        One volume map per element; ignored when ``c1_continuous`` is set.

    boundary_dimension : int, optional
        Boundary dimension, defaults to ``ndim - 1``. Zero is allowed for scalar
        (order zero) traces: point rows pair vertex value functionals through the
        endpoint tables.

    c1_continuous : bool, default: False
        Reference-frame pairing; ``element_maps`` is ignored.

    packed : bool, default: False
        Also return one packed row tuple per element.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry to get the quadrature rules from.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry to get the traced basis tables from.

    Returns
    -------
    tuple
        ``(common_specs, common_integration, matrices, packed)`` with one dense
        matrix and one packed row tuple per element; ``common_specs`` is ``None``
        when the form order exceeds the boundary dimension. ``sides`` holds the
        element's index in ``element_specs``; the coefficients are the dense
        matrix entries in row-major order.
    """
    ...

def compute_kform_boundary_load(
    test_specs: KFormSpecs,
    element_spec: KFormSpecs,
    element_map: SpaceMap,
    collections: tuple[npt.ArrayLike, ...],
    npts: int,
    element_id: int,
    boundary_id: int,
    data: Callable[..., npt.ArrayLike] | Sequence[Callable[..., npt.ArrayLike]],
    /,
    surface_measure: bool = False,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> npt.NDArray[np.double]:
    """Assemble the physical boundary load of one element face.

    Computes the metric-free chain integral of the components of a k-form
    datum (element frame, ``k = element_spec.order + 1``) against the trace of
    the element (k-1)-form basis on a codimension-1 boundary face. For each
    traced face component with element-frame axes ``J_e`` and fixed normal
    axis ``a``, the only contributing datum component is ``J_e | {a}``:

    ``b[j] = s * o * (-1)^{|{i in J_e : i < a}|} * sum_p w_p u_{J_e | {a}}(g_p) B_j(g_p)``

    where ``s`` and ``a`` are the side and index of the fixed normal axis of
    the face, ``o`` the orientation sign of the mapped component, ``w_p`` the
    reference face quadrature weights, ``u`` the sampled datum component and
    ``B_j`` the element (k-1)-form basis of the traced component. When ``k``
    equals the element dimension (a single datum component) this reduces to
    the scalar chain integral ``s * o * (-1)^a * sum_p w_p data(g_p) B_j(g_p)``:
    the natural boundary term of the mixed formulation implementing the weak
    Dirichlet condition ``u = data``.

    Parameters
    ----------
    test_specs : KFormSpecs
        Test (k-1)-form specification on the canonical boundary space.

    element_spec : KFormSpecs
        Volume k-form specification for the selected element. The datum order
        is ``element_spec.order + 1``.

    element_map : SpaceMap
        Volume map for the selected element. Its restricted face map provides the
        face geometry and quadrature.

    collections : tuple of array_like
        Boundary-ID arrays for mesh objects of dimensions 1 through N. The last
        collection contains the N-dimensional elements.

    npts : int
        Number of mesh points represented implicitly by point IDs.

    element_id : int
        Element containing the selected boundary.

    boundary_id : int
        Mesh boundary-object ID on the selected element.

    data : Callable or sequence of Callables
        Datum components in element-frame component order: one callable per
        ``k``-form component (``math.comb(element_spec.dimension, k)`` of
        them), each called with one coordinate array per element dimension and
        returning one value per face quadrature point (scalars broadcast). A
        bare callable is accepted when ``k`` equals the element dimension (a
        single component). 0-form data is not covered; impose it strongly
        instead.

        The quadrature points are the *canonical* face tensor-product nodes
        of the restricted element map's rule, in canonical-face point order
        (fixed normal axis first), mapped through the restricted element map.
        They coincide with the restricted face map's integration points only
        because the same rule and cardinality are used; consumers matching the
        ``data`` evaluations against other sample sets must match by
        position, not assume a particular index order.

    surface_measure : bool, default: False
        Integrate the data with the mapped face Jacobian (physical surface
        measure) instead of the metric-free chain integral.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry to get the face quadrature rules from.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry to get the traced basis table from.

    Returns
    -------
    numpy.ndarray
        Dense load vector over the flattened element (k-1)-form degrees of
        freedom.
    """
    ...

def incidence_kform_operator(
    specs: KFormSpecs,
    values: npt.NDArray[np.double],
    transpose: bool = False,
    right: bool = False,
    *,
    out: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Apply the incidence operator on the k-form.

    Parameters
    ----------
    specs : KFormSpecs
        Specifications of the input k-form on which this operator is to be applied on.

    values : array
        Degrees of freedom of all components of the input, flattened into one axis.
        A 1D array is a single set of DoFs; in a 2D array that axis is the first one
        when applying from the left (default) and the last one when ``right`` is set,
        the other axis repeating the operator.

    transpose : bool, default: False
        Apply the transpose of the incidence operator instead.

    right : bool, default: False
        Apply the incidence operator from the right side: the input is multiplied by
        the operator on the right. This is equivalent to applying the transposed
        operator from the left to the transposed input, then transposing the result
        back.

    out : array, optional
        Array to which the result is written. Its degree-of-freedom axis must have
        the size of the output degrees of freedom: the first axis when applying
        from the left, the last axis when ``right`` is set. A 2D input requires a
        2D output whose repetition axis matches the input's.

    Returns
    -------
    array
        Degrees of freedom of the image of the input under the incidence operator:
        by default the (k + 1)-form derivative of the input k-form; with exactly one
        of ``transpose`` or ``right`` the operator runs from the (k + 1)-form space
        to the k-form space. When ``out`` is given it is returned, otherwise a new
        array holds the result.
    """
    ...

def incidence_matrix(basis_specs: BasisSpecs) -> npt.NDArray[np.double]:
    """Return the incidence matrix to transfer derivative degrees of freedom.

    Parameters
    ----------
    basis_specs : BasisSpecs
        Basis specs for which this incidence matrix should be computed.

    Returns
    -------
    array
        Incidence matrix of shape ``(order, order + 1)`` (``order`` the basis order
        of ``basis_specs``): maps the primal degrees of freedom of that space to the
        degrees of freedom of its derivative, which lies in the space one order
        less.
    """
    ...

def incidence_operator(
    x: npt.ArrayLike, specs: BasisSpecs, axis: int = 0
) -> npt.NDArray[np.double]:
    """Apply the incidence operator to an array of degrees of freedom along an axis.

    Parameters
    ----------
    x : array_like
        Array of degrees of freedom to apply the incidence operator to. The axis
        selected by ``axis`` must have size ``specs.order + 1``.

    specs : BasisSpecs
        Specifications for basis that determine what set of polynomial is used to take
        the derivative.

    axis : int, default: 0
        Axis along which to apply the incidence operator.

    Returns
    -------
    array
        Array of degrees of freedom that is the result of applying the incidence
        operator along the specified axis; that axis shrinks from
        ``specs.order + 1`` to ``specs.order``.
    """
    ...

def packed_kform_constraints_to_csr(
    packed: tuple[
        npt.NDArray[np.uintp],
        npt.NDArray[np.uint64],
        npt.NDArray[np.uint32],
        npt.NDArray[np.uintp],
        npt.NDArray[np.double],
    ],
    specs: KFormSpecs,
    element_count: int,
    /,
) -> tuple[
    npt.NDArray[np.double],
    npt.NDArray[np.intp],
    npt.NDArray[np.uintp],
]:
    """Convert packed global k-form rows to CSR constructor arrays.

    Parameters
    ----------
    packed : tuple of (array, array, array, array, array)
        ``(row_offsets, element_ids, components, local_dofs, coefficients)``
        with dtypes ``uintp``, ``uint64``, ``uint32``, ``uintp`` and ``float64``.
        ``row_offsets`` starts at zero, is non-decreasing and ends at the entry
        count; ``element_ids`` and ``components`` stay below ``element_count``
        and the component count of ``specs``.

    specs : KFormSpecs
        Element k-form specification that numbers the columns inside each
        element.

    element_count : int
        Number of elements referenced by ``element_ids``.

    Returns
    -------
    tuple of (array, array, array)
        ``(data, indices, indptr)`` for direct use with
        ``scipy.sparse.csr_matrix``. Columns are element-major:
        ``element_id`` times the element's total DoF count, plus the
        component's start inside the element, plus the local DoF.
    """
    ...

def compute_mass_matrix(
    space_in: FunctionSpace,
    space_out: FunctionSpace,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> npt.NDArray[np.double]:
    """Compute the mass matrix between two function spaces.

    Parameters
    ----------
    space_in : FunctionSpace
        Function space for the input functions.

    space_out : FunctionSpace
        Function space for the output functions.

    integration : IntegrationSpace or SpaceMap
        Integration space used to compute the mass matrix or a space mapping.
        If the integration space is provided, the integration is done on the
        reference domain. If the mapping is defined instead, the integration
        space of the mapping is used, along with the integration being done
        on the mapped domain instead.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry used to retrieve the integration rules.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry used to retrieve the basis specifications.

    Returns
    -------
    array
        Mass matrix as a 2D array, which maps the primal degrees of freedom of the input
        function space to dual degrees of freedom of the output function space.
    """
    ...

def compute_gradient_mass_matrix(
    space_in: FunctionSpace,
    space_out: FunctionSpace,
    integration: IntegrationSpace | SpaceMap,
    /,
    idx_in: int,
    idx_out: int,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> npt.NDArray[np.double]:
    """Compute the mass matrix between two function spaces.

    The purpose of this function is to compute the matrix, which transfers
    the contribution of derivative along the reference space dimension
    to the physical space derivative.

    Parameters
    ----------
    space_in : FunctionSpace
        Function space for the input functions.

    space_out : FunctionSpace
        Function space for the output functions.

    integration : IntegrationSpace or SpaceMap
        Integration space used to compute the mass matrix or a space mapping.
        If the integration space is provided, the integration is done on the
        reference domain. If the mapping is defined instead, the integration
        space of the mapping is used, along with the integration being done
        on the mapped domain instead.

    idx_in : int
        Index of the reference-space dimension along which the input functions are
        differentiated.

    idx_out : int
        Index of the output-space dimension on which the derivative component is
        returned. Without a space map only ``idx_in == idx_out`` is non-zero.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        Registry used to retrieve the integration rules.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        Registry used to retrieve the basis specifications.

    Returns
    -------
    array
        Mass matrix as a 2D array, which maps the primal degrees of freedom of the input
        function space to dual degrees of freedom of the output function space.
    """
    ...

def transform_contravariant_to_target(
    smap: SpaceMap,
    components: npt.ArrayLike,
    *,
    out: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Transform contravariant vector components from reference to target domain.

    Since the basis of 1-forms are covectors, which are as the name implies covarying,
    the values of components are contravarying. Once transformed to the target domain,
    the 1-form can be lowered to a tangent vector field trivially.

    Parameters
    ----------
    smap : SpaceMap
        Mapping from the reference space to the physical space to use to transform the
        components.

    components : array_like
        Array whose first dimension indexes the components in the reference space and
        has length ``input_dimensions``. The remaining dimensions must match the
        integration grid of the space map (``order + 1`` nodes per reference dimension).

    out : array, optional
        Array to use to write the resulting transformed components to. If it is not
        specified, a new array is created.

    Returns
    -------
    array
        Array of transformed contravariant components. If the ``out`` parameter was given,
        a new reference to it is returned, otherwise a reference to the newly created
        output array is returned.
    """
    ...

def transform_covariant_to_target(
    smap: SpaceMap,
    components: npt.ArrayLike,
    *,
    out: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Transform covariant 1-form components from reference to target domain.

    Parameters
    ----------
    smap : SpaceMap
        Mapping from the reference space to the physical space to use to transform the
        components.

    components : array_like
        Array whose first dimension indexes the components in the reference space and
        has length ``input_dimensions``. The remaining dimensions must match the
        integration grid of the space map (``order + 1`` nodes per reference dimension).

    out : array, optional
        Array to use to write the resulting transformed components to. If it is not
        specified, a new array is created.

    Returns
    -------
    array
        Array of transformed covariant components. If the ``out`` parameter was given,
        a new reference to it is returned, otherwise a reference to the newly created
        output array is returned.
    """
    ...

def transform_kform_to_target(
    order: int,
    smap: SpaceMap,
    components: npt.ArrayLike,
    *,
    out: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Transform k-form values based on a space mapping.

    Parameters
    ----------
    order : int
        Order of the k-form being transformed.

    smap : SpaceMap
        Mapping between the reference and target domain to use.

    components : array_like
        Array with values of components of the k-form in the reference domain at
        integration points associated with the space mapping.

    out : array, optional
        Array to use to store the output in.

    Returns
    -------
    array
        Array with values of the components in the physical space.
    """
    ...

def transform_kform_to_target_sampled(
    order: int,
    smap: SampledSpaceMap,
    components: npt.ArrayLike,
    *,
    out: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Transform k-form values based on a sampled space mapping.

    0-forms do not need a coordinate transformation. This function therefore
    accepts only orders greater than zero; handle order-zero values directly.

    Parameters
    ----------
    order : int
        Order of the k-form being transformed. Must be at least 1.

    smap : SampledSpaceMap
        Mapping between the reference and target domain to use.

    components : array_like
        Array with values of components of the k-form in the reference domain at
        the sampled points associated with the space mapping.

    out : array, optional
        Array to use to store the output in.

    Returns
    -------
    array
        Array with values of the components in the physical space.
    """
    ...

def transform_kform_component_to_target(
    order: int,
    smap: SpaceMap,
    component: npt.ArrayLike,
    index: int,
    *,
    out: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Transform k-form values based on a space mapping.

    Parameters
    ----------
    order : int
        Order of the k-form being transformed.

    smap : SpaceMap
        Mapping between the reference and target domain to use.

    component : array_like
        Values of the component in the reference domain at the integration points
        of the space map. Leading dimensions are batch dimensions; the trailing
        dimensions must match the integration grid and the batch dimensions are
        preserved in the output.

    index : int
        Index of the component that is to be computed.

    out : array, optional
        Array to use to store the output in.

    Returns
    -------
    array
        Array with values of the components in the physical space.
    """
    ...
