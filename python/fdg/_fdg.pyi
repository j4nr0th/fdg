from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Self, final

import numpy as np
import numpy.typing as npt

from fdg.enum_type import _BasisTypeHint, _IntegrationMethodHint

@final
class IntegrationRegistry:
    """Registry for integration rules.

    This registry contains all available integration rules and caches them for
    efficient retrieval.
    """

    def __new__(cls) -> Self: ...
    def usage(self) -> tuple[IntegrationSpecs, ...]: ...
    def clear(self) -> None: ...

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

    def __new__(
        cls,
        order: int,
        /,
        method: _IntegrationMethodHint = "gauss",
    ) -> Self: ...
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
        self, registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY
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
        self, registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY
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
    """Registry for basis specifications.

    This registry contains all available basis sets and caches them for efficient
    retrieval.
    """

    def __new__(cls) -> Self: ...
    def usage(self) -> tuple[tuple[BasisSpecs, IntegrationSpecs], ...]:
        """Return the basis-integration pairs that are held by the registry.

        Returns
        -------
        tuple of (BasisSpecs, IntegrationSpecs)
            Tuple of basis-integration specifications pair for each of basis set
            held in the registry.
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

    def __new__(self, n: int, /, *idx: int): ...
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
        """Normalize the basis by splitting the sign."""
        ...

@final
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
    def basis_type(self) -> _BasisTypeHint:
        """Type of the basis used for the set."""
        ...

    @property
    def order(self) -> int:
        """Order of the basis in the set."""
        ...

    def values(self, x: npt.ArrayLike, /) -> npt.NDArray[np.double]:
        """Evaluate basis functions at given locations.

        Parameters
        ----------
        x : array_like
            Locations where the basis functions should be evaluated.

        Returns
        -------
        array
            Array of basis function values at the specified locations.
            It has one more dimension than ``x``, with the last dimension
            corresponding to the basis function index.
        """
        ...

    def derivatives(self, x: npt.ArrayLike, /) -> npt.NDArray[np.double]:
        """Evaluate basis function derivatives at given locations.

        Parameters
        ----------
        x : array_like
            Locations where the basis function derivatives should be evaluated.

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
            will be created and returned. It should have the same shape as ``x``,
            but with an extra dimension added, the length of which is the total
            number of basis functions in the function space.

        Returns
        -------
        array
            Array of basis function values at the specified locations.
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

        transpose : bool, defaul: False
            Order the array so that axes indexing the integration points come before
            the ones indexing the bases.

        integration_registry : IntegrationRegistry, defaul: DEFAULT_INTEGRATION_REGISTRY
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
        idim: Sequence[int],
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
        idim : Sequence[int]
            Dimensions in which the derivative should be computed. All values
            should appear at most once.
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
        *,
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
            Registry used to retrieve the integration rules.

        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Registry used to retrieve the basis specifications.

        Returns
        -------
        DegreesOfFreedom
            Degrees of freedom using Lagrange basis of specified orders.
        """
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
        of freedom to get the DoFs corresponding to a praticular component.

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

    # TODO: test this method!
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

    Parameters are given through the ``from_corners`` and ``from_collections``
    class methods; the type itself cannot be instantiated directly.
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
        """Boundary-ID arrays of the mesh objects of every dimension (copies)."""
        ...

    def element_object(self, element_id: int, axis: Sequence[int], /) -> int:
        """Look up the global ID of the object at a position within one element.

        Parameters
        ----------
        element_id : int
            ID of the element.

        axis : sequence of int
            Axis specification of length ``ndim``; entry ``i`` is 0 for a free
            axis, or ``i + 1`` / ``-(i + 1)`` to fix the axis at its end / start
            side.

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
            Dimension of the objects.

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
            Dimension of the objects.

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

    def compute_kform_boundary_constraints(
        self,
        test_specs: KFormSpecs,
        element_spec: KFormSpecs,
        element_map: SpaceMap,
        element_id: int,
        boundary_id: int,
        /,
    ) -> tuple[
        npt.NDArray[np.uintp],
        npt.NDArray[np.uint32],
        npt.NDArray[np.uintp],
        npt.NDArray[np.double],
    ]:
        """Assemble physical k-form boundary constraints for one boundary object.

        Identical to the free function :func:`compute_kform_boundary_constraints`,
        but the mesh collections and point count are taken from the mesh itself.

        Parameters
        ----------
        test_specs : KFormSpecs
            Test k-form specification on the canonical boundary space.

        element_spec : KFormSpecs
            Volume k-form specification for the selected element.

        element_map : SpaceMap
            Volume map for the selected element. Its restricted face map provides
            the k-form pullbacks and physical measure.

        element_id : int
            Element containing the selected boundary.

        boundary_id : int
            Mesh boundary-object ID on the selected element.

        Returns
        -------
        tuple of arrays
            Row offsets, element component indices, local DoF indices, and
            coefficients for the packed constraint rows.
        """
        ...

@final
class CoordinateMap:
    """Mapping between reference and physical coordinates.

    This is type is a glorified wrapper around
    :meth:`DegreesOfFreedom.reconstruct_at_integration_points()`
    that represents a coordinate mapping for one dimension. In N-dimensional space,
    N such maps are used to represent the full mapping.

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
        /,
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
        """Retrieve the gradient of the coordinate map in given dimension."""
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
            Index of the dimension for which the map shoudl be returned.

        Returns
        -------
        CoordinateMap
            Map used for the specified coordinate.
        """
        ...

    @property
    def integration_space(self) -> IntegrationSpace:
        """Integration space used by the map."""
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
        mapping space (as counted by :meth:`SpaceMap.output_dimensions`)
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
            Order of the k-form for which this is to be done.

        Returns
        -------
        array
            Array with three axis. The first indexes over the input basis, the second
            over output basis, and the last one over integration points.
        """
        ...

    def boundary(
        self,
        idim: int,
        end: bool = False,
        integration_space: IntegrationSpace | None = None,
        /,
    ) -> SpaceMap:
        """Extract a space map restricted to a reference-space boundary.

        Parameters
        ----------
        idim : int
            Index of the reference dimension that is fixed.

        end : bool, default: False
            Select the upper boundary at ``+1`` when true; otherwise select the lower
            boundary at ``-1``.

        integration_space : IntegrationSpace, optional
            Face integration space used to sample the extracted map. When omitted,
            the volume integration space with the fixed axis removed is used.

        Returns
        -------
        SpaceMap
            Mapping from the remaining reference dimensions to the same physical
            coordinates. This map provides the tangential pullback and positive
            surface measure for forms on this element face.
        """
        ...

def _scale_array_boundary(arr: npt.ArrayLike, /) -> npt.NDArray[np.double]:
    """Scale the array based on how many N-dimensional boundaries an entry appears.

    Parameters
    ----------
    arr : array_like
        Array to scale.

    Returns
    -------
    array
        Scaled array.
    """
    ...

def compute_kform_mass_matrix(
    smap: SpaceMap,
    order: int,
    left_bases: FunctionSpace,
    right_bases: FunctionSpace,
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

    left_bases : FunctionSpace
        Function space of 0-forms used as test forms.

    right_bases : FunctionSpace
        Function space of 0-forms used as trial forms.

    integration_registry : IntegrationRegistry, optional
        Registry to get the integration rules from.

    basis_registry : BasisRegistry, optional
        Registry to get the basis from.

    Returns
    -------
    array
        Mass matrix for inner product of two k-forms.
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
    left_bases: FunctionSpace,
    right_bases: FunctionSpace,
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
        Order of the k-form for which this is to be done.

    left_bases : FunctionSpace
        Function space of 0-forms used as test forms.

    right_bases : FunctionSpace
        Function space of 0-forms used as trial forms.

    vector_field_components : array
        Vector field components involved in the interior product.

    int_registry : IntegrationRegistry, optional
        Registry to get the integration rules from.

    basis_registry : BasisRegistry, optional
        Registry to get the basis from.

    Returns
    -------
    array
        Mass matrix for inner product of two k-forms, where the right one has the interior
        product with the vector field applied to it.
    """
    ...

def compute_kform_boundary_constraints(
    test_specs: KFormSpecs,
    element_spec: KFormSpecs,
    element_map: SpaceMap,
    collections: tuple[npt.ArrayLike, ...],
    npts: int,
    element_id: int,
    boundary_id: int,
) -> tuple[
    npt.NDArray[np.uintp],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uintp],
    npt.NDArray[np.double],
]:
    """Assemble physical k-form boundary constraints.

    Parameters
    ----------
    test_specs : KFormSpecs
        Test k-form specification on the canonical boundary space.

    element_spec : KFormSpecs
        Volume k-form specification for the selected element.

    element_map : SpaceMap
        Volume map for the selected element. Its restricted face map provides the
        k-form pullbacks and physical measure.

    collections : tuple of array_like
        Boundary-ID arrays for mesh objects of dimensions 1 through N. The last
        collection contains the N-dimensional elements.

    npts : int
        Number of mesh points represented implicitly by point IDs.

    element_id : int
        Element containing the selected boundary.

    boundary_id : int
        Mesh boundary-object ID on the selected element.

    Returns
    -------
    tuple of arrays
        Row offsets, element component indices, local DoF indices, and coefficients
        for the packed constraint rows.
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
    the scalar chain integral ``s * (-1)^a * sum_p w_p data(g_p) B_j(g_p)``:
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
        bare callable is accepted when ``k`` equals the element dimension
        (a single component). 0-form data is not covered; impose it strongly
        instead.

        The quadrature points are the *canonical* face tensor-product nodes
        of the restricted element map's rule, in canonical-face point order
        (fixed normal axis first), mapped through the restricted element map.
        They coincide with the restricted face map's integration points only
        because the same rule and cardinality are used; consumers matching
        the ``data`` evaluations against other sample sets must match by
        position, not assume a particular index order.

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
        Array which contains the degrees of freedom of all components flattened along the
        last axis. Treated as a row-major matrix or a vector, depending if 1D or 2D.

    transpose : bool, default: False
        Apply the transpose of the incidence operator instead.

    right : bool, default: False
        Apply the incidence operator from the right side. This is equivalent to the
        transpose of the operator to the left to the transpose of the input, then
        transposing the result back.

    out : array, optional
        Array to which the result is written to. The first axis must have the same size
        as the number of output degrees of freedom of the resulting k-form. If the input
        was 2D, this must be as well, with the last axis matching the input's last axis.

    Returns
    -------
    array
        Values of the degrees of freedom of the derivative of the input k-form. When an
        output array is specified through the parameters, another reference to it is
        returned, otherwise a new array is created to hold the result and returned.
    """
    ...

def incidence_matrix(specs: BasisSpecs) -> npt.NDArray[np.double]:
    """Return the incidence matrix to transfer derivative degrees of freedom.

    Parameters
    ----------
    specs : BasisSpecs
        Basis specs for which this incidence matrix should be computed.

    Returns
    -------
    array
        One dimensional incidence matrix. It transfers primal degrees of freedom
        for a derivative to a function space one order less than the original.
    """
    ...

def incidence_operator(
    val: npt.ArrayLike, /, specs: BasisSpecs, axis: int = 0
) -> npt.NDArray[np.double]:
    """Apply the incidence operator to an array of degrees of freedom along an axis.

    Parameters
    ----------
    val : array_like
        Array of degrees of freedom to apply the incidence operator to.

    specs : BasisSpecs
        Specifications for basis that determine what set of polynomial is used to take
        the derivative.

    axis : int, default: 0
        Axis along which to apply the incidence operator along.

    Returns
    -------
    array
        Array of degrees of freedom that is the result of applying the incidence operator,
        along the specified axis.
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
        Mass matrix as a 2D array, which maps the primal degress of freedom of the input
        function space to dual degrees of freedom of the output function space.
    """
    ...

def compute_gradient_mass_matrix(
    space_in: FunctionSpace,
    space_out: FunctionSpace,
    integration: IntegrationSpace | SpaceMap,
    /,
    idim_in: int,
    idim_out: int,
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

    idim_im : int
        Index of the dimension to take the derivative of the input space on.

    idim_out : int
        Index of the output space on which the component of the derivative should
        be returned on.

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
        Array where the first dimension indexes the components in the reference space. All
        other dimensions will be treated as if flattened.

    out : array, optional
        Array to used to write the resulting transformed components to. If it is not
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
        Array where the first dimension indexes the components in the reference space. All
        other dimensions will be treated as if flattened.

    out : array, optional
        Array to used to write the resulting transformed components to. If it is not
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
        Values of component in the reference domain at integration points associated
        with the space mapping.

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
