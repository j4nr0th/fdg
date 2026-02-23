from __future__ import annotations

from collections.abc import Sequence
from typing import Self, final

import numpy as np
import numpy.typing as npt

from interplib.enum_type import _BasisTypeHint, _IntegrationMethodHint

def lagrange1d(
    roots: npt.ArrayLike, x: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
) -> npt.NDArray[np.double]:
    r"""Evaluate Lagrange polynomials.

    This function efficiently evaluates Lagrange basis polynomials, defined by

    .. math::

       \mathcal{L}^n_i (x) = \prod\limits_{j=1, j \neq i}^{n} \frac{x - x_j}{x_i - x_j},

    where the ``roots`` specifies the zeros of the Polynomials
    :math:`\{x_1, \dots, x_n\}`.

    Parameters
    ----------
    roots : array_like
       Roots of Lagrange polynomials.
    x : array_like
       Points where the polynomials should be evaluated.
    out : array, optional
       Array where the results should be written to. If not given, a new one will be
       created and returned. It should have the same shape as ``x``, but with an extra
       dimension added, the length of which is ``len(roots)``.

    Returns
    -------
    array
       Array of Lagrange polynomial values at positions specified by ``x``.

    Examples
    --------
    This example here shows the most basic use of the function to evaluate Lagrange
    polynomials. First, let us define the roots.

    .. jupyter-execute::

        >>> import numpy as np
        >>>
        >>> order = 7
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))

    Next, we can evaluate the polynomials at positions. Here the interval between the
    roots is chosen.

    .. jupyter-execute::

        >>> from interplib import lagrange1d
        >>>
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)
        >>> yvals = lagrange1d(roots, xpos)

    Note that if we were to give an output array to write to, it would also be the
    return value of the function (as in no copy is made).

    .. jupyter-execute::

        >>> yvals is lagrange1d(roots, xpos, yvals)
        True

    Now we can plot these polynomials.

    .. jupyter-execute::

        >>> from matplotlib import pyplot as plt
        >>>
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"$\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()

    Accuracy is retained even at very high polynomial order. The following
    snippet shows that even at absurdly high order of 51, the results still
    have high accuracy and don't suffer from rounding errors. It also performs
    well (in this case, the 52 polynomials are each evaluated at 1025 points).

    .. jupyter-execute::

        >>> from time import perf_counter
        >>> order = 51
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)
        >>> t0 = perf_counter()
        >>> yvals = lagrange1d(roots, xpos)
        >>> t1 = perf_counter()
        >>> print(f"Calculations took {t1 - t0: e} seconds.")
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"$\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> # plt.legend() # No, this is too long
        >>> plt.grid()
        >>> plt.show()
    """
    ...

def dlagrange1d(
    roots: npt.ArrayLike, x: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
) -> npt.NDArray[np.double]:
    r"""Evaluate derivatives of Lagrange polynomials.

    This function efficiently evaluates Lagrange basis polynomials derivatives, defined by

    .. math::

       \frac{d \mathcal{L}^n_i (x)}{d x} =
       \sum\limits_{j=0,j \neq i}^n \prod\limits_{k=0, k \neq i, k \neq j}^{n}
       \frac{1}{x_i - x_j} \cdot \frac{x - x_k}{x_i - x_k},

    where the ``roots`` specifies the zeros of the Polynomials
    :math:`\{x_0, \dots, x_n\}`.

    Parameters
    ----------
    roots : array_like
       Roots of Lagrange polynomials.
    x : array_like
       Points where the derivatives of polynomials should be evaluated.
    out : array, optional
       Array where the results should be written to. If not given, a new one will be
       created and returned. It should have the same shape as ``x``, but with an extra
       dimension added, the length of which is ``len(roots)``.

    Returns
    -------
    array
       Array of Lagrange polynomial derivatives at positions specified by ``x``.

    Examples
    --------
    This example here shows the most basic use of the function to evaluate derivatives of
    Lagrange polynomials. First, let us define the roots.

    .. jupyter-execute::

        >>> import numpy as np
        >>>
        >>> order = 7
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))

    Next, we can evaluate the polynomials at positions. Here the interval between the
    roots is chosen.

    .. jupyter-execute::

        >>> from interplib import dlagrange1d
        >>>
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)
        >>> yvals = dlagrange1d(roots, xpos)

    Note that if we were to give an output array to write to, it would also be the
    return value of the function (as in no copy is made).

    .. jupyter-execute::

        >>> yvals is dlagrange1d(roots, xpos, yvals)
        True

    Now we can plot these polynomials.

    .. jupyter-execute::

        >>> from matplotlib import pyplot as plt
        >>>
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"${{\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\prime$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()

    Accuracy is retained even at very high polynomial order. The following
    snippet shows that even at absurdly high order of 51, the results still
    have high accuracy and don't suffer from rounding errors. It also performs
    well (in this case, the 52 polynomials are each evaluated at 1025 points).

    .. jupyter-execute::

        >>> from time import perf_counter
        >>> order = 51
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)
        >>> t0 = perf_counter()
        >>> yvals = dlagrange1d(roots, xpos)
        >>> t1 = perf_counter()
        >>> print(f"Calculations took {t1 - t0: e} seconds.")
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"${{\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\prime$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> # plt.legend() # No, this is too long
        >>> plt.grid()
        >>> plt.show()
    """
    ...

def d2lagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def bernstein1d(n: int, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute Bernstein polynomials of given order at given locations.

    Parameters
    ----------
    n : int
       Order of polynomials used.
    x : (M,) array_like
       Flat array of locations where the values should be interpolated.

    Returns
    -------
    (M, n) arr
       Matrix containing values of Bernstein polynomial :math:`B^M_j(x_i)` as the
       element ``array[i, j]``.
    """
    ...

def bernstein_coefficients(x: npt.ArrayLike, /) -> npt.NDArray[np.double]:
    """Compute Bernstein polynomial coefficients from a power series polynomial.

    Parameters
    ----------
    x : array_like
       Coefficients of the polynomial from 0-th to the highest order.

    Returns
    -------
    array
       Array of coefficients of Bernstein polynomial series.
    """
    ...

def compute_gll(
    order: int, max_iter: int = 10, tol: float = 1e-15
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Gauss-Legendre-Lobatto integration nodes and weights.

    If you are often re-using these, consider caching them.

    Parameters
    ----------
    order : int
       Order of the scheme. The number of node-weight pairs is one more.
    max_iter : int, default: 10
       Maximum number of iterations used to further refine the values.
    tol : float, default: 1e-15
       Tolerance for stopping the refinement of the nodes.

    Returns
    -------
    array
       Array of ``order + 1`` integration nodes on the interval :math:`[-1, +1]`.
    array
       Array of integration weights which correspond to the nodes.
    """
    ...
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

    method : interplib.IntegrationMethod, default: "gauss"
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
        registry : interplib.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
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
        registry : interplib.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
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
    def usage(self) -> tuple[tuple[BasisSpecs, IntegrationSpecs], ...]: ...
    def clear(self) -> None: ...

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
    basis_type : interplib._typing.BasisType
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
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> npt.NDArray[np.double]:
        """Return values of basis at integration points.

        Parameters
        ----------
        integration : IntegrationSpace
            Integration space, the nodes of which are used to evaluate basis at.

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

        registry : interplib.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
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

        registry : interplib.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
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
        """Values of the degrees of freedom."""
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

@final
class KFormSpecs:
    """Differential k-form specification."""

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
        """Get the function space for a component."""
        ...

    def get_component_basis(self, idx: int) -> CovectorBasis:
        """Get covector basis bundle for a component."""
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
    """Degrees of freedom of a k-form."""

    def __new__(cls, specs: KFormSpecs) -> Self: ...
    @property
    def specs(self) -> KFormSpecs:
        """KFormSpecs : Specifications of the k-form."""
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
        """Values of the coordinate map at the integration points."""
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

def compute_kform_mass_matrix(
    smap: SpaceMap,
    order: int,
    left_bases: FunctionSpace,
    right_bases: FunctionSpace,
    *,
    int_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
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

    int_registry : IntegrationRegistry, optional
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
        Array of transformed covariant components. If the ``out`` parameter was given,
        a new reference to it is returned, otherwise a reference to the newly created
        output array is returned.
    """
    ...
@final
class GeoID:
    """Type used to identify a geometrical object with an index and orientation.

    Parameters
    ----------
    index : int
        Index of the geometrical object.
    reversed : any, default: False
        The object's orientation should be reversed.
    """

    def __new__(cls, index: int, reverse: object = False) -> Self: ...
    @property
    def index(self) -> int:
        """Index of the object referenced by id."""
        ...
    @property
    def reversed(self) -> bool:
        """Is the orientation of the object reversed."""
        ...

    def __bool__(self) -> bool: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __neg__(self) -> GeoID: ...

@final
class Line:
    """Geometrical object, which connects two points.

    Parameters
    ----------
    begin : GeoID or int
        ID of the point where the line beings.
    end : GeoID or int
        ID of the point where the line ends.
    """

    def __new__(cls, begin: GeoID | int, end: GeoID | int) -> Self: ...
    @property
    def begin(self) -> GeoID:
        """ID of the point where the line beings."""
        ...
    @property
    def end(self) -> GeoID:
        """ID of the point where the line ends."""
        ...

    def __array__(self, dtype=None, copy=None) -> npt.NDArray: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Surface:
    """Two dimensional geometrical object, which is bound by lines.

    Parameters
    ----------
    *ids : GeoID or int
        Ids of the lines which are the boundary of the surface.
    """

    def __new__(cls, *ids: GeoID | int) -> Self: ...
    def __array__(self, dtype=None, copy=None) -> npt.NDArray: ...
    def __getitem__(self, idx: int) -> GeoID: ...
    def __len__(self) -> int: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Manifold:
    """A manifold of a finite number of dimensions."""

    @property
    def dimension(self) -> int:
        """Dimension of the manifold."""
        ...

@final
class Manifold1D(Manifold):
    """One dimensional manifold."""

    @property
    def n_lines(self) -> int:
        """Number of lines in the manifold."""
        ...

    @property
    def n_points(self) -> int:
        """Number of points in the manifold."""
        ...

    def get_line(self, index: GeoID | int, /) -> Line:
        """Get the line of the specified ID."""
        ...

    def find_line(self, line: Line) -> GeoID:
        """Find the ID of the specified line."""
        ...

    @classmethod
    def line_mesh(cls, segments: int, /) -> Manifold1D:
        """Create a new Manifold1D which represents a line.

        Parameters
        ----------
        segments : int
            Number of segments the line is split into. There will be one more point.

        Returns
        -------
        Manifold1D
            Manifold that represents the topology of the line.
        """
        ...

    def compute_dual(self) -> Manifold1D:
        """Compute the dual to the manifold.

        Returns
        -------
        Manifold1D
            The dual to the manifold.
        """
        ...

class Manifold2D(Manifold):
    """A manifold of a finite number of dimensions."""

    @property
    def dimension(self) -> int:
        """Dimension of the manifold."""
        ...

    @property
    def n_points(self) -> int:
        """Number of points in the mesh."""
        ...

    @property
    def n_lines(self) -> int:
        """Number of lines in the mesh."""
        ...

    @property
    def n_surfaces(self) -> int:
        """Number of surfaces in the mesh."""
        ...

    def get_line(self, index: int | GeoID, /) -> Line:
        """Get the line from the mesh.

        Parameters
        ----------
        index : int or GeoID
           Id of the line to get in 1-based indexing or GeoID. If negative, the
           orientation will be reversed.

        Returns
        -------
        Line
           Line object corresponding to the ID.
        """
        ...

    def get_surface(self, index: int | GeoID, /) -> Surface:
        """Get the surface from the mesh.

        Parameters
        ----------
        index : int or GeoID
           Id of the surface to get in 1-based indexing or GeoID. If negative,
           the orientation will be reversed.

        Returns
        -------
        Surface
           Surface object corresponding to the ID.
        """

    @classmethod
    def from_irregular(
        cls,
        n_points: int,
        line_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
        surface_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
    ) -> Self:
        """Create Manifold2D from surfaces with non-constant number of lines.

        Parameters
        ----------
        n_points : int
            Number of points in the mesh.
        line_connectivity : (N, 2) array_like
            Connectivity of points which form lines in 0-based indexing.
        surface_connectivity : Sequence of array_like
            Sequence of arrays specifying connectivity of mesh surfaces in 1-based
            indexing, where a negative value means that the line's orientation is
            reversed.

        Returns
        -------
        Self
            Two dimensional manifold.
        """
        ...

    @classmethod
    def from_regular(
        cls,
        n_points: int,
        line_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
        surface_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
    ) -> Self:
        """Create Manifold2D from surfaces with constant number of lines.

        Parameters
        ----------
        n_points : int
            Number of points in the mesh.
        line_connectivity : (N, 2) array_like
            Connectivity of points which form lines in 0-based indexing.
        surface_connectivity : array_like
            Two dimensional array-like object specifying connectivity of mesh
            surfaces in 1-based indexing, where a negative value means that
            the line's orientation is reversed.

        Returns
        -------
        Self
            Two dimensional manifold.
        """
        ...

    def compute_dual(self) -> Manifold2D:
        """Compute the dual to the manifold.

        A dual of each k-dimensional object in an n-dimensional space is a
        (n-k)-dimensional object. This means that duals of surfaces are points,
        duals of lines are also lines, and that the duals of points are surfaces.

        A dual line connects the dual points which correspond to surfaces which
        the line was a part of. Since the change over a line is computed by
        subtracting the value at the beginning from that at the end, the dual point
        which corresponds to the primal surface where the primal line has a
        positive orientation is the end point of the dual line and conversely the end
        dual point is the one corresponding to the surface which contained the primal
        line in the negative orientation. Since lines may only be contained in a
        single primal surface, they may have an invalid ID as either their beginning or
        their end. This can be used to determine if the line is actually a boundary of
        the manifold.

        A dual surface corresponds to a point and contains dual lines which correspond
        to primal lines, which contained the primal point of which the dual surface is
        the result of. The orientation of dual lines in this dual surface is positive if
        the primal line of which they are duals originated in the primal point in question
        and negative if it was their end point.

        Returns
        -------
        Manifold2D
            Dual manifold.
        """
        ...

    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
