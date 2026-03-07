"""Types to simplify specifying domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from fdg._fdg import (
    DEFAULT_BASIS_REGISTRY,
    DEFAULT_INTEGRATION_REGISTRY,
    BasisRegistry,
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationRegistry,
    IntegrationSpace,
    IntegrationSpecs,
    SpaceMap,
)
from fdg.degrees_of_freedom import reconstruct
from fdg.enum_type import BasisType, IntegrationMethod
from fdg.integration import Integrable, integrate_callable


def dofs_from_boundary_pairs(
    *boundaries: tuple[DegreesOfFreedom, DegreesOfFreedom],
) -> DegreesOfFreedom:
    """Create new DoFs from its boundaries via multilinear interpolation."""
    # First check the inputs make sense
    ndim_in = len(boundaries)
    max_orders = np.zeros(ndim_in, np.uintc)

    for i, (b1, b2) in enumerate(boundaries):
        if type(b1) is not DegreesOfFreedom or type(b2) is not DegreesOfFreedom:
            raise TypeError("Both boundaries must be DegreesOfFreedom.")

        if b1.function_space.dimension != b2.function_space.dimension:
            raise ValueError(
                f"One or both boundaries for dimension {i} do not have matching "
                "number of input dimensions."
            )
        elif b1.function_space.dimension + 1 != ndim_in:
            raise ValueError(
                f"Number of physical dimensions for boundary {i} does not"
                " match the number specified expected based on the boundary count."
            )

        for idim in range(0, i):
            max_orders[idim] = max(
                max_orders[idim],
                b1.function_space.orders[idim],
                b2.function_space.orders[idim],
            )

        for idim in range(i + 1, ndim_in):
            max_orders[idim] = max(
                max_orders[idim],
                b1.function_space.orders[idim - 1],
                b2.function_space.orders[idim - 1],
            )

    # Create new boundary pairs that are have correct orders based on the new
    # function space, which can exactly represent all boundaries.
    function_space = FunctionSpace(
        *(
            BasisSpecs(BasisType.LAGRNAGE_GAUSS_LOBATTO, int(order))
            for order in max_orders
        )
    )
    output_dofs = DegreesOfFreedom(function_space)
    out_vals = output_dofs.values
    scale = np.zeros(out_vals.shape)
    for idim, (b1, b2) in enumerate(boundaries):
        r1 = b1.lagrange_projection((*max_orders[:idim], *max_orders[idim + 1 :]))
        r2 = b2.lagrange_projection((*max_orders[:idim], *max_orders[idim + 1 :]))
        # Reverse the orientation of the second boundary
        for i in range(ndim_in - 1):
            r2 = r2.reverse_orientation(i)
        bv1 = np.expand_dims(r1.values, axis=idim)
        bv2 = np.expand_dims(r2.values, axis=idim)
        nodes = IntegrationSpecs(
            function_space.orders[idim], IntegrationMethod.GAUSS_LOBATTO
        ).nodes()
        bf1 = np.expand_dims(
            (1 - nodes) / 2, axis=(*range(0, idim), *range(idim + 1, ndim_in))
        )
        bf2 = np.expand_dims(
            (1 + nodes) / 2, axis=(*range(0, idim), *range(idim + 1, ndim_in))
        )
        out_vals += bv1 * bf1 + bv2 * bf2
        scale += bf1 + bf2

    # Deal correct scaling
    scaled_vals = out_vals / scale
    output_dofs.values = scaled_vals
    return output_dofs


def _array_axis_slice(a: npt.NDArray, idx: int, axis: int):
    """Take a slice from a numpy array along the specified axis."""
    slices: list[slice | int] = [slice(None)] * a.ndim
    slices[axis] = slice(idx, idx + 1) if idx >= 0 else slice(idx, idx - 1)
    return a[tuple(slices)]


@dataclass(frozen=True)
class HypercubeDomain:
    """Base type for all domains.

    Parameters
    ----------
    *dofs : DegreesOfFreedom
        Degrees of freedom for each of the output coordinates.
    """

    dofs: tuple[DegreesOfFreedom, ...]

    def __init__(self, *dofs: DegreesOfFreedom) -> None:
        if not len(dofs):
            raise ValueError("At least one coordinate must have its DoFs specified.")

        ndim_in = 0
        for i, d in enumerate(dofs):
            if type(dofs) is None:
                raise TypeError(
                    f"Argument {i} was not {DegreesOfFreedom}, but {type(dofs)}."
                )
            if ndim_in == 0:
                ndim_in = d.function_space.dimension
            elif d.function_space.dimension != ndim_in:
                raise ValueError(
                    f"Function spaces of the DoFs {i} does not have the same input "
                    "dimension as the rest!"
                )

        object.__setattr__(self, "dofs", dofs)

    @property
    def ndim_physical(self) -> int:
        """Number of physical dimensions of the domain."""
        return len(self.dofs)

    @property
    def ndim_reference(self) -> int:
        """Number of reference dimensions of the domain."""
        if not self.dofs:
            return 0
        return self.dofs[0].function_space.dimension

    def __call__(
        self,
        space: IntegrationSpace,
        /,
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> SpaceMap:
        """Create a space map based on the integration space.

        Parameters
        ----------
        space : IntegratinoSpace
            Integration space to base the space map on.

        integration_registry : IntegrationRegistry, optional
            Integration registry to use for retrieving the integration nodes and weights.

        basis_registry : BasisRegistry
            Basis registry to use for retrieving basis values from.

        Returns
        -------
        SpaceMap
            Space mapping of the domain for the specified integration space.
        """
        return SpaceMap(
            *(
                CoordinateMap(dof, space, integration_registry, basis_registry)
                for dof in self.dofs
            )
        )

    @property
    def endpoints(self) -> tuple[npt.NDArray[np.double], ...]:
        """Return the end points of the domain."""
        int_space = IntegrationSpace(
            *(
                IntegrationSpecs(1, "gauss-lobatto")
                for _idim in range(self.ndim_reference)
            )
        )
        return tuple(
            dof.reconstruct_at_integration_points(int_space) for dof in self.dofs
        )

    def boundary(self, idim: int, end: bool = False) -> HypercubeDomain:
        """Extract a boundary."""
        dofs = [dof.plane_projection(idim, +1.0 if end else -1.0) for dof in self.dofs]
        return HypercubeDomain(*dofs)

    def compute_size(
        self,
        int_space: IntegrationSpace | None = None,
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> float:
        """Compute the size of the domain.

        Parameters
        ----------
        int_space : IntegrationSpace, optional
            Integration space to use for computing the size of the domain. If it is not
            given, a new one will be created, such that the size is computed exactly.

        integration_registry : IntegrationRegistry, optional
            Integration registry to use for retrieving the integration nodes and weights.

        basis_registry : BasisRegistry
            Basis registry to use for retrieving basis values from.

        Returns
        -------
        float
            Size of the domain. For a 1D domain this is the lenght, for a 2D domain this
            is the surface area, for a 3D domain it is the volume, and so on.
        """
        if int_space is None:
            int_space = IntegrationSpace(
                *(
                    IntegrationSpecs((order + 1) // 2)
                    for order in self.dofs[0].function_space.orders
                )
            )
        smap = self(
            int_space,
            integration_registry=integration_registry,
            basis_registry=basis_registry,
        )
        return float(np.sum(int_space.weights(integration_registry) * smap.determinant))

    def sample(self, *x: npt.NDArray[np.double]) -> tuple[npt.NDArray[np.double], ...]:
        """Sample coordinates in the physical domain.

        Parameters
        ----------
        *x : array
            Arrays of coordinate positions to evaluate the points in domain at.

        Returns
        -------
        tuple of array
            Arrays with the shape of ``x``, containing values of coordinates
            at the specified points.
        """
        return tuple(reconstruct(dof, *x) for dof in self.dofs)

    @property
    def function_space(self) -> FunctionSpace:
        """Function space used by all the DoFs."""
        return self.dofs[0].function_space

    def integrate(
        self,
        fn: Integrable,
        int_space: IntegrationSpace,
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> float:
        """Integrates the callable.

        Parameters
        ----------
        fn : Integrable
            Callable to integrate.

        int_space : IntegrationSpace
            Integration space to use for integration.

        integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
            Integration registry to use for retrieving integration rules.

        basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
            Basis registry to use for retrieving basis values.

        Returns
        -------
        float
            Result of integrating the callable on the domain.
        """
        return integrate_callable(
            fn,
            self(
                int_space,
                integration_registry=integration_registry,
                basis_registry=basis_registry,
            ),
            registry=integration_registry,
        )

    def subregion(self, *ranges: tuple[float, float]) -> HypercubeDomain:
        """Split self into a sub-region of the domain.

        Parameters
        ----------
        *ranges : (float, float)
            Range of the domain to include for each dimension.

        Returns
        -------
        HypercubeDomain
            Subregion of the domain, where the boundaries are determined from where the
            ``ranges`` parameters constrain the original domain.
        """
        n_dim_ref = self.ndim_reference
        if len(ranges) < n_dim_ref:
            raise ValueError(f"At most {n_dim_ref} pairs of divisions can be specified.")
        limits: list[tuple[float, float]] = [(float(vl), float(vh)) for vl, vh in ranges]
        while len(limits) < n_dim_ref:
            limits.append((-1.0, +1.0))

        shape = self.dofs[0].shape

        grid = np.meshgrid(
            *(np.linspace(vl, vh, n) for n, (vl, vh) in zip(shape, limits, strict=True)),
            indexing="ij",
        )
        new_fs = FunctionSpace(
            *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, s - 1) for s in shape)
        )
        new_dofs: list[DegreesOfFreedom] = list()
        for new_vals in self.sample(*grid):
            new_dofs.append(DegreesOfFreedom(new_fs, new_vals))

        return HypercubeDomain(*new_dofs)

    @staticmethod
    def from_boundary_pairs(
        *boundaries: tuple[HypercubeDomain, HypercubeDomain],
    ) -> HypercubeDomain:
        """Create a new domain from its boundaries via multilinear interpolation."""
        # First check the inputs make sense
        ndim_in = len(boundaries)
        ndim_out = 0

        for i, (b1, b2) in enumerate(boundaries):
            if not isinstance(b1, HypercubeDomain) or not isinstance(b2, HypercubeDomain):
                raise TypeError("Both boundaries must be HyperCubes.")

            if b1.ndim_physical != b2.ndim_physical:
                raise ValueError(
                    f"The number of physical dimensions for boundaries of dimension {i}"
                    " does not match between the two boundaries."
                )

            if b1.ndim_reference + 1 != ndim_in or b2.ndim_reference + 1 != ndim_in:
                raise ValueError(
                    f"One or both boundaries for dimension {i} do not have the correct "
                    "number of input dimensions."
                )
            if ndim_out == 0:
                ndim_out = b1.ndim_physical
            elif b1.ndim_physical != ndim_out:
                raise ValueError(
                    f"Number of physical dimensions for boundary {i} does not"
                    " match the number specified by previous boundaries."
                )

        dofs = [
            dofs_from_boundary_pairs(
                *((b1.dofs[idim], b2.dofs[idim]) for b1, b2 in boundaries)
            )
            for idim in range(ndim_out)
        ]

        return HypercubeDomain(*dofs)


@dataclass(frozen=True)
class Line(HypercubeDomain):
    """One dimensional object connecting two points.

    Parameters
    ----------
    *knots : npt.ArrayLike
        Values of knot points, that are used to interpolate the position
        along the line using Bernstein polynomials. All knots must have
        the same number of entries, but their number is not limited.
    """

    knots: npt.NDArray[np.double]

    def __init__(self, *knots: npt.ArrayLike) -> None:
        pts = np.array(knots)
        if pts.ndim != 2:
            raise ValueError("Line must be specified by an array of points.")
        if pts.shape[0] < 2:
            raise ValueError("At least two points must be given for a line.")
        ndim = pts.shape[1]
        basis = BasisSpecs(BasisType.BERNSTEIN, pts.shape[0] - 1)
        func_space = FunctionSpace(basis)
        dofs: list[DegreesOfFreedom] = list()
        for idim in range(ndim):
            dofs.append(DegreesOfFreedom(func_space, pts[:, idim]))
        object.__setattr__(self, "knots", pts)
        super().__init__(*dofs)

    @property
    def start(self) -> npt.NDArray[np.double]:
        """The start point of the line."""
        return self.knots[0, :]

    @property
    def end(self) -> npt.NDArray[np.double]:
        """The end point of the line."""
        return self.knots[-1, :]

    def reverse(self) -> Line:
        """Reverse the orientation of the line.

        Returns
        -------
        Line
            Line which has its orientation flipped.
        """
        return Line(*np.flip(self.knots, axis=0))


class Quad(HypercubeDomain):
    """Two dimensional object with four corners.

    Parameters
    ----------
    bottom : Line
        Bottom boundary along which the second dimension is -1. Starts where
        the left boundary ends and ends where the right boundary starts.

    right : Line
        Right boundary along which the first dimension is +1. Starts where
        the bottom boundary ends and ends where the top boundary starts.

    top : Line
        Top boundary along which the second dimension is +1. Starts where
        the right boundary ends and ends where the left boundary starts.

    left : Line
        Left boundary along which the first dimension is -1. Starts where
        the top boundary ends and ends where the bottom boundary starts.
    """

    def __init__(self, bottom: Line, right: Line, top: Line, left: Line) -> None:
        # Check we're dealing with the real types
        for line in (bottom, right, top, left):
            if type(line) is not Line:
                raise TypeError(f"Only {Line} objects can be used as inputs for a Quad")

        # Check the surface is closed
        if np.any(bottom.end != right.start):
            raise ValueError("The right side does not start where the bottom ends.")
        if np.any(right.end != top.start):
            raise ValueError("The top side does not start where the right ends.")
        if np.any(top.end != left.start):
            raise ValueError("The left side does not start where the top ends.")
        if np.any(left.end != bottom.start):
            raise ValueError("The bottom side does not start where the left ends.")

        # Determine the function spaces we're dealing with
        fs_b = bottom.function_space
        fs_r = right.function_space
        fs_t = top.function_space
        fs_l = left.function_space
        assert fs_b.dimension == 1
        assert fs_r.dimension == 1
        assert fs_t.dimension == 1
        assert fs_l.dimension == 1

        # Find the highest orders we must represent
        max_h = max((1, fs_b.orders[0], fs_t.orders[0]))  # horizontal edges
        max_v = max((1, fs_r.orders[0], fs_l.orders[0]))  # vertical edges

        fs_quad = FunctionSpace(
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, max_h),
            BasisSpecs(BasisType.LAGRANGE_UNIFORM, max_v),
        )

        xh = np.linspace(-1, +1, max_h + 1)
        xv = np.linspace(-1, +1, max_v + 1)

        coords_c1 = bottom.sample(xh)
        coords_c2 = right.sample(xv)
        coords_c3 = top.sample(np.flip(xh))
        coords_c4 = left.sample(np.flip(xv))

        gx, gy = np.meshgrid(xh, xv)  # TODO: check if this gives correct results

        new_dofs: list[DegreesOfFreedom] = list()

        p_bl = bottom.start
        p_br = right.start
        p_tr = top.start
        p_tl = left.start
        # TODO: fix
        for c1, c2, c3, c4, bl, br, tr, tl in zip(
            coords_c1,
            coords_c2,
            coords_c3,
            coords_c4,
            p_bl,
            p_br,
            p_tr,
            p_tl,
            strict=True,
        ):
            dof_vals = (
                c1[None, :] * (1 - gy) / 2
                + c2[:, None] * (1 + gx) / 2
                + c3[None, :] * (1 + gy) / 2
                + c4[:, None] * (1 - gx) / 2
            ) - (
                bl * (1 - gy) / 2 * (1 - gx) / 2
                + br * (1 - gy) / 2 * (1 + gx) / 2
                + tr * (1 + gy) / 2 * (1 + gx) / 2
                + tl * (1 + gy) / 2 * (1 - gx) / 2
            )
            new_dofs.append(DegreesOfFreedom(fs_quad, dof_vals.T))

        super().__init__(*new_dofs)

    @classmethod
    def from_corners(
        cls,
        bottom_left: npt.ArrayLike,
        bottom_right: npt.ArrayLike,
        top_right: npt.ArrayLike,
        top_left: npt.ArrayLike,
    ) -> Self:
        """Create a new (linear) Quad based on four corners.

        Parameters
        ----------
        bottom_left : npt.ArrayLike
            Bottom left corner.

        bottom_right : npt.ArrayLike
            Bottom right corner.

        top_right : npt.ArrayLike
            Top right corner.

        top_left : npt.ArrayLike
            Top left corner.

        Returns
        -------
        Quad
            Quad domain that has straight lines for its edges.
        """
        return cls(
            bottom=Line(bottom_left, bottom_right),
            right=Line(bottom_right, top_right),
            top=Line(top_right, top_left),
            left=Line(top_left, bottom_left),
        )
