"""Types to simplify specifying domains."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
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
    """Create new DoFs from opposite boundary pairs by transfinite blending.

    Each pair contains the boundaries at the negative and positive side of one
    reference axis.  Boundary coordinates must use the remaining reference axes
    in their natural order; in particular, this function does not infer or alter
    boundary orientations.
    """
    ndim_in = len(boundaries)
    if ndim_in == 0:
        raise ValueError("At least one boundary pair must be specified.")

    max_orders = np.zeros(ndim_in, dtype=np.uintc)
    for idim, pair in enumerate(boundaries):
        if len(pair) != 2:
            raise ValueError("Every boundary must be specified as a pair.")
        b1, b2 = pair
        if type(b1) is not DegreesOfFreedom or type(b2) is not DegreesOfFreedom:
            raise TypeError("Both boundaries must be DegreesOfFreedom.")
        expected_dimension = ndim_in - 1
        if (
            b1.function_space.dimension != expected_dimension
            or b2.function_space.dimension != expected_dimension
        ):
            raise ValueError(
                f"Boundary pair {idim} must have {expected_dimension} input dimensions."
            )

        for parent_axis in range(ndim_in):
            if parent_axis == idim:
                continue
            boundary_axis = parent_axis if parent_axis < idim else parent_axis - 1
            max_orders[parent_axis] = max(
                max_orders[parent_axis],
                b1.function_space.orders[boundary_axis],
                b2.function_space.orders[boundary_axis],
            )

    function_space = FunctionSpace(
        *(
            BasisSpecs(BasisType.LAGRANGE_GAUSS_LOBATTO, int(order))
            for order in max_orders
        )
    )
    output_dofs = DegreesOfFreedom(function_space)
    out_vals = output_dofs.values
    corrected_boundaries: list[tuple[DegreesOfFreedom, DegreesOfFreedom]] = []
    for idim, (b1, b2) in enumerate(boundaries):
        boundary_orders = (*max_orders[:idim], *max_orders[idim + 1 :])
        corrected_boundaries.append(
            (
                b1.lagrange_projection(boundary_orders),
                b2.lagrange_projection(boundary_orders),
            )
        )

    nodes = tuple(
        IntegrationSpecs(int(order), IntegrationMethod.GAUSS_LOBATTO).nodes()
        for order in max_orders
    )
    weights = tuple(
        (
            (1 - node) / 2,
            (1 + node) / 2,
        )
        for node in nodes
    )

    for subset_size in range(1, ndim_in + 1):
        coefficient = -1 if subset_size % 2 == 0 else 1
        for axes in combinations(range(ndim_in), subset_size):
            base_axis = axes[0]
            for signs in product((0, 1), repeat=subset_size):
                intersection = corrected_boundaries[base_axis][signs[0]]
                for removed, (axis, side) in enumerate(
                    zip(axes[1:], signs[1:], strict=True)
                ):
                    local_axis = axis - 1 - removed
                    intersection = intersection.plane_projection(
                        local_axis, -1.0 if side == 0 else +1.0
                    )
                value_shape: list[int] = []
                value_axis = 0
                for axis in range(ndim_in):
                    if axis in axes:
                        value_shape.append(1)
                    else:
                        value_shape.append(intersection.values.shape[value_axis])
                        value_axis += 1
                value = intersection.values.reshape(value_shape)

                blend = 1.0
                for axis, side in zip(axes, signs, strict=True):
                    weight_shape = [1] * ndim_in
                    weight_shape[axis] = int(max_orders[axis]) + 1
                    blend = blend * weights[axis][side].reshape(weight_shape)
                out_vals += coefficient * value * blend

    return output_dofs


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
            if type(d) is not DegreesOfFreedom:
                raise TypeError(
                    f"Argument {i} was not {DegreesOfFreedom}, but {type(d)}."
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
        if len(ranges) > n_dim_ref:
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
        """Create a new domain from its boundaries by transfinite blending."""
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

        _validate_boundary_intersections(tuple(boundaries))
        dofs = [
            dofs_from_boundary_pairs(
                *((b1.dofs[idim], b2.dofs[idim]) for b1, b2 in boundaries)
            )
            for idim in range(ndim_out)
        ]

        return HypercubeDomain(*dofs)


def _boundary_target_orders(
    boundaries: tuple[tuple[HypercubeDomain, HypercubeDomain], ...],
) -> tuple[int, ...]:
    """Find common parent-axis orders for a collection of boundary pairs."""
    ndim_reference = len(boundaries)
    orders = np.zeros(ndim_reference, dtype=np.uintc)
    for idim, (boundary_start, boundary_end) in enumerate(boundaries):
        for parent_axis in range(ndim_reference):
            if parent_axis == idim:
                continue
            boundary_axis = parent_axis if parent_axis < idim else parent_axis - 1
            orders[parent_axis] = max(
                orders[parent_axis],
                boundary_start.function_space.orders[boundary_axis],
                boundary_end.function_space.orders[boundary_axis],
            )
    return tuple(int(order) for order in orders)


def _project_boundary(
    boundary: HypercubeDomain, idim: int, target_orders: tuple[int, ...]
) -> HypercubeDomain:
    """Project one boundary onto the common parent-axis orders."""
    boundary_orders = (*target_orders[:idim], *target_orders[idim + 1 :])
    return HypercubeDomain(
        *(dof.lagrange_projection(boundary_orders) for dof in boundary.dofs)
    )


def _validate_boundary_intersections(
    boundaries: tuple[tuple[HypercubeDomain, HypercubeDomain], ...],
) -> None:
    """Check that every pair of neighboring boundaries has a common trace."""
    target_orders = _boundary_target_orders(boundaries)
    projected = tuple(
        tuple(_project_boundary(boundary, idim, target_orders) for boundary in pair)
        for idim, pair in enumerate(boundaries)
    )

    for idim, jdim in combinations(range(len(boundaries)), 2):
        local_j_on_i = jdim if jdim < idim else jdim - 1
        local_i_on_j = idim if idim < jdim else idim - 1
        for boundary_i, side_i in enumerate((-1.0, +1.0)):
            for boundary_j, side_j in enumerate((-1.0, +1.0)):
                traces_i = [
                    dof.plane_projection(local_j_on_i, side_j).values
                    for dof in projected[idim][boundary_i].dofs
                ]
                traces_j = [
                    dof.plane_projection(local_i_on_j, side_i).values
                    for dof in projected[jdim][boundary_j].dofs
                ]
                if any(
                    left.shape != right.shape
                    or not np.allclose(left, right, rtol=1e-10, atol=1e-12)
                    for left, right in zip(traces_i, traces_j, strict=True)
                ):
                    raise ValueError(
                        f"Boundary intersections for axes {idim} and {jdim} do not match."
                    )


def _domain_from_boundary_points(
    points: npt.ArrayLike, ndim_reference: int
) -> HypercubeDomain:
    """Fit a boundary map through a tensor-product array of physical points."""
    values = np.asarray(points, dtype=np.double)
    if values.ndim != ndim_reference + 1:
        raise ValueError(
            f"Boundary points must have {ndim_reference} reference axes and one "
            "physical-coordinate axis."
        )
    if any(size < 2 for size in values.shape[:-1]):
        raise ValueError("At least two points are required along every boundary axis.")
    if values.shape[-1] == 0:
        raise ValueError("At least one physical coordinate must be provided.")

    function_space = FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, size - 1) for size in values.shape[:-1])
    )
    return HypercubeDomain(
        *(
            DegreesOfFreedom(function_space, values[..., icoord])
            for icoord in range(values.shape[-1])
        )
    )


class Hypercube(HypercubeDomain):
    """N-dimensional hypercube assembled from opposite boundary pairs.

    Parameters
    ----------
    *boundaries : tuple of HypercubeDomain
        One ``(negative, positive)`` boundary pair for every reference axis.
        Each boundary must use the remaining reference axes in ascending order.

    Notes
    -----
    Boundaries are combined with inclusion-exclusion blending.  Consequently,
    curved boundaries are preserved when all neighboring boundary intersections
    agree, while each boundary may have a different polynomial order.
    """

    def __init__(self, *boundaries) -> None:
        if not boundaries:
            raise ValueError("At least one boundary pair must be specified.")
        if all(isinstance(boundary, HypercubeDomain) for boundary in boundaries):
            if len(boundaries) % 2:
                raise ValueError("An even number of boundaries must be specified.")
            boundaries = tuple(zip(boundaries[::2], boundaries[1::2], strict=True))
        elif any(not isinstance(pair, tuple) or len(pair) != 2 for pair in boundaries):
            raise TypeError(
                "Boundaries must be HypercubeDomain objects or pairs of them."
            )

        ndim_reference = len(boundaries)
        ndim_physical = boundaries[0][0].ndim_physical
        for idim, (b1, b2) in enumerate(boundaries):
            if not isinstance(b1, HypercubeDomain) or not isinstance(b2, HypercubeDomain):
                raise TypeError(
                    f"Boundary pair {idim} must contain HypercubeDomain objects."
                )
            if b1.ndim_physical != ndim_physical or b2.ndim_physical != ndim_physical:
                raise ValueError(
                    f"Boundary pair {idim} does not have {ndim_physical} physical "
                    "dimensions."
                )
            if b1.ndim_reference != ndim_reference - 1:
                raise ValueError(
                    f"Boundary pair {idim} must have {ndim_reference - 1} "
                    "reference dimensions."
                )
            if b2.ndim_reference != ndim_reference - 1:
                raise ValueError(
                    f"Boundary pair {idim} must have {ndim_reference - 1} "
                    "reference dimensions."
                )
        boundaries = tuple(boundaries)
        _validate_boundary_intersections(boundaries)

        dofs = [
            dofs_from_boundary_pairs(
                *((b1.dofs[icoord], b2.dofs[icoord]) for b1, b2 in boundaries)
            )
            for icoord in range(ndim_physical)
        ]
        super().__init__(*dofs)

    @classmethod
    def from_corners(cls, *corners: npt.ArrayLike) -> Self:
        """Create a multilinear hypercube from its corner coordinates.

        Parameters
        ----------
        *corners : npt.ArrayLike
            The ``2**N`` physical corner coordinates.  Corner ``k`` is on the
            positive side of reference axis ``a`` when bit ``a`` of ``k`` is set.

        Returns
        -------
        Hypercube
            Hypercube with linear interpolation in every reference axis.
        """
        points = np.asarray(corners[0] if len(corners) == 1 else corners, dtype=np.double)
        if points.ndim != 2 or points.shape[0] < 2:
            raise ValueError("At least two coordinate-valued corners must be given.")
        if points.shape[0] & (points.shape[0] - 1):
            raise ValueError("The number of corners must be a power of two.")

        ndim_reference = points.shape[0].bit_length() - 1
        function_space = FunctionSpace(
            *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(ndim_reference))
        )
        # np.reshape uses the first index as the most significant one, whereas
        # hypercube corner IDs use reference axis zero as the least significant bit.
        corner_grid = points.reshape(
            (2,) * ndim_reference + (points.shape[1],)
        ).transpose((*reversed(range(ndim_reference)), ndim_reference))
        dofs = [
            DegreesOfFreedom(function_space, corner_grid[..., icoord])
            for icoord in range(points.shape[1])
        ]
        result = cls.__new__(cls)
        HypercubeDomain.__init__(result, *dofs)
        return result

    @classmethod
    def from_boundary_points(
        cls, *boundaries: tuple[npt.ArrayLike, npt.ArrayLike]
    ) -> Self:
        """Create a hypercube from tensor-product points on its boundaries.

        Parameters
        ----------
        *boundaries : tuple of array_like
            One ``(negative, positive)`` pair for every reference axis.  Each
            array has shape ``(n_0, ..., n_(N-2), n_physical)`` and stores points
            on a boundary at uniform reference coordinates.  Its axes correspond
            to the parent axes other than the boundary axis, in ascending order.
            Intersections of neighboring boundaries must contain the same points.

        Returns
        -------
        Hypercube
            Hypercube whose boundary maps interpolate the supplied points.

        Notes
        -----
        The points are treated as Lagrange-uniform nodal values.  Boundaries with
        different nodal orders are projected to common orders before their shared
        traces are checked and blended.
        """
        if not boundaries:
            raise ValueError("At least one boundary pair must be specified.")

        ndim_reference = len(boundaries)
        boundary_domains: list[tuple[HypercubeDomain, HypercubeDomain]] = []
        for idim, pair in enumerate(boundaries):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(f"Boundary pair {idim} must contain two point arrays.")
            boundary_domains.append(
                (
                    _domain_from_boundary_points(pair[0], ndim_reference - 1),
                    _domain_from_boundary_points(pair[1], ndim_reference - 1),
                )
            )

        return cls(*boundary_domains)


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
