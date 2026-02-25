"""Types to simplify specifying domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from interplib._interp import (
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
from interplib.degrees_of_freedom import reconstruct
from interplib.enum_type import BasisType
from interplib.integration import Integrable, integrate_callable


def _array_axis_slice(a: npt.NDArray, idx: int, axis: int):
    """Take a slice from a numpy array along the specified axis."""
    slices: list[slice | int] = [slice(None)] * a.ndim
    slices[axis] = slice(idx, idx + 1) if idx >= 0 else slice(idx, idx - 1)
    return a[tuple(slices)]


@dataclass(frozen=True)
class HypercubeDomain:
    """Base type for all domains."""

    dofs: tuple[DegreesOfFreedom, ...]

    def __init__(self, *dofs: DegreesOfFreedom) -> None:
        if not len(dofs):
            raise ValueError("At least one coordinate must have its DoFs specified.")

        fs: FunctionSpace | None = None
        for i, d in enumerate(dofs):
            if type(dofs) is None:
                raise TypeError(
                    f"Argument {i} was not {DegreesOfFreedom}, but {type(dofs)}."
                )
            if fs is None:
                fs = d.function_space
            elif d.function_space != fs:
                raise ValueError(
                    f"Function spaces of the DoFs {i} does not match the rest!"
                )

        object.__setattr__(self, "dofs", dofs)

    @property
    def ndim_physical(self) -> int:
        """Number of physical dimensions of the domain."""
        return len(self.dofs)

    @property
    def ndim_reference(self) -> int:
        """Number of reference dimensions of the domain."""
        return len(self.dofs[0].shape)

    def __call__(
        self,
        space: IntegrationSpace,
        /,
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> SpaceMap:
        """Create a space map based on the integration space."""
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

    def compute_size(
        self,
        int_space: IntegrationSpace | None = None,
        *,
        integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
        basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
    ) -> float:
        """Compute the size of the domain."""
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
        """Split self into a sub-region of the domain."""
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

    # @classmethod
    # def from_boundaries(cls, *boundaries: HypercubeDomain) -> Self:
    #     """Create a domain from its boundaries."""
    #     nbnd = len(boundaries)
    #     if nbnd & 1:
    #         raise ValueError("Domain must have an even number of boundaries.")

    #     ndim_in = nbnd // 2
    #     if ndim_in <= 1:
    #         raise ValueError(
    #             "Domain must have at least two dimensions for this way of constructing
    #  it"
    #             " to work."
    #         )

    #     for ib, bnd in enumerate(boundaries):
    #         if bnd.ndim_reference + 1 != ndim_in:
    #             raise ValueError(
    #                 f"Boundary {ib} is not defined on the reference domain with "
    #                 f"{ndim_in - 1} dimension as is expected for a new domain."
    #             )
    #         if bnd.ndim_physical != boundaries[0].ndim_physical:
    #             raise ValueError(
    #                 f"Boundary {ib} does not have the matching number of physical "
    #                 "dimension with other boundaries that have "
    #                 f"{boundaries[0].ndim_physical}"
    #             )

    #     start_boundaries = boundaries[:ndim_in]
    #     end_boundaries = boundaries[ndim_in:]

    #     max_orders = np.zeros(ndim_in, dtype=np.uint64)
    #     for idim in range(ndim_in):
    #         start = start_boundaries[idim]
    #         end = end_boundaries[idim]
    #         orders = np.max(
    #             (start.function_space.orders, end.function_space.orders), axis=0
    #         )
    #         padded = np.concatenate((orders[:idim], (0,), orders[idim:]))
    #         max_orders = np.max((max_orders, padded), axis=0)

    #     fs_target = FunctionSpace(
    #         *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, int(order)) for order in max_orders
    # )
    #     )
    #     sample_points = np.meshgrid(
    #         *(np.linspace(-1, +1, order + 1) for order in max_orders)
    #     )
    #     start_bnd_pts = [
    #         start_boundaries[idim].sample(
    #             *(_array_axis_slice(pts, 0, idim) for pts in sample_points)
    #         )
    #         for idim in range(ndim_in)
    #     ]
    #     end_bnd_pts = [
    #         end_boundaries[idim].sample(
    #             *(_array_axis_slice(pts, 0, idim) for pts in sample_points)
    #         )
    #         for idim in range(ndim_in)
    #     ]

    #     # TODO: check for adjacency between neighboring boundaries

    #     raise NotImplementedError


@dataclass(frozen=True)
class Line(HypercubeDomain):
    """One dimensional object connecting two points."""

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
        """Reverse the orientation of the line."""
        return Line(*np.flip(self.knots, axis=0))


class Quad(HypercubeDomain):
    """Two dimensional object with four corners."""

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
        """Create a new (linear) Quad based on four corners."""
        return cls(
            bottom=Line(bottom_left, bottom_right),
            right=Line(bottom_right, top_right),
            top=Line(top_right, top_left),
            left=Line(top_left, bottom_left),
        )
