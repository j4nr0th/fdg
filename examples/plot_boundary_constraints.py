r"""
.. currentmodule:: fdg

Boundary Trace Mass Matrices
============================

Each row of a boundary mass matrix is a trace inner product,

.. math::

   r_i(q_e) = (v_i, \operatorname{tr}_e q_e)_{F_e},

where :math:`v_i` are the windowed common Legendre test functions of the
shared face: the per-axis minimum order of the incident elements, reduced by
two on every axis that carries no covector of the k-form component
(``axis_skip=2``).  The columns pair with the element's trace degrees of
freedom, so one matrix collects the boundary geometry and metric factors of
one element trace in a single per-object operator.

:func:`compute_kform_boundary_mass_matrices` assembles one matrix per
incident element of a shared boundary object in a single call and returns the
rows in a packed CSR-like representation.  This example assembles the shared
edge of two quadrilaterals and the shared face of two hexahedra for every
k-form order living on that face, prints the packed row data, and verifies
that both incident elements pair a constant trace with the same values.

The printed checks run before the figures are created, and the boundary
geometry of both setups is plotted afterwards.
"""  # noqa: D205 D400

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    KFormSpecs,
    SpaceMap,
    compute_kform_boundary_mass_matrices,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#: Element k-form order. The windowed common test space reduces the per-axis
#: order by two on axes that carry no component covector, so the order is
#: chosen high enough to leave visible rows on the shared face.
ELEMENT_ORDER = 3


def make_element_basis(ndim: int) -> FunctionSpace:
    """Make the uniform element k-form basis of the demo order."""
    return FunctionSpace(
        *(BasisSpecs(BasisType.LAGRANGE_UNIFORM, ELEMENT_ORDER) for _ in range(ndim))
    )


def make_2d_maps() -> tuple[list[SpaceMap], FunctionSpace]:
    """Make two translated affine quadrilateral maps."""
    basis = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    integration = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))
    y_values = [-1, 1, -1, 1]
    maps = [
        SpaceMap(
            CoordinateMap(DegreesOfFreedom(basis, x_values), integration),
            CoordinateMap(DegreesOfFreedom(basis, y_values), integration),
        )
        for x_values in ([-1, -1, 0, 0], [0, 0, 1, 1])
    ]
    return maps, basis


def make_3d_maps() -> tuple[
    list[SpaceMap], FunctionSpace, tuple[tuple[list[float], ...], ...]
]:
    """Make two translated affine hexahedral maps."""
    basis = FunctionSpace(*(BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1) for _ in range(3)))
    integration = IntegrationSpace(*(IntegrationSpecs(3) for _ in range(3)))
    x_values_0 = [-1, -1, 1, 1, -1, -1, 1, 1]
    y_values_0 = [-1, 1, -1, 1, -1, 1, -1, 1]
    z_values = [-1, -1, -1, -1, 1, 1, 1, 1]
    coordinate_values = (
        ([value + 1 for value in x_values_0], y_values_0, z_values),
        ([value + 1 for value in y_values_0], x_values_0, z_values),
    )
    maps = [
        SpaceMap(
            *(
                CoordinateMap(DegreesOfFreedom(basis, values), integration)
                for values in values_set
            )
        )
        for values_set in coordinate_values
    ]
    return maps, basis, coordinate_values  # type: ignore[return-value]


def boundary_mass_matrices(
    order: int,
    basis: FunctionSpace,
    maps: list[SpaceMap],
    orientations: tuple[Sequence[int], Sequence[int]],
    boundary_dimension: int,
) -> list[np.ndarray]:
    """Assemble and print one shared-boundary mass matrix per incident element.

    The rows of every matrix are the windowed common Legendre test space of
    the shared face; the columns are the trace degrees of freedom of the
    element k-form.  The orientation records fix the shared face inside each
    element: the first entry is the signed one-based fixed axis, the
    remaining entries list the tangential axes.
    """
    specs = [KFormSpecs(order, basis) for _ in maps]
    common_specs, _common_integration, matrices, packed = (
        compute_kform_boundary_mass_matrices(
            specs,
            orientations,
            [element_map.integration_space for element_map in maps],
            axis_skip=(2,) * boundary_dimension,
            boundary_dimension=boundary_dimension,
            packed=True,
        )
    )
    print(
        f"  common test space: k={common_specs.order}, "
        f"rows={matrices[0].shape[0]}, trace DoFs per element={matrices[0].shape[1]}"
    )
    for element_id, (matrix, rows) in enumerate(zip(matrices, packed, strict=True)):
        row_offsets, _sides, components, local_dofs, coefficients = rows
        print(
            f"  element {element_id}: shape={matrix.shape}, "
            f"nnz={int(np.count_nonzero(matrix))}, packed_entries={coefficients.size}"
        )
        print(f"    offsets={row_offsets.tolist()}")
        support = np.nonzero(coefficients)[0]
        print(f"    first components={components[support][:8].tolist()}")
        print(f"    first local DoFs={local_dofs[support][:8].tolist()}")
        print(f"    first coefficients={coefficients[support][:8]}")
    return matrices


def report_2d() -> None:
    """Assemble the shared-edge boundary mass matrices of two quadrilaterals."""
    maps, _ = make_2d_maps()
    basis = make_element_basis(2)
    print("2D shared edge (boundary dimension 1)")
    for order in range(2):
        print(f"  k={order} element k-form:")
        matrices = boundary_mass_matrices(order, basis, maps, ((1, 2), (-1, 2)), 1)
        if order == 0:
            ones = [np.ones(matrix.shape[1]) for matrix in matrices]
            residual = np.max(np.abs(matrices[0] @ ones[0] - matrices[1] @ ones[1]))
            print(f"    constant trace pairing residual = {residual:.3e}")


def report_3d() -> None:
    """Assemble the shared-face boundary mass matrices of two hexahedra."""
    maps, _, _ = make_3d_maps()
    basis = make_element_basis(3)
    print("3D shared face (boundary dimension 2)")
    for order in range(3):
        print(f"  k={order} element k-form:")
        matrices = boundary_mass_matrices(order, basis, maps, ((1, 2, 3), (-1, 2, 3)), 2)
        if order == 0:
            ones = [np.ones(matrix.shape[1]) for matrix in matrices]
            residual = np.max(np.abs(matrices[0] @ ones[0] - matrices[1] @ ones[1]))
            print(f"    constant trace pairing residual = {residual:.3e}")


def plot_geometry() -> None:
    """Plot the adjacent 2D quadrilaterals and 3D hexahedra."""
    fig, ax_2d = plt.subplots(figsize=(12, 5))
    ax_2d.plot([-1, 0, 0, -1, -1], [-1, -1, 1, 1, -1], "o-", label="element A")
    ax_2d.plot([0, 1, 1, 0, 0], [-1, -1, 1, 1, -1], "o-", label="element B")
    ax_2d.axvline(0, color="black", linestyle="--", label="shared edge")
    ax_2d.set(aspect="equal", title="2D shared boundary", xlabel="x", ylabel="y")
    ax_2d.legend()

    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    for x0, x1, color in ((-1, 0, "tab:blue"), (0, 1, "tab:orange")):
        vertices = np.array(
            [
                [x0, -1, -1],
                [x1, -1, -1],
                [x1, 1, -1],
                [x0, 1, -1],
                [x0, -1, 1],
                [x1, -1, 1],
                [x1, 1, 1],
                [x0, 1, 1],
            ]
        )
        faces = [
            vertices[list(index)]
            for index in (
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (3, 2, 6, 7),
                (0, 3, 7, 4),
                (1, 2, 6, 5),
            )
        ]
        ax_3d.add_collection3d(
            Poly3DCollection(faces, alpha=0.12, facecolor=color, edgecolor=color)
        )
    ax_3d.set(
        xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1), title="3D shared face"
    )
    fig.tight_layout()
    plt.show()


report_2d()
report_3d()
plot_geometry()
