r"""
.. currentmodule:: fdg

Boundary Constraints and Continuity
===================================

A boundary row is a trace inner product,

.. math::

   r_i(u_e) = (v_i, \operatorname{tr}_e u_e)_{F_e}.

For two adjacent elements, continuity is checked with

.. math::

   T_A u_A - T_B u_B = 0,

where each :math:`T` is assembled by
:func:`compute_kform_boundary_constraints`.  The function returns a packed
row representation rather than choosing a global sparse-matrix numbering, so
this example converts each local operator to a SciPy CSR array explicitly.

The printed checks run before the figures are created.  They cover point and
line traces in 2D, and point, line, and face traces in 3D.  Every valid test
and element k-form order is assembled; scalar and tangential one-form traces
are compared numerically across the shared objects.
"""  # noqa: D205 D400

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
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
    compute_kform_boundary_constraints,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import csr_array


def packed_to_sparse(
    result: tuple[npt.NDArray[np.generic], ...], specs: KFormSpecs
) -> csr_array:
    """Convert one element's packed rows to a local sparse operator."""
    row_offsets, components, local_dofs, coefficients = result
    columns = np.empty(components.size, dtype=np.uintp)
    for component in range(specs.component_count):
        component_entries = components == component
        component_slice = specs.get_component_slice(component)
        columns[component_entries] = component_slice.start + local_dofs[component_entries]
    rows = np.repeat(
        np.arange(row_offsets.size - 1), np.diff(row_offsets).astype(np.intp)
    )
    return csr_array(
        (coefficients, (rows, columns)),
        shape=(row_offsets.size - 1, int(np.sum(specs.component_dof_counts))),
    )


def make_test_specs(dimension: int, order: int) -> KFormSpecs:
    """Create a Legendre test space on a canonical boundary."""
    return KFormSpecs(
        order,
        FunctionSpace(*(BasisSpecs(BasisType.LEGENDRE, 1) for _ in range(dimension))),
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


def make_2d_collections() -> tuple[np.ndarray, np.ndarray]:
    """Return two quadrilaterals sharing edge 2 and points 2/3."""
    lines = np.array(
        [[0, 1], [0, 2], [2, 3], [1, 3], [2, 4], [4, 5], [3, 5]],
        dtype=np.uint64,
    )
    elements = np.array([[0, 1, 2, 3], [2, 4, 5, 6]], dtype=np.uint64)
    return lines, elements


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


def make_3d_collections() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two hexahedra sharing face 2, line 0, and point 0."""
    lines = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 8],
            [0, 8],
            [1, 9],
            [5, 10],
            [4, 11],
        ],
        dtype=np.uint64,
    )
    faces = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 9, 4, 8],
            [1, 10, 5, 9],
            [2, 11, 6, 10],
            [3, 8, 7, 11],
            [0, 17, 12, 16],
            [9, 18, 13, 17],
            [4, 18, 14, 19],
            [8, 19, 15, 16],
            [12, 13, 14, 15],
        ],
        dtype=np.uint64,
    )
    elements = np.array([[5, 2, 0, 3, 4, 1], [2, 6, 7, 10, 8, 9]], dtype=np.uint64)
    return lines, faces, elements


def boundary_operator(
    test_specs: KFormSpecs,
    element_specs: KFormSpecs,
    element_map: SpaceMap,
    collections: tuple[np.ndarray, ...],
    npts: int,
    element_id: int,
    boundary_id: int,
) -> csr_array:
    """Assemble and print one local boundary operator."""
    result = compute_kform_boundary_constraints(
        test_specs,
        element_specs,
        element_map,
        collections,
        npts,
        element_id,
        boundary_id,
    )
    matrix = packed_to_sparse(result, element_specs)
    row_offsets, components, local_dofs, coefficients = result
    print(
        f"  element {element_id}: rows={matrix.shape[0]}, cols={matrix.shape[1]}, "
        f"nnz={matrix.nnz}, packed_entries={coefficients.size}"
    )
    print(f"    offsets={row_offsets.tolist()}")
    print(f"    first coefficients={coefficients[: min(8, coefficients.size)]}")
    print(f"    first components={components[: min(8, components.size)]}")
    print(f"    first local DoFs={local_dofs[: min(8, local_dofs.size)]}")
    return matrix


def report_2d() -> None:
    """Print 2D point and edge continuity checks."""
    maps, basis = make_2d_maps()
    collections = make_2d_collections()
    print("2D polynomial k-form: every component is the constant polynomial 1")
    for face_dim, boundary_id, label in ((0, 2, "point"), (1, 2, "edge")):
        print(f"2D shared {label} boundary (object {boundary_id})")
        for order in range(face_dim + 1):
            test_specs = make_test_specs(face_dim, order)
            element_specs = KFormSpecs(order, basis)
            matrices = [
                boundary_operator(
                    test_specs,
                    element_specs,
                    maps[element_id],
                    collections,
                    6,
                    element_id,
                    boundary_id,
                )
                for element_id in range(2)
            ]
            element_values = [
                np.ones(int(np.sum(element_specs.component_dof_counts))) for _ in maps
            ]
            traces = [
                matrix @ element_values[element_id]
                for element_id, matrix in enumerate(matrices)
            ]
            residual = np.max(np.abs(traces[0] - traces[1]))
            print(f"  k={order}: continuity residual = {residual:.3e}")


def report_3d() -> None:
    """Print 3D point, line, and face assembly and continuity checks."""
    maps, basis, _ = make_3d_maps()
    collections = make_3d_collections()
    print("3D polynomial k-form: every component is the constant polynomial 1")
    for face_dim, boundary_id, label in ((0, 0, "point"), (1, 0, "line"), (2, 2, "face")):
        print(f"3D shared {label} boundary (object {boundary_id})")
        for order in range(face_dim + 1):
            test_specs = make_test_specs(face_dim, order)
            element_specs = KFormSpecs(order, basis)
            matrices = [
                boundary_operator(
                    test_specs,
                    element_specs,
                    maps[element_id],
                    collections,
                    12,
                    element_id,
                    boundary_id,
                )
                for element_id in range(2)
            ]
            element_values = [
                np.ones(int(np.sum(element_specs.component_dof_counts))) for _ in maps
            ]
            traces = [
                matrix @ element_values[element_id]
                for element_id, matrix in enumerate(matrices)
            ]
            if order == 0:
                residual = np.max(np.abs(traces[0] - traces[1]))
                print(f"  k=0 constant test mode residual = {residual:.3e}")
            elif face_dim == 1:
                residual = np.max(np.abs(traces[0] - traces[1]))
                print(f"  k=1 tangential one-form residual = {residual:.3e}")
            else:
                print(
                    f"  k={order}: assembled all face components; compare with "
                    "orientation-matched local DoFs"
                )


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
