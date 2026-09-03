"""Global boundary, periodic, and transformed trace constraints."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from fdg._fdg import (
    KFormSpecs,
    SpaceMap,
    transform_kform_component_to_target,
)

if TYPE_CHECKING:
    from fdg._fdg import Mesh

BoundaryCallable = Callable[..., npt.ArrayLike]
BoundaryData = BoundaryCallable | Sequence[BoundaryCallable]
PackedRows = tuple[
    npt.NDArray[np.uintp],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uintp],
    npt.NDArray[np.double],
]


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """Prescribed data on one or more outer boundary faces.

    Parameters
    ----------
    faces : sequence of int
        IDs of codimension-one outer boundary faces. The IDs are the same as
        those returned by :meth:`Mesh.iterate_boundary` for ``mdim=ndim-1``.
    data : callable or sequence of callable
        Physical k-form data. A callable receives one coordinate array per
        physical dimension and returns values on those points. For a k-form,
        provide one callable per ambient physical k-form component in the
        canonical combination order. A scalar 0-form therefore uses one
        callable.
    """

    faces: tuple[int, ...]
    data: BoundaryData


@dataclass(frozen=True, slots=True)
class BoundaryPair:
    """Periodic, mirrored, or axis-rotated pair of outer boundary faces.

    Parameters
    ----------
    left_face, right_face : int
        IDs of the paired codimension-one outer boundary faces. The pair
        direction determines how ``axis_map`` is applied.
    axis_map : sequence of int
        Signed permutation of the boundary canonical axes. Entry ``i`` gives
        the right-face axis corresponding to left-face axis ``i``; a negative
        entry reverses that axis.
    """

    left_face: int
    right_face: int
    axis_map: Sequence[int]

    def __post_init__(self) -> None:
        """Normalize scalar IDs and the signed axis map."""
        object.__setattr__(self, "left_face", int(self.left_face))
        object.__setattr__(self, "right_face", int(self.right_face))
        object.__setattr__(self, "axis_map", tuple(int(value) for value in self.axis_map))


@dataclass(frozen=True, slots=True)
class BoundaryPairGroup:
    """Pair two ordered collections of outer boundary faces.

    ``left_faces[i]`` is paired with ``right_faces[i]``. The collections must
    have equal lengths; their order supplies the correspondence because a
    topological mesh has no geometry from which to infer it. Groups are useful
    for one periodic boundary of a structured mesh, such as all faces on
    ``x=-1`` paired with all faces on ``x=+1``.
    """

    left_faces: Sequence[int]
    right_faces: Sequence[int]
    axis_map: Sequence[int]

    def __post_init__(self) -> None:
        """Normalize face IDs and reject unmatched collections."""
        left_faces = tuple(int(face) for face in self.left_faces)
        right_faces = tuple(int(face) for face in self.right_faces)
        if len(left_faces) != len(right_faces):
            raise ValueError("Grouped periodic boundary sides must have equal lengths.")
        object.__setattr__(self, "left_faces", left_faces)
        object.__setattr__(self, "right_faces", right_faces)
        object.__setattr__(self, "axis_map", tuple(int(value) for value in self.axis_map))

    def expand(self) -> tuple[BoundaryPair, ...]:
        """Expand this group into one pair descriptor per face patch."""
        return tuple(
            BoundaryPair(left, right, self.axis_map)
            for left, right in zip(self.left_faces, self.right_faces, strict=True)
        )


def _normalize_boundary_conditions(
    conditions: Mapping[int, BoundaryData] | Sequence[BoundaryCondition] | None,
) -> list[tuple[int, BoundaryData]]:
    """Normalize mapping and record forms into face-data pairs.

    Parameters
    ----------
    conditions : mapping[int, BoundaryData] or sequence[BoundaryCondition] or None
        Prescribed data keyed by outer face ID, or records that associate one
        callable (or callable sequence) with one or more outer faces. ``None``
        represents no prescribed boundary data.

    Returns
    -------
    list of tuple[int, BoundaryData]
        One ``(face_id, data)`` pair per prescribed face, preserving the input
        order.

    Raises
    ------
    TypeError
        If a sequence contains something other than ``BoundaryCondition``.
    ValueError
        If a face is prescribed more than once.
    """
    if conditions is None:
        return []
    if isinstance(conditions, Mapping):
        return [(int(face), data) for face, data in conditions.items()]
    result: list[tuple[int, BoundaryData]] = []
    for condition in conditions:
        if not isinstance(condition, BoundaryCondition):
            raise TypeError(
                "boundary_conditions entries must be BoundaryCondition objects."
            )
        result.extend((int(face), condition.data) for face in condition.faces)
    faces = [face for face, _ in result]
    if len(faces) != len(set(faces)):
        raise ValueError("Each boundary face may have only one prescribed condition.")
    return result


def _normalize_periodic_pairs(
    pairs: Sequence[BoundaryPair | BoundaryPairGroup] | None,
) -> list[BoundaryPair]:
    """Expand periodic pair groups into individual face-pair descriptors.

    Parameters
    ----------
    pairs : sequence[BoundaryPair or BoundaryPairGroup] or None
        Explicit outer-face pairs and, optionally, ordered groups of pairs.
        ``None`` represents no periodic identifications.

    Returns
    -------
    list of BoundaryPair
        Individual pair descriptors in input order, with each
        ``BoundaryPairGroup`` expanded by :meth:`BoundaryPairGroup.expand`.

    Raises
    ------
    TypeError
        If an item is not a ``BoundaryPair`` or ``BoundaryPairGroup``.
    """
    if pairs is None:
        return []
    result: list[BoundaryPair] = []
    for pair in pairs:
        if isinstance(pair, BoundaryPair):
            result.append(pair)
        elif isinstance(pair, BoundaryPairGroup):
            result.extend(pair.expand())
        else:
            raise TypeError(
                "periodic_pairs entries must be BoundaryPair or "
                "BoundaryPairGroup objects."
            )
    return result


def _validate_axis_map(axis_map: Sequence[int], mdim: int) -> tuple[int, ...]:
    """Validate and normalize a signed permutation of trace axes.

    Parameters
    ----------
    axis_map : sequence of int
        Mapping from left-object axes to right-object axes. Absolute values
        must be the one-based permutation ``1..mdim``; signs encode reversals.
    mdim : int
        Dimension of the object whose axes are being mapped.

    Returns
    -------
    tuple of int
        The normalized signed axis permutation.

    Raises
    ------
    ValueError
        If the map has the wrong length or is not a signed permutation.
    """
    result = tuple(int(value) for value in axis_map)
    if len(result) != mdim or sorted(abs(value) for value in result) != list(
        range(1, mdim + 1)
    ):
        raise ValueError(
            f"Boundary axis map in dimension {mdim} must be a signed permutation "
            f"of 1..{mdim}."
        )
    return result


def _boundary_records(
    mesh: Mesh,
) -> dict[tuple[int, int], tuple[npt.NDArray[np.uint64], npt.NDArray[np.int8]]]:
    """Index all outer-boundary objects by dimension and object ID.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose boundary objects are enumerated.

    Returns
    -------
    dict[tuple[int, int], tuple[ndarray, ndarray]]
        A mapping from ``(object_dimension, object_id)`` to the element IDs
        containing that object and their corresponding orientation records.
        The arrays are the values returned by ``mesh.iterate_boundary_all``.
    """
    return {
        (mdim, int(object_id)): (element_ids, orientations)
        for mdim, object_id, element_ids, orientations in mesh.iterate_boundary_all()
    }


def _object_descendants(mesh: Mesh, mdim: int, object_id: int):
    """Yield an object and every recursively contained boundary descendant.

    Parameters
    ----------
    mesh : Mesh
        Mesh containing the object.
    mdim : int
        Dimension of the starting object.
    object_id : int
        ID of the starting object in the ``mdim`` collection.

    Yields
    ------
    tuple[int, int]
        ``(dimension, object_id)`` for the starting object and each distinct
        lower-dimensional boundary object reachable from it.

    Notes
    -----
    A visited set prevents a descendant shared by multiple boundary paths
    from being yielded more than once.
    """
    seen: set[tuple[int, int]] = set()

    def visit(current_dim: int, current_id: int):
        """Yield an object and recursively visit each boundary child."""
        key = (current_dim, current_id)
        if key in seen:
            return
        seen.add(key)
        yield key
        if current_dim == 0:
            return
        boundaries = np.asarray(mesh.collections[current_dim - 1][current_id])
        for axis in range(current_dim):
            yield from visit(current_dim - 1, int(boundaries[axis]))
            yield from visit(current_dim - 1, int(boundaries[current_dim + axis]))

    yield from visit(mdim, object_id)


def _select_periodic_object_relations(
    relations: Mapping[tuple[int, int, int], tuple[int, ...]],
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    """Keep an acyclic spanning forest of periodic object relations.

    Parameters
    ----------
    relations : mapping[tuple[int, int, int], tuple[int, ...]]
        Candidate relations keyed by ``(mdim, left_id, right_id)`` and mapped
        by their signed axis permutations.

    Returns
    -------
    dict[tuple[int, int, int], tuple[int, ...]]
        The candidate relations with any edge that would close a cycle
        removed. Iteration order determines which redundant edge is dropped.

    Notes
    -----
    Connectivity is tracked independently for each object dimension. The
    union-find structure is used only to remove redundant equations; it does
    not alter the supplied axis maps.
    """
    parents: dict[tuple[int, int], tuple[int, int]] = {}

    def find(key: tuple[int, int]) -> tuple[int, int]:
        """Return the union-find root for an object, with path compression."""
        parent = parents.setdefault(key, key)
        if parent != key:
            parent = find(parent)
            parents[key] = parent
        return parent

    selected: dict[tuple[int, int, int], tuple[int, ...]] = {}
    for key, axis_map in relations.items():
        mdim, left_id, right_id = key
        left_key = (mdim, left_id)
        right_key = (mdim, right_id)
        if find(left_key) == find(right_key):
            continue
        selected[key] = axis_map
        parents[find(left_key)] = find(right_key)
    return selected


def _restrict_map(
    element_map: SpaceMap, orientation: npt.NDArray[np.int8], ndim: int, mdim: int
) -> SpaceMap:
    """Restrict an element map to an oriented boundary object.

    Parameters
    ----------
    element_map : SpaceMap
        Full element map with ``ndim`` reference dimensions.
    orientation : ndarray[int8]
        Element orientation record. Its first ``ndim - mdim`` entries select
        the fixed boundary axes and sides; a negative value selects the lower
        side and a positive value selects the upper side.
    ndim : int
        Dimension of the element reference domain.
    mdim : int
        Dimension of the resulting boundary map.

    Returns
    -------
    SpaceMap
        The map obtained by applying the fixed restrictions in the order
        required by the nested boundary-map API.
    """
    result = element_map
    for fixed_orientation in orientation[: ndim - mdim][::-1]:
        result = result.boundary(
            abs(int(fixed_orientation)) - 1, int(fixed_orientation) > 0
        )
    return result


def _object_relations(
    mesh: Mesh, pair: BoundaryPair
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    """Derive a periodic axis map for each pair of boundary descendants.

    Parameters
    ----------
    mesh : Mesh
        Mesh containing the two outer faces.
    pair : BoundaryPair
        Outer-face pair whose signed axis map is propagated to lower strata.

    Returns
    -------
    dict[tuple[int, int, int], tuple[int, ...]]
        Relations keyed by ``(mdim, left_object_id, right_object_id)``. Each
        value is the induced signed permutation of the object axes.

    Raises
    ------
    ValueError
        If a supplied or induced axis map is not a signed permutation, or if
        the same object pair is reached with inconsistent maps.
    """
    ndim = mesh.ndim
    current = [(ndim - 1, int(pair.left_face), int(pair.right_face), pair.axis_map)]
    result: dict[tuple[int, int, int], tuple[int, ...]] = {}
    while current:
        mdim, left_id, right_id, raw_axis_map = current.pop()
        axis_map = _validate_axis_map(raw_axis_map, mdim)
        key = (mdim, left_id, right_id)
        previous = result.get(key)
        if previous is not None:
            if previous != axis_map:
                raise ValueError(
                    f"Periodic object pair {key[1:]} has inconsistent axis maps."
                )
            continue
        result[key] = axis_map
        if mdim == 0:
            continue

        left_boundaries = np.asarray(mesh.collections[mdim - 1][left_id])
        right_boundaries = np.asarray(mesh.collections[mdim - 1][right_id])
        for left_axis, mapped_axis in enumerate(axis_map):
            right_axis = abs(mapped_axis) - 1
            remaining_left = [axis for axis in range(mdim) if axis != left_axis]
            remaining_right = [axis for axis in range(mdim) if axis != right_axis]
            child_axis_map = tuple(
                (1 if axis_map[parent_axis] > 0 else -1)
                * (remaining_right.index(abs(axis_map[parent_axis]) - 1) + 1)
                for parent_axis in remaining_left
            )
            for side in (0, 1):
                right_side = side if mapped_axis > 0 else 1 - side
                left_child = int(left_boundaries[side * mdim + left_axis])
                right_child = int(right_boundaries[right_side * mdim + right_axis])
                current.append((mdim - 1, left_child, right_child, child_axis_map))
    return result


def _component_relation(
    mdim: int, order: int, left_component: int, axis_map: Sequence[int]
) -> tuple[int, int]:
    """Map a left k-form component through a signed axis permutation.

    Parameters
    ----------
    mdim : int
        Dimension of the trace object.
    order : int
        k-form degree.
    left_component : int
        Canonical combination-order component index on the left object.
    axis_map : sequence of int
        Signed permutation mapping left axes to right axes.

    Returns
    -------
    tuple[int, int]
        The right-object canonical component index and the pullback sign.
        The sign includes axis reversals and the permutation parity required
        to restore canonical wedge order.
    """
    components = list(combinations(range(mdim), order))
    left_axes = components[left_component]
    mapped_axes = [abs(axis_map[axis]) - 1 for axis in left_axes]
    inversions = sum(
        mapped_axes[i] > mapped_axes[j]
        for i in range(len(mapped_axes))
        for j in range(i + 1, len(mapped_axes))
    )
    sign = (-1) ** inversions
    for axis in left_axes:
        sign *= 1 if axis_map[axis] > 0 else -1
    return components.index(tuple(sorted(mapped_axes))), sign


def _component_rows(
    result: tuple[np.ndarray, ...],
    test_spec: KFormSpecs,
    component: int,
    element_id: int,
    scale: float,
) -> list[list[tuple[int, int, int, float]]]:
    """Extract packed rows belonging to one trace-test component.

    Parameters
    ----------
    result : tuple of ndarray
        Four-array result of the one-boundary constraint method, excluding
        the side field: row offsets, component IDs, local DoF IDs, and
        coefficients.
    test_spec : KFormSpecs
        Test-space specification that supplies component row counts.
    component : int
        Canonical k-form component whose rows should be extracted.
    element_id : int
        Global element ID to attach to every extracted entry.
    scale : float
        Factor applied to every extracted coefficient.

    Returns
    -------
    list of list of tuple
        Rows represented as ``(element_id, component, local_dof,
        coefficient)`` entries.
    """
    row_offsets, components, local_dofs, coefficients = result
    counts = np.asarray(test_spec.component_dof_counts)
    row_start = int(np.sum(counts[:component]))
    row_end = row_start + int(counts[component])
    return [
        [
            (
                element_id,
                int(components[index]),
                int(local_dofs[index]),
                scale * float(coefficients[index]),
            )
            for index in range(int(row_offsets[row]), int(row_offsets[row + 1]))
        ]
        for row in range(row_start, row_end)
    ]


def _pack(
    rows: Sequence[Sequence[tuple[int, int, int, float]]], rhs: Sequence[float]
) -> tuple[PackedRows, npt.NDArray[np.double]]:
    """Pack mutable row entries and right-hand sides into NumPy arrays.

    Parameters
    ----------
    rows : sequence of sequences of tuple
        Constraint rows. Each entry is ``(element_id, component, local_dof,
        coefficient)``.
    rhs : sequence of float
        One right-hand-side value per row.

    Returns
    -------
    tuple
        ``(packed_rows, rhs_array)``, where ``packed_rows`` contains CSR-like
        row offsets followed by one array for each entry field.

    Raises
    ------
    RuntimeError
        If the number of rows differs from the number of right-hand sides.
    """
    if len(rows) != len(rhs):
        raise RuntimeError("Constraint rows and right-hand side have different sizes.")
    row_offsets = np.zeros(len(rows) + 1, dtype=np.uintp)
    element_ids: list[int] = []
    components: list[int] = []
    local_dofs: list[int] = []
    coefficients: list[float] = []
    for row, entries in enumerate(rows):
        element_ids.extend(entry[0] for entry in entries)
        components.extend(entry[1] for entry in entries)
        local_dofs.extend(entry[2] for entry in entries)
        coefficients.extend(entry[3] for entry in entries)
        row_offsets[row + 1] = len(element_ids)
    return (
        (
            row_offsets,
            np.asarray(element_ids, dtype=np.uint64),
            np.asarray(components, dtype=np.uint32),
            np.asarray(local_dofs, dtype=np.uintp),
            np.asarray(coefficients, dtype=np.double),
        ),
        np.asarray(rhs, dtype=np.double),
    )


def _unpack(
    packed: PackedRows,
) -> list[list[tuple[int, int, int, float]]]:
    """Expand packed constraint arrays into a list of entry rows.

    Parameters
    ----------
    packed : PackedRows
        ``(row_offsets, element_ids, components, local_dofs, coefficients)``
        as returned by a mesh constraint method.

    Returns
    -------
    list of list of tuple
        One list per row, with entries represented as
        ``(element_id, component, local_dof, coefficient)``.
    """
    row_offsets, element_ids, components, local_dofs, coefficients = packed
    return [
        [
            (
                int(element_ids[index]),
                int(components[index]),
                int(local_dofs[index]),
                float(coefficients[index]),
            )
            for index in range(int(row_offsets[row]), int(row_offsets[row + 1]))
        ]
        for row in range(row_offsets.size - 1)
    ]


def _data_functions(
    data: BoundaryData, component_count: int
) -> tuple[BoundaryCallable, ...]:
    """Normalize scalar or component-wise boundary data callables.

    Parameters
    ----------
    data : callable or sequence of callable
        One physical boundary-data callable, or one callable per ambient
        k-form component.
    component_count : int
        Required number of component callables.

    Returns
    -------
    tuple of callable
        Normalized component-wise callables.

    Raises
    ------
    ValueError
        If the callable count is wrong or any item is not callable.
    """
    if callable(data):
        functions = (data,)
    else:
        functions = tuple(data)
    if len(functions) != component_count or not all(callable(fn) for fn in functions):
        raise ValueError(
            f"Boundary data must provide {component_count} callable physical "
            "k-form components."
        )
    return functions


def _physical_data_values(
    data: BoundaryData, boundary_map: SpaceMap, ndim: int, order: int
) -> np.ndarray:
    """Evaluate physical k-form boundary data on mapped quadrature points.

    Parameters
    ----------
    data : callable or sequence of callable
        Physical component functions. A function receives one coordinate
        array per ambient dimension.
    boundary_map : SpaceMap
        Map whose coordinate values provide the evaluation points.
    ndim : int
        Ambient physical dimension and number of coordinate arguments.
    order : int
        k-form degree, which determines the required component count.

    Returns
    -------
    ndarray
        Values with shape ``(comb(ndim, order), *point_shape)`` in canonical
        k-form component order.

    Raises
    ------
    ValueError
        If a callable returns values that cannot broadcast to the mapped
        point shape.
    """
    functions = _data_functions(data, comb(ndim, order))
    coordinates = tuple(
        np.asarray(boundary_map.coordinate_map(axis).values) for axis in range(ndim)
    )
    point_shape = coordinates[0].shape if coordinates else ()
    values = []
    for function in functions:
        value = np.asarray(function(*coordinates), dtype=np.double)
        try:
            values.append(np.broadcast_to(value, point_shape))  # type: ignore
        except ValueError as error:
            raise ValueError(
                "Boundary data callable returned values with an incompatible shape."
            ) from error
    return np.asarray(values, dtype=np.double)


def _boundary_dual_values(
    data: BoundaryData,
    test_spec: KFormSpecs,
    boundary_map: SpaceMap,
    ndim: int,
    mdim: int,
) -> np.ndarray:
    """Assemble prescribed boundary data into test-space dual moments.

    Parameters
    ----------
    data : callable or sequence of callable
        Physical k-form boundary data.
    test_spec : KFormSpecs
        Test trace space. Its basis and k-form degree determine the moments.
    boundary_map : SpaceMap
        Map from the canonical boundary reference object to physical space.
    ndim : int
        Ambient physical dimension.
    mdim : int
        Dimension of the boundary object.

    Returns
    -------
    ndarray
        One dual value per test row, concatenated by canonical test component.
        A zero-dimensional object contributes its point value directly;
        higher-dimensional objects include the mapped measure, basis, and
        k-form pullback.

    Raises
    ------
    ValueError
        If the physical data shape is incompatible with the mapped trace.
    """
    physical_values = _physical_data_values(data, boundary_map, ndim, test_spec.order)
    if mdim == 0:
        return np.asarray([physical_values[0].reshape(-1)[0]], dtype=np.double)

    integration = boundary_map.integration_space
    weights = np.asarray(integration.weights()) * np.abs(
        np.asarray(boundary_map.determinant)
    )
    result: list[np.ndarray] = []
    for component in range(test_spec.component_count):
        basis = test_spec.get_component_function_space(
            component
        ).values_at_integration_nodes(integration, transpose=True)
        if test_spec.order == 0:
            expected_shape = (1, *basis.shape[mdim:])
            if physical_values.shape != expected_shape:
                raise ValueError(
                    "Boundary data component shape does not match the physical trace map."
                )
            physical_component = physical_values[0].reshape(
                (1,) * mdim + physical_values[0].shape
            )
            weight = weights.reshape((1,) * mdim + weights.shape)
            weighted = basis * physical_component * weight
            value = np.sum(weighted, axis=tuple(range(mdim, weighted.ndim)))
        else:
            transformed = np.asarray(
                transform_kform_component_to_target(
                    test_spec.order, boundary_map, basis, component
                )
            )
            target_axis = mdim
            expected_shape = (
                transformed.shape[target_axis],
                *transformed.shape[target_axis + 1 :],
            )
            if physical_values.shape != expected_shape:
                raise ValueError(
                    "Boundary data component shape does not match the physical trace map."
                )
            physical_components = physical_values.reshape(
                (1,) * mdim + physical_values.shape
            )
            weight = weights.reshape((1,) * (mdim + 1) + weights.shape)
            weighted = transformed * physical_components * weight
            value = np.sum(weighted, axis=tuple(range(target_axis, weighted.ndim)))
        result.append(np.asarray(value).reshape(-1))
    return np.concatenate(result).astype(np.double, copy=False)


def _basis_change(left_space, right_space, axis_map: Sequence[int]) -> np.ndarray:
    """Compute the coefficient map between two related trace test spaces.

    Parameters
    ----------
    left_space, right_space : function-space-like
        Component test spaces on the left and right periodic objects.
    axis_map : sequence of int
        Signed permutation mapping left reference axes to right reference axes.

    Returns
    -------
    ndarray
        Matrix of shape ``(right_dof_count, left_dof_count)``. Multiplying
        right-space evaluations by this matrix reproduces the mapped
        left-space evaluations.

    Raises
    ------
    ValueError
        If the two spaces are not related by the supplied axis map to the
        numerical tolerance used by the least-squares fit.
    """
    mdim = len(axis_map)
    if mdim == 0:
        return np.ones((1, 1), dtype=np.double)
    left_orders = tuple(int(order) for order in left_space.orders)
    right_orders = tuple(int(order) for order in right_space.orders)
    left_count = int(np.prod(np.asarray(left_orders) + 1, dtype=np.intp))
    right_count = int(np.prod(np.asarray(right_orders) + 1, dtype=np.intp))
    nodes = tuple(np.linspace(-1.0, 1.0, max(order, 0) + 2) for order in left_orders)
    left_grid = np.meshgrid(*nodes, indexing="ij")
    right_coordinates: list[np.ndarray | None] = [None] * mdim
    for left_axis, mapped_axis in enumerate(axis_map):
        right_coordinates[abs(mapped_axis) - 1] = (
            1.0 if mapped_axis > 0 else -1.0
        ) * left_grid[left_axis]
    left_values = np.asarray(left_space.evaluate(*left_grid)).reshape(-1, left_count)
    right_values = np.asarray(
        right_space.evaluate(*(coordinate for coordinate in right_coordinates))
    ).reshape(-1, right_count)
    coefficients, *_ = np.linalg.lstsq(right_values, left_values, rcond=None)
    residual = np.max(np.abs(right_values @ coefficients - left_values), initial=0.0)
    scale = max(float(np.max(np.abs(left_values), initial=0.0)), 1.0)
    if residual > 1.0e-9 * scale:
        raise ValueError(
            "Periodic boundary test spaces are not related by the supplied axis map."
        )
    return coefficients


def _append_boundary_rows(
    mesh: Mesh,
    maps: Sequence[SpaceMap],
    element_specs: Sequence[KFormSpecs],
    test_specs: Sequence[Sequence[Sequence[KFormSpecs]]],
    sources: Mapping[tuple[int, int], Sequence[BoundaryData]],
    records: Mapping[
        tuple[int, int], tuple[npt.NDArray[np.uint64], npt.NDArray[np.int8]]
    ],
    rows: list[list[tuple[int, int, int, float]]],
    rhs: list[float],
) -> None:
    """Append prescribed-boundary rows for selected objects.

    Parameters
    ----------
    mesh : Mesh
        Mesh defining boundary objects and incident-element records.
    maps : sequence of SpaceMap
        Full element maps indexed by global element ID.
    element_specs : sequence of KFormSpecs
        Volume trial specifications indexed by global element ID.
    test_specs : nested sequence of KFormSpecs
        Test spaces indexed as ``test_specs[mdim][object_id][component]``.
    sources : mapping
        Boundary data keyed by ``(object_dimension, object_id)``.
    records : mapping
        Boundary incident-element IDs and orientations keyed like ``sources``.
    rows, rhs : list
        Mutable output lists receiving packed-row entries and right-hand-side
        values. Existing shared and periodic rows are preserved.

    Notes
    -----
    Rows are imposed on the lowest-ID incident element. If several selected
    outer faces reach one object, their prescribed moments must agree.
    """
    ndim = mesh.ndim
    for mdim, object_id, _, _ in mesh.iterate_boundary_all():
        key = (mdim, int(object_id))
        object_sources = sources.get(key)
        object_tests = test_specs[mdim][int(object_id)]
        if not object_sources:
            continue
        if not object_tests:
            continue
        element_ids, orientations = records[key]
        element_id = int(element_ids[0])
        boundary_map = _restrict_map(maps[element_id], orientations[0], ndim, mdim)
        for component, test_spec in enumerate(object_tests):
            local_result = mesh.compute_kform_boundary_constraints(
                test_spec,
                element_specs[element_id],
                maps[element_id],
                element_id,
                int(object_id),
            )
            local_rows = _component_rows(
                local_result, test_spec, component, element_id, +1.0
            )
            candidates = [
                _boundary_dual_values(data, test_spec, boundary_map, ndim, mdim)
                for data in object_sources
            ]
            component_counts = np.asarray(test_spec.component_dof_counts)
            start = int(np.sum(component_counts[:component]))
            end = start + int(component_counts[component])
            values = candidates[0][start:end]
            if any(
                not np.allclose(values, candidate[start:end], rtol=1.0e-10, atol=1.0e-11)
                for candidate in candidates[1:]
            ):
                raise ValueError(
                    f"Boundary data disagree on the shared boundary object {object_id}."
                )
            if len(local_rows) != values.size:
                raise RuntimeError("Boundary trace rows and data have different sizes.")
            rows.extend(local_rows)
            rhs.extend(float(value) for value in values)


def _append_periodic_rows(
    mesh: Mesh,
    maps: Sequence[SpaceMap],
    element_specs: Sequence[KFormSpecs],
    test_specs: Sequence[Sequence[Sequence[KFormSpecs]]],
    relations: Mapping[tuple[int, int, int], tuple[int, ...]],
    records: Mapping[
        tuple[int, int], tuple[npt.NDArray[np.uint64], npt.NDArray[np.int8]]
    ],
    rows: list[list[tuple[int, int, int, float]]],
    rhs: list[float],
) -> None:
    """Append acyclic periodic rows for related boundary objects.

    Parameters
    ----------
    mesh : Mesh
        Mesh containing the related boundary objects.
    maps : sequence of SpaceMap
        Full element maps indexed by global element ID.
    element_specs : sequence of KFormSpecs
        Volume trial specifications indexed by global element ID.
    test_specs : nested sequence of KFormSpecs
        Test spaces indexed as ``test_specs[mdim][object_id][component]``.
    relations : mapping
        Signed-axis relations keyed by ``(mdim, left_id, right_id)``.
    records : mapping
        Boundary incident-element IDs and orientations keyed by object.
    rows, rhs : list
        Mutable output lists receiving periodic rows and zero right-hand sides.

    Notes
    -----
    Local traces are batched when they share test and element specifications.
    Component pullback signs and basis changes are then applied to the right
    side of each periodic equation.
    """
    order = element_specs[0].order
    relation_items = sorted(
        relations.items(), key=lambda item: (-item[0][0], item[0][1], item[0][2])
    )
    requests: dict[tuple[int, int, int], tuple[KFormSpecs, int, int]] = {}
    validated_relations = []
    for (mdim, left_id, right_id), axis_map in relation_items:
        left_tests = test_specs[mdim][left_id]
        right_tests = test_specs[mdim][right_id]
        component_count = comb(mdim, order)
        if not left_tests and not right_tests:
            continue
        if len(left_tests) != component_count or len(right_tests) != component_count:
            raise ValueError(
                "Periodic boundary objects must provide matching test specifications."
            )
        left_elements, _ = records[(mdim, left_id)]
        right_elements, _ = records[(mdim, right_id)]
        left_element = int(left_elements[0])
        right_element = int(right_elements[0])
        validated_relations.append(
            (
                mdim,
                left_id,
                right_id,
                axis_map,
                left_tests,
                right_tests,
                left_element,
                right_element,
            )
        )
        for left_component, left_test_spec in enumerate(left_tests):
            right_component, _ = _component_relation(
                mdim, order, left_component, axis_map
            )
            right_test_spec = right_tests[right_component]
            requests.setdefault(
                (id(left_test_spec), left_element, left_id),
                (left_test_spec, left_element, left_id),
            )
            requests.setdefault(
                (id(right_test_spec), right_element, right_id),
                (right_test_spec, right_element, right_id),
            )

    batches: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for key, (_, element_id, _) in requests.items():
        test_spec = requests[key][0]
        batches.setdefault((id(test_spec), id(element_specs[element_id])), []).append(key)

    local_results: dict[tuple[int, int, int], tuple[np.ndarray, ...]] = {}
    for keys in batches.values():
        test_spec, _, _ = requests[keys[0]]
        element_spec = element_specs[requests[keys[0]][1]]
        batch = mesh.compute_kform_boundary_constraints_batch(
            test_spec,
            element_spec,
            [maps[requests[key][1]] for key in keys],
            [requests[key][1] for key in keys],
            [requests[key][2] for key in keys],
        )
        batch_offsets, _, batch_components, batch_dofs, batch_coefficients = batch
        rows_per_item = int(np.sum(test_spec.component_dof_counts))
        for index, key in enumerate(keys):
            row_start = index * rows_per_item
            row_end = row_start + rows_per_item
            entry_start = int(batch_offsets[row_start])
            entry_end = int(batch_offsets[row_end])
            local_results[key] = (
                np.asarray(
                    batch_offsets[row_start : row_end + 1] - entry_start,
                    dtype=np.uintp,
                ),
                batch_components[entry_start:entry_end],
                batch_dofs[entry_start:entry_end],
                batch_coefficients[entry_start:entry_end],
            )

    for (
        mdim,
        left_id,
        right_id,
        axis_map,
        left_tests,
        right_tests,
        left_element,
        right_element,
    ) in validated_relations:
        for left_component, left_test_spec in enumerate(left_tests):
            right_component, form_sign = _component_relation(
                mdim, order, left_component, axis_map
            )
            right_test_spec = right_tests[right_component]
            left_result = local_results[(id(left_test_spec), left_element, left_id)]
            right_result = local_results[(id(right_test_spec), right_element, right_id)]
            left_rows = _component_rows(
                left_result, left_test_spec, left_component, left_element, +1.0
            )
            right_rows = _component_rows(
                right_result, right_test_spec, right_component, right_element, +1.0
            )
            left_space = left_test_spec.get_component_function_space(left_component)
            right_space = right_test_spec.get_component_function_space(right_component)
            basis_change = _basis_change(left_space, right_space, axis_map)
            if basis_change.shape != (len(right_rows), len(left_rows)):
                raise ValueError(
                    "Periodic boundary test spaces have incompatible row counts."
                )
            for left_row, left_entries in enumerate(left_rows):
                entries = list(left_entries)
                for right_row, right_entries in enumerate(right_rows):
                    factor = -form_sign * float(basis_change[right_row, left_row])
                    if factor == 0.0:
                        continue
                    entries.extend(
                        (element, component, dof, factor * coefficient)
                        for element, component, dof, coefficient in right_entries
                    )
                rows.append(entries)
                rhs.append(0.0)


def _compute_kform_global_constraints(
    mesh: Mesh,
    element_specs: Sequence[KFormSpecs],
    element_maps: Sequence[SpaceMap],
    test_specs: Sequence[Sequence[Sequence[KFormSpecs]]],
    boundary_conditions: Mapping[int, BoundaryData]
    | Sequence[BoundaryCondition]
    | None = None,
    periodic_pairs: Sequence[BoundaryPair | BoundaryPairGroup] | None = None,
) -> tuple[PackedRows, npt.NDArray[np.double]]:
    """Assemble shared, prescribed, and periodic global trace constraints.

    Parameters
    ----------
    mesh : Mesh
        Conforming hypercube mesh supplying topology and boundary orientation.
    element_specs : sequence of KFormSpecs
        One volume specification per mesh element.
    element_maps : sequence of SpaceMap
        One physical element map per mesh element.
    test_specs : nested sequence of KFormSpecs
        Explicit trace test spaces indexed as
        ``test_specs[mdim][object_id][component]``.
    boundary_conditions : mapping, sequence, or None, optional
        Prescribed physical data on outer faces. Mapping keys are face IDs;
        record sequences use ``BoundaryCondition`` objects.
    periodic_pairs : sequence of BoundaryPair or BoundaryPairGroup, optional
        Explicit outer-face identifications and signed axis maps.

    Returns
    -------
    tuple
        ``(packed_rows, rhs)``. ``packed_rows`` is the five-array global row
        representation, and ``rhs`` contains one value per row.

    Notes
    -----
    Shared-object continuity is assembled first. Prescribed and periodic rows
    are appended afterward, with periodic relations reduced to an acyclic
    spanning forest.
    """
    shared = mesh.compute_kform_continuity_constraints(
        element_specs, element_maps, test_specs
    )
    rows = _unpack(shared)
    rhs = [0.0] * len(rows)
    records = _boundary_records(mesh)
    ndim = mesh.ndim
    boundary_faces = {
        int(object_id) for _, object_id, _, _ in mesh.iterate_boundary(ndim - 1)
    }

    sources: dict[tuple[int, int], list[BoundaryData]] = {}
    for face, data in _normalize_boundary_conditions(boundary_conditions):
        if face not in boundary_faces:
            raise ValueError(f"Boundary face {face} is not an outer boundary face.")
        for key in _object_descendants(mesh, ndim - 1, face):
            sources.setdefault(key, []).append(data)

    relations: dict[tuple[int, int, int], tuple[int, ...]] = {}
    seen_face_pairs: set[tuple[int, int]] = set()
    seen_faces: set[int] = set()
    for pair in _normalize_periodic_pairs(periodic_pairs):
        left_face = int(pair.left_face)
        right_face = int(pair.right_face)
        if left_face not in boundary_faces or right_face not in boundary_faces:
            raise ValueError("Periodic pairs must contain outer boundary face IDs.")
        if left_face == right_face:
            raise ValueError("A periodic boundary face cannot be paired with itself.")
        if left_face in seen_faces or right_face in seen_faces:
            raise ValueError("A boundary face cannot appear in multiple periodic pairs.")
        face_pair = (left_face, right_face)
        if face_pair in seen_face_pairs or (right_face, left_face) in seen_face_pairs:
            raise ValueError("Periodic boundary face pairs must be unique.")
        seen_face_pairs.add(face_pair)
        seen_faces.update((left_face, right_face))
        for key, axis_map in _object_relations(mesh, pair).items():
            previous = relations.get(key)
            if previous is not None and previous != axis_map:
                raise ValueError(f"Periodic object pair {key[1:]} is inconsistent.")
            relations[key] = axis_map
    relations = _select_periodic_object_relations(relations)

    periodic_objects = {
        (mdim, object_id)
        for mdim, left, right in relations
        for object_id in (left, right)
    }
    overlap = set(sources).intersection(periodic_objects)
    if overlap:
        mdim, object_id = sorted(overlap)[0]
        raise ValueError(
            f"Boundary object {object_id} cannot have both prescribed and "
            "periodic constraints."
        )

    _append_boundary_rows(
        mesh, element_maps, element_specs, test_specs, sources, records, rows, rhs
    )
    _append_periodic_rows(
        mesh, element_maps, element_specs, test_specs, relations, records, rows, rhs
    )
    return _pack(rows, rhs)


def compute_kform_global_constraints(
    mesh: Mesh,
    element_specs: Sequence[KFormSpecs],
    element_maps: Sequence[SpaceMap],
    test_specs: Sequence[Sequence[Sequence[KFormSpecs]]],
    boundary_conditions: Mapping[int, BoundaryData]
    | Sequence[BoundaryCondition]
    | None = None,
    periodic_pairs: Sequence[BoundaryPair | BoundaryPairGroup] | None = None,
) -> tuple[PackedRows, npt.NDArray[np.double]]:
    """Assemble global shared, prescribed, and periodic trace rows.

    Parameters
    ----------
    mesh : Mesh
        Conforming hypercube mesh supplying topology and boundary orientation.
    element_specs : sequence of KFormSpecs
        One volume specification per mesh element.
    element_maps : sequence of SpaceMap
        One physical element map per mesh element.
    test_specs : nested sequence of KFormSpecs
        Explicit trace test spaces indexed as
        ``test_specs[mdim][object_id][component]``.
    boundary_conditions : mapping, sequence, or None, optional
        Prescribed physical k-form data on outer faces.
    periodic_pairs : sequence of BoundaryPair or BoundaryPairGroup, optional
        Explicit outer-face identifications with signed axis maps.

    Returns
    -------
    tuple
        ``(packed_rows, rhs)``: the five packed row arrays and one right-hand
        side value per row. Shared and periodic rows have zero right-hand side.
    """
    return _compute_kform_global_constraints(
        mesh,
        element_specs,
        element_maps,
        test_specs,
        boundary_conditions,
        periodic_pairs,
    )
