"""Block-structured hybridized solves on top of the hybsol block solver.

The trace constraints turn one small problem per element into the saddle-point
system :math:`[[A, N], [N^T, 0]]`. It goes to hybsol as a block system, so no
Schur complement is formed and the multipliers fall out of the same solve.

One block per element holds that element's operator. A constraint row can be
absorbed into it -- the row's equation takes a row of the block, its multiplier
a column -- which borders the block into :math:`[[A_e, C^T], [C, 0]]`. That is
what makes a singular operator usable: the 0-form stiffness ``D M D^T`` has the
constants in its nullspace and cannot lead an unpivoted factorization. Rows
nobody absorbs form blocks of their own, grouped per interface object, whose
diagonals hybsol fills once the element blocks are gone.

hybsol does not pivot, so every leading principal minor of every diagonal block
has to be nonzero. Rows and columns inside an element block share one schedule
that walks the degrees of freedom and, right after the last one a row touches
with a nonzero coefficient, places that row's equation and its multiplier.

What the caller owes: every row reaches a block, and the rows absorbed into one
element are independent on it. A singular operator needs at least one absorbed
row, and a 0-form problem needs a prescribed row per connected component.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from hybsol import BlockSystem, Precision

from fdg._fdg import (
    KFormSpecs,
    MeshGeometry,
    MeshKFormSpecs,
    SpaceMap,
    compute_kform_incidence_matrix,
    compute_kform_mass_matrix,
    incidence_kform_operator,
)
from fdg.boundary_conditions import PackedRows

__all__ = [
    "ElementBlockBuilder",
    "HybridizedSolution",
    "laplace_stiffness",
    "mixed_block",
    "solve_hybridized",
]


@dataclass(frozen=True, slots=True)
class HybridizedSolution:
    """Solution of one block-structured hybridized system.

    ``element_dofs`` holds the degrees of freedom of every element in element
    order and ``multipliers`` the multiplier of every constraint row in row
    order; ``constraint_residual`` is the largest deviation of the constraints
    from their right-hand side.
    """

    element_dofs: tuple[npt.NDArray[np.double], ...]
    multipliers: npt.NDArray[np.double]
    constraint_residual: float


class ElementBlockBuilder(Protocol):
    """Assemble the local system of one element.

    Returns the block in the field order of the specification structure. The
    constrained field has to come first, because the packed rows address it.
    """

    def __call__(
        self,
        element_id: int,
        field_specs: Sequence[KFormSpecs],
        element_map: SpaceMap,
    ) -> npt.NDArray[np.double]:
        """Return the ``(n, n)`` block of one element."""
        ...


def laplace_stiffness(
    element_id: int,  # noqa: ARG001 - part of the builder protocol
    field_specs: Sequence[KFormSpecs],
    element_map: SpaceMap,
) -> npt.NDArray[np.double]:
    """Assemble the stiffness matrix of a 0-form Laplace element.

    The operator is ``D^T M_1 D``, the one-form mass matrix pulled back
    through the incidence operator. It is **singular**: constants lie in its
    nullspace, so it needs absorbed constraint rows and a prescribed row per
    connected component to gauge the solution.

    Parameters
    ----------
    element_id : int
        Unused; the operator depends only on the specifications and the map.
    field_specs : sequence of KFormSpecs
        Field group of the element. Exactly one field is read.
    element_map : SpaceMap
        Space map supplying the geometry of the quadrature.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` element stiffness matrix.

    Raises
    ------
    ValueError
        If the field group does not hold exactly one field.
    """
    if len(field_specs) != 1:
        raise ValueError(
            f"laplace_stiffness needs exactly one field, got {len(field_specs)}."
        )
    space = field_specs[0].base_space
    mass = np.asarray(compute_kform_mass_matrix(element_map, 1, space, space))
    incidence = np.asarray(compute_kform_incidence_matrix(space, 0))
    return incidence.T @ mass @ incidence


def mixed_block(
    element_id: int,  # noqa: ARG001 - part of the builder protocol
    field_specs: Sequence[KFormSpecs],
    element_map: SpaceMap,
) -> npt.NDArray[np.double]:
    """Assemble the local system of a mixed Poisson element.

    The block is ``[[M_q, D^T], [D, 0]]`` with the trial degrees of freedom
    first, the order the constraint rows address. It is nonsingular for the
    top-degree pairs (``q = n - 1``, ``u = n``), so it needs no absorbed row.

    Parameters
    ----------
    element_id : int
        Unused; the block depends only on the specifications and the map.
    field_specs : sequence of KFormSpecs
        Field group of the element: the trial k-form first, the test k-form
        second.
    element_map : SpaceMap
        Space map supplying the geometry of the quadrature.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` element block, with the trial degrees of freedom first.

    Raises
    ------
    ValueError
        If the field group does not hold exactly two fields.
    """
    if len(field_specs) != 2:
        raise ValueError(
            f"mixed_block needs a trial and a test field, got {len(field_specs)}."
        )
    trial, test = field_specs
    q_mass = np.asarray(
        compute_kform_mass_matrix(
            element_map, trial.order, trial.base_space, trial.base_space
        )
    )
    u_mass = np.asarray(
        compute_kform_mass_matrix(
            element_map, test.order, test.base_space, test.base_space
        )
    )
    derivative = np.asarray(incidence_kform_operator(trial, u_mass, right=True))
    derivative_transpose = np.asarray(
        incidence_kform_operator(trial, u_mass, transpose=True)
    )
    nu = derivative.shape[0]
    nq = derivative.shape[1]
    block = np.zeros((nq + nu, nq + nu))
    block[:nq, :nq] = q_mass
    block[nq:, :nq] = derivative
    block[:nq, nq:] = derivative_transpose
    return block


class _Layout:
    """Which block holds which row, multiplier and degree of freedom."""

    def __init__(
        self,
        constraints: PackedRows,
        component_offsets: npt.NDArray[np.intp],
        element_sizes: npt.NDArray[np.intp],
    ) -> None:
        row_offsets, element_ids, components, local_dofs, coefficients = constraints
        self.row_count = int(row_offsets.size - 1)
        self.element_count = int(element_sizes.size)
        starts = row_offsets.astype(np.intp)
        local = component_offsets[components.astype(np.intp)] + local_dofs.astype(np.intp)
        self.entries: list[list[tuple[int, int, float]]] = [
            list(
                zip(
                    element_ids[s:e].tolist(),
                    local[s:e].tolist(),
                    coefficients[s:e].tolist(),
                )
            )
            for s, e in zip(starts[:-1], starts[1:])
        ]

        self.element_sizes = element_sizes
        self.owner = self._assign_owners()
        self.top = np.array(
            [
                max(
                    (
                        index
                        for element, index, coefficient in entries
                        if element == owner and coefficient != 0.0
                    ),
                    default=-1,
                )
                for entries, owner in zip(self.entries, self.owner)
            ],
            dtype=np.intp,
        )
        absorbed_mask = self.owner >= 0
        if absorbed_mask.any() and int(self.top[absorbed_mask].min()) < 0:
            raise ValueError("a constraint row does not touch its owner element.")

        self.absorbed: list[list[int]] = [
            [row for row in range(self.row_count) if int(self.owner[row]) == element]
            for element in range(self.element_count)
        ]
        self.groups = self._group_unowned()

        self.block_sizes: list[int] = []
        self.block_rows: list[list[int]] = []
        self.dof_at: list[npt.NDArray[np.intp]] = []
        self.equation_at = np.full(self.row_count, -1, dtype=np.intp)
        self.multiplier_at = np.full(self.row_count, -1, dtype=np.intp)
        for element in range(self.element_count):
            size = int(element_sizes[element])
            mine = sorted(self.absorbed[element], key=lambda row: int(self.top[row]))
            positions = np.empty(size, dtype=np.intp)
            cursor = 0
            for dof in range(size):
                positions[dof] = cursor
                cursor += 1
                for row in mine:
                    if int(self.top[row]) == dof:
                        self.equation_at[row] = cursor
                        self.multiplier_at[row] = cursor
                        cursor += 1
            self.block_sizes.append(cursor)
            self.block_rows.append(mine)
            self.dof_at.append(positions)
        for rows in self.groups:
            self.block_sizes.append(len(rows))
            self.block_rows.append(rows)
            self.dof_at.append(np.empty(0, dtype=np.intp))
        self.block_offsets = np.concatenate(
            ([0], np.cumsum(self.block_sizes)), dtype=np.intp
        )
        self.total = int(self.block_offsets[-1])

    def _assign_owners(self) -> npt.NDArray[np.intp]:
        """Absorb rows into the elements that can still use them.

        Every element gets one row, then a row is absorbed only where it grows
        the rank of the rows already there, which keeps them independent on
        that element. The rest keep blocks of their own.
        """
        rows_of_element: list[list[int]] = [[] for _ in range(self.element_count)]
        bases: list[list[np.ndarray]] = [[] for _ in range(self.element_count)]
        owner = np.full(self.row_count, -1, dtype=np.intp)

        def elements_of(row: int) -> list[int]:
            """Return the elements the row touches with a nonzero coefficient."""
            return sorted(
                {
                    element
                    for element, _, coefficient in self.entries[row]
                    if coefficient != 0.0
                }
            )

        def restriction(row: int, element: int) -> np.ndarray:
            """Return the row as a vector over the element's degrees of freedom."""
            size = self.element_sizes[element]
            vector = np.zeros(size)
            for other, index, coefficient in self.entries[row]:
                if other == element:
                    vector[index] += coefficient
            return vector

        def grows_rank(row: int, element: int) -> bool:
            """Whether the row is independent of the ones already absorbed."""
            residual = restriction(row, element)
            scale = float(np.linalg.norm(residual))
            if scale == 0.0:
                return False
            for vector in bases[element]:
                residual -= float(residual @ vector) * vector
            return float(np.linalg.norm(residual)) > 1.0e-10 * scale

        def absorb(row: int, element: int) -> None:
            """Give the row to the element and extend its orthonormal basis."""
            residual = restriction(row, element)
            for vector in bases[element]:
                residual -= float(residual @ vector) * vector
            norm = float(np.linalg.norm(residual))
            if norm > 0.0:
                bases[element].append(residual / norm)
            owner[row] = element
            rows_of_element[element].append(row)

        # A singular operator cannot lead the factorization until something
        # borders it, so every element gets a row first. The first row always
        # grows the rank, so coverage is unconditional.
        for element in range(self.element_count):
            if rows_of_element[element]:
                continue
            spare = [
                row
                for row in range(self.row_count)
                if owner[row] < 0 and element in elements_of(row)
            ]
            if spare:
                absorb(spare[0], element)

        # Then absorb a row only where it still grows the rank. One that grows
        # nothing anywhere keeps a block of its own: a dependent element block
        # would meet a zero pivot.
        for row in range(self.row_count):
            if owner[row] >= 0:
                continue
            pool = [element for element in elements_of(row) if grows_rank(row, element)]
            if pool:
                absorb(row, min(pool, key=lambda e: (len(rows_of_element[e]), e)))
        return owner

    def _group_unowned(self) -> list[list[int]]:
        """Group the rows nobody absorbed per interface object."""
        free = [row for row in range(self.row_count) if int(self.owner[row]) < 0]
        groups: list[list[int]] = []
        members: list[set[int]] = []
        for row in free:
            touches = {element for element, _, coefficient in self.entries[row]}
            for index, seen in enumerate(members):
                if touches & seen:
                    groups[index].append(row)
                    members[index] |= touches
                    break
            else:
                groups.append([row])
                members.append(set(touches))
        return groups

    def block_of_row(self, row: int) -> int:
        """Block index holding the equation and the multiplier of a row."""
        if int(self.owner[row]) >= 0:
            return int(self.owner[row])
        return self.element_count + next(
            index for index, rows in enumerate(self.groups) if row in rows
        )

    def position_of_row(self, row: int) -> int:
        """Slot of a row's equation and multiplier inside its block."""
        block = self.block_of_row(row)
        if block < self.element_count:
            return int(self.equation_at[row])
        return self.groups[block - self.element_count].index(row)

    def system(
        self,
        blocks: Sequence[npt.NDArray[np.double]],
        precision: Precision | str,
    ) -> BlockSystem:
        """Assemble the hybsol block system of the augmented operator."""
        data: dict[tuple[int, int], npt.NDArray[np.double]] = {}

        def put(key: tuple[int, int], i: int, j: int, value: float) -> None:
            """Accumulate one entry into a block, creating it on first use."""
            if key not in data:
                data[key] = np.zeros((self.block_sizes[key[0]], self.block_sizes[key[1]]))
            data[key][i, j] += value

        for element in range(self.element_count):
            positions = self.dof_at[element]
            values = np.zeros((self.block_sizes[element],) * 2)
            values[np.ix_(positions, positions)] = blocks[element]
            data[element, element] = values
            for row in self.absorbed[element]:
                equation = int(self.equation_at[row])
                multiplier = int(self.multiplier_at[row])
                for other, local, coefficient in self.entries[row]:
                    if other != element:
                        continue
                    column = int(positions[local])
                    put((element, element), column, multiplier, coefficient)
                    put((element, element), equation, column, coefficient)

        for row in range(self.row_count):
            block = self.block_of_row(row)
            slot = self.position_of_row(row)
            for element, local, coefficient in self.entries[row]:
                column = int(self.dof_at[element][local])
                if block == element:
                    continue
                put((element, block), column, slot, coefficient)
                put((block, element), slot, column, coefficient)

        # A group couples to element blocks but has no diagonal of its own,
        # and hybsol wants every block row to have one.
        for block in range(len(self.block_sizes)):
            data.setdefault(
                (block, block),
                np.zeros((self.block_sizes[block], self.block_sizes[block])),
            )

        pattern: dict[tuple[int, int], npt.NDArray[np.double]] = {}
        for key, values in data.items():
            if key in pattern:
                continue
            row_block, col_block = key
            pattern[key] = values
            if row_block != col_block:
                pattern[col_block, row_block] = values.T
        rows, cols, payload = [], [], []
        for (row_block, col_block), values in sorted(pattern.items()):
            rows.append(np.array([row_block], dtype=np.intp))
            cols.append(np.array([col_block], dtype=np.intp))
            payload.append(np.ascontiguousarray(values).ravel())
        return BlockSystem.from_blocks(
            self.block_sizes,
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(payload),
            precision=precision,
        )

    def rhs(
        self,
        element_rhs: npt.NDArray[np.double],
        element_offsets: npt.NDArray[np.intp],
        constraint_rhs: npt.NDArray[np.double],
    ) -> npt.NDArray[np.double]:
        """Scatter the caller's right-hand side into the block layout."""
        rhs = np.zeros(self.total)
        for element in range(self.element_count):
            base = int(self.block_offsets[element])
            positions = self.dof_at[element]
            start = int(element_offsets[element])
            rhs[base + positions] = element_rhs[start : start + positions.size]
        for row in range(self.row_count):
            block = int(self.block_offsets[self.block_of_row(row)])
            rhs[block + self.position_of_row(row)] = constraint_rhs[row]
        return rhs

    def unpack(
        self, solution: npt.NDArray[np.double]
    ) -> tuple[list[npt.NDArray[np.double]], npt.NDArray[np.double]]:
        """Recover element degrees of freedom and multipliers from a solution."""
        element_dofs = []
        for element in range(self.element_count):
            base = int(self.block_offsets[element])
            positions = self.dof_at[element]
            element_dofs.append(np.array(solution[base + positions]))
        multipliers = np.zeros(self.row_count)
        for row in range(self.row_count):
            block = int(self.block_offsets[self.block_of_row(row)])
            multipliers[row] = solution[block + self.position_of_row(row)]
        return element_dofs, multipliers


def solve_hybridized(
    geometry: MeshGeometry,
    specs: MeshKFormSpecs,
    element_rhs: npt.NDArray[np.double],
    constraints: PackedRows,
    constraint_rhs: npt.NDArray[np.double],
    builder: ElementBlockBuilder,
    precision: Precision | str = Precision.DOUBLE,
    n_threads: int = 0,
) -> HybridizedSolution:
    """Solve the augmented block system of a mesh with trace constraints.

    Parameters
    ----------
    geometry : MeshGeometry
        Batched geometry, one space map per element.
    specs : MeshKFormSpecs
        Batched k-form specifications, one field group per element. The field
        order of a group is the row and column order of its element block.
    element_rhs : numpy.ndarray
        Right-hand side of every element, concatenated in element order and
        matching the block sizes the builder returns.
    constraints : PackedRows
        Packed constraint rows: row offsets, element ids, element-frame
        components, local degrees of freedom and coefficients.
    constraint_rhs : numpy.ndarray
        One right-hand-side value per constraint row.
    builder : ElementBlockBuilder
        Called once per element to assemble its block.
    precision : hybsol.Precision or str, default: "double"
        Floating-point type the system stores its blocks in.
    n_threads : int, default: 0
        OpenMP threads used by the decomposition; ``0`` selects the OpenMP
        default and ``1`` runs it serially.

    Returns
    -------
    HybridizedSolution
        Element degrees of freedom, multipliers and the constraint residual.

    Raises
    ------
    ValueError
        If the stores disagree on the element count, the packed rows are
        malformed, a right-hand side has the wrong length, or the builder
        returns a block that is not a square matrix.
    RuntimeError
        If the solution does not satisfy the constraints, which means an
        element block is singular -- a 0-form operator needs at least one
        absorbed row, and the problem needs a prescribed row to gauge it.
    """
    element_count = geometry.element_count
    if specs.element_count != element_count:
        raise ValueError(
            f"specs describe {specs.element_count} elements, but geometry holds "
            f"{element_count}."
        )

    labels = specs.labels
    blocks = []
    for element in range(element_count):
        field_specs = [specs.field_specs(element, label) for label in labels]
        block = np.asarray(builder(element, field_specs, geometry.space_map(element)))
        if block.ndim != 2 or block.shape[0] != block.shape[1]:
            raise ValueError(
                f"builder returned a {block.ndim}-D or non-square block for element "
                f"{element}."
            )
        blocks.append(block)
    element_sizes = np.array([block.shape[0] for block in blocks], dtype=np.intp)
    element_offsets = np.concatenate(([0], np.cumsum(element_sizes))).astype(np.intp)
    total_dofs = int(element_offsets[-1])

    trial = specs.field_specs(0, labels[0])
    component_offsets = np.array(
        [
            int(trial.get_component_slice(component).start)
            for component in range(trial.component_count)
        ],
        dtype=np.intp,
    )
    layout = _Layout(constraints, component_offsets, element_sizes)

    element_rhs = np.asarray(element_rhs, dtype=np.double)
    constraint_rhs = np.asarray(constraint_rhs, dtype=np.double)
    if element_rhs.size != total_dofs:
        raise ValueError(
            f"element right-hand side has {element_rhs.size} entries, expected "
            f"{total_dofs}."
        )
    if constraint_rhs.size != layout.row_count:
        raise ValueError(
            f"constraint right-hand side has {constraint_rhs.size} entries, expected "
            f"{layout.row_count}."
        )

    system = layout.system(blocks, precision)
    rhs = layout.rhs(element_rhs, element_offsets, constraint_rhs)
    decomposition = system.decompose(n_threads=n_threads)
    element_dofs, multipliers = layout.unpack(
        decomposition.solve(rhs, n_threads=n_threads)
    )

    residual = 0.0
    for row, entries in enumerate(layout.entries):
        value = -constraint_rhs[row]
        for element, local, coefficient in entries:
            value += coefficient * element_dofs[element][local]
        residual = max(residual, abs(value))
    tolerance = 1.0e-6 if str(precision) == str(Precision.SINGLE) else 1.0e-9
    if residual > tolerance:
        raise RuntimeError(
            f"the solution does not satisfy the constraints (residual {residual:.3e}); "
            "an element block is probably singular."
        )
    return HybridizedSolution(
        element_dofs=tuple(element_dofs),
        multipliers=multipliers,
        constraint_residual=residual,
    )
