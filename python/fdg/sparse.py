"""Batched scipy assembly of the direct continuity transfer.

:func:`scatter_csc` drives the C core's ``_scatter_triplets`` over element
batches and sums each batch's COO triplets into one CSC matrix. Batching in
Python keeps the C binding free of SciPy while bounding the working set to
one batch's triplets plus the stored result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array, csc_array

if TYPE_CHECKING:
    from fdg._fdg import DirectDofMap

#: Lossy scipy upstream stubs: every sparse type spells ``Any``.
SparseMatrix = Any

#: Triplets one batch emits; roughly 100 MB of rows, columns, and values.
BATCH_TRIPLETS = 1 << 22


def scatter_csc(
    transfer: DirectDofMap,
    local_matrices: NDArray[np.float64],
    n_threads: int = 0,
) -> SparseMatrix:
    """Scatter row-major per-element local matrices into the global CSC matrix.

    Element ``e`` contributes ``value_r * m_i_j * value_c`` over every
    entry pair ``r``, ``c`` of every local degree-of-freedom pair ``i``,
    ``j``. Explicit zeros are kept and the COO-to-CSC conversion sums
    the duplicates. ``local_matrices`` is one flat C-contiguous float64
    array holding one row-major ``n_e x n_e`` block per element in
    element order. Elements are processed in batches of :data:`BATCH_TRIPLETS`
    triplets each, so the working set stays bounded by one batch plus
    the stored result; an element whose own triplets exceed the batch
    target forms an oversized batch, which makes its per-element
    triplets the memory floor. ``n_threads`` picks the worker count;
    ``0`` uses the OpenMP default, and the result is identical for any
    thread count. Returns the assembled ``scipy.sparse.csc_array`` of
    shape ``(global_dof_count, global_dof_count)``.
    """
    element_offsets = np.asarray(transfer.element_offsets, dtype=np.int64)
    # Per-element triplet count: one per entry pair of every local DoF pair.
    counts = (
        np.diff(np.asarray(transfer.entry_offsets, dtype=np.int64)[element_offsets]) ** 2
    )

    result: SparseMatrix | None = None
    first = 0
    while first < counts.size:
        # Greedy fill; a single element past the target forms its own batch.
        batch_triplets = int(counts[first])
        stop = first + 1
        while stop < counts.size and batch_triplets + int(counts[stop]) <= BATCH_TRIPLETS:
            batch_triplets += int(counts[stop])
            stop += 1
        rows, cols, values = transfer._scatter_triplets(
            local_matrices, int(first), int(stop), n_threads=n_threads
        )
        first = stop
        if rows.size == 0:
            continue
        batch = coo_array(
            (values, (rows, cols)),
            shape=(transfer.global_dof_count, transfer.global_dof_count),
        ).tocsc()
        result = batch if result is None else result + batch
    if result is None:
        return csc_array((transfer.global_dof_count, transfer.global_dof_count))
    return result
