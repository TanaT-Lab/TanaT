#!/usr/bin/env python3
"""
Numba kernels for LinearPairwiseSequenceMetric.

All functions are @njit (no Python objects). They operate on int32-encoded
feature arrays and float32 distance buffers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit, prange
from numba import float32 as nb_float32

# ---------------------------------------------------------------------------
# Aggregation kernels
# ---------------------------------------------------------------------------


@njit
def _numba_mean(values, n):
    """Mean of the first *n* values.

    Args:
        values: float32 buffer (length >= n).
        n:      Number of valid entries to consider.

    Returns:
        Mean as float32.  Returns 0.0 when n == 0.
    """
    if n == 0:
        return nb_float32(0.0)
    s = nb_float32(0.0)
    for i in range(n):
        s += values[i]
    return s / nb_float32(n)


@njit
def _numba_sum(values, n):
    """Sum of the first *n* values.

    Args:
        values: float32 buffer (length >= n).
        n:      Number of valid entries to consider.

    Returns:
        Sum as float32.
    """
    s = nb_float32(0.0)
    for i in range(n):
        s += values[i]
    return s


# ---------------------------------------------------------------------------
# Pairwise kernels
# ---------------------------------------------------------------------------


@njit
def compute_single_pair(
    arr_a,
    arr_b,
    len_a,
    len_b,
    dist_kernel,
    context,
    aggregator,
    padding_penalty,
):
    """Compute distance for a single pair of int32-encoded sequences.

    Empty-sequence rules (mirrors the Python path):

    1. Both empty → ``nan`` (undefined distance).
    2. One empty + ``padding_penalty`` is ``nan`` → ``nan`` (undefined).
    3. One empty + ``padding_penalty`` is set → all positions are padded.

    Normal path:

    4. Iterate over ``min(len_a, len_b)`` aligned positions and call
       ``dist_kernel`` for each pair.
    5. If ``padding_penalty`` is not ``nan`` and lengths differ, append
       ``padding_penalty`` for each extra position of the longer sequence.
    6. Aggregate the distance buffer with ``aggregator``.

    Args:
        arr_a:           int32 array for the first sequence.
        arr_b:           int32 array for the second sequence.
        len_a:           Length of the first sequence.
        len_b:           Length of the second sequence.
        dist_kernel:     Numba-compiled entity-level distance function.
        context:         Opaque tuple forwarded to ``dist_kernel``.
        aggregator:      Numba-compiled aggregation function
                         ``(values, n) -> float32``.
        padding_penalty: ``float32`` penalty for unmatched positions.
                         Use ``np.nan`` to disable padding (undefined result
                         when lengths differ).

    Returns:
        Aggregated float32 distance, or ``nan`` for undefined pairs.
    """
    # --- Early exits for undefined pairs ---
    if len_a == 0 and len_b == 0:
        return nb_float32(np.nan)
    if (len_a == 0 or len_b == 0) and np.isnan(padding_penalty):
        return nb_float32(np.nan)

    min_len = min(len_a, len_b)
    max_len = max(len_a, len_b)

    # Allocate a float32 buffer for all distances (overlap + padding)
    values = np.empty(max_len, dtype=np.float32)
    count = 0

    # Aligned positions
    for i in range(min_len):
        values[count] = dist_kernel(arr_a[i], arr_b[i], context)
        count += 1

    # Padding for the extra positions of the longer sequence
    if max_len > min_len and not np.isnan(padding_penalty):
        for _ in range(max_len - min_len):
            values[count] = padding_penalty
            count += 1

    return aggregator(values, count)


@njit(parallel=True)
def compute_pairwise_chunk(
    result,
    start,
    end,
    arrays,
    lengths,
    dist_kernel,
    context,
    aggregator,
    padding_penalty,
):
    """Compute a chunk of rows ``[start, end)`` of the distance matrix.

    Only fills ``result[i, j]`` and ``result[j, i]`` for ``i`` in
    ``[start, end)`` and ``j`` in ``[i+1, n)``. Uses ``prange`` on the
    chunk rows.  The diagonal is **not** set by this kernel.

    Args:
        result:          The full (n × n) memmap/array. Only rows
                         ``[start:end]`` are written.
        start:           First row index of the chunk (inclusive).
        end:             Last row index of the chunk (exclusive).
        arrays:          Per-sequence int32 encoded arrays.
        lengths:         Per-sequence lengths.
        dist_kernel:     Entity distance kernel.
        context:         Opaque context for the kernel.
        aggregator:      Aggregation kernel.
        padding_penalty: Padding value (NaN = no padding).
    """
    n = len(lengths)
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        for j in range(i + 1, n):
            d = compute_single_pair(
                arrays[i],
                arrays[j],
                lengths[i],
                lengths[j],
                dist_kernel,
                context,
                aggregator,
                padding_penalty,
            )
            result[i, j] = d
            result[j, i] = d


@njit(parallel=True)
def compute_pairwise_matrix(
    result,
    arrays,
    lengths,
    dist_kernel,
    context,
    aggregator,
    padding_penalty,
):
    """Fill the upper triangle of *result* using ``prange`` for parallelism.

    Uses ``prange`` on the outer loop so Numba can parallelise row-wise.
    Each (i, j) pair is written to both ``result[i, j]`` and
    ``result[j, i]`` (symmetry).  The diagonal is left at ``0.0``
    (zeroed by the caller).

    Args:
        result:          Pre-allocated float32 (n × n) array.
        arrays:          ``numba.typed.List`` of int32 arrays, one per
                         sequence (ordered by ``pool.unique_ids``).
        lengths:         int32 array of sequence lengths.
        dist_kernel:     Numba-compiled entity-level distance function.
        context:         Opaque tuple forwarded to ``dist_kernel``.
        aggregator:      Numba-compiled aggregation function.
        padding_penalty: float32 padding value (``nan`` = no padding).
    """
    n = len(lengths)
    for i in prange(n):  # pylint: disable=not-an-iterable
        for j in range(i + 1, n):
            d = compute_single_pair(
                arrays[i],
                arrays[j],
                lengths[i],
                lengths[j],
                dist_kernel,
                context,
                aggregator,
                padding_penalty,
            )
            result[i, j] = d
            result[j, i] = d


# ---------------------------------------------------------------------------
# Aggregation kernel registry
# ---------------------------------------------------------------------------

_AGG_NUMBA_KERNELS: dict[str, Callable] = {
    "mean": _numba_mean,
    "sum": _numba_sum,
}
