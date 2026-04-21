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
def compute_pairwise_matrix(
    result,
    start,
    end,
    arrays_a,
    lengths_a,
    arrays_b,
    lengths_b,
    dist_kernel,
    context,
    aggregator,
    padding_penalty,
    symmetric,
):
    """Unified chunk kernel for memmap paths.

    Processes rows ``[start, end)`` of *result*.  Behaviour mirrors
    :func:`compute_pairwise_matrix`: when ``symmetric`` is ``True``, only
    the upper triangle of the chunk is computed and values are mirrored;
    when ``False``, every cell in the chunk rows is computed.

    The diagonal is set by this kernel when ``symmetric`` is ``True``.

    Args:
        result:          The full (n × n) or (chunk × k) memmap/array.
        start:           First row index of the chunk (inclusive).
        end:             Last row index of the chunk (exclusive).
        arrays_a:        Encoded sequences for the row pool.
        lengths_a:       Sequence lengths for the row pool.
        arrays_b:        Encoded sequences for the column pool.
        lengths_b:       Sequence lengths for the column pool.
        dist_kernel:     Entity distance kernel.
        context:         Opaque context for the kernel.
        aggregator:      Aggregation kernel.
        padding_penalty: Padding value (NaN = no padding).
        symmetric:       When ``True``, exploit the upper-triangle + mirror
                         optimisation (same-pool square matrix only).
                         When ``False``, compute every cell in the row range
                         Required for rectangular cross-pool chunks.
    """
    k = len(lengths_b)
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        if symmetric:
            result[i, i] = compute_single_pair(
                arrays_a[i],
                arrays_b[i],
                lengths_a[i],
                lengths_b[i],
                dist_kernel,
                context,
                aggregator,
                padding_penalty,
            )
            for j in range(i + 1, k):
                d = compute_single_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    aggregator,
                    padding_penalty,
                )
                result[i, j] = d
                result[j, i] = d
        else:
            for j in range(k):
                result[i, j] = compute_single_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    aggregator,
                    padding_penalty,
                )


# ---------------------------------------------------------------------------
# Aggregation kernel registry
# ---------------------------------------------------------------------------

_AGG_NUMBA_KERNELS: dict[str, Callable] = {
    "mean": _numba_mean,
    "sum": _numba_sum,
}
