#!/usr/bin/env python3
"""
Numba kernels for DTWSequenceMetric.

All functions are @njit (no Python objects).  They operate on int32-encoded
feature arrays produced by the entity metric's ``prepare_batch_data``.

Uses a 2-row rolling DP with optional Sakoe-Chiba band (``window ≥ 0``).
Pass ``window = -1`` to disable the band constraint (full DTW).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba import float32 as nb_float32


@njit
def compute_dtw_pair(
    arr_a,
    arr_b,
    len_a,
    len_b,
    dist_kernel,
    context,
    window,
    normalize,
):
    """Compute DTW distance for a single pair of int32-encoded sequences.

    Uses a 2-row rolling DP with optional Sakoe-Chiba band.

    Args:
        arr_a:       int32-encoded sequence A.
        arr_b:       int32-encoded sequence B.
        len_a:       Length of A.
        len_b:       Length of B.
        dist_kernel: Numba entity distance kernel.
        context:     Opaque context tuple forwarded to ``dist_kernel``.
        window:      Sakoe-Chiba band half-width.  ``-1`` = no constraint.
        normalize:   When ``True``, divide by ``len_a + len_b``.

    Returns:
        float32 DTW distance, or ``nan`` when either sequence is empty.
    """
    n, m = len_a, len_b
    if n == 0 or m == 0:
        return nb_float32(np.nan)

    INF = nb_float32(np.inf)
    prev = np.full(m + 1, np.inf, dtype=np.float32)
    curr = np.full(m + 1, np.inf, dtype=np.float32)
    prev[0] = nb_float32(0.0)

    for i in range(n):
        for jj in range(m + 1):
            curr[jj] = INF

        if window >= 0:
            j_lo = i - window
            j_hi = i + window
            j_lo = max(j_lo, 0)
            j_hi = min(j_hi, m - 1)
        else:
            j_lo = 0
            j_hi = m - 1

        for j in range(j_lo, j_hi + 1):
            cost = dist_kernel(arr_a[i], arr_b[j], context)
            best = prev[j]
            if prev[j + 1] < best:
                best = prev[j + 1]
            if curr[j] < best:
                best = curr[j]
            curr[j + 1] = cost + best

        for jj in range(m + 1):
            prev[jj] = curr[jj]

    d = prev[m]
    if normalize:
        d = d / nb_float32(n + m)
    return d


@njit(parallel=True)
def compute_dtw_matrix(
    result,
    start,
    end,
    arrays_a,
    lengths_a,
    arrays_b,
    lengths_b,
    dist_kernel,
    context,
    window,
    normalize,
    symmetric,
):
    """Parallel DTW matrix kernel.

    Processes rows ``[start, end)``.
    """
    k = len(lengths_b)
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        if symmetric:
            result[i, i] = compute_dtw_pair(
                arrays_a[i],
                arrays_b[i],
                lengths_a[i],
                lengths_b[i],
                dist_kernel,
                context,
                window,
                normalize,
            )
            for j in range(i + 1, k):
                d = compute_dtw_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    window,
                    normalize,
                )
                result[i, j] = d
                result[j, i] = d
        else:
            for j in range(k):
                result[i, j] = compute_dtw_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    window,
                    normalize,
                )
