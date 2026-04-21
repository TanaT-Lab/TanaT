#!/usr/bin/env python3
"""
Numba kernels for LCSSequenceMetric.

All functions are @njit (no Python objects).  They operate on int32-encoded
feature arrays produced by the entity metric's ``prepare_batch_data``.

Output-mode integer encoding (mirrors ``_MODE_MAP`` in metric.py):
    0 → length
    1 → distance
    2 → normalized
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba import float32 as nb_float32


@njit
def compute_lcs_pair(
    arr_a,
    arr_b,
    len_a,
    len_b,
    dist_kernel,
    context,
    threshold,
    mode,
):
    """Compute LCS distance for a single pair of int32-encoded sequences.

    Uses a space-optimised 2-row rolling DP.  Two entities are considered
    equal when their distance ≤ ``threshold``.

    Args:
        arr_a:       int32-encoded sequence A.
        arr_b:       int32-encoded sequence B.
        len_a:       Length of A.
        len_b:       Length of B.
        dist_kernel: Numba entity distance kernel.
        context:     Opaque context tuple forwarded to ``dist_kernel``.
        threshold:   Equality threshold (float32).
        mode:        Integer output mode (0/1/2).

    Returns:
        float32 result.
    """
    if len_a == 0 or len_b == 0:
        lcs_len = nb_float32(0.0)
    else:
        prev = np.zeros(len_b + 1, dtype=np.float32)
        curr = np.zeros(len_b + 1, dtype=np.float32)
        for i in range(len_a):
            for j in range(len_b):
                if dist_kernel(arr_a[i], arr_b[j], context) <= threshold:
                    curr[j + 1] = prev[j] + nb_float32(1.0)
                else:
                    curr[j + 1] = max(curr[j], prev[j + 1])
            # swap prev ↔ curr, reset curr
            for j in range(len_b + 1):
                prev[j] = curr[j]
                curr[j] = nb_float32(0.0)
        lcs_len = prev[len_b]

    n = nb_float32(len_a)
    m = nb_float32(len_b)

    if mode == 0:  # length
        return lcs_len
    if mode == 1:  # distance
        return n + m - nb_float32(2.0) * lcs_len
    # normalized
    total = n + m
    if total == nb_float32(0.0):
        return nb_float32(0.0)
    return nb_float32(1.0) - nb_float32(2.0) * lcs_len / total


@njit(parallel=True)
def compute_lcs_matrix(
    result,
    start,
    end,
    arrays_a,
    lengths_a,
    arrays_b,
    lengths_b,
    dist_kernel,
    context,
    threshold,
    mode,
    symmetric,
):
    """Parallel LCS matrix kernel.

    Processes rows ``[start, end)``.
    """
    k = len(lengths_b)
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        if symmetric:
            result[i, i] = compute_lcs_pair(
                arrays_a[i],
                arrays_b[i],
                lengths_a[i],
                lengths_b[i],
                dist_kernel,
                context,
                threshold,
                mode,
            )
            for j in range(i + 1, k):
                d = compute_lcs_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    threshold,
                    mode,
                )
                result[i, j] = d
                result[j, i] = d
        else:
            for j in range(k):
                result[i, j] = compute_lcs_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    threshold,
                    mode,
                )
