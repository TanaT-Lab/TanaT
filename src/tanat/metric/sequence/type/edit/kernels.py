#!/usr/bin/env python3
"""
Numba kernels for EditSequenceMetric.

All functions are @njit (no Python objects).  They operate on int32-encoded
feature arrays produced by the entity metric's ``prepare_batch_data``.

Implementation: 2-row rolling Needleman-Wunsch DP.  Only two rows of the
full (n+1) × (m+1) matrix are kept in memory at any time.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba import float32 as nb_float32


@njit
def compute_edit_pair(
    arr_a,
    arr_b,
    len_a,
    len_b,
    dist_kernel,
    context,
    indel_cost,
    normalize,
):
    """Compute Needleman-Wunsch edit distance for a single pair.

    Uses a 2-row rolling DP (O(m) space, O(n×m) time).

    Args:
        arr_a:       int32-encoded sequence A.
        arr_b:       int32-encoded sequence B.
        len_a:       Length of A.
        len_b:       Length of B.
        dist_kernel: Numba entity distance kernel (substitution cost).
        context:     Opaque context tuple forwarded to ``dist_kernel``.
        indel_cost:  Cost per insertion / deletion (float32).
        normalize:   When ``True``, divide result by ``max(len_a, len_b)``.

    Returns:
        float32 edit distance.
    """
    n, m = len_a, len_b
    ic = nb_float32(indel_cost)

    if n == 0 and m == 0:
        return nb_float32(0.0)
    if n == 0:
        d = ic * nb_float32(m)
        return d / nb_float32(m) if normalize else d
    if m == 0:
        d = ic * nb_float32(n)
        return d / nb_float32(n) if normalize else d

    prev = np.empty(m + 1, dtype=np.float32)
    curr = np.empty(m + 1, dtype=np.float32)

    for j in range(m + 1):
        prev[j] = nb_float32(j) * ic

    for i in range(1, n + 1):
        curr[0] = nb_float32(i) * ic
        for j in range(1, m + 1):
            sub = dist_kernel(arr_a[i - 1], arr_b[j - 1], context)
            curr[j] = min(prev[j - 1] + sub, prev[j] + ic, curr[j - 1] + ic)
        for j in range(m + 1):
            prev[j] = curr[j]

    d = prev[m]
    if normalize:
        max_len = n if n > m else m
        if max_len > 0:
            d = d / nb_float32(max_len)
    return d


@njit(parallel=True)
def compute_edit_matrix(
    result,
    start,
    end,
    arrays_a,
    lengths_a,
    arrays_b,
    lengths_b,
    dist_kernel,
    context,
    indel_cost,
    normalize,
    symmetric,
):
    """Parallel Edit matrix kernel.

    Processes rows ``[start, end)``.
    """
    k = len(lengths_b)
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        if symmetric:
            result[i, i] = compute_edit_pair(
                arrays_a[i],
                arrays_b[i],
                lengths_a[i],
                lengths_b[i],
                dist_kernel,
                context,
                indel_cost,
                normalize,
            )
            for j in range(i + 1, k):
                d = compute_edit_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    indel_cost,
                    normalize,
                )
                result[i, j] = d
                result[j, i] = d
        else:
            for j in range(k):
                result[i, j] = compute_edit_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    indel_cost,
                    normalize,
                )
