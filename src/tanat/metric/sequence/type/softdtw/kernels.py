#!/usr/bin/env python3
"""
Numba kernels for SoftDTWSequenceMetric.

All functions are @njit (no Python objects).  They operate on int32-encoded
feature arrays produced by the entity metric's ``prepare_batch_data``.

Implementation: full (n+2) × (m+2) DP matrix with soft-minimum operator.
The log-sum-exp trick is used for numerical stability.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numba import float32 as nb_float32


@njit
def _softmin3(a, b, c, gamma):
    """Numerically stable soft-minimum of three float32 values.

    soft-min(a, b, c; γ) = −γ · log(exp(−a/γ) + exp(−b/γ) + exp(−c/γ))

    Uses the log-sum-exp trick.  Returns ``+inf`` when all inputs are
    ``+inf``.

    Args:
        a, b, c: Input values (float32).
        gamma:   Regularisation parameter (float32 > 0).

    Returns:
        float32 soft-minimum.
    """
    INF = nb_float32(np.inf)
    if a == INF and b == INF and c == INF:
        return INF

    g = nb_float32(gamma)
    neg_g = -g
    sa = -a / g if a != INF else nb_float32(-np.inf)
    sb = -b / g if b != INF else nb_float32(-np.inf)
    sc = -c / g if c != INF else nb_float32(-np.inf)

    max_s = sa
    if sb > max_s:
        max_s = sb
    if sc > max_s:
        max_s = sc

    log_sum = max_s + math.log(
        math.exp(sa - max_s) + math.exp(sb - max_s) + math.exp(sc - max_s)
    )
    return nb_float32(neg_g * log_sum)


@njit
def compute_softdtw_pair(
    arr_a,
    arr_b,
    len_a,
    len_b,
    dist_kernel,
    context,
    gamma,
):
    """Compute SoftDTW distance for a single pair of int32-encoded sequences.

    Uses the full (n+2) × (m+2) DP matrix.

    Args:
        arr_a:       int32-encoded sequence A.
        arr_b:       int32-encoded sequence B.
        len_a:       Length of A.
        len_b:       Length of B.
        dist_kernel: Numba entity distance kernel.
        context:     Opaque context tuple forwarded to ``dist_kernel``.
        gamma:       Regularisation parameter (float32 > 0).

    Returns:
        float32 SoftDTW distance, or ``nan`` when either sequence is empty.
    """
    n, m = len_a, len_b
    if n == 0 or m == 0:
        return nb_float32(np.nan)

    INF = nb_float32(np.inf)
    # R is (n+2) × (m+2); indices shifted by 1 vs the DP positions.
    R = np.full((n + 2, m + 2), np.inf, dtype=np.float32)
    R[0, 0] = nb_float32(0.0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_kernel(arr_a[i - 1], arr_b[j - 1], context)
            R[i, j] = cost + _softmin3(R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma)

    return R[n, m]


@njit(parallel=True)
def compute_softdtw_matrix(
    result,
    start,
    end,
    arrays_a,
    lengths_a,
    arrays_b,
    lengths_b,
    dist_kernel,
    context,
    gamma,
    symmetric,
):
    """Parallel SoftDTW matrix kernel.

    Processes rows ``[start, end)``.
    """
    k = len(lengths_b)
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        if symmetric:
            result[i, i] = compute_softdtw_pair(
                arrays_a[i],
                arrays_b[i],
                lengths_a[i],
                lengths_b[i],
                dist_kernel,
                context,
                gamma,
            )
            for j in range(i + 1, k):
                d = compute_softdtw_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    gamma,
                )
                result[i, j] = d
                result[j, i] = d
        else:
            for j in range(k):
                result[i, j] = compute_softdtw_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    gamma,
                )
