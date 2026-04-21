#!/usr/bin/env python3
"""
Numba kernels for LCPSequenceMetric.

All functions are @njit (no Python objects).  They operate on int32-encoded
feature arrays produced by the entity metric's ``prepare_batch_data``.

Output-mode integer encoding (mirrors ``_MODE_MAP`` in metric.py):
    0 → length
    1 → distance
    2 → normalized
"""

from __future__ import annotations

from numba import njit, prange
from numba import float32 as nb_float32

# ---------------------------------------------------------------------------
# Single-pair kernel
# ---------------------------------------------------------------------------


@njit
def compute_lcp_pair(
    arr_a,
    arr_b,
    len_a,
    len_b,
    dist_kernel,
    context,
    threshold,
    mode,
):
    """Compute LCP distance for a single pair of int32-encoded sequences.

    Scans aligned positions until the first mismatch (entity distance >
    ``threshold``) and counts the common prefix length.  Then applies the
    requested output ``mode``:

    * ``0`` (length)      → raw prefix count.
    * ``1`` (distance)    → ``len_a + len_b − 2 × lcp``.
    * ``2`` (normalized)  → ``1 − 2 × lcp / (len_a + len_b)``.

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
    if len_a == 0 and len_b == 0:
        return nb_float32(0.0)

    lcp_len = nb_float32(0.0)
    min_len = min(len_a, len_b)
    for i in range(min_len):
        if dist_kernel(arr_a[i], arr_b[i], context) > threshold:
            break
        lcp_len += nb_float32(1.0)

    n = nb_float32(len_a)
    m = nb_float32(len_b)

    if mode == 0:  # length
        return lcp_len
    if mode == 1:  # distance
        return n + m - nb_float32(2.0) * lcp_len
    # normalized
    total = n + m
    if total == nb_float32(0.0):
        return nb_float32(0.0)
    return nb_float32(1.0) - nb_float32(2.0) * lcp_len / total


# ---------------------------------------------------------------------------
# Full-matrix kernel
# ---------------------------------------------------------------------------


@njit(parallel=True)
def compute_lcp_matrix(
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
    """Parallel LCP matrix kernel.

    Processes rows ``[start, end)``.

    Args:
        result:    Full (n × n) or (chunk × k) memmap/array.
        start:     First row index (inclusive).
        end:       Last row index (exclusive).
        arrays_a:  Encoded sequences for the row pool.
        lengths_a: Lengths for the row pool.
        arrays_b:  Encoded sequences for the column pool.
        lengths_b: Lengths for the column pool.
        dist_kernel: Entity distance kernel.
        context:   Opaque context tuple.
        threshold: Equality threshold.
        mode:      Integer output mode (0/1/2).
        symmetric: When ``True``, exploit upper-triangle + mirror.
    """
    k = len(lengths_b)
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        if symmetric:
            result[i, i] = compute_lcp_pair(
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
                d = compute_lcp_pair(
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
                result[i, j] = compute_lcp_pair(
                    arrays_a[i],
                    arrays_b[j],
                    lengths_a[i],
                    lengths_b[j],
                    dist_kernel,
                    context,
                    threshold,
                    mode,
                )
