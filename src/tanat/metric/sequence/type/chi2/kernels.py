#!/usr/bin/env python3
"""
Numba kernels for Chi2SequenceMetric.

Chi² operates on pre-computed histogram arrays (one float32 row per
sequence), built in Python via Polars.  The Numba kernels only perform the
O(n² × V) pairwise chi-squared distance computations.

Histogram layout: ``hists[i]`` is a float32 vector of raw (unnormalised)
weights of length ``n_cats`` for sequence ``i``.  The kernel normalises
internally (division by ``sum(hists[i])``).
"""

from __future__ import annotations

import math

from numba import njit, prange
from numba import float32 as nb_float32


@njit
def compute_chi2_pair(hist_a, hist_b, n_cats):
    """Compute chi-squared distance between two (unnormalised) histograms.

    .. math::

        d(a, b) = \\sqrt{\\sum_j \\frac{(p_{aj} - p_{bj})^2}{p_{aj} + p_{bj}}}

    where ``p`` values are proportions (sum-normalised weights).

    Args:
        hist_a:  float32 weight array of length ``n_cats`` for sequence A.
        hist_b:  float32 weight array of length ``n_cats`` for sequence B.
        n_cats:  Number of categories (length of the histogram vectors).

    Returns:
        float32 chi-squared distance.
    """
    total_a = nb_float32(0.0)
    total_b = nb_float32(0.0)
    for k in range(n_cats):
        total_a += hist_a[k]
        total_b += hist_b[k]

    if total_a == nb_float32(0.0) and total_b == nb_float32(0.0):
        return nb_float32(0.0)
    if total_a == nb_float32(0.0) or total_b == nb_float32(0.0):
        return nb_float32(1.0)

    result = nb_float32(0.0)
    for k in range(n_cats):
        pa = hist_a[k] / total_a
        pb = hist_b[k] / total_b
        denom = pa + pb
        if denom > nb_float32(0.0):
            diff = pa - pb
            result += diff * diff / denom
    return nb_float32(math.sqrt(result))


@njit(parallel=True)
def compute_chi2_matrix(result, start, end, hists, n_cats, symmetric):
    """Parallel Chi2 matrix kernel.

    Processes rows ``[start, end)``.
    """
    n = result.shape[0]
    for i in prange(start, end):  # pylint: disable=not-an-iterable
        result[i, i] = compute_chi2_pair(hists[i], hists[i], n_cats)
        if symmetric:
            for j in range(i + 1, n):
                d = compute_chi2_pair(hists[i], hists[j], n_cats)
                result[i, j] = d
                result[j, i] = d
