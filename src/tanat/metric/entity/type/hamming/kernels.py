#!/usr/bin/env python3
"""
Numba kernels for HammingEntityMetric.

These are pure Numba functions (no Python objects, no classes).
They receive integer-encoded feature values and a context tuple.
"""

from __future__ import annotations

from numba import njit
from numba import float32 as nb_float32


@njit
def hamming_dist_simple(a, b, context):
    """1.0 if different, 0.0 if equal. context is ignored.

    Args:
        a: Integer code for the first entity's feature value.
        b: Integer code for the second entity's feature value.
        context: Unused (empty tuple for this kernel).

    Returns:
        ``0.0`` if ``a == b``, ``1.0`` otherwise.
    """
    return nb_float32(1.0) if a != b else nb_float32(0.0)


@njit
def hamming_dist_weighted(a, b, context):
    """Lookup in cost matrix: context[0][a, b].

    Args:
        a: Integer code for the first entity's feature value.
        b: Integer code for the second entity's feature value.
        context: One-element tuple containing the (V × V) float32 cost matrix.

    Returns:
        ``context[0][a, b]``: the pre-built pairwise cost.
    """
    cost_matrix = context[0]
    return cost_matrix[a, b]
