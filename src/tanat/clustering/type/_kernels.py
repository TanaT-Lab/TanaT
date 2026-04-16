#!/usr/bin/env python3
"""
Numba-optimized kernels for the PAM (Partition Around Medoids) algorithm.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange

# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------


@jit(nopython=True)
def compute_min_distances(
    dist_matrix: np.ndarray,
    unselected: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute min and second-min distances from each unselected to any medoid.

    Returns:
        ``(min_dist, second_min_dist, closest_medoid_idx)`` arrays of
        length ``len(unselected)``.
    """
    n_unsel = len(unselected)
    n_sel = len(selected)

    min_dist = np.full(n_unsel, np.inf)
    second_min_dist = np.full(n_unsel, np.inf)
    closest_medoid_idx = np.zeros(n_unsel, dtype=np.int64)

    for i in range(n_unsel):
        u = unselected[i]
        for j in range(n_sel):
            m = selected[j]
            d = dist_matrix[u, m]
            if d < min_dist[i]:
                second_min_dist[i] = min_dist[i]
                min_dist[i] = d
                closest_medoid_idx[i] = j
            elif d < second_min_dist[i]:
                second_min_dist[i] = d

    return min_dist, second_min_dist, closest_medoid_idx


@jit(nopython=True, parallel=True)
def compute_swap_cost(
    dist_matrix: np.ndarray,
    unselected: np.ndarray,
    selected: np.ndarray,
    min_dist: np.ndarray,
    second_min_dist: np.ndarray,
    closest_medoid_idx: np.ndarray,
) -> np.ndarray:
    """Cost matrix for all (medoid, non-medoid) swap pairs.

    ``cost[mi, hi]`` is the total change in clustering cost if
    ``selected[mi]`` is replaced by ``unselected[hi]``.
    Negative values indicate an improving swap.

    Returns:
        Cost matrix of shape ``(len(selected), len(unselected))``.
    """
    n_sel = len(selected)
    n_unsel = len(unselected)
    cost = np.zeros((n_sel, n_unsel))

    for mi in prange(n_sel):  # pylint: disable=not-an-iterable
        medoid = selected[mi]

        # Precompute: distance from old medoid to nearest *remaining* medoid.
        d_medoid_remaining = np.inf
        for j in range(n_sel):
            if j == mi:
                continue
            d_medoid_remaining = min(
                d_medoid_remaining, dist_matrix[medoid, selected[j]]
            )

        for hi in range(n_unsel):
            h = unselected[hi]
            d_medoid_h = dist_matrix[medoid, h]

            delta = 0.0

            # h becomes a medoid: its own cost drops to 0.
            delta -= min_dist[hi]

            # Old medoid becomes a non-medoid: assign to nearest remaining or h.
            delta += min(d_medoid_remaining, d_medoid_h)

            # Every other non-medoid: keep best assignment or switch to h.
            for ui in range(n_unsel):
                if ui == hi:
                    continue
                u = unselected[ui]
                d_uh = dist_matrix[u, h]
                if closest_medoid_idx[ui] == mi:
                    # mi was the closest medoid; now use second-best or h.
                    new_d = min(second_min_dist[ui], d_uh)
                else:
                    new_d = min(min_dist[ui], d_uh)
                delta += new_d - min_dist[ui]

            cost[mi, hi] = delta

    return cost


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def pam_build_optimized(
    dist_matrix: np.ndarray, n_clusters: int
) -> tuple[list[int], list[int]]:
    """BUILD phase: greedy initial medoid selection.

    Returns:
        ``(selected, unselected)``: lists of medoid and non-medoid indices.
    """
    n = dist_matrix.shape[0]
    all_indices = list(range(n))

    # First medoid: minimise total distance to all objects.
    first_medoid = int(np.argmin(dist_matrix.sum(axis=1)))
    selected = [first_medoid]
    unselected = [i for i in all_indices if i != first_medoid]

    # d_min[u] = distance from object u to its nearest current medoid.
    d_min = dist_matrix[:, first_medoid].copy().astype(float)

    for _ in range(1, n_clusters):
        best_candidate = unselected[0]
        best_gain = float("-inf")

        for c in unselected:
            gain = 0.0
            for u in unselected:
                if u == c:
                    continue
                gain += max(0.0, d_min[u] - dist_matrix[u, c])
            if gain > best_gain:
                best_gain = gain
                best_candidate = c

        selected.append(best_candidate)
        unselected.remove(best_candidate)

        # Update d_min for remaining unselected objects.
        for u in unselected:
            d_min[u] = min(d_min[u], dist_matrix[u, best_candidate])

    return selected, unselected


def pam_swap_optimized(
    selected: list[int],
    unselected: list[int],
    dist_matrix: np.ndarray,
) -> tuple[int, int] | None:
    """SWAP phase: find the best improving (medoid, non-medoid) swap.

    Returns:
        ``(medoid_idx, non_medoid_idx)`` to swap, or ``None`` if converged.
    """
    if not unselected or not selected:
        return None

    sel_arr = np.array(selected, dtype=np.int64)
    unsel_arr = np.array(unselected, dtype=np.int64)

    min_dist, second_min_dist, closest_medoid_idx = compute_min_distances(
        dist_matrix, unsel_arr, sel_arr
    )

    cost_matrix = compute_swap_cost(
        dist_matrix, unsel_arr, sel_arr, min_dist, second_min_dist, closest_medoid_idx
    )

    # Find the most beneficial swap (most negative cost change).
    flat_idx = int(np.argmin(cost_matrix))
    mi, hi = divmod(flat_idx, len(unselected))
    best_cost = cost_matrix[mi, hi]

    if best_cost < -1e-10:
        return (selected[mi], unselected[hi])
    return None
