#!/usr/bin/env python3
"""
Cluster: value object representing a single cluster after fit().
"""

from __future__ import annotations


class Cluster:
    """A cluster of items produced by a :class:`~tanat.clustering.Clusterer`.

    Immutable value object: populated once after ``fit()`` and never mutated.

    Args:
        cluster_id: Integer identifier for the cluster.
        items:      List of item IDs belonging to this cluster.
    """

    def __init__(self, cluster_id: int, items: list) -> None:
        self._id = cluster_id
        self._items = list(items)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> int:
        """Cluster identifier."""
        return self._id

    @property
    def items(self) -> list:
        """Item IDs belonging to this cluster."""
        return list(self._items)

    @property
    def size(self) -> int:
        """Number of items in the cluster."""
        return len(self._items)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Cluster(id={self.id}, size={self.size})"

    def __contains__(self, item) -> bool:
        """Check if an item ID belongs to this cluster."""
        return item in self._items

    def __len__(self) -> int:
        return self.size
