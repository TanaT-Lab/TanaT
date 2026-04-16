#!/usr/bin/env python3
"""
Tests for Cluster value object.
"""

from __future__ import annotations

from tanat.clustering import Cluster


class TestClusterProperties:
    """Basic property access."""

    def test_id(self) -> None:
        """id property returns the cluster_id passed at construction."""
        c = Cluster(cluster_id=0, items=["a", "b"])
        assert c.id == 0

    def test_items_returns_copy(self) -> None:
        """items returns a copy; mutating it leaves the internal list unchanged."""
        c = Cluster(cluster_id=1, items=["x", "y", "z"])
        items = c.items
        items.append("w")  # mutation on the returned copy
        assert len(c.items) == 3  # internal list is untouched

    def test_size(self) -> None:
        """size equals the number of items."""
        c = Cluster(cluster_id=2, items=["a", "b", "c"])
        assert c.size == 3

    def test_empty_cluster(self) -> None:
        """Cluster with no items has size 0 and an empty items list."""
        c = Cluster(cluster_id=0, items=[])
        assert c.size == 0
        assert c.items == []


class TestClusterDunder:
    """__repr__, __contains__, __len__."""

    def test_contains_true(self) -> None:
        """__contains__ returns True for a member item."""
        c = Cluster(cluster_id=0, items=["a", "b", "c"])
        assert "b" in c

    def test_contains_false(self) -> None:
        """__contains__ returns False for a non-member item."""
        c = Cluster(cluster_id=0, items=["a", "b"])
        assert "z" not in c

    def test_len(self) -> None:
        """__len__ equals the number of items."""
        c = Cluster(cluster_id=0, items=["a", "b", "c", "d"])
        assert len(c) == 4
