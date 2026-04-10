#!/usr/bin/env python3
"""Tests: DistanceMatrix"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tanat.metric.matrix import DistanceMatrix


class TestDistanceMatrixCreation:
    """Construction and basic properties."""

    def test_init_basic(self) -> None:
        """Shape and ids are stored correctly on construction."""
        data = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
        dm = DistanceMatrix(data, ids=[1, 2])
        assert dm.shape == (2, 2)
        assert dm.ids == [1, 2]

    def test_empty_factory(self) -> None:
        """empty() creates a zero matrix with the right shape and ids."""
        dm = DistanceMatrix.empty([10, 20, 30])
        assert dm.shape == (3, 3)
        assert np.all(dm.data == 0.0)
        assert dm.ids == [10, 20, 30]

    def test_empty_dtype(self) -> None:
        """empty() respects an explicit dtype."""
        dm = DistanceMatrix.empty([1, 2], dtype="float64")
        assert dm.data.dtype == np.float64

    def test_invalid_non_square(self) -> None:
        """Non-square data raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            DistanceMatrix(np.zeros((2, 3), dtype="float32"), ids=[1, 2])

    def test_invalid_ids_mismatch(self) -> None:
        """ids length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="len\\(ids\\)"):
            DistanceMatrix(np.zeros((2, 2), dtype="float32"), ids=[1])

    def test_len(self) -> None:
        """__len__ returns the number of identifiers."""
        dm = DistanceMatrix.empty([1, 2, 3])
        assert len(dm) == 3

    def test_repr(self) -> None:
        """__repr__ contains the class name."""
        dm = DistanceMatrix.empty([1, 2])
        assert "DistanceMatrix" in repr(dm)


class TestDistanceMatrixConversions:
    """to_numpy and to_frame."""

    def test_to_numpy_returns_array(self) -> None:
        """to_numpy() returns the raw array unchanged."""
        data = np.array([[0.0, 0.5], [0.5, 0.0]], dtype="float32")
        dm = DistanceMatrix(data, ids=["a", "b"])
        out = dm.to_numpy()
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, data)

    def test_to_frame_shape(self) -> None:
        """to_frame() produces a pandas DataFrame with ids as index and columns."""
        dm = DistanceMatrix.empty(["x", "y", "z"])
        df = dm.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["x", "y", "z"]
        assert list(df.columns) == ["x", "y", "z"]

    def test_to_frame_values(self) -> None:
        """to_frame() preserves distance values at the correct (row, col) positions."""
        data = np.array([[0.0, 0.3], [0.3, 0.0]], dtype="float32")
        dm = DistanceMatrix(data, ids=[1, 2])
        df = dm.to_frame()
        assert df.loc[1, 2] == pytest.approx(0.3)
        assert df.loc[2, 1] == pytest.approx(0.3)
        assert df.loc[1, 1] == 0.0

    def test_symmetry_preserved(self) -> None:
        """to_numpy() is symmetric when constructed from a symmetric array."""
        data = np.array(
            [[0.0, 0.7, 0.3], [0.7, 0.0, 0.5], [0.3, 0.5, 0.0]], dtype="float32"
        )
        dm = DistanceMatrix(data, ids=[1, 2, 3])
        np.testing.assert_array_equal(dm.to_numpy(), dm.to_numpy().T)
