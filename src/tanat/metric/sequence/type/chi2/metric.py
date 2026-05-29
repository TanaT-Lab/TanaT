#!/usr/bin/env python3
"""
Chi2SequenceMetric: Chi-squared distance between state-time distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from tanat_utils import settings_dataclass as dataclass

from .....metadata.feature import CategoricalInfo
from ...base import SequenceMetric
from ....matrix import DistanceMatrix
from ...._storage import save_progress
from .kernels import compute_chi2_cross_matrix, compute_chi2_matrix, compute_chi2_pair

if TYPE_CHECKING:
    from .....sequence.base.sequence import Sequence
    from .....sequence.base.pool import SequencePool
    from ...._storage import StorageOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _histogram_weight_expr(sequence: Sequence | SequencePool) -> pl.Expr:
    """Build the per-entity weight expression for a sequence view."""
    time_cols = sequence.settings.get_time_columns()
    if len(time_cols) == 2:
        start_col, end_col = time_cols
        weight_expr = pl.col(end_col) - pl.col(start_col)
        if sequence.metadata.time_index.is_datetime:
            weight_expr = weight_expr.dt.total_seconds()
        return weight_expr.cast(pl.Float64)
    return pl.lit(1.0)


def _histogram_agg_lf(
    sequence: Sequence | SequencePool,
    feature: str,
) -> pl.LazyFrame:
    """Build the lazy long-form histogram aggregation for *feature*."""
    id_col = sequence.settings.id_column
    # pylint: disable=protected-access
    return (
        sequence._frames.temporal(  # pylint: disable=protected-access
            features=[feature]
        )
        .with_columns(
            pl.col(feature).cast(pl.Utf8).alias(feature),
            _histogram_weight_expr(sequence).alias("__weight__"),
        )
        .group_by([id_col, feature])
        .agg(pl.col("__weight__").sum().alias("__total_weight__"))
    )


def _histogram_vocab(agg_df: pl.DataFrame, feature: str) -> list[str]:
    """Extract the sorted vocabulary from an aggregated histogram frame."""
    if agg_df.is_empty():
        return []
    return sorted(
        agg_df.get_column(feature).drop_nulls().unique().to_list(),
        key=str,
    )


def _materialize_histogram_matrix(
    sequence: Sequence | SequencePool,
    agg_df: pl.DataFrame,
    feature: str,
    vocab: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Materialize a dense histogram matrix from aggregated histogram rows."""
    # pylint: disable=protected-access
    ids_df = sequence._id_lf.collect()
    resolved_vocab = _histogram_vocab(agg_df, feature) if vocab is None else vocab

    if not resolved_vocab:
        return np.zeros((ids_df.height, 0), dtype=np.float32), resolved_vocab

    if agg_df.is_empty():
        return (
            np.zeros((ids_df.height, len(resolved_vocab)), dtype=np.float32),
            resolved_vocab,
        )

    id_col = sequence.settings.id_column
    pivot_df = agg_df.pivot(on=feature, index=id_col, values="__total_weight__")
    result_df = ids_df.join(pivot_df, on=id_col, how="left")

    existing_columns = set(result_df.columns)
    exprs = [
        (
            pl.col(category).fill_null(0.0)
            if category in existing_columns
            else pl.lit(0.0)
        ).alias(category)
        for category in resolved_vocab
    ]
    return result_df.select(exprs).to_numpy().astype(np.float32), resolved_vocab


def _build_histogram(
    sequence: Sequence | SequencePool,
    feature: str,
    vocab: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a (n_sequences × n_categories) weight matrix for *feature*.

    Each sequence's weight per category is the total time spent in that state
    (``end - start`` in seconds for datetime, 1.0 per event otherwise).
    Rows are ordered and typed according to ``sequence._id_lf``.

    Returns:
        ``(hists, vocab)``: float32 array of shape ``(n, n_cats)`` and the
        sorted list of category strings.
    """
    agg_df = _histogram_agg_lf(sequence, feature).collect()
    return _materialize_histogram_matrix(sequence, agg_df, feature, vocab=vocab)


def _build_cross_histograms(
    pool_rows: Sequence | SequencePool,
    pool_cols: Sequence | SequencePool,
    feature: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build aligned histogram matrices for two sequence views."""
    agg_df = pl.concat(
        [
            _histogram_agg_lf(pool_rows, feature).with_columns(
                pl.lit("rows").alias("__side__")
            ),
            _histogram_agg_lf(pool_cols, feature).with_columns(
                pl.lit("cols").alias("__side__")
            ),
        ]
    )
    combined_df = agg_df.collect()
    vocab = _histogram_vocab(combined_df, feature)
    hists_rows, _ = _materialize_histogram_matrix(
        pool_rows,
        combined_df.filter(pl.col("__side__") == "rows").drop("__side__"),
        feature,
        vocab=vocab,
    )
    hists_cols, _ = _materialize_histogram_matrix(
        pool_cols,
        combined_df.filter(pl.col("__side__") == "cols").drop("__side__"),
        feature,
        vocab=vocab,
    )
    return hists_rows, hists_cols, vocab


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Chi2Settings:
    """Settings for :class:`Chi2SequenceMetric`.

    Args:
        entity_feature: Categorical feature name used as the histogram key
            (same semantics as :class:`~tanat.metric.entity.HammingEntityMetric`).
            ``None`` \u2192 resolved from the first entity feature of the sequence
            at :meth:`validate_composition` time.
    """

    entity_feature: str | None = None


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class Chi2SequenceMetric(SequenceMetric, register_name="chi2"):
    """Chi-squared distance between the state-time distributions of two sequences.

    Rather than comparing sequences element-by-element, Chi² compares the
    *proportion of time* (or event count) spent in each categorical state.

    * **Event** sequences: each event contributes a weight of 1.
    * **Interval / State** sequences: each entity contributes
      ``end − start`` as its weight.

    The per-category proportions are computed independently for each sequence,
    then compared via the Chi-squared distance formula:

    .. math::

        d(a, b) = \\sqrt{\\sum_j \\frac{(p_{aj} - p_{bj})^2}{p_{aj} + p_{bj}}}

    .. note::
        Chi² does **not** use an entity metric; ``entity_metric`` is absent
        from its settings.  The ``validate_composition`` method checks only
        that the requested feature is present.

    Empty-sequence behaviour:

    * **Both empty** → ``0.0`` (identical empty distributions).
    * **One empty** → ``1.0`` (maximally different distributions).

    Example::

        chi2 = Chi2SequenceMetric(entity_feature="status")
        d    = chi2(seq_a, seq_b)
        dm   = chi2.compute_matrix(pool)
    """

    SETTINGS_CLASS = Chi2Settings
    MEMMAP_SUPPORT = True

    def __init__(
        self,
        entity_feature: str | None = None,
        *,
        store_path: str | Path | None = None,
        chunk_size: int = 500,
        resume: bool = True,
        dtype: str = "float32",
    ) -> None:
        if store_path is not None:
            storage_options: dict | None = {
                "store_path": store_path,
                "chunk_size": chunk_size,
                "resume": resume,
                "dtype": dtype,
            }
        else:
            storage_options = None
        super().__init__(
            settings=Chi2Settings(entity_feature=entity_feature),
            storage=storage_options,
        )

    def _resolve_feature(self, seq: Sequence) -> str:
        """Return the target feature name, resolving from seq if needed."""
        if self.settings.entity_feature is not None:
            return self.settings.entity_feature
        return seq.settings.entity_features[0]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def validate_composition(
        self, seq_a: Sequence, seq_b: Sequence | None = None
    ) -> None:
        """Resolve and validate the target feature.

        If ``entity_feature`` was not specified, resolves to the first entity
        feature of ``seq_a`` and stores it in :attr:`target_feature`.
        Then checks that the feature is present and categorical in every
        provided sequence.

        Args:
            seq_a: Primary sequence.
            seq_b: Optional second sequence.

        Raises:
            KeyError:  If the feature is absent from a sequence.
            TypeError: If the feature is not categorical.
        """
        feature = self._resolve_feature(seq_a)

        for seq in (s for s in (seq_a, seq_b) if s is not None):
            info = seq.metadata.feature_info(feature)
            if info is None:
                raise KeyError(
                    f"Feature '{feature}' not found in sequence entity features. "
                    f"Available: {sorted(seq.settings.entity_features)}"
                )
            if not isinstance(info, CategoricalInfo):
                raise TypeError(
                    f"Chi2SequenceMetric requires a Categorical or Enum feature, "
                    f"got '{feature}' ({type(info).__name__}). "
                    f"Cast it first to pl.Categorical or pl.Enum."
                )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _compute(self, seq_a: Sequence, seq_b: Sequence) -> float:
        """Compute Chi-squared distance between two sequence distributions.

        Args:
            seq_a: First sequence.
            seq_b: Second sequence.

        Returns:
            Chi-squared distance >= 0.
        """
        feature = self._resolve_feature(seq_a)
        hists_a, hists_b, vocab = _build_cross_histograms(seq_a, seq_b, feature)
        return float(compute_chi2_pair(hists_a[0], hists_b[0], len(vocab)))

    # ------------------------------------------------------------------
    # Numba batch protocol
    # ------------------------------------------------------------------

    def prepare_batch_data(self, pool: SequencePool) -> tuple:
        """Build histogram arrays for all sequences in *pool*.

        Returns:
            ``(hists, n_cats)`` where *hists* is a float32 numpy array of
            shape ``(n, n_cats)`` containing raw (unnormalised) weights, with
            rows ordered to match ``pool.unique_ids``.
        """
        feature = self.settings.entity_feature or pool.metadata.entity_features[0].name
        hists, vocab = _build_histogram(pool, feature)
        return hists, len(vocab)

    # ------------------------------------------------------------------
    # Matrix computation: Numba optimisation
    # ------------------------------------------------------------------

    def _compute_matrix_impl(
        self,
        pool: SequencePool,
        *,
        storage: StorageOptions | None = None,
        result=None,
        is_resuming: bool = False,
        completed: int = 0,
    ) -> DistanceMatrix:
        """Build histograms then run the parallel Numba Chi2 kernel.

        Chi2 has no entity metric, so the Numba path is always used
        (no Python fallback needed).
        """
        hists, n_cats = self.prepare_batch_data(pool)
        n = len(hists)

        if result is None:
            result = np.full((n, n), np.nan, dtype=np.float32)

        chunk_size = storage.chunk_size if storage is not None else n
        chunks = list(range(0, n, chunk_size))

        with self._create_progress_bar(total=len(chunks), desc="Chunks") as pbar:
            for chunk_idx, chunk_start in enumerate(chunks):
                chunk_end = min(chunk_start + chunk_size, n)
                if is_resuming and chunk_idx < completed:
                    pbar.update(1)
                    continue
                compute_chi2_matrix(result, chunk_start, chunk_end, hists, n_cats, True)
                if storage is not None:
                    result.flush()
                    completed += 1
                    save_progress(storage, completed, status="computing")
                pbar.update(1)

        if storage is not None:
            result.flush()
            save_progress(storage, completed, status="complete")

        return DistanceMatrix(result, pool.unique_ids)

    def _compute_cross_matrix_impl(
        self,
        pool_rows: SequencePool,
        pool_cols: SequencePool,
    ) -> np.ndarray:
        """Compute the asymmetric Chi2 matrix between two pools."""
        feature = (
            self.settings.entity_feature or pool_rows.metadata.entity_features[0].name
        )
        hists_rows, hists_cols, vocab = _build_cross_histograms(
            pool_rows, pool_cols, feature
        )
        result = np.empty((len(hists_rows), len(hists_cols)), dtype=np.float32)
        if result.size == 0:
            return result

        compute_chi2_cross_matrix(result, hists_rows, hists_cols, len(vocab))
        return result
