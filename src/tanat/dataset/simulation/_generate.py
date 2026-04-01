#!/usr/bin/env python3
"""Private generation helpers shared across simulation functions."""

from __future__ import annotations

from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VOCAB: list[str] = ["A", "B", "C", "D", "E"]
_TYPE_CYCLE: list[str] = ["numeric", "categorical", "boolean"]
_DEFAULT_TIME_RANGE: tuple[datetime, datetime] = (
    datetime(2000, 1, 1),
    datetime(2025, 1, 1),
)
_DAY_US: int = 86_400 * 1_000_000


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def _resolve_feature_names(features: int | list[str], prefix: str) -> list[str]:
    """Resolve a feature spec to a list of column names.

    Args:
        features: Number of features (auto-named) or explicit name list.
        prefix: Prefix used for auto-naming (e.g. ``"f"`` or ``"s"``).

    Returns:
        List of column name strings.
    """
    if isinstance(features, int):
        return [f"{prefix}_{i}" for i in range(features)]
    return list(features)


def _assign_feature_types(names: list[str]) -> list[tuple[str, str]]:
    """Assign types to feature names cycling through numeric/categorical/boolean.

    Args:
        names: Ordered list of column names.

    Returns:
        List of ``(name, type)`` pairs where type is one of
        ``"numeric"``, ``"categorical"``, ``"boolean"``.
    """
    return [(name, _TYPE_CYCLE[i % 3]) for i, name in enumerate(names)]


def _generate_features(
    n: int,
    typed_names: list[tuple[str, str]],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate raw feature arrays for ``n`` rows.

    Args:
        n: Number of rows to generate.
        typed_names: List of ``(name, type)`` pairs from
            :func:`_assign_feature_types`.
        rng: NumPy random generator.

    Returns:
        Mapping of column name to generated array.
    """
    result: dict[str, np.ndarray] = {}
    for name, ftype in typed_names:
        if ftype == "numeric":
            result[name] = rng.standard_normal(n)
        elif ftype == "categorical":
            result[name] = rng.choice(_VOCAB, size=n).astype(object)
        else:  # boolean
            result[name] = rng.random(n) > 0.5
    return result


# ---------------------------------------------------------------------------
# Length helpers
# ---------------------------------------------------------------------------


def _sample_lengths(
    n_ids: int,
    length_range: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a sequence length for each ID.

    Args:
        n_ids: Number of IDs.
        length_range: (min, max) number of rows per ID (inclusive).
        rng: NumPy random generator.

    Returns:
        Int64 array of shape ``(n_ids,)`` with lengths in [min, max].
    """
    lo, hi = length_range
    return rng.integers(lo, hi + 1, size=n_ids)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _to_us(dt: datetime) -> int:
    """Convert a datetime to microseconds since the Unix epoch.

    Args:
        dt: Datetime to convert.

    Returns:
        Integer microsecond offset from 1970-01-01.
    """
    return int((dt - datetime(1970, 1, 1)).total_seconds() * 1_000_000)


def _sample_event_times(
    lengths: np.ndarray,
    time_range: tuple[datetime, datetime],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample event timestamps uniformly in ``time_range``, sorted per ID.

    Args:
        lengths: Number of events per ID.
        time_range: (start, end) bounds for timestamps.
        rng: NumPy random generator.

    Returns:
        Flat ``datetime64[us]`` array of length ``sum(lengths)``.
    """
    lo = _to_us(time_range[0])
    hi = _to_us(time_range[1])
    total = int(lengths.sum())
    raw = rng.integers(lo, hi, size=total)
    offset = 0
    for length in lengths:
        raw[offset : offset + length].sort()
        offset += int(length)
    return raw.view(dtype="datetime64[us]")


def _sample_interval_times(
    lengths: np.ndarray,
    duration_range: tuple[int, int],
    allow_overlaps: bool,
    time_range: tuple[datetime, datetime],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample interval start/end arrays.

    With ``allow_overlaps=True`` starts are sampled freely (sorted for
    readability) and durations are independent. With
    ``allow_overlaps=False`` each interval starts after the previous
    end, guaranteeing ``start[i+1] >= end[i]`` within every ID.

    Args:
        lengths: Number of intervals per ID.
        duration_range: (min, max) interval duration in days.
        allow_overlaps: When True intervals may overlap within an ID.
        time_range: (start, end) bounds for timestamps.
        rng: NumPy random generator.

    Returns:
        Tuple ``(starts, ends)`` as ``datetime64[us]`` arrays of length
        ``sum(lengths)``.
    """
    lo = _to_us(time_range[0])
    hi = _to_us(time_range[1])
    d_lo = duration_range[0] * _DAY_US
    d_hi = duration_range[1] * _DAY_US
    total = int(lengths.sum())
    starts_raw = np.empty(total, dtype=np.int64)
    ends_raw = np.empty(total, dtype=np.int64)
    offset = 0
    for length in lengths:
        length = int(length)
        if allow_overlaps:
            s = np.sort(rng.integers(lo, hi, size=length))
            durations = rng.integers(d_lo, d_hi + 1, size=length)
            e = s + durations
        else:
            s = np.empty(length, dtype=np.int64)
            e = np.empty(length, dtype=np.int64)
            current = int(rng.integers(lo, hi))
            for k in range(length):
                s[k] = current
                dur = int(rng.integers(d_lo, d_hi + 1))
                e[k] = current + dur
                gap = int(rng.integers(0, 5 * _DAY_US + 1))
                current = e[k] + gap
        starts_raw[offset : offset + length] = s
        ends_raw[offset : offset + length] = e
        offset += length
    return (
        starts_raw.view(dtype="datetime64[us]"),
        ends_raw.view(dtype="datetime64[us]"),
    )


def _sample_state_times(
    lengths: np.ndarray,
    duration_range: tuple[int, int],
    time_range: tuple[datetime, datetime],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample contiguous state start/end arrays.

    States within each ID are strictly contiguous: ``end[i] == start[i+1]``.
    The last state's ``end`` is set to ``time_range[1]``.

    Args:
        lengths: Number of states per ID.
        duration_range: (min, max) state duration in days.
        time_range: (start, end) bounds for timestamps.
        rng: NumPy random generator.

    Returns:
        Tuple ``(starts, ends)`` as ``datetime64[us]`` arrays of length
        ``sum(lengths)``.
    """
    lo = _to_us(time_range[0])
    hi = _to_us(time_range[1])
    d_lo = duration_range[0] * _DAY_US
    d_hi = duration_range[1] * _DAY_US
    total = int(lengths.sum())
    starts_raw = np.empty(total, dtype=np.int64)
    ends_raw = np.empty(total, dtype=np.int64)
    offset = 0
    for length in lengths:
        length = int(length)
        s_arr = np.empty(length, dtype=np.int64)
        e_arr = np.empty(length, dtype=np.int64)
        current = int(rng.integers(lo, hi))
        for k in range(length - 1):
            s_arr[k] = current
            dur = int(rng.integers(d_lo, d_hi + 1))
            current = current + dur
            e_arr[k] = current
        s_arr[length - 1] = current
        e_arr[length - 1] = hi
        starts_raw[offset : offset + length] = s_arr
        ends_raw[offset : offset + length] = e_arr
        offset += length
    return (
        starts_raw.view(dtype="datetime64[us]"),
        ends_raw.view(dtype="datetime64[us]"),
    )


# ---------------------------------------------------------------------------
# Sequence preparation helper
# ---------------------------------------------------------------------------


def _prepare_sequence(
    n_ids: int,
    seq_length_range: tuple[int, int],
    features: int | list[str],
    time_range: tuple[datetime, datetime] | None,
    seed: int | None,
) -> tuple[
    np.random.Generator,
    tuple[datetime, datetime],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Common setup shared by all ``simulate_*`` functions.

    Args:
        n_ids: Number of distinct sequence IDs.
        seq_length_range: (min, max) rows per ID.
        features: Feature count or name list.
        time_range: Optional (start, end) datetime bounds.
        seed: Random seed.

    Returns:
        Tuple of ``(rng, time_range, id_col, lengths, feat_data)``.
    """
    rng = np.random.default_rng(seed)
    tr = time_range if time_range is not None else _DEFAULT_TIME_RANGE
    ids = np.arange(1, n_ids + 1, dtype=np.int64)
    lengths = _sample_lengths(n_ids, seq_length_range, rng)
    id_col = np.repeat(ids, lengths)
    typed = _assign_feature_types(_resolve_feature_names(features, prefix="f"))
    feat_data = _generate_features(int(lengths.sum()), typed, rng)
    return rng, tr, id_col, lengths, feat_data
