#!/usr/bin/env python3
"""Local fixtures for criterion tests."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _temporal_ids(pool) -> set:
    """IDs that actually have temporal data (excludes static-only IDs)."""
    id_col = pool.settings.id_column
    return set(pool.temporal_data(fmt="polars")[id_col].unique().to_list())


# ---------------------------------------------------------------------------
# Sentinel IDs (used only to build fixtures below)
# ---------------------------------------------------------------------------

#: A sequence that contains at least one 'error' status row.
HAS_ERROR_ID = 1

#: A sequence that contains NO 'error' status row.
NO_ERROR_ID = 2


# ---------------------------------------------------------------------------
# Sequence fixtures (extracted from the shared session-scoped interval_pool)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def seq_with_error(interval_pool):
    """A Sequence that contains at least one 'error' status row (id=1)."""
    return interval_pool[HAS_ERROR_ID]


@pytest.fixture(scope="session")
def seq_without_error(interval_pool):
    """A Sequence that contains no 'error' status row (id=2)."""
    return interval_pool[NO_ERROR_ID]


@pytest.fixture(scope="session")
def seq_with_adjacent_error_ok(interval_pool):
    """A Sequence where 'error' appears directly before 'ok' (id=1)."""
    return interval_pool[HAS_ERROR_ID]


@pytest.fixture(scope="session")
def seq_short(interval_pool):
    """A Sequence with exactly 3 entity rows (id=6)."""
    return interval_pool[6]


@pytest.fixture(scope="session")
def seq_long(interval_pool):
    """A Sequence with more than 6 entity rows (id=1)."""
    return interval_pool[HAS_ERROR_ID]
