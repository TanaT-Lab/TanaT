#!/usr/bin/env python3
"""
Local fixtures for tests/zeroing/.

Every test that calls set_t0() mutates the pool (_t0_setter replacement +
cache clear).  Session-scoped pools must never be mutated; all zeroing tests
work on a fresh copy produced by this fixture.
"""

from __future__ import annotations

import pytest

from tanat.sequence.type.interval.sequence import IntervalSequence
from tanat.sequence.type.event.sequence import EventSequence
from tanat.sequence.type.state.sequence import StateSequence

# ---------------------------------------------------------------------------
# Class maps
# ---------------------------------------------------------------------------

_SEQ_CLS: dict = {
    "interval": IntervalSequence,
    "event": EventSequence,
    "state": StateSequence,
}

# ID guaranteed to have temporal rows (present in sequence_main.parquet)
_ID_WITH_DATA = 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pool_copy(pools_dict: dict, pool_type: str):
    """Fresh copy of the session pool; safe to call set_t0()."""
    return pools_dict[pool_type].copy()


@pytest.fixture
def standalone_seq(stores_dict: dict, pool_type: str):
    """Standalone Sequence for ID=1; guaranteed to have temporal rows."""
    cls = _SEQ_CLS[pool_type]
    return cls(id_value=_ID_WITH_DATA, store=stores_dict[pool_type])
