#!/usr/bin/env python3
"""Local fixtures for tests/metadata/: standalone Sequence and Trajectory objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from tanat.sequence.type.interval.sequence import IntervalSequence
from tanat.sequence.type.event.sequence import EventSequence
from tanat.sequence.type.state.sequence import StateSequence
from tanat.sequence.type.interval.entity import IntervalEntity
from tanat.sequence.type.event.entity import EventEntity
from tanat.sequence.type.state.entity import StateEntity
from tanat.trajectory.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Class maps
# ---------------------------------------------------------------------------

_SEQ_CLS: dict = {
    "interval": IntervalSequence,
    "event": EventSequence,
    "state": StateSequence,
}

_ENTITY_CLS: dict = {
    "interval": IntervalEntity,
    "event": EventEntity,
    "state": StateEntity,
}

# ---------------------------------------------------------------------------
# ID / rank constants
#
# _ID_PARTIAL  (id=1)  : present in sequence data only, absent from static.csv
# _ID_COMPLETE (id=11) : present in both sequence and static data → real stats
# ---------------------------------------------------------------------------

_ID_PARTIAL = 1
_ID_COMPLETE = 11
_ENTITY_RANK = 0  # row index within a sequence used for standalone entity tests

# ---------------------------------------------------------------------------
# Parametrized type fixture
# ---------------------------------------------------------------------------


@pytest.fixture(params=["interval", "event", "state"])
def seq_type(request: pytest.FixtureRequest) -> str:
    """Parametrized over the three sequence types."""
    return request.param


# ---------------------------------------------------------------------------
# Standalone object fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def standalone_seqs(stores_dict: dict, seq_type: str) -> dict:
    """Dict {"partial": seq(id=1), "complete": seq(id=11)} for the current seq_type."""
    cls = _SEQ_CLS[seq_type]
    return {
        "partial": cls(id_value=_ID_PARTIAL, store=stores_dict[seq_type]),
        "complete": cls(id_value=_ID_COMPLETE, store=stores_dict[seq_type]),
    }


@pytest.fixture
def standalone_entities(stores_dict: dict, seq_type: str) -> dict:
    """Dict {"partial": entity(id=1, rank=0), "complete": entity(id=11, rank=0)} for the current seq_type."""
    cls = _ENTITY_CLS[seq_type]
    return {
        "partial": cls(
            id_value=_ID_PARTIAL, rank=_ENTITY_RANK, store=stores_dict[seq_type]
        ),
        "complete": cls(
            id_value=_ID_COMPLETE, rank=_ENTITY_RANK, store=stores_dict[seq_type]
        ),
    }


@pytest.fixture
def standalone_trajs(traj_store: Path) -> dict:
    """Dict {"partial": traj(id=1), "complete": traj(id=11)}."""
    return {
        "partial": Trajectory(id_value=_ID_PARTIAL, store=traj_store),
        "complete": Trajectory(id_value=_ID_COMPLETE, store=traj_store),
    }
