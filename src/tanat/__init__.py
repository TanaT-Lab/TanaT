#!/usr/bin/env python3
"""
TanaT - Temporal Analysis of Trajectories
"""

import logging
from pathlib import Path

from tanat_utils import check_latest_version

from .core import registry as _registry
from .core import context
from .store.factory import StoreFactory
from .core.workspace import Workspace
from .sequence.base.pool import SequencePool
from .trajectory.pool import TrajectoryPool

from .sequence.shortcuts import build_events, build_intervals, build_states
from .trajectory.shortcuts import build_trajectories

logging.getLogger("tanat").addHandler(logging.NullHandler())
LOGGER = logging.getLogger(__name__)


# Workspace utility functions
def set_workspace(path: str):
    """
    Set the active workspace to a new directory.
    """
    new_ws = Workspace(path)
    context.set_active_ws_instance(new_ws)
    LOGGER.info("Workspace updated to: %s", new_ws.root)


def get_workspace():
    """
    Get the active workspace instance.
    """
    return context.get_workspace()


##  --- INTERNAL BUILDER REGISTRY ---
def _sequence_pool_builder(path: Path) -> SequencePool:
    store = StoreFactory.from_path(path)
    sequence_type = store.get_sequence_type()
    seqpool_cls = SequencePool.get_registered(sequence_type)
    return seqpool_cls(store=store)


_registry.register_factory("sequence", _sequence_pool_builder)
_registry.register_factory("trajectory", lambda path: TrajectoryPool(store=path))


# Check for updates
# Emit a warning if a new version is available on PyPI
check_latest_version("tanat")
