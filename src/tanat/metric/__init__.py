#!/usr/bin/env python3
"""
Metric Module.
"""

from .matrix import DistanceMatrix
from ._storage import StorageOptions

## -- Entity Metrics --
from .entity import (
    EntityMetric,
    HammingEntityMetric,
    HammingSettings,
)

## -- Sequence Metrics --
from .sequence import (
    SequenceMetric,
    LinearPairwiseSequenceMetric,
    LinearPairwiseSettings,
)

__all__ = [
    "DistanceMatrix",
    "StorageOptions",
    ## -- Entity Metrics --
    "EntityMetric",
    "HammingEntityMetric",
    "HammingSettings",
    ## -- Sequence Metrics --
    "SequenceMetric",
    "LinearPairwiseSequenceMetric",
    "LinearPairwiseSettings",
]
