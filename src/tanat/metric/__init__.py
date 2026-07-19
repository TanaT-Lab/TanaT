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
    Chi2SequenceMetric,
    Chi2Settings,
    DTWSequenceMetric,
    DTWSettings,
    EditSequenceMetric,
    EditSettings,
    LCPSequenceMetric,
    LCPSettings,
    LCSSequenceMetric,
    LCSSettings,
    LinearPairwiseSequenceMetric,
    LinearPairwiseSettings,
    SoftDTWSequenceMetric,
    SoftDTWSettings,
)

## -- Trajectory Metrics --
from .trajectory import (
    TrajectoryMetric,
    AggregationTrajectoryMetric,
    AggregationSettings,
)

## -- Static Metrics --
from .static import StaticMetric

__all__ = [
    "DistanceMatrix",
    "StorageOptions",
    ## -- Entity Metrics --
    "EntityMetric",
    "HammingEntityMetric",
    "HammingSettings",
    ## -- Sequence Metrics --
    "SequenceMetric",
    "Chi2SequenceMetric",
    "Chi2Settings",
    "DTWSequenceMetric",
    "DTWSettings",
    "EditSequenceMetric",
    "EditSettings",
    "LCPSequenceMetric",
    "LCPSettings",
    "LCSSequenceMetric",
    "LCSSettings",
    "LinearPairwiseSequenceMetric",
    "LinearPairwiseSettings",
    "SoftDTWSequenceMetric",
    "SoftDTWSettings",
    ## -- Trajectory Metrics --
    "TrajectoryMetric",
    "AggregationTrajectoryMetric",
    "AggregationSettings",
    ## -- Static Metrics --
    "StaticMetric",
]
