#!/usr/bin/env python3
"""
Criterion module: filtering primitives for sequences, entities, and trajectories.
"""

from .base import Criterion, CriterionLevel, CriterionLevelError, CriterionError
from .type.length import LengthCriterion
from .type.entity import EntityCriterion
from .type.static import StaticCriterion
from .type.pattern import PatternCriterion, WILDCARD, ANY
from .type.time import TimeCriterion
from .type.rank import RankCriterion

__all__ = [
    "Criterion",
    "CriterionLevel",
    "CriterionLevelError",
    "CriterionError",
    "LengthCriterion",
    "EntityCriterion",
    "StaticCriterion",
    "PatternCriterion",
    "WILDCARD",
    "ANY",
    "TimeCriterion",
    "RankCriterion",
]
