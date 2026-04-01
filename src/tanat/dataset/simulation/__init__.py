#!/usr/bin/env python3
"""Simulation sub-package: synthetic data generation."""

from .events import simulate_events
from .intervals import simulate_intervals
from .states import simulate_states
from .static import simulate_static
from .trajectories import simulate_trajectories

__all__ = [
    "simulate_events",
    "simulate_intervals",
    "simulate_states",
    "simulate_static",
    "simulate_trajectories",
]
