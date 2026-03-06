#!/usr/bin/env python3
"""
Centralised column-name constants for the Trajectory Store layout.
"""

from typing import Final


class TrajectorySchema:
    """Internal column names used by the trajectory store."""

    # --- Trajectory index ---
    TRAJ_ID: Final[str] = "_traj_id"

    class Files:
        """Physical file names for a trajectory store on disk."""

        CORE: Final[str] = "core.json"
        METADATA: Final[str] = "metadata.json"
        TRAJECTORY_INDEX: Final[str] = "trajectory_index.arrow"
        STATIC_FEATURES: Final[str] = "static_features.arrow"
        DIR_STORES: Final[str] = "stores"
