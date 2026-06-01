"""
Custom Trajectory Metric
========================

Learn how to implement a custom trajectory-level distance metric by subclassing
:class:`~tanat.metric.trajectory.base.TrajectoryMetric`.

A trajectory metric computes a scalar distance between two
:class:`~tanat.trajectory.trajectory.Trajectory` objects and can produce a
full pairwise :class:`~tanat.metric.DistanceMatrix` over a
:class:`~tanat.trajectory.pool.TrajectoryPool`.

**Minimal contract**

1. Declare a ``SETTINGS_CLASS`` dataclass (use ``tanat_utils.settings_dataclass``).
2. Implement ``_compute(traj_a, traj_b)``: return a non-negative ``float``.

The base class handles ``__call__``, ``compute_matrix``, and
``compute_cross_matrix`` automatically.
"""

# %% [markdown]
# Setup
# -----

# %%
from tanat_utils import settings_dataclass as dataclass

from tanat import build_events, build_states
from tanat.dataset import simulate_events, simulate_states
from tanat.trajectory.shortcuts import build_trajectories
from tanat.metric.trajectory.base import TrajectoryMetric

# %% [markdown]
# Data
# ----
#
# A trajectory pool combining an event sequence (``"visits"``) and a state
# sequence (``"status"``) for the same set of individuals.

# %%
N_IDS = 20
SEED = 42

events_df = simulate_events(n_ids=N_IDS, features=["score"], seed=SEED)
states_df = simulate_states(n_ids=N_IDS, features=["phase"], seed=SEED)

event_pool = build_events(temporal_data=events_df, id_column="id", time_column="time")
state_pool = build_states(
    temporal_data=states_df, id_column="id", start_column="start", end_column="end"
)

tpool = build_trajectories({"visits": event_pool, "status": state_pool})

# %%
print(tpool)

# %% [markdown]
# Define a custom trajectory metric
# -----------------------------------
#
# We implement a **total-length-diff** metric: the distance between two
# trajectories is the sum of absolute length differences across all shared
# sequence aliases. Simple by design, its purpose is to show the minimal pattern.


# %%
@dataclass
class TotalLengthDiffSettings:
    """Settings for :class:`TotalLengthDiffTrajectoryMetric`.

    Args:
        normalize: If ``True``, divide each per-alias difference by the length
            of the longer sub-sequence before summing.
    """

    normalize: bool = False


class TotalLengthDiffTrajectoryMetric(
    TrajectoryMetric, register_name="total_length_diff"
):
    """Trajectory distance as the sum of per-alias sequence-length differences.

    For each alias present in both trajectories:
    ``diff_alias = |len(alias_a) - len(alias_b)|``

    The final distance is the sum (or normalised sum) of those values.
    """

    SETTINGS_CLASS = TotalLengthDiffSettings

    def __init__(self, normalize: bool = False) -> None:
        super().__init__(settings=TotalLengthDiffSettings(normalize=normalize))

    # ------------------------------------------------------------------
    # Required implementation
    # ------------------------------------------------------------------

    def _compute(self, traj_a, traj_b) -> float:
        """Sum of per-alias absolute length differences."""
        total = 0.0
        for alias in traj_a:
            if alias not in traj_b:
                continue
            la = len(traj_a[alias])
            lb = len(traj_b[alias])
            diff = abs(la - lb)
            if self.settings.normalize:
                denom = max(la, lb)
                diff = (diff / denom) if denom > 0 else 0.0
            total += diff
        return float(total)


# %% [markdown]
# Instantiate and inspect
# -----------------------

# %%
metric = TotalLengthDiffTrajectoryMetric(normalize=True)
print(metric)

# %% [markdown]
# Compute a distance between two trajectories
# --------------------------------------------

# %%
ids = tpool.unique_ids
traj_a = tpool[ids[0]]
traj_b = tpool[ids[1]]

print(
    f"Trajectory A | visits: {len(traj_a['visits'])}, status: {len(traj_a['status'])}"
)
print(
    f"Trajectory B | visits: {len(traj_b['visits'])}, status: {len(traj_b['status'])}"
)
print(f"Distance: {metric(traj_a, traj_b):.4f}")

# %% [markdown]
# Compute a full pairwise distance matrix
# ----------------------------------------
#
# ``compute_matrix`` is inherited from the base class and works out of the
# box once ``_compute`` is implemented.

# %%
dm = metric.compute_matrix(tpool)
dm.to_frame().head()
