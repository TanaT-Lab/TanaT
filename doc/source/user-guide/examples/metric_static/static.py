"""
Static Metrics: Compare sequences and trajectories
=================================================

This example illustrates how to define a static metrics, which computes
distances between trajectories or sequences with static data (i.e. non-temporal
data).

We show that
"""

# %%
# Setup
# -----

import polars as pl
import matplotlib.pyplot as plt

from tanat import build_events, build_trajectories
from tanat.dataset import simulate_trajectories, simulate_static
from tanat.metric.static import StaticMetric
from tanat.metric.entity import HammingEntityMetric
from tanat.metric.sequence import EditSequenceMetric
from tanat.metric import AggregationTrajectoryMetric

# %%
# Generate synthetic trajectory data
# -----------------------------------
#
# Simulate some synthetic data for this example. We generate
# a sequence pool `event_pool` which contains events and static data
# for 10 individuas. Then, we also generate a trajectory pool with
# # the same data.

N_TRAJ = 10
SEED = 42

data = simulate_trajectories(
    sequences={
        "events": {"type": "event", "n_ids": N_TRAJ, "features": ["value", "category"]}
    },
    seed=SEED,
)

static_df = simulate_static(n_ids=N_TRAJ, features=["age", "group"], seed=0)

# Build pools for each sequence type
events_pool = build_events(
    temporal_data=data["events"],
    id_column="id",
    time_column="time",
    static_data=static_df,
)

events_pool.cast_features({"category": pl.Categorical})

# Build trajectory pool
traj_pool = build_trajectories(
    pools={"events": events_pool}, static_data=static_df, id_column="id"
)

print(traj_pool)

# %%
# Default static metric
# -------------------------
#
# The default static metric implements a kind of hamming distance
# stategy: it is 1 when at least one static attributes holds different
# values, and 0 if they are all similar.
#

defaut_static_metric = StaticMetric()

# we can now compare two sequences:

seqs_ids = events_pool.unique_ids
seq_a = events_pool[seqs_ids[0]]
seq_b = events_pool[seqs_ids[1]]

print("sequences:")
print(seq_a.static_data())
print(seq_b.static_data())

value = defaut_static_metric(seq_a, seq_b)

print(f"metric value between {seqs_ids[0]} and {seqs_ids[1]}: {value:.1f}.")

# %%
# The default metric compares all attributes, only one that differ is
# necessary to consider the static data are different.
seq_c = events_pool[seqs_ids[3]]
print(seq_c.static_data())

value = defaut_static_metric(seq_a, seq_c)
print(f"metric value between {seqs_ids[0]} and {seqs_ids[3]}: {value:.1f}.")


# %%
# Nonetheless, similar data leads to a 0-valued metric.
value = defaut_static_metric(seq_a, seq_a)
print(f"metric value: {value:.1f}.")

# %%
# Define a custom static metric
# --------------------------------
#
# Let us now imagine that we would like to compare only the age of the
# individuals, but we do not care about the group.
# We could drop the `group` static feature (see
# :func:`~tanat.trajectory.pool.TrajectoryPool.drop_static_features`)
# but we would loose it for future analysis.
#
# In this case, it is possible to define a custom static metric through
# the definition of a comparison function.
# This function must have two parameters that are dictionary with attributes
# names as keys.
#
# In our example, we simply evalute the absolute difference between the
# ages.


def age_cmp(s1, s2):
    """Comparison of age static features"""
    return abs(s1["age"] - s2["age"])


custom_static_metric = StaticMetric(cmp_fnct=age_cmp)


# Le us now compare the same sequences as before ...
seqs_ids = events_pool.unique_ids
seq_a = events_pool[seqs_ids[0]]
seq_b = events_pool[seqs_ids[1]]
seq_c = events_pool[seqs_ids[3]]

print("sequences:")
print(seq_a.static_data())
print(seq_b.static_data())
print(seq_c.static_data())

value_ab = custom_static_metric(seq_a, seq_b)
value_ac = custom_static_metric(seq_a, seq_c)

print(f"custom metric values: {value_ab:.1f}, {value_ac:.1f}.")

# %%
# Compute static distance between pools
# ----------------------------------------
#
# It is also possible to compute the distance matrix based on
# a pool of sequences (or trajectories) by simply calling the
# :func:`~tanat.metric.static.StaticMetric.compute_matrix` function.
#

mat = custom_static_metric.compute_matrix(events_pool)

print(mat)


# %%
# Compute static distance between trajectories
# -----------------------------------------------
# The same :class:`~tanat.metric.static.StaticMetric` objects can
# be used also for trajectories. You simply have to take care that
# the static feature exists and that the custom comparison function
# suits the data.

traj_ids = traj_pool.unique_ids
traj_a = traj_pool[traj_ids[0]]
traj_b = traj_pool[traj_ids[1]]

dist = custom_static_metric(traj_a, traj_b)
print(f"\nDistance between {traj_ids[0]} and {traj_ids[1]}: {dist:.1f}")


# It is also possible to use the default static metric and to compute
# a distance matrix in a similar way as for sequences.

# %%
# Use static metric in AggregationMetric
# ----------------------------------------
# A :class:`~tanat.metric.static.StaticMetric` can also be used in
# an :class:`~tanat.metric.trajectory.AggregationMetric`. This trajectory
# metric defines a distance as an aggregate of sequence metrics.
# In addition, it is possible to define a static metric as an additional
# metric to aggregate (with it own weight).
#
# Let us define a trajectory metric that involves the static data and for
# that, we first need a sequence metric ...

seq_metric = EditSequenceMetric(entity_metric=HammingEntityMetric("category"))

# then the trajectory metric is defined by the sequence metric required
# for the `events` type, and we add a `static_metric` with a very small weight
# (the comparison of ages leads to values between 0 and 120).

traj_metric = AggregationTrajectoryMetric(
    sequence_metrics={"events": seq_metric},
    static_metric=custom_static_metric,
    static_metric_weight=0.01,
)

traj_ids = traj_pool.unique_ids
traj_a = traj_pool[traj_ids[0]]
traj_b = traj_pool[traj_ids[1]]

dist = traj_metric(traj_a, traj_b)
print(f"\nDistance between {traj_ids[0]} and {traj_ids[1]}: {dist:.1f}")


#
# warning ::
#   It is possible to use static metric only in a trajectory metric but
#   not directly with a `SequenceMetric`.
#
