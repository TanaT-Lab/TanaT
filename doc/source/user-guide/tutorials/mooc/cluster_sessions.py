"""
Clustering learning sessions by action patterns
=================================================

**Scenario:** Building on :doc:`explore_sessions`, you want to group
sessions with similar action sequences together to identify recurring
learning strategies.  This is the second tutorial in the
:ref:`MOOC series <mooc_tutorials>`.

**Concepts covered:**

- Rebuild the session pool with :func:`~tanat.build_states` (self-contained)
- Compute edit distances with
  :class:`~tanat.metric.HammingEntityMetric` +
  :class:`~tanat.metric.EditSequenceMetric` (Optimal Matching)
- Cluster with :class:`~tanat.clustering.HierarchicalClusterer`
- Inspect cluster membership via
  :attr:`~tanat.clustering.Clusterer.clusters`
- Visualise per-cluster state distributions with a faceted plot
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import pandas as pd
import polars as pl

from tanat import build_states
from tanat.clustering import HierarchicalClusterer
from tanat.criterion import LengthCriterion
from tanat.dataset import access
from tanat.metric import EditSequenceMetric, HammingEntityMetric
from tanat.visualization import SequenceVisualizer

# %% [markdown]
# Rebuild the filtered session pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Self-contained rebuild (see :doc:`explore_sessions` for details).

# %%
INACTIVITY = pd.Timedelta("2h")

df = access("mooc_events")
df["timecreated"] = pd.to_datetime(df["timecreated"])
df = df.sort_values(["user", "timecreated"])

df["session"] = (
    (df["user"] != df["user"].shift()) | (df["timecreated"].diff() > INACTIVITY)
).cumsum()

sessions = df[["user", "session"]].drop_duplicates()
df["position"] = df.groupby("session").cumcount()

pool = build_states(
    df[["session", "position", "Action"]],
    id_column="session",
    start_column="position",
    static_data=sessions,
    store_name="mooc_sessions_store",
)
# ``pl.Categorical`` enables consistent colour-coding across visualisations
# and is required by the metric module.
pool.cast_features({"Action": pl.Categorical}, is_static=False)

ids_keep = pool.which(LengthCriterion(ge=2, le=40))
pool_filtered = pool.subset(ids_keep)

# %%
print(pool_filtered)

# %% [markdown]
# Step 1: Define the sequence metric
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use **Optimal Matching** (edit distance), the standard metric for
# sequence analysis in the social sciences.
#
# :class:`~tanat.metric.HammingEntityMetric` compares two actions at the
# same position: distance 0 if they share the same type, 1 otherwise.
# :class:`~tanat.metric.EditSequenceMetric` extends this to full sequences
# by counting insertions, deletions, and substitutions.
#
# .. tip::
#
#    You can provide a custom substitution cost matrix to
#    :class:`~tanat.metric.HammingEntityMetric` to reflect domain knowledge
#    about action similarity (e.g. "Course_view" is closer to "Group_work"
#    than to "Feedback").  See the API reference for details.

# %%
entity_metric = HammingEntityMetric(entity_feature="Action")
sequence_metric = EditSequenceMetric(
    entity_metric=entity_metric,
    indel_cost=1.0,
)

# %% [markdown]
# Step 2: Cluster sessions
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We group sessions into 5 clusters using complete-linkage hierarchical
# clustering.  After :meth:`~tanat.clustering.HierarchicalClusterer.fit`,
# the cluster label is automatically added as a static feature under
# ``session_cluster``.

# %%
clusterer = HierarchicalClusterer(
    metric=sequence_metric,
    n_clusters=5,
    linkage="complete",
    cluster_column="session_cluster",
)
clusterer.fit(pool_filtered)

# %% [markdown]
# Step 3: Inspect cluster membership
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :attr:`~tanat.clustering.Clusterer.clusters` exposes the fitted
# :class:`~tanat.clustering.Cluster` objects directly.

# %%
for cluster in clusterer.clusters:
    print(cluster)

# %%
# Cluster labels are also stored as a static feature for downstream use.
pool_filtered.static_data().head()

# %% [markdown]
# Step 4: Faceted distribution per cluster
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Each panel shows how action proportions evolve across positions for one
# cluster.  Structural differences between clusters reveal distinct learning
# strategies.

# %%

# fmt: off
SequenceVisualizer.distribution(bin_size=1) \
    .title("Action distribution by learning cluster") \
    .x_axis(label="Position in session") \
    .facet(by="session_cluster", is_static=True, cols=3, share_y=True) \
    .draw(pool_filtered, entity_feature="Action") \
    .show()
# fmt: on
