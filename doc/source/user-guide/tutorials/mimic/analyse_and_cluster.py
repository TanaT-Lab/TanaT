"""
Analysing and clustering cohort sequences
===========================================

**Scenario:** Starting from the cohort prepared in
:doc:`filter_and_prepare`, you want to
quantify how similar the admission sequences are across patients, group them
into clusters with shared patterns, and visualise the results.

**Concepts covered:**

- Compute a pairwise sequence distance matrix with
  :class:`~tanat.metric.HammingEntityMetric` +
  :class:`~tanat.metric.LinearPairwiseSequenceMetric`
- Cluster with :class:`~tanat.clustering.HierarchicalClusterer`
- Inspect cluster membership
- Produce a **faceted timeline** coloured by cluster
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl

from tanat.clustering import HierarchicalClusterer
from tanat.criterion import (
    EntityCriterion,
    LengthCriterion,
)
from tanat.dataset import access
from tanat.metric import HammingEntityMetric, LinearPairwiseSequenceMetric
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.visualization import SequenceVisualizer

# %% [markdown]
# Rebuild the prepared cohort
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Self-contained rebuild using the builder API
# (see :doc:`explore_a_cohort` and :doc:`filter_and_prepare` for details).

# %%
DB = f"sqlite:///{access('mimic4')}"

pool = IntervalSequencePool(
    store=(
        IntervalSequencePool.builder()
        .add_sql(
            DB,
            "SELECT subject_id, admittime, dischtime,"
            "       admission_type, admission_location"
            ' FROM "hosp/admissions"',
            id_column="subject_id",
            start_column="admittime",
            end_column="dischtime",
            features=["admission_type", "admission_location"],
        )
        .add_sql(
            DB,
            'SELECT subject_id, gender, anchor_age AS age FROM "hosp/patients"',
            id_column="subject_id",
            is_static=True,
            features=["gender", "age"],
        )
        .build("admissions_store", exist_ok=True)
    )
)
# ``pl.Categorical`` is required by the metric and clustering modules, and
# enables consistent colour-coding across all visualisations.
pool.cast_features({"admission_type": pl.Categorical}, is_static=False)

# %%
print(pool)

# %% [markdown]
# Cohort selection
# ~~~~~~~~~~~~~~~~
#
# Same selection as :doc:`filter_and_prepare`: patients with at least 2 admissions who
# experienced at least one emergency, aligned to their first emergency (T0).

# %%
ids_cohort = pool.which(LengthCriterion(ge=2)) & pool.which(
    EntityCriterion(query=pl.col("admission_type") == "EW EMER.")
)
print(f"[Intersection]      → {len(ids_cohort)} IDs")

# %%
cohort = pool.subset(ids_cohort)
# Align sequences to first emergency admission (T0).
cohort.set_t0(query=pl.col("admission_type") == "EW EMER.", anchor="start")
print(cohort)

# %% [markdown]
# Step 1: Define the sequence metric
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`~tanat.metric.HammingEntityMetric` compares two admissions at the same
# position: distance 0 if they share the same type, 1 otherwise.
# :class:`~tanat.metric.LinearPairwiseSequenceMetric` aggregates these
# entity-level distances along the full sequence. The ``padding_penalty``
# penalises sequences of different lengths.

# %%
entity_metric = HammingEntityMetric(entity_feature="admission_type")
sequence_metric = LinearPairwiseSequenceMetric(
    entity_metric=entity_metric,
    padding_penalty=1.0,
)

# %% [markdown]
# Step 2: Compute the distance matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
dist_matrix = sequence_metric.compute_matrix(cohort)

# %%
dm = dist_matrix.data
print(f"Distance matrix shape : {dist_matrix.shape}")

# %% [markdown]
# Step 3: Hierarchical clustering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We group patients into 3 clusters using complete-linkage hierarchical
# clustering. After :meth:`~tanat.clustering.HierarchicalClusterer.fit`,
# the cluster label is automatically added as a static feature on the pool
# under the name given by ``cluster_column``.

# %%
clusterer = HierarchicalClusterer(
    metric=sequence_metric,
    n_clusters=3,
    linkage="complete",
    cluster_column="adm_cluster",
)
clusterer.fit(cohort)

# %% [markdown]
# Step 4: Inspect cluster membership
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :attr:`~tanat.clustering.Clusterer.clusters` exposes the fitted
# :class:`~tanat.clustering.Cluster` objects directly. Each provides a
# ``size`` and the list of patient ``items``.  The cluster label is also
# available as a static feature (``adm_cluster``) for downstream filtering
# or visualisation.

# %%
for cluster in clusterer.clusters:
    print(cluster)

# %%
# Cluster labels are also stored as a static feature for downstream use.
cohort.static_data().head()


# %% [markdown]
# Step 5: Faceted timeline coloured by cluster
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Each panel shows the admission sequences of one cluster, aligned to T0.
# This makes it easy to spot structural differences between groups.

# %%
# fmt: off
SequenceVisualizer.timeline(time_mode="relative", allow_large=True) \
    .title("Admission sequences by cluster") \
    .x_axis(label="Admissions from first emergency (T0)") \
    .colors("tab10") \
    .facet(by="adm_cluster", is_static=True, cols=2, share_y=False) \
    .draw(cohort, entity_feature="admission_type") \
    .show()
# fmt: on
