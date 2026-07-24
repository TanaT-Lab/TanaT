"""
Custom Entity Metric
====================

Learn how to implement a custom entity-level distance metric by subclassing
:class:`~tanat.metric.entity.base.EntityMetric`.

An entity metric computes a scalar distance between two individual entities
(atomic observations in a sequence).  It is the basic building block used
by most sequence-level metrics.

**Minimal contract**

1. Declare a ``SETTINGS_CLASS`` dataclass (use ``tanat_utils.settings_dataclass``).
2. Implement ``validate_entity(ent_a, ent_b)``: raise ``TypeError`` / ``KeyError``
   when the entities are incompatible with the metric.
3. Implement ``_compute(ent_a, ent_b)``: return a non-negative ``float``.

The public ``__call__`` in the base class invokes ``validate_entity`` then
``_compute`` automatically.
"""

# %% [markdown]
# Setup
# -----

# %%
import polars as pl
from tanat_utils import settings_dataclass as dataclass

from tanat import build_events
from tanat.dataset import simulate_events
from tanat.metric.entity.base import EntityMetric

# %% [markdown]
# Data
# ----
#
# A small event pool with a ``score`` numeric feature and a ``status``
# categorical feature.

# %%
raw = simulate_events(n_ids=20, features=["score", "status"], seed=0)
pool = build_events(temporal_data=raw, id_column="id", time_column="time")
pool.cast_features({"score": pl.Float32})

# %%
print(pool)

# %% [markdown]
# Define a custom entity metric
# -----------------------------
#
# We implement a **normalised absolute difference** on a numeric feature.
# ``dist(a, b) = |a.score - b.score| / scale``


# %%
@dataclass
class ScoreDiffSettings:
    """Settings for :class:`ScoreDiffEntityMetric`.

    Args:
        entity_feature: Numeric feature to compare.
        scale: Normalisation constant (``1.0`` means raw absolute diff).
    """

    entity_feature: str = "score"
    scale: float = 1.0


class ScoreDiffEntityMetric(EntityMetric, register_name="score_diff"):
    """Normalised absolute difference on a numeric entity feature."""

    SETTINGS_CLASS = ScoreDiffSettings

    def __init__(self, entity_feature: str = "score", scale: float = 1.0) -> None:
        super().__init__(
            settings=ScoreDiffSettings(entity_feature=entity_feature, scale=scale)
        )

    # ------------------------------------------------------------------
    # Required implementations
    # ------------------------------------------------------------------

    def validate_entity(self, ent_a, ent_b=None) -> None:
        """Check that both entities expose the expected numeric feature."""
        self._validate_entity_instance(ent_a, ent_b)
        feat = self.settings.entity_feature
        for ent in (e for e in (ent_a, ent_b) if e is not None):
            if feat not in ent.data():
                raise KeyError(
                    f"Feature {feat!r} not found in entity. "
                    f"Available: {list(ent.data().keys())}"
                )

    def _compute(self, ent_a, ent_b) -> float:
        """Return normalised absolute difference of the configured feature."""
        feat = self.settings.entity_feature
        val_a = float(ent_a[feat])
        val_b = float(ent_b[feat])
        return abs(val_a - val_b) / self.settings.scale


# %% [markdown]
# Instantiate and inspect
# -----------------------

# %%
metric = ScoreDiffEntityMetric(entity_feature="score", scale=100.0)
print(metric)

# %% [markdown]
# Compute a distance between two entities
# ----------------------------------------

# %%
ids = pool.unique_ids
ent_a = pool[ids[0]][0]
ent_b = pool[ids[1]][0]

print(f"score A : {ent_a['score']}")
print(f"score B : {ent_b['score']}")
print(f"distance: {metric(ent_a, ent_b):.4f}")

# %% [markdown]
# Compute distances over several pairs
# -------------------------------------

# %%
print("Sample pairwise distances")
print("-" * 40)
for i in range(5):
    ea = pool[ids[i]][0]
    eb = pool[ids[i + 1]][0]
    d = metric(ea, eb)
    print(f"  {ea['score']:6.1f}  vs  {eb['score']:6.1f}  ->  {d:.4f}")

# %% [markdown]
# Use the custom metric inside a sequence metric
# -----------------------------------------------
#
# Any :class:`~tanat.metric.entity.base.EntityMetric` can be passed as the
# ``entity_metric`` argument to sequence-level metrics such as
# :class:`~tanat.metric.sequence.LinearPairwiseSequenceMetric`.

# %%
from tanat.metric.sequence import LinearPairwiseSequenceMetric

lp = LinearPairwiseSequenceMetric(entity_metric=metric)
print(lp)

# %%
seq_a = pool[ids[0]]
seq_b = pool[ids[1]]
dist = lp(seq_a, seq_b)
print(f"LinearPairwise distance: {dist:.4f}")

# %%
dm = lp.compute_matrix(pool)
dm.to_frame().head()


# .. note::
#
#       Such an custom metric is not optimized for large datasets. TanaT has mechanisms
#       based on the Numba framework.
#       For implementing an
#       computationaly efficient metric, we invite the developers to contact the core
#       developer team or to look at the code of the implemented entity metrics (especially
#       hamming).
