"""
Filtering and preparing a cohort
==================================

**Scenario:** Starting from the admission pool built in
:doc:`explore_a_cohort`, you want
to isolate a clinically relevant sub-cohort, align all sequences to a shared
reference point (the first emergency admission, called T0), and trim to a fixed
observation window around that anchor.

**Concepts covered:**

- Combine :class:`~tanat.criterion.LengthCriterion` and
  :class:`~tanat.criterion.EntityCriterion` to select a sub-cohort
- Detect admission-type progressions with
  :class:`~tanat.criterion.PatternCriterion`
- Set a T0 reference with :meth:`~tanat.sequence.base.pool.SequencePool.set_t0`
- Convert to an event pool with
  :meth:`~tanat.sequence.IntervalSequencePool.as_event`
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl

from tanat.criterion import (
    ANY,
    EntityCriterion,
    LengthCriterion,
    PatternCriterion,
)
from tanat.dataset import access
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.visualization import SequenceVisualizer

# %% [markdown]
# Rebuild the admission pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Self-contained rebuild using the builder API (see :doc:`explore_a_cohort` for details).

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
# Step 1: Select the study cohort
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We focus on patients who:
#
# 1. Have **at least 2 admissions**.
# 2. Experienced **at least one emergency admission**, which marks the highest-acuity patients.
#
# :meth:`~tanat.sequence.base.pool.SequencePool.which` returns a set of patient IDs.
# The ``&`` operator computes the intersection of both criteria.

# %%
ids_multi = pool.which(LengthCriterion(ge=2))
ids_emergency = pool.which(
    EntityCriterion(query=pl.col("admission_type") == "EW EMER.")
)
ids_cohort = ids_multi & ids_emergency
print(f"[Intersection]      → {len(ids_cohort)} IDs")

# %%
cohort = pool.subset(ids_cohort)

print(cohort)

# %% [markdown]
# Step 2: Detect emergency-to-elective progressions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`~tanat.criterion.PatternCriterion` matches patients whose admission
# sequence contains a given ordered pattern. :data:`~tanat.criterion.ANY`
# acts as a wildcard that matches any single admission.
#
# Here we use it to identify patients who transitioned from an emergency
# admission to an elective one at some point, suggesting clinical stabilisation.
# This is an **exploratory query**, the result is not used to filter the cohort
# further, but illustrates how pattern-based selection works.

# %%
ids_stabilised = cohort.which(
    PatternCriterion(
        feature="admission_type",
        pattern=["EW EMER.", ANY, "ELECTIVE"],
    )
)

# %%
print(ids_stabilised)

# %% [markdown]
# Step 3: Anchor sequences to T0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We set T0 at the **start of the first emergency admission** of each patient.
# This alignment ensures that position 0 always corresponds to the index
# emergency event, making cross-patient comparisons meaningful.

# %%
cohort.set_t0(query=pl.col("admission_type") == "EW EMER.", anchor="start")
print(cohort.t0_data().head(5))

# %% [markdown]
# Step 4: Visualise the cohort aligned to T0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With T0 set, all sequences are now relative to the first emergency admission.
# The timeline makes it easy to compare individual admission patterns across patients.

# %%

# fmt: off
SequenceVisualizer.timeline(time_mode="relative", display_unit="days", allow_large=True) \
    .title("Admission sequences aligned to first emergency (T0)") \
    .x_axis(label="Days from first emergency admission") \
    .colors("tab10") \
    .draw(cohort, entity_feature="admission_type") \
    .show()
# fmt: on

# %% [markdown]
# Step 5: Convert to an event pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For some analyses you only need the admission *timestamp*, not its duration.
# :meth:`~tanat.sequence.IntervalSequencePool.as_event` converts the interval pool to an
# :class:`~tanat.sequence.EventSequencePool`, replacing the ``(start, end)``
# pair with a single ``time`` column.

# %%
event_pool = cohort.as_event(anchor="start")

# %%
print(event_pool)

# %% [markdown]
# The temporal data now contains a single ``time`` column instead of
# ``start`` and ``end``.

# %%
print(event_pool.temporal_data().head(5))
