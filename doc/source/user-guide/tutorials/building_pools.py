"""
Building pools from multiple sources
======================================

Combine **three file-based data sources** (two Parquet files and one CSV) that
share the same schema into a single :class:`~tanat.sequence.IntervalSequencePool`,
then compose a :class:`~tanat.trajectory.pool.TrajectoryPool` on top.

**What you will learn:**

- ``add_parquet`` / ``add_csv``: ingest data from files
- Multi-source chaining: three ``.add_*()`` calls → one store
- ``is_static=True`` for per-individual static features (``bmi``, ``site``)
- Builder option: ``sort_anchor``
- Trajectory composition with ``TrajectoryPool.builder()``
- Workspace registration and ``pool.save()``

.. note::

   SQL sources (``add_sql``) follow the exact same pattern and accept the same
   column-mapping parameters. They require ``connectorx``; install it with
   ``pip install 'tanat[sql]'``. See :doc:`../../reference/builder` for details.

**Scenario:** Admission records arrive from three upstream systems (a hospital
export, a supplementary cohort, and a legacy CSV extract).  All three share
the schema ``severity_score`` (float) / ``ward`` (categorical) and are merged
into a single pool, then linked to a procedures pool through a ``TrajectoryPool``.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import tempfile
from pathlib import Path
import datetime

import pandas as pd
import polars as pl

from tanat import get_workspace, set_workspace
from tanat.dataset.simulation import (
    simulate_events,
    simulate_intervals,
    simulate_static,
)
from tanat.sequence.type.event.pool import EventSequencePool
from tanat.sequence.type.interval.pool import IntervalSequencePool
from tanat.trajectory.pool import TrajectoryPool

# %% [markdown]
# Workspace setup
# ~~~~~~~~~~~~~~~
#
# A **workspace** registers every store built below under a short name so
# any script can reload them later without tracking file paths.

# %%
set_workspace("~/.tanat_workspace/building_pools_tutorial")
ws = get_workspace()
ws.clear()
print(ws)

# %% [markdown]
# Generate source files
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``simulate_intervals`` generates one row per admission.  Feature types follow
# the **numeric → categorical → boolean** cycle, so:
#
# - ``severity_score`` → float  (clinical severity score at admission)
# - ``ward``           → categorical  (care unit, values ``{A, B, C, D, E}``)
#
# Static features follow the same cycle:
#
# - ``bmi``  → float
# - ``site`` → categorical  (care site, values ``{A, B, C, D, E}``)
#
# Patient IDs are prefixed with the source letter (``"a"``, ``"b"``, ``"c"``)
# to avoid collisions when the three files are merged.
# The simulation outputs columns ``id / start / end`` by default; these match
# the builder's defaults so **no column mapping is needed**.


# %%
def _prefix_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix the ``id`` column with a source identifier."""
    return df.assign(id=prefix + df["id"].astype(str))


# Feature schemas: defined once, reused across simulate_* and builder calls
ADMISSION_FEATURES = ["severity_score", "ward"]  # float, categorical
STATIC_FEATURES = ["bmi", "site"]  # float, categorical
PROCEDURE_FEATURES = ["priority", "procedure"]  # float, categorical
TIME_RANGE = (datetime.datetime(2000, 1, 1), datetime.datetime(2000, 12, 31))
SEED = 42

tmpdir = Path(tempfile.mkdtemp())

# %%
# Simulate admissions in different sources (parquets, CSV, ...)

# Source A
src_a = simulate_intervals(
    n_ids=100,
    features=ADMISSION_FEATURES,
    time_range=TIME_RANGE,
    seed=SEED,
)
src_a = _prefix_ids(src_a, "a")

# Source B
src_b = simulate_intervals(
    n_ids=50,
    features=ADMISSION_FEATURES,
    time_range=TIME_RANGE,
    seed=SEED + 1,
)
src_b = _prefix_ids(src_b, "b")

## Source C
src_c = simulate_intervals(
    n_ids=30,
    features=ADMISSION_FEATURES,
    time_range=TIME_RANGE,
    seed=SEED + 2,
)
src_c = _prefix_ids(src_c, "c")

# %%

# Simulate static demographics (one row per patient, all sources)
df_static = simulate_static(n_ids=100, features=STATIC_FEATURES, seed=SEED + 10)
tmp_static_A = _prefix_ids(df_static, "a")
df_static_B = simulate_static(n_ids=50, features=STATIC_FEATURES, seed=SEED + 11)
tmp_static_B = _prefix_ids(df_static_B, "b")
df_static_C = simulate_static(n_ids=30, features=STATIC_FEATURES, seed=SEED + 12)
tmp_static_C = _prefix_ids(df_static_C, "c")

# Static single DataFrame
static_df = pd.concat([tmp_static_A, tmp_static_B, tmp_static_C])


# %%
# Write to disk: two Parquet files + one CSV (mimicking three upstream systems)
parquet_a = tmpdir / "hospital_export.parquet"
parquet_b = tmpdir / "supplementary_cohort.parquet"
csv_c = tmpdir / "legacy_extract.csv"
parquet_static = tmpdir / "demographics.parquet"

src_a.to_parquet(parquet_a, index=False)
src_b.to_parquet(parquet_b, index=False)
src_c.to_csv(csv_c, index=False)
static_df.to_parquet(parquet_static, index=False)

for label, df in [
    ("A (Parquet)", src_a),
    ("B (Parquet)", src_b),
    ("C (CSV)   ", src_c),
]:
    print(f"Source {label}: {df['id'].nunique()} patients")

# %% [markdown]
# Build the admissions pool (multi-source)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Three ``.add_*()`` calls on the same builder merge all rows into one store.
# A fourth call with ``is_static=True`` attaches per-patient demographics.
# Column mapping is omitted; the default ``id``/``start``/``end`` names
# already match the simulation output.

# %%
admissions_path = (
    IntervalSequencePool.builder(sort_anchor="start")
    # Source A: hospital export
    .add_parquet(
        str(parquet_a),
        id_column="id",
        start_column="start",
        end_column="end",
        features=["severity_score", "ward"],
    )
    # Source B: supplementary cohort
    .add_parquet(
        str(parquet_b),
        id_column="id",
        start_column="start",
        end_column="end",
        features=["severity_score", "ward"],
    )
    # Source C: legacy CSV extract
    .add_csv(
        str(csv_c),
        id_column="id",
        start_column="start",
        end_column="end",
        features=["severity_score", "ward"],
        try_parse_dates=True,
    )
    # Static demographics
    .add_parquet(
        str(parquet_static), is_static=True, id_column="id", features=["bmi", "site"]
    ).build("admissions")
)

# %% [markdown]
# Inspect the admissions pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
admissions = IntervalSequencePool(store=admissions_path)
print(admissions)

# %%
print(f"Total patients : {len(admissions)}")
admissions.temporal_data().head(5)

# %%
admissions.static_data().head(5)

# %% [markdown]
# Builder option: ``sort_anchor``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``sort_anchor`` controls how intervals are ordered within each sequence:
# ``"start"`` (default), ``"end"``, or ``"middle"`` (midpoint).
#
# We build all three variants into a dict, then display the same patient's
# sequence under each ordering.

# %%
anchor_pools = {
    anchor: IntervalSequencePool(
        store=IntervalSequencePool.builder(sort_anchor=anchor)
        .add_parquet(
            str(parquet_a),
            id_column="id",
            start_column="start",
            end_column="end",
            features=["severity_score", "ward"],
        )
        .build(f"admissions_{anchor}", exist_ok=True)
    )
    for anchor in ("start", "end", "middle")
}

# %%

# sort_anchor = "start"
pid = admissions.unique_ids[0]
anchor_pools["start"][pid].temporal_data()

# %%

# sort_anchor = "end"
anchor_pools["end"][pid].temporal_data()

# %%

# sort_anchor = "middle"
anchor_pools["middle"][pid].temporal_data()

# %% [markdown]
# Build a procedures pool
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# An :class:`~tanat.sequence.type.event.pool.EventSequencePool` stores
# single-timestamp events.  Two Parquet files are merged into one pool.
#
# Feature schema: ``priority`` (float), ``procedure`` (categorical, values A–E)

# %%
proc_a = simulate_events(n_ids=100, features=PROCEDURE_FEATURES, seed=SEED + 20)
proc_a = _prefix_ids(proc_a, "a")

proc_b = simulate_events(n_ids=80, features=PROCEDURE_FEATURES, seed=SEED + 21)
proc_b = _prefix_ids(proc_b, "b")
parquet_proc_a = tmpdir / "procedures_a.parquet"
parquet_proc_b = tmpdir / "procedures_b.parquet"
proc_a.to_parquet(parquet_proc_a, index=False)
proc_b.to_parquet(parquet_proc_b, index=False)

# %%
procedures_path = (
    EventSequencePool.builder()
    .add_parquet(
        str(parquet_proc_a),
        id_column="id",
        time_column="time",
        features=["priority", "procedure"],
    )
    .add_parquet(
        str(parquet_proc_b),
        id_column="id",
        time_column="time",
        features=["priority", "procedure"],
    )
    .build("procedures")
)

# %%
procedures = EventSequencePool(store=procedures_path)
print(procedures)

# %% [markdown]
# Compose a TrajectoryPool
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# A :class:`~tanat.trajectory.pool.TrajectoryPool` groups multiple sequence
# pools under a shared ID space.  Each pool is registered under an **alias**::
#
#   tpool["admissions"]          → IntervalSequencePool (full pool)
#   tpool[id]                    → Trajectory (one patient)
#   tpool[id]["admissions"]      → IntervalSequence (one sequence)
#   tpool[id]["admissions"][0]   → IntervalEntity (one entity)

# %%
traj_path = (
    TrajectoryPool.builder()
    .add("admissions", admissions)
    .add("procedures", procedures)
    .build("patient_trajectories", exist_ok=True)
)

# %%
tpool = TrajectoryPool(store=traj_path)
print(tpool)
print(f"{len(tpool)} patients with at least one sequence")

# %% [markdown]
# Navigate a single trajectory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
first_id = tpool.unique_ids[0]
traj = tpool[first_id]

print(f"Patient {first_id!r}")
for alias in ["admissions", "procedures"]:
    seq = traj[alias]
    print(f"  {alias:<12}: {len(seq)} rows")

# %%
traj["admissions"].temporal_data().head(3)

# %% [markdown]
# Workspace: reload without tracking paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# All stores are registered in the workspace by name.
# Reload them in any script without knowing the file path.

# %%
print(ws)

# %%
admissions_reloaded = ws["admissions"]
print(f"Reloaded: {len(admissions_reloaded)} patients")

# %% [markdown]
# Save a modified pool
# ~~~~~~~~~~~~~~~~~~~~
#
# ``pool.save()`` materialises any pending lazy transformations into a new
# store registered under the given name.

# %%
admissions.cast_features({"ward": pl.Categorical})

# %%
saved_path = admissions.save("admissions_optimised", overwrite=True)
print("Saved to", saved_path)
