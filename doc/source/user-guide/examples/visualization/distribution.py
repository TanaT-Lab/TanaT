"""
Distribution
============

Visualize state-occupancy distributions over time with
:class:`~tanat.visualization.SequenceVisualizer`.

Each time bin shows how many (or what fraction of) sequences occupy each state
at that moment, using **occupancy-based binning**: a state segment contributes
to every bin it overlaps.

.. note::
    Compatible with **state** pools only.
    Other pool types raise ``UnsupportedSequenceTypeError``.
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl

from tanat import build_states
from tanat.dataset import simulate_states, simulate_static
from tanat.visualization import SequenceVisualizer

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# :func:`~tanat.dataset.simulation.states.simulate_states` produces strictly
# contiguous states (``end[i] == start[i+1]``). The second feature (``status``)
# is categorical; it labels the occupancy areas.

# %%
temporal = simulate_states(
    n_ids=80,
    seq_length_range=(4, 12),
    features=["value", "status"],
    seed=42,
)
print(temporal.shape, temporal.columns.tolist())

# %%
temporal.head()

# %% [markdown]
# Build the pool
# ~~~~~~~~~~~~~~
#
# :func:`~tanat.sequence.shortcuts.build_states` accepts an explicit ``end_column``
# when the data is already contiguous; :func:`~tanat.dataset.simulate_states`
# always guarantees.

# %%
pool = build_states(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
)

# %%
pool.cast_features({"status": pl.Categorical}, is_static=False)
print(pool)

# %% [markdown]
# Default: percentage stacked area
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``mode="percentage"`` (default) renders each state's share summing to 100%
# per bin, the classic state-sequence distribution chart.

# %%

# fmt: off
SequenceVisualizer.distribution(mode="percentage", bin_size="1mo") \
    .title("State distribution over time (%, monthly bins)") \
    .x_axis(label="Date", rotation=30, autofmt_xdate=True) \
    .y_axis(label="% of sequences") \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Modes: count, proportion, percentage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. list-table::
#    :header-rows: 1
#
#    * - ``mode=``
#      - y-axis
#      - Sum per bin
#    * - ``"count"``
#      - raw number of sequences
#      - varies
#    * - ``"proportion"``
#      - fraction (0–1)
#      - 1.0
#    * - ``"percentage"``
#      - percent (0–100)
#      - 100

# %%

# Raw counts
# fmt: off
SequenceVisualizer.distribution(mode="count", bin_size="3mo") \
    .title("State distribution: count (3-month bins)") \
    .x_axis(label="Date", rotation=30, autofmt_xdate=True) \
    .y_axis(label="Number of sequences") \
    .colors("tab10") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %%

# Proportion (0-1 scale)
# fmt: off
SequenceVisualizer.distribution(mode="proportion", bin_size="3mo") \
    .title("State distribution: proportion (3-month bins)") \
    .x_axis(label="Date", rotation=30, autofmt_xdate=True) \
    .y_axis(label="Proportion") \
    .colors("Pastel1") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Bin size
# ~~~~~~~~
#
# ``bin_size`` accepts any Polars duration string: ``"1d"``, ``"1w"``,
# ``"1mo"``, ``"1y"``. Use coarser bins for long time horizons, finer
# bins for detailed short-term patterns.

# %%

# Yearly bins, smoothest view for long horizons
# fmt: off
SequenceVisualizer.distribution(mode="percentage", bin_size="1y") \
    .title("State distribution (1-year bins)") \
    .x_axis(label="Year", rotation=30) \
    .y_axis(label="% of sequences") \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Flat (unstacked) fills
# ~~~~~~~~~~~~~~~~~~~~~~
#
# ``stacked=False`` renders overlapping transparent fills, easier to compare
# individual state curves without stacking artefacts.

# %%

# fmt: off
SequenceVisualizer.distribution(mode="proportion", bin_size="1mo", stacked=False) \
    .title("State distribution: flat fills (monthly bins)") \
    .x_axis(label="Date", rotation=30, autofmt_xdate=True) \
    .y_axis(label="Proportion") \
    .marker(alpha=0.4) \
    .colors("tab10") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Single sequence
# ~~~~~~~~~~~~~~~
#
# Pass a :class:`~tanat.sequence.Sequence` directly to inspect
# one individual's state occupancy.

# %%
seq = pool[pool.unique_ids[0]]
print(f"ID {seq.id_value}: {len(seq)} states")

# %%

# fmt: off
SequenceVisualizer.distribution(mode="count", bin_size="1mo") \
    .title(f"State occupancy, sequence {seq.id_value}") \
    .x_axis(label="Date", rotation=30, autofmt_xdate=True) \
    .y_axis(label="Occupied") \
    .colors("Set2") \
    .draw(seq, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Faceting
# ~~~~~~~~
#
# ``.facet()`` splits the chart into a grid of panels, one per unique value
# of a chosen feature. Here we attach per-sequence static data and facet on
# ``group``.

# %%
static_df = simulate_static(n_ids=80, features=["age", "group"], seed=0)
pool.add_static_features(static_df)
pool.cast_features({"group": pl.Categorical}, is_static=True)

# %%

# fmt: off
SequenceVisualizer.distribution(mode="percentage", bin_size="1mo") \
    .facet(by="group", is_static=True, cols=3) \
    .title("State distribution faceted by group") \
    .x_axis(label="Date", rotation=30, autofmt_xdate=True) \
    .y_axis(label="% of sequences") \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Inspect ``prepare_data()``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``prepare_data()`` returns the binned Polars DataFrame before rendering.
# Columns: ``__BIN__``, ``__LABEL__``, ``__VALUE__``, and optionally ``__COLOR__``.

# %%
builder = SequenceVisualizer.distribution(mode="percentage", bin_size="1mo")
df = builder.prepare_data(pool, entity_feature="status")
print(f"Shape: {df.shape}, schema: {dict(df.schema)}")
df.head(10)
