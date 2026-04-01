"""
Barplot
=======

Aggregate entity features across a :class:`~tanat.sequence.SequencePool` with
:class:`~tanat.visualization.SequenceVisualizer`.

Three aggregation modes are available:

- ``show_as="count"``: raw occurrences per label (all pool types)
- ``show_as="rate"``: relative frequency, bars sum to 1 (all pool types)
- ``show_as="duration"``: total cumulated duration per label (interval / state pools only)
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import polars as pl

from tanat import build_intervals
from tanat.dataset import simulate_intervals, simulate_static
from tanat.visualization import SequenceVisualizer

# %% [markdown]
# Simulate data
# ~~~~~~~~~~~~~
#
# :func:`~tanat.dataset.simulation.intervals.simulate_intervals` produces one row
# per interval. The second feature (``status``) is categorical; it groups the bars.

# %%
temporal = simulate_intervals(
    n_ids=80,
    seq_length_range=(3, 12),
    features=["value", "status"],
    seed=42,
)
print(temporal.shape, temporal.columns.tolist())

# %%
temporal.head()

# %% [markdown]
# Build the pool
# ~~~~~~~~~~~~~~

# %%
pool = build_intervals(
    temporal_data=temporal,
    id_column="id",
    start_column="start",
    end_column="end",
)

# %%
pool.cast_features({"status": pl.Categorical}, is_static=False)
print(pool)

# %% [markdown]
# Count: occurrences per label
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``show_as="count"`` (default) counts how many intervals carry each label.

# %%

# fmt: off
SequenceVisualizer.barplot(show_as="count") \
    .title("Interval count by status") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Rate: relative frequency
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``show_as="rate"`` normalises counts so bars sum to 1.
# Combine with ``sort="descending"`` to put the most frequent label first.

# %%

# fmt: off
SequenceVisualizer.barplot(show_as="rate", sort="descending") \
    .title("Relative frequency by status (descending)") \
    .y_axis(label="Rate") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Duration: total time per label
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``show_as="duration"`` sums ``end − start`` per label.
# ``display_unit`` converts the result to a human-readable time unit.
#
# .. note::
#    Duration mode requires an interval or state pool.
#    Event pools (point observations) have no duration.

# %%

# fmt: off
SequenceVisualizer.barplot(show_as="duration", display_unit="hours") \
    .title("Total duration per status (hours)") \
    .y_axis(label="Hours") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Horizontal orientation
# ~~~~~~~~~~~~~~~~~~~~~~
#
# ``orientation="horizontal"`` flips the axes, handy when label names are long.

# %%

# fmt: off
SequenceVisualizer.barplot(
    show_as="count",
    orientation="horizontal",
    sort="descending",
) \
    .title("Interval count by status (horizontal)") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Color customization
# ~~~~~~~~~~~~~~~~~~~
#
# The ``.colors()`` method accepts three formats:
#
# - **Named colormap** string: ``"Set2"``, ``"tab10"``, ``"Pastel1"``, …
# - **Dict** mapping label → hex color
# - **No argument** (default): matplotlib default color cycle

# %%

# Named colormap
# fmt: off
SequenceVisualizer.barplot(show_as="count") \
    .colors("Set2") \
    .title("Count (Set2 palette)") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %%

# Explicit dict: one color per label
palette = {
    "A": "#2ecc71",
    "B": "#e74c3c",
    "C": "#3498db",
    "D": "#f39c12",
    "E": "#9b59b6",
}

# %%

# fmt: off
SequenceVisualizer.barplot(show_as="count") \
    .colors(palette) \
    .title("Count (custom dict palette)") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Single sequence
# ~~~~~~~~~~~~~~~
#
# Pass a :class:`~tanat.sequence.Sequence` directly for a per-individual view.

# %%
seq = pool[pool.unique_ids[0]]
print(f"ID {seq.id_value}: {len(seq)} intervals")

# %%

# fmt: off
SequenceVisualizer.barplot(show_as="count") \
    .title(f"Status counts, sequence {seq.id_value}") \
    .colors("Set2") \
    .draw(seq, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Layout and style
# ~~~~~~~~~~~~~~~~

# %%

# Grid + capped y-axis
# fmt: off
SequenceVisualizer.barplot(show_as="rate", sort="descending") \
    .figsize(8, 4) \
    .grid() \
    .x_axis(rotation=30) \
    .y_axis(limit_max=1, label="Rate") \
    .colors("Set2") \
    .title("Rate (grid, capped y-axis)") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %%

# Slim bars with a visible edge
# fmt: off
SequenceVisualizer.barplot(show_as="count") \
    .colors("Set2") \
    .marker(bar_width=0.5, alpha=0.85, edge_color="#333333") \
    .title("Count (slim bars with edge)") \
    .draw(pool, entity_feature="status") \
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
SequenceVisualizer.barplot(show_as="count") \
    .facet(by="group", is_static=True, cols=3) \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Inspect ``prepare_data()``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``prepare_data()`` returns the aggregated Polars DataFrame before rendering.
# The result is cached: calling ``.draw()`` on the same builder reuses it.

# %%
builder = SequenceVisualizer.barplot(show_as="rate", sort="descending")
df = builder.prepare_data(pool, entity_feature="status")
df
