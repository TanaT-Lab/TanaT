"""
Spanplot
========

Visualize segment-duration distributions with
:class:`~tanat.visualization.SequenceVisualizer`.

Three chart styles and two grouping dimensions are available:

- ``kind``: ``"box"`` (default), ``"violin"``, or ``"strip"``
- ``group_by``: ``"category"`` (one column per label) or ``"id"`` (one column per sequence)

.. note::
    Compatible with **interval** and **state** pools only.
    Event pools have no duration; passing one raises ``UnsupportedSequenceTypeError``.
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
# per interval. The second feature (``status``) is categorical; it groups the
# duration boxes.

# %%
temporal = simulate_intervals(
    n_ids=80,
    seq_length_range=(4, 15),
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
# Box plot (default)
# ~~~~~~~~~~~~~~~~~~
#
# ``kind="box"`` (default) renders a standard box-and-whisker plot.
# Groups are sorted by ascending median duration.

# %%

# fmt: off
SequenceVisualizer.spanplot(kind="box", display_unit="hours") \
    .title("Duration distribution by status (box)") \
    .y_axis(label="Duration (h)") \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Violin plot
# ~~~~~~~~~~~
#
# ``kind="violin"`` shows the full kernel-density estimate, more informative
# when the distribution is multimodal or skewed.

# %%

# fmt: off
SequenceVisualizer.spanplot(kind="violin", display_unit="hours") \
    .title("Duration distribution by status (violin)") \
    .y_axis(label="Duration (h)") \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Strip plot
# ~~~~~~~~~~
#
# ``kind="strip"`` renders individual points with horizontal jitter,
# ideal for spotting outliers and showing raw data density.

# %%

# fmt: off
SequenceVisualizer.spanplot(kind="strip", display_unit="hours") \
    .title("Duration distribution by status (strip)") \
    .y_axis(label="Duration (h)") \
    .marker(alpha=0.4, point_size=3.5) \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Group by sequence ID
# ~~~~~~~~~~~~~~~~~~~~
#
# ``group_by="id"`` shows one distribution per sequence ID.
# We work on a small subset to keep the chart readable.

# %%
small_pool = pool.subset(ids=pool.unique_ids[:15])

# %%

# Box: one distribution per ID
# fmt: off
SequenceVisualizer.spanplot(group_by="id", kind="box", display_unit="hours") \
    .title("Duration per sequence ID (box)") \
    .y_axis(label="Duration (h)") \
    .x_axis(rotation=45) \
    .colors("tab20") \
    .draw(small_pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Sort order
# ~~~~~~~~~~
#
# ``sort`` controls the group ordering on the x-axis:
#
# - ``"ascending"``: ascending median duration (default)
# - ``"descending"``: descending median duration
# - ``"alphabetic"``: alphabetical label order

# %%

# Descending: largest median first
# fmt: off
SequenceVisualizer.spanplot(kind="box", display_unit="hours", sort="descending") \
    .title("sort='descending': largest median first") \
    .y_axis(label="Duration (h)") \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Horizontal orientation
# ~~~~~~~~~~~~~~~~~~~~~~
#
# ``orientation="horizontal"`` moves group labels to the y-axis,
# especially useful when label names are long.

# %%

# fmt: off
SequenceVisualizer.spanplot(
    kind="box",
    display_unit="hours",
    orientation="horizontal",
) \
    .title("Duration distribution (horizontal box)") \
    .x_axis(label="Duration (h)") \
    .colors("Pastel1") \
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
SequenceVisualizer.spanplot(kind="strip", display_unit="hours") \
    .title(f"Duration distribution, sequence {seq.id_value}") \
    .y_axis(label="Duration (h)") \
    .colors("tab10") \
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
SequenceVisualizer.spanplot(kind="box", display_unit="hours") \
    .facet(by="group", is_static=True, cols=3) \
    .title("Duration distribution faceted by group") \
    .y_axis(label="Duration (h)") \
    .colors("Set2") \
    .draw(pool, entity_feature="status") \
    .show()
# fmt: on

# %% [markdown]
# Inspect ``prepare_data()``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``prepare_data()`` returns the flat Polars DataFrame before rendering.
# Each row is one segment; ``__DURATION__`` holds the computed duration.

# %%
builder = SequenceVisualizer.spanplot(display_unit="hours")
df = builder.prepare_data(pool, entity_feature="status")
df.head()
