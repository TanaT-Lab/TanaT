"""
Exploring learner activity sequences
======================================

**Scenario:** You have interaction logs from a Moodle LMS and want to
understand how learners engage with course material.

**Concepts covered:**

- Load an event log with :func:`~tanat.dataset.access`
- Detect learning sessions from inactivity gaps
- Build a :class:`~tanat.sequence.StateSequencePool` with :func:`~tanat.build_states`
- Filter sequences by length with :class:`~tanat.criterion.LengthCriterion`
- Visualise action distributions, timelines, and state distributions
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import random

import pandas as pd
import polars as pl

from tanat import build_states
from tanat.criterion import LengthCriterion
from tanat.dataset import access
from tanat.visualization import SequenceVisualizer

# %% [markdown]
# Load and prepare the event log
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :func:`~tanat.dataset.access` returns the MOOC dataset as a pandas
# DataFrame.  Each row is a single learner interaction recorded by a Moodle
# LMS (~100 k events, ~118 learners).

# %%
df = access("mooc_events")
print(f"{len(df)} events  ·  {df['user'].nunique()} learners")
df.head()

# %% [markdown]
# Step 1: Session detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Learning sessions are not labelled in the log.  We define a session as a
# continuous period of activity: a **new session** begins when the same
# learner is idle for more than 2 hours, or when a different user appears.
#
# Each session receives a unique integer id that will serve as the sequence
# identifier in TanaT.

# %%
INACTIVITY = pd.Timedelta("2h")

df["timecreated"] = pd.to_datetime(df["timecreated"])
df = df.sort_values(["user", "timecreated"])
df["session"] = (
    (df["user"] != df["user"].shift()) | (df["timecreated"].diff() > INACTIVITY)
).cumsum()

print(f"Detected {df['session'].nunique()} sessions")

# %%
# Static table: one row per session with the learner identifier.
sessions = df[["user", "session"]].drop_duplicates()

# %% [markdown]
# Step 2: Build the sequence pool
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Each session becomes one sequence.  We use
# :func:`~tanat.build_states` with a **within-session position index**
# as the time axis (0 = first event, 1 = second, …).  This abstracts away
# calendar time and focuses on the order of actions.
#
# The ``sessions`` table (one row per session) is passed as ``static_data``
# so the learner identifier is attached to each sequence.

# %%

# Add a within-session position index.
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

# %%
print(pool)

# %% [markdown]
# Step 3: Filter by length
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The session length distribution is skewed: some outlier sessions contain
# hundreds of events.  We keep sessions with **2 to 40 actions**, which
# covers the majority of learners while removing single-click noise and
# unrealistically long sessions.

# %%
ids_keep = pool.which(LengthCriterion(ge=2, le=40))
pool_filtered = pool.subset(ids_keep)

# %%
print(pool_filtered)

# %% [markdown]
# Step 4: Action distribution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A bar plot shows the frequency of each action type across all sessions,
# giving a first overview of what learners do most.

# %%

# fmt: off
SequenceVisualizer.barplot(sort="descending") \
    .title("Action type distribution") \
    .draw(pool_filtered, entity_feature="Action") \
    .show()
# fmt: on

# %% [markdown]
# Step 5: Sample timeline
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We draw 30 random sessions side by side.  Each row is one session;
# each coloured block is one action at a given position.

# %%
random.seed(42)
sample_ids = random.sample(sorted(pool_filtered.unique_ids), 30)
sample = pool_filtered.subset(sample_ids)

# fmt: off
SequenceVisualizer.timeline() \
    .title("30 random learning sessions") \
    .x_axis(label="Position in session") \
    .draw(sample, entity_feature="Action") \
    .show()
# fmt: on

# %% [markdown]
# Step 6: State distribution over position
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The distribution plot shows how action proportions shift across positions,
# revealing how learners typically start and end their sessions.

# %%

# fmt: off
SequenceVisualizer.distribution(bin_size=1) \
    .title("Action distribution over session progress") \
    .x_axis(label="Position in session") \
    .draw(pool_filtered, entity_feature="Action") \
    .show()
# fmt: on
