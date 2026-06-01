"""
Custom Sequence Metric
======================

Learn how to implement a custom sequence-level distance metric by subclassing
:class:`~tanat.metric.sequence.base.SequenceMetric`.

A sequence metric computes a scalar distance between two sequences and can
produce a full pairwise :class:`~tanat.metric.DistanceMatrix` over a pool.

**Minimal contract**

1. Declare a ``SETTINGS_CLASS`` dataclass (use ``tanat_utils.settings_dataclass``).
2. Implement ``_compute(seq_a, seq_b)``: return a non-negative ``float``.
3. Implement ``validate_composition(seq_a, seq_b)``: raise early if the metric
   is incompatible with the given sequences (feature mismatch, wrong type, …).

The base class handles ``__call__``, ``compute_matrix``, and
``compute_cross_matrix`` automatically.

.. note::
   For performance-critical cases you can additionally override
   ``_compute_matrix_impl`` with a vectorised or Numba-backed kernel.
   This is not required for a minimal working implementation.
"""

# %% [markdown]
# Setup
# -----

# %%
from tanat_utils import settings_dataclass as dataclass

from tanat import build_states
from tanat.dataset import simulate_states
from tanat.metric.sequence.base import SequenceMetric

# %% [markdown]
# Data
# ----

# %%
raw = simulate_states(n_ids=30, features=["status"], seed=42)
pool = build_states(
    temporal_data=raw, id_column="id", start_column="start", end_column="end"
)

# %%
print(pool)

# %% [markdown]
# Define a custom sequence metric
# --------------------------------
#
# We implement a **length-difference** metric: the distance between two
# sequences is the absolute difference of their lengths (number of entities).
# This intentionally simple metric requires no entity-level building block
# and illustrates the minimal implementation pattern.


# %%
@dataclass
class LengthDiffSettings:
    """Settings for :class:`LengthDiffSequenceMetric`.

    Args:
        normalize: If ``True``, divide by the length of the longer sequence
            so that the result is in ``[0, 1]``.
    """

    normalize: bool = False


class LengthDiffSequenceMetric(SequenceMetric, register_name="length_diff"):
    """Distance metric based on the difference in sequence lengths.

    ``dist(seq_a, seq_b) = |len(seq_a) - len(seq_b)|``

    With ``normalize=True``:
    ``dist = |len(seq_a) - len(seq_b)| / max(len(seq_a), len(seq_b))``
    """

    SETTINGS_CLASS = LengthDiffSettings

    def __init__(self, normalize: bool = False) -> None:
        super().__init__(settings=LengthDiffSettings(normalize=normalize))

    # ------------------------------------------------------------------
    # Required implementations
    # ------------------------------------------------------------------

    def validate_composition(self, seq_a, seq_b=None) -> None:
        """No entity-feature constraint: any sequence type is accepted."""
        # Length is always available, nothing to validate here.

    def _compute(self, seq_a, seq_b) -> float:
        """Absolute (or normalised) difference of sequence lengths."""
        la, lb = len(seq_a), len(seq_b)
        diff = abs(la - lb)
        if self.settings.normalize:
            denom = max(la, lb)
            return float(diff / denom) if denom > 0 else 0.0
        return float(diff)


# %% [markdown]
# Instantiate and inspect
# -----------------------

# %%
metric = LengthDiffSequenceMetric(normalize=True)
print(metric)

# %% [markdown]
# Compute a distance between two sequences
# -----------------------------------------

# %%
ids = pool.unique_ids
seq_a = pool[ids[0]]
seq_b = pool[ids[1]]

print(f"len(seq_a) = {len(seq_a)},  len(seq_b) = {len(seq_b)}")
print(f"distance   = {metric(seq_a, seq_b):.4f}")

# %% [markdown]
# Compute a full pairwise distance matrix
# ----------------------------------------
#
# ``compute_matrix`` is inherited from the base class and works out of the
# box once ``_compute`` is implemented.

# %%
dm = metric.compute_matrix(pool)
dm.to_frame().head()
