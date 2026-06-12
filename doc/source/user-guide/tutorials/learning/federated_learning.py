"""
Federated learning with Fed-BioMed
==================================

**Scenario :** Several hospitals hold patient observation records.
They want to train a shared model **without ever sharing their raw data**.

This tutorial walks through the full pipeline in two stages:

**Stage A · Local prototype**

1. Generate a synthetic multi-centre dataset with ``FedPatientJourney``
2. Build a TanaT event pool
3. Convert patient trajectories into temporal tensors
4. Train a tiny PyTorch model locally (just to validate the pipeline)

**Stage B · Federated deployment**

5. Move the preprocessing into a Fed-BioMed ``TrainingPlan``
6. Register each centre's data on its own node
7. Run a federated ``Experiment`` with ``FedAverage``
"""

# %% [markdown]
# Stage A · Local prototype
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# %% [markdown]
# 1 · Imports
# -----------

# %%
import polars as pl
import torch
import torch.nn as nn

from tanat_synthea import FedPatientJourney
from tanat.sequence.shortcuts import build_events

# %% [markdown]
# 2 · Generate synthetic patient journeys
# ---------------------------------------
#
# ``FedPatientJourney.demo()`` simulates up 3 centres (US / FR / GB,
# 60 patients each) and runs Synthea to produce realistic patient records.

# %%

# Generates 3 centres with 60 patients each
fpj = FedPatientJourney.demo()
fpj.generate()

# %%
print(fpj)

# %% [markdown]
# 3 · Inspect one centre
# ----------------------
#
# In a real federated setup each centre lives on a separate node.
# Here we work with centre 0 locally to validate the full pipeline
# before federating.

# %%
center = fpj.center(0)

static_data = center.static_data()
observation_data = center.temporal_data(record_types=["observation"])

print("Static data  :", static_data.shape)
print("Observations :", observation_data.shape)

# %% [markdown]
# 4 · Build a TanaT event pool
# ----------------------------
#
# ``build_events`` converts the raw observation table into a structured
# :class:`~tanat.sequence.EventSequencePool` indexed by patient and time.

# %%
pool = build_events(
    observation_data,
    id_column="PATIENT",
    time_column="DATE",
    static_data=static_data,
    store_name="observations",
)

# %%
print(pool)

# %%
# Temporal data (one row per observation event)
print(pool.temporal_data()[["PATIENT", "DATE", "CODE"]].head())

# %%
# Static data (one row per patient)
print(pool.static_data()[["PATIENT", "GENDER", "BIRTHDATE"]].head())

# %% [markdown]
# 5 · Convert events into a temporal tensor
# -----------------------------------------
#
# :meth:`~tanat.sequence.base.pool.SequencePool.to_tensor` projects each
# patient's history onto a shared daily time axis and returns a 3-tuple
# ``(tensor, patient_ids, feature_names)``.
#
# Tensor dimensions:
#
# - **N** : patients
# - **M** : time bins (one per calendar day, capped at ``max_bins``)
# - **K** : one-hot encoded observation codes
#
# We keep only the top 30 most frequent codes before converting to a tensor.
# This keeps the tutorial manageable and avoids an overly sparse OHE vocabulary
# while preserving the most common observation patterns.

# %%
TOP_K = 30
top_codes = (
    pool.temporal_data()
    .groupby("CODE")
    .size()
    .sort_values(ascending=False)
    .head(TOP_K)
    .index.tolist()
)
print(f"Retaining {TOP_K} codes out of {pool.temporal_data()['CODE'].nunique()}")
print("Top codes:", top_codes[:10], "...")

# %%
from tanat.criterion import EntityCriterion

pool.filter_entities(
    EntityCriterion(query=pl.col("CODE").is_in(top_codes)),
    inplace=True,
)

# %%
BIN_SIZE = "360D"  # one bin = 360 days

pool.cast_features({"CODE": pl.Categorical})
tensor, patient_ids, feature_names = pool.to_tensor(
    features="CODE",
    bin_size=BIN_SIZE,
    fill_value=0,
    ohe=True,
)

N, M, K = tensor.shape
print(f"Tensor shape : {tensor.shape}")
print(f"Patients   : {N}")
print(f"Time bins  : {M}")
print(f"Features   : {K}")
print(f"Sparsity   : {(tensor == 0).mean():.1%} empty bins")
print(f"Non-zero cells : {(tensor != 0).sum()}")
print(f"Patients with events : {(tensor.sum(axis=(1, 2)) != 0).sum()} / {N}")

# %% [markdown]
# 6 · Build a learning-ready dataset
# ----------------------------------
#
# We collapse the time axis by **averaging** across bins. A quick way to
# turn the 3D tensor into a flat feature matrix.
#
# A real model could consume the full ``(N, M, K)`` tensor.
# This step is intentionally simplified.


# %%
X = tensor.mean(axis=1)  # (N, M, K) → (N, K)

# Binary target: male = 1, female = 0
y_df = pool.apply(
    exprs=(pl.col("GENDER") == "M").cast(pl.Int8).alias("target"),
    is_static=True,
)
y = y_df["target"].to_numpy()

print("X shape :", X.shape)
print("y shape :", y.shape)
print(f"Class balance : {y.mean():.0%} positive")

# %% [markdown]
# 7 · Train a local model
# -----------------------
#
# A three-layer MLP trained for 10 epochs to verify that the tensor feeds
# cleanly into a standard PyTorch loop.

# %%
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(K, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid(),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(10):
    optimizer.zero_grad()
    pred = model(X_t).squeeze()
    loss = loss_fn(pred, y_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1:2d}  loss={loss.item():.4f}")

# %% [markdown]
# Stage B · Federated deployment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# This stage shows how the same preprocessing pipeline is executed on each
# node locally, while only model weights are shared.
#
# .. tip::
#
#    **Key concept: what travels on the wire?**
#    In Fed-BioMed only the **model weights** move between nodes and the
#    researcher. Raw data never leaves the hospital.
#
# .. code-block:: text
#
#           ┌───────────────────────────────────────┐
#           │             RESEARCHER                │
#           │        1) defines the model           │
#           │    2) aggregates weights (FedAvg)     │
#           └───────────────────────────────────────┘
#                    ▲          ▲          ▲
#                    │          │          │
#              weights only: raw data stays local
#                    │          │          │
#          ┌─────────┘      ┌───┘          └──┐
#          ▼                ▼                 ▼
#   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
#   │  Node 0 · US  │  │  Node 1 · FR  │  │  Node 2 · GB  │
#   │  local data   │  │  local data   │  │  local data   │
#   │      +        │  │      +        │  │      +        │
#   │local training │  │local training │  │local training │
#   └───────────────┘  └───────────────┘  └───────────────┘
#
# .. note::
#
#    Prerequisites:
#
#    .. code-block:: bash
#
#       pip install fedbiomed tanat tanat-synthea
#
#       # one terminal per node
#       fedbiomed node -p fbm-node-0 start
#       fedbiomed node -p fbm-node-1 start
#       fedbiomed node -p fbm-node-2 start
#
#       # researcher (this notebook)
#       fedbiomed researcher start

# %% [markdown]
# 8 · Register data on each node
# ------------------------------
#
# On each node terminal, declare the local data with a shared **tag**.
# The tag is how the Researcher discovers which nodes hold compatible data.
#
# .. code-block:: bash
#
#    # Node 0 terminal
#    fedbiomed node -p fbm-node-0 dataset add \\
#        --name "tanat_observations" \\
#        --tags tanat-obs \\
#        --data-type csv \\
#        --path /path/to/center_0/
#
#    # Node 1 terminal  (same tag, different path)
#    fedbiomed node -p fbm-node-1 dataset add \\
#        --name "tanat_observations" \\
#        --tags tanat-obs \\
#        --data-type csv \\
#        --path /path/to/center_1/
#
#    # Node 2 terminal
#    fedbiomed node -p fbm-node-2 dataset add \\
#        --name "tanat_observations" \\
#        --tags tanat-obs \\
#        --data-type csv \\
#        --path /path/to/center_2/

# %% [markdown]
# 9 · Define the TrainingPlan
# ---------------------------
#
# A ``TorchTrainingPlan`` wraps:
#
# - ``init_model``: the PyTorch model (identical to Stage A)
# - ``init_optimizer``: the optimiser
# - ``init_dependencies``: imports that must be available on each node
# - ``training_data``: **the TanaT preprocessing pipeline from Stage A**
# - ``training_step``: the loss computation
#
# Here the new part is simple: each node reads its own local data,
# applies the same preprocessing, and returns a training-ready dataset.
# Raw data never leaves the node; only model weights are aggregated.

# %% [markdown]
# .. code-block:: python
#
#    from fedbiomed.common.training_plans import TorchTrainingPlan
#    from fedbiomed.common.data import DataManager
#    import polars as pl
#    import torch
#    import torch.nn as nn
#    import torch.nn.functional as F
#
#    from tanat.sequence.shortcuts import build_events
#
#    class TanaTObsTrainingPlan(TorchTrainingPlan):
#
#        # ------------------------------------------------------------------
#        # Model (same MLP as Stage A)
#        # ------------------------------------------------------------------
#        def init_model(self, model_args: dict):
#
#            in_dim = model_args["in_features"]
#
#            class MLP(nn.Module):
#                def __init__(self, in_dim):
#                    super().__init__()
#                    self.net = nn.Sequential(
#                        nn.Linear(in_dim, 32),
#                        nn.ReLU(),
#                        nn.Linear(32, 1),
#                        nn.Sigmoid(),
#                    )
#                def forward(self, x):
#                    return self.net(x).squeeze(-1)
#
#            return MLP(in_dim)
#
#        # ------------------------------------------------------------------
#        # Optimiser
#        # ------------------------------------------------------------------
#        def init_optimizer(self, optimizer_args: dict):
#            return torch.optim.Adam(
#                self.parameters(),
#                lr=optimizer_args.get("lr", 1e-3),
#            )
#
#        # ------------------------------------------------------------------
#        # Node-side dependencies
#        # ------------------------------------------------------------------
#        def init_dependencies(self):
#            return [
#                "import polars as pl",
#                "from tanat.sequence.shortcuts import build_events",
#            ]
#
#        # ------------------------------------------------------------------
#        # Data loading (runs locally on each node/TanaT pipeline from Stage A)
#        # ------------------------------------------------------------------
#        def training_data(self):
#
#            center = self.dataset_path   # path injected by Fed-BioMed
#
#            static_data = center.static_data()
#            observation_data = center.temporal_data(record_types=["observation"])
#
#            pool = build_events(
#                observation_data,
#                id_column="PATIENT",
#                time_column="DATE",
#                static_data=static_data,
#            )
#
#            pool.cast_features({"CODE": pl.Categorical})
#
#            tensor, _, _ = pool.to_tensor(
#                features="CODE",
#                bin_size="1D",
#                max_bins=90,
#                fill_value=0,
#                ohe=True,
#            )
#
#            X = tensor.mean(axis=1)    # (N, M, K) → (N, K)
#
#            y_df = pool.apply(
#                exprs=(pl.col("GENDER") == "M").cast(pl.Int8).alias("target"),
#                is_static=True,
#            )
#            y = y_df["target"].to_numpy()
#
#            return DataManager(
#                dataset=torch.tensor(X, dtype=torch.float32),
#                target=torch.tensor(y, dtype=torch.float32),
#            )
#
#        # ------------------------------------------------------------------
#        # Loss
#        # ------------------------------------------------------------------
#        def training_step(self, data, target):
#            return F.binary_cross_entropy(self.forward(data), target)

# %% [markdown]
# 10 · Run the federated experiment
# ---------------------------------
#
# .. code-block:: python
#
#    from fedbiomed.researcher.federated_workflows import Experiment
#    from fedbiomed.researcher.aggregators.fedavg import FedAverage
#
#    exp = Experiment(
#        tags=["tanat-obs"],
#        model_args={"in_features": K},          # K from Stage A
#        training_plan_class=TanaTObsTrainingPlan,
#        training_args={
#            "loader_args": {"batch_size": 16},
#            "optimizer_args": {"lr": 1e-3},
#            "epochs": 5,
#        },
#        round_limit=10,
#        aggregator=FedAverage(),
#    )
#
#    exp.run()
#
#    # Inspect results
#    final_model = exp.training_plan().model()
#    final_model.eval()
#
#    # Loss per round (averaged over nodes)
#    for round_id, replies in exp.training_replies().items():
#        losses = [r["loss"] for r in replies.values() if "loss" in r]
#        if losses:
#            print(f"Round {round_id:2d}  avg loss = {sum(losses) / len(losses):.4f}")
