"""
Learning clinical phenotypes with SWoTTeD
==========================================

**Scenario:** You want to discover latent *clinical phenotypes*, i.e. recurrent
temporal patterns of medical procedures, directly from raw MIMIC-IV data,
without any supervision.

This tutorial shows a complete end-to-end pipeline:

1. Ingest procedure events from MIMIC-IV into a TanaT
   :class:`~tanat.sequence.EventSequencePool`.
2. Restrict to the most frequent procedure codes to keep the tensor tractable.
3. Call :meth:`~tanat.sequence.base.pool.SequencePool.to_tensor` with
   ``ohe=True`` to obtain a dense ``(N, M, K)`` array alongside patient IDs
   and feature names, all in a single call.
4. Feed the tensor to `SWoTTeD <https://link.springer.com/article/10.1007/s10994-024-06545-8>`_, a
   dictionary-learning model that decomposes the population into *R*
   phenotypes, each with a characteristic temporal signature.
5. Interpret the result using the ``feature_names`` returned by
   :meth:`~tanat.sequence.base.pool.SequencePool.to_tensor`.

.. note::

   SWoTTeD is **not** bundled with TanaT.  Install it separately::

       pip install swotted

   SQL ingestion also requires ``connectorx``::

       pip install 'tanat[sql]'

**Concepts covered:**

- :class:`~tanat.sequence.EventSequencePool` from a SQL source
- Feature frequency filtering with
  :meth:`~tanat.sequence.base.pool.SequencePool.temporal_data`
- :meth:`~tanat.sequence.base.pool.SequencePool.to_tensor` with OHE
- Phenotype interpretation using ``ids`` and ``feature_names``
"""

# %% [markdown]
# Imports
# ~~~~~~~

# %%
import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from swotted import fastSWoTTeDDataset, fastSWoTTeDModule, fastSWoTTeDTrainer

from tanat.criterion import EntityCriterion
from tanat.dataset import access
from tanat.sequence.type.event.pool import EventSequencePool

# %% [markdown]
# Step 1: Discover the top procedure codes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We build a single pool over all procedure codes, count code frequencies,
# then restrict to the **top 30 most frequent codes**.  Keeping only frequent
# codes avoids an extremely sparse OHE tensor
# (352 codes x 92 patients would be ~99% zeros).

# %%
DB = f"sqlite:///{access('mimic4')}"

pool = EventSequencePool(
    store=(
        EventSequencePool.builder()
        .add_sql(
            DB,
            'SELECT subject_id, chartdate, icd_code FROM "hosp/procedures_icd"',
            id_column="subject_id",
            time_column="chartdate",
            features=["icd_code"],
        )
        .build("procedures_store", exist_ok=True)
    )
)

# %%
TOP_K = 30

top_codes = (
    pool.temporal_data()
    .groupby("icd_code")
    .size()
    .sort_values(ascending=False)
    .head(TOP_K)
    .index.tolist()
)
print(f"Retaining {TOP_K} codes out of {pool.temporal_data()['icd_code'].nunique()}")
print("Top codes:", top_codes[:10], "...")

# %% [markdown]
# Step 2: Restrict the pool to the top codes and fix the vocabulary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :meth:`~tanat.sequence.base.pool.SequencePool.filter_entities` with an
# :class:`~tanat.criterion.EntityCriterion` prunes every event row whose
# ``icd_code`` is not in ``top_codes``.
#

# %%
pool.filter_entities(
    EntityCriterion(query=pl.col("icd_code").is_in(top_codes)),
    inplace=True,
)

# pl.Enum fixes the vocabulary to exactly TOP_K codes: OHE will produce one
# column per code, named after the code itself (e.g. "icd_code_02HV33Z").
pool.cast_features({"icd_code": pl.Enum(top_codes)})

print(pool)

# %% [markdown]
# Step 3: Encode as a dense 3-D tensor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :meth:`~tanat.sequence.base.pool.SequencePool.to_tensor` projects every
# patient's procedure history onto a shared daily time axis and returns a
# **3-tuple** ``(arr, ids, feature_names)``:
#
# - ``arr``: shape ``(N, M, K)``, i.e. N patients x M daily bins x K OHE codes.
# - ``ids``: the N patient identifiers aligned with axis 0.
# - ``feature_names``: the K column labels aligned with axis 2.
#
# ``ohe=True`` one-hot encodes ``icd_code`` in-place; ``fill_value=0``
# replaces empty bins with zeros (no procedure recorded that day).

# %%
BIN_SIZE = "1D"  # one bin per calendar day
MAX_BINS = 90  # cap at 90 days (covers > 95 % of stays)

arr, ids, feature_names = pool.to_tensor(
    features="icd_code",
    bin_size=BIN_SIZE,
    max_bins=MAX_BINS,
    fill_value=0,
    ohe=True,
)

print(f"Tensor shape : {arr.shape}")  # (N, MAX_BINS, K)
print(f"Patients     : {len(ids)}")
print(f"Features     : {len(feature_names)}")
print(f"Sparsity     : {(arr == 0).mean():.4%} empty bins")

# %% [markdown]
# Step 4: Prepare the tensor for SWoTTeD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# SWoTTeD's :class:`~swotted.fastSWoTTeDModule` expects a tensor of shape
# ``(N, K, T)``
# ``(N, K, T)`` : patients × features × time
# The :meth:`to_tensor` returns ``(N, T, K)``.
# A single ``transpose`` aligns the axes.

# %%

# (N, T, K)  →  (N, K, T)
X = torch.from_numpy(arr.transpose(0, 2, 1).astype(np.float32))
print(f"SWoTTeD input shape : {X.shape}")  # (N, K, MAX_BINS)

# %% [markdown]
# Step 5: Train SWoTTeD
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We search for ``R = 5`` phenotypes, each described by a temporal window of
# ``Tw = 7`` days.  Training runs for 50 epochs on CPU, fast enough on this
# small cohort.

# %%
R = 5  # number of phenotypes to discover
Tw = 7  # temporal window width (days)

N_patients, K_codes, T_days = X.shape

swotted_cfg = OmegaConf.create(
    {
        "model": {
            "non_succession": True,
            "sparsity": 0.1,
            "rank": R,
            "twl": Tw,
            "N": K_codes,
            "metric": "Bernoulli",  # binary OHE data → Bernoulli loss
        },
        "training": {
            "batch_size": N_patients,
            "nepochs": 50,
            "lr": 1e-2,
        },
        "predict": {
            "nepochs": 20,
            "lr": 1e-2,
        },
    }
)

device = torch.device("cpu")
model = fastSWoTTeDModule(swotted_cfg).to(device)

loader = DataLoader(
    fastSWoTTeDDataset(X.to(device)),
    batch_size=N_patients,
    shuffle=False,
    collate_fn=lambda x: x,
)

trainer = fastSWoTTeDTrainer(
    fast_dev_run=False,
    max_epochs=swotted_cfg.training.nepochs,
    accelerator="cpu",
)
trainer.fit(model=model, train_dataloaders=loader)

# %% [markdown]
# Step 6: Extract and interpret the learned phenotypes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :meth:`~swotted.fastSWoTTeDModule.reorderPhenotypes` reorders the *R*
# phenotypes by their activation strength.
# Each phenotype is a ``(K, Tw)`` matrix, i.e. a temporal signature over
# the ``K`` procedure codes.
#
# ``feature_names`` returned by :meth:`to_tensor` gives us the code label for
# each row, so we can read off *which procedures* drive each phenotype and
# *when* during the window they tend to occur.

# %%
phenotypes, pathways = model.reorderPhenotypes(model.Ph.detach().cpu(), tw=Tw)
phenotypes = phenotypes.detach().numpy()  # (R, K, Tw): temporal signature per phenotype
pathways = (
    pathways.detach().numpy()
)  # (N, R, T'): activation of each phenotype per patient

print(f"Phenotypes shape : {phenotypes.shape}")
print(f"Pathways shape   : {pathways.shape}")

# %%
# For each phenotype, print the top-3 most active procedure codes.
# ``phenotypes`` has shape (R, K, Tw): sum over the time axis
# to get the overall "weight" of each code in each phenotype.
code_weights = phenotypes.sum(axis=-1)  # (R, K)

print("\nTop procedures per phenotype:")
for r in range(R):
    top_idx = np.argsort(code_weights[r])[::-1][:3]
    # Strip the "icd_code_" prefix added by OHE for readability.
    top_codes_r = [feature_names[i].removeprefix("icd_code_") for i in top_idx]
    print(f"  Phenotype {r + 1}: {top_codes_r}")

# %% [markdown]
# Step 7: Assign phenotypes back to patient IDs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``ids`` (returned alongside the tensor) maps axis-0 indices back to
# patient identifiers, so downstream analyses can join phenotype assignments
# to any other pool or dataframe.

# %%
# ``pathways`` shape is (N, R, T'); take argmax over R (axis=1) to get the
# dominant phenotype index for each patient × time bin, then keep the
# most frequent dominant phenotype across time bins (majority vote).
from scipy.stats import mode

dominant_per_patient = mode(pathways.argmax(axis=1), axis=1).mode  # shape: (N,)

patient_phenotypes = dict(zip(ids, dominant_per_patient.tolist()))
print("\nPatient -> dominant phenotype (first 10):")
for pid, ph in list(patient_phenotypes.items())[:10]:
    print(f"  {pid} -> phenotype {ph + 1}")
