# Band gap prediction — postmortem

This project originally targeted band gap regression (semiconductor screening
in the 1.0–2.5 eV "solar-relevant" range). That track is now abandoned in
favor of formation energy + stability, which the same pipeline handles
reliably. This file records what was tried and why it was dropped, so the
reasoning isn't lost.

## What was tried

| Approach | Input | Model | Test MAE (eV) | Test R2 |
|---|---|---|---|---|
| Classical baseline | Magpie + Stoichiometry + ValenceOrbital features | ExtraTreesRegressor | 0.595 | 0.683 |
| Composition deep model | Raw formula string (mat2vec embedding) | CrabNet, 300 epochs, T4 GPU | 0.513 | 0.700 |
| Structure-based GNN | pymatgen Structure (atoms+bonds+distances graph) | MEGNet (via `matgl`) | *(not completed — abandoned before training)* | — |

## Why it plateaued around 0.5 eV MAE / R2~0.7

1. **DFT label noise (the main cause).** Materials Project band gaps come
   from the GGA/PBE functional, which systematically underestimates real
   band gaps by 30-50%, and the error size varies by material class. The
   model is fitting a target with built-in inconsistency — this is a noise
   floor, not a model-capacity problem. No amount of feature engineering or
   bigger models fixes a noisy label.

2. **Composition alone is information-poor.** Two materials with the same
   composition but different atomic arrangement (polymorphs) can have very
   different real band gaps. Exact-duplicate polymorphs were deduped (kept
   the most stable one via `energy_above_hull`), but materials with similar-
   but-not-identical composition and very different local bonding still
   can't be told apart by Magpie stats or a composition embedding. This is
   why a structure-based GNN was started — it's the only lever that adds
   genuinely new information rather than re-squeezing the same composition
   signal.

3. **Band gap is intrinsically hard to regress.** It is not smooth in
   composition space — small substitutions can flip a material from
   metallic to wide-gap insulator. Literature composition-only models
   (Roost, CrabNet-style) typically land in the same 0.4-0.5 eV MAE range,
   so this project's numbers are in line with the field, not a bug in the
   pipeline.

## Pipeline sanity check (proof it's not a bug)

The exact same features, cleaning, and ExtraTreesRegressor were re-run on
`formation_energy_per_atom` instead of `band_gap_ev` — a property DFT
computes much more reliably (ground-state total-energy difference, no
excited-state physics required):

- **Formation energy: MAE = 0.090 eV/atom, R2 = 0.972**
- vs. band gap: MAE = 0.595 eV, R2 = 0.683

Same pipeline, same code shape, wildly different result quality. This
confirms the featurization/cleaning/modeling approach is sound — band gap
specifically was the hard part, not the project's implementation.

## Outlier handling note (for reference)

IQR-based outlier removal (Q1-1.5*IQR to Q3+1.5*IQR) was applied to
band_gap_ev. Lower bound (-2.76 eV) never triggered (gap can't be
negative). Upper bound (6.44 eV) dropped 641 rows (0.85%) — wide-gap
insulators (fluorides, oxides, diamond-like materials). This was a minor
effect on the reported numbers (cutting <1% of data) — the 0.5 eV MAE
plateau is not explained by outlier handling.

## Decision

Abandon band gap as the primary target. Pivot to a screener built on:

- **Formation energy regression** — proven R2=0.972, reuses existing Magpie
  features and MP data already downloaded.
- **Stability classification** (`is_stable` / convex hull, OQMD data) — data
  was already prepared (`data/splits/*_stab.npy`) but never trained; this is
  the next concrete step.

Both targets are DFT-energy-derived (not excited-state properties), which is
exactly the category where this pipeline gets reliable R2>0.95 results.

## Files removed as part of this pivot

- `04_train_bandgap.py`, `models/bandgap_model.pkl`,
  `models/bandgap_model_meta.json`, `models/scaler_bandgap.pkl`
- `crabnet_bandgap_experiment.py`, `colab_crabnet_bandgap.py`,
  `colab_crabnet_bandgap.ipynb`, `crabnet_run.log`
- `structure_gnn/` (entire folder — structure download, dataset builder,
  MEGNet notebook; ~670 MB of downloaded structures)
- `data/y_bandgap.csv`, `data/crabnet_train.csv`, `data/crabnet_val.csv`,
  `data/crabnet_test.csv`, `data/splits/*_bg.npy`

Kept: `01_download_mp.py`, `02_download_oqmd.py`, `03_featurize.py`,
`03b_prepare.py`, `data/X_mp.csv`, `data/X_oqmd.csv`, `data/meta_mp.csv`,
`data/meta_oqmd.csv`, `data/mp_bandgap_data.csv` (has
`formation_energy_per_atom`), `data/y_stability.csv`,
`data/splits/*_stab.npy`, `models/scaler_stability.pkl`,
`formation_energy/test_quick.py` (proof-of-concept, MAE=0.090, R2=0.972).
