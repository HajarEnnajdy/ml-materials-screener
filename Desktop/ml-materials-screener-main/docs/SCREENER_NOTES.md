# Combined screener — what it is and its limits

## What was built

- `04_train_stability.py` — RandomForestClassifier on OQMD-derived features
  (`data/splits/X_*_stab.npy`, 136 features). Test accuracy 94.3%, ROC AUC
  0.978, AP 0.926. Plots: `plots/stability_*.png`.
- `formation_energy/04_train_formation_energy.py` — ExtraTreesRegressor on
  MP-derived features (146 features). Test MAE 0.090 eV/atom, R2 0.972.
  Plots: `plots/fe_*.png`.
- `screener.py` — inference module. Featurizes ANY formula string on the
  fly (matminer: MAGPIE + Stoichiometry + ValenceOrbital, same stack as
  `03_featurize.py`) and runs both models. This is what lets the project
  score formulas that were never in Materials Project or OQMD at all.
- `05_screen_candidates.py` — CLI entry point. Two modes:
  - `--formulas-file path` — score a list of formulas you already have.
  - `--elements Ga,N,Al,In,O` — generate binary/ternary candidates
    combinatorially within an element set and rank them.
  Outputs a ranked CSV + a formation-energy-vs-stability scatter plot.

Verified the inference pipeline matches training-time features exactly
(re-featurized Al2O3 from scratch vs. its precomputed row in `X_mp.csv`:
max diff ~1e-14) — the on-the-fly featurization is not a source of error.

## Important caveat: what "stability" actually means here

OQMD's `stable_label` (the stability classifier's training target) is
defined narrowly:

```
label = 1  if stability <= 0.0 eV/atom above OQMD's own calculated hull
label = 0  if stability >  0.1 eV/atom above hull
(0.0, 0.1] eV/atom is dropped as ambiguous — see 02_download_oqmd.py:71-77
```

This means the model predicts **"is this exact OQMD-calculated entry
sitting on OQMD's own convex hull"** — not "is this compound real /
synthesizable in the world." Concrete examples found while testing the
screener:

| Formula | OQMD stability (eV/atom above hull) | OQMD label | Real-world status |
|---|---|---|---|
| AlN | 0.116–0.117 | unstable (0) | Extremely common, stable, industrially produced |
| Ga2O3 | 0.288 | unstable (0) | Stable, widely studied wide-gap semiconductor |
| Al2O3 | 0.280 (single OQMD entry) | unstable (0) | Corundum — one of the most stable oxides known |

These aren't model errors — the classifier reproduces OQMD's own number
faithfully (verified directly against `X_oqmd.csv` + `y_stability.csv`).
The gap is between OQMD's specific DFT entry/hull construction and
real-world chemical stability. A handful of well-known stable compounds
sit just outside OQMD's strict ≤0.0 cutoff, likely due to functional
accuracy limits, missing competing phases in OQMD's hull at the time of
calculation, or the specific polymorph OQMD calculated not being the true
ground state.

**Practical implication for screening**: treat `stability_proba` as a
relative ranking signal ("more vs. less likely to be near-hull, by OQMD's
own standard"), not an absolute yes/no on real-world makeability. A
probability of 0.3-0.4 is not strong evidence a compound can't be made —
borderline-stable real materials can legitimately score there. Don't
filter with a hard `>=0.5` cutoff for final decisions; look at relative
ranking among candidates, and treat very low scores (<0.1, like AlN's
0.087 reproduction above) with more suspicion only when corroborated by
the formation energy being unusually high too.

## Fixes applied

- **Probability calibration**: `04_train_stability.py` now wraps the
  RandomForest in `CalibratedClassifierCV` (isotonic, fit on the
  validation set). Brier score improved 0.0450 -> 0.0441. The saved
  `models/stability_model.pkl` is the calibrated version.
- **Charge-balance filter**: `05_screen_candidates.py`'s combinatorial
  generator now drops formulas with no valid charge-balanced
  oxidation-state assignment (via pymatgen `oxi_state_guesses`) by
  default. On a 5-element test set this cut 320 -> 135 candidates,
  removing chemically nonsensical stoichiometries like `Ga3Al2O3`.
- **`is_promising` flag fixed**: originally required `stability_proba >=
  0.5`, which — given the OQMD strictness above — almost never fires even
  for real materials. Changed to relative ranking (top 30% by stability
  probability AND below-median formation energy within the candidate
  batch). Re-tested on Ga-N-Al-In-O: now correctly surfaces Al2O3, Ga2O3,
  GaAlO3 etc. as top "promising" candidates.

## Status

Closes out the formation-energy + stability pivot described in
`BANDGAP_POSTMORTEM.md`. The combined screener works end-to-end and was
validated against known chemistry (Al2O3, Ga2O3 correctly rank at the
top of a generated Ga-N-Al-In-O candidate pool).
