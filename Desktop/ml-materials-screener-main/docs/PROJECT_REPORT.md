# ML Materials Screener — Full Project Report

## 1. One-paragraph abstract

This project is a machine-learning pipeline and interactive web application
that screens inorganic materials directly from their **chemical formula** —
predicting **formation energy**, **thermodynamic stability**, and the
**metal / semiconductor / insulator** class without running any expensive
quantum-mechanical (DFT) calculation. On top of these models it ranks real
materials as candidate **solar-cell absorbers** using a physically-grounded
score, and exposes everything through a web interface with an interactive 3D
crystal-structure viewer. The work also includes a rigorous, documented
investigation of *why band gap is hard to predict*, which produced two
generalizable scientific lessons (label-noise vs. pipeline error, and
training/test distribution shift).

---

## 2. Motivation / the problem

Discovering a new functional material (e.g. a solar absorber) traditionally
requires either lab synthesis or DFT simulation — both slow and expensive.
DFT takes hours-to-days of supercomputer time *per material*, so scanning a
large chemical space exhaustively is impractical.

**Goal:** train ML models that take a cheap input (the composition / formula)
and predict the properties that decide whether a material is worth a closer
look — so a researcher can rank thousands of candidates in seconds and only
spend DFT/lab effort on the best ones.

---

## 3. Data sources

| Source | Size | What it provides | Access |
|---|---|---|---|
| **Materials Project (MP)** | 132,596 compounds (≤4 elements) | DFT-PBE band gap, formation energy, stability (energy above hull), is_metal, direct/indirect gap, space group | `mp-api` (REST, API key) |
| **OQMD** (Open Quantum Materials DB) | 36,794 compounds | DFT stability label (on convex hull or not) | REST API |
| **expt_gap** (Zhuo et al. 2018) | 6,354 compounds | *Experimentally measured* band gaps | via `matminer` |

Three sources are used because each is best for a different target: MP for
formation energy + structure, OQMD for stability labels, and the experimental
set for the band-gap investigation.

---

## 4. Methodology

### 4.1 Featurization (composition → numbers)
Every formula is turned into a fixed-length numeric vector using
[matminer](https://hackingmaterials.lbl.gov/matminer/):
- **Magpie** (132 features): statistics (min/max/range/mean/avg-dev/mode) of
  22 elemental properties (electronegativity, atomic radius, valence, …).
- **Stoichiometry** + **ValenceOrbital**: element-fraction norms and s/p/d/f
  valence-electron counts.
- **Total: 146 composition-only features per formula.**

Composition-only (no 3D structure) is a deliberate choice: it is fast and
works for *hypothetical* compounds that have never been made or simulated —
essential for true screening.

### 4.2 Cleaning (identical across all models, for fair comparison)
1. Drop NaNs.
2. **Dedup polymorphs:** keep the most stable polymorph per formula (lowest
   energy above hull) — composition features are identical for polymorphs, so
   keeping several injects label noise.
3. **IQR outlier removal** on the regression target.
4. **Variance threshold** (drops near-constant features; stability model).
5. **70/15/15 train/val/test split**, fixed `random_state=42`.
6. **StandardScaler** fit on train only (no leakage).

---

## 5. The production models (the deliverables)

### Model A — Formation-energy regressor
- **File:** `formation_energy/05_train_with_metals.py`
- **Algorithm:** ExtraTreesRegressor (300 trees), 146 scaled features.
- **Trained on:** full MP set *including metals* (~87k after cleaning).
- **Performance:** Test **MAE = 0.112 eV/atom, R² = 0.955**; on the metal
  subset alone MAE = 0.035 eV/atom.
- *Interpretation:* formation energy is a ground-state DFT quantity that PBE
  computes reliably, so the model is excellent.

### Model B — Stability classifier
- **File:** `04_train_stability.py`
- **Algorithm:** RandomForestClassifier (balanced) + **isotonic probability
  calibration**, 136 scaled features.
- **Trained on:** OQMD (on-hull = stable vs not).
- **Performance:** Test **accuracy 94.3%, ROC-AUC 0.977, average precision
  0.92**, Brier 0.044 after calibration.

### Model C — Metal / semiconductor / insulator classifier
- **File:** `bandgap_v2/03_train_mp_classifier.py`
- **Algorithm:** RandomForestClassifier, 146 raw features.
- **Trained on:** full MP set (with metals), labels from the gap value
  (metal = 0, semiconductor 0–3 eV, insulator > 3 eV).
- **Performance:** Test **accuracy 84.1%**; errors are almost entirely
  between *adjacent* classes (metal↔semiconductor), metal↔insulator confusion
  is near zero — physically sensible.

All three feed a single inference module, `screener.py::predict_candidates`,
which featurizes any formula on the fly and returns all predictions.

---

## 6. The band-gap investigation (scientific contribution)

Band gap is the property that actually defines "is this a semiconductor," so
predicting it was the original goal. It turned out to be hard, and chasing it
down produced the most scientifically interesting results.

### 6.1 Three attempts on DFT-PBE labels — all plateaued
| Approach | Method | Test MAE | R² |
|---|---|---|---|
| Classical | Magpie → ExtraTrees | 0.595 eV | 0.683 |
| Composition deep model | CrabNet (transformer, 300 epochs, GPU) | 0.513 eV | 0.700 |
| Structure GNN | MEGNet (prototyped) | — | — |

### 6.2 Diagnosis — it's the *labels*, not the pipeline
The same pipeline was re-run with the target swapped to **formation energy**:
**R² jumped from 0.68 to 0.97.** This controlled experiment proves the band-gap
ceiling comes from **DFT-PBE label noise** (PBE underestimates real gaps by
30–50%, unevenly), *not* from a flaw in the modeling. (Documented in
`BANDGAP_POSTMORTEM.md`.)

### 6.3 Experimental labels + a distribution-shift lesson
Retraining on **experimentally measured** gaps (`expt_gap`) gave a regressor
MAE 0.429 eV (28% better than PBE) and an 88%-accurate metal/semi/insulator
classifier. **But** when that experimental-trained classifier was applied to
our MP data it failed badly (called 65% of real non-metals "metal") — a
textbook case of **covariate shift** (the experimental population is ~50%
metals, the MP population very different). The fix was to retrain the
classifier *in-distribution* on MP itself (Model C above). (Documented in
`bandgap_v2/TRANSFER_TEST_NOTES.md`.)

**Two transferable lessons demonstrated with controlled experiments:**
1. A model's accuracy ceiling can come from label quality, not algorithm.
2. A model is only trustworthy on the distribution it was trained on.

---

## 7. The application

### 7.1 Inference + screening (`screener.py`, `05_screen_candidates.py`)
`predict_candidates(formulas)` featurizes *any* formula (verified to match
training features to ~1e-14) and returns formation energy, stability
probability, and semiconductor class. Three ways to supply candidates:
- **From elements** — combinatorially generate binary/ternary formulas within
  an element set, filtered to charge-balanced stoichiometries.
- **Custom formulas** — score a user-supplied list.
- **Dataset / solar mode** — see below.

### 7.2 Solar-absorber ranking (novel application layer)
For real MP materials, the app ranks candidate single-junction **solar
absorbers** with a physically-grounded score:
```
solar_score = gap_fit × direct_bonus × stability_factor × abundance_factor
```
- **gap_fit:** peaks at the Shockley–Queisser optimum **1.34 eV**, using the
  measured gap corrected for PBE underestimation (× 1.4).
- **direct_bonus:** ×1.25 for direct-gap materials (absorb light better).
- **stability_factor:** favors materials on the convex hull.
- **abundance_factor:** penalizes toxic (Pb/Cd/As…) and radioactive elements
  for practical terrestrial PV.

The top results are ~1.3 eV direct-gap, stable, earth-abundant compounds —
exactly the physical target.

### 7.3 Interactive web interface
- **Frontend:** React (Vite), a "crystallographic worksheet" visual design.
- **Backend:** FastAPI wrapping the models + MP structure lookups.
- **Per-result card:** formula, semiconductor class, formation energy,
  stability, (and band gap + direct/indirect in solar mode).
- **3D crystal viewer** (3Dmol.js) for any material that exists in MP.
- **Auto-generated structure descriptions** via **Robocrystallographer**
  (e.g. "Al₂O₃ is Corundum structured, crystallizes in R-3c, Al is bonded to
  six O forming edge/face/corner-sharing octahedra…").
- **Chemical name** (e.g. GaN → "Gallium nitride") + crystal system / space
  group.

---

## 8. What is new / what this project brings

1. **A working composition-only screener** for three properties at once
   (formation energy, stability, semiconductor class) that runs on *any*
   formula in milliseconds — no DFT required.
2. **A physically-grounded solar-absorber ranking** combining four
   independent criteria (gap optimum, direct gap, stability, element
   practicality) over real materials.
3. **A rigorous, documented band-gap study** that turns a "failure" into two
   generalizable ML lessons (label-noise diagnosis via a controlled
   target-swap; distribution-shift demonstrated and then fixed).
4. **A complete product**, not just notebooks: trained/saved models, an
   inference API, and an interactive 3D web UI with literature-grade
   structure descriptions.

---

## 9. Honest limitations

- **Band gap values are DFT-PBE × 1.4**, a single global correction — fine
  for ranking, not a precise per-material gap (true per-material corrections
  range ~1.1–1.8).
- **Solar/gap ranking only applies to real MP materials** (which have a
  computed gap); hypothetical formulas get class + stability only, since we
  have no reliable gap-*value* regressor that generalizes.
- **Stability labels are OQMD's strict on-hull definition** — a few real,
  synthesizable materials (AlN, Ga₂O₃) score "unstable," so stability is best
  read as a *relative ranking* signal.
- **Model accuracies hold on their training distribution** (MP/OQMD-like
  composition space); exotic chemistries are extrapolation.
- The **element abundance/toxicity list is an opinionated heuristic**
  (documented in code), deliberately downranking e.g. GaAs/CdTe for
  *terrestrial* PV.

---

## 10. Tech stack

Python (scikit-learn, matminer, pymatgen, pandas, mp-api, robocrys),
FastAPI + uvicorn (backend), React + Vite + 3Dmol.js (frontend),
matplotlib/seaborn (figures). Models persisted with joblib.

## 11. Repository map

```
01_download_mp.py / 06_download_mp_full.py   MP download (with metals)
02_download_oqmd.py                          OQMD stability download
03_featurize.py / 07_featurize_full.py       matminer featurization
03b_prepare.py                               cleaning + splits
04_train_stability.py                        Model B (stability, calibrated)
formation_energy/05_train_with_metals.py     Model A (formation energy)
bandgap_v2/03_train_mp_classifier.py         Model C (metal/semi/insulator)
bandgap_v2/01_train_expt_bandgap.py          experimental-label band-gap study
bandgap_v2/02_test_on_mp_data.py             transfer-failure test
screener.py                                  inference core (all 3 models)
05_screen_candidates.py                      CLI screener
api/main.py                                  FastAPI backend (+ solar ranking,
                                             3D structure, robocrys, naming)
frontend/                                    React UI
models/  bandgap_v2/models/                  saved models + metrics JSON
plots/   bandgap_v2/plots/                   diagnostic figures
*_NOTES.md / *_POSTMORTEM.md                 design + decision records
```
