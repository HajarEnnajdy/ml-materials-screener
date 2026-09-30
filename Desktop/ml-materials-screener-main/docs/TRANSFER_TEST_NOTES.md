# Band gap classifier — transfer test result (NEGATIVE)

## What was tested
The metal/semiconductor/insulator classifier (`01_train_expt_bandgap.py`,
88% test accuracy) was trained AND tested only on the experimental
`expt_gap` dataset. `02_test_on_mp_data.py` applies it to the project's own
75,683 Materials Project compounds to check whether it generalizes.

## Result: it does NOT transfer
Every MP compound here is a real non-metal (the MP download excluded metals,
gap >= 0.01 eV). Yet the experimental-trained classifier predicts "metal"
for **65% of them**, including **55% of wide-gap insulators (PBE gap 3-10
eV)**. A 3-10 eV material is unambiguously not a metal, so this is a
systematic failure, not a near-boundary disagreement.

Predicted-"metal" rate by true PBE gap:

| PBE gap (eV) | % called metal | n |
|---|---|---|
| 0–0.5 | 74.5% | 14,546 |
| 0.5–1 | 70.8% | 10,651 |
| 1–2 | 67.1% | 17,570 |
| 2–3 | 61.8% | 14,123 |
| 3–10 | 54.9% | 18,793 |

## Why
Covariate (distribution) shift. The experimental training set is ~50%
metals and covers a different population of chemistries; the MP set is 0%
metals and is far larger and broader. The composition→class associations the
model learned on the experimental population do not hold on the MP
population. Test accuracy of 88% was genuine but is only valid on data drawn
from the same distribution as the training set.

## Takeaway / options
- **Do NOT** wire the experimental-trained classifier into the screener UI
  for scoring MP-style candidates — it would show unreliable labels.
- The experimental classifier remains a valid standalone result *on
  experimental-like inputs* (88%), useful for the report's narrative.
- To get a metal/semiconductor/insulator label that is reliable **on the MP
  data we actually screen**, train the classifier IN-DISTRIBUTION on MP's
  own labels. MP exposes a direct `is_metal` flag (a clean label for the
  metal vs non-metal split, independent of the noisy exact-gap value), and
  classification is robust to PBE's gap-value noise. This requires
  re-downloading MP *including* metals (currently filtered out at
  gap >= 0.01 in `01_download_mp.py`).

This is the band-gap analogue of the lesson in `../BANDGAP_POSTMORTEM.md`:
a model is only trustworthy on the distribution it was trained on.

## RESOLUTION — in-distribution classifier (`03_train_mp_classifier.py`)
Re-downloaded MP *including* metals (`../06_download_mp_full.py` →
132,596 compounds, 51% metals) and trained the classifier in-distribution
on MP's own labels (metal = gap 0, semiconductor 0–3 eV, insulator > 3).

Result: **84.1% test accuracy**, and — crucially — it now identifies metals
correctly (metal recall 87%, precision 90%) instead of the experimental
model's failure. Errors are almost entirely between adjacent classes;
metal↔insulator confusion is near-zero (127 + 31 of 17,601), which is
physically correct. Saved as `models/mp_bandgap_classifier.pkl`. THIS is the
classifier to use for screening MP-style candidates.

The experimental-trained classifier (`bandgap_classifier.pkl`, 88% on
experimental data) stays as a separate artifact, valid only on
experimental-like inputs.

Formation energy was also retrained on the with-metals set
(`../formation_energy/05_train_with_metals.py`): overall MAE 0.090 → 0.112
eV/atom and R² 0.972 → 0.955 (modestly worse because metals broaden the
population), but it is now in-distribution everywhere — MAE on the metal
subset is 0.035 eV/atom, a region the old model had never seen.
