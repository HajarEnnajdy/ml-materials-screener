"""
07_featurize_full.py
====================
Featurize the with-metals MP dataset (data/mp_full_data.csv) with the same
Magpie + Stoichiometry + ValenceOrbital stack used everywhere else.

Outputs:
    data/X_mp_full.csv      — 146 feature columns
    data/meta_mp_full.csv   — material_id, formula, band_gap_ev, is_metal,
                              formation_energy_per_atom, energy_above_hull
(rows aligned).
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from screener import load_featurizer

print("Loading data/mp_full_data.csv...")
df = pd.read_csv("data/mp_full_data.csv")
print(f"  {len(df):,} compounds")

featurizer, converter = load_featurizer()
feature_cols = featurizer.feature_labels()

print("Converting formulas to compositions...")
t0 = time.time()
comps, keep = [], []
for i, f in enumerate(df["formula"]):
    try:
        comps.append(converter.featurize(f)[0])
        keep.append(i)
    except Exception:
        pass
df = df.iloc[keep].reset_index(drop=True)
df["composition"] = comps
print(f"  {len(df):,} valid formulas  [{time.time()-t0:.0f}s]")

print("Computing features (a few minutes)...")
t1 = time.time()
df = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=True, inplace=False)
before = len(df)
df = df.dropna(subset=feature_cols).reset_index(drop=True)
print(f"  {len(df):,} featurized ({before-len(df)} dropped for NaN)  [{time.time()-t1:.0f}s]")

meta_cols = ["material_id", "formula", "band_gap_ev", "is_metal",
             "formation_energy_per_atom", "energy_above_hull"]
df[feature_cols].to_csv("data/X_mp_full.csv", index=False)
df[meta_cols].to_csv("data/meta_mp_full.csv", index=False)

n_metal = int((df["band_gap_ev"] == 0).sum())
print(f"\nSaved data/X_mp_full.csv  ({len(df):,} x {len(feature_cols)})")
print(f"Saved data/meta_mp_full.csv")
print(f"  metals: {n_metal:,} ({100*n_metal/len(df):.1f}%)  non-metals: {len(df)-n_metal:,}")
