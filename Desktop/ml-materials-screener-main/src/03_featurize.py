"""
03_featurize.py
===============
Converts chemical formulas into numerical features using
matminer's MAGPIE featurizer (132 features per compound).

This runs on BOTH datasets:
    data/mp_bandgap_data.csv   → data/X_mp.csv + data/y_bandgap.csv
    data/oqmd_stability_data.csv → data/X_oqmd.csv + data/y_stability.csv

What MAGPIE features are:
    For each formula (e.g. TiO2), it looks up properties of every
    element inside it (Ti, O) and computes statistics:
    min, max, range, mean, avg_dev, mode
    ...across 22 elemental properties like:
    electronegativity, atomic radius, ionization energy,
    valence electrons, melting point, etc.
    Result: 22 properties × 6 statistics = 132 numbers per formula.

Requirements:
    pip install matminer pandas numpy

Usage:
    python 03_featurize.py                     # both datasets
    python 03_featurize.py --dataset mp        # only Materials Project
    python 03_featurize.py --dataset oqmd      # only OQMD
    python 03_featurize.py --test              # 500 rows only (quick check)
"""

import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")   # suppress matminer progress noise in logs

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Featurize MP and OQMD data using MAGPIE features"
    )
    p.add_argument(
        "--mp-input",    default="data/mp_bandgap_data.csv",
        help="Path to Materials Project CSV (default: data/mp_bandgap_data.csv)"
    )
    p.add_argument(
        "--oqmd-input",  default="data/oqmd_stability_data.csv",
        help="Path to OQMD CSV (default: data/oqmd_stability_data.csv)"
    )
    p.add_argument(
        "--output-dir",  default="data",
        help="Folder to save output files (default: data/)"
    )
    p.add_argument(
        "--dataset",     choices=["mp", "oqmd", "both"], default="both",
        help="Which dataset to featurize (default: both)"
    )
    p.add_argument(
        "--test",        action="store_true",
        help="Test mode: only featurize first 500 rows of each file"
    )
    return p.parse_args()

# ── Load matminer ─────────────────────────────────────────────────────────────

def load_featurizer():
    """
    Load the composition-based featurizer stack from matminer.

    MAGPIE alone (132 features: 22 elemental properties x 6 stats) misses
    bonding-character signal. We stack two more composition-only
    featurizers on top — they work directly off the pymatgen Composition
    object (no extra conversion step needed):
      - Stoichiometry   : Lp-norms of element fractions (alloy-vs-ordered signal)
      - ValenceOrbital  : s/p/d/f valence electron counts + fractions

    NOTE: IonProperty was tried too (ionic character from oxidation-state
    guesses) but Composition.oxi_state_guesses() has a combinatorial
    worst-case for formulas with several distinct elements — a handful of
    compounds in this dataset took minutes each, stalling the whole batch.
    Not worth it here; dropped.
    """
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Stoichiometry, ValenceOrbital,
        )
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.conversions import StrToComposition
    except ImportError:
        print("ERROR: matminer not installed.")
        print("Run: pip install matminer")
        sys.exit(1)

    print("Loading composition featurizers (MAGPIE + Stoichiometry + ValenceOrbital)...")
    featurizer = MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        ValenceOrbital(),
    ])
    featurizer.set_n_jobs(1)   # single thread — more stable on Windows
    converter  = StrToComposition()

    n_features = len(featurizer.feature_labels())
    print(f"Featurizer stack generates {n_features} features per formula\n")

    return featurizer, converter

# ── Core featurization ────────────────────────────────────────────────────────

def featurize(df, formula_col, featurizer, converter, dataset_name):
    """
    Convert formula column → numerical feature columns.

    Steps:
        1. StrToComposition: "TiO2" → Composition object
        2. featurizer:       Composition → numbers

    Returns DataFrame with only the feature columns.
    Rows with invalid formulas are dropped.
    """
    print(f"  Step 1/2 — Converting formulas to Composition objects...")
    t0  = time.time()
    df  = df.copy()

    # Convert string formula to pymatgen Composition object
    df, failed = _safe_convert(df, formula_col, converter)
    print(f"  Converted {len(df):,} formulas  "
          f"({failed} invalid formulas dropped)  [{time.time()-t0:.1f}s]")

    print(f"  Step 2/2 — Computing MAGPIE features (this takes a few minutes)...")
    t1  = time.time()

    # Compute all 132 MAGPIE features
    df  = featurizer.featurize_dataframe(
        df,
        col_id="composition",
        ignore_errors=True,   # skip any remaining bad rows
        inplace=False,
    )
    print(f"  Features computed  [{time.time()-t1:.1f}s]")

    # Get feature column names
    feature_cols = featurizer.feature_labels()

    # Drop rows where any feature is NaN (failed featurization)
    before = len(df)
    df     = df.dropna(subset=feature_cols)
    after  = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN features")

    print(f"  Final: {after:,} compounds × {len(feature_cols)} features\n")
    return df, feature_cols

def _safe_convert(df, formula_col, converter):
    """Convert formula strings to Composition, track failures."""
    from matminer.featurizers.conversions import StrToComposition

    results   = []
    failed    = 0

    for formula in df[formula_col]:
        try:
            row = converter.featurize(formula)
            results.append(row[0])
        except Exception:
            results.append(None)
            failed += 1

    df = df.copy()
    df["composition"] = results
    df = df[df["composition"].notna()].reset_index(drop=True)
    return df, failed

# ── Process Materials Project ─────────────────────────────────────────────────

def process_mp(input_path, output_dir, featurizer, converter, test_mode):
    """
    Load MP data, featurize, save:
        X_mp.csv      — 132 feature columns (input to Model 1)
        y_bandgap.csv — band_gap_ev column  (target for Model 1)
        meta_mp.csv   — formula + material_id for reference
    """
    path = Path(input_path)
    if not path.exists():
        print(f"  SKIP: {path} not found.")
        print(f"  Run 01_download_mp.py first.\n")
        return False

    print("=" * 60)
    print("  Materials Project — Band Gap Dataset")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from {path}")

    if test_mode:
        df = df.head(500)
        print(f"  TEST MODE: using first 500 rows only")

    # Drop metals (band_gap = 0) and rows without band gap
    before = len(df)
    df     = df[df["band_gap_ev"] > 0.0].dropna(subset=["band_gap_ev"])
    print(f"  After removing metals: {len(df):,} rows "
          f"({before - len(df)} metals removed)")

    # Remove extreme outliers (band gap > 10 eV are wide insulators, not useful)
    df = df[df["band_gap_ev"] <= 10.0]
    print(f"  After removing wide insulators (>10 eV): {len(df):,} rows\n")

    # Print band gap distribution
    _print_bandgap_distribution(df["band_gap_ev"])

    # Featurize
    t_total = time.time()
    df_feat, feature_cols = featurize(df, "formula", featurizer, converter, "MP")
    print(f"  Total featurization time: {time.time()-t_total:.1f}s")

    # Save outputs
    out   = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # X — features only
    X = df_feat[feature_cols]
    X.to_csv(out / "X_mp.csv", index=False)

    # y — band gap target
    y = df_feat[["band_gap_ev"]].rename(columns={"band_gap_ev": "target"})
    y.to_csv(out / "y_bandgap.csv", index=False)

    # meta — formula + id for traceability
    meta_cols = [c for c in ["formula", "material_id", "band_gap_ev",
                              "is_stable", "energy_above_hull", "chemsys"]
                 if c in df_feat.columns]
    df_feat[meta_cols].to_csv(out / "meta_mp.csv", index=False)

    print(f"\n  Saved:")
    print(f"    {out}/X_mp.csv       — {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"    {out}/y_bandgap.csv  — {len(y):,} band gap targets")
    print(f"    {out}/meta_mp.csv    — formula + metadata\n")

    # Sanity check
    _sanity_check(X, y, "MP band gap")
    return True

# ── Process OQMD ──────────────────────────────────────────────────────────────

def process_oqmd(input_path, output_dir, featurizer, converter, test_mode):
    """
    Load OQMD data, featurize, save:
        X_oqmd.csv        — 132 feature columns (input to Model 2)
        y_stability.csv   — stable_label column  (target for Model 2)
        meta_oqmd.csv     — formula + stability for reference
    """
    path = Path(input_path)
    if not path.exists():
        print(f"  SKIP: {path} not found.")
        print(f"  Run 02_download_oqmd.py first.\n")
        return False

    print("=" * 60)
    print("  OQMD — Stability Dataset")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from {path}")

    if test_mode:
        df = df.head(500)
        print(f"  TEST MODE: using first 500 rows only")

    # Keep only rows with a valid label
    before = len(df)
    df     = df.dropna(subset=["stable_label", "formula"])
    df     = df[df["stable_label"].isin([0, 1])]
    print(f"  Valid labelled rows: {len(df):,} "
          f"({before - len(df)} rows without label dropped)")

    # Print class balance
    n1 = (df["stable_label"] == 1).sum()
    n0 = (df["stable_label"] == 0).sum()
    print(f"  Stable   (1): {n1:,}  ({100*n1/len(df):.1f}%)")
    print(f"  Unstable (0): {n0:,}  ({100*n0/len(df):.1f}%)\n")

    # Featurize
    t_total = time.time()
    df_feat, feature_cols = featurize(df, "formula", featurizer, converter, "OQMD")
    print(f"  Total featurization time: {time.time()-t_total:.1f}s")

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = df_feat[feature_cols]
    X.to_csv(out / "X_oqmd.csv", index=False)

    y = df_feat[["stable_label"]].rename(columns={"stable_label": "target"})
    y.to_csv(out / "y_stability.csv", index=False)

    meta_cols = [c for c in ["formula", "entry_id", "stability",
                              "delta_e", "stable_label"] if c in df_feat.columns]
    df_feat[meta_cols].to_csv(out / "meta_oqmd.csv", index=False)

    print(f"\n  Saved:")
    print(f"    {out}/X_oqmd.csv         — {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"    {out}/y_stability.csv    — {len(y):,} stability labels")
    print(f"    {out}/meta_oqmd.csv      — formula + metadata\n")

    _sanity_check(X, y, "OQMD stability")
    return True

# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_bandgap_distribution(series):
    """Print a simple ASCII histogram of band gaps."""
    bins   = [(0,1),(1,1.5),(1.5,2),(2,2.5),(2.5,3),(3,5),(5,10)]
    labels = ["0–1","1–1.5","1.5–2","2–2.5","2.5–3","3–5","5–10"]
    total  = len(series)
    print("  Band gap distribution:")
    for (lo, hi), label in zip(bins, labels):
        n   = ((series >= lo) & (series < hi)).sum()
        bar = "█" * int(40 * n / total)
        marker = " ← solar range" if label in ["1–1.5","1.5–2","2–2.5"] else ""
        print(f"    {label:>7} eV : {n:>6,}  {bar}{marker}")
    print()

def _sanity_check(X, y, name):
    """Basic checks to confirm outputs are usable."""
    print(f"  Sanity check — {name}:")

    # Check for NaN
    nan_X = X.isnull().sum().sum()
    nan_y = y.isnull().sum().sum()
    print(f"    NaN in X : {nan_X}  {'✓' if nan_X == 0 else '✗ WARNING'}")
    print(f"    NaN in y : {nan_y}  {'✓' if nan_y == 0 else '✗ WARNING'}")

    # Check shapes match
    match = len(X) == len(y)
    print(f"    Rows match (X vs y): {'✓' if match else '✗ MISMATCH'}")

    # Check feature variance (zero variance = useless feature)
    zero_var = (X.std() == 0).sum()
    print(f"    Zero-variance features: {zero_var}  "
          f"{'✓' if zero_var == 0 else f'(will be removed during training)'}")

    # Show feature value range
    print(f"    Feature value range: "
          f"[{X.values.min():.2f}, {X.values.max():.2f}]")
    print()

# ── Feature list printer ──────────────────────────────────────────────────────

def print_feature_overview(featurizer):
    """Show what the features actually are."""
    labels = featurizer.feature_labels()

    # Extract unique MAGPIE property names (strip statistic prefix);
    # other featurizers (Stoichiometry/ValenceOrbital/IonProperty) have
    # short labels that don't match this pattern and are just counted.
    props = set()
    other = 0
    for label in labels:
        # Format: "MagpieData mean Electronegativity" → "Electronegativity"
        if label.startswith("MagpieData"):
            parts = label.split()
            if len(parts) >= 3:
                props.add(parts[-1])
        else:
            other += 1

    print("\n" + "="*60)
    print(f"  Feature Overview ({len(labels)} total)")
    print("="*60)
    print(f"  MAGPIE: 22 elemental properties x 6 statistics = {6*len(props)} features")
    print(f"\n  Statistics computed per formula (MAGPIE):")
    print(f"    minimum, maximum, range, mean, avg_dev, mode")
    print(f"\n  Elemental properties used (MAGPIE):")
    for i, prop in enumerate(sorted(props)):
        print(f"    {i+1:2}. {prop}")
    print(f"\n  Plus {other} features from Stoichiometry / ValenceOrbital / IonProperty")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  03_featurize.py — MAGPIE Feature Engineering")
    print("="*60)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Test mode    : {args.test}")
    print("="*60 + "\n")

    # Load featurizer once — used for both datasets
    featurizer, converter = load_featurizer()

    # Show what features we're computing
    print_feature_overview(featurizer)

    start   = time.time()
    success = []

    if args.dataset in ("mp", "both"):
        ok = process_mp(
            args.mp_input, args.output_dir,
            featurizer, converter, args.test
        )
        if ok:
            success.append("MP")

    if args.dataset in ("oqmd", "both"):
        ok = process_oqmd(
            args.oqmd_input, args.output_dir,
            featurizer, converter, args.test
        )
        if ok:
            success.append("OQMD")

    # Final summary
    total = time.time() - start
    print("=" * 60)
    print("  All done!")
    print("=" * 60)
    print(f"  Completed    : {', '.join(success) if success else 'nothing'}")
    print(f"  Total time   : {total:.0f}s ({total/60:.1f} min)")
    print(f"\n  Output files in '{args.output_dir}/':")
    out = Path(args.output_dir)
    for f in sorted(out.glob("*.csv")):
        mb = f.stat().st_size / 1024 / 1024
        rows = sum(1 for _ in open(f)) - 1
        print(f"    {f.name:<25} {rows:>8,} rows  {mb:.1f} MB")

    print(f"\n  Next steps:")
    print(f"    python 04_train_bandgap.py    ← trains Model 1")
    print(f"    python 05_train_stability.py  ← trains Model 2")
    print()

if __name__ == "__main__":
    main()
