"""
screener.py
===========
Core inference module: given raw formula strings (formulas that may not
exist in any training set), featurize them the same way as the training
pipeline (matminer: MAGPIE + Stoichiometry + ValenceOrbital) and run both
trained models:

    - formation_energy_model.pkl  (ExtraTreesRegressor, 146 features)
    - stability_model.pkl         (RandomForestClassifier, 136 features —
      the variance-thresholded subset used by 03b_prepare.py for OQMD)

This is what actually makes the project a "screener" instead of a backtest
on already-known materials — it can score formulas that were never in
Materials Project or OQMD at all.

Usage as a library:
    from screener import predict_candidates
    df = predict_candidates(["GaN", "InP", "Al2O3", "Fe3O4"])
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_featurizer():
    """Same featurizer stack as 03_featurize.py — must match exactly,
    since the trained models expect these specific columns."""
    try:
        from matminer.featurizers.composition import (
            ElementProperty, Stoichiometry, ValenceOrbital,
        )
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.conversions import StrToComposition
    except ImportError:
        print("ERROR: matminer not installed. Run: pip install matminer")
        sys.exit(1)

    featurizer = MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        ValenceOrbital(),
    ])
    featurizer.set_n_jobs(1)
    converter = StrToComposition()
    return featurizer, converter


def featurize_formulas(formulas):
    """
    formulas: list[str] -> DataFrame with one row per VALID formula,
    columns = all 146 Magpie+Stoichiometry+ValenceOrbital features,
    plus a 'formula' column. Invalid formulas are dropped (and reported).
    """
    featurizer, converter = load_featurizer()

    df = pd.DataFrame({"formula": formulas})
    compositions = []
    valid_idx = []
    for i, f in enumerate(df["formula"]):
        try:
            comp = converter.featurize(f)[0]
            compositions.append(comp)
            valid_idx.append(i)
        except Exception:
            pass

    n_failed = len(formulas) - len(valid_idx)
    if n_failed:
        print(f"  Skipped {n_failed} invalid formula(s)")

    df = df.iloc[valid_idx].reset_index(drop=True)
    df["composition"] = compositions

    df = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=True, inplace=False)
    feature_cols = featurizer.feature_labels()

    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    if before != len(df):
        print(f"  Dropped {before - len(df)} formula(s) with NaN features")

    return df, feature_cols


def predict_candidates(formulas, models_dir="models", splits_dir="data/splits"):
    """
    Returns a DataFrame: formula, formation_energy_pred_ev_atom,
    stability_proba, predicted_stable, plus an overall 'is_promising' flag
    (stable AND formation energy below the training-set median — a simple
    default; override the threshold yourself for your actual use case).
    """
    print(f"Featurizing {len(formulas)} candidate formula(s)...")
    df, feature_cols = featurize_formulas(formulas)
    if len(df) == 0:
        print("No valid formulas to score.")
        return pd.DataFrame()

    # ── Formation energy model: all 146 raw features ───────────────────────
    fe_model = joblib.load(f"{models_dir}/formation_energy_model.pkl")
    fe_scaler = joblib.load(f"{models_dir}/scaler_formation_energy.pkl")
    X_fe = df[feature_cols].values
    X_fe_scaled = fe_scaler.transform(X_fe)
    fe_pred = fe_model.predict(X_fe_scaled)

    # ── Stability model: 136-column variance-thresholded subset ────────────
    # The stability model was trained on SCALED features (03b_prepare.py fit
    # scaler_stability.pkl on the train split), so candidates must be scaled
    # with the same scaler before predict_proba. Skipping this fed raw,
    # wrong-magnitude features to the model and produced near-constant
    # garbage probabilities (~0.32 for everything).
    with open(f"{splits_dir}/feature_names.json") as f:
        stab_feature_names = json.load(f)
    stab_model = joblib.load(f"{models_dir}/stability_model.pkl")
    stab_scaler = joblib.load(f"{models_dir}/scaler_stability.pkl")
    X_stab = stab_scaler.transform(df[stab_feature_names].values)
    stab_proba = stab_model.predict_proba(X_stab)[:, 1]
    stab_pred = (stab_proba >= 0.5).astype(int)

    # ── Band gap class: metal / semiconductor / insulator ──────────────────
    # In-distribution classifier trained on the full (with-metals) MP set
    # (bandgap_v2/03_train_mp_classifier.py). RandomForest on the raw 146
    # features (no scaling). See bandgap_v2/TRANSFER_TEST_NOTES.md for why
    # this one is used instead of the experimental-trained classifier.
    clf_path = Path(__file__).parent / "bandgap_v2" / "models" / "mp_bandgap_classifier.pkl"
    bg_classes = ["metal", "semiconductor", "insulator"]
    bg_model = joblib.load(clf_path)
    bg_idx = bg_model.predict(df[feature_cols].values)
    bg_proba = bg_model.predict_proba(df[feature_cols].values)
    bg_class = [bg_classes[i] for i in bg_idx]
    bg_conf = bg_proba.max(axis=1)
    semi_proba = bg_proba[:, bg_classes.index("semiconductor")]

    result = pd.DataFrame({
        "formula": df["formula"],
        "formation_energy_pred_ev_atom": fe_pred,
        "stability_proba": stab_proba,
        "predicted_stable": stab_pred.astype(bool),
        "bandgap_class": bg_class,
        "bandgap_confidence": bg_conf,
        "semiconductor_proba": semi_proba,
    })

    # "is_promising" uses RELATIVE ranking, not predicted_stable (proba>=0.5),
    # on purpose. OQMD's own stability label is strict (only entries at or
    # below 0.0 eV/atom above its hull count as stable — see
    # SCREENER_NOTES.md), so even well-known real materials like Al2O3 score
    # well under 0.5. A hard 0.5 cutoff would flag almost nothing as
    # promising regardless of candidate quality. Top-30%-by-stability-proba
    # AND below-median formation energy is a much more useful signal for
    # ranking a candidate pool against itself.
    stab_cutoff = result["stability_proba"].quantile(0.70)
    fe_median = np.median(fe_pred)
    result["is_promising"] = (
        (result["stability_proba"] >= stab_cutoff)
        & (result["formation_energy_pred_ev_atom"] <= fe_median)
    )

    return result.sort_values(
        ["stability_proba", "formation_energy_pred_ev_atom"],
        ascending=[False, True],
    ).reset_index(drop=True)


if __name__ == "__main__":
    test_formulas = ["GaN", "InP", "Al2O3", "Fe3O4", "NotAFormulaXYZ"]
    out = predict_candidates(test_formulas)
    print("\n" + out.to_string(index=False))
