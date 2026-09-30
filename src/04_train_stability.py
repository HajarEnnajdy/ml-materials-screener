"""
04_train_stability.py
======================
Trains the stability classifier (is the material on the convex hull?) on
the OQMD-derived features already prepared by 03b_prepare.py
(data/splits/X_*_stab.npy). Formation-energy-adjacent target (thermodynamic
stability), so — like formation energy — this is expected to be reliable,
unlike band gap.

Outputs:
    models/stability_model.pkl
    models/stability_model_meta.json
    plots/stability_confusion_matrix.png
    plots/stability_roc_curve.png
    plots/stability_precision_recall.png
    plots/stability_feature_importance.png
    plots/stability_feature_correlation.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, brier_score_loss,
)

RANDOM_STATE = 42
Path("plots").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

print("Loading prepared stability splits...")
X_train = np.load("data/splits/X_train_stab.npy")
X_val   = np.load("data/splits/X_val_stab.npy")
X_test  = np.load("data/splits/X_test_stab.npy")
y_train = np.load("data/splits/y_train_stab.npy")
y_val   = np.load("data/splits/y_val_stab.npy")
y_test  = np.load("data/splits/y_test_stab.npy")

with open("data/splits/feature_names.json") as f:
    feature_names = json.load(f)

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print(f"Train class balance: {y_train.mean()*100:.1f}% stable")

# ── Train ───────────────────────────────────────────────────────────────────
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train)

val_acc = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)
print(f"Val accuracy : {val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# ── Calibrate probabilities ───────────────────────────────────────────────────
# Raw RandomForest predict_proba is a vote fraction, not a true probability —
# it's skewed by the 21%-stable class imbalance (see SCREENER_NOTES.md for why
# this matters: the screener uses these probabilities for ranking, so a
# 0.5 cutoff should mean something close to "50% chance," not just "more
# trees voted yes"). Isotonic calibration on the held-out val set fixes this
# without retraining the underlying forest.
print("\nCalibrating probabilities on validation set (isotonic)...")
calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
calibrated_model.fit(X_val, y_val)

raw_proba_test = model.predict_proba(X_test)[:, 1]
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print(f"Brier score before calibration: {brier_score_loss(y_test, raw_proba_test):.4f}")
print(f"Brier score after calibration : {brier_score_loss(y_test, y_proba):.4f}  (lower is better)")

report = classification_report(y_test, y_pred, target_names=["unstable", "stable"])
print("\n" + report)

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["unstable", "stable"], yticklabels=["unstable", "stable"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Stability classifier — confusion matrix")
plt.tight_layout()
plt.savefig("plots/stability_confusion_matrix.png", dpi=150)
plt.close()

# ── ROC curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stability classifier — ROC curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/stability_roc_curve.png", dpi=150)
plt.close()

# ── Precision-recall curve ────────────────────────────────────────────────────
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Stability classifier — Precision-Recall curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/stability_precision_recall.png", dpi=150)
plt.close()

# ── Feature importance (top 20) ───────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
top20 = importances.head(20)
plt.figure(figsize=(8, 7))
sns.barplot(x=top20.values, y=top20.index, color="steelblue")
plt.xlabel("Feature importance")
plt.title("Stability classifier — top 20 features")
plt.tight_layout()
plt.savefig("plots/stability_feature_importance.png", dpi=150)
plt.close()

# ── Correlation matrix of top 20 features ────────────────────────────────────
X_train_df = pd.DataFrame(X_train, columns=feature_names)
corr = X_train_df[top20.index].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Stability classifier — correlation matrix (top 20 features)")
plt.tight_layout()
plt.savefig("plots/stability_feature_correlation.png", dpi=150)
plt.close()

# ── Save calibrated model + metadata ─────────────────────────────────────────
# Save the calibrated wrapper (not the raw forest) — this is what screener.py
# should load, so predict_proba returns calibrated probabilities.
joblib.dump(calibrated_model, "models/stability_model.pkl")
meta = {
    "model": "RandomForestClassifier + isotonic calibration (CalibratedClassifierCV)",
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "val_accuracy": val_acc,
    "test_accuracy": test_acc,
    "test_roc_auc": roc_auc,
    "test_average_precision": ap,
    "brier_score_before_calibration": float(brier_score_loss(y_test, raw_proba_test)),
    "brier_score_after_calibration": float(brier_score_loss(y_test, y_proba)),
    "n_features": len(feature_names),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
}
with open("models/stability_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 50)
print("  Stability classifier results")
print("=" * 50)
print(f"  Test accuracy : {test_acc:.4f}")
print(f"  ROC AUC       : {roc_auc:.4f}")
print(f"  Average prec. : {ap:.4f}")
print("\nSaved: models/stability_model.pkl, models/stability_model_meta.json")
print("Saved plots to plots/stability_*.png")
