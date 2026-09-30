# 03b_prepare.py — Preparation complete avant entrainement
import pandas as pd
import numpy as np
import json
import os
import joblib
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════
# ETAPE 1 — CHARGEMENT ET AUDIT
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("  ETAPE 1 — AUDIT")
print("=" * 60)

X_mp      = pd.read_csv("data/X_mp.csv")
y_bg      = pd.read_csv("data/y_bandgap.csv")
X_oqmd    = pd.read_csv("data/X_oqmd.csv")
y_stab    = pd.read_csv("data/y_stability.csv")
meta_mp   = pd.read_csv("data/meta_mp.csv")
meta_oqmd = pd.read_csv("data/meta_oqmd.csv")

print(f"X_mp    shape : {X_mp.shape}")
print(f"y_bg    shape : {y_bg.shape}")
print(f"X_oqmd  shape : {X_oqmd.shape}")
print(f"y_stab  shape : {y_stab.shape}")
print(f"Match MP      : {len(X_mp) == len(y_bg)}")
print(f"Match OQMD    : {len(X_oqmd) == len(y_stab)}")
print(f"NaN dans X_mp : {X_mp.isnull().sum().sum()}")
print(f"NaN dans X_oqmd: {X_oqmd.isnull().sum().sum()}")
print(f"Band gap min  : {y_bg['target'].min():.3f} eV")
print(f"Band gap max  : {y_bg['target'].max():.3f} eV")
print(f"Band gap mean : {y_bg['target'].mean():.3f} eV")
n1 = (y_stab["target"] == 1).sum()
n0 = (y_stab["target"] == 0).sum()
print(f"Stables   (1) : {n1} ({100*n1/len(y_stab):.1f}%)")
print(f"Instables (0) : {n0} ({100*n0/len(y_stab):.1f}%)")

cols_match = set(X_mp.columns) == set(X_oqmd.columns)
print(f"Memes features: {cols_match}")

# ═══════════════════════════════════════════════════════
# ETAPE 2 — NETTOYAGE
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ETAPE 2 — NETTOYAGE")
print("=" * 60)

# -- Model 1 : supprimer NaN
mask_nan = X_mp.isnull().any(axis=1) | y_bg.isnull().any(axis=1)
X_mp = X_mp[~mask_nan].reset_index(drop=True)
y_bg = y_bg[~mask_nan].reset_index(drop=True)
print(f"MP apres suppression NaN    : {len(X_mp):,} lignes")

# -- Model 1 : supprimer doublons (garder le polymorphe le plus stable)
if "formula" in meta_mp.columns:
    meta_mp = meta_mp[~mask_nan].reset_index(drop=True)

    # Plusieurs lignes peuvent partager la meme formule (polymorphes) avec
    # des band gaps tres differents alors que les features MAGPIE (basees
    # uniquement sur la composition) sont IDENTIQUES. Garder une ligne au
    # hasard ("first") injecte du bruit dans les labels. On garde donc le
    # polymorphe le plus stable (energy_above_hull la plus basse).
    if "energy_above_hull" in meta_mp.columns:
        sort_order = meta_mp["energy_above_hull"].fillna(np.inf)
    else:
        sort_order = pd.Series(np.arange(len(meta_mp)), index=meta_mp.index)

    order_idx = (
        pd.DataFrame({"formula": meta_mp["formula"], "_sort": sort_order})
        .sort_values(["formula", "_sort"], kind="stable")
        .index
    )

    meta_mp = meta_mp.loc[order_idx].reset_index(drop=True)
    X_mp    = X_mp.loc[order_idx].reset_index(drop=True)
    y_bg    = y_bg.loc[order_idx].reset_index(drop=True)

    dup_mask = meta_mp["formula"].duplicated(keep="first")
    X_mp = X_mp[~dup_mask].reset_index(drop=True)
    y_bg = y_bg[~dup_mask].reset_index(drop=True)
    meta_mp = meta_mp[~dup_mask].reset_index(drop=True)
    print(f"Doublons supprimes (polymorphe le + stable garde): {dup_mask.sum():,}")
    print(f"MP apres suppression doublons: {len(X_mp):,} lignes")

# -- Features variance nulle (entraine sur MP, applique aux deux)
selector = VarianceThreshold(threshold=0.01)
X_mp_clean = pd.DataFrame(
    selector.fit_transform(X_mp),
    columns=X_mp.columns[selector.get_support()]
)
n_removed = X_mp.shape[1] - X_mp_clean.shape[1]
print(f"Features supprimees (var=0) : {n_removed}")
print(f"Features restantes          : {X_mp_clean.shape[1]}")

# -- Model 2 : meme nettoyage
mask_nan2 = X_oqmd.isnull().any(axis=1) | y_stab.isnull().any(axis=1)
X_oqmd    = X_oqmd[~mask_nan2].reset_index(drop=True)
y_stab    = y_stab[~mask_nan2].reset_index(drop=True)
meta_oqmd = meta_oqmd[~mask_nan2].reset_index(drop=True)
print(f"OQMD apres suppression NaN  : {len(X_oqmd):,} lignes")

# -- Model 2 : supprimer doublons (meme probleme que MP : les features
#    MAGPIE sont basees uniquement sur la composition, donc deux polymorphes
#    de la meme formule sont des doublons EXACTS en X. Garder le plus stable
#    (stability la plus basse) au lieu d'un choix arbitraire.
if "formula" in meta_oqmd.columns and "stability" in meta_oqmd.columns:
    order_idx2 = (
        meta_oqmd[["formula", "stability"]]
        .sort_values(["formula", "stability"], kind="stable")
        .index
    )
    meta_oqmd = meta_oqmd.loc[order_idx2].reset_index(drop=True)
    X_oqmd    = X_oqmd.loc[order_idx2].reset_index(drop=True)
    y_stab    = y_stab.loc[order_idx2].reset_index(drop=True)

    dup_mask2 = meta_oqmd["formula"].duplicated(keep="first")
    X_oqmd    = X_oqmd[~dup_mask2].reset_index(drop=True)
    y_stab    = y_stab[~dup_mask2].reset_index(drop=True)
    meta_oqmd = meta_oqmd[~dup_mask2].reset_index(drop=True)
    print(f"Doublons OQMD supprimes (polymorphe le + stable garde): {dup_mask2.sum():,}")
    print(f"OQMD apres suppression doublons: {len(X_oqmd):,} lignes")

# -- Aligner les colonnes OQMD sur MP (CRITIQUE)
X_oqmd_clean = X_oqmd[X_mp_clean.columns]
print(f"Features OQMD alignees      : {X_oqmd_clean.shape[1]}")

# ═══════════════════════════════════════════════════════
# ETAPE 3 — OUTLIERS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ETAPE 3 — OUTLIERS")
print("=" * 60)

y_values = y_bg["target"].values
Q1  = np.percentile(y_values, 25)
Q3  = np.percentile(y_values, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Q1={Q1:.3f}  Q3={Q3:.3f}  IQR={IQR:.3f}")
print(f"Seuil bas : {lower:.3f} eV")
print(f"Seuil haut: {upper:.3f} eV")

mask_ok = (y_values >= lower) & (y_values <= upper)
n_out   = (~mask_ok).sum()
print(f"Outliers supprimes : {n_out} ({100*n_out/len(y_values):.1f}%)")

X_mp_clean = X_mp_clean[mask_ok].reset_index(drop=True)
y_bg       = y_bg[mask_ok].reset_index(drop=True)
print(f"MP apres outliers  : {len(X_mp_clean):,} lignes")

# Clip valeurs extremes dans OQMD (sans supprimer de lignes)
X_oqmd_clean = X_oqmd_clean.clip(
    lower=X_oqmd_clean.quantile(0.001),
    upper=X_oqmd_clean.quantile(0.999),
    axis=1
)
print(f"OQMD features clippees aux percentiles 0.1% - 99.9%")

# ═══════════════════════════════════════════════════════
# ETAPE 4 + 5 — NORMALISATION + SPLIT
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ETAPE 4+5 — SPLIT + NORMALISATION")
print("=" * 60)

Path("models").mkdir(exist_ok=True)
Path("data/splits").mkdir(parents=True, exist_ok=True)

# ── MODEL 1 ──
X_arr = X_mp_clean.values
y_arr = y_bg["target"].values

X_train,  X_temp,  y_train,  y_temp  = train_test_split(
    X_arr, y_arr, test_size=0.30, random_state=RANDOM_STATE)
X_val,    X_test,  y_val,    y_test  = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

print(f"Model 1 — Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

scaler_mp = StandardScaler()
scaler_mp.fit(X_train)                          # fit sur train UNIQUEMENT
X_train_s = scaler_mp.transform(X_train)
X_val_s   = scaler_mp.transform(X_val)
X_test_s  = scaler_mp.transform(X_test)

joblib.dump(scaler_mp, "models/scaler_bandgap.pkl")
print("Scaler band gap sauvegarde")

np.save("data/splits/X_train_bg.npy", X_train_s)
np.save("data/splits/X_val_bg.npy",   X_val_s)
np.save("data/splits/X_test_bg.npy",  X_test_s)
np.save("data/splits/y_train_bg.npy", y_train)
np.save("data/splits/y_val_bg.npy",   y_val)
np.save("data/splits/y_test_bg.npy",  y_test)

# ── MODEL 2 ──
X_arr2 = X_oqmd_clean.values
y_arr2 = y_stab["target"].values

X_train2, X_temp2, y_train2, y_temp2 = train_test_split(
    X_arr2, y_arr2, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y_arr2)
X_val2,   X_test2, y_val2,   y_test2 = train_test_split(
    X_temp2, y_temp2, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp2)

print(f"Model 2 — Train: {len(X_train2):,}  Val: {len(X_val2):,}  Test: {len(X_test2):,}")
print(f"Stables dans train: {y_train2.sum()}/{len(y_train2)} ({100*y_train2.mean():.1f}%)")

scaler_oqmd = StandardScaler()
scaler_oqmd.fit(X_train2)
X_train2_s = scaler_oqmd.transform(X_train2)
X_val2_s   = scaler_oqmd.transform(X_val2)
X_test2_s  = scaler_oqmd.transform(X_test2)

joblib.dump(scaler_oqmd, "models/scaler_stability.pkl")
print("Scaler stabilite sauvegarde")

np.save("data/splits/X_train_stab.npy", X_train2_s)
np.save("data/splits/X_val_stab.npy",   X_val2_s)
np.save("data/splits/X_test_stab.npy",  X_test2_s)
np.save("data/splits/y_train_stab.npy", y_train2)
np.save("data/splits/y_val_stab.npy",   y_val2)
np.save("data/splits/y_test_stab.npy",  y_test2)

# Sauvegarder noms des features
with open("data/splits/feature_names.json", "w") as f:
    json.dump(list(X_mp_clean.columns), f)
print("Noms des features sauvegardes")

# ═══════════════════════════════════════════════════════
# ETAPE 6 — VERIFICATION FINALE
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ETAPE 6 — VERIFICATION FINALE")
print("=" * 60)

checks = [
    ("Tailles train/y coherentes Model 1",  len(X_train_s)  == len(y_train)),
    ("Tailles train/y coherentes Model 2",  len(X_train2_s) == len(y_train2)),
    ("Aucun NaN dans X_train Model 1",      not np.isnan(X_train_s).any()),
    ("Aucun NaN dans X_train Model 2",      not np.isnan(X_train2_s).any()),
    ("Scaler band gap sauvegarde",          os.path.exists("models/scaler_bandgap.pkl")),
    ("Scaler stabilite sauvegarde",         os.path.exists("models/scaler_stability.pkl")),
    ("Meme nb features MP et OQMD",         X_train_s.shape[1] == X_train2_s.shape[1]),
    ("Band gap moyen entre 1 et 4 eV",      1.0 < y_train.mean() < 4.0),
    ("Equilibre stables entre 30% et 70%",  0.30 < y_train2.mean() < 0.70),
]

all_ok = True
for name, result in checks:
    status = " OK  " if result else "ECHEC"
    print(f"  [{status}] {name}")
    if not result:
        all_ok = False

print("=" * 60)
if all_ok:
    print("  TOUT EST OK — Lancez maintenant : python 04_train_bandgap.py")
else:
    print("  ATTENTION — Corrigez les erreurs ci-dessus avant de continuer")
print("=" * 60)