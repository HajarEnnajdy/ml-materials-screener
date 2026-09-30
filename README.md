# ML Materials Screener

Predict **formation energy**, **thermodynamic stability**, and the
**metal / semiconductor / insulator** class of an inorganic material straight
from its chemical formula — no DFT calculation required — and rank candidates
as **solar-cell absorbers**. Includes a React web app with an interactive 3D
crystal-structure viewer.

> Full write-up: [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

## What it does
- **3 models from composition alone:** formation energy (R² 0.955),
  stability (94% acc, ROC-AUC 0.98), semiconductor class (84% acc).
- **Screening modes:** generate candidates from an element set, score your
  own formula list, or rank real Materials Project materials as solar
  absorbers (Shockley–Queisser gap fit + direct-gap + stability + element
  abundance).
- **Web UI:** 3D structure viewer (3Dmol.js), auto-generated structure
  descriptions (Robocrystallographer), chemical names + space groups.

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
cp .env.example .env      # then put your free Materials Project API key in it
```

### 2. Build data + models (regenerated locally — not stored in git)
```bash
python 01_download_mp.py        # or 06_download_mp_full.py (includes metals)
python 02_download_oqmd.py
python 03_featurize.py          # and 07_featurize_full.py
python 03b_prepare.py
python 04_train_stability.py
python formation_energy/05_train_with_metals.py
python bandgap_v2/03_train_mp_classifier.py
```

### 3. Run the app
```bash
# terminal 1 — backend
python -m uvicorn api.main:app --port 8001
# terminal 2 — frontend
cd frontend && npm install && npm run dev
```
See [`docs/RUN_UI.md`](docs/RUN_UI.md) for details.

## Why data/ and models/ aren't in the repo
They are large (multi-GB models, 100+ MB feature tables) and fully
**regenerable** from the scripts above, so they're git-ignored. The download
scripts pull from the Materials Project / OQMD public APIs.

## Documentation
- [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) — full report (data, models, methods, findings)
- [`docs/BANDGAP_POSTMORTEM.md`](docs/BANDGAP_POSTMORTEM.md) — why DFT-PBE band gap was abandoned
- [`docs/TRANSFER_TEST_NOTES.md`](docs/TRANSFER_TEST_NOTES.md) — distribution-shift study + fix
- [`docs/SCREENER_NOTES.md`](docs/SCREENER_NOTES.md) — screener design + stability caveats
- [`docs/RUN_UI.md`](docs/RUN_UI.md) — running the web app

## Tech stack
Python (scikit-learn, matminer, pymatgen, mp-api, robocrys) ·
FastAPI · React + Vite + 3Dmol.js
