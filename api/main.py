"""
api/main.py
============
FastAPI backend for the screener UI.

Endpoints:
    POST /api/screen          -> run predict_candidates() on a formula list
                                  or a combinatorially-generated candidate set
    GET  /api/structure/{formula} -> fetch the real crystal structure (CIF)
                                  for a formula, if it exists in Materials
                                  Project. Returns 404 if not found — there
                                  is NO fake/predicted structure for
                                  hypothetical formulas, on purpose (see
                                  SCREENER_NOTES.md: composition models say
                                  nothing about 3D structure).

Run with:
    uvicorn api.main:app --reload --port 8000
"""
import os
import sys
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from screener import predict_candidates  # noqa: E402
from importlib import import_module
import importlib.util

# 05_screen_candidates.py starts with a digit -> can't `import` normally
spec = importlib.util.spec_from_file_location("screen_candidates", ROOT / "05_screen_candidates.py")
screen_candidates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen_candidates)
generate_combinatorial = screen_candidates.generate_combinatorial


def load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

app = FastAPI(title="Material Screener API")
app.add_middleware(
    CORSMiddleware,
    # allow any localhost port — Vite may use 5173, 5174, ... if a port is busy
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreenRequest(BaseModel):
    formulas: list[str] | None = None
    elements: list[str] | None = None
    dataset_first_n: int | None = None
    max_stoich: int = 4
    include_ternary: bool = True
    require_charge_balance: bool = True


# ── Solar-suitability ranking over real MP materials ─────────────────────────
# Element practicality for large-scale terrestrial PV (earth-abundant / non-toxic).
_RADIOACTIVE = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
                "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm"}
_TOXIC = {"Pb", "Cd", "Hg", "As", "Tl", "Be", "Sb"}
_RARE = {"Te", "In", "Ga", "Ge", "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm",
         "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Ru", "Rh",
         "Pd", "Ag", "Re", "Os", "Ir", "Pt", "Au", "Hf", "Ta"}
IDEAL_GAP = 1.34   # Shockley–Queisser optimum (eV)
PBE_CORRECTION = 1.4   # rough scissor: true gap ≈ PBE gap × 1.4


def _abundance_factor(formula: str) -> float:
    from pymatgen.core import Composition
    try:
        syms = {e.symbol for e in Composition(formula).elements}
    except Exception:
        return 1.0
    if syms & _RADIOACTIVE:
        return 0.05
    if syms & _TOXIC:
        return 0.30
    if syms & _RARE:
        return 0.70
    return 1.0


@lru_cache(maxsize=8)
def _solar_ranked(n: int):
    """Top-n real MP materials ranked for single-junction solar absorbers.
    solar_score = gap_fit × direct_bonus × stability_factor × abundance_factor
    (see the route docstring / discussion for the physics)."""
    df = pd.read_csv(ROOT / "data" / "mp_bandgap_data.csv")
    # most-stable polymorph per formula
    df = df.sort_values("energy_above_hull", na_position="last").drop_duplicates("formula")

    true_gap = df["band_gap_ev"] * PBE_CORRECTION
    gap_fit = np.exp(-(((true_gap - IDEAL_GAP) / 0.5) ** 2))
    df = df.assign(true_gap=true_gap, gap_fit=gap_fit)
    df = df[df["gap_fit"] > 0.1]   # prefilter to a plausible gap window first

    direct_bonus = np.where(df["is_gap_direct"] == True, 1.25, 1.0)  # noqa: E712
    eah = df["energy_above_hull"].fillna(1.0).clip(lower=0)
    stability_factor = np.exp(-eah / 0.1)
    abundance = df["formula"].map(_abundance_factor)

    df = df.assign(
        solar_score=df["gap_fit"] * direct_bonus * stability_factor * abundance,
        direct_bonus=direct_bonus,
    )
    top = df.sort_values("solar_score", ascending=False).head(n)
    return top[["formula", "band_gap_ev", "true_gap", "is_gap_direct",
                "energy_above_hull", "solar_score"]].copy()


def df_to_records(df: pd.DataFrame):
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.post("/api/screen")
def screen(req: ScreenRequest):
    solar = None
    if req.dataset_first_n:
        # Dataset mode = "best solar semiconductors": rank REAL MP materials by
        # solar suitability (measured gap near 1.34 eV after PBE correction,
        # direct-gap bonus, stability, earth-abundant/non-toxic), then attach
        # the model predictions for display.
        solar = _solar_ranked(req.dataset_first_n)
        formulas = solar["formula"].tolist()
    elif req.formulas:
        formulas = req.formulas
    elif req.elements:
        formulas = generate_combinatorial(
            req.elements,
            max_stoich=req.max_stoich,
            include_ternary=req.include_ternary,
            require_charge_balance=req.require_charge_balance,
        )
    else:
        raise HTTPException(400, "Provide 'formulas', 'elements', or 'dataset_first_n'")

    if len(formulas) > 2000:
        raise HTTPException(400, f"{len(formulas)} candidates is too many for one request — narrow your element set or stoichiometry range")

    results = predict_candidates(formulas)
    if results.empty:
        return {"results": [], "count": 0}

    if solar is not None:
        # merge real solar data onto the model predictions, keep solar order
        results = solar.merge(results, on="formula", how="left")
        results = results.sort_values("solar_score", ascending=False)

    return {"results": df_to_records(results), "count": len(results)}


@lru_cache(maxsize=512)
def _lookup_material_id(formula: str):
    """Most-stable MP entry for this formula, if any (lowest energy_above_hull)."""
    meta = pd.read_csv(ROOT / "data" / "meta_mp.csv")
    matches = meta[meta["formula"] == formula]
    if matches.empty:
        return None
    matches = matches.sort_values("energy_above_hull", na_position="last")
    return matches.iloc[0]["material_id"]


# anion stems for simple inorganic naming (element symbol -> "...ide" stem)
_ANION_STEMS = {
    "O": "oxide", "S": "sulfide", "Se": "selenide", "Te": "telluride",
    "N": "nitride", "P": "phosphide", "As": "arsenide", "Sb": "antimonide",
    "C": "carbide", "Si": "silicide", "B": "boride", "H": "hydride",
    "F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide",
}


def chemical_name(formula: str):
    """Best-effort common chemical name for simple compounds.
    Reliable for elements and binaries; returns None for complex compounds
    (ternary+), where no safe simple name can be generated."""
    from pymatgen.core import Composition
    try:
        els = list(Composition(formula).elements)
    except Exception:
        return None
    if len(els) == 1:
        return els[0].long_name
    if len(els) == 2:
        cation, anion = sorted(els, key=lambda e: e.X)  # higher electronegativity = anion
        stem = _ANION_STEMS.get(anion.symbol)
        if stem:
            return f"{cation.long_name} {stem}"
    return None  # ternary+ : rely on the structural identity instead


@lru_cache(maxsize=256)
def _fetch_cif(material_id: str):
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise HTTPException(500, "MP_API_KEY not configured on server")
    from mp_api.client import MPRester
    from pymatgen.io.cif import CifWriter

    with MPRester(api_key) as mpr:
        doc = mpr.materials.summary.search(
            material_ids=[material_id], fields=["structure", "symmetry"]
        )
    if not doc:
        return None
    structure = doc[0].structure
    sym = doc[0].symmetry
    crystal_system = str(sym.crystal_system) if sym and sym.crystal_system else None
    spacegroup = getattr(sym, "symbol", None) if sym else None
    sg_number = getattr(sym, "number", None) if sym else None
    return {
        "cif": str(CifWriter(structure)),
        "n_atoms": len(structure),
        "crystal_system": crystal_system,
        "spacegroup": spacegroup,
        "sg_number": sg_number,
    }


@app.get("/api/structure/{formula}")
def get_structure(formula: str):
    material_id = _lookup_material_id(formula)
    if material_id is None:
        raise HTTPException(404, f"'{formula}' has no known Materials Project entry — no real structure available (hypothetical composition)")

    result = _fetch_cif(material_id)
    if result is None:
        raise HTTPException(404, "Structure lookup failed")
    return {
        "formula": formula,
        "name": chemical_name(formula),
        "material_id": material_id,
        "mp_url": f"https://materialsproject.org/materials/{material_id}",
        **result,
    }


@lru_cache(maxsize=256)
def _robocrys_description(material_id: str):
    """Robocrystallographer natural-language description of the structure.
    Cached per material_id — generation takes ~5s (bonding analysis)."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        return None
    from mp_api.client import MPRester
    from robocrys import StructureCondenser, StructureDescriber

    with MPRester(api_key) as mpr:
        doc = mpr.materials.summary.search(material_ids=[material_id], fields=["structure"])
    if not doc:
        return None
    condensed = StructureCondenser().condense_structure(doc[0].structure)
    return StructureDescriber().describe(condensed)


@app.get("/api/description/{material_id}")
def get_description(material_id: str):
    try:
        text = _robocrys_description(material_id)
    except Exception as e:
        raise HTTPException(500, f"Description generation failed: {e}")
    if not text:
        raise HTTPException(404, "No description available")
    return {"material_id": material_id, "description": text}


@app.get("/api/health")
def health():
    return {"status": "ok"}
