"""
06_download_mp_full.py
======================
Like 01_download_mp.py but INCLUDES metals (band_gap == 0).

The original download filtered band_gap >= 0.01, removing all metals. That
left two blind spots:
  - the formation energy model never saw metallic compositions
    (extrapolates for them), and
  - there were no metals to train a metal/semiconductor/insulator classifier.

This pulls the full <=4-element set with no lower gap bound and keeps the
is_metal flag, so both can be fixed in-distribution.

Saves data/mp_full_data.csv.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

MAX_ELEMENTS = 4
MAX_GAP = 10.0


def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv()


def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("ERROR: MP_API_KEY not set in .env")
        sys.exit(1)
    from mp_api.client import MPRester

    fields = [
        "material_id", "formula_pretty", "band_gap",
        "formation_energy_per_atom", "energy_above_hull",
        "is_stable", "is_metal", "nelements", "symmetry",
    ]

    print("Downloading MP (INCLUDING metals, <=4 elements)... this takes a while")
    start = time.time()
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(0.0, MAX_GAP),          # 0.0 lower bound keeps metals
            num_elements=(1, MAX_ELEMENTS),
            deprecated=False,
            fields=fields,
            chunk_size=1000,
        )
    print(f"Downloaded {len(docs):,} compounds in {time.time()-start:.0f}s")

    rows = []
    for d in docs:
        if d.band_gap is None or d.formation_energy_per_atom is None:
            continue
        sg = getattr(d.symmetry, "symbol", None) if d.symmetry is not None else None
        rows.append({
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "band_gap_ev": round(d.band_gap, 4),
            "is_metal": bool(d.is_metal) if d.is_metal is not None else (d.band_gap == 0),
            "formation_energy_per_atom": round(d.formation_energy_per_atom, 4),
            "energy_above_hull": round(d.energy_above_hull, 4) if d.energy_above_hull is not None else None,
            "is_stable": bool(d.is_stable) if d.is_stable is not None else None,
            "nelements": int(d.nelements) if d.nelements is not None else None,
            "spacegroup": sg,
        })

    df = pd.DataFrame(rows)
    out = Path("data/mp_full_data.csv")
    df.to_csv(out, index=False)

    n_metal = int((df["band_gap_ev"] == 0).sum())
    print(f"\nSaved {len(df):,} compounds to {out}")
    print(f"  metals (gap==0)     : {n_metal:,} ({100*n_metal/len(df):.1f}%)")
    print(f"  non-metals (gap>0)  : {len(df)-n_metal:,}")


if __name__ == "__main__":
    main()
