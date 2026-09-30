"""
01_download_mp.py
=================
Downloads band gap data from the Materials Project database.
Saves results to data/mp_bandgap_data.csv

Requirements:
    pip install mp-api pandas

Usage:
    python 01_download_mp.py --api-key YOUR_KEY_HERE
    # OR set environment variable:
    export MP_API_KEY="your_key_here"
    python 01_download_mp.py
"""

import os
import sys
import argparse
import time
import pandas as pd
from pathlib import Path


def load_dotenv():
    """Load key=value pairs from a .env file in the project root, if present."""
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


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download band gap data from Materials Project"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Your Materials Project API key. "
             "If not given, reads MP_API_KEY env variable.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mp_bandgap_data.csv",
        help="Output CSV file path (default: data/mp_bandgap_data.csv)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.01,
        help="Minimum band gap in eV — filters out metals (default: 0.01)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=10.0,
        help="Maximum band gap in eV — filters out wide insulators (default: 10.0)",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=4,
        help="Maximum number of elements per compound (default: 4)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of results per API request (default: 1000)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: download only 200 compounds to verify setup",
    )
    return parser.parse_args()


# ── Download function ─────────────────────────────────────────────────────────

def download_mp_data(api_key, min_gap, max_gap, max_elements, chunk_size, test_mode):
    """
    Downloads materials from Materials Project with band gap in [min_gap, max_gap].

    Returns a pandas DataFrame with columns:
        material_id, formula, band_gap, formation_energy_per_atom,
        energy_above_hull, is_stable, is_gap_direct, nelements,
        chemsys, volume, density, spacegroup
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: mp-api not installed. Run: pip install mp-api")
        sys.exit(1)

    print("\n" + "="*60)
    print("  Materials Project — Band Gap Downloader")
    print("="*60)
    print(f"  Band gap range : {min_gap} – {max_gap} eV")
    print(f"  Max elements   : {max_elements}")
    print(f"  Chunk size     : {chunk_size}")
    print(f"  Test mode      : {test_mode}")
    print("="*60 + "\n")

    fields_to_fetch = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
        "is_gap_direct",
        "is_metal",
        "nelements",
        "chemsys",
        "volume",
        "density",
        "symmetry",
    ]

    print("Connecting to Materials Project API...")
    start = time.time()

    with MPRester(api_key) as mpr:

        search_kwargs = dict(
            band_gap=(min_gap, max_gap),
            num_elements=(1, max_elements),
            deprecated=False,
            fields=fields_to_fetch,
            chunk_size=chunk_size,
        )

        if test_mode:
            # In test mode, limit to a small number for quick verification
            search_kwargs["num_chunks"] = 1
            search_kwargs["chunk_size"] = 200
            print("TEST MODE: downloading 200 compounds only...\n")
        else:
            print("Downloading — this takes 5–15 minutes for the full dataset...\n")

        docs = mpr.materials.summary.search(**search_kwargs)

    elapsed = time.time() - start
    print(f"\nDownloaded {len(docs):,} compounds in {elapsed:.1f}s")

    return docs


# ── Process into DataFrame ────────────────────────────────────────────────────

def docs_to_dataframe(docs):
    """Convert list of SummaryDoc objects to a clean pandas DataFrame."""
    print("\nProcessing results into DataFrame...")

    rows = []
    skipped = 0

    for doc in docs:
        # Skip if band gap is missing or zero (metallic)
        if doc.band_gap is None or doc.band_gap <= 0:
            skipped += 1
            continue

        # Extract spacegroup symbol safely
        spacegroup = None
        if doc.symmetry is not None:
            spacegroup = getattr(doc.symmetry, "symbol", None)

        rows.append({
            "material_id":              str(doc.material_id),
            "formula":                  doc.formula_pretty,
            "band_gap_ev":              round(doc.band_gap, 4),
            "formation_energy_per_atom": round(doc.formation_energy_per_atom, 4)
                                         if doc.formation_energy_per_atom is not None else None,
            "energy_above_hull":        round(doc.energy_above_hull, 4)
                                         if doc.energy_above_hull is not None else None,
            "is_stable":                bool(doc.is_stable) if doc.is_stable is not None else None,
            "is_gap_direct":            bool(doc.is_gap_direct) if doc.is_gap_direct is not None else None,
            "is_metal":                 bool(doc.is_metal) if doc.is_metal is not None else None,
            "nelements":                int(doc.nelements),
            "chemsys":                  doc.chemsys,
            "volume_ang3":              round(doc.volume, 3) if doc.volume is not None else None,
            "density_g_cm3":            round(doc.density, 4) if doc.density is not None else None,
            "spacegroup":               spacegroup,
        })

    df = pd.DataFrame(rows)
    print(f"Skipped {skipped:,} compounds with missing/zero band gap")
    print(f"Kept {len(df):,} compounds")
    return df


# ── Summary statistics ────────────────────────────────────────────────────────

def print_summary(df):
    """Print useful summary statistics about the downloaded dataset."""
    print("\n" + "="*60)
    print("  Dataset Summary")
    print("="*60)
    print(f"  Total compounds        : {len(df):,}")
    print(f"  Unique chemical systems: {df['chemsys'].nunique():,}")
    print(f"  Band gap range         : {df['band_gap_ev'].min():.3f} – {df['band_gap_ev'].max():.3f} eV")
    print(f"  Mean band gap          : {df['band_gap_ev'].mean():.3f} eV")
    print(f"  Median band gap        : {df['band_gap_ev'].median():.3f} eV")
    print(f"\n  Solar-relevant (1.0–2.5 eV): "
          f"{len(df[(df['band_gap_ev'] >= 1.0) & (df['band_gap_ev'] <= 2.5)]):,} compounds")

    print("\n  Compounds by number of elements:")
    for n, count in df['nelements'].value_counts().sort_index().items():
        bar = "█" * (count // max(1, len(df) // 40))
        print(f"    {n} elements: {count:>6,}  {bar}")

    if df['is_stable'].notna().any():
        n_stable = df['is_stable'].sum()
        print(f"\n  Stable (on convex hull): {n_stable:,} "
              f"({100*n_stable/len(df):.1f}%)")

    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print("\nERROR: No API key found.")
        print("Either pass --api-key YOUR_KEY or set: export MP_API_KEY='your_key'")
        print("Get your free key at: https://materialsproject.org (Dashboard)")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    docs = download_mp_data(
        api_key=api_key,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        max_elements=args.max_elements,
        chunk_size=args.chunk_size,
        test_mode=args.test,
    )

    # Convert to DataFrame
    df = docs_to_dataframe(docs)

    # Print summary
    print_summary(df)

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path.resolve()}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nDone! Next step: run 02_download_oqmd.py")
    print("Or run 03_featurize.py to start building features from this data.\n")


if __name__ == "__main__":
    main()
