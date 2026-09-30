"""
05_screen_candidates.py
========================
The combined screener entry point. Two modes:

    1. --formulas-file path.txt   — score a list of formulas you already
       have in mind (one per line, or a CSV with a 'formula' column).
    2. --elements Ga,N,Al,In,O    — generate candidate binary/ternary
       compositions from an element set and rank them (real discovery
       use case: "what should I look at within this element set?").

Either way: featurizes with the same matminer stack used in training,
runs both trained models (screener.py), and writes a ranked CSV +
a scatter plot of the results.

Usage:
    python 05_screen_candidates.py --elements Ga,N,Al,In,O
    python 05_screen_candidates.py --formulas-file my_candidates.txt
"""
import argparse
import math
from itertools import combinations, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from screener import predict_candidates


def is_charge_balanced(formula):
    """
    True if pymatgen finds at least one oxidation-state assignment that
    balances to neutral (common-valence guesses only — fast). Filters out
    combinatorial noise like 'Ga3Al2O3' before it ever reaches the models,
    so model calls aren't wasted on stoichiometries with no plausible ionic
    structure.
    """
    from pymatgen.core import Composition
    try:
        guesses = Composition(formula).oxi_state_guesses(max_sites=-1)
        return len(guesses) > 0
    except Exception:
        return False


def generate_combinatorial(elements, max_stoich=4, include_ternary=True, require_charge_balance=True):
    """
    Generates candidate formulas for all binary (and optionally ternary)
    combinations of the given elements, with reduced integer stoichiometries
    up to max_stoich (e.g. AB, AB2, A2B3, ...).

    By default, filters to formulas with at least one charge-balanced
    oxidation-state assignment (require_charge_balance=True) — without
    this, brute-force combinatorics generates mostly chemically
    implausible stoichiometries (e.g. 'Ga3Al2O3') that waste model calls
    and clutter results with predictable "unstable" predictions.
    """
    formulas = set()
    ratios = list(range(1, max_stoich + 1))

    for a, b in combinations(elements, 2):
        for r1, r2 in product(ratios, ratios):
            g = math.gcd(r1, r2)
            x, y = r1 // g, r2 // g
            f = f"{a}{x if x > 1 else ''}{b}{y if y > 1 else ''}"
            formulas.add(f)

    if include_ternary and len(elements) >= 3:
        for a, b, c in combinations(elements, 3):
            for r1, r2, r3 in product(ratios[:3], ratios[:3], ratios[:3]):
                g = math.gcd(math.gcd(r1, r2), r3)
                x, y, z = r1 // g, r2 // g, r3 // g
                f = f"{a}{x if x > 1 else ''}{b}{y if y > 1 else ''}{c}{z if z > 1 else ''}"
                formulas.add(f)

    if require_charge_balance:
        before = len(formulas)
        formulas = {f for f in formulas if is_charge_balanced(f)}
        print(f"  Charge-balance filter: {before:,} -> {len(formulas):,} formulas")

    return sorted(formulas)


def load_formulas_from_file(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        col = "formula" if "formula" in df.columns else df.columns[0]
        return df[col].dropna().astype(str).tolist()
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Screen candidate materials with both trained models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--formulas-file", type=str, help="Path to a .txt (one formula/line) or .csv (column 'formula')")
    group.add_argument("--elements", type=str, help="Comma-separated element list, e.g. Ga,N,Al,In,O")
    parser.add_argument("--max-stoich", type=int, default=4, help="Max stoichiometric ratio for combinatorial generation (default: 4)")
    parser.add_argument("--no-ternary", action="store_true", help="Skip ternary combinations (binaries only)")
    parser.add_argument("--no-charge-filter", action="store_true", help="Skip the charge-balance filter (keep chemically implausible stoichiometries)")
    parser.add_argument("--output", type=str, default="screening_results.csv")
    parser.add_argument("--top", type=int, default=20, help="How many top candidates to print")
    args = parser.parse_args()

    if args.formulas_file:
        formulas = load_formulas_from_file(args.formulas_file)
        print(f"Loaded {len(formulas):,} formulas from {args.formulas_file}")
    else:
        elements = [e.strip() for e in args.elements.split(",")]
        print(f"Generating candidates from elements: {elements}")
        formulas = generate_combinatorial(
            elements,
            max_stoich=args.max_stoich,
            include_ternary=not args.no_ternary,
            require_charge_balance=not args.no_charge_filter,
        )
        print(f"Generated {len(formulas):,} candidate formulas")

    results = predict_candidates(formulas)
    if results.empty:
        print("No valid candidates scored.")
        return

    results.to_csv(args.output, index=False)
    print(f"\nSaved full results to {args.output}")

    print(f"\nTop {args.top} promising candidates (stable + low formation energy):")
    promising = results[results["is_promising"]].head(args.top)
    print(promising.to_string(index=False) if len(promising) else "  (none flagged as promising)")

    # ── Plot: formation energy vs stability probability ────────────────────
    plt.figure(figsize=(7, 6))
    colors = results["is_promising"].map({True: "seagreen", False: "lightgray"})
    plt.scatter(
        results["formation_energy_pred_ev_atom"],
        results["stability_proba"],
        c=colors, alpha=0.6, s=20,
    )
    plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Stability decision boundary")
    plt.xlabel("Predicted formation energy (eV/atom)")
    plt.ylabel("Predicted stability probability")
    plt.title(f"Screening results — {len(results):,} candidates\n(green = flagged promising)")
    plt.legend()
    plt.tight_layout()
    plot_path = "plots/screening_results.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
