"""
02_download_oqmd.py  (fast version)
=====================================
Downloads stability data from OQMD.
No API key needed.

Changes from previous version:
- Bigger batches (1000 instead of 500) = fewer requests = faster
- Longer timeout (60s instead of 30s) = fewer fake timeouts
- Parallel requests using ThreadPoolExecutor = 3x faster
- Progress bar shows estimated time remaining

Usage:
    python 02_download_oqmd.py --test        # 2000 compounds, ~30 seconds
    python 02_download_oqmd.py --max 50000   # ~10 minutes
    python 02_download_oqmd.py               # full dataset ~45 min
"""

import sys, time, argparse, requests, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL   = "http://oqmd.org/oqmdapi/formationenergy"
FIELDS     = "name,entry_id,ntypes,delta_e,stability,band_gap"
BATCH_SIZE = 1000   # bigger = fewer requests = faster
TIMEOUT    = 60     # seconds — generous to avoid fake timeouts
WORKERS    = 3      # parallel requests (don't go higher — be polite)

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",       default="data/oqmd_stability_data.csv")
    p.add_argument("--max",          type=int, default=None)
    p.add_argument("--max-elements", type=int, default=4)
    p.add_argument("--test",         action="store_true")
    return p.parse_args()

# ── Fetch one batch ───────────────────────────────────────────────────────────

def fetch_batch(offset, limit, retries=3):
    params = {
        "fields": FIELDS,
        "limit":  limit,
        "offset": offset,
        "format": "json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return offset, r.json().get("data", [])
        except requests.exceptions.Timeout:
            wait = 3 * (attempt + 1)
            print(f"  Timeout at offset {offset}, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Error at offset {offset}: {e}")
            time.sleep(3)
    print(f"  SKIPPING offset {offset} after {retries} attempts")
    return offset, []

# ── Process one record ────────────────────────────────────────────────────────

def process(rec, max_elements):
    if not rec.get("name") or rec.get("stability") is None:
        return None
    if rec.get("ntypes") and rec["ntypes"] > max_elements:
        return None

    stability = rec["stability"]
    if stability <= 0.0:
        label = 1
    elif stability > 0.1:
        label = 0
    else:
        return None  # borderline — skip

    return {
        "formula":      rec["name"],
        "entry_id":     rec.get("entry_id"),
        "ntypes":       rec.get("ntypes"),
        "delta_e":      round(rec["delta_e"],   5) if rec.get("delta_e")  is not None else None,
        "stability":    round(stability, 5),
        "band_gap_ev":  round(rec["band_gap"],  4) if rec.get("band_gap") is not None else None,
        "stable_label": label,
    }

# ── Main download — parallel ──────────────────────────────────────────────────

def download(max_compounds, max_elements, test_mode):
    if test_mode:
        target = 2000
        print("TEST MODE: 2000 compounds\n")
    else:
        target = max_compounds or 815_000

    # Build list of all (offset, batch_size) jobs upfront
    jobs = []
    offset = 0
    while offset < target:
        size = min(BATCH_SIZE, target - offset)
        jobs.append((offset, size))
        offset += size

    total_jobs = len(jobs)
    print(f"Total batches to fetch : {total_jobs}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Parallel workers       : {WORKERS}")
    print(f"Estimated time         : ~{max(1, total_jobs // (WORKERS * 6))} minutes\n")
    print(f"{'Done':>6}  {'Fetched':>9}  {'Kept':>8}  {'Stable':>8}  {'Unstable':>9}  {'Speed':>8}")
    print("-" * 60)

    all_records   = []
    total_fetched = 0
    jobs_done     = 0
    start         = time.time()

    # Process in chunks of WORKERS batches at a time
    chunk_size = WORKERS * 4
    for chunk_start in range(0, len(jobs), chunk_size):
        chunk = jobs[chunk_start : chunk_start + chunk_size]

        # Fetch this chunk in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(fetch_batch, off, sz): off for off, sz in chunk}
            for fut in as_completed(futures):
                off, data = fut.result()
                results[off] = data

        # Process in offset order (keeps CSV consistent)
        for off, sz in chunk:
            raw = results.get(off, [])
            for rec in raw:
                p = process(rec, max_elements)
                if p:
                    all_records.append(p)
            total_fetched += len(raw)
            jobs_done     += 1

        elapsed   = time.time() - start
        speed     = total_fetched / elapsed if elapsed > 0 else 0
        n_stable  = sum(1 for r in all_records if r["stable_label"] == 1)
        n_unstable= sum(1 for r in all_records if r["stable_label"] == 0)

        pct = 100 * jobs_done / total_jobs
        print(f"{pct:>5.1f}%  {total_fetched:>9,}  {len(all_records):>8,}  "
              f"{n_stable:>8,}  {n_unstable:>9,}  {speed:>6.0f}/s")

        # Checkpoint every 10k compounds
        if total_fetched % 10_000 < chunk_size * BATCH_SIZE and all_records:
            cp = Path("data/oqmd_checkpoint.csv")
            cp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_records).to_csv(cp, index=False)
            print(f"  [Checkpoint: {len(all_records):,} records]")

    return all_records

# ── Summary ───────────────────────────────────────────────────────────────────

def summary(df):
    print("\n" + "="*50)
    print("  OQMD Download Complete")
    print("="*50)
    n1 = (df.stable_label == 1).sum()
    n0 = (df.stable_label == 0).sum()
    print(f"  Total records   : {len(df):,}")
    print(f"  Stable  (1)     : {n1:,}  ({100*n1/len(df):.1f}%)")
    print(f"  Unstable (0)    : {n0:,}  ({100*n0/len(df):.1f}%)")
    gdf = df.band_gap_ev.dropna()
    if len(gdf):
        print(f"  With band gap   : {len(gdf):,}")
    print("="*50)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "="*50)
    print("  OQMD Fast Downloader")
    print("="*50)
    print(f"  Output      : {args.output}")
    print(f"  Max elements: {args.max_elements}")
    print(f"  Max records : {args.max or 'all'}")
    print(f"  Test mode   : {args.test}")
    print("="*50 + "\n")

    print("Checking connection...")
    try:
        r = requests.get(f"{BASE_URL}?fields=name&limit=1&format=json", timeout=15)
        r.raise_for_status()
        print("Connected!\n")
    except Exception as e:
        print(f"Cannot connect: {e}")
        sys.exit(1)

    records = download(args.max, args.max_elements, args.test)

    if not records:
        print("No records downloaded.")
        sys.exit(1)

    df = pd.DataFrame(records)
    summary(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    mb = out.stat().st_size / 1024 / 1024
    print(f"\nSaved : {out.resolve()}")
    print(f"Size  : {mb:.1f} MB  |  Rows: {len(df):,}")
    print("\nNext step: run 03_featurize.py\n")

if __name__ == "__main__":
    main()
