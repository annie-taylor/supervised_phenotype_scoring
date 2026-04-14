#!/usr/bin/env python3
"""
run_pipeline.py — Batch wrapper for the supervised_phenotype_scoring pipeline.

Automates prepare_batch.py + export_batch.py for a list of nest-father ×
genetic-father pairings, in two phases separated by a manual prescreen step.

Phase 1  (before prescreen)
    For each pairing: prepare_batch.py → export_batch.py
    Run prescreen_app.py on each batch before proceeding to Phase 2.

Phase 2  (after prescreen)
    For each pairing: find the prescreen CSV from Phase 1, then
    prepare_batch.py --exclude-csv → export_batch.py on the new batch.

Usage
-----
::

    # Phase 1: build and export all batches
    python run_pipeline.py pairings.csv --phase 1

    # (run prescreen_app.py on each batch, label all spectrograms)

    # Phase 2: rebuild with exclusions and re-export
    python run_pipeline.py pairings.csv --phase 2

Pairings CSV
------------
Two columns, one pairing per row.  A header row is optional.

    nest_father,genetic_father
    pk24bu3,wh88br85
    ab12cd3,ef45gh6

Pass-through options (forwarded to prepare_batch.py / export_batch.py)
-----------------------------------------------------------------------
--snippets-per-bird N
--workers N
--dpi N
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCORING_DIR = Path(__file__).resolve().parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pairings(csv_path: Path) -> list[tuple[str, str]]:
    """
    Load (nest_father, genetic_father) pairs from a CSV file.
    Skips blank lines and comment lines starting with ``#``.
    Accepts files with or without a header row.
    """
    pairings = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            nf, gf = row[0].strip(), row[1].strip()
            if nf.lower() in ("nest_father", "nf") and gf.lower() in ("genetic_father", "gf"):
                continue  # skip header row
            pairings.append((nf, gf))
    return pairings


def find_latest_batch_dir(output_dir: Path, nf: str, gf: str) -> Path | None:
    """Return the most recently created batch directory for a pairing."""
    dirs = sorted(output_dir.glob(f"{nf}_{gf}_*"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def find_prescreen_csv(batch_dir: Path) -> Path | None:
    """Return the most recent prescreen CSV in a batch directory, or None."""
    csvs = sorted(batch_dir.glob("prescreen_*.csv"), reverse=True)
    return csvs[0] if csvs else None


def today_batch_dir(output_dir: Path, nf: str, gf: str) -> Path:
    """Return the expected batch directory path for today's date."""
    return output_dir / f"{nf}_{gf}_{datetime.now().strftime('%Y%m%d')}"


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess command, stream output, return True on success."""
    print(f"\n  >>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCORING_DIR))
    if result.returncode != 0:
        print(f"  ✗ {label} failed (exit {result.returncode})")
        return False
    print(f"  ✓ {label} done")
    return True


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1(pairings: list[tuple[str, str]], args: argparse.Namespace,
           output_dir: Path) -> None:
    """Build and export initial batches for all pairings."""
    print(f"\n{'='*60}")
    print(f"PHASE 1 — Building {len(pairings)} batch(es)")
    print(f"{'='*60}")

    succeeded, failed = [], []

    for nf, gf in pairings:
        label = f"{nf} × {gf}"
        print(f"\n--- {label} ---")

        # prepare_batch.py
        cmd = [sys.executable, "prepare_batch.py",
               "--nest-father", nf, "--genetic-father", gf]
        if args.snippets_per_bird:
            cmd += ["--snippets-per-bird", str(args.snippets_per_bird)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if not run(cmd, f"prepare_batch [{label}]"):
            failed.append(label)
            continue

        # export_batch.py — use today's batch dir
        batch_dir = today_batch_dir(output_dir, nf, gf)
        if not batch_dir.exists():
            print(f"  ✗ expected batch dir not found: {batch_dir}")
            failed.append(label)
            continue

        cmd = [sys.executable, "export_batch.py", str(batch_dir)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.dpi:
            cmd += ["--dpi", str(args.dpi)]
        if not run(cmd, f"export_batch [{label}]"):
            failed.append(label)
            continue

        succeeded.append((label, batch_dir))

    _print_summary(succeeded, failed, phase=1)

    if succeeded:
        print("\nNext steps:")
        for label, batch_dir in succeeded:
            print(f"  python prescreen_app.py {batch_dir}")
        print("\nOnce all batches are prescreened, run:")
        print(f"  python run_pipeline.py {args.pairings_csv} --phase 2")


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2(pairings: list[tuple[str, str]], args: argparse.Namespace,
           output_dir: Path) -> None:
    """Rebuild batches with prescreen exclusions and re-export."""
    print(f"\n{'='*60}")
    print(f"PHASE 2 — Rebuilding {len(pairings)} batch(es) with exclusions")
    print(f"{'='*60}")

    succeeded, failed, skipped = [], [], []

    for nf, gf in pairings:
        label = f"{nf} × {gf}"
        print(f"\n--- {label} ---")

        # Find the Phase 1 batch dir and prescreen CSV
        phase1_dir = find_latest_batch_dir(output_dir, nf, gf)
        if phase1_dir is None:
            print(f"  ✗ no existing batch directory found — run Phase 1 first")
            failed.append(label)
            continue

        prescreen_csv = find_prescreen_csv(phase1_dir)
        if prescreen_csv is None:
            print(f"  ✗ no prescreen CSV found in {phase1_dir}")
            print(f"     run: python prescreen_app.py {phase1_dir}")
            skipped.append(label)
            continue

        print(f"  prescreen CSV: {prescreen_csv.name}")

        # prepare_batch.py --exclude-csv
        cmd = [sys.executable, "prepare_batch.py",
               "--nest-father", nf, "--genetic-father", gf,
               "--exclude-csv", str(prescreen_csv)]
        if args.snippets_per_bird:
            cmd += ["--snippets-per-bird", str(args.snippets_per_bird)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if not run(cmd, f"prepare_batch --exclude-csv [{label}]"):
            failed.append(label)
            continue

        # export_batch.py — use today's batch dir (may be same or new)
        batch_dir = today_batch_dir(output_dir, nf, gf)
        if not batch_dir.exists():
            print(f"  ✗ expected batch dir not found: {batch_dir}")
            failed.append(label)
            continue

        cmd = [sys.executable, "export_batch.py", str(batch_dir)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.dpi:
            cmd += ["--dpi", str(args.dpi)]
        if not run(cmd, f"export_batch [{label}]"):
            failed.append(label)
            continue

        succeeded.append((label, batch_dir))

    _print_summary(succeeded, failed, phase=2, skipped=skipped)

    if succeeded:
        print("\nBatches ready for upload to EC2:")
        for label, batch_dir in succeeded:
            print(f"  {batch_dir.name}")
        print("\nUpload each batch (run in PowerShell):")
        for _, batch_dir in succeeded:
            print(f'  scp -i "C:\\Users\\Eric\\.ssh\\scoring-key" -r '
                  f'"{batch_dir}" '
                  f'ubuntu@<public-ip>:~/supervised_phenotype_scoring/batches/')


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(succeeded: list, failed: list, phase: int,
                   skipped: list | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"Phase {phase} summary")
    print(f"  Succeeded : {len(succeeded)}")
    if skipped:
        print(f"  Skipped   : {len(skipped)} (prescreen not yet complete)")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("  Failed pairings:")
        for f in failed:
            print(f"    - {f}")
    if skipped:
        print("  Skipped pairings (run prescreen first):")
        for s in skipped:
            print(f"    - {s}")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch wrapper for the supervised_phenotype_scoring pipeline."
    )
    parser.add_argument("pairings_csv",
                        help="CSV with nest_father,genetic_father columns")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True,
                        help="1 = build + export;  2 = rebuild with exclusions + export")
    parser.add_argument("--snippets-per-bird", type=int,
                        help="Override config snippets_per_bird")
    parser.add_argument("--workers", type=int,
                        help="Override config n_workers")
    parser.add_argument("--dpi", type=int,
                        help="Override export DPI for PNG spectrograms")
    args = parser.parse_args()

    pairings_path = Path(args.pairings_csv).resolve()
    if not pairings_path.exists():
        print(f"Error: pairings CSV not found: {pairings_path}")
        sys.exit(1)

    pairings = load_pairings(pairings_path)
    if not pairings:
        print("Error: no pairings found in CSV.")
        sys.exit(1)

    print(f"Loaded {len(pairings)} pairing(s) from {pairings_path.name}")

    # Read output_dir from config
    cfg_path = SCORING_DIR / "config.json"
    import json
    with open(cfg_path) as f:
        output_dir = Path(json.load(f)["output_dir"])

    if args.phase == 1:
        phase1(pairings, args, output_dir)
    else:
        phase2(pairings, args, output_dir)


if __name__ == "__main__":
    main()
