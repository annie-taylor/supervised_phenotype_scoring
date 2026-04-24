#!/usr/bin/env python3
"""
analyze_filter.py — Evaluate the automated song filter against prescreen labels.

For each prescreened snippet, re-runs ``score_song_candidate()`` on the
original audio and records the raw feature values (``max_segments``,
``n_windows_passing``).  Outputs a feature CSV, a distribution plot, and a
table of threshold candidates ranked by F1 score.

Usage
-----
    # one or more specific batches
    python analyze_filter.py batches/bk37wh86_rd75wh72_20260414

    # all batches that have a prescreen CSV
    python analyze_filter.py --all-batches

    # custom output directory
    python analyze_filter.py --all-batches --out results/my_analysis

Output
------
``<out_dir>/features.csv``
    One row per labeled snippet (rendering_error excluded).
    Columns: uid, batch, label, max_segments, n_windows_passing, threshold,
    passed_current.

``<out_dir>/distributions.png``
    Overlapping histograms of max_segments and n_windows_passing split by
    song / not_song label, with current threshold lines marked.

Threshold candidates are printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tables

SCORING_DIR = Path(__file__).resolve().parent
if str(SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_DIR))

import family_spec_generation as fsg

DEFAULT_MIN_SEGMENTS = 8
DEFAULT_MIN_WINDOWS  = 3


# ── Batch discovery ───────────────────────────────────────────────────────────

def find_prescreen_csv(batch_dir: Path) -> Path | None:
    """Return the most recent prescreen_*.csv in batch_dir, or None."""
    csvs = sorted(batch_dir.glob("prescreen_*.csv"))
    return csvs[-1] if csvs else None


def discover_batches(batches_root: Path) -> list[Path]:
    """Return all batch dirs under batches_root that have a prescreen CSV."""
    return [
        d for d in sorted(batches_root.iterdir())
        if d.is_dir() and find_prescreen_csv(d) is not None
    ]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_manifest(h5_path: Path) -> tuple[dict[str, dict], float]:
    """Return ({uid: {source_file, snippet_start_s, snippet_duration_s}}, snippet_duration_s)."""
    rows: dict[str, dict] = {}
    duration_s = 8.0
    with tables.open_file(str(h5_path), mode="r") as h5:
        cfg_attrs = h5.root.config._v_attrs
        if "snippet_duration_s" in cfg_attrs._v_attrnames:
            duration_s = float(cfg_attrs["snippet_duration_s"])
        for row in h5.root.manifest.iterrows():
            uid = row["uid"].decode()
            rows[uid] = {
                "source_file":      row["source_file"].decode(),
                "snippet_start_s":  float(row["snippet_start_s"]),
                "snippet_duration_s": float(row["snippet_duration_s"]),
            }
    return rows, duration_s


def load_prescreen_labels(csv_path: Path) -> dict[str, str]:
    """Return {uid: label} from a prescreen CSV."""
    labels: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["uid"]] = row["label"]
    return labels


def collect_records(batch_dirs: list[Path]) -> list[dict]:
    """
    For each batch, join prescreen labels with manifest metadata.
    Returns a flat list of dicts (rendering_error rows excluded).
    """
    records: list[dict] = []
    for batch_dir in batch_dirs:
        csv_path = find_prescreen_csv(batch_dir)
        h5_path  = batch_dir / "batch.h5"
        if csv_path is None or not h5_path.exists():
            print(f"  skip {batch_dir.name}: missing batch.h5 or prescreen CSV")
            continue

        manifest, _ = load_manifest(h5_path)
        labels       = load_prescreen_labels(csv_path)

        for uid, label in labels.items():
            if label == "rendering_error":
                continue
            if uid not in manifest:
                print(f"  warning: uid {uid[:8]}… not in manifest — skipping")
                continue
            rec = {"uid": uid, "batch": batch_dir.name, "label": label}
            rec.update(manifest[uid])
            records.append(rec)

        print(f"  {batch_dir.name}: {sum(1 for r in records if r['batch'] == batch_dir.name)} usable labels")
    return records


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(records: list[dict]) -> list[dict]:
    """
    Re-run score_song_candidate() on each snippet's audio window.
    Skips records where the source file is inaccessible.
    Returns enriched records with max_segments, n_windows_passing, threshold,
    passed_current added.
    """
    results: list[dict] = []
    n_skipped = 0

    for i, rec in enumerate(records):
        src  = rec["source_file"]
        if not Path(src).exists():
            n_skipped += 1
            continue

        try:
            audio_full, sr = fsg.read_audio_file(src)
        except Exception as e:
            print(f"  could not read {Path(src).name}: {e}")
            n_skipped += 1
            continue

        start_s  = rec["snippet_start_s"]
        dur_s    = rec["snippet_duration_s"]
        start_i  = int(round(start_s * sr))
        end_i    = start_i + int(round(dur_s * sr))
        snippet  = audio_full[start_i:end_i]

        if len(snippet) == 0:
            n_skipped += 1
            continue

        res = fsg.score_song_candidate(snippet, sr)

        out = dict(rec)
        out["max_segments"]      = res["max_segments"]
        out["n_windows_passing"] = res.get("n_windows_passing", 0)
        out["threshold"]         = round(float(res["threshold"]), 6)
        out["passed_current"]    = res["passed"]
        results.append(out)

        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            print(f"  {i + 1}/{len(records)} snippets processed", end="\r")

    print()
    if n_skipped:
        print(f"  {n_skipped} snippet(s) skipped (missing or unreadable audio)")
    return results


# ── Output ────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "uid", "batch", "label",
    "max_segments", "n_windows_passing", "threshold", "passed_current",
]


def write_csv(results: list[dict], out_dir: Path) -> Path:
    path = out_dir / "features.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return path


def plot_distributions(results: list[dict], out_dir: Path) -> Path:
    songs    = [r for r in results if r["label"] == "song"]
    notsongs = [r for r in results if r["label"] == "not_song"]

    seg_song    = [r["max_segments"]      for r in songs]
    seg_noise   = [r["max_segments"]      for r in notsongs]
    win_song    = [r["n_windows_passing"] for r in songs]
    win_noise   = [r["n_windows_passing"] for r in notsongs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Song filter feature distributions  "
        f"(n_song={len(songs)}, n_not_song={len(notsongs)})",
        fontsize=13,
    )

    max_seg = max(max(seg_song, default=0), max(seg_noise, default=0), DEFAULT_MIN_SEGMENTS + 5)
    bins_seg = range(0, max_seg + 2)

    ax = axes[0]
    ax.hist(seg_noise, bins=bins_seg, alpha=0.6, label="not_song", color="tomato")
    ax.hist(seg_song,  bins=bins_seg, alpha=0.6, label="song",     color="steelblue")
    ax.axvline(DEFAULT_MIN_SEGMENTS, color="black", linestyle="--",
               label=f"current threshold ({DEFAULT_MIN_SEGMENTS})")
    ax.set_xlabel("max_segments (notes in best 2-s window)")
    ax.set_ylabel("count")
    ax.set_title("max_segments")
    ax.legend()

    max_win = max(max(win_song, default=0), max(win_noise, default=0), DEFAULT_MIN_WINDOWS + 3)
    bins_win = range(0, max_win + 2)

    ax = axes[1]
    ax.hist(win_noise, bins=bins_win, alpha=0.6, label="not_song", color="tomato")
    ax.hist(win_song,  bins=bins_win, alpha=0.6, label="song",     color="steelblue")
    ax.axvline(DEFAULT_MIN_WINDOWS, color="black", linestyle="--",
               label=f"current threshold ({DEFAULT_MIN_WINDOWS})")
    ax.set_xlabel("n_windows_passing (windows with sustained activity)")
    ax.set_ylabel("count")
    ax.set_title("n_windows_passing")
    ax.legend()

    plt.tight_layout()
    path = out_dir / "distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def print_threshold_table(results: list[dict]) -> None:
    """Grid search over (min_segments, min_windows) and print top candidates by F1."""
    truth = [r["label"] == "song" for r in results]
    n     = len(truth)
    if n == 0:
        return

    rows: list[dict] = []
    for ms in range(1, 21):
        for mw in range(1, 11):
            soft = ceil(ms * 0.6)
            preds = [
                r["max_segments"] >= ms and r["n_windows_passing"] >= mw
                for r in results
            ]
            tp = sum(p and t for p, t in zip(preds, truth))
            fp = sum(p and not t for p, t in zip(preds, truth))
            fn = sum(not p and t for p, t in zip(preds, truth))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            rows.append({
                "min_segments": ms, "min_windows": mw,
                "precision": precision, "recall": recall, "f1": f1,
            })

    rows.sort(key=lambda r: -r["f1"])

    print("\n── Current filter performance ──────────────────────────────────────")
    current = next(
        (r for r in rows
         if r["min_segments"] == DEFAULT_MIN_SEGMENTS
         and r["min_windows"] == DEFAULT_MIN_WINDOWS),
        None,
    )
    if current:
        print(f"  min_segments={DEFAULT_MIN_SEGMENTS}, min_windows={DEFAULT_MIN_WINDOWS}"
              f"  →  precision={current['precision']:.3f}  "
              f"recall={current['recall']:.3f}  F1={current['f1']:.3f}")

    print("\n── Top 10 threshold candidates (by F1) ─────────────────────────────")
    print(f"  {'min_seg':>7}  {'min_win':>7}  {'precision':>9}  {'recall':>6}  {'F1':>6}")
    for r in rows[:10]:
        marker = " ◄ current" if (
            r["min_segments"] == DEFAULT_MIN_SEGMENTS
            and r["min_windows"] == DEFAULT_MIN_WINDOWS
        ) else ""
        print(f"  {r['min_segments']:>7}  {r['min_windows']:>7}"
              f"  {r['precision']:>9.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}{marker}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the automated song filter against prescreen labels."
    )
    parser.add_argument(
        "batch_dirs", nargs="*",
        help="Batch directories to analyze (e.g. batches/bk37wh86_rd75wh72_20260414)",
    )
    parser.add_argument(
        "--all-batches", action="store_true",
        help="Auto-discover all batch directories that have a prescreen CSV",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: results/filter_analysis_<YYYYMMDD>)",
    )
    args = parser.parse_args()

    if not args.batch_dirs and not args.all_batches:
        parser.error("Provide at least one batch_dir or use --all-batches")

    if args.all_batches:
        batches_root = SCORING_DIR / "batches"
        batch_dirs   = discover_batches(batches_root)
        print(f"Found {len(batch_dirs)} batch(es) with prescreen CSV")
    else:
        batch_dirs = [Path(p).resolve() for p in args.batch_dirs]

    out_dir = Path(args.out) if args.out else (
        SCORING_DIR / "results" / f"filter_analysis_{date.today().strftime('%Y%m%d')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nCollecting labeled snippets...")
    records = collect_records(batch_dirs)
    if not records:
        print("No usable records found — check that batch.h5 and prescreen CSVs exist.")
        return
    print(f"  {len(records)} total labeled snippets across {len(batch_dirs)} batch(es)")

    print("\nExtracting features (loading audio)...")
    results = extract_features(records)
    if not results:
        print("No features extracted — are audio files accessible?")
        return
    print(f"  {len(results)} snippets with features")

    csv_path  = write_csv(results, out_dir)
    plot_path = plot_distributions(results, out_dir)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {plot_path}")

    print_threshold_table(results)


if __name__ == "__main__":
    main()
