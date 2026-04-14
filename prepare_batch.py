#!/usr/bin/env python3
"""
prepare_batch.py — Build a scoring batch for one nest-father × genetic-father pairing.

Creates  E:/scoring/batches/<nf>_<gf>_<YYYYMMDD>/batch.h5  containing:
  - Anonymized spectrograms  (EArray, uniform shape: n_snippets × freq_bins × time_bins)
  - Raw audio snippets       (VLArray, float32, one row per snippet)
  - Manifest table           (UUID → bird identity — never exposed to scorers)
  - Shared frequency axis    (Array, Hz)

Spectrograms are computed in parallel; HDF5 is written serially from the main
process (tables / PyTables does not support concurrent writes).

Usage
-----
::

    python prepare_batch.py --nest-father pk24bu3 --genetic-father wh88br85
    python prepare_batch.py --nest-father pk24bu3 --genetic-father wh88br85 --snippets-per-bird 8 --workers 6

Next step: run export_batch.py to generate PNGs and WAVs from the HDF5.
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tables

# ── Paths ───────────────────────────────────────────────────────────────────────

SCORING_DIR  = Path(__file__).resolve().parent
DEFAULT_CFG  = SCORING_DIR / "config.json"


# ── Config ──────────────────────────────────────────────────────────────────────

def load_config(path: Path = DEFAULT_CFG) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("snippets_per_bird",  6)
    cfg.setdefault("snippet_duration_s", 8.0)
    cfg.setdefault("min_gap_s",          5.0)
    cfg.setdefault("edge_s",             1.0)
    cfg.setdefault("seed",               42)
    cfg.setdefault("target_sr",          32000)
    cfg.setdefault("n_workers",          4)
    cfg.setdefault("spectrogram", {
        "nfft": 1024, "hop": 256,
        "min_freq": 400, "max_freq": 10000,
        "p_low": 2, "p_high": 98,
    })
    return cfg


# ── Exclusion list ───────────────────────────────────────────────────────────────

def load_exclude_set(csv_path: Path) -> set[tuple[str, float]]:
    """
    Load a set of (source_file, snippet_start_s) pairs to exclude from a
    prescreen CSV.  Only rows with label ``not_song`` or ``rendering_error``
    are excluded; ``song`` rows are ignored.

    The start time is rounded to 2 decimal places for robust matching across
    floating-point representations.
    """
    exclude: set[tuple[str, float]] = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label", "not_song") != "song":
                exclude.add((row["source_file"],
                             round(float(row["snippet_start_s"]), 2)))
    return exclude


# ── Bird registry ────────────────────────────────────────────────────────────────

def _parse_bird_list(s: Any) -> list[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [b.strip() for b in str(s).split(";") if b.strip()]


def load_bird_registry(fm_dir: str, nest_father: str, genetic_father: str) -> dict:
    """
    Return bird_registry dict for the requested pairing.
    Keys are bird IDs; values are {role, nest_father, genetic_father}.
    """
    fm = Path(fm_dir)
    pair_df = pd.read_csv(fm / "nest_gen_pair_offspring_summary.csv")
    gf_df   = pd.read_csv(fm / "genetic_father_offspring_summary.csv")

    gf_hr_pool = {
        row["Genetic Father"]: _parse_bird_list(row.get("HR Birds", ""))
        for _, row in gf_df.iterrows()
    }

    row_df = pair_df[
        (pair_df["Nest Father"]    == nest_father) &
        (pair_df["Genetic Father"] == genetic_father)
    ]
    if row_df.empty:
        raise ValueError(
            f"Pairing {nest_father} × {genetic_father} not found in "
            "nest_gen_pair_offspring_summary.csv"
        )

    row = row_df.iloc[0]
    nf, gf = str(row["Nest Father"]), str(row["Genetic Father"])

    xf_birds   = [b for b in _parse_bird_list(row.get("XF Birds",  "")) if b not in (nf, gf)]
    hr_nf_pool = [b for b in gf_hr_pool.get(nf, [])                     if b not in (nf, gf)]
    hr_gf_pool = [b for b in gf_hr_pool.get(gf, [])                     if b not in (nf, gf)]

    registry: dict = {}

    def reg(bird: str, role: str) -> None:
        if bird and bird not in registry:
            registry[bird] = {"role": role, "nest_father": nf, "genetic_father": gf}

    reg(nf, "nest_father")
    reg(gf, "genetic_father")
    for b in xf_birds:   reg(b, "xf")
    for b in hr_nf_pool: reg(b, "hr_nest")
    for b in hr_gf_pool: reg(b, "hr_genetic")

    return registry


# ── Duration helper ──────────────────────────────────────────────────────────────

def get_audio_duration(filepath: str) -> float | None:
    """
    Return audio duration in seconds without loading the full file.
    Uses soundfile.info() when available (fast); falls back to WAV header
    parsing, then to None (caller will estimate from file size).
    """
    try:
        import soundfile as sf
        return sf.info(filepath).duration
    except Exception:
        pass

    # WAV header fallback: read sample-rate and data-chunk size
    try:
        with open(filepath, "rb") as fh:
            header = fh.read(44)
        if len(header) >= 44 and header[:4] == b"RIFF":
            sr         = int.from_bytes(header[24:28], "little")
            bits       = int.from_bytes(header[34:36], "little")
            data_bytes = int.from_bytes(header[40:44], "little")
            if sr > 0 and bits > 0:
                return data_bytes / (sr * (bits // 8))
    except Exception:
        pass

    return None


# ── Snippet planning ─────────────────────────────────────────────────────────────

def _sample_positions(
    duration_s: float,
    snippet_s: float,
    n: int,
    min_gap_s: float,
    edge_s: float,
    rng: np.random.Generator,
) -> list[float]:
    """
    Sample up to n non-overlapping start positions from a single recording.
    Positions are drawn uniformly from [edge_s, duration_s - snippet_s - edge_s]
    with a minimum separation of  min_gap_s + snippet_s  between starts
    (so snippets don't overlap and have breathing room between them).
    """
    lo = edge_s
    hi = duration_s - snippet_s - edge_s
    if hi <= lo:
        return []

    min_sep = snippet_s + min_gap_s
    positions: list[float] = []
    for _ in range(n):
        for _ in range(100):   # max attempts per position
            pos = rng.uniform(lo, hi)
            if all(abs(pos - p) >= min_sep for p in positions):
                positions.append(pos)
                break
    return sorted(positions)


def plan_snippets(
    registry:         dict,
    audio_candidates: dict,
    cfg:              dict,
    rng:              np.random.Generator,
    exclude_set:      set[tuple[str, float]] | None = None,
) -> list[dict]:
    """
    For each bird, allocate up to cfg['snippets_per_bird'] snippet tasks.
    Snippets are distributed across candidate files from largest to smallest;
    multiple snippets may come from the same file if it is long enough.

    Parameters
    ----------
    exclude_set : set of (source_file, snippet_start_s) tuples, optional
        Snippets whose (source_file, round(start_s, 2)) key is present in
        this set are skipped.  Populated from a prescreen CSV via
        ``load_exclude_set()``.

    Returns a flat list of task dicts (one per snippet, with a fresh UUID each).
    """
    N          = cfg["snippets_per_bird"]
    dur        = cfg["snippet_duration_s"]
    gap        = cfg["min_gap_s"]
    edge       = cfg["edge_s"]
    project_dir = cfg["project_dir"]
    spec_cfg   = cfg["spectrogram"]
    target_sr  = cfg["target_sr"]

    all_tasks: list[dict] = []

    for bird, meta in registry.items():
        candidates = audio_candidates.get(bird, [])
        existing   = [c for c in candidates if Path(c["filepath"]).exists()]
        if not existing:
            print(f"  {bird}: no accessible audio — skipping")
            continue

        # Sort descending by size (proxy for duration)
        existing.sort(key=lambda c: c.get("size_mb", 0), reverse=True)

        bird_tasks: list[dict] = []
        remaining = N

        for cand in existing:
            if remaining <= 0:
                break
            fp = cand["filepath"]

            # Get actual duration; fall back to size estimate (1 MB ≈ 40 s at 32kHz 16-bit)
            file_dur = get_audio_duration(fp)
            if file_dur is None:
                file_dur = cand.get("size_mb", 0) * 40.0

            if file_dur < dur + 2 * edge:
                continue   # file too short for even one snippet

            # How many non-overlapping snippets fit?
            max_from_file = max(1, int((file_dur - dur - 2 * edge) / (dur + gap)) + 1)
            n_this        = min(remaining, max_from_file)

            positions = _sample_positions(file_dur, dur, n_this, gap, edge, rng)
            if not positions:
                continue

            for start_s in positions:
                if exclude_set and (fp, round(start_s, 2)) in exclude_set:
                    print(f"    excluded: {Path(fp).name} @ {start_s:.2f}s")
                    continue
                bird_tasks.append({
                    "uid":           str(uuid.uuid4()),
                    "source_file":   fp,
                    "start_s":       float(start_s),
                    "duration_s":    float(dur),
                    "bird_id":       bird,
                    "role":          meta["role"],
                    "nest_father":   meta["nest_father"],
                    "genetic_father": meta["genetic_father"],
                    # passed to worker subprocess
                    "project_dir":   project_dir,
                    "spec_cfg":      spec_cfg,
                    "target_sr":     target_sr,
                })

            remaining -= len(positions)

        n_planned = N - remaining
        print(f"  {bird} ({meta['role']}): {n_planned} snippet(s) planned")
        all_tasks.extend(bird_tasks)

    return all_tasks


# ── Worker (runs in subprocess) ──────────────────────────────────────────────────

def compute_snippet(task: dict) -> dict:
    """
    Extract an 8-second audio slice, resample to target_sr, compute spectrogram.

    Designed to run inside a ProcessPoolExecutor worker — all imports are local
    so the subprocess doesn't need pre-loaded state.

    Returns a dict with keys:
        uid, spec (float32 ndarray), audio (float32 ndarray),
        f (float32 ndarray), error (str or None).
    """
    import sys as _sys
    from math import gcd as _gcd
    from pathlib import Path as _Path

    # Make family_spec_generation importable from the project directory
    proj = task["project_dir"]
    if proj not in _sys.path:
        _sys.path.insert(0, proj)
    import family_spec_generation as fsg
    from scipy.signal import resample_poly as _rsp

    uid = task["uid"]

    # ── Read audio ───────────────────────────────────────────────────────────
    try:
        audio_full, sr = fsg.read_audio_file(task["source_file"])
    except Exception as e:
        return {"uid": uid, "error": f"read: {e}"}

    audio_full = audio_full.astype(np.float32)

    # ── Slice snippet ────────────────────────────────────────────────────────
    start_samp = int(task["start_s"] * sr)
    want_samp  = int(task["duration_s"] * sr)
    end_samp   = start_samp + want_samp

    if start_samp >= len(audio_full):
        # Planned start is beyond actual file length — shift to end
        end_samp   = len(audio_full)
        start_samp = max(0, end_samp - want_samp)

    snippet = audio_full[start_samp:end_samp].copy()
    del audio_full

    if len(snippet) < want_samp * 0.5:
        return {"uid": uid, "error": "snippet shorter than 50 % of requested duration"}

    # ── Resample to target SR ─────────────────────────────────────────────────
    target_sr = task["target_sr"]
    if sr != target_sr:
        g       = _gcd(int(sr), int(target_sr))
        snippet = _rsp(snippet, target_sr // g, sr // g).astype(np.float32)
        sr      = target_sr

    # ── Exact-length enforce (truncate or zero-pad) ───────────────────────────
    exact_len = int(task["duration_s"] * sr)
    if len(snippet) > exact_len:
        snippet = snippet[:exact_len]
    elif len(snippet) < exact_len:
        snippet = np.pad(snippet.astype(np.float32), (0, exact_len - len(snippet)))

    # ── Spectrogram ──────────────────────────────────────────────────────────
    scfg = task["spec_cfg"]
    try:
        spec, f, _ = fsg.make_song_spectrogram(
            snippet.astype(np.float64), sr,
            nfft=scfg["nfft"],
            hop=scfg["hop"],
            min_freq=scfg["min_freq"],
            max_freq=scfg["max_freq"],
            p_low=scfg["p_low"],
            p_high=scfg["p_high"],
            max_duration_s=task["duration_s"] + 2,   # no truncation for 8-s clips
        )
    except Exception as e:
        return {"uid": uid, "error": f"spectrogram: {e}"}

    if spec.size == 0:
        return {"uid": uid, "error": "empty spectrogram"}

    return {
        "uid":   uid,
        "spec":  spec.astype(np.float32),
        "audio": snippet,
        "f":     f.astype(np.float32),
        "error": None,
    }


# ── HDF5 schema ──────────────────────────────────────────────────────────────────

class _ManifestRow(tables.IsDescription):
    uid              = tables.StringCol(36)
    spec_idx         = tables.Int32Col()
    bird_id          = tables.StringCol(32)
    role             = tables.StringCol(16)
    nest_father      = tables.StringCol(32)
    genetic_father   = tables.StringCol(32)
    source_file      = tables.StringCol(512)
    snippet_start_s  = tables.Float32Col()
    snippet_duration_s = tables.Float32Col()


# ── HDF5 writer ──────────────────────────────────────────────────────────────────

def write_hdf5(
    h5_path:  Path,
    tasks:    list[dict],
    results:  dict,          # uid → compute_snippet output
    cfg:      dict,
    pairing:  dict,
    salt:     str,
) -> int:
    """
    Write all successful snippets to batch.h5.

    Notes
    -----
    HDF5 layout::

        /config      group   attrs: pairing, created, salt, seed, target_sr,
                             snippet_duration_s, spectrogram, freq_bins, time_bins
        /freq_axis   Array   (freq_bins,)  Hz
        /specs       EArray  (n x freq_bins x time_bins)  float32, blosc-5
        /audio       VLArray float32 rows (one per snippet)
        /manifest    Table   uid, spec_idx, bird_id, role, ...

    Returns the number of snippets written.
    """
    good = [r for r in results.values() if not r.get("error")]
    if not good:
        raise RuntimeError("No successful snippets — nothing to write.")

    ref       = good[0]
    freq_bins, time_bins = ref["spec"].shape
    n_total   = len(good)
    blosc     = tables.Filters(complevel=5, complib="blosc")

    with tables.open_file(str(h5_path), mode="w", title="scoring_batch") as h5:

        # ── Config metadata ──────────────────────────────────────────────────
        grp = h5.create_group("/", "config")
        grp._v_attrs["pairing"]            = json.dumps(pairing)
        grp._v_attrs["created"]            = datetime.now().isoformat()
        grp._v_attrs["salt"]               = salt
        grp._v_attrs["seed"]               = cfg["seed"]
        grp._v_attrs["target_sr"]          = cfg["target_sr"]
        grp._v_attrs["snippet_duration_s"] = cfg["snippet_duration_s"]
        grp._v_attrs["spectrogram"]        = json.dumps(cfg["spectrogram"])
        grp._v_attrs["freq_bins"]          = freq_bins
        grp._v_attrs["time_bins"]          = time_bins

        # ── Shared frequency axis ────────────────────────────────────────────
        h5.create_array("/", "freq_axis", ref["f"],
                        title="Frequency axis (Hz)")

        # ── Spectrogram EArray ───────────────────────────────────────────────
        specs = h5.create_earray(
            "/", "specs",
            tables.Float32Atom(),
            shape=(0, freq_bins, time_bins),
            filters=blosc,
            expectedrows=n_total,
            title="Spectrograms (n_snippets × freq_bins × time_bins)",
        )

        # ── Audio VLArray ────────────────────────────────────────────────────
        audio_vl = h5.create_vlarray(
            "/", "audio",
            tables.Float32Atom(),
            filters=blosc,
            title="Audio snippets (float32, one row per snippet)",
        )

        # ── Manifest table ───────────────────────────────────────────────────
        manifest = h5.create_table(
            "/", "manifest",
            _ManifestRow,
            filters=blosc,
            title="UUID → bird identity (decode key)",
            expectedrows=n_total,
        )

        mrow     = manifest.row
        spec_idx = 0

        # Write in task order so spec_idx stays consistent
        uid_to_task = {t["uid"]: t for t in tasks}
        for task in tasks:
            uid = task["uid"]
            res = results.get(uid)
            if res is None or res.get("error"):
                continue

            specs.append(res["spec"][np.newaxis])   # (1, freq, time) → appended
            audio_vl.append(res["audio"])

            mrow["uid"]               = uid
            mrow["spec_idx"]          = spec_idx
            mrow["bird_id"]           = task["bird_id"]
            mrow["role"]              = task["role"]
            mrow["nest_father"]       = task["nest_father"]
            mrow["genetic_father"]    = task["genetic_father"]
            mrow["source_file"]       = task["source_file"]
            mrow["snippet_start_s"]   = task["start_s"]
            mrow["snippet_duration_s"] = task["duration_s"]
            mrow.append()
            spec_idx += 1

        manifest.flush()

    return spec_idx


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an anonymized scoring batch for one pairing."
    )
    parser.add_argument("--nest-father",    required=True,
                        help="Nest-father bird ID (e.g. pk24bu3)")
    parser.add_argument("--genetic-father", required=True,
                        help="Genetic-father bird ID (e.g. wh88br85)")
    parser.add_argument("--config", default=str(DEFAULT_CFG),
                        help="Path to config.json")
    parser.add_argument("--snippets-per-bird", type=int,
                        help="Override config snippets_per_bird")
    parser.add_argument("--workers", type=int,
                        help="Override config n_workers")
    parser.add_argument("--exclude-csv", default=None,
                        help="Path to a prescreen CSV; snippets labelled "
                             "not_song or rendering_error are excluded")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.snippets_per_bird:
        cfg["snippets_per_bird"] = args.snippets_per_bird
    if args.workers:
        cfg["n_workers"] = args.workers

    nf, gf  = args.nest_father, args.genetic_father
    pairing = {"nest_father": nf, "genetic_father": gf}
    salt    = secrets.token_hex(16)

    print(f"\n=== Preparing batch: {nf} × {gf} ===")
    print(f"Snippets/bird: {cfg['snippets_per_bird']}  |  "
          f"Duration: {cfg['snippet_duration_s']}s  |  "
          f"Workers: {cfg['n_workers']}")

    # ── Output directory ─────────────────────────────────────────────────────
    date_str  = datetime.now().strftime("%Y%m%d")
    batch_dir = Path(cfg["output_dir"]) / f"{nf}_{gf}_{date_str}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "sessions").mkdir(exist_ok=True)
    h5_path = batch_dir / "batch.h5"

    if h5_path.exists():
        print(f"NOTE: {h5_path} already exists — will overwrite.")

    # ── Load bird registry ────────────────────────────────────────────────────
    print("\nLoading bird registry...")
    registry = load_bird_registry(cfg["ref_dir"], nf, gf)
    role_counts = {r: sum(1 for v in registry.values() if v["role"] == r)
                   for r in ["nest_father", "genetic_father", "xf", "hr_nest", "hr_genetic"]}
    print(f"  {len(registry)} birds: " +
          "  ".join(f"{r}={n}" for r, n in role_counts.items() if n))

    # ── Load audio candidates ─────────────────────────────────────────────────
    print("\nLoading audio candidates cache...")
    with open(cfg["audio_candidates_cache"]) as f:
        audio_candidates = json.load(f)
    found = sum(1 for b in registry if b in audio_candidates)
    print(f"  {found}/{len(registry)} birds have audio candidates")

    # ── Load exclusion list ───────────────────────────────────────────────────
    exclude_set = None
    if args.exclude_csv:
        exclude_path = Path(args.exclude_csv)
        exclude_set  = load_exclude_set(exclude_path)
        print(f"\nExclusion list: {len(exclude_set)} snippet(s) from {exclude_path.name}")

    # ── Plan snippets ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(cfg["seed"])
    print("\nPlanning snippets...")
    tasks = plan_snippets(registry, audio_candidates, cfg, rng, exclude_set)

    if not tasks:
        print("\nNo snippets could be planned — check audio paths and candidates cache.")
        return

    print(f"\n{len(tasks)} snippet tasks planned.")

    # ── Compute spectrograms (parallel) ───────────────────────────────────────
    print(f"\nComputing spectrograms ({cfg['n_workers']} workers)...")
    results: dict = {}
    n_errors = 0

    with ProcessPoolExecutor(max_workers=cfg["n_workers"]) as pool:
        futures = {pool.submit(compute_snippet, t): t["uid"] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res      = fut.result()
            uid      = res["uid"]
            results[uid] = res
            if res.get("error"):
                n_errors += 1
                print(f"  [{i:3d}/{len(tasks)}] ERR  {uid[:8]}…  {res['error']}")
            elif i % 20 == 0 or i == len(tasks):
                print(f"  [{i:3d}/{len(tasks)}] ok")

    n_ok = len(tasks) - n_errors
    print(f"\n{n_ok} succeeded, {n_errors} failed.")

    if n_ok == 0:
        print("Nothing to write — exiting.")
        return

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    print(f"\nWriting {h5_path} ...")
    n_written = write_hdf5(h5_path, tasks, results, cfg, pairing, salt)

    # ── Public batch index (no bird identity) ─────────────────────────────────
    index = {
        "pairing":    pairing,
        "created":    datetime.now().isoformat(),
        "n_snippets": n_written,
        "batch_dir":  str(batch_dir),
        "h5_file":    str(h5_path),
    }
    (batch_dir / "batch_index.json").write_text(json.dumps(index, indent=2))

    print(f"\nDone. {n_written} snippets written to:")
    print(f"  {h5_path}")
    print(f"\nNext: python export_batch.py {batch_dir}")


if __name__ == "__main__":
    main()
