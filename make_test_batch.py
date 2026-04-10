#!/usr/bin/env python3
"""
make_test_batch.py — Generate a synthetic batch.h5 for pipeline testing.

Creates a minimal but structurally complete batch using random spectrograms
and audio — no real audio files or bird registry CSVs needed.

Default: 4 birds (nest_father, genetic_father, 2 xf), 3 snippets each = 12 rows.

Usage
-----
    python make_test_batch.py
    python make_test_batch.py --out-dir E:/scoring/batches/test_batch
    python make_test_batch.py --birds 6 --snippets 4

Then test each downstream script:
    python export_batch.py     E:/scoring/batches/test_batch
    python ranking_app.py      E:/scoring/batches/test_batch
    python analyze_rankings.py E:/scoring/batches/test_batch
    python printout_generator.py E:/scoring/batches/test_batch
"""

from __future__ import annotations

import argparse
import json
import secrets
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import tables

# ── Defaults ─────────────────────────────────────────────────────────────────────

DEFAULT_OUT  = Path("E:/scoring/batches/test_batch")
NF_ID        = "testbird_nf"
GF_ID        = "testbird_gf"
FREQ_BINS    = 128     # matches nfft=1024, sr=32000, min_freq=400, max_freq=10000
TIME_BINS    = 100     # ~8 s at hop=256, sr=32000
AUDIO_LEN    = 256000  # 8 s × 32000 Hz
TARGET_SR    = 32000
SNIPPET_S    = 8.0


# ── HDF5 schema (mirrors prepare_batch.py) ───────────────────────────────────────

class _ManifestRow(tables.IsDescription):
    uid               = tables.StringCol(36)
    spec_idx          = tables.Int32Col()
    bird_id           = tables.StringCol(32)
    role              = tables.StringCol(16)
    nest_father       = tables.StringCol(32)
    genetic_father    = tables.StringCol(32)
    source_file       = tables.StringCol(512)
    snippet_start_s   = tables.Float32Col()
    snippet_duration_s = tables.Float32Col()


# ── Synthetic data generators ────────────────────────────────────────────────────

def synthetic_spec(rng: np.random.Generator, role: str) -> np.ndarray:
    """
    Generate a plausible-looking (freq_bins × time_bins) float32 spectrogram.

    Each role gets a slightly different intensity profile so that the Elo
    analysis produces non-trivial spread (nest_father > genetic_father > xf).
    """
    base = rng.standard_normal((FREQ_BINS, TIME_BINS)).astype(np.float32)

    # Low-frequency noise floor
    base[:FREQ_BINS // 4, :] -= 2.0

    # Role-specific amplitude offset — creates a detectable signal difference
    offsets = {
        "nest_father":    2.0,
        "genetic_father": 1.0,
        "xf":             0.0,
        "hr_nest":        0.5,
        "hr_genetic":    -0.5,
    }
    base += offsets.get(role, 0.0)
    return base


def synthetic_audio(rng: np.random.Generator) -> np.ndarray:
    """Generate a short burst of band-limited noise as a stand-in for song audio."""
    audio = rng.standard_normal(AUDIO_LEN).astype(np.float32) * 0.05
    # Add a tone sweep to make spectrograms visually interesting
    t = np.linspace(0, SNIPPET_S, AUDIO_LEN, endpoint=False, dtype=np.float32)
    freq = 2000 + 3000 * t / SNIPPET_S   # sweep 2–5 kHz
    audio += 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio


def synthetic_freq_axis() -> np.ndarray:
    """Return a Hz axis consistent with the default spectrogram params."""
    return np.linspace(400, 10000, FREQ_BINS, dtype=np.float32)


# ── Batch assembly ────────────────────────────────────────────────────────────────

def make_bird_list(n_birds: int) -> list[tuple[str, str]]:
    """
    Return [(bird_id, role), ...].
    Always includes nest_father and genetic_father; extras are xf.
    """
    birds = [
        (NF_ID, "nest_father"),
        (GF_ID, "genetic_father"),
    ]
    for i in range(n_birds - 2):
        birds.append((f"xf_bird_{i+1:02d}", "xf"))
    return birds


def write_batch(
    out_dir:        Path,
    birds:          list[tuple[str, str]],
    snippets_each:  int,
    seed:           int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sessions").mkdir(exist_ok=True)
    h5_path = out_dir / "batch.h5"

    rng  = np.random.default_rng(seed)
    salt = secrets.token_hex(16)

    pairing = {"nest_father": NF_ID, "genetic_father": GF_ID}
    cfg_spectrogram = {
        "nfft": 1024, "hop": 256,
        "min_freq": 400, "max_freq": 10000,
        "p_low": 2, "p_high": 98,
    }

    # Pre-generate all specs and audio so we know the shapes before opening HDF5
    tasks = []
    for bird_id, role in birds:
        for s in range(snippets_each):
            tasks.append({
                "uid":      str(uuid.uuid4()),
                "bird_id":  bird_id,
                "role":     role,
                "start_s":  float(s * (SNIPPET_S + 5.0)),
                "spec":     synthetic_spec(rng, role),
                "audio":    synthetic_audio(rng),
            })

    blosc = tables.Filters(complevel=5, complib="blosc")
    freq_axis = synthetic_freq_axis()

    with tables.open_file(str(h5_path), mode="w", title="test_scoring_batch") as h5:

        # Config group
        grp = h5.create_group("/", "config")
        grp._v_attrs["pairing"]            = json.dumps(pairing)
        grp._v_attrs["created"]            = datetime.now().isoformat()
        grp._v_attrs["salt"]               = salt
        grp._v_attrs["seed"]               = seed
        grp._v_attrs["target_sr"]          = TARGET_SR
        grp._v_attrs["snippet_duration_s"] = SNIPPET_S
        grp._v_attrs["spectrogram"]        = json.dumps(cfg_spectrogram)
        grp._v_attrs["freq_bins"]          = FREQ_BINS
        grp._v_attrs["time_bins"]          = TIME_BINS

        # Frequency axis
        h5.create_array("/", "freq_axis", freq_axis, title="Frequency axis (Hz)")

        # Spectrogram EArray
        specs = h5.create_earray(
            "/", "specs",
            tables.Float32Atom(),
            shape=(0, FREQ_BINS, TIME_BINS),
            filters=blosc,
            expectedrows=len(tasks),
            title="Spectrograms (n_snippets × freq_bins × time_bins)",
        )

        # Audio VLArray
        audio_vl = h5.create_vlarray(
            "/", "audio",
            tables.Float32Atom(),
            filters=blosc,
            title="Audio snippets (float32)",
        )

        # Manifest table
        manifest = h5.create_table(
            "/", "manifest",
            _ManifestRow,
            filters=blosc,
            title="UUID → bird identity",
            expectedrows=len(tasks),
        )

        mrow = manifest.row
        for idx, t in enumerate(tasks):
            specs.append(t["spec"][np.newaxis])
            audio_vl.append(t["audio"])

            mrow["uid"]               = t["uid"]
            mrow["spec_idx"]          = idx
            mrow["bird_id"]           = t["bird_id"]
            mrow["role"]              = t["role"]
            mrow["nest_father"]       = NF_ID
            mrow["genetic_father"]    = GF_ID
            mrow["source_file"]       = f"synthetic/{t['bird_id']}_snippet{idx}.wav"
            mrow["snippet_start_s"]   = t["start_s"]
            mrow["snippet_duration_s"] = SNIPPET_S
            mrow.append()

        manifest.flush()

    print(f"  {len(tasks)} snippets written ({len(birds)} birds × {snippets_each} each)")
    return h5_path


# ── Fake session writer ───────────────────────────────────────────────────────────

def write_fake_sessions(
    out_dir:  Path,
    h5_path:  Path,
    n_sessions: int = 2,
    rounds_each: int = 3,
    batch_size: int = 4,
) -> None:
    """
    Write synthetic session JSON files so analyze_rankings.py has something to read.
    Each session simulates a different scorer making slightly different rankings.
    """
    # Read UIDs from the HDF5
    uid_list: list[str] = []
    with tables.open_file(str(h5_path), mode="r") as h5:
        for row in h5.root.manifest.iterrows():
            uid_list.append(row["uid"].decode())

    sessions_dir = out_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(999)

    for sess_i in range(n_sessions):
        scorer = f"TestScorer{sess_i + 1}"
        trait  = "stereotypy"
        started = datetime.now().isoformat()

        # Shuffle UIDs to simulate pool
        pool = uid_list.copy()
        rng.shuffle(pool)

        rounds = []
        for r in range(rounds_each):
            batch_start = r * batch_size
            presented   = pool[batch_start: batch_start + batch_size]
            if not presented:
                break

            # Shuffle the presented batch to simulate a ranking
            ranking = presented.copy()
            rng.shuffle(ranking)

            rounds.append({
                "round":     r + 1,
                "presented": presented,
                "ranking":   ranking,
                "elapsed_s": float(rng.uniform(15, 90)),
                "timestamp": datetime.now().isoformat(),
            })

        session = {
            "sid":       str(uuid.uuid4()),
            "scorer":    scorer,
            "trait":     trait,
            "platform":  "local",
            "batch_id":  out_dir.name,
            "pairing":   {"nest_father": NF_ID, "genetic_father": GF_ID},
            "started":   started,
            "pool_pos":  len(pool),
            "batch_size": batch_size,
            "rounds":    rounds,
        }

        fname = f"{scorer}_{trait}_{started[:10]}.json"
        (sessions_dir / fname).write_text(json.dumps(session, indent=2))
        print(f"  Wrote {fname}  ({len(rounds)} rounds)")


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic batch.h5 for pipeline testing."
    )
    parser.add_argument("--out-dir",   default=str(DEFAULT_OUT),
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--birds",     type=int, default=4,
                        help="Total number of birds (min 2; extras are xf role)")
    parser.add_argument("--snippets",  type=int, default=3,
                        help="Snippets per bird")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-sessions", action="store_true",
                        help="Skip writing fake session JSONs")
    args = parser.parse_args()

    if args.birds < 2:
        parser.error("--birds must be >= 2 (need at least nest_father + genetic_father)")

    out_dir = Path(args.out_dir)
    birds   = make_bird_list(args.birds)

    print(f"\n=== Synthetic test batch ===")
    print(f"Output:   {out_dir}")
    print(f"Birds:    {args.birds}  ({', '.join(r for _, r in birds)})")
    print(f"Snippets: {args.snippets} per bird = {args.birds * args.snippets} total")

    print("\nWriting batch.h5 ...")
    h5_path = write_batch(out_dir, birds, args.snippets, args.seed)
    print(f"  → {h5_path}")

    if not args.no_sessions:
        print("\nWriting fake session JSONs ...")
        write_fake_sessions(out_dir, h5_path)

    print(f"""
Done.  Run the pipeline in order:

  1. Export spectrograms + audio:
       python E:/scoring/export_batch.py {out_dir}

  2. Open the ranking app:
       python E:/scoring/ranking_app.py {out_dir}
       # browse to http://localhost:5000/

  3. Analyze rankings (uses fake sessions already written):
       python E:/scoring/analyze_rankings.py {out_dir}

  4. Generate printout HTML:
       python E:/scoring/printout_generator.py {out_dir}
""")


if __name__ == "__main__":
    main()
