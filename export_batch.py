#!/usr/bin/env python3
"""
export_batch.py — Generate PNGs and WAVs from a scoring batch HDF5.

Reads  batch.h5  and writes:
  export/spectrograms/<uuid>.png   — spectrogram images (for GUI + printout)
  export/audio/<uuid>.wav          — 8-second audio clips (for GUI playback)
  export/manifest_public.json      — list of UUIDs with no bird-identity info

Re-running with different visual parameters regenerates only the PNGs without
touching the HDF5 or audio files (use --force to overwrite existing PNGs).

Usage
-----
    python export_batch.py E:/scoring/batches/pk24bu3_wh88br85_20260410
    python export_batch.py <batch_dir> --dpi 150 --figsize 8 2.5 --cmap inferno
    python export_batch.py <batch_dir> --force --workers 6
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tables
from scipy.io import wavfile


# ── PNG worker (runs in subprocess) ─────────────────────────────────────────────

def _write_png(args: tuple) -> tuple[str, str | None]:
    """
    Worker: open HDF5 (read-only), read one spectrogram, save PNG.
    Returns (uid, error_string_or_None).
    """
    h5_path, spec_idx, uid, out_path, params = args

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tables as _tables
        import numpy as _np

        with _tables.open_file(h5_path, mode="r") as h5:
            spec = h5.root.specs[spec_idx]          # (freq_bins, time_bins)
            f    = h5.root.freq_axis[:]              # Hz
            cfg  = json.loads(h5.root.config._v_attrs["spectrogram"])
            dur  = float(h5.root.config._v_attrs["snippet_duration_s"])

        fig, ax = plt.subplots(figsize=params["figsize"])
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            cmap=params["cmap"],
            vmin=0, vmax=1,
            extent=[0, dur, f[0] / 1000, f[-1] / 1000],
            interpolation="nearest",
        )
        ax.set_xlabel("Time (s)", fontsize=8, color="#aaa")
        ax.set_ylabel("kHz",      fontsize=8, color="#aaa")
        ax.tick_params(labelsize=7, colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        fig.patch.set_facecolor("#111")
        ax.set_facecolor("#111")
        plt.tight_layout(pad=0.4)
        fig.savefig(out_path, dpi=params["dpi"],
                    bbox_inches="tight", facecolor="#111")
        plt.close(fig)
        return uid, None

    except Exception as e:
        return uid, str(e)


# ── WAV writer ────────────────────────────────────────────────────────────────────

def export_audio(h5_path: Path, audio_dir: Path, force: bool) -> int:
    """Write WAV files for all snippets. Returns count written."""
    written = 0
    with tables.open_file(str(h5_path), mode="r") as h5:
        target_sr = int(h5.root.config._v_attrs["target_sr"])
        for row in h5.root.manifest.iterrows():
            uid = row["uid"].decode()
            out = audio_dir / f"{uid}.wav"
            if out.exists() and not force:
                continue
            audio = h5.root.audio[int(row["spec_idx"])]
            # float32 → int16 for standard WAV
            audio_i16 = (audio * 32767).clip(-32767, 32767).astype(np.int16)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wavfile.write(str(out), target_sr, audio_i16)
            written += 1
    return written


# ── Public manifest (no bird identity) ───────────────────────────────────────────

def write_public_manifest(h5_path: Path, export_dir: Path) -> list[str]:
    """Write export/manifest_public.json — UIDs only, safe to share with scorers."""
    uids = []
    with tables.open_file(str(h5_path), mode="r") as h5:
        for row in h5.root.manifest.iterrows():
            uids.append(row["uid"].decode())
    out = export_dir / "manifest_public.json"
    out.write_text(json.dumps({"uids": uids, "n": len(uids)}, indent=2))
    return uids


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PNGs and WAVs from a scoring batch HDF5."
    )
    parser.add_argument("batch_dir", help="Path to batch directory containing batch.h5")
    parser.add_argument("--dpi",     type=int,   default=120,
                        help="PNG resolution in DPI (default 120)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[7.0, 2.0],
                        metavar=("W", "H"),
                        help="Figure size in inches, width height (default 7 2)")
    parser.add_argument("--cmap",    default="magma",
                        help="Matplotlib colormap (default magma)")
    parser.add_argument("--force",   action="store_true",
                        help="Overwrite existing PNG/WAV files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for PNG generation (default 4)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip WAV export (PNGs only)")
    args = parser.parse_args()

    batch_dir  = Path(args.batch_dir).resolve()
    h5_path    = batch_dir / "batch.h5"
    export_dir = batch_dir / "export"
    spec_dir   = export_dir / "spectrograms"
    audio_dir  = export_dir / "audio"

    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found.")
        return

    spec_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "figsize": tuple(args.figsize),
        "dpi":     args.dpi,
        "cmap":    args.cmap,
    }

    # ── Build PNG task list ────────────────────────────────────────────────────
    print(f"\nReading manifest from {h5_path} ...")
    tasks = []
    with tables.open_file(str(h5_path), mode="r") as h5:
        n_total = h5.root.manifest.nrows
        for row in h5.root.manifest.iterrows():
            uid      = row["uid"].decode()
            spec_idx = int(row["spec_idx"])
            out_path = spec_dir / f"{uid}.png"
            if out_path.exists() and not args.force:
                continue
            tasks.append((str(h5_path), spec_idx, uid, str(out_path), params))

    n_skip = n_total - len(tasks)
    print(f"  {n_total} snippets total | {n_skip} PNGs already exist | "
          f"{len(tasks)} to generate")

    # ── Generate PNGs (parallel) ───────────────────────────────────────────────
    if tasks:
        print(f"\nGenerating {len(tasks)} PNGs ({args.workers} workers) ...")
        n_errors = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_write_png, t): t[2] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                uid, err = fut.result()
                if err:
                    n_errors += 1
                    print(f"  [{i:4d}/{len(tasks)}] ERR {uid[:8]}…  {err}")
                elif i % 50 == 0 or i == len(tasks):
                    print(f"  [{i:4d}/{len(tasks)}] ok")
        print(f"  Done — {len(tasks) - n_errors} written, {n_errors} errors.")
    else:
        print("  All PNGs up to date.")

    # ── Export audio ───────────────────────────────────────────────────────────
    if not args.no_audio:
        print(f"\nExporting audio to {audio_dir} ...")
        n_wav = export_audio(h5_path, audio_dir, args.force)
        print(f"  {n_wav} WAV files written.")
    else:
        print("\nSkipping audio export (--no-audio).")

    # ── Public manifest ────────────────────────────────────────────────────────
    uids = write_public_manifest(h5_path, export_dir)
    print(f"\nPublic manifest written: {len(uids)} UUIDs.")
    print(f"\nExport complete: {export_dir}")
    print("Next: python ranking_app.py " + str(batch_dir))


if __name__ == "__main__":
    main()
