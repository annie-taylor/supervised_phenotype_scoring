#!/usr/bin/env python3
"""
prescreen_app.py — Pre-screening app for inspecting batch spectrograms.

Renders each spectrogram from batch.h5 as an interactive Plotly heatmap
(magma colorscale, full frequency resolution) and lets a reviewer label each
snippet as song, not-song, or rendering-error using on-screen buttons or
keyboard shortcuts.

Labels are written to ``<batch_dir>/prescreen_<YYYYMMDD>.csv`` after each
decision. If a prescreen CSV already exists for the batch, already-labelled
snippets are skipped so the session can be resumed at any time.

Usage
-----
    python prescreen_app.py <batch_dir>
    python prescreen_app.py <batch_dir> --port 5001

Output CSV columns
------------------
uid, bird_id, role, source_file, snippet_start_s, label

Labels
------
song            Snippet contains a full song — suitable for scoring.
not_song        Snippet contains noise, calls, or non-song vocalisation.
rendering_error Plotly heatmap shows a display / normalisation artefact.

Snippets labelled ``not_song`` or ``rendering_error`` can be excluded from
future batches and used to diagnose gaps in the upstream song filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import tables
from flask import Flask, jsonify, redirect, render_template, request, url_for

SCORING_DIR = Path(__file__).resolve().parent

# Maximum number of time columns sent to the browser per spectrogram.
# hop=1 produces ~255 000 columns for an 8-second clip; downsampling to
# MAX_TIME_COLS keeps the JSON payload under ~5 MB while preserving the
# full frequency resolution and any normalisation artefacts.
MAX_TIME_COLS = 2000


# ── Batch loading ─────────────────────────────────────────────────────────────

def load_batch(batch_dir: Path) -> dict:
    """Load manifest and config from batch.h5."""
    h5_path = batch_dir / "batch.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"batch.h5 not found in {batch_dir}")

    uid_meta = {}
    with tables.open_file(str(h5_path), mode="r") as h5:
        cfg_attrs  = h5.root.config._v_attrs
        spec_cfg   = json.loads(cfg_attrs["spectrogram"])
        target_sr  = float(cfg_attrs["target_sr"])
        freq_axis  = h5.root.freq_axis[:]

        for row in h5.root.manifest.iterrows():
            uid = row["uid"].decode()
            uid_meta[uid] = {
                "bird_id":         row["bird_id"].decode(),
                "role":            row["role"].decode(),
                "source_file":     row["source_file"].decode(),
                "snippet_start_s": float(row["snippet_start_s"]),
                "spec_idx":        int(row["spec_idx"]),
            }

    return {
        "h5_path":   h5_path,
        "uid_meta":  uid_meta,
        "spec_cfg":  spec_cfg,
        "target_sr": target_sr,
        "freq_axis": freq_axis,
        "uids":      list(uid_meta.keys()),
    }


# ── Prescreen CSV ─────────────────────────────────────────────────────────────

def prescreen_csv_path(batch_dir: Path) -> Path:
    return batch_dir / f"prescreen_{date.today().strftime('%Y%m%d')}.csv"


def load_existing_labels(csv_path: Path) -> dict[str, str]:
    """Return {uid: label} for any already-labelled snippets."""
    if not csv_path.exists():
        return {}
    labels = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["uid"]] = row["label"]
    return labels


def write_labels_csv(csv_path: Path, labels: dict[str, str], uid_meta: dict) -> None:
    """Rewrite the entire prescreen CSV from the current labels dict."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uid", "bird_id", "role",
                        "source_file", "snippet_start_s", "label"],
        )
        writer.writeheader()
        for uid, label in labels.items():
            meta = uid_meta[uid]
            writer.writerow({
                "uid":             uid,
                "bird_id":         meta["bird_id"],
                "role":            meta["role"],
                "source_file":     meta["source_file"],
                "snippet_start_s": meta["snippet_start_s"],
                "label":           label,
            })


# ── Spectrogram data route ────────────────────────────────────────────────────

def read_spec_from_h5(batch: dict, uid: str) -> dict:
    """
    Read a spectrogram from batch.h5 and downsample the time axis to at most
    MAX_TIME_COLS columns.  Returns {z, x, y} for Plotly.
    """
    meta     = batch["uid_meta"][uid]
    hop      = batch["spec_cfg"]["hop"]
    sr       = batch["target_sr"]
    freq     = batch["freq_axis"]

    with tables.open_file(str(batch["h5_path"]), mode="r") as h5:
        spec = h5.root.specs[meta["spec_idx"]]   # (freq_bins, time_bins)

    n_time  = spec.shape[1]
    stride  = max(1, n_time // MAX_TIME_COLS)
    spec_ds = spec[:, ::stride]
    t       = (np.arange(0, n_time, stride) * hop / sr).tolist()

    return {
        "z": np.round(spec_ds, 4).tolist(),
        "x": np.round(t, 4).tolist(),
        "y": np.round(freq, 1).tolist(),
    }


# ── Flask app ─────────────────────────────────────────────────────────────────

def create_app(batch_dir: Path) -> Flask:
    app = Flask(__name__, template_folder=str(SCORING_DIR / "templates"),
                static_folder=str(SCORING_DIR / "static"))
    app.secret_key = os.urandom(24)

    batch    = load_batch(batch_dir)
    csv_path = prescreen_csv_path(batch_dir)

    all_uids = batch["uids"]
    existing = load_existing_labels(csv_path)

    # Start cursor at the first unlabeled snippet
    first_unlabeled = next(
        (i for i, u in enumerate(all_uids) if u not in existing),
        len(all_uids) - 1,
    )

    # Mutable state — single-user app, no locking needed
    state = {
        "all_uids": all_uids,
        "cursor":   first_unlabeled,
        "labels":   dict(existing),       # {uid: label} — updated on every submission
        "total":    len(all_uids),
    }

    @app.route("/")
    def index():
        cursor = state["cursor"]
        uid    = state["all_uids"][cursor]
        meta   = batch["uid_meta"][uid]
        return render_template(
            "prescreen.html",
            uid=uid,
            uid_short=uid[:8] + "…",
            done=len(state["labels"]),
            total=state["total"],
            source_file=Path(meta["source_file"]).name,
            snippet_start_s=round(meta["snippet_start_s"], 2),
            role=meta["role"],
            cursor=cursor,
            has_prev=cursor > 0,
            has_next=cursor < state["total"] - 1,
            current_label=state["labels"].get(uid),
        )

    @app.route("/goto")
    def goto():
        try:
            idx = int(request.args.get("idx", state["cursor"]))
        except ValueError:
            idx = state["cursor"]
        state["cursor"] = max(0, min(idx, state["total"] - 1))
        return redirect(url_for("index"))

    @app.route("/spec_data/<uid>")
    def spec_data(uid: str):
        if uid not in batch["uid_meta"]:
            return "Not found", 404
        try:
            data = read_spec_from_h5(batch, uid)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify(data)

    @app.route("/label", methods=["POST"])
    def label():
        data  = request.get_json()
        uid   = data.get("uid", "")
        lbl   = data.get("label", "")
        valid = {"song", "not_song", "rendering_error"}

        if uid not in batch["uid_meta"] or lbl not in valid:
            return jsonify({"error": "invalid"}), 400

        state["labels"][uid] = lbl
        write_labels_csv(csv_path, state["labels"], batch["uid_meta"])

        # Advance cursor to next snippet (or stay at end)
        state["cursor"] = min(state["cursor"] + 1, state["total"] - 1)

        remaining = sum(1 for u in state["all_uids"] if u not in state["labels"])
        return jsonify({"ok": True, "remaining": remaining})

    @app.route("/done")
    def done():
        return render_template(
            "prescreen_done.html",
            total=state["total"],
            csv_path=str(csv_path),
        )

    return app


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-screen batch spectrograms before scoring."
    )
    parser.add_argument("batch_dir", help="Path to batch directory")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to serve on (default 5001)")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    app       = create_app(batch_dir)

    print(f"\n=== Prescreen App ===")
    print(f"Batch:  {batch_dir.name}")
    print(f"Open in browser: http://localhost:{args.port}/\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
