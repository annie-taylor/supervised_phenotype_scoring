#!/usr/bin/env python3
"""
ranking_app.py — Local web app for anonymized song ranking sessions.

Loads a prepared batch directory, serves spectrogram images and audio clips,
and lets scorers drag-and-drop songs into ranked order.  Rankings are saved
as JSON session files in  <batch_dir>/sessions/.

The session format is identical to the MTurk-compatible format so sessions
collected locally and via MTurk can be pooled in analyze_rankings.py.

Usage
-----
    python ranking_app.py E:/scoring/batches/pk24bu3_wh88br85_20260410
    python ranking_app.py <batch_dir> --port 5001 --batch-size 7

Dependencies
------------
    pip install flask
    Sortable.js: place Sortable.min.js in E:/scoring/static/
    (auto-downloaded on first run if internet is available)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import tables
from flask import (Flask, jsonify, redirect, render_template,
                   request, send_file, session, url_for)

# ── Constants ────────────────────────────────────────────────────────────────────

SCORING_DIR  = Path(__file__).resolve().parent
SORTABLE_URL = "https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"
SORTABLE_JS  = SCORING_DIR / "static" / "Sortable.min.js"

TRAITS = ["stereotypy", "repeat_propensity"]

ROLE_ORDER = ["nest_father", "genetic_father", "xf", "hr_nest", "hr_genetic"]

# ── Sortable.js auto-download ────────────────────────────────────────────────────

def ensure_sortable() -> None:
    if SORTABLE_JS.exists():
        return
    print(f"Downloading Sortable.js from {SORTABLE_URL} ...")
    try:
        urllib.request.urlretrieve(SORTABLE_URL, SORTABLE_JS)
        print("  OK")
    except Exception as e:
        print(f"  WARNING: could not download Sortable.js: {e}")
        print(f"  Download manually to {SORTABLE_JS}")


# ── Batch loading ────────────────────────────────────────────────────────────────

def load_batch(batch_dir: Path) -> dict:
    """
    Load batch metadata from batch.h5.  Returns a dict with:
        uid_meta   : {uid: {role, bird_id, spec_idx}}  (no scorer-visible identity)
        pairing    : {nest_father, genetic_father}
        export_dir : Path
        h5_path    : Path
        sessions_dir: Path
    """
    h5_path    = batch_dir / "batch.h5"
    export_dir = batch_dir / "export"
    sessions_dir = batch_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    if not h5_path.exists():
        raise FileNotFoundError(f"batch.h5 not found in {batch_dir}")
    if not (export_dir / "spectrograms").exists():
        raise FileNotFoundError(
            f"Export directory not found.  Run export_batch.py {batch_dir} first."
        )

    uid_meta = {}
    with tables.open_file(str(h5_path), mode="r") as h5:
        pairing = json.loads(h5.root.config._v_attrs["pairing"])
        for row in h5.root.manifest.iterrows():
            uid = row["uid"].decode()
            uid_meta[uid] = {
                "role":     row["role"].decode(),
                "bird_id":  row["bird_id"].decode(),   # kept server-side only
                "spec_idx": int(row["spec_idx"]),
            }

    return {
        "uid_meta":    uid_meta,
        "pairing":     pairing,
        "h5_path":     h5_path,
        "export_dir":  export_dir,
        "sessions_dir": sessions_dir,
    }


# ── Session pool builder ─────────────────────────────────────────────────────────

def build_session_pool(uid_meta: dict, batch_size: int, seed: int) -> list[str]:
    """
    Create a shuffled pool of UIDs arranged so that consecutive windows of
    batch_size contain songs from varied roles.

    Strategy: interleave by role (round-robin), which guarantees that any
    consecutive window of len(ROLE_ORDER) items spans all present roles.
    """
    rng    = np.random.default_rng(seed)
    by_role: dict[str, list] = {r: [] for r in ROLE_ORDER}
    for uid, meta in uid_meta.items():
        role = meta["role"]
        if role in by_role:
            by_role[role].append(uid)
        else:
            by_role.setdefault(role, []).append(uid)

    # Shuffle within each role
    for uids in by_role.values():
        rng.shuffle(uids)

    # Interleave: pull one from each non-empty role in turn
    pool: list[str] = []
    roles = [uids for uids in by_role.values() if uids]
    i = 0
    while any(roles):
        lst = roles[i % len(roles)]
        if lst:
            pool.append(lst.pop(0))
        i += 1

    return pool


# ── In-memory session store ───────────────────────────────────────────────────────
# Keyed by Flask session["sid"].  Lost on app restart (acceptable for local use).

_sessions: dict[str, dict] = {}
_lock = threading.Lock()


def new_scoring_session(scorer: str, trait: str, batch: dict,
                        batch_size: int) -> str:
    """Create a new scoring session, return its sid."""
    sid  = str(uuid.uuid4())
    seed = int(time.time() * 1000) % (2**31)
    pool = build_session_pool(batch["uid_meta"], batch_size, seed)

    state = {
        "sid":        sid,
        "scorer":     scorer,
        "trait":      trait,
        "platform":   "local",
        "batch_id":   batch["h5_path"].parent.name,
        "pairing":    batch["pairing"],
        "started":    datetime.now().isoformat(),
        "pool":       pool,
        "pool_pos":   0,
        "batch_size": batch_size,
        "rounds":     [],
    }
    with _lock:
        _sessions[sid] = state
    return sid


def get_session(sid: str) -> dict | None:
    with _lock:
        return _sessions.get(sid)


def save_session_to_disk(state: dict, sessions_dir: Path) -> None:
    """Append-write session JSON after each round."""
    fname = (f"{state['scorer']}_{state['trait']}_"
             f"{state['started'][:10]}.json")
    # Replace chars invalid in filenames
    fname = fname.replace(":", "-").replace(" ", "_")
    out   = sessions_dir / fname
    with open(out, "w") as f:
        # Write only the serialisable subset (no pool — too large)
        payload = {k: v for k, v in state.items() if k != "pool"}
        json.dump(payload, f, indent=2)


def next_batch(state: dict) -> list[str] | None:
    """
    Pull the next batch_size UIDs from the session pool.
    Returns None when pool is exhausted.
    """
    pos  = state["pool_pos"]
    size = state["batch_size"]
    pool = state["pool"]
    if pos >= len(pool):
        return None
    batch = pool[pos: pos + size]
    state["pool_pos"] = pos + size
    return batch if batch else None


# ── Flask app factory ─────────────────────────────────────────────────────────────

def create_app(batch_dir: Path, batch_size: int, cfg_mode: str) -> Flask:
    app = Flask(__name__, template_folder=str(SCORING_DIR / "templates"),
                static_folder=str(SCORING_DIR / "static"))
    app.secret_key = os.urandom(24)

    batch = load_batch(batch_dir)
    app.config["BATCH"]      = batch
    app.config["BATCH_SIZE"] = batch_size
    app.config["MODE"]       = cfg_mode

    nf = batch["pairing"]["nest_father"]
    gf = batch["pairing"]["genetic_father"]

    # ── Landing page ─────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html",
                               pairing=f"{nf} × {gf}",
                               traits=TRAITS)

    # ── Start session ─────────────────────────────────────────────────────────
    @app.route("/start", methods=["POST"])
    def start():
        scorer = request.form.get("scorer", "").strip()
        trait  = request.form.get("trait", "").strip()
        if not scorer:
            return redirect(url_for("index"))
        if trait not in TRAITS:
            trait = TRAITS[0]

        sid = new_scoring_session(scorer, trait, batch, batch_size)
        session["sid"] = sid

        # Pre-load first batch
        state = get_session(sid)
        state["current_batch"] = next_batch(state)
        return redirect(url_for("rank"))

    # ── Ranking interface ─────────────────────────────────────────────────────
    @app.route("/rank")
    def rank():
        sid   = session.get("sid")
        state = get_session(sid)
        if not state:
            return redirect(url_for("index"))

        current = state.get("current_batch")
        if not current:
            return redirect(url_for("done"))

        total    = len(state["pool"])
        done_so_far = state["pool_pos"] - len(current)
        round_num   = len(state["rounds"]) + 1

        # Build display data — only short UID prefix shown to scorer
        songs = [{"uid": uid, "uid_short": uid[:8] + "…"}
                 for uid in current]

        return render_template(
            "rank.html",
            songs=songs,
            trait=state["trait"],
            scorer=state["scorer"],
            round_num=round_num,
            completed=done_so_far,
            total=total,
            pairing=f"{nf} × {gf}",
        )

    # ── Submit ranking ────────────────────────────────────────────────────────
    @app.route("/submit", methods=["POST"])
    def submit():
        sid   = session.get("sid")
        state = get_session(sid)
        if not state:
            return jsonify({"error": "session not found"}), 400

        data     = request.get_json()
        ranking  = data.get("ranking", [])
        elapsed  = data.get("elapsed_s", None)
        presented = state.get("current_batch", [])

        round_record = {
            "round":     len(state["rounds"]) + 1,
            "presented": presented,
            "ranking":   ranking,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        state["rounds"].append(round_record)

        # Save to disk after every round
        save_session_to_disk(state, batch["sessions_dir"])

        # Advance to next batch
        state["current_batch"] = next_batch(state)

        if state["current_batch"] is None:
            return jsonify({"done": True})
        return jsonify({"done": False})

    # ── Skip current batch ────────────────────────────────────────────────────
    @app.route("/skip")
    def skip():
        sid   = session.get("sid")
        state = get_session(sid)
        if state:
            state["current_batch"] = next_batch(state)
        return redirect(url_for("rank"))

    # ── Done page ─────────────────────────────────────────────────────────────
    @app.route("/done")
    def done():
        sid   = session.get("sid")
        state = get_session(sid)
        n_rounds = len(state["rounds"]) if state else 0
        n_songs  = n_rounds * batch_size
        scorer   = state["scorer"] if state else "Unknown"
        trait    = state["trait"]  if state else ""
        return render_template("done.html",
                               scorer=scorer, trait=trait,
                               n_rounds=n_rounds, n_songs=n_songs,
                               pairing=f"{nf} × {gf}")

    # ── Asset routes ──────────────────────────────────────────────────────────
    @app.route("/spec/<uid>")
    def serve_spec(uid: str):
        # Validate UID is in the batch (prevents path traversal)
        if uid not in batch["uid_meta"]:
            return "Not found", 404
        path = batch["export_dir"] / "spectrograms" / f"{uid}.png"
        if not path.exists():
            return "PNG not exported yet — run export_batch.py", 404
        return send_file(str(path), mimetype="image/png")

    @app.route("/audio/<uid>")
    def serve_audio(uid: str):
        if uid not in batch["uid_meta"]:
            return "Not found", 404
        path = batch["export_dir"] / "audio" / f"{uid}.wav"
        if not path.exists():
            return "WAV not exported yet — run export_batch.py", 404
        return send_file(str(path), mimetype="audio/wav")

    return app


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local web app for song ranking sessions."
    )
    parser.add_argument("batch_dir", help="Path to batch directory")
    parser.add_argument("--port",       type=int,   default=5000)
    parser.add_argument("--batch-size", type=int,   default=7,
                        help="Songs per ranking round (default 7)")
    parser.add_argument("--mode",       default="local",
                        choices=["local", "hosted"])
    args = parser.parse_args()

    ensure_sortable()

    batch_dir = Path(args.batch_dir).resolve()
    app       = create_app(batch_dir, args.batch_size, args.mode)

    print(f"\n=== Song Ranking App ===")
    print(f"Batch:      {batch_dir.name}")
    print(f"Batch size: {args.batch_size} songs/round")
    print(f"Open in browser: http://localhost:{args.port}/\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
