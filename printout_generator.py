#!/usr/bin/env python3
"""
printout_generator.py — Generate per-bird anonymized HTML pages for printing.

Each page shows one bird's spectrograms with blank rank boxes and a notes
field.  Bird identity is hidden; a separate answer-key page maps UIDs back
to bird info.

Layout per bird:
  - Header: anonymized bird code, trait, scorer line
  - Grid: 5-8 spectrogram panels, each with a rank box and notes line
  - Footer: cumulative ranked list for easy data entry after scoring

Last page of the output: answer key (uid → bird_id, role, pairing).

Usage
-----
    python printout_generator.py E:/scoring/batches/pk24bu3_wh88br85_20260410
    python printout_generator.py <batch_dir> --snippets-per-bird 6 --trait stereotypy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
import base64

import tables


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _img_b64(path: Path) -> str:
    """Embed a PNG as a base64 data-URI so the HTML is fully self-contained."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: monospace; font-size: 11pt;
  background: #fff; color: #000;
}
@media print {
  .page-break { page-break-after: always; }
  .no-print   { display: none; }
}
.bird-page {
  padding: 18mm 16mm 12mm;
  max-width: 280mm;
  page-break-after: always;
}
.page-header {
  border-bottom: 2px solid #000; padding-bottom: 6px; margin-bottom: 14px;
}
.page-header h2 { font-size: 1.1em; }
.meta-row {
  display: flex; gap: 40px; font-size: 0.82em;
  color: #333; margin-top: 4px;
}
.meta-row span { display: flex; gap: 8px; align-items: center; }
.blank { border-bottom: 1px solid #000; min-width: 120px; display: inline-block; }

/* spec grid */
.spec-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110mm, 1fr));
  gap: 10mm; margin-top: 10px;
}
.spec-panel {
  border: 1px solid #ccc; border-radius: 3px; padding: 6px;
}
.spec-panel img {
  width: 100%; height: auto; display: block;
  border: 1px solid #eee;
}
.panel-footer {
  margin-top: 5px; font-size: 0.78em;
}
.uid-code {
  color: #888; font-size: 0.72em;
  word-break: break-all; margin-bottom: 3px;
}
.rank-row {
  display: flex; align-items: center; gap: 6px; margin-top: 3px;
}
.rank-box {
  width: 28px; height: 22px; border: 1.5px solid #000;
  border-radius: 2px; display: inline-block;
}
.notes-line {
  border-bottom: 1px solid #aaa; margin-top: 5px; height: 16px;
  width: 100%;
}

/* ranked list at bottom of page */
.ranked-list {
  margin-top: 16px; border-top: 1px solid #ccc;
  padding-top: 10px; font-size: 0.82em;
}
.ranked-list h3 { font-size: 0.9em; margin-bottom: 6px; }
.rank-entry {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 4px;
}
.rank-entry .num { font-weight: bold; min-width: 20px; }
.rank-entry .uid { color: #666; min-width: 80px; }
.rank-entry .line { flex: 1; border-bottom: 1px solid #ccc; }

/* answer key */
.answer-key { padding: 18mm 16mm; }
.answer-key h1 { font-size: 1.2em; margin-bottom: 14px;
                 border-bottom: 2px solid #000; padding-bottom: 6px; }
.key-table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
.key-table th, .key-table td {
  border: 1px solid #ccc; padding: 4px 8px; text-align: left;
}
.key-table th { background: #f0f0f0; }
"""


def build_bird_page(
    bird_anon_id: str,
    uid_entries:  list[dict],   # list of {uid, img_b64}
    trait:        str,
    snippets_per_bird: int,
) -> str:
    """Return HTML string for one bird's printout page."""
    trait_label = trait.replace("_", " ").title()

    panels = []
    for entry in uid_entries[:snippets_per_bird]:
        uid = entry["uid"]
        img = entry["img_b64"]
        panel = f"""
        <div class="spec-panel">
          <div class="uid-code">{uid}</div>
          <img src="{img}" alt="spectrogram">
          <div class="panel-footer">
            <div class="rank-row">
              <span style="font-size:0.8em">Rank:</span>
              <div class="rank-box"></div>
            </div>
            <div class="notes-line"></div>
          </div>
        </div>"""
        panels.append(panel)

    # Ranked list section at bottom
    n = len(uid_entries[:snippets_per_bird])
    rank_entries = "".join(
        f'<div class="rank-entry">'
        f'  <span class="num">{i}.</span>'
        f'  <span class="uid">{uid_entries[i-1]["uid"] if i-1 < len(uid_entries) else ""}</span>'
        f'  <div class="line"></div>'
        f'</div>'
        for i in range(1, n + 1)
    )

    return f"""
    <div class="bird-page">
      <div class="page-header">
        <h2>Bird: {bird_anon_id}</h2>
        <div class="meta-row">
          <span>Trait: <strong>{trait_label}</strong></span>
          <span>Scorer: <span class="blank"></span></span>
          <span>Date: <span class="blank"></span></span>
        </div>
      </div>
      <div class="spec-grid">{''.join(panels)}</div>
      <div class="ranked-list">
        <h3>Ranked order (most → least {trait_label})</h3>
        {rank_entries}
      </div>
    </div>"""


def build_answer_key(manifest_rows: list[dict], pairing: dict) -> str:
    """Return HTML for the answer-key page."""
    nf = pairing["nest_father"]
    gf = pairing["genetic_father"]
    rows_html = "".join(
        f"<tr>"
        f"<td>{r['uid']}</td>"
        f"<td>{r['bird_anon_id']}</td>"
        f"<td>{r['bird_id']}</td>"
        f"<td>{r['role']}</td>"
        f"<td>{r['source_file_basename']}</td>"
        f"<td>{r['snippet_start_s']:.1f}</td>"
        f"</tr>"
        for r in manifest_rows
    )
    return f"""
    <div class="answer-key">
      <h1>Answer Key — {nf} × {gf}</h1>
      <p style="font-size:0.8em; color:#666; margin-bottom:12px;">
        Keep this page separate from the scoring sheets.
      </p>
      <table class="key-table">
        <thead>
          <tr>
            <th>UUID</th>
            <th>Anon ID</th>
            <th>Bird ID</th>
            <th>Role</th>
            <th>Source file</th>
            <th>Start (s)</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-bird anonymized HTML printout pages."
    )
    parser.add_argument("batch_dir")
    parser.add_argument("--snippets-per-bird", type=int, default=6)
    parser.add_argument("--trait", default="stereotypy",
                        choices=["stereotypy", "repeat_propensity"])
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: batch_dir/printout_<trait>.html)")
    args = parser.parse_args()

    batch_dir  = Path(args.batch_dir).resolve()
    h5_path    = batch_dir / "batch.h5"
    spec_dir   = batch_dir / "export" / "spectrograms"
    out_path   = Path(args.output) if args.output else \
                 batch_dir / f"printout_{args.trait}.html"

    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found.")
        return
    if not spec_dir.exists():
        print(f"ERROR: run export_batch.py {batch_dir} first.")
        return

    # ── Load manifest ─────────────────────────────────────────────────────────
    print(f"Loading manifest from {h5_path} ...")
    by_bird: dict[str, list] = defaultdict(list)    # bird_id → list of rows
    all_rows = []

    with tables.open_file(str(h5_path), mode="r") as h5:
        pairing = json.loads(h5.root.config._v_attrs["pairing"])
        for row in h5.root.manifest.iterrows():
            uid      = row["uid"].decode()
            bird_id  = row["bird_id"].decode()
            entry = {
                "uid":                uid,
                "bird_id":            bird_id,
                "role":               row["role"].decode(),
                "source_file_basename": Path(row["source_file"].decode()).name,
                "snippet_start_s":    float(row["snippet_start_s"]),
                "spec_idx":           int(row["spec_idx"]),
            }
            by_bird[bird_id].append(entry)
            all_rows.append(entry)

    # ── Assign anonymous IDs to birds ─────────────────────────────────────────
    # Shuffle birds so role order is not obvious from page order
    import random, hashlib
    salt = ""
    with tables.open_file(str(h5_path), mode="r") as h5:
        salt = h5.root.config._v_attrs["salt"]

    bird_ids_shuffled = sorted(
        by_bird.keys(),
        key=lambda b: hashlib.sha256((salt + b).encode()).hexdigest()
    )
    bird_anon_map = {
        bird: f"Bird-{hashlib.sha256((salt + bird).encode()).hexdigest()[:8].upper()}"
        for bird in bird_ids_shuffled
    }

    # Add anon ID to all_rows
    for r in all_rows:
        r["bird_anon_id"] = bird_anon_map[r["bird_id"]]

    # ── Build pages ───────────────────────────────────────────────────────────
    print(f"Building pages for {len(by_bird)} birds ...")
    pages = []

    for bird_id in bird_ids_shuffled:
        entries = by_bird[bird_id][:args.snippets_per_bird]
        anon_id = bird_anon_map[bird_id]

        # Embed PNGs as base64
        for e in entries:
            png_path = spec_dir / f"{e['uid']}.png"
            if png_path.exists():
                e["img_b64"] = _img_b64(png_path)
            else:
                e["img_b64"] = ""   # missing PNG — blank

        pages.append(build_bird_page(anon_id, entries, args.trait,
                                     args.snippets_per_bird))

    # Answer key as last page
    pages.append(build_answer_key(all_rows, pairing))

    nf = pairing["nest_father"]
    gf = pairing["genetic_father"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Printout — {nf} × {gf} — {args.trait}</title>
  <style>{CSS}</style>
</head>
<body>
{''.join(pages)}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"\nWritten: {out_path}")
    print("Open in browser and use File → Print (set to A4/Letter landscape) to save as PDF.")


if __name__ == "__main__":
    main()
