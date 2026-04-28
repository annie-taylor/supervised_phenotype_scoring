#!/usr/bin/env python3
"""
analyze_rankings.py — Aggregate session JSON rankings into Elo scores and IRR stats.

Reads all session files from <batch_dir>/sessions/, converts each round's
ordered ranking into pairwise comparisons, updates Elo ratings, then produces:

  results/<batch_id>_<trait>_elo.csv          — per-snippet Elo scores with metadata
  results/<batch_id>_<trait>_birds.csv        — per-bird averages by role
  results/<batch_id>_<trait>_consistency.csv  — per-snippet rank consistency stats
  results/<batch_id>_<trait>_irr.csv          — pairwise Kendall τ between scorers
  results/<batch_id>_<trait>_summary.txt      — human-readable digest

  results/<batch_id>_<trait>_bird_elo.png         — per-bird Elo bar chart
  results/<batch_id>_<trait>_snippet_elo.png       — per-snippet Elo strip plot by role
  results/<batch_id>_<trait>_rank_consistency.png  — mean rank vs SD scatter
  results/<batch_id>_<trait>_scorer_agreement.png  — pairwise scorer rank scatter

Elo parameters follow standard chess convention (K=32, base=400) but applied
to ranking tournaments rather than individual matches.

When --scoring-mode is given, output filenames include the mode
(e.g. bk37wh86_rd75wh72_20260414_same_tutor_stereotypy_elo.csv) so that
results from different modes do not overwrite each other.

Usage
-----
    python analyze_rankings.py E:/scoring/batches/pk24bu3_wh88br85_20260410
    python analyze_rankings.py <batch_dir> --trait stereotypy --k 32 --min-rounds 2
    python analyze_rankings.py <batch_dir> --trait all          # run both traits
    python analyze_rankings.py <batch_dir> --scoring-mode same_tutor --no-plots
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

# Windows cmd/PowerShell default to cp1252; force UTF-8 so Unicode in
# summary output (tau, em-dashes, etc.) prints correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")          # headless; must come before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tables

# ── Constants ─────────────────────────────────────────────────────────────────────

ELO_START   = 1500.0
ELO_BASE    = 400.0
ELO_K       = 32.0

# Rounds shorter than this are flagged as suspiciously fast
MIN_ROUND_SECONDS = 5.0

ROLE_ORDER  = ["nest_father", "genetic_father", "xf", "hr_nest", "hr_genetic"]

ROLE_LABELS = {
    "nest_father":    "Nest father",
    "genetic_father": "Genetic father",
    "xf":             "Cross-foster offspring",
    "hr_nest":        "Home-reared (nest-type)",
    "hr_genetic":     "Home-reared (genetic-type)",
}

ROLE_COLORS = {
    "nest_father":    "#4878CF",
    "genetic_father": "#E07B39",
    "xf":             "#2CA25F",
    "hr_nest":        "#8B6CAC",
    "hr_genetic":     "#CC4444",
}


# ── Elo helpers ───────────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected win probability for A against B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / ELO_BASE))


def update_elo(ratings: dict[str, float], winner: str, loser: str,
               k: float = ELO_K) -> None:
    """Update ratings in-place for one pairwise comparison (winner beats loser)."""
    ra = ratings[winner]
    rb = ratings[loser]
    ea = expected_score(ra, rb)
    ratings[winner] += k * (1.0 - ea)
    ratings[loser]  += k * (0.0 - (1.0 - ea))


def ranking_to_pairs(ranking: list[str]) -> list[tuple[str, str]]:
    """
    Convert an ordered ranking (index 0 = best = most of trait) into all
    (winner, loser) pairs implied by the ordering.
    """
    pairs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            pairs.append((ranking[i], ranking[j]))   # i ranked higher than j
    return pairs


# ── Session loading ───────────────────────────────────────────────────────────────

def load_sessions(sessions_dir: Path,
                  trait: Optional[str] = None) -> list[dict]:
    """Load all session JSON files; optionally filter by trait."""
    sessions = []
    for p in sorted(sessions_dir.glob("*.json")):
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
            continue
        if trait and s.get("trait") != trait:
            continue
        sessions.append(s)
    return sessions


def flag_fast_rounds(sessions: list[dict], min_s: float = MIN_ROUND_SECONDS
                     ) -> int:
    """
    Mark rounds with elapsed_s < min_s as suspicious.
    Returns count of flagged rounds (for reporting); mutates round dicts in-place.
    """
    n = 0
    for sess in sessions:
        for rnd in sess.get("rounds", []):
            elapsed = rnd.get("elapsed_s")
            if elapsed is not None and elapsed < min_s:
                rnd["_flagged"] = True
                n += 1
    return n


# ── Elo computation ───────────────────────────────────────────────────────────────

def collect_flagged_uids(sessions: list[dict]) -> set[str]:
    """Return the set of UIDs flagged as noise/call by any scorer in any round."""
    flagged: set[str] = set()
    for sess in sessions:
        for rnd in sess.get("rounds", []):
            flagged.update(rnd.get("flagged", []))
    return flagged


def compute_elo(sessions: list[dict],
                all_uids: set[str],
                k: float = ELO_K,
                skip_flagged: bool = True,
                excluded_uids: set[str] | None = None) -> dict[str, float]:
    """
    Run all pairwise comparisons across all sessions/rounds and return
    final Elo ratings per UID.

    UIDs in *excluded_uids* are not initialised and are silently dropped from
    every round's ranking before generating pairs, so they never influence the
    ratings of other snippets.
    """
    excluded = excluded_uids or set()
    ratings  = {uid: ELO_START for uid in all_uids if uid not in excluded}
    n_pairs  = 0

    for sess in sessions:
        for rnd in sess.get("rounds", []):
            if skip_flagged and rnd.get("_flagged"):
                continue
            ranking = [uid for uid in rnd.get("ranking", []) if uid not in excluded]
            for winner, loser in ranking_to_pairs(ranking):
                if winner in ratings and loser in ratings:
                    update_elo(ratings, winner, loser, k=k)
                    n_pairs += 1

    print(f"  Processed {n_pairs:,} pairwise comparisons from "
          f"{sum(len(s.get('rounds',[])) for s in sessions)} rounds.")
    return ratings


# ── Inter-rater reliability ───────────────────────────────────────────────────────

def scorer_ranking(sessions: list[dict],
                   uids_in_common: list[str]) -> dict[str, dict[str, float]]:
    """
    For each scorer, build a mean rank per UID (lower = more of trait = better).
    Only UIDs that appear in at least one of that scorer's rounds are included.
    Returns {scorer: {uid: mean_rank}}.
    """
    # Accumulate (sum_rank, count) per scorer per uid
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for sess in sessions:
        scorer = sess.get("scorer", "unknown")
        for rnd in sess.get("rounds", []):
            if rnd.get("_flagged"):
                continue
            ranking = rnd.get("ranking", [])
            for pos, uid in enumerate(ranking, start=1):
                acc[scorer][uid].append(float(pos))

    result = {}
    for scorer, uid_ranks in acc.items():
        result[scorer] = {uid: float(np.mean(ranks))
                          for uid, ranks in uid_ranks.items()}
    return result


def kendall_tau(rank_a: dict[str, float],
                rank_b: dict[str, float]) -> Optional[float]:
    """
    Compute Kendall's τ-b between two scorer dicts on their shared UIDs.
    Returns None if fewer than 2 shared UIDs.
    """
    shared = sorted(set(rank_a) & set(rank_b))
    if len(shared) < 2:
        return None

    a = [rank_a[u] for u in shared]
    b = [rank_b[u] for u in shared]

    # Count concordant / discordant pairs
    n = len(shared)
    nc = nd = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da * db > 0:
                nc += 1
            elif da * db < 0:
                nd += 1
            # ties contribute 0

    # τ-b denominator accounts for ties
    def tie_count(v):
        from collections import Counter
        c = Counter(v)
        return sum(t * (t - 1) // 2 for t in c.values())

    n_pairs = n * (n - 1) // 2
    t1 = tie_count(a)
    t2 = tie_count(b)
    denom = math.sqrt((n_pairs - t1) * (n_pairs - t2))
    if denom == 0:
        return None
    return (nc - nd) / denom


def compute_irr(sessions: list[dict]) -> list[dict]:
    """
    Compute pairwise Kendall τ between all scorer pairs.
    Returns list of dicts with scorer_a, scorer_b, tau, n_shared_uids.
    """
    scorer_ranks = scorer_ranking(sessions, [])
    scorers = sorted(scorer_ranks.keys())
    rows = []
    for sa, sb in combinations(scorers, 2):
        shared = set(scorer_ranks[sa]) & set(scorer_ranks[sb])
        tau = kendall_tau(scorer_ranks[sa], scorer_ranks[sb])
        rows.append({
            "scorer_a":     sa,
            "scorer_b":     sb,
            "tau":          round(tau, 4) if tau is not None else None,
            "n_shared_uids": len(shared),
        })
    return rows


# ── Rank consistency ──────────────────────────────────────────────────────────────

def compute_rank_consistency(
    sessions: list[dict],
    excluded_uids: set[str] | None = None,
) -> dict[str, dict]:
    """
    For each UID, collect all rank positions across (scorer, round) appearances.

    Normalised rank = (position − 1) / (n_ranked − 1), mapped to [0, 1]
    where 0 = ranked first (most of trait) and 1 = ranked last.
    Rounds with only one ranked item are excluded from normalised positions
    (raw position is still recorded).

    Parameters
    ----------
    sessions : list[dict]
        Session dicts, already processed by ``flag_fast_rounds``.
    excluded_uids : set[str], optional
        UIDs excluded as noise/call — omitted from all computations.

    Returns
    -------
    dict
        Maps uid ->{
            "positions":      list[int]   — raw 1-based positions within round
            "norm_positions": list[float] — normalised [0, 1] positions
            "scorers":        set[str]    — scorers who ranked this UID
            "n_appearances":  int
            "n_scorers":      int
        }
    """
    excluded = excluded_uids or set()
    acc: dict[str, dict] = {}

    for sess in sessions:
        scorer = sess.get("scorer", "unknown")
        for rnd in sess.get("rounds", []):
            if rnd.get("_flagged"):
                continue
            ranking  = [u for u in rnd.get("ranking", []) if u not in excluded]
            n_ranked = len(ranking)
            for pos0, uid in enumerate(ranking):          # pos0 is 0-based
                if uid not in acc:
                    acc[uid] = {
                        "positions":      [],
                        "norm_positions": [],
                        "scorers":        set(),
                    }
                acc[uid]["positions"].append(pos0 + 1)    # store as 1-based
                acc[uid]["scorers"].add(scorer)
                if n_ranked > 1:
                    acc[uid]["norm_positions"].append(pos0 / (n_ranked - 1))

    for uid, info in acc.items():
        info["n_appearances"] = len(info["positions"])
        info["n_scorers"]     = len(info["scorers"])

    return acc


# ── Aggregation helpers ───────────────────────────────────────────────────────────

def bird_averages(elo_scores: dict[str, float],
                  uid_meta: dict[str, dict]) -> dict[str, dict]:
    """
    Average Elo scores by bird_id.
    Returns {bird_id: {mean_elo, sd_elo, n_snippets, role}}.
    """
    by_bird: dict[str, list[float]] = defaultdict(list)
    bird_role: dict[str, str] = {}
    for uid, score in elo_scores.items():
        meta = uid_meta.get(uid)
        if not meta:
            continue
        bird_id = meta["bird_id"]
        by_bird[bird_id].append(score)
        bird_role[bird_id] = meta["role"]

    result = {}
    for bird_id, scores in by_bird.items():
        result[bird_id] = {
            "bird_id":    bird_id,
            "role":       bird_role[bird_id],
            "mean_elo":   round(float(np.mean(scores)), 2),
            "sd_elo":     round(float(np.std(scores)), 2),
            "n_snippets": len(scores),
        }
    return result


def role_summary(bird_avgs: dict[str, dict]) -> dict[str, dict]:
    """
    Summarize mean/sd/n at the role level.
    """
    by_role: dict[str, list[float]] = defaultdict(list)
    for info in bird_avgs.values():
        by_role[info["role"]].append(info["mean_elo"])

    result = {}
    for role, vals in by_role.items():
        result[role] = {
            "role":     role,
            "n_birds":  len(vals),
            "mean_elo": round(float(np.mean(vals)), 2),
            "sd_elo":   round(float(np.std(vals)), 2),
        }
    return result


# ── CSV / text writers ────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(no data)\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(str(k) for k in keys)]
    for r in rows:
        lines.append(",".join("" if r[k] is None else str(r[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_flagged_csv(
    flagged_uids: set[str],
    uid_meta:     dict[str, dict],
    sessions:     list[dict],
    results_dir:  Path,
    output_stem:  str,
) -> None:
    """Write flagged noise/call snippets to CSV for exclusion from future batches."""
    if not flagged_uids:
        return
    flag_scorers: dict[str, set] = defaultdict(set)
    for sess in sessions:
        scorer = sess.get("scorer", "unknown")
        for rnd in sess.get("rounds", []):
            for uid in rnd.get("flagged", []):
                flag_scorers[uid].add(scorer)

    rows = []
    for uid in sorted(flagged_uids):
        meta = uid_meta.get(uid, {})
        rows.append({
            "uid":               uid,
            "bird_id":           meta.get("bird_id",         ""),
            "role":              meta.get("role",             ""),
            "source_file":       meta.get("source_file",     ""),
            "snippet_start_s":   meta.get("snippet_start_s", ""),
            "n_scorers_flagged": len(flag_scorers.get(uid, set())),
            "flagged_by":        ";".join(sorted(flag_scorers.get(uid, set()))),
        })
    write_csv(rows, results_dir / f"{output_stem}_flagged.csv")
    print(f"  Wrote {len(rows)} flagged snippets -> {output_stem}_flagged.csv")


def write_consistency_csv(
    consistency: dict[str, dict],
    uid_meta:    dict[str, dict],
    results_dir: Path,
    output_stem: str,
) -> None:
    """
    Write per-snippet rank consistency stats to CSV.

    Columns: uid, bird_id, role, n_appearances, n_scorers,
             mean_norm_rank, sd_norm_rank, scorers.
    """
    rows = []
    for uid, info in consistency.items():
        meta = uid_meta.get(uid, {})
        npos = info["norm_positions"]
        rows.append({
            "uid":            uid,
            "bird_id":        meta.get("bird_id", ""),
            "role":           meta.get("role", ""),
            "n_appearances":  info["n_appearances"],
            "n_scorers":      info["n_scorers"],
            "mean_norm_rank": round(float(np.mean(npos)), 4) if npos else "",
            "sd_norm_rank":   round(float(np.std(npos)),  4) if len(npos) > 1 else "",
            "scorers":        ";".join(sorted(info["scorers"])),
        })
    rows.sort(key=lambda r: (
        float(r["mean_norm_rank"]) if r["mean_norm_rank"] != "" else 999.0
    ))
    write_csv(rows, results_dir / f"{output_stem}_consistency.csv")
    print(f"  Wrote consistency stats for {len(rows)} snippets "
          f"→ {output_stem}_consistency.csv")


def write_summary(
    output_stem:     str,
    trait:           str,
    pairing:         dict,
    sessions:        list[dict],
    n_flagged:       int,
    n_noise_flagged: int,
    elo_scores:      dict[str, float],
    bird_avgs:       dict[str, dict],
    irr_rows:        list[dict],
    results_dir:     Path,
) -> None:
    nf = pairing["nest_father"]
    gf = pairing["genetic_father"]
    n_rounds = sum(len(s.get("rounds", [])) for s in sessions)
    scorers  = sorted({s.get("scorer", "?") for s in sessions})

    role_stats = role_summary(bird_avgs)

    noise_note = (f"  ({n_noise_flagged} excluded as noise/call)"
                  if n_noise_flagged else "")
    lines = [
        f"=== Ranking Analysis ===",
        f"Batch:   {output_stem}",
        f"Pairing: {nf} × {gf}",
        f"Trait:   {trait.replace('_', ' ').title()}",
        f"",
        f"Sessions: {len(sessions)}",
        f"Scorers:  {', '.join(scorers)}",
        f"Rounds:   {n_rounds}   ({n_flagged} flagged as too-fast)",
        f"UIDs:     {len(elo_scores)}{noise_note}",
        f"",
        f"--- Elo scores by role (higher = more of trait) ---",
    ]

    for role in ROLE_ORDER:
        if role not in role_stats:
            continue
        s = role_stats[role]
        label = ROLE_LABELS.get(role, role)
        lines.append(f"  {label:<30}  n={s['n_birds']:>3}  "
                     f"mean={s['mean_elo']:>7.1f}  sd={s['sd_elo']:>6.1f}")

    # Offspring vs fathers
    xf_vals  = [v["mean_elo"] for v in bird_avgs.values() if v["role"] == "xf"]
    nf_vals  = [v["mean_elo"] for v in bird_avgs.values() if v["role"] == "nest_father"]
    gf_vals  = [v["mean_elo"] for v in bird_avgs.values() if v["role"] == "genetic_father"]

    lines += ["", "--- Offspring vs. fathers ---"]
    if xf_vals and nf_vals:
        diff = np.mean(xf_vals) - np.mean(nf_vals)
        lines.append(f"  XF − nest_father:    {diff:+.1f} Elo points")
    if xf_vals and gf_vals:
        diff = np.mean(xf_vals) - np.mean(gf_vals)
        lines.append(f"  XF − genetic_father: {diff:+.1f} Elo points")

    if irr_rows:
        lines += ["", "--- Inter-rater reliability (Kendall τ) ---"]
        for r in irr_rows:
            tau_str = f"{r['tau']:.3f}" if r["tau"] is not None else "n/a"
            lines.append(f"  {r['scorer_a']} vs {r['scorer_b']}: "
                         f"τ = {tau_str}  (n={r['n_shared_uids']} shared songs)")

    lines += ["", f"Output files: {results_dir}"]

    txt = "\n".join(lines) + "\n"
    out = results_dir / f"{output_stem}_{trait}_summary.txt"
    out.write_text(txt, encoding="utf-8")
    print(txt)


# ── Plot helpers ──────────────────────────────────────────────────────────────────

def _role_color(role: str) -> str:
    """Return the hex colour for a role string."""
    return ROLE_COLORS.get(role, "#888888")


def plot_bird_elo(
    bird_avgs:   dict[str, dict],
    trait:       str,
    results_dir: Path,
    output_stem: str,
) -> None:
    """
    Bar chart of per-bird mean Elo ± SD, grouped and coloured by role.

    Parameters
    ----------
    bird_avgs : dict
        Output of ``bird_averages()``.
    trait : str
        Trait label used in the plot title.
    results_dir : Path
        Directory where the PNG is saved.
    output_stem : str
        File-name prefix (batch_id or batch_id_scoring_mode).
    """
    rows = sorted(bird_avgs.values(), key=lambda x: (
        ROLE_ORDER.index(x["role"]) if x["role"] in ROLE_ORDER else 99,
        -x["mean_elo"],
    ))
    if not rows:
        return

    labels = [r["bird_id"]  for r in rows]
    means  = [r["mean_elo"] for r in rows]
    sds    = [r["sd_elo"]   for r in rows]
    colors = [_role_color(r["role"]) for r in rows]

    fig, ax = plt.subplots(figsize=(max(5, len(rows) * 0.85), 4))
    x = range(len(rows))
    ax.bar(x, means, yerr=sds, color=colors, capsize=4,
           error_kw={"linewidth": 1.2}, edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Elo")
    ax.set_title(f"{output_stem}\n{trait.replace('_', ' ').title()} — per-bird Elo")
    ax.axhline(ELO_START, color="grey", linewidth=0.8, linestyle="--",
               label="baseline (1500)")

    seen_roles = dict.fromkeys(r["role"] for r in rows)
    patches = [mpatches.Patch(color=_role_color(role),
                               label=ROLE_LABELS.get(role, role))
               for role in seen_roles]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    fig.tight_layout()
    out = results_dir / f"{output_stem}_{trait}_bird_elo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {out.name}")


def plot_snippet_elo(
    elo_scores:  dict[str, float],
    uid_meta:    dict[str, dict],
    trait:       str,
    results_dir: Path,
    output_stem: str,
) -> None:
    """
    Strip plot of per-snippet Elo scores grouped by role.

    Each point is one snippet. A horizontal line marks the per-role median.
    The baseline Elo (1500) is shown as a dashed reference.

    Parameters
    ----------
    elo_scores : dict
        Output of ``compute_elo()``.
    uid_meta : dict
        Maps uid ->metadata dict (must contain "role").
    trait : str
        Trait label used in the plot title.
    results_dir : Path
        Directory where the PNG is saved.
    output_stem : str
        File-name prefix.
    """
    by_role: dict[str, list[float]] = defaultdict(list)
    for uid, score in elo_scores.items():
        role = uid_meta.get(uid, {}).get("role", "unknown")
        by_role[role].append(score)

    roles = [r for r in ROLE_ORDER if r in by_role]
    if not roles:
        return

    fig, ax = plt.subplots(figsize=(max(4, len(roles) * 1.6), 4))
    rng = np.random.default_rng(42)
    for xi, role in enumerate(roles):
        vals   = by_role[role]
        jitter = rng.uniform(-0.2, 0.2, len(vals))
        ax.scatter(xi + jitter, vals, color=_role_color(role),
                   s=35, alpha=0.75, linewidths=0)
        ax.plot([xi - 0.28, xi + 0.28], [np.median(vals)] * 2,
                color="black", linewidth=2.0)

    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([ROLE_LABELS.get(r, r) for r in roles],
                        rotation=20, ha="right", fontsize=9)
    ax.axhline(ELO_START, color="grey", linewidth=0.8, linestyle="--",
               label="baseline (1500)")
    ax.legend(fontsize=8)
    ax.set_ylabel("Elo")
    ax.set_title(f"{output_stem}\n{trait.replace('_', ' ').title()} — snippet Elo by role")
    fig.tight_layout()
    out = results_dir / f"{output_stem}_{trait}_snippet_elo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {out.name}")


def plot_rank_consistency(
    consistency: dict[str, dict],
    uid_meta:    dict[str, dict],
    trait:       str,
    results_dir: Path,
    output_stem: str,
) -> None:
    """
    Scatter of mean normalised rank (x) vs SD (y), one point per UID.

    Normalised rank runs from 0 (ranked first / most of trait) to 1 (ranked
    last / least of trait).  High SD means the snippet was ranked inconsistently
    across appearances.  Point size scales with number of appearances.  UIDs
    that appear only once (SD undefined) are drawn as ×.

    Parameters
    ----------
    consistency : dict
        Output of ``compute_rank_consistency()``.
    uid_meta : dict
        Maps uid ->metadata dict.
    trait : str
        Trait label used in the plot title.
    results_dir : Path
        Directory where the PNG is saved.
    output_stem : str
        File-name prefix.
    """
    uids = [u for u, d in consistency.items() if d["norm_positions"]]
    if not uids:
        return

    xs     = [float(np.mean(consistency[u]["norm_positions"])) for u in uids]
    ys     = [float(np.std(consistency[u]["norm_positions"]))   for u in uids]
    sizes  = [20 + 18 * consistency[u]["n_appearances"]         for u in uids]
    colors = [_role_color(uid_meta.get(u, {}).get("role", ""))  for u in uids]
    multi  = [consistency[u]["n_appearances"] > 1               for u in uids]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Single-appearance points (SD = 0 by definition — drawn as ×)
    sx = [x for x, m in zip(xs, multi) if not m]
    sy = [y for y, m in zip(ys, multi) if not m]
    sc = [c for c, m in zip(colors, multi) if not m]
    if sx:
        ax.scatter(sx, sy, c=sc, s=30, alpha=0.45, marker="x",
                   linewidths=1.5, label="1 appearance")

    # Multi-appearance points
    for x, y, s, c, m in zip(xs, ys, sizes, colors, multi):
        if m:
            ax.scatter(x, y, c=c, s=s, alpha=0.80,
                       edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Mean normalised rank  (0 = most of trait, 1 = least)")
    ax.set_ylabel("SD of normalised rank  (lower = more consistent)")
    ax.set_title(f"{output_stem}\n{trait.replace('_', ' ').title()} — rank consistency")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.03, None)

    seen_roles = dict.fromkeys(uid_meta.get(u, {}).get("role", "") for u in uids)
    patches = [mpatches.Patch(color=_role_color(role),
                               label=ROLE_LABELS.get(role, role))
               for role in seen_roles if role]
    if patches:
        ax.legend(handles=patches, fontsize=8)

    fig.tight_layout()
    out = results_dir / f"{output_stem}_{trait}_rank_consistency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {out.name}")


def plot_scorer_agreement(
    scorer_ranks: dict[str, dict[str, float]],
    uid_meta:     dict[str, dict],
    irr_rows:     list[dict],
    trait:        str,
    results_dir:  Path,
    output_stem:  str,
) -> None:
    """
    Pairwise scorer-agreement scatter plots, one panel per scorer pair.

    Each point is a UID scored by both scorers. Axes are mean rank position
    (lower = ranked higher / more of trait). The identity line shows perfect
    agreement. Kendall τ is annotated in each panel.

    Parameters
    ----------
    scorer_ranks : dict
        Output of ``scorer_ranking()``.
    uid_meta : dict
        Maps uid ->metadata dict.
    irr_rows : list[dict]
        Output of ``compute_irr()`` — used to look up pre-computed τ values.
    trait : str
        Trait label used in the plot title.
    results_dir : Path
        Directory where the PNG is saved.
    output_stem : str
        File-name prefix.
    """
    scorers = sorted(scorer_ranks.keys())
    pairs   = list(combinations(scorers, 2))
    if not pairs:
        return

    tau_lookup = {(r["scorer_a"], r["scorer_b"]): r["tau"] for r in irr_rows}

    ncols = min(len(pairs), 3)
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2 * ncols, 4.0 * nrows),
                              squeeze=False)

    for idx, (sa, sb) in enumerate(pairs):
        ax     = axes[idx // ncols][idx % ncols]
        shared = sorted(set(scorer_ranks[sa]) & set(scorer_ranks[sb]))
        if len(shared) < 2:
            ax.set_visible(False)
            continue

        xa     = [scorer_ranks[sa][u] for u in shared]
        xb     = [scorer_ranks[sb][u] for u in shared]
        colors = [_role_color(uid_meta.get(u, {}).get("role", "")) for u in shared]

        ax.scatter(xa, xb, c=colors, s=35, alpha=0.78,
                   edgecolors="white", linewidths=0.5)

        lo = min(xa + xb); hi = max(xa + xb)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.4)

        tau_key = (sa, sb) if (sa, sb) in tau_lookup else (sb, sa)
        tau_val = tau_lookup.get(tau_key)
        tau_str = f"τ = {tau_val:.3f}" if tau_val is not None else "τ = n/a"
        ax.text(0.05, 0.95, tau_str, transform=ax.transAxes, fontsize=9,
                va="top")

        ax.set_xlabel(f"{sa}  mean rank", fontsize=9)
        ax.set_ylabel(f"{sb}  mean rank", fontsize=9)
        ax.set_title(f"{sa} vs {sb}", fontsize=10)

    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"{output_stem}\n{trait.replace('_', ' ').title()} — scorer agreement",
        fontsize=11,
    )
    fig.tight_layout()
    out = results_dir / f"{output_stem}_{trait}_scorer_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────────

def analyze_one_trait(
    batch_dir:    Path,
    trait:        str,
    uid_meta:     dict[str, dict],
    pairing:      dict,
    k:            float,
    min_rounds:   int,
    results_dir:  Path,
    scoring_mode: str | None = None,
    no_plots:     bool = False,
) -> None:
    batch_id    = batch_dir.name
    output_stem = (f"{batch_id}_{scoring_mode}" if scoring_mode else batch_id)

    sessions_dir = (batch_dir / "sessions" / scoring_mode
                    if scoring_mode else batch_dir / "sessions")

    print(f"\n--- {trait} ---")
    sessions = load_sessions(sessions_dir, trait=trait)
    if not sessions:
        print(f"  No session files found for trait '{trait}' in {sessions_dir}")
        return

    # Check minimum rounds per scorer
    scorer_counts: dict[str, int] = defaultdict(int)
    for s in sessions:
        scorer_counts[s.get("scorer", "?")] += len(s.get("rounds", []))
    for scorer, n in scorer_counts.items():
        if n < min_rounds:
            print(f"  WARNING: scorer '{scorer}' has only {n} round(s) "
                  f"(--min-rounds={min_rounds})")

    n_flagged = flag_fast_rounds(sessions)
    if n_flagged:
        print(f"  Flagged {n_flagged} suspiciously fast rounds (<{MIN_ROUND_SECONDS}s)")

    # Collect noise/call flags — excluded from Elo entirely
    noise_uids = collect_flagged_uids(sessions)
    if noise_uids:
        print(f"  Excluding {len(noise_uids)} noise/call snippet(s) flagged by scorers")
        write_flagged_csv(noise_uids, uid_meta, sessions, results_dir, output_stem)

    all_uids   = set(uid_meta.keys())
    elo_scores = compute_elo(sessions, all_uids, k=k, excluded_uids=noise_uids)

    # Per-snippet CSV
    snippet_rows = []
    for uid, score in sorted(elo_scores.items(), key=lambda x: -x[1]):
        meta = uid_meta.get(uid, {})
        snippet_rows.append({
            "uid":     uid,
            "elo":     round(score, 2),
            "bird_id": meta.get("bird_id", ""),
            "role":    meta.get("role", ""),
        })
    write_csv(snippet_rows, results_dir / f"{output_stem}_{trait}_elo.csv")

    # Per-bird CSV
    bird_avgs = bird_averages(elo_scores, uid_meta)
    bird_rows = sorted(bird_avgs.values(), key=lambda x: -x["mean_elo"])
    write_csv(bird_rows, results_dir / f"{output_stem}_{trait}_birds.csv")

    # Rank consistency CSV
    consistency = compute_rank_consistency(sessions, excluded_uids=noise_uids)
    write_consistency_csv(consistency, uid_meta, results_dir, f"{output_stem}_{trait}")

    # IRR
    irr_rows = compute_irr(sessions)
    write_csv(irr_rows, results_dir / f"{output_stem}_{trait}_irr.csv")

    # Summary text
    write_summary(
        output_stem, trait, pairing,
        sessions, n_flagged, len(noise_uids),
        elo_scores, bird_avgs, irr_rows,
        results_dir,
    )

    # Plots
    if not no_plots:
        plot_bird_elo(bird_avgs, trait, results_dir, output_stem)
        plot_snippet_elo(elo_scores, uid_meta, trait, results_dir, output_stem)
        plot_rank_consistency(consistency, uid_meta, trait, results_dir, output_stem)
        sranks = scorer_ranking(sessions, [])
        plot_scorer_agreement(sranks, uid_meta, irr_rows, trait, results_dir, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate session JSON rankings into Elo scores and IRR stats."
    )
    parser.add_argument("batch_dir", help="Path to batch directory")
    parser.add_argument("--trait", default="all",
                        help="Trait to analyze: stereotypy, repeat_propensity, or 'all'")
    parser.add_argument("--k",    type=float, default=ELO_K,
                        help=f"Elo K-factor (default {ELO_K})")
    parser.add_argument("--min-rounds", type=int, default=1,
                        help="Warn if a scorer has fewer than this many rounds")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: batch_dir/../results/)")
    parser.add_argument("--scoring-mode", default=None,
                        choices=["all", "same_tutor"],
                        help="Load sessions from sessions/<mode>/ subdirectory. "
                             "Omit to load from sessions/ root (legacy behaviour).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating PNG plots.")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    h5_path   = batch_dir / "batch.h5"

    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found.")
        return

    # ── Load uid_meta and pairing from HDF5 ──────────────────────────────────
    uid_meta = {}
    with tables.open_file(str(h5_path), mode="r") as h5:
        pairing = json.loads(h5.root.config._v_attrs["pairing"])
        for row in h5.root.manifest.iterrows():
            uid = row["uid"].decode()
            uid_meta[uid] = {
                "bird_id":         row["bird_id"].decode(),
                "role":            row["role"].decode(),
                "spec_idx":        int(row["spec_idx"]),
                "source_file":     row["source_file"].decode(),
                "snippet_start_s": float(row["snippet_start_s"]),
            }

    print(f"Loaded {len(uid_meta)} UIDs from {h5_path.name}")

    # ── Results directory ─────────────────────────────────────────────────────
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = batch_dir.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    traits = ["stereotypy", "repeat_propensity"] if args.trait == "all" \
             else [args.trait]

    for trait in traits:
        analyze_one_trait(
            batch_dir    = batch_dir,
            trait        = trait,
            uid_meta     = uid_meta,
            pairing      = pairing,
            k            = args.k,
            min_rounds   = args.min_rounds,
            results_dir  = results_dir,
            scoring_mode = args.scoring_mode,
            no_plots     = args.no_plots,
        )

    print(f"\nDone. Results in: {results_dir}")


if __name__ == "__main__":
    main()
