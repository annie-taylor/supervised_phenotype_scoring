# supervised_phenotype_scoring

A local web application for anonymised, supervised scoring of birdsong
phenotype traits by human raters.

Songs are extracted from audio recordings, stored in an HDF5 batch file,
served via a drag-and-drop ranking interface, and aggregated into per-bird
Elo scores with inter-rater reliability statistics.

Designed for cross-foster experiments where scorer identity knowledge would
bias trait ratings (stereotypy, repeat propensity, etc.).

---

## Table of contents

1. [Installation](#installation)
2. [Data preparation](#data-preparation)
3. [Pipeline overview](#pipeline-overview)
4. [Step-by-step usage](#step-by-step-usage)
5. [Scoring app](#scoring-app)
6. [Analysis](#analysis)
7. [Hosted / MTurk mode](#hosted--mturk-mode)
8. [Building the docs](#building-the-docs)

---

## Installation

### Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

### Create the conda environment

```bash
git clone <repo-url> supervised_phenotype_scoring
cd supervised_phenotype_scoring
conda env create -f environment.yml
conda activate supervised_phenotype_scoring
```

This installs Python 3.11 plus all required packages.
The `evfuncs` pip package is optional and only needed if your audio files are
in `.cbin` format.  The `boto3` package is only needed for S3 upload
(`upload_batch.py`).

### Verify the install

```bash
python make_test_batch.py
python export_batch.py batches/test_batch
python analyze_rankings.py batches/test_batch
```

You should see spectrogram PNGs appear in `batches/test_batch/export/spectrograms/`
and a summary printed to the console.

---

## Data preparation

### What you need

| File | Description |
|------|-------------|
| `audio_candidates_cache.json` | `{bird_id: [{filepath, size_mb}, ...]}` — pre-screened audio paths per bird |
| `nest_gen_pair_offspring_summary.csv` | One row per nest-father × genetic-father pairing with XF bird lists |
| `genetic_father_offspring_summary.csv` | Hand-reared bird lists per genetic father |

These files are produced by the upstream `file_management` pipeline.
Update `config.json` to point to their locations before running
`prepare_batch.py`.

### config.json

```json
{
  "project_dir":            "/path/to/project",
  "file_management_dir":    "/path/to/project/file_management",
  "audio_candidates_cache": "/path/to/audio_candidates_cache.json",
  "output_dir":             "/path/to/scoring/batches",

  "snippets_per_bird":  6,
  "snippet_duration_s": 8.0,
  "min_gap_s":          5.0,
  "edge_s":             1.0,
  "target_sr":          32000,

  "spectrogram": {
    "nfft": 1024, "hop": 256,
    "min_freq": 400, "max_freq": 10000,
    "p_low": 2, "p_high": 98
  },

  "n_workers": 4
}
```

---

## Pipeline overview

```
prepare_batch.py     audio → HDF5 batch (spectrograms + audio + manifest)
      ↓
export_batch.py      HDF5 → PNG spectrograms + WAV audio clips
      ↓
ranking_app.py       Flask drag-and-drop scoring interface
      ↓
analyze_rankings.py  session JSON files → Elo scores + IRR
```

Optionally, `printout_generator.py` produces a self-contained HTML file
suitable for printing and hand-scoring, and `upload_batch.py` pushes
exported assets to S3 for remote scoring sessions.

---

## Step-by-step usage

### 1. Build the batch

```bash
python prepare_batch.py --nest-father <nf_id> --genetic-father <gf_id>
# e.g.:
python prepare_batch.py --nest-father pk24bu3 --genetic-father wh88br85
```

This writes `batches/<nf>_<gf>_<YYYYMMDD>/batch.h5`.

Options:
- `--snippets-per-bird N` — override config (default 6)
- `--workers N` — parallel spectrogram workers (default 4)

### 2. Export spectrograms and audio

```bash
python export_batch.py batches/pk24bu3_wh88br85_20260410
```

Writes PNG and WAV files to `batches/.../export/`.

Options:
- `--dpi 150` — higher-resolution PNGs
- `--no-audio` — skip WAV export
- `--force` — overwrite existing files
- `--workers N`

### 3. Run the scoring app

```bash
python ranking_app.py batches/pk24bu3_wh88br85_20260410
```

Open `http://localhost:5000/` in a browser.  Enter your name, choose a
trait, and drag song cards into ranked order (leftmost = most of trait).
Rankings are saved after every round to `batches/.../sessions/`.

Options:
- `--port 5001`
- `--batch-size 7` — songs per ranking round

### 4. Generate the print-scoring sheet (optional)

```bash
python printout_generator.py batches/pk24bu3_wh88br85_20260410
```

Opens `batches/.../printout_stereotypy.html` — use File → Print in the
browser to produce a PDF.  Bird identities are hidden; an answer key is
appended as the last page.

### 5. Analyze rankings

```bash
python analyze_rankings.py batches/pk24bu3_wh88br85_20260410
# or for both traits:
python analyze_rankings.py batches/pk24bu3_wh88br85_20260410 --trait all
```

Writes to `results/`:
- `<batch>_<trait>_elo.csv` — per-snippet Elo scores
- `<batch>_<trait>_birds.csv` — per-bird averages by role
- `<batch>_<trait>_irr.csv` — pairwise Kendall τ between scorers
- `<batch>_<trait>_summary.txt` — human-readable digest

---

## Scoring app

### How scoring works

Each round presents `--batch-size` song cards (default 7).  Cards can be
dragged into order and reordered freely before submitting.  Click a
spectrogram image to play its audio clip.

The progress bar tracks songs ranked so far in the session.  Sessions can
be stopped at any time; progress is saved after every submitted round.

### Anonymisation

Song cards display only an 8-character UID prefix — no bird identity, no
role, no source file information.  UIDs are generated fresh for each batch
using a random salt so that the same song appearing in two batches will have
different UIDs.

The full uid → bird identity mapping is stored only in `batch.h5` and in
`printout_<trait>.html` answer key pages.

### Session files

Each scorer's session is saved as:

```
batches/<batch>/sessions/<scorer>_<trait>_<date>.json
```

The format is:

```json
{
  "scorer":    "Annie",
  "trait":     "stereotypy",
  "platform":  "local",
  "rounds": [
    {
      "round":     1,
      "presented": ["<uid>", ...],
      "ranking":   ["<uid>", ...],
      "elapsed_s": 42.1,
      "timestamp": "2026-04-10T14:32:01"
    }
  ]
}
```

This format is compatible with the MTurk session format so local and
crowdsourced sessions can be pooled in `analyze_rankings.py`.

---

## Analysis

### Elo scoring

Each submitted ranking is decomposed into all implied pairwise comparisons
(rank 1 beats rank 2, rank 1 beats rank 3, …, rank n−1 beats rank n).
Elo ratings are updated for every comparison using the standard formula
(K = 32, base = 400, start = 1500).

Higher Elo = more of the trait (leftmost = rank 1 = most stereotyped, etc.).

### Inter-rater reliability

Kendall's τ-b is computed between every pair of scorers on their shared
songs.  τ ≈ 1 means perfect agreement; τ ≈ 0 means no agreement.

Rounds completed in under 5 seconds are flagged as suspicious and excluded
from analysis.

### Role-level comparison

`analyze_rankings.py` reports mean Elo by role and prints the difference
between cross-foster offspring (xf) and each father type, making it easy
to assess whether song trait expression is closer to the nest father or
genetic father.

---

## Hosted / MTurk mode

To run scoring sessions remotely:

```bash
# 1. Upload assets to S3
python upload_batch.py batches/<batch> --bucket my-bucket --prefix scoring/<batch>

# 2. Run the app in hosted mode (reads image/audio from S3)
python ranking_app.py batches/<batch> --mode hosted
```

`upload_batch.py` writes `batches/<batch>/config_hosted.json` with the S3
base URLs after a successful upload.  Use `--dry-run` to preview what would
be uploaded.

---

## Building the docs

```bash
conda activate supervised_phenotype_scoring
cd docs
sphinx-build -b html . _build/html
# then open docs/_build/html/index.html
```

API documentation is generated automatically from the NumPy-style
docstrings in each module.  To add documentation for a new module, create
`docs/api/<module>.rst` following the pattern of the existing files.
