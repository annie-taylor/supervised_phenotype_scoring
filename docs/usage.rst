Usage Guide
===========

This page describes the full workflow from raw audio to scored results,
including the prescreen quality-control step.

.. contents:: Contents
   :local:
   :depth: 2

----

Preparing a pairings file
--------------------------

Create a plain CSV listing the nest-father × genetic-father pairings you want
to score.  A header row is optional.

.. code-block:: text

    nest_father,genetic_father
    pk24bu3,wh88br85
    ab12cd3,ef45gh6

Save this as e.g. ``pairings.csv`` in the project root.

----

Single-family workflow
-----------------------

Use this when processing one pairing at a time.

Step 1 — Build the batch
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python prepare_batch.py --nest-father pk24bu3 --genetic-father wh88br85

Writes ``batches/pk24bu3_wh88br85_<YYYYMMDD>/batch.h5``.

Options:

- ``--snippets-per-bird N`` — override config (default 6)
- ``--workers N`` — parallel spectrogram workers
- ``--exclude-csv path/to/prescreen_<date>.csv`` — skip snippets labelled
  ``not_song`` or ``rendering_error`` in a prior prescreen run
- ``--existing-batch path/to/batch.h5`` — carry over valid snippets from a
  prior run and only compute the shortfall per bird (use with ``--exclude-csv``)

Step 2 — Export spectrograms and audio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python export_batch.py batches/pk24bu3_wh88br85_20260414

Writes PNG spectrograms and WAV clips to ``batches/.../export/``.

Options:

- ``--dpi 150`` — higher-resolution PNGs
- ``--no-audio`` — skip WAV export
- ``--force`` — overwrite existing files
- ``--workers N``

Step 3 — Prescreen the batch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the prescreen app locally to label every spectrogram before expert scorers
see the batch.  Use keyboard shortcuts for fast review.

.. code-block:: bash

    python prescreen_app.py batches/pk24bu3_wh88br85_20260414

Open ``http://localhost:5001`` in a browser.  For each spectrogram press:

- **S** — song (keep)
- **N** — not song (exclude)
- **E** — rendering error (exclude)

Labels are saved to ``batches/.../prescreen_<YYYYMMDD>.csv`` after each
decision.  Close the browser and restart to resume mid-session.

Step 4 — Rebuild excluding non-song snippets, topping up where possible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass both ``--exclude-csv`` and ``--existing-batch`` so the script carries
over valid snippets from the Phase 1 batch and only recomputes the shortfall.

.. code-block:: bash

    python prepare_batch.py --nest-father pk24bu3 --genetic-father wh88br85 \
        --exclude-csv batches/pk24bu3_wh88br85_20260414/prescreen_20260414.csv \
        --existing-batch batches/pk24bu3_wh88br85_20260414/batch.h5

What this does:

- Loads all valid (non-excluded) snippets from the existing HDF5 — no recomputation
- For each bird below the target count, samples additional positions from
  audio files, respecting the min-gap constraint against existing positions
- Writes a new HDF5 combining carried-over and newly computed snippets

Then re-export the cleaned batch:

.. code-block:: bash

    python export_batch.py batches/pk24bu3_wh88br85_20260414 --force

When using ``run_pipeline.py --phase 2``, the ``--existing-batch`` flag is
passed automatically — no manual intervention needed.

Step 5 — Run the scoring app
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python ranking_app.py batches/pk24bu3_wh88br85_20260414

Open ``http://localhost:5000/``.  Enter a scorer name, choose a trait, and
drag song cards into ranked order.  Rankings are saved after every round.

For remote scoring over EC2, see :ref:`ec2-deployment`.

Step 6 — Analyze rankings
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python analyze_rankings.py batches/pk24bu3_wh88br85_20260414 --trait all

When ``--scoring-mode`` is given, the mode is included in all output filenames
(e.g. ``<batch>_same_tutor_<trait>_elo.csv``) so results from different modes
do not overwrite each other.

Options:

- ``--trait <name>`` — ``stereotypy``, ``repeat_propensity``, or ``all`` (default)
- ``--scoring-mode <mode>`` — ``same_tutor`` or ``all``; loads sessions from
  ``sessions/<mode>/`` and prefixes outputs with the mode name
- ``--k <float>`` — Elo K-factor (default 32)
- ``--min-rounds <n>`` — warn if a scorer has fewer than n rounds
- ``--no-plots`` — skip PNG plot generation
- ``--output-dir <path>`` — override the default ``results/`` directory

Outputs written to ``results/``:

.. list-table::
   :header-rows: 1

   * - File
     - Description
   * - ``<stem>_<trait>_elo.csv``
     - Per-snippet Elo scores, sorted highest to lowest
   * - ``<stem>_<trait>_birds.csv``
     - Per-bird mean Elo ± SD, sorted by role then score
   * - ``<stem>_<trait>_consistency.csv``
     - Per-snippet rank consistency: mean and SD of normalised rank position
       across all (scorer, round) appearances
   * - ``<stem>_<trait>_irr.csv``
     - Pairwise Kendall τ-b between scorers on shared UIDs
   * - ``<stem>_<trait>_summary.txt``
     - Human-readable digest: role means, offspring vs. father gaps, IRR
   * - ``<stem>_flagged.csv``
     - Noise/call snippets flagged during scoring
   * - ``<stem>_<trait>_bird_elo.png``
     - Bar chart of per-bird mean Elo ± SD, coloured by role
   * - ``<stem>_<trait>_snippet_elo.png``
     - Strip plot of per-snippet Elo by role with median line
   * - ``<stem>_<trait>_rank_consistency.png``
     - Scatter: mean normalised rank vs. SD, one point per snippet
   * - ``<stem>_<trait>_scorer_agreement.png``
     - Pairwise scorer rank scatter with Kendall τ annotation

``<stem>`` is ``<batch_id>`` when no scoring mode is given, or
``<batch_id>_<scoring_mode>`` otherwise.

Interactive exploration
^^^^^^^^^^^^^^^^^^^^^^^

Open ``explore_rankings.ipynb`` in Jupyter for an interactive walkthrough of
all analyses.  Set ``BATCH_DIR``, ``SCORING_MODE``, and ``TRAIT`` at the top
and run all cells.  The notebook covers:

- Per-bird and per-snippet ranking tables
- All four plots inline
- Within-scorer vs. across-scorer consistency breakdown
- Side-by-side scoring-mode comparison (``same_tutor`` vs ``all``)

----

Multi-family workflow
----------------------

Use ``run_pipeline.py`` to process many pairings with a single command.
The two-phase design pauses between build and rebuild so you can prescreen
each batch in between.

Phase 1 — Build and export all batches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python run_pipeline.py pairings.csv --phase 1

For each pairing this runs ``prepare_batch.py`` then ``export_batch.py``.
At the end it prints the ``prescreen_app.py`` command for each batch.

Prescreen — review each batch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the prescreen app on each batch and label all spectrograms before
continuing.  Phase 2 will skip any pairing that has no prescreen CSV.

Phase 2 — Rebuild with exclusions and re-export
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python run_pipeline.py pairings.csv --phase 2

For each pairing this finds the most recent prescreen CSV, runs
``prepare_batch.py --exclude-csv``, then re-runs ``export_batch.py``.
At the end it prints the ``scp`` commands to upload each batch to EC2.

Pass-through options (both phases):

- ``--workers N``
- ``--snippets-per-bird N``
- ``--dpi N``

----

.. _ec2-deployment:

EC2 deployment for remote scorers
-----------------------------------

After Phase 2, upload each clean batch to EC2 so expert scorers can access
the app from any machine without a local installation.

.. code-block:: powershell

    # Upload batch (run in PowerShell on local machine)
    scp -i "C:\Users\Eric\.ssh\scoring-key" -r `
      "E:\scoring\batches\<batch_name>" `
      ubuntu@<public-ip>:~/supervised_phenotype_scoring/batches/

    # SSH in and launch the app
    ssh -i "C:\Users\Eric\.ssh\scoring-key" ubuntu@<public-ip>

.. code-block:: bash

    # On EC2
    conda activate supervised_phenotype_scoring
    cd ~/supervised_phenotype_scoring
    nohup python ranking_app.py batches/<batch_name> > app.log 2>&1 &

Share ``http://<public-ip>:5000`` with scorers.  No installation required on
their end.

After scoring, retrieve session files and analyze locally:

.. code-block:: powershell

    # Retrieve sessions (PowerShell)
    scp -i "C:\Users\Eric\.ssh\scoring-key" -r `
      ubuntu@<public-ip>:~/supervised_phenotype_scoring/batches/<batch_name>/sessions `
      "E:\scoring\batches\<batch_name>\"

.. code-block:: bash

    # Analyze locally
    python analyze_rankings.py batches/<batch_name> --trait all

See ``docs/ec2_deployment_guide.txt`` for the full EC2 setup walkthrough.

----

Scoring app interface
----------------------

- Drag cards to rank songs left (most of trait) to right (least).
- Click a spectrogram image to play its audio clip.
- Click **A** or **B** on any card to load it into a side-by-side comparison
  panel below (interactive Plotly heatmap — scroll to zoom, drag to pan).
- Click **flag noise/call** to exclude a snippet from the ranking.
  Flagged snippets are shown with a red border and excluded from Elo scoring.

----

Output files reference
------------------------

.. list-table::
   :header-rows: 1

   * - File
     - Created by
     - Description
   * - ``batch.h5``
     - ``prepare_batch.py``
     - HDF5 batch: spectrograms, audio, manifest
   * - ``batch_index.json``
     - ``prepare_batch.py``
     - Public batch metadata (no bird identity)
   * - ``prescreen_<date>.csv``
     - ``prescreen_app.py``
     - Per-snippet labels: song / not_song / rendering_error
   * - ``export/spectrograms/<uid>.png``
     - ``export_batch.py``
     - Spectrogram PNGs (magma colorscale)
   * - ``export/audio/<uid>.wav``
     - ``export_batch.py``
     - 8-second audio clips
   * - ``sessions/<scorer>_<trait>_<date>.json``
     - ``ranking_app.py``
     - Per-scorer ranking records
   * - ``results/<stem>_<trait>_elo.csv``
     - ``analyze_rankings.py``
     - Per-snippet Elo scores
   * - ``results/<stem>_<trait>_birds.csv``
     - ``analyze_rankings.py``
     - Per-bird mean Elo ± SD by role
   * - ``results/<stem>_<trait>_consistency.csv``
     - ``analyze_rankings.py``
     - Per-snippet rank consistency stats
   * - ``results/<stem>_<trait>_irr.csv``
     - ``analyze_rankings.py``
     - Pairwise Kendall τ between scorers
   * - ``results/<stem>_<trait>_summary.txt``
     - ``analyze_rankings.py``
     - Human-readable digest
   * - ``results/<stem>_flagged.csv``
     - ``analyze_rankings.py``
     - Noise/call snippets flagged during scoring
   * - ``results/<stem>_<trait>_bird_elo.png``
     - ``analyze_rankings.py``
     - Per-bird Elo bar chart
   * - ``results/<stem>_<trait>_snippet_elo.png``
     - ``analyze_rankings.py``
     - Per-snippet Elo strip plot by role
   * - ``results/<stem>_<trait>_rank_consistency.png``
     - ``analyze_rankings.py``
     - Mean rank vs. SD scatter
   * - ``results/<stem>_<trait>_scorer_agreement.png``
     - ``analyze_rankings.py``
     - Pairwise scorer rank scatter
