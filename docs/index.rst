supervised_phenotype_scoring
============================

A local web application for **anonymised, supervised scoring of birdsong phenotype traits**
(stereotypy, repeat propensity, …) by human raters.

Songs are extracted from audio recordings, stored in an HDF5 batch, served via a
drag-and-drop ranking interface with interactive Plotly comparison panels, and
aggregated into per-bird Elo scores with inter-rater reliability statistics.

See the project README on GitHub for installation, configuration, and step-by-step usage.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/family_spec_generation
   api/prepare_batch
   api/export_batch
   api/prescreen_app
   api/run_pipeline
   api/ranking_app
   api/analyze_rankings
   api/printout_generator
   api/upload_batch
   api/make_test_batch

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
