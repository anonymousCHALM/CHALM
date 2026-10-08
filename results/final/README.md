# CHALM result package

The training seeds are **13, 29, and 31**. `metrics.csv` gives each
method and horizon, with a relative path to its raw metrics file.
`chalm_seed_summary.csv` gives the mean and sample standard deviation across
the three CHALM checkpoints. Baselines without trained checkpoints have one
fixed evaluation.

`raw/jobs/` contains 51 completed job folders. Each `complete.json` lists
SHA-256 hashes for the copied artifacts. `provenance/run_manifest.json` records
the three checkpoint hashes and fixed evaluation settings; `source_map.json`
records the compact table hash. The three-seed summary reports the mean and
sample standard deviation across the three trained checkpoints.

LongMemEval token F1 and exact match are local diagnostics. LoCoMo's
category-aware QA score includes its adversarial abstention category.

From the project root, run `python experiments/verify_results.py` to check
the packaged table and any locally available raw jobs. Use
`python experiments/verify_results.py --require-raw` for a strict raw-artifact
checksum audit or `--package-only` for a fresh public checkout. Completed raw jobs are included. Intermediate checkpoints, redundant error
files and regenerable stream exports are omitted. Release artifact hashes
verify the anonymized metadata; original fingerprints preserve run provenance.
Run `python experiments/run_final.py --dry-run` to inspect
the independent reproduction schedule. A full run writes to
`results/reproduction/` and requires the models, datasets, and CUDA setup
described in the root README.

