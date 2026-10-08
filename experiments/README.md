# CHALM experiments

See the root README for setup and all output locations.

## Published studies

Original protocol: `final_study.yaml` (51 jobs, checkpoints 13/29/31). Follow-up: 102 completed jobs under `results/followup_study`. Priority hybrid retrieval runs precede uncentered-prefix and other ablations, BM25, score ordering, cue-free and newest-wins conditions. Shared baseline checkpoint loading is recorded for reproducibility, not treated as training variation.

```powershell
python experiments/verify_publication.py
python experiments/reproduce.py --plan
python experiments/reproduce.py --output results/reproduction/followup
python experiments/run_final.py --dry-run
```

Each follow-up job contains `config.yaml`, `command.json`, `complete.json`, and `runs/<run>/predictions.jsonl`, metrics, per-type rows, dataset manifest and runtime diagnostics. Summaries are `per_run.csv`, `per_seed.csv`, `means.csv`, `inventory.json`, `RESULTS.md`. Means average streams within checkpoints before sample SD. Metric names distinguish synthetic correctness, diagnostic F1/exact match and LoCoMo QA scoring.

`paired_significance.py` reproduces original comparisons; `paired_followup.py` reproduces hybrid/BM25 comparisons. They use the same paired bootstrap/sign-randomization implementation; Holm corrects aggregate rows only. Sources and hashes are saved beside `paired_tests.csv`. Analysis does not rerun models.

Completed newly generated jobs are hash-verified on restart. Interrupted jobs restart individually. Use one writer per output directory and do not edit source during a suite. Published historical receipts remain provenance; publication verification covers the normalized, distributed metadata.
