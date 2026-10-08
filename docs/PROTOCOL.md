# CHALM evaluation guide

The root README contains installation, directory structure and reproduction commands.

## Result editions

- `results/final`: 51-job original study, three trained checkpoints, synthetic/released-data baselines, prefix controls, capacity and resource diagnostics.
- `results/evaluation_study`: two initial hybrid runs and original paired comparisons.
- `results/followup_study`: all 102 completed evaluations, dense/BM25 hybrid full history and FIFO, all-checkpoint ablations, BM25, score ordering, cue-free corrections and newest-wins.

Run `python experiments/verify_publication.py` after `git lfs pull`. The publication manifest checks shipped bytes. Intermediate weights/error-only copies and incomplete attempts are excluded; original numerical outputs and final weights remain unchanged. Metadata paths are normalized for a standalone checkout.

## Scores and paired inference

Synthetic accuracy uses the recorded answer matching rule. LongMemEval token F1 and exact match are local diagnostics on 450 held-out questions, not the official judge score. LoCoMo QA scoring covers 1,986 questions and five categories in ten conversations. Per-type scores are in run `per_type.csv` and summary CSVs.

The original significance file compares CHALM with Growing RAG and the local LinearRAG baseline. Follow-up tests compare CHALM with hybrid full-history at 5,000 turns and both released benchmarks, and BM25-cache on both released benchmarks. All requested comparisons use seeds 13/29/31. Paired scripts reject mismatched question IDs, queries, references and conflicting duplicates. LoCoMo resamples conversation clusters; other datasets resample questions. Holm applies only to aggregate comparisons. Inference is conditional on the three fitted checkpoints.

## Reproduction

`experiments/reproduce.py` replays recorded follow-up jobs with the shipped weights. `experiments/run_final.py` retrains and evaluates the original fixed protocol in a separate directory. Per-seed retraining is available through `reproduce.py --train-seed`. Use fresh output directories for changed source/configs; historical source fingerprints describe the original execution.

Memory bytes describe bounded user-memory payload, not total process/GPU memory. Resource samples, phases and environment files accompany original runs. Baseline names identify the implementation actually evaluated; local LinearRAG is not a claim of equivalence to every Graph RAG implementation.
