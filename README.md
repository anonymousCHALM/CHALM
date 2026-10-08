# CHALM: Centered Hybrid Augmented Liquid Memory

A bounded recurrent memory and exact-text sidecar with dense/BM25 fusion for long-horizon conversational question answering. This repository contains the CHALM implementation, three trained checkpoints, complete saved predictions, result tables and reproducible protocols.

## Start here

| Resource | Contents |
|---|---|
| [Protocol guide](docs/PROTOCOL.md) | Protocol, metrics, verification and interpretation |
| [Original scores](results/final/metrics.csv) | Original study: training seeds 13/29/31 and shared baselines |
| [Three-seed summary](results/final/chalm_seed_summary.csv) | Means and sample SDs |
| [Follow-up results](results/followup_study/RESULTS.md) | Hybrid/BM25 retrieval, prefix ablations, ordering, cue-free and newest-wins |
| [Original paired tests](results/evaluation_study/significance/RESULTS.md) | CHALM versus Growing RAG and local Graph RAG |
| [Hybrid/BM25 paired tests](results/followup_study/significance/RESULTS.md) | All three checkpoints, per-seed and aggregate inference |
| [Hybrid LoCoMo QA](results/followup_study/HYBRID_LOCOMO_QA.md) | Full-history and FIFO QA scores |

## Repository structure

```text
CHALM/
|-- configs/                      # Model and evaluation settings
|-- data/                         # Dataset acquisition instructions
|-- docs/PROTOCOL.md               # Metric definitions and interpretation
|-- experiments/                  # Replay, verification and paired tests
|-- scripts/                      # Training and evaluation entry points
|-- src/liquid_memory_agents/      # CHALM, reader, encoders and baselines
|-- tests/unit/                   # CPU correctness and export checks
|-- results/
|   |-- final/                    # Original 51-job study and checkpoints
|   |-- evaluation_study/         # Initial hybrid runs and paired tests
|   |-- followup_study/           # Complete 102-job evaluation suite
|   |-- publication_manifest.json # SHA256 inventory of shipped results
|-- pyproject.toml
|-- requirements.txt
```

Result paths use neutral study names consistently across provenance and analysis scripts. Each run includes predictions, metrics, question-type scores, config, dataset manifest and runtime/memory diagnostics. Receipts identify seeds and checksums. Publication metadata normalizes original workstation paths and omits references to removed rendering tools. Saved scores, prediction contents and trained checkpoint parameters are unchanged. Checkpoint metadata uses portable paths.

## Installation

Python 3.12 was used for the recorded runs. GPU replay requires a CUDA-capable device. The Windows requirements select the recorded CUDA 12.8 PyTorch family; use a suitable CUDA-enabled PyTorch build for your system.

```powershell
git lfs install
git clone https://github.com/anonymousCHALM/CHALM.git
cd CHALM
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest tests/unit -q
```

Checkpoints and JSONL prediction/resource files use Git LFS. Run `git lfs pull` if they contain pointer text. The three final checkpoints total about 606 MB; saved predictions total about 728 MB. Downloaded datasets, intermediate epoch checkpoints, duplicate error-only predictions and incomplete attempts are excluded. [The omission list](results/publication_omissions.json) records excluded intermediate artifacts. Original local archives are retained.

Download benchmark inputs:

```powershell
python scripts/download_data.py --dataset longmemeval
python scripts/download_data.py --dataset locomo
```

## Verify and recompute saved results (no GPU runs)

```powershell
python experiments/verify_publication.py
python experiments/verify_results.py --package-only
python experiments/summarize_results.py --output results/followup_study
python experiments/paired_significance.py
python experiments/paired_followup.py
```

The publication verifier checks shipped file SHA256 hashes, three checkpoint paths and completed-study counts. Historical receipt lists also describe intentionally omitted intermediate files, so use the publication verifier for the distributed artifact set. Recomputed summaries are new analysis outputs and can change publication hashes; verify the downloaded checkout before regenerating them.

Both significance CSVs report per-seed and aggregate differences, paired percentile-bootstrap 95% CIs and two-sided paired sign-randomization raw p-values with 10,000 draws. **Holm correction covers aggregate comparisons only**; seed-row adjusted p is blank. LoCoMo resamples conversations. Aggregate inference averages checkpoints within each question and is conditional on those fitted checkpoints. Scores and differences in CSVs are fractions; multiply by 100 for percentages or percentage points.

## Reproduce evaluations (GPU required)

Replay the 102 follow-up evaluations with the included checkpoints into a fresh output directory:

```powershell
python experiments/reproduce.py --plan
python experiments/reproduce.py --output results/reproduction/followup
```

Select a subset with `--group priority`, `ablations`, `retrieval`, `ordering`, `cue-free`, `newest-wins-original`, or `newest-wins-cue_free`. Completed new jobs are hash-verified and skipped on resume; interrupted individual evaluations restart. Do not edit code/configs during a suite or run two writers against the same output directory. A source change requires a fresh output directory. Archived receipts are provenance, not resume targets for edited publication source.

Reproduce the original 51-job study, including training and all original baselines:

```powershell
python experiments/run_final.py --dry-run
python experiments/run_final.py
```

This uses `experiments/final_study.yaml` and writes to `results/reproduction/`, leaving published scores intact. To retrain only one checkpoint with its recorded settings:

```powershell
python experiments/reproduce.py --train-seed 13 --output results/reproduction/retrain
```

Repeat for seeds 29 and 31. Newly trained weights remain separate from the published checkpoints. For direct evaluation, use `scripts/run_synthetic.py` or `scripts/run_real_benchmark.py` with `--config` and `--checkpoint`; the study runners are preferred because they apply the recorded sample counts, selection and seeds.

## Protocol and metrics

- Training checkpoints: **13, 29, 31**. Standard synthetic seed: **29001**, horizons **100/500/1,000/5,000**, with **300/225/150/150** questions.
- Cue-free synthetic seeds: **11/13/17**, 50 questions each at 1,000 and 5,000 turns. The generator is included independently in this repository.
- Reader: **Qwen/Qwen2.5-3B-Instruct**; encoder: **sentence-transformers/all-MiniLM-L6-v2**. Reader and encoder are frozen.
- Memory: eight 256-dimensional recurrent slots, 512 text entries, 1,024-byte text cap, top-eight reads, redundancy threshold 0.82. CHALM fuses dense and BM25 ranks with RRF, then orders selected records by timestamp.
- Centered prefix: **2p - p0**. Ablations include text-only, centered-zero, uncentered **p**, difference **p - p0**, Pure Liquid and Lexical-only where specified by the study.
- Hybrid full-history and FIFO baselines use the same dense/BM25 read without a prefix. BM25-only tests cover full history and the adaptive cache. Ordering-off presents fused results by score.
- Newest-wins is evaluation only: at full capacity, similarity >= 0.82 replaces the closest record without a correction cue. The original cue-gated rule remains the default.
- LongMemEval: questions **50-499**, 450 total; diagnostic token F1/exact match, not the upstream LLM-judged score.
- LoCoMo: **1,986 questions**, ten conversations, local category-aware QA scoring plus diagnostic F1/exact match. Released textual image captions are used.
- Means +/- sample SD use training-checkpoint replicates after averaging any evaluation streams within each checkpoint. Shared baselines have no training-seed SD.
- Logical user-memory payload is **1,318,912 bytes**; it is not GPU VRAM. Raw phase/resource/latency files provide the recorded resource measurements.

Original hardware was an RTX 5080. Library or hardware changes can alter generated answers; compare the recorded settings and metric definitions as well as the scores.

## Anonymity audit

```powershell
python scripts/anonymize_repo.py
```

The audit checks source, result metadata and checkpoint metadata. Original benchmark text can contain ordinary words or place names matching audit terms; questions, references and predictions remain unchanged. Environment files record relevant dependency versions without personal editable-install URLs or workstation paths.
