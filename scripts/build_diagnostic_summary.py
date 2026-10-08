"""Package aggregate evaluation subgroup and resource diagnostics from raw runs."""

from pathlib import Path
import csv, json, hashlib, statistics as st
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1] / "results/final"
groups = defaultdict(list)
sources = {}


def rows(p):
    sources[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return list(csv.DictReader(p.open(newline="", encoding="utf-8")))


for seed in (13, 29, 31):
    for dataset in ("longmemeval", "locomo"):
        p = (
            ROOT
            / f"raw/jobs/main/{dataset}/chalm/seed_{seed}/attempts/attempt_001/evaluation_artifacts/per_type.csv"
        )
        metric = "diagnostic_token_f1" if dataset == "longmemeval" else "locomo_category_qa_score"
        for r in rows(p):
            groups[(dataset, r["question_type"], metric)].append(100 * float(r[metric]))
    attempt = ROOT / f"raw/jobs/main/synthetic/chalm/seed_{seed}/attempts/attempt_001"
    for filename, metric, scale in [
        ("memory.csv", "bytes", 1e6),
        ("latency.csv", "end_to_end_ms", 1),
        ("latency.csv", "read_ms", 1),
    ]:
        r = [r for r in rows(attempt / "evaluation_artifacts" / filename) if r["horizon"] == "5000"]
        groups[("synthetic", "5000", metric)].append(st.mean(float(x[metric]) for x in r) / scale)
    p = attempt / "phases.jsonl"
    sources[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    phases = [json.loads(l) for l in p.open()]
    groups[("synthetic", "5000", "torch_peak_GB")].append(
        max(
            g["peak_allocated_bytes"] for r in phases if r.get("horizon") == 5000 for g in r["cuda"]
        )
        / 1e9
    )
with (ROOT / "diagnostic_summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "subgroup", "metric", "mean", "sample_sd", "training_checkpoints"])
    for key, values in sorted(groups.items()):
        assert len(values) == 3, (key, values)
        w.writerow([*key, st.mean(values), st.stdev(values), len(values)])
(ROOT / "provenance/diagnostic_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
print("Packaged", len(groups), "aggregate subgroup/resource diagnostics")
