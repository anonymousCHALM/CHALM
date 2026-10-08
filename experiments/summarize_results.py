"""Summarize completed evaluation jobs, with stream means before training-seed SD."""

from pathlib import Path
import argparse, csv, json, statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
TRAINED = {
    "chalm_score_order",
    "uncentered",
    "chalm",
    "pure_liquid",
    "difference_prefix",
    "hybrid_text",
    "centered_zero",
}


def write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "results/evaluation_study")
    a = p.parse_args()
    rows = []
    inventory = []
    for receipt in sorted((a.output / "jobs").rglob("complete.json")):
        d = json.loads(receipt.read_text())
        j = d["job"]
        if j["dataset"] == "train":
            continue
        run = receipt.parent / d["result"]
        groups = defaultdict(list)
        for line in (run / "predictions.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            metric = (
                "correct"
                if j["dataset"] == "synthetic"
                else (
                    "locomo_official_qa_score"
                    if j["dataset"] == "locomo"
                    else "token_f1_diagnostic"
                )
            )
            metrics = (
                ["correct"]
                if j["dataset"] == "synthetic"
                else ["token_f1_diagnostic", "correct"]
                + (["locomo_official_qa_score"] if j["dataset"] == "locomo" else [])
            )
            for metric in metrics:
                for qtype in ["all", r["question_type"]]:
                    groups[(r.get("horizon", ""), qtype, metric)].append(float(r[metric]))
        for (h, t, m), v in groups.items():
            rows.append(
                dict(
                    group=j["group"],
                    dataset=j["dataset"],
                    agent=j["agent"],
                    training_seed=j["training_seed"] if j["agent"] in TRAINED else "",
                    loaded_checkpoint_seed=j["training_seed"],
                    evaluation_seed=j["evaluation_seed"],
                    horizon=h,
                    question_type=t,
                    metric=m,
                    n=len(v),
                    value=statistics.mean(v),
                    predictions=str((run / "predictions.jsonl").relative_to(a.output)),
                )
            )
        inventory.append(dict(**j, run=str(run.relative_to(a.output)), artifacts=d["artifacts"]))
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / "per_run.csv", rows)
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(
            r[k]
            for k in [
                "group",
                "dataset",
                "agent",
                "training_seed",
                "horizon",
                "question_type",
                "metric",
            ]
        )
        buckets[key].append(r["value"])
    per_seed = [
        dict(
            zip(
                [
                    "group",
                    "dataset",
                    "agent",
                    "training_seed",
                    "horizon",
                    "question_type",
                    "metric",
                ],
                k,
            ),
            evaluation_replicates=len(v),
            value=statistics.mean(v),
        )
        for k, v in buckets.items()
    ]
    write(a.output / "per_seed.csv", per_seed)
    buckets = defaultdict(list)
    for r in per_seed:
        buckets[
            tuple(r[k] for k in ["group", "dataset", "agent", "horizon", "question_type", "metric"])
        ].append(r["value"])
    means = [
        dict(
            zip(["group", "dataset", "agent", "horizon", "question_type", "metric"], k),
            n_rows=len(v),
            mean=statistics.mean(v),
            sample_sd=statistics.stdev(v) if len(v) > 1 else "",
        )
        for k, v in buckets.items()
    ]
    write(a.output / "means.csv", means)
    (a.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    table = [
        "# evaluation experiment results",
        "",
        "Scores are fractions. Evaluation streams are averaged within training seeds before mean/sample SD. Shared controls are not independent training replicates. Partial suites are marked by their counts.",
        "",
        "| Group | Dataset | Method | Horizon | Metric | Rows | Mean | SD |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for r in means:
        if r["question_type"] == "all":
            table.append(
                f"| {r['group']} | {r['dataset']} | {r['agent']} | {r['horizon']} | {r['metric']} | {r['n_rows']} | {r['mean']:.4f} | {r['sample_sd']} |"
            )
    (a.output / "RESULTS.md").write_text("\n".join(table) + "\n")
    print("Wrote per_run.csv, per_seed.csv, means.csv, RESULTS.md, inventory.json to", a.output)


if __name__ == "__main__":
    main()
