"""Aggregate completed jobs only; paired intervals are conditional on checkpoints."""

import argparse
from collections import defaultdict
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def jsonlines(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def device_sample_summary(samples):
    """Summarize finite device readings; unsupported measurements stay missing."""
    devices = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        raw = sample.get("whole_device_samples_index_mib_util_percent_watts", "")
        for fields in csv.reader(raw.splitlines()):
            if len(fields) != 4:
                continue
            device = fields[0].strip()
            for label, value in zip(
                ("memory_mib", "utilization_percent", "power_watts"), fields[1:]
            ):
                readings = devices[device][label]
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                if math.isfinite(numeric):
                    readings.append(numeric)
    rows = []
    for device, values in devices.items():
        row = {"phase": "whole_device_sampled", "device": device}
        for label, readings in values.items():
            row[f"{label}_samples"] = len(readings)
            if readings:
                row[f"{label}_mean"] = float(np.mean(readings))
                row[f"{label}_max"] = max(readings)
        rows.append(row)
    return rows


def bleu1(reference, prediction):
    from nltk.translate.bleu_score import sentence_bleu

    tokens = lambda s: re.findall(r"\w+", s.casefold())
    reference, prediction = tokens(reference), tokens(prediction)
    return (
        float(sentence_bleu([reference], prediction, weights=(1.0,)))
        if reference and prediction
        else 0.0
    )


def paired(left, right, dataset, samples=10000):
    a = {r["example_id"]: r for r in left}
    b = {r["example_id"]: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("Paired predictions must have unique, identical question IDs")
    field = {
        "synthetic": "correct",
        "longmemeval": "token_f1_diagnostic",
        "locomo": "locomo_official_qa_score",
    }[dataset]
    groups = defaultdict(list)
    for key in a:
        if a[key].get("reference") != b[key].get("reference") or a[key].get("query") != b[key].get(
            "query"
        ):
            raise ValueError(f"Question mismatch: {key}")
        group = key.rsplit("-qa-", 1)[0] if dataset == "locomo" else key
        groups[group].append(float(a[key][field]) - float(b[key][field]))
    totals = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(groups), size=(samples, len(groups)))
    deltas = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "paired_questions": len(a),
        "independent_units": len(groups),
        "delta": float(totals.sum() / counts.sum()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "interval": "conversation-cluster bootstrap"
        if dataset == "locomo"
        else "paired-example bootstrap",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=Path("results/study"))
    args = parser.parse_args()
    suite = args.suite
    report = suite / "reports"
    report.mkdir(parents=True, exist_ok=True)
    summaries, resource_rows, per_type, training, parameters = [], [], [], [], []
    cost_rows = []
    auxiliary_rows = []
    lexical_rows = []
    records = {}
    fingerprints = {}
    for receipt_path in sorted((suite / "jobs").rglob("complete.json")):
        receipt = json.loads(receipt_path.read_text())
        job = receipt["job"]
        root = receipt_path.parent
        result = root / receipt["result"]
        labels = {
            "job": job["key"],
            "dataset": job["dataset"],
            "training_seed": job.get("training_seed", job["seed"]),
            "experiment_group": job.get(
                "group", "training" if job["dataset"] == "train" else "main"
            ),
            "capacity": job.get("capacity", 512),
        }
        manifest = json.loads((result / "dataset_manifest.json").read_text())
        fingerprint = (
            manifest.get("content_sha256"),
            manifest.get("source_sha256"),
            manifest.get("sha256"),
        )
        previous = fingerprints.setdefault(job["dataset"], fingerprint)
        if previous != fingerprint:
            raise ValueError(f"Dataset/split changed across jobs: {job['key']}")
        if job["dataset"] == "train":
            for row in jsonlines(result / "validation_epochs.jsonl"):
                training.append({**labels, **row})
        else:
            with (result / "metrics.csv").open(newline="") as handle:
                summaries.extend({**labels, **r} for r in csv.DictReader(handle))
            with (result / "per_type.csv").open(newline="") as handle:
                per_type.extend({**labels, **r} for r in csv.DictReader(handle))
            records[job["key"]] = (job, jsonlines(result / "predictions.jsonl"))
            diagnostic = records[job["key"]][1]
            for horizon in {r.get("horizon", "") for r in diagnostic}:
                selected = [
                    r
                    for r in diagnostic
                    if r.get("horizon", "") == horizon
                    and r.get("question_type") != "category-5"
                    and isinstance(r.get("reference"), str)
                    and isinstance(r.get("prediction"), str)
                ]
                if selected:
                    lexical_rows.append(
                        {
                            **labels,
                            "horizon": horizon,
                            "examples": len(selected),
                            "mean_sentence_bleu1_local": float(
                                np.mean([bleu1(r["reference"], r["prediction"]) for r in selected])
                            ),
                            "protocol": "lowercase word tokens; single reference; excludes LoCoMo category 5; not upstream BLEU",
                        }
                    )
            for filename in ("memory.csv", "latency.csv"):
                if not (result / filename).exists():
                    continue
                with (result / filename).open(newline="") as handle:
                    values = list(csv.DictReader(handle))
                for horizon in {r.get("horizon", "") for r in values}:
                    selected = [r for r in values if r.get("horizon", "") == horizon]
                    for field in (
                        "bytes",
                        "update_ms",
                        "read_ms",
                        "generation_ms",
                        "end_to_end_ms",
                    ):
                        if not selected or field not in selected[0]:
                            continue
                        data = np.array([float(r[field]) for r in selected])
                        cost_rows.append(
                            {
                                **labels,
                                "horizon": horizon,
                                "metric": field,
                                "mean": float(data.mean()),
                                "max": float(data.max()),
                                "p50": float(np.quantile(data, 0.5)),
                                "p95": float(np.quantile(data, 0.95)),
                                "p99": float(np.quantile(data, 0.99)),
                            }
                        )
        attempt = root / receipt["attempt"]
        auxiliary = attempt / "auxiliary_calls.jsonl"
        if auxiliary.exists():
            calls = jsonlines(auxiliary)
            for stage in sorted({r["stage"] for r in calls}):
                selected = [r for r in calls if r["stage"] == stage]
                auxiliary_rows.append(
                    {
                        **labels,
                        "stage": stage,
                        "calls": len(selected),
                        "input_tokens": sum(
                            r["input_tokens"] for r in selected if r["status"] == "generated"
                        ),
                        "rejected_input_tokens": sum(
                            r["input_tokens"]
                            for r in selected
                            if r["status"] == "input_over_budget"
                        ),
                        "output_tokens": sum(r["output_tokens"] for r in selected),
                        "generation_seconds": sum(r["wall_seconds"] for r in selected),
                        "over_budget_calls": sum(
                            r["status"] == "input_over_budget" for r in selected
                        ),
                        "token_limit_calls": sum(r.get("hit_token_limit", False) for r in selected),
                    }
                )
        for component, values in json.loads((attempt / "parameters.json").read_text()).items():
            parameters.append({**labels, "component": component, **values})
        groups = defaultdict(list)
        for row in jsonlines(attempt / "phases.jsonl"):
            groups[(row["phase"], row.get("horizon"))].append(row)
        for (phase, horizon), rows in groups.items():
            seconds = np.array([r["wall_seconds"] for r in rows])
            gpu = [g for r in rows for g in r["cuda"]]
            resource_rows.append(
                {
                    **labels,
                    "phase": phase,
                    "horizon": horizon,
                    "calls": len(rows),
                    "wall_seconds_sum": float(seconds.sum()),
                    "wall_ms_mean": float(seconds.mean() * 1000),
                    "wall_ms_p50": float(np.quantile(seconds, 0.5) * 1000),
                    "wall_ms_p95": float(np.quantile(seconds, 0.95) * 1000),
                    "wall_ms_p99": float(np.quantile(seconds, 0.99) * 1000),
                    "items_per_second": sum(r.get("items", 1) for r in rows)
                    / max(float(seconds.sum()), 1e-12),
                    "output_tokens_per_phase_second": sum(
                        sum(r.get("output_tokens_including_eos", [])) for r in rows
                    )
                    / max(float(seconds.sum()), 1e-12),
                    "cpu_seconds_sum": sum(r["cpu_seconds"] for r in rows),
                    "text_input_tokens_sum": sum(sum(r.get("text_input_tokens", [])) for r in rows),
                    "output_tokens_sum": sum(
                        sum(r.get("output_tokens_including_eos", [])) for r in rows
                    ),
                    "peak_torch_allocated_bytes": max(
                        (g["peak_allocated_bytes"] for g in gpu), default=0
                    ),
                    "peak_torch_reserved_bytes": max(
                        (g["peak_reserved_bytes"] for g in gpu), default=0
                    ),
                    "peak_incremental_allocated_bytes": max(
                        (g["incremental_peak_allocated_bytes"] for g in gpu), default=0
                    ),
                }
            )
        samples = jsonlines(attempt / "resource_samples.jsonl")
        resource_rows.append(
            {
                **labels,
                "phase": "whole_job_sampled_rss",
                "calls": len(samples),
                "sampled_peak_rss_bytes": max(
                    (r.get("process_rss_bytes", 0) for r in samples), default=0
                ),
            }
        )
        resource_rows.extend({**labels, **row} for row in device_sample_summary(samples))
    comparisons = []
    for key, (job, rows) in records.items():
        if job["agent"] != "chalm" or "capacity" in job:
            continue
        dataset, seed = job["dataset"], job["seed"]
        references = [
            f"{dataset}_{name}"
            for name in (
                "rag",
                "bounded_rag",
                "lexical_only",
                "centered_zero",
                "hybrid_text",
                "linearrag_local",
            )
        ]
        references += [f"{dataset}_pure_liquid_{seed}"]
        horizons = sorted({r.get("horizon", 0) for r in rows})
        for baseline in references:
            if baseline not in records:
                continue
            for horizon in horizons:
                a = [r for r in rows if r.get("horizon", 0) == horizon]
                b = [r for r in records[baseline][1] if r.get("horizon", 0) == horizon]
                comparisons.append(
                    {
                        "dataset": dataset,
                        "training_seed": seed,
                        "horizon": horizon,
                        "baseline": baseline,
                        **paired(a, b, dataset),
                    }
                )
    write_csv(report / "metrics_by_job.csv", summaries)
    write_csv(
        report / "main_comparisons.csv", [r for r in summaries if r["experiment_group"] == "main"]
    )
    write_csv(
        report / "ablation_results.csv",
        [r for r in summaries if r["experiment_group"].startswith("ablations/")],
    )
    write_csv(report / "per_type.csv", per_type)
    write_csv(report / "resources.csv", resource_rows)
    write_csv(report / "validation_epochs.csv", training)
    write_csv(report / "parameters.csv", parameters)
    write_csv(report / "memory_latency.csv", cost_rows)
    write_csv(report / "auxiliary_calls.csv", auxiliary_rows)
    write_csv(report / "lexical_diagnostics.csv", lexical_rows)
    write_csv(report / "paired_comparisons.csv", comparisons)
    aggregates = defaultdict(list)
    for row in summaries:
        metric = {
            "synthetic": "accuracy",
            "longmemeval": "diagnostic_token_f1",
            "locomo": "locomo_official_qa_score",
        }[row["dataset"]]
        if row.get(metric) not in (None, ""):
            aggregates[
                (row["dataset"], row["agent"], row.get("horizon", ""), row["capacity"], metric)
            ].append(float(row[metric]))
    seed_rows = [
        {
            "dataset": k[0],
            "agent": k[1],
            "horizon": k[2],
            "capacity": k[3],
            "metric": k[4],
            "runs": len(v),
            "mean": float(np.mean(v)),
            "sample_std": float(np.std(v, ddof=1)) if len(v) > 1 else "",
        }
        for k, v in aggregates.items()
    ]
    write_csv(report / "training_seed_summary.csv", seed_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset in ("synthetic", "longmemeval", "locomo"):
        selected = [r for r in seed_rows if r["dataset"] == dataset and r["capacity"] == 512]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if dataset == "synthetic":
            for name in sorted({r["agent"] for r in selected}):
                values = sorted(
                    [r for r in selected if r["agent"] == name], key=lambda r: int(r["horizon"])
                )
                ax.plot(
                    [int(r["horizon"]) for r in values],
                    [r["mean"] for r in values],
                    marker="o",
                    label=name,
                )
            ax.set_xlabel("Conversation turns")
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.bar([r["agent"] for r in selected], [r["mean"] for r in selected])
            ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel(selected[0]["metric"])
        fig.tight_layout()
        fig.savefig(report / f"{dataset}.pdf")
        fig.savefig(report / f"{dataset}.png", dpi=180)
        plt.close(fig)
    index = suite / "run_index.json"
    status = "Coverage unknown: no scheduled job index."
    if index.exists():
        scheduled = json.loads(index.read_text())
        completed = sum(
            (suite / "jobs" / job["folder"] / "complete.json").exists() for job in scheduled
        )
        state = "COMPLETE" if completed == len(scheduled) else "PARTIAL"
        status = f"{state}: {completed}/{len(scheduled)} scheduled jobs completed."
    (report / "README.txt").write_text(
        f"{status} Baselines are evaluated once, not three independent replicates.\n"
        "Training-seed summaries condition on one fixed dataset. CIs are descriptive, unadjusted, conditional on checkpoints.\n"
        "LoCoMo has ten independent conversations. LongMemEval scores are local diagnostics, not official judge accuracy.\n"
        "Torch peaks are per-device phase maxima, not persistent memory or whole-process VRAM.\n"
        "Device power/utilization samples include other processes; RSS is sampled, not an exact peak.\n",
        encoding="utf-8",
    )
    print(f"Reports: {report}")


if __name__ == "__main__":
    main()
