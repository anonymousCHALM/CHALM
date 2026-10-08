"""Paired follow-up inference from saved predictions; no model evaluation."""

from collections import defaultdict
import csv, json, hashlib
import numpy as np
from paired_significance import ROOT, inference


def main():
    data = defaultdict(dict)
    sources = {}
    rows = []
    missing = []

    def load(path, dataset, agent, seed, label, horizons=None):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        metric = (
            "correct"
            if dataset == "synthetic"
            else ("locomo_official_qa_score" if dataset == "locomo" else "token_f1_diagnostic")
        )
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["agent"] != agent or (horizons and r.get("horizon") not in horizons):
                continue
            key = (label, dataset, r.get("horizon", ""), seed)
            item = (float(r[metric]), r["query"], json.dumps(r["reference"], sort_keys=True))
            old = data[key].get(r["example_id"])
            if old is not None and old != item:
                raise ValueError(f"Conflicting duplicate {key}")
            data[key][r["example_id"]] = item

    for suite in ["evaluation_study", "followup_study"]:
        for receipt in sorted((ROOT / "results" / suite / "jobs").rglob("complete.json")):
            d = json.loads(receipt.read_text())
            j = d["job"]
            a = j["agent"]
            g = j["group"]
            label = None
            seed = j["training_seed"]
            if j["dataset"] == "train":
                continue
            if suite == "followup_study" and g == "ablations" and a == "chalm":
                label = "chalm"
            if suite == "followup_study" and a in ["hybrid_growing_rag", "bm25_cache"]:
                label = a
                seed = None
            if label:
                load(
                    receipt.parent / d["result"] / "predictions.jsonl", j["dataset"], a, seed, label
                )
    comparisons = [
        (
            "chalm",
            "hybrid_growing_rag",
            [("synthetic", 5000), ("longmemeval", ""), ("locomo", "")],
            True,
        ),
        ("chalm", "bm25_cache", [("longmemeval", ""), ("locomo", "")], True),
    ]
    expected = [13, 29, 31]
    for left, right, settings, shared in comparisons:
        for dataset, h in settings:
            aligned = []
            ids0 = None
            used = []
            for seed in expected:
                l = data.get((left, dataset, h, seed))
                r = data.get((right, dataset, h, None if shared else seed))
                if l is None or r is None:
                    missing.append(
                        dict(
                            left=left,
                            right=right,
                            dataset=dataset,
                            horizon=h,
                            training_seed=seed,
                            reason="Saved prediction set missing",
                        )
                    )
                    continue
                if set(l) != set(r):
                    raise ValueError(
                        f"Unmatched IDs {left} {right} {dataset} {h} {seed}: {len(l)} vs {len(r)}"
                    )
                ids = sorted(l)
                if ids0 is not None and ids != ids0:
                    raise ValueError("Checkpoint question sets differ")
                ids0 = ids
                for k in ids:
                    if l[k][1:] != r[k][1:]:
                        raise ValueError(f"Query/reference mismatch {k}")
                delta = np.array([l[k][0] - r[k][0] for k in ids])
                aligned.append(delta)
                used.append(seed)
                metric = (
                    "correct"
                    if dataset == "synthetic"
                    else (
                        "locomo_official_qa_score" if dataset == "locomo" else "token_f1_diagnostic"
                    )
                )
                rows.append(
                    dict(
                        dataset=dataset,
                        horizon=h,
                        left=left,
                        right=right,
                        training_seed=seed,
                        metric=metric,
                        checkpoints=str(seed),
                        complete=True,
                        **inference(delta, ids, dataset, 10000, 20261001),
                    )
                )
            if aligned:
                rows.append(
                    dict(
                        dataset=dataset,
                        horizon=h,
                        left=left,
                        right=right,
                        training_seed="mean",
                        metric=metric,
                        checkpoints=",".join(map(str, used)),
                        complete=used == expected,
                        **inference(np.mean(aligned, axis=0), ids0, dataset, 10000, 20261001),
                    )
                )
    for r in rows:
        r["p_holm"] = ""
    order = sorted(
        [i for i, r in enumerate(rows) if r["training_seed"] == "mean"],
        key=lambda i: rows[i]["p_raw"],
    )
    prev = 0
    for rank, i in enumerate(order):
        prev = max(prev, min(1, rows[i]["p_raw"] * (len(order) - rank)))
        rows[i]["p_holm"] = prev
    out = ROOT / "results/followup_study/significance"
    out.mkdir(exist_ok=True)
    with (out / "paired_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                samples=10000,
                seed=20261001,
                sources=sources,
                missing=missing,
                method="Same paired percentile bootstrap and paired cluster sign randomization; LoCoMo clusters conversations. Average checkpoints within question; Holm across aggregate rows only. Partial aggregates explicitly flagged.",
            ),
            indent=2,
        )
    )
    md = [
        "# Follow-up paired significance",
        "",
        "Differences/CI are fractions. Holm applies only to aggregate comparisons; per-seed adjusted p is blank. Aggregates are conditional on fitted checkpoints. LongMemEval uses diagnostic token F1; LoCoMo uses QA score.",
        "",
        "| Dataset | Horizon | Comparison | Seed | Checkpoints | Complete | Difference | 95% CI | Raw p | Holm p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['dataset']} | {r['horizon']} | {r['left']} vs {r['right']} | {r['training_seed']} | {r['checkpoints']} | {r['complete']} | {r['delta']:.5f} | [{r['ci_low']:.5f}, {r['ci_high']:.5f}] | {r['p_raw']:.5g} | {r['p_holm']} |"
        )
    if missing:
        md += ["", "## Missing saved predictions", ""] + [str(x) for x in missing]
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("CHALM", len(rows), "rows;", len(missing), "missing checkpoint comparisons")


if __name__ == "__main__":
    main()
