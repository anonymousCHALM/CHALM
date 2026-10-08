"""Strictly paired bootstrap CIs and paired randomization tests on saved predictions.

Training checkpoints are averaged per question before aggregate inference.
LoCoMo resamples conversations, preserving the question-weighted estimand.
"""

from pathlib import Path
from collections import defaultdict
import argparse, csv, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED = {"lexical_only", "rag", "linearrag_local"}


def inference(deltas, ids, dataset, samples, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(deltas, dtype=float)
    clusters = defaultdict(list)
    for i, k in enumerate(ids):
        clusters[k.rsplit("-qa-", 1)[0] if dataset == "locomo" else k].append(i)
    values = [x[v] for v in clusters.values()]
    sums = np.array([v.sum() for v in values])
    sizes = np.array([len(v) for v in values])
    n = len(values)
    draws = rng.integers(0, n, (samples, n))
    boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    signs = rng.choice([-1, 1], size=(samples, n))
    null = (signs * sums).sum(axis=1) / len(x)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = (1 + np.count_nonzero(np.abs(null) >= abs(x.mean()) - 1e-12)) / (samples + 1)
    return dict(
        n_questions=len(x),
        n_clusters=n,
        delta=float(x.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        p_raw=float(p),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args()
    sources = json.loads((ROOT / "experiments/paired_sources.json").read_text())
    data = defaultdict(dict)
    hashes = {}
    for spec in sources:
        path = ROOT / spec["path"]
        hashes[spec["path"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            agent = r["agent"]
            if agent not in spec["agents"]:
                continue
            seed = None if agent in SHARED else spec["training_seed"]
            dataset = spec["dataset"]
            key = (spec["suite"], dataset, r.get("horizon", ""), agent, seed)
            metric = (
                "correct"
                if dataset == "synthetic"
                else ("locomo_official_qa_score" if dataset == "locomo" else "token_f1_diagnostic")
            )
            item = (float(r[metric]), r["query"], json.dumps(r["reference"], sort_keys=True))
            old = data[key].get(r["example_id"])
            if old is not None and old != item:
                raise ValueError(f"Conflicting duplicate {key} {r['example_id']}")
            data[key][r["example_id"]] = item
    comparisons = [("main", "chalm", "rag"), ("main", "chalm", "linearrag_local")]
    rows = []
    for suite, left, right in comparisons:
        settings = sorted({(d, str(h)) for s, d, h, ag, seed in data if s == suite and ag == left})
        for dataset, hs in settings:
            h = int(hs) if hs else ""
            seeds = sorted(
                seed for s, d, hh, ag, seed in data if (s, d, hh, ag) == (suite, dataset, h, left)
            )
            aligned = []
            ids0 = None
            for seed in seeds:
                l = data[(suite, dataset, h, left, seed)]
                rr = data.get((suite, dataset, h, right, None if right in SHARED else seed))
                if rr is None:
                    raise ValueError(f"Missing comparator {right} {dataset} {h} {seed}")
                if set(l) != set(rr):
                    raise ValueError(f"Unmatched IDs: {dataset} {h} {seed}: {len(l)} vs {len(rr)}")
                ids = sorted(l)
                if ids0 is not None and ids != ids0:
                    raise ValueError("Checkpoint question sets differ")
                ids0 = ids
                for k in ids:
                    if l[k][1:] != rr[k][1:]:
                        raise ValueError(f"Query/reference mismatch: {k}")
                delta = np.array([l[k][0] - rr[k][0] for k in ids])
                aligned.append(delta)
                stats = inference(delta, ids, dataset, a.samples, a.seed)
                rows.append(
                    dict(
                        suite=suite,
                        dataset=dataset,
                        horizon=h,
                        left=left,
                        right=right,
                        training_seed=seed,
                        **stats,
                    )
                )
            stats = inference(np.mean(aligned, axis=0), ids0, dataset, a.samples, a.seed)
            rows.append(
                dict(
                    suite=suite,
                    dataset=dataset,
                    horizon=h,
                    left=left,
                    right=right,
                    training_seed="mean",
                    **stats,
                )
            )
    # Only aggregate comparisons belong to the Holm family.
    for row in rows:
        row["p_holm"] = ""
    order = sorted(
        [i for i, r in enumerate(rows) if r["training_seed"] == "mean"],
        key=lambda i: rows[i]["p_raw"],
    )
    prev = 0
    for rank, i in enumerate(order):
        prev = max(prev, min(1, rows[i]["p_raw"] * (len(order) - rank)))
        rows[i]["p_holm"] = prev
    out = ROOT / "results/evaluation_study/significance"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "paired_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                samples=a.samples,
                seed=a.seed,
                sources=hashes,
                method="Paired percentile bootstrap CI; two-sided paired cluster sign randomization p; Holm across aggregate comparisons only. Aggregate averages checkpoints within question; conditional on fitted checkpoints. LoCoMo conversation clusters.",
            ),
            indent=2,
        )
    )
    md = [
        "# Paired significance results",
        "",
        "Differences and CIs are fractions (multiply by 100 for percentage points). CIs are pointwise; Holm correction applies only to aggregate comparisons; per-seed adjusted p is blank. Aggregate inference is conditional on the fitted checkpoints, not a population claim over training seeds. LoCoMo has only ten conversation clusters.",
        "",
        "| Dataset | Horizon | Comparison | Seed | Delta | 95% CI | Raw p | Holm p |",
        "|---|---:|---|---|---:|---|---:|---:|",
    ]
    for v in rows:
        md.append(
            f"| {v['dataset']} | {v['horizon']} | {v['left']} vs {v['right']} | {v['training_seed']} | {v['delta']:.4f} | [{v['ci_low']:.4f}, {v['ci_high']:.4f}] | {v['p_raw']:.4g} | {v['p_holm']} |"
        )
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print(out, len(rows), "comparisons")


if __name__ == "__main__":
    main()
