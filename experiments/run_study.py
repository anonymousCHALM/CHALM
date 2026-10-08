"""Plan/run resumable evaluation experiments; never changes saved result packages."""

from pathlib import Path
import argparse, hashlib, json, subprocess, sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from liquid_memory_agents.utils.config import load_config


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def jobs(protocol, group):
    seeds = protocol["training_seeds"]
    out = []

    def add(g, d, a, s, e=29001, style="original", h=None):
        out.append(
            dict(
                group=g,
                dataset=d,
                agent=a,
                training_seed=s,
                evaluation_seed=e,
                correction_style=style,
                horizons=h,
            )
        )

    for d in ["synthetic", "longmemeval", "locomo"]:
        for a in ["hybrid_growing_rag", "hybrid_fifo_rag"]:
            add("priority", d, a, seeds[0])
        for a in ["bm25_cache", "bm25_growing_rag"]:
            add("retrieval", d, a, seeds[0])
        for s in seeds:
            for a in [
                "chalm",
                "hybrid_text",
                "centered_zero",
                "uncentered",
                "difference_prefix",
                "lexical_only",
            ]:
                add("ablations", d, a, s)
        add("ordering", d, "chalm_score_order", seeds[0])
    priority = {
        "priority": 0,
        "cue-free": 0,
        "ordering": 1,
        "retrieval": 2,
        "ablations": 3,
        "extra-seeds": 4,
    }
    return sorted(
        [j for j in out if j["group"] == group or (group == "all" and j["group"] != "extra-seeds")],
        key=lambda j: priority[j["group"]],
    )


def key(j):
    return f"{j['group']}/{j['dataset']}/{j['agent']}/checkpoint_{j['training_seed']}/eval_{j['evaluation_seed']}"


def run(j, p, suite, checkpoints):
    legacy = suite / "jobs" / key(j)
    # Retain completed original jobs; compact paths avoid Windows MAX_PATH.
    job_id = hashlib.sha256(key(j).encode()).hexdigest()[:12]
    folder = legacy if (legacy / "complete.json").is_file() else suite / "jobs" / job_id
    done = folder / "complete.json"
    config = load_config(
        ROOT / "configs" / ("base.yaml" if j["dataset"] == "train" else j["dataset"] + ".yaml")
    )
    config["run"].update(seed=j["evaluation_seed"], output_root=str(folder / "runs"))
    config["protocol"] = {**j, "project": p["project"], "version": 1}
    if j["dataset"] == "synthetic":
        config["dataset"].pop("test_seeds", None)
        config["dataset"]["correction_style"] = j["correction_style"]
        config["dataset"]["examples_by_horizon"] = {100: 300, 500: 225, 1000: 150, 5000: 150}
        if j["horizons"]:
            config["dataset"]["turns"] = j["horizons"]
    elif j["dataset"] != "train":
        config["dataset"]["input"] = str(ROOT / config["dataset"]["input"])
    config["liquid"].setdefault("lexical_memory", {})["replacement_rule"] = j.get(
        "replacement_rule", "cue_gated"
    )
    if j["dataset"] == "synthetic" and j["correction_style"] == "cue_free":
        config["dataset"]["examples_by_horizon"] = {1000: 50, 5000: 50}
    config.setdefault("evaluation", {}).update(agents=[j["agent"]], qa_batch_size=1)
    checkpoint = checkpoints.get(j["training_seed"])
    if j["dataset"] != "train" and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(
            f"Missing checkpoint for seed {j['training_seed']}; see experiments/evaluation_study.yaml"
        )
    sources = {
        str(f.relative_to(ROOT)): sha(f)
        for directory in ["src", "scripts", "experiments"]
        for f in sorted((ROOT / directory).rglob("*.py"))
        if "__pycache__" not in str(f)
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            dict(
                config=config,
                checkpoint=sha(checkpoint) if j["dataset"] != "train" else None,
                sources=sources,
                data_sha256=sha(config["dataset"]["input"])
                if j["dataset"] in {"longmemeval", "locomo"}
                else None,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt["fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Protocol/code changed for completed job {folder}; choose a new --output directory"
            )
        for name, digest in receipt["artifacts"].items():
            if sha(folder / name) != digest:
                raise RuntimeError(f"Changed artifact {folder / name}")
        if j["dataset"] == "train":
            checkpoints[j["training_seed"]] = folder / receipt["checkpoint"]
        print("[Verified skip]", key(j), flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    cfg = folder / "config.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=True))
    script = (
        "train_liquid.py"
        if j["dataset"] == "train"
        else ("run_synthetic.py" if j["dataset"] == "synthetic" else "run_real_benchmark.py")
    )
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", str(cfg)]
    if j["dataset"] != "train":
        cmd += ["--checkpoint", str(checkpoint)]
    if j["dataset"] == "longmemeval":
        cmd += ["--start-index", "50"]
    (folder / "command.json").write_text(json.dumps(cmd, indent=2))
    before = set((folder / "runs").glob("*"))
    print("[Run]", key(j), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    created = [
        f
        for f in (folder / "runs").glob("*")
        if f not in before and (f / "dataset_manifest.json").exists()
    ]
    if len(created) != 1:
        raise RuntimeError(f"Expected one result folder: {created}")
    result = created[0]
    required = ["config.yaml", "dataset_manifest.json"] + (
        ["checkpoints/best.pt"] if j["dataset"] == "train" else ["predictions.jsonl", "metrics.csv"]
    )
    required += [
        str(f.relative_to(result))
        for f in result.rglob("*")
        if f.is_file() and str(f.relative_to(result)) not in required
    ]
    artifacts = {str((result / f).relative_to(folder)): sha(result / f) for f in required}
    receipt = dict(
        job=j,
        fingerprint=fingerprint,
        artifacts=artifacts,
        result=str(result.relative_to(folder)),
        source_sha256=sources,
    )
    if j["dataset"] == "train":
        checkpoints[j["training_seed"]] = result / "checkpoints/best.pt"
        receipt["checkpoint"] = str(checkpoints[j["training_seed"]].relative_to(folder))
    done.write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default="all",
        choices=[
            "all",
            "priority",
            "retrieval",
            "ablations",
            "ordering",
            "cue-free",
            "extra-seeds",
        ],
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p = yaml.safe_load((ROOT / "experiments/evaluation_study.yaml").read_text())
    selected = jobs(p, args.group)
    if not selected:
        parser.error("No jobs for this project/group")
    if args.plan:
        for j in selected:
            print(key(j))
        print(f"{len(selected)} jobs; no models loaded or outputs written.")
        return
    suite = (args.output or ROOT / p["output"]).resolve()
    suite.mkdir(parents=True, exist_ok=True)
    checkpoints = {int(k): ROOT / v for k, v in p["checkpoints"].items()}
    for j in selected:
        run(j, p, suite, checkpoints)
    subprocess.run(
        [sys.executable, str(ROOT / "experiments/summarize_results.py"), "--output", str(suite)],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
