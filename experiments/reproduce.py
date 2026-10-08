"""Replay completed CHALM follow-up evaluations using the shipped checkpoints."""

import argparse, json, sys, yaml, subprocess
from pathlib import Path
from run_study import ROOT, run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--group", default="all")
    ap.add_argument("--output", type=Path, default=ROOT / "results/reproduction/followup")
    ap.add_argument("--train-seed", type=int, choices=[13, 29, 31])
    a = ap.parse_args()
    p = yaml.safe_load((ROOT / "experiments/evaluation_study.yaml").read_text())
    checkpoints = {int(k): ROOT / v for k, v in p["checkpoints"].items()}
    if a.train_seed:
        checkpoint = checkpoints[a.train_seed]
        cfg = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
        cfg["run"].update(seed=a.train_seed, output_root=str(a.output.resolve() / "training"))
        cfg.pop("execution", None)
        if a.plan:
            print("Train seed", a.train_seed, "from recorded training settings")
            return
        a.output.mkdir(parents=True, exist_ok=True)
        path = a.output / f"train_seed_{a.train_seed}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_liquid.py"),
                "--config",
                str(path.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
        return
    selected = []
    for receipt in sorted((ROOT / "results/followup_study/jobs").rglob("complete.json")):
        j = json.loads(receipt.read_text())["job"]
        if a.group == "all" or j["group"] == a.group:
            selected.append(j)
    print(len(selected), "evaluation jobs", flush=True)
    for i, j in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}]", j, flush=True)
        if not a.plan:
            run(j, p, a.output.resolve(), checkpoints)
    if not a.plan:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/summarize_results.py"),
                "--output",
                str(a.output.resolve()),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
