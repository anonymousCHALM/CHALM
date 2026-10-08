"""Reproducible main comparisons and explicitly separated ablations."""

from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from liquid_memory_agents.utils.config import load_config


def relay_output(stream, terminal, log):
    """Preserve tqdm carriage returns instead of universal-newline conversion."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            break
        text = decoder.decode(chunk)
        terminal.write(text)
        terminal.flush()
        log.write(text)
        log.flush()
    tail = decoder.decode(b"", final=True)
    if tail:
        terminal.write(tail)
        terminal.flush()
        log.write(tail)
        log.flush()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hash():
    paths = sorted(
        set((ROOT / "src").rglob("*.py"))
        | set((ROOT / "scripts").glob("*.py"))
        | set((ROOT / "configs").glob("*.yaml"))
    )
    return hashlib.sha256(
        json.dumps({str(p.relative_to(ROOT)): digest(p) for p in paths}, sort_keys=True).encode()
    ).hexdigest()


def make_jobs(protocol):
    jobs = []
    seeds = protocol["training_seeds"]
    for seed in seeds:
        jobs.append({"key": f"train_{seed}", "seed": seed, "dataset": "train"})
    for dataset in ("synthetic", "longmemeval", "locomo"):
        for agent in protocol["baselines"] + protocol["shared_ablations"]:
            group = "main" if agent in protocol["baselines"] else "ablations/pathways"
            jobs.append(
                {
                    "key": f"{dataset}_{agent}",
                    "seed": seeds[0],
                    "dataset": dataset,
                    "agent": agent,
                    "group": group,
                }
            )
        for seed in seeds:
            for agent in protocol["checkpoint_methods"]:
                group = "main" if agent == protocol["main_method"] else "ablations/pathways"
                jobs.append(
                    {
                        "key": f"{dataset}_{agent}_{seed}",
                        "seed": seed,
                        "dataset": dataset,
                        "agent": agent,
                        "group": group,
                    }
                )
    for capacity in protocol.get("capacity_sweep", []):
        for agent in protocol["capacity_methods"]:
            jobs.append(
                {
                    "key": f"capacity_{capacity}_{agent}",
                    "seed": seeds[0],
                    "dataset": "synthetic",
                    "agent": agent,
                    "capacity": capacity,
                    "group": "ablations/capacity",
                }
            )
    for job in jobs:
        if job["dataset"] == "train":
            job["folder"] = f"training/seed_{job['seed']}"
            job["training_seed"] = job["seed"]
        else:
            dependent = job["agent"] in protocol["checkpoint_methods"]
            job["training_seed"] = job["seed"] if dependent else None
            capacity = f"/entries_{job['capacity']}" if "capacity" in job else ""
            identity = f"/seed_{job['seed']}" if dependent else ""
            job["folder"] = f"{job['group']}/{job['dataset']}{capacity}/{job['agent']}{identity}"
    return jobs


def sample_resources(pid, stop, target):
    import psutil

    with target.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            row = {"time_utc": datetime.now(timezone.utc).isoformat()}
            try:
                process = psutil.Process(pid)
                row["process_rss_bytes"] = process.memory_info().rss
                row["process_cpu_seconds"] = sum(process.cpu_times()[:2])
            except psutil.Error:
                break
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                row["whole_device_samples_index_mib_util_percent_watts"] = result.stdout.strip()
                if result.returncode:
                    row["gpu_sampling_error"] = result.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as error:
                row["gpu_sampling_error"] = str(error)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            stop.wait(1)


def run_job(job, protocol, suite):
    directory = suite / "jobs" / job.get("folder", job["key"])
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "complete.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        for relative, expected in receipt["artifacts"].items():
            path = directory / relative
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"Completed artifact missing or changed: {path}")
        print(f"[Skip verified] {job['key']}", flush=True)
        return
    dataset = job["dataset"]
    config = load_config(
        ROOT / "configs" / ("base.yaml" if dataset == "train" else f"{dataset}.yaml")
    )
    attempts = directory / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        attempt = attempts / f"attempt_{number:03d}"
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            number += 1
    config["run"].update(
        seed=job["seed"] if dataset == "train" else protocol["evaluation_seed"],
        output_root=str(attempt),
    )
    config["training"].update(
        data_seed=protocol["training_data_seed"],
        epochs=protocol["training_epochs"],
        examples=protocol["training_examples"],
    )
    config["protocol"] = {
        "training_seed": job["training_seed"],
        "fixed_evaluation_seed": protocol["evaluation_seed"],
        "study": "chalm",
    }
    if dataset != "train":
        config.setdefault("evaluation", {}).update(agents=[job["agent"]], qa_batch_size=1)
    if dataset == "synthetic":
        config["dataset"]["examples_by_horizon"] = protocol["synthetic_examples_by_horizon"]
        config["dataset"].pop("test_seeds", None)
    if "capacity" in job:
        config["liquid"]["lexical_memory"]["capacity"] = job["capacity"]
        config["bounded_rag"]["capacity"] = job["capacity"]
    config_path = attempt / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    script = (
        "train_liquid.py"
        if dataset == "train"
        else ("run_synthetic.py" if dataset == "synthetic" else "run_real_benchmark.py")
    )
    command = [sys.executable, str(ROOT / "scripts" / script), "--config", str(config_path)]
    if dataset != "train":
        train_job = next(j for j in make_jobs(protocol) if j["key"] == f"train_{job['seed']}")
        train_dir = suite / "jobs" / train_job["folder"]
        receipt = json.loads((train_dir / "complete.json").read_text())
        checkpoint = train_dir / receipt["checkpoint"]
        if digest(checkpoint) != receipt["artifacts"][receipt["checkpoint"]]:
            raise RuntimeError(f"Training checkpoint changed: {checkpoint}")
        command += ["--checkpoint", str(checkpoint)]
        if dataset == "longmemeval":
            command += ["--start-index", "50"]
    env = dict(
        os.environ,
        CHALM_PROFILE_PATH=str(attempt / "phases.jsonl"),
        CHALM_RUN_NAME="training_artifacts" if dataset == "train" else "evaluation_artifacts",
        PYTHONUNBUFFERED="1",
    )
    (attempt / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    print(f"[Run] {job['key']} -> {attempt}", flush=True)
    start = time.perf_counter()
    with (attempt / "terminal.log").open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=-1
        )
        stop = threading.Event()
        sampler = threading.Thread(
            target=sample_resources,
            args=(process.pid, stop, attempt / "resource_samples.jsonl"),
            daemon=True,
        )
        sampler.start()
        try:
            relay_output(process.stdout, sys.stdout, log)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            stop.set()
            sampler.join(timeout=5)
    (attempt / "runtime.json").write_text(
        json.dumps({"exit_code": code, "wall_seconds": time.perf_counter() - start}, indent=2)
    )
    if code:
        raise RuntimeError(f"Job failed; previous completed jobs are preserved: {job['key']}")
    manifests = list(attempt.glob("*/dataset_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one completed dataset manifest for {job['key']}")
    result = manifests[0].parent
    required = ["dataset_manifest.json", "config.yaml"]
    required += (
        ["checkpoints/best.pt", "training_metrics.json"]
        if dataset == "train"
        else ["predictions.jsonl", "metrics.csv", "memory.csv", "latency.csv"]
    )
    artifacts = {}
    for relative in required:
        path = result / relative
        artifacts[str(path.relative_to(directory))] = digest(path)
    receipt = {
        "job": job,
        "result": str(result.relative_to(directory)),
        "attempt": str(attempt.relative_to(directory)),
        "artifacts": artifacts,
    }
    if dataset == "train":
        receipt["checkpoint"] = str((result / "checkpoints/best.pt").relative_to(directory))
    complete.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/final_study.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument(
        "--seed",
        type=int,
        help="Run shared jobs and one training seed before moving to the next seed",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol = yaml.safe_load(args.config.read_text())
    if (
        len(set(protocol["training_seeds"])) != len(protocol["training_seeds"])
        or not protocol["training_seeds"]
    ):
        raise ValueError("Training seeds must be nonempty and distinct")
    if args.seed is not None and args.seed not in protocol["training_seeds"]:
        parser.error(f"--seed must be one of {protocol['training_seeds']}")
    jobs = make_jobs(protocol)
    jobs = [
        j
        for j in jobs
        if (args.phase == "all" or (j["dataset"] == "train") == (args.phase == "train"))
        and (args.seed is None or j["training_seed"] in (None, args.seed))
    ]
    for j in jobs:
        print(j["folder"])
    print(
        f"{len(jobs)} jobs; baselines run once; one fixed evaluation corpus; no response-cache reuse."
    )
    if args.dry_run:
        return
    for name in ("longmemeval", "locomo"):
        path = Path(load_config(ROOT / "configs" / f"{name}.yaml")["dataset"]["input"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run scripts/download_data.py --dataset {name}"
            )
    import torch

    __import__("psutil")
    __import__("rank_bm25")
    if "linearrag_local" in protocol["baselines"]:
        import spacy

        model = load_config(ROOT / "configs/base.yaml")["linearrag_local"]["ner_model"]
        spacy.load(model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full suite; check your PyTorch installation")
    suite = ROOT / protocol["output"]
    suite.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": protocol,
        "source_sha256": source_hash(),
        "datasets": {
            name: digest(load_config(ROOT / "configs" / f"{name}.yaml")["dataset"]["input"])
            for name in ("longmemeval", "locomo")
        },
    }
    # JSON turns YAML integer horizon keys into strings; normalize before resume checks.
    manifest = json.loads(json.dumps(manifest))
    path = suite / "suite_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Source, protocol, or data changed; use a new output directory")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (suite / "run_index.json").write_text(
        json.dumps(make_jobs(protocol), indent=2), encoding="utf-8"
    )
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    packages_path = suite / "packages.txt"
    if packages_path.exists() and packages_path.read_text() != packages:
        raise RuntimeError(
            "Installed packages changed. Restore the environment or start a new suite directory."
        )
    packages_path.write_text(packages)
    for index, job in enumerate(jobs, 1):
        print(f"\n[Suite {index}/{len(jobs)}]", flush=True)
        run_job(job, protocol, suite)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_results.py"), "--suite", str(suite)],
        check=True,
    )


if __name__ == "__main__":
    main()
