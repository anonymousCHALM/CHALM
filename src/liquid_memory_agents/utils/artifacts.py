from __future__ import annotations

import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_RUN_FILES = {
    "config.yaml",
    "environment.json",
    "hardware.json",
    "dataset_manifest.json",
    "predictions.jsonl",
    "metrics.csv",
    "per_type.csv",
    "latency.csv",
    "memory.csv",
    "errors.jsonl",
}


def create_run_directory(root: str | Path, label: str) -> Path:
    name = os.environ.get("CHALM_RUN_NAME")
    if name:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("CHALM_RUN_NAME must be a single directory name")
        destination = Path(root) / name
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "logs").mkdir()
        (destination / "checkpoints").mkdir()
        return destination
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(root) / f"{stamp}_{label}"
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = Path(root) / f"{stamp}_{label}_{suffix}"
    destination.mkdir(parents=True)
    (destination / "logs").mkdir()
    (destination / "checkpoints").mkdir()
    return destination


def write_environment(run_dir: str | Path) -> None:
    import torch

    run_dir = Path(run_dir)
    packages = []
    for name in (
        "torch",
        "transformers",
        "sentence-transformers",
        "accelerate",
        "numpy",
        "pyyaml",
        "pandas",
        "scikit-learn",
        "faiss-cpu",
        "tqdm",
        "nltk",
        "requests",
        "matplotlib",
        "tokenizers",
        "safetensors",
        "huggingface-hub",
        "rank-bm25",
        "psutil",
        "networkx",
        "scipy",
        "spacy",
        "en-core-web-sm",
    ):
        try:
            packages.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            pass
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    hardware = {
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "torch_threads": torch.get_num_threads(),
        "gpu_total_memory_bytes": [
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ],
    }
    (run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
