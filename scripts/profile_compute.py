"""Separate diagnostic: counted PyTorch FLOPs, not total hardware TFLOPS."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_synthetic import build_agents, generate, observe_sequence
from liquid_memory_agents.datasets.synthetic import generate_examples
from liquid_memory_agents.embeddings import SentenceTransformerEncoder, LexicalAugmentedEncoder
from liquid_memory_agents.llm import TransformersAnswerReader
from liquid_memory_agents.utils import seed_everything
from liquid_memory_agents.utils.config import load_config


def profile_call(function):
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda = torch.cuda.is_available()
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities, with_flops=True, record_shapes=True
    ) as profile:
        start = time.perf_counter()
        function()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    events = profile.key_averages()
    counted = sum(e.flops or 0 for e in events)
    return {
        "supported_operator_flops": counted,
        "supported_operator_tflop": counted / 1e12,
        "instrumented_wall_seconds": elapsed,
        "supported_operator_tflop_per_wall_second": counted / max(elapsed, 1e-12) / 1e12,
        "scope": "partial PyTorch operator estimates, CPU and CUDA; not achieved GPU TFLOPS",
        "uncounted_operator_names": sorted({e.key for e in events if not e.flops}),
        "counted_operators": [
            {"name": e.key, "calls": e.count, "flops": e.flops} for e in events if e.flops
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--agents", nargs="+", default=["chalm", "rag", "bounded_rag", "linearrag_local"]
    )
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.turns < 1 or args.repeats < 1:
        parser.error("turns and repeats must be positive")
    os.chdir(ROOT)
    os.environ.pop("CHALM_PROFILE_PATH", None)
    seed_everything(29001)
    config = load_config(ROOT / "configs/base.yaml")
    config["evaluation"] = {"agents": args.agents}
    output = ROOT / "results/compute_diagnostics" / f"turns_{args.turns}"
    output.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (output / f"attempt_{attempt:03d}").exists():
        attempt += 1
    output = output / f"attempt_{attempt:03d}"
    output.mkdir()
    encoder = SentenceTransformerEncoder(config["embedding"]["model"])
    reader = TransformersAnswerReader(
        config["answer_llm"]["model"],
        max_input_tokens=config["answer_llm"]["max_input_tokens"],
        max_new_tokens=config["answer_llm"]["max_new_tokens"],
    )
    agents = build_agents(
        encoder,
        LexicalAugmentedEncoder(encoder, config["liquid"]["lexical_bytes"]),
        reader,
        config,
        args.checkpoint,
    )
    example = generate_examples(1, args.turns, 29001)[0]
    observations = [
        ({"role": role, "content": text}, timestamp, session)
        for text, role, timestamp, session in zip(
            example.turns, example.roles, example.timestamps, example.session_ids
        )
    ]
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "config": config,
                "checkpoint": str(args.checkpoint),
                "example": asdict(example),
                "repeats": args.repeats,
                "torch_version": torch.__version__,
            },
            indent=2,
        )
    )
    for agent in agents:
        print(f"[Compute diagnostic] Warmup {agent.name}", flush=True)
        observe_sequence(agent, observations)
        generate(agent, example.query)
        for repeat in range(args.repeats):
            agent.reset()
            for phase, call in [
                ("history_update", lambda: observe_sequence(agent, observations)),
                ("query", lambda: generate(agent, example.query)),
            ]:
                row = {"agent": agent.name, "phase": phase, "repeat": repeat, **profile_call(call)}
                with (output / "counted_compute.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(
                    f"{agent.name} {phase}: {row['supported_operator_tflop']:.4f} counted TFLOP",
                    flush=True,
                )
    print(f"Diagnostic only, not primary latency or total FLOPs: {output}")


if __name__ == "__main__":
    main()
