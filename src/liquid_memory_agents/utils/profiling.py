"""Opt-in synchronized phase measurements, excluding model initialization."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from functools import wraps

import torch


def record_parameters(modules):
    target = os.environ.get("CHALM_PROFILE_PATH")
    if not target:
        return
    values = {
        name: {
            "parameters": sum(p.numel() for p in module.parameters()),
            "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in module.parameters()),
            "resolved_model_revision": getattr(
                getattr(module, "config", None), "_commit_hash", None
            ),
        }
        for name, module in modules.items()
    }
    path = Path(target).with_name("parameters.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def profiled_agent(phase):
    def decorate(function):
        @wraps(function)
        def wrapped(agent, *args, **kwargs):
            count = len(args[0]) if args and isinstance(args[0], list) else 1
            with measure(
                phase,
                agent=agent.name,
                items=count,
                horizon=getattr(agent, "_profile_horizon", None),
            ) as extra:
                result = function(agent, *args, **kwargs)
                if phase.startswith("query"):
                    extra["text_input_tokens"] = getattr(
                        agent.answerer, "last_input_token_counts", []
                    )
                    extra["output_tokens_including_eos"] = getattr(
                        agent.answerer, "last_output_token_counts", []
                    )
                return result

        return wrapped

    return decorate


@contextmanager
def measure(phase, **labels):
    target = os.environ.get("CHALM_PROFILE_PATH")
    if not target:
        yield {}
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    for d in devices:
        torch.cuda.synchronize(d)
        torch.cuda.reset_peak_memory_stats(d)
    baseline = {d: torch.cuda.memory_allocated(d) for d in devices}
    started, cpu = time.perf_counter(), time.process_time()
    status = "ok"
    extra = {}
    try:
        yield extra
    except BaseException:
        status = "failed"
        raise
    finally:
        for d in devices:
            torch.cuda.synchronize(d)
        row = {
            "phase": phase,
            **labels,
            **extra,
            "status": status,
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu,
            "cuda": [
                {
                    "device": d,
                    "baseline_allocated_bytes": baseline[d],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(d),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(d),
                    "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated(d)
                    - baseline[d],
                }
                for d in devices
            ],
        }
        try:
            import psutil

            row["process_rss_bytes_at_end"] = psutil.Process().memory_info().rss
        except ImportError:
            row["process_rss_bytes_at_end"] = None
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
