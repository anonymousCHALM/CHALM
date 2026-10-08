"""Short real-model integration check; no training checkpoint or scored claims."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_synthetic import build_agents, generate, observe_sequence
from liquid_memory_agents.embeddings import SentenceTransformerEncoder
from liquid_memory_agents.llm import TransformersAnswerReader
from liquid_memory_agents.utils.config import load_config
from liquid_memory_agents.utils import seed_everything


def main():
    os.chdir(ROOT)
    config = load_config(ROOT / "configs/base.yaml")
    config["evaluation"] = {"agents": ["linearrag_local"]}
    parent = ROOT / "results/smoke/baselines"
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        output = parent / f"attempt_{index:03d}"
        try:
            output.mkdir()
            break
        except FileExistsError:
            index += 1
    os.environ["CHALM_PROFILE_PATH"] = str(output / "phases.jsonl")
    seed_everything(7)
    print(f"[Preflight] {output}", flush=True)
    encoder = SentenceTransformerEncoder(config["embedding"]["model"])
    reader = TransformersAnswerReader(
        config["answer_llm"]["model"],
        max_input_tokens=config["answer_llm"]["max_input_tokens"],
        max_new_tokens=config["answer_llm"]["max_new_tokens"],
    )
    agents = build_agents(encoder, None, reader, config, None)
    history = [
        ("Alice lives in Paris.", "2026-01-01", "s1"),
        ("Alice works for Beacon Labs.", "2026-01-02", "s2"),
        ("Her project access code is ZX-482.", "2026-01-03", "s3"),
        ("Correction: Alice now lives in Berlin, not Paris.", "2026-01-04", "s4"),
    ]
    results = []
    for agent in agents:
        print(f"[Preflight] Building {agent.name}", flush=True)
        observe_sequence(agent, history)
        prediction, context, metadata, read_ms, generation_ms = generate(
            agent, "Where does Alice currently live?"
        )
        if not context:
            raise RuntimeError(
                f"{agent.name} returned empty context; inspect extraction before full run"
            )
        results.append(
            {
                "agent": agent.name,
                "prediction": prediction,
                "context": context,
                "metadata": metadata,
                "read_ms": read_ms,
                "generation_ms": generation_ms,
            }
        )
        print(f"{agent.name}: {prediction}; diagnostics={metadata}", flush=True)
    (output / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        "Integration completed. Inspect predictions and JSON failures; this is not a quality benchmark."
    )


if __name__ == "__main__":
    main()
