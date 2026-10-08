from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.agents import (
    BoundedRAGMemoryAgent,
    HybridMemoryAgent,
    LexicalMemoryAgent,
    RAGMemoryAgent,
    VanillaAgent,
    WindowMemoryAgent,
)
from liquid_memory_agents.liquid.lexical_memory import BoundedLexicalMemory
from liquid_memory_agents.datasets.synthetic import generate_examples
from liquid_memory_agents.embeddings import LexicalAugmentedEncoder, SentenceTransformerEncoder
from liquid_memory_agents.liquid.checkpoint import load_checkpoint
from liquid_memory_agents.llm import TransformersAnswerReader
from liquid_memory_agents.llm.prompts import render_prompt
from liquid_memory_agents.utils import create_run_directory, seed_everything, write_environment
from liquid_memory_agents.utils.config import load_config
from liquid_memory_agents.agents.chalm import CHALMMemoryAgent
from liquid_memory_agents.utils.profiling import profiled_agent, record_parameters

NEW_AGENTS = {
    "chalm",
    "centered_zero",
    "hybrid_text",
    "uncentered",
    "difference_prefix",
    "chalm_score_order",
}
RECENT_AGENTS = {"linearrag_local"}


def uses_prefix(agent):
    return isinstance(agent, HybridMemoryAgent) and agent.name != "hybrid_text"


def load_hybrid(
    encoder,
    reader,
    config,
    checkpoint,
    condition_name="hybrid_base",
    allow_untrained=False,
):
    dtype_name = config["liquid"].get("storage_dtype", "float32")
    dtype = getattr(torch, dtype_name, None)
    if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(f"unsupported liquid storage dtype: {dtype_name}")
    lexical = config["liquid"].get("lexical_memory", {})
    lexical_enabled = condition_name != "pure_liquid"
    agent_class = CHALMMemoryAgent if condition_name in NEW_AGENTS else HybridMemoryAgent
    agent = agent_class(
        encoder,
        reader,
        **{
            key: config["liquid"][key]
            for key in ("state_dim", "slots", "prefix_tokens", "tau_min", "tau_max", "dt")
        },
        storage_dtype=dtype,
        **{
            key: config["liquid"][key]
            for key in ("adaptive_tau", "update", "memory_decay")
            if key in config["liquid"]
        },
        lexical_enabled=lexical_enabled,
        lexical_capacity=lexical.get("capacity", 128),
        lexical_top_k=lexical.get("top_k", 8),
        lexical_max_text_bytes=lexical.get("max_text_bytes", 1024),
        lexical_redundancy_threshold=lexical.get("redundancy_threshold", 0.82),
        lexical_prefix_scale=lexical.get("prefix_scale", 0.1),
        lexical_correction_cues=lexical.get("correction_cues"),
        condition_name=condition_name,
    )
    payload = load_checkpoint(
        checkpoint,
        {"cell": agent.memory.cell, "reader": agent.reader, "adapter": agent.adapter},
    )
    checkpoint_config = payload.get("config", {})
    is_untrained = (
        checkpoint_config.get("initialization_only") or checkpoint_config.get("trained") is False
    )
    expected_trained = config.get("training", {}).get("trained", True)
    if is_untrained and not allow_untrained:
        raise RuntimeError(
            "Untrained Liquid checkpoint rejected from primary evaluation. "
            "Use --allow-untrained only for the labeled trained-vs-random ablation."
        )
    if expected_trained is False and not is_untrained:
        raise RuntimeError(
            "The trained=false ablation requires an initialization-only checkpoint; "
            "a trained checkpoint was supplied."
        )
    trained_liquid = checkpoint_config.get("liquid", {})
    for key, default in (
        ("state_dim", 256),
        ("slots", 8),
        ("prefix_tokens", 16),
        ("lexical_bytes", 256),
        ("adaptive_tau", True),
        ("update", "adaptive"),
    ):
        expected = config["liquid"].get(key, default)
        actual = trained_liquid.get(key, default)
        if actual != expected:
            raise RuntimeError(
                f"Checkpoint/config mismatch for liquid.{key}: checkpoint={actual!r}, "
                f"evaluation={expected!r}"
            )
    expected_correction = config.get("training", {}).get("correction_loss", True)
    actual_correction = checkpoint_config.get("training", {}).get("correction_loss", True)
    if expected_correction != actual_correction:
        raise RuntimeError(
            "Checkpoint/config mismatch for training.correction_loss: "
            f"checkpoint={actual_correction}, evaluation={expected_correction}"
        )
    agent.checkpoint_metadata = {
        "path": str(checkpoint),
        "seed": payload.get("seed"),
        "dataset_hashes": payload.get("dataset_hashes", {}),
        "validation_metrics": payload.get("validation_metrics", {}),
        "epoch": payload.get("epoch"),
    }
    return agent


def build_agents(
    semantic_encoder, liquid_encoder, reader, config, checkpoint, allow_untrained=False
):
    allowed = {
        "vanilla",
        "window",
        "rag",
        "bounded_rag",
        "pure_liquid",
        "lexical_only",
        "hybrid_base",
        "hybrid_base_zero_prefix",
        "hybrid_base_random_prefix",
        "hybrid_base_permuted_prefix",
    }
    requested = config.get("evaluation", {}).get(
        "agents",
        ["vanilla", "window", "rag", "bounded_rag", "chalm", "linearrag_local"],
    )
    allowed.update(["hybrid_growing_rag", "hybrid_fifo_rag", "bm25_growing_rag", "bm25_cache"])
    allowed.update(NEW_AGENTS)
    allowed.update(RECENT_AGENTS)
    unknown = set(requested) - allowed
    if unknown:
        raise ValueError(f"Unknown agents requested: {sorted(unknown)}")
    make_rag = lambda: RAGMemoryAgent(
        semantic_encoder,
        reader,
        config["rag"]["top_k"],
        config["rag"].get("max_turn_chars", 2000),
        config["rag"].get("max_context_chars", 12000),
    )
    bounded_rag = config.get("bounded_rag", {})
    make_bounded_rag = lambda: BoundedRAGMemoryAgent(
        semantic_encoder,
        reader,
        top_k=bounded_rag.get("top_k", config["rag"]["top_k"]),
        capacity=bounded_rag.get("capacity", 512),
        max_text_bytes=bounded_rag.get("max_text_bytes", 1024),
        max_context_chars=bounded_rag.get(
            "max_context_chars", config["rag"].get("max_context_chars", 12000)
        ),
    )
    make_hybrid_base = lambda: load_hybrid(
        liquid_encoder,
        reader,
        config,
        checkpoint,
        condition_name="hybrid_base",
        allow_untrained=allow_untrained,
    )
    make_hybrid_base_zero = lambda: load_hybrid(
        liquid_encoder,
        reader,
        config,
        checkpoint,
        condition_name="hybrid_base_zero_prefix",
        allow_untrained=allow_untrained,
    )
    make_hybrid_base_random = lambda: load_hybrid(
        liquid_encoder,
        reader,
        config,
        checkpoint,
        condition_name="hybrid_base_random_prefix",
        allow_untrained=allow_untrained,
    )
    make_hybrid_base_permuted = lambda: load_hybrid(
        liquid_encoder,
        reader,
        config,
        checkpoint,
        condition_name="hybrid_base_permuted_prefix",
        allow_untrained=allow_untrained,
    )
    make_pure_liquid = lambda: load_hybrid(
        liquid_encoder,
        reader,
        config,
        checkpoint,
        condition_name="pure_liquid",
        allow_untrained=allow_untrained,
    )
    lexical = config["liquid"].get("lexical_memory", {})
    make_lexical = lambda: LexicalMemoryAgent(
        semantic_encoder,
        reader,
        capacity=lexical.get("capacity", 512),
        top_k=lexical.get("top_k", 8),
        max_text_bytes=lexical.get("max_text_bytes", 1024),
        redundancy_threshold=lexical.get("redundancy_threshold", 0.82),
        correction_cues=lexical.get("correction_cues"),
    )
    builders = {
        "vanilla": lambda: VanillaAgent(reader),
        "window": lambda: WindowMemoryAgent(reader),
        "rag": make_rag,
        "bounded_rag": make_bounded_rag,
        "pure_liquid": make_pure_liquid,
        "lexical_only": make_lexical,
        "hybrid_base": make_hybrid_base,
        "hybrid_base_zero_prefix": make_hybrid_base_zero,
        "hybrid_base_random_prefix": make_hybrid_base_random,
        "hybrid_base_permuted_prefix": make_hybrid_base_permuted,
    }
    builders.update(
        {
            name: (
                lambda name=name: load_hybrid(
                    liquid_encoder,
                    reader,
                    config,
                    checkpoint,
                    condition_name=name,
                    allow_untrained=allow_untrained,
                )
            )
            for name in NEW_AGENTS
        }
    )
    if "linearrag_local" in requested:
        from liquid_memory_agents.agents.graph_rag import LinearRAGLocalAgent

        builders["linearrag_local"] = lambda: LinearRAGLocalAgent(
            semantic_encoder,
            reader,
            top_k=config["rag"]["top_k"],
            max_turn_chars=config["rag"].get("max_turn_chars", 2000),
            max_context_chars=config["rag"].get("max_context_chars", 12000),
            **config.get("linearrag_local", {}),
        )
    from liquid_memory_agents.agents.retrieval_controls import (
        HybridGrowingRAG,
        HybridFIFORAG,
        BM25GrowingRAG,
        BM25Cache,
    )

    for name, cls in [
        ("hybrid_growing_rag", HybridGrowingRAG),
        ("hybrid_fifo_rag", HybridFIFORAG),
        ("bm25_growing_rag", BM25GrowingRAG),
    ]:

        def make(cls=cls):
            if cls is HybridFIFORAG:
                return cls(
                    semantic_encoder,
                    reader,
                    top_k=bounded_rag.get("top_k", 8),
                    capacity=bounded_rag.get("capacity", 512),
                    max_text_bytes=bounded_rag.get("max_text_bytes", 1024),
                    max_context_chars=config["rag"].get("max_context_chars", 12000),
                )
            return cls(
                semantic_encoder,
                reader,
                top_k=config["rag"]["top_k"],
                max_turn_chars=config["rag"].get("max_turn_chars", 2000),
                max_context_chars=config["rag"].get("max_context_chars", 12000),
            )

        builders[name] = make
    builders["bm25_cache"] = lambda: BM25Cache(
        semantic_encoder,
        reader,
        capacity=lexical.get("capacity", 512),
        top_k=lexical.get("top_k", 8),
        max_text_bytes=lexical.get("max_text_bytes", 1024),
        redundancy_threshold=lexical.get("redundancy_threshold", 0.82),
        correction_cues=lexical.get("correction_cues"),
    )
    agents = [builders[name]() for name in requested]
    rule = lexical.get("replacement_rule", "cue_gated")
    if rule not in {"cue_gated", "newest_wins"}:
        raise ValueError(f"Unknown replacement_rule: {rule}")
    for agent in agents:
        sidecar = getattr(agent, "lexical_memory", None)
        if sidecar is None and isinstance(getattr(agent, "memory", None), BoundedLexicalMemory):
            sidecar = agent.memory
        if sidecar is not None:
            sidecar.replacement_rule = rule
    modules = {"answer_model": reader.model} if hasattr(reader, "model") else {}
    if hasattr(semantic_encoder, "model"):
        modules["sentence_encoder"] = semantic_encoder.model
    for agent in agents:
        if isinstance(agent, HybridMemoryAgent):
            for label, module in (
                ("cell", agent.memory.cell),
                ("reader", agent.reader),
                ("adapter", agent.adapter),
            ):
                module.eval()
                module.requires_grad_(False)
                modules[f"{agent.name}_{label}"] = module
    record_parameters(modules)
    return agents


@profiled_agent("query")
def generate(agent, query, query_timestamp=None):
    started = time.perf_counter()
    memory_query = f"[Question date: {query_timestamp}] {query}" if query_timestamp else query
    context, metadata = agent.get_memory_context(memory_query)
    read_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    if uses_prefix(agent):
        prefix_scale = (
            agent.lexical_prefix_scale if context and agent.lexical_memory is not None else 1.0
        )
        prediction = agent.answerer.answer_with_prefix(
            query,
            agent._last_read["prefix"] * prefix_scale,
            query_timestamp,
            context=context,
        )
    else:
        prediction = agent.answerer.answer(query, context, query_timestamp)
    generation_ms = (time.perf_counter() - started) * 1000
    metadata = {
        **metadata,
        "generation_hit_token_limit": bool(
            getattr(agent.answerer, "last_generation_hit_limit", False)
        ),
    }
    return prediction, context, metadata, read_ms, generation_ms


@profiled_agent("query_batch")
def generate_batch(agent, examples):
    """Generate a query batch from one already-built, query-independent memory."""
    prepared = []
    for example in examples:
        started = time.perf_counter()
        memory_query = (
            f"[Question date: {example.question_date}] {example.question}"
            if example.question_date
            else example.question
        )
        context, metadata = agent.get_memory_context(memory_query)
        read_ms = (time.perf_counter() - started) * 1000
        prefix = None
        if uses_prefix(agent):
            prefix_scale = (
                agent.lexical_prefix_scale if context and agent.lexical_memory is not None else 1.0
            )
            prefix = agent._last_read["prefix"].detach().clone() * prefix_scale
        prepared.append((context, metadata, read_ms, prefix))

    started = time.perf_counter()
    if uses_prefix(agent):
        predictions = agent.answerer.answer_with_prefix_batch(
            [
                (example.question, item[3], example.question_date, item[0])
                for example, item in zip(examples, prepared)
            ]
        )
    else:
        predictions = agent.answerer.answer_batch(
            [
                (example.question, item[0], example.question_date)
                for example, item in zip(examples, prepared)
            ]
        )
    amortized_generation_ms = (time.perf_counter() - started) * 1000 / max(1, len(examples))
    limits = getattr(agent.answerer, "last_generation_hit_limits", None)
    results = []
    for index, (prediction, (context, metadata, read_ms, _prefix)) in enumerate(
        zip(predictions, prepared)
    ):
        metadata = {
            **metadata,
            "generation_hit_token_limit": (
                bool(limits[index]) if limits and index < len(limits) else False
            ),
        }
        results.append(
            (
                prediction,
                context,
                metadata,
                read_ms,
                amortized_generation_ms,
            )
        )
    return results


def value_match(reference: str, prediction: str) -> bool:
    tokens = re.findall(r"\b[\w'-]+\b", prediction.casefold())
    reference_tokens = re.findall(r"\b[\w'-]+\b", reference.casefold())
    if reference in {"yes", "no"}:
        return bool(tokens) and tokens[0] == reference
    width = len(reference_tokens)
    return any(tokens[i : i + width] == reference_tokens for i in range(len(tokens) - width + 1))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@profiled_agent("history_update")
def observe_sequence(agent, observations) -> None:
    if hasattr(agent, "observe_many"):
        agent.observe_many(observations)
    else:
        for turn, timestamp, session_id in observations:
            agent.observe(turn, timestamp, session_id)
    if hasattr(agent, "finalize_observation"):
        agent.finalize_observation()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--examples",
        type=int,
        help="Override examples per horizon for a targeted diagnostic run",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=(
            "hybrid_growing_rag",
            "hybrid_fifo_rag",
            "bm25_growing_rag",
            "bm25_cache",
            "uncentered",
            "difference_prefix",
            "chalm_score_order",
            *sorted(NEW_AGENTS),
            *sorted(RECENT_AGENTS),
            "vanilla",
            "window",
            "rag",
            "bounded_rag",
            "pure_liquid",
            "lexical_only",
            "hybrid_base",
            "hybrid_base_zero_prefix",
            "hybrid_base_random_prefix",
            "hybrid_base_permuted_prefix",
        ),
        help="Override the evaluated memory conditions",
    )
    parser.add_argument("--allow-untrained", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(
            f"checkpoint does not exist: {args.checkpoint}. Train first and use the "
            "actual results/<run_id>/checkpoints/best.pt path."
        )
    config = load_config(args.config)
    config["run_kind"] = "evaluation"
    if args.agents:
        config.setdefault("evaluation", {})["agents"] = args.agents
    horizons = (
        [int(config["dataset"].get("smoke_horizon", 20))]
        if args.smoke
        else config["dataset"]["turns"]
    )
    configured_counts = config["dataset"].get("examples_by_horizon", {})
    counts = {
        int(horizon): (
            int(args.examples)
            if args.examples is not None
            else (
                int(config["dataset"].get("smoke_examples", 4))
                if args.smoke
                else int(
                    configured_counts.get(
                        str(horizon),
                        configured_counts.get(int(horizon), config["dataset"].get("examples", 100)),
                    )
                )
            )
        )
        for horizon in horizons
    }
    if min(counts.values()) < 1:
        parser.error("--examples must be positive")
    config["execution"] = {
        "smoke": args.smoke,
        "horizons": horizons,
        "examples_per_horizon": counts,
        "checkpoint": str(args.checkpoint.resolve()),
        "agents": config.get("evaluation", {}).get("agents", []),
    }
    seed_everything(config["run"]["seed"])
    run_dir = create_run_directory(config["run"]["output_root"], "synthetic")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    write_environment(run_dir)
    semantic_encoder = SentenceTransformerEncoder(config["embedding"]["model"])
    liquid_encoder = LexicalAugmentedEncoder(
        semantic_encoder, int(config["liquid"].get("lexical_bytes", 256))
    )
    reader = TransformersAnswerReader(
        config["answer_llm"]["model"],
        max_input_tokens=config["answer_llm"]["max_input_tokens"],
        max_new_tokens=config["answer_llm"]["max_new_tokens"],
        abstention_response=config["answer_llm"].get("abstention_response", "I do not know."),
    )
    agents = build_agents(
        semantic_encoder,
        liquid_encoder,
        reader,
        config,
        args.checkpoint,
        allow_untrained=args.allow_untrained,
    )
    predictions, latency, memory = [], [], []
    content_hash = hashlib.sha256()
    all_ids = []
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:

        def record(agent, example, update_ms):
            prediction, context, diagnostics, read_ms, generation_ms = generate(
                agent, example.query
            )
            row = {
                "example_id": example.example_id,
                "agent": agent.name,
                "horizon": horizon,
                "question_type": example.question_type,
                "query": example.query,
                "reference": example.answer,
                "prediction": prediction,
                "correct": value_match(example.answer.casefold(), prediction),
                "prompt": render_prompt(example.query, context),
                "memory_context": context,
                "diagnostics": diagnostics,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            predictions.append(row)
            latency.append(
                {
                    "example_id": example.example_id,
                    "agent": agent.name,
                    "horizon": horizon,
                    "update_ms": update_ms,
                    "read_ms": read_ms,
                    "generation_ms": generation_ms,
                    "end_to_end_ms": update_ms + read_ms + generation_ms,
                    "model_loading_policy": "one shared preloaded model",
                }
            )
            memory.append(
                {
                    "example_id": example.example_id,
                    "agent": agent.name,
                    "horizon": horizon,
                    "bytes": agent.memory_size_bytes(),
                }
            )

        for horizon in horizons:
            examples = generate_examples(
                counts[int(horizon)],
                horizon,
                config["run"]["seed"],
                correction_style=config["dataset"].get("correction_style", "original"),
            )
            for example in examples:
                content_hash.update(json.dumps(asdict(example), sort_keys=True).encode("utf-8"))
            all_ids.extend(example.example_id for example in examples)
            for agent in agents:
                agent._profile_horizon = horizon
                for example in tqdm(examples, desc=f"{horizon} turns / {agent.name}"):
                    agent.reset()
                    update_started = time.perf_counter()
                    observe_sequence(
                        agent,
                        [
                            (
                                {"role": role, "content": turn},
                                timestamp,
                                session_id,
                            )
                            for turn, role, timestamp, session_id in zip(
                                example.turns,
                                example.roles,
                                example.timestamps,
                                example.session_ids,
                            )
                        ],
                    )
                    update_ms = (time.perf_counter() - update_started) * 1000
                    record(agent, example, update_ms)
    metrics = []
    for horizon in horizons:
        for agent in agents:
            rows = [x for x in predictions if x["horizon"] == horizon and x["agent"] == agent.name]
            metrics.append(
                {
                    "horizon": horizon,
                    "agent": agent.name,
                    "examples": len(rows),
                    "accuracy": sum(x["correct"] for x in rows) / len(rows),
                }
            )
    write_csv(run_dir / "metrics.csv", metrics)
    question_types = sorted({row["question_type"] for row in predictions})
    write_csv(
        run_dir / "per_type.csv",
        [
            {
                "agent": a.name,
                "question_type": q,
                "accuracy": sum(x["correct"] for x in rows) / len(rows),
            }
            for a in agents
            for q in question_types
            if (
                rows := [x for x in predictions if x["agent"] == a.name and x["question_type"] == q]
            )
        ],
    )
    write_csv(run_dir / "latency.csv", latency)
    write_csv(run_dir / "memory.csv", memory)
    (run_dir / "errors.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in predictions if not x["correct"]),
        encoding="utf-8",
    )
    evaluation_hash = hashlib.sha256(json.dumps(all_ids, sort_keys=True).encode()).hexdigest()
    checkpoint_agent = next(
        (agent for agent in agents if hasattr(agent, "checkpoint_metadata")),
        None,
    )
    training_hashes = (
        checkpoint_agent.checkpoint_metadata["dataset_hashes"]
        if checkpoint_agent is not None
        else {}
    )
    if evaluation_hash in set(training_hashes.values()):
        raise RuntimeError("evaluation manifest hash overlaps a checkpoint training dataset hash")
    manifest = {
        "example_ids": all_ids,
        "sha256": evaluation_hash,
        "content_sha256": content_hash.hexdigest(),
        "checkpoint_training_hashes": training_hashes,
        "checkpoint_seed": (
            checkpoint_agent.checkpoint_metadata["seed"] if checkpoint_agent is not None else None
        ),
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed immutable run: {run_dir}")


if __name__ == "__main__":
    main()
