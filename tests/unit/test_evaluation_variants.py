from types import SimpleNamespace
import torch
from liquid_memory_agents.agents.rag import RAGMemoryAgent, TurnRecord


class Encoder:
    dimension = 3

    def encode(self, text):
        if isinstance(text, list):
            return torch.stack([self.encode(t) for t in text])
        return torch.tensor([1.0, 0.2 if "apple" in text else 0.7, 0.3])


class Answerer:
    embedding_dimension = 5

    def answer(self, *args, **kwargs):
        return "test"


def test_timestamp_preserves_selected_records_and_orders_ties():
    a = RAGMemoryAgent(Encoder(), Answerer(), top_k=3)
    a.turns = [
        TurnRecord(str(i), torch.tensor([1.0, float(i), 0.0]), t, str(i))
        for i, t in enumerate(["2024-03", "2024-01", "2024-01"])
    ]
    _, before = a.get_memory_context("q")
    a.timestamp_order = True
    context, after = a.get_memory_context("q")
    assert before["turn_indices"] == after["turn_indices"]
    assert (
        context.index("[Session id: 1]")
        < context.index("[Session id: 2]")
        < context.index("[Session id: 0]")
    )


from liquid_memory_agents.agents.chalm import CHALMMemoryAgent, hybrid_indices
from liquid_memory_agents.agents.retrieval_controls import HybridGrowingRAG, HybridFIFORAG


class LiquidEncoder(Encoder):
    semantic_encoder = Encoder()

    def encode_with_semantic(self, text):
        return self.encode(text), self.encode(text)


def test_prefix_formulas_and_state_immutability():
    torch.manual_seed(4)
    a = CHALMMemoryAgent(
        LiquidEncoder(), Answerer(), state_dim=4, slots=2, prefix_tokens=2, condition_name="chalm"
    )
    a.observe("apple", timestamp="2024-01", session_id="a")
    state = a.memory.state.clone()
    latent, _ = a.encoder.encode_with_semantic("q")
    p = a.adapter(a.reader(latent, state)[0])
    p0 = a.adapter(a.reader(latent, torch.zeros_like(state))[0])
    for name, expected in [
        ("chalm", 2 * p - p0),
        ("uncentered", p),
        ("difference_prefix", p - p0),
        ("centered_zero", torch.zeros_like(p)),
        ("chalm_score_order", 2 * p - p0),
    ]:
        a.name = name
        a.get_memory_context("q")
        assert torch.allclose(a._last_read["prefix"], expected)
        assert torch.equal(state, a.memory.state)


def test_full_history_and_fifo_read_only_and_capacity():
    for cls in [HybridGrowingRAG, HybridFIFORAG]:
        a = cls(Encoder(), Answerer(), top_k=2, **({"capacity": 2} if cls is HybridFIFORAG else {}))
        a.observe_many(
            [
                ({"content": text}, t, str(i))
                for i, (text, t) in enumerate(
                    [("apple", "2024-03"), ("pear", "2024-01"), ("apple apple", "2024-02")]
                )
            ]
        )
        n = len(a.turns)
        context, meta = a.get_memory_context("apple")
        assert len(a.turns) == n == (2 if cls is HybridFIFORAG else 3)
        ts = [a.turns[i].timestamp for i in meta["turn_indices"]]
        assert ts == sorted(ts)
        assert not meta["prefix"]


def test_bm25_selection_does_not_follow_dense_ranks():
    m = SimpleNamespace(
        count=3,
        top_k=1,
        embeddings=torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
        texts=[
            "session=a; timestamp=1; content=unique apple",
            "session=b; timestamp=2; content=pear",
            "session=c; timestamp=3; content=orange",
        ],
    )
    assert hybrid_indices(m, "unique", torch.tensor([1.0, 0.0]), mode="bm25") == [0]


def test_evaluation_exports_with_stub_reader(tmp_path, monkeypatch):
    # This verifies runner wiring and artifact format, not model accuracy.
    import importlib.util, sys, yaml, json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "review_smoke_runner", root / "scripts/run_synthetic.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from liquid_memory_agents.embeddings import LexicalAugmentedEncoder

    class StubEncoder(Encoder):
        def __init__(self, *args, **kwargs):
            pass

    class StubReader(Answerer):
        def __init__(self, *args, **kwargs):
            pass

        def answer_with_prefix(self, *args, **kwargs):
            return "test"

    config = m.load_config(root / "configs/synthetic.yaml")
    config["run"]["output_root"] = str(tmp_path / "runs")
    config["liquid"].update(state_dim=4, slots=2, prefix_tokens=2)
    config["liquid"]["lexical_memory"]["capacity"] = 8
    config["dataset"].update(smoke_examples=2, smoke_horizon=10)
    enc = LexicalAugmentedEncoder(StubEncoder(), config["liquid"]["lexical_bytes"])
    from liquid_memory_agents.agents.hybrid_base import HybridMemoryAgent as Agent

    agent = Agent(enc, StubReader(), state_dim=4, slots=2, prefix_tokens=2)
    names = [
        "chalm",
        "centered_zero",
        "hybrid_text",
        "uncentered",
        "difference_prefix",
        "chalm_score_order",
        "lexical_only",
        "hybrid_growing_rag",
        "hybrid_fifo_rag",
        "bm25_cache",
        "bm25_growing_rag",
    ]
    checkpoint = tmp_path / "test.pt"
    torch.save(
        dict(
            format_version=1,
            modules={
                k: v.state_dict()
                for k, v in [
                    ("cell", agent.memory.cell),
                    ("reader", agent.reader),
                    ("adapter", agent.adapter),
                ]
            },
            config=config,
            seed=13,
            dataset_hashes={},
        ),
        checkpoint,
    )
    config.setdefault("evaluation", {})["agents"] = names
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(m, "SentenceTransformerEncoder", StubEncoder)
    monkeypatch.setattr(m, "TransformersAnswerReader", StubReader)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_synthetic.py", "--config", str(cfg), "--checkpoint", str(checkpoint), "--smoke"],
    )
    m.main()
    predictions = next((tmp_path / "runs").rglob("predictions.jsonl"))
    rows = [json.loads(x) for x in predictions.read_text().splitlines()]
    assert len(rows) == 2 * len(names)
    assert {x["agent"] for x in rows} == set(names)
    for name in ["metrics.csv", "config.yaml", "dataset_manifest.json"]:
        assert (predictions.parent / name).is_file()
