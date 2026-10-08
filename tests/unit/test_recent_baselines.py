from types import SimpleNamespace

import pytest
import torch

from liquid_memory_agents.agents.graph_rag import LinearRAGLocalAgent


class Encoder:
    dimension = 3

    def encode(self, text):
        if isinstance(text, list):
            return torch.stack([self.encode(t) for t in text])
        return torch.nn.functional.normalize(
            torch.tensor([1.0, float("alice" in text.lower()), float("paris" in text.lower())]),
            dim=0,
        )


class NER:
    def __call__(self, text):
        entities = [
            SimpleNamespace(text=n, label_="PERSON")
            for n in ("Alice", "Paris")
            if n.lower() in text.lower()
        ]
        sentence = SimpleNamespace(text=text, ents=entities)
        return SimpleNamespace(ents=entities, sents=[sentence])

    def pipe(self, texts):
        return map(self, texts)


class Reader:
    pass


def test_graph_build_query_and_reset():
    agent = LinearRAGLocalAgent(Encoder(), Reader(), ner=NER(), bridge_threshold=0)
    agent.observe_many(
        [
            ("Alice visited Paris.", "2026-01-01", "a"),
            ("Paris hosted the event.", "2026-01-02", "b"),
        ]
    )
    with pytest.raises(RuntimeError, match="finalize"):
        agent.get_memory_context("Alice?")
    agent.finalize_observation()
    size = agent.memory_size_bytes()
    nodes = list(agent.graph.nodes)
    context, meta = agent.get_memory_context("Where did Alice go?")
    assert "Paris" in context
    assert meta["retrieval_mode"] == "semantic_bridge_then_ppr"
    assert set(meta["retrieved_session_ids"]) == {"a", "b"}
    assert nodes == list(agent.graph.nodes)
    assert agent.memory_size_bytes() == size
    assert agent.get_memory_context("Which event?")[1]["retrieval_mode"] == "dense_no_entity_seed"
    agent.reset()
    agent.finalize_observation()
    assert agent.get_memory_context("Alice?")[0] == ""
    assert not agent.entities and not agent.turns
