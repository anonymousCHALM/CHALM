"""Controlled LinearRAG-style adaptation; see docs/BASELINE_PROTOCOL.md."""

from collections import defaultdict
import math

import networkx as nx
import torch
from torch.nn import functional as F

from .rag import RAGMemoryAgent


def normalize(values):
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 1e-8 else torch.ones_like(values)


class LinearRAGLocalAgent(RAGMemoryAgent):
    name = "linearrag_local"

    def __init__(
        self,
        encoder,
        answerer,
        *,
        ner=None,
        ner_model="en_core_web_sm",
        hops=3,
        bridge_threshold=0.4,
        sentences_per_entity=3,
        damping=0.5,
        passage_ratio=2.0,
        passage_weight=0.05,
        **kwargs,
    ):
        super().__init__(encoder, answerer, **kwargs)
        if ner is None:
            import spacy

            # No regex fallback: an unavailable NER model must fail visibly.
            ner = spacy.load(ner_model)
        self.ner = ner
        self.ner_model = ner_model
        self.hops, self.bridge_threshold = hops, bridge_threshold
        self.sentences_per_entity = sentences_per_entity
        self.damping, self.passage_ratio, self.passage_weight = (
            damping,
            passage_ratio,
            passage_weight,
        )
        self.reset()

    def reset(self):
        super().reset()
        self.graph = nx.Graph()
        self.entities, self.sentences, self.sentence_entities = [], [], []
        self.entity_sentences = defaultdict(list)
        self.entity_vectors = torch.empty(0, self.encoder.dimension)
        self.sentence_vectors = torch.empty(0, self.encoder.dimension)
        self._indexed_count = -1

    @staticmethod
    def entities_in(doc):
        return sorted(
            {
                e.text.casefold().strip()
                for e in doc.ents
                if e.label_ not in {"CARDINAL", "ORDINAL"} and e.text.strip()
            }
        )

    def finalize_observation(self):
        if self._indexed_count == len(self.turns):
            return
        # Rebuild only at the observation boundary, never using a query.
        self.graph = nx.Graph()
        self.entities, self.sentences, self.sentence_entities = [], [], []
        self.entity_sentences = defaultdict(list)
        entity_ids = {}
        for i, doc in enumerate(self.ner.pipe([r.text for r in self.turns])):
            self.graph.add_node(("p", i))
            for sentence in doc.sents:
                names = self.entities_in(sentence)
                if not names:
                    continue
                sid = len(self.sentences)
                self.sentences.append(sentence.text)
                ids = []
                for name in names:
                    if name not in entity_ids:
                        entity_ids[name] = len(self.entities)
                        self.entities.append(name)
                    eid = entity_ids[name]
                    ids.append(eid)
                    self.entity_sentences[eid].append(sid)
                    self.graph.add_edge(("p", i), ("e", eid), weight=1.0)
                self.sentence_entities.append(ids)
            # Adjacent passages are linked as in the released implementation.
            if i:
                self.graph.add_edge(("p", i - 1), ("p", i), weight=1.0)
        for i, record in enumerate(self.turns):
            neighbors = [n for n in self.graph[("p", i)] if n[0] == "e"]
            counts = {
                n: max(1, record.text.casefold().count(self.entities[n[1]])) for n in neighbors
            }
            total = sum(counts.values())
            for node, count in counts.items():
                self.graph[("p", i)][node]["weight"] = count / total
        self.entity_vectors = self._embed(self.entities)
        self.sentence_vectors = self._embed(self.sentences)
        self._indexed_count = len(self.turns)

    def _embed(self, texts):
        return (
            F.normalize(self.encoder.encode(texts).float(), dim=-1)
            if texts
            else torch.empty(0, self.encoder.dimension)
        )

    def get_memory_context(self, query):
        if self._indexed_count != len(self.turns):
            raise RuntimeError("Call finalize_observation before querying graph memory")
        if not self.turns:
            return "", {"retrieved_session_ids": []}
        query_vector = F.normalize(self.encoder.encode(query).float(), dim=-1)
        passage_scores = torch.stack([r.embedding for r in self.turns]) @ query_vector
        seeds = self.entities_in(self.ner(query))
        active = {}
        if seeds and self.entities:
            similarities = self.entity_vectors @ self._embed(seeds).T
            for column in range(len(seeds)):
                i = int(similarities[:, column].argmax())
                active[i] = (max(0.0, float(similarities[i, column])), 1)
        if not active:
            indices = torch.argsort(passage_scores, descending=True, stable=True)[
                : self.top_k
            ].tolist()
            mode = "dense_no_entity_seed"
        else:
            sentence_scores = self.sentence_vectors @ query_vector
            frontier = dict(active)
            visited = set()
            accumulated = {i: value[0] for i, value in active.items()}
            for level in range(2, self.hops + 1):
                next_frontier = {}
                for eid, (strength, _) in sorted(frontier.items()):
                    if strength < self.bridge_threshold:
                        continue
                    sentences = sorted(
                        (s for s in self.entity_sentences[eid] if s not in visited),
                        key=lambda s: (-float(sentence_scores[s]), s),
                    )[: self.sentences_per_entity]
                    for sid in sentences:
                        visited.add(sid)
                        value = strength * float(sentence_scores[sid])
                        if value < self.bridge_threshold:
                            continue
                        for other in self.sentence_entities[sid]:
                            accumulated[other] = accumulated.get(other, 0.0) + value
                            next_frontier[other] = (value, level)
                active.update(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break
            initial = {node: 0.0 for node in self.graph}
            for eid, value in accumulated.items():
                initial[("e", eid)] = value
            dense = normalize(passage_scores)
            for i, record in enumerate(self.turns):
                bonus = sum(
                    strength * math.log1p(record.text.casefold().count(self.entities[eid])) / level
                    for eid, (strength, level) in active.items()
                )
                initial[("p", i)] = (
                    self.passage_ratio * float(dense[i]) + math.log1p(bonus)
                ) * self.passage_weight
            ranks = nx.pagerank(
                self.graph,
                alpha=self.damping,
                personalization=initial,
                weight="weight",
                max_iter=500,
                tol=1e-8,
            )
            indices = sorted(range(len(self.turns)), key=lambda i: (-ranks[("p", i)], i))[
                : self.top_k
            ]
            mode = "semantic_bridge_then_ppr"
        blocks = [
            f"[Session id: {self.turns[i].session_id}]\n{self.turns[i].text[: self.max_turn_chars]}"
            for i in reversed(indices)
        ]
        context = "\n\n".join(blocks)[-self.max_context_chars :]
        self._last = {
            "turn_indices": indices,
            "retrieved_session_ids": list(dict.fromkeys(self.turns[i].session_id for i in indices)),
            "retrieval_mode": mode,
            "activated_entities": len(active),
            "graph_nodes": len(self.graph),
            "graph_edges": self.graph.number_of_edges(),
            "adaptation": "LinearRAG-style; MiniLM + spaCy-sm + shared Qwen",
        }
        return context, dict(self._last)

    def memory_size_bytes(self):
        # Logical payload, not Python/NetworkX RSS; undirected edge: two int64 IDs + float32 weight.
        vectors = sum(
            v.numel() * v.element_size() for v in (self.entity_vectors, self.sentence_vectors)
        )
        text = sum(len(s.encode("utf-8")) for s in self.entities + self.sentences)
        incidence = 8 * sum(len(ids) for ids in self.sentence_entities)
        return (
            super().memory_size_bytes()
            + vectors
            + text
            + incidence
            + self.graph.number_of_edges() * 20
            + len(self.graph) * 16
        )
