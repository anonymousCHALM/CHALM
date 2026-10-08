"""CHALM: Centered Hybrid Augmented Liquid Memory; inherits hybrid memory writes."""

from __future__ import annotations

import re

import torch
from rank_bm25 import BM25Okapi

from .hybrid_base import HybridMemoryAgent


def hybrid_indices(memory, query, embedding, mode="hybrid", timestamp_order=True):
    n = memory.count
    if not n:
        return []
    dense = memory.embeddings[:n] @ torch.nn.functional.normalize(embedding.float(), dim=-1)
    order = torch.argsort(dense, descending=True, stable=True).tolist()

    def tokenize(text):
        text = re.sub(r"['\u2019]s\b", "", text.casefold())
        return re.findall(r"[\w]+(?:[-][\w]+)*", text)

    corpus = [tokenize(t.split("; content=", 1)[-1]) for t in memory.texts[:n]]
    if any(corpus):
        sparse = BM25Okapi(corpus).get_scores(tokenize(query))
        sparse_order = sorted(range(n), key=lambda i: (-float(sparse[i]), i))
        scores = torch.zeros(n)
        weights = torch.tensor([1 / (61 + i) for i in range(n)])
        for ranking in (sparse_order,) if mode == "bm25" else (order, sparse_order):
            scores[ranking] += weights
        order = torch.argsort(scores, descending=True, stable=True).tolist()
    if mode == "bm25" and not any(corpus):
        order = list(range(n))
    if not timestamp_order:
        return order[: memory.top_k]
    return sorted(
        order[: memory.top_k],
        key=lambda i: (memory.texts[i].split("; timestamp=", 1)[-1].split("; content=", 1)[0], i),
    )


class CHALMMemoryAgent(HybridMemoryAgent):
    PREFIX_CONTROLS = {
        **HybridMemoryAgent.PREFIX_CONTROLS,
        "chalm": "learned",
        "uncentered": "learned",
        "difference_prefix": "learned",
        "chalm_score_order": "learned",
        "centered_zero": "zero",
        "hybrid_text": "zero",
    }

    @torch.no_grad()
    def get_memory_context(self, query):
        latent, semantic = self.encoder.encode_with_semantic(query)
        raw_query = re.sub(r"^\[Question date: .*?\] ", "", query)
        ids = hybrid_indices(
            self.lexical_memory,
            raw_query,
            semantic,
            timestamp_order=self.name != "chalm_score_order",
        )
        context = "\n\n".join(
            f"[Lexical memory entry {i}]\n{self.lexical_memory.texts[i]}" for i in ids
        )
        state = self.memory.state.float()
        read, weights = self.reader(latent.float(), state)
        prefix = self.adapter(read)
        if self.name in {"chalm", "difference_prefix", "chalm_score_order"}:
            null_read, _ = self.reader(latent.float(), torch.zeros_like(state))
            prefix = (1 if self.name == "difference_prefix" else 2) * prefix - self.adapter(
                null_read
            )
        if self.name in {"centered_zero", "hybrid_text"}:
            prefix = torch.zeros_like(prefix)
        self._last_read = {"prefix": prefix, "attention": weights.cpu().tolist()}
        return context, {
            "retrieved_indices": ids,
            "retrieved_session_ids": [
                self.lexical_memory.texts[i].split(";", 1)[0].removeprefix("session=") for i in ids
            ],
            "read_policy": self.name,
            "prefix_norm": float(prefix.norm()),
            "sparse_index_policy": "rebuilt transiently from bounded cache",
        }
