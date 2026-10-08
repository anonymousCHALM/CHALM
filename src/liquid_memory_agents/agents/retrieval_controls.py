"""Prefix-free retrieval controls for the evaluation experiment protocol."""

from types import SimpleNamespace
import re
import torch
from .rag import RAGMemoryAgent, BoundedRAGMemoryAgent
from .lexical import LexicalMemoryAgent
from .chalm import hybrid_indices


class RankedRead:
    def get_memory_context(self, query):
        if not self.turns:
            return "", {"retrieved_session_ids": []}
        raw = re.sub(r"^\[Question date: .*?\] ", "", query)
        # Ranking uses existing stored keys/text. Timestamp serialization below
        # is used only for stable temporal ordering, not for lexical matching.
        proxy = SimpleNamespace(
            count=len(self.turns),
            top_k=self.top_k,
            embeddings=torch.stack([t.embedding for t in self.turns]),
            texts=[
                f"session={t.session_id}; timestamp={t.timestamp or ''}; content={t.text.split('] ', 1)[-1].split(': ', 1)[-1]}"
                for t in self.turns
            ],
        )
        ids = hybrid_indices(proxy, raw, self.encoder.encode(query).cpu(), mode=self.retrieval_mode)
        blocks = []
        for i in ids:
            record = self.turns[i]
            text = record.text
            if len(text) > self.max_turn_chars:
                half = max(1, (self.max_turn_chars - 20) // 2)
                text = text[:half] + "\n...[clipped]...\n" + text[-half:]
            blocks.append(f"[Session id: {record.session_id}]\n{text}")
        context = "\n\n".join(blocks)
        if len(context) > self.max_context_chars:
            context = context[-self.max_context_chars :]
        self._last = {
            "turn_indices": ids,
            "retrieved_session_ids": [self.turns[i].session_id for i in ids],
            "retrieval_mode": self.retrieval_mode,
            "order": "timestamp",
            "prefix": False,
        }
        return context, dict(self._last)


class HybridGrowingRAG(RankedRead, RAGMemoryAgent):
    name = "hybrid_growing_rag"
    retrieval_mode = "hybrid"


class HybridFIFORAG(RankedRead, BoundedRAGMemoryAgent):
    name = "hybrid_fifo_rag"
    retrieval_mode = "hybrid"


class BM25GrowingRAG(RankedRead, RAGMemoryAgent):
    name = "bm25_growing_rag"
    retrieval_mode = "bm25"


class BM25Cache(LexicalMemoryAgent):
    name = "bm25_cache"

    def get_memory_context(self, query):
        raw = re.sub(r"^\[Question date: .*?\] ", "", query)
        ids = hybrid_indices(self.memory, raw, self.encoder.encode(query).cpu(), mode="bm25")
        return "\n\n".join(f"[Lexical memory entry {i}]\n{self.memory.texts[i]}" for i in ids), {
            "retrieved_indices": ids,
            "retrieval_mode": "bm25",
            "order": "timestamp",
            "prefix": False,
        }
