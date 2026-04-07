"""
Recall - Vector-based Long-Term Memory for Multi-Agent Systems
Based on: "Vector Storage Based Long-term Memory Research on LLM"

Components:
  - SentenceTransformerEmbedder — semantic dense vectors (all-MiniLM-L6-v2)
  - BM25Retriever               — sparse keyword retrieval, min-max normalised
  - Hybrid retrieval            — α·cosine + (1−α)·BM25, fused per query
  - Ebbinghaus forgetting curve — R(t) = e^(-t/λ), recall strengthens λ
  - Deduplication               — cosine similarity threshold on store
  - Persistence                 — atomic JSON save/load
"""

import math
import time
import uuid
import json
import re
import os
import logging
from typing import Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger("recall.memory")


# ─────────────────────────────────────────────────────────────────
#  SentenceTransformerEmbedder
# ─────────────────────────────────────────────────────────────────

class SentenceTransformerEmbedder:
    """
    Semantic embedder using sentence-transformers (all-MiniLM-L6-v2 by default).

    Understands meaning — paraphrases and synonyms score highly:
      "I was charged twice"  ≈  "double billing on my account"
      "can't log in"         ≈  "authentication failure"
      "API rate limit"       ≈  "429 error too many requests"

    Model is downloaded once (~80MB) and cached in ~/.cache/huggingface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        logger.info("Loading sentence-transformers/%s…", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedder ready (dim=%d)", self.dim)

    def _embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def fit_transform(self, text: str) -> np.ndarray:
        """Embed a document for storage."""
        return self._embed(text)

    def transform(self, text: str) -> np.ndarray:
        """Embed a query for retrieval."""
        return self._embed(text)


# ─────────────────────────────────────────────────────────────────
#  BM25Retriever
# ─────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 keyword retriever with:
    - Per-query min-max score normalisation (results comparable with cosine)
    - delete() method to stay in sync with pruned segments
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._docs:      list[str]       = []
        self._doc_ids:   list[str]       = []
        self._tokenized: list[list[str]] = []
        self._df:        dict[str, int]  = defaultdict(int)
        self._avgdl:     float           = 0.0
        self._id_to_idx: dict[str, int]  = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def add(self, text: str, doc_id: str):
        if doc_id in self._id_to_idx:
            return
        tokens = self._tokenize(text)
        idx = len(self._docs)
        self._id_to_idx[doc_id] = idx
        self._docs.append(text)
        self._doc_ids.append(doc_id)
        self._tokenized.append(tokens)
        for t in set(tokens):
            self._df[t] += 1
        total = sum(len(d) for d in self._tokenized)
        self._avgdl = total / max(len(self._tokenized), 1)

    def delete(self, doc_id: str):
        """Remove a document from the index (call after pruning)."""
        idx = self._id_to_idx.pop(doc_id, None)
        if idx is None:
            return
        old_tokens = self._tokenized[idx]
        for t in set(old_tokens):
            self._df[t] = max(0, self._df[t] - 1)
        self._tokenized[idx] = []
        self._doc_ids[idx]   = None
        total = sum(len(d) for d in self._tokenized)
        self._avgdl = total / max(sum(1 for d in self._tokenized if d), 1)

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        active = sum(1 for d in self._tokenized if d)
        if not active:
            return []
        q_tokens = self._tokenize(text)
        N = active
        scores = []
        for i, doc_tokens in enumerate(self._tokenized):
            if not doc_tokens or self._doc_ids[i] is None:
                continue
            tf_map: dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1
            dl = len(doc_tokens)
            score = 0.0
            for t in q_tokens:
                if t not in tf_map:
                    continue
                df = self._df.get(t, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf  = tf_map[t]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl + 1e-9))
                score += idf * num / (den + 1e-9)
            scores.append((self._doc_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        if top:
            min_s = min(s for _, s in top)
            max_s = max(s for _, s in top)
            rng   = max_s - min_s
            if rng > 1e-9:
                top = [(doc_id, (s - min_s) / rng) for doc_id, s in top]
            else:
                top = [(doc_id, 1.0) for doc_id, _ in top]

        return top


# ─────────────────────────────────────────────────────────────────
#  MemorySegment
# ─────────────────────────────────────────────────────────────────

class MemorySegment:
    """One stored chunk of memory with Ebbinghaus tracking."""

    MEMORY_TYPES = ("knowledge", "dialog", "task")

    LAMBDA_INIT = {
        "knowledge": 24.0,
        "task":      15.0,
        "dialog":    10.0,
    }

    def __init__(
        self,
        text: str,
        memory_type: str,
        source_agent: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
        lambda_init: Optional[float] = None,
    ):
        assert memory_type in self.MEMORY_TYPES, f"Invalid type: {memory_type}"
        self.id           = str(uuid.uuid4())[:8]
        self.text         = text
        self.memory_type  = memory_type
        self.source_agent = source_agent
        self.vector       = vector
        self.metadata     = metadata or {}
        self.created_at   = time.time()
        self.last_accessed = time.time()
        self.lambda_forget = lambda_init or self.LAMBDA_INIT.get(memory_type, 1.0)
        self.recall_count  = 0

    def retention(self) -> float:
        """R(t) = e^(-t/λ) — t elapsed from created_at in hours."""
        elapsed_h = (time.time() - self.created_at) / 3600
        return math.exp(-elapsed_h / (self.lambda_forget + 1e-9))

    def on_recalled(self):
        """Spaced repetition: λ += 1 per recall → slower forgetting."""
        self.recall_count  += 1
        self.lambda_forget += 1.0
        self.last_accessed  = time.time()

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "text":          self.text,
            "memory_type":   self.memory_type,
            "source_agent":  self.source_agent,
            "created_at":    self.created_at,
            "last_accessed": self.last_accessed,
            "lambda_forget": self.lambda_forget,
            "recall_count":  self.recall_count,
            "retention":     round(self.retention(), 3),
            "metadata":      self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict, vector: Optional[np.ndarray] = None) -> "MemorySegment":
        seg = cls(
            text=d["text"], memory_type=d["memory_type"],
            source_agent=d["source_agent"], vector=vector,
            metadata=d.get("metadata", {}),
            lambda_init=d.get("lambda_forget", 1.0),
        )
        seg.id            = d["id"]
        seg.created_at    = d["created_at"]
        seg.last_accessed = d["last_accessed"]
        seg.lambda_forget = d["lambda_forget"]
        seg.recall_count  = d["recall_count"]
        return seg

    def __repr__(self):
        return (f"<MemorySegment id={self.id} type={self.memory_type} "
                f"ret={self.retention():.2f} λ={self.lambda_forget:.1f}>")


# ─────────────────────────────────────────────────────────────────
#  Recall — The Memory Core
# ─────────────────────────────────────────────────────────────────

class Recall:
    """
    Universal long-term memory bank for multi-agent systems.

    Retrieval: hybrid SentenceTransformer (dense) + BM25 (sparse)
      S = α·cosine + (1−α)·BM25   (α=0.6 by default)

    1. Storage    — three typed banks (knowledge/dialog/task) + BM25 index
    2. Retrieval  — hybrid dense+sparse, min-max normalised, intent-filtered
    3. Decay      — Ebbinghaus R(t)=e^(-t/λ), recall strengthens λ
    4. Dedup      — cosine threshold prevents near-duplicate storage
    5. Persistence — atomic JSON save/load across restarts
    """

    def __init__(
        self,
        forget_threshold: float = 0.05,
        dedup_threshold:  float = 0.92,
        embedder=None,
        persist_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.forget_threshold = forget_threshold
        self.dedup_threshold  = dedup_threshold
        self.persist_path     = persist_path
        self.verbose          = verbose

        model_name    = os.environ.get("RECALL_ST_MODEL", "all-MiniLM-L6-v2")
        self.embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)

        self._banks: dict[str, dict[str, MemorySegment]] = {
            "knowledge": {}, "dialog": {}, "task": {},
        }
        self._bm25: dict[str, BM25Retriever] = {
            "knowledge": BM25Retriever(),
            "dialog":    BM25Retriever(),
            "task":      BM25Retriever(),
        }
        self.stats = {"stored": 0, "retrieved": 0, "pruned": 0, "deduped": 0}

        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # ── 1. STORAGE ────────────────────────────────────────────────

    def store(
        self,
        text: str,
        memory_type: str,
        source_agent: str,
        metadata: Optional[dict] = None,
        lambda_init: Optional[float] = None,
        agent_filter: Optional[str] = None,
    ) -> Optional[MemorySegment]:
        """Store a memory segment. Returns None if deduplicated."""
        vec = self.embedder.fit_transform(text)

        if self._is_duplicate(vec, memory_type, agent_filter=source_agent):
            self.stats["deduped"] += 1
            if self.verbose:
                logger.debug("Deduped [%s] from %s: %s…", memory_type, source_agent, text[:50])
            return None

        seg = MemorySegment(
            text=text, memory_type=memory_type,
            source_agent=source_agent, vector=vec,
            metadata=metadata or {}, lambda_init=lambda_init,
        )
        self._banks[memory_type][seg.id] = seg
        self._bm25[memory_type].add(text, seg.id)
        self.stats["stored"] += 1

        if self.verbose:
            logger.info("Stored [%s] from %s: %s%s", memory_type, source_agent, text[:60], '…' if len(text) > 60 else '')

        if self.persist_path:
            self._save(self.persist_path)

        return seg

    def _is_duplicate(self, vec: np.ndarray, memory_type: str, agent_filter: Optional[str] = None) -> bool:
        bank = self._banks[memory_type]
        for seg in bank.values():
            if agent_filter and seg.source_agent != agent_filter:
                continue
            if seg.vector is None:
                continue
            if float(np.dot(vec, seg.vector)) >= self.dedup_threshold:
                return True
        return False

    # ── 2. RETRIEVAL ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        time_weight: float = 0.3,
        agent_filter: Optional[str] = None,
        alpha: float = 0.6,
        min_score: float = 0.10,
    ) -> list[MemorySegment]:
        """
        Hybrid retrieval: semantic cosine (dense) + BM25 (sparse).
        S = alpha * cosine + (1-alpha) * bm25_normalised
        """
        types     = [memory_type] if memory_type else list(self._banks.keys())
        query_vec = self.embedder.transform(query)
        dense_scores: dict[str, float] = {}
        bm25_scores:  dict[str, float] = {}

        for mtype in types:
            bank = self._banks[mtype]
            if not bank:
                continue

            for seg_id, seg in bank.items():
                if agent_filter and seg.source_agent != agent_filter:
                    continue
                if seg.vector is None:
                    continue
                cos_sim = float(np.dot(query_vec, seg.vector))
                if mtype in ("dialog", "task"):
                    elapsed_h = (time.time() - seg.created_at) / 3600
                    cos_sim  *= math.exp(-time_weight * elapsed_h)
                dense_scores[seg_id] = cos_sim

            for seg_id, bm25_score in self._bm25[mtype].query(query, top_k=top_k * 3):
                seg = bank.get(seg_id)
                if seg is None:
                    continue
                if agent_filter and seg.source_agent != agent_filter:
                    continue
                bm25_scores[seg_id] = bm25_score

        all_ids = set(dense_scores) | set(bm25_scores)
        fused: dict[str, float] = {
            seg_id: alpha * dense_scores.get(seg_id, 0.0) + (1 - alpha) * bm25_scores.get(seg_id, 0.0)
            for seg_id in all_ids
        }

        ranked  = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results = []
        for seg_id, score in ranked[:top_k]:
            if score < min_score:
                break
            seg = None
            for mtype in types:
                seg = self._banks[mtype].get(seg_id)
                if seg:
                    break
            if seg:
                seg.on_recalled()
                results.append(seg)

        self.stats["retrieved"] += len(results)
        return results

    # ── 3. PRUNE ──────────────────────────────────────────────────

    def prune_forgotten(self) -> int:
        """Remove segments below retention threshold. Syncs BM25 index."""
        pruned = 0
        for mtype, bank in self._banks.items():
            to_delete = [sid for sid, seg in bank.items() if seg.retention() < self.forget_threshold]
            for sid in to_delete:
                del bank[sid]
                self._bm25[mtype].delete(sid)
                pruned += 1
        self.stats["pruned"] += pruned
        if self.verbose and pruned:
            logger.info("Pruned %d forgotten segments", pruned)
        if pruned and self.persist_path:
            self._save(self.persist_path)
        return pruned

    # ── 4. PERSISTENCE ────────────────────────────────────────────

    def _save(self, path: str):
        """Persist all segments to JSON (vectors excluded — recomputed on load)."""
        data = {"stats": self.stats, "segments": [seg.to_dict() for seg in self.get_all()]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def _load(self, path: str):
        """Load segments from JSON, recomputing vectors via SentenceTransformer."""
        try:
            with open(path) as f:
                data = json.load(f)
            self.stats = data.get("stats", self.stats)
            loaded = 0
            for d in data.get("segments", []):
                mtype = d.get("memory_type")
                if mtype not in self._banks:
                    continue
                if d.get("retention", 1.0) < self.forget_threshold:
                    continue
                vec = self.embedder.fit_transform(d["text"])
                seg = MemorySegment.from_dict(d, vector=vec)
                self._banks[mtype][seg.id] = seg
                self._bm25[mtype].add(seg.text, seg.id)
                loaded += 1
            if self.verbose:
                logger.info("Loaded %d segments from %s", loaded, path)
        except Exception as e:
            if self.verbose:
                logger.warning("Could not load %s: %s", path, e)

    def save(self, path: Optional[str] = None):
        self._save(path or self.persist_path or "recall_memory.json")

    def load(self, path: Optional[str] = None):
        self._load(path or self.persist_path or "recall_memory.json")

    # ── Utilities ──────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "banks":             {k: len(v) for k, v in self._banks.items()},
            "total_segments":    sum(len(v) for v in self._banks.values()),
            "stats":             self.stats,
            "segments_by_agent": self._count_by_agent(),
        }

    def _count_by_agent(self) -> dict:
        counts: dict[str, int] = defaultdict(int)
        for bank in self._banks.values():
            for seg in bank.values():
                counts[seg.source_agent] += 1
        return dict(counts)

    def get_all(self, memory_type: Optional[str] = None) -> list[MemorySegment]:
        types = [memory_type] if memory_type else list(self._banks.keys())
        result = []
        for t in types:
            result.extend(self._banks[t].values())
        return sorted(result, key=lambda s: s.created_at)

    def export_json(self) -> str:
        return json.dumps([s.to_dict() for s in self.get_all()], indent=2)
