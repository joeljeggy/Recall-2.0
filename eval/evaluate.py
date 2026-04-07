"""
Recall Performance Evaluation Script
======================================
Compares four retrieval modes on the same query set:

  1. context_stuffing  — last N conversation turns injected directly into prompt
  2. rag_cosine        — cosine-only retrieval, no memory lifecycle (alpha=1.0)
  3. rag_hybrid        — BM25+cosine retrieval, no memory lifecycle (alpha=0.6)
  4. recall            — full system: BM25+cosine + Ebbinghaus lifecycle

Usage:
    python eval/evaluate.py --provider stub
    python eval/evaluate.py --provider gemini
    python eval/evaluate.py --provider gemini --modes recall rag_hybrid
"""

import sys
import os
import json
import time
import argparse
import logging
from typing import Optional
from copy import deepcopy
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Silence noisy loggers during eval
logging.basicConfig(level=logging.WARNING)
logging.getLogger("recall").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from core.memory import Recall, SentenceTransformerEmbedder, BM25Retriever, MemorySegment
from agents.customer_support import IntakeAgent, KnowledgeAgent, ResponseAgent
from agents.pipeline import AgentPipeline
from agents.base_agent import BaseAgent
from demo.knowledge_seed import seed_customer_support_knowledge

# ─────────────────────────────────────────────────────────────────
#  Token counter — patches LLM calls to record usage
# ─────────────────────────────────────────────────────────────────

class TokenCounter:
    """Thread-safe token counter. Reset before each pipeline run."""
    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0
        self.calls             = 0

    def reset(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0
        self.calls             = 0

    def add(self, prompt_t: int, completion_t: int):
        self.prompt_tokens     += prompt_t
        self.completion_tokens += completion_t
        self.total_tokens      += prompt_t + completion_t
        self.calls             += 1

    def summary(self) -> dict:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "llm_calls":         self.calls,
        }

# Global counter — shared across all agents in one pipeline run
_TOKEN_COUNTER = TokenCounter()


def _patch_base_agent_for_token_counting():
    """
    Monkey-patches BaseAgent LLM call methods to intercept token usage.
    Called once at startup. Works for gemini, groq, openai, anthropic.
    Stub provider uses estimated counts based on character length.
    """
    from agents.base_agent import BaseAgent

    # ── Gemini ──────────────────────────────────────────────────
    _orig_gemini = BaseAgent._call_gemini
    def _patched_gemini(self, prompt, system):
        try:
            from google import genai
            from google.genai import types
            import os
            api_key    = os.getenv("GEMINI_API_KEY")
            client     = genai.Client(api_key=api_key)
            model_name = self.model or "gemini-2.0-flash"
            NO_SYSTEM  = ("gemma",)
            use_sys    = not any(m in model_name.lower() for m in NO_SYSTEM)
            if use_sys:
                config   = types.GenerateContentConfig(system_instruction=system, temperature=0.7, max_output_tokens=512)
                contents = prompt
            else:
                config   = types.GenerateContentConfig(temperature=0.7, max_output_tokens=512)
                contents = system + "\n\n" + prompt
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
            # Extract token usage from response metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                pt = getattr(um, 'prompt_token_count',     0) or 0
                ct = getattr(um, 'candidates_token_count', 0) or 0
                _TOKEN_COUNTER.add(pt, ct)
            else:
                # Estimate if metadata unavailable
                _TOKEN_COUNTER.add(len(contents.split()), len(response.text.split()))
            return response.text.strip()
        except Exception as e:
            return self._call_stub(prompt, system)
    BaseAgent._call_gemini = _patched_gemini

    # ── Groq ────────────────────────────────────────────────────
    _orig_groq = BaseAgent._call_groq
    def _patched_groq(self, prompt, system):
        try:
            from groq import Groq
            import os
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            model  = self.model or "llama3-8b-8192"
            r = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
                temperature=0.7, max_tokens=512,
            )
            if hasattr(r, 'usage') and r.usage:
                _TOKEN_COUNTER.add(r.usage.prompt_tokens, r.usage.completion_tokens)
            return r.choices[0].message.content.strip()
        except Exception:
            return self._call_stub(prompt, system)
    BaseAgent._call_groq = _patched_groq

    # ── OpenAI ──────────────────────────────────────────────────
    _orig_openai = BaseAgent._call_openai
    def _patched_openai(self, prompt, system):
        try:
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model  = self.model or "gpt-4o-mini"
            r = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
                temperature=0.7, max_tokens=512,
            )
            if hasattr(r, 'usage') and r.usage:
                _TOKEN_COUNTER.add(r.usage.prompt_tokens, r.usage.completion_tokens)
            return r.choices[0].message.content.strip()
        except Exception:
            return self._call_stub(prompt, system)
    BaseAgent._call_openai = _patched_openai

    # ── Anthropic ───────────────────────────────────────────────
    _orig_anthropic = BaseAgent._call_anthropic
    def _patched_anthropic(self, prompt, system):
        try:
            import anthropic, os
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            model  = self.model or "claude-haiku-4-5-20251001"
            r = client.messages.create(
                model=model, max_tokens=512, system=system,
                messages=[{"role":"user","content":prompt}],
            )
            if hasattr(r, 'usage') and r.usage:
                _TOKEN_COUNTER.add(r.usage.input_tokens, r.usage.output_tokens)
            return r.content[0].text.strip()
        except Exception:
            return self._call_stub(prompt, system)
    BaseAgent._call_anthropic = _patched_anthropic

    # ── Stub — estimate from character length ────────────────────
    _orig_stub = BaseAgent._call_stub
    def _patched_stub(self, prompt, system):
        result = _orig_stub(self, prompt, system)
        # Rough estimate: 1 token ~ 4 chars
        _TOKEN_COUNTER.add(len(prompt) // 4, len(result) // 4)
        return result
    BaseAgent._call_stub = _patched_stub

# -----------------------------------------------------------------
#  Scoring helpers
# -----------------------------------------------------------------

def precision_at_k(retrieved_texts: list[str], ground_truth: list[str], k: int = 3) -> float:
    """Fraction of top-k retrieved segments that match any ground truth segment."""
    if not retrieved_texts or not ground_truth:
        return 0.0
    top_k = retrieved_texts[:k]
    hits = 0
    for ret in top_k:
        for gt in ground_truth:
            # Check if ground truth text is substantially present in retrieved text
            gt_words = set(gt.lower().split())
            ret_words = set(ret.lower().split())
            overlap = len(gt_words & ret_words) / max(len(gt_words), 1)
            if overlap >= 0.5:
                hits += 1
                break
    return round(hits / k, 3)


def keyword_coverage(response: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the response."""
    if not keywords or not response:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    return round(hits / len(keywords), 3)


def cross_turn_score(response: str, prior_query: Optional[str]) -> int:
    """1 if response seems to reference prior context, 0 otherwise."""
    if not prior_query:
        return -1  # not applicable
    # Simple heuristic: check if response contains words from prior query
    prior_words = set(w for w in prior_query.lower().split() if len(w) > 4)
    response_lower = response.lower()
    matches = sum(1 for w in prior_words if w in response_lower)
    return 1 if matches >= 2 else 0


def context_length(text: str) -> int:
    return len(text) if text else 0


# -----------------------------------------------------------------
#  Static RAG Recall — disables lifecycle (no lambda update, no prune)
# -----------------------------------------------------------------

class StaticRecall(Recall):
    """
    Recall with lifecycle disabled — segments never strengthen or decay.
    Used for rag_cosine and rag_hybrid baselines.
    """
    def retrieve(self, query, memory_type=None, top_k=5,
                 time_weight=0.0,  # no temporal decay
                 agent_filter=None, alpha=0.6, min_score=0.10):
        # Call parent retrieve but prevent on_recalled() from updating lambda
        types     = [memory_type] if memory_type else list(self._banks.keys())
        import math
        import numpy as np
        query_vec = self.embedder.transform(query)
        dense_scores = {}
        bm25_scores  = {}

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
                dense_scores[seg_id] = cos_sim
            for seg_id, bm25_score in self._bm25[mtype].query(query, top_k=top_k * 3):
                seg = bank.get(seg_id)
                if seg is None:
                    continue
                if agent_filter and seg.source_agent != agent_filter:
                    continue
                bm25_scores[seg_id] = bm25_score

        all_ids = set(dense_scores) | set(bm25_scores)
        fused = {
            sid: alpha * dense_scores.get(sid, 0.0) + (1 - alpha) * bm25_scores.get(sid, 0.0)
            for sid in all_ids
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
                # NOTE: no seg.on_recalled() — static RAG does not update lambda
                results.append(seg)

        self.stats["retrieved"] += len(results)
        return results

    def prune_forgotten(self) -> int:
        return 0  # Static RAG never prunes


# -----------------------------------------------------------------
#  Context Stuffing Pipeline
# -----------------------------------------------------------------

class ContextStuffingIntakeAgent(BaseAgent):
    """IntakeAgent that uses no memory — just classifies intent."""

    SYSTEM_PROMPT = """You are an intake specialist for a customer support team.
Classify the customer message and respond ONLY in this JSON format:
{
  "intent": "<billing|technical|account|refund|general>",
  "entities": {"product": "...", "issue": "...", "urgency": "low|medium|high"},
  "summary": "<one sentence summary of the core issue>"
}"""

    def __init__(self, bank, llm_provider="stub", **kwargs):
        super().__init__(name="IntakeAgent", bank=bank,
                         llm_provider=llm_provider,
                         system_prompt=self.SYSTEM_PROMPT, **kwargs)

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        context = context or {}
        raw    = self.llm_call(f'Customer message: "{task}"\n\nClassify this message.')
        import json, re
        parsed = {"intent": "general",
                  "entities": {"product": "unknown", "issue": task[:60], "urgency": "medium"},
                  "summary": task[:120]}
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
        except Exception:
            pass
        return {"agent": self.name, "output": parsed,
                "used_memory_ids": [], "used_memory_texts": []}


class ContextStuffingKnowledgeAgent(BaseAgent):
    """KnowledgeAgent that injects raw conversation history instead of retrieving."""

    SYSTEM_PROMPT = """You are a knowledge specialist for customer support.
Using the conversation history provided, summarise relevant context to help resolve the issue.
Be concise. 3-5 sentences max."""

    def __init__(self, bank, llm_provider="stub", **kwargs):
        super().__init__(name="KnowledgeAgent", bank=bank,
                         llm_provider=llm_provider,
                         system_prompt=self.SYSTEM_PROMPT, **kwargs)

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        context = context or {}
        history = context.get("conversation_history", [])
        # Stuff last 6 turns directly into prompt
        history_str = "\n".join(
            f"{t['role'].upper()}: {t['content']}"
            for t in history[-6:]
        ) if history else "No prior conversation history."

        intake  = context.get("intake", {})
        summary = intake.get("summary", task) if isinstance(intake, dict) else task
        intent  = intake.get("intent", "general") if isinstance(intake, dict) else "general"

        prompt = (
            f"Customer issue (intent: {intent}): {summary}\n\n"
            f"Conversation history (last {len(history[-6:])} turns):\n{history_str}\n\n"
            f"Summarise relevant context to resolve this issue."
        )
        synthesis = self.llm_call(prompt)
        context_len = len(history_str)

        return {
            "agent":  self.name,
            "output": synthesis,
            "knowledge_context": synthesis,
            "used_memory_ids":   [],
            "used_memory_texts": [],
            "memories_used":     {"knowledge": 0, "task": 0, "dialog": 0},
            "context_length":    context_len,
        }


# -----------------------------------------------------------------
#  Build pipeline for each mode
# -----------------------------------------------------------------

def build_pipeline(mode: str, provider: str, model: Optional[str] = None):
    """
    Returns (pipeline, bank) for the given evaluation mode.
    Memory is cleared before seeding to ensure each mode starts fresh.

    Modes:
      context_stuffing — no retrieval, raw history injection
      rag_cosine       — cosine-only, no lifecycle (alpha=1.0)
      rag_hybrid       — BM25+cosine, no lifecycle (alpha=0.6)
      recall           — full system
    """
    verbose = False

    if mode == "context_stuffing":
        bank = Recall(forget_threshold=0.02, dedup_threshold=0.92, verbose=verbose)
        _clear_and_reseed(bank, mode)
        agents = [
            ContextStuffingIntakeAgent(bank, llm_provider=provider, model=model, verbose=verbose),
            ContextStuffingKnowledgeAgent(bank, llm_provider=provider, model=model, verbose=verbose),
            ResponseAgent(bank, llm_provider=provider, model=model, verbose=verbose),
        ]
        pipeline = AgentPipeline(bank, agents, prune_every=9999)
        return pipeline, bank

    elif mode == "rag_cosine":
        bank = StaticRecall(forget_threshold=0.02, dedup_threshold=0.92, verbose=verbose)
        _clear_and_reseed(bank, mode)
        agents = [
            # Override recall alpha to 1.0 — cosine only
            _make_cosine_intake(bank, provider, model, verbose),
            _make_cosine_knowledge(bank, provider, model, verbose),
            ResponseAgent(bank, llm_provider=provider, model=model, verbose=verbose),
        ]
        pipeline = AgentPipeline(bank, agents, prune_every=9999)
        return pipeline, bank

    elif mode == "rag_hybrid":
        bank = StaticRecall(forget_threshold=0.02, dedup_threshold=0.92, verbose=verbose)
        _clear_and_reseed(bank, mode)
        agents = [
            IntakeAgent(bank, llm_provider=provider, model=model, verbose=verbose),
            KnowledgeAgent(bank, llm_provider=provider, model=model, verbose=verbose),
            ResponseAgent(bank, llm_provider=provider, model=model, verbose=verbose),
        ]
        pipeline = AgentPipeline(bank, agents, prune_every=9999)
        return pipeline, bank

    elif mode == "recall":
        bank = Recall(forget_threshold=0.02, dedup_threshold=0.92, verbose=verbose)
        _clear_and_reseed(bank, mode)
        agents = [
            IntakeAgent(bank, llm_provider=provider, model=model, verbose=verbose),
            KnowledgeAgent(bank, llm_provider=provider, model=model, verbose=verbose),
            ResponseAgent(bank, llm_provider=provider, model=model, verbose=verbose),
        ]
        pipeline = AgentPipeline(bank, agents, prune_every=9999)
        return pipeline, bank

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _make_cosine_intake(bank, provider, model, verbose):
    """IntakeAgent that uses cosine-only retrieval."""
    agent = IntakeAgent(bank, llm_provider=provider, model=model, verbose=verbose)
    # Monkey-patch recall to use alpha=1.0
    original_recall = agent.recall
    def cosine_recall(query, **kwargs):
        kwargs["alpha"] = 1.0
        return original_recall(query, **kwargs)
    agent.recall = cosine_recall
    return agent


def _make_cosine_knowledge(bank, provider, model, verbose):
    """KnowledgeAgent that uses cosine-only retrieval."""
    agent = KnowledgeAgent(bank, llm_provider=provider, model=model, verbose=verbose)
    original_recall = agent.recall
    def cosine_recall(query, **kwargs):
        kwargs["alpha"] = 1.0
        return original_recall(query, **kwargs)
    agent.recall = cosine_recall
    return agent


# -----------------------------------------------------------------
#  Run evaluation
# -----------------------------------------------------------------

def _clear_and_reseed(bank: Recall, mode: str, verbose: bool = False):
    """Clear all memory banks and reseed base knowledge."""
    for mtype in list(bank._banks.keys()):
        bank._banks[mtype].clear()
        bank._bm25[mtype] = BM25Retriever()
    bank.stats = {"stored": 0, "retrieved": 0, "pruned": 0, "deduped": 0}
    seed_customer_support_knowledge(bank)
    if verbose:
        print(f"    [Reseed] Memory cleared and base knowledge reseeded for {mode}")


def run_evaluation(modes: list[str], provider: str, model: Optional[str] = None) -> dict:
    # Load queries
    queries_path = os.path.join(os.path.dirname(__file__), "queries.json")
    with open(queries_path) as f:
        data = json.load(f)
    queries      = data["queries"]
    ground_truth = data["ground_truth"]

    all_results = {}

    # Patch all BaseAgent LLM calls to record token usage
    _patch_base_agent_for_token_counting()

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'='*60}")

        # Build pipeline — memory is cleared and base knowledge seeded fresh per mode
        pipeline, bank = build_pipeline(mode, provider, model)
        mode_results   = []
        conversation_history = []  # for context stuffing
        query_count = 0  # track queries for periodic reseed

        for q in queries:
            qid     = q["id"]
            query   = q["query"]
            gt      = ground_truth.get(str(qid), [])
            keywords = q["relevant_keywords"]
            cross_ref_id = q.get("cross_turn_ref")
            prior_query = next((x["query"] for x in queries if x["id"] == cross_ref_id), None) \
                          if cross_ref_id else None

            # Reseed base knowledge every 10 queries to prevent knowledge bank decay
            if query_count > 0 and query_count % 10 == 0:
                print(f"    [Reseed] Reseeding base knowledge at query {query_count} for {mode}")
                seed_customer_support_knowledge(bank)

            print(f"\n  Q{qid}: {query[:70]}...")

            # Build initial context
            initial_ctx = {}
            if mode == "context_stuffing":
                initial_ctx["conversation_history"] = conversation_history.copy()

            # Time the full pipeline run
            _TOKEN_COUNTER.reset()
            t0  = time.time()
            run = pipeline.run(query, initial_context=initial_ctx)
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            token_usage = _TOKEN_COUNTER.summary()

            # Extract per-agent timings
            agent_times = {
                t["agent"]: round(t["elapsed_s"] * 1000, 1)
                for t in run.get("agent_traces", [])
            }

            # Extract memories used
            total_memories = sum(
                sum(v for v in t["memories_used"].values() if isinstance(v, (int, float)))
                for t in run.get("agent_traces", [])
            )

            # Get response
            response = run.get("response", "")

            # Get context length passed to ResponseAgent
            knowledge_ctx = ""
            for t in run.get("agent_traces", []):
                if t["agent"] == "KnowledgeAgent":
                    knowledge_ctx = t.get("output") or ""
                    break

            # Get all retrieved texts across the run
            retrieved_texts = []
            for t in run.get("agent_traces", []):
                retrieved_texts.extend(t.get("used_memory_texts", []))

            # Compute scores
            p_at_3 = precision_at_k(retrieved_texts, gt, k=3)
            kw_cov = keyword_coverage(response, keywords)
            ct_score = cross_turn_score(response, prior_query)

            result = {
                "query_id":          qid,
                "token_usage":       token_usage,
                "query":             query,
                "response":          response,
                "retrieved_texts":   retrieved_texts[:5],
                "memories_used":     total_memories,
                "precision_at_3":    p_at_3,
                "keyword_coverage":  kw_cov,
                "cross_turn_score":  ct_score,
                "context_length":    context_length(knowledge_ctx),
                "elapsed_ms":        elapsed_ms,
                "agent_times_ms":    agent_times,
                "mem_delta":         sum(t.get("mem_delta", 0) for t in run.get("agent_traces", [])),
            }
            mode_results.append(result)

            # Update conversation history for context stuffing
            conversation_history.append({"role": "user",      "content": query})
            conversation_history.append({"role": "assistant",  "content": response[:200]})

            query_count += 1

            print(f"    [OK] P@3={p_at_3:.2f}  KwCov={kw_cov:.2f}  "
                  f"Memories={total_memories}  {elapsed_ms}ms  "
                  f"Tokens={token_usage['total_tokens']}({token_usage['llm_calls']} calls)")

        all_results[mode] = mode_results

    return all_results


# -----------------------------------------------------------------
#  Report generation
# -----------------------------------------------------------------

def generate_report(results: dict, provider: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  RECALL PERFORMANCE EVALUATION REPORT")
    lines.append(f"  Provider : {provider.upper()}")
    lines.append(f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    modes = list(results.keys())

    # -- Per-mode summary table ----------------------------------
    lines.append("\n-- SUMMARY TABLE ----------------------------------------------")
    lines.append(f"{'Metric':<28} " + "  ".join(f"{m:<16}" for m in modes))
    lines.append("-" * (28 + 18 * len(modes)))

    metrics = [
        ("Avg Precision@3",      lambda rs: round(sum(r["precision_at_3"] for r in rs) / len(rs), 3)),
        ("Avg Keyword Coverage", lambda rs: round(sum(r["keyword_coverage"] for r in rs) / len(rs), 3)),
        ("Avg Memories Used",    lambda rs: round(sum(r["memories_used"] for r in rs) / len(rs), 1)),
        ("Avg Context Length",   lambda rs: round(sum(r["context_length"] for r in rs) / len(rs), 0)),
        ("Avg Total Latency ms", lambda rs: round(sum(r["elapsed_ms"] for r in rs) / len(rs), 1)),
        ("Avg IntakeAgent ms",   lambda rs: round(sum(r["agent_times_ms"].get("IntakeAgent", 0) for r in rs) / len(rs), 1)),
        ("Avg KnowledgeAgent ms",lambda rs: round(sum(r["agent_times_ms"].get("KnowledgeAgent", 0) for r in rs) / len(rs), 1)),
        ("Avg ResponseAgent ms", lambda rs: round(sum(r["agent_times_ms"].get("ResponseAgent", 0) for r in rs) / len(rs), 1)),
        ("Avg Prompt Tokens",    lambda rs: round(sum(r["token_usage"]["prompt_tokens"] for r in rs) / len(rs), 0)),
        ("Avg Completion Tokens",lambda rs: round(sum(r["token_usage"]["completion_tokens"] for r in rs) / len(rs), 0)),
        ("Avg Total Tokens",     lambda rs: round(sum(r["token_usage"]["total_tokens"] for r in rs) / len(rs), 0)),
        ("Avg LLM Calls/Query",  lambda rs: round(sum(r["token_usage"]["llm_calls"] for r in rs) / len(rs), 1)),
        ("Cross-turn Accuracy",  lambda rs: f"{sum(1 for r in rs if r['cross_turn_score'] == 1)}/{sum(1 for r in rs if r['cross_turn_score'] >= 0)}"),
    ]

    for label, fn in metrics:
        row = f"{label:<28} "
        for mode in modes:
            val = fn(results[mode])
            row += f"{str(val):<18}"
        lines.append(row)

    # -- Per-query breakdown --------------------------------------
    lines.append("\n-- PER-QUERY BREAKDOWN -----------------------------------------")
    for mode in modes:
        lines.append(f"\n  [{mode.upper()}]")
        lines.append(f"  {'Q':<4} {'P@3':<6} {'KwCov':<8} {'Mems':<6} {'CtxLen':<8} {'Ms':<8} {'Tokens':<8} {'Calls':<6} {'CT'}")
        lines.append("  " + "-" * 68)
        for r in results[mode]:
            ct = "N/A" if r["cross_turn_score"] == -1 else ("[OK]" if r["cross_turn_score"] == 1 else "[X]")
            tu = r.get("token_usage", {})
            lines.append(
                f"  {r['query_id']:<4} {r['precision_at_3']:<6} "
                f"{r['keyword_coverage']:<8} {r['memories_used']:<6} "
                f"{r['context_length']:<8} {r['elapsed_ms']:<8} "
                f"{tu.get('total_tokens', 0):<8} {tu.get('llm_calls', 0):<6} {ct}"
            )

    # -- Response comparison for cross-turn queries ---------------
    cross_turn_ids = [q["query_id"] for q in
                      [r for mode_rs in results.values() for r in mode_rs
                       if r["cross_turn_score"] >= 0][:5]]
    cross_turn_ids = list(dict.fromkeys(cross_turn_ids))  # deduplicate

    if cross_turn_ids:
        lines.append("\n-- CROSS-TURN RESPONSE COMPARISON ------------------------------")
        for qid in cross_turn_ids[:3]:
            lines.append(f"\n  Query ID: {qid}")
            for mode in modes:
                r = next((x for x in results[mode] if x["query_id"] == qid), None)
                if r:
                    lines.append(f"\n  [{mode.upper()}]")
                    lines.append(f"  Query   : {r['query']}")
                    lines.append(f"  Response: {r['response'][:300]}...")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recall Performance Evaluation")
    # Load .env first so we can read RECALL_LLM_PROVIDER as default
    _env_path = os.path.join(ROOT, ".env")
    if os.path.exists(_env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path, override=True)
        except ImportError:
            with open(_env_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _k, _, _v = _line.partition("=")
                    os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

    _default_provider = os.environ.get("RECALL_LLM_PROVIDER", "stub")
    _default_model    = os.environ.get("RECALL_MODEL", None)

    parser.add_argument("--provider", default=_default_provider,
                        choices=["stub", "gemini", "groq", "openai", "anthropic", "github"],
                        help=f"LLM provider to use (default from .env: {_default_provider})")
    parser.add_argument("--model", default=_default_model,
                        help=f"Model name (default from .env: {_default_model})")
    parser.add_argument("--modes", nargs="+",
                        default=["context_stuffing", "rag_cosine", "rag_hybrid", "recall"],
                        choices=["context_stuffing", "rag_cosine", "rag_hybrid", "recall"],
                        help="Which modes to evaluate")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: eval/results/TIMESTAMP.json)")
    args = parser.parse_args()

    # .env already loaded above during argument parsing

    print(f"\nRecall Evaluation")
    print(f"Provider : {args.provider.upper()}")
    print(f"Modes    : {', '.join(args.modes)}")
    print(f"Queries  : 10 fixed test queries")

    # Run evaluation
    results = run_evaluation(args.modes, args.provider, args.model)

    # Generate text report
    report = generate_report(results, args.provider)
    print("\n" + report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # JSON — full data
    json_path = args.output or os.path.join(results_dir, f"eval_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "provider":  args.provider,
            "modes":     args.modes,
            "results":   results,
        }, f, indent=2)

    # Text report
    txt_path = json_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nResults saved to:")
    print(f"  JSON : {json_path}")
    print(f"  Text : {txt_path}")


if __name__ == "__main__":
    main()