"""
Customer Support Demo Agents
─────────────────────────────
Three agents sharing a single Recall instance:

  1. IntakeAgent      — Classifies intent, then recalls relevant past dialogs
  2. KnowledgeAgent   — Searches all memory banks, builds answer context
  3. ResponseAgent    — Drafts reply using context passed from KnowledgeAgent
                        (no independent recall — avoids double-counting)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, re, logging
from typing import Optional
from agents.base_agent import BaseAgent
from core.memory import Recall

logger = logging.getLogger("recall.agents.customer_support")


# ────────────────────────────────────────────────────────────────
#  1. IntakeAgent
# ────────────────────────────────────────────────────────────────

class IntakeAgent(BaseAgent):
    """
    First stage: classify intent, THEN recall past dialogs using the
    extracted summary (not the raw message) to improve retrieval quality.
    """

    SYSTEM_PROMPT = """You are an intake specialist for a customer support team.
Classify the customer message and respond ONLY in this JSON format:
{
  "intent": "<billing|technical|account|refund|general>",
  "entities": {"product": "...", "issue": "...", "urgency": "low|medium|high"},
  "summary": "<one sentence summary of the core issue>"
}"""

    def __init__(self, bank: Recall, llm_provider: str = "stub", **kwargs):
        super().__init__(name="IntakeAgent", bank=bank,
                         llm_provider=llm_provider,
                         system_prompt=self.SYSTEM_PROMPT, **kwargs)

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        logger.info("IntakeAgent processing: %s", task[:80])

        # Step 1 — classify first (no recall yet)
        raw    = self.llm_call(f'Customer message: "{task}"\n\nClassify this message.')
        parsed = self._parse_intake(raw, task)
        intent  = parsed["intent"]
        summary = parsed["summary"]

        # Step 2 — recall using the extracted summary, not the raw message
        #           This gives much more relevant results
        past_dialogs  = self.recall(f"{intent}: {summary}", memory_type="dialog", top_k=3)
        memory_ctx    = self.format_memories(past_dialogs)

        # Step 3 — store this dialog exchange
        dialog_text = f"Customer [{intent}]: {task} | Summary: {summary}"
        new_dialog = self.remember_dialog(dialog_text, metadata={
            "intent": intent,
            "urgency": parsed["entities"].get("urgency", "medium"),
            "used_memory_ids":   [m.id   for m in past_dialogs],
            "used_memory_texts": [m.text[:80] for m in past_dialogs],
        })

        logger.info("IntakeAgent -> intent=%s, urgency=%s", intent, parsed['entities'].get('urgency','?'))
        return {
            "agent":             self.name,
            "output":            parsed,
            "raw_llm":           raw,
            "used_memory_ids":   [m.id   for m in past_dialogs],
            "used_memory_texts": [m.text for m in past_dialogs],
            # Pass the ID of the dialog we just stored so KnowledgeAgent can exclude it
            "current_dialog_id": new_dialog.id if new_dialog else None,
        }

    def _parse_intake(self, raw: str, original: str) -> dict:
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        intent  = "general"
        for kw, label in [
            ("billing","billing"),("invoice","billing"),("payment","billing"),("charge","billing"),
            ("error","technical"),("crash","technical"),("bug","technical"),("api","technical"),("429","technical"),
            ("refund","refund"),("cancel","account"),("password","account"),("login","account"),
        ]:
            if kw in original.lower():
                intent = label
                break
        urgency = "high" if any(w in original.lower() for w in ["urgent","asap","immediately","broken","down"]) else "medium"
        return {
            "intent":   intent,
            "entities": {"product": "unknown", "issue": original[:60], "urgency": urgency},
            "summary":  original[:120],
        }


# ────────────────────────────────────────────────────────────────
#  2. KnowledgeAgent
# ────────────────────────────────────────────────────────────────

class KnowledgeAgent(BaseAgent):
    """
    Second stage: searches all memory banks and synthesises context.
    Passes all retrieved memory metadata to the pipeline for the UI.
    """

    SYSTEM_PROMPT = """You are a knowledge retrieval specialist.
Given a customer issue and retrieved memory context, synthesise the most
relevant information to help resolve the issue.
Be concise and factual. 3-5 sentences max. Plain text only."""

    def __init__(self, bank: Recall, llm_provider: str = "stub", **kwargs):
        super().__init__(name="KnowledgeAgent", bank=bank,
                         llm_provider=llm_provider,
                         system_prompt=self.SYSTEM_PROMPT, **kwargs)

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        context = context or {}
        intake  = context.get("intake", {})
        intent  = intake.get("intent", "general") if isinstance(intake, dict) else "general"
        summary = intake.get("summary", task)      if isinstance(intake, dict) else task

        logger.info("KnowledgeAgent processing: %s", summary[:60])

        # Search all three banks using enriched query
        # Exclude the dialog segment IntakeAgent just stored this run (it's noise, not history)
        current_dialog_id = context.get("current_dialog_id")
        query         = f"{intent}: {summary}"

        # Thresholds are embedder-aware:
        #   SentenceTransformer — real semantic cosine (0.3–0.7 for related, 0.05–0.15 unrelated)
        #     → low threshold fine, semantics already do the filtering
        #   StableHashEmbedder  — bag-of-words random projections, BM25 dominates scoring
        #     → needs higher threshold to cut false positives from shared stopwords
        is_semantic = "Sentence" in type(self.bank.embedder).__name__
        k_min  = 0.03 if is_semantic else 0.08   # knowledge: broad, keep low
        dt_min = 0.05 if is_semantic else 0.10   # dialog/task: intent filter handles the rest

        knowledge_mems  = self.recall(query, memory_type="knowledge", top_k=3, min_score=k_min)
        # Pull extra candidates so intent + current_dialog filters don't leave us empty
        task_mems_raw   = self.recall(query, memory_type="task",   top_k=6, min_score=dt_min)
        dialog_mems_raw = self.recall(query, memory_type="dialog", top_k=6, min_score=dt_min)

        # Intent-filter: skip segments whose stored intent clearly doesn't match.
        # Segments without stored intent (e.g. seeded facts) are always kept.
        # "refund" is treated as compatible with "billing" — same domain.
        COMPATIBLE = {
            "billing": {"billing", "refund"},
            "refund":  {"billing", "refund"},
            "account": {"account"},
            "technical": {"technical"},
            "general": None,  # None = accept any
        }
        allowed = COMPATIBLE.get(intent)  # None means accept all

        def intent_ok(seg) -> bool:
            seg_intent = seg.metadata.get("intent")
            if seg_intent is None:
                return True   # no stored intent — keep it
            if allowed is None:
                return True   # general query — keep everything
            return seg_intent in allowed

        task_mems   = [m for m in task_mems_raw   if intent_ok(m)][:2]
        # Filter both by intent AND exclude the dialog stored this very run
        dialog_mems = [m for m in dialog_mems_raw
                       if intent_ok(m) and m.id != current_dialog_id][:2]

        all_mems   = knowledge_mems + task_mems + dialog_mems
        used_ids   = [m.id   for m in all_mems]
        used_texts = [m.text for m in all_mems]

        all_context = (
            "=== Knowledge Base ===\n"   + self.format_memories(knowledge_mems) + "\n\n"
            "=== Past Resolutions ===\n" + self.format_memories(task_mems)      + "\n\n"
            "=== Similar Dialogs ===\n"  + self.format_memories(dialog_mems)
        )

        prompt = (
            f"Customer issue (intent: {intent}): {summary}\n\n"
            f"Retrieved memory context:\n{all_context}\n\n"
            f"Synthesise the relevant information to help resolve this issue."
        )
        synthesis = self.llm_call(prompt)

        # Store synthesis as task memory
        self.remember_task(
            f"Issue [{intent}]: {summary} | Context: {synthesis[:200]}",
            metadata={
                "intent":            intent,
                "used_memory_ids":   used_ids,
                "used_memory_texts": [t[:80] for t in used_texts],
            },
        )

        logger.info("KnowledgeAgent -> used %d memories (k:%d t:%d d:%d)",
                     len(all_mems), len(knowledge_mems), len(task_mems), len(dialog_mems))
        return {
            "agent":  self.name,
            "output": synthesis,
            "memories_used": {
                "knowledge": len(knowledge_mems),
                "task":      len(task_mems),
                "dialog":    len(dialog_mems),
            },
            "used_memory_ids":   used_ids,
            "used_memory_texts": used_texts,
            # Pass synthesised context forward so ResponseAgent doesn't need to re-recall
            "knowledge_context": synthesis,
        }


# ────────────────────────────────────────────────────────────────
#  3. ResponseAgent
# ────────────────────────────────────────────────────────────────

class ResponseAgent(BaseAgent):
    """
    Final stage: drafts customer-facing reply.

    Uses knowledge_context passed through the pipeline context — does NOT
    independently recall from memory. This avoids double-counting recall
    stats and keeps the Ebbinghaus tracking accurate.
    """

    SYSTEM_PROMPT = """You are a friendly and professional customer support representative.
Write a helpful reply to the customer using the context provided.
- Acknowledge their issue warmly
- Provide a clear, actionable solution or next step
- Keep it concise (3-5 sentences)
- End with an offer to help further
- Do NOT mention internal systems, AI, or memory"""

    def __init__(self, bank: Recall, llm_provider: str = "stub", **kwargs):
        super().__init__(name="ResponseAgent", bank=bank,
                         llm_provider=llm_provider,
                         system_prompt=self.SYSTEM_PROMPT, **kwargs)

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        context  = context or {}
        intake   = context.get("intake", {})
        summary  = intake.get("summary", task)      if isinstance(intake, dict) else task
        intent   = intake.get("intent", "general")  if isinstance(intake, dict) else "general"
        urgency  = intake.get("entities", {}).get("urgency", "medium") if isinstance(intake, dict) else "medium"

        # Use knowledge context from KnowledgeAgent — no independent recall
        knowledge = context.get("knowledge_context") or context.get("knowledge", "")

        logger.info("ResponseAgent processing: intent=%s, urgency=%s", intent, urgency)

        prompt = (
            f"Customer message: {task}\n"
            f"Summary: {summary}\n"
            f"Intent: {intent} | Urgency: {urgency}\n\n"
            f"Knowledge context:\n{knowledge}\n\n"
            f"Write the customer-facing reply."
        )
        response = self.llm_call(prompt)

        logger.info("ResponseAgent -> %d chars", len(response))
        return {
            "agent":           self.name,
            "output":          response,
            "used_memory_ids": [],   # ResponseAgent doesn't recall independently
        }
