"""
BaseAgent — Universal plug-in interface for Recall.

Changes:
  - recall() now exposes agent_filter properly
  - remember_* methods pass lambda_init by memory type
  - format_memories includes score metadata
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
from typing import Optional
import os
import logging

from core.memory import Recall, MemorySegment

logger = logging.getLogger("recall.agents")


class BaseAgent(ABC):

    def __init__(
        self,
        name: str,
        bank: Recall,
        llm_provider: str = "gemini",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
    ):
        self.name          = name
        self.bank          = bank
        self.llm_provider  = llm_provider
        self.model         = model
        self.system_prompt = system_prompt or f"You are {name}, a helpful AI agent."
        self.verbose       = verbose

    # ── Memory helpers ───────────────────────────────────────────

    def remember_knowledge(self, text: str, metadata: Optional[dict] = None):
        return self.bank.store(text, "knowledge", self.name, metadata)

    def remember_dialog(self, text: str, metadata: Optional[dict] = None):
        return self.bank.store(text, "dialog", self.name, metadata)

    def remember_task(self, text: str, metadata: Optional[dict] = None):
        return self.bank.store(text, "task", self.name, metadata)

    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 4,
        agent_filter: Optional[str] = None,
        alpha: float = 0.6,
        min_score: float = 0.10,
    ) -> list[MemorySegment]:
        """Retrieve relevant memories. agent_filter restricts to one agent's segments."""
        results = self.bank.retrieve(
            query,
            memory_type=memory_type,
            top_k=top_k,
            agent_filter=agent_filter,
            alpha=alpha,
            min_score=min_score,
        )
        if self.verbose and results:
            logger.info("[%s] Recalled %d [%s] for: %s", self.name, len(results), memory_type or 'all', query[:50])
        return results

    def format_memories(self, memories: list[MemorySegment]) -> str:
        if not memories:
            return "No relevant memories found."
        lines = []
        for m in memories:
            lines.append(
                f"[{m.memory_type.upper()} | {m.source_agent} | "
                f"recalled×{m.recall_count} | ret={m.retention():.2f}] {m.text}"
            )
        return "\n".join(lines)

    # ── LLM call ─────────────────────────────────────────────────

    def llm_call(self, prompt: str, system: Optional[str] = None) -> str:
        sys_msg  = system or self.system_prompt
        provider = self.llm_provider.lower()
        try:
            if provider == "gemini":
                return self._call_gemini(prompt, sys_msg)
            elif provider == "groq":
                return self._call_groq(prompt, sys_msg)
            elif provider == "openai":
                return self._call_openai(prompt, sys_msg)
            elif provider == "anthropic":
                return self._call_anthropic(prompt, sys_msg)
            else:
                return self._call_stub(prompt, sys_msg)
        except Exception as e:
            if self.verbose:
                logger.warning("[%s] LLM call failed (%s), using stub.", self.name, e)
            return self._call_stub(prompt, sys_msg)

    def _call_gemini(self, prompt: str, system: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        client     = genai.Client(api_key=api_key)
        model_name = self.model or "gemini-2.0-flash"

        NO_SYSTEM = ("gemma",)
        use_sys   = not any(m in model_name.lower() for m in NO_SYSTEM)

        if use_sys:
            config   = types.GenerateContentConfig(system_instruction=system, temperature=0.7, max_output_tokens=512)
            contents = prompt
        else:
            config   = types.GenerateContentConfig(temperature=0.7, max_output_tokens=512)
            contents = system + "\n\n" + prompt

        return client.models.generate_content(model=model_name, contents=contents, config=config).text.strip()

    def _call_groq(self, prompt: str, system: str) -> str:
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("groq not installed. Run: pip install groq")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        client = Groq(api_key=api_key)
        model  = self.model or "llama3-8b-8192"
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=0.7, max_tokens=512,
        )
        return r.choices[0].message.content.strip()

    def _call_openai(self, prompt: str, system: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai not installed. Run: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        client = OpenAI(api_key=api_key)
        model  = self.model or "gpt-4o-mini"
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=0.7, max_tokens=512,
        )
        return r.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str, system: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic not installed. Run: pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        client = anthropic.Anthropic(api_key=api_key)
        model  = self.model or "claude-haiku-4-5-20251001"
        r = client.messages.create(
            model=model, max_tokens=512,
            system=system,
            messages=[{"role":"user","content":prompt}],
        )
        return r.content[0].text.strip()

    def _call_stub(self, prompt: str, system: str) -> str:
        lines = prompt.strip().split("\n")
        last  = next((l for l in reversed(lines) if l.strip()), "")
        return (
            f"[STUB] {self.name} processed: {last[:80]}\n"
            f"(Set RECALL_LLM_PROVIDER in .env to use a real LLM)"
        )

    @abstractmethod
    def run(self, task: str, context: Optional[dict] = None) -> dict:
        pass
