"""
DocGeneratorAgent — Generates structured documents from natural language prompts.

Produces JSON document structure which the web layer converts to .docx,
also stores document history in Recall task memory.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
from typing import Optional
from agents.base_agent import BaseAgent
from core.memory import Recall


class DocGeneratorAgent(BaseAgent):

    SYSTEM_PROMPT = """You are a professional document writer. Given a prompt, generate a well-structured document.

ALWAYS respond with valid JSON only — no markdown, no explanation, just the JSON object:
{
  "title": "Document Title",
  "type": "report|letter|memo|proposal|summary|guide",
  "sections": [
    {
      "heading": "Section Title",
      "content": "Section body text. Can be multiple sentences.",
      "bullets": ["optional", "bullet", "points"]
    }
  ],
  "metadata": {
    "author": "Recall AI",
    "date": "today",
    "summary": "One sentence summary of the document"
  }
}

Write professional, complete content. Minimum 3 sections. Make it genuinely useful."""

    def __init__(self, bank: Recall, llm_provider: str = "stub", **kwargs):
        super().__init__(
            name="DocGeneratorAgent",
            bank=bank,
            llm_provider=llm_provider,
            system_prompt=self.SYSTEM_PROMPT,
            **kwargs,
        )

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        # Recall any similar past documents for context
        past_docs = self.recall(task, memory_type="task", top_k=2)
        memory_ctx = self.format_memories(past_docs) if past_docs else ""

        prompt = f"Generate a document for: {task}"
        if memory_ctx:
            prompt += f"\n\nFor reference, similar past documents:\n{memory_ctx}"

        raw = self.llm_call(prompt)
        doc_data = self._parse_doc(raw, task)

        # Store in task memory
        summary = doc_data.get("metadata", {}).get("summary", task[:100])
        self.remember_task(
            f"Generated document: '{doc_data['title']}' — {summary}",
            metadata={"doc_type": doc_data.get("type", "document"), "prompt": task[:100]},
        )

        return {
            "agent":  self.name,
            "output": doc_data,
            "raw":    raw,
        }

    def _parse_doc(self, raw: str, original_prompt: str) -> dict:
        """Parse LLM JSON output, with fallback structure."""
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        # Fallback: build minimal structure from stub/failed response
        return {
            "title": f"Document: {original_prompt[:60]}",
            "type": "report",
            "sections": [
                {
                    "heading": "Overview",
                    "content": f"This document addresses: {original_prompt}",
                    "bullets": []
                },
                {
                    "heading": "Details",
                    "content": raw[:400] if raw else "Content will appear here once an LLM provider is configured.",
                    "bullets": []
                },
                {
                    "heading": "Next Steps",
                    "content": "Review the document and make any necessary adjustments.",
                    "bullets": ["Review content", "Share with stakeholders", "Implement recommendations"]
                }
            ],
            "metadata": {
                "author": "Recall AI",
                "date": time.strftime("%Y-%m-%d"),
                "summary": f"Document generated for: {original_prompt[:80]}"
            }
        }
