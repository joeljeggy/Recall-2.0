"""
ChatbotAgent — Conversational agent with Recall long-term memory.

Only recalls its own past dialog exchanges — deliberately isolated from
the customer support knowledge bank so it doesn't bleed irrelevant facts.
Tracks which memory segments were used per message for history display.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from agents.base_agent import BaseAgent
from core.memory import Recall


class ChatbotAgent(BaseAgent):

    SYSTEM_PROMPT = (
        "You are a helpful, friendly AI assistant with long-term memory.\n"
        "You remember past conversations and can reference them naturally.\n"
        "When you recall something from a past conversation, weave it in naturally without announcing it.\n"
        "IMPORTANT: Only use information the user has actually told you. Never guess or infer personal details.\n"
        "If you don't know something (like the user's name), just say so honestly.\n"
        "Be concise, warm, and genuinely helpful."
    )

    def __init__(self, bank: Recall, llm_provider: str = "stub", **kwargs):
        super().__init__(
            name="ChatbotAgent",
            bank=bank,
            llm_provider=llm_provider,
            system_prompt=self.SYSTEM_PROMPT,
            **kwargs,
        )
        self._session_history: list[dict] = []

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        # Only recall this agent's own past dialog
        all_dialogs  = self.recall(task, memory_type="dialog", top_k=5)
        past_dialogs = [m for m in all_dialogs if m.source_agent == self.name]

        # Capture IDs of memories actually used
        used_memory_ids   = [m.id   for m in past_dialogs]
        used_memory_texts = [m.text for m in past_dialogs]

        # Build memory context string
        memory_ctx = ""
        if past_dialogs:
            memory_ctx = "Relevant past conversations:\n" + self.format_memories(past_dialogs)

        # Build session history string (last 3 exchanges)
        history_str = ""
        if self._session_history:
            lines = []
            for t in self._session_history[-6:]:
                role = "User" if t["role"] == "user" else "Assistant"
                lines.append(role + ": " + t["content"])
            history_str = "\n".join(lines)

        # Assemble prompt
        parts = []
        if memory_ctx:
            parts.append("[Relevant past conversations]\n" + memory_ctx)
        if history_str:
            parts.append("[Current conversation]\n" + history_str)
        parts.append("User: " + task + "\nAssistant:")
        prompt = "\n\n".join(parts)

        response = self.llm_call(prompt)

        # Store exchange with metadata about which memories were used
        exchange = "User: " + task + " | Assistant: " + response[:200]
        seg = self.remember_dialog(exchange, metadata={
            "agent":            self.name,
            "used_memory_ids":  used_memory_ids,
            "used_memory_texts": [t[:80] for t in used_memory_texts],
            "user_message":     task,
        })

        self._session_history.append({"role": "user",      "content": task})
        self._session_history.append({"role": "assistant", "content": response})

        return {
            "agent":             self.name,
            "output":            response,
            "memories_used":     len(past_dialogs),
            "used_memory_ids":   used_memory_ids,
            "used_memory_texts": used_memory_texts,
            "stored_seg_id":     seg.id,
        }

    def clear_session(self):
        """Clear current session history (long-term memory is preserved)."""
        self._session_history = []
