"""
AgentPipeline — Orchestrates multi-agent task execution with Recall.
Captures full per-run traces for the web UI.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import uuid
from typing import Optional
from core.memory import Recall
from agents.base_agent import BaseAgent


class AgentPipeline:
    def __init__(self, bank: Recall, agents: list[BaseAgent], prune_every: int = 10):
        self.bank        = bank
        self.agents      = agents
        self.prune_every = prune_every
        self._task_count = 0
        self.run_history: list[dict] = []   # full trace of every run

    def run(self, task: str, initial_context: Optional[dict] = None) -> dict:
        self._task_count += 1
        run_id  = str(uuid.uuid4())[:8]
        context = initial_context or {}
        start   = time.time()

        agent_traces = []

        for agent in self.agents:
            agent_start = time.time()
            mem_before  = self.bank.summary()["total_segments"]

            result = agent.run(task, context=context)

            mem_after = self.bank.summary()["total_segments"]
            elapsed   = round(time.time() - agent_start, 3)

            # Normalise memories_used to always be a dict
            mu = result.get("memories_used", {})
            if isinstance(mu, int):
                mu = {"total": mu}

            agent_traces.append({
                "agent":         agent.name,
                "elapsed_s":     elapsed,
                "output":        result.get("output"),
                "memories_used": mu,
                "mem_delta":     mem_after - mem_before,
                "used_memory_ids":   result.get("used_memory_ids", []),
                "used_memory_texts": result.get("used_memory_texts", []),
            })

            # Enrich context for next agent
            context[agent.name.lower().replace("agent", "").strip()] = result.get("output")
            if agent.name == "IntakeAgent":
                context["intake"] = result["output"]
                # Pass current dialog ID so KnowledgeAgent can exclude it from recall
                if result.get("current_dialog_id"):
                    context["current_dialog_id"] = result["current_dialog_id"]
            elif agent.name == "KnowledgeAgent":
                context["knowledge"] = result["output"]
                context["knowledge_context"] = result.get("knowledge_context", result.get("output",""))

        total_elapsed = round(time.time() - start, 3)

        pruned = 0
        if self._task_count % self.prune_every == 0:
            pruned = self.bank.prune_forgotten()

        final_response = next(
            (t["output"] for t in reversed(agent_traces) if t["output"]),
            "No response generated."
        )

        run = {
            "run_id":        run_id,
            "task_number":   self._task_count,
            "task":          task,
            "response":      final_response,
            "agent_traces":  agent_traces,
            "elapsed_s":     total_elapsed,
            "pruned":        pruned,
            "memory_summary": self.bank.summary(),
            "timestamp":     time.time(),
        }
        self.run_history.append(run)
        return run

    def memory_report(self):
        summary = self.bank.summary()
        print(f"\n{'='*60}\n  RECALL MEMORY REPORT\n{'='*60}")
        print(f"  Total: {summary['total_segments']} segments")
        for btype, count in summary["banks"].items():
            print(f"    {btype:12s}: {count}")
        print(f"{'='*60}\n")
