# Recall — Vector-Based Long-Term Memory for Multi-Agent LLM Systems

Based on the paper: _"Vector Storage Based Long-term Memory Research on LLM"_ (Li et al., 2024)

## Architecture

```
Customer Query
      │
      ▼
┌─────────────┐    stores dialog     ┌──────────────┐
│ IntakeAgent │ ─────────────────▶  │              │
│  (classify) │    recalls dialog    │    Recall    │
│             │ ◀─────────────────  │  Memory Bank │
└──────┬──────┘                     │              │
       │ intent + summary            │  knowledge   │
       ▼                            │  dialog      │
┌─────────────────┐ recalls all 3  │  task        │
│ KnowledgeAgent  │ ◀────────────  │              │
│  (synthesise)   │ stores task    └──────────────┘
│                 │ ─────────────▶
└────────┬────────┘
         │ knowledge_context
         ▼
┌─────────────────┐
│ ResponseAgent   │
│  (draft reply)  │
└─────────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env              # fill in your API key
python web/app.py --provider gemini
# Open http://localhost:5000
```

## File Structure

```
Recall/
├── core/
│   ├── memory.py          ← Recall engine: embedder, BM25, Ebbinghaus
│   └── __init__.py
├── agents/
│   ├── base_agent.py      ← BaseAgent: recall/remember/llm_call helpers
│   ├── customer_support.py← IntakeAgent, KnowledgeAgent, ResponseAgent
│   └── pipeline.py        ← AgentPipeline: orchestration + run traces
├── demo/
│   └── knowledge_seed.py  ← Seeds 19 customer support facts
├── web/
│   ├── app.py             ← Flask REST API
│   └── static/
│       └── index.html     ← Single-page UI (4 pages)
├── main.py                ← CLI entry point
├── .env.example           ← API key + embedder config template
├── .gitignore
├── requirements.txt
└── README.md
```

## Memory Banks

| Bank        | Stored by                      | Content                        |
| ----------- | ------------------------------ | ------------------------------ |
| `knowledge` | Seeder / agents                | Facts, policies, documentation |
| `dialog`    | IntakeAgent                    | Customer queries and intent    |
| `task`      | KnowledgeAgent / ResponseAgent | Syntheses and resolutions      |

## What Makes This Work

- **Hybrid retrieval** — dense cosine similarity (stable hash vectors) + sparse BM25, fused with proper min-max normalisation
- **Ebbinghaus forgetting curve** — `R(t) = e^(-t/λ)`. λ starts higher for knowledge (2.0) vs dialog (1.0). Each recall increments λ, making frequently-used memories harder to forget
- **Deduplication** — segments with cosine similarity > 0.92 against existing segments are rejected at write time
- **Persistence** — memory survives server restarts via JSON (atomic writes)
- **No double-counting** — ResponseAgent uses KnowledgeAgent's context directly, not independent recall
