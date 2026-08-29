<div align="center">

# Recall

**Vector-Based Long-Term Memory & Spaced Repetition for Multi-Agent LLM Systems**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask API](https://img.shields.io/badge/Flask-3.0%2B-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-Sentence--Transformers-FFD21E?style=flat-square)](https://www.sbert.net/)
[![LLM Providers](https://img.shields.io/badge/LLMs-Gemini%20%7C%20Groq%20%7C%20OpenAI%20%7C%20Claude-8A2BE2?style=flat-square)](#supported-llm-providers)

[Overview](#overview) • [Architecture](#architecture) • [Key Features](#key-features) • [Quick Start](#quick-start) • [Web Dashboard & API](#web-dashboard--api) • [Evaluation](#performance--evaluation) • [Configuration](#configuration) • [Project Structure](#project-structure)

</div>

---

**Recall** is a high-performance long-term memory framework built for multi-agent LLM systems. It combines dense semantic search and sparse BM25 keyword retrieval with cognitive memory dynamics—specifically an **Ebbinghaus forgetting curve** and **spaced repetition reinforcement**—enabling agents to retain critical context, forget stale data, and coordinate without context bloat or memory double-counting.

> [!TIP]
> You can run Recall completely offline without API keys using the built-in `stub` provider, or connect it to Gemini, Groq, OpenAI, Anthropic, or GitHub Models.

---

## Overview

Traditional LLM memory approaches either stuff entire conversation histories into prompts (leading to high costs and attention degradation) or rely on static vector databases that treat every memory identically regardless of age or frequency of use.

Recall implements biological memory principles for AI agents:
- **Hierarchical Memory Banks**: Separates factual knowledge, conversational dialogue, and synthesised agent reasoning.
- **Hybrid Retrieval**: Combines semantic embeddings with BM25 lexical search using min-max score normalisation.
- **Cognitive Decay**: Memory retention decays exponentially over time ($R(t) = e^{-t/\lambda}$).
- **Spaced Repetition**: Each recall event strengthens a memory's half-life ($\lambda \leftarrow \lambda + 1.0$), making useful information permanent while unused memories are automatically pruned.
- **Write-Time Deduplication**: Prevents database bloat by rejecting memories that exceed cosine similarity thresholds.

---

## Architecture

```
                      Customer Query
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                      IntakeAgent                       │
│  - Classifies intent & extracts structured entities    │
│  - Recalls related past dialogs by issue summary       │
│  - Stores new conversation record into dialog bank     │
└───────────┬────────────────────────────────┬───────────┘
            │ intent + summary               │
            ▼                                │
┌────────────────────────────────────────┐   │   ┌───────────────────────────────┐
│             KnowledgeAgent             │   │   │          Recall Core          │
│  - Queries knowledge, dialog, & tasks  │◀──┼──▶│                               │
│  - Applies domain intent compatibility │   │   │  ┌─────────────────────────┐  │
│  - Synthesises context for resolution  │   │   │  │  knowledge  (λ₀ = 24.0) │  │
│  - Stores synthesis into task bank     │───┼──▶│  ├─────────────────────────┤  │
└───────────┬────────────────────────────┘   │   │  │  dialog     (λ₀ = 10.0) │  │
            │ knowledge_context              │   │  ├─────────────────────────┤  │
            ▼                                │   │  │  task       (λ₀ = 15.0) │  │
┌────────────────────────────────────────┐   │   │  └─────────────────────────┘  │
│             ResponseAgent              │   │   │                               │
│  - Drafts customer-facing solution     │   │   │  • Hybrid Dense + BM25 Search │
│  - Uses passed context directly        │   │   │  • Ebbinghaus Decay & Pruning │
│  - Prevents recall double-counting     │   └───│  • Write-Time Deduplication   │
└────────────────────────────────────────┘       └───────────────────────────────┘
```

### Memory Banks

| Bank | Stored By | Initial Half-Life ($\lambda_0$) | Content |
| :--- | :--- | :--- | :--- |
| **`knowledge`** | Knowledge Seeder / System | `24.0` hrs | Domain policies, FAQs, platform constraints, product specs |
| **`dialog`** | `IntakeAgent` | `10.0` hrs | User interactions, classified customer intents, extracted entities |
| **`task`** | `KnowledgeAgent` | `15.0` hrs | Multi-agent reasoning traces, synthesised contexts, resolutions |

---

## Key Features

- **Hybrid Dense + Sparse Search**: Fuses dense `sentence-transformers` embeddings with BM25 lexical scores:
  $$S = \alpha \cdot \text{CosineSim}(q, d) + (1 - \alpha) \cdot \text{BM25}_{\text{norm}}(q, d)$$
- **Ebbinghaus Forgetting & Spaced Repetition**: Memory retention follows $R(t) = e^{-t/\lambda}$. Every successful retrieval reinforces stability ($\lambda \leftarrow \lambda + 1$), preserving high-value facts while low-retention segments are pruned.
- **Smart Deduplication**: Vector-level deduplication (cosine threshold $\ge 0.92$) prevents redundant writes.
- **Zero Double-Counting**: `ResponseAgent` utilizes synthesised context directly from `KnowledgeAgent`, keeping access counts and decay curves accurate.
- **Real-Time Web Dashboard**: Modern UI with dark/light themes, live SSE streaming execution traces, visual retention graphs, and memory inspection.
- **Multi-LLM Provider Support**: Native adapters for Google Gemini, Groq, OpenAI, Anthropic Claude, GitHub Models, and offline Stubs.
- **Production Ready**: Atomic JSON persistence, in-memory IP rate limiting, input validation, and structured logging.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- `pip` or virtual environment manager

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Recall-2.0.git
cd Recall-2.0

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file and configure your chosen provider:

```bash
cp .env.example .env
```

Edit `.env` with your API key:

```env
# Choose provider: gemini | groq | openai | anthropic | github | stub
RECALL_LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
RECALL_MODEL=gemini-2.0-flash
```

> [!NOTE]
> If no API key is provided, Recall defaults to `stub` mode, allowing you to test the complete memory pipeline and UI locally without external API dependencies.

### 4. Running the Web UI

Start the Flask application server:

```bash
python web/app.py --provider gemini --port 5000
```

Open your browser at **`http://localhost:5000`** to access the dashboard.

### 5. Running the CLI & Interactive Mode

Run pre-configured demo queries through the terminal:

```bash
python main.py --provider gemini
```

Or launch interactive conversational mode:

```bash
python main.py --interactive --provider gemini
```

Within interactive mode, type `memory` to print a live memory breakdown or `quit` to exit.

---

## Web Dashboard & API

The web interface provides real-time visibility into multi-agent execution and vector memory state:

- **Pipeline Execution**: Submit queries, monitor real-time agent reasoning via Server-Sent Events (SSE), and inspect retrieved memories.
- **Memory Visualizer**: Real-time charts for memory type distributions, agent contributions, and the dynamic Ebbinghaus decay curve.
- **Segment Explorer**: Filter, search, and inspect individual vectors, retention scores, and access counts.
- **Execution History**: Audit full execution traces, latency breakdowns, and memory state deltas.

```
                      Recall Dashboard Pages
  ┌─────────────────────────────────────────────────────────────┐
  │  [ Pipeline ]      Live multi-agent execution & SSE trace   │
  │  [ Visualizer ]    Memory analytics & Ebbinghaus curve      │
  │  [ Segments ]      Filterable memory bank explorer          │
  │  [ Run History ]   Trace logs, per-agent latency & diffs    │
  └─────────────────────────────────────────────────────────────┘
```

### REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/pipeline` | Execute the full 3-agent pipeline synchronously |
| `POST` | `/api/pipeline/stream` | Stream pipeline execution events via Server-Sent Events (SSE) |
| `GET` | `/api/runs` | Retrieve recent pipeline execution traces and latency metrics |
| `GET` | `/api/memory/stats` | Retrieve aggregate memory counts, retention buckets, and lifecycle stats |
| `GET` | `/api/memory/history` | List stored memory segments with optional `?type=` and `?limit=` filters |
| `POST` | `/api/memory/prune` | Trigger manual pruning of decayed memory segments ($R < \text{threshold}$) |
| `POST` | `/api/memory/seed` | Seed default customer support domain knowledge into the memory bank |
| `GET` | `/api/health` | Health check endpoint returning status, active provider, and segment counts |

---

## Performance & Evaluation

Recall includes an evaluation suite that benchmarks four memory paradigms against 36 curated test cases covering intent classification, ambiguous phrasing, and multi-turn conversational cross-references.

### Evaluation Modes

1. **`context_stuffing`**: Injects raw conversation history turns directly into the LLM context window.
2. **`rag_cosine`**: Standard dense vector retrieval ($\alpha = 1.0$) without memory lifecycle or decay.
3. **`rag_hybrid`**: Dense cosine + sparse BM25 retrieval ($\alpha = 0.6$) without decay or spaced repetition.
4. **`recall`**: Complete Recall engine (Hybrid retrieval + Ebbinghaus decay + Spaced repetition + Dedup).

### Running Benchmarks

```bash
# Run full benchmark across all 4 modes
python eval/evaluate.py --provider gemini

# Compare specific modes
python eval/evaluate.py --provider gemini --modes recall rag_hybrid

# Run offline evaluation with stub provider
python eval/evaluate.py --provider stub
```

### Evaluated Metrics

- **Precision@K**: Relevance of retrieved memory chunks against annotated ground truth.
- **Keyword Coverage**: Proportion of expected domain terms in agent responses.
- **Context Length & Token Usage**: Total prompt and completion tokens consumed per query.
- **Latency Breakdown**: End-to-end response time and per-agent execution times.
- **Cross-Turn Accuracy**: Resolution capability on multi-turn referential queries (e.g., *"Has my refund been processed yet?"*).

---

## Configuration

All configuration options can be defined in `.env` or passed via CLI flags:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RECALL_LLM_PROVIDER` | `gemini` | LLM backend: `gemini`, `groq`, `openai`, `anthropic`, `github`, or `stub` |
| `RECALL_MODEL` | Provider default | Specific model name (e.g., `gemini-2.0-flash`, `gpt-4o-mini`, `llama3-8b-8192`) |
| `RECALL_ST_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`) |
| `RECALL_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `RECALL_RATE_LIMIT` | `30` | Maximum API requests allowed per client IP within window |
| `RECALL_RATE_WINDOW`| `60` | Rate limit window in seconds |
| `PORT` | `5000` | Port for the Flask web server |

### Supported LLM Providers

| Provider | Required Environment Variable | Recommended Model |
| :--- | :--- | :--- |
| **Google Gemini** | `GEMINI_API_KEY` | `gemini-2.0-flash`, `gemma-3-27b-it` |
| **Groq** | `GROQ_API_KEY` | `llama3-8b-8192`, `mixtral-8x7b-32768` |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-4o` |
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |
| **GitHub Models** | `GITHUB_TOKEN` | `gpt-4o-mini` |
| **Stub** | _None_ | Offline rule-based mock responses |

---

## Project Structure

```
Recall/
├── agents/                     # Multi-agent implementations
│   ├── base_agent.py           # Base agent class with memory & LLM caller methods
│   ├── customer_support.py     # IntakeAgent, KnowledgeAgent, and ResponseAgent
│   └── pipeline.py             # Pipeline orchestrator & SSE event generator
├── core/                       # Core memory engine
│   ├── memory.py               # Recall class, BM25 retriever, embedder, Ebbinghaus decay
│   └── __init__.py
├── demo/                       # Demo seeding data
│   └── knowledge_seed.py       # Pre-seeded enterprise knowledge facts
├── eval/                       # Benchmark & evaluation suite
│   ├── evaluate.py             # Evaluation harness, token counter & metric calculator
│   └── queries.json            # 36 benchmark queries with ground truth annotations
├── web/                        # Web dashboard & API
│   ├── app.py                  # Flask REST API, SSE streaming & rate limiter
│   └── static/                 # Single-page frontend
│       ├── index.html          # Dashboard markup & SVG iconography
│       ├── app.js              # State management, SSE consumer & canvas renderer
│       └── styles.css          # Theme system & responsive layout styles
├── .env.example                # Template configuration file
├── main.py                     # CLI demo & interactive runner
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
