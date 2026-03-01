"""
Recall Web Server — Customer Support Pipeline UI
Run: python web/app.py [--port 5000] [--provider gemini]
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _load_env():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path): return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and val: os.environ[key] = val
_load_env()

import argparse
import time
from flask import Flask, request, jsonify, send_from_directory

from core.memory import Recall
from agents.customer_support import IntakeAgent, KnowledgeAgent, ResponseAgent
from agents.pipeline import AgentPipeline
from demo.knowledge_seed import seed_customer_support_knowledge

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR)

bank: Recall = None
pipeline: AgentPipeline = None
_provider: str = "stub"

def init_agents(provider: str = "stub"):
    global bank, pipeline, _provider
    provider = provider or os.environ.get("RECALL_LLM_PROVIDER", "stub")
    model    = os.environ.get("RECALL_MODEL") or None
    _provider = provider

    persist_path = os.path.join(ROOT, 'recall_memory.json')
    bank = Recall(forget_threshold=0.02, dedup_threshold=0.92,
                  persist_path=persist_path, verbose=False)
    agents = [
        IntakeAgent(bank,    llm_provider=provider, model=model, verbose=False),
        KnowledgeAgent(bank, llm_provider=provider, model=model, verbose=False),
        ResponseAgent(bank,  llm_provider=provider, model=model, verbose=False),
    ]
    pipeline = AgentPipeline(bank, agents, prune_every=20)
    print(f"✅ Recall pipeline ready | provider={provider.upper()}")


# ── Static ────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ── Pipeline ──────────────────────────────────────────────────
@app.route("/api/pipeline", methods=["POST"])
def run_pipeline():
    data  = request.get_json()
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    run = pipeline.run(query)
    # Serialise — ensure agent traces are JSON-safe
    def safe(v):
        if isinstance(v, dict):  return {k: safe(x) for k,x in v.items()}
        if isinstance(v, list):  return [safe(x) for x in v]
        return v
    return jsonify(safe(run))

@app.route("/api/runs", methods=["GET"])
def get_runs():
    def safe(v):
        if isinstance(v, dict):  return {k: safe(x) for k,x in v.items()}
        if isinstance(v, list):  return [safe(x) for x in v]
        return v
    return jsonify(safe(list(reversed(pipeline.run_history))))


# ── Memory ────────────────────────────────────────────────────
@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    segs = bank.get_all()
    by_type  = {"knowledge": 0, "dialog": 0, "task": 0}
    by_agent: dict = {}
    ret_buckets = {"high (>0.8)": 0, "mid (0.4–0.8)": 0, "low (<0.4)": 0}
    recall_total = 0
    for s in segs:
        by_type[s.memory_type] = by_type.get(s.memory_type, 0) + 1
        by_agent[s.source_agent] = by_agent.get(s.source_agent, 0) + 1
        r = s.retention()
        if r > 0.8:   ret_buckets["high (>0.8)"] += 1
        elif r > 0.4: ret_buckets["mid (0.4–0.8)"] += 1
        else:         ret_buckets["low (<0.4)"] += 1
        recall_total += s.recall_count
    return jsonify({
        "total": len(segs), "by_type": by_type, "by_agent": by_agent,
        "retention_buckets": ret_buckets, "total_recalls": recall_total,
        "deduped":    bank.stats.get("deduped", 0),
        "lifecycle":  bank.stats,
        "embedder":   type(bank.embedder).__name__,
        "provider": _provider,
    })

@app.route("/api/memory/history", methods=["GET"])
def memory_history():
    mtype = request.args.get("type")
    limit = int(request.args.get("limit", 150))
    segs  = bank.get_all(memory_type=mtype)
    segs  = sorted(segs, key=lambda s: s.created_at, reverse=True)[:limit]
    return jsonify([{
        "id":                s.id,
        "text":              s.text[:200],
        "memory_type":       s.memory_type,
        "source_agent":      s.source_agent,
        "retention":         round(s.retention(), 3),
        "recall_count":      s.recall_count,
        "lambda_forget":     round(s.lambda_forget, 2),
        "created_at":        s.created_at,
        "last_accessed":     s.last_accessed,
        "used_memory_ids":   s.metadata.get("used_memory_ids", []),
        "used_memory_texts": s.metadata.get("used_memory_texts", []),
    } for s in segs])

@app.route("/api/memory/prune", methods=["POST"])
def prune():
    pruned = bank.prune_forgotten()
    return jsonify({"pruned": pruned, "remaining": bank.summary()["total_segments"]})

@app.route("/api/memory/seed", methods=["POST"])
def reseed():
    n = seed_customer_support_knowledge(bank)
    return jsonify({"seeded": n})


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int, default=5000)
    parser.add_argument("--provider", default=None,
                        choices=["stub","groq","openai","anthropic","gemini"])
    args = parser.parse_args()
    init_agents(args.provider)
    print(f"\n🌐 Recall → http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port, host="0.0.0.0")
