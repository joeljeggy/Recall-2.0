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
import json
import logging
import sys
import time
from flask import Flask, request, jsonify, send_from_directory, Response

# ── Logging setup ──────────────────────────────────────────────
def _setup_logging():
    level = os.environ.get("RECALL_LOG_LEVEL", "INFO").upper()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger("recall")
    root.setLevel(getattr(logging, level, logging.INFO))
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False

_setup_logging()

from core.memory import Recall
from agents.customer_support import IntakeAgent, KnowledgeAgent, ResponseAgent
from agents.pipeline import AgentPipeline
from demo.knowledge_seed import seed_customer_support_knowledge

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR)

logger = logging.getLogger("recall.web")

# ── Rate Limiting ─────────────────────────────────────────────
import threading
from collections import defaultdict as _dd

class RateLimiter:
    """Simple in-memory per-IP rate limiter (no external deps)."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = _dd(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            self._hits[key] = [t for t in self._hits[key] if now - t < self.window]
            if len(self._hits[key]) >= self.max_requests:
                return False
            self._hits[key].append(now)
            return True

_limiter = RateLimiter(
    max_requests=int(os.environ.get("RECALL_RATE_LIMIT", "30")),
    window_seconds=int(os.environ.get("RECALL_RATE_WINDOW", "60")),
)

# ── Input validation helpers ──────────────────────────────────
MAX_QUERY_LENGTH = 2000

def _get_client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()

def _validate_query(data) -> tuple[str | None, tuple | None]:
    """Validate pipeline query input. Returns (query, None) or (None, error_response)."""
    if data is None:
        return None, (jsonify({"error": "Request body must be JSON"}), 400)
    query = data.get("query", "")
    if not isinstance(query, str):
        return None, (jsonify({"error": "query must be a string"}), 400)
    query = query.strip()
    if not query:
        return None, (jsonify({"error": "query is required and cannot be empty"}), 400)
    if len(query) > MAX_QUERY_LENGTH:
        return None, (jsonify({"error": f"query exceeds maximum length of {MAX_QUERY_LENGTH} characters"}), 400)
    return query, None

# ── Global error handlers ─────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error"}), 500

@app.before_request
def check_rate_limit():
    if request.path.startswith("/api/"):
        ip = _get_client_ip()
        if not _limiter.is_allowed(ip):
            logger.warning("Rate limit exceeded for %s on %s", ip, request.path)
            return jsonify({"error": "Rate limit exceeded. Try again later."}), 429

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
    logger.info("Recall pipeline ready | provider=%s", provider.upper())


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
    data  = request.get_json(silent=True)
    query, err = _validate_query(data)
    if err:
        return err
    try:
        run = pipeline.run(query)
    except Exception as e:
        logger.exception("Pipeline execution failed for query: %s", query[:80])
        return jsonify({"error": f"Pipeline execution failed: {str(e)}"}), 500
    # Serialise — ensure agent traces are JSON-safe
    def safe(v):
        if isinstance(v, dict):  return {k: safe(x) for k,x in v.items()}
        if isinstance(v, list):  return [safe(x) for x in v]
        return v
    return jsonify(safe(run))


@app.route("/api/pipeline/stream", methods=["POST"])
def run_pipeline_stream():
    """SSE endpoint — streams agent progress events in real-time."""
    data  = request.get_json(silent=True)
    query, err = _validate_query(data)
    if err:
        return err

    def generate():
        try:
            for event in pipeline.run_streaming(query):
                event_type = event.get("event", "message")
                # JSON-safe serialisation
                payload = json.dumps(event, default=str)
                yield f"event: {event_type}\ndata: {payload}\n\n"
        except Exception as e:
            logger.exception("SSE stream error for query: %s", query[:80])
            error_payload = json.dumps({"event": "error", "message": str(e)})
            yield f"event: error\ndata: {error_payload}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # disable nginx buffering
            "Connection": "keep-alive",
        },
    )

@app.route("/api/runs", methods=["GET"])
def get_runs():
    try:
        def safe(v):
            if isinstance(v, dict):  return {k: safe(x) for k,x in v.items()}
            if isinstance(v, list):  return [safe(x) for x in v]
            return v
        return jsonify(safe(list(reversed(pipeline.run_history))))
    except Exception as e:
        logger.exception("Failed to fetch run history")
        return jsonify({"error": "Failed to fetch run history"}), 500


# ── Health ────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "provider": _provider,
        "memory_segments": bank.summary()["total_segments"] if bank else 0,
    })


# ── Memory ────────────────────────────────────────────────────
@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    try:
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
    except Exception as e:
        logger.exception("Failed to fetch memory stats")
        return jsonify({"error": "Failed to fetch memory stats"}), 500

@app.route("/api/memory/history", methods=["GET"])
def memory_history():
    try:
        mtype = request.args.get("type")
        if mtype and mtype not in ("knowledge", "dialog", "task"):
            return jsonify({"error": f"Invalid memory type: {mtype}. Must be knowledge, dialog, or task"}), 400
        try:
            limit = int(request.args.get("limit", 150))
            limit = max(1, min(limit, 500))  # clamp to [1, 500]
        except (ValueError, TypeError):
            return jsonify({"error": "limit must be a positive integer"}), 400
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
    except Exception as e:
        if isinstance(e, tuple):  # re-raise HTTP errors
            raise
        logger.exception("Failed to fetch memory history")
        return jsonify({"error": "Failed to fetch memory history"}), 500

@app.route("/api/memory/prune", methods=["POST"])
def prune():
    try:
        pruned = bank.prune_forgotten()
        return jsonify({"pruned": pruned, "remaining": bank.summary()["total_segments"]})
    except Exception as e:
        logger.exception("Failed to prune memory")
        return jsonify({"error": "Failed to prune memory"}), 500

@app.route("/api/memory/seed", methods=["POST"])
def reseed():
    try:
        n = seed_customer_support_knowledge(bank)
        return jsonify({"seeded": n})
    except Exception as e:
        logger.exception("Failed to seed knowledge")
        return jsonify({"error": "Failed to seed knowledge"}), 500


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int, default=5000)
    parser.add_argument("--provider", default=None,
                        choices=["stub","groq","openai","anthropic","gemini"])
    args = parser.parse_args()
    init_agents(args.provider)
    logger.info("Recall -> http://localhost:%d", args.port)
    app.run(debug=False, port=args.port, host="0.0.0.0")
