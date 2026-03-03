"""
Recall Customer Support Demo
══════════════════════════════
Demonstrates a 3-agent pipeline (Intake → Knowledge → Response)
sharing a single Recall long-term memory instance.

Usage:
    python main.py                      # runs full demo with stub LLM
    python main.py --provider groq      # use Groq (set GROQ_API_KEY)
    python main.py --provider openai    # use OpenAI (set OPENAI_API_KEY)
    python main.py --provider anthropic # use Anthropic (set ANTHROPIC_API_KEY)
    python main.py --interactive        # interactive chat mode
"""

import sys
import os
import argparse
import logging

# ── Logging setup ─────────────────────────────────────────────────────────────
def _setup_logging():
    """Configure structured logging for the entire Recall application."""
    level = os.environ.get("RECALL_LOG_LEVEL", "INFO").upper()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger("recall")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(handler)
    # Prevent duplicate logs from propagating to root logger
    root.propagate = False

_setup_logging()
logger = logging.getLogger("recall.main")

# Make sure imports resolve from project root (works on Windows, Mac, Linux)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Load .env ────────────────────────────────────────────────────────────────
def _load_env():
    """Load .env file — works with or without python-dotenv installed."""
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        logger.warning("No .env file found at %s", env_path)
        logger.info("Rename .env.example to .env and fill in your keys")
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)  # override=True forces reload
        logger.info("Loaded .env via python-dotenv")
    except ImportError:
        # Manual fallback — no package needed
        loaded = []
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")  # strip quotes too
                if key and val:
                    os.environ[key] = val  # always set, even if already exists
                    loaded.append(key)
        logger.info("Loaded .env — keys: %s", ', '.join(loaded))

_load_env()
# ─────────────────────────────────────────────────────────────────────────────

from core.memory import Recall
from agents.customer_support import IntakeAgent, KnowledgeAgent, ResponseAgent
from agents.pipeline import AgentPipeline
from demo.knowledge_seed import seed_customer_support_knowledge


DEMO_QUERIES = [
    "Hi, I was charged twice this month on my credit card. This is really frustrating!",
    "I can't log in to my account. I keep getting an error saying 'invalid credentials' even though I just reset my password.",
    "How do I add more team members to my account? We just hired 3 people.",
    "My API requests are failing with a 429 error. Our app goes down every evening.",
    "I want to cancel my subscription. How do I do that and will I get a refund?",
]


def build_pipeline(provider: str = None) -> AgentPipeline:
    """Build the shared Recall + 3-agent pipeline."""
    # Provider priority: CLI arg → .env → default stub
    if provider is None:
        provider = os.environ.get("RECALL_LLM_PROVIDER", "stub")
    model = os.environ.get("RECALL_MODEL") or None

    logger.info("Initializing Recall Multi-Agent Customer Support System")

    bank = Recall(forget_threshold=0.02, verbose=True)
    seed_customer_support_knowledge(bank)

    agents = [
        IntakeAgent(bank,    llm_provider=provider, model=model, verbose=True),
        KnowledgeAgent(bank, llm_provider=provider, model=model, verbose=True),
        ResponseAgent(bank,  llm_provider=provider, model=model, verbose=True),
    ]

    pipeline = AgentPipeline(bank, agents, prune_every=10)
    logger.info("Pipeline ready | LLM provider: %s", provider.upper())
    logger.info("Agents: %s", ' -> '.join(a.name for a in agents))
    logger.info("Memory seeded: %d segments", bank.summary()['total_segments'])
    return pipeline


def run_demo(provider: str = None):
    pipeline = build_pipeline(provider)

    print("\n" + "═" * 60)
    print("  DEMO: Processing 5 customer support queries")
    print("  Watch how memory accumulates across queries!")
    print("═" * 60)

    for query in DEMO_QUERIES:
        pipeline.run(query)
        input("\n  ⏎ Press Enter for next query...")

    pipeline.memory_report()

    print("\n" + "═" * 60)
    print("  MEMORY EFFECT: Running a similar billing query again...")
    print("═" * 60)
    pipeline.run("I think there's a mistake on my latest invoice.")
    pipeline.memory_report()


def run_interactive(provider: str = None):
    pipeline = build_pipeline(provider)

    print("\n" + "═" * 60)
    print("  INTERACTIVE MODE — Customer Support Pipeline")
    print("  Type a customer message and press Enter.")
    print("  Commands: 'memory' = show memory state | 'quit' = exit")
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "memory":
            pipeline.memory_report()
            continue

        pipeline.run(user_input)


def main():
    parser = argparse.ArgumentParser(description="Recall Customer Support Demo")
    parser.add_argument("--provider", default=None,
                        choices=["stub", "gemini", "groq", "openai", "anthropic"],
                        help="LLM provider (overrides .env RECALL_LLM_PROVIDER)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.provider)
    else:
        run_demo(args.provider)


if __name__ == "__main__":
    main()
