"""
Knowledge seeder — pre-populates Recall with domain knowledge.
In a real system this could load from a CRM, docs, or FAQ database.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from core.memory import Recall

logger = logging.getLogger("recall.demo")


def seed_customer_support_knowledge(bank: Recall):
    """Seed the knowledge bank with common customer support facts."""

    knowledge_base = [
        # Billing
        "Invoices are generated on the 1st of each month and sent via email.",
        "Customers can update billing details from Account Settings > Billing.",
        "Refunds for billing errors are processed within 5-7 business days.",
        "Annual plan subscribers receive a 20% discount compared to monthly billing.",
        "Failed payments trigger an automatic retry after 3 days; the account is paused after 2 failures.",

        # Technical
        "The service supports Chrome, Firefox, Safari and Edge (latest 2 versions).",
        "Clearing browser cache and cookies resolves 80% of login issues.",
        "API rate limit is 1000 requests/minute; upgrade to Enterprise for higher limits.",
        "Planned maintenance windows are every Sunday 2-4am UTC; status at status.example.com.",
        "Two-factor authentication can be enabled under Security Settings.",

        # Account
        "Passwords must be 8+ characters with at least one number and one special character.",
        "Account deletion requests are processed within 30 days per data regulations.",
        "Users can export all their data from Account Settings > Data Export.",
        "Sub-accounts (team members) can be added under Team Management.",
        "Free tier allows up to 3 projects; Pro plan allows unlimited projects.",

        # Refund / Cancellation
        "Cancellations take effect at end of current billing period; no partial refunds.",
        "Enterprise contracts require 30-day written notice for cancellation.",
        "Refund policy: full refund within 14 days of purchase if not satisfied.",
        "Downgrade from Pro to Free retains data but limits features immediately.",
    ]

    seeded = 0
    for fact in knowledge_base:
        bank.store(fact, memory_type="knowledge", source_agent="KnowledgeSeeder")
        seeded += 1

    logger.info("Seeded %d knowledge facts into Recall", seeded)
    return seeded
