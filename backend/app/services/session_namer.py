"""
Session Namer Service (OpenAI)
------------------------------
Generates a short, meaningful session title from the user's first query
using OpenAI.

Add to .env:
    OPENAI_API_KEY=your_key_here

Usage:
    from app.services.session_namer import generate_session_name

    name = generate_session_name("What are the latest treatments for Crohn's disease?")
    # → "Crohn's Disease Treatment Options"
"""

import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

_MODEL = os.getenv("SESSION_NAMER_MODEL", "gpt-4.1-mini")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=_OPENAI_KEY)

_PROMPT = """\
You are a chat session title generator.

Given a user's research query, produce a concise, descriptive session title.

Rules:
- 3 to 6 words maximum
- Title case
- No quotes
- No punctuation
- No trailing period
- No filler words like "Query About" or "Question On"
- Capture the core medical or research topic

Examples:

Query: "What are the latest treatments for Crohn's disease?"
Title: Crohn's Disease Treatment Options

Query: "Summarize PMID 41785024"
Title: PMID 41785024 Summary

Query: "Compare efficacy of Drug A vs Drug B in rheumatoid arthritis"
Title: Drug A vs Drug B Arthritis

Query: "What does the FDA say about accelerated approval?"
Title: FDA Accelerated Approval Guidance

Now generate a title for this query:

Query: "{query}"
Title:
"""


def generate_session_name(query: str, fallback: str | None = None) -> str:
    """
    Generate a short session title from a user query.
    """

    if not query or not query.strip():
        return fallback or "New Chat"

    query = query.strip()

    _fallback = fallback or (query[:50] + ("..." if len(query) > 50 else ""))

    if not _OPENAI_KEY:
        logger.warning("OPENAI_API_KEY not set — using fallback name")
        return _fallback

    try:
        response = client.responses.create(
            model=_MODEL,
            input=_PROMPT.format(query=query[:500]),
            max_output_tokens=20,
            temperature=0.2,
        )

        raw = response.output_text.strip()

        name = re.sub(r"^[\"\\\']+|[\"\\\']+$", "", raw).strip().rstrip(".")

        if not name:
            return _fallback

        if len(name) > 60:
            name = name[:60].rsplit(" ", 1)[0] + "..."

        logger.info("Session named: %r (from query: %r)", name, query[:60])

        return name

    except Exception as e:
        logger.warning("Session naming failed, using fallback: %s", e)
        return _fallback
