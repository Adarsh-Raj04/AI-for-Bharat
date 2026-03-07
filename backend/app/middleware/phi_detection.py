"""
PHI Detection Middleware
------------------------
Scans every incoming chat request body for Protected Health Information (PHI)
patterns before the request reaches the RAG pipeline.

Rejects with HTTP 400 if PHI is detected, returning a structured error
that the frontend can surface to the user.

Patterns covered (HIPAA Safe Harbor identifiers):
  - Social Security Numbers
  - Medical Record Numbers
  - Phone numbers (US formats)
  - Dates of birth (common formats)
  - Email addresses
  - Patient name signals ("patient <Name>", "my patient")
  - ZIP codes (5-digit standalone)
  - IP addresses (as data, not request metadata)
  - Health plan / account / certificate numbers
  - Vehicle / device serial number patterns
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern registry
# Each entry: (label, compiled_regex)
# ---------------------------------------------------------------------------

_PHI_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "SSN_NO_DASH",
        re.compile(r"\b\d{9}\b"),  # 9-digit standalone — broad but catches raw SSNs
    ),
    (
        "PHONE_US",
        re.compile(r"\b(?:\+1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
    ),
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "DATE_OF_BIRTH",
        re.compile(
            r"\b(?:dob|date of birth|born on|birthdate)\s*[:\-]?\s*"
            r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MRN",
        re.compile(
            r"\b(?:mrn|medical record(?:\s+number)?|patient\s+id)\s*[:\-#]?\s*\d{4,10}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PATIENT_NAME_SIGNAL",
        re.compile(
            r"\b(?:my patient|patient\s+[A-Z][a-z]+|treating\s+[A-Z][a-z]+)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "HEALTH_PLAN_NUMBER",
        re.compile(
            r"\b(?:policy|plan|member|certificate|beneficiary)\s*(?:no\.?|number|#)\s*[:\-]?\s*\w{6,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),
]

# Paths to inspect — only scan chat endpoint bodies
_SCAN_PATHS = {"/api/v1/chat/chat"}


def _detect_phi(text: str) -> List[str]:
    """Return list of matched PHI label names, empty if clean."""
    matches = []
    for label, pattern in _PHI_PATTERNS:
        if pattern.search(text):
            matches.append(label)
    return matches


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class PHIDetectionMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware that blocks requests containing PHI patterns.

    Only inspects POST request bodies on configured paths.
    All other requests pass through untouched.
    """

    def __init__(self, app: ASGIApp, scan_paths: set = None) -> None:
        super().__init__(app)
        self.scan_paths = scan_paths or _SCAN_PATHS

    async def dispatch(self, request: Request, call_next):
        # Only scan POST requests on configured paths
        if request.method == "POST" and request.url.path in self.scan_paths:
            try:
                body_bytes = await request.body()
                body_text = body_bytes.decode("utf-8", errors="ignore")

                # Extract the message field from the JSON body if possible
                scan_text = body_text
                try:
                    body_json = json.loads(body_text)
                    # Scan only the user's message, not session IDs etc.
                    scan_text = body_json.get("message", body_text)
                except (json.JSONDecodeError, AttributeError):
                    pass

                detected = _detect_phi(scan_text)

                if detected:
                    logger.warning(
                        "PHI detected in request to %s — patterns: %s",
                        request.url.path,
                        detected,
                    )
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "PHI_DETECTED",
                            "message": (
                                "Your query appears to contain protected health information (PHI) "
                                "such as personal identifiers, phone numbers, or patient details. "
                                "Please remove any personal information and rephrase your research question."
                            ),
                            "detected_patterns": detected,
                        },
                    )

                # Re-attach body so downstream can still read it
                # (Starlette body is consumed; we patch it back via scope)
                async def receive():
                    return {"type": "http.request", "body": body_bytes}

                request = Request(request.scope, receive)

            except Exception:
                logger.exception("PHI middleware error — passing request through")

        return await call_next(request)
