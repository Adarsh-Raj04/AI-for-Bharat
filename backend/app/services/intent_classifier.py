"""
Intent Classifier - Categorizes incoming queries and routes them appropriately
"""

import re
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Query intent types"""

    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    FACTUAL_LOOKUP = "factual_lookup"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    ADVERSE_EVENTS = "adverse_events"
    CLINICAL_TRIAL = "clinical_trial"
    DRUG_INTERACTION = "drug_interaction"
    GENERAL_QA = "general_qa"


# ---------------------------------------------------------------------------
# Hard-rule pre-checks — run BEFORE pattern scoring
# If any of these match, intent is forced regardless of score
# ---------------------------------------------------------------------------
HARD_RULE_INTENTS = [
    # PMID reference of any phrasing → always SUMMARIZATION
    (QueryIntent.SUMMARIZATION, re.compile(r"\bpmid:?\s*\d{7,9}\b", re.IGNORECASE)),
    # NCT number → always CLINICAL_TRIAL
    (QueryIntent.CLINICAL_TRIAL, re.compile(r"\bnct\s*\d{6,8}\b", re.IGNORECASE)),
]


class IntentClassifier:
    """
    Classifies user queries into intent categories.
    Uses hard rules first, then rule-based keyword scoring.
    """

    def __init__(self):
        self.intent_patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[QueryIntent, List[str]]:
        return {
            QueryIntent.SUMMARIZATION: [
                r"\b(summarize|summarise|summary|summaries)\b",
                r"\b(overview|abstract|tldr|tl;dr)\b",
                r"\b(key findings?|main findings?|key results?)\b",
                r"\b(give me (a|an) (summary|overview|brief))\b",
                r"\b(what (does|did) (the|this) (paper|study|article|research) (say|find|show|report))\b",
                r"\b(can you (summarize|summarise|give|provide))\b",
            ],
            QueryIntent.COMPARISON: [
                r"\b(compare|comparison|versus|vs\.?|difference between)\b",
                r"\b(which is (better|more effective|safer))\b",
                r"\b(how do(es)? .+ compare (to|with))\b",
                r"\b(advantages? (and|vs) disadvantages?)\b",
                r"\b(pros and cons)\b",
            ],
            QueryIntent.FACTUAL_LOOKUP: [
                r"\b(what is|define|definition of|meaning of)\b",
                r"\b(mechanism of action|moa)\b",
                r"\b(how (does|do|is|are) .+ work)\b",
            ],
            QueryIntent.REGULATORY_COMPLIANCE: [
                r"\b(fda|ema|regulatory|approval|guidance|compliance)\b",
                r"\b(clinical trial (phase|protocol|design))\b",
                r"\b(ich guidelines?|gcp|glp)\b",
                r"\b(label(l?ing)?|package insert)\b",
                r"\b(black box warning|contraindication)\b",
            ],
            QueryIntent.ADVERSE_EVENTS: [
                r"\b(adverse (event|reaction|effect)|side effect|safety)\b",
                r"\b(toxicity|toxic)\b",
                r"\b(drug safety|pharmacovigilance)\b",
            ],
            QueryIntent.CLINICAL_TRIAL: [
                r"\b(clinical trial|randomized controlled trial|rct)\b",
                r"\b(phase [i1-3]{1,3}|phase [1-3])\b",
                r"\b(randomized|placebo|double-blind|open-label)\b",
                r"\b(primary endpoint|secondary endpoint)\b",
            ],
            QueryIntent.DRUG_INTERACTION: [
                r"\b(drug interaction|interact(s|ion) with)\b",
                r"\b(can i take .+ with)\b",
                r"\b(combination (therapy|treatment))\b",
                r"\b(contraindicated with)\b",
            ],
        }

    def classify(self, query: str) -> Tuple[QueryIntent, float, Dict]:
        """
        Classify query intent.

        Step 1: Hard rules — PMID/NCT force a specific intent immediately.
        Step 2: Pattern scoring — highest score wins.
        Step 3: Fallback to GENERAL_QA.
        """
        # ── Step 1: Hard rules ────────────────────────────────────────────
        for forced_intent, pattern in HARD_RULE_INTENTS:
            if pattern.search(query):
                logger.info(
                    "Hard rule matched → intent forced to %s for query: %s",
                    forced_intent.value,
                    query[:80],
                )
                return (
                    forced_intent,
                    1.0,
                    {
                        "matched_patterns": [pattern.pattern],
                        "forced_by_hard_rule": True,
                    },
                )

        # ── Step 2: Pattern scoring ───────────────────────────────────────
        query_lower = query.lower()
        intent_scores: Dict[QueryIntent, int] = {}
        matched_patterns: Dict[QueryIntent, List[str]] = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            matches = []
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
                    matches.append(pattern)
            if score > 0:
                intent_scores[intent] = score
                matched_patterns[intent] = matches

        if not intent_scores:
            return QueryIntent.GENERAL_QA, 0.5, {"matched_patterns": []}

        primary_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[primary_intent]
        total_patterns = len(self.intent_patterns[primary_intent])
        confidence = min(0.95, 0.6 + (max_score / total_patterns) * 0.35)

        metadata = {
            "matched_patterns": matched_patterns.get(primary_intent, []),
            "all_scores": {k.value: v for k, v in intent_scores.items()},
            "query_length": len(query.split()),
        }

        logger.info(
            "Classified query as %s (confidence %.2f) | scores: %s",
            primary_intent.value,
            confidence,
            {k.value: v for k, v in intent_scores.items()},
        )

        return primary_intent, confidence, metadata

    def get_routing_config(self, intent: QueryIntent) -> Dict:
        configs = {
            QueryIntent.SUMMARIZATION: {
                "top_k": 5,  # bumped from 3 — give reranker more to work with
                "rerank_top_k": 3,  # bumped from 2 — was dropping the target doc
                "max_tokens": 1500,
                "temperature": 0.3,
                "source_types": ["pubmed", "clinical_trial"],
            },
            QueryIntent.COMPARISON: {
                "top_k": 12,
                "rerank_top_k": 6,
                "max_tokens": 2500,
                "temperature": 0.4,
                "source_types": ["pubmed", "clinical_trial", "fda"],
            },
            QueryIntent.FACTUAL_LOOKUP: {
                "top_k": 8,
                "rerank_top_k": 4,
                "max_tokens": 1500,
                "temperature": 0.2,
                "source_types": ["pubmed", "fda", "textbook"],
            },
            QueryIntent.REGULATORY_COMPLIANCE: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.2,
                "source_types": ["fda", "ema", "regulatory"],
            },
            QueryIntent.ADVERSE_EVENTS: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.3,
                "source_types": ["pubmed", "fda", "clinical_trial"],
            },
            QueryIntent.CLINICAL_TRIAL: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.3,
                "source_types": ["clinical_trial", "pubmed"],
            },
            QueryIntent.DRUG_INTERACTION: {
                "top_k": 8,
                "rerank_top_k": 4,
                "max_tokens": 1500,
                "temperature": 0.2,
                "source_types": ["pubmed", "fda", "drugbank"],
            },
            QueryIntent.GENERAL_QA: {
                "top_k": 10,
                "rerank_top_k": 5,
                "max_tokens": 2000,
                "temperature": 0.5,
                "source_types": None,
            },
        }
        return configs.get(intent, configs[QueryIntent.GENERAL_QA])


_intent_classifier = None


def get_intent_classifier() -> IntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
