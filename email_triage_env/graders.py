"""
Deterministic graders for all three tasks in SupportDesk-OpenEnv.

Each grader takes an action dict and a ground-truth dict and returns
a score in [0.0, 1.0] plus a feedback dict explaining the score.

Design principles
-----------------
* Fully deterministic — no LLM calls, no randomness.
* Partial credit wherever meaningful (not just binary).
* Clear feedback dict so agents can learn from mistakes.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case + collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _contains(haystack: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(haystack)


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token similarity between two strings."""
    a_tokens = set(_normalize(a).split())
    b_tokens = set(_normalize(b).split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


# ---------------------------------------------------------------------------
# Grader 1 — email_classify  (EASY)
# ---------------------------------------------------------------------------

PRIORITY_LEVELS = ["urgent", "high", "normal", "low"]
CATEGORY_SET = {"billing", "technical", "account", "general", "complaint"}


def grade_classify(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a 'classify' action.

    Scoring:
    - Priority (60% weight):
        • Exact match → 1.0
        • Off by 1 level → 0.5
        • Off by 2 levels → 0.2
        • Off by 3+ levels → 0.0
        • Missing / invalid → 0.0
    - Category (40% weight):
        • Exact match → 1.0
        • Otherwise → 0.0

    Returns score in [0.0, 1.0] and a detailed feedback dict.
    """
    feedback: Dict[str, Any] = {}

    # --- Priority scoring ---
    pred_priority = (action.get("priority") or "").strip().lower()
    true_priority = ground_truth.get("priority", "").lower()
    valid_priority = pred_priority in PRIORITY_LEVELS

    if not valid_priority:
        priority_score = 0.0
        feedback["priority"] = {
            "predicted": pred_priority or None,
            "expected": true_priority,
            "score": 0.0,
            "reason": "Invalid or missing priority value",
        }
    elif pred_priority == true_priority:
        priority_score = 1.0
        feedback["priority"] = {
            "predicted": pred_priority,
            "expected": true_priority,
            "score": 1.0,
            "reason": "Exact match",
        }
    else:
        try:
            pred_idx = PRIORITY_LEVELS.index(pred_priority)
            true_idx = PRIORITY_LEVELS.index(true_priority)
            distance = abs(pred_idx - true_idx)
        except ValueError:
            distance = 3

        partial_map = {1: 0.5, 2: 0.2}
        priority_score = partial_map.get(distance, 0.0)
        feedback["priority"] = {
            "predicted": pred_priority,
            "expected": true_priority,
            "score": priority_score,
            "reason": f"Off by {distance} priority level(s) — partial credit",
        }

    # --- Category scoring ---
    pred_category = (action.get("category") or "").strip().lower()
    true_category = ground_truth.get("category", "").lower()

    if pred_category == true_category:
        category_score = 1.0
        feedback["category"] = {
            "predicted": pred_category,
            "expected": true_category,
            "score": 1.0,
            "reason": "Exact match",
        }
    else:
        category_score = 0.0
        feedback["category"] = {
            "predicted": pred_category or None,
            "expected": true_category,
            "score": 0.0,
            "reason": "Incorrect category",
        }

    # --- Combined score ---
    score = round(0.6 * priority_score + 0.4 * category_score, 4)
    feedback["total_score"] = score
    return score, feedback


# ---------------------------------------------------------------------------
# Grader 2 — email_extract  (MEDIUM)
# ---------------------------------------------------------------------------

EXTRACT_FIELDS = [
    "customer_id",
    "issue_type",
    "urgency_signals",
    "affected_product",
    "requested_action",
]

# Per-field weights (must sum to 1.0)
FIELD_WEIGHTS = {
    "customer_id": 0.20,
    "issue_type": 0.25,
    "urgency_signals": 0.25,
    "affected_product": 0.15,
    "requested_action": 0.15,
}


def _grade_urgency_signals(predicted: Any, expected: List[str]) -> Tuple[float, str]:
    """
    Score the urgency_signals field specifically.
    Predicted may be a list or a comma-separated string.
    We use token-level recall: fraction of expected signals that have a
    fuzzy match in the predicted list (Jaccard ≥ 0.3).
    """
    if isinstance(predicted, str):
        pred_list = [s.strip() for s in predicted.split(",") if s.strip()]
    elif isinstance(predicted, list):
        pred_list = [str(s).strip() for s in predicted if s]
    else:
        return 0.0, "urgency_signals must be a list or comma-separated string"

    if not pred_list:
        return 0.0, "No urgency signals provided"

    matched = 0
    for exp_signal in expected:
        best = max((_token_overlap(exp_signal, p) for p in pred_list), default=0.0)
        if best >= 0.3:
            matched += 1

    recall = matched / len(expected) if expected else 1.0
    # Also reward precision lightly (penalize spam)
    precision = matched / len(pred_list) if pred_list else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return round(f1, 4), f"Matched {matched}/{len(expected)} expected signals (F1={f1:.2f})"


def _grade_text_field(predicted: Any, expected: str) -> Tuple[float, str]:
    """
    Score a free-text field using token Jaccard similarity.
    Exact match → 1.0, partial match → proportional, no overlap → 0.0.
    """
    if predicted is None:
        return 0.0, "Field missing"
    pred_str = str(predicted).strip()
    if not pred_str:
        return 0.0, "Empty field"
    # Check for exact match first
    if _normalize(pred_str) == _normalize(expected):
        return 1.0, "Exact match"
    sim = _token_overlap(pred_str, expected)
    # Threshold: sim < 0.2 → no credit; sim in [0.2, 0.8) → partial; ≥ 0.8 → near-full
    if sim < 0.2:
        return 0.0, f"No meaningful overlap with expected '{expected}' (Jaccard={sim:.2f})"
    elif sim >= 0.8:
        return round(min(sim, 1.0), 4), f"Near-exact match (Jaccard={sim:.2f})"
    else:
        return round(sim * 0.8, 4), f"Partial match (Jaccard={sim:.2f})"


def grade_extract(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Grade an 'extract' action.

    Each of 5 fields is scored independently using field-specific logic,
    then combined via weighted average.
    """
    extracted = action.get("extracted_info") or {}
    if not isinstance(extracted, dict):
        return 0.0, {"error": "extracted_info must be a JSON object", "total_score": 0.0}

    feedback: Dict[str, Any] = {"fields": {}}
    weighted_total = 0.0

    for field in EXTRACT_FIELDS:
        pred_val = extracted.get(field)
        true_val = ground_truth.get(field)
        weight = FIELD_WEIGHTS[field]

        if field == "urgency_signals":
            field_score, reason = _grade_urgency_signals(pred_val, true_val or [])
        elif field == "customer_id":
            # Exact match only for IDs
            pred_str = str(pred_val).strip() if pred_val else ""
            true_str = str(true_val).strip()
            if _normalize(pred_str) == _normalize(true_str):
                field_score, reason = 1.0, "Exact match"
            else:
                field_score, reason = 0.0, f"Expected '{true_str}', got '{pred_str}'"
        else:
            field_score, reason = _grade_text_field(pred_val, str(true_val or ""))

        weighted_total += weight * field_score
        feedback["fields"][field] = {
            "predicted": pred_val,
            "expected": true_val,
            "score": field_score,
            "weight": weight,
            "reason": reason,
        }

    score = round(weighted_total, 4)
    feedback["total_score"] = score
    return score, feedback


# ---------------------------------------------------------------------------
# Grader 3 — email_respond  (HARD)
# ---------------------------------------------------------------------------

# Scoring breakdown for respond grader:
#   required_terms present (40%)  — each term = equal share
#   no forbidden phrases (15%)    — binary: full credit if none present
#   professional format (20%)     — greeting + closing present
#   factual accuracy (25%)        — key facts from correct_facts dict

FORMAL_GREETINGS = [
    "dear", "hello", "hi ", "good morning", "good afternoon",
    "good evening", "greetings", "thank you for"
]
FORMAL_CLOSINGS = [
    "sincerely", "regards", "best regards", "kind regards",
    "thank you", "thanks", "warm regards", "yours", "cheers"
]
MIN_RESPONSE_WORDS = 40
MAX_RESPONSE_WORDS = 600


def _has_greeting(text: str) -> bool:
    lowered = _normalize(text)
    return any(lowered.startswith(g) or ("\n" + g) in ("\n" + lowered) for g in FORMAL_GREETINGS)


def _has_closing(text: str) -> bool:
    lowered = _normalize(text)
    last_200 = lowered[-200:]
    return any(c in last_200 for c in FORMAL_CLOSINGS)


def grade_respond(action: Dict[str, Any], email_meta: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a 'respond' action.

    Scoring sub-components (must all be deterministic):
    1. Required terms coverage (40%): fraction of required_terms present in response.
    2. No forbidden phrases (15%): binary — all absent → 1.0, any present → 0.0.
    3. Professional format (20%): greeting (10%) + closing (10%).
    4. Factual accuracy (25%): fraction of correct_facts values present in response.
    """
    response_text = action.get("response_text") or ""
    feedback: Dict[str, Any] = {}

    if not response_text.strip():
        return 0.0, {"error": "Empty response", "total_score": 0.0}

    word_count = len(response_text.split())
    feedback["word_count"] = word_count

    # 1. Required terms (40%)
    required_terms: List[str] = email_meta.get("required_terms", [])
    if required_terms:
        matched_terms = [t for t in required_terms if _contains(response_text, t)]
        terms_score = len(matched_terms) / len(required_terms)
        feedback["required_terms"] = {
            "matched": matched_terms,
            "missing": [t for t in required_terms if t not in matched_terms],
            "score": round(terms_score, 4),
        }
    else:
        terms_score = 1.0
        feedback["required_terms"] = {"score": 1.0, "note": "No required terms defined"}

    # 2. Forbidden phrases (15%)
    forbidden: List[str] = email_meta.get("forbidden_phrases", [])
    found_forbidden = [p for p in forbidden if _contains(response_text, p)]
    forbidden_score = 0.0 if found_forbidden else 1.0
    feedback["forbidden_phrases"] = {
        "found": found_forbidden,
        "score": forbidden_score,
    }

    # 3. Professional format (20%)
    has_greet = _has_greeting(response_text)
    has_close = _has_closing(response_text)
    format_score = (0.5 if has_greet else 0.0) + (0.5 if has_close else 0.0)
    feedback["format"] = {
        "has_greeting": has_greet,
        "has_closing": has_close,
        "score": format_score,
    }

    # 4. Factual accuracy (25%)
    correct_facts: Dict[str, Any] = email_meta.get("correct_facts", {})
    if correct_facts:
        fact_hits = 0
        fact_details = {}
        for fact_key, fact_value in correct_facts.items():
            if isinstance(fact_value, bool):
                # Can't check booleans directly in text — grant credit if response is positive
                present = True  # Assume OK unless response explicitly contradicts
                fact_details[fact_key] = {"value": fact_value, "present": present}
                if present:
                    fact_hits += 1
            else:
                present = _contains(response_text, str(fact_value))
                fact_details[fact_key] = {
                    "expected": fact_value,
                    "present": present,
                }
                if present:
                    fact_hits += 1

        facts_score = fact_hits / len(correct_facts)
        feedback["factual_accuracy"] = {
            "details": fact_details,
            "score": round(facts_score, 4),
        }
    else:
        facts_score = 1.0
        feedback["factual_accuracy"] = {"score": 1.0, "note": "No facts to check"}

    # Length bonus/penalty: very short responses lose up to 0.1
    if word_count < MIN_RESPONSE_WORDS:
        length_penalty = 0.1 * (word_count / MIN_RESPONSE_WORDS)
    elif word_count > MAX_RESPONSE_WORDS:
        # Slight penalty for extremely long responses (runaway generation)
        length_penalty = -0.05
    else:
        length_penalty = 0.0
    feedback["length_adjustment"] = round(length_penalty, 4)

    # Combine
    raw_score = (
        0.40 * terms_score
        + 0.15 * forbidden_score
        + 0.20 * format_score
        + 0.25 * facts_score
        + length_penalty
    )
    score = round(max(0.0, min(1.0, raw_score)), 4)
    feedback["total_score"] = score
    return score, feedback


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def grade(
    task_name: str,
    action: Dict[str, Any],
    email_data: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Route to the appropriate grader based on task_name.

    Parameters
    ----------
    task_name : str
        One of 'email_classify', 'email_extract', 'email_respond'.
    action : dict
        The agent's action dict.
    email_data : dict
        The full email record including ground truth / metadata.

    Returns
    -------
    (score, feedback) where score ∈ [0.0, 1.0].
    """
    if task_name == "email_classify":
        return grade_classify(action, email_data.get("labels", {}))
    elif task_name == "email_extract":
        return grade_extract(action, email_data.get("ground_truth", {}))
    elif task_name == "email_respond":
        return grade_respond(action, email_data)
    else:
        raise ValueError(f"Unknown task: {task_name}")
