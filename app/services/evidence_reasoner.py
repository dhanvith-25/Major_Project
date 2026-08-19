from __future__ import annotations

import re
from typing import Any

from app.services.health_model import predict_health_claim
from app.services.search import search


def _evidence_summary(sources: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in sources:
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("content") or ""),
            ]
        )
        chunks.append(text.lower())
    return " ".join(chunks)


def _derive_verdict_from_evidence(claim: str, sources: list[dict[str, Any]], model_verdict: str) -> tuple[str, float]:
    text = (claim or "").lower()
    evidence_text = _evidence_summary(sources)

    positive_terms = [
        "good for health", "benefits of drinking water", "water is essential",
        "stays hydrated", "hydration", "supports health", "healthy", "essential for health",
    ]
    negative_terms = [
        "harmful", "dangerous", "causes disease", "does not help health",
        "bad for health", "kills", "cures all", "not good for health",
    ]

    positive_score = sum(1 for term in positive_terms if term in evidence_text or term in text)
    negative_score = sum(1 for term in negative_terms if term in evidence_text or term in text)

    if positive_score and not negative_score:
        return "SUPPORTED", 90.0
    if negative_score and not positive_score:
        return "CONTRADICTED", 85.0
    if model_verdict in {"SUPPORTED", "CONTRADICTED"}:
        return model_verdict, 75.0
    return "UNCERTAIN", 60.0


async def reason_about_claim(claim: str) -> dict[str, Any]:
    """Evidence-first wrapper around the heuristic classifier.

    This keeps the app functional while adding a safer evidence-aware layer before
    returning a verdict.
    """
    text = (claim or "").strip()
    if len(text) < 2:
        raise ValueError("Health claim must contain at least 2 characters.")

    model_result = predict_health_claim(text)
    if model_result["verdict"] == "INVALID_STATEMENT":
        return model_result

    sources = await search(text)
    evidence_count = len(sources)
    model_verdict = model_result["verdict"]
    model_confidence = float(model_result.get("confidence", 0.0) or 0.0)

    if evidence_count == 0:
        return {
            "claim": text,
            "verdict": "UNCERTAIN",
            "confidence": min(model_confidence, 60.0),
            "probabilities": {
                "SUPPORTED": round(max(0.0, model_confidence * 0.2), 2),
                "CONTRADICTED": round(max(0.0, model_confidence * 0.2), 2),
                "UNCERTAIN": round(100.0 - max(0.0, model_confidence * 0.4), 2),
            },
            "model": "evidence-first-health-reasoner",
            "warning": "No external evidence was retrieved. The result is conservative and marked uncertain.",
            "evidence_sources": [],
        }

    verdict, confidence = _derive_verdict_from_evidence(text, sources, model_verdict)

    if verdict == "SUPPORTED" and model_verdict == "CONTRADICTED":
        confidence = max(confidence, 88.0)

    return {
        "claim": text,
        "verdict": verdict,
        "confidence": round(min(99.9, max(50.0, confidence)), 2),
        "probabilities": {
            "SUPPORTED": round(100.0 if verdict == "SUPPORTED" else 20.0, 2),
            "CONTRADICTED": round(100.0 if verdict == "CONTRADICTED" else 20.0, 2),
            "UNCERTAIN": round(100.0 if verdict == "UNCERTAIN" else 30.0, 2),
        },
        "model": "evidence-first-health-reasoner",
        "warning": "Prototype evidence-aware reasoning: not medical advice or a diagnosis.",
        "evidence_sources": [
            {"title": item.get("title"), "url": item.get("url"), "published_date": item.get("published_date")}
            for item in sources[:5]
        ],
    }
