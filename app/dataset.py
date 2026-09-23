import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.db import connection

INDEX_FILE = Path("data/faiss_index.bin")
METADATA_FILE = Path("data/faiss_metadata.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

_model: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.IndexFlatIP] = None
_claims_metadata: List[Dict[str, Any]] = []


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def _get_db_rows() -> List[Dict[str, Any]]:
    c = connection()
    try:
        rows = c.execute("SELECT * FROM verified_claims ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()


def _get_db_state_token() -> str:
    c = connection()
    try:
        row = c.execute(
            "SELECT COUNT(*) as count, MAX(updated_at) as max_updated FROM verified_claims"
        ).fetchone()
        return f"{row['count']}_{row['max_updated']}"
    except Exception:
        return "0_none"
    finally:
        c.close()


def refresh_index() -> Dict[str, Any]:
    """Rebuilds the FAISS index from verified_claims table and saves to disk cache."""
    global _faiss_index, _claims_metadata

    rows = _get_db_rows()
    if not rows:
        _faiss_index = None
        _claims_metadata = []
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
        return {"status": "empty", "total_claims": 0}

    model = get_model()
    claims_text = [r["canonical_claim"] for r in rows]

    embeddings = model.encode(claims_text, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _faiss_index = index
    _claims_metadata = rows

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    state_token = _get_db_state_token()
    cache_payload = {
        "state_token": state_token,
        "claims_metadata": rows
    }
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(cache_payload, f)

    return {"status": "refreshed", "total_claims": len(rows), "dimension": dimension}


def _ensure_index_loaded() -> None:
    global _faiss_index, _claims_metadata

    if _faiss_index is not None and len(_claims_metadata) > 0:
        return

    current_state_token = _get_db_state_token()

    if INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "rb") as f:
                cached_data = pickle.load(f)
            if cached_data.get("state_token") == current_state_token:
                index = faiss.read_index(str(INDEX_FILE))
                _faiss_index = index
                _claims_metadata = cached_data.get("claims_metadata", [])
                return
        except Exception:
            pass

    refresh_index()


def get_top_k_matches(
    query: str,
    k: int = 5,
    threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Retrieves top-K semantic matches from FAISS using cosine similarity."""
    _ensure_index_loaded()

    if _faiss_index is None or _faiss_index.ntotal == 0 or not _claims_metadata:
        return []

    effective_threshold = threshold if threshold is not None else 0.0

    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)

    search_k = min(k, _faiss_index.ntotal)
    scores, indices = _faiss_index.search(q_emb, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_claims_metadata):
            continue

        cosine_sim = float(score)
        if cosine_sim < effective_threshold:
            continue

        row = _claims_metadata[idx]
        sources_list = json.loads(row["sources_json"] or "[]") if isinstance(row["sources_json"], str) else (row["sources_json"] or [])

        results.append({
            "id": int(row["id"]),
            "canonical_claim": row["canonical_claim"],
            "answer": row["answer"],
            "verdict": row["verdict"],
            "similarity": round(cosine_sim * 100, 2),
            "similarity_score": round(cosine_sim, 4),
            "validation_score": float(row["validation_score"]),
            "sources": sources_list,
        })

    return results


def find_match(query: str, threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Single top-match retrieval if cosine similarity exceeds threshold."""
    min_thresh = threshold if threshold is not None else settings().dataset_match_threshold
    matches = get_top_k_matches(query, k=1, threshold=min_thresh)
    return matches[0] if matches else None
