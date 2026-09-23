import csv
import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.dataset import find_match, get_top_k_matches, refresh_index
from app.db import (
    add_training_example,
    generate_query_report,
    init_db,
    list_claims,
    list_logs,
    log_query,
    training_status,
    update_query_response,
    upsert_claim,
)
from app.models import (
    HealthPredictionRequest,
    QueryReportResponse,
    QueryRequest,
    QueryResponse,
    SemanticMatchResponse,
    SemanticSearchRequest,
    SourceResponse,
    TrainResponse,
)
from app.services.evidence_reasoner import reason_about_claim
from app.services.health_model import (
    predict_health_claim,
    predict_health_claims,
    train_health_model,
)
from app.services.llm import call, choose_route
from app.services.search import search
from app.services.symptom_checker import symptom_checker
from app.services.validator import score, validate
from app.training.pipeline import train_if_ready


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(title=settings().app_name, version=settings().version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.get("/api/status")
def api_status():
    return {
        "name": settings().app_name,
        "version": settings().version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    s = settings()
    return {
        "status": "ok",
        "database": s.db_path,
        "thresholds": {
            "auto_ingest": s.auto_ingest_threshold,
            "training": s.training_threshold,
            "dataset_match": s.dataset_match_threshold,
            "training_batch": s.training_batch_size,
        },
        "llm_providers_configured": {
            "cheap": s.cheap_configured,
            "medium": s.medium_configured,
            "premium": s.premium_configured,
            "validator": s.validator_configured,
        },
        "models": {
            "cheap": s.cheap_model,
            "medium": s.medium_model,
            "premium": s.premium_model,
            "validator": s.validator_model,
            "training_base": s.training_base_model,
        },
        "training": {"enabled": s.training_enabled, **training_status()},
    }


@app.post("/health/predict")
async def health_predict(req: HealthPredictionRequest):
    return await reason_about_claim(req.claim)


@app.post("/health/train")
def health_train():
    return train_health_model()


@app.get("/health/dataset")
def health_dataset():
    from app.services.health_model import training_examples

    return {"examples": training_examples(), "count": len(training_examples())}


@app.post("/health/upload")
async def health_upload(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file.")
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(413, "CSV file must be smaller than 5 MB.")
    try:
        text = raw.decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(400, "Could not read the CSV file as UTF-8.") from exc
    if not rows:
        raise HTTPException(400, "CSV file has no data rows.")
    columns = {str(column).strip().lower() for column in rows[0] if column}
    claim_column = next(
        (name for name in ("claim", "text", "query") if name in columns), None
    )
    if not claim_column:
        raise HTTPException(400, "CSV must contain a claim, text, or query column.")
    original_column = next(
        column for column in rows[0] if str(column).strip().lower() == claim_column
    )
    claims = [str(row.get(original_column) or "").strip() for row in rows]
    if any(not claim for claim in claims):
        raise HTTPException(400, "Every row must contain a non-empty claim.")
    predictions = predict_health_claims(claims)
    label_column = next(
        (column for column in rows[0] if str(column).strip().lower() == "label"), None
    )
    results = []
    correct = 0
    labeled = 0
    for index, (row, prediction) in enumerate(zip(rows, predictions), 1):
        expected = (
            str(row.get(label_column) or "").strip().upper() if label_column else None
        )
        is_correct = (
            expected in ("SUPPORTED", "CONTRADICTED", "UNCERTAIN")
            and expected == prediction["verdict"]
        )
        if expected in ("SUPPORTED", "CONTRADICTED", "UNCERTAIN"):
            labeled += 1
            correct += int(is_correct)
        results.append(
            {
                "row": index,
                "claim": claims[index - 1],
                "expected": expected,
                "verdict": prediction["verdict"],
                "confidence": prediction["confidence"],
                "correct": is_correct if expected else None,
            }
        )
    summary = {
        "rows": len(results),
        "labeled_rows": labeled,
        "counts": {
            label: sum(item["verdict"] == label for item in results)
            for label in ("SUPPORTED", "CONTRADICTED", "UNCERTAIN")
        },
    }
    if labeled:
        summary["accuracy"] = round(correct / labeled * 100, 2)
    return {
        "filename": file.filename,
        "claim_column": original_column,
        "summary": summary,
        "results": results,
    }


async def _generate_query_payload(q: str, route_hint: str | None = None):
    q = q.strip()
    top_matches = get_top_k_matches(q, k=5)
    hit = (
        top_matches[0]
        if (
            top_matches
            and top_matches[0]["similarity_score"] >= settings().dataset_match_threshold
        )
        else None
    )

    formatted_semantic_matches = [
        {
            "id": m["id"],
            "canonical_claim": m["canonical_claim"],
            "answer": m["answer"],
            "verdict": m["verdict"],
            "similarity": m["similarity"],
            "similarity_score": m["similarity_score"],
            "validation_score": m["validation_score"],
            "sources": [
                {
                    "title": s.get("title") or "Verified source",
                    "url": s.get("url"),
                    "published_date": s.get("published_date"),
                }
                for s in m["sources"]
                if isinstance(s, dict) and s.get("url")
            ],
        }
        for m in top_matches
    ]

    if hit:
        return {
            "query": q,
            "answer": hit["answer"],
            "verdict": hit["verdict"],
            "status": "VERIFIED_DATASET",
            "validation_score": hit["validation_score"],
            "validation_status": "VERIFIED_DATASET",
            "validation_reason": "Retrieved from Satarka verified dataset via FAISS semantic search.",
            "explanation": f"This claim semantically matches a previously verified claim in the project dataset with {hit['similarity']}% similarity.",
            "route": "verified_dataset",
            "model_used": "faiss_all-MiniLM-L6-v2",
            "dataset_hit": True,
            "dataset_similarity": hit["similarity_score"],
            "semantic_matches": formatted_semantic_matches,
            "sources": [
                {
                    "title": "Verified source",
                    "url": x["url"],
                    "published_date": x.get("published_date"),
                }
                for x in hit["sources"]
                if isinstance(x, dict) and x.get("url")
            ],
            "created_at": None,
        }

    initial = route_hint or choose_route(q)
    order = {
        "cheap": ["cheap", "medium", "premium"],
        "medium": ["medium", "premium", "cheap"],
        "premium": ["premium", "medium", "cheap"],
    }[initial]
    answer = model = None
    errors = []
    for route in order:
        available = {
            "cheap": settings().cheap_configured,
            "medium": settings().medium_configured,
            "premium": settings().premium_configured,
        }[route]
        if not available:
            continue
        try:
            answer, model = call(
                route,
                "You are Satarka's answer generator. Do not invent sources or confidence percentages.",
                q,
            )
            initial = route
            break
        except Exception as e:
            errors.append(f"{route}: {e}")

    if not answer:
        raise HTTPException(502, "No LLM route succeeded. " + "; ".join(errors))

    try:
        sources = await search(q)
    except Exception as e:
        sources = []
        search_error = str(e)
    else:
        search_error = None

    if not sources:
        return {
            "query": q,
            "answer": answer,
            "verdict": "UNCERTAIN",
            "status": "INSUFFICIENT_EVIDENCE",
            "validation_score": 0,
            "validation_status": "INSUFFICIENT_EVIDENCE",
            "validation_reason": "No independent evidence was retrieved; claim was not stored."
            + (f" Search error: {search_error}" if search_error else ""),
            "explanation": "The system could not retrieve independent supporting sources, so it cannot safely confirm or contradict this claim.",
            "route": initial,
            "model_used": model,
            "dataset_hit": False,
            "dataset_similarity": top_matches[0]["similarity_score"] if top_matches else None,
            "semantic_matches": formatted_semantic_matches,
            "sources": [],
            "created_at": None,
        }

    if not settings().validator_configured:
        raise HTTPException(502, "Validator is not configured.")

    try:
        r = validate(q, answer, sources)
        val, status, reason, components = score(r)
    except Exception as e:
        raise HTTPException(502, f"Evidence validation failed: {e}") from e

    evidence = [e for e in r.get("evidence", []) if isinstance(e, dict) and e.get("url")]
    payload = {
        "query": q,
        "answer": str(r.get("answer") or answer),
        "verdict": str(r.get("verdict", "UNCERTAIN")),
        "status": status,
        "validation_score": val,
        "validation_status": status,
        "validation_reason": reason,
        "validation_components": components,
        "explanation": str(r.get("explanation") or reason),
        "route": initial,
        "model_used": model,
        "dataset_hit": False,
        "dataset_similarity": top_matches[0]["similarity_score"] if top_matches else None,
        "semantic_matches": formatted_semantic_matches,
        "sources": [
            {
                "title": s.get("title") or s.get("url") or "Source",
                "url": s.get("url"),
                "published_date": s.get("published_date"),
            }
            for s in sources
            if s.get("url")
        ],
        "evidence": evidence,
        "created_at": None,
    }
    return payload


@app.post("/symptom-check")
async def symptom_check_endpoint(
    image: UploadFile | None = File(default=None), symptoms: str = Form(default="")
):
    if image is None:
        return symptom_checker(None, symptoms)

    try:
        image_bytes = await image.read()
        if not image_bytes:
            return {
                **symptom_checker(None, symptoms),
                "ocr_status": "No image data received",
                "ocr_warning": "No image content was uploaded. Please upload a valid PNG/JPG image or type the symptoms manually.",
            }

        uploaded = io.BytesIO(image_bytes)
        uploaded.name = image.filename or "upload.png"
        return symptom_checker(uploaded, symptoms)
    except Exception as exc:
        fallback = symptom_checker(None, symptoms)
        fallback["ocr_status"] = "Upload failed"
        fallback["ocr_warning"] = (
            f"Image upload failed: {exc}. Please retry with a valid PNG/JPG image or enter symptoms manually."
        )
        return fallback


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    q = req.query.strip()
    payload = await _generate_query_payload(q)

    semantic_matches = [
        SemanticMatchResponse(
            id=m["id"],
            canonical_claim=m["canonical_claim"],
            answer=m["answer"],
            verdict=m["verdict"],
            similarity=m["similarity"],
            similarity_score=m["similarity_score"],
            validation_score=m["validation_score"],
            sources=[
                SourceResponse(
                    title=s.get("title") or "Verified source",
                    url=s["url"],
                    published_date=s.get("published_date"),
                )
                for s in m.get("sources", [])
                if isinstance(s, dict) and s.get("url")
            ],
        )
        for m in payload.get("semantic_matches", [])
    ]

    if payload.get("status") == "VERIFIED_DATASET":
        response = QueryResponse(
            answer=payload["answer"],
            verdict=payload["verdict"],
            route=payload["route"],
            model_used=payload.get("model_used"),
            dataset_hit=payload.get("dataset_hit", False),
            dataset_similarity=payload.get("dataset_similarity"),
            semantic_matches=semantic_matches,
            validation_score=payload.get("validation_score"),
            validation_status=payload.get("validation_status", "VERIFIED_DATASET"),
            validation_reason=payload.get("validation_reason"),
            explanation=payload.get("explanation"),
            sources=[
                SourceResponse(
                    title=s["title"],
                    url=s["url"],
                    published_date=s.get("published_date"),
                )
                for s in payload.get("sources", [])
                if s.get("url")
            ],
            claim_id=None,
        )
        log_query(
            q,
            payload["route"],
            payload.get("model_used"),
            bool(payload.get("dataset_hit", False)),
            payload.get("validation_score"),
            payload.get("status"),
            verdict=payload.get("verdict"),
            answer=payload.get("answer"),
            response_payload=payload,
        )
        return response

    if payload.get("status") == "INSUFFICIENT_EVIDENCE":
        log_query(
            q,
            payload["route"],
            payload.get("model_used"),
            False,
            payload.get("validation_score", 0),
            payload.get("status"),
            verdict=payload.get("verdict"),
            answer=payload.get("answer"),
            response_payload=payload,
        )
        return QueryResponse(
            answer=payload["answer"],
            verdict=payload["verdict"],
            route=payload["route"],
            model_used=payload.get("model_used"),
            dataset_hit=False,
            dataset_similarity=payload.get("dataset_similarity"),
            semantic_matches=semantic_matches,
            validation_score=payload.get("validation_score", 0),
            validation_status=payload.get("validation_status", "INSUFFICIENT_EVIDENCE"),
            validation_reason=payload.get("validation_reason"),
            explanation=payload.get("explanation"),
            sources=[],
        )

    log_query(
        q,
        payload["route"],
        payload.get("model_used"),
        False,
        payload.get("validation_score"),
        payload.get("status"),
        verdict=payload.get("verdict"),
        answer=payload.get("answer"),
        response_payload=payload,
    )
    evidence = [e for e in payload.get("evidence", []) if isinstance(e, dict) and e.get("url")]
    return QueryResponse(
        answer=payload["answer"],
        verdict=payload["verdict"],
        route=payload["route"],
        model_used=payload.get("model_used"),
        dataset_hit=False,
        dataset_similarity=payload.get("dataset_similarity"),
        semantic_matches=semantic_matches,
        validation_score=payload.get("validation_score"),
        validation_status=payload.get("validation_status") or payload.get("status"),
        validation_reason=payload.get("validation_reason"),
        validation_components=payload.get("validation_components", {}),
        explanation=payload.get("explanation"),
        evidence=evidence,
        sources=[
            SourceResponse(
                title=s["title"],
                url=s["url"],
                published_date=s.get("published_date"),
            )
            for s in payload.get("sources", [])
            if s.get("url")
        ],
    )


@app.post("/dataset/search", response_model=list[SemanticMatchResponse])
def search_dataset(req: SemanticSearchRequest):
    matches = get_top_k_matches(req.query, k=req.k, threshold=req.threshold)
    return [
        SemanticMatchResponse(
            id=m["id"],
            canonical_claim=m["canonical_claim"],
            answer=m["answer"],
            verdict=m["verdict"],
            similarity=m["similarity"],
            similarity_score=m["similarity_score"],
            validation_score=m["validation_score"],
            sources=[
                SourceResponse(
                    title=s.get("title") or "Verified source",
                    url=s["url"],
                    published_date=s.get("published_date"),
                )
                for s in m["sources"]
                if isinstance(s, dict) and s.get("url")
            ],
        )
        for m in matches
    ]


@app.post("/dataset/refresh-index")
def refresh_dataset_index():
    return refresh_index()


@app.get("/report", response_model=QueryReportResponse)
async def report(limit: int = 20):
    rows = list_logs(max(1, min(int(limit), 1000)))
    for row in rows:
        payload = row.get("response_json") or {}
        if not payload.get("answer") and not row.get("answer"):
            try:
                refreshed = await _generate_query_payload(row.get("query") or "")
                update_query_response(
                    row["id"],
                    row.get("query") or "",
                    refreshed.get("route") or row.get("route"),
                    refreshed.get("model_used") or row.get("model_used"),
                    refreshed.get("dataset_hit", bool(row.get("dataset_hit"))),
                    refreshed.get("validation_score", row.get("validation_score")),
                    refreshed.get("status") or row.get("status"),
                    verdict=refreshed.get("verdict"),
                    answer=refreshed.get("answer"),
                    response_payload=refreshed,
                )
            except Exception:
                pass
    return generate_query_report(limit)


@app.get("/dataset")
def dataset():
    return list_claims()


@app.get("/logs")
def logs(limit: int = 100):
    return list_logs(max(1, min(limit, 1000)))


@app.get("/training/status")
def training():
    return {
        "training": training_status(),
        "config": {
            "enabled": settings().training_enabled,
            "base_model": settings().training_base_model,
            "batch_size": settings().training_batch_size,
            "threshold": settings().training_threshold,
        },
    }


@app.post("/training/run", response_model=TrainResponse)
def run_training(force: bool = False):
    r = train_if_ready(force)
    return TrainResponse(
        status=r["status"],
        examples=r.get("examples", 0),
        output_dir=r.get("output_dir"),
        message=r["message"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=int(os.getenv("SATARKA_PORT", "8000")),
        reload=True,
    )
