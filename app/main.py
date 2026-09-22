from contextlib import asynccontextmanager
import os
import csv
import io
from fastapi import FastAPI,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.config import settings
from app.db import init_db,log_query,upsert_claim,add_training_example,list_claims,list_logs,training_status,generate_query_report,update_query_response
from app.dataset import find_match
from app.models import HealthPredictionRequest,QueryRequest,QueryResponse,SourceResponse,TrainResponse,QueryReportResponse
from app.services.llm import choose_route,call
from app.services.search import search
from app.services.validator import validate,score
from app.training.pipeline import train_if_ready
from app.services.health_model import predict_health_claim,predict_health_claims,train_health_model
from app.services.evidence_reasoner import reason_about_claim

@asynccontextmanager
async def lifespan(app):
    init_db(); yield
app=FastAPI(title=settings().app_name,version=settings().version,lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173","http://localhost:4173","http://127.0.0.1:4173"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"])

@app.get("/")
def root():return FileResponse("app/static/index.html")

@app.get("/api/status")
def api_status():return {"name":settings().app_name,"version":settings().version,"status":"running","docs":"/docs"}

@app.get("/health")
def health():
    s=settings()
    return {"status":"ok","database":s.db_path,
      "thresholds":{"auto_ingest":s.auto_ingest_threshold,"training":s.training_threshold,"dataset_match":s.dataset_match_threshold,"training_batch":s.training_batch_size},
      "llm_providers_configured":{"cheap":s.cheap_configured,"medium":s.medium_configured,"premium":s.premium_configured,"validator":s.validator_configured},
      "models":{"cheap":s.cheap_model,"medium":s.medium_model,"premium":s.premium_model,"validator":s.validator_model,"training_base":s.training_base_model},
      "training":{"enabled":s.training_enabled,**training_status()}}

@app.post("/health/predict")
async def health_predict(req:HealthPredictionRequest):
    return await reason_about_claim(req.claim)

@app.post("/health/train")
def health_train():
    return train_health_model()

@app.get("/health/dataset")
def health_dataset():
    from app.services.health_model import training_examples
    return {"examples": training_examples(), "count": len(training_examples())}

@app.post("/health/upload")
async def health_upload(file:UploadFile):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400,"Upload a CSV file.")
    raw=await file.read()
    if len(raw)>5*1024*1024:
        raise HTTPException(413,"CSV file must be smaller than 5 MB.")
    try:
        text=raw.decode("utf-8-sig")
        rows=list(csv.DictReader(io.StringIO(text)))
    except (UnicodeDecodeError,csv.Error) as exc:
        raise HTTPException(400,"Could not read the CSV file as UTF-8.") from exc
    if not rows:
        raise HTTPException(400,"CSV file has no data rows.")
    columns={str(column).strip().lower() for column in rows[0] if column}
    claim_column=next((name for name in ("claim","text","query") if name in columns),None)
    if not claim_column:
        raise HTTPException(400,"CSV must contain a claim, text, or query column.")
    original_column=next(column for column in rows[0] if str(column).strip().lower()==claim_column)
    claims=[str(row.get(original_column) or "").strip() for row in rows]
    if any(not claim for claim in claims):
        raise HTTPException(400,"Every row must contain a non-empty claim.")
    predictions=predict_health_claims(claims)
    label_column=next((column for column in rows[0] if str(column).strip().lower()=="label"),None)
    results=[]; correct=0; labeled=0
    for index,(row,prediction) in enumerate(zip(rows,predictions),1):
        expected=str(row.get(label_column) or "").strip().upper() if label_column else None
        is_correct=expected in ("SUPPORTED","CONTRADICTED","UNCERTAIN") and expected==prediction["verdict"]
        if expected in ("SUPPORTED","CONTRADICTED","UNCERTAIN"):
            labeled+=1; correct+=int(is_correct)
        results.append({"row":index,"claim":claims[index-1],"expected":expected,"verdict":prediction["verdict"],"confidence":prediction["confidence"],"correct":is_correct if expected else None})
    summary={"rows":len(results),"labeled_rows":labeled,"counts":{label:sum(item["verdict"]==label for item in results) for label in ("SUPPORTED","CONTRADICTED","UNCERTAIN")}}
    if labeled: summary["accuracy"]=round(correct/labeled*100,2)
    return {"filename":file.filename,"claim_column":original_column,"summary":summary,"results":results}

async def _generate_query_payload(q: str, route_hint: str | None = None):
    q=q.strip(); hit=find_match(q)
    if hit:
        return {
            "query": q,
            "answer": hit["answer"],
            "verdict": hit["verdict"],
            "status": "VERIFIED_DATASET",
            "validation_score": hit["validation_score"],
            "validation_status": "VERIFIED_DATASET",
            "validation_reason": "Retrieved from Satarka verified dataset.",
            "explanation": "This result came from a previously verified claim in the project dataset.",
            "route": "verified_dataset",
            "model_used": None,
            "dataset_hit": True,
            "sources": [{"title": "Verified source", "url": x["url"], "published_date": x.get("published_date")} for x in hit["sources"] if x.get("url")],
            "created_at": None,
        }
    initial=route_hint or choose_route(q)
    order={"cheap":["cheap","medium","premium"],"medium":["medium","premium","cheap"],"premium":["premium","medium","cheap"]}[initial]
    answer=model=None; errors=[]
    for route in order:
        available={"cheap":settings().cheap_configured,"medium":settings().medium_configured,"premium":settings().premium_configured}[route]
        if not available:continue
        try:
            answer,model=call(route,"You are Satarka's answer generator. Do not invent sources or confidence percentages.",q); initial=route; break
        except Exception as e:errors.append(f"{route}: {e}")
    if not answer:
        raise HTTPException(502,"No LLM route succeeded. "+"; ".join(errors))
    try:sources=await search(q)
    except Exception as e:sources=[]; search_error=str(e)
    else:search_error=None
    if not sources:
        return {
            "query": q,
            "answer": answer,
            "verdict": "UNCERTAIN",
            "status": "INSUFFICIENT_EVIDENCE",
            "validation_score": 0,
            "validation_status": "INSUFFICIENT_EVIDENCE",
            "validation_reason": "No independent evidence was retrieved; claim was not stored."+(f" Search error: {search_error}" if search_error else ""),
            "explanation": "The system could not retrieve independent supporting sources, so it cannot safely confirm or contradict this claim.",
            "route": initial,
            "model_used": model,
            "dataset_hit": False,
            "sources": [],
            "created_at": None,
        }
    if not settings().validator_configured:raise HTTPException(502,"Validator is not configured.")
    try:r=validate(q,answer,sources); val,status,reason,components=score(r)
    except Exception as e:raise HTTPException(502,f"Evidence validation failed: {e}") from e
    evidence=[e for e in r.get("evidence",[]) if isinstance(e,dict) and e.get("url")]
    payload={
        "query": q,
        "answer": str(r.get("answer") or answer),
        "verdict": str(r.get("verdict","UNCERTAIN")),
        "status": status,
        "validation_score": val,
        "validation_status": status,
        "validation_reason": reason,
        "validation_components": components,
        "explanation": str(r.get("explanation") or reason),
        "route": initial,
        "model_used": model,
        "dataset_hit": False,
        "sources": [{"title": s.get("title") or s.get("url") or "Source", "url": s.get("url"), "published_date": s.get("published_date")} for s in sources if s.get("url")],
        "evidence": evidence,
        "created_at": None,
    }
    return payload

@app.post("/query",response_model=QueryResponse)
async def query(req:QueryRequest):
    q=req.query.strip(); payload=await _generate_query_payload(q)
    if payload.get("status") == "VERIFIED_DATASET":
        response = QueryResponse(answer=payload["answer"],verdict=payload["verdict"],route=payload["route"],dataset_hit=payload.get("dataset_hit",False),dataset_similarity=None,validation_score=payload.get("validation_score"),validation_status=payload.get("validation_status","VERIFIED_DATASET"),validation_reason=payload.get("validation_reason"),explanation=payload.get("explanation"),sources=[SourceResponse(title=s["title"],url=s["url"],published_date=s.get("published_date")) for s in payload.get("sources",[]) if s.get("url")],claim_id=None)
        log_query(q,payload["route"],payload.get("model_used"),bool(payload.get("dataset_hit",False)),payload.get("validation_score"),payload.get("status"),verdict=payload.get("verdict"),answer=payload.get("answer"),response_payload=payload)
        return response
    if payload.get("status") == "INSUFFICIENT_EVIDENCE":
        log_query(q,payload["route"],payload.get("model_used"),False,payload.get("validation_score",0),payload.get("status"),verdict=payload.get("verdict"),answer=payload.get("answer"),response_payload=payload)
        return QueryResponse(answer=payload["answer"],verdict=payload["verdict"],route=payload["route"],model_used=payload.get("model_used"),dataset_hit=False,validation_score=payload.get("validation_score",0),validation_status=payload.get("validation_status","INSUFFICIENT_EVIDENCE"),validation_reason=payload.get("validation_reason"),explanation=payload.get("explanation"),sources=[])
    log_query(q,payload["route"],payload.get("model_used"),False,payload.get("validation_score"),payload.get("status"),verdict=payload.get("verdict"),answer=payload.get("answer"),response_payload=payload)
    evidence=[e for e in payload.get("evidence",[]) if isinstance(e,dict) and e.get("url")]
    return QueryResponse(answer=payload["answer"],verdict=payload["verdict"],route=payload["route"],model_used=payload.get("model_used"),dataset_hit=False,validation_score=payload.get("validation_score"),validation_status=payload.get("validation_status") or payload.get("status"),validation_reason=payload.get("validation_reason"),validation_components=payload.get("validation_components",{}),explanation=payload.get("explanation"),evidence=evidence,sources=[SourceResponse(title=s["title"],url=s["url"],published_date=s.get("published_date")) for s in payload.get("sources",[]) if s.get("url")])

@app.get("/report", response_model=QueryReportResponse)
async def report(limit:int=20):
    rows=list_logs(max(1,min(int(limit),1000)))
    for row in rows:
        payload=row.get("response_json") or {}
        if (not payload.get("answer") and not row.get("answer")) or (not payload.get("response") and row.get("query")):
            try:
                refreshed = await _generate_query_payload(row.get("query") or "")
                update_query_response(row["id"], row.get("query") or "", refreshed.get("route") or row.get("route"), refreshed.get("model_used") or row.get("model_used"), refreshed.get("dataset_hit", bool(row.get("dataset_hit"))), refreshed.get("validation_score", row.get("validation_score")), refreshed.get("status") or row.get("status"), verdict=refreshed.get("verdict"), answer=refreshed.get("answer"), response_payload=refreshed)
            except Exception:
                pass
    return generate_query_report(limit)

@app.get("/dataset")
def dataset():return list_claims()
@app.get("/logs")
def logs(limit:int=100):return list_logs(max(1,min(limit,1000)))
@app.get("/training/status")
def training():return {"training":training_status(),"config":{"enabled":settings().training_enabled,"base_model":settings().training_base_model,"batch_size":settings().training_batch_size,"threshold":settings().training_threshold}}
@app.post("/training/run",response_model=TrainResponse)
def run_training(force:bool=False):
    r=train_if_ready(force); return TrainResponse(status=r["status"],examples=r.get("examples",0),output_dir=r.get("output_dir"),message=r["message"])
if __name__=="__main__":
    import uvicorn
    uvicorn.run("app.main:app",host="127.0.0.1",port=int(os.getenv("SATARKA_PORT","8000")),reload=True)
