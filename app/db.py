import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from app.config import settings

def now(): return datetime.now(timezone.utc).isoformat()

def normalize_verdict(verdict=None, status=None):
    raw = str(verdict or "").strip().upper()
    if not raw:
        raw = str(status or "").strip().upper()
    mapping = {
        "VERIFIED_DATASET": "SUPPORTED",
        "VERIFIED": "SUPPORTED",
        "SUPPORTED": "SUPPORTED",
        "CONTRADICTED": "CONTRADICTED",
        "UNCERTAIN": "UNCERTAIN",
        "INVALID_STATEMENT": "INVALID_STATEMENT",
        "INSUFFICIENT_EVIDENCE": "UNCERTAIN",
        "LOW_CONFIDENCE": "UNCERTAIN",
        "ERROR": "UNCERTAIN",
        "UNKNOWN": "UNKNOWN",
    }
    return mapping.get(raw, raw or "UNKNOWN")

def connection():
    p=Path(settings().db_path); p.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(str(p)); c.row_factory=sqlite3.Row; return c

def init_db():
    c=connection()
    try:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS verified_claims(
          id INTEGER PRIMARY KEY AUTOINCREMENT, canonical_claim TEXT NOT NULL,
          answer TEXT NOT NULL, verdict TEXT NOT NULL, validation_score REAL NOT NULL,
          source_count INTEGER NOT NULL DEFAULT 0, sources_json TEXT NOT NULL DEFAULT '[]',
          model_used TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_claims_updated ON verified_claims(updated_at);
        CREATE TABLE IF NOT EXISTS training_examples(
          id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id INTEGER NOT NULL UNIQUE,
          prompt TEXT NOT NULL, completion TEXT NOT NULL, validation_score REAL NOT NULL,
          created_at TEXT NOT NULL, trained INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS training_runs(
          id INTEGER PRIMARY KEY AUTOINCREMENT, base_model TEXT NOT NULL,
          output_dir TEXT NOT NULL, examples_count INTEGER NOT NULL,
          status TEXT NOT NULL, metrics_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS query_log(
          id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, route TEXT NOT NULL,
          model_used TEXT, dataset_hit INTEGER NOT NULL, validation_score REAL,
          status TEXT NOT NULL, error TEXT, verdict TEXT, answer TEXT,
          response_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
        """); c.commit()
        existing_cols={row[1] for row in c.execute("PRAGMA table_info(query_log)").fetchall()}
        for col_name, ddl in {
            "verdict":"ALTER TABLE query_log ADD COLUMN verdict TEXT",
            "answer":"ALTER TABLE query_log ADD COLUMN answer TEXT",
            "response_json":"ALTER TABLE query_log ADD COLUMN response_json TEXT NOT NULL DEFAULT '{}'",
        }.items():
            if col_name not in existing_cols:
                c.execute(ddl)
        c.commit()
    finally: c.close()

def log_query(q,route,model,hit,score,status,error=None,verdict=None,answer=None,response_payload=None):
    response_payload=response_payload or {}
    if verdict is None and isinstance(response_payload,dict):
        verdict=response_payload.get("verdict")
    if answer is None and isinstance(response_payload,dict):
        answer=response_payload.get("answer")
    normalized_verdict = normalize_verdict(verdict, status)
    full_payload={
        "query": q,
        "route": route,
        "model_used": model,
        "dataset_hit": bool(hit),
        "validation_score": score,
        "status": status,
        "verdict": normalized_verdict,
        "answer": answer,
        "error": error,
        "created_at": now(),
        **response_payload,
    }
    full_payload["verdict"] = normalize_verdict(full_payload.get("verdict"), full_payload.get("status"))
    c=connection()
    try:
        c.execute("INSERT INTO query_log(query,route,model_used,dataset_hit,validation_score,status,error,verdict,answer,response_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (q,route,model,int(hit),score,status,error,full_payload.get("verdict"),answer or "",json.dumps(full_payload),full_payload.get("created_at") or now())); c.commit()
    finally: c.close()

def upsert_claim(claim,answer,verdict,score,sources,model):
    c=connection()
    try:
        r=c.execute("SELECT id FROM verified_claims WHERE lower(canonical_claim)=lower(?) LIMIT 1",(claim,)).fetchone()
        if r:
            cid=int(r["id"])
            c.execute("UPDATE verified_claims SET answer=?,verdict=?,validation_score=?,source_count=?,sources_json=?,model_used=?,updated_at=? WHERE id=?",
                      (answer,verdict,score,len(sources),json.dumps(sources),model,now(),cid))
        else:
            cur=c.execute("INSERT INTO verified_claims(canonical_claim,answer,verdict,validation_score,source_count,sources_json,model_used,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                          (claim,answer,verdict,score,len(sources),json.dumps(sources),model,now(),now()))
            cid=int(cur.lastrowid)
        c.commit()
        try:
            from app.dataset import refresh_index
            refresh_index()
        except Exception:
            pass
        return cid
    finally: c.close()

def add_training_example(cid,prompt,completion,score):
    c=connection()
    try:
        c.execute("INSERT OR IGNORE INTO training_examples(claim_id,prompt,completion,validation_score,created_at) VALUES(?,?,?,?,?)",
                  (cid,prompt,completion,score,now())); c.commit()
    finally: c.close()

def pending_training():
    c=connection()
    try: return c.execute("SELECT * FROM training_examples WHERE trained=0 ORDER BY id").fetchall()
    finally: c.close()

def mark_trained(ids):
    if not ids:return
    c=connection()
    try:
        c.executemany("UPDATE training_examples SET trained=1 WHERE id=?",[(int(i),) for i in ids]); c.commit()
    finally:c.close()

def training_status():
    c=connection()
    try:
        a=c.execute("SELECT COUNT(*) n FROM training_examples").fetchone()["n"]
        b=c.execute("SELECT COUNT(*) n FROM training_examples WHERE trained=0").fetchone()["n"]
        return {"total_examples":int(a),"pending_examples":int(b)}
    finally:c.close()

def list_claims():
    c=connection()
    try:return [dict(r) for r in c.execute("SELECT * FROM verified_claims ORDER BY updated_at DESC").fetchall()]
    finally:c.close()

def list_logs(limit):
    c=connection()
    try:
        logs=[]
        for row in c.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT ?",(limit,)).fetchall():
            item=dict(row)
            raw=item.get("response_json") or "{}"
            try: item["response_json"]=json.loads(raw)
            except Exception: item["response_json"]={}
            logs.append(item)
        return logs
    finally:c.close()


def update_query_response(log_id, q, route, model, hit, score, status, error=None, verdict=None, answer=None, response_payload=None):
    response_payload=response_payload or {}
    updated_at=now()
    answer = answer or response_payload.get("answer") or ""
    verdict = normalize_verdict(verdict or response_payload.get("verdict"), status)
    payload = {
        "query": q,
        "route": route,
        "model_used": model,
        "dataset_hit": bool(hit),
        "validation_score": score,
        "status": status,
        "verdict": verdict,
        "answer": answer,
        "error": error,
        "created_at": updated_at,
        **response_payload,
    }
    payload["verdict"] = normalize_verdict(payload.get("verdict"), payload.get("status"))
    c=connection()
    try:
        c.execute("UPDATE query_log SET route=?, model_used=?, dataset_hit=?, validation_score=?, status=?, error=?, verdict=?, answer=?, response_json=?, created_at=? WHERE id=?",
                  (route, model, int(bool(hit)), score, status, error, payload.get("verdict"), answer or "", json.dumps(payload), updated_at, int(log_id)))
        c.commit()
    finally: c.close()


def generate_query_report(limit=20):
    limit = max(1, min(int(limit), 1000))
    all_rows = list_logs(1000)
    visible_rows = all_rows[:limit]
    summary = {}
    queries = []

    for row in all_rows:
        payload = row.get("response_json") or {}
        if not isinstance(payload, dict):
            payload = {}
        verdict = normalize_verdict(row.get("verdict") or payload.get("verdict"), row.get("status") or payload.get("status"))
        if not payload.get("answer") and row.get("answer"):
            payload["answer"] = row.get("answer")
        if not payload.get("query"):
            payload["query"] = row.get("query")
        if not payload.get("route"):
            payload["route"] = row.get("route")
        if not payload.get("model_used"):
            payload["model_used"] = row.get("model_used")
        if payload.get("validation_score") is None:
            payload["validation_score"] = row.get("validation_score")
        if payload.get("created_at") is None:
            payload["created_at"] = row.get("created_at")
        payload["verdict"] = normalize_verdict(payload.get("verdict") or row.get("verdict"), payload.get("status") or row.get("status"))

        response_payload = {
            "answer": payload.get("answer") or row.get("answer") or "",
            "verdict": payload.get("verdict") or verdict,
            "status": payload.get("status") or row.get("status"),
            "validation_score": payload.get("validation_score", row.get("validation_score")),
            "validation_status": payload.get("validation_status") or payload.get("status") or row.get("status"),
            "explanation": payload.get("explanation") or payload.get("validation_reason") or "",
            "model_used": payload.get("model_used") or row.get("model_used"),
            "route": payload.get("route") or row.get("route"),
            "dataset_hit": bool(row.get("dataset_hit") or payload.get("dataset_hit")),
            "created_at": payload.get("created_at") or row.get("created_at"),
            "error": payload.get("error") or row.get("error"),
            "sources": payload.get("sources") or [],
            "evidence": payload.get("evidence") or [],
        }
        response_payload["verdict"] = normalize_verdict(response_payload.get("verdict"), response_payload.get("status"))
        summary[response_payload["verdict"]] = summary.get(response_payload["verdict"], 0) + 1

    for row in visible_rows:
        payload = row.get("response_json") or {}
        if not isinstance(payload, dict):
            payload = {}
        if not payload.get("answer") and row.get("answer"):
            payload["answer"] = row.get("answer")
        if not payload.get("query"):
            payload["query"] = row.get("query")
        if not payload.get("route"):
            payload["route"] = row.get("route")
        if not payload.get("model_used"):
            payload["model_used"] = row.get("model_used")
        if payload.get("validation_score") is None:
            payload["validation_score"] = row.get("validation_score")
        if payload.get("created_at") is None:
            payload["created_at"] = row.get("created_at")
        payload["verdict"] = normalize_verdict(payload.get("verdict") or row.get("verdict"), payload.get("status") or row.get("status"))

        response_payload = {
            "answer": payload.get("answer") or row.get("answer") or "",
            "verdict": payload.get("verdict") or normalize_verdict(row.get("verdict"), row.get("status")),
            "status": payload.get("status") or row.get("status"),
            "validation_score": payload.get("validation_score", row.get("validation_score")),
            "validation_status": payload.get("validation_status") or payload.get("status") or row.get("status"),
            "explanation": payload.get("explanation") or payload.get("validation_reason") or "",
            "model_used": payload.get("model_used") or row.get("model_used"),
            "route": payload.get("route") or row.get("route"),
            "dataset_hit": bool(row.get("dataset_hit") or payload.get("dataset_hit")),
            "created_at": payload.get("created_at") or row.get("created_at"),
            "error": payload.get("error") or row.get("error"),
            "sources": payload.get("sources") or [],
            "evidence": payload.get("evidence") or [],
        }
        response_payload["verdict"] = normalize_verdict(response_payload.get("verdict"), response_payload.get("status"))

        query_entry = {
            "id": row.get("id"),
            "query": payload.get("query") or row.get("query"),
            "route": payload.get("route") or row.get("route"),
            "model_used": payload.get("model_used") or row.get("model_used"),
            "dataset_hit": bool(row.get("dataset_hit") or payload.get("dataset_hit")),
            "validation_score": payload.get("validation_score", row.get("validation_score")),
            "status": payload.get("status") or row.get("status"),
            "verdict": response_payload["verdict"],
            "response": response_payload,
            "answer": response_payload["answer"],
            "created_at": response_payload["created_at"],
            "error": response_payload["error"],
            "time": response_payload["created_at"],
            "date": (response_payload["created_at"] or "").split("T")[0] if response_payload.get("created_at") else "",
        }
        queries.append(query_entry)

    return {"generated_at": now(), "total_queries": len(all_rows), "summary": dict(sorted(summary.items())), "queries": queries}

