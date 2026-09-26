import json, re, sqlite3
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


def normalize_review_status(status=None):
    value = str(status or "pending").strip().lower()
    if value not in {"pending", "approved", "rejected"}:
        return "pending"
    return value


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
          model_used TEXT, review_status TEXT NOT NULL DEFAULT 'pending',
          review_note TEXT DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_claims_updated ON verified_claims(updated_at);
        CREATE TABLE IF NOT EXISTS claim_reviews(
          id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending', note TEXT DEFAULT '',
          reviewed_by TEXT DEFAULT 'admin', created_at TEXT NOT NULL, reviewed_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS training_examples(
          id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id INTEGER NOT NULL UNIQUE,
          prompt TEXT NOT NULL, completion TEXT NOT NULL, validation_score REAL NOT NULL,
          review_status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
          trained INTEGER NOT NULL DEFAULT 0);
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
        CREATE TABLE IF NOT EXISTS personal_reports(
          id INTEGER PRIMARY KEY AUTOINCREMENT, original_filename TEXT NOT NULL,
          stored_path TEXT NOT NULL, mime_type TEXT, file_size INTEGER NOT NULL,
          extracted_text TEXT NOT NULL, extraction_method TEXT NOT NULL,
          test_type TEXT NOT NULL, test_type_confidence REAL NOT NULL DEFAULT 0,
          report_test_name TEXT, patient_name TEXT, doctor_name TEXT, reported_date TEXT,
          created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_personal_reports_type_date
          ON personal_reports(test_type, reported_date DESC);
        CREATE TABLE IF NOT EXISTS personal_measurements(
          id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER NOT NULL,
          test_name TEXT NOT NULL, value_raw TEXT NOT NULL, value_num REAL,
          unit TEXT, reference_low REAL, reference_high REAL, reference_range TEXT,
          status TEXT NOT NULL DEFAULT 'unknown', created_at TEXT NOT NULL,
          FOREIGN KEY(report_id) REFERENCES personal_reports(id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS idx_personal_measurements_report
          ON personal_measurements(report_id);
        CREATE INDEX IF NOT EXISTS idx_personal_measurements_name
          ON personal_measurements(test_name);
        """); c.commit()
        existing_cols={row[1] for row in c.execute("PRAGMA table_info(query_log)").fetchall()}
        for col_name, ddl in {
            "verdict":"ALTER TABLE query_log ADD COLUMN verdict TEXT",
            "answer":"ALTER TABLE query_log ADD COLUMN answer TEXT",
            "response_json":"ALTER TABLE query_log ADD COLUMN response_json TEXT NOT NULL DEFAULT '{}'",
        }.items():
            if col_name not in existing_cols:
                c.execute(ddl)
        claim_cols={row[1] for row in c.execute("PRAGMA table_info(verified_claims)").fetchall()}
        for col_name, ddl in {
            "review_status":"ALTER TABLE verified_claims ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'",
            "review_note":"ALTER TABLE verified_claims ADD COLUMN review_note TEXT DEFAULT ''",
        }.items():
            if col_name not in claim_cols:
                c.execute(ddl)
        example_cols={row[1] for row in c.execute("PRAGMA table_info(training_examples)").fetchall()}
        for col_name, ddl in {
            "review_status":"ALTER TABLE training_examples ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'",
        }.items():
            if col_name not in example_cols:
                c.execute(ddl)
        personal_report_cols={row[1] for row in c.execute("PRAGMA table_info(personal_reports)").fetchall()}
        for col_name, ddl in {
            "report_test_name":"ALTER TABLE personal_reports ADD COLUMN report_test_name TEXT",
        }.items():
            if col_name not in personal_report_cols:
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
        r=c.execute("SELECT id, review_status FROM verified_claims WHERE lower(canonical_claim)=lower(?) LIMIT 1",(claim,)).fetchone()
        review_status = "pending"
        if r and r["review_status"]:
            review_status = normalize_review_status(r["review_status"])
        if r:
            cid=int(r["id"])
            c.execute("UPDATE verified_claims SET answer=?,verdict=?,validation_score=?,source_count=?,sources_json=?,model_used=?,review_status=?,updated_at=? WHERE id=?",
                      (answer,verdict,score,len(sources),json.dumps(sources),model,review_status,now(),cid))
        else:
            cur=c.execute("INSERT INTO verified_claims(canonical_claim,answer,verdict,validation_score,source_count,sources_json,model_used,review_status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                          (claim,answer,verdict,score,len(sources),json.dumps(sources),model,review_status,now(),now()))
            cid=int(cur.lastrowid)
        c.commit()
        if float(score) >= 95.0:
            add_training_example(cid, claim, answer, score)
            c.execute("UPDATE training_examples SET review_status='pending', trained=0 WHERE claim_id=?", (cid,))
            c.execute("UPDATE verified_claims SET review_status='pending' WHERE id=?", (cid,))
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
        c.execute("INSERT OR IGNORE INTO training_examples(claim_id,prompt,completion,validation_score,review_status,created_at) VALUES(?,?,?,?,?,?)",
                  (cid,prompt,completion,score,"pending",now())); c.commit()
    finally: c.close()


def pending_training():
    c=connection()
    try: return c.execute("SELECT * FROM training_examples WHERE trained=0 AND review_status='approved' ORDER BY id").fetchall()
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
        approved=c.execute("SELECT COUNT(*) n FROM training_examples WHERE review_status='approved'").fetchone()["n"]
        pending=c.execute("SELECT COUNT(*) n FROM training_examples WHERE review_status='pending'").fetchone()["n"]
        rejected=c.execute("SELECT COUNT(*) n FROM training_examples WHERE review_status='rejected'").fetchone()["n"]
        return {"total_examples":int(a),"pending_examples":int(b),"approved_examples":int(approved),"pending_review":int(pending),"rejected_examples":int(rejected)}
    finally:c.close()


def review_dashboard():
    c=connection()
    try:
        rows=c.execute("SELECT * FROM verified_claims ORDER BY updated_at DESC").fetchall()
        items=[]
        for row in rows:
            items.append({
                "id": row["id"],
                "canonical_claim": row["canonical_claim"],
                "answer": row["answer"],
                "verdict": row["verdict"],
                "validation_score": row["validation_score"],
                "review_status": normalize_review_status(row["review_status"]),
                "review_note": row["review_note"] or "",
                "updated_at": row["updated_at"],
            })
        summary={"total": len(items), "pending": sum(1 for x in items if x["review_status"]=="pending"), "approved": sum(1 for x in items if x["review_status"]=="approved"), "rejected": sum(1 for x in items if x["review_status"]=="rejected")}
        return {"summary": summary, "claims": items}
    finally:c.close()


def set_claim_review_status(claim_id, status, note=""):
    normalized = normalize_review_status(status)
    c=connection()
    try:
        row=c.execute("SELECT * FROM verified_claims WHERE id=?", (int(claim_id),)).fetchone()
        if not row:
            return {"claim_id": int(claim_id), "status": "not_found", "review_status": normalized}
        c.execute("UPDATE verified_claims SET review_status=?, review_note=?, updated_at=? WHERE id=?",
                  (normalized, note, now(), int(claim_id)))
        c.execute("INSERT INTO claim_reviews(claim_id,status,note,reviewed_by,created_at,reviewed_at) VALUES(?,?,?,?,?,?)",
                  (int(claim_id), normalized, note, "admin", now(), now()))
        c.execute("UPDATE training_examples SET review_status=?, trained=0 WHERE claim_id=?", (normalized, int(claim_id)))
        c.commit()
        return {"claim_id": int(claim_id), "status": "updated", "review_status": normalized}
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


def _personal_report_rows(c, where="", params=()):
    rows = [dict(row) for row in c.execute(
        "SELECT * FROM personal_reports " + where + " ORDER BY COALESCE(reported_date, created_at) DESC, id DESC",
        params,
    ).fetchall()]
    if not rows:
        return []
    report_ids = [row["id"] for row in rows]
    placeholders = ",".join("?" for _ in report_ids)
    measurements = c.execute(
        "SELECT * FROM personal_measurements WHERE report_id IN (" + placeholders + ") ORDER BY id",
        report_ids,
    ).fetchall()
    by_report = {report_id: [] for report_id in report_ids}
    for measurement in measurements:
        by_report[measurement["report_id"]].append(dict(measurement))
    for row in rows:
        row["measurements"] = by_report[row["id"]]
        # Original paths and the complete OCR/PDF text are persisted locally for
        # search, but they are not needed by the browser response.
        row.pop("stored_path", None)
        row.pop("extracted_text", None)
    return rows


def create_personal_report(report, measurements):
    """Persist a local report and all extracted measurements in one transaction."""
    c = connection()
    try:
        created_at = now()
        cur = c.execute(
            """INSERT INTO personal_reports(
                original_filename, stored_path, mime_type, file_size, extracted_text,
                extraction_method, test_type, test_type_confidence, report_test_name, patient_name,
                doctor_name, reported_date, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                report["original_filename"], report["stored_path"], report.get("mime_type"),
                int(report["file_size"]), report["extracted_text"], report["extraction_method"],
                report["test_type"], float(report.get("test_type_confidence", 0)), report.get("report_test_name"),
                report.get("patient_name"), report.get("doctor_name"), report.get("reported_date"),
                created_at,
            ),
        )
        report_id = int(cur.lastrowid)
        c.executemany(
            """INSERT INTO personal_measurements(
                report_id, test_name, value_raw, value_num, unit, reference_low,
                reference_high, reference_range, status, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    report_id, item["test_name"], item["value_raw"], item.get("value_num"),
                    item.get("unit"), item.get("reference_low"), item.get("reference_high"),
                    item.get("reference_range"), item.get("status", "unknown"), created_at,
                )
                for item in measurements
            ],
        )
        c.commit()
        return _personal_report_rows(c, "WHERE id=?", (report_id,))[0]
    finally:
        c.close()


def list_personal_reports(query="", test_type="", limit=100):
    """List reports, optionally searching report metadata, OCR/PDF text, and test names."""
    limit = max(1, min(int(limit), 500))
    query = str(query or "").strip().lower()
    test_type = str(test_type or "").strip()
    filters, params = [], []
    if query:
        pattern = f"%{query}%"
        filters.append(
            """(
                lower(original_filename) LIKE ? OR lower(test_type) LIKE ? OR
                lower(COALESCE(patient_name,'')) LIKE ? OR lower(COALESCE(doctor_name,'')) LIKE ? OR
                lower(extracted_text) LIKE ? OR id IN (
                    SELECT report_id FROM personal_measurements WHERE lower(test_name) LIKE ?
                )
            )"""
        )
        params.extend([pattern] * 6)
    if test_type:
        filters.append("test_type = ?")
        params.append(test_type)
    where = "WHERE " + " AND ".join(filters) if filters else ""
    c = connection()
    try:
        rows = _personal_report_rows(c, where, tuple(params))
        return rows[:limit]
    finally:
        c.close()


def get_personal_report(report_id):
    c = connection()
    try:
        rows = _personal_report_rows(c, "WHERE id=?", (int(report_id),))
        return rows[0] if rows else None
    finally:
        c.close()


def personal_reports_summary(query=""):
    """Build a date-aware trend view across all matching stored reports."""
    reports = list_personal_reports(query=query, limit=500)
    groups = {}
    for report in reports:
        effective_date = report.get("reported_date") or (report.get("created_at") or "")[:10]
        for measurement in report.get("measurements", []):
            key = (measurement["test_name"].strip().lower(), (measurement.get("unit") or "").lower())
            group = groups.setdefault(
                key,
                {
                    "test_name": measurement["test_name"],
                    "unit": measurement.get("unit"),
                    "records": [],
                    "test_types": set(),
                },
            )
            group["test_types"].add(report["test_type"])
            group["records"].append(
                {
                    "report_id": report["id"], "reported_date": effective_date,
                    "value_raw": measurement["value_raw"], "value_num": measurement.get("value_num"),
                    "status": measurement.get("status", "unknown"),
                    "reference_range": measurement.get("reference_range"),
                    "filename": report["original_filename"],
                }
            )

    metrics = []
    for group in groups.values():
        records = sorted(group["records"], key=lambda row: row["reported_date"] or "")
        numeric = [row for row in records if row.get("value_num") is not None]
        latest = records[-1]
        first = numeric[0] if numeric else None
        latest_numeric = numeric[-1] if numeric else None
        change = None
        direction = "insufficient data"
        if first and latest_numeric and len(numeric) >= 2:
            change = round(latest_numeric["value_num"] - first["value_num"], 4)
            direction = "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
        elif latest_numeric:
            direction = "single result"
        metrics.append(
            {
                "test_name": group["test_name"], "unit": group["unit"],
                "test_types": sorted(group["test_types"]), "latest": latest,
                "highest": max(numeric, key=lambda row: row["value_num"]) if numeric else None,
                "lowest": min(numeric, key=lambda row: row["value_num"]) if numeric else None,
                "change": change, "direction": direction, "records": records,
                "normal_count": sum(row["status"] == "normal" for row in records),
                "high_count": sum(row["status"] == "high" for row in records),
                "low_count": sum(row["status"] == "low" for row in records),
            }
        )
    metrics.sort(key=lambda metric: metric["test_name"].lower())
    return {
        "total_reports": len(reports),
        "total_measurements": sum(len(report["measurements"]) for report in reports),
        "metrics": metrics,
    }


def answer_personal_records_question(question):
    """Answer change questions deterministically from the user's saved measurements."""
    summary = personal_reports_summary()
    tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(question or "").lower())
        if len(token) > 2 and token not in {
            "what", "when", "where", "have", "with", "from", "that", "this", "your", "about",
            "show", "tell", "give", "does", "changed", "change", "latest", "report", "reports",
            "results", "result", "personal", "record", "records", "been", "much", "were", "which",
        }
    }
    candidates = []
    for metric in summary["metrics"]:
        searchable = " ".join([metric["test_name"], *metric["test_types"]]).lower()
        if not tokens or any(token in searchable for token in tokens):
            candidates.append(metric)
    candidates = candidates[:6]
    if not candidates:
        return {
            "answer": "No matching measurement was found in your saved personal reports.",
            "metrics": [], "total_reports": summary["total_reports"],
        }

    statements = []
    for metric in candidates:
        latest = metric["latest"]
        unit = f" {metric['unit']}" if metric.get("unit") else ""
        if metric["change"] is not None:
            first = metric["records"][0]
            delta = f"{abs(metric['change']):g}{unit}"
            statements.append(
                f"{metric['test_name']} {metric['direction']} by {delta}, from "
                f"{first['value_raw']}{unit} on {first['reported_date']} to "
                f"{latest['value_raw']}{unit} on {latest['reported_date']}"
            )
        else:
            statements.append(
                f"{metric['test_name']} has one recorded result: {latest['value_raw']}{unit} "
                f"on {latest['reported_date']}"
            )
        if latest.get("status") in {"high", "low"}:
            statements[-1] += f" (marked {latest['status']} against the report range)"
        statements[-1] += "."
    return {
        "answer": " ".join(statements) + " This summarizes saved report values and is not medical advice.",
        "metrics": candidates, "total_reports": summary["total_reports"],
    }


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
