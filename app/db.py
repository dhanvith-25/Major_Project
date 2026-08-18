import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from app.config import settings

def now(): return datetime.now(timezone.utc).isoformat()
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
          status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL);
        """); c.commit()
    finally: c.close()

def log_query(q,route,model,hit,score,status,error=None):
    c=connection()
    try:
        c.execute("INSERT INTO query_log(query,route,model_used,dataset_hit,validation_score,status,error,created_at) VALUES(?,?,?,?,?,?,?,?)",
                  (q,route,model,int(hit),score,status,error,now())); c.commit()
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
        c.commit(); return cid
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
    try:return [dict(r) for r in c.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT ?",(limit,)).fetchall()]
    finally:c.close()
