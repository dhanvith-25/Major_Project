import re, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import settings
from app.db import connection

def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ",s.lower())).strip()
def find_match(query):
    c=connection()
    try: rows=c.execute("SELECT * FROM verified_claims ORDER BY updated_at DESC").fetchall()
    finally:c.close()
    if not rows:return None
    corpus=[norm(r["canonical_claim"]) for r in rows]
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),sublinear_tf=True)
    m=v.fit_transform(corpus+[norm(query)])
    sims=cosine_similarity(m[-1],m[:-1]).ravel(); i=int(sims.argmax()); sim=float(sims[i])
    if sim<settings().dataset_match_threshold:return None
    r=rows[i]
    return {"id":int(r["id"]),"answer":r["answer"],"verdict":r["verdict"],
            "similarity":round(sim*100,2),"validation_score":float(r["validation_score"]),
            "sources":json.loads(r["sources_json"] or "[]")}
