import json
import os
import re
import socket
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from app.config import settings
from app.services.health_model import predict_health_claim
from app.services.llm import call

AUTH = {
    "who": 100,
    "government": 95,
    "peer_reviewed": 95,
    "university": 90,
    "medical_institution": 90,
    "established_fact_checker": 85,
    "major_news": 75,
    "news": 65,
    "blog": 30,
    "social_media": 10,
    "unknown": 10,
}


def authority(x: str | None) -> int:
    source = str(x or "unknown").lower().replace("-", "_").replace(" ", "_")
    if "who" in source or "world_health" in source:
        return AUTH["who"]
    if "government" in source or "gov" in source or "public_health" in source:
        return AUTH["government"]
    if "peer_review" in source or "systematic_review" in source:
        return AUTH["peer_reviewed"]
    if "university" in source or "academic" in source:
        return AUTH["university"]
    if "medical" in source or "health_agency" in source or "health_website" in source:
        return AUTH["medical_institution"]
    if "fact_check" in source or "factchecker" in source:
        return AUTH["established_fact_checker"]
    if "news" in source:
        return AUTH["news"]
    if "blog" in source:
        return AUTH["blog"]
    if "social" in source:
        return AUTH["social_media"]
    return AUTH.get(source, AUTH["unknown"])


def recency(d: str | None) -> float:
    if not d:
        return 50.0
    try:
        date_value = datetime.strptime(str(d)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_days = max(0, (datetime.now(timezone.utc) - date_value).days)
        if age_days <= 30:
            return 100.0
        if age_days <= 180:
            return 95.0
        if age_days <= 365:
            return 90.0
        if age_days <= 730:
            return 80.0
        if age_days <= 1825:
            return 65.0
        return 45.0
    except Exception:
        return 40.0


def parse_json(t: str):
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise RuntimeError("Validator returned invalid JSON.")
        return json.loads(m.group(0))


def _domain_reputation(url: str | None) -> dict[str, float | str | bool]:
    if not url:
        return {"domain": "", "whois_age_years": 0.0, "ssl_trust": 0.0, "score": 0.0, "is_trusted": False}

    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    if not host:
        return {"domain": "", "whois_age_years": 0.0, "ssl_trust": 0.0, "score": 0.0, "is_trusted": False}

    domain = host.split(":")[0]
    ssl_trust = 0.0
    whois_age_years = 0.0

    try:
        import whois  # type: ignore

        info = whois.whois(domain)
        created = info.get("creation_date") or info.get("created")
        if created:
            if isinstance(created, list):
                created = created[0]
            if isinstance(created, str):
                try:
                    created_date = datetime.strptime(created[:10], "%Y-%m-%d")
                    whois_age_years = max(0.0, (datetime.now(timezone.utc) - created_date.replace(tzinfo=timezone.utc)).days / 365.25)
                except Exception:
                    whois_age_years = 0.0
    except Exception:
        whois_age_years = 0.0

    if domain.endswith((".gov", ".edu", ".org")) and whois_age_years == 0.0:
        whois_age_years = 8.0
    elif domain.endswith(".com") and whois_age_years == 0.0:
        whois_age_years = 3.0

    base_score = 0.0
    if whois_age_years >= 10:
        base_score += 45
    elif whois_age_years >= 5:
        base_score += 35
    elif whois_age_years >= 2:
        base_score += 20
    elif whois_age_years >= 1:
        base_score += 10
    else:
        base_score += 5

    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                issuer = cert.get("issuer") or ()
                issuer_str = " ".join(
                    str(v[0][1]) for v in issuer if isinstance(v, tuple) and len(v) > 0
                )
                if any(name in issuer_str for name in ("Let's Encrypt", "GlobalSign", "DigiCert", "Amazon", "Google Trust")):
                    ssl_trust = 90.0
                else:
                    ssl_trust = 70.0
    except Exception:
        ssl_trust = 55.0 if domain.endswith((".gov", ".edu", ".org")) else 35.0

    base_score += ssl_trust * 0.5
    score = min(100.0, max(0.0, base_score))
    return {
        "domain": domain,
        "whois_age_years": round(whois_age_years, 2),
        "ssl_trust": round(ssl_trust, 2),
        "score": round(score, 2),
        "is_trusted": score >= 60.0,
    }


def _fact_check_tool_score(claim: str) -> float:
    api_key = getattr(settings(), "google_factcheck_api_key", "") or os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
    if not api_key or not claim:
        return 0.0

    try:
        response = httpx.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": claim, "key": api_key},
            timeout=8.0,
        )
        if response.status_code != 200:
            return 0.0
        payload = response.json()
        claims = payload.get("claims") or []
        if not claims:
            return 0.0

        total = 0.0
        for item in claims[:3]:
            reviews = item.get("claimReview") or []
            if not reviews:
                continue
            for review in reviews:
                rating = str(review.get("textualRating") or "").lower()
                if any(term in rating for term in ("true", "correct", "supported")):
                    total += 100.0
                elif any(term in rating for term in ("mostly true", "mixed")):
                    total += 60.0
                elif any(term in rating for term in ("false", "misleading", "incorrect")):
                    total += 20.0
                else:
                    total += 50.0
        return round(min(100.0, total / max(1, len(claims))), 2)
    except Exception:
        return 0.0


def _score_authority_for_evidence(evidence: list[dict], claim_text: str = "") -> float:
    if not evidence:
        return 0.0

    fact_check_boost = _fact_check_tool_score(claim_text)
    source_scores: list[float] = []
    for item in evidence:
        source_type = str(item.get("source_type") or item.get("type") or "unknown").lower()
        url = str(item.get("url") or "")
        base = authority(source_type)
        if url:
            domain_info = _domain_reputation(url)
            if domain_info.get("is_trusted"):
                base = max(base, int(domain_info.get("score", 0)))
            else:
                base = min(100, int(base * 0.7 + float(domain_info.get("score", 0)) * 0.3))
        source_scores.append(float(base))

    if fact_check_boost:
        source_scores = [max(score, fact_check_boost * 0.6) for score in source_scores]

    return round(sum(source_scores) / len(source_scores), 2)
def validate(q,answer,sources):
    health_prediction=predict_health_claim(q)
    if health_prediction["verdict"]=="INVALID_STATEMENT":
        return {"canonical_claim":q,"verdict":"INVALID_STATEMENT","answer":"The input is not a health-related statement.","evidence":[],"_health_prediction":health_prediction}
    ev="\n".join(f"SOURCE {i}\nTITLE:{s['title']}\nURL:{s['url']}\nDATE:{s.get('published_date')}\nCONTENT:{s['content'][:6000]}" for i,s in enumerate(sources,1))
    p=f"""Query:{q}\nGenerated answer:{answer}\nML health-model prediction:{health_prediction["verdict"]} ({health_prediction["confidence"]}% confidence)\nEvidence:{ev}
Return ONLY JSON with canonical_claim, verdict (SUPPORTED|CONTRADICTED|UNCERTAIN), answer, explanation, and evidence array.
The explanation must be 2-4 sentences in plain language. Explain what the evidence says,
how it relates to the user's claim, and why that supports, contradicts, or cannot confirm the verdict.
Each evidence item must contain url, relationship (SUPPORTS|CONTRADICTS|IRRELEVANT), relevance 0-100, source_type, published_date, reason.
Never invent evidence and never report model confidence. Treat the ML prediction as a signal, not as proof; use only the supplied evidence for the final verdict."""
    raw,model=call("validator","You are Satarka's independent evidence validator. Use only supplied evidence.",p,1800)
    r=parse_json(raw); r["_model"]=model; r["_health_prediction"]=health_prediction; return r
def score(r):
    if r.get("verdict")=="INVALID_STATEMENT":
        return 0,"INVALID_STATEMENT","The input is not a health-related statement.",{}
    ev=[e for e in r.get("evidence",[]) if e.get("relationship") in ("SUPPORTS","CONTRADICTS")]
    if not ev:return 0,"INSUFFICIENT_EVIDENCE","No directly relevant evidence.",{}
    sup=[e for e in ev if e.get("relationship")=="SUPPORTS"]; con=[e for e in ev if e.get("relationship")=="CONTRADICTS"]
    rel=sum(max(0,min(100,float(e.get("relevance",0)))) for e in ev)/len(ev)
    auth=sum(authority(e.get("source_type","unknown")) for e in ev)/len(ev)
    recent=sum(recency(e.get("published_date")) for e in ev)/len(ev)
    agreement=len(sup)/(len(sup)+len(con))*100
    domains={re.search(r"https?://([^/]+)",str(e.get("url",""))).group(1).lower() for e in ev if re.search(r"https?://([^/]+)",str(e.get("url","")))}
    independence=min(100,len(domains)/len(ev)*100)
    final=round(max(0,min(100,rel*.30+auth*.25+agreement*.20+recent*.15+independence*.10)),2)
    if sup and con: status,reason="CONFLICTING","Supporting and contradicting evidence was found."
    elif final>=settings().auto_ingest_threshold: status,reason="VERIFIED","Passed automatic verification threshold."
    elif final>=75: status,reason="SUPPORTED","Evidence supports the claim, but the score is below the automatic ingestion threshold."
    else: status,reason="LOW_CONFIDENCE","Evidence is insufficient."
    return final,status,reason,{"relevance":round(rel,2),"authority":round(auth,2),"agreement":round(agreement,2),"recency":round(recent,2),"independence":round(independence,2)}
