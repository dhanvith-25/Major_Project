import json,re
from datetime import datetime,timezone
from app.services.llm import call
from app.config import settings
from app.services.health_model import predict_health_claim
AUTH={"who":100,"government":95,"peer_reviewed":95,"university":90,"medical_institution":90,"established_fact_checker":85,"major_news":75,"news":65,"blog":30,"social_media":10,"unknown":10}
def authority(x):return AUTH.get(str(x).lower(),10)
def recency(d):
    if not d:return 50
    try:
        age=max(0,(datetime.now(timezone.utc)-datetime.strptime(str(d)[:10],"%Y-%m-%d").replace(tzinfo=timezone.utc)).days)
        return 100 if age<=30 else 95 if age<=180 else 90 if age<=365 else 80 if age<=730 else 65 if age<=1825 else 45
    except:return 40
def parse_json(t):
    try:return json.loads(t)
    except:
        m=re.search(r"\{.*\}",t,re.S)
        if not m:raise RuntimeError("Validator returned invalid JSON.")
        return json.loads(m.group(0))
def validate(q,answer,sources):
    health_prediction=predict_health_claim(q)
    if health_prediction["verdict"]=="INVALID_STATEMENT":
        return {"canonical_claim":q,"verdict":"INVALID_STATEMENT","answer":"The input is not a health-related statement.","evidence":[],"_health_prediction":health_prediction}
    ev="\n".join(f"SOURCE {i}\nTITLE:{s['title']}\nURL:{s['url']}\nDATE:{s.get('published_date')}\nCONTENT:{s['content'][:6000]}" for i,s in enumerate(sources,1))
    p=f"""Query:{q}\nGenerated answer:{answer}\nML health-model prediction:{health_prediction["verdict"]} ({health_prediction["confidence"]}% confidence)\nEvidence:{ev}
Return ONLY JSON with canonical_claim, verdict (SUPPORTED|CONTRADICTED|UNCERTAIN), answer, and evidence array.
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
    elif final>=90: status,reason="SUPPORTED","Strong evidence, below automatic ingestion threshold."
    else: status,reason="LOW_CONFIDENCE","Evidence is insufficient."
    return final,status,reason,{"relevance":round(rel,2),"authority":round(auth,2),"agreement":round(agreement,2),"recency":round(recent,2),"independence":round(independence,2)}
