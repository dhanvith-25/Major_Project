from app.config import settings


def cfg(route):
    s=settings()
    if route in ("cheap","medium"):
        return {"key":s.groq_key,"base":"https://api.groq.com/openai/v1",
                "model":s.cheap_model if route=="cheap" else s.medium_model}
    return {"key":s.openrouter_key,"base":"https://openrouter.ai/api/v1",
            "model":s.premium_model if route=="premium" else s.validator_model}
def call(route,system,prompt,max_tokens=1500):
    c=cfg(route)
    if not c["key"]: raise RuntimeError(f"{route} provider is not configured.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for LLM routes. Install requirements.txt.") from exc
    r=OpenAI(api_key=c["key"],base_url=c["base"]).chat.completions.create(
      model=c["model"],messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
      temperature=0.1,max_tokens=max_tokens)
    if not r.choices or not r.choices[0].message.content: raise RuntimeError(f"{route} returned no content.")
    return r.choices[0].message.content.strip(),c["model"]
def choose_route(q):
    q=q.lower(); hard=["medical","health","legal","scientific","study","evidence","misinformation","disinformation","latest","current","verify","true"]
    med=["compare","difference","analyze","explain","why","how"]
    n=len(q.split())+15*sum(x in q for x in hard)+7*sum(x in q for x in med)
    return "premium" if n>=40 else "medium" if n>=18 else "cheap"
