import httpx
from app.config import settings
async def search(query):
    if not settings().tavily_api_key:return []
    payload={"api_key":settings().tavily_api_key,"query":query,"search_depth":"advanced","max_results":6,"include_answer":False}
    async with httpx.AsyncClient(timeout=30) as client:
        r=await client.post("https://api.tavily.com/search",json=payload); r.raise_for_status(); data=r.json()
    results=data.get("results",[]) if isinstance(data,dict) else []
    return [{"title":str(x.get("title") or ""),"url":str(x.get("url") or ""),"content":str(x.get("content") or ""),"published_date":x.get("published_date")} for x in results if isinstance(x,dict)]
