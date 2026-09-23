import httpx
from app.config import settings


async def search(query):
    api_key = settings().tavily_api_key
    if not api_key:
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 6,
        "include_answer": False,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return []

    results = data.get("results", []) if isinstance(data, dict) else []
    return [
        {
            "title": str(x.get("title") or ""),
            "url": str(x.get("url") or ""),
            "content": str(x.get("content") or ""),
            "published_date": x.get("published_date"),
        }
        for x in results
        if isinstance(x, dict)
    ]
