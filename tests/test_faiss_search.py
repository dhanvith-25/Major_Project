from fastapi.testclient import TestClient
from app.main import app
from app.db import upsert_claim, init_db
from app.dataset import refresh_index, get_top_k_matches

client = TestClient(app)


def test_faiss_semantic_retrieval_and_endpoints():
    init_db()

    cid1 = upsert_claim(
        claim="Vaccines significantly reduce the risk of severe COVID-19 hospitalization.",
        answer="Vaccinations stimulate antibody production and protect against severe COVID-19 complications.",
        verdict="SUPPORTED",
        score=96.0,
        sources=[{"title": "CDC Guidance", "url": "https://www.cdc.gov/covid"}],
        model="test-model"
    )

    cid2 = upsert_claim(
        claim="Drinking bleach kills internal viruses and cures diseases.",
        answer="Bleach is toxic and caustic. Ingesting it causes severe internal injury and death.",
        verdict="CONTRADICTED",
        score=99.0,
        sources=[{"title": "FDA Warning", "url": "https://www.fda.gov/bleach-warning"}],
        model="test-model"
    )

    ref_res = refresh_index()
    assert ref_res["status"] == "refreshed"
    assert ref_res["total_claims"] >= 2

    top_matches = get_top_k_matches("Does getting vaccinated stop bad COVID infection?", k=5)
    assert len(top_matches) > 0
    top_hit = top_matches[0]
    assert "covid" in top_hit["canonical_claim"].lower() or "vaccine" in top_hit["canonical_claim"].lower()
    assert top_hit["verdict"] == "SUPPORTED"
    assert top_hit["similarity_score"] > 0.6

    response = client.post("/dataset/search", json={"query": "Is bleach safe to drink for viral infection?", "k": 3})
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert results[0]["verdict"] == "CONTRADICTED"
    assert "bleach" in results[0]["canonical_claim"].lower()

    response_ref = client.post("/dataset/refresh-index")
    assert response_ref.status_code == 200
    assert response_ref.json()["status"] == "refreshed"
