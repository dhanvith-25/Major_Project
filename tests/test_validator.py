from app.services.validator import score, validate


def test_validator_health_gate_rejects_unrelated_input_without_llm():
    result = validate("Hello, how are you?", "", [])
    assert result["verdict"] == "INVALID_STATEMENT"
    assert score(result)[1] == "INVALID_STATEMENT"


def test_validator_returns_weighted_5_metric_score():
    example = {
        "verdict": "SUPPORTED",
        "answer": "Vaccines reduce severe COVID-19 hospitalization risk.",
        "explanation": "Evidence is strong and consistent.",
        "evidence": [
            {
                "url": "https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)",
                "relationship": "SUPPORTS",
                "relevance": 95,
                "source_type": "government",
                "published_date": "2024-02-01",
                "reason": "WHO guidance on COVID-19 vaccines."
            },
            {
                "url": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/index.html",
                "relationship": "SUPPORTS",
                "relevance": 90,
                "source_type": "government",
                "published_date": "2024-04-10",
                "reason": "CDC guidance on vaccine protection."
            }
        ]
    }
    final_score, status, reason, components = score(example)
    assert 0 <= final_score <= 100
    assert status in {"VERIFIED", "SUPPORTED", "LOW_CONFIDENCE"}
    assert set(components.keys()) >= {"relevance", "authority", "agreement", "recency", "independence"}
    assert final_score >= 70
