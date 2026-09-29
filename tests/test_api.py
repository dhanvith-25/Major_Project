from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_frontend_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Health Claim Checker" in response.text


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_prediction():
    response = client.post("/health/predict", json={"claim": "Antibiotics cure the common cold"})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "CONTRADICTED"
    assert 0 <= body["confidence"] <= 100


def test_health_prediction_rejects_short_claim():
    response = client.post("/health/predict", json={"claim": "x"})
    assert response.status_code == 422


def test_health_prediction_rejects_unrelated_statement():
    response = client.post("/health/predict", json={"claim": "Hello, how are you?"})
    assert response.status_code == 200
    assert response.json()["verdict"] == "INVALID_STATEMENT"


def test_training_reads_csv_dataset(tmp_path):
    from app.services.health_model import train_health_model

    csv_path = tmp_path / "custom_claims.csv"
    csv_path.write_text(
        "claim,label\n"
        "Vaccination reduces severe disease risk,SUPPORTED\n"
        "Antibiotics cure all infections,CONTRADICTED\n"
        "This treatment may help some people,UNCERTAIN\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "custom_model.joblib"

    result = train_health_model(model_path=model_path, dataset_path=csv_path)

    assert result["status"] == "trained"
    assert result["examples"] == 3
    assert model_path.exists()


def test_query_report_summary_includes_history():
    from app.db import log_query

    log_query(
        "Vaccines prevent severe disease",
        "verified_dataset",
        "demo-model",
        True,
        0.91,
        "VERIFIED",
        verdict="SUPPORTED",
        answer="Vaccines reduce severe illness in many populations.",
        response_payload={
            "query": "Vaccines prevent severe disease",
            "answer": "Vaccines reduce severe illness in many populations.",
            "verdict": "SUPPORTED",
            "status": "VERIFIED",
            "validation_score": 0.91,
            "route": "verified_dataset",
            "model_used": "demo-model",
            "created_at": "2026-09-21T10:00:00+00:00",
        },
    )
    log_query(
        "Antibiotics treat the common cold",
        "llm_fallback",
        "demo-model",
        False,
        0.0,
        "INSUFFICIENT_EVIDENCE",
        verdict="UNCERTAIN",
        answer="Antibiotics are not recommended for a viral cold.",
        response_payload={
            "query": "Antibiotics treat the common cold",
            "answer": "Antibiotics are not recommended for a viral cold.",
            "verdict": "UNCERTAIN",
            "status": "INSUFFICIENT_EVIDENCE",
            "validation_score": 0.0,
            "route": "llm_fallback",
            "model_used": "demo-model",
            "created_at": "2026-09-21T11:00:00+00:00",
        },
    )

    response = client.get("/report")

    assert response.status_code == 200
    body = response.json()
    assert body["total_queries"] >= 2
    assert body["summary"]["SUPPORTED"] >= 1
    assert body["summary"]["UNCERTAIN"] >= 1
    assert body["queries"][0]["query"]
    assert body["queries"][0]["response"]["answer"]
    assert body["queries"][0]["response"]["validation_score"] is not None


def test_symptom_checker_supports_manual_symptoms(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import app.db as db
    from app.services import symptom_checker as checker_module

    monkeypatch.setattr(db, "settings", lambda: SimpleNamespace(db_path=str(tmp_path / "symptoms.db")))
    db.init_db()
    monkeypatch.setattr(
        checker_module,
        "_predict_disease",
        lambda text: {
            "cause": "Influenza",
            "risk_level": "medium",
            "prevention": "Rest and drink fluids.",
            "explanation": "A viral respiratory illness.",
            "confidence": 0.8,
        },
    )

    response = client.post("/symptom-check", data={"symptoms": "fever, cough, sore throat"})

    assert response.status_code == 200
    body = response.json()
    assert body["cause"] == "Influenza"
    assert body["prevention"] == "Rest and drink fluids."
    assert body["risk_level"] == "medium"
    assert body["disclaimer"] == "This is not a medical diagnosis. Please consult a doctor. Results may be inaccurate."
    assert body["manual_symptoms"] == ["fever", "cough", "sore throat"]
    assert body["summary"]
    assert "suggested_causes" in body
    assert isinstance(body["suggested_causes"], list)
    assert db.get_symptom_mapping("cough, fever, sore throat")["mapped_disease"] == "Influenza"


def test_symptom_database_hit_skips_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import app.db as db
    from app.services import symptom_checker as checker_module

    monkeypatch.setattr(db, "settings", lambda: SimpleNamespace(db_path=str(tmp_path / "symptoms.db")))
    db.init_db()
    db.save_symptom_mapping(
        "cough, fever",
        "Known respiratory illness",
        "high",
        "Seek clinical advice.",
        "Stored explanation.",
    )
    monkeypatch.setattr(
        checker_module,
        "_predict_disease",
        lambda text: pytest.fail("model should not run for a database hit"),
    )

    result = checker_module.symptom_checker(symptoms="fever, cough")

    assert result["cause"] == "Known respiratory illness"
    assert result["risk_level"] == "high"
    assert result["prevention"] == "Seek clinical advice."
    assert result["disclaimer"]


def test_symptom_checker_accepts_uploaded_image(tmp_path, monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace

    from PIL import Image

    import app.db as db
    from app.services import symptom_checker as checker_module

    monkeypatch.setattr(db, "settings", lambda: SimpleNamespace(db_path=str(tmp_path / "symptoms.db")))
    db.init_db()
    monkeypatch.setattr(
        checker_module,
        "_predict_disease",
        lambda text: {
            "cause": "Influenza",
            "risk_level": "medium",
            "prevention": "Rest and drink fluids.",
            "explanation": "A viral respiratory illness.",
            "confidence": 0.8,
        },
    )

    image_buffer = BytesIO()
    Image.new("RGB", (80, 60), "white").save(image_buffer, format="PNG")
    image_buffer.seek(0)

    response = client.post(
        "/symptom-check",
        files={"image": ("symptoms.png", image_buffer.getvalue(), "image/png")},
        data={"symptoms": "fever, cough"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["manual_symptoms"] == ["fever", "cough"]
    assert "ocr_status" in body
    assert isinstance(body["suggested_causes"], list)
