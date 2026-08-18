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
