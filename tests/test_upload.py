from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_csv_upload_returns_predictions_and_accuracy():
    csv_data = b"claim,label\nAntibiotics cure the common cold,CONTRADICTED\nRegular physical activity provides health benefits,SUPPORTED\n"
    response = client.post("/health/upload", files={"file": ("claims.csv", BytesIO(csv_data), "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["rows"] == 2
    assert body["summary"]["accuracy"] == 100.0


def test_csv_upload_requires_claim_column():
    response = client.post("/health/upload", files={"file": ("claims.csv", BytesIO(b"name\nhello\n"), "text/csv")})
    assert response.status_code == 400
