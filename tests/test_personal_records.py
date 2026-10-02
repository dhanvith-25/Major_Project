from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.db as db
import app.main as main
from app.services.personal_records import (
    classify_test_type,
    extract_report_details,
    validate_report_upload,
)


def test_personal_report_extraction_parses_patient_and_measurements():
    text = """Patient Name: Asha Rao
Referring Doctor: Dr Mehta
Report Date: 20/09/2026
Fasting Glucose 110 mg/dL 70 - 99 H
Vitamin D: 25 ng/mL 30 - 100 L
"""

    test_type, confidence = classify_test_type(text)
    details = extract_report_details(text, test_type)

    assert test_type == "Blood Sugar"
    assert confidence > 0
    assert details["patient_name"] == "Asha Rao"
    assert details["doctor_name"] == "Dr Mehta"
    assert details["reported_date"] == "2026-09-20"
    assert len(details["measurements"]) == 2
    assert details["measurements"][0]["status"] == "high"
    assert details["measurements"][1]["status"] == "low"


def test_personal_report_validation_rejects_an_unsupported_file():
    try:
        validate_report_upload("report.txt", "text/plain", b"not a report")
    except ValueError as exc:
        assert "PDF" in str(exc)
    else:
        raise AssertionError("Unsupported uploads must be rejected")


def test_deleting_personal_report_removes_file_and_extracted_details(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    report_path = report_dir / "report.pdf"
    report_path.write_bytes(b"local report")
    test_settings = SimpleNamespace(
        db_path=str(tmp_path / "personal.db"),
        personal_reports_dir=str(report_dir),
    )
    monkeypatch.setattr(db, "settings", lambda: test_settings)
    monkeypatch.setattr(main, "settings", lambda: test_settings)
    db.init_db()
    report = db.create_personal_report(
        {
            "original_filename": "report.pdf",
            "stored_path": str(report_path),
            "file_size": report_path.stat().st_size,
            "extracted_text": "Glucose 100 mg/dL",
            "extraction_method": "pdf_text",
            "test_type": "Blood Sugar",
        },
        [{"test_name": "Glucose", "value_raw": "100", "value_num": 100}],
    )

    response = TestClient(main.app).delete(f"/personal-reports/{report['id']}")

    assert response.status_code == 200
    assert not report_path.exists()
    assert db.get_personal_report(report["id"]) is None
    assert db.personal_reports_summary()["total_measurements"] == 0
    connection = db.connection()
    try:
        assert connection.execute("SELECT COUNT(*) FROM personal_measurements").fetchone()[0] == 0
    finally:
        connection.close()
