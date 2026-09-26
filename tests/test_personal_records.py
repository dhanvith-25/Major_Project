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
