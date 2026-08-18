from app.services.validator import score, validate


def test_validator_health_gate_rejects_unrelated_input_without_llm():
    result = validate("Hello, how are you?", "", [])
    assert result["verdict"] == "INVALID_STATEMENT"
    assert score(result)[1] == "INVALID_STATEMENT"
