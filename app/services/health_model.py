"""Small, local health-claim classifier for research prototyping.

The examples below are synthetic paraphrases of a small set of general health
claims. They are not a clinical dataset and must not be used for diagnosis,
treatment, or patient decisions. A production system needs a curated,
clinician-reviewed dataset and evidence retrieval.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/health_claim_model.joblib")
DATASET_PATH = Path("data/health_claims_dataset.csv")
LABELS = ("SUPPORTED", "CONTRADICTED", "UNCERTAIN")
HEALTH_TERMS = {
    "allergy", "antibiotic", "anxiety", "asthma", "blood", "cancer", "cold", "condition",
    "covid", "dehydration", "disease", "doctor", "dose", "drug", "exercise", "fever",
    "health", "heart", "infection", "injury", "medicine", "medication", "mental", "pain",
    "patient", "pregnant", "prescription", "protein", "symptom", "supplement", "surgery",
    "treatment", "vaccine", "virus", "vitamin", "water", "weight", "wellness", "sleep",
    "smoking", "skin", "physical", "blood pressure", "handwashing", "cure", "illness",
}

_SEEDS = [
    ("Drinking water is good for health", "SUPPORTED"),
    ("Is drinking water good for health?", "SUPPORTED"),
    ("Water is essential for normal body functions", "SUPPORTED"),
    ("Drinking water helps the body stay hydrated", "SUPPORTED"),
    ("Regular physical activity provides health benefits", "SUPPORTED"),
    ("Vaccination can reduce the risk of severe infectious disease", "SUPPORTED"),
    ("Antibiotics treat bacterial infections when prescribed appropriately", "SUPPORTED"),
    ("Handwashing helps reduce the spread of infections", "SUPPORTED"),
    ("Sleep supports physical and mental health", "SUPPORTED"),
    ("Smoking increases the risk of serious disease", "SUPPORTED"),
    ("Sunscreen helps protect skin from ultraviolet radiation", "SUPPORTED"),
    ("Oral rehydration can help replace fluids during diarrheal illness", "SUPPORTED"),
    ("Antibiotics cure the common cold", "CONTRADICTED"),
    ("Drinking hot water cures all diseases", "CONTRADICTED"),
    ("Drinking water replaces all medical treatment", "CONTRADICTED"),
    ("Drinking hot water kills every virus inside the body", "CONTRADICTED"),
    ("Garlic prevents all infections", "CONTRADICTED"),
    ("A detox drink removes every toxin from the body", "CONTRADICTED"),
    ("One vitamin guarantees that a person will never get sick", "CONTRADICTED"),
    ("Home remedies can replace all cancer treatment", "CONTRADICTED"),
   
    ("A social-media post is enough evidence to change treatment", "UNCERTAIN"),
    ("Routine blood pressure checks can help identify high blood pressure", "SUPPORTED"),
    ("Regular physical activity provides health benefits", "SUPPORTED"),
    ("Vaccination can reduce the risk of severe infectious disease", "SUPPORTED"),
    ("Antibiotics treat bacterial infections when prescribed appropriately", "SUPPORTED"),
    ("Handwashing helps reduce the spread of infections", "SUPPORTED"),
    ("Sleep supports physical and mental health", "SUPPORTED"),
    ("Smoking increases the risk of serious disease", "SUPPORTED"),
    ("Sunscreen helps protect skin from ultraviolet radiation", "SUPPORTED"),
    ("Oral rehydration can help replace fluids during diarrheal illness", "SUPPORTED"),
    ("Routine blood pressure checks can help identify high blood pressure", "SUPPORTED"),
    ("Eating vegetables can contribute to a healthy diet", "SUPPORTED"),
    ("Regular exercise can improve cardiovascular fitness", "SUPPORTED"),
    ("Physical activity can help maintain healthy body weight", "SUPPORTED"),
    ("Smoking is associated with increased lung cancer risk", "SUPPORTED"),
    ("Secondhand smoke can harm health", "SUPPORTED"),
    ("Excessive alcohol consumption can damage health", "SUPPORTED"),

]

def training_examples() -> list[dict[str, str]]:
    """Return exactly 100 synthetic examples for the first prototype model."""
    examples = []
    for claim, label in _SEEDS:
        variants = (
            claim,
            f"Is it true that {claim.lower()}?",
            f"A health post says that {claim.lower()}.",
            f"Please verify this health claim: {claim.lower()}.",
        )
        examples.extend({"claim": variant, "label": label} for variant in variants)
    return examples


def _load_examples_from_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return training_examples()

    with csv_path.open("r", newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} is missing a header row.")

        fieldnames = {str(field).strip().lower() for field in reader.fieldnames}
        if "claim" not in fieldnames or "label" not in fieldnames:
            raise ValueError("CSV must contain 'claim' and 'label' columns.")

        claim_field = next(field for field in reader.fieldnames if str(field).strip().lower() == "claim")
        label_field = next(field for field in reader.fieldnames if str(field).strip().lower() == "label")

        examples = []
        for row in reader:
            claim = str(row.get(claim_field) or "").strip()
            label = str(row.get(label_field) or "").strip().upper()
            if not claim or not label:
                continue
            if label not in LABELS:
                raise ValueError(f"Invalid label '{label}' in {csv_path}. Allowed labels: {LABELS}")
            examples.append({"claim": claim, "label": label})

    if not examples:
        raise ValueError(f"CSV file {csv_path} does not contain any valid training rows.")
    return examples


def train_health_model(model_path: Path = MODEL_PATH, dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    if dataset_path.exists():
        examples = _load_examples_from_csv(dataset_path)
    else:
        examples = training_examples()
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", newline="", encoding="utf-8") as dataset_file:
            writer = csv.DictWriter(dataset_file, fieldnames=("id", "claim", "label"))
            writer.writeheader()
            writer.writerows({"id": index, **row} for index, row in enumerate(examples, 1))

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit([row["claim"] for row in examples], [row["label"] for row in examples])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {
        "status": "trained",
        "examples": len(examples),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
    }


def _load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        train_health_model()
    return joblib.load(MODEL_PATH)


def is_health_related(claim: str) -> bool:
    normalized = " ".join(claim.lower().split())
    return any(term in normalized for term in HEALTH_TERMS)


def predict_health_claim(claim: str) -> dict[str, Any]:
    """Classify one claim; low-confidence predictions are returned as UNCERTAIN."""
    text = claim.strip()
    if len(text) < 2:
        raise ValueError("Health claim must contain at least 2 characters.")
    if not is_health_related(text):
        return {
            "claim": text,
            "verdict": "INVALID_STATEMENT",
            "confidence": 100.0,
            "probabilities": {},
            "model": "health-domain-filter",
            "warning": "This statement does not appear to be health-related.",
        }
    return _predict_with_model(text, _load_model())


def _predict_with_model(text: str, model: Pipeline) -> dict[str, Any]:
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best_index = int(probabilities.argmax())
    confidence = round(float(probabilities[best_index]) * 100, 2)
    predicted = str(classes[best_index]) if confidence >= 65 else "UNCERTAIN"
    return {
        "claim": text,
        "verdict": predicted,
        "confidence": confidence,
        "probabilities": {str(label): round(float(probabilities[i]) * 100, 2) for i, label in enumerate(classes)},
        "model": "tfidf-logistic-regression-health-prototype",
        "warning": "Synthetic research prototype; not medical advice or a diagnosis.",
    }


def predict_health_claims(claims: list[str]) -> list[dict[str, Any]]:
    """Predict a batch while loading the trained model only once."""
    model = _load_model()
    predictions = []
    for claim in claims:
        text = claim.strip()
        if not text:
            continue
        if not is_health_related(text):
            predictions.append({"claim": text, "verdict": "INVALID_STATEMENT", "confidence": 100.0, "probabilities": {}, "model": "health-domain-filter", "warning": "This statement does not appear to be health-related."})
        else:
            predictions.append(_predict_with_model(text, model))
    return predictions
