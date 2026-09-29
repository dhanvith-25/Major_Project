import io
import re
from functools import lru_cache
from typing import Any, Iterable

from app.db import get_symptom_mapping, save_symptom_mapping


DISCLAIMER = "This is not a medical diagnosis. Please consult a doctor. Results may be inaccurate."
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

DISEASE_PROFILES: dict[str, dict[str, str]] = {
    "Allergic rhinitis": {
        "risk_level": "low",
        "prevention": "Avoid known allergens when possible and discuss persistent symptoms with a healthcare professional.",
        "explanation": "Allergic rhinitis is an immune response to airborne allergens and can cause sneezing, nasal symptoms, and itchy or watery eyes.",
    },
    "Asthma": {
        "risk_level": "high",
        "prevention": "Follow your clinician's asthma plan and avoid known triggers. Seek urgent care for severe or worsening breathing difficulty.",
        "explanation": "Asthma can narrow and inflame the airways, causing wheezing, chest tightness, cough, or shortness of breath.",
    },
    "Common cold": {
        "risk_level": "low",
        "prevention": "Rest, drink fluids, wash hands often, and limit close contact while unwell. Contact a clinician if symptoms worsen or persist.",
        "explanation": "A common cold is a viral upper-respiratory infection that may cause a runny nose, cough, and sore throat.",
    },
    "Dehydration": {
        "risk_level": "medium",
        "prevention": "Drink fluids regularly. Seek medical advice for persistent vomiting, confusion, fainting, or inability to keep fluids down.",
        "explanation": "Dehydration occurs when the body loses more fluid than it takes in and may cause thirst, dizziness, and dark urine.",
    },
    "Gastroenteritis": {
        "risk_level": "medium",
        "prevention": "Wash hands carefully, use safe food and water, and replace fluids. Seek care for severe pain, blood, or dehydration signs.",
        "explanation": "Gastroenteritis irritates the stomach and intestines and can cause nausea, vomiting, diarrhea, or abdominal discomfort.",
    },
    "Influenza": {
        "risk_level": "medium",
        "prevention": "Stay home while feverish, wash hands, and discuss vaccination and treatment options with a healthcare professional.",
        "explanation": "Influenza is a viral respiratory illness that can cause fever, chills, body aches, cough, and fatigue.",
    },
    "Migraine": {
        "risk_level": "medium",
        "prevention": "Rest in a quiet, dark room and note possible triggers. Seek urgent care for a sudden severe headache or new neurological symptoms.",
        "explanation": "Migraine is a neurological condition that can cause recurring headache, nausea, and sensitivity to light or sound.",
    },
    "Sinusitis": {
        "risk_level": "low",
        "prevention": "Rest, drink fluids, and seek medical advice for severe symptoms, worsening pain, or symptoms that do not improve.",
        "explanation": "Sinusitis is inflammation of the sinuses and may cause facial pressure, congestion, and headache.",
    },
    "Strep throat": {
        "risk_level": "medium",
        "prevention": "Avoid sharing drinks and wash hands. A clinician can determine whether testing or treatment is needed.",
        "explanation": "Strep throat is a bacterial throat infection that may cause sore throat, fever, and pain with swallowing.",
    },
}

CLASSIFIER_EXAMPLES: list[tuple[str, str]] = [
    ("runny nose cough sneezing mild sore throat", "Common cold"),
    ("nasal congestion sneezing and a mild cough", "Common cold"),
    ("sore throat runny nose and fatigue", "Common cold"),
    ("fever chills body aches cough and fatigue", "Influenza"),
    ("sudden fever headache muscle aches and cough", "Influenza"),
    ("high temperature chills weakness and dry cough", "Influenza"),
    ("fever sore throat painful swallowing swollen glands", "Strep throat"),
    ("pain when swallowing and sore throat without cough", "Strep throat"),
    ("swollen neck glands fever and throat pain", "Strep throat"),
    ("sneezing itchy eyes watery eyes and runny nose", "Allergic rhinitis"),
    ("itchy nose repeated sneezing and watery eyes", "Allergic rhinitis"),
    ("seasonal nasal congestion and eye itching", "Allergic rhinitis"),
    ("wheezing shortness of breath chest tightness and cough", "Asthma"),
    ("chest tightness wheezing and difficulty breathing", "Asthma"),
    ("breathlessness with wheezing and recurring cough", "Asthma"),
    ("thirst dry mouth dizziness and dark urine", "Dehydration"),
    ("dizziness weakness and reduced urination", "Dehydration"),
    ("thirst fatigue and dry mouth after fluid loss", "Dehydration"),
    ("nausea vomiting diarrhea and stomach cramps", "Gastroenteritis"),
    ("diarrhea abdominal pain and vomiting", "Gastroenteritis"),
    ("stomach cramps nausea and loose stools", "Gastroenteritis"),
    ("throbbing headache nausea and light sensitivity", "Migraine"),
    ("recurring headache with sensitivity to sound", "Migraine"),
    ("severe headache nausea and visual sensitivity", "Migraine"),
    ("facial pressure nasal congestion and headache", "Sinusitis"),
    ("sinus pressure blocked nose and facial pain", "Sinusitis"),
    ("nasal congestion headache and pressure around the face", "Sinusitis"),
]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


SYMPTOM_RULES: dict[str, list[str]] = {
    "Common cold": ["runny nose", "cough", "sneezing", "mild fever", "sore throat", "fatigue"],
    "Influenza / flu": ["fever", "body aches", "chills", "cough", "fatigue", "headache"],
    "Allergic reaction": ["sneezing", "itchy eyes", "runny nose", "watery eyes", "cough"],
    "Strep throat": ["sore throat", "fever", "swollen glands", "pain when swallowing"],
    "Asthma / breathing irritation": ["wheezing", "shortness of breath", "chest tightness", "cough"],
    "Dehydration": ["dry mouth", "dizziness", "fatigue", "dark urine", "nausea", "thirst"],
    "Migraine": ["headache", "nausea", "light sensitivity", "sensitivity to sound"],
    "Gastrointestinal illness": ["nausea", "vomiting", "stomach pain", "diarrhea", "bloating"],
    "Sinus infection": ["facial pain", "nasal congestion", "headache", "fever", "pressure in face"],
}


def _normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9\s-]", " ", (value or "").lower()).strip()


def _parse_symptoms(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,;\n]+|\band\b", text)
    cleaned: list[str] = []
    for part in parts:
        normalized = _normalize_phrase(part)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _canonical_symptoms(symptoms: Iterable[str]) -> str:
    return ", ".join(sorted(set(symptom for symptom in symptoms if symptom)))


def _embed_texts(texts: list[str], tokenizer: Any, model: Any, torch: Any, device: Any) -> Any:
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


@lru_cache(maxsize=1)
def _load_pubmedbert_classifier() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL)
    model = AutoModel.from_pretrained(PUBMEDBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    example_texts = [text for text, _ in CLASSIFIER_EXAMPLES]
    labels = [label for _, label in CLASSIFIER_EXAMPLES]
    embeddings = _embed_texts(example_texts, tokenizer, model, torch, device)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(embeddings, labels)
    return tokenizer, model, classifier, torch, device


def _predict_disease(symptom_text: str) -> dict[str, Any]:
    tokenizer, model, classifier, torch, device = _load_pubmedbert_classifier()
    embedding = _embed_texts([symptom_text], tokenizer, model, torch, device)
    probabilities = classifier.predict_proba(embedding)[0]
    best_index = int(probabilities.argmax())
    disease = str(classifier.classes_[best_index])
    profile = DISEASE_PROFILES[disease]
    return {
        "cause": disease,
        "risk_level": profile["risk_level"],
        "prevention": profile["prevention"],
        "explanation": profile["explanation"],
        "confidence": float(probabilities[best_index]),
    }


def _extract_ocr_text(image_file: Any) -> dict[str, Any]:
    if image_file is None:
        return {"ocr_text": "", "ocr_status": "No image provided", "ocr_warning": ""}

    try:
        content = image_file.read()
    except Exception:
        content = b""

    if not content:
        return {"ocr_text": "", "ocr_status": "Image file was empty", "ocr_warning": "No image content was uploaded."}

    if Image is None or pytesseract is None:
        return {
            "ocr_text": "",
            "ocr_status": "OCR is unavailable",
            "ocr_warning": "OCR support is not installed on this server. Please install Tesseract/Pillow or enter symptoms manually; the symptom analysis will still work without image text extraction.",
        }

    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        text = pytesseract.image_to_string(img)
        cleaned = text.strip()
        if cleaned:
            return {"ocr_text": cleaned, "ocr_status": "OCR completed", "ocr_warning": ""}
        return {
            "ocr_text": "",
            "ocr_status": "No readable text found",
            "ocr_warning": "The uploaded image did not contain readable text. Please upload a sharper PNG/JPG image or type the symptoms manually.",
        }
    except Exception as exc:  # pragma: no cover - runtime dependency may be missing
        return {
            "ocr_text": "",
            "ocr_status": "OCR processing failed",
            "ocr_warning": f"OCR could not process the image: {exc}. Please upload a clear PNG/JPG image or enter symptoms manually.",
        }


def _match_causes(symptoms: Iterable[str]) -> list[dict[str, Any]]:
    symptom_list = [sym for sym in symptoms if sym]
    matched: list[dict[str, Any]] = []
    seen: set[str] = set()

    for cause, keywords in SYMPTOM_RULES.items():
        matched_keywords = [sym for sym in symptom_list if any(sym in keyword or keyword in sym for keyword in keywords)]
        if not matched_keywords:
            continue

        if cause in seen:
            continue
        seen.add(cause)

        confidence = "high" if len(matched_keywords) >= 2 else "medium"
        matched.append({
            "cause": cause,
            "confidence": confidence,
            "matched_symptoms": sorted(set(matched_keywords)),
        })

    matched.sort(key=lambda item: (0 if item["confidence"] == "high" else 1, item["cause"]))
    return matched


def symptom_checker(image_file: Any = None, symptoms: str = "") -> dict[str, Any]:
    ocr_result = _extract_ocr_text(image_file)
    manual_symptoms = _parse_symptoms(symptoms)
    ocr_symptoms = _parse_symptoms(ocr_result.get("ocr_text", ""))
    all_symptoms = list(dict.fromkeys(manual_symptoms + ocr_symptoms))

    canonical_symptoms = _canonical_symptoms(all_symptoms)
    mapping = get_symptom_mapping(canonical_symptoms) if canonical_symptoms else None
    prediction: dict[str, Any] | None = None
    analysis_method = "symptoms_db"

    if mapping:
        prediction = {
            "cause": mapping["mapped_disease"],
            "risk_level": mapping["risk_level"],
            "prevention": mapping["prevention_notes"],
            "explanation": mapping["explanation"],
            "confidence": 1.0,
        }
    elif canonical_symptoms:
        analysis_method = "pubmedbert"
        try:
            prediction = _predict_disease(canonical_symptoms)
        except Exception:
            prediction = None
            analysis_method = "unavailable"

        if prediction:
            try:
                save_symptom_mapping(
                    canonical_symptoms,
                    prediction["cause"],
                    prediction["risk_level"],
                    prediction["prevention"],
                    prediction["explanation"],
                )
            except Exception:
                pass

    if prediction is None:
        if all_symptoms:
            cause = "Unable to determine"
            risk_level = "medium"
            prevention = "The symptom model is unavailable. Please consult a healthcare professional for assessment."
            explanation = "No reliable symptom mapping was available for this input."
            summary = f"{prevention} {DISCLAIMER}"
            suggested_causes: list[dict[str, Any]] = []
        else:
            cause = "No symptoms provided"
            risk_level = "low"
            prevention = "Enter symptoms or upload a clear image containing symptom text."
            explanation = "No symptoms were entered or detected."
            summary = f"{prevention} {DISCLAIMER}"
            suggested_causes = []
    else:
        cause = prediction["cause"]
        risk_level = prediction["risk_level"]
        prevention = prediction["prevention"]
        explanation = prediction["explanation"]
        confidence = float(prediction.get("confidence", 0.0))
        suggested_causes = [{
            "cause": cause,
            "confidence": "high" if confidence >= 0.75 else "medium" if confidence >= 0.45 else "low",
            "matched_symptoms": all_symptoms,
        }]
        summary = f"Possible cause: {cause}. {prevention} {DISCLAIMER}"

    return {
        "cause": cause,
        "prevention": prevention,
        "risk_level": risk_level,
        "disclaimer": DISCLAIMER,
        "explanation": explanation,
        "analysis_method": analysis_method,
        "ocr_text": ocr_result.get("ocr_text", ""),
        "ocr_status": ocr_result.get("ocr_status", "No image provided"),
        "ocr_warning": ocr_result.get("ocr_warning", ""),
        "manual_symptoms": manual_symptoms,
        "detected_symptoms": all_symptoms,
        "suggested_causes": suggested_causes,
        "summary": summary,
    }
