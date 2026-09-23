import io
import re
from typing import Any, Iterable

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

    suggested_causes = _match_causes(all_symptoms)

    if not all_symptoms:
        summary = "No symptoms were entered or detected. Please add symptoms manually or upload an image with readable text."
    elif not suggested_causes:
        summary = "The provided symptoms do not strongly match the built-in symptom logic. Consider consulting a clinician for a more detailed assessment."
    else:
        top = suggested_causes[0]["cause"]
        summary = f"Based on the supplied symptoms, the most likely cause is {top.lower()}. Additional symptoms and medical review may still be needed."

    return {
        "ocr_text": ocr_result.get("ocr_text", ""),
        "ocr_status": ocr_result.get("ocr_status", "No image provided"),
        "ocr_warning": ocr_result.get("ocr_warning", ""),
        "manual_symptoms": manual_symptoms,
        "detected_symptoms": all_symptoms,
        "suggested_causes": suggested_causes,
        "summary": summary,
    }
