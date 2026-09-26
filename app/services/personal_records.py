"""Local processing helpers for personal laboratory reports.

The feature deliberately does not send report contents to a third-party service.
PDFs are read with pypdf and images are processed with the locally installed
Tesseract engine through pytesseract.
"""

from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.config import settings


class ReportProcessingError(ValueError):
    """A user-facing error raised while validating or extracting a report."""


ALLOWED_EXTENSIONS = {
    ".pdf": {"application/pdf", "application/x-pdf"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg", "image/jpg"},
    ".jpeg": {"image/jpeg", "image/jpg"},
    ".webp": {"image/webp"},
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
EXTRACTED_TEXT_LIMIT = 500_000

# Small, local seed corpus.  This provides a predictable ML classification step
# without transmitting sensitive report text or requiring a downloaded model.
_TRAINING_EXAMPLES = {
    "Blood Sugar": [
        "fasting blood sugar glucose fasting plasma glucose random blood sugar",
        "serum glucose post prandial glucose diabetes blood sugar",
    ],
    "HbA1c": [
        "hba1c glycated hemoglobin estimated average glucose diabetes control",
        "glycosylated haemoglobin hb a1c percentage",
    ],
    "Complete Blood Count (CBC)": [
        "complete blood count cbc hemoglobin wbc rbc platelet hematocrit",
        "differential count neutrophils lymphocytes mch mcv rdw",
    ],
    "Vitamin D": [
        "vitamin d 25 oh vitamin d total vitamin d3 cholecalciferol",
        "25 hydroxy vitamin d deficiency ng ml",
    ],
    "Lipid Profile": [
        "lipid profile cholesterol triglycerides hdl ldl vldl",
        "total cholesterol non hdl cholesterol cardiac risk ratio",
    ],
    "Thyroid Function": [
        "thyroid profile tsh free t3 free t4 thyroxine triiodothyronine",
        "thyroid stimulating hormone hypothyroidism",
    ],
    "Liver Function": [
        "liver function test bilirubin alt ast sgpt sgot alkaline phosphatase albumin",
        "hepatic panel total protein globulin",
    ],
    "Kidney Function": [
        "kidney renal function creatinine urea bun uric acid egfr",
        "renal profile blood urea nitrogen",
    ],
    "Iron Studies": [
        "iron studies ferritin serum iron tibc transferrin saturation",
        "iron binding capacity anaemia ferritin",
    ],
    "Urine Analysis": [
        "urine routine urinalysis urine protein urine sugar specific gravity",
        "urine microscopy pus cells epithelial cells urine examination",
    ],
}

_classifier_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
_classifier_labels = [
    label for label, examples in _TRAINING_EXAMPLES.items() for _ in examples
]
_classifier_samples = [
    sample for examples in _TRAINING_EXAMPLES.values() for sample in examples
]
_classifier = LogisticRegression(max_iter=500, random_state=42)
_classifier.fit(_classifier_vectorizer.fit_transform(_classifier_samples), _classifier_labels)


def validate_report_upload(filename: str | None, content_type: str | None, raw: bytes) -> str:
    """Validate extension, MIME type, size, and basic file signature."""

    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ReportProcessingError("Upload a PDF, PNG, JPG, JPEG, or WEBP laboratory report.")
    if not raw:
        raise ReportProcessingError("The uploaded report is empty.")
    if len(raw) > settings().personal_report_max_bytes:
        size_mb = settings().personal_report_max_bytes // (1024 * 1024)
        raise ReportProcessingError(f"Report must be smaller than {size_mb} MB.")

    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime and mime != "application/octet-stream" and mime not in ALLOWED_EXTENSIONS[suffix]:
        raise ReportProcessingError("The file type does not match its filename extension.")
    if suffix == ".pdf":
        if not raw.lstrip().startswith(b"%PDF-"):
            raise ReportProcessingError("The uploaded PDF is not valid.")
    else:
        try:
            with Image.open(BytesIO(raw)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ReportProcessingError("The uploaded image is not valid.") from exc
    return suffix


def store_report_file(filename: str | None, raw: bytes, suffix: str) -> tuple[str, str]:
    """Store the original file with a generated name, never the user supplied name."""

    target_dir = Path(settings().personal_reports_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}{suffix}"
    path = target_dir / stored_name
    path.write_bytes(raw)
    return str(path), Path(filename or f"report{suffix}").name


def extract_report_text(path: str | Path, suffix: str) -> tuple[str, str]:
    """Return extracted text and the local method used to obtain it."""

    source = Path(path)
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(source))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            method = "pdf_text"
        else:
            import pytesseract

            with Image.open(source) as image:
                text = pytesseract.image_to_string(image)
            method = "tesseract_ocr"
    except Exception as exc:
        kind = "PDF text" if suffix == ".pdf" else "image OCR"
        raise ReportProcessingError(f"{kind} extraction failed: {exc}") from exc

    cleaned = normalize_text(text)
    if not cleaned:
        hint = "Use a text-based PDF" if suffix == ".pdf" else "Use a clearer image with readable text"
        raise ReportProcessingError(f"No readable text was found. {hint} and try again.")
    return cleaned[:EXTRACTED_TEXT_LIMIT], method


def normalize_text(text: str) -> str:
    return "\n".join(
        re.sub(r"[\t\f\v ]+", " ", line).strip()
        for line in str(text or "").replace("\r", "\n").split("\n")
        if line.strip()
    )


def classify_test_type(text: str) -> tuple[str, float]:
    """Classify report text locally with a compact TF-IDF / logistic model."""

    sample = (text or "")[:20_000].lower()
    if not sample:
        return "General Laboratory Report", 0.0
    probabilities = _classifier.predict_proba(_classifier_vectorizer.transform([sample]))[0]
    index = int(probabilities.argmax())
    label = str(_classifier.classes_[index])
    confidence = float(probabilities[index])

    # Reinforce exact clinical markers that commonly survive imperfect OCR.
    markers = {
        "Blood Sugar": ("glucose", "blood sugar", "fasting sugar", "post prandial"),
        "HbA1c": ("hba1c", "hb a1c", "glycated"),
        "Complete Blood Count (CBC)": ("complete blood count", "cbc", "hemoglobin", "platelet"),
        "Vitamin D": ("vitamin d", "25-oh", "25 hydroxy"),
        "Lipid Profile": ("lipid", "cholesterol", "triglyceride", "ldl", "hdl"),
        "Thyroid Function": ("thyroid", "tsh", "free t3", "free t4"),
        "Liver Function": ("bilirubin", "sgpt", "sgot", "alkaline phosphatase"),
        "Kidney Function": ("creatinine", "blood urea", "egfr", "renal"),
        "Iron Studies": ("ferritin", "tibc", "transferrin", "serum iron"),
        "Urine Analysis": ("urinalysis", "urine routine", "urine examination"),
    }
    keyword_scores = {
        name: sum(marker in sample for marker in words) for name, words in markers.items()
    }
    best_keyword = max(keyword_scores, key=keyword_scores.get)
    if keyword_scores[best_keyword] and (
        best_keyword != label or keyword_scores[best_keyword] >= 2
    ):
        label = best_keyword
        confidence = max(confidence, min(0.98, 0.62 + keyword_scores[label] * 0.12))
    return label, round(confidence * 100, 1)


def extract_report_details(text: str, classified_test_type: str) -> dict:
    """Extract common laboratory fields and structured measurement rows."""

    patient_name = _find_labeled_value(text, ("patient name", "patient", "name"))
    doctor_name = _find_labeled_value(
        text, ("referring doctor", "referred by", "consultant", "doctor")
    )
    date_value = _find_labeled_value(
        text, ("report date", "collection date", "sample date", "date")
    )
    reported_date = normalize_report_date(date_value or _find_first_date(text))
    test_name = _find_labeled_value(text, ("test name", "investigation", "test"))
    measurements = extract_measurements(text)
    return {
        "patient_name": patient_name,
        "doctor_name": doctor_name,
        "reported_date": reported_date,
        "test_name": test_name or classified_test_type,
        "measurements": measurements,
    }


def _find_labeled_value(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        pattern = re.compile(
            rf"(?:^|\n)\s*{re.escape(label)}\s*(?:name)?\s*[:#\-]\s*([^\n|]{{2,100}})",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            value = re.split(r"\s{2,}|\s+(?:age|sex|gender|id)\s*[:#\-]", match.group(1), 1, re.I)[0]
            value = value.strip(" -:;")
            if value and not re.search(r"\b(?:date|doctor|age|gender)\b", value, re.I):
                return value[:100]
    return None


_DATE_PATTERNS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
)


def _find_first_date(text: str) -> str | None:
    match = re.search(
        r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b",
        text,
    )
    return match.group(0) if match else None


def normalize_report_date(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value.strip())
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


_MEASUREMENT_START = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9()/%+.,' \-]{1,80}?)(?:\s*[:=]\s*|\s{2,})(?P<value>[<>≤≥]?\s*\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-zµμ%][A-Za-z0-9µμ%/^.\-]*)?",
    re.IGNORECASE,
)
_MEASUREMENT_FALLBACK = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9()/%+.,' \-]{1,80}?)\s+(?P<value>[<>≤≥]?\s*\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-zµμ%][A-Za-z0-9µμ%/^.\-]*)?",
    re.IGNORECASE,
)
_RANGE = re.compile(
    r"(?P<range>(?:\(?\s*\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*\)?)|(?:[<>≤≥]\s*\d+(?:\.\d+)?))",
    re.IGNORECASE,
)
_EXCLUDED_MEASUREMENT_NAMES = {
    "patient name", "patient", "doctor", "report date", "date", "age", "gender",
    "sex", "sample type", "lab no", "laboratory", "page", "method",
}


def extract_measurements(text: str) -> list[dict]:
    """Parse typical lab result lines into values, units, and reference ranges."""

    measurements: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()
    for line in normalize_text(text).splitlines():
        match = _MEASUREMENT_START.match(line) or _MEASUREMENT_FALLBACK.match(line)
        if not match:
            continue
        name = re.sub(r"\s+", " ", match.group("name")).strip(" -:.")
        normalized_name = name.lower()
        if normalized_name in _EXCLUDED_MEASUREMENT_NAMES or len(name) < 2:
            continue
        raw_value = match.group("value").replace(" ", "")
        try:
            value_num = float(re.sub(r"^[<>≤≥]", "", raw_value))
        except ValueError:
            continue
        unit = (match.group("unit") or "").strip() or None
        tail = line[match.end():]
        range_match = _RANGE.search(tail)
        reference_range = range_match.group("range").strip() if range_match else None
        low, high = parse_reference_range(reference_range)
        flag_match = re.search(r"\b(HIGH|LOW|NORMAL|H|L)\b", tail, re.IGNORECASE)
        status = derive_measurement_status(value_num, low, high, flag_match.group(1) if flag_match else None)
        key = (normalized_name, raw_value, unit)
        if key in seen:
            continue
        seen.add(key)
        measurements.append(
            {
                "test_name": name[:100],
                "value_raw": raw_value,
                "value_num": value_num,
                "unit": unit,
                "reference_low": low,
                "reference_high": high,
                "reference_range": reference_range,
                "status": status,
            }
        )
    return measurements[:200]


def parse_reference_range(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
    if "-" in value or "–" in value or "to" in value.lower():
        return (numbers[0], numbers[1]) if len(numbers) >= 2 else (None, None)
    if value.lstrip().startswith(("<", "≤")) and numbers:
        return None, numbers[0]
    if value.lstrip().startswith((">", "≥")) and numbers:
        return numbers[0], None
    return None, None


def derive_measurement_status(
    value: float, low: float | None, high: float | None, printed_flag: str | None
) -> str:
    flag = (printed_flag or "").upper()
    if flag in {"L", "LOW"}:
        return "low"
    if flag in {"H", "HIGH"}:
        return "high"
    if flag == "NORMAL":
        return "normal"
    if low is not None and value < low:
        return "low"
    if high is not None and value > high:
        return "high"
    if low is not None or high is not None:
        return "normal"
    return "unknown"
