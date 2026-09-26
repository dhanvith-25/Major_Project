# Satarka Backend v3
Install:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy .env.example to .env and add newly rotated API keys.
Run: python -m app.main
Docs: http://127.0.0.1:8081/docs

React/Vite frontend (recommended for development):
cd frontend
npm install
npm run dev

The Vite application is available at http://127.0.0.1:5173/ and uses the API
at http://127.0.0.1:8081 by default. Set VITE_API_URL if the API is hosted
elsewhere.

Personal reports:
POST /personal-reports/upload with a `file` field (PDF, PNG, JPG, JPEG, or
WEBP; 10 MB maximum). Reports are stored only in `data/personal_reports/` and
their extracted fields and measurements are stored in SQLite. Use:
- GET /personal-reports/search?q=sugar
- GET /personal-reports/summary
- POST /personal-reports/ask with `{"question":"How has my blood sugar changed?"}`

PDFs use local text extraction; images use local Tesseract OCR. Install the
Tesseract executable and put it on PATH to process image reports.

Run tests:
pytest -q

Health classifier API:
POST /health/predict with {"claim":"Antibiotics cure the common cold"}

Training is OFF by default. For GPU training:
pip install -r requirements-training.txt
Set TRAINING_ENABLED=true
Then POST /training/run?force=true, or wait for the configured batch size.
Default training model: Qwen/Qwen2.5-0.5B-Instruct, using LoRA.
The validator remains independent; the model does not validate itself.
