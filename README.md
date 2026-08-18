# Satarka Backend v3
Install:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy .env.example to .env and add newly rotated API keys.
Run: python -m app.main
Docs: http://127.0.0.1:8081/docs

Health claim frontend:
http://127.0.0.1:8081/

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
