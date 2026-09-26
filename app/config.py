import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

class Settings:
    app_name="Satarka Hybrid Verification API"; version="3.0.0"
    db_path=os.getenv("SATARKA_DB","data/satarka.db")
    auto_ingest_threshold=float(os.getenv("AUTO_INGEST_THRESHOLD","95"))
    training_threshold=float(os.getenv("TRAINING_THRESHOLD","95"))
    training_batch_size=int(os.getenv("TRAINING_BATCH_SIZE","100"))
    dataset_match_threshold=float(os.getenv("DATASET_MATCH_THRESHOLD","0.88"))
    tavily_api_key=os.getenv("TAVILY_API_KEY","").strip()
    google_factcheck_api_key=os.getenv("GOOGLE_FACTCHECK_API_KEY","").strip()
    groq_key=os.getenv("GROQ_API_KEY","").strip()
    openrouter_key=os.getenv("OPENROUTER_API_KEY","").strip()
    cheap_model=os.getenv("CHEAP_MODEL","llama-3.1-8b-instant").strip()
    medium_model=os.getenv("MEDIUM_MODEL","llama-3.3-70b-versatile").strip()
    premium_model=os.getenv("PREMIUM_MODEL","openai/gpt-oss-120b").strip()
    validator_model=os.getenv("VALIDATOR_MODEL","openai/gpt-oss-120b").strip()
    training_base_model=os.getenv("TRAINING_BASE_MODEL","Qwen/Qwen2.5-0.5B-Instruct").strip()
    training_output_dir=os.getenv("TRAINING_OUTPUT_DIR","models/satarka_adapter").strip()
    training_enabled=os.getenv("TRAINING_ENABLED","false").lower()=="true"
    personal_reports_dir=os.getenv("PERSONAL_REPORTS_DIR", "data/personal_reports").strip()
    personal_report_max_bytes=int(os.getenv("PERSONAL_REPORT_MAX_BYTES", str(10 * 1024 * 1024)))
    @property
    def cheap_configured(self): return bool(self.groq_key and self.cheap_model)
    @property
    def medium_configured(self): return bool(self.groq_key and self.medium_model)
    @property
    def premium_configured(self): return bool(self.openrouter_key and self.premium_model)
    @property
    def validator_configured(self): return bool(self.openrouter_key and self.validator_model)
    @property
    def factcheck_configured(self): return bool(self.google_factcheck_api_key)

@lru_cache
def settings(): return Settings()
