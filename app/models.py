from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=5000)


class HealthPredictionRequest(BaseModel):
    claim: str = Field(..., min_length=2, max_length=5000)


class SourceResponse(BaseModel):
    title: str
    url: str
    published_date: Optional[str] = None


class EvidenceResponse(BaseModel):
    url: str
    relationship: str
    relevance: float = 0
    source_type: Optional[str] = None
    published_date: Optional[str] = None
    reason: str = ""


class SemanticMatchResponse(BaseModel):
    id: int
    canonical_claim: str
    answer: str
    verdict: str
    similarity: float
    similarity_score: float
    validation_score: float
    sources: list[SourceResponse] = Field(default_factory=list)


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=5000)
    k: int = Field(default=5, ge=1, le=50)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    verdict: str
    route: str
    model_used: Optional[str] = None
    dataset_hit: bool
    dataset_similarity: Optional[float] = None
    semantic_matches: list[SemanticMatchResponse] = Field(default_factory=list)
    validation_score: Optional[float] = None
    validation_status: str
    validation_reason: Optional[str] = None
    validation_components: dict[str, float] = Field(default_factory=dict)
    sources: list[SourceResponse] = Field(default_factory=list)
    evidence: list[EvidenceResponse] = Field(default_factory=list)
    explanation: Optional[str] = None
    ingested_into_dataset: bool = False
    queued_for_training: bool = False
    claim_id: Optional[int] = None


class QueryReportItem(BaseModel):
    id: int
    query: str
    route: str
    model_used: Optional[str] = None
    dataset_hit: bool = False
    validation_score: Optional[float] = None
    status: str
    verdict: str
    answer: str = ""
    created_at: Optional[str] = None
    error: Optional[str] = None
    response: dict = Field(default_factory=dict)


class QueryReportResponse(BaseModel):
    generated_at: str
    total_queries: int
    summary: dict[str, int] = Field(default_factory=dict)
    queries: list[QueryReportItem] = Field(default_factory=list)


class TrainResponse(BaseModel):
    status: str
    examples: int
    output_dir: Optional[str] = None
    message: str


class ReviewStatusUpdate(BaseModel):
    claim_id: int
    status: str = Field(default="pending", pattern="^(approved|rejected|pending)$")
    note: str = ""


class TrainingMetrics(BaseModel):
    rouge_l: Optional[float] = None
    rouge_2: Optional[float] = None
    bleu: Optional[float] = None
    eval_loss: Optional[float] = None
    final_model_dir: Optional[str] = None


class AdminReviewClaim(BaseModel):
    id: int
    canonical_claim: str
    answer: str
    verdict: str
    validation_score: float
    review_status: str
    review_note: str = ""
    updated_at: Optional[str] = None


class AdminReviewDashboard(BaseModel):
    summary: dict[str, int] = Field(default_factory=dict)
    claims: list[AdminReviewClaim] = Field(default_factory=list)
