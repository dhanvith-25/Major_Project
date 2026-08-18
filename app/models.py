from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel): query:str=Field(...,min_length=2,max_length=5000)
class HealthPredictionRequest(BaseModel): claim:str=Field(...,min_length=2,max_length=5000)
class SourceResponse(BaseModel):
    title:str; url:str; published_date:Optional[str]=None
class QueryResponse(BaseModel):
    answer:str; verdict:str; route:str; model_used:Optional[str]=None
    dataset_hit:bool; dataset_similarity:Optional[float]=None
    validation_score:Optional[float]=None; validation_status:str
    validation_reason:Optional[str]=None
    validation_components:dict[str,float]=Field(default_factory=dict)
    sources:list[SourceResponse]=Field(default_factory=list)
    ingested_into_dataset:bool=False
    queued_for_training:bool=False; claim_id:Optional[int]=None
class TrainResponse(BaseModel):
    status:str; examples:int; output_dir:Optional[str]=None; message:str
