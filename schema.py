# schema.py
from pydantic import BaseModel, Field, model_validator
from typing import List, Any

class SLRQueryContext(BaseModel):
    """
    The intermediate data contract running through the pipeline.
    Pure data carrier. Active validation and filtering are strictly 
    deferred downstream to Stage 5: The Validation Sieve.
    """
    technology: List[str] = Field(default_factory=list)
    domain: List[str] = Field(default_factory=list)
    comparison: List[str] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalizes any casing adjustments made by the local model
            return {k.lower(): v for k, v in data.items()}
        return data