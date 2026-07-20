# schema.py
from pydantic import BaseModel, Field
from typing import List

class LiteralSpanExtraction(BaseModel):
    """
    Pass 1 of the new extractor.

    Only identify literal phrases from the research question.
    No classification.
    No semantic interpretation.
    """

    phrases: List[str] = Field(
        default_factory=list,
        description=(
            "Literal phrases copied exactly from the research question. "
            "Do not classify them. "
            "Do not generalize them. "
            "Do not invent new terms."
        ),
    )

class SLRExtractionContract(BaseModel):
    """
    Gateway extraction schema. Forces strict role separation between 
    the core target variables and the comparative baseline controls.
    """
    primary_paradigm: List[str] = Field(
        ..., 
        description="Literal method, technology, framework, or paradigm spans. Never include comparative baselines."
    )
    comparator_baseline: List[str] = Field(
        ..., 
        description="Literal methods, baselines, or controls explicitly compared in the question."
    )
    domain_context: List[str] = Field(
        ..., 
        description="Literal population, setting, deployment environment, or problem-domain spans."
    )
    outcome_variables: List[str] = Field(
        ..., 
        description="Literal tasks, outcomes, measures, or goals requested by the question."
    )

class SLRQueryContext(BaseModel):
    """
    Downstream operational schema. Maintained for perfect backwards-compatibility 
    with generator.py, compiler.py, and validator.py modules.
    """
    technology: List[str] = Field(default_factory=list)
    domain: List[str] = Field(default_factory=list)
    comparison: List[str] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
