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
        description="The innovative core technology, framework, or paradigm being evaluated (e.g., 'Zero Trust', 'Federated Learning'). NEVER include comparative baselines here."
    )
    comparator_baseline: List[str] = Field(
        ..., 
        description="The traditional methods, legacy baselines, or alternative control configurations being compared against (e.g., 'Perimeter Security', 'Centralized Database')."
    )
    domain_context: List[str] = Field(
        ..., 
        description="The deployment environment, specific industry sector, or problem domain (e.g., 'Robotic Surgery', 'Cloud Networks')."
    )
    outcome_variables: List[str] = Field(
        ..., 
        description="The target metrics, vulnerabilities, or engineering goals being measured (e.g., 'Device Compromise', 'Deployment Latency')."
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