from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import instructor
import re

# Keep your imports—these are the "engine"
from extractor import extract_5_facets
from generator import expand_base_synonyms
from acronym_expander import expand_acronym_layer
from classifier import classify_extracted_context
from registries import inject_implicit_academic_layers
from ontology_expander import expand_ontology_layer
from comparator_registry import expand_comparator_registry
from validator import run_validation_sieve
from compiler import compile_boolean_query
from schema import SLRQueryContext

app = FastAPI(title="SLR Query Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

local_client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local"),
    mode=instructor.Mode.MD_JSON
)
LOCAL_MODEL = "qwen2.5:7b"

class QuestionRequest(BaseModel):
    question: str

def compress_schema_for_ieee(context: SLRQueryContext) -> SLRQueryContext:
    import copy
    compressed = copy.deepcopy(context)
    merged_tech = list(context.technology[:2]) + list(context.comparison[:1])
    compressed.technology = [t.replace("*", "") for t in merged_tech if t]
    compressed.domain = [t.replace("*", "") for t in context.domain[:2]]
    compressed.outcomes = [t.replace("*", "") for t in context.outcomes[:2]]
    compressed.comparison = []
    compressed.context = []
    return compressed

@app.post("/generate")
async def generate(req: QuestionRequest):
    try:
        # STAGE 1: Extract
        raw = extract_5_facets(local_client, LOCAL_MODEL, req.question)
        s1 = SLRQueryContext(
            technology=raw.primary_paradigm,
            comparison=raw.comparator_baseline,
            domain=raw.domain_context,
            context=[],
            outcomes=raw.outcome_variables
        )

        # STAGE 2: Pipeline
        s2 = expand_base_synonyms(local_client, LOCAL_MODEL, s1)
        s3 = expand_acronym_layer(s2)
        primary_domain = classify_extracted_context(s3)
        s3_hydrated = inject_implicit_academic_layers(s3, primary_domain)
        s4 = expand_ontology_layer(s3_hydrated, primary_domain)
        s4_compared = expand_comparator_registry(s4)
        s5 = run_validation_sieve(s4_compared)

        # STAGE 3: Compile
        base_query = compile_boolean_query(s5).replace("\n", " ")
        ieee_safe = compress_schema_for_ieee(s5)
        ieee_query = compile_boolean_query(ieee_safe).replace("\n", " ")

        return {
            "status": "success",
            "google_scholar": base_query,
            "scopus": f"TITLE-ABS-KEY({base_query})",
            "web_of_science": f"TS=({base_query})",
            "ieee_xplore": ieee_query,
            "pubmed": re.sub(r'"([^"]+)"', r'"\1"[tiab]', base_query),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)