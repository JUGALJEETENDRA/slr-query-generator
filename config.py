DEFAULT_MODEL = "qwen2.5:3b"

# Hybrid two-stage screening configuration
HYBRID_SCREENING_ENABLED = False  # default keeps behavior unchanged
FIRST_STAGE_MODEL = "qwen2.5:3b"
SECOND_STAGE_MODEL = "qwen2.5:7b"

# Escalation mode (confidence-driven hooks). For now default: only MAYBE papers.
# Future: could re-screen low-confidence REJECTs.
HYBRID_ESCALATE_ON = "MAYBE_ONLY"  # "MAYBE_ONLY" | "MAYBE_AND_LOW_CONF_REJECT"
