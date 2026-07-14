"""Small production configuration for the local-AI-first architecture."""

import os


MODEL_TIER = os.getenv("MODEL_TIER", "auto")
RESOURCE_PROFILE = os.getenv("RESOURCE_PROFILE", "balanced")
COMPACT_FAST_MODEL = os.getenv("COMPACT_FAST_MODEL", "qwen2.5:3b")
COMPACT_STRONG_MODEL = os.getenv("COMPACT_STRONG_MODEL", COMPACT_FAST_MODEL)
BALANCED_FAST_MODEL = os.getenv("BALANCED_FAST_MODEL", "qwen3:8b")
BALANCED_STRONG_MODEL = os.getenv("BALANCED_STRONG_MODEL", BALANCED_FAST_MODEL)
PERFORMANCE_FAST_MODEL = os.getenv("PERFORMANCE_FAST_MODEL", "qwen3:8b")
PERFORMANCE_STRONG_MODEL = os.getenv("PERFORMANCE_STRONG_MODEL", PERFORMANCE_FAST_MODEL)
LOCAL_CHECKPOINT_INTERVAL = int(os.getenv("LOCAL_CHECKPOINT_INTERVAL", "25"))
DEV_SCREENING_ROW_LIMIT = None
ENABLE_EXTERNAL_ENGINES = os.getenv("ENABLE_EXTERNAL_ENGINES", "false").lower() in {"1", "true", "yes"}

# Compatibility aliases for query generation and old clients. Screening no longer
# uses caller-selected stage models or a rule-based two-stage mode.
DEFAULT_MODEL = BALANCED_FAST_MODEL
FIRST_STAGE_MODEL = PERFORMANCE_FAST_MODEL
SECOND_STAGE_MODEL = PERFORMANCE_STRONG_MODEL
TWO_STAGE_SCREENING_ENABLED = False
GEMINI_WEB_BATCH_SIZE = 5
GEMINI_WEB_PROFILE_DIR = "browser_profiles/gemini"
