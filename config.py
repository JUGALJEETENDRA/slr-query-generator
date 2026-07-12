DEFAULT_MODEL = "qwen2.5:3b"

# Optional emergency development cap for every screening path.
# Keep None for production: uploaded CSVs are screened in full by default.
DEV_SCREENING_ROW_LIMIT = None

# Two-stage screening configuration
TWO_STAGE_SCREENING_ENABLED = False  # default keeps behavior unchanged
FIRST_STAGE_MODEL = "qwen2.5:3b"
SECOND_STAGE_MODEL = "qwen2.5:7b"
LOCAL_CHECKPOINT_INTERVAL = 25

# Gemini Web Automation screening configuration
GEMINI_WEB_BATCH_SIZE = 5
GEMINI_WEB_PROFILE_DIR = "browser_profiles/gemini"

# Model-powered semantic judge configuration.
# Modes: off, fast, balanced, full. Heavy models are loaded lazily and fall
# back gracefully when unavailable in the local environment.
ENABLE_MODEL_JUDGES = True
ENABLE_RERANKER_JUDGE = True
ENABLE_NLI_JUDGE = True
ENABLE_ZERO_SHOT_JUDGE = True
ENABLE_LLM_JUDGE = False
ENABLE_HF_MODEL_LOADING = False
ENABLE_HF_MODEL_DOWNLOAD = False
MODEL_JUDGE_MODE = "balanced"
MODEL_JUDGE_PROFILE = "light"
MODEL_JUDGE_TIMEOUT_SECONDS = 8.0
MAX_LLM_DIRECTIONAL_ROWS = 100
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
ZERO_SHOT_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

# Performance controls. Defaults preserve the current architecture.
SCREENING_PIPELINE_MODE = "current"
ENABLE_BATCH_LLM_JUDGE = False
ENABLE_AGGRESSIVE_LLM_GATING = False
ENABLE_SEMANTIC_FRAME_CACHE = False
ENABLE_CURRENT_MODE_CACHE = False
LLM_BATCH_SIZE = 5
LLM_BATCH_MAX_CHARS = 12000
LLM_BATCH_TIMEOUT_SECONDS = 30.0
ENABLE_PARALLEL_SCREENING = False
SCREENING_WORKERS = 1
OLLAMA_MAX_CONCURRENT = 1
ENABLE_PERFORMANCE_PROFILE = True
