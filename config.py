DEFAULT_MODEL = "qwen2.5:3b"

# Temporary development cap for every screening path.
# Set to None to process full datasets.
DEV_SCREENING_ROW_LIMIT = 100

# Two-stage screening configuration
TWO_STAGE_SCREENING_ENABLED = False  # default keeps behavior unchanged
FIRST_STAGE_MODEL = "qwen2.5:3b"
SECOND_STAGE_MODEL = "qwen2.5:7b"
LOCAL_CHECKPOINT_INTERVAL = 25

# Gemini Web Automation screening configuration
GEMINI_WEB_BATCH_SIZE = 5
GEMINI_WEB_PROFILE_DIR = "browser_profiles/gemini"
