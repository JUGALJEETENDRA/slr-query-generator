@echo off
setlocal
cd /d "%~dp0"

rem gpt-oss:20b deployment profile for the 128 GB laptop.
set "LOCAL_MODEL=gpt-oss:20b"
set "GPT_OSS_MODEL=gpt-oss:20b"
set "OLLAMA_NUM_CTX=8192"
set "OLLAMA_NUM_PREDICT=768"
set "OLLAMA_NUM_THREAD=8"
set "OLLAMA_TEMPERATURE=0.1"
set "OLLAMA_KEEP_ALIVE=30m"
set "OLLAMA_REQUEST_TIMEOUT_SECONDS=900"
set "OLLAMA_MAX_CONCURRENT=1"
set "ENABLE_PARALLEL_SCREENING=false"
set "SCREENING_WORKERS=1"

where ollama >nul 2>nul || (
  echo Ollama is not installed or is not on PATH. Install it on this laptop first.
  exit /b 1
)

echo Checking whether gpt-oss:20b is already installed...
ollama list | findstr /I /C:"gpt-oss:20b" >nul || (
  echo Downloading gpt-oss:20b. This happens only on the 128 GB laptop.
  ollama pull gpt-oss:20b || exit /b 1
)

echo Starting the SLR service with gpt-oss:20b...
start "" "http://localhost:8000/docs"
python server.py
