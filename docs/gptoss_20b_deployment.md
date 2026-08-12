# gpt-oss-20b deployment: 128 GB laptop

This repository is ready to use `gpt-oss:20b` through Ollama without putting
model weights, caches, or a GPU dependency in the project. The application
architecture is:

```text
Browser / API client
        |
        v
FastAPI (server.py)
        |
        v
SLR screening workflow (bulk_screen.py, screener.py)
        |
        v
LocalInferenceEngine -> ollama_client.py -> Ollama HTTP API
        |
        v
gpt-oss:20b weights on the 128 GB laptop
```

The 8 GB laptop is intentionally not part of the inference path. Use it to edit
or test the application; copy this repository to the 128 GB laptop for model
download and end-to-end execution.

## One-time setup on the 128 GB laptop

1. Install 64-bit Python and Ollama for Windows, then copy/clone this repository.
2. From the repository folder, install the app dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Start `run_gptoss_20b.bat`. It checks for Ollama, downloads
   `gpt-oss:20b` only if missing, applies the CPU-safe profile, and starts the
   app. The first download needs internet access and sufficient free disk space.

Open `http://localhost:8000/docs` after it starts. The endpoint
`GET /health/local-model` must show both `reachable: true` and
`model_installed: true` before starting a long CSV job.

## End-to-end smoke test

With the server running, use a second PowerShell window:

```powershell
Invoke-RestMethod http://localhost:8000/health/local-model | ConvertTo-Json

$body = @{
  question = 'How is machine learning used to automate systematic literature review screening?'
  title = 'Machine learning assisted title and abstract screening for systematic reviews'
  abstract = 'We evaluate classifiers that prioritize and screen candidate studies during systematic reviews.'
  processing_engine = 'local'
  model = 'gpt-oss:20b'
} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/screen -Method Post -ContentType 'application/json' -Body $body
```

This verifies the complete application path, not merely that the model can
produce text. For a full test, upload a small CSV with `Title` and `Abstract`
columns to `/screen_csv`, wait for `/progress` to report `completed`, and
inspect the generated CSV in `outputs/`.

## Resource policy

The model is an open-weight MoE model with 21B total parameters but 3.6B active
parameters per token. OpenAI states that the quantized 20B weights require about
16 GB memory; this profile uses an 8k context and one request at a time to leave
headroom for Windows, the app, and the KV cache. A slow CPU is acceptable for
correctness testing, but expect low throughput; do not enable parallel screening
until a serial run is stable.

`gpt-oss` is self-hosted: it is not provided through the OpenAI API or ChatGPT.
See the official [gpt-oss announcement](https://openai.com/index/introducing-gpt-oss/)
for the model architecture, memory statement, and supported local runtimes.
