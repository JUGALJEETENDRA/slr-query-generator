# LitSync-XDQ research benchmark

LitSync-XDQ combines the repository's 100 capability questions with 20 difficult cross-domain
questions. Every model invocation runs in a child process, is terminated at 15 seconds, records
`TIMEOUT`, and is checkpointed immediately.

The hard-case `known_relevant_papers` arrays are intentionally empty until a human assessor or a
published review dataset supplies gold relevance labels. Search results must not be mislabeled as
ground truth. Retrieval recall, F3, and WSS@95 remain unavailable until that field is curated.

Install research challengers manually because downloads exceed the 15-second command rule:

```powershell
ollama pull qwen3.5:4b
ollama run hf.co/mradermacher/Autobool-Qwen4b-No-reasoning-GGUF:Q4_K_M
ollama pull llama3.1:8b
```

Enter `/bye` after the AutoBool import. Quick smoke run:

```powershell
python benchmark/query_generator/run_xdq.py --model qwen3.5:4b --strategy grounded --limit 3
```

Full grounded comparison and direct-prompt ablation:

```powershell
python benchmark/query_generator/run_xdq.py --strategy grounded
python benchmark/query_generator/run_xdq.py --strategy direct
```
