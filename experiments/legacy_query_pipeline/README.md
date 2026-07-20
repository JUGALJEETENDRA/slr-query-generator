# Historical ontology query baseline

This opt-in experiment reproduces LitSync's former hand-written domain registries. It is not a
production query generator and may emit expansions unsupported by the input or retrieved corpus.

Run it only for an explicit historical comparison:

```powershell
python benchmark/query_generator/run_benchmark.py --strategy historical_ontology
```

The deprecated `litsync_workflow` name resolves here only when the benchmark runner explicitly
registers experimental strategies.
