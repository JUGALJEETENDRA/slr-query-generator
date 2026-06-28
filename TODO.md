# TODO - Regression Fixes

## Step 1: Runtime regression (screener.py)
- Make LLM rationale generation optional.
- Introduce a flag (e.g., `generate_reason: bool = True`).
- Default must preserve existing application behavior.
- Benchmark execution must set the flag to `False`.

## Step 2: Comparator regression (semantic_comparator.py)
- Adjust decision policy only.
- Keep task identity conflict checks.
- When there is a canonical identity conflict, do NOT automatically REJECT.
- Allow strong semantic evidence to return MAYBE instead.
- Preserve true conflicts (e.g., prediction ≠ diagnosis) via compatibility checks.

## Step 3: Validation
- Run `python -m py_compile`.
- Run the 100-paper benchmark.
- Confirm runtime and KEEP/MAYBE/REJECT distribution are close to baseline.

