# Query Regression Tests

Add `.json` files in this directory to extend the permanent regression suite.
The benchmark runner discovers them automatically.

Example:

```json
[
  {
    "id": "comparator_loss_example",
    "category": "comparator_loss",
    "description": "The generated query must preserve the named comparator.",
    "query_contains_all": ["traditional screening"],
    "query_contains_none": ["unrelated parent concept"]
  }
]
```

Supported fields:

- `id`
- `category`
- `description`
- `query_contains_all`
- `query_contains_none`
- `max_runtime_ms`
