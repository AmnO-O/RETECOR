# Evaluation Pipeline Tests

This directory contains the automated tests for `eval_pipeline`. The suite uses
Python's standard-library `unittest` framework and does not require network
access, model files, or third-party packages.

## Run the tests

Run this command from the `starter_kit/` directory:

```powershell
python -m unittest discover -s eval_pipeline\tests -p "test_*.py" -v
```

The tests can also be run from the repository root by adjusting the discovery
path and ensuring `starter_kit/` is on `PYTHONPATH`.

## Test modules

| File | Coverage |
| --- | --- |
| `test_metrics.py` | Golden-value tests for nDCG, precision, recall, MAP, MRR, temporal coverage, and deterministic behavior. |
| `test_validation.py` | TREC run validation, including malformed ranks and scores, duplicate ranks/documents, missing topics, unknown IDs, and warnings. |
| `test_report.py` | End-to-end preprocessing and scoring, missing-topic handling, unknown-document filtering, reranking, NaN handling, report serialization, and agreement with the starter-kit scorer. |
| `test_taxonomy.py` | Derived query-type classification for temporal scope, period count, and step class. |
| `dummy_data.py` | Seeded synthetic data and helpers used by pipeline-level tests; it does not contain tests itself. |

## Test data and expectations

The fixtures are generated in memory or in temporary files. This keeps the
suite deterministic and prevents changes to released RETECO data. Seeded tests
use fixed random seeds and verify hand-computed metric values where practical.

The suite checks that:

- `IDCG = 0` produces an nDCG value of `0.0`, not `NaN`.
- Invalid numeric values and structural errors are reported as validation
  errors rather than raising unexpected exceptions.
- Duplicate document IDs keep the first occurrence and produce a warning.
- Unknown document IDs are discarded with a warning.
- Missing qrels topics remain in scoring and receive zero scores.
- Topics are reranked by score before metrics are calculated.
- Reports preserve the expected `reteco-eval-report/1` schema and include all
  topics, including missing ones.
