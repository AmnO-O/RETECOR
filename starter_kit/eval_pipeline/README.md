# eval_pipeline

Dependency-free evaluation pipeline for RETECO sub-track 1a (Temporal Retrieval):
TREC run validation, official scoring, per-domain / per-query-type analysis, and a
unified `test_results.json` report. Reproduces the official nDCG@10 metric exactly
(verified against the starter-kit scorer on cardano dev: 0.1108).

- Pure standard library — no PyPI dependencies, no GPU, no JDK.
- UTF-8 safe on Windows (`utf-8-sig` readers; no reliance on `PYTHONUTF8=1`).

## Install / requirements

Python 3.9+ (developed on 3.12). No `pip install` needed.

## Usage

```powershell
# single domain
python -m eval_pipeline.pipeline `
  --run "..\runs\cardano_1a_dev.txt" `
  --qrels "..\data\track1_tempo\cardano\qrels_dev.txt" `
  --corpus "..\data\track1_tempo\cardano\documents.jsonl" `
  --steps "..\data\track1_tempo\cardano\steps_dev.jsonl" `
  --domain cardano --split dev --tag myrun --outdir results

# all domains at once (runs-dir files named <domain>_1a_<split>.txt)
python -m eval_pipeline.pipeline `
  --all-domains --data-dir "..\data" --runs-dir "..\runs" `
  --split dev --tag myrun --outdir results
```

Good runs print `RESULT: VALID  nDCG@10: ...` and exit 0.
Format violations print the error list and exit 1.

Run from the `starter_kit/` directory (or anywhere on `sys.path`); it must be
invoked as a module, not as a script file (`python -m eval_pipeline.pipeline`).

## Tests

```powershell
python -m unittest discover -s eval_pipeline\tests -p "test_*.py" -v
```

Seeded dummy data covers golden values (hand-computed nDCG/AP/MRR), all validation
edge cases (column counts, duplicate ranks, non-finite scores, missing topics,
unknown qids/docids), the derived taxonomy, the report schema, and parity with the
starter-kit `ir_metrics.py`.

## Scoring conventions (Track 1 official rules)

- Runs are TREC format: `qid Q0 docid rank score tag` — 6 columns, one line per
  (topic, doc) pair, ranked in strictly increasing rank order.
- Unknown doc ids (not in the domain corpus) are discarded with a warning.
- Duplicate (topic, doc) pairs keep the first occurrence with a warning.
- Topics present in the qrels but absent from the run score 0.0 and are included
  in the macro average.
- Unknown qids in the run are skipped with a warning.
- Non-finite scores, wrong column counts, and duplicate ranks are hard errors.
- Official macro-average is over domains.

## Metrics

Primary (official): **nDCG@10**. Ranked by score descending.

Additional reported metrics:

| Metric | Definition |
| --- | --- |
| `P@k`, `Recall@k` | Precision / recall at rank k |
| `MAP@k` | Mean average precision, truncated at top-k |
| `MRR` | Mean reciprocal rank of the first relevant result |
| `nDCG@k` | Official primary metric for sub-track 1a |
| `TR@k`, `TP@k`, `TC@k` | Temporal recall / precision / coverage at rank k |
| `NDCG\|FC@k` | nDCG conditioned on full temporal coverage (>1 gold document required) |

`IDCG = 0` topics are scored `0.0`, never skipped.

## Report schema

`test_results.json` (schema `reteco-eval-report/1`):

- `meta` — run/qrels/corpus/steps paths, domain, split, tag, created time.
- `params` — evaluation parameters (`k`).
- `validation` — `valid`, `errors[]`, `warnings[]`, `missing_topics[]`,
  `invalid_only_topics[]`, line/topic counts.
- `metrics_macro` / `per_domain` — macro metrics over topics (single domain) or
  over domains (all-domains mode).
- `per_query_type` — metrics grouped by derived `period_bucket`, `n_periods`,
  `temporal_scope`, `step_class` (aggregated over topics).
- `query_types` — derived classification per topic id.
- `per_topic` — per-topic metrics, `num_gold`, `num_retrieved`, and gold doc ids.

All `null` / missing values serialize to JSON `null` (no NaN fields).

## Derived query-type taxonomy

Rule-based, auditable classes derived from the public `steps_*.jsonl` and query
text (the released data has no explicit labels):

- `period_bucket` — `single` / `multi` (number of required time periods).
- `n_periods` — step count (flat curated-sample steps degrade to 1).
- `temporal_scope` — finest explicit time unit in the query text:
  day > month > year > decade > century, else `point`.
- `step_class` — keyword class of the step instruction:
  `period` / `event` / `trend` / `compare` / `other`.

## Files

| File | Purpose |
| --- | --- |
| `dataio.py` | UTF-8/BOM-safe readers for runs, qrels, corpus, steps |
| `run_validation.py` | TREC format checker + edge-case diagnostics |
| `metrics.py` | All ranking + temporal metrics |
| `analysis.py` | Preprocessing (dedupe, discard, rerank) + scoring + aggregation |
| `taxonomy.py` | Derived query-type classification rules |
| `report.py` | `test_results.json` builder |
| `pipeline.py` | CLI entry point (single-domain and `--all-domains`) |
| `corpus_tools.py` | Corpus cleaning / dedupe / parent-mapped chunking (see below) |

### Corpus prep (`corpus_tools.py`)

Clean, deduplicate, and window a raw `documents.jsonl` before indexing. Short
docs pass through untouched (atomic scoring unit), long tail is split into
overlapping windows mapped back to the parent id via a chunk map.

```powershell
python -m eval_pipeline.corpus_tools ..\data\track1_tempo\cardano\documents.jsonl `
  --qrels ..\data\track1_tempo\cardano\qrels_dev.txt `
  --out corpus_clean.jsonl --map chunk_map.tsv

# options: --chunk-min-words 1500 --window 300 --overlap 50 --cap-words 20000
```

`--qrels` protects gold doc ids from content-dedupe (cardano has gold docs that
are exact copies of each other). Result: 87,201 → 56,611 records, 0/45 gold docs
lost on cardano dev.