# data_chunking

Prepare a RETECO Track 1 domain corpus for retrieval: **clean → dedupe → chunk**

The scoring unit in Track 1a is the *document id* in `documents.jsonl` (qrels
reference ids like `iota/470_68d2fff7515d039b_93.txt`), so this tool never
re-ids atomic docs. Short docs pass through unchanged; only the long tail is
split into overlapping windows that map back to their parent via `chunk_map.tsv`.

```
raw documents.jsonl
   │  clean_text()        strip bbcode / URLs / HTML / emoji, collapse ws
   ▼
clean text
   │  content sha1 dedupe keep-first  (qrels gold ids exempt — see below)
   ▼
unique docs
   │  truncate_content()  cap pathological giants (>20k words)
   ▼
capped docs
   │  chunk_doc()         w=300, overlap=50, min_words=1500
   ▼
{id, parent_id, content} records  +  chunk_map.tsv
```

## Quick start

```powershell
# from starter_kit/
$env:PYTHONUTF8=1

python -m data_chunking.chunker iota --data-dir ..\reteco_data
python -m data_chunking.chunker cardano --data-dir ..\reteco_data --sample 5
```

Outputs land in `data_chunking\out\<domain>\`:

| file | contents |
| --- | --- |
| `corpus_clean.jsonl` | `{"id","parent_id","content"}` records, ready to index |
| `chunk_map.tsv` | `chunk_id<TAB>parent_id` for every long-tail window |
| `stats.json` | counts: corpus → records, dropped, chunk windows, gold coverage |

`stats.json` also reports `gold_total` / `gold_after_clean` / `gold_missing` so
you can prove no gold document was lost by cleaning or dedupe.

## Options

| flag | default | meaning |
| --- | --- | --- |
| `--window` | 300 | words per chunk window (long-tail only) |
| `--overlap` | 50 | overlapping words between consecutive windows |
| `--chunk-min-words` | 1500 | docs above this split; below → single atomic record |
| `--cap-words` | 20000 | hard truncation for pathological giant docs |
| `--split` | dev | which qrels to load (`dev` / `train`) |
| `--qrels <file>` | auto | explicit qrels; its ids are protected from dedupe |
| `--no-dedupe` | off | disable content-hash deduping |
| `--sample N` | 3 | print N random records to eyeball quality (0 = off) |

## Why the defaults

Measured on the released data (`reteco_data/track1_tempo/*/report_data.md`):

- **Corpus docs are passage-scale.** Cardano median ~130 words, IOTA ~152
  words, almost all single-paragraph. Chunking normal docs only adds context
  bleed, so they stay atomic.
- **Gold answers are ~300 words** (median), ~2.4× the corpus median — a 300-word
  window is comfortably bigger than any gold doc, so windowing doesn't truncate
  answers.
- **The tail is extreme:** Cardano p99 = 18.7k chars, IOTA p99 = 200k chars,
  with 5 MB / 2.5 MB stitched giants. `min_words=1500` catches the tail;
  `cap_words=20000` bounds the giants so they stop dominating BM25 tf and search
  latency.
- **Exact-content duplication** is huge in some domains (Cardano ~46% of the
  corpus, IOTA ~7%). Dedupe keeps the first occurrence per sha1.

## The qrels-protection rule

Some domains contain **gold docs that are exact content copies of one another**
(e.g. Cardano `4052572a_6040` == `6045` == `6046`). A naive content-dedupe would
drop valid gold targets. Every id appearing in the qrels is therefore exempt
from dedupe, and unprotected copies of the same content are dropped instead.

## Indexing a windowed corpus

The retriever searches `corpus_clean.jsonl` (chunk ids) but must return
**parent ids** in its run file, because the official scorer (`nDCG@10`) is keyed
on document ids. Resolve chunk → parent with `chunk_map.tsv` when writing the
run:

```
chunk_id           -> parent_id
iota/longer.txt|1  -> iota/longer.txt
iota/longer.txt|2  -> iota/longer.txt
```

Only `parent_id`s appear in a valid submission. Chunks from the same parent can
otherwise both be ranked in the top-k; collapse to the parent before scoring or
accept the duplicate-parent entries for the doc-level aggregation.

## Tests

```powershell
python -m unittest discover -s data_chunking\tests -p "test_*.py" -v
```

Coverage: domain-dir discovery, gold-protection through dedupe, duplicate
dropping when unprotected, window chunk-map correctness, stats file, garbage
(cleans-to-empty) dropping.

## Relationship to eval_pipeline

`data_chunking` is a thin domain-level CLI over the reusable primitives in
`eval_pipeline/corpus_tools.py` (`clean_text`, `chunk_doc`,
`truncate_content`, `run`). Scoring/evaluation lives in `eval_pipeline`;
chunking + retrieval-side corpus prep lives here.