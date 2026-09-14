# RETECO Walkthrough — Sub-track 1a (Temporal Retrieval)

Step-by-step guide for setting up the environment, getting the data, and running
the baseline **on Windows (PowerShell, CPU-only)**. Everything below was run and
verified on this machine.

---

## 1. Repository layout (what/where)

```text
RETECO/
├── starter_kit/               # baselines, scorer, format checker
│   ├── bm25.py                # pure-Python BM25 (no deps)
│   ├── bm25_baseline.py       # produces a TREC run file (per domain)
│   ├── run_release.py         # driver: baseline + scoring over the release layout
│   ├── scorer.py              # official scorer (nDCG@10 + temporal diagnostics)
│   ├── ir_metrics.py          # metric implementations (pure Python)
│   ├── official_baseline.py   # exact upstream reproduction (needs JDK + pyserini)
│   ├── format_checker.py      # validates TREC submissions
│   ├── requirements.txt
│   └── walkthrough.md         # this file
├── reteco_data/               # the download (created in step 3)
├── docs/                      # website source
└── README.md
```

---

## 2. Virtual environment

Created a Python venv **inside `starter_kit/`**:

```powershell
# from C:\CODE_SOMETHING\Reteco\RETECO\starter_kit
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
```

### Install packages

```powershell
# core deps (download + scoring + BM25 model)
.\.venv\Scripts\python.exe -m pip install pyarrow huggingface_hub tqdm gensim --quiet
```

> **`pytrec_eval` cannot be installed on this machine** (no Windows wheel / needs a
> C++ compiler). It is only required to *byte-reproduce* the published baseline.
> For normal development you do NOT need it — `scorer.py` + `ir_metrics.py`
> implement the same nDCG@10 in pure Python. The official ranking metric is
> **nDCG@10**, computed with `pytrec_eval` on the organisers' side; locally the
> pure-Python scorer agrees.

> **`pyserini`** (used by `official_baseline.py`) additionally needs a JDK. Skip it
> unless you want the exact-upstream reproduction.

---

## 3. Mandatory Windows fix: UTF-8

The starter kit scripts open files **without** an explicit encoding, and Windows
defaults to `cp1252`, which crashes on the data (contains non-ASCII).

**Always set this before running any script** in this project:

```powershell
$env:PYTHONUTF8=1
```

If you don't, you get:

```text
UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d ...
```

---

## 4. Download the dataset (~4.5 GB)

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\hf.exe download DataScience-UIBK/RETECO-SemEval2027 `
    --repo-type dataset --local-dir ..\reteco_data
```

Notes:
- The `hf.exe` CLI lives at `.venv\Scripts\hf.exe` (a venv script); the
  `huggingface_hub.commands` module path does **not** exist in v1.30.
- Unauthenticated downloads hit **HTTP 429** rate limits and retry automatically
  (they succeed). Let it run.
- Result: `reteco_data/track1_tempo/{domain}/...` (13 domains) and
  `reteco_data/track2_recor/{domain}/...` (11 domains). Each domain has
  `documents.jsonl`, `examples_{train,dev}.jsonl`, `steps_{train,dev}.jsonl`,
  `qrels_{train,dev}.txt`, `qrels_steps_{train,dev}.txt`, etc.

---

## 5. Quick sanity check on the bundled sample data (tiny, no download)

Contains only 5 TEMPO examples / 26 gold passages — **format check only**, not a
real benchmark:

```powershell
$env:PYTHONUTF8=1
New-Item -ItemType Directory -Force -Path ".\runs_sample" | Out-Null
.\.venv\Scripts\python.exe bm25_baseline.py --track 1 `
    --corpus "..\..\RETECO_curated_sample_data\track1_tempo\documents.jsonl" `
    --queries "..\..\RETECO_curated_sample_data\track1_tempo\examples.jsonl" `
    --out ".\runs_sample\1a_sample.txt"

.\.venv\Scripts\python.exe scorer.py --track 1 `
    --run ".\runs_sample\1a_sample.txt" `
    --qrels "..\..\RETECO_curated_sample_data\track1_tempo\qrels.txt" `
    --steps "..\..\RETECO_curated_sample_data\track1_tempo\steps.jsonl"
```

You should see the official metric line:

```text
nDCG@10  0.7821  <-- OFFICIAL
```

(Don't read into the number — 5 positive-only docs.)

---

## 6. Run the baseline over the real data — one domain first

The pure-Python BM25 is **slow on CPU** over large corpora (a full 13-domain run
did not finish in 1 hour on this machine). Always pick a small domain first, e.g.
`iota` (10k docs) or `cardano`:

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_release.py `
    --data ..\reteco_data --split dev --track1 cardano
```

### DANGER — `--track1` / `--track2` with no values means "zero domains"

`run_release.py` uses `nargs="*"` with **no default**, so:

```powershell
# WRONG — silently skips every domain:
python run_release.py --data ..\reteco_data --split dev --track1

# WRONG — joins all names into one path:
$doms = (Get-ChildItem ..\reteco_data\track1_tempo -Directory).Name -join " "
python run_release.py --data ..\reteco_data --split dev --track1 $doms

# CORRECT — pass each domain name as a separate token:
$doms = Get-ChildItem ..\reteco_data\track1_tempo -Directory | Select-Object -ExpandProperty Name
python run_release.py --data ..\reteco_data --split dev --track1 $doms
```

With no `--track1` at all the script runs **all Track 1 + all Track 2** (very slow
on CPU). To skip Track 2, there is no clean flag — pass `--track2` followed by a
single non-existent name, or just let Track 1 domains be selected.

Output per domain lands in `runs_release/track1/{domain}/1a_dev.txt` (TREC run)
and the score summary in `runs_release/baseline_dev.json`.

---

## 7. Score an existing run file

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe scorer.py --track 1 `
    --run .\runs_release\track1\cardano\1a_dev.txt `
    --qrels ..\reteco_data\track1_tempo\cardano\qrels_dev.txt `
    --steps ..\reteco_data\track1_tempo\cardano\steps_dev.jsonl
```

Output example (per-domain):

```text
# RETECO Track 1 -- 37 topics scored
  nDCG@10        0.1108  <-- OFFICIAL
  TP@10          0.0
  TR@10          0.0
  TC@10          0.0
  NDCG|FC@10     0.0
  num_full_coverage 0
```

Reference per-domain numbers: `starter_kit/BASELINE_RESULTS.md`.

---

## 8. Individual script usage (not through the driver)

### BM25 baseline → TREC run file

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe bm25_baseline.py --track 1 `
    --corpus ..\reteco_data\track1_tempo\cardano\documents.jsonl `
    --queries ..\reteco_data\track1_tempo\cardano\examples_dev.jsonl `
    --out .\my_run_1a.txt --top-k 100
```

Argments: `--track 1|2`, `--level query|step` (1b), `--strategy current|history`
(2a), `--top-k`, `--k1 0.9`, `--b 0.4`.

### Format checker (validate a submission before sending it)

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe format_checker.py --run .\my_run_1a.txt
```

---

## 9. The official baseline (optional, JDK required)

Exact reproduction of the published numbers needs `pyserini` (Lucene analyzer) +
`gensim` + `pytrec_eval` + a JDK. Not usable on this machine yet (no JDK, no
Windows wheel for `pytrec_eval`). Skip unless you want bit-exact reproduction.

---

## 10. Gotchas collected during this setup

| Problem | Fix |
| --- | --- |
| `UnicodeDecodeError: 'charmap' codec` | `$env:PYTHONUTF8=1` before every run |
| `pytrec_eval` fails to pip install | Not needed for dev; official scorer is pure Python |
| `hf ... -m huggingface_hub.commands...` fails | Use `.venv\Scripts\hf.exe download ...` |
| HTTP 429 during download | Normal when unauthenticated; it retries |
| `run_release.py --track1` runs nothing | `nargs="*"`; pass explicit domain names |
| Full 13-domain run hangs on CPU | Pure-Python BM25 is slow; work per-domain on small corpora first |
| `python eval_pipeline\pipeline.py` → `ImportError: attempted relative import` | Use `-m eval_pipeline.pipeline`; run via `python -m` not a direct script |
| `topic '\ufeff...' not present in qrels` | Run file saved with a UTF-8 BOM (e.g. Notepad/PowerShell); readers now strip it, resave using ASCII/UTF-8 no BOM |

---

## 11. Evaluation pipeline (`eval_pipeline/`)

Dependency-free scorer + TREC format checker + `test_results.json` reporter for
Track 1 dev/test. Reproduces the official nDCG@10 exactly.

```powershell
$env:PYTHONUTF8=1

# single domain (run/gold/corpus paths)
.\.venv\Scripts\python.exe -m eval_pipeline.pipeline `
  --run "runs_release\track1\cardano\1a_dev.txt" `
  --qrels "..\reteco_data\track1_tempo\cardano\qrels_dev.txt" `
  --corpus "..\reteco_data\track1_tempo\cardano\documents.jsonl" `
  --steps "..\reteco_data\track1_tempo\cardano\steps_dev.jsonl" `
  --domain cardano --split dev --tag bm25 --outdir results_cardano

# all domains: expects runs-dir files named <domain>_1a_<split>.txt
.\.venv\Scripts\python.exe -m eval_pipeline.pipeline `
  --all-domains --data-dir "..\reteco_data" --runs-dir "runs_all" `
  --split dev --tag bm25 --outdir results_all
```

Reports are per-topic, per-domain and per derived query type
(`period_bucket` / `n_periods` / `temporal_scope` / `step_class`), plus the
extra track metrics `TR@10 TP@10 TC@10 NDCG|FC@10`. Good runs print
`RESULT: VALID nDCG@10: ...` and exit 0; format violations (bad columns,
duplicate ranks, non-finite scores, missing topics) print a validation error
list and exit 1.

Unit tests with seeded dummy data:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s eval_pipeline\tests -p "test_*.py" -v
```

## 12. Where the real headroom is (why we're doing this)

Reference BM25 (dev): **1a = 0.0967**, 1b = 0.1063. Temporal grounding is not a
keyword problem — dense/hybrid retrieval, temporal-aware query expansion, and
LLM reranking are typical ways to push nDCG@10 well above the BM25 floor.

On a **CPU-only** machine, realistic directions:
- Sparse improvements to BM25 (term weighting, query expansion) — cheap, no GPU.
- Dense retrieval with a small model (e.g. MiniLM / bge-small) — feasible on CPU
  for the smallest domain corpora first.
- LLM/API-based reranking of a BM25 candidate list — no local GPU needed.
- Use the step decomposition (1b) to build better 1a queries.