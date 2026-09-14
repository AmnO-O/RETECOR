# RETECO Sub-track 1a — Baseline Analysis & Proposed Architecture

## 1. Task recap

**Sub-track 1a (Temporal Retrieval):** given a temporal information need and a
domain corpus, return a ranked list of document IDs. The official metric is
**nDCG@10** (binary relevance, macro-averaged over domains). A passage is "good"
only if it is topically relevant **and** temporally aligned with the query's time
periods — a passage that *mentions* the topic in an unrelated year does not count.

**Inputs per domain (train/dev):**
- `documents.jsonl` — `{id, content}`
- `examples_{split}.jsonl` — `{id, query, gold_ids, gold_answers}`
- `qrels_{split}.txt` — TREC qrels (topic id = query `id`, e.g. `124973_5`)
- `steps_{split}.jsonl` — the same queries decomposed into steps (public on
  train/dev; useful for analysis, not given for 1a on the hidden test)

---

## 2. Baseline architecture (what the starter kit does)

### 2.1 The pipeline

```text
documents.jsonl ──► tokenize ──────────────────────────────────┐
                                                                │
examples.jsonl ──► query text (verbatim) ──► tokenize ──► BM25 score every doc ──► top-100 ──► TREC run ──► nDCG@10
```

Four moving parts, all from `bm25.py` / `bm25_baseline.py` /
`official_baseline.py`:

| Component | Implementation |
| --- | --- |
| **Tokenizer** | lowercase + `[A-Za-z0-9]+` regex. No stemming, no stopword removal, no numbers handling. (`bm25.py:12`) |
| **Scorer** | Okapi BM25: `Σ idf(w) · f·(k1+1) / (f + k1·(1 − b + b·dl/avgdl))` with `k1=0.9, b=0.4`, BM25+ non-negative idf `log(1 + (N−n+0.5)/(n+0.5))`. (`bm25.py:35-46`) |
| **Retrieval** | Brute-force: score **every document** in the corpus against the query, return top-100 (starter) / top-1000 (official). (`bm25.py:51`) |
| **Query construction** | The raw `query` field, verbatim. No expansion, no time extraction. (`bm25_baseline.py:49`) |

The **official baseline** (`official_baseline.py`) is the identical retriever
reimplemented with the upstream stack for bit-exact reproducibility: Lucene
analyzer via `pyserini`, gensim `LuceneBM25Model(k1=0.9, b=0.4)`, top-1000, scored
with `pytrec_eval` (`ndcg_cut_10`).

### 2.2 Reference numbers (dev)

| Sub-track | train | dev |
| --- | ---: | ---: |
| 1a Temporal retrieval | 0.0879 | 0.0967 |
| 1b Step-wise retrieval | 0.0852 | 0.1063 |
| 2a (turn only / +history) | 0.1837 / 0.4539 | 0.1827 / 0.4379 |

Per-domain (measured locally, dev): cardano 0.1108, genealogy 0.1258,
economics 0.0385, bitcoin 0.0126. (Numbers in `starter_kit/runs_release/` and
`BASELINE_RESULTS.md`.)

### 2.3 Why the baseline is weak (this is the point of the task)

1. **No temporal understanding.** `"Why did the transaction count jump in
   April–May 2023?"` is scored against documents that must discuss that *window*.
   BM25 matches surface terms (`transaction`, `block`) and ignores "when".
2. **Query-vocabulary mismatch.** Reasoning-heavy temporal queries rarely share
   the exact wording of the relevant evidence. Lexical matching has no
   *semantic* bridge.
3. **No period coverage objective.** 1a is best satisfied by retrieving at least
   one good doc for **every** required period — the baseline has no notion of
   "period" at all. Diagnostic `TC@10 = 0.0` on our local runs says it covers
   almost no required periods.
4. **Crude tokenizer.** No stemming/casing/number normalization fragments the
   representation of years and canonical names (`Apr 2023` vs `april 2023`).

Conclusion: the baseline is a **topical** ranker; 1a is a **topical + temporal**
task. ~0.09 is the "free" floor, and the headroom is large.

---

## 3. Proposed architecture ("the ideal system")

A staged pipeline that keeps every stage measurable and lets us ship CPU-only.

```text
                    ┌───────────────────────────┐
  query ───────────►│ Stage 1 · Query analysis  │──► temporal constraints T_q
                    │  (temporal parse + steps) │        topic/manifesto query m_q
                    └───────────┬───────────────┘
                                │
                                ▼
                    ┌───────────────────────────┐
                    │ Stage 2 · Candidate recall│◄──── documents.jsonl (indexed)
                    │  BM25 (improved)          │
                    │  + dense (small bi-enc)   │
                    │  + RRF fusion             │────► top-k candidates per query
                    └───────────┬───────────────┘
                                │
                                ▼
                    ┌───────────────────────────┐
                    │ Stage 3 · Temporal rerank │
                    │  cross-encoder / LLM      │◄──── T_q (time alignment check)
                    │  "topical AND in-period?" │
                    └───────────┬───────────────┘
                                │
                                ▼
                    ┌───────────────────────────┐
                    │ Stage 4 · Coverage packing│─► final ranking (all periods ≥1 doc)
                    └───────────┬───────────────┘
                                │
                                ▼
                        TREC run ──► scorer.py ──► nDCG@10, TC@10, TP@10
```

### Stage 1 — Query analysis (the stage that makes the difference)

Goal: turn a free-text temporal query into `(temporal constraints T_q, retrieval query m_q)`.

- **Temporal constraint extraction**
  - Rule-based (sentence transformers on spans + date regexes + temporal-commons
    gazetteers) to extract: explicit windows (`April–May 2023`), point events,
    trend predicates (`jump`, `decline`), and period boundaries.
  - Normalize to a canonical representation: `(entity, property, window)` e.g.
    `(bitcoin_transactions_per_block, count, 2023-04→2023-05)`.
  - Supervision source (public on train/dev): `steps_*.jsonl` gives each query's
    decomposition and per-step gold periods — ideal training signal for both the
    parser and (later) the reranker.
- **Retrieval query rewrite**: `m_q` = original query + normalized time anchors +
  domain glossary terms. Yes: on 1a, nothing stops us from *using* the public
  step files during development to train/validate this stage; hidden test queries
  are self-contained, so the extractor must be robust without them.

### Stage 2 — Candidate recall (maximize coverage, accept noise)

We want high *recall across periods*, then precision later. Options, ordered by
CPU-friendliness:

1. **Improved BM25** (the core, CPU-free):
   - Stemming + stopword list + dedicated year/number normalization so 2023 ≠
     20230 and `Apr` = `april`.
   - Query = backoff combination of `m_q` and per-period sub-queries.
2. **Dense bi-encoder**, only if time permits: a small model (bge-small /
   MiniLM, ~100MB) over the *candidate set* of one domain at a time. On CPU this
   is feasible for small domains (iota 10k docs, law 43k) and prohibitive for
   history (356k). Dense pays off mainly on vocabulary-mismatch queries.
3. **Fusion**: Reciprocal Rank Fusion (RRF) over BM25 + dense lists — cheap,
   robust, no tuning.

Each query keeps `top-K` (K ≈ 200–500) candidates **and** the per-dimension
sub-scores that stage 3 and 4 need.

### Stage 3 — Temporal reranking (the precision stage)

Re-rank stage-2 candidates with a *temporal-aware* scorer, not just textbook
similarity:

- **Temporal-aware cross-encoder** (CPU, fine): per-query, fine-tune or few-shot
  a small cross-encoder on positive/negative pairs derived from the public
  qrels. Signal: `"does this passage discuss {topic} during {window}?"`.
  (Cross-encoders are relevance-only; temporal checks need the *grid* passed in.)
- **LLM reranker** (no GPU needed — API): rerank top-200 with a prompt that
  receives the query, `T_q` and the candidate; emit a binary/aligned score and
  *which period(s) the passage satisfies*. This directly feeds stage 4. This is
  also the natural source for `nDCG|FC`: passages labeled per-period let us
  cover every required period deliberately.
- Fallback/pre-check: keep everything the baseline likes (BM25) and only
  reorder, never prune aggressively at this stage.

### Stage 4 — Coverage packing (the RETECO-specific trick)

Since the official metric is nDCG@10 and the diagnostics measure period coverage:

- Maintain `T_q` = list of required periods, each with a ranked list of
  per-period candidates from stage 3.
- Build the final top-10 by interleaving: pick the best-scoring doc for each
  uncovered period first, then fill remaining slots by global score. This
  directly targets `TC@10 → 1` and `nDCG|FC@10`.
- If no temporal labels are available (cold start), fall back to pure score
  ordering; the packing mode is a graded on/off we validate on dev.

### Stage 5 — Evaluation & iteration loop

- Local scores via `scorer.py` on every run: `nDCG@10` (official), plus `TP@10`,
  `TR@10`, `TC@10`, `NDCG|FC@10` to see **where** we win/lose.
- Iterate on **train**, and only measure on **dev** (never tune on dev).
  Expected value ladder:
  1. Improved tokenizer + queries on BM25 (target: dev ≥ 0.15).
  2. + query rewrite/time anchors (target: ≥ 0.20).
  3. + LLM/cross-encoder rerank (target: 0.25–0.35).
  4. + coverage packing (target: TC@10 →1, nDCG higher on fully-covered topics).
- These 0.09 → 0.3 targets are directional, not promises — everything is
  validated empirically.

---

## 4. Effort / risk table

| Stage | CPU cost | Est. gain | Risk |
| --- | --- | --- | --- |
| 1 · Query analysis | low | high | parser quality; hidden queries differ |
| 2 · Improved BM25 | low | medium | saturates quickly |
| 2 · Dense (small domains) | medium | medium | slow on large domains |
| 3 · LLM rerank (API) | n/a (no GPU) | high | API cost; rate limits; judge consistency |
| 3 · Cross-encoder | medium/low | medium | needs qrels-based training pairs |
| 4 · Coverage packing | low | medium | needs per-period labels |

**CPU-only constraints we bake in:** dense retrieval only on small domains; the
whole recall+rerank path for large domains (history, genealogy, politics) uses
BM25 + API reranking; every expensive component is per-domain-cached like the
official baseline's `results.json`.

---

## 5. Immediate next steps (in order)

1. Re-run baseline for one small domain and confirm numbers (done: cardano 0.11).
2. Stage 1 v0: temporal-normalization + query-rewrite module, eval offline by
   comparing `TC@10` on dev. **No training needed.**
3. Stage 2 v1: improved tokenizer + stemming + per-period sub-queries; measure
   1a on cardano vs 0.1108.
4. Stage 3 v1: prompt-based LLM rerank over stage-2 candidates on cardano,
   measure delta and cost-per-topic before scaling to larger domains.
5. If signal is real, add coverage packing (stage 4) and scale to all domains
   per-domain with caching.

Deliverables are a single config-driven pipeline producing TREC run files
(`qid Q0 docid rank score tag`), validated by `format_checker.py` and `scorer.py`,
CPU-only, and reproducible from the venv documented in `walkthrough.md`.