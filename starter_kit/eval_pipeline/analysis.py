"""Score + aggregate a TREC run against qrels.

Handles the official pre-processing rules before any metric is computed:
  * duplicate docids -> keep the first occurrence
  * docids absent from the corpus -> discarded
  * qrels topics absent from the run -> scored as empty list (all metrics 0,
    TC undefined), still counted in the macro averages
  * ranking trust -> ranks are ignored; topics are re-ranked by descending
    score with a deterministic tie-break (docid ascending)
  * nan/inf scores invalidate their topic (kept out of the run, counted)
"""
from collections import defaultdict

from .metrics import metrics_for_topic, mean
from . import run_validation


def preprocess_run(lines, qrels, corpus_ids=None):
    """Return per-topic ranked lists, deduped, filtered, re-ranked by score.

    Returns (ranked, invalid_topics) where ranked is {qid: [docids...]}.
    Invalid topics (nan/inf score present) are excluded and returned separately.
    """
    grouped = defaultdict(list)
    for line in lines:
        if line["qid"] not in qrels:
            continue
        if corpus_ids is not None and line["docid"] not in corpus_ids:
            continue
        grouped[line["qid"]].append((line["score"], line["docid"]))

    ranked, invalid = {}, set()
    for qid, rows in grouped.items():
        # any non-finite score poisons the whole topic: the run file is broken
        try:
            parsed = [(float(s), d) for s, d in rows]
        except ValueError:
            invalid.add(qid)
            continue
        if not all(_finite(s) for s, _ in parsed):
            invalid.add(qid)
            continue
        seen, ordered = set(), []
        for s, d in sorted(parsed, key=lambda t: (-t[0], t[1])):
            if d not in seen:
                seen.add(d)
                ordered.append(d)
        ranked[qid] = ordered
    return ranked, invalid


def _finite(v):
    import math
    return math.isfinite(v)


def score_topics(ranked, qrels, period_gold=None, k=10):
    """Compute all metrics per topic.

    ``period_gold`` maps qid -> list of gold-id sets (required periods) for the
    temporal diagnostics. Missing topics (in qrels, not ranked) are scored with
    an empty ranking.
    """
    out = {}
    for qid, gold in qrels.items():
        docs = ranked.get(qid, [])
        periods = (period_gold or {}).get(qid, {}).get("periods") if period_gold else None
        out[qid] = metrics_for_topic(docs, gold, period_gold=periods, k=k)
    return out


# ------------------------------------------------------------ aggregation ----
_METRIC_KEYS = None


def _metric_names(metrics):
    """Float metric columns to average (drops bookkeeping keys)."""
    skip = {"num_gold", "num_retrieved"}
    return [k for k in ("nDCG@10", "Recall@10", "P@10", "MAP@10", "MRR",
                        "TR@10", "TP@10", "TC@10", "NDCG|FC@10")
            if f"{k}" in metrics or k in metrics]


def aggregate(metric_map):
    """Average per-topic metrics into a single summary dict.

    Averaging is per-topic then across topics (the diagnostic macro). TC@10 and
    NDCG|FC@10 exclude topics where the value is None (no periods / no full
    coverage), matching the official diagnostic definition.
    """
    if not metric_map:
        return {}
    n = len(metric_map)
    keys = [k for k, v in next(iter(metric_map.values())).items()
            if isinstance(v, (int, float)) and k not in ("num_gold", "num_retrieved")]
    keys = [k for k in ("nDCG@10", "Recall@10", "P@10", "MAP@10", "MRR",
                        "TR@10", "TP@10", "TC@10", "NDCG|FC@10")
            if k in keys]
    out = {"num_topics": n}
    for key in keys:
        out[key] = round(mean(m.get(key) for m in metric_map.values()), 4)
    return out