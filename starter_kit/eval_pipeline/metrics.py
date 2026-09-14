"""Retrieval metrics (pure Python), aligned with pytrec_eval semantics.

All functions take ``ranked`` as a list of docids in rank order and ``gold`` as
an iterable/set of relevant docids. Relevance is binary (1/0) as in the RETECO
qrels. These implement the official RETECO metric set:

  * nDCG@k   -- official leaderboard metric
  * Recall@k, Precision@k, MAP@k, MRR -- standard diagnostics
  * TP@k / TR@k / TC@k / nDCG|FC@k -- temporal diagnostics (Track 1)

Conventions (match the upstream/pytrec_eval behaviour):
  * nDCG: gain 1 at top of gain log2(i+1); IDCG from min(len(gold), k) rels.
  * A topic with zero attainable gain (IDCG == 0) scores 0.0 -- never NaN.
  * A topic with no relevant docs contributes 0.0 to recall, MRR and MAP.
  * MAP@k uses map_cut semantics: cutoff applies to AP, P@r still uses r.
  * TC@k returns None when the topic has no defined periods (excluded).
"""
from math import log2


# ---------------------------------------------------------- core metrics ----
def _dcg(rels):
    """DCG of a list of binary relevance grades (rel 1 at rank i+1)."""
    return sum(r / log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked, gold, k=10):
    """Binary nDCG@k. gold: set/iterable of relevant docids."""
    gold = set(gold)
    grades = [1.0 if d in gold else 0.0 for d in ranked[:k]]
    idcg = _dcg([1.0] * min(len(gold), k))
    return (_dcg(grades) / idcg) if idcg > 0 else 0.0


def precision_at_k(ranked, gold, k=10):
    """P@k = |relevant in top-k| / k  (0 when k == 0)."""
    gold = set(gold)
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked[:k] if d in gold) / k


def recall_at_k(ranked, gold, k=10):
    """Recall@k = |relevant in top-k| / |relevant|  (0 when no gold)."""
    gold = set(gold)
    if not gold:
        return 0.0
    return sum(1 for d in ranked[:k] if d in gold) / len(gold)


def reciprocal_rank(ranked, gold):
    """MRR contribution for one topic: 1/rank of first relevant (0 if none)."""
    gold = set(gold)
    for i, d in enumerate(ranked, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranked, gold, k=10):
    """AP@k with map_cut semantics (cutoff on AP, P@r uses r <= min(k, len))."""
    gold = set(gold)
    if not gold:
        return 0.0
    hit = 0
    total = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in gold:
            hit += 1
            total += hit / i
    return total / len(gold)


def map_at_k(ranked_by_topic, gold_by_topic, k=10):
    """Macro-averaged MAP@k over topics."""
    vals = [average_precision_at_k(ranked_by_topic[q], gold_by_topic[q], k)
            for q in gold_by_topic if q in ranked_by_topic]
    return mean(vals)


# ----------------------------------------------- temporal diagnostics (T1) ----
def temporal_relevance_at_k(ranked, gold, k=10):
    """TR@k: fraction of the top-k that is relevant (temporally, per oracle)."""
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in gold) / len(top)


def temporal_precision_at_k(ranked, gold, k=10):
    """TP@k: position-weighted precision, w_i = 1/log2(i+1)."""
    top = ranked[:k]
    num = den = 0.0
    for i, d in enumerate(top):
        w = 1.0 / log2(i + 2)
        den += w
        if d in gold:
            num += w
    return (num / den) if den > 0 else 0.0


def temporal_coverage_at_k(ranked, period_gold, k=10):
    """TC@k: fraction of required periods covered by >=1 doc in the top-k.

    ``period_gold`` is a list of gold-id sets (one per required period).
    Returns None when there are no defined periods (excluded from TC stats).
    """
    if not period_gold:
        return None
    top = set(ranked[:k])
    covered = sum(1 for gs in period_gold if top & set(gs))
    return covered / len(period_gold)


def ndcg_full_coverage(ranked, gold, period_gold, k=10):
    """nDCG@k but only returned when every required period is covered (else None)."""
    tc = temporal_coverage_at_k(ranked, period_gold, k)
    if tc != 1.0:
        return None
    return ndcg_at_k(ranked, gold, k)


# -------------------------------------------------------- aggregation ---------
def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def metrics_for_topic(ranked, gold, period_gold=None, k=10):
    """All official + diagnostic metrics for a single ranked topic.

    Returns a dict of floats (period-gated values may be None when undefined).
    """
    out = {
        "nDCG@%d" % k: ndcg_at_k(ranked, gold, k),
        "Recall@%d" % k: recall_at_k(ranked, gold, k),
        "P@%d" % k: precision_at_k(ranked, gold, k),
        "MAP@%d" % k: average_precision_at_k(ranked, gold, k),
        "MRR": reciprocal_rank(ranked, gold),
        "TR@%d" % k: temporal_relevance_at_k(ranked, gold, k),
        "TP@%d" % k: temporal_precision_at_k(ranked, gold, k),
        "TC@%d" % k: temporal_coverage_at_k(ranked, period_gold, k),
        "NDCG|FC@%d" % k: ndcg_full_coverage(ranked, gold, period_gold, k),
        "num_gold": len(gold),
        "num_retrieved": len(ranked),
    }
    return out