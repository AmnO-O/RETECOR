"""TREC run format validation.

Rules enforced (contract of the competition platform checker):
  * 6 whitespace-separated columns per line
  * rank is a positive integer, unique within a topic
  * score is a finite float, non-increasing with rank (warning)
  * duplicate docids within a topic -> warning, first occurrence kept on scoring
  * qrels topics absent from the run -> `missing_topics` (scored 0)
  * run lines whose qid is not in qrels -> warning
  * run lines whose docid is not in the corpus -> warning (discarded on scoring)
  * nan/inf scores -> error

``validate()`` returns a dict, never raises on malformed data, and always
reports every issue found (not just the first).
"""
import math
from collections import defaultdict


def _parse_rank(s, lineno, errors):
    try:
        r = int(s)
    except ValueError:
        errors.append(f"line {lineno}: rank must be a integer, got {s!r}")
        return None
    if r <= 0:
        errors.append(f"line {lineno}: rank must be positive, got {r}")
        return None
    return r


def _parse_score(s, lineno, errors):
    try:
        v = float(s)
    except ValueError:
        errors.append(f"line {lineno}: score must be a float, got {s!r}")
        return None
    if math.isnan(v) or math.isinf(v):
        errors.append(f"line {lineno}: score must be finite, got {s!r}")
        return None
    return v


def validate(lines, qrels=None, corpus_ids=None):
    """Validate parsed run lines (from dataio.read_run).

    Args:
        lines:      list of dicts from dataio.read_run (raw, 6 cols guaranteed).
        qrels:      optional dict qid -> set(docids) for cross-checks.
        corpus_ids: optional set of valid docids.

    Returns dict:
        valid, errors: [str], warnings: [str], missing_topics: [qid],
        topics_in_run: {qid: [docids]}  (docids in order, deduped to first use),
        nonfinite_qids: set of topic ids containing a nan/inf score.
    """
    errors, warnings = [], []

    per_topic_ranks = defaultdict(set)
    per_topic_last_score = defaultdict(float)
    per_topic_seen_docs = defaultdict(set)
    topics_in_run = {}
    seen_topics = []
    nonfinite_qids = set()

    valid_topics = set(qrels) if qrels else None

    for line in lines:
        lineno, qid, q0, docid = line["lineno"], line["qid"], line["q0"], line["docid"]

        if q0 != "Q0":
            warnings.append(f"line {lineno}: column 2 is {q0!r}, expected 'Q0'")

        rank = _parse_rank(line["rank"], lineno, errors)
        score = _parse_score(line["score"], lineno, errors)
        if score is None:
            nonfinite_qids.add(qid)

        # topic-in-run tracking (before unknown-qid filter so warnings still flag)
        if qid not in per_topic_seen_docs:
            seen_topics.append(qid)

        if valid_topics is not None and qid not in valid_topics:
            warnings.append(f"line {lineno}: topic {qid!r} not present in qrels")
            continue

        if rank is not None:
            if rank in per_topic_ranks[qid]:
                errors.append(f"line {lineno}: duplicate rank {rank} in topic {qid!r}")
            per_topic_ranks[qid].add(rank)

        if docid in per_topic_seen_docs[qid]:
            warnings.append(
                f"line {lineno}: duplicate docid {docid!r} in topic {qid!r} "
                f"-- later occurrences ignored on scoring")
        else:
            per_topic_seen_docs[qid].add(docid)
            topics_in_run.setdefault(qid, []).append(docid)

        if corpus_ids is not None and docid not in corpus_ids:
            warnings.append(f"line {lineno}: docid {docid!r} not in corpus -- "
                            f"discarded on scoring")

        if score is not None and qid in per_topic_last_score:
            if score > per_topic_last_score[qid] + 1e-9:
                warnings.append(f"line {lineno}: score {score} increases with rank "
                                f"in topic {qid!r} (scores must be non-increasing)")
        if score is not None:
            per_topic_last_score[qid] = score

    missing_topics = []
    if qrels is not None:
        missing_topics = sorted(set(qrels) - set(per_topic_seen_docs))

    # topics that appear in the run but only via invalid (nonfinite) lines:
    invalid_only = sorted(q for q in seen_topics
                          if q in nonfinite_qids and q not in per_topic_seen_docs)

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "missing_topics": missing_topics,
        "invalid_only_topics": invalid_only,
        "topics_in_run": topics_in_run,
        "nonfinite_qids": sorted(nonfinite_qids),
        "num_lines": len(lines),
        "num_topics_in_run": len(per_topic_seen_docs),
    }