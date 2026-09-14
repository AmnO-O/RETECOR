"""Seeded dummy-data generator for tests (no trained model, no net access)."""
import random


def line(qid, doc, rank, score):
    return " ".join([qid, "Q0", doc, str(rank), str(score), "dummy_run"])


def make_qrels(rng, topics, docs):
    """-> {qid: set(relevant docids)}."""
    return {qid: set(rng.sample(list(docs), rng.randint(1, 3))) for qid in topics}


def make_run_lines(rng, topics, docs, dedupe_topic=None, bad_qid=None,
                   nan_topic=None, k=12):
    """TREC run lines with optional injected edge cases.

    Returns (lines, expected_ranked) where expected_ranked maps qid to the doc
    list the pipeline SHOULD score after dedupe/nan/missing handling.
    """
    lines = []
    scores_used = {}
    expected = {}
    for qid in topics:
        if qid == bad_qid:
            expected[qid] = []
            continue
        ordered, prev = [], None
        for i in range(k):
            doc = rng.choice(list(docs))
            score = random_score(rng, prev)
            prev = score
            s = line(qid, doc, i + 1, score)
            if qid == nan_topic and i == 0:
                lines.append(line(qid, doc, 1, "nan"))
            else:
                lines.append(s)
                ordered.append(doc)
        if qid == dedupe_topic and ordered:
            dup = ordered[-1]
            lines.append(line(qid, dup, k + 1, 0.0))
            # pipeline keeps first occurrence; the dup line is ident anyway
        expected[qid] = ordered
    return lines, expected


def random_score(rng, prev):
    if prev is None:
        return round(rng.random() * 100, 4)
    return round(max(0.0, prev - rng.random() * 10), 4)


def make_steps(rng, topics, docs, path, query="When did events X happen?"):
    """Write steps.jsonl in the nested release layout; return qid->type map.

    Even topics get 2 periods (multi), odd topics 1 (single), with a
    "during"/"compare" step_instruction and a "2023"/"in April 2023" query so
    the taxonomy has something to classify.
    """
    import json
    rows = []
    types = {}
    for i, qid in enumerate(topics):
        n = 2 if i % 2 == 0 else 1
        q = "what happened in April 2023 compared to 2022?" if i % 2 == 0 \
            else "when did the event happen in the 1990s?"
        steps = [{
            "step_id": f"{qid}_step{j}",
            "step": f"Identify period {j + 1}",
            "step_instruction": ("during the time period" if j == 0 else "compare"),
            "gold_ids": list(rng.sample(list(docs), 2)),
        } for j in range(n)]
        rows.append({"id": qid, "query": q, "steps": steps})
        types[qid] = {"n_periods": n,
                      "period_bucket": "multi" if n > 1 else "single",
                      "temporal_scope": "month" if i % 2 == 0 else "decade",
                      "step_class": "period" if n == 2 else "event"}
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return types


def make_dataset(seed=42):
    """-> (topics, docs, qrels, run_lines, expected_ranked)."""
    rng = random.Random(seed)
    topics = [f"t{i}" for i in range(1, 11)]
    docs = [f"d{i}" for i in range(1, 41)]
    qrels = make_qrels(rng, topics, docs)
    lines, expected = make_run_lines(rng, topics, docs,
                                     dedupe_topic="t3", bad_qid="t4",
                                     nan_topic="t7")
    return topics, docs, qrels, lines, expected