"""UTF-8-safe readers for JSONL / JSON / qrels / TREC run files.

The starter-kit scripts open files without an explicit encoding, which breaks on
Windows (default cp1252). Everything here opens with ``encoding="utf-8-sig"`` so the
pipeline does not depend on ``PYTHONUTF8=1``.
"""
import json
from collections import defaultdict


def iter_jsonl(path):
    """Yield parsed JSON objects from a JSONL file (blank lines skipped)."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def read_qrels(path):
    """Return {qid: set(docids)} with relevance > 0 only."""
    gold = defaultdict(set)
    with open(path, encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed qrels line: {ln!r}")
            qid, _iter, docid, rel = parts[0], parts[1], parts[2], parts[3]
            if int(rel) > 0:
                gold[qid].add(docid)
    return gold


def load_corpus_ids(path, doc_key="id"):
    """Return the set of document ids present in a documents.jsonl corpus."""
    ids = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)[doc_key])
    return ids


def read_run(path):
    """Parse a TREC run file.

    Returns a list of Line objects (dicts): lineno, qid, q0, docid, rank, score,
    tag. Column count is checked here so downstream code can assume 6 columns.
    Malformed lines raise ValueError with the line number.
    """
    lines = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) != 6:
                raise ValueError(
                    f"line {lineno}: expected 6 columns, got {len(cols)}: {line[:80]!r}")
            qid, q0, docid, rank, score, tag = cols
            lines.append({
                "lineno": lineno,
                "qid": qid,
                "q0": q0,
                "docid": docid,
                "rank": rank,
                "score": score,
                "tag": tag,
            })
    return lines


def read_steps(path):
    """Return {qid: {"step_count": int, "periods": [set(gold ids)], "instructions": [str]}}.

    Handles both release layouts:
      nested  steps_*.jsonl : {"id", "query", "steps": [{step_id, step,
                            step_instruction, gold_ids}, ...]}
      flat    curated sample : {"id", "query", "gold_ids": [...]}
    """
    out = {}
    for rec in iter_jsonl(path):
        qid = rec.get("id")
        if qid is None:
            continue
        periods, instructions = [], []
        steps = rec.get("steps")
        if isinstance(steps, list):            # release layout (nested)
            for st in steps:
                if isinstance(st, dict):
                    periods.append(set(st.get("gold_ids") or []))
                    instructions.append(str(st.get("step_instruction") or ""))
        elif rec.get("gold_ids"):              # curated sample (flat, 1 period)
            periods.append(set(rec["gold_ids"]))
        out[qid] = {
            "step_count": len(periods),
            "periods": periods,
            "instructions": instructions,
            "query": rec.get("query") or "",
        }
    return out
