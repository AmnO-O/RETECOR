"""Assemble the unified test_results.json report."""
import datetime as _dt
import json

from . import __version__, SCHEMA
from .analysis import aggregate

OFFICIAL_METRIC = "nDCG@10"


def build_report(*, meta, validation, metric_map, gold_map, per_domain=None,
                 per_query_type=None, query_types=None):
    """Assemble the report dict.

    meta:           {run_file, qrels, corpus, steps, domain, split, tag, k}
    validation:     dict from run_validation.validate()
    metric_map:     {qid: metrics dict} from analysis.score_topics (all topics,
                    including scored-zero missing ones)
    gold_map:       {qid: sorted gold docids} for reference
    per_domain:     {domain: {group_value: summary}} (optional)
    per_query_type: {group_key: {group_value: summary}} (optional)
    query_types:    {qid: taxonomy dict} (optional)
    """
    return {
        "schema": SCHEMA,
        "pipeline_version": __version__,
        "created_utc": meta["created_utc"],
        "meta": {k: meta[k] for k in (
            "run_file", "qrels", "corpus", "steps", "domain", "split", "tag")
            if k in meta},
        "params": {"k": meta["k"]},
        "validation": _validation_summary(validation),
        "official_metric": OFFICIAL_METRIC,
        "metrics_macro": aggregate(metric_map),
        "per_domain": per_domain or {},
        "per_query_type": per_query_type or {},
        "query_types": query_types or {},
        "per_topic": {qid: {"metrics": metric_map[qid],
                            "gold": sorted(gold_map.get(qid, []))}
                      for qid in metric_map},
        "missing_topics": validation.get("missing_topics", []),
    }


def _validation_summary(validation):
    """Version-stable view of validation (lists truncated already upstream)."""
    return {
        "valid": validation["valid"],
        "errors": validation["errors"],
        "warnings": validation["warnings"],
        "missing_topics": validation["missing_topics"],
        "invalid_only_topics": validation["invalid_only_topics"],
        "num_lines": validation["num_lines"],
        "num_topics_in_run": validation["num_topics_in_run"],
    }


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def now_utc_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")