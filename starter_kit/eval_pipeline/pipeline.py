#!/usr/bin/env python3
"""RETECO Track 1 evaluation pipeline CLI.

Validate a TREC run, score all official + diagnostic metrics, produce
per-domain and per-query-type analysis, and write test_results.json.

Single-domain usage:
    python eval_pipeline/pipeline.py \
        --run runs/cardano_1a_dev.trec \
        --qrels ../reteco_data/track1_tempo/cardano/qrels_dev.txt \
        --corpus ../reteco_data/track1_tempo/cardano/documents.jsonl \
        --steps ../reteco_data/track1_tempo/cardano/steps_dev.jsonl \
        --domain cardano --split dev --k 10 --outdir results/

All-domains usage (globs track1_tempo/*):
    python eval_pipeline/pipeline.py --all-domains \
        --data-dir ../reteco_data --runs-dir runs_dev \
        --split dev --k 10 --outdir results/
    # expects a run file per domain: runs_dev/<domain>_1a_dev.trec

Exit code: 0 valid run, 1 = invalid or files missing.
"""
import argparse
import os
import sys

from . import __version__
from . import analysis as analysis_mod
from .dataio import load_corpus_ids, read_qrels, read_run, read_steps
from .report import build_report, now_utc_iso, write_report
from .run_validation import validate
from .taxonomy import types_from_steps

TRACK1_ROOT = "track1_tempo"


def _run_single(args):
    if not os.path.isfile(args.run):
        _die(f"run file not found: {args.run}")
    for path, label in ((args.qrels, "qrels"), (args.corpus, "corpus"),
                        (args.steps, "steps")):
        if path and not os.path.isfile(path):
            _die(f"{label} file not found: {path}")

    lines = read_run(args.run)
    qrels = read_qrels(args.qrels)
    corpus_ids = load_corpus_ids(args.corpus) if args.corpus else None
    steps = read_steps(args.steps) if args.steps else {}
    query_types = types_from_steps(steps) if steps else {}

    validation = validate(lines, qrels=qrels, corpus_ids=corpus_ids)
    ranked, invalid = analysis_mod.preprocess_run(lines, qrels, corpus_ids)
    metric_map = analysis_mod.score_topics(ranked, qrels, steps, args.k)
    gold_map = {qid: sorted(gold) for qid, gold in qrels.items()}

    per_domain = {args.domain: analysis_mod.aggregate(metric_map)}

    per_query_type = _group_aggregates(metric_map, query_types)

    meta = {
        "run_file": args.run, "qrels": args.qrels,
        "corpus": args.corpus, "steps": args.steps,
        "domain": args.domain, "split": args.split, "tag": args.tag,
        "k": args.k, "created_utc": now_utc_iso(),
    }
    report = build_report(meta=meta, validation=validation,
                          metric_map=metric_map, gold_map=gold_map,
                          per_domain=per_domain,
                          per_query_type=per_query_type,
                          query_types=query_types)
    return report


def _group_aggregates(metric_map, query_types):
    """Aggregate per-topic metrics by groups of a derived taxonomy key."""
    groups = {
        "period_bucket": {}, "n_periods": {}, "temporal_scope": {}, "step_class": {},
    }
    for qid, meta in (query_types or {}).items():
        if qid not in metric_map:
            continue
        for key, val in meta.items():
            groups[key].setdefault(val, {})[qid] = metric_map[qid]
    out = {}
    for key, buckets in groups.items():
        out[key] = {str(val): analysis_mod.aggregate(items)
                    for val, items in buckets.items()}
    return out


def _run_all_domains(args):
    t1 = os.path.join(args.data_dir, TRACK1_ROOT)
    if not os.path.isdir(t1):
        _die(f"data-dir track1_tempo not found: {t1}")
    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        _die(f"runs-dir not found: {runs_dir}")

    domains = sorted(os.listdir(t1))
    per_domain, missing_runs = {}, []
    for dom in domains:
        ddir = os.path.join(t1, dom)
        if not os.path.isdir(ddir):
            continue
        run = os.path.join(runs_dir, f"{dom}_1a_{args.split}.trec")
        if not os.path.isfile(run):
            run = os.path.join(runs_dir, f"{dom}_1a_{args.split}.txt")
        if not os.path.isfile(run):
            missing_runs.append(dom)
            continue
        sub_meta = {
            "run_file": run,
            "qrels": os.path.join(ddir, f"qrels_{args.split}.txt"),
            "corpus": os.path.join(ddir, "documents.jsonl"),
            "steps": os.path.join(ddir, f"steps_{args.split}.jsonl"),
            "domain": dom, "split": args.split, "tag": args.tag,
            "k": args.k, "created_utc": now_utc_iso(),
        }
        sub = argparse.Namespace(**sub_meta)
        sub.run = sub.run_file   # _run_single reads args.run
        report = _run_single(sub)
        per_domain[dom] = report
        print(f"[{dom}] nDCG@10 {report['metrics_macro']['nDCG@10']} "
              f"({report['metrics_macro']['num_topics']} topics)",
              flush=True)

    # official macro: average of per-domain means
    official = _per_domain_macro(per_domain)
    report = {
        "schema": report["schema"] if per_domain else _schema(),
        "pipeline_version": __version__,
        "created_utc": now_utc_iso(),
        "meta": {"data_dir": args.data_dir, "runs_dir": runs_dir,
                 "split": args.split, "k": args.k, "tag": args.tag,
                 "mode": "all-domains"},
        "params": {"k": args.k},
        "official_metric": "nDCG@10",
        "metrics_macro": official,
        "per_domain": per_domain,
        "missing_runs": missing_runs,
        "per_query_type": {},
        "query_types": {},
        "per_topic": {},
        "missing_topics": [],
    }
    return report


def _per_domain_macro(per_domain):
    import statistics
    keys = ("nDCG@10", "Recall@10", "P@10", "MAP@10", "MRR", "TR@10", "TP@10",
            "TC@10", "NDCG|FC@10")
    out = {"num_topics": 0, "num_domains": len(per_domain),
           "average_unit": "domain (official)"}
    seqs = {k: [] for k in keys}
    for rep in per_domain.values():
        m = rep["metrics_macro"]
        for k in keys:
            if k in m:
                seqs[k].append(m[k])
        out["num_topics"] += m.get("num_topics", 0)
    for k in keys:
        if seqs[k]:
            out[k] = round(statistics.mean(seqs[k]), 4)
    return out


def _schema():
    from . import SCHEMA
    return SCHEMA


def _die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RETECO evaluation pipeline")
    ap.add_argument("--run", help="TREC run file")
    ap.add_argument("--qrels", help="qrels file")
    ap.add_argument("--corpus", help="documents.jsonl (validates + discards unknown docids)")
    ap.add_argument("--steps", help="steps jsonl (temporal diagnostics + query types)")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--all-domains", action="store_true",
                    help="scan data-dir/track1_tempo for every domain")
    ap.add_argument("--data-dir", help="reteco_data root (with --all-domains)")
    ap.add_argument("--runs-dir", help="dir containing <domain>_1a_<split>.trec files")
    args = ap.parse_args(argv)

    if args.all_domains:
        if not (args.data_dir and args.runs_dir):
            ap.error("--all-domains requires --data-dir and --runs-dir")
        report = _run_all_domains(args)
    else:
        for req in ("run", "qrels"):
            if getattr(args, req) is None:
                ap.error(f"--{req} is required (or use --all-domains)")
        report = _run_single(args)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "test_results.json")
    write_report(report, out)
    print(f"wrote {out}")

    valid = report.get("validation", {}).get("valid", True)
    if valid:
        print(f"RESULT: VALID   nDCG@{args.k}: "
              f"{report['metrics_macro'].get('nDCG@' + str(args.k))}")
        return 0
    print("RESULT: INVALID (see validation.errors in the report)")
    return 1


if __name__ == "__main__":
    sys.exit(main())