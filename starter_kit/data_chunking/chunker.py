"""Per-domain corpus prep CLI for RETECO Track 1a.

Cleans, dedupes (with qrels gold protection) and chunks one domain's
``documents.jsonl``, then writes:

  * ``corpus_clean.jsonl`` — records ``{id, parent_id, content}``
  * ``chunk_map.tsv`` — ``chunk_id<TAB>parent_id`` for long-tail windows
  * ``stats.json`` — counts + gold-coverage verification

Usage::

    python -m data_chunking.chunker iota --data-dir ..\\reteco_data
    python -m data_chunking.chunker cardano --data-dir ..\\reteco_data --sample 5
"""
import argparse
import json
import os
import random
import sys

from . import __version__

DOMAIN_ALIASES = {"iota", "cardano", "bitcoin", "monero", "economics", "law",
                  "politics", "history", "quant_finance", "travel",
                  "workplace", "genealogy", "science_math"}


def _find_domain_dir(data_dir, domain):
    candidates = [
        os.path.join(data_dir, "track1_tempo", domain),
        os.path.join(data_dir, "track2_recor", domain),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"no corpus dir found for {domain!r} under {data_dir!r}")


def build_stats(docs_path, qrels_path, out_dir, window=300, overlap=50,
                min_words=1500, cap_words=20000, dedupe=True,
                seed=1337, sample=3, log=print):
    """Run the corpus cleanup pipeline and persist artifacts + a stats file.

    Returns the stats dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, "corpus_clean.jsonl")
    map_path = os.path.join(out_dir, "chunk_map.tsv")
    stats_path = os.path.join(out_dir, "stats.json")

    protected = set()
    if qrels_path and os.path.isfile(qrels_path):
        from eval_pipeline.dataio import read_qrels
        for qids in read_qrels(qrels_path).values():
            protected |= qids

    from eval_pipeline import corpus_tools
    records, chunks, dropped = corpus_tools.run(
        docs_path, clean_path, map_path, window=window, overlap=overlap,
        min_words=min_words, cap_words=cap_words, dedupe=dedupe,
        protected_ids=protected, log=log)

    # gold coverage check
    gold_total = len(protected)
    indexed = set()
    with open(clean_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                indexed.add(json.loads(line)["parent_id"])
    missing_gold = sorted(protected - indexed)

    stats = {
        "domain": os.path.basename(os.path.dirname(docs_path)),
        "params": {"window": window, "overlap": overlap,
                   "min_words": min_words, "cap_words": cap_words,
                   "dedupe": dedupe},
        "corpus_records": None,
        "records_out": records, "chunk_windows": chunks, "dropped": dropped,
        "unique_docs": records - chunks,
        "gold_total": gold_total,
        "gold_after_clean": gold_total - len(missing_gold),
        "gold_missing": missing_gold,
    }
    with open(docs_path, encoding="utf-8") as f:
        stats["corpus_records"] = sum(1 for _ in f if _.strip())

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log(f"wrote {stats_path}")

    if sample:
        _print_sample(clean_path, seed=seed, n=sample, log=log)
    return stats


def _print_sample(clean_path, seed, n, log=print):
    rows = []
    with open(clean_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        log("(no records to sample)")
        return
    rng = random.Random(seed)
    log("\n--- sample chunks ---")
    for r in rng.sample(rows, min(n, len(rows))):
        cid, pid = r["id"], r["parent_id"]
        tag = " [WINDOW]" if cid != pid else ""
        txt = r["content"].replace("\n", " ")[:180]
        log(f"  {cid}{tag} (parent {pid})\n    {txt}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="clean + dedupe + chunk one RETECO domain")
    ap.add_argument("domain", choices=sorted(DOMAIN_ALIASES))
    ap.add_argument("--data-dir", default=r"..\reteco_data",
                    help="reteco_data root (default: ..\\reteco_data)")
    ap.add_argument("--out-dir", default=r"data_chunking\out",
                    help="where to write outputs (default: data_chunking\\out, "
                         "relative to the starter_kit cwd)")
    ap.add_argument("--qrels", default=None,
                    help="qrels file; defaults to qrels_dev.txt in the domain dir")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--chunk-min-words", type=int, default=1500)
    ap.add_argument("--cap-words", type=int, default=20000)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--sample", type=int, default=3, help="chunks to eyeball (0 = off)")
    a = ap.parse_args(argv)

    ddir = _find_domain_dir(a.data_dir, a.domain)
    docs_path = os.path.join(ddir, "documents.jsonl")
    qrels_path = a.qrels or os.path.join(ddir, f"qrels_{a.split}.txt")
    out_dir = os.path.join(a.out_dir, a.domain)
    log = print
    log(f"data_chunking v{__version__}")
    log(f"domain dir : {ddir}")
    log(f"documents  : {docs_path}")
    log(f"qrels      : {qrels_path}")
    log(f"params     : window={a.window} overlap={a.overlap} "
        f"min_words={a.chunk_min_words} cap_words={a.cap_words}")

    stats = build_stats(
        docs_path, qrels_path, out_dir,
        window=a.window, overlap=a.overlap, min_words=a.chunk_min_words,
        cap_words=a.cap_words, dedupe=not a.no_dedupe, sample=a.sample,
        log=log)
    n0, n1 = stats["corpus_records"], stats["records_out"]
    log("\nsize change: {} -> {} records ({}%)".format(
        n0, n1, round(n1 / n0 * 100, 1) if n0 else 0))


if __name__ == "__main__":
    sys.exit(main())