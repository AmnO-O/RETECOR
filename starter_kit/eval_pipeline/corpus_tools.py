"""Corpus cleaning / deduplication / chunking for RETECO retrieval corpora.

Design (track 1a):
  * Atomic scoring unit = the document id in ``documents.jsonl``; qrels use ids
    like ``cardano/<num>_<hex>_<n>.txt``. Chunking never re-ids an atomic doc:
    short docs pass through unchanged, long docs are split into windows that map
    back to their parent id via a chunk map (``chunk_id  parent_id``).
  * Streaming: a 255 MB corpus is processed line-by-line (bounded RAM).
  * Pure standard library.

Typical invocation::

    python -m eval_pipeline.corpus_tools retaco/track1_tempo/cardano/documents.jsonl \\
        --out corpus_clean.jsonl --map chunk_map.tsv
"""
import argparse
import hashlib
import json
import re

RE_HTML = re.compile(r"<[a-zA-Z/][^>]*>")
RE_BBCODE = re.compile(r"\[/?[a-z0-9]+\]")
RE_URL = re.compile(r"\[url=[^\]]*\](.*?)\[/url\]|\[url\](.*?)\[/url\]"
                    r"|https?://\S+", re.I)
RE_EMOJI = re.compile(
    r"[\U0001F000-\U0001FAFF\U00002702-\U000027B0\U00002600-\U000026FF"
    r"\U0000FE0F\u2764]+")
RE_WS = re.compile(r"\s+")


def clean_text(text):
    """Strip HTML / bbcode / URLs; collapse emoji + whitespace.

    URL handling first so ``[url=...]label[/url]`` keeps the label; ``https://``
    tokens are removed entirely.
    """
    if not text:
        return text
    t = RE_URL.sub(lambda m: (m.group(1) or m.group(2) or "").strip(), text)
    t = RE_BBCODE.sub(" ", t)
    t = RE_HTML.sub(" ", t)
    t = RE_EMOJI.sub(" ", t)
    return RE_WS.sub(" ", t).strip()


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_doc(doc, window=300, overlap=50, min_words=1500):
    """Split one doc into overlapping word windows.

    ``len(words) <= min_words`` → a single record whose ``id == parent_id``.
    Longer docs → windows with ``id == "<parent>|<n>"``. Returns a list of
    ``{"id", "parent_id", "content"}`` dicts.
    """
    words = doc["content"].split()
    if len(words) <= min_words:
        return [{"id": doc["id"], "parent_id": doc["id"],
                 "content": doc["content"]}]
    step = max(1, window - overlap)
    recs = []
    for n, start in enumerate(range(0, len(words), step), 1):
        recs.append({
            "id": f"{doc['id']}|{n}",
            "parent_id": doc["id"],
            "content": " ".join(words[start:start + window]),
        })
        if start + window >= len(words):
            break
    return recs


def truncate_content(doc, cap_words=20000):
    """Cap pathological docs at cap_words (no-op otherwise)."""
    words = doc["content"].split()
    if len(words) > cap_words:
        doc = dict(doc)
        doc["content"] = " ".join(words[:cap_words])
    return doc


def run(docs_path, out_path, map_path=None, window=300, overlap=50,
        min_words=1500, cap_words=20000, dedupe=True, protected_ids=(),
        log=print):
    """Stream documents.jsonl -> cleaned, deduped, capped, windowed records.

    Writes ``{"id","parent_id","content"}`` lines to ``out_path`` and
    ``chunk_id<TAB>parent_id`` lines for windows to ``map_path``.
    ``protected_ids`` (iterable of doc ids present in qrels) are exempt from
    dedupe so gold targets are never dropped even when content-duplicated.
    Returns (records, chunk_windows, dropped).
    """
    from .dataio import iter_jsonl

    seen = set()
    records = chunks = dropped = 0
    protected = set(protected_ids)
    with open(out_path, "w", encoding="utf-8") as out:
        with (open(map_path, "w", encoding="utf-8")
              if map_path else _null_ctx()) as mp:
            for d in iter_jsonl(docs_path):
                c = d.get("content", "")
                if not c:
                    dropped += 1
                    continue
                c = clean_text(c)
                if not c:
                    dropped += 1
                    continue
                doc_id = d["id"]
                if dedupe:
                    h = _sha1(c)
                    # protected (gold) docs always pass and seed the seen set;
                    # unprotected duplicates thereafter are dropped
                    if h in seen and doc_id not in protected:
                        dropped += 1
                        continue
                    seen.add(h)
                doc = truncate_content({"id": doc_id, "content": c}, cap_words)
                for r in chunk_doc(doc, window=window, overlap=overlap,
                                   min_words=min_words):
                    records += 1
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if r["id"] != r["parent_id"]:
                        chunks += 1
                        mp.write(f"{r['id']}\t{r['parent_id']}\n")
    log(f"records={records} chunk_windows={chunks} dropped={dropped} "
        f"unique_docs={records - chunks}")
    return records, chunks, dropped


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, *_):
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="clean + dedupe + window a RETECO documents.jsonl corpus")
    ap.add_argument("docs", help="path to documents.jsonl")
    ap.add_argument("--out", default="corpus_clean.jsonl",
                    help="output records jsonl (default: corpus_clean.jsonl)")
    ap.add_argument("--map", default="chunk_map.tsv",
                    help="chunk map output; pass --no-map to disable")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--chunk-min-words", type=int, default=1500)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--cap-words", type=int, default=20000)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--qrels", help="qrels file; its doc ids are exempt from dedupe")
    a = ap.parse_args(argv)

    from .dataio import read_qrels
    protected = set()
    if a.qrels:
        for qids in read_qrels(a.qrels).values():
            protected |= qids
    run(a.docs, a.out, None if a.no_map else a.map,
        window=a.window, overlap=a.overlap, min_words=a.chunk_min_words,
        cap_words=a.cap_words, dedupe=not a.no_dedupe,
        protected_ids=protected, log=print)


if __name__ == "__main__":
    main()