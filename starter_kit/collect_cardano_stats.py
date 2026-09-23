import json, io, re, collections, statistics, os

DIR = r"..\reteco_data\track1_tempo\cardano"

# ---- file inventory ----
print("=== FILE INVENTORY ===")
for fn in sorted(os.listdir(DIR)):
    p = os.path.join(DIR, fn)
    n = 0
    if fn.endswith(".jsonl"):
        with io.open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f if _.strip())
    elif fn.endswith(".txt"):
        with io.open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f if _.strip())
    print(f"  {fn}: {os.path.getsize(p)/1e6:8.2f} MB, {n} lines/records")

CORPUS = os.path.join(DIR, "documents.jsonl")

# ---- corpus profile ----
lens = []
words = []
paras = []
sents = []
lang_marker = 0
bb = url = html = 0
RE_PSEUDO = re.compile(r"\[/?[a-z0-9]+\]")
RE_URL = re.compile(r"https?://|\[url")
RE_HTML = re.compile(r"<[a-zA-Z/][^>]*>")
RE_LANG = re.compile(r"[\u0400-\u4DBF\uA960-\uA97F\uAC00-\uD7AF\u3040-\u30FF\uAC00-\uD7AF]")
id_styles = collections.Counter()
total_chars = 0
with io.open(CORPUS, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        c = d["content"]
        n = len(c)
        lens.append(n); total_chars += n
        words.append(len(c.split()))
        ps = [p for p in c.split("\n") if p.strip()]
        paras.append(len(ps))
        sents.append(sum(1 for p in ps for _ in p.rstrip().split(". ")))
        if RE_PSEUDO.search(c): bb += 1
        if RE_URL.search(c): url += 1
        if RE_HTML.search(c): html += 1
        if RE_LANG.search(c): lang_marker += 1
        m = re.match(r"^(cardano/)(.+)$", d["id"])
        if m and "_" in m.group(2):
            pre = m.group(2).split("_")[0]
            id_styles[("num_hex" if re.fullmatch(r"[0-9a-f]+", pre) else pre)] += 1

N = len(lens)
def pct(x, q):
    xs = sorted(x); return xs[min(len(xs)-1, int(len(xs)*q))]
print("=== CORPUS PROFILE ===")
print(f"  docs: {N}   total chars: {total_chars:,}   total words: {sum(words):,}")
print(f"  chars: min={min(lens)} med={pct(lens,.5)} mean={statistics.mean(lens):.0f} p90={pct(lens,.9)} p95={pct(lens,.95)} p99={pct(lens,.99)} max={max(lens)}")
print(f"  words: min={min(words)} med={pct(words,.5)} mean={statistics.mean(words):.0f} p90={pct(words,.9)} p95={pct(words,.95)} p99={pct(words,.99)} max={max(words)}")
print(f"  paras: med={pct(paras,.5)} mean={statistics.mean(paras):.2f} max={max(paras)}")
print(f"  sents: med={pct(sents,.5)} mean={statistics.mean(sents):.1f} p99={pct(sents,.99)} max={max(sents)}")
print(f"  contamination: bbcode={bb} url={url} html={html} nonCJK={lang_marker}")
print(f"  empty docs: {sum(1 for n in lens if n == 0)}")
zh = collections.Counter()
for line in io.open(CORPUS, encoding="utf-8"):
    c = json.loads(line)["content"]
    for ch in re.findall(r"[\u3400-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0400-\u04FF\u0600-\u06FF\u0900-\u097F]", c):
        pass
# rough id style - skip, low value. print id styles
print("  id styles (prefix part before first '_'):", dict(id_styles.most_common(6)))

# ---- qrels ----
print("=== QRELS ===")
gold = collections.defaultdict(set)
with io.open(os.path.join(DIR, "qrels_dev.txt"), encoding="utf-8") as f:
    for ln in f:
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0: gold[p[0]].add(p[2])
print(f"  dev: {len(gold)} queries; gold/query min={min(len(v) for v in gold.values())} med={statistics.median(len(v) for v in gold.values())} mean={statistics.mean(len(v) for v in gold.values()):.2f} max={max(len(v) for v in gold.values())}")
uniq_gold = {g for v in gold.values() for g in v}
print(f"  unique gold: {len(uniq_gold)}")
print("  qrels column check (first col iter rank rel):")
with io.open(os.path.join(DIR, "qrels_dev.txt"), encoding="utf-8") as f:
    for ln in f:
        p = ln.split()
        if p: break
print("   sample:", p)

# gold doc stats + contamination + resolvability
docs_by_id = {}
with io.open(CORPUS, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line); docs_by_id[d["id"]] = d["content"]
res = [g for g in uniq_gold if g in docs_by_id]
gl = [len(docs_by_id[g]) for g in res]
gw = [len(docs_by_id[g].split()) for g in res]
gbb = sum(1 for g in res if RE_PSEUDO.search(docs_by_id[g]))
gur = sum(1 for g in res if RE_URL.search(docs_by_id[g]))
print("=== GOLD DOCS (dev) ===")
print(f"  resolvable in corpus: {len(res)}/{len(uniq_gold)}")
print(f"  chars: min={min(gl)} med={pct(gl,.5)} mean={statistics.mean(gl):.0f} max={max(gl)}")
print(f"  words: min={min(gw)} med={pct(gw,.5)} mean={statistics.mean(gw):.0f} max={max(gw)}")
print(f"  contaminated: bbcode={gbb} url={gur}")

# ---- train ----
gold_tr = collections.defaultdict(set)
with io.open(os.path.join(DIR, "qrels_train.txt"), encoding="utf-8") as f:
    for ln in f:
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0: gold_tr[p[0]].add(p[2])
print("=== QRELS TRAIN ===")
print(f"  train: {len(gold_tr)} queries; gold/query med={statistics.median(len(v) for v in gold_tr.values())} mean={statistics.mean(len(v) for v in gold_tr.values()):.2f} max={max(len(v) for v in gold_tr.values())}")
ug_tr = {g for v in gold_tr.values() for g in v}
print(f"  unique gold: {len(ug_tr)}; dev/train unique gold overlap with dev: {len(uniq_gold & ug_tr)}")

# ---- guidance (temporal metadata) ----
print("=== GUIDANCE ===")
tempo = nt = 0
gran = collections.Counter()
is_tp = collections.Counter()
for split in ("dev", "train"):
    g = collections.Counter(); pth = os.path.join(DIR, f"guidance_{split}.jsonl")
    if not os.path.exists(pth): continue
    for line in io.open(pth, encoding="utf-8"):
        r = json.loads(line)
        qg = r.get("query_guidance", {})
        gran[qg.get("expected_granularity", "") or "unspecified"] += 1
        is_tp[qg.get("is_temporal_query")] += 1
print(f"  is_temporal_query: {dict(is_tp)}")
print(f"  expected_granularity: {dict(gran)}")

# query text stats from steps file (contains query field)
print("=== QUERY TEXT ===")
for split in ("dev", "train"):
    ql = qw = 0; nq = 0
    pth = os.path.join(DIR, f"steps_{split}.jsonl")
    qs = []
    for line in io.open(pth, encoding="utf-8"):
        r = json.loads(line)
        q = r.get("query", ""); ql += len(q); qw += len(q.split()); nq += 1
        qs.append(q)
    print(f"  {split}: {nq} queries; chars mean={ql/nq:.0f} med={pct([len(q) for q in qs],.5)}; words mean={qw/nq:.0f} med={pct([len(q.split()) for q in qs],.5)} max={pct([len(q.split()) for q in qs],1)}")

# steps count per query
print("=== STEPS ===")
for split in ("dev", "train"):
    sc = []
    for line in io.open(os.path.join(DIR, f"steps_{split}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        sc.append(0 if not isinstance(r.get("steps"), list) else len(r["steps"]))
    print(f"  {split}: steps/query min={min(sc)} med={pct(sc,.5)} mean={statistics.mean(sc):.2f} max={max(sc)}")