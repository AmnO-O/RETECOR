"""Unit tests for eval_pipeline.corpus_tools (cleaning, dedupe, chunking)."""
import json
import os
import shutil
import tempfile
import unittest

from eval_pipeline import corpus_tools as C


def _text(length, fill="word"):
    return " ".join(f"{fill}_{i}" for i in range(length))


class TestCleanText(unittest.TestCase):

    def test_bbcode_stripped(self):
        self.assertEqual(
            C.clean_text("hello [b]world[/b] end"),
            "hello world end")

    def test_url_tokens_removed(self):
        self.assertEqual(
            C.clean_text("see https://example.com/a?b=1 for more"),
            "see for more")

    def test_url_bbcode_keeps_label(self):
        self.assertEqual(
            C.clean_text("[url=https://x.io]my label[/url] done"),
            "my label done")

    def test_html_stripped(self):
        self.assertEqual(
            C.clean_text("<p>para <b>bold</b></p> tail"),
            "para bold tail")

    def test_whitespace_collapsed(self):
        self.assertEqual(C.clean_text("a\t b\n  c   d"), "a b c d")

    def test_emoji_removed(self):
        txt = "spam \U0001F0CF\u2764\U0001F600 more"
        self.assertEqual(C.clean_text(txt), "spam more")

    def test_empty(self):
        self.assertEqual(C.clean_text(""), "")
        self.assertEqual(C.clean_text("[b][/b]  \n http://x"), "")


class TestChunkDoc(unittest.TestCase):

    def test_short_doc_passes_through(self):
        recs = C.chunk_doc({"id": "cardano/a1.txt", "content": _text(10)},
                           min_words=1500)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["id"], "cardano/a1.txt")
        self.assertEqual(recs[0]["parent_id"], "cardano/a1.txt")
        self.assertEqual(recs[0]["content"].split(), _text(10).split())

    def test_long_doc_windows_map_to_parent(self):
        n_words = 5000
        recs = C.chunk_doc({"id": "cardano/b2.txt", "content": _text(n_words)},
                           window=300, overlap=50, min_words=1000)
        self.assertGreater(len(recs), 1)
        self.assertTrue(recs[0]["id"].endswith("|1"))
        for r in recs:
            self.assertEqual(r["parent_id"], "cardano/b2.txt")
            self.assertTrue(r["id"].startswith("cardano/b2.txt|"))
            n = len(r["content"].split())
            self.assertLessEqual(n, 300)
            self.assertGreaterEqual(n, 250)  # window minus one step

    def test_window_contiguity(self):
        words = _text(1000).split()
        recs = C.chunk_doc({"id": "c", "content": _text(1000)},
                           window=300, overlap=50, min_words=500)
        self.assertEqual(len(recs), 4)
        # starts advance by (window - overlap) = 250
        for i, r in enumerate(recs):
            start = i * 250
            end = min(start + 300, 1000)
            self.assertEqual(r["content"].split(), words[start:end])
        # last window spans the tail
        self.assertEqual(recs[3]["content"].split(), words[750:1000])

    def test_min_words_zero_chunks_everything(self):
        recs = C.chunk_doc({"id": "d", "content": _text(20)},
                           window=5, overlap=2, min_words=0)
        self.assertGreater(len(recs), 1)
        for r in recs:
            self.assertEqual(r["parent_id"], "d")


class TestTruncate(unittest.TestCase):

    def test_cap(self):
        doc = {"id": "x", "content": _text(3000)}
        capped = C.truncate_content(doc, cap_words=100)
        self.assertEqual(len(capped["content"].split()), 100)
        self.assertEqual(capped["content"].split()[0], "word_0")

    def test_noop_below_cap(self):
        doc = {"id": "x", "content": _text(50)}
        self.assertIs(C.truncate_content(doc, cap_words=100), doc)


class TestRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.corpus = os.path.join(self.tmp, "documents.jsonl")
        self.out = os.path.join(self.tmp, "clean.jsonl")
        self.mapp = os.path.join(self.tmp, "chunk_map.tsv")
        docs = [
            {"id": "c1", "content": _text(30)},                    # atomic short
            {"id": "c2", "content": "dup " + _text(30)},           # unique
            {"id": "c2dup", "content": "dup " + _text(30)},        # exact dup
            {"id": "c3", "content": "[b]big[/b] " + _text(3000)},  # long -> windows
            {"id": "cempty", "content": "http://x"},               # cleans to ""
        ]
        with open(self.corpus, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json_dumps(d) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_run_dedupe_and_chunk(self):
        records, chunks, dropped = C.run(
            self.corpus, self.out, self.mapp, window=300, overlap=50,
            min_words=1500, cap_words=20000, dedupe=True, log=lambda *_: None)
        # c1 atomic, c2 unique atomic, c2dup dropped, c3 chunked, cempty dropped
        self.assertEqual(dropped, 2)
        self.assertGreater(records, 2)
        self.assertGreaterEqual(chunks, 1)

        with open(self.mapp, encoding="utf-8") as f:
            rows = f.read().splitlines()
        self.assertEqual(len(rows), chunks)
        for row in rows:
            chunk_id, parent = row.split("\t")
            self.assertTrue(chunk_id.startswith("c3|"))
            self.assertEqual(parent, "c3")

    def test_no_dedupe_keeps_dupes(self):
        records, chunks, dropped = C.run(
            self.corpus, self.out, None, window=300, overlap=50,
            min_words=1500, cap_words=20000, dedupe=False, log=lambda *_: None)
        self.assertEqual(dropped, 1)  # only cempty
        # c1 + c2 + c2dup atomic + c3 windows
        self.assertGreaterEqual(records, 3)

    def test_protected_ids_exempt_from_dedupe(self):
        # c2dup duplicates c2; protecting c2dup keeps it
        records, chunks, dropped = C.run(
            self.corpus, self.out, None, window=300, overlap=50,
            min_words=1500, cap_words=20000, dedupe=True,
            protected_ids={"c2dup"}, log=lambda *_: None)
        self.assertEqual(dropped, 1)  # only cempty
        parents = []
        for line in open(self.out, encoding="utf-8"):
            parents.append(json.loads(line)["parent_id"])
        self.assertIn("c2dup", parents)
        self.assertIn("c2", parents)


def json_dumps(rec):
    import json
    return json.dumps(rec, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()