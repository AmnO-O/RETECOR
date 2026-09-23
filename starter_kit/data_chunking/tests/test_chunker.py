import os
import shutil
import tempfile
import unittest
import json

from data_chunking import chunker
from data_chunking.chunker import build_stats


def _text(length, fill="word"):
    return " ".join(f"{fill}_{i}" for i in range(length))


def _write_corpus(ddir, docs):
    with open(os.path.join(ddir, "documents.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _write_qrels(ddir, rows):
    with open(os.path.join(ddir, "qrels_dev.txt"), "w", encoding="utf-8") as f:
        for qid, docid in rows:
            f.write(f"{qid} 0 {docid} 1\n")


class TestFindDomainDir(unittest.TestCase):

    def test_track1_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "track1_tempo", "iota"))
            got = chunker._find_domain_dir(tmp, "iota")
            self.assertTrue(got.endswith(os.path.join("track1_tempo", "iota")))

    def test_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            chunker._find_domain_dir("C:\\does_not_exist", "iota")


class TestBuildStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ddir = os.path.join(self.tmp, "domain_iota")
        os.makedirs(self.ddir)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_gold_protected_through_dedupe_and_chunks(self):
        docs = [
            {"id": "iota/short_1.txt", "content": _text(20)},
            # exact duplicate of short_1 but referenced by qrels -> must survive
            {"id": "iota/short_dup.txt", "content": _text(20)},
            # long doc -> gets windowed, parent id must survive
            {"id": "iota/longer.txt", "content": _text(3000)},
            # cleans to empty -> dropped
            {"id": "iota/garbage.txt", "content": "[b][/b] http://x"},
        ]
        _write_corpus(self.ddir, docs)
        _write_qrels(self.ddir, [("q1", "iota/short_dup.txt"),
                                 ("q1", "iota/longer.txt")])
        out = os.path.join(self.tmp, "out")
        stats = build_stats(os.path.join(self.ddir, "documents.jsonl"),
                            os.path.join(self.ddir, "qrels_dev.txt"), out,
                            window=300, overlap=50, min_words=1500,
                            cap_words=20000, dedupe=True, sample=0,
                            log=lambda *_: None)
        self.assertEqual(stats["gold_total"], 2)
        self.assertEqual(stats["gold_after_clean"], 2)
        self.assertEqual(stats["gold_missing"], [])
        self.assertEqual(stats["dropped"], 1)  # garbage only
        # short_1 and short_dup both kept (dup protected), longer windowed
        self.assertGreaterEqual(stats["records_out"], 3)
        self.assertEqual(stats["domain"], "domain_iota")

        # chunk map contains window mapping for longer.txt
        map_path = os.path.join(out, "chunk_map.tsv")
        with open(map_path, encoding="utf-8") as f:
            rows = f.read().splitlines()
        self.assertTrue(rows)
        for row in rows:
            cid, pid = row.split("\t")
            self.assertTrue(cid.startswith("iota/longer.txt|"))
            self.assertEqual(pid, "iota/longer.txt")

    def test_unprotected_duplicate_is_dropped(self):
        docs = [
            {"id": "iota/a.txt", "content": _text(20)},
            {"id": "iota/b.txt", "content": _text(20)},  # dup, not gold
        ]
        _write_corpus(self.ddir, docs)
        _write_qrels(self.ddir, [("q1", "iota/a.txt")])
        out = os.path.join(self.tmp, "out")
        stats = build_stats(os.path.join(self.ddir, "documents.jsonl"),
                            os.path.join(self.ddir, "qrels_dev.txt"), out,
                            dedupe=True, sample=0, log=lambda *_: None)
        self.assertEqual(stats["dropped"], 1)
        self.assertEqual(stats["records_out"], 1)
        self.assertEqual(stats["gold_missing"], [])


if __name__ == "__main__":
    unittest.main()