"""Pipeline-level tests: end-to-end scoring through the real report path,
missing topics score zero, dedupe keep-first, unknown-docid discard, rerank by
score, nan handling, and the report schema."""
import json
import os
import random
import tempfile
import unittest

from eval_pipeline import analysis as A
from eval_pipeline import report as R
from eval_pipeline.dataio import read_run, read_steps
from eval_pipeline.metrics import ndcg_at_k
from eval_pipeline.run_validation import validate
from eval_pipeline.taxonomy import types_from_steps
from eval_pipeline.tests import dummy_data


def _read(lines):
    """Parse TREC strings through the real dataio path (temp file)."""
    with tempfile.NamedTemporaryFile("w", suffix=".trec", delete=False,
                                     encoding="utf-8") as f:
        f.write("\n".join(lines))
        name = f.name
    try:
        return read_run(name)
    finally:
        os.unlink(name)


class TestPipelineBehaviour(unittest.TestCase):
    def _score(self, lines, qrels, corpus_ids=None, steps_path=None, k=10):
        validation = validate(_read(lines), qrels=qrels, corpus_ids=corpus_ids)
        ranked, _ = A.preprocess_run(_read(lines), qrels, corpus_ids)
        steps = read_steps(steps_path) if steps_path else {}
        metric_map = A.score_topics(ranked, qrels, steps, k)
        return validation, ranked, metric_map

    def test_dedupe_keep_first(self):
        lines = [dummy_data.line("q1", "d1", 1, 2.0),
                 dummy_data.line("q1", "d1", 2, 1.0)]
        qrels = {"q1": {"d1"}}
        validation, ranked, metric_map = self._score(lines, qrels)
        self.assertEqual(ranked["q1"], ["d1"])
        self.assertAlmostEqual(metric_map["q1"]["nDCG@10"], 1.0)
        self.assertTrue(any("duplicate docid" in w
                            for w in validation["warnings"]))

    def test_missing_topic_scores_zero_is_counted(self):
        qrels = {"q1": {"d1"}, "q2": {"d2"}}
        lines = [dummy_data.line("q1", "d1", 1, 1.0)]
        validation, ranked, metric_map = self._score(lines, qrels)
        self.assertEqual(validation["missing_topics"], ["q2"])
        self.assertEqual(metric_map["q2"]["nDCG@10"], 0.0)
        macro = A.aggregate(metric_map)
        self.assertLess(macro["nDCG@10"], 1.0)

    def test_unknown_docid_discarded(self):
        qrels = {"q1": {"d1", "d2"}}
        lines = [dummy_data.line("q1", "zzz", 1, 9.0),
                 dummy_data.line("q1", "d1", 2, 1.0)]
        validation, ranked, metric_map = self._score(
            lines, qrels, corpus_ids={"d1", "d2"})
        self.assertEqual(ranked["q1"], ["d1"])
        # 2 relevant, only d1 retrieved -> nDCG = 1 / (1 + 1/log2(3))
        self.assertAlmostEqual(metric_map["q1"]["nDCG@10"],
                               1.0 / (1.0 + 1.0 / 1.584962500721156), places=6)
        self.assertTrue(any("not in corpus" in w
                            for w in validation["warnings"]))

    def test_rerank_by_score_not_rank(self):
        qrels = {"q1": {"d1"}}
        lines = [dummy_data.line("q1", "d1", 1, 0.1),
                 dummy_data.line("q1", "other", 2, 9.0)]
        _, ranked, _ = self._score(lines, qrels)
        self.assertEqual(ranked["q1"], ["other", "d1"])

    def test_nan_topic_excluded_and_zero(self):
        qrels = {"q1": {"d1"}}
        lines = [dummy_data.line("q1", "d1", 1, "nan")]
        validation, ranked, metric_map = self._score(lines, qrels)
        self.assertFalse(validation["valid"])
        self.assertEqual(metric_map["q1"]["nDCG@10"], 0.0)

    def test_seeded_golden_end_to_end(self):
        # fully deterministic dummy dataset, check the official metric computes
        topics, docs, qrels, lines, expected = dummy_data.make_dataset(seed=42)
        # build an ideal run to validate nDCG=1 for every topic
        ideal = []
        for qid in set(qrels):
            rel = sorted(qrels[qid])
            for i, d in enumerate(rel, 1):
                ideal.append(dummy_data.line(qid, d, i, 100.0 - i))
        _, ranked, metric_map = self._score(ideal, qrels, corpus_ids=set(docs))
        for qid in qrels:
            self.assertAlmostEqual(metric_map[qid]["nDCG@10"], 1.0,
                                   places=6, msg=qid)


class TestReportSchema(unittest.TestCase):
    def test_report_roundtrip(self):
        topics, docs, qrels, lines, _ = dummy_data.make_dataset(seed=3)
        grading = {q: set(g) for q, g in qrels.items()}
        with tempfile.TemporaryDirectory() as tmp:
            steps_path = os.path.join(tmp, "steps.jsonl")
            dummy_data.make_steps(random.Random(1), topics, docs, steps_path)
            lines_parsed = _read(lines)
            validation = validate(lines_parsed, qrels=qrels, corpus_ids=set(docs))
            ranked, _ = A.preprocess_run(lines_parsed, qrels, set(docs))
            metric_map = A.score_topics(ranked, qrels, read_steps(steps_path), 10)
            report = R.build_report(
                meta={"run_file": "run.trec", "qrels": "qrels.txt", "corpus": None,
                      "steps": steps_path, "domain": "dummy", "split": "dev",
                      "tag": "r", "k": 10, "created_utc": R.now_utc_iso()},
                validation=validation, metric_map=metric_map,
                gold_map=grading, per_query_type={},
                query_types=types_from_steps(read_steps(steps_path)))
            out = os.path.join(tmp, "test_results.json")
            R.write_report(report, out)
            with open(out, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["schema"], "reteco-eval-report/1")
            self.assertEqual(set(loaded["per_topic"]), set(qrels))
            self.assertEqual(set(loaded["query_types"]), set(qrels))
            # every topic present, including the injected missing one (t4)
            self.assertIn("t4", loaded["per_topic"])
            self.assertIn("t4", loaded["missing_topics"])


class TestMetricsAgreementWithStarterKit(unittest.TestCase):
    def test_ndcg_matches_ir_metrics(self):
        """Same nDCG values as the existing starter-kit scorer."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        from ir_metrics import ndcg_at_k as starter_ndcg
        ranked = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]
        for gold in ({"d2"}, {"d1", "d4"}, set(), {"d7", "d2", "d9"}):
            self.assertAlmostEqual(ndcg_at_k(ranked, gold, 10),
                                   starter_ndcg(ranked, gold, 10))


if __name__ == "__main__":
    unittest.main()