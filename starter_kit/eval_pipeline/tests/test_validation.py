"""Validation edge-case tests for run_validation.validate()."""
import unittest

from eval_pipeline.run_validation import validate


def ln(qid, doc, rank, score, q0="Q0"):
    return {"lineno": 1, "qid": qid, "q0": q0, "docid": doc,
            "rank": str(rank), "score": str(score), "tag": "run"}


class TestStructure(unittest.TestCase):
    def test_well_formed(self):
        qrels = {"q1": {"d1", "d2"}}
        r = validate([ln("q1", "d1", 1, 1.0), ln("q1", "d2", 2, 0.5)], qrels=qrels)
        self.assertTrue(r["valid"])
        self.assertEqual(r["missing_topics"], [])

    def test_duplicate_rank_is_error(self):
        r = validate([ln("q1", "d1", 1, 1.0), ln("q1", "d2", 1, 0.5)])
        self.assertFalse(r["valid"])
        self.assertTrue(any("duplicate rank" in e for e in r["errors"]))

    def test_nonpositive_rank_is_error(self):
        self.assertFalse(validate([ln("q1", "d1", 0, 1.0)])["valid"])
        self.assertFalse(validate([ln("q1", "d1", -2, 1.0)])["valid"])

    def test_non_numeric_rank_score(self):
        self.assertFalse(validate([ln("q1", "d1", "x", 1.0)])["valid"])
        self.assertFalse(validate([ln("q1", "d1", 1, "abc")])["valid"])

    def test_nan_inf_score_is_error(self):
        self.assertFalse(validate([ln("q1", "d1", 1, "nan")])["valid"])
        self.assertFalse(validate([ln("q1", "d1", 1, "inf")])["valid"])


class TestCrossChecks(unittest.TestCase):
    def test_duplicate_docid_warns_keeps_first(self):
        r = validate([ln("q1", "d1", 1, 1.0), ln("q1", "d1", 2, 0.5)])
        self.assertTrue(r["valid"])
        self.assertTrue(any("duplicate docid" in w for w in r["warnings"]))
        self.assertEqual(r["topics_in_run"]["q1"], ["d1"])

    def test_missing_topic_flagged(self):
        qrels = {"q1": {"d1"}, "q2": {"d2"}}
        r = validate([ln("q1", "d1", 1, 1.0)], qrels=qrels)
        self.assertEqual(r["missing_topics"], ["q2"])

    def test_unknown_qid_warns_and_excluded(self):
        qrels = {"q1": {"d1"}}
        r = validate([ln("q1", "d1", 1, 1.0), ln("ghost", "d9", 1, 1.0)], qrels=qrels)
        self.assertTrue(any("not present in qrels" in w for w in r["warnings"]))
        self.assertEqual(list(r["topics_in_run"]), ["q1"])

    def test_unknown_docid_warns(self):
        r = validate([ln("q1", "d1", 1, 1.0)], qrels={"q1": {"d1"}},
                     corpus_ids={"d2"})
        self.assertTrue(any("not in corpus" in w for w in r["warnings"]))

    def test_score_increase_warns(self):
        r = validate([ln("q1", "d1", 1, 0.5), ln("q1", "d2", 2, 0.9)])
        self.assertTrue(any("increases with rank" in w for w in r["warnings"]))

    def test_qu0_not_q0_warns(self):
        r = validate([ln("q1", "d1", 1, 1.0, q0="id")])
        self.assertTrue(any("expected 'Q0'" in w for w in r["warnings"]))

    def test_both_missing_cases_together(self):
        qrels = {"q1": {"d1"}, "q2": {"d2"}}
        r = validate([ln("q1", "d1", 1, 1.0), ln("ghost", "d9", 1, 1.0)], qrels=qrels)
        self.assertEqual(r["missing_topics"], ["q2"])
        self.assertIn("ghost", [w for w in r["warnings"] if "not present" in w][0])


class TestInputSafety(unittest.TestCase):
    def test_empty_lines_ok(self):
        r = validate([])
        self.assertTrue(r["valid"])
        self.assertEqual(r["num_lines"], 0)

    def test_qrels_none(self):
        r = validate([ln("q1", "d1", 1, 1.0)])
        self.assertTrue(r["valid"])

    def test_no_exceptions_on_garbage(self):
        rows = [{"lineno": i, "qid": "q", "q0": "Q0", "docid": "d",
                 "rank": "??", "score": "abc", "tag": "t"} for i in (1, 2)]
        r = validate(rows)
        self.assertFalse(r["valid"])
        self.assertTrue(r["errors"])


if __name__ == "__main__":
    unittest.main()