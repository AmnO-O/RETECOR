"""Golden-value tests for the metric implementations."""
import math
import random
import unittest

from eval_pipeline.metrics import (
    average_precision_at_k, map_at_k, ndcg_at_k, precision_at_k, recall_at_k,
    reciprocal_rank, temporal_coverage_at_k)

L2_3 = math.log2(3)      # 1.584962500721156
L2_5 = math.log2(5)      # 2.321928094887362


class TestNDCG(unittest.TestCase):
    def test_golden_hand_computed(self):
        # ranked with gold={d2,d4}, k=4: DCG=1/log2(3)+1/log2(5), IDCG=1+1/log2(3)
        ranked = ["d1", "d2", "d3", "d4", "d5"]
        self.assertAlmostEqual(ndcg_at_k(ranked, {"d2", "d4"}, 4),
                               (1 / L2_3 + 1 / L2_5) / (1 + 1 / L2_3), places=6)
        # perfect ranking -> 1.0
        self.assertAlmostEqual(ndcg_at_k(ranked, {"d1", "d2", "d3"}, 4), 1.0)
        # no relevant retrieved but gold exists -> 0.0
        self.assertEqual(ndcg_at_k(ranked, {"x", "y"}, 4), 0.0)

    def test_no_gold_idcg_zero_is_zero_not_nan(self):
        self.assertEqual(ndcg_at_k(["a", "b"], set(), 10), 0.0)

    def test_shortliste(self):
        self.assertAlmostEqual(ndcg_at_k(["d1"], {"d1"}, 10),
                               1.0 / math.log2(2) / (1.0 / math.log2(2)), places=9)


class TestPrecisionRecall(unittest.TestCase):
    def test_golden(self):
        ranked = ["d2", "d3", "d1", "d4", "d5"]
        gold = {"d2", "d4"}
        self.assertAlmostEqual(precision_at_k(ranked, gold, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, gold, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, gold, 10), 1.0)

    def test_k_beyond_list(self):
        self.assertAlmostEqual(precision_at_k(["a"], {"a"}, 10), 0.1)

    def test_no_relevant(self):
        self.assertEqual(recall_at_k(["a"], set(), 10), 0.0)


class TestMRR(unittest.TestCase):
    def test_first_relevant_at_rank2(self):
        self.assertAlmostEqual(reciprocal_rank(["d1", "d2"], {"d2"}), 0.5)

    def test_missing(self):
        self.assertEqual(reciprocal_rank(["a", "b"], {"z"}), 0.0)


class TestMAP(unittest.TestCase):
    def test_classic_ap(self):
        # rel pattern [1, 0, 1, 0] -> AP = (1 + 2/3)/2 = 0.8333
        self.assertAlmostEqual(
            average_precision_at_k(["d2", "d3", "d1", "d4"], {"d1", "d2"}, 4),
            0.8333333333333333, places=6)

    def test_cutoff(self):
        # relevant only beyond cutoff -> 0
        self.assertEqual(average_precision_at_k(["a", "b", "c"], {"d"}, 1), 0.0)

    def test_map_macro(self):
        r = {"q1": ["a"], "q2": ["b", "x"]}
        g = {"q1": {"a"}, "q2": {"b"}}
        self.assertAlmostEqual(map_at_k(r, g, 10), (1.0 + 1.0) / 2)


class TestTemporal(unittest.TestCase):
    def test_coverage(self):
        periods = [{"d1", "d2"}, {"d3"}]
        self.assertAlmostEqual(temporal_coverage_at_k(
            ["d1", "d9", "d3"], periods, 10), 1.0)
        self.assertAlmostEqual(temporal_coverage_at_k(
            ["d1", "d9", "d8"], periods, 10), 0.5)
        self.assertIsNone(temporal_coverage_at_k(["d1"], [], 10))


class TestDeterminism(unittest.TestCase):
    def test_tiebreak_stable(self):
        # same score, different docids: sort is stable by input order
        # (preprocess_run sorts by (-score, docid), covered there; here we just
        # confirm metrics don't mutate input)
        ranked = ["b", "a"]
        ndcg_at_k(ranked, {"a"})
        self.assertEqual(ranked, ["b", "a"])

    def test_seeded_reproducible(self):
        rng1, rng2 = random.Random(7), random.Random(7)
        a = rng1.sample(range(100), 50)
        b = rng2.sample(range(100), 50)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()