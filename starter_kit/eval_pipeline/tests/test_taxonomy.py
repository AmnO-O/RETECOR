"""Taxonomy (derived query-type) unit tests + the dummy-data edge generator."""
import unittest

from eval_pipeline.taxonomy import classify_query, temporal_scope, step_class


class TestTemporalScope(unittest.TestCase):
    # rule: finest explicit time unit wins, day > month > year > decade > century
    def test_century_when_alone(self):
        self.assertEqual(temporal_scope("What happened in the 20th century?"),
                         "century")

    def test_decade(self):
        self.assertEqual(temporal_scope("events of the 1980s"), "decade")

    def test_year(self):
        self.assertEqual(temporal_scope("in 1967"), "year")

    def test_month_beats_year(self):
        self.assertEqual(temporal_scope("jump in April-May 2023"), "month")

    def test_day_beats_month(self):
        self.assertEqual(
            temporal_scope("on December 7 1941 (Dec 1941)"),
            "day")

    def test_finest_wins_across_mixed_units(self):
        # mentions century/decade/year too, but a month is the finest needed
        self.assertEqual(
            temporal_scope("20th century, 1920s, 1923, April 1923"),
            "month")

    def test_point_when_nothing(self):
        self.assertEqual(temporal_scope("why is the sky blue?"), "point")


class TestStepClass(unittest.TestCase):
    def test_period(self):
        self.assertEqual(step_class(["look at time window between 1940 and 1945"]),
                         "period")

    def test_trend(self):
        self.assertEqual(step_class(["check the evolution / decline of prices"]),
                         "trend")

    def test_compare(self):
        self.assertEqual(step_class(["compare the two decades"]), "compare")

    def test_event(self):
        self.assertEqual(step_class(["explain why the event occurred"]), "event")

    def test_other_fallback(self):
        self.assertEqual(step_class(["just retrieve everything"]), "other")


class TestClassify(unittest.TestCase):
    def test_multi(self):
        t = classify_query("cases in 1996-1997?", 2, ["during the period", "compare"])
        self.assertEqual(t["period_bucket"], "multi")
        self.assertEqual(t["n_periods"], 2)
        self.assertEqual(t["temporal_scope"], "year")

    def test_single(self):
        t = classify_query("what caused the 1931 fire?", 0, [])
        # no steps -> single bucket, point scope (no explicit time unit pattern)
        self.assertEqual(t["period_bucket"], "single")


if __name__ == "__main__":
    unittest.main()