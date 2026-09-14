"""Derived query-type taxonomy for RETECO Track 1 queries.

The released data has no explicit query-type label, so we derive auditable,
rule-based groups from the public ``steps_*.jsonl`` and the query text:

  * n_periods         -- number of required time periods (= number of steps,
                         flat sample data degrades to 1).
  * period_bucket     -- 'single' / 'multi' (a direct proxy for how much
                         temporal coverage a system must achieve).
  * temporal_scope    -- coarsest time unit detected in the query text:
                         century/decade/year/month/day/point.
  * step_class        -- keyword class over the step instructions:
                         period/event/trend/compare/other.

Everything here is pure string rules so it can be unit-tested and audited.
"""
import re

_YEARS = r"(?:1[0-9]{3}|20[0-9]{2})"
_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?")
# a day number must not be the start of a 4-digit year: guard with (?!\d)
_DAY = r"\d{1,2}(?!\d)(?:st|nd|rd|th)?"

RE_DAY = re.compile(
    rf"\b(?:{_DAY})\s+(?:of\s+)?(?:{_MONTHS})\b|"          # 7 December 1941
    rf"\b(?:{_MONTHS})\s+(?:the\s+)?{_DAY}"
    rf"(?:\s*,?(?:\s+|\s)*(?:{_YEARS}))?\b", re.I)          # December 7, 1941
RE_MONTH = re.compile(rf"\b(?:{_MONTHS})\b", re.I)
RE_YEAR = re.compile(rf"\b{_YEARS}\b")
RE_DECADE = re.compile(r"\b(?:1[0-9]|20)[0-9]{2}s\b")
RE_CENTURY = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\s+century\b", re.I)

# keyword -> class for step instructions
_STEP_CLASS_RULES = [
    (("period", "between", "during", "throughout", "from", "until", "when in",
      "which period", "time window"), "period"),
    (("trend", "increase", "decrease", "evolution", "develop",
      "change", "grow", "decline", "rise", "fall", "jump"), "trend"),
    (("compare", "comparison", "differen", "versus", "vs "), "compare"),
    (("event", "happen", "occur", "reason", "why", "explain", "cause"), "event"),
]

# finest explicit time unit wins: day > month > year > decade > century.
_TEMPORAL_ORDER = [("century", RE_CENTURY), ("decade", RE_DECADE),
                   ("year", RE_YEAR), ("month", RE_MONTH), ("day", RE_DAY)]


def temporal_scope(query_text):
    """Most specific time unit present: day > month > year > decade > century."""
    text = query_text or ""
    for scope, rx in reversed(_TEMPORAL_ORDER):
        if rx.search(text):
            return scope
    return "point"


def step_class(instructions):
    """Classify a query from its (public) step instructions."""
    text = " ".join(instructions).lower()
    for keywords, cls in _STEP_CLASS_RULES:
        if any(kw in text for kw in keywords):
            return cls
    return "other"


def classify_query(query_text, step_count, instructions):
    """Produce the query-type dict for one qid."""
    scope = temporal_scope(query_text)
    return {
        "period_bucket": "multi" if step_count and step_count > 1 else "single",
        "n_periods": step_count or 1,
        "temporal_scope": scope,
        "step_class": step_class(instructions),
    }


def types_from_steps(steps_data):
    """Build {qid: query-type dict} from dataio.read_steps output."""
    out = {}
    for qid, info in steps_data.items():
        out[qid] = classify_query(info.get("query") or "",
                                  info.get("step_count") or 0,
                                  info.get("instructions") or [])
    return out