"""The evaluation runner: failures are recorded fairly and their cost is still counted."""

import json
from pathlib import Path

import pytest

from brightroom.baselines import GradingError, grade_reply
from brightroom.catalogue import load_catalogue
from brightroom.evaluate import run_case, summarise
from brightroom.llm import LLMResult
from brightroom.scoring import load_cases

CAT = load_catalogue()
CASES = {c.id: c for c in load_cases(Path(__file__).resolve().parents[1] / "eval" / "cases" / "dev.json")}
AUDIT_Q1 = json.dumps(
    {
        "kind": "proposal",
        "venue_id": "V1",
        "service_ids": ["S1"],
        "stated_total": 320,
        "reasons": [],
        "flagged_assumptions": ["accessibility"],
        "asks_about": ["accessibility"],
        "declines": [],
        "alternative_totals": [],
        "venues_offered": ["V1"],
        "promises": [],
    }
)


class ScriptedClient:
    model = "scripted"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def complete(self, messages, *, schema=None, schema_name="output", max_tokens=1500) -> LLMResult:
        self.requests.append(messages)
        return LLMResult(text=self.replies.pop(0), model=self.model, prompt_tokens=100, completion_tokens=50, cost=0.001, seconds=0.1)


def test_grader_gets_one_corrective_call_after_malformed_json():
    grader = ScriptedClient("{not json", AUDIT_Q1)
    outcome, calls = grade_reply(CASES["Q1"].enquiry, "Studio €240 + coffee €80 = €320.", grader, CAT)
    assert outcome.route == "proposal" and outcome.total == 320
    assert len(calls) == 2 and "not a valid audit" in grader.requests[1][-1]["content"]


def test_grader_rejects_an_audit_with_missing_fields():
    partial = json.dumps({"kind": "proposal", "venue_id": "V1"})
    with pytest.raises(GradingError) as error:
        grade_reply(CASES["Q1"].enquiry, "reply", ScriptedClient(partial, partial), CAT)
    assert len(error.value.calls) == 2


def test_a_grader_slip_is_not_scored_against_the_baseline():
    record = run_case(CASES["Q1"], "naive", ScriptedClient("Studio €240 + coffee €80 = €320."), ScriptedClient("{not json", AUDIT_Q1), CAT)
    assert record["usable"] and "error" not in record
    assert record["grader_cost"] == pytest.approx(0.002)


def test_a_grader_that_fails_twice_is_reported_as_a_grader_error():
    record = run_case(CASES["Q1"], "naive", ScriptedClient("A reply."), ScriptedClient("{bad", "{bad"), CAT)
    assert record["grader_error"] and not record["usable"]
    assert record["cost"] == pytest.approx(0.001) and record["grader_cost"] == pytest.approx(0.002)
    assert "could not be audited by the grader" in summarise([record | {"repeat": 0}])


def test_a_failed_extraction_still_counts_its_cost():
    record = run_case(CASES["Q1"], "pipeline", ScriptedClient("garbage", "still garbage"), None, CAT)
    assert "ExtractionError" in record["error"]
    assert record["llm_calls"] == 2 and record["cost"] == pytest.approx(0.002) and record["tokens"] == 300
