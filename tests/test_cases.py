"""The test cases themselves are checked: given the ideal brief, the engine must reach the
expected outcome. A failure here means the answer key or the engine is wrong, before any model
is involved."""

from pathlib import Path

import pytest

from brightroom.catalogue import load_catalogue
from brightroom.pipeline import run_from_brief
from brightroom.scoring import extraction_errors, load_cases, outcome_from_run, score

CAT = load_catalogue()
CASES = load_cases(Path(__file__).resolve().parents[1] / "eval" / "cases" / "dev.json")


def test_case_ids_are_unique_and_the_five_starters_are_present():
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))
    assert [case.id for case in CASES if case.source == "case"] == ["Q1", "Q2", "Q3", "Q4", "Q5"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_engine_reaches_the_expected_outcome_from_the_ideal_brief(case):
    result = run_from_brief(case.enquiry, case.brief.to_brief(), None, CAT, "template", case.brief.issues())
    failed = [f"{r.name}: {r.detail}" for r in score(case.expect, outcome_from_run(result)) if not r.passed]
    assert not failed, failed
    assert result.draft_violations == []


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_ideal_brief_matches_itself(case):
    assert extraction_errors(case.brief, case.brief.to_brief()) == []


def test_scorer_marks_the_errors_that_would_mislead_a_client():
    from brightroom.scoring import Expect, Outcome, critical_error, usable

    expect = Expect(route="infeasible", total=940, unmet=["budget"], forbid_venues=["V1"])
    hidden = score(expect, Outcome(route="proposal", total=780, venues_offered=["V1"], promises=["discount_offered"]))
    failed = {r.name: r.critical for r in hidden if not r.passed}
    assert failed == {"route": True, "total": True, "reasons": False, "no_unsuitable_venue": True, "no_promise": True}
    assert critical_error(hidden) and not usable(hidden)
    wrong_route = score(Expect(route="proposal"), Outcome(route="clarify"))
    assert not usable(wrong_route) and not critical_error(wrong_route)  # over-cautious, but nothing wrong is sent
