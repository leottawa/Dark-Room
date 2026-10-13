"""Tests for the model-facing steps, using a scripted client in place of OpenRouter."""

import json

import pytest

from brightroom.catalogue import load_catalogue
from brightroom.draft import FOOTER, Prose, prose_facts, render_draft, template_prose
from brightroom.engine import assess
from brightroom.extract import ExtractionError, audit, brief_schema, extract_brief, parse_brief
from brightroom.llm import LLMResult
from brightroom.models import Accessibility, Brief, Budget, DateMention, Duration, Headcount, OtherRequest, ServiceWish
from brightroom.pipeline import run_from_brief, run_pipeline
from brightroom.textutil import numbers_in, quote_in
from brightroom.verify import check_draft, check_prose
from brightroom.view import highlight

CAT = load_catalogue()
Q1 = "10 people, 4 hours, coffee for everyone. Maximum budget €350."
Q5 = "30 people for 6 hours with lunch. We can spend €1,500. Please confirm it today."
GOOD_PROSE = json.dumps({"opening": "Thank you for your enquiry about a workshop for 10 people.", "body": "A draft package matching your request follows, with a point for you to confirm.", "closing": "We look forward to hearing from you."})


class ScriptedClient:
    model = "scripted"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def complete(self, messages, *, schema=None, schema_name="output", max_tokens=1500) -> LLMResult:
        self.requests.append(messages)
        return LLMResult(text=self.replies.pop(0), model=self.model, prompt_tokens=100, completion_tokens=50, cost=0.0001, seconds=0.2)


def q1_brief(headcount_quote="10 people") -> Brief:
    return Brief(
        headcount=Headcount(min=10, max=10, quote=headcount_quote),
        duration=Duration(hours=4, quote="4 hours"),
        budget=Budget(amount=350, basis="total", quote="Maximum budget €350"),
        services=[ServiceWish(service_id="S1", need="required", quote="coffee for everyone")],
    )


def q5_brief() -> Brief:
    return Brief(
        headcount=Headcount(min=30, max=30, quote="30 people"),
        duration=Duration(hours=6, quote="6 hours"),
        budget=Budget(amount=1500, basis="total", quote="We can spend €1,500"),
        services=[ServiceWish(service_id="S2", need="required", quote="with lunch")],
        other_requests=[OtherRequest(category="booking_or_date_confirmation", description="confirm it today", need="required", quote="Please confirm it today")],
    )


# --- Text helpers ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("€1,500 for 30 people", {1500, 30}), ("12.50 each", {12.5}), ("4,5 hours", {4.5}), ("up to 1.5k", {1500}), ("ends at 350.", {350}), ("none", set())],
)
def test_numbers_in(text, expected):
    assert numbers_in(text) == expected


def test_spaced_thousands_only_when_loose():
    assert 1500 in numbers_in("budget 1 500 €", loose=True)
    assert 1500 not in numbers_in("budget 1 500 €")


def test_quote_matching_ignores_case_and_typography():
    assert quote_in("“Coffee for Everyone.”", Q1)
    assert not quote_in("coffee and lunch", Q1)
    assert not quote_in("...", Q1)


# --- Extraction -----------------------------------------------------------------------------------


def test_schema_is_strict_everywhere():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(brief_schema(CAT))


def test_clean_extraction_takes_one_call_and_raises_no_issue():
    client = ScriptedClient(q1_brief().model_dump_json())
    extraction = extract_brief(Q1, client, CAT)
    assert len(extraction.calls) == 1 and extraction.issues == []
    assert extraction.brief.headcount.max == 10
    assert "Enquiry: " + Q1 in client.requests[0][1]["content"]


def test_misquote_triggers_one_corrective_call():
    client = ScriptedClient(q1_brief("ten delegates").model_dump_json(), q1_brief().model_dump_json())
    extraction = extract_brief(Q1, client, CAT)
    assert len(extraction.calls) == 2 and extraction.issues == []
    assert "not exact spans" in client.requests[1][-1]["content"]


def test_persistent_misquote_is_kept_and_flagged():
    bad = q1_brief("ten delegates").model_dump_json()
    extraction = extract_brief(Q1, ScriptedClient(bad, bad), CAT)
    assert [i.kind for i in extraction.issues] == ["quote_not_found"]
    assert assess(extraction.brief, CAT, extraction.issues).warnings


def test_unusable_output_twice_raises():
    with pytest.raises(ExtractionError) as error:
        extract_brief(Q1, ScriptedClient("not json", '{"services": [{"service_id": "S9"}]}'), CAT)
    assert len(error.value.calls) == 2


def test_parse_rejects_what_the_engine_cannot_use():
    for broken in ({"headcount": {"min": 0, "max": 0}}, {"budget": {"amount": -5}}, {"venue": {"venue_id": "V9", "need": "required"}}):
        with pytest.raises(ValueError):
            parse_brief(json.dumps(broken), CAT)


def test_parse_merges_a_repeated_service_keeping_the_stronger_wish():
    raw = {"services": [{"service_id": "S3", "need": "preferred"}, {"service_id": "S3", "need": "required"}]}
    brief = parse_brief(json.dumps(raw), CAT)
    assert [(s.service_id, s.need) for s in brief.services] == [("S3", "required")]


def test_values_not_literally_in_the_quote_are_marked_interpreted():
    enquiry = "We are a dozen people and want a full day with food."
    brief = Brief(
        headcount=Headcount(min=12, max=12, quote="a dozen people"),
        duration=Duration(hours=8, quote="a full day"),
        services=[ServiceWish(service_id="S2", need="required", quote="with food")],
    )
    issues = audit(brief, enquiry, CAT)
    assert [(i.field, i.kind) for i in issues] == [("headcount", "interpreted"), ("duration", "interpreted"), ("services.S2", "interpreted")]
    assert issues[0].message == 'We read "a dozen people" as 12 attendees.'


def test_values_without_a_quote_are_flagged():
    brief = Brief(accessibility=Accessibility(need="required"), date=DateMention(mentioned=True, quote="in March"))
    assert [i.field for i in audit(brief, Q1, CAT)] == ["accessibility", "date"]


# --- Drafting and verification -----------------------------------------------------------------------


def facts_for(brief: Brief):
    assessment = assess(brief, CAT)
    return assessment, prose_facts(assessment, brief, CAT)


@pytest.mark.parametrize("brief", [q1_brief(), q5_brief(), Brief()])
def test_template_prose_passes_verification_on_every_route(brief):
    assessment, facts = facts_for(brief)
    prose = template_prose(assessment, brief)
    assert check_prose(prose, facts, CAT) == []
    assert check_draft(render_draft(assessment, brief, prose, CAT), assessment) == []


@pytest.mark.parametrize(
    "body, code",
    [
        ("Your package comes to €300 in total.", "unsupported_figure"),
        ("We can confirm the Studio for you.", "commitment"),
        ("The Studio is available on your date.", "commitment"),
        ("We are happy to offer a discount.", "commitment"),
        ("The Loft would suit your team.", "item_not_in_draft"),
        ("Parking is included.", "unlisted_item"),
        ("- Studio\n- Coffee", "format"),
    ],
)
def test_verifier_catches_what_must_not_reach_a_client(body, code):
    _, facts = facts_for(q1_brief())
    prose = Prose(opening="Thank you for your enquiry.", body=body, closing="We look forward to hearing from you.")
    assert code in [v.code for v in check_prose(prose, facts, CAT)]


def test_verifier_accepts_figures_and_items_from_the_facts():
    _, facts = facts_for(q1_brief())
    prose = Prose(opening="Thank you for your enquiry for 10 people.", body="The Studio with a coffee break comes to €320, within your €350 budget.", closing="Do tell us if anything should change.")
    assert check_prose(prose, facts, CAT) == []


def test_draft_for_q1_shows_the_package_the_assumption_and_the_other_scenario():
    brief = q1_brief()
    assessment = assess(brief, CAT)
    draft = render_draft(assessment, brief, template_prose(assessment, brief), CAT)
    for expected in (
        "- Studio (4-hour session, up to 12 people, stairs only, Wi-Fi included): €240",
        "- Coffee break, 10 x €8: €80. One coffee break for every attendee.",
        "Total: €320",
        "(€30 remaining)",
        "Step-free access was not mentioned: the Studio is stairs only.",
        "If step-free access is needed: Loft, coffee break (10 x €8); 10 attendees, 4-hour session; total €500.",
        FOOTER,
    ):
        assert expected in draft


def test_draft_for_q5_explains_and_offers_the_nearest_package():
    brief = q5_brief()
    assessment = assess(brief, CAT)
    draft = render_draft(assessment, brief, template_prose(assessment, brief), CAT)
    assert "30 attendees exceed the capacity of our largest venue, the Loft (maximum 24)." in draft
    assert "6 hours requested; our venues are offered for 4 hours and no longer-duration price exists." in draft
    assert "A. Shorten the session to 4 hours and limit attendance to 24: total €852" in draft
    assert "We cannot confirm or book at this stage" in draft
    assert "Proposed package" not in draft


# --- Whole pipeline -------------------------------------------------------------------------------


def test_pipeline_end_to_end_with_model_prose():
    result = run_pipeline(Q1, ScriptedClient(q1_brief().model_dump_json(), GOOD_PROSE), CAT)
    assert result.assessment.status == "provisional"
    assert result.prose_source == "model" and result.draft_violations == []
    assert "with a point for you to confirm" in result.draft and "Total: €320" in result.draft
    assert [s.name for s in result.steps] == ["extract", "assess", "write", "verify"]
    assert result.llm_calls == 2 and result.cost == pytest.approx(0.0002)


def test_prose_that_fails_twice_falls_back_to_the_template():
    bad = json.dumps({"opening": "Thanks.", "body": "Your date is confirmed and the total is €300.", "closing": "Bye."})
    result = run_pipeline(Q1, ScriptedClient(q1_brief().model_dump_json(), bad, bad), CAT)
    assert result.prose_source == "template_fallback"
    assert len(result.prose_violations) == 2
    assert "confirmed" not in result.draft.replace(FOOTER, "") and "€300" not in result.draft


def test_prose_corrected_on_the_second_try_is_used():
    bad = json.dumps({"opening": "Thanks.", "body": "We can book this for you.", "closing": "Bye."})
    client = ScriptedClient(q1_brief().model_dump_json(), bad, GOOD_PROSE)
    result = run_pipeline(Q1, client, CAT)
    assert result.prose_source == "model_retry"
    assert "broke these rules" in client.requests[2][1]["content"]


def test_edited_brief_reruns_without_a_model():
    brief = q1_brief()
    brief.accessibility = Accessibility(need="required", quote=None)
    result = run_from_brief(Q1, brief, None, CAT)
    assert result.assessment.route == "infeasible"  # the Loft at €500 is over the €350 budget
    assert result.prose_source == "template" and result.llm_calls == 0


# --- Regressions -------------------------------------------------------------------------------------


def loft_brief() -> Brief:
    return Brief(
        headcount=Headcount(min=20, max=20, quote="20 people"),
        duration=Duration(hours=4, quote="4 hours"),
        budget=Budget(amount=700, basis="total", quote="Budget €700"),
        services=[ServiceWish(service_id="S1", need="required", quote="coffee")],
    )


@pytest.mark.parametrize(
    "opening",
    [
        "Thank you for thinking of our studio for your team's workshop.",  # BrightRoom calls itself a studio
        "20 colleagues for a workshop sounds wonderful, and thank you for your enquiry.",  # not a list item
    ],
)
def test_verifier_does_not_reject_ordinary_sentences(opening):
    _, facts = facts_for(loft_brief())
    prose = Prose(opening=opening, body="A draft package matching your request follows.", closing="We look forward to hearing from you.")
    assert check_prose(prose, facts, CAT) == []


@pytest.mark.parametrize("body, code", [("The Studio would suit your team.", "item_not_in_draft"), ("1. Loft\n2. Coffee", "format"), ("- Loft", "format")])
def test_verifier_still_catches_the_other_venue_and_lists(body, code):
    _, facts = facts_for(loft_brief())
    prose = Prose(opening="Thank you for your enquiry.", body=body, closing="We look forward to hearing from you.")
    assert code in [v.code for v in check_prose(prose, facts, CAT)]


def test_short_event_draft_says_it_is_priced_as_a_full_session():
    brief = q1_brief()
    brief.duration = Duration(hours=2, quote="2 hours")
    assessment = assess(brief, CAT)
    draft = render_draft(assessment, brief, template_prose(assessment, brief), CAT)
    assert "2 hours requested; venues are booked as 4-hour sessions at the listed price." in draft
    assert check_draft(draft, assessment) == []


def test_template_prose_states_a_headcount_range_as_a_range():
    brief = Brief(headcount=Headcount(min=10, max=14), duration=Duration(hours=4))
    assessment = assess(brief, CAT)
    assert template_prose(assessment, brief).opening == "Thank you for your enquiry about a workshop for 10 to 14 people."


def test_highlight_finds_quotes_whatever_the_typography():
    enquiry = "Hi — we’re 12 people,  for 4 hours."
    assert highlight(enquiry, ["we're 12 people", "for 4 hours."]) == "Hi — <mark>we’re 12 people</mark>,  <mark>for 4 hours</mark>."
    assert highlight("<b>10</b> people", ["10"]) == "&lt;b&gt;<mark>10</mark>&lt;/b&gt; people"


def test_openrouter_client_retries_a_body_that_is_not_json(monkeypatch):
    import httpx

    from brightroom import llm

    replies = iter([httpx.Response(200, text="<html>Bad gateway</html>"), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"cost": 0.001}})])
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    client = llm.OpenRouterClient(api_key="test")
    client._http = httpx.Client(transport=httpx.MockTransport(lambda request: next(replies)))
    result = client.complete([{"role": "user", "content": "hi"}])
    assert result.text == "ok" and result.cost == 0.001
