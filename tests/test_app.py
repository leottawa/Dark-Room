"""Smoke test of the review screen, run headless with no API key."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from brightroom.catalogue import load_catalogue
from brightroom.models import Accessibility, Brief, Budget, Duration, Headcount, ServiceWish
from brightroom.pipeline import run_from_brief

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.chdir(tmp_path)  # audit logs go to a temporary folder
    return AppTest.from_file(APP, default_timeout=30).run()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def html(app) -> str:
    """Text of the custom HTML blocks (verdict card, assessment, draft preview)."""
    bodies = (element.proto.body for element in app.get("html"))
    return "\n".join(body for body in bodies if not body.lstrip().startswith("<style"))


def test_brief_entered_by_hand_reaches_a_verdict_and_gates_approval(app):
    assert not app.exception
    assert button(app, "Prepare draft").disabled  # no key: nothing can be sent to a model

    app.text_area(key="enquiry").set_value("10 people, coffee, budget 350").run()
    button(app, "Enter brief by hand").click().run()
    assert not app.exception
    assert "MORE INFORMATION NEEDED" in html(app)

    next(n for n in app.number_input if n.label.startswith("Attendees, from")).set_value(10)
    next(n for n in app.number_input if n.label.startswith("Budget")).set_value(350.0)
    next(s for s in app.selectbox if s.label == "Coffee break").set_value("required")
    button(app, "Recalculate with this brief").click().run()
    assert not app.exception
    verdict = next(block for block in html(app).split("\n") if "br-verdict" in block)
    assert "PROVISIONAL" in verdict and "€320" in verdict
    draft = next(t for t in app.text_area if t.label.startswith("Edit if needed"))
    assert "Total: €320" in draft.value
    assert "Total: €320" in html(app)  # the email preview shows the same text

    approve = button(app, "Approve draft")
    assert approve.disabled
    for box in app.checkbox:
        if box.label.startswith("I "):
            box.check()
    app.run()
    approve = button(app, "Approve draft")
    assert not approve.disabled
    approve.click().run()
    assert not app.exception
    assert "nothing has been sent or booked" in app.success[0].value
    assert len(list(Path("runs").glob("*_approved.json"))) == 1


def test_model_box_defaults_to_the_model_set_in_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("BRIGHTROOM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert next(t for t in app.text_input if t.label.startswith("Model")).value == "openai/gpt-4o-mini"


def test_recalculating_an_untouched_brief_keeps_the_clients_budget_words(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.chdir(tmp_path)
    enquiry = "10 people, 4 hours, coffee. We can spend 350."
    # The model may leave the budget basis empty; the engine reads it as a total.
    brief = Brief(
        headcount=Headcount(min=10, max=10, quote="10 people"),
        duration=Duration(hours=4, quote="4 hours"),
        budget=Budget(amount=350, basis=None, quote="We can spend 350"),
        services=[ServiceWish(service_id="S1", need="required", quote="coffee")],
    )
    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state["result"] = run_from_brief(enquiry, brief, None, load_catalogue(), "template")
    app.session_state["version"] = 1
    app.session_state["shown_at"] = time.time()
    app.run()
    button(app, "Recalculate with this brief").click().run()
    assert not app.exception
    budget = app.session_state["result"].brief.budget
    assert (budget.amount, budget.basis, budget.quote) == (350, None, "We can spend 350")


ROUTES = {
    "fits": ("FITS THE BRIEF", dict(headcount=Headcount(min=10, max=10), duration=Duration(hours=4), budget=Budget(amount=350, basis="total"), accessibility=Accessibility(need="not_needed"), services=[ServiceWish(service_id="S1")])),
    "provisional": ("PROVISIONAL", dict(headcount=Headcount(min=10, max=14), duration=Duration(hours=4), budget=Budget(amount=600, basis="total"), services=[ServiceWish(service_id="S1")])),
    "over budget": ("CANNOT MEET THE BRIEF", dict(headcount=Headcount(min=20, max=20), duration=Duration(hours=4), budget=Budget(amount=850, basis="total"), services=[ServiceWish(service_id="S1"), ServiceWish(service_id="S2")])),
    "no venue": ("CANNOT MEET THE BRIEF", dict(headcount=Headcount(min=30, max=30), duration=Duration(hours=6), budget=Budget(amount=1500, basis="total"), services=[ServiceWish(service_id="S2")])),
    "clarify": ("MORE INFORMATION NEEDED", dict()),
}


@pytest.mark.parametrize("route", ROUTES)
def test_every_verdict_renders_its_card_figures_and_draft(route, monkeypatch, tmp_path):
    label, fields = ROUTES[route]
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.chdir(tmp_path)
    result = run_from_brief("An enquiry.", Brief(**fields), None, load_catalogue(), "template")
    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state["result"] = result
    app.session_state["version"] = 1
    app.session_state["shown_at"] = time.time()
    app.run()
    assert not app.exception
    page = html(app)
    assert label in page and 'class="br-figures"' in page
    assert ('class="br-option"' in page) == bool(result.assessment.alternatives)
    assert "Subject:" not in page  # the preview shows the subject in the email header, not as a body line
