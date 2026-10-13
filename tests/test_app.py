"""Smoke test of the review screen, run headless with no API key."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.chdir(tmp_path)  # audit logs go to a temporary folder
    return AppTest.from_file(APP, default_timeout=30).run()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_brief_entered_by_hand_reaches_a_verdict_and_gates_approval(app):
    assert not app.exception
    assert button(app, "Prepare draft").disabled  # no key: nothing can be sent to a model

    app.text_area(key="enquiry").set_value("10 people, coffee, budget 350").run()
    button(app, "Enter brief by hand").click().run()
    assert not app.exception
    assert "MORE INFORMATION NEEDED" in app.info[0].value

    next(n for n in app.number_input if n.label.startswith("Attendees, from")).set_value(10)
    next(n for n in app.number_input if n.label.startswith("Budget")).set_value(350.0)
    next(s for s in app.selectbox if s.label == "Coffee break").set_value("required")
    button(app, "Recalculate with this brief").click().run()
    assert not app.exception
    assert "PROVISIONAL" in app.warning[0].value and "€320" in app.warning[0].value
    draft = next(t for t in app.text_area if t.label.startswith("Edit if needed"))
    assert "Total: €320" in draft.value

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
    import time

    from brightroom.catalogue import load_catalogue
    from brightroom.models import Brief, Budget, Duration, Headcount, ServiceWish
    from brightroom.pipeline import run_from_brief

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
