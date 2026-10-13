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


# --- Public deployment: the owner's demo key and keys pasted by visitors ---------------------------------

DEMO_KEY = "sk-or-v1-demo-key-for-tests-0123456789"


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """A clean server: no key in the environment, empty shared caches, audit logs in a temp folder."""
    import streamlit as st

    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("BRIGHTROOM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    st.cache_resource.clear()
    yield monkeypatch
    st.cache_resource.clear()


def rendered(app) -> str:
    """Everything a visitor can read on the page, apart from what they typed into the key field."""
    parts = [m.value for m in app.markdown] + [c.value for c in app.caption] + [html(app)]
    parts += [t.value for t in app.text_input if "API key" not in t.label] + [t.value for t in app.text_area]
    return "\n".join(str(p) for p in parts)


def key_box(app):
    return next(t for t in app.text_input if "API key" in t.label)


def model_box(app):
    return next(t for t in app.text_input if t.label.startswith("Model"))


def test_demo_key_locks_the_model_and_is_never_shown(fresh):
    fresh.setenv("OPENROUTER_API_KEY", DEMO_KEY)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert model_box(app).disabled and model_box(app).value == "mistralai/mistral-small-3.2-24b-instruct"
    assert key_box(app).value == "" and "optional" in key_box(app).label
    assert "Demo key active" in rendered(app) and "15 model runs left" in rendered(app)
    assert DEMO_KEY not in rendered(app)
    app.text_area(key="enquiry").set_value("10 people, coffee, budget 350").run()
    assert not button(app, "Prepare draft").disabled


def test_demo_key_can_come_from_streamlit_secrets(fresh):
    app = AppTest.from_file(APP, default_timeout=30)
    app.secrets["OPENROUTER_API_KEY"] = DEMO_KEY
    app.run()
    assert not app.exception and "Demo key active" in rendered(app)


def test_a_pasted_key_unlocks_the_model_and_stays_in_the_session(fresh):
    fresh.setenv("OPENROUTER_API_KEY", DEMO_KEY)
    app = AppTest.from_file(APP, default_timeout=30).run()
    key_box(app).set_value("sk-or-v1-visitor-key").run()
    assert not model_box(app).disabled
    assert app.session_state["own_client"].api_key == "sk-or-v1-visitor-key"
    assert "Using your key" in rendered(app) and "sk-or-v1-visitor-key" not in rendered(app)
    other = AppTest.from_file(APP, default_timeout=30).run()  # another visitor's session
    assert "own_client" not in other.session_state and "Demo key active" in rendered(other)


def test_without_any_key_the_brief_can_still_be_entered_by_hand(fresh):
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert "No API key" in rendered(app) and not model_box(app).disabled
    app.text_area(key="enquiry").set_value("10 people").run()
    assert button(app, "Prepare draft").disabled and not button(app, "Enter brief by hand").disabled


def test_the_demo_limit_stops_model_runs(fresh):
    fresh.setenv("OPENROUTER_API_KEY", DEMO_KEY)
    fresh.setenv("BRIGHTROOM_DEMO_RUNS_PER_SESSION", "0")
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_area(key="enquiry").set_value("10 people").run()
    assert "Demo limit reached" in rendered(app) and button(app, "Prepare draft").disabled


def test_a_rejected_key_is_explained_and_the_demo_run_is_counted(fresh):
    from brightroom import llm

    def reject(self, *args, **kwargs):
        raise llm.LLMError('OpenRouter HTTP 401: {"error":{"message":"User not found.","code":401}}')

    fresh.setattr(llm.OpenRouterClient, "complete", reject)
    fresh.setenv("OPENROUTER_API_KEY", DEMO_KEY)
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_area(key="enquiry").set_value("10 people, coffee, budget 350").run()
    button(app, "Prepare draft").click().run()
    assert not app.exception
    assert "OpenRouter rejected the API key" in app.error[0].value
    assert app.session_state["demo_runs"] == 1 and "14 model runs left" in rendered(app)
