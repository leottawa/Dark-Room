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
