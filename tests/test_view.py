"""The HTML blocks of the review screen: right content, and client text always escaped."""

from brightroom.catalogue import load_catalogue
from brightroom.engine import assess
from brightroom.models import Brief, Budget, Duration, Headcount, ServiceWish
from brightroom.view import brief_html, brief_rows, email_html, figures_html, header_html, options_html, package_html, quote_box_html, verdict_html

CAT = load_catalogue()
HOSTILE = '<img src=x onerror="alert(1)"> & <b>bold</b>'


def test_header_marks_finished_steps_and_the_current_one():
    page = header_html(2)
    assert page.count('class="br-step done"') == 2 and page.count('class="br-step current"') == 1
    assert header_html(4).count('class="br-step done"') == 4


def test_verdict_shows_the_route_label_and_the_reason_without_its_prefix():
    page = verdict_html("infeasible", "Cannot meet the brief as stated: €90 over budget", "Read by model")
    assert "CANNOT MEET THE BRIEF" in page and ">€90 over budget<" in page
    assert "as stated:" not in page


def test_package_is_an_invoice_with_quantities_and_the_total():
    a = assess(Brief(headcount=Headcount(min=20, max=20), duration=Duration(hours=4), budget=Budget(amount=850, basis="total"), services=[ServiceWish(service_id="S1"), ServiceWish(service_id="S2")]), CAT)
    page = package_html(a.package, CAT, [CAT.price_note])
    assert "20 × €8" in page and "20 × €18" in page and ">€940<" in page
    assert page.count("<tr><td>") == 3  # venue, coffee and lunch rows; header and total rows differ


def test_options_say_whether_they_change_a_requirement():
    a = assess(Brief(headcount=Headcount(min=20, max=20), duration=Duration(hours=4), budget=Budget(amount=850, basis="total"), services=[ServiceWish(service_id="S1"), ServiceWish(service_id="S2")]), CAT)
    page = options_html(a.alternatives, CAT)
    assert page.count('class="br-option"') == len(a.alternatives)
    assert "Keeps every requirement" in page and "Changes a requirement" in page


def test_email_preview_lifts_the_subject_and_keeps_the_body():
    page = email_html("Subject: Draft proposal\n\nHello,\n\nTotal: €320")
    assert 'class="br-email-subject">Draft proposal<' in page
    assert "Hello,\n\nTotal: €320" in page and "Subject:" not in page


def test_client_text_is_escaped_everywhere_it_appears():
    assert "<img" not in quote_box_html(HOSTILE, [])
    assert "<img" not in email_html(f"Subject: {HOSTILE}\n\n{HOSTILE}")
    brief = Brief(headcount=Headcount(min=10, max=10, quote=HOSTILE))
    assert "<img" not in brief_html(brief_rows(brief, [], CAT))
    assert "<img" not in figures_html([("Label", HOSTILE, HOSTILE, "red")])
    assert "&lt;img" in email_html(HOSTILE)
