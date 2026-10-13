"""Review screen for the salesperson.

The verdict is shown before the draft, the brief sits beside the client's own words, and nothing
leaves the tool: the salesperson approves, copies and sends by hand.
"""

from __future__ import annotations

import time

import streamlit as st

from brightroom.catalogue import hours, load_catalogue, money
from brightroom.extract import ExtractionError
from brightroom.llm import DEFAULT_MODEL, LLMError, OpenRouterClient, api_key_available
from brightroom.models import Accessibility, Brief, Budget, DateMention, Duration, Headcount, OtherRequest, ServiceWish, VenueWish
from brightroom.pipeline import RunResult, run_from_brief, run_pipeline, save_run
from brightroom.view import brief_rows, edit_share, highlight, quotes_of

CAT = load_catalogue()
SAMPLES = {
    "Q1": "10 people, 4 hours, coffee for everyone. Maximum budget €350.",
    "Q2": "20 people, 4 hours, coffee and lunch are required. Total budget €850.",
    "Q3": "10 people, 4 hours, wheelchair access, coffee and a projector. Budget €600.",
    "Q4": "Can you organise a team workshop next month? Please send me your best offer.",
    "Q5": "30 people for 6 hours with lunch. We can spend €1,500. Please confirm it today.",
}
ICONS = {"pass": ":green[:material/check_circle:]", "fail": ":red[:material/cancel:]", "unknown": ":orange[:material/help:]", "n/a": ":gray[:material/remove:]"}
CATEGORIES = ["unlisted_service", "discount", "booking_or_date_confirmation", "vegetarian_vegan", "allergen_guarantee", "other"]

st.set_page_config(page_title="BrightRoom proposal desk", page_icon=":material/event_note:", layout="wide")
state = st.session_state
state.setdefault("result", None)
state.setdefault("version", 0)
state.setdefault("enquiry", "")
state.setdefault("approved", None)


@st.cache_resource
def get_client(model: str, key_present: bool) -> OpenRouterClient | None:
    # key_present is part of the cache key so that adding the key to .env takes effect on reload.
    return OpenRouterClient(model=model) if key_present else None


def show(result: RunResult, event: str) -> None:
    state.result = result
    state.version += 1
    state.shown_at = time.time()
    state.approved = None
    save_run(result, event=event)


# --- Sidebar ----------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model (OpenRouter id)", value=DEFAULT_MODEL)
    client = get_client(model, api_key_available())
    if client:
        st.caption(":green[:material/check_circle:] API key found")
    else:
        st.caption(":orange[:material/warning:] No API key: add OPENROUTER_API_KEY to .env. You can still enter a brief by hand.")
    prose_mode = st.radio("Connecting sentences", ["model", "template"], format_func={"model": "Written by the model, then verified", "template": "Fixed template (no model call)"}.get)
    st.divider()
    st.subheader("What this tool does not do")
    st.markdown(
        "- It does not send anything: you copy the approved draft yourself.\n"
        "- It does not check availability, hold dates or book venues.\n"
        "- It offers nothing outside the catalogue: no discounts, no unlisted services.\n"
        "- It can misread an enquiry: compare the brief with the client's words."
    )

# --- Enquiry ----------------------------------------------------------------------------------------

st.title("BrightRoom proposal desk")
st.caption("From a client enquiry to a draft the salesperson can check and approve.")

picked = st.pills("Load a case enquiry", list(SAMPLES), selection_mode="single")
if picked and state.get("picked") != picked:
    state.picked, state.enquiry = picked, SAMPLES[picked]
enquiry = st.text_area("Client enquiry", key="enquiry", height=110, placeholder="Paste the client's message here.")

left, right, _ = st.columns([1, 1, 3])
if left.button("Prepare draft", type="primary", disabled=not (client and enquiry.strip()), width="stretch"):
    try:
        with st.spinner("Reading the enquiry and preparing the draft…"):
            show(run_pipeline(enquiry, client, CAT, prose_mode), "draft")
    except (ExtractionError, LLMError) as error:
        st.error(f"The enquiry could not be read automatically ({error}). Enter the brief by hand below.")
        show(run_from_brief(enquiry, Brief(), None, CAT, "template"), "manual")
if right.button("Enter brief by hand", width="stretch"):
    show(run_from_brief(enquiry, Brief(), None, CAT, "template"), "manual")

result: RunResult | None = state.result
if result is None:
    st.stop()
a, brief, version = result.assessment, result.brief, state.version

# --- Verdict first ------------------------------------------------------------------------------------

st.divider()
banner = {"fits": st.success, "provisional": st.warning, "infeasible": st.error, "clarify": st.info}[a.status]
label = {"fits": "FITS THE BRIEF", "provisional": "PROVISIONAL", "infeasible": "CANNOT MEET THE BRIEF", "clarify": "MORE INFORMATION NEEDED"}[a.status]
banner(f"**{label}**  \n{a.headline}")
for warning in a.warnings:
    st.warning(f":material/report: Unverified reading. {warning}")

understood, decided = st.columns(2, gap="large")

with understood:
    st.subheader("What the client wrote")
    with st.container(border=True):
        st.markdown(highlight(result.enquiry, quotes_of(brief)) or "*No enquiry text.*", unsafe_allow_html=True)
    st.caption("Highlighted words are the ones the brief is based on.")
    st.subheader("What we understood")
    st.dataframe(brief_rows(brief, result.issues, CAT), hide_index=True, width="stretch")

    with st.expander("Correct the brief", expanded=result.enquiry.strip() == "" or a.route == "clarify" and not client):
        with st.form(f"brief_{version}"):
            c1, c2, c3 = st.columns(3)
            n_min = c1.number_input("Attendees, from (0 = not stated)", min_value=0, value=brief.headcount.min or 0)
            n_max = c2.number_input("Attendees, to", min_value=0, value=brief.headcount.max or 0)
            approximate = c3.checkbox("Approximate", value=brief.headcount.approximate)
            c1, c2, c3 = st.columns(3)
            duration = c1.number_input("Hours (0 = not stated)", min_value=0.0, step=0.5, value=float(brief.duration.hours or 0))
            amount = c2.number_input("Budget in € (0 = not stated)", min_value=0.0, step=10.0, value=float(brief.budget.amount or 0))
            basis = c3.selectbox("Budget is", ["total", "per_person"], index=1 if brief.budget.basis == "per_person" else 0, format_func=lambda b: b.replace("_", " "))
            c1, c2, c3 = st.columns(3)
            needs = ["not_mentioned", "not_needed", "required", "preferred"]
            access = c1.selectbox("Step-free access", needs, index=needs.index(brief.accessibility.need), format_func=lambda n: n.replace("_", " "))
            venue_ids = [None, *CAT.venue_ids]
            venue_id = c2.selectbox("Venue asked for", venue_ids, index=venue_ids.index(brief.venue.venue_id), format_func=lambda v: "none" if v is None else CAT.venue(v).name)
            venue_need = c3.selectbox("Venue is", ["required", "preferred"], index=1 if brief.venue.need == "preferred" else 0)
            current = {s.service_id: s for s in brief.services}
            choices = {}
            for column, service in zip(st.columns(len(CAT.services)), CAT.services):
                options = ["not requested", "required", "preferred"]
                was = current[service.id].need if service.id in current else "not requested"
                choices[service.id] = column.selectbox(service.name, options, index=options.index(was))
            date_mentioned = st.checkbox("A date or timing was mentioned", value=brief.date.mentioned)
            labels = [f"{r.category.replace('_', ' ')}: {r.description} ({r.need})" for r in brief.other_requests]
            kept = st.multiselect("Other requests to keep", labels, default=labels)
            c1, c2, c3 = st.columns([2, 2, 1])
            new_text = c1.text_input("Add another request (optional)")
            new_category = c2.selectbox("Its kind", CATEGORIES, format_func=lambda c: c.replace("_", " "))
            new_need = c3.selectbox("It is", ["preferred", "required"])
            submitted = st.form_submit_button("Recalculate with this brief", type="primary")

        if submitted:
            changed: set[str] = set()

            def keep(field: str, same: bool, quote: str | None) -> str | None:
                if not same:
                    changed.add(field)
                return quote if same else None

            lo, hi = (n_min or n_max or None), (n_max or n_min or None)
            old = brief
            new = Brief(
                language=old.language,
                headcount=Headcount(min=lo, max=hi, approximate=approximate, quote=keep("headcount", (lo, hi, approximate) == (old.headcount.min, old.headcount.max, old.headcount.approximate), old.headcount.quote)),
                duration=Duration(hours=duration or None, quote=keep("duration", (duration or None) == old.duration.hours, old.duration.quote)),
                budget=Budget(
                    amount=amount or None,
                    basis=basis if amount else None,
                    approximate=old.budget.approximate if amount else False,
                    quote=keep("budget", (amount or None, basis if amount else None) == (old.budget.amount, old.budget.basis), old.budget.quote),
                ),
                accessibility=Accessibility(need=access, quote=keep("accessibility", access == old.accessibility.need, old.accessibility.quote)),
                venue=VenueWish(
                    venue_id=venue_id,
                    need=venue_need if venue_id else "not_mentioned",
                    quote=keep("venue", (venue_id, venue_need if venue_id else "not_mentioned") == (old.venue.venue_id, old.venue.need), old.venue.quote),
                ),
                services=[
                    ServiceWish(
                        service_id=sid,
                        need=need,
                        beyond_terms=current[sid].beyond_terms if sid in current and current[sid].need == need else None,
                        quote=keep(f"services.{sid}", sid in current and current[sid].need == need, current[sid].quote if sid in current else None),
                    )
                    for sid, need in choices.items()
                    if need != "not requested"
                ],
                date=DateMention(mentioned=date_mentioned, quote=keep("date", date_mentioned == old.date.mentioned, old.date.quote)),
                other_requests=[r for r, text in zip(old.other_requests, labels) if text in kept]
                + ([OtherRequest(category=new_category, description=new_text.strip(), need=new_need)] if new_text.strip() else []),
            )
            changed |= {f"services.{sid}" for sid in current if choices[sid] == "not requested"}
            other_changed = len(new.other_requests) != len(old.other_requests) or bool(new_text.strip())
            issues = [i for i in result.issues if i.field not in changed and not (other_changed and i.field.startswith("other_requests"))]
            with st.spinner("Recalculating…"):
                show(run_from_brief(result.enquiry, new, client, CAT, prose_mode, issues), "corrected")
            st.rerun()

with decided:
    st.subheader("Checks")
    for check in a.checks:
        st.markdown(f"{ICONS[check.status]} **{check.name.capitalize()}**: {check.detail}")
    if not a.checks:
        st.markdown(f"{ICONS['unknown']} Nothing can be checked until the number of attendees is known.")
    for unmet in a.unmet:
        if unmet.kind not in ("capacity", "accessibility", "duration", "budget"):
            st.markdown(f"{ICONS['fail']} **Request**: {unmet.detail}")

    if a.package:
        st.subheader("Package proposed" if a.route == "proposal" else "The request as priced")
        rows = [{"Item": line.name, "Quantity": line.quantity, "Unit price": money(line.unit_price), "Amount": money(line.amount), "Conditions": line.conditions} for line in a.package.lines]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.markdown(f"**Total: {money(a.package.total)}** for {a.package.headcount} attendees, {hours(a.package.duration_hours)}-hour session")

    if a.assumptions:
        st.subheader("Assumptions")
        for item in a.assumptions:
            marker = ":orange[:material/priority_high:] " if item.sensitive else ""
            st.markdown(f"- {marker}{item.text}")
        if any(item.sensitive for item in a.assumptions):
            st.caption(":orange[:material/priority_high:] could change the venue or the verdict if wrong.")
    if a.scenarios:
        st.subheader("If the assumption is wrong")
        for scenario in a.scenarios:
            icon = ICONS["pass" if scenario.feasible else "fail"]
            price = f": {scenario.package.venue_name}, total {money(scenario.package.total)}" if scenario.package else ""
            st.markdown(f"{icon} **{scenario.label}**{price}. {scenario.note}")
    if a.alternatives:
        st.subheader("Options to offer")
        for letter, alt in zip("ABCDEFGH", a.alternatives):
            flag = "  :orange[changes a stated requirement]" if alt.changes_requirement else ""
            st.markdown(f"**{letter}. {alt.label}**: total {money(alt.package.total)}{flag}  \n{' '.join(alt.trade_offs)}")
    if a.addons:
        st.subheader("Optional additions")
        for addon in a.addons:
            st.markdown(f"- {addon.name}: {money(addon.amount)}. {addon.reason}")
    if a.notices:
        st.subheader("Requests outside the catalogue or your authority")
        for notice in a.notices:
            icon = ICONS["pass"] if notice.offerable else ICONS["fail"]
            quote = f'  \n:gray[Client: "{notice.quote}"]' if notice.quote else ""
            st.markdown(f"{icon} {notice.text}{quote}")
    if a.questions:
        st.subheader("Questions for the client")
        for question in a.questions:
            st.markdown(f"- {'**Blocking:** ' if question.blocking else ''}{question.text}")

# --- Draft and approval ---------------------------------------------------------------------------------

st.divider()
st.subheader("Draft for the client")
sources = {
    "model": "Connecting sentences written by the model and verified.",
    "model_retry": "Connecting sentences rewritten once after failing verification, then verified.",
    "template_fallback": "The model's sentences failed verification twice: the fixed template was used instead.",
    "template": "Fixed template sentences (no model call).",
}
st.caption(f"{sources[result.prose_source]} All figures, items, assumptions and notices are inserted by code.")
for violation in result.draft_violations:
    st.error(f"Draft check failed: {violation.detail}")
text = st.text_area("Edit if needed, then approve", value=result.draft, height=520, key=f"draft_{version}")

acknowledgements = ["I have compared the brief with the client's words.", f"I have read the verdict: {a.headline}."]
if any(item.sensitive for item in a.assumptions):
    acknowledgements.append("I have read the points that still need the client's confirmation.")
if a.warnings:
    acknowledgements.append("I have checked the readings that could not be verified.")
acknowledgements.append("I know that availability has not been checked and nothing is booked.")
ticks = [st.checkbox(item, key=f"ack_{version}_{i}") for i, item in enumerate(acknowledgements)]

if st.button("Approve draft", type="primary", disabled=not all(ticks) or state.approved == text):
    review_seconds = round(time.time() - state.shown_at, 1)
    save_run(result, event="approved", extra={"review_seconds": review_seconds, "edit_share": edit_share(result.draft, text), "final_draft": text})
    state.approved = text
if state.approved:
    st.success("Approved. Copy the text below into your email: nothing has been sent or booked.")
    st.code(state.approved, language=None, wrap_lines=True)

with st.expander("Run details"):
    st.dataframe(
        [{"Step": s.name, "Seconds": round(s.seconds, 2), "Model calls": s.llm_calls, "Tokens in": s.prompt_tokens, "Tokens out": s.completion_tokens, "Cost ($)": round(s.cost, 6)} for s in result.steps],
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Total {result.seconds:.2f} s, {result.llm_calls} model calls, ${result.cost:.6f}. Model: {result.model or 'none'}.")
    for attempt, violations in enumerate(result.prose_violations, 1):
        for violation in violations:
            st.markdown(f"- Attempt {attempt}: {violation.code}: {violation.detail}")
