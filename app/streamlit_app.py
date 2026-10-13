"""Review screen for the salesperson.

The verdict is shown before the draft, the brief sits beside the client's own words, and nothing
leaves the tool: the salesperson approves, copies and sends by hand.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from datetime import date
from pathlib import Path

import streamlit as st

# The package lives in src/. Put it on the path so the app also runs where the project itself is not
# installed: Streamlit Community Cloud, or `streamlit run app/streamlit_app.py` from a plain checkout.
SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brightroom.catalogue import hours, load_catalogue, money  # noqa: E402
from brightroom.extract import ExtractionError  # noqa: E402
from brightroom.llm import DEFAULT_MODEL, LLMError, OpenRouterClient, api_key_available  # noqa: E402
from brightroom.models import Accessibility, Assessment, Brief, Budget, DateMention, Duration, Headcount, OtherRequest, ServiceWish, VenueWish  # noqa: E402
from brightroom.pipeline import RunResult, run_from_brief, run_pipeline, save_run  # noqa: E402
from brightroom.view import (  # noqa: E402
    STEPS,
    brief_html,
    brief_rows,
    catalogue_html,
    checks_html,
    edit_share,
    email_html,
    figures_html,
    header_html,
    items_html,
    label_html,
    options_html,
    package_html,
    quote_box_html,
    quotes_of,
    sidebar_brand_html,
    verdict_html,
)

CAT = load_catalogue()
STYLE = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
SAMPLES = {
    "Q1": "10 people, 4 hours, coffee for everyone. Maximum budget €350.",
    "Q2": "20 people, 4 hours, coffee and lunch are required. Total budget €850.",
    "Q3": "10 people, 4 hours, wheelchair access, coffee and a projector. Budget €600.",
    "Q4": "Can you organise a team workshop next month? Please send me your best offer.",
    "Q5": "30 people for 6 hours with lunch. We can spend €1,500. Please confirm it today.",
}
SAMPLE_LABELS = {"Q1": "Q1 · Routine", "Q2": "Q2 · Over budget", "Q3": "Q3 · Step-free access", "Q4": "Q4 · Vague request", "Q5": "Q5 · Too big, too long"}
CATEGORIES = ["unlisted_service", "discount", "booking_or_date_confirmation", "vegetarian_vegan", "allergen_guarantee", "other"]
CONSTRAINTS = ("capacity", "accessibility", "duration", "budget")
PROSE_SOURCES = {
    "model": "Connecting sentences written by the model and verified.",
    "model_retry": "Connecting sentences rewritten once after failing verification, then verified.",
    "template_fallback": "The model's sentences failed verification twice: the fixed template was used instead.",
    "template": "Fixed template sentences (no model call).",
}
INTRO = (
    '<div class="br-intro">'
    '<div class="br-intro-card"><div class="br-intro-n">1</div><div><b>Read</b>'
    "<p>The model turns the client's message into a brief, quoting the client's own words for every value.</p></div></div>"
    '<div class="br-intro-card"><div class="br-intro-n">2</div><div><b>Check</b>'
    "<p>Code prices the package from the approved catalogue and checks capacity, access, duration and budget.</p></div></div>"
    '<div class="br-intro-card"><div class="br-intro-n">3</div><div><b>Draft</b>'
    "<p>Every figure in the email is inserted by code. You review, edit and approve before anything is sent.</p></div></div>"
    "</div>"
)

st.set_page_config(page_title="BrightRoom proposal desk", page_icon=":material/event_note:", layout="wide")
st.html(f"<style>{STYLE}</style>")
state = st.session_state
state.setdefault("result", None)
state.setdefault("version", 0)
state.setdefault("enquiry", "")
state.setdefault("approved", None)
state.setdefault("flash", None)
state.setdefault("demo_runs", 0)


def setting(name: str, default: str = "") -> str:
    """A value from Streamlit secrets (when deployed) or from the environment / local .env file."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets.toml: running locally
        value = None
    if value is None or str(value).strip() == "":
        value = os.environ.get(name, default)
    return str(value).strip()


def demo_key() -> str:
    """The owner's OpenRouter key, set as a deployment secret or in .env. Never shown or logged."""
    api_key_available()  # loads a local .env into the environment, if there is one
    return setting("OPENROUTER_API_KEY")


def int_setting(name: str, default: int) -> int:
    try:
        return max(0, int(float(setting(name, str(default)))))
    except ValueError:
        return default


# The demo key pays for every visitor, so its model is fixed and its use is capped. A run costs about
# $0.0002 with the default model, so the daily cap bounds the spend at a few cents a day.
DEMO_MODEL = setting("BRIGHTROOM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
DEMO_RUNS_PER_SESSION = int_setting("BRIGHTROOM_DEMO_RUNS_PER_SESSION", 15)
DEMO_RUNS_PER_DAY = int_setting("BRIGHTROOM_DEMO_RUNS_PER_DAY", 300)


@st.cache_resource
def demo_client(model: str, fingerprint: str) -> OpenRouterClient:
    # Shared by every visitor. The key is read here rather than passed in, so it never becomes a
    # cache key; the fingerprint only makes a changed key take effect.
    return OpenRouterClient(api_key=demo_key(), model=model)


@st.cache_resource
def demo_usage() -> dict:
    """Model runs on the demo key today, across all visitors of this server."""
    return {"lock": threading.Lock(), "day": date.today().isoformat(), "runs": 0}


def demo_runs_left() -> int:
    usage = demo_usage()
    with usage["lock"]:
        if usage["day"] != date.today().isoformat():
            usage["day"], usage["runs"] = date.today().isoformat(), 0
        left_today = DEMO_RUNS_PER_DAY - usage["runs"]
    return max(0, min(left_today, DEMO_RUNS_PER_SESSION - state.demo_runs))


def spend_demo_run() -> None:
    usage = demo_usage()
    with usage["lock"]:
        usage["runs"] += 1
    state.demo_runs += 1


def own_client(api_key: str, model: str) -> OpenRouterClient:
    """A client for a key the visitor pasted: kept in their session only, never in a shared cache."""
    current = state.get("own_client")
    if current is None or current.api_key != api_key or current.model != model:
        state.own_client = OpenRouterClient(api_key=api_key, model=model)
    return state.own_client


def failure_reason(error: Exception) -> str:
    text = str(error)
    if "HTTP 401" in text:
        return "OpenRouter rejected the API key: check it and try again"
    if "HTTP 402" in text:
        return "the OpenRouter account has no credit left"
    return text


def show(result: RunResult, event: str) -> None:
    state.result = result
    state.version += 1
    state.shown_at = time.time()
    state.approved = None
    state.flash = None
    save_run(result, event=event)


def run_line(result: RunResult) -> str:
    """One line on how the result was produced, for the verdict card."""
    calls = result.llm_calls
    read = any(step.name == "extract" for step in result.steps)
    parts = [f"Read by {(result.model or 'the model').split('/')[-1]}" if read else "Brief entered by hand"]
    parts.append(f"{calls} model call{'s' if calls != 1 else ''}" if calls else "no model call")
    parts.append(f"{result.seconds:.1f} s")
    if calls:
        parts.append(f"${result.cost:.4f}")
    return " · ".join(parts)


def budget_figure(a: Assessment, brief: Brief) -> tuple[str, str, str, str]:
    limit, pkg = a.budget_total, a.package
    if limit is None:
        stated = brief.budget.amount
        if stated is None:
            return ("Budget", "Not stated", "ask the client", "amber")
        return ("Budget", money(stated), "per person" if brief.budget.basis == "per_person" else "", "")
    note = "total ceiling" + (" (approximate)" if brief.budget.approximate else "")
    tone = ""
    if pkg is not None:
        gap = limit - pkg.total
        if abs(gap) < 0.005:
            note, tone = "exactly on budget", "green"
        elif gap > 0:
            note, tone = f"{money(gap)} to spare", "green"
        else:
            note, tone = f"{money(-gap)} over budget", "red"
    return ("Budget", money(limit), note, tone)


def verdict_figures(a: Assessment, brief: Brief) -> list[tuple[str, str, str, str]]:
    """Four figures that let the salesperson read the verdict at a glance."""
    pkg = a.package
    if a.route == "clarify":
        services = ", ".join(CAT.service(s.service_id).name.lower() for s in brief.services)
        return [
            ("Attendees", "Missing", "needed before any price", "red"),
            budget_figure(a, brief),
            ("Services asked for", str(len(brief.services)) if brief.services else "None yet", services, ""),
            ("Questions to ask", str(len(a.questions)), "listed in the draft", ""),
        ]
    if pkg is not None:
        venue = CAT.venue(pkg.venue_id)
        access = "step-free" if venue.wheelchair_accessible else venue.access_note
        last = (
            ("Attendees", str(pkg.headcount), "priced for this headcount", "")
            if a.route == "proposal"
            else ("Options to offer", str(len(a.alternatives)), "see the cards below", "")
        )
        count = len(pkg.service_ids)
        contents = "venue only" if count == 0 else f"venue + {count} service{'s' if count != 1 else ''}"
        return [
            ("Total" if a.route == "proposal" else "Total as requested", money(pkg.total), contents, ""),
            budget_figure(a, brief),
            ("Venue", f"{pkg.venue_name} · {hours(pkg.duration_hours)} h", f"up to {venue.max_people} people · {access}", ""),
            last,
        ]
    duration = brief.duration.hours or CAT.standard_duration
    closest = a.alternatives[0] if a.alternatives else None
    return [
        ("Requested", f"{brief.headcount.max} · {hours(duration)} h", "attendees · hours", ""),
        budget_figure(a, brief),
        ("Options to offer", str(len(a.alternatives)), "see the cards below", ""),
        ("Closest option", money(closest.package.total) if closest else "None", closest.label if closest else "", "amber" if closest else ""),
    ]


def assessment_html(a: Assessment) -> str:
    """Everything the engine decided, as one block for the right-hand column."""
    if a.checks:
        blocks = [checks_html(a.checks)]
    else:
        blocks = [items_html([("amber", "Nothing can be checked until the number of attendees is known.", "", "")])]
    requests = [u for u in a.unmet if u.kind not in CONSTRAINTS]
    if requests:
        blocks += [label_html("Requests we cannot meet"), items_html([("red", u.detail, "", "") for u in requests])]
    if a.package:
        title = "Package proposed" if a.route == "proposal" else "The request as priced"
        blocks += [label_html(title), package_html(a.package, CAT, [CAT.price_note])]
    if a.assumptions:
        rows = [("amber" if x.sensitive else "gray", x.text, "", "to confirm" if x.sensitive else "") for x in a.assumptions]
        blocks += [label_html("Assumptions"), items_html(rows)]
    if a.scenarios:
        rows = []
        for s in a.scenarios:
            text = f"{s.label}: {s.package.venue_name}, total {money(s.package.total)}" if s.package else s.label
            rows.append(("green" if s.feasible else "red", text, s.note, ""))
        blocks += [label_html("If an assumption is wrong"), items_html(rows)]
    if a.addons:
        rows = [("navy", f"{x.name}: {money(x.amount)}", x.reason, "") for x in a.addons]
        blocks += [label_html("Optional additions"), items_html(rows)]
    if a.notices:
        rows = [("green" if n.offerable else "red", n.text, f'Client: "{n.quote}"' if n.quote else "", "") for n in a.notices]
        blocks += [label_html("Outside the catalogue or your authority"), items_html(rows)]
    if a.questions:
        rows = [("blue", q.text, "", "blocking" if q.blocking else "") for q in a.questions]
        blocks += [label_html("Questions for the client"), items_html(rows)]
    return "".join(blocks)


# --- Sidebar ------------------------------------------------------------------------------------------

with st.sidebar:
    st.html(sidebar_brand_html())
    st.html('<div class="br-side-label">Model access</div>')
    server_key = demo_key()
    pasted = st.text_input(
        "Your OpenRouter API key" + (" (optional)" if server_key else ""),
        type="password",
        key="own_key",
        placeholder="sk-or-v1-…",
        help="Used only in this browser session to call OpenRouter. It is not saved, logged or shared.",
    ).strip()
    locked = bool(server_key) and not pasted
    typed_model = st.text_input(
        "Model (OpenRouter id)",
        value=DEMO_MODEL,
        disabled=locked,
        key="model_locked" if locked else "model_own",  # the locked field always shows the demo model
        help="The demo key always uses this model. Paste your own key to choose another." if locked else None,
    ).strip()
    using_demo = False
    if pasted:
        client = own_client(pasted, typed_model or DEMO_MODEL)
        st.badge("Using your key", icon=":material/key:", color="green")
    elif server_key and demo_runs_left() > 0:
        client = demo_client(DEMO_MODEL, hashlib.sha256(server_key.encode()).hexdigest()[:12])
        using_demo = True
        left = demo_runs_left()
        st.badge("Demo key active", icon=":material/check_circle:", color="green")
        st.caption(f"{left} model run{'s' if left != 1 else ''} left in this session on the demo key. Paste your own key for more, or to choose another model.")
    elif server_key:
        client = None
        st.badge("Demo limit reached", icon=":material/hourglass_empty:", color="orange")
        st.caption("Paste your own OpenRouter key to continue, or enter a brief by hand.")
    else:
        client = None
        st.badge("No API key", icon=":material/warning:", color="orange")
        st.caption("Paste an OpenRouter key above ([get one](https://openrouter.ai/keys)), or enter a brief by hand.")
    prose_mode = st.radio(
        "Connecting sentences",
        ["model", "template"],
        format_func={"model": "Written by the model, then verified", "template": "Fixed template (no model call)"}.get,
    )
    st.html('<div class="br-side-label">Approved catalogue</div>' + catalogue_html(CAT))
    st.caption(CAT.price_note)
    st.html('<div class="br-side-label">What this tool does not do</div>')
    st.markdown(
        "- It does not send anything: you copy the approved draft yourself.\n"
        "- It does not check availability, hold dates or book venues.\n"
        "- It offers nothing outside the catalogue: no discounts, no unlisted services.\n"
        "- It can misread an enquiry: compare the brief with the client's words.\n"
        "- Enquiries are sent to OpenRouter to be read: use fictional data."
    )

# --- Header and enquiry ---------------------------------------------------------------------------------

stage = 0 if state.result is None else (len(STEPS) if state.approved else 2)
st.html(header_html(stage))

with st.container(border=True):
    st.subheader(":material/mail: Client enquiry", anchor=False)
    picked = st.pills("Try an enquiry from the case", list(SAMPLES), format_func=SAMPLE_LABELS.get, selection_mode="single")
    if picked and state.get("picked") != picked:
        state.picked, state.enquiry = picked, SAMPLES[picked]
    enquiry = st.text_area("Client enquiry", key="enquiry", height=110, placeholder="Paste the client's message here.", label_visibility="collapsed")
    actions = st.container(horizontal=True, gap="small")
    prepare = actions.button("Prepare draft", type="primary", icon=":material/auto_awesome:", disabled=not (client and enquiry.strip()))
    manual = actions.button("Enter brief by hand", icon=":material/edit_note:")
    if not client:
        st.caption(":material/info: Without an API key the model cannot read enquiries: enter the brief by hand instead.")
    if prepare:
        if using_demo:
            spend_demo_run()
        try:
            with st.spinner("Reading the enquiry and preparing the draft…"):
                show(run_pipeline(enquiry, client, CAT, prose_mode), "draft")
        except (ExtractionError, LLMError) as error:
            show(run_from_brief(enquiry, Brief(), None, CAT, "template"), "manual")
            state.flash = f"The enquiry could not be read automatically ({failure_reason(error)}). Enter the brief by hand below."
        st.rerun()
    if manual:
        show(run_from_brief(enquiry, Brief(), None, CAT, "template"), "manual")
        st.rerun()
    if state.flash:
        st.error(state.flash, icon=":material/error:")

result: RunResult | None = state.result
if result is None:
    st.html(INTRO)
    st.stop()
a, brief, version = result.assessment, result.brief, state.version

# --- Verdict first ------------------------------------------------------------------------------------

st.html(verdict_html(a.status, a.headline, run_line(result)) + figures_html(verdict_figures(a, brief)))
for warning in a.warnings:
    st.warning(f"Unverified reading. {warning}", icon=":material/report:")

understood, decided = st.columns(2, gap="large")

with understood:
    st.subheader(":material/forum: What the client asked for", anchor=False)
    st.html(quote_box_html(result.enquiry, quotes_of(brief)))
    st.caption("Highlighted words are the ones the brief is based on.")
    st.html(label_html("What we understood") + brief_html(brief_rows(brief, result.issues, CAT)))

    with st.expander(":material/edit: Correct the brief", expanded=result.enquiry.strip() == "" or a.route == "clarify" and not client):
        with st.form(f"brief_{version}", border=False):
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
            service_columns = st.columns(2)
            for index, service in enumerate(CAT.services):
                options = ["not requested", "required", "preferred"]
                was = current[service.id].need if service.id in current else "not requested"
                choices[service.id] = service_columns[index % 2].selectbox(service.name, options, index=options.index(was))
            date_mentioned = st.checkbox("A date or timing was mentioned", value=brief.date.mentioned)
            labels = [f"{r.category.replace('_', ' ')}: {r.description} ({r.need})" for r in brief.other_requests]
            kept = st.multiselect("Other requests to keep", labels, default=labels)
            c1, c2, c3 = st.columns([2, 2, 1])
            new_text = c1.text_input("Add another request (optional)")
            new_category = c2.selectbox("Its kind", CATEGORIES, format_func=lambda c: c.replace("_", " "))
            new_need = c3.selectbox("It is", ["preferred", "required"])
            submitted = st.form_submit_button("Recalculate with this brief", type="primary", icon=":material/refresh:")

        if submitted:
            changed: set[str] = set()

            def keep(field: str, same: bool, quote: str | None) -> str | None:
                if not same:
                    changed.add(field)
                return quote if same else None

            lo, hi = (n_min or n_max or None), (n_max or n_min or None)
            old = brief
            # The form shows an unstated budget basis as "total" (the engine reads it that way), so
            # submitting it untouched must not count as an edit and drop the client's quote.
            budget_same = (amount or None) == old.budget.amount and (not amount or basis == (old.budget.basis or "total"))
            new = Brief(
                language=old.language,
                headcount=Headcount(min=lo, max=hi, approximate=approximate, quote=keep("headcount", (lo, hi, approximate) == (old.headcount.min, old.headcount.max, old.headcount.approximate), old.headcount.quote)),
                duration=Duration(hours=duration or None, quote=keep("duration", (duration or None) == old.duration.hours, old.duration.quote)),
                budget=Budget(
                    amount=amount or None,
                    basis=old.budget.basis if budget_same else (basis if amount else None),
                    approximate=old.budget.approximate if amount else False,
                    quote=keep("budget", budget_same, old.budget.quote),
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
            if using_demo and prose_mode == "model":
                spend_demo_run()
            with st.spinner("Recalculating…"):
                show(run_from_brief(result.enquiry, new, client, CAT, prose_mode, issues), "corrected")
            st.rerun()

with decided:
    st.subheader(":material/fact_check: How we checked it", anchor=False)
    st.html(assessment_html(a))

if a.alternatives:
    st.subheader(":material/alt_route: Options to offer the client", anchor=False)
    st.html(options_html(a.alternatives, CAT))

# --- Draft and approval ---------------------------------------------------------------------------------

st.subheader(":material/drafts: Draft for the client", anchor=False)
st.caption(f"{PROSE_SOURCES[result.prose_source]} All figures, items, assumptions and notices are inserted by code.")
for violation in result.draft_violations:
    st.error(f"Draft check failed: {violation.detail}")

draft_column, approve_column = st.columns([1.75, 1], gap="large")
with draft_column:
    preview_tab, edit_tab = st.tabs([":material/visibility: Preview", ":material/edit: Edit"])
    with edit_tab:
        text = st.text_area("Edit if needed, then approve", value=result.draft, height=520, key=f"draft_{version}")
    with preview_tab:
        st.html(email_html(text))

with approve_column:
    with st.container(border=True):
        st.markdown("**Before you approve**")
        acknowledgements = ["I have compared the brief with the client's words.", f"I have read the verdict: {a.headline}."]
        if any(item.sensitive for item in a.assumptions):
            acknowledgements.append("I have read the points that still need the client's confirmation.")
        if a.warnings:
            acknowledgements.append("I have checked the readings that could not be verified.")
        acknowledgements.append("I know that availability has not been checked and nothing is booked.")
        ticks = [st.checkbox(item, key=f"ack_{version}_{i}") for i, item in enumerate(acknowledgements)]
        if st.button("Approve draft", type="primary", icon=":material/task_alt:", disabled=not all(ticks) or state.approved == text, width="stretch"):
            review_seconds = round(time.time() - state.shown_at, 1)
            save_run(result, event="approved", extra={"review_seconds": review_seconds, "edit_share": edit_share(result.draft, text), "final_draft": text})
            state.approved = text
            st.rerun()
        if state.approved:
            st.success("Approved. Copy the text below into your email: nothing has been sent or booked.", icon=":material/check_circle:")
        else:
            st.caption("Tick every point to enable approval. Approving saves an audit record; it sends nothing.")

if state.approved:
    st.markdown("**Approved text**: use the copy button at the top right of the box.")
    st.code(state.approved, language=None, wrap_lines=True)

with st.expander(":material/timer: Run details: time, model calls and cost"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Time", f"{result.seconds:.1f} s")
    m2.metric("Model calls", result.llm_calls)
    m3.metric("Tokens", f"{result.tokens:,}")
    m4.metric("Cost", f"${result.cost:.4f}")
    st.dataframe(
        [{"Step": s.name, "Seconds": round(s.seconds, 2), "Model calls": s.llm_calls, "Tokens in": s.prompt_tokens, "Tokens out": s.completion_tokens, "Cost ($)": round(s.cost, 6)} for s in result.steps],
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Model: {result.model or 'none'}.")
    for attempt, violations in enumerate(result.prose_violations, 1):
        for violation in violations:
            st.markdown(f"- Attempt {attempt}: {violation.code}: {violation.detail}")
