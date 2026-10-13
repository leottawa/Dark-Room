"""Presentation helpers for the review screen, kept free of Streamlit so they can be tested."""

from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher

from .catalogue import Catalogue, hours, money
from .models import Brief, FieldIssue
from .textutil import _PUNCTUATION

# Typographic variants the extraction audit treats as equal (see textutil.normalise).
_VARIANTS = {"'": "['‘’]", '"': '["“”]', "-": "[-–—]"}

NOT_STATED = "not stated"


def brief_rows(brief: Brief, issues: list[FieldIssue], cat: Catalogue) -> list[dict]:
    """One row per brief field: what was understood, the client's words, and how far to trust it."""
    by_field: dict[str, list[FieldIssue]] = {}
    for issue in issues:
        by_field.setdefault(issue.field, []).append(issue)

    def row(field: str, label: str, value: str | None, quote: str | None) -> dict:
        kinds = {i.kind for i in by_field.get(field, [])}
        if value is None:
            status = NOT_STATED
        elif "quote_not_found" in kinds:
            status = "check: quote not found"
        elif "interpreted" in kinds:
            status = "interpreted"
        elif quote:
            status = "quoted"
        else:
            status = "entered by hand"
        return {"Field": label, "Understood as": value or "—", "Client's words": quote or "—", "Status": status}

    hc = brief.headcount
    attendees = None
    if hc.max is not None:
        attendees = str(hc.max) if hc.min == hc.max else f"{hc.min} to {hc.max}"
        attendees += " (approximate)" if hc.approximate else ""
    budget = None
    if brief.budget.amount is not None:
        budget = money(brief.budget.amount) + (" per person" if brief.budget.basis == "per_person" else " in total")
        budget += " (approximate)" if brief.budget.approximate else ""
    venue = None
    if brief.venue.venue_id:
        venue = f"{cat.venue(brief.venue.venue_id).name} ({brief.venue.need})"
    access = None if brief.accessibility.need == "not_mentioned" else brief.accessibility.need.replace("_", " ")

    rows = [
        row("headcount", "Attendees", attendees, hc.quote),
        row("duration", "Duration", None if brief.duration.hours is None else f"{hours(brief.duration.hours)} hours", brief.duration.quote),
        row("budget", "Budget", budget, brief.budget.quote),
        row("accessibility", "Step-free access", access, brief.accessibility.quote),
        row("venue", "Venue asked for", venue, brief.venue.quote),
    ]
    if not brief.services:
        rows.append(row("services", "Services", None, None))
    for wish in brief.services:
        value = wish.need + (f"; beyond catalogue terms: {wish.beyond_terms}" if wish.beyond_terms else "")
        rows.append(row(f"services.{wish.service_id}", cat.service(wish.service_id).name, value, wish.quote))
    rows.append(row("date", "Date", "mentioned" if brief.date.mentioned else None, brief.date.quote))
    for index, request in enumerate(brief.other_requests):
        label = f"Other: {request.category.replace('_', ' ')}"
        rows.append(row(f"other_requests.{index}", label, f"{request.description} ({request.need})", request.quote))
    return rows


def quotes_of(brief: Brief) -> list[str]:
    parts = [brief.headcount, brief.duration, brief.budget, brief.accessibility, brief.venue, brief.date, *brief.services, *brief.other_requests]
    return [p.quote for p in parts if p.quote]


def highlight(enquiry: str, quotes: list[str]) -> str:
    """The enquiry as HTML with every quoted span marked, so the reviewer sees what was read."""
    spans: list[tuple[int, int]] = []
    for quote in quotes:
        pattern = _quote_pattern(quote)
        match = re.search(pattern, enquiry, re.IGNORECASE) if pattern else None
        if match:
            spans.append(match.span())
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out, cursor = [], 0
    for start, end in merged:
        out.append(html.escape(enquiry[cursor:start]))
        out.append(f"<mark>{html.escape(enquiry[start:end])}</mark>")
        cursor = end
    out.append(html.escape(enquiry[cursor:]))
    return "".join(out).replace("\n", "<br>")


def _quote_pattern(quote: str) -> str:
    """A pattern that finds the quote the way the audit does: ignoring case, typography and spacing."""
    text = unicodedata.normalize("NFKC", quote).translate(_PUNCTUATION).strip(" \"'.,;:!?…")
    parts = []
    for token in re.split(r"(\s+)", text):
        if token.isspace():
            parts.append(r"\s+")
        elif token:
            parts.append("".join(_VARIANTS.get(ch, re.escape(ch)) for ch in token))
    return "".join(parts)


def edit_share(original: str, final: str) -> float:
    """Share of the draft the reviewer changed before approving: 0 is untouched, 1 is rewritten."""
    return round(1 - SequenceMatcher(None, original, final).ratio(), 3)


# --- HTML blocks for the review screen (styled by app/style.css) ------------------------------------

VERDICT = {
    "fits": ("green", "✓", "FITS THE BRIEF"),
    "provisional": ("amber", "!", "PROVISIONAL"),
    "infeasible": ("red", "✕", "CANNOT MEET THE BRIEF"),
    "clarify": ("blue", "?", "MORE INFORMATION NEEDED"),
}
STEPS = ["Enquiry", "Verdict", "Draft", "Approved"]
CHECK_ICONS = {"pass": "✓", "fail": "✕", "unknown": "?", "n/a": "–"}
DOT_ICONS = {"green": "✓", "red": "✕", "amber": "!", "blue": "?", "navy": "•", "gray": "•"}
# How far to trust each brief field: (chip colour, chip text).
STATUS_CHIPS = {
    "quoted": ("green", "Quoted"),
    "interpreted": ("amber", "Interpreted"),
    "check: quote not found": ("red", "Check the quote"),
    "entered by hand": ("gray", "Entered by hand"),
    NOT_STATED: ("gray", "Not stated"),
}


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def header_html(stage: int) -> str:
    """Brand header with the four-step progress; steps before `stage` are done, `stage` is current."""
    steps = []
    for index, label in enumerate(STEPS):
        state = "done" if index < stage else ("current" if index == stage else "todo")
        mark = "✓" if state == "done" else str(index + 1)
        steps.append(f'<div class="br-step {state}"><span class="n">{mark}</span>{label}</div>')
    progress = '<span class="br-step-line"></span>'.join(steps)
    return (
        '<div class="br-header"><div class="br-brand"><div class="br-logo">BR</div><div>'
        '<div class="br-title">BrightRoom proposal desk</div>'
        '<div class="br-subtitle">From a client enquiry to a draft the salesperson can check and approve.</div>'
        f'</div></div><div class="br-steps">{progress}</div></div>'
    )


def verdict_html(status: str, headline: str, run_line: str = "") -> str:
    """The verdict, shown before anything else. The pill names the route; the headline gives the reason."""
    tone, icon, label = VERDICT[status]
    reason = headline.split(": ", 1)[1] if ": " in headline else headline
    reason = reason[:1].upper() + reason[1:]
    return (
        f'<div class="br-verdict {tone}"><div class="br-verdict-top">'
        f'<span class="br-pill {tone}">{icon}&nbsp; {label}</span><span class="br-run">{_e(run_line)}</span></div>'
        f'<div class="br-headline">{_e(reason)}</div></div>'
    )


def figures_html(figures: list[tuple[str, str, str, str]]) -> str:
    """Equal-height tiles of (label, value, note, note tone); the tone colours the note line."""
    tiles = []
    for label, value, note, tone in figures:
        note_html = f'<div class="n {tone}">{_e(note)}</div>' if note else ""
        tiles.append(f'<div class="br-figure"><div class="l">{_e(label)}</div><div class="v">{_e(value)}</div>{note_html}</div>')
    return f'<div class="br-figures">{"".join(tiles)}</div>'


def quote_box_html(enquiry: str, quotes: list[str]) -> str:
    marked = highlight(enquiry, quotes) or '<span class="empty">No enquiry text.</span>'
    return f'<div class="br-quote">{marked}</div>'


def brief_html(rows: list[dict]) -> str:
    """The rows of brief_rows as a list that wraps: field, reading with the client's words, trust chip."""
    out = []
    for row in rows:
        tone, chip = STATUS_CHIPS.get(row["Status"], ("gray", row["Status"]))
        words = row["Client's words"]
        quote = f'<span class="q">“{_e(words)}”</span>' if words and words != "—" else ""
        out.append(
            f'<div class="br-brief-row"><div class="f">{_e(row["Field"])}</div>'
            f'<div class="v">{_e(row["Understood as"])}{quote}</div>'
            f'<div class="s"><span class="br-tag {tone}">{_e(chip)}</span></div></div>'
        )
    return f'<div class="br-brief">{"".join(out)}</div>'


def checks_html(checks) -> str:
    tiles = "".join(
        f'<div class="br-check {c.status.replace("/", "")}"><span class="ic">{CHECK_ICONS[c.status]}</span>'
        f'<div><div class="name">{_e(c.name.capitalize())}</div><div class="detail">{_e(c.detail)}</div></div></div>'
        for c in checks
    )
    return f'<div class="br-checks">{tiles}</div>'


def label_html(text: str) -> str:
    return f'<div class="br-label">{_e(text)}</div>'


def items_html(items: list[tuple[str, str, str, str]]) -> str:
    """A list of (tone, text, detail, tag) rows; detail and tag may be empty."""
    rows = []
    for tone, text, detail, tag in items:
        sub = f'<span class="br-sub">{_e(detail)}</span>' if detail else ""
        badge = f'<span class="br-tag amber">{_e(tag)}</span>' if tag else ""
        rows.append(f'<li><span class="br-dot {tone}">{DOT_ICONS.get(tone, "•")}</span><span>{_e(text)}{badge}{sub}</span></li>')
    return f'<ul class="br-items">{"".join(rows)}</ul>'


def package_summary(pkg, cat: Catalogue) -> str:
    services = ", ".join(line.name.lower() for line in pkg.lines if line.kind == "service") or "venue only"
    return f"{pkg.venue_name} · {pkg.headcount} attendees · {hours(pkg.duration_hours)} h · {services}"


def package_html(pkg, cat: Catalogue, notes: list[str] = ()) -> str:
    """The package as an invoice: one row per line, quantities shown, total underlined."""
    rows = []
    for line in pkg.lines:
        if line.kind == "venue":
            quantity = "1 session"
        elif cat.service(line.item_id).pricing == "per_person":
            quantity = f"{line.quantity} × {money(line.unit_price)}"
        else:
            quantity = "1"
        rows.append(
            f'<tr><td>{_e(line.name)}<span class="cond">{_e(line.conditions)}</span></td>'
            f'<td class="num">{_e(quantity)}</td><td class="num">{_e(money(line.amount))}</td></tr>'
        )
    total = (
        f'<tr class="total"><td>Total</td><td class="num meta">{pkg.headcount} attendees · {hours(pkg.duration_hours)} h</td>'
        f'<td class="num">{_e(money(pkg.total))}</td></tr>'
    )
    head = '<thead><tr><th>Item</th><th class="num">Quantity</th><th class="num">Amount</th></tr></thead>'
    notes_html = "".join(f'<div class="br-note">{_e(note)}</div>' for note in notes)
    return f'<table class="br-invoice">{head}<tbody>{"".join(rows)}{total}</tbody></table>{notes_html}'


def options_html(alternatives, cat: Catalogue) -> str:
    cards = []
    for letter, alt in zip("ABCDEFGH", alternatives):
        if alt.changes_requirement:
            tag = '<span class="br-tag amber">Changes a requirement</span>'
        else:
            tag = '<span class="br-tag green">Keeps every requirement</span>'
        trade_offs = "".join(f"<li>{_e(t)}</li>" for t in alt.trade_offs)
        cards.append(
            f'<div class="br-option"><div class="br-option-head"><span class="br-letter">{letter}</span>{tag}</div>'
            f'<div class="br-option-title">{_e(alt.label)}</div>'
            f'<div class="br-option-total">{_e(money(alt.package.total))}</div>'
            f'<div class="br-option-includes">{_e(package_summary(alt.package, cat))}</div>'
            f"<ul>{trade_offs}</ul></div>"
        )
    return f'<div class="br-options">{"".join(cards)}</div>'


def email_html(text: str) -> str:
    """The draft as the client would read it. The first line is the subject when it starts with "Subject:"."""
    lines = text.split("\n")
    subject = ""
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return (
        '<div class="br-email"><div class="br-email-head"><div><b>From</b> The BrightRoom team &nbsp;·&nbsp; <b>To</b> the client</div>'
        f'<div class="br-email-subject">{_e(subject) or "(no subject)"}</div></div>'
        f'<div class="br-email-body">{_e(chr(10).join(lines))}</div></div>'
    )


def sidebar_brand_html() -> str:
    return (
        '<div class="br-side-brand"><div class="br-logo">BR</div><div>'
        '<div class="br-side-title">BrightRoom</div><div class="br-side-sub">Proposal desk</div></div></div>'
    )


def catalogue_html(cat: Catalogue) -> str:
    """The approved catalogue as a compact price list for the sidebar."""
    rows = []
    for v in cat.venues:
        access = "step-free" if v.wheelchair_accessible else v.access_note
        detail = f"{hours(v.duration_hours)} h · up to {v.max_people} people · {access}"
        rows.append((v.name, detail, money(v.price)))
    for s in cat.services:
        per_person = s.pricing == "per_person"
        rows.append((s.name, "per person" if per_person else "per booking", money(s.price) + ("/person" if per_person else "")))
    body = "".join(f'<tr><td>{_e(name)}<span class="cond">{_e(detail)}</span></td><td class="num">{_e(price)}</td></tr>' for name, detail, price in rows)
    return f'<table class="br-cat">{body}</table>'
