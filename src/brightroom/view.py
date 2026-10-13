"""Presentation helpers for the review screen, kept free of Streamlit so they can be tested."""

from __future__ import annotations

import html
import re
from difflib import SequenceMatcher

from .catalogue import Catalogue, hours, money
from .models import Brief, FieldIssue

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
        match = re.search(re.escape(quote.strip()), enquiry, re.IGNORECASE)
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


def edit_share(original: str, final: str) -> float:
    """Share of the draft the reviewer changed before approving: 0 is untouched, 1 is rewritten."""
    return round(1 - SequenceMatcher(None, original, final).ratio(), 3)
