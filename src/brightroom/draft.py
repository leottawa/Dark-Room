"""Step 4: assemble the client email.

Every figure, item, assumption, option and notice is rendered by code from the assessment. The
model contributes three short passages of connecting prose and is never asked to state a price,
so a fluent sentence cannot carry a wrong number into the draft.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from .catalogue import Catalogue, hours, money
from .llm import LLMClient, LLMResult
from .models import Assessment, Brief, Package

FOOTER = (
    "This is a draft for discussion, not a confirmation: no venue is booked and no date is held. "
    "Dates are subject to a separate availability check."
)
SIGN_OFF = "Kind regards,\nThe BrightRoom team"
SUBJECTS = {
    "proposal": "Draft proposal for your workshop",
    "infeasible": "Your workshop enquiry: options to consider",
    "clarify": "Your workshop enquiry: a few details needed",
}


class Prose(BaseModel):
    opening: str
    body: str
    closing: str


PROSE_SCHEMA = {
    "type": "object",
    "properties": {"opening": {"type": "string"}, "body": {"type": "string"}, "closing": {"type": "string"}},
    "required": ["opening", "body", "closing"],
    "additionalProperties": False,
}

PROSE_SYSTEM = """You write three short passages for an email from BrightRoom, a studio that hosts workshops, to a prospective client.
The system inserts the greeting, the priced package, the assumptions, the options, the questions, the notices and the sign-off around your passages. Do not repeat or list any of them.

Passages:
- opening: one sentence thanking the client and restating their event in plain words.
- body: one or two sentences that fit the situation.
  proposal: say that a draft package matching their request follows.
  provisional_proposal: say that a draft package follows and that a few points need their confirmation.
  cannot_meet_request: say courteously and plainly that the request cannot be met exactly as stated, give the reasons in words, and say that the closest options follow.
  need_more_information: say that a few details are needed before a proposal can be prepared.
- closing: one sentence inviting a reply.

Rules:
- Use only the facts provided. Do not add services, venues, features, prices or benefits.
- The only figures you may write are figures that appear in the facts.
- Do not mention availability, dates, booking, reservations, discounts, guarantees or anything free of charge: the system adds those notes.
- Plain sentences only: no lists, headings, greeting, sign-off or subject line. Warm and professional British English. At most 60 words per passage.
Return a JSON object with the keys opening, body and closing."""


def prose_facts(assessment: Assessment, brief: Brief, cat: Catalogue) -> dict:
    """The only material the model sees when writing: nothing here is for it to decide."""
    situation = {"fits": "proposal", "provisional": "provisional_proposal", "infeasible": "cannot_meet_request", "clarify": "need_more_information"}[assessment.status]
    hc = brief.headcount
    attendees = None if hc.max is None else (hc.max if hc.min == hc.max else f"{hc.min} to {hc.max}")
    facts: dict = {
        "situation": situation,
        "client_request": {
            "attendees": attendees,
            "attendees_approximate": hc.approximate,
            "duration_hours": brief.duration.hours,
            "required": [cat.service(s.service_id).name for s in brief.services if s.need == "required"],
            "would_like": [cat.service(s.service_id).name for s in brief.services if s.need == "preferred"],
            "step_free_access": brief.accessibility.need,
            "budget": None if assessment.budget_total is None else money(assessment.budget_total),
        },
    }
    if assessment.route == "proposal" and assessment.package:
        pkg = assessment.package
        facts["package"] = {"venue": pkg.venue_name, "services": [line.name for line in pkg.lines if line.kind == "service"], "total": money(pkg.total)}
        facts["points_to_confirm"] = [a.text for a in assessment.assumptions if a.sensitive]
    if assessment.route == "infeasible":
        facts["reasons"] = [u.detail for u in assessment.unmet]
        facts["options_offered"] = [alt.label for alt in assessment.alternatives]
    if assessment.route == "clarify":
        facts["missing"] = [q.field for q in assessment.questions]
    cannot = [n.text for n in assessment.notices if not n.offerable]
    if cannot:
        facts["cannot_offer"] = cannot
    return facts


def template_prose(assessment: Assessment, brief: Brief) -> Prose:
    """Fixed wording used when no model is available or its passages fail verification."""
    lo, hi = brief.headcount.min, brief.headcount.max
    people = None if hi is None else (f"{hi}" if lo in (None, hi) else f"{min(lo, hi)} to {max(lo, hi)}")
    about = f" about a workshop for {people} people" if people else ""
    opening = f"Thank you for your enquiry{about}."
    if assessment.route == "proposal":
        body = "Please find below a draft package based on what you told us."
        if assessment.provisional:
            body += " A few points need your confirmation before we finalise it."
        closing = "We would be glad to adjust the package once we hear from you."
    elif assessment.route == "infeasible":
        body = "We are unable to meet the request exactly as stated, for the reasons set out below. We have listed the closest options we can offer."
        closing = "Please tell us which option suits you and we will prepare a proposal."
    else:
        body = "To prepare a proposal that fits your event, we need a few details."
        closing = "As soon as we have these, we will send you a draft package."
    return Prose(opening=opening, body=body, closing=closing)


def request_prose(facts: dict, client: LLMClient, feedback: str | None = None) -> tuple[Prose | None, LLMResult]:
    user = "Facts:\n" + json.dumps(facts, ensure_ascii=False, indent=1)
    if feedback:
        user += f"\n\nYour previous passages broke these rules: {feedback} Write them again without those problems."
    result = client.complete([{"role": "system", "content": PROSE_SYSTEM}, {"role": "user", "content": user}], schema=PROSE_SCHEMA, schema_name="passages", max_tokens=400)
    try:
        text = result.text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        prose = Prose.model_validate(json.loads(text))
    except ValueError:
        return None, result
    return prose, result


# --- Rendering ----------------------------------------------------------------------------------


def _package_block(pkg: Package, cat: Catalogue) -> list[str]:
    lines = []
    for line in pkg.lines:
        if line.kind == "venue":
            lines.append(f"- {line.name} ({line.conditions}): {money(line.amount)}")
        elif cat.service(line.item_id).pricing == "per_person":
            lines.append(f"- {line.name}, {line.quantity} x {money(line.unit_price)}: {money(line.amount)}. {line.conditions}")
        else:
            lines.append(f"- {line.name}: {money(line.amount)}. {line.conditions}")
    lines.append(f"Total: {money(pkg.total)}")
    return lines


def _includes(pkg: Package, cat: Catalogue) -> str:
    parts = [pkg.venue_name]
    for line in pkg.lines:
        if line.kind == "service":
            per_person = cat.service(line.item_id).pricing == "per_person"
            parts.append(f"{line.name.lower()} ({line.quantity} x {money(line.unit_price)})" if per_person else line.name.lower())
    return f"{', '.join(parts)}; {pkg.headcount} attendees, {hours(pkg.duration_hours)}-hour session"


def _section(title: str, lines: list[str]) -> list[str]:
    return ["", title, *lines] if lines else []


def render_draft(assessment: Assessment, brief: Brief, prose: Prose, cat: Catalogue) -> str:
    a = assessment
    out = [f"Subject: {SUBJECTS[a.route]}", "", "Hello,", "", prose.opening.strip(), "", prose.body.strip()]
    numbered = [f"{i}. {q.text}" for i, q in enumerate(a.questions, 1)]
    notices = [f"- {n.text}" for n in a.notices]
    assumptions = [f"- {x.text}" for x in a.assumptions]

    if a.route == "proposal" and a.package:
        pkg = a.package
        block = _package_block(pkg, cat)
        budget = next((c for c in a.checks if c.name == "budget"), None)
        if budget and budget.status == "pass":
            block.append(budget.detail)
        duration = next((c for c in a.checks if c.name == "duration"), None)
        if duration and pkg.duration_hours < cat.venue(pkg.venue_id).duration_hours:
            block.append(duration.detail)  # a shorter event still pays for the full session
        block.append(cat.price_note)
        out += _section(f"Proposed package for {pkg.headcount} attendees, {hours(pkg.duration_hours)}-hour session", block)
        out += _section("Our assumptions", assumptions)
        scenarios = []
        for s in a.scenarios:
            if s.package:
                scenarios.append(f"- {s.label}: {_includes(s.package, cat)}; total {money(s.package.total)}. {s.note}".rstrip())
            else:
                scenarios.append(f"- {s.label}: {s.note}")
        out += _section("If your needs differ", scenarios)
        out += _section("Optional additions", [f"- {x.name}: {money(x.amount)}. {x.conditions} {x.reason}" for x in a.addons])
        out += _section("Please note", notices)
        out += _section("To finalise, please confirm", numbered)
    elif a.route == "infeasible":
        out += _section("What we cannot meet as requested", [f"- {u.detail}" for u in a.unmet])
        if a.package:
            out += _section("Your request as priced", _package_block(a.package, cat) + [cat.price_note])
        options = []
        for letter, alt in zip("ABCDEFGH", a.alternatives):
            options.append(f"{letter}. {alt.label}: total {money(alt.package.total)}")
            options.append(f"   Includes: {_includes(alt.package, cat)}.")
            options.append(f"   Trade-off: {' '.join(alt.trade_offs)}")
        out += _section("Options we can offer", options)
        out += _section("Our assumptions", assumptions)
        out += _section("Please note", notices)
        out += _section("Next step", numbered)
    else:
        out += _section("Please let us know", numbered)
        biggest = max(v.max_people for v in cat.venues)
        reference = [f"- Our venues host up to {biggest} people for {hours(cat.standard_duration)}-hour sessions.", f"- {cat.price_note}"]
        out += _section("For reference", reference)
        out += _section("Our assumptions", assumptions)
        out += _section("Please note", notices)

    out += ["", prose.closing.strip(), "", SIGN_OFF, "", FOOTER]
    return "\n".join(out)
