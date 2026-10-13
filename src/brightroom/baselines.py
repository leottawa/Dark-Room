"""Single-prompt baselines and the grader that turns their free-text replies into an Outcome.

naive: what a salesperson does with a chat model today: paste the catalogue and the enquiry and
ask for a proposal. engineered: the same single call with every case rule spelled out, the
strongest fair competitor to the pipeline. Both use the same model and the same catalogue text.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from .catalogue import Catalogue, money
from .llm import LLMClient, LLMResult
from .scoring import Outcome

REASONS = ["capacity", "duration", "budget", "accessibility", "unlisted_service", "allergen_guarantee", "beyond_terms"]
FIELDS = ["headcount", "services", "budget", "accessibility", "duration", "date"]
REFUSALS = ["discount", "booking_or_date_confirmation", "allergen_guarantee", "unlisted_service", "vegetarian_vegan", "date"]
PROMISES = ["booking_or_date_confirmed", "availability_promised", "discount_offered", "unlisted_service_offered", "allergen_guarantee_given", "price_not_in_catalogue"]


def catalogue_text(cat: Catalogue) -> str:
    """The catalogue as printed in the case starter pack."""
    lines = ["APPROVED CATALOGUE"]
    lines += [f"{v.id} · {v.name}: {money(v.price)}. {v.conditions}." for v in cat.venues]
    lines += [f"{s.id} · {s.name}: {money(s.price)}{'/person' if s.pricing == 'per_person' else ''}. {s.conditions}" for s in cat.services]
    lines.append(f"{cat.price_note} {cat.terms}")
    return "\n".join(lines)


ENGINEERED_SYSTEM = """You are a sales assistant at {business}, a studio that organises half-day workshops. You turn a client enquiry into a draft reply for a salesperson to review.

{catalogue}

Work through these steps and show them:
1. BRIEF: list headcount, duration, budget, required services, optional preferences, accessibility needs, and what is missing.
2. If the headcount is missing, do not propose a package: write a short reply asking for the missing details.
3. Otherwise choose one venue that fits the headcount, the accessibility need and the duration. Add each required service. Itemise every price and compute the total.
4. VERDICT: state whether the package meets every requirement and the budget. If anything cannot be met, say so explicitly and offer alternatives with their trade-offs. Never drop or change a required item silently to make the numbers work.
5. Label every assumption you make.
6. REPLY: write the email to the client.

Rules: use only the catalogue. Do not invent services, prices or discounts. This is a draft: do not book a venue, confirm a date or promise availability."""


def run_baseline(kind: str, enquiry: str, client: LLMClient, cat: Catalogue) -> LLMResult:
    if kind == "naive":
        messages = [{"role": "user", "content": f"{catalogue_text(cat)}\n\nClient enquiry: {enquiry.strip()}\n\nWrite a proposal for this client."}]
    elif kind == "engineered":
        system = ENGINEERED_SYSTEM.format(business=cat.business, catalogue=catalogue_text(cat))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Client enquiry: {enquiry.strip()}"}]
    else:
        raise ValueError(f"unknown baseline: {kind}")
    return client.complete(messages, max_tokens=1500)


def _array(values: list[str]) -> dict:
    return {"type": "array", "items": {"type": "string", "enum": values}}


def grader_schema(cat: Catalogue) -> dict:
    number = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    properties = {
        "kind": {"type": "string", "enum": ["proposal", "cannot_meet", "clarification", "other"]},
        "venue_id": {"anyOf": [{"type": "string", "enum": cat.venue_ids}, {"type": "null"}]},
        "service_ids": _array(cat.service_ids),
        "stated_total": number,
        "reasons": _array(REASONS),
        "flagged_assumptions": _array(FIELDS),
        "asks_about": _array(FIELDS),
        "declines": _array(REFUSALS),
        "alternative_totals": {"type": "array", "items": {"type": "number"}},
        "venues_offered": _array(cat.venue_ids),
        "promises": _array(PROMISES),
    }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


GRADER_SYSTEM = """You audit a reply that was drafted for a client of {business}. Record what the reply says. Do not correct it and do not judge whether it is right.

{catalogue}

Fields:
- kind: "proposal" if the reply offers one priced package as its main answer; "cannot_meet" if it says the request cannot be met as stated (it may then offer alternatives); "clarification" if it asks questions and offers no priced package; otherwise "other".
- venue_id: the venue of the main package offered, or of the request as priced; null if none.
- service_ids: the catalogue services included in that main package.
- stated_total: the total price the reply states for that main package; null if none is stated.
- reasons: the reasons the reply gives for not meeting the request.
- flagged_assumptions: the items the reply explicitly labels as an assumption or asks the client to confirm.
- asks_about: the items the reply asks the client a question about.
- declines: what the reply explicitly says it cannot offer, confirm or guarantee. Use "date" when it says dates are subject to an availability check.
- alternative_totals: the total price of each alternative package offered besides the main one.
- venues_offered: every venue offered in the main package or in an alternative.
- promises: commitments the reply makes. booking_or_date_confirmed: says the venue or date is booked, held or confirmed. availability_promised: says the venue is available. discount_offered: offers any reduction. unlisted_service_offered: offers something that is not in the catalogue. allergen_guarantee_given: guarantees allergen-free food. price_not_in_catalogue: states a unit price that differs from the catalogue.
Return only the JSON object."""


class GradingError(RuntimeError):
    """The grader could not audit a reply. Carries the calls made, so their cost is still counted."""

    def __init__(self, message: str, calls: list[LLMResult]):
        super().__init__(message)
        self.calls = calls


class _Audit(BaseModel):
    kind: Literal["proposal", "cannot_meet", "clarification", "other"]
    venue_id: str | None
    service_ids: list[str]
    stated_total: float | None
    reasons: list[str]
    flagged_assumptions: list[str]
    asks_about: list[str]
    declines: list[str]
    alternative_totals: list[float]
    venues_offered: list[str]
    promises: list[str]


def _parse_audit(text: str) -> _Audit:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return _Audit.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(str(exc)[:300]) from exc


def grade_reply(enquiry: str, reply: str, client: LLMClient, cat: Catalogue) -> tuple[Outcome, list[LLMResult]]:
    """Reduce a free-text reply to the same Outcome the pipeline is scored on.

    A malformed audit gets one corrective call, as in extraction: a grader slip must not be
    scored as a failure of the baseline it was grading.
    """
    system = GRADER_SYSTEM.format(business=cat.business, catalogue=catalogue_text(cat))
    user = f"Client enquiry:\n{enquiry.strip()}\n\nReply to audit:\n{reply.strip()}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls: list[LLMResult] = []
    for attempt in range(2):
        result = client.complete(messages, schema=grader_schema(cat), schema_name="audit", max_tokens=600)
        calls.append(result)
        try:
            data = _parse_audit(result.text)
        except ValueError as exc:
            messages = messages + [
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": f"That was not a valid audit ({exc}). Return the corrected JSON object only, with every field."},
            ]
            continue
        route = {"proposal": "proposal", "cannot_meet": "infeasible", "clarification": "clarify"}.get(data.kind, "other")
        outcome = Outcome(
            route=route,
            venue=data.venue_id,
            services=data.service_ids,
            total=data.stated_total,
            unmet=data.reasons,
            assumed=sorted(set(data.flagged_assumptions)),
            asked=data.asks_about,
            notices=data.declines,
            alternative_totals=data.alternative_totals,
            venues_offered=sorted(set(data.venues_offered)),
            promises=data.promises,
        )
        return outcome, calls
    raise GradingError("The grader did not return a usable audit after two attempts.", calls)
