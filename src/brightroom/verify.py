"""Step 5: check the draft before a person sees it.

The model's passages are the only free text in the email, so they are the only place an
unsupported figure, an invented service or a promise could enter. The checks are deliberately
strict: a false alarm costs one rewrite or a plainer template, a miss could reach a client.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from .catalogue import Catalogue, money
from .draft import FOOTER, Prose
from .models import Assessment
from .textutil import numbers_in

MAX_WORDS = 70

# Wording that commits BrightRoom to something only a person can decide.
_COMMITMENTS = [
    r"\b(we|i)\s+(can|will|could|would|shall|are able to|are happy to|are pleased to|have)?\s*(confirm|book|reserve|hold|guarantee|secure)\b",
    r"\b(is|are|has been|have been|will be)\s+(now\s+)?(confirmed|booked|reserved|held|guaranteed|secured)\b",
    r"\byour\s+(booking|reservation)\b",
    r"\b(discount\w*|reduction|complimentary|free of charge|for free|at no (extra |additional )?(cost|charge)|on the house|guarantee\w*)\b",
    r"\b(availab\w+|reserv\w+)\b",
]
# Common extras that are not in the catalogue: a tripwire for invented services.
_EXTRAS = [
    "parking", "photograph", "microphone", "flipchart", "whiteboard", "breakfast", "dinner", "drinks", "wine",
    "champagne", "snack", "transport", "shuttle", "hotel", "accommodation", "sound system", "livestream",
    "video conferenc", "decoration", "music", "catering team", "gift",
]


class Violation(BaseModel):
    code: str
    detail: str


def check_prose(prose: Prose, facts: dict, cat: Catalogue) -> list[Violation]:
    """Violations in the model's passages, judged only against the facts it was given."""
    violations: list[Violation] = []
    text = " ".join([prose.opening, prose.body, prose.closing])
    lowered = text.lower()
    given = json.dumps(facts, ensure_ascii=False).lower()

    allowed = numbers_in(given)
    stray = sorted(n for n in numbers_in(text) if not any(abs(n - ok) < 0.005 for ok in allowed))
    if stray:
        violations.append(Violation(code="unsupported_figure", detail=f"Figures not in the facts: {', '.join(f'{n:g}' for n in stray)}."))

    for pattern in _COMMITMENTS:
        match = re.search(pattern, lowered)
        if match:
            violations.append(Violation(code="commitment", detail=f'Do not write "{match.group(0)}": the system adds notes on availability, bookings and prices.'))

    for item in [*cat.venues, *cat.services]:
        name = item.name.lower()
        if re.search(rf"\b{re.escape(name)}\b", lowered) and name not in given:
            violations.append(Violation(code="item_not_in_draft", detail=f'"{item.name}" is not part of this draft.'))
    for extra in _EXTRAS:
        if extra in lowered and extra not in given:
            violations.append(Violation(code="unlisted_item", detail=f'"{extra}" is not in the catalogue or the facts.'))

    for name, passage in (("opening", prose.opening), ("body", prose.body), ("closing", prose.closing)):
        if not passage.strip():
            violations.append(Violation(code="empty", detail=f"The {name} is empty."))
        if len(passage.split()) > MAX_WORDS:
            violations.append(Violation(code="too_long", detail=f"The {name} is longer than {MAX_WORDS} words."))
        if "\n" in passage.strip() or re.search(r"^\s*[-*\d]+[.)]?\s", passage):
            violations.append(Violation(code="format", detail=f"The {name} must be plain sentences."))
    return violations


def check_draft(draft: str, assessment: Assessment) -> list[Violation]:
    """Invariants of the assembled email. These hold by construction; the check guards regressions."""
    violations: list[Violation] = []

    def require(condition: bool, code: str, detail: str) -> None:
        if not condition:
            violations.append(Violation(code=code, detail=detail))

    require(FOOTER in draft, "missing_footer", "The draft and availability notice is missing.")
    a = assessment
    if a.route == "proposal" and a.package:
        require(f"Total: {money(a.package.total)}" in draft, "missing_total", "The total is missing.")
        for line in a.package.lines:
            require(f"- {line.name}" in draft, "missing_line", f"{line.name} is missing from the package.")
        for assumption in a.assumptions:
            require(assumption.text in draft, "missing_assumption", f"Assumption not shown: {assumption.text}")
    if a.route == "infeasible":
        for unmet in a.unmet:
            require(unmet.detail in draft, "missing_reason", f"Reason not shown: {unmet.detail}")
        for alt in a.alternatives:
            require(all(t in draft for t in alt.trade_offs), "missing_trade_off", f"Trade-off not shown for: {alt.label}")
    if a.route == "clarify":
        for question in a.questions:
            require(question.text in draft, "missing_question", f"Question not shown: {question.text}")
    for notice in a.notices:
        require(notice.text in draft, "missing_notice", f"Notice not shown: {notice.text}")
    return violations
