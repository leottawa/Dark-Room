"""Test cases and scoring, shared by every arm of the evaluation.

Each case carries an expected outcome written by hand from the catalogue rules, and the brief a
careful reader would extract. Every arm is reduced to the same Outcome and judged by the same
checks, so a pass means the same thing whichever system produced the reply.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .models import Accessibility, Brief, Budget, DateMention, Duration, FieldIssue, Headcount, OtherRequest, ServiceWish, VenueWish
from .pipeline import RunResult

Route = Literal["clarify", "proposal", "infeasible"]


class Expect(BaseModel):
    route: Route
    venue: str | None = None
    services: list[str] | None = None
    total: float | None = None
    unmet: list[str] = []
    assume: list[str] = []
    ask: list[str] = []
    notices: list[str] = []
    alternative_totals: list[float] = []
    forbid_venues: list[str] = []


class IdealBrief(BaseModel):
    """The brief a careful reader would take from the enquiry, in compact form."""

    n: list[int] | None = None  # [min, max]
    approximate: bool = False
    hours: float | None = None
    budget: float | None = None
    basis: Literal["total", "per_person"] | None = None
    access: Literal["required", "preferred", "not_needed", "not_mentioned"] = "not_mentioned"
    venue: str | None = None
    venue_need: Literal["required", "preferred", "not_mentioned"] = "not_mentioned"
    required: list[str] = []
    preferred: list[str] = []
    beyond: list[str] = []
    date: bool = False
    others: list[list[str]] = []  # [category, need]
    # Fields whose value is a reading of looser words ("with food" as lunch), to be confirmed.
    interpreted: list[str] = []
    # Other readings that are also right, by field name: {"hours": [null, 4]}.
    also_ok: dict[str, list] = {}

    def to_brief(self) -> Brief:
        services = [ServiceWish(service_id=s, need="required", beyond_terms="beyond the listed terms" if s in self.beyond else None) for s in self.required]
        services += [ServiceWish(service_id=s, need="preferred") for s in self.preferred]
        return Brief(
            headcount=Headcount(min=self.n[0], max=self.n[1], approximate=self.approximate) if self.n else Headcount(),
            duration=Duration(hours=self.hours),
            budget=Budget(amount=self.budget, basis=(self.basis or "total") if self.budget is not None else None),
            accessibility=Accessibility(need=self.access),
            venue=VenueWish(venue_id=self.venue, need=self.venue_need),
            services=services,
            date=DateMention(mentioned=self.date),
            other_requests=[OtherRequest(category=category, description=category.replace("_", " "), need=need) for category, need in self.others],
        )


    def issues(self) -> list[FieldIssue]:
        return [FieldIssue(field=name, kind="interpreted", message=f"Reading of {name} to confirm.") for name in self.interpreted]


class Case(BaseModel):
    id: str
    category: str
    source: Literal["case", "added"]
    enquiry: str
    brief: IdealBrief
    expect: Expect
    note: str = ""


def load_cases(path: str | Path) -> list[Case]:
    return [Case.model_validate(item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


class Outcome(BaseModel):
    """What a reply amounts to, whichever system wrote it."""

    route: str
    venue: str | None = None
    services: list[str] = []
    total: float | None = None
    unmet: list[str] = []
    assumed: list[str] = []
    asked: list[str] = []
    notices: list[str] = []
    alternative_totals: list[float] = []
    venues_offered: list[str] = []
    promises: list[str] = []


class CheckResult(BaseModel):
    name: str
    passed: bool
    critical: bool = False
    detail: str = ""


def outcome_from_run(result: RunResult) -> Outcome:
    a = result.assessment
    offered = [alt.package.venue_id for alt in a.alternatives]
    if a.route == "proposal" and a.package:
        offered.append(a.package.venue_id)
    return Outcome(
        route=a.route,
        venue=a.package.venue_id if a.package else None,
        services=a.package.service_ids if a.package else [],
        total=a.package.total if a.package else None,
        unmet=[u.kind for u in a.unmet],
        assumed=sorted({x.field.split(".")[0] for x in a.assumptions if x.sensitive}),
        asked=[q.field.split(".")[0] for q in a.questions],
        notices=[n.category for n in a.notices],
        alternative_totals=[alt.package.total for alt in a.alternatives],
        venues_offered=sorted(set(offered)),
        promises=[],
    )


def score(expect: Expect, got: Outcome) -> list[CheckResult]:
    """Critical checks are the errors that would mislead a client if the draft were sent."""
    results: list[CheckResult] = []

    def check(name: str, passed: bool, detail: str, critical: bool = False) -> None:
        results.append(CheckResult(name=name, passed=passed, critical=critical and not passed, detail="" if passed else detail))

    # Proposing a package when the request cannot be met, or cannot yet be priced, hides the problem.
    hides_problem = got.route == "proposal" and expect.route != "proposal"
    check("route", got.route == expect.route, f"expected {expect.route}, got {got.route}", critical=hides_problem)
    if expect.venue is not None:
        check("venue", got.venue == expect.venue, f"expected {expect.venue}, got {got.venue}", critical=True)
    if expect.services is not None:
        check("services", set(got.services) == set(expect.services), f"expected {sorted(expect.services)}, got {sorted(got.services)}", critical=True)
    if expect.total is not None:
        ok = got.total is not None and abs(got.total - expect.total) < 0.01
        check("total", ok, f"expected {expect.total:g}, got {got.total}", critical=True)
    if expect.route == "infeasible":
        check("reasons", set(got.unmet) == set(expect.unmet), f"expected {sorted(expect.unmet)}, got {sorted(got.unmet)}")
    missing = sorted(set(expect.assume) - set(got.assumed))
    check("assumptions_flagged", not missing, f"not flagged: {missing}")
    spurious = sorted(set(got.assumed) - set(expect.assume))
    check("no_needless_assumption", not spurious, f"flagged without need: {spurious}")
    missing = sorted(set(expect.ask) - set(got.asked))
    check("questions", not missing, f"not asked: {missing}")
    missing = sorted(set(expect.notices) - set(got.notices))
    check("notices", not missing, f"not stated: {missing}")
    missing = [t for t in expect.alternative_totals if not any(abs(t - x) < 0.01 for x in got.alternative_totals)]
    check("alternatives", not missing, f"alternatives not offered at totals: {missing}")
    bad = sorted(set(expect.forbid_venues) & set(got.venues_offered))
    check("no_unsuitable_venue", not bad, f"offered {bad}", critical=True)
    check("no_promise", not got.promises, f"promised: {got.promises}", critical=True)
    return results


def usable(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)


def critical_error(results: list[CheckResult]) -> bool:
    return any(r.critical for r in results)


def extraction_errors(ideal: IdealBrief, got: Brief) -> list[str]:
    """Names of the brief fields the model read differently from the ideal brief."""
    found = {
        "n": [got.headcount.min, got.headcount.max] if got.headcount.max is not None else None,
        "approximate": got.headcount.approximate,
        "hours": got.duration.hours,
        "budget": got.budget.amount,
        "basis": got.budget.basis,
        "access": got.accessibility.need,
        "venue": got.venue.venue_id,
        "venue_need": got.venue.need,
        "required": sorted(s.service_id for s in got.services if s.need == "required"),
        "preferred": sorted(s.service_id for s in got.services if s.need == "preferred"),
        "beyond": sorted(s.service_id for s in got.services if s.beyond_terms),
        "date": got.date.mentioned,
        "others": _requests([r.category, r.need] for r in got.other_requests),
    }
    wanted = ideal.model_dump(exclude={"interpreted", "also_ok"})
    wanted["basis"] = (ideal.basis or "total") if ideal.budget is not None else None
    wanted["others"] = _requests(ideal.others)
    for key in ("required", "preferred", "beyond"):
        wanted[key] = sorted(wanted[key])
    return [name for name, value in found.items() if value != wanted[name] and value not in ideal.also_ok.get(name, [])]


def _requests(pairs) -> list[list[str]]:
    """Other requests as comparable pairs. Whether a request is required only matters where it
    can block a proposal, so the need is ignored for the other categories."""
    blocking = {"unlisted_service", "allergen_guarantee"}
    unique = {(category, need if category in blocking else "*") for category, need in pairs}
    return [list(pair) for pair in sorted(unique)]
