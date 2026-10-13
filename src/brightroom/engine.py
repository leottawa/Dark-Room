"""Deterministic core: routing, venue choice, pricing, verdict and alternatives.

Nothing here calls a model. Every decision that would be costly to get wrong (a total, a venue
that cannot host the event, a must-have quietly dropped) is computed from the catalogue, so the
same brief always produces the same assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Sequence

from .catalogue import Catalogue, Service, Venue, hours, money
from .models import (
    Addon,
    Alternative,
    Assessment,
    Assumption,
    Brief,
    Check,
    Evaluation,
    FieldIssue,
    Line,
    Notice,
    Package,
    Question,
    Scenario,
    Unmet,
)

MAX_ALTERNATIVES = 5
CONSTRAINTS = ("capacity", "accessibility", "duration")


@dataclass(frozen=True)
class Spec:
    """A fully specified request: all the pricing rules need, with nothing left to interpret."""

    headcount: int
    duration: float
    required: tuple[str, ...] = ()
    accessible: bool = False
    venue_id: str | None = None
    budget: float | None = None
    budget_per_person: float | None = None

    @property
    def budget_limit(self) -> float | None:
        if self.budget is not None:
            return self.budget
        if self.budget_per_person is not None:
            return self.budget_per_person * self.headcount
        return None


# --- Pricing one fully specified request ------------------------------------------------------


def evaluate(spec: Spec, cat: Catalogue) -> Evaluation:
    """Pick the cheapest venue that can host the request, price it and check every constraint."""
    candidates = [v for v in cat.venues if spec.venue_id in (None, v.id)]
    if not candidates:
        raise ValueError(f"unknown venue id: {spec.venue_id}")
    failures = {v.id: _venue_failures(v, spec) for v in candidates}
    eligible = [v for v in candidates if not failures[v.id]]
    venue = min(eligible, key=lambda v: (v.price, v.max_people, v.id)) if eligible else None
    package = _package(venue, spec, cat) if venue else None
    checks = _checks(spec, candidates, failures, venue, package)
    feasible = venue is not None and all(c.status != "fail" for c in checks)
    return Evaluation(package=package, checks=checks, feasible=feasible)


def _venue_failures(venue: Venue, spec: Spec) -> list[str]:
    failed = []
    if spec.headcount > venue.max_people:
        failed.append("capacity")
    if spec.accessible and not venue.wheelchair_accessible:
        failed.append("accessibility")
    if spec.duration > venue.duration_hours:
        failed.append("duration")
    return failed


def _service_line(service: Service, headcount: int) -> Line:
    quantity = headcount if service.pricing == "per_person" else 1
    return Line(
        item_id=service.id,
        name=service.name,
        kind="service",
        quantity=quantity,
        unit_price=service.price,
        amount=service.price * quantity,
        conditions=service.conditions,
    )


def _package(venue: Venue, spec: Spec, cat: Catalogue) -> Package:
    lines = [
        Line(
            item_id=venue.id,
            name=venue.name,
            kind="venue",
            quantity=1,
            unit_price=venue.price,
            amount=venue.price,
            conditions=venue.conditions,
        )
    ]
    lines += [_service_line(cat.service(sid), spec.headcount) for sid in spec.required]
    return Package(
        venue_id=venue.id,
        venue_name=venue.name,
        headcount=spec.headcount,
        duration_hours=spec.duration,
        lines=lines,
        total=sum(line.amount for line in lines),
    )


def _checks(
    spec: Spec,
    candidates: list[Venue],
    failures: dict[str, list[str]],
    venue: Venue | None,
    package: Package | None,
) -> list[Check]:
    pinned = spec.venue_id is not None

    def fails_everywhere(name: str) -> bool:
        return all(name in failures[v.id] for v in candidates)

    # When no venue fits and no single constraint rules out every venue, the constraints clash
    # with each other: each one that rules out some venue is reported as failing.
    clash = venue is None and not any(fails_everywhere(c) for c in CONSTRAINTS)

    def failed(name: str) -> bool:
        if venue is not None:
            return False
        return fails_everywhere(name) or (clash and any(name in failures[v.id] for v in candidates))

    clash_note = " No single venue meets all the requirements together." if clash else ""
    n, d = spec.headcount, spec.duration
    checks = []

    if failed("capacity"):
        biggest = max(candidates, key=lambda v: v.max_people)
        scope = f"the requested {biggest.name}" if pinned else f"our largest venue, the {biggest.name}"
        detail = f"{n} attendees exceed the capacity of {scope} (maximum {biggest.max_people}).{clash_note}"
        checks.append(Check(name="capacity", status="fail", detail=detail))
    elif venue is not None:
        detail = f"{n} attendees; the {venue.name} holds up to {venue.max_people}."
        checks.append(Check(name="capacity", status="pass", detail=detail))
    else:
        checks.append(Check(name="capacity", status="pass", detail=f"{n} attendees can be hosted."))

    if not spec.accessible:
        checks.append(Check(name="accessibility", status="n/a", detail="Step-free access not requested."))
    elif failed("accessibility"):
        if pinned:
            detail = f"The requested {candidates[0].name} is {candidates[0].access_note}.{clash_note}"
        else:
            detail = f"No venue offers step-free access for this request.{clash_note}"
        checks.append(Check(name="accessibility", status="fail", detail=detail))
    elif venue is not None:
        detail = f"The {venue.name} is {venue.access_note}."
        checks.append(Check(name="accessibility", status="pass", detail=detail))
    else:
        checks.append(Check(name="accessibility", status="pass", detail="A step-free venue exists."))

    if failed("duration"):
        longest = max(v.duration_hours for v in candidates)
        detail = (
            f"{hours(d)} hours requested; our venues are offered for {hours(longest)} hours "
            f"and no longer-duration price exists.{clash_note}"
        )
        checks.append(Check(name="duration", status="fail", detail=detail))
    elif venue is not None and d < venue.duration_hours:
        detail = (
            f"{hours(d)} hours requested; venues are booked as {hours(venue.duration_hours)}-hour "
            "sessions at the listed price."
        )
        checks.append(Check(name="duration", status="pass", detail=detail))
    else:
        checks.append(Check(name="duration", status="pass", detail=f"{hours(d)}-hour session."))

    limit = spec.budget_limit
    if limit is None:
        checks.append(Check(name="budget", status="unknown", detail="No budget stated."))
    elif package is None:
        detail = "Cannot be priced: no venue meets the request."
        checks.append(Check(name="budget", status="unknown", detail=detail))
    elif package.total <= limit + 1e-9:
        left = limit - package.total
        margin = "exactly on budget" if left < 0.005 else f"{money(left)} remaining"
        detail = f"Total {money(package.total)} is within the budget of {money(limit)} ({margin})."
        checks.append(Check(name="budget", status="pass", detail=detail))
    else:
        over = package.total - limit
        detail = f"Total {money(package.total)} is {money(over)} above the budget of {money(limit)}."
        checks.append(Check(name="budget", status="fail", detail=detail))
    return checks


# --- From a brief to an assessment ------------------------------------------------------------


def assess(brief: Brief, cat: Catalogue, issues: Sequence[FieldIssue] = ()) -> Assessment:
    """Route the brief (clarify, proposal or infeasible) and build everything the reviewer sees."""
    warnings = [i.message for i in issues if i.kind == "quote_not_found"]
    lo, hi = _headcount_bounds(brief)
    if hi is None:
        return _clarify(brief, cat, issues, warnings)

    spec = _base_spec(brief, cat, hi)
    ev = evaluate(spec, cat)
    blocked = _blocking_requests(brief, cat)
    if ev.feasible and not blocked:
        return _proposal(brief, cat, issues, warnings, spec, ev, lo)
    return _infeasible(brief, cat, issues, warnings, spec, ev, blocked)


def _headcount_bounds(brief: Brief) -> tuple[int | None, int | None]:
    lo, hi = brief.headcount.min, brief.headcount.max
    lo, hi = (lo if lo is not None else hi), (hi if hi is not None else lo)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _required_services(brief: Brief) -> tuple[str, ...]:
    return tuple(dict.fromkeys(s.service_id for s in brief.services if s.need == "required"))


def _base_spec(brief: Brief, cat: Catalogue, headcount: int) -> Spec:
    per_person = brief.budget.basis == "per_person"
    return Spec(
        headcount=headcount,
        duration=brief.duration.hours or cat.standard_duration,
        required=_required_services(brief),
        accessible=brief.accessibility.need == "required",
        venue_id=brief.venue.venue_id if brief.venue.need == "required" else None,
        budget=None if per_person else brief.budget.amount,
        budget_per_person=brief.budget.amount if per_person else None,
    )


def _blocking_requests(brief: Brief, cat: Catalogue) -> list[Unmet]:
    """Must-haves that no catalogue package can deliver, whatever the venue or budget."""
    unmet = []
    for req in brief.other_requests:
        policy = cat.policies[req.category]
        if req.need == "required" and not policy.offerable and policy.blocks_when_required:
            short = f"{_cap(req.description)}: not available"
            detail = f"{_cap(req.description)}: {policy.statement}"
            unmet.append(Unmet(kind=req.category, short=short, detail=detail, option=f"without {_uncap(req.description)}"))
    for wish in brief.services:
        if wish.need == "required" and wish.beyond_terms:
            service = cat.service(wish.service_id)
            short = f"{service.name} only as listed ({service.conditions.rstrip('.')})"
            detail = f"{service.name}: requested beyond the catalogue terms ({wish.beyond_terms}). Listed terms: {service.conditions}"
            unmet.append(Unmet(kind="beyond_terms", short=short, detail=detail, option=f"with the {service.name.lower()} as listed"))
    return unmet


def _notices(brief: Brief, cat: Catalogue, package: Package | None) -> list[Notice]:
    notices = []
    for req in brief.other_requests:
        policy = cat.policies[req.category]
        statement, offerable, subject = policy.statement, policy.offerable, ""
        if req.category in ("unlisted_service", "other"):
            subject = _cap(req.description)
        needs = policy.requires_service
        if needs and (package is None or needs not in package.service_ids):
            statement = f"{policy.statement} This applies to {cat.service(needs).name.lower()}, which is not part of this request."
            offerable = False
        notices.append(Notice(category=req.category, statement=statement, subject=subject, offerable=offerable, quote=req.quote))
    for wish in brief.services:
        if wish.beyond_terms:
            service = cat.service(wish.service_id)
            statement = f"Requested beyond the catalogue terms ({wish.beyond_terms}). We can offer it only as listed: {service.conditions}"
            notices.append(Notice(category="beyond_terms", statement=statement, subject=service.name, quote=wish.quote))
    if brief.date.mentioned and not any(n.category == "booking_or_date_confirmation" for n in notices):
        notices.append(Notice(category="date", statement=cat.policies["date"].statement, quote=brief.date.quote))
    unique: dict[tuple[str, str], Notice] = {}
    for notice in notices:
        unique.setdefault((notice.category, notice.subject.lower()), notice)
    return list(unique.values())


def _common_assumptions(brief: Brief, cat: Catalogue, spec: Spec, lo: int | None, issues: Sequence[FieldIssue]) -> list[Assumption]:
    out = [Assumption(field=i.field, text=i.message, sensitive=True) for i in issues if i.kind == "interpreted"]
    n = spec.headcount
    if lo is not None and lo != n:
        out.append(Assumption(field="headcount", text=f"Headcount given as {lo} to {n}: priced for {n}, the upper bound.", sensitive=True))
    elif brief.headcount.approximate:
        quoted = f' ("{brief.headcount.quote}")' if brief.headcount.quote else ""
        out.append(Assumption(field="headcount", text=f"Headcount is approximate{quoted}: priced for {n}.", sensitive=True))
    if brief.duration.hours is None:
        text = f"No duration stated: priced for our standard {hours(cat.standard_duration)}-hour session."
        out.append(Assumption(field="duration", text=text))
    budget = brief.budget
    if budget.amount is None:
        if budget.quote:
            text = f'Budget described as "{budget.quote}" with no figure: treated as not stated.'
        else:
            text = "No budget stated."
        out.append(Assumption(field="budget", text=text))
    else:
        if budget.basis == "per_person":
            text = f"Budget of {money(budget.amount)} per person for {n} attendees: {money(budget.amount * n)} in total."
            out.append(Assumption(field="budget", text=text))
        elif budget.basis is None:
            out.append(Assumption(field="budget", text=f"Budget of {money(budget.amount)} taken as a total for the event."))
        if budget.approximate:
            limit = spec.budget_limit or 0.0
            text = f"Budget stated approximately: treated as a ceiling of {money(limit)}."
            out.append(Assumption(field="budget", text=text))
    return out


def _proposal(brief, cat, issues, warnings, spec: Spec, ev: Evaluation, lo: int | None) -> Assessment:
    spec, ev, addons, notes = _apply_preferences(brief, cat, spec, ev)
    package = ev.package
    assert package is not None
    venue = cat.venue(package.venue_id)
    assumptions = _common_assumptions(brief, cat, spec, lo, issues) + notes
    scenarios: list[Scenario] = []
    questions: list[Question] = []

    increment = sum(cat.service(sid).price for sid in package.service_ids if cat.service(sid).pricing == "per_person")
    headroom = _headroom(spec, ev, cat)
    uncertain_headcount = brief.headcount.approximate or (lo is not None and lo != spec.headcount)
    if uncertain_headcount:
        extra = f"; each additional attendee adds {money(increment)}" if increment else ""
        if headroom > spec.headcount:
            text = f"The package holds as quoted for up to {headroom} attendees{extra}."
        else:
            text = f"The package does not hold for more than {spec.headcount} attendees."
        assumptions.append(Assumption(field="headcount", text=text, sensitive=True))
        questions.append(Question(field="headcount", text="Could you confirm the final number of attendees?"))
    if lo is not None and lo != spec.headcount:
        low = evaluate(replace(spec, headcount=lo), cat)
        scenarios.append(_scenario(f"With {lo} attendees", low))

    if brief.accessibility.need == "not_mentioned" and not venue.wheelchair_accessible:
        text = f"Step-free access was not mentioned: the {venue.name} is {venue.access_note}."
        assumptions.append(Assumption(field="accessibility", text=text, sensitive=True))
        pinned = brief.venue.venue_id if brief.venue.need == "required" else None
        accessible = evaluate(replace(spec, accessible=True, venue_id=pinned), cat)
        scenarios.append(_scenario("If step-free access is needed", accessible))
        text = f"Does anyone attending need step-free access? The {venue.name} is {venue.access_note}."
        questions.append(Question(field="accessibility", text=text))

    if not brief.services:
        text = "No catering or equipment was requested: the package covers the venue only."
        assumptions.append(Assumption(field="services", text=text))
        addons += [_addon(cat.service(sid), spec.headcount, "Available on request.", requested=False) for sid in cat.service_ids]
        names = ", ".join(s.name.lower() for s in cat.services)
        questions.append(Question(field="services", text=f"Would you like to add catering or equipment? We offer: {names}."))

    for issue in issues:
        if issue.kind == "interpreted":
            questions.append(Question(field=issue.field, text=f"Could you confirm our reading? {issue.message}"))
    if brief.budget.amount is None:
        questions.append(Question(field="budget", text="Do you have a budget ceiling for the event?"))
    if not brief.date.mentioned:
        questions.append(Question(field="date", text="Which date or dates do you have in mind? We will then check availability."))

    provisional = any(a.sensitive for a in assumptions)
    limit = spec.budget_limit
    against = f" of the {money(limit)} budget" if limit is not None else ", no budget stated"
    if provisional:
        count = len({a.field for a in assumptions if a.sensitive})
        headline = f"Provisional: {money(package.total)}{against}, with {count} point{'s' if count != 1 else ''} to confirm"
    else:
        headline = f"Fits the brief: {money(package.total)}{against}"
    return Assessment(
        route="proposal",
        provisional=provisional,
        headline=headline,
        package=package,
        checks=ev.checks,
        assumptions=assumptions,
        scenarios=scenarios,
        addons=addons,
        notices=_notices(brief, cat, package),
        questions=questions,
        warnings=warnings,
        budget_total=limit,
        headroom=headroom,
        per_person_increment=increment,
    )


def _apply_preferences(brief: Brief, cat: Catalogue, spec: Spec, ev: Evaluation) -> tuple[Spec, Evaluation, list[Addon], list[Assumption]]:
    """Add what the client would like, as far as the stated requirements and budget allow.

    Preferences never displace a requirement: each is tried on top of the feasible package and
    kept only if the result is still feasible.
    """
    notes: list[Assumption] = []
    addons: list[Addon] = []
    assert ev.package is not None

    wish = brief.venue
    if wish.need == "preferred" and wish.venue_id and ev.package.venue_id != wish.venue_id:
        trial = replace(spec, venue_id=wish.venue_id)
        result = evaluate(trial, cat)
        name = cat.venue(wish.venue_id).name
        if result.feasible:
            spec, ev = trial, result
            notes.append(Assumption(field="venue", text=f"The {name} is used, as preferred."))
        else:
            reason = _first_failure(result)
            notes.append(Assumption(field="venue", text=f"The preferred {name} is not used: {reason}"))

    assert ev.package is not None
    if brief.accessibility.need == "preferred" and not cat.venue(ev.package.venue_id).wheelchair_accessible:
        pinned = wish.venue_id if wish.need == "required" else None
        trial = replace(spec, accessible=True, venue_id=pinned)
        result = evaluate(trial, cat)
        if result.feasible:
            spec, ev = trial, result
            notes.append(Assumption(field="accessibility", text="A step-free venue is used, as preferred."))
        else:
            reason = _first_failure(result)
            text = f"Step-free access was preferred but is not included: {reason}"
            notes.append(Assumption(field="accessibility", text=text, sensitive=True))

    preferred = dict.fromkeys(s.service_id for s in brief.services if s.need == "preferred" and s.service_id not in spec.required)
    for sid in preferred:
        service = cat.service(sid)
        if spec.budget_limit is None:
            addons.append(_addon(service, spec.headcount, "Not included: no budget was stated, so it is offered as an option.", requested=True))
            continue
        trial = replace(spec, required=spec.required + (sid,))
        result = evaluate(trial, cat)
        if result.feasible:
            spec, ev = trial, result
            notes.append(Assumption(field="services", text=f"{service.name} is included, as it fits within the budget."))
        else:
            addons.append(_addon(service, spec.headcount, "Not included: it would take the total above the budget.", requested=True))
    return spec, ev, addons, notes


def _addon(service: Service, headcount: int, reason: str, requested: bool) -> Addon:
    line = _service_line(service, headcount)
    return Addon(service_id=service.id, name=service.name, amount=line.amount, conditions=service.conditions, reason=reason, requested=requested)


def _first_failure(ev: Evaluation) -> str:
    return next((c.detail for c in ev.checks if c.status == "fail"), "it does not fit the request.")


def _scenario(label: str, ev: Evaluation) -> Scenario:
    failing = [c.detail for c in ev.checks if c.status == "fail"]
    if ev.package is None:
        return Scenario(label=label, feasible=False, note=" ".join(failing))
    budget = next(c for c in ev.checks if c.name == "budget")
    note = " ".join(failing) if failing else (budget.detail if budget.status == "pass" else "")
    return Scenario(label=label, feasible=ev.feasible, package=ev.package, note=note)


def _headroom(spec: Spec, ev: Evaluation, cat: Catalogue) -> int:
    """Largest headcount for which the same venue still hosts and the budget still holds."""
    assert ev.package is not None
    best = spec.headcount
    for n in range(spec.headcount + 1, max(v.max_people for v in cat.venues) + 1):
        trial = evaluate(replace(spec, headcount=n), cat)
        if not (trial.feasible and trial.package and trial.package.venue_id == ev.package.venue_id):
            break
        best = n
    return best


def _infeasible(brief, cat, issues, warnings, spec: Spec, ev: Evaluation, blocked: list[Unmet]) -> Assessment:
    lo, _ = _headcount_bounds(brief)
    unmet = [Unmet(kind=c.name, short=_short_reason(c, spec, ev, cat), detail=c.detail) for c in ev.checks if c.status == "fail"]
    unmet += blocked
    questions = [Question(field=i.field, text=f"Could you confirm our reading? {i.message}") for i in issues if i.kind == "interpreted"]
    alternatives = _alternatives(spec, ev, cat, blocked)
    if alternatives:
        questions.append(Question(field="alternatives", text="Which of the options above would suit you best?"))
    assumptions = _common_assumptions(brief, cat, spec, lo, issues)
    return Assessment(
        route="infeasible",
        provisional=any(a.sensitive for a in assumptions),
        headline="Cannot meet the brief as stated: " + "; ".join(u.short for u in unmet),
        package=ev.package,
        checks=ev.checks,
        unmet=unmet,
        assumptions=assumptions,
        notices=_notices(brief, cat, ev.package),
        alternatives=alternatives,
        questions=questions,
        warnings=warnings,
        budget_total=spec.budget_limit,
    )


def _short_reason(check: Check, spec: Spec, ev: Evaluation, cat: Catalogue) -> str:
    pool = [v for v in cat.venues if spec.venue_id in (None, v.id)]
    if check.name == "capacity":
        return f"{spec.headcount} attendees exceed the maximum of {max(v.max_people for v in pool)}"
    if check.name == "duration":
        return f"{hours(spec.duration)} hours exceed the {hours(max(v.duration_hours for v in pool))}-hour session"
    if check.name == "accessibility":
        return "no step-free venue for this request"
    limit = spec.budget_limit
    if ev.package is not None and limit is not None:
        return f"{money(ev.package.total - limit)} over budget"
    return "over budget"


def _alternatives(spec: Spec, ev: Evaluation, cat: Catalogue, blocked: list[Unmet]) -> list[Alternative]:
    """Nearest packages the catalogue can deliver, each with what the client gives up.

    Step-free access is never traded away. Limits the catalogue cannot stretch (longest session,
    largest venue) are applied first; budget options are then offered on top of them.
    """
    alternatives: list[Alternative] = []
    seen: set[tuple] = set()
    blocked_notes = [f"{u.short}." for u in blocked]

    def offer(label: str, trial: Spec, trade_offs: list[str], changes_requirement: bool) -> bool:
        result = evaluate(trial, cat)
        if not (result.feasible and result.package):
            return False
        pkg = result.package
        key = (pkg.venue_id, pkg.headcount, pkg.duration_hours, tuple(pkg.service_ids))
        if key in seen:
            return False
        seen.add(key)
        alternatives.append(
            Alternative(
                label=label,
                package=pkg,
                trade_offs=trade_offs + blocked_notes,
                changes_requirement=changes_requirement or bool(blocked),
            )
        )
        return True

    if ev.feasible and blocked:
        offer(f"Proceed {_join([u.option for u in blocked])}", spec, [], True)
        return alternatives

    if spec.venue_id:
        other = evaluate(replace(spec, venue_id=None), cat)
        if other.feasible and other.package and other.package.venue_id != spec.venue_id:
            requested = cat.venue(spec.venue_id).name
            offer(f"Use the {other.package.venue_name} instead of the {requested}", replace(spec, venue_id=None), ["A different venue from the one requested."], True)

    pool = [v for v in cat.venues if spec.venue_id in (None, v.id) and (v.wheelchair_accessible or not spec.accessible)]
    if not pool:
        if spec.venue_id:
            # The requested venue can never meet the request (no step-free access), so the nearest
            # options are found at the other venues, with the change of venue stated as a trade-off.
            requested = cat.venue(spec.venue_id).name
            free = replace(spec, venue_id=None)
            for alt in _alternatives(free, evaluate(free, cat), cat, blocked):
                key = (alt.package.venue_id, alt.package.headcount, alt.package.duration_hours, tuple(alt.package.service_ids))
                if key in seen:
                    continue
                seen.add(key)
                label = f"Use the {alt.package.venue_name} instead of the {requested}, and {_uncap(alt.label)}"
                trade_offs = ["A different venue from the one requested.", *alt.trade_offs]
                alternatives.append(Alternative(label=label, package=alt.package, trade_offs=trade_offs, changes_requirement=True))
        return alternatives[:MAX_ALTERNATIVES]
    base, steps, forced = spec, [], []
    longest = max(v.duration_hours for v in pool)
    if base.duration > longest:
        base = replace(base, duration=longest)
        steps.append(f"shorten the session to {hours(longest)} hours")
        forced.append(f"{hours(spec.duration - longest)} hours shorter than requested.")
    largest = max(v.max_people for v in pool if v.duration_hours >= base.duration)
    if base.headcount > largest:
        base = replace(base, headcount=largest)
        steps.append(f"limit attendance to {largest}")
        forced.append(f"{spec.headcount - largest} fewer attendees than requested.")
    if steps:
        offer(_cap(_join(steps)), base, forced, True)

    priced = evaluate(base, cat)
    limit = base.budget_limit
    if priced.package is None or limit is None or priced.package.total <= limit + 1e-9:
        return alternatives[:MAX_ALTERNATIVES]

    total = priced.package.total
    per_head = f" ({money(total / base.headcount)} per person)" if base.budget_per_person is not None else ""
    label = _cap(_join(steps + [f"increase the budget to {money(total)}{per_head}"]))
    no_ceiling = replace(base, budget=None, budget_per_person=None)
    offer(label, no_ceiling, forced + [f"{money(total - limit)} above the stated budget."], bool(steps))

    for n in range(base.headcount - 1, 0, -1):
        trial = replace(base, headcount=n)
        if evaluate(trial, cat).feasible:
            other_steps = [step for step in steps if not step.startswith("limit attendance")]
            label = _cap(_join(other_steps + [f"reduce attendance to {n}"]))
            cut = [f"{spec.headcount - n} fewer attendees than requested."]
            kept = [note for note in forced if "attendees" not in note]
            offer(label, trial, kept + cut, True)
            break

    for size in range(1, len(base.required) + 1):
        found = False
        for dropped in combinations(base.required, size):
            trial = replace(base, required=tuple(s for s in base.required if s not in dropped))
            names = _join([cat.service(s).name.lower() for s in dropped])
            label = _cap(_join(steps + [f"leave out the {names}"]))
            note = f"Removes a stated requirement: {names}."
            found = offer(label, trial, forced + [note], True) or found
        if found:
            break
    return alternatives[:MAX_ALTERNATIVES]


def _clarify(brief: Brief, cat: Catalogue, issues: Sequence[FieldIssue], warnings: list[str]) -> Assessment:
    questions = [Question(field="headcount", text="How many people will attend?", blocking=True)]
    if not brief.services:
        names = ", ".join(s.name.lower() for s in cat.services)
        questions.append(Question(field="services", text=f"Which services would you like? We offer: {names}."))
    if brief.budget.amount is None:
        questions.append(Question(field="budget", text="Do you have a budget ceiling for the event?"))
    if brief.accessibility.need == "not_mentioned":
        questions.append(Question(field="accessibility", text="Does anyone attending need step-free access?"))
    if brief.duration.hours is None:
        text = f"Our sessions run for {hours(cat.standard_duration)} hours: does that suit your event?"
        questions.append(Question(field="duration", text=text))
    if not brief.date.mentioned:
        questions.append(Question(field="date", text="Which date or dates do you have in mind? We will then check availability."))
    questions += [Question(field=i.field, text=f"Could you confirm our reading? {i.message}") for i in issues if i.kind == "interpreted"]
    assumptions = [Assumption(field=i.field, text=i.message, sensitive=True) for i in issues if i.kind == "interpreted"]
    return Assessment(
        route="clarify",
        headline="Cannot quote yet: the number of attendees is missing",
        assumptions=assumptions,
        notices=_notices(brief, cat, None),
        questions=questions,
        warnings=warnings,
    )


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _uncap(text: str) -> str:
    return text[:1].lower() + text[1:]
