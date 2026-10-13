"""Engine tests. Expected figures are worked out by hand from the case catalogue, not by the engine."""

from itertools import combinations

import pytest

from brightroom.catalogue import load_catalogue
from brightroom.engine import Spec, assess, evaluate
from brightroom.models import (
    Accessibility,
    Brief,
    Budget,
    DateMention,
    Duration,
    FieldIssue,
    Headcount,
    OtherRequest,
    ServiceWish,
    VenueWish,
)

CAT = load_catalogue()


def brief(n=None, hours=4, required=(), preferred=(), budget=None, basis="total", access="not_mentioned", **extra) -> Brief:
    services = [ServiceWish(service_id=s, need="required") for s in required]
    services += [ServiceWish(service_id=s, need="preferred") for s in preferred]
    return Brief(
        headcount=Headcount(min=n, max=n) if n else Headcount(),
        duration=Duration(hours=hours),
        budget=Budget(amount=budget, basis=basis if budget is not None else None),
        accessibility=Accessibility(need=access),
        services=services,
        **extra,
    )


def totals(assessment) -> list[float]:
    return [alt.package.total for alt in assessment.alternatives]


# --- The five starter enquiries ---------------------------------------------------------------


def test_q1_studio_fits_but_is_provisional_on_access():
    a = assess(brief(10, required=["S1"], budget=350), CAT)
    assert a.route == "proposal"
    assert a.package.venue_id == "V1"
    assert a.package.total == 320  # 240 + 10 x 8
    assert a.status == "provisional"
    assert any(x.field == "accessibility" and x.sensitive for x in a.assumptions)
    scenario = next(s for s in a.scenarios if "step-free" in s.label)
    assert scenario.package.venue_id == "V2"
    assert scenario.package.total == 500  # 420 + 80: over the 350 budget
    assert not scenario.feasible


def test_q2_over_budget_keeps_both_must_haves():
    a = assess(brief(20, required=["S1", "S2"], budget=850), CAT)
    assert a.route == "infeasible"
    assert a.package.venue_id == "V2"
    assert a.package.total == 940  # 420 + 160 + 360
    assert a.package.service_ids == ["S1", "S2"]
    assert [u.kind for u in a.unmet] == ["budget"]
    assert "€90 over budget" in a.headline
    # raise budget; 16 people (420 + 16 x 26); no coffee (420 + 360); no lunch (420 + 160)
    assert totals(a) == [940, 836, 780, 580]
    assert [alt.changes_requirement for alt in a.alternatives] == [False, True, True, True]
    assert a.alternatives[1].package.headcount == 16


def test_q3_wheelchair_forces_the_loft():
    a = assess(brief(10, required=["S1", "S3"], budget=600, access="required"), CAT)
    assert a.route == "proposal"
    assert a.package.venue_id == "V2"
    assert a.package.total == 535  # 420 + 80 + 35
    assert a.status == "fits"
    assert not a.scenarios


def test_q4_no_headcount_means_questions_only():
    a = assess(Brief(date=DateMention(mentioned=True, quote="next month")), CAT)
    assert a.route == "clarify"
    assert a.package is None
    fields = [q.field for q in a.questions]
    assert fields[0] == "headcount" and a.questions[0].blocking
    assert {"services", "budget", "accessibility"} <= set(fields)
    assert "date" not in fields
    assert any(n.category == "date" for n in a.notices)


def test_q5_capacity_and_duration_fail_despite_ample_budget():
    confirm = OtherRequest(category="booking_or_date_confirmation", description="confirm today", need="required")
    a = assess(brief(30, hours=6, required=["S2"], budget=1500, other_requests=[confirm]), CAT)
    assert a.route == "infeasible"
    assert a.package is None
    assert {u.kind for u in a.unmet} == {"capacity", "duration"}
    assert any(n.category == "booking_or_date_confirmation" for n in a.notices)
    first = a.alternatives[0]
    assert first.package.headcount == 24 and first.package.duration_hours == 4
    assert first.package.total == 852  # 420 + 24 x 18
    assert len(first.trade_offs) == 2


# --- Boundaries --------------------------------------------------------------------------------


@pytest.mark.parametrize("n, venue, total", [(12, "V1", 336), (13, "V2", 524), (24, "V2", 612)])
def test_capacity_boundaries(n, venue, total):
    a = assess(brief(n, required=["S1"]), CAT)
    assert (a.package.venue_id, a.package.total) == (venue, total)


def test_25_people_exceed_every_venue():
    a = assess(brief(25, required=["S1"]), CAT)
    assert a.route == "infeasible" and [u.kind for u in a.unmet] == ["capacity"]


def test_total_exactly_on_budget_fits_and_one_euro_less_does_not():
    assert assess(brief(10, required=["S1"], budget=320), CAT).route == "proposal"
    over = assess(brief(10, required=["S1"], budget=319), CAT)
    assert over.route == "infeasible" and "€1 over budget" in over.headline


def test_flat_items_are_charged_once():
    a = assess(brief(8, required=["S3", "S4"]), CAT)
    assert a.package.total == 555  # 240 + 35 + 280
    assert all(line.quantity == 1 for line in a.package.lines)


def test_shorter_event_is_priced_as_a_full_session():
    a = assess(brief(10, hours=2, required=["S1"]), CAT)
    assert a.route == "proposal" and a.package.total == 320
    assert "4-hour sessions" in next(c.detail for c in a.checks if c.name == "duration")


def test_longer_event_offers_the_standard_session():
    a = assess(brief(10, hours=6, required=["S1"]), CAT)
    assert a.route == "infeasible" and [u.kind for u in a.unmet] == ["duration"]
    assert a.alternatives[0].package.duration_hours == 4 and a.alternatives[0].package.total == 320


# --- Budgets -----------------------------------------------------------------------------------


def test_per_person_budget_is_converted_and_scales_with_headcount():
    a = assess(brief(10, required=["S2"], budget=40, basis="per_person"), CAT)
    assert a.budget_total == 400
    assert a.route == "infeasible" and a.package.total == 420  # 240 + 180
    labels = [alt.label for alt in a.alternatives]
    assert labels[0] == "Increase the budget to €420 (€42 per person)"
    # Fewer people lowers the ceiling as fast as the cost, so it is never offered here.
    assert not any("attendance" in label for label in labels)
    assert 240 in totals(a)


def test_missing_budget_gives_a_proposal_with_an_unknown_budget_check():
    a = assess(brief(10, required=["S1"]), CAT)
    assert a.route == "proposal"
    assert next(c for c in a.checks if c.name == "budget").status == "unknown"
    assert "budget" in [q.field for q in a.questions]


# --- Preferences never displace requirements -----------------------------------------------------


def test_preferred_service_is_included_only_when_it_fits():
    fits = assess(brief(10, required=["S1"], preferred=["S3"], budget=360), CAT)
    assert fits.package.total == 355 and not fits.addons
    tight = assess(brief(10, required=["S1"], preferred=["S3"], budget=350), CAT)
    assert tight.package.total == 320
    assert [(x.service_id, x.amount) for x in tight.addons] == [("S3", 35)]


def test_preferred_service_without_budget_is_offered_as_an_option():
    a = assess(brief(10, required=["S1"], preferred=["S2"]), CAT)
    assert a.package.total == 320
    assert [(x.service_id, x.amount) for x in a.addons] == [("S2", 180)]


def test_preferred_access_is_honoured_when_affordable():
    a = assess(brief(10, required=["S1"], budget=600, access="preferred"), CAT)
    assert a.package.venue_id == "V2" and a.package.total == 500
    b = assess(brief(10, required=["S1"], budget=350, access="preferred"), CAT)
    assert b.package.venue_id == "V1" and b.status == "provisional"


# --- Uncertain headcount -------------------------------------------------------------------------


def test_range_is_priced_at_the_upper_bound_with_the_lower_bound_shown():
    b = brief(required=["S1"])
    b.headcount = Headcount(min=10, max=14)
    a = assess(b, CAT)
    assert a.package.headcount == 14 and a.package.total == 532  # 420 + 112
    low = next(s for s in a.scenarios if "10 attendees" in s.label)
    assert low.package.total == 320
    assert a.status == "provisional"


def test_approximate_headcount_reports_headroom():
    b = brief(15, required=["S2"], budget=700)
    b.headcount.approximate = True
    a = assess(b, CAT)
    assert a.package.total == 690  # 420 + 270
    assert a.headroom == 15  # 16 people would cost 708
    assert a.per_person_increment == 18
    assert "headcount" in [q.field for q in a.questions]


# --- Venue wishes --------------------------------------------------------------------------------


def test_required_venue_too_small_suggests_the_other_one():
    a = assess(brief(15, required=["S1"], venue=VenueWish(venue_id="V1", need="required")), CAT)
    assert a.route == "infeasible" and [u.kind for u in a.unmet] == ["capacity"]
    assert a.alternatives[0].label == "Use the Loft instead of the Studio"
    assert a.alternatives[0].package.total == 540  # 420 + 120
    assert a.alternatives[1].package.venue_id == "V1" and a.alternatives[1].package.headcount == 12


# --- Requests outside the catalogue ---------------------------------------------------------------


def test_required_unlisted_service_blocks_the_proposal():
    photo = OtherRequest(category="unlisted_service", description="a photographer", need="required")
    a = assess(brief(10, required=["S1"], budget=400, other_requests=[photo]), CAT)
    assert a.route == "infeasible"
    assert [u.kind for u in a.unmet] == ["unlisted_service"]
    assert a.alternatives[0].label == "Proceed without a photographer"
    assert a.alternatives[0].package.total == 320 and a.alternatives[0].changes_requirement


def test_preferred_unlisted_service_is_only_a_notice():
    parking = OtherRequest(category="unlisted_service", description="parking", need="preferred")
    a = assess(brief(10, required=["S1"], other_requests=[parking]), CAT)
    assert a.route == "proposal"
    assert [n.text for n in a.notices if n.category == "unlisted_service"] == ["Parking: This is not part of our catalogue and cannot be included."]


def test_discount_and_confirmation_never_block_a_proposal():
    asks = [
        OtherRequest(category="discount", description="10% off", need="required"),
        OtherRequest(category="booking_or_date_confirmation", description="confirm today", need="required"),
    ]
    a = assess(brief(10, required=["S1"], budget=350, other_requests=asks), CAT)
    assert a.route == "proposal" and a.package.total == 320
    assert {n.category for n in a.notices} == {"discount", "booking_or_date_confirmation"}


def test_allergen_guarantee_cannot_be_promised():
    nuts = OtherRequest(category="allergen_guarantee", description="guaranteed nut-free lunch", need="required")
    a = assess(brief(10, required=["S2"], other_requests=[nuts]), CAT)
    assert a.route == "infeasible" and [u.kind for u in a.unmet] == ["allergen_guarantee"]


def test_vegetarian_is_available_only_with_lunch():
    veg = OtherRequest(category="vegetarian_vegan", description="vegetarian option", need="preferred")
    with_lunch = assess(brief(10, required=["S2"], other_requests=[veg]), CAT)
    assert with_lunch.notices[0].offerable
    without = assess(brief(10, required=["S1"], other_requests=[veg]), CAT)
    assert not without.notices[0].offerable and "not part of this request" in without.notices[0].statement


def test_service_beyond_catalogue_terms_is_flagged():
    b = brief(10)
    b.services = [ServiceWish(service_id="S4", need="required", beyond_terms="facilitator for the full 4 hours")]
    a = assess(b, CAT)
    assert a.route == "infeasible" and [u.kind for u in a.unmet] == ["beyond_terms"]
    assert a.alternatives[0].label == "Proceed with the facilitator as listed"
    assert a.alternatives[0].package.total == 520


# --- Step-free access is never traded away --------------------------------------------------------


def test_alternatives_keep_step_free_access():
    a = assess(brief(10, required=["S1"], budget=400, access="required"), CAT)
    assert a.route == "infeasible" and a.package.total == 500
    assert totals(a) == [500]  # only a higher budget works; the cheaper Studio is never offered
    assert all(alt.package.venue_id == "V2" for alt in a.alternatives)


def test_too_many_people_with_step_free_access_is_capped_at_the_loft():
    a = assess(brief(30, required=["S1"], access="required"), CAT)
    assert a.alternatives[0].package.venue_id == "V2" and a.alternatives[0].package.headcount == 24


# --- Extraction issues reach the reviewer ----------------------------------------------------------


def test_interpreted_values_become_points_to_confirm():
    issue = FieldIssue(field="headcount", kind="interpreted", message='Read "a dozen" as 12 attendees.')
    a = assess(brief(12, required=["S1"], access="required"), CAT, [issue])
    assert a.status == "provisional"
    assert any(q.field == "headcount" for q in a.questions)


def test_unverified_quotes_become_warnings():
    issue = FieldIssue(field="budget", kind="quote_not_found", message="Budget quote not found in the enquiry.")
    assert assess(brief(10, budget=500), CAT, [issue]).warnings == [issue.message]


# --- Exhaustive check of the pricing rules against an independent calculation -----------------------

VENUES = {"V1": (240, 12, False), "V2": (420, 24, True)}
SERVICES = {"S1": ("pp", 8), "S2": ("pp", 18), "S3": ("flat", 35), "S4": ("flat", 280)}
SUBSETS = [c for size in range(5) for c in combinations(SERVICES, size)]


def expected(n, duration, services, accessible, budget):
    """Venue and total straight from the case rules, or None when nothing can host the event."""
    if duration > 4:
        return None
    options = [(price, vid) for vid, (price, cap, step_free) in VENUES.items() if n <= cap and (step_free or not accessible)]
    if not options:
        return None
    price, vid = min(options)
    total = price + sum(SERVICES[s][1] * (n if SERVICES[s][0] == "pp" else 1) for s in services)
    return vid, total, budget is None or total <= budget


@pytest.mark.parametrize("duration", [2, 4, 6])
@pytest.mark.parametrize("accessible", [False, True])
def test_every_combination_matches_the_case_rules(duration, accessible):
    for n in range(1, 41):
        for services in SUBSETS:
            for budget in (None, 300, 500, 850, 1500):
                spec = Spec(headcount=n, duration=duration, required=services, accessible=accessible, budget=budget)
                result = evaluate(spec, CAT)
                want = expected(n, duration, services, accessible, budget)
                if want is None:
                    assert result.package is None and not result.feasible
                    continue
                vid, total, within = want
                assert result.package.venue_id == vid
                assert result.package.total == total == sum(line.amount for line in result.package.lines)
                assert result.package.service_ids == list(services)
                assert result.feasible == within


def test_alternatives_are_always_deliverable_and_honest():
    for n in (5, 12, 13, 20, 24, 30):
        for services in SUBSETS:
            for budget in (200, 400, 700):
                for accessible in (False, True):
                    for duration in (4, 6):
                        a = assess(brief(n, hours=duration, required=services, budget=budget, access="required" if accessible else "not_mentioned"), CAT)
                        for alt in a.alternatives:
                            pkg = alt.package
                            price, cap, step_free = VENUES[pkg.venue_id]
                            assert pkg.headcount <= cap and pkg.duration_hours <= 4
                            assert step_free or not accessible
                            assert pkg.total <= budget or "increase the budget" in alt.label.lower()
                            dropped = set(services) - set(pkg.service_ids)
                            changed = dropped or pkg.headcount != n or pkg.duration_hours != duration
                            assert alt.changes_requirement == bool(changed)
                            assert alt.trade_offs


# --- Regressions -------------------------------------------------------------------------------------


def test_required_studio_with_step_free_access_still_gets_options():
    # The Studio has stairs, so it can never host this group; the Loft needs a bigger budget.
    studio = VenueWish(venue_id="V1", need="required")
    a = assess(brief(11, required=["S1"], budget=45, access="required", venue=studio), CAT)
    assert a.route == "infeasible"
    assert [alt.label for alt in a.alternatives] == ["Use the Loft instead of the Studio, and increase the budget to €508"]
    alt = a.alternatives[0]
    assert alt.package.total == 508  # 420 + 11 x 8
    assert alt.trade_offs[0] == "A different venue from the one requested." and alt.changes_requirement


def test_required_studio_for_a_large_step_free_group_offers_the_loft_at_capacity():
    studio = VenueWish(venue_id="V1", need="required")
    a = assess(brief(42, required=["S1"], access="required", venue=studio), CAT)
    assert a.alternatives[0].label == "Use the Loft instead of the Studio, and limit attendance to 24"
    assert a.alternatives[0].package.total == 612  # 420 + 24 x 8
    assert "18 fewer attendees than requested." in a.alternatives[0].trade_offs


def test_every_infeasible_brief_offers_at_least_one_deliverable_option():
    for venue in (None, VenueWish(venue_id="V1", need="required"), VenueWish(venue_id="V2", need="required")):
        for n in (5, 12, 13, 20, 30):
            for services in SUBSETS:
                for budget in (None, 45, 400, 900):
                    for access in ("required", "not_needed"):
                        for duration in (4, 6):
                            extra = {"venue": venue} if venue else {}
                            a = assess(brief(n, hours=duration, required=services, budget=budget, access=access, **extra), CAT)
                            if a.route != "infeasible":
                                continue
                            assert a.alternatives, a.headline
                            for alt in a.alternatives:
                                _, cap, step_free = VENUES[alt.package.venue_id]
                                assert alt.package.headcount <= cap and alt.package.duration_hours <= 4
                                assert step_free or access != "required"
