"""Data shapes passed between pipeline steps.

The Brief is what the LLM extracts; everything from Evaluation down is produced by code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Need = Literal["required", "preferred", "not_mentioned"]
Wish = Literal["required", "preferred"]
RequestCategory = Literal[
    "discount",
    "booking_or_date_confirmation",
    "vegetarian_vegan",
    "allergen_guarantee",
    "unlisted_service",
    "other",
]


# --- Step 1 output: the structured brief ------------------------------------------------------


class Headcount(BaseModel):
    min: int | None = None
    max: int | None = None
    approximate: bool = False
    quote: str | None = None


class Duration(BaseModel):
    hours: float | None = None
    quote: str | None = None


class Budget(BaseModel):
    amount: float | None = None
    basis: Literal["total", "per_person"] | None = None
    approximate: bool = False
    quote: str | None = None


class Accessibility(BaseModel):
    # "not_needed" is the client saying so; "not_mentioned" leaves the question open.
    need: Literal["required", "preferred", "not_needed", "not_mentioned"] = "not_mentioned"
    quote: str | None = None


class VenueWish(BaseModel):
    venue_id: str | None = None
    need: Need = "not_mentioned"
    quote: str | None = None


class ServiceWish(BaseModel):
    service_id: str
    need: Wish = "required"
    beyond_terms: str | None = None
    quote: str | None = None


class DateMention(BaseModel):
    mentioned: bool = False
    quote: str | None = None


class OtherRequest(BaseModel):
    category: RequestCategory
    description: str
    need: Wish = "preferred"
    quote: str | None = None


class Brief(BaseModel):
    language: str = "en"
    headcount: Headcount = Headcount()
    duration: Duration = Duration()
    budget: Budget = Budget()
    accessibility: Accessibility = Accessibility()
    venue: VenueWish = VenueWish()
    services: list[ServiceWish] = []
    date: DateMention = DateMention()
    other_requests: list[OtherRequest] = []


class FieldIssue(BaseModel):
    """Something the salesperson should look at in the extracted brief."""

    field: str
    kind: Literal["quote_not_found", "interpreted", "corrected"]
    message: str


# --- Steps 2-3 output: the assessment ---------------------------------------------------------


class Line(BaseModel):
    item_id: str
    name: str
    kind: Literal["venue", "service"]
    quantity: int
    unit_price: float
    amount: float
    conditions: str


class Package(BaseModel):
    venue_id: str
    venue_name: str
    headcount: int
    duration_hours: float
    lines: list[Line]
    total: float

    @property
    def service_ids(self) -> list[str]:
        return [line.item_id for line in self.lines if line.kind == "service"]


class Check(BaseModel):
    name: Literal["capacity", "accessibility", "duration", "budget"]
    status: Literal["pass", "fail", "unknown", "n/a"]
    detail: str


class Evaluation(BaseModel):
    package: Package | None
    checks: list[Check]
    feasible: bool


class Unmet(BaseModel):
    kind: str
    short: str
    detail: str
    # How an alternative that gives this up is worded, e.g. "without a photographer".
    option: str = ""


class Assumption(BaseModel):
    field: str
    text: str
    # A sensitive assumption could change the venue or the verdict if it is wrong.
    sensitive: bool = False


class Scenario(BaseModel):
    label: str
    feasible: bool
    package: Package | None = None
    note: str = ""


class Addon(BaseModel):
    service_id: str
    name: str
    amount: float
    conditions: str
    reason: str
    requested: bool


class Notice(BaseModel):
    category: str
    statement: str
    subject: str = ""
    offerable: bool = False
    quote: str | None = None

    @property
    def text(self) -> str:
        return f"{self.subject}: {self.statement}" if self.subject else self.statement


class Alternative(BaseModel):
    label: str
    package: Package
    trade_offs: list[str]
    changes_requirement: bool


class Question(BaseModel):
    field: str
    text: str
    blocking: bool = False


class Assessment(BaseModel):
    route: Literal["clarify", "proposal", "infeasible"]
    provisional: bool = False
    headline: str = ""
    package: Package | None = None
    checks: list[Check] = []
    unmet: list[Unmet] = []
    assumptions: list[Assumption] = []
    scenarios: list[Scenario] = []
    addons: list[Addon] = []
    notices: list[Notice] = []
    alternatives: list[Alternative] = []
    questions: list[Question] = []
    warnings: list[str] = []
    budget_total: float | None = None
    headroom: int | None = None
    per_person_increment: float = 0.0

    @property
    def status(self) -> Literal["clarify", "fits", "provisional", "infeasible"]:
        if self.route == "proposal":
            return "provisional" if self.provisional else "fits"
        return self.route
