"""The approved catalogue, loaded from data so prices and rules change without touching code."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "catalogue.json"


class Venue(BaseModel):
    id: str
    name: str
    price: float
    duration_hours: float
    max_people: int
    wheelchair_accessible: bool
    access_note: str
    included: list[str] = []
    keywords: list[str] = []

    @property
    def conditions(self) -> str:
        parts = [f"{hours(self.duration_hours)}-hour session", f"up to {self.max_people} people", self.access_note]
        parts += [f"{item} included" for item in self.included]
        return ", ".join(parts)


class Service(BaseModel):
    id: str
    name: str
    pricing: Literal["per_person", "flat"]
    price: float
    conditions: str
    keywords: list[str] = []


class Policy(BaseModel):
    offerable: bool
    statement: str
    requires_service: str | None = None
    # True when a client who *requires* this cannot be given a compliant proposal
    # (a missing photographer changes the event; a refused discount does not).
    blocks_when_required: bool = False


class Catalogue(BaseModel):
    business: str
    currency_symbol: str
    price_note: str
    terms: str
    venues: list[Venue]
    services: list[Service]
    policies: dict[str, Policy]

    @model_validator(mode="after")
    def _check(self) -> "Catalogue":
        ids = [v.id for v in self.venues] + [s.id for s in self.services]
        if len(ids) != len(set(ids)):
            raise ValueError("catalogue ids must be unique")
        if not self.venues:
            raise ValueError("catalogue needs at least one venue")
        if any(item.price < 0 for item in [*self.venues, *self.services]):
            raise ValueError("catalogue prices must not be negative")
        return self

    def venue(self, venue_id: str) -> Venue:
        return next(v for v in self.venues if v.id == venue_id)

    def service(self, service_id: str) -> Service:
        return next(s for s in self.services if s.id == service_id)

    @property
    def venue_ids(self) -> list[str]:
        return [v.id for v in self.venues]

    @property
    def service_ids(self) -> list[str]:
        return [s.id for s in self.services]

    @property
    def standard_duration(self) -> float:
        """The session length assumed when the client states none."""
        return min(v.duration_hours for v in self.venues)


def load_catalogue(path: str | Path | None = None) -> Catalogue:
    path = Path(path or os.environ.get("BRIGHTROOM_CATALOGUE") or DEFAULT_PATH)
    return Catalogue.model_validate(json.loads(path.read_text(encoding="utf-8")))


def money(amount: float, symbol: str = "€") -> str:
    if abs(amount - round(amount)) < 0.005:
        return f"{symbol}{round(amount):,}"
    return f"{symbol}{amount:,.2f}"


def hours(value: float) -> str:
    return f"{value:g}"
