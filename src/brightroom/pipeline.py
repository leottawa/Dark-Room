"""The fixed route from enquiry to reviewed draft.

    1 extract (model) -> 2-3 assess (code) -> 4 write (model + code) -> 5 verify (code) -> 6 person

The order never depends on what a step returns, which is why this is an automation and not an
agent: there is no decision about what to do next for a model to get wrong.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .catalogue import Catalogue
from .draft import Prose, prose_facts, render_draft, request_prose, template_prose
from .engine import assess
from .extract import extract_brief
from .llm import LLMClient, LLMResult
from .models import Assessment, Brief, FieldIssue
from .verify import Violation, check_draft, check_prose

ProseSource = Literal["model", "model_retry", "template_fallback", "template"]


class Step(BaseModel):
    name: str
    seconds: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


class RunResult(BaseModel):
    enquiry: str
    model: str | None
    brief: Brief
    issues: list[FieldIssue]
    assessment: Assessment
    prose: Prose
    prose_source: ProseSource
    prose_violations: list[list[Violation]] = []
    draft: str
    draft_violations: list[Violation] = []
    steps: list[Step] = []
    created_at: str = ""

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    @property
    def cost(self) -> float:
        return sum(s.cost for s in self.steps)

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    @property
    def tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)


def _step(name: str, started: float, calls: list[LLMResult] = ()) -> Step:
    return Step(
        name=name,
        seconds=time.perf_counter() - started,
        llm_calls=len(calls),
        prompt_tokens=sum(c.prompt_tokens for c in calls),
        completion_tokens=sum(c.completion_tokens for c in calls),
        cost=sum(c.cost for c in calls),
    )


def run_pipeline(enquiry: str, client: LLMClient, cat: Catalogue, prose_mode: Literal["model", "template"] = "model") -> RunResult:
    started = time.perf_counter()
    extraction = extract_brief(enquiry, client, cat)
    first = _step("extract", started, extraction.calls)
    return _from_brief(enquiry, extraction.brief, extraction.issues, client, cat, prose_mode, [first])


def run_from_brief(
    enquiry: str,
    brief: Brief,
    client: LLMClient | None,
    cat: Catalogue,
    prose_mode: Literal["model", "template"] = "model",
    issues: list[FieldIssue] | None = None,
) -> RunResult:
    """Steps 2 to 5 for a brief a person has entered or corrected.

    Fields the person changed need no audit; pass the issues of the fields they left alone.
    """
    return _from_brief(enquiry, brief, issues or [], client, cat, prose_mode, [])


def _from_brief(enquiry, brief, issues, client, cat, prose_mode, steps: list[Step]) -> RunResult:
    started = time.perf_counter()
    assessment = assess(brief, cat, issues)
    steps.append(_step("assess", started))

    started = time.perf_counter()
    facts = prose_facts(assessment, brief, cat)
    prose, source, history, calls = _write_prose(assessment, brief, facts, client if prose_mode == "model" else None, cat)
    draft = render_draft(assessment, brief, prose, cat)
    steps.append(_step("write", started, calls))

    started = time.perf_counter()
    draft_violations = check_draft(draft, assessment)
    steps.append(_step("verify", started))

    return RunResult(
        enquiry=enquiry,
        model=getattr(client, "model", None),
        brief=brief,
        issues=list(issues),
        assessment=assessment,
        prose=prose,
        prose_source=source,
        prose_violations=history,
        draft=draft,
        draft_violations=draft_violations,
        steps=steps,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _write_prose(assessment, brief, facts, client, cat) -> tuple[Prose, ProseSource, list[list[Violation]], list[LLMResult]]:
    """Model passages if they pass verification within two tries, otherwise the fixed template."""
    fallback = template_prose(assessment, brief)
    if client is None:
        return fallback, "template", [], []
    history: list[list[Violation]] = []
    calls: list[LLMResult] = []
    feedback = None
    for attempt in range(2):
        try:
            prose, result = request_prose(facts, client, feedback)
        except Exception:  # a drafting failure must never block the reviewer: the template is safe
            break
        calls.append(result)
        violations = [Violation(code="invalid_json", detail="The passages were not valid JSON.")] if prose is None else check_prose(prose, facts, cat)
        history.append(violations)
        if prose is not None and not violations:
            return prose, ("model" if attempt == 0 else "model_retry"), history, calls
        feedback = " ".join(v.detail for v in violations)
    return fallback, "template_fallback", history, calls


def save_run(result: RunResult, directory: str | Path = "runs", event: str = "draft", extra: dict | None = None) -> Path:
    """Append-only audit trail: what came in, what was decided, what a person did with it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = directory / f"{stamp}_{event}.json"
    record = {"event": event, "seconds": result.seconds, "cost": result.cost, **(extra or {}), "run": result.model_dump()}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
