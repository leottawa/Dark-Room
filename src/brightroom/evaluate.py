"""Run every arm over a set of cases and report task success, latency and cost.

    uv run python -m brightroom.evaluate --arms pipeline naive engineered --repeats 3

Arms
    oracle             ideal brief -> engine: no model; the ceiling the pipeline can reach
    pipeline           the full pipeline (model reads, code decides, model writes, code verifies)
    pipeline_template  the pipeline with template sentences: one model call instead of two
    naive              one prompt: catalogue + enquiry + "write a proposal"
    engineered         one prompt with every rule spelled out

Every raw reply is saved, so each score can be checked by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .baselines import grade_reply, run_baseline
from .catalogue import Catalogue, load_catalogue
from .llm import DEFAULT_MODEL, LLMClient, OpenRouterClient
from .pipeline import run_from_brief, run_pipeline
from .scoring import Case, critical_error, extraction_errors, load_cases, outcome_from_run, score, usable

ARMS = ["oracle", "pipeline", "pipeline_template", "naive", "engineered"]
ROOT = Path(__file__).resolve().parents[2]


def run_case(case: Case, arm: str, client: LLMClient | None, grader: LLMClient | None, cat: Catalogue) -> dict:
    record: dict = {"case": case.id, "category": case.category, "source": case.source, "arm": arm, "seconds": 0.0, "cost": 0.0, "llm_calls": 0, "tokens": 0}
    started = time.perf_counter()
    try:
        if arm == "oracle":
            result = run_from_brief(case.enquiry, case.brief.to_brief(), None, cat, "template", case.brief.issues())
            outcome, text = outcome_from_run(result), result.draft
        elif arm in ("pipeline", "pipeline_template"):
            result = run_pipeline(case.enquiry, client, cat, "model" if arm == "pipeline" else "template")
            outcome, text = outcome_from_run(result), result.draft
            record |= {
                "seconds": result.seconds,
                "cost": result.cost,
                "llm_calls": result.llm_calls,
                "tokens": result.tokens,
                "prose_source": result.prose_source,
                "extraction_errors": extraction_errors(case.brief, result.brief),
                "brief": result.brief.model_dump(),
                "issues": [i.model_dump() for i in result.issues],
            }
        else:
            reply = run_baseline(arm, case.enquiry, client, cat)
            text = reply.text
            record |= {"seconds": reply.seconds, "cost": reply.cost, "llm_calls": 1, "tokens": reply.prompt_tokens + reply.completion_tokens}
            outcome, grading = grade_reply(case.enquiry, text, grader, cat)
            record["grader_cost"] = grading.cost
    except Exception as error:  # a failed run is a result, not a reason to stop the evaluation
        record |= {"error": f"{type(error).__name__}: {error}", "usable": False, "critical": False, "checks": []}
        record["seconds"] = record["seconds"] or time.perf_counter() - started
        return record
    checks = score(case.expect, outcome)
    record |= {"usable": usable(checks), "critical": critical_error(checks), "outcome": outcome.model_dump(), "checks": [c.model_dump() for c in checks], "text": text}
    return record


def summarise(records: list[dict]) -> str:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_arm[record["arm"]].append(record)
    lines = [f"{'arm':<18}{'runs':>5}{'usable':>9}{'critical':>10}{'errors':>8}{'median s':>10}{'mean $':>11}{'calls':>7}"]
    for arm, rows in by_arm.items():
        share = sum(r["usable"] for r in rows) / len(rows)
        lines.append(
            f"{arm:<18}{len(rows):>5}{share:>9.0%}{sum(r['critical'] for r in rows):>10}{sum('error' in r for r in rows):>8}"
            f"{statistics.median(r['seconds'] for r in rows):>10.2f}{statistics.mean(r['cost'] for r in rows):>11.6f}{statistics.mean(r['llm_calls'] for r in rows):>7.1f}"
        )
    lines.append("")
    lines.append("Usable share by category")
    categories = sorted({r["category"] for r in records})
    lines.append(f"{'arm':<18}" + "".join(f"{c[:14]:>16}" for c in categories))
    for arm, rows in by_arm.items():
        cells = []
        for category in categories:
            group = [r for r in rows if r["category"] == category]
            cells.append(f"{sum(r['usable'] for r in group)}/{len(group)}" if group else "-")
        lines.append(f"{arm:<18}" + "".join(f"{cell:>16}" for cell in cells))
    lines.append("")
    lines.append("Failed checks (count of runs)")
    for arm, rows in by_arm.items():
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            for check in row.get("checks", []):
                if not check["passed"]:
                    counts[check["name"] + ("!" if check["critical"] else "")] += 1
        detail = ", ".join(f"{name} {count}" for name, count in sorted(counts.items(), key=lambda item: -item[1])) or "none"
        lines.append(f"{arm:<18}{detail}")
    repeats = max(sum(1 for r in rows if r["case"] == rows[0]["case"]) for rows in by_arm.values())
    if repeats > 1:
        lines.append("")
        lines.append(f"Consistency over {repeats} runs per case (always / sometimes / never usable)")
        for arm, rows in by_arm.items():
            per_case: dict[str, list[bool]] = defaultdict(list)
            for row in rows:
                per_case[row["case"]].append(row["usable"])
            always = sum(all(v) for v in per_case.values())
            never = sum(not any(v) for v in per_case.values())
            lines.append(f"{arm:<18}{always} / {len(per_case) - always - never} / {never}")
    lines.append("")
    lines.append("! marks a critical failure: an error that would mislead the client if the draft were sent.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases" / "dev.json"))
    parser.add_argument("--arms", nargs="+", default=["oracle", "pipeline", "naive", "engineered"], choices=ARMS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=None, help=f"OpenRouter model id (default: BRIGHTROOM_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--grader-model", default=None, help="Model that audits baseline replies (default: the same model)")
    parser.add_argument("--only", nargs="*", help="Case ids to run")
    parser.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = parser.parse_args()

    cat = load_catalogue()
    cases = [c for c in load_cases(args.cases) if not args.only or c.id in args.only]
    needs_model = any(arm != "oracle" for arm in args.arms)
    client = OpenRouterClient(model=args.model) if needs_model else None
    grader = OpenRouterClient(model=args.grader_model or args.model) if needs_model else None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(args.out) / f"{stamp}_{Path(args.cases).stem}"
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "started": stamp,
        "cases": args.cases,
        "arms": args.arms,
        "repeats": args.repeats,
        "model": client.model if client else None,
        "grader_model": grader.model if grader else None,
        "temperature": 0,
        "seed": client.seed if client else None,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    records: list[dict] = []
    with (folder / "records.jsonl").open("w", encoding="utf-8") as log:
        for repeat in range(args.repeats):
            for case in cases:
                for arm in args.arms:
                    record = run_case(case, arm, client, grader, cat) | {"repeat": repeat}
                    records.append(record)
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                    mark = "ok  " if record["usable"] else ("CRIT" if record["critical"] else "fail")
                    failed = ", ".join(c["name"] for c in record["checks"] if not c["passed"]) or record.get("error", "")
                    print(f"{mark} {case.id:<4} {arm:<18} {record['seconds']:>6.2f}s  {failed}")
    summary = summarise(records)
    (folder / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary + f"\n\nSaved to {folder}")


if __name__ == "__main__":
    main()
