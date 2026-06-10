"""CLI: python -m hybrid_rag.cli {ingest|ask|demo}"""
from __future__ import annotations

import argparse
import json
import textwrap

from .models import Answer

DEMO_QUESTIONS = [
    # hybrid, DB -> docs
    "Acme Corp has reported compressor failures on their AeroFlow X200 units. "
    "How many X200 units have they purchased since 2025, and are these "
    "failures covered under warranty? Include any relevant service bulletins.",
    # pure SQL
    "What was Acme Corp's total order value in 2025?",
    # pure PDF
    "What does the warranty policy say about units operated below 5 degrees C?",
    # hybrid, docs -> DB (graph hop resolves the bulletin to a product)
    "Which customers purchased the product affected by service bulletin "
    "FSB-2025-03, and what corrective action does it recommend?",
]


def _wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(p, width=92, initial_indent=indent, subsequent_indent=indent)
        for p in text.split("\n") if p.strip())


def render(answer: Answer, show_trace: bool = False) -> str:
    r = answer.routing
    lines = [
        "=" * 96,
        _wrap(f"Q: {answer.question}", ""),
        "-" * 96,
        f"Route   : {r.route.upper()}  (confidence {r.confidence:.2f}, "
        f"engine {r.engine})",
        f"Entities: {', '.join(r.matched_entities) or '-'}",
    ]
    if r.signals.get("bridged_entities"):
        lines.append(f"Bridged : {', '.join(r.signals['bridged_entities'])} "
                     f"(present in DB and documents)")
    if answer.trace.get("entity_expansion"):
        for ent, why in answer.trace["entity_expansion"].items():
            lines.append(f"KG hop  : +{ent}  ({why})")
    for ex in answer.trace.get("kg_expansion", []):
        lines.append(f"KG chunk: +{ex['chunk']}  (via {ex['via']})")
    mark = "verified" if answer.verified else "NOT verified"
    lines += [
        "-" * 96,
        f"Answer  ({answer.engine}, {mark}):",
        _wrap(answer.text),
        "",
        "Sources:",
    ]
    for entry in answer.bibliography():
        lines.append(_wrap(entry, "  "))
    if answer.warnings:
        lines.append("Audit notes:")
        for w in answer.warnings:
            lines.append(f"  - {w}")
    if show_trace:
        lines += ["", "Trace:", json.dumps(answer.trace, indent=2, default=str)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="hybrid_rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="(re)build DB, PDFs, indexes and graph")
    ask_p = sub.add_parser("ask", help="answer one question")
    ask_p.add_argument("question")
    ask_p.add_argument("--trace", action="store_true")
    ask_p.add_argument("--json", action="store_true")
    sub.add_parser("demo", help="run the four scripted demo questions")
    args = parser.parse_args()

    if args.cmd == "ingest":
        from .ingest import run_ingest
        run_ingest()
        return

    from .pipeline import Pipeline
    pipe = Pipeline()

    if args.cmd == "ask":
        answer = pipe.ask(args.question)
        if args.json:
            print(answer.model_dump_json(indent=2))
        else:
            print(render(answer, show_trace=args.trace))
        return

    if args.cmd == "demo":
        for q in DEMO_QUESTIONS:
            print(render(pipe.ask(q)))
            print()


if __name__ == "__main__":
    main()
