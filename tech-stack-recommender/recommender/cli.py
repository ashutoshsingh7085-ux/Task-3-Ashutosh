"""Command-line interface for the Tech Stack Recommender."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from .engine import Recommender, RecommendationResult

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "raw_skills.csv"


def format_result(result: RecommendationResult) -> str:
    lines: list[str] = []
    if result.cold_start:
        lines.append("None of your skills matched our dataset (cold start).")
        lines.append("Showing trending career paths instead:\n")
    else:
        lines.append(f"Top {len(result.recommendations)} career paths for: "
                     f"{', '.join(result.known_skills)}\n")
    for rank, rec in enumerate(result.recommendations, start=1):
        header = f"{rank}. {rec.role}"
        if not result.cold_start:
            header += f"  -  {rec.percent}% match"
        lines.append(header)
        if rec.matched:
            lines.append(f"     You have : {', '.join(rec.matched)}")
        if rec.missing:
            lines.append(f"     Learn next: {', '.join(rec.missing)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _warn_unknown(engine: Recommender, unknown: list[str]) -> None:
    for skill in unknown:
        hint = difflib.get_close_matches(skill, engine.vocabulary, n=1)
        suffix = f" Did you mean '{hint[0]}'?" if hint else ""
        print(f"  ! '{skill}' is not in our vocabulary and was ignored.{suffix}")


def _interactive(engine: Recommender, min_inputs: int) -> list[str]:
    print("Tech Stack Recommender")
    print(f"Enter at least {min_inputs} skills, separated by commas.")
    print("Tip: run with --list-skills to see the full vocabulary.\n")
    while True:
        raw = input("Your skills > ")
        skills = [s for s in raw.split(",") if s.strip()]
        known, unknown = engine.parse_skills(skills)
        _warn_unknown(engine, unknown)
        if len(known) + len(unknown) >= min_inputs:
            return skills
        print(f"  Please enter at least {min_inputs} skills.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recommender", description="Recommend tech roles from your skills."
    )
    parser.add_argument("--skills", help='Comma-separated, e.g. "Python,Cloud,Automation"')
    parser.add_argument("--top", type=int, default=3, help="Number of results (default 3)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to skills CSV")
    parser.add_argument("--list-skills", action="store_true", help="Print the vocabulary and exit")
    args = parser.parse_args(argv)

    engine = Recommender.from_csv(args.data)

    if args.list_skills:
        print("\n".join(engine.vocabulary))
        return 0

    min_inputs = 3
    if args.skills:
        skills = [s for s in args.skills.split(",") if s.strip()]
    else:
        skills = _interactive(engine, min_inputs)

    try:
        result = engine.recommend(skills, top_n=args.top, min_inputs=min_inputs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.skills:
        _warn_unknown(engine, result.unknown_skills)
    print()
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
