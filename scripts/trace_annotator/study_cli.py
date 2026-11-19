#!/usr/bin/env python3
"""Initialize, draft, review, and serve a matched annotation study."""
import argparse
import json
from pathlib import Path

from study import REPO, Study, audit_criteria_attachment, initialize


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--study", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--criteria-attachment", type=Path)
    draft = sub.add_parser("draft")
    scope = draft.add_mutually_exclusive_group()
    scope.add_argument("--pilot", action="store_true")
    scope.add_argument("--all", action="store_true")
    draft.add_argument("--workers", type=int, default=2, choices=range(1, 9))
    draft.add_argument("--limit", type=int)
    approve = sub.add_parser("approve-pilot")
    approve.add_argument("--reviewer", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5056)
    sub.add_parser("status")
    args = p.parse_args()
    if args.command == "init":
        result = initialize(args.study, criteria_attachment=args.criteria_attachment)
        if args.criteria_attachment:
            audit_criteria_attachment(Study(args.study), args.criteria_attachment)
        print(
            f"Imported {len(result['runs'])} runs; {len(result['pilot_ids'])} pilot runs"
        )
        return
    study = Study(args.study)
    if args.command == "draft":
        if (study.root / "attached_criteria.md").exists():
            audit_criteria_attachment(study, study.root / "attached_criteria.md")
        from dotenv import load_dotenv
        from judge import run_drafts
        from openai import OpenAI

        load_dotenv(REPO / ".env")
        client = OpenAI(timeout=240, max_retries=2)
        results = run_drafts(
            study, client, pilot=not args.all, workers=args.workers, limit=args.limit
        )
        if any(r["status"] not in ("drafted", "skipped") for r in results):
            raise SystemExit(1)
    elif args.command == "approve-pilot":
        from judge import approve_pilot

        approve_pilot(study, args.reviewer)
    elif args.command == "serve":
        from study_web import create_app

        create_app(args.study).run(
            host=args.host, port=args.port, threaded=True, debug=False
        )
    else:
        from collections import Counter

        print(
            json.dumps(
                {
                    "study": str(study.root),
                    "model": study.config["model"],
                    "reasoning_effort": study.config["reasoning_effort"],
                    "runs": len(study.runs),
                    "states": Counter(r["review_state"] for r in study.list_runs()),
                    "pilot_approved": study.config["pilot_approved"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
