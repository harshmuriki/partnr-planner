"""Inspect immutable simulator scores and Astra's separate incident annotations."""

import csv
import io
from collections import Counter
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from study import Study, read_json


def create_app(root):
    gui = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(gui / "templates"),
        static_folder=str(gui / "static"),
    )
    study = Study(root)

    def selected():
        keys = ("condition", "task", "memory", "uncertainty", "repetition")
        return [
            r
            for r in study.runs.values()
            if all(
                not request.args.get(k) or str(r[k]) == request.args[k] for k in keys
            )
        ]

    def analysis(rid):
        return read_json(study.path("analyses", rid))

    @app.get("/")
    def index():
        return render_template("astra.html")

    @app.get("/api/rubric")
    def rubric():
        return Response((study.root / "rubric.md").read_text(), mimetype="text/plain")

    @app.get("/api/status")
    def stats():
        rows = selected()
        groups = {}
        tags = {"failures": Counter(), "inefficiencies": Counter()}
        annotated = 0
        for r in rows:
            group = groups.setdefault(
                r["condition"],
                {"n": 0, "annotated": 0, "outcomes": Counter(), "execution": Counter()},
            )
            group["n"] += 1
            group["outcomes"][r["simulator"]["outcome"]] += 1
            group["execution"][r["status"]] += 1
            a = analysis(r["id"])
            if a:
                annotated += 1
                group["annotated"] += 1
                for family in tags:
                    tags[family].update({i["label"] for i in a["annotation"][family]})
        return jsonify(
            total=len(rows),
            annotated=annotated,
            groups=groups,
            tags=tags,
            model=study.config["model"],
            reasoning_effort=study.config["reasoning_effort"],
            simulator_outcomes=Counter(r["simulator"]["outcome"] for r in rows),
        )

    @app.get("/api/runs")
    def runs():
        return jsonify(
            [
                {
                    k: r[k]
                    for k in (
                        "id",
                        "run_id",
                        "condition",
                        "task",
                        "memory",
                        "uncertainty",
                        "repetition",
                        "status",
                    )
                }
                | {
                    "outcome": r["simulator"]["outcome"],
                    "simulator_percent": r["simulator"]["metrics"].get(
                        "task_percent_complete"
                    ),
                    "annotated": study.path("analyses", r["id"]).exists(),
                    "error": (
                        read_json(study.path("errors", r["id"]))
                        if not study.path("analyses", r["id"]).exists()
                        else None
                    ),
                }
                for r in selected()
            ]
        )

    @app.get("/api/run")
    def run():
        rid = request.args["id"]
        if rid not in study.runs:
            abort(404)
        return jsonify(run=study.runs[rid], analysis=analysis(rid))

    @app.get("/api/evidence")
    def evidence():
        rid = request.args["id"]
        if rid not in study.runs:
            abort(404)
        wanted = set(request.args.get("events", "").split(","))
        return jsonify([e for e in study.evidence(rid)["events"] if e["id"] in wanted])

    @app.get("/api/packet")
    def packet():
        rid = request.args["id"]
        if rid not in study.runs:
            abort(404)
        path = study.path("astra_packets", rid)
        if not path.exists():
            abort(404)
        return send_file(
            path,
            as_attachment=True,
            download_name=study.runs[rid]["run_id"] + "_astra_input.json",
        )

    @app.get("/api/export")
    def export():
        rows = []
        for r in selected():
            a = analysis(r["id"])
            sim = r["simulator"]
            rows.append(
                {
                    "run_id": r["id"],
                    "condition": r["condition"],
                    "task": r["task"],
                    "memory": r["memory"],
                    "uncertainty": r["uncertainty"],
                    "repetition": r["repetition"],
                    "execution": r["status"],
                    "simulator_outcome": sim["outcome"],
                    "simulator_percent": sim["metrics"].get("task_percent_complete"),
                    "simulator_criteria": jsonify_text(sim["criteria"]),
                    "annotated": bool(a),
                    "astra_annotation": jsonify_text(a["annotation"]) if a else "",
                }
            )
        stream = io.StringIO()
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {
                    k: (
                        "'" + v
                        if isinstance(v, str) and v.startswith(("=", "+", "-", "@"))
                        else v
                    )
                    for k, v in row.items()
                }
                for row in rows
            )
        return Response(
            stream.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=astra_annotations.csv"
            },
        )

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app


def jsonify_text(value):
    import json

    return json.dumps(value, ensure_ascii=False)
