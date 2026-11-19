"""Separate study UI: legacy annotations and inference stay untouched."""

from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from judge import approve_pilot
from study import Study, export_csv


def create_app(root):
    gui = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(gui / "templates"),
        static_folder=str(gui / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

    def study():
        return Study(root)

    @app.before_request
    def same_origin_writes():
        if request.method in ("POST", "PUT", "DELETE"):
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
                abort(403)
            if not request.is_json:
                abort(415)

    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(KeyError)
    def not_found(error):
        return jsonify(error="Unknown run or missing required field"), 404

    @app.errorhandler(RuntimeError)
    def conflict(error):
        return jsonify(error=str(error)), 409

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/")
    def index():
        return render_template("study.html")

    @app.get("/api/meta")
    def meta():
        s = study()
        return jsonify(
            model=s.config["model"],
            reasoning_effort=s.config["reasoning_effort"],
            codebook=s.codebook(),
            pilot_ids=s.config["pilot_ids"],
            pilot_approved=s.config["pilot_approved"],
        )

    @app.get("/api/runs")
    def runs():
        return jsonify(study().list_runs())

    @app.get("/api/stats")
    def stats():
        return jsonify(study().aggregate(request.args))

    @app.get("/api/run")
    def run():
        s, rid = study(), request.args["id"]
        return jsonify(
            run=s.runs[rid],
            evidence=s.evidence(rid),
            blank=s.blank(rid),
            **s.annotation(rid),
        )

    @app.put("/api/review")
    def review():
        payload = request.get_json()
        return jsonify(
            study().save_review(
                payload["run_id"],
                payload["annotation"],
                payload["revision"],
                payload.get("expected_draft_sha256"),
            )
        )

    @app.post("/api/category")
    def category():
        payload = request.get_json()
        return jsonify(study().add_category(payload["category"], payload["reviewer"]))

    @app.post("/api/approve-pilot")
    def approve():
        approve_pilot(study(), request.get_json()["reviewer"])
        return jsonify(ok=True)

    @app.get("/api/export")
    def export():
        s = study()
        kind = request.args.get("kind", "runs")
        if kind == "json":
            ids = s.aggregate(request.args)["run_ids"]
            return jsonify(
                schema_version=2,
                codebook=s.codebook(),
                stats=s.aggregate(request.args),
                runs=[{"run": s.runs[rid], **s.annotation(rid)} for rid in ids],
            )
        if kind not in ("runs", "criteria", "incidents"):
            abort(400)
        return Response(
            export_csv(s, kind, request.args),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{kind}.csv"'},
        )

    @app.get("/api/asset")
    def asset():
        s, rid = study(), request.args["id"]
        evidence = s.evidence(rid)
        relative = request.args["path"]
        # Only inventoried artifacts from this exact run, never an adjacent repetition.
        allowed = set(evidence["images"] + evidence["reports"]) | set(
            evidence["files_sha256"]
        )
        base = Path(s.runs[rid]["path"]).resolve()
        target = (base / relative).resolve()
        if (
            relative not in allowed
            or not target.is_relative_to(base)
            or not target.is_file()
        ):
            abort(404)
        response = send_file(target, as_attachment=target.suffix == ".html")
        if target.suffix == ".html":
            response.headers["Content-Security-Policy"] = "sandbox"
        return response

    return app
