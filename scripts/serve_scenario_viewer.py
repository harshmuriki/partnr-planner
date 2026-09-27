#!/usr/bin/env python3
"""Serve the scenario viewer and its shared, durable human-verification records."""

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.scenario_ground_truth import GroundTruthService
API_PATH = "/baseline_evaluation_v3/api/human-reviews"
REVIEW_FILE = ROOT / "baseline_evaluation_v3/human_verification.json"


class ReviewStore:
    def __init__(self, path, variants):
        self.path = Path(path)
        self.variants = set(variants)

    def read(self):
        if not self.path.exists():
            return {"version": 1, "reviews": {}}
        data = json.loads(self.path.read_text())
        if data.get("version") != 1 or not isinstance(data.get("reviews"), dict):
            raise ValueError("Invalid human-verification file; refusing to overwrite it")
        for record in data["reviews"].values():
            if (not isinstance(record, dict) or type(record.get("verified")) is not bool
                    or not isinstance(record.get("updated_at"), str)):
                raise ValueError("Invalid human-verification record")
        return data

    def update(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        importing = set(payload) == {"import"}
        if importing:
            records = payload["import"]
            if not isinstance(records, dict):
                raise ValueError("Expected imported reviews")
            changes = {}
            for variant, timestamp in records.items():
                if not isinstance(timestamp, str):
                    raise ValueError("Invalid review timestamp")
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("Review timestamp must include a timezone")
                changes[variant] = {"verified": True, "updated_at": parsed.isoformat()}
        else:
            if set(payload) != {"variant_id", "verified"} or type(payload["verified"]) is not bool:
                raise ValueError("Expected variant_id and boolean verified")
            if not isinstance(payload["variant_id"], str):
                raise ValueError("Invalid variant ID")
            changes = {payload["variant_id"]: {
                "verified": payload["verified"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }}
        if not set(changes).issubset(self.variants):
            raise ValueError("Unknown scenario")
        # Separate lock file survives atomic replacement and coordinates processes.
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            for variant, record in changes.items():
                # Keep unchecked records, so old browser imports cannot re-check them.
                if not importing or variant not in data["reviews"]:
                    data["reviews"][variant] = record
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent,
                                                 prefix=".human-verification-", delete=False) as out:
                    name = out.name
                    json.dump(data, out, indent=2, sort_keys=True)
                    out.write("\n")
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(name, self.path)
            finally:
                if name and os.path.exists(name):
                    os.unlink(name)
            return data


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, store, ground_truth=None, **kwargs):
        self.store = store
        self.ground_truth = ground_truth
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def ground_truth_route(self):
        import re
        return re.fullmatch(r'/baseline_evaluation_v3/api/ground-truth/(T[1-7]-(?:ACC|INC|OUT)-[A-Z]+)(?:/(sandbox|action|steps|record|rooms|furniture|notes|frame))?', urlsplit(self.path).path)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.ground_truth_route()
        if route:
            if self.ground_truth is None:
                return self.send_json({'error': 'Ground truth service unavailable'}, 503)
            try:
                variant, operation = route.groups()
                if operation == 'frame':
                    frame = self.ground_truth.worker.frame() if self.ground_truth.variant == variant else None
                    if not frame:
                        return self.send_error(404, 'No live frame yet')
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    return
                if operation:
                    return self.send_error(405)
                return self.send_json(self.ground_truth.state(variant))
            except (ValueError, OSError, KeyError) as error:
                return self.send_json({'error': str(error)}, 400)
        if urlsplit(self.path).path != API_PATH:
            return super().do_GET()
        try:
            self.send_json(self.store.read())
        except (OSError, ValueError, TypeError, AttributeError):
            self.send_json({"error": "Could not read the shared verification file"}, 500)

    def do_POST(self):
        route = self.ground_truth_route()
        if urlsplit(self.path).path != API_PATH and not route:
            return self.send_error(404)
        # JSON-only, same-origin writes; no permissive CORS on this local service.
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self.send_json({"error": "Cross-origin writes are not allowed"}, 403)
        if self.headers.get_content_type() != "application/json":
            return self.send_json({"error": "Expected application/json"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                return self.send_json({"error": "Invalid request size"}, 413)
            payload = json.loads(self.rfile.read(length))
            if route:
                if self.ground_truth is None:
                    return self.send_json({'error': 'Ground truth service unavailable'}, 503)
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object')
                variant, operation = route.groups()
                if operation == 'sandbox':
                    return self.send_json(self.ground_truth.sandbox(variant), 202)
                if operation == 'record':
                    return self.send_json(self.ground_truth.record(variant), 202)
                if operation == 'steps':
                    return self.send_json(self.ground_truth.edit_steps(variant, payload))
                if operation == 'action':
                    return self.send_json(self.ground_truth.action(variant, payload), 202)
                if operation == 'notes':
                    return self.send_json(self.ground_truth.save_notes(variant, payload))
                if operation == 'furniture':
                    return self.send_json(self.ground_truth.furniture(variant, payload))
                if operation == 'rooms':
                    return self.send_json(self.ground_truth.rooms(variant, payload))
                return self.send_error(405)
            self.send_json(self.store.update(payload))
        except (ValueError, TypeError, AttributeError) as error:
            self.send_json({"error": str(error)}, 400)
        except OSError:
            self.send_json({"error": "Could not save the shared verification file"}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="localhost", help="localhost serves IPv4 and IPv6 loopback in one process")
    parser.add_argument("--habitat-python", help="Python executable from the Habitat environment")
    args = parser.parse_args()
    with (ROOT / "baseline_evaluation_v3/variant_object_assets.csv").open() as csv_file:
        variants = {row["Variant_ID"] for row in csv.DictReader(csv_file)}
    store = ReviewStore(REVIEW_FILE, variants)
    class IPv6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

        def server_bind(self):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            super().server_bind()

    addresses = ['127.0.0.1', '::1'] if args.bind == 'localhost' else [args.bind]
    with ExitStack() as stack:
        # Bind before initializing state: a duplicate launch must not interrupt a run.
        servers = [stack.enter_context((IPv6Server if ':' in address else ThreadingHTTPServer)(
            (address, args.port), ViewerHandler)) for address in addresses]
        from scripts.scenario_ground_truth import HabitatWorker
        ground_truth = GroundTruthService(worker=HabitatWorker(args.habitat_python))
        for server in servers:
            server.RequestHandlerClass = partial(ViewerHandler, store=store, ground_truth=ground_truth)
        for server in servers[1:]:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Scenario viewer on port {args.port}; shared checks: {REVIEW_FILE}", flush=True)
        try:
            servers[0].serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            for server in servers[1:]:
                server.shutdown()
            ground_truth.worker.close()


if __name__ == "__main__":
    main()
