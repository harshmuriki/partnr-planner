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
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.scenario_ground_truth import GroundTruthSessions
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

        def apply(reviews):
            for variant, record in changes.items():
                # Keep unchecked records, so old browser imports cannot re-check them.
                if importing and variant in reviews:
                    continue
                previous = reviews.get(variant, {})
                # A re-recording stays flagged until someone verifies the variant again.
                if not record["verified"] and previous.get("needs_review"):
                    record.update(needs_review=True, rerecorded_at=previous["rerecorded_at"])
                reviews[variant] = record
        return self._write(apply)

    def flag_rerecorded(self, variant):
        """Revoke verification after a saved ground truth is replaced, until re-verified."""
        if variant not in self.variants:
            return None
        stamp = datetime.now(timezone.utc).isoformat()
        return self._write(lambda reviews: reviews.__setitem__(variant, {
            "verified": False, "needs_review": True, "rerecorded_at": stamp, "updated_at": stamp}))

    def _write(self, apply):
        # Separate lock file survives atomic replacement and coordinates processes.
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            apply(data["reviews"])
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

    def send_head(self):
        """Serve byte ranges so the browser can seek without downloading the whole MP4."""
        self._video_bytes_remaining = None
        path = Path(self.translate_path(self.path))
        if path.suffix.lower() != '.mp4' or not path.is_file():
            return super().send_head()
        import re
        stream = path.open('rb')
        size = os.fstat(stream.fileno()).st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get('Range') if not self.headers.get('If-Range') else None
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
            try:
                if not match or not any(match.groups()):
                    raise ValueError('Invalid range')
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    if int(last) <= 0:
                        raise ValueError('Invalid suffix')
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    raise ValueError('Unsatisfiable range')
                status = 206
            except ValueError:
                stream.close()
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
        self.send_response(status)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        self.send_header('Last-Modified', self.date_time_string(os.fstat(stream.fileno()).st_mtime))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        stream.seek(start)
        self._video_bytes_remaining = max(0, end - start + 1)
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self._video_bytes_remaining is None:
                return super().copyfile(source, outputfile)
            remaining = self._video_bytes_remaining
            while remaining:
                block = source.read(min(1 << 16, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Normal when seeking or leaving a video mid-download.

    def ground_truth_route(self):
        import re
        return re.fullmatch(r'/baseline_evaluation_v3/api/ground-truth/(T[1-7]-(?:ACC|INC|OUT)-[A-Z]+)(?:/(sandbox|close|action|steps|record|rooms|furniture|notes|packs|frame))?', urlsplit(self.path).path)

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
                    session_id = parse_qs(urlsplit(self.path).query).get('session_id', [None])[0]
                    frame = self.ground_truth.frame(variant, session_id)
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
                if operation not in ('sandbox', 'close', 'record', 'steps', 'action', 'notes', 'furniture', 'rooms', 'packs'):
                    return self.send_error(405)
                result = self.ground_truth.dispatch(variant, operation, payload)
                return self.send_json(result, 202 if operation in ('sandbox', 'record', 'action') or result.get('running') else 200)
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
    parser.add_argument("--max-sandboxes", type=int, default=2, help="Maximum concurrent Habitat recording sessions (default: 2)")
    args = parser.parse_args()
    if args.max_sandboxes < 1:
        parser.error("--max-sandboxes must be at least 1")
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
        ground_truth = GroundTruthSessions(max_sessions=args.max_sandboxes, worker_factory=lambda: HabitatWorker(args.habitat_python),
                                           on_rerecord=store.flag_rerecorded)
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
            ground_truth.close()


if __name__ == "__main__":
    main()
