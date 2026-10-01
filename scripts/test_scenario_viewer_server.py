"""Regression checks for durable shared human review state."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from scripts.serve_scenario_viewer import ReviewStore


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "reviews.json"
        self.variants = {"T1-ACC-BASE", "T2-ACC-BASE"}
        self.store = ReviewStore(self.path, self.variants)

    def test_survives_restart_and_uncheck(self):
        self.store.update({"variant_id": "T1-ACC-BASE", "verified": True})
        restarted = ReviewStore(self.path, self.variants)
        self.assertTrue(restarted.read()["reviews"]["T1-ACC-BASE"]["verified"])
        restarted.update({"variant_id": "T1-ACC-BASE", "verified": False})
        self.assertFalse(self.store.read()["reviews"]["T1-ACC-BASE"]["verified"])

    def test_concurrent_updates_preserve_both_scenarios(self):
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda variant: self.store.update({
                "variant_id": variant, "verified": True,
            }), self.variants))
        self.assertEqual(set(self.store.read()["reviews"]), self.variants)

    def test_legacy_import_never_overwrites_shared_decision(self):
        self.store.update({"variant_id": "T1-ACC-BASE", "verified": False})
        imported = {variant: "2026-09-16T12:00:00Z" for variant in self.variants}
        self.store.update({"import": imported})
        self.store.update({"import": imported})
        records = self.store.read()["reviews"]
        self.assertFalse(records["T1-ACC-BASE"]["verified"])
        self.assertTrue(records["T2-ACC-BASE"]["verified"])

    def test_invalid_updates_leave_file_unchanged(self):
        self.store.update({"variant_id": "T1-ACC-BASE", "verified": True})
        before = self.path.read_bytes()
        for payload in [{"variant_id": "../bad", "verified": True},
                        {"variant_id": "T1-ACC-BASE", "verified": "false"},
                        {"import": {"T1-ACC-BASE": "not a timestamp"}}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.store.update(payload)
            self.assertEqual(self.path.read_bytes(), before)

    def test_rerecording_revokes_verification_until_rechecked(self):
        self.store.update({"variant_id": "T1-ACC-BASE", "verified": True})
        self.store.flag_rerecorded("T1-ACC-BASE")
        record = self.store.read()["reviews"]["T1-ACC-BASE"]
        self.assertEqual((record["verified"], record["needs_review"]), (False, True))
        self.store.update({"variant_id": "T1-ACC-BASE", "verified": False})
        self.assertTrue(self.store.read()["reviews"]["T1-ACC-BASE"]["needs_review"])
        self.store.update({"import": {"T1-ACC-BASE": "2026-09-16T12:00:00Z"}})
        self.assertFalse(self.store.read()["reviews"]["T1-ACC-BASE"]["verified"])
        self.store.update({"variant_id": "T1-ACC-BASE", "verified": True})
        self.assertNotIn("needs_review", self.store.read()["reviews"]["T1-ACC-BASE"])
        self.assertIsNone(self.store.flag_rerecorded("../bad"))

    def test_corrupt_file_is_not_replaced(self):
        self.path.write_text("broken")
        with self.assertRaises(ValueError):
            self.store.update({"variant_id": "T1-ACC-BASE", "verified": True})
        self.assertEqual(self.path.read_text(), "broken")


class VideoRangeTests(unittest.TestCase):
    def test_video_ranges_and_full_download(self):
        from functools import partial
        from http.server import ThreadingHTTPServer
        import threading
        from unittest.mock import patch
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        from scripts.serve_scenario_viewer import ViewerHandler

        with tempfile.TemporaryDirectory() as directory, patch('scripts.serve_scenario_viewer.ROOT', Path(directory)):
            content = bytes(range(256)) * 8
            (Path(directory) / 'video.mp4').write_bytes(content)
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(ViewerHandler, store=None))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}/video.mp4?v=checksum'
            try:
                for header, start, end in [('bytes=10-25', 10, 25), ('bytes=2040-', 2040, 2047),
                                           ('bytes=-12', 2036, 2047), ('bytes=2040-9999', 2040, 2047)]:
                    with self.subTest(header=header), urlopen(Request(url, headers={'Range': header})) as response:
                        self.assertEqual(response.status, 206)
                        self.assertEqual(response.headers['Content-Range'], f'bytes {start}-{end}/{len(content)}')
                        self.assertEqual(response.read(), content[start:end + 1])
                with urlopen(url) as response:
                    self.assertEqual(response.headers['Accept-Ranges'], 'bytes')
                    self.assertEqual(response.read(), content)
                with urlopen(Request(url, method='HEAD')) as response:
                    self.assertEqual(int(response.headers['Content-Length']), len(content))
                    self.assertEqual(response.read(), b'')
                for header in ['bytes=3000-', 'bytes=-0', 'bytes=20-10', 'bytes=bad']:
                    with self.subTest(header=header), self.assertRaises(HTTPError) as error:
                        urlopen(Request(url, headers={'Range': header}))
                    self.assertEqual(error.exception.code, 416)
                    self.assertEqual(error.exception.headers['Content-Range'], f'bytes */{len(content)}')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
