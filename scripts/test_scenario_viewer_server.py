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

    def test_corrupt_file_is_not_replaced(self):
        self.path.write_text("broken")
        with self.assertRaises(ValueError):
            self.store.update({"variant_id": "T1-ACC-BASE", "verified": True})
        self.assertEqual(self.path.read_text(), "broken")


if __name__ == "__main__":
    unittest.main()
