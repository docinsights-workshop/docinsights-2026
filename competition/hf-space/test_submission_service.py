import datetime as dt
import hashlib
import json
import tempfile
import threading
import unittest
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
from scoring import SubmissionError
from submission_service import HubTestConfigLoader, SubmissionService, TrustedTestConfig
from test_policy import OAuthIdentity, TestReleasePolicy
from test_store import TestReceipt, TestStoreError


VALIDATION_LABELS = [
    {"instance_id": "val-1", "answer": "42", "evidence": ["b1"]},
]
VALIDATION_ROWS = [
    {"instance_id": "val-1", "answer": "42", "evidence": ["b1"]},
]
NOW = dt.datetime(2026, 9, 5, 12, 0, tzinfo=dt.timezone.utc)
PROFILE = {
    "sub": "server-oauth-subject",
    "preferred_username": "server-user",
    "email": "server@example.org",
    "email_verified": True,
}
TEST_LABELS = [
    {"instance_id": "test-1", "answer": "42", "evidence": ["b1"]},
]
TEST_TASKS = [
    {
        "instance_id": "test-1",
        "user_query": "What is the answer?",
        "document_pdf": "test/documents/test-1.pdf",
    },
]
TEST_ROWS = [
    {"instance_id": "test-1", "answer": "42", "evidence": ["b1"]},
]
TRUSTED_POLICY = TestReleasePolicy(
    release_id="trusted-release",
    task_manifest_sha256="a" * 64,
    gold_sha256="b" * 64,
    open_at=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
    close_at=dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc),
    enabled=True,
)
TEST_META = {
    "team": " Fixture Team ",
    "participant_names": " Alice Example,\n Bob Example ",
    "submission_name": " final-run ",
    "contact": "client@example.org",
    "release_id": "client-release",
    "task_manifest_sha256": "c" * 64,
    "gold_sha256": "d" * 64,
    "gold_path": "../../private/client-labels.jsonl",
    "scoring_gold_sha256": "e" * 64,
    "scoring_private_revision": "client-private-revision",
    "scoring_public_revision": "client-public-revision",
    "scoring_public_repo_id": "client/public-repo",
    "scoring_task_manifest_path": "../../client/tasks.jsonl",
    "metrics": {"answer_accuracy": 0.0},
    "identity": {"sub": "client-subject"},
    "attempt": 99,
}


class FileProbe:
    def __init__(self):
        self.was_read = False

    @property
    def name(self):
        self.was_read = True
        raise AssertionError("the file must not be read")


class RecordingStore:
    def __init__(self, attempts=(1,), exact_attempt=None):
        self.attempts = iter(attempts)
        self.exact_attempt = exact_attempt
        self.submissions = []
        self.history_identity = None
        self.lookup_requests = []

    def find_exact_attempt(self, identity, metadata, predictions):
        self.lookup_requests.append((identity, metadata, predictions))
        return self.exact_attempt

    def submit(self, identity, metadata, predictions, metrics):
        self.submissions.append(
            {
                "identity": identity,
                "metadata": metadata,
                "predictions": predictions,
                "metrics": metrics,
            }
        )
        attempt = next(self.attempts)
        return TestReceipt(True, attempt, f"receipt-{attempt}", f"2026-09-05T12:00:0{attempt}Z")

    def account_history(self, identity):
        self.history_identity = identity
        return [
            {
                "attempt_number": 1,
                "submission_id": "receipt-1",
                "submission_name": "first",
                "submitted_at": "2026-09-05T12:00:01Z",
                "metrics": {
                    "answer_accuracy": 1.0,
                    "evidence_exact_match": 1.0,
                    "evidence_f1": 1.0,
                    "examples": 1,
                    "per_example": [{"instance_id": "secret-test-id"}],
                },
                "predictions": [{"answer": "private prediction"}],
            },
            {
                "attempt_number": 2,
                "submission_id": "receipt-2",
                "submission_name": "second",
                "submitted_at": "2026-09-05T12:00:02Z",
                "metrics": {
                    "answer_accuracy": 0.5,
                    "evidence_f1": 0.5,
                    "per_example": [{"instance_id": "secret-test-id"}],
                },
            },
        ]


def test_file(rows=TEST_ROWS):
    upload = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for row in rows:
        upload.write(json.dumps(row) + "\n")
    upload.close()
    return SimpleNamespace(name=upload.name)


def configured_service(store=None, loader=None):
    store = store or RecordingStore()
    loader = loader or (
        lambda now: TrustedTestConfig(
            policy=TRUSTED_POLICY,
            labels=TEST_LABELS,
            scoring_gold_sha256=TRUSTED_POLICY.gold_sha256,
            private_revision="private-sha",
            public_revision="f" * 40,
            public_repo_id="public/repo",
            task_manifest_path="test/tasks.jsonl",
        )
    )
    return SubmissionService(
        validation_submitter=lambda file_obj, metadata: app.evaluate_submission(
            file_obj,
            metadata["team"],
            metadata["contact"],
            metadata["submission_name"],
            metadata.get("participant_names"),
        ),
        test_store=store,
        test_config_loader=loader,
        now_provider=lambda: NOW,
    )


class LegacyValidationCharacterizationTests(unittest.TestCase):
    def test_validation_leaderboard_fixture_preserves_exact_rendered_row(self):
        rows = [
            {
                "team": "Fixture Team",
                "contact": "lead@example.org",
                "submission_name": "baseline",
                "submitted_at": "2026-09-05T12:00:00Z",
                "answer_accuracy": 1.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 1,
                "attempts": 2,
            }
        ]

        with patch.object(app, "_load_leaderboard_rows", return_value=rows):
            rendered = app.leaderboard_html()

        self.assertIn(
            """<tr>
                <td class=\"leaderboard-rank\">1</td>
                <td>Fixture Team</td>
                <td>baseline</td>
                <td class=\"leaderboard-attempts\">2</td>
                <td class=\"leaderboard-metric\">100.00%</td>
                <td class=\"leaderboard-metric\">100.00%</td>
                <td class=\"leaderboard-date\">2026-09-05 12:00:00</td>
            </tr>""",
            rendered,
        )

    def test_validation_callback_preserves_metrics_identity_persistence_and_response(self):
        upload = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        for row in VALIDATION_ROWS:
            upload.write(json.dumps(row) + "\n")
        upload.close()
        persisted = {}

        def capture_upload(*, path_or_fileobj, path_in_repo, **kwargs):
            persisted["payload"] = json.loads(Path(path_or_fileobj).read_text(encoding="utf-8"))
            persisted["path_in_repo"] = path_in_repo
            Path(path_or_fileobj).unlink(missing_ok=True)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"datetime\.datetime\.utcnow\(\) is deprecated",
                category=DeprecationWarning,
            )
            with (
                patch.object(app, "WRITE_TOKEN", "fixture-token"),
                patch.object(app, "SUBMISSIONS_REPO_ID", "private/fixture"),
                patch.object(app, "_load_gold_rows", return_value=VALIDATION_LABELS),
                patch.object(app, "_update_leaderboard", return_value=1),
                patch.object(app, "upload_file", side_effect=capture_upload),
            ):
                result = app.evaluate_submission(
                    SimpleNamespace(name=upload.name),
                    "  Fixture Team  ",
                    "  lead@example.org  ",
                    "  baseline  ",
                    " Alice Example,\n Bob Example ",
                )

        self.assertEqual(set(result), {"value", "visible", "__type__"})
        self.assertTrue(result["visible"])
        self.assertEqual(result["__type__"], "update")
        self.assertEqual(
            set(result["value"]),
            {
                "answer_accuracy",
                "evidence_exact_match",
                "evidence_f1",
                "examples",
                "message",
            },
        )
        self.assertEqual(
            {key: result["value"][key] for key in (
                "answer_accuracy",
                "evidence_exact_match",
                "evidence_f1",
                "examples",
            )},
            {
                "answer_accuracy": 1.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 1,
            },
        )
        self.assertEqual(set(persisted["payload"]), {"leaderboard", "metrics", "predictions"})
        self.assertEqual(persisted["payload"]["predictions"], VALIDATION_ROWS)
        self.assertEqual(
            persisted["payload"]["metrics"],
            {
                "answer_accuracy": 1.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 1,
                "per_example": [
                    {
                        "instance_id": "val-1",
                        "answer_exact_match": 1.0,
                        "evidence_exact_match": 1.0,
                        "evidence_f1": 1.0,
                    }
                ],
            },
        )
        self.assertEqual(
            {
                key: persisted["payload"]["leaderboard"][key]
                for key in ("team", "contact", "participant_names", "submission_name")
            },
            {
                "team": "Fixture Team",
                "contact": "lead@example.org",
                "participant_names": "Alice Example, Bob Example",
                "submission_name": "baseline",
            },
        )
        self.assertTrue(persisted["path_in_repo"].startswith("submissions/"))


class SplitAwareServiceTests(unittest.TestCase):
    def test_validation_service_compatibility_fixture_preserves_exact_public_metrics(self):
        upload = test_file(VALIDATION_ROWS)
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        metadata = {
            "team": "Fixture Team",
            "contact": "lead@example.org",
            "submission_name": "baseline",
            "participant_names": "Alice Example",
        }

        with (
            patch.object(app, "_load_gold_rows", return_value=VALIDATION_LABELS),
            patch.object(app, "_persist_submission", return_value="legacy persistence"),
        ):
            result = configured_service().submit_for_split(
                "validation", upload, metadata, None
            )

        self.assertEqual(
            result,
            {
                "value": {
                    "answer_accuracy": 1.0,
                    "evidence_exact_match": 1.0,
                    "evidence_f1": 1.0,
                    "examples": 1,
                    "message": "legacy persistence",
                },
                "visible": True,
                "__type__": "update",
            },
        )

    def test_validation_does_not_require_oauth(self):
        upload = test_file(VALIDATION_ROWS)
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        metadata = {
            "team": "Fixture Team",
            "contact": "lead@example.org",
            "submission_name": "baseline",
            "participant_names": "Alice Example",
        }
        with (
            patch.object(app, "_load_gold_rows", return_value=VALIDATION_LABELS),
            patch.object(app, "_persist_submission", return_value="legacy persistence"),
        ):
            result = configured_service().submit_for_split(
                "validation", upload, metadata, None
            )

        self.assertEqual(result["value"]["answer_accuracy"], 1.0)
        self.assertEqual(result["value"]["message"], "legacy persistence")

    def test_test_rejects_missing_oauth_before_reading_file(self):
        unreadable = FileProbe()

        with self.assertRaisesRegex(SubmissionError, "Sign in"):
            configured_service().submit_for_split("test", unreadable, TEST_META, None)

        self.assertFalse(unreadable.was_read)

    def test_test_rejects_unavailable_configuration_before_reading_file(self):
        unreadable = FileProbe()
        service = configured_service(
            loader=lambda now: (_ for _ in ()).throw(RuntimeError("private config detail"))
        )

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            service.submit_for_split("test", unreadable, TEST_META, PROFILE)

        self.assertFalse(unreadable.was_read)

    def test_service_rechecks_fresh_window_after_loading_config_before_file_read(self):
        close_at = NOW + dt.timedelta(seconds=1)
        policy = TestReleasePolicy(
            release_id=TRUSTED_POLICY.release_id,
            task_manifest_sha256=TRUSTED_POLICY.task_manifest_sha256,
            gold_sha256=TRUSTED_POLICY.gold_sha256,
            open_at=TRUSTED_POLICY.open_at,
            close_at=close_at,
            enabled=True,
        )
        clock_values = iter((NOW, close_at))
        unreadable = FileProbe()
        service = SubmissionService(
            validation_submitter=lambda file_obj, metadata: None,
            test_store=RecordingStore(),
            test_config_loader=lambda now: TrustedTestConfig(
                policy=policy,
                labels=TEST_LABELS,
                scoring_gold_sha256=policy.gold_sha256,
                private_revision="private-sha",
                public_revision="f" * 40,
                public_repo_id="public/repo",
                task_manifest_path="test/tasks.jsonl",
            ),
            now_provider=lambda: next(clock_values),
        )

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            service.submit_for_split("test", unreadable, TEST_META, PROFILE)

        self.assertFalse(unreadable.was_read)

    def test_closed_split_allowlist_rejects_client_gold_path_and_attempt_identity(self):
        unreadable = FileProbe()

        with self.assertRaises(SubmissionError) as caught:
            configured_service().submit_for_split(
                "../../private", unreadable, TEST_META, PROFILE
            )

        self.assertEqual(str(caught.exception), "Select validation or test.")
        self.assertFalse(unreadable.was_read)
        self.assertNotIn("../../private", str(caught.exception))

    def test_test_uses_only_oauth_identity_and_server_release_labels(self):
        upload = test_file()
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        store = RecordingStore()
        service = configured_service(store=store)

        result = service.submit_for_split("test", upload, TEST_META, PROFILE)

        self.assertEqual(
            result,
            {
                "accepted": True,
                "attempt": 1,
                "receipt": "receipt-1",
                "answer_accuracy": 1.0,
                "evidence_f1": 1.0,
                "accepted_at": "2026-09-05T12:00:01Z",
            },
        )
        recorded = store.submissions[0]
        self.assertEqual(
            recorded["identity"], OAuthIdentity.from_profile(PROFILE)
        )
        self.assertEqual(
            recorded["metadata"],
            {
                "release_id": "trusted-release",
                "task_manifest_sha256": "a" * 64,
                "scoring_gold_sha256": "b" * 64,
                "scoring_private_revision": "private-sha",
                "scoring_public_revision": "f" * 40,
                "scoring_public_repo_id": "public/repo",
                "scoring_task_manifest_path": "test/tasks.jsonl",
                "team": "Fixture Team",
                "participant_names": "Alice Example, Bob Example",
                "submission_name": "final-run",
            },
        )
        self.assertEqual(recorded["metrics"]["answer_accuracy"], 1.0)
        self.assertEqual(recorded["metrics"]["evidence_f1"], 1.0)
        self.assertEqual(recorded["predictions"], TEST_ROWS)
        self.assertNotIn("now", recorded)

    def test_attempt_two_direct_response_withholds_all_metrics(self):
        upload = test_file()
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        service = configured_service(store=RecordingStore(attempts=(2,)))

        result = service.submit_for_split("test", upload, TEST_META, PROFILE)

        self.assertEqual(
            result,
            {
                "accepted": True,
                "attempt": 2,
                "receipt": "receipt-2",
                "score": "withheld",
                "accepted_at": "2026-09-05T12:00:02Z",
            },
        )
        self.assertNotIn("answer_accuracy", result)
        self.assertNotIn("evidence_f1", result)

    def test_exact_retry_uses_persisted_metrics_without_calling_scorer(self):
        upload = test_file()
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        persisted = {
            "attempt_number": 1,
            "submission_id": "persisted-receipt",
            "submission_name": "original-name",
            "submitted_at": "2026-09-05T12:00:09Z",
            "metrics": {
                "answer_accuracy": 0.25,
                "evidence_exact_match": 0.0,
                "evidence_f1": 0.5,
                "examples": 1,
                "per_example": [{"instance_id": "private-id"}],
            },
            "predictions": [{"answer": "private-answer"}],
        }
        store = RecordingStore(exact_attempt=persisted)
        service = configured_service(store=store)

        with patch(
            "submission_service.score_predictions",
            side_effect=AssertionError("the scorer must not run for an exact retry"),
        ) as scorer:
            result = service.submit_for_split("test", upload, TEST_META, PROFILE)

        self.assertEqual(
            result,
            {
                "accepted": True,
                "attempt": 1,
                "receipt": "persisted-receipt",
                "answer_accuracy": 0.25,
                "evidence_f1": 0.5,
                "accepted_at": "2026-09-05T12:00:09Z",
            },
        )
        scorer.assert_not_called()
        self.assertEqual(store.submissions, [])
        self.assertEqual(len(store.lookup_requests), 1)

    def test_racing_exact_retry_response_uses_the_store_record_not_rescored_metrics(self):
        upload = test_file()
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        persisted = {
            "attempt_number": 1,
            "submission_id": "winning-race-receipt",
            "submission_name": "original-name",
            "submitted_at": "2026-09-05T12:00:11Z",
            "metrics": {"answer_accuracy": 0.25, "evidence_f1": 0.5},
        }

        class RacingRetryStore(RecordingStore):
            def submit(self, identity, metadata, predictions, metrics):
                return TestReceipt(
                    True,
                    1,
                    "winning-race-receipt",
                    "2026-09-05T12:00:11Z",
                    matched_attempt=persisted,
                )

        service = configured_service(store=RacingRetryStore())
        result = service.submit_for_split("test", upload, TEST_META, PROFILE)

        self.assertEqual(result["receipt"], "winning-race-receipt")
        self.assertEqual(result["answer_accuracy"], 0.25)
        self.assertEqual(result["evidence_f1"], 0.5)

    def test_oversized_test_upload_is_rejected_before_file_read_or_scoring(self):
        upload = test_file()
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        with Path(upload.name).open("r+b") as handle:
            handle.truncate(10 * 1024 * 1024 + 1)
        store = RecordingStore()
        service = configured_service(store=store)

        with (
            patch(
                "submission_service.Path.read_text",
                side_effect=AssertionError("oversized upload must not be read"),
            ) as read_text,
            patch("submission_service.score_predictions") as scorer,
            self.assertRaisesRegex(SubmissionError, "could not be accepted"),
        ):
            service.submit_for_split("test", upload, TEST_META, PROFILE)

        read_text.assert_not_called()
        scorer.assert_not_called()
        self.assertEqual(store.submissions, [])

    def test_test_row_limit_is_enforced_before_scoring_or_persistence(self):
        upload = test_file([{} for _ in range(10_001)])
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        store = RecordingStore()
        service = configured_service(store=store)

        with (
            patch("submission_service.score_predictions") as scorer,
            self.assertRaisesRegex(SubmissionError, "could not be accepted"),
        ):
            service.submit_for_split("test", upload, TEST_META, PROFILE)

        scorer.assert_not_called()
        self.assertEqual(store.submissions, [])

    def test_test_answer_and_evidence_bounds_precede_scoring_and_persistence(self):
        cases = {
            "answer characters": {
                "instance_id": "test-1",
                "answer": "x" * 4_097,
                "evidence": ["b1"],
            },
            "evidence identifiers": {
                "instance_id": "test-1",
                "answer": "42",
                "evidence": [f"b{index}" for index in range(129)],
            },
            "evidence identifier characters": {
                "instance_id": "test-1",
                "answer": "42",
                "evidence": ["b" * 257],
            },
        }
        for name, row in cases.items():
            with self.subTest(name=name):
                upload = test_file([row])
                self.addCleanup(Path(upload.name).unlink, missing_ok=True)
                store = RecordingStore()
                service = configured_service(store=store)
                with (
                    patch("submission_service.score_predictions") as scorer,
                    self.assertRaisesRegex(SubmissionError, "could not be accepted"),
                ):
                    service.submit_for_split("test", upload, TEST_META, PROFILE)
                scorer.assert_not_called()
                self.assertEqual(store.submissions, [])

    def test_scoring_capacity_timeout_is_one_second_and_consumes_no_attempt(self):
        class UnavailableScoringSlot:
            def __init__(self):
                self.timeouts = []
                self.release_calls = 0

            def acquire(self, *, timeout):
                self.timeouts.append(timeout)
                return False

            def release(self):
                self.release_calls += 1

        upload = test_file()
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        store = RecordingStore()
        service = configured_service(store=store)
        limiter = UnavailableScoringSlot()
        service.scoring_semaphore = limiter

        with (
            patch("submission_service.score_predictions") as scorer,
            self.assertRaisesRegex(SubmissionError, "could not be accepted"),
        ):
            service.submit_for_split("test", upload, TEST_META, PROFILE)

        self.assertEqual(limiter.timeouts, [1.0])
        self.assertEqual(limiter.release_calls, 0)
        scorer.assert_not_called()
        self.assertEqual(store.submissions, [])

    def test_default_test_scoring_gate_allows_only_two_concurrent_scores(self):
        release_scorers = threading.Event()
        two_scorers_entered = threading.Event()
        active_lock = threading.Lock()
        active = 0
        max_active = 0

        def blocking_scorer(predictions, labels):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    two_scorers_entered.set()
            try:
                was_released = release_scorers.wait(timeout=3)
                if not was_released:
                    raise AssertionError("scoring gate fixture timed out")
                return {
                    "answer_accuracy": 1.0,
                    "evidence_exact_match": 1.0,
                    "evidence_f1": 1.0,
                    "examples": 1,
                    "per_example": [],
                }
            finally:
                with active_lock:
                    active -= 1

        class ConcurrentStore:
            def find_exact_attempt(self, identity, metadata, predictions):
                return None

            def submit(self, identity, metadata, predictions, metrics):
                return TestReceipt(True, 1, f"receipt-{id(predictions)}", "2026-09-05T12:00:01Z")

        uploads = [
            test_file(
                [
                    {
                        "instance_id": "test-1",
                        "answer": f"answer-{index}",
                        "evidence": ["b1"],
                    }
                ]
            )
            for index in range(3)
        ]
        for upload in uploads:
            self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        service = configured_service(store=ConcurrentStore())

        try:
            with (
                patch("submission_service.score_predictions", new=blocking_scorer),
                ThreadPoolExecutor(max_workers=3) as pool,
            ):
                futures = [
                    pool.submit(
                        service.submit_for_split,
                        "test",
                        upload,
                        TEST_META,
                        PROFILE,
                    )
                    for upload in uploads
                ]
                self.assertTrue(two_scorers_entered.wait(timeout=1))
                completed, pending = wait(
                    futures,
                    timeout=1.5,
                    return_when=FIRST_COMPLETED,
                )
                self.assertEqual(len(completed), 1)
                with self.assertRaisesRegex(SubmissionError, "could not be accepted"):
                    next(iter(completed)).result()
                self.assertEqual(len(pending), 2)
                release_scorers.set()
                accepted = []
                rejected = []
                for future in futures:
                    try:
                        accepted.append(future.result(timeout=2))
                    except SubmissionError as exc:
                        rejected.append(exc)
        finally:
            release_scorers.set()

        self.assertEqual(max_active, 2)
        self.assertEqual(len(rejected), 1)
        self.assertIn("could not be accepted", str(rejected[0]))
        self.assertEqual(len(accepted), 2)
        self.assertTrue(all(result["accepted"] for result in accepted))

    def test_fourth_distinct_attempt_reveals_only_participant_safe_receipts(self):
        private_attempts = (
            {
                "attempt_number": 1,
                "submission_id": "receipt-1",
                "submission_name": "first",
                "submitted_at": "2026-09-05T12:00:01Z",
                "metrics": {"answer_accuracy": 1.0, "evidence_f1": 0.75},
            },
            {
                "attempt_number": 2,
                "submission_id": "receipt-2",
                "submission_name": "second",
                "submitted_at": "2026-09-05T12:00:02Z",
                "metrics": {"answer_accuracy": 0.125, "evidence_f1": 0.25},
            },
            {
                "attempt_number": 3,
                "submission_id": "receipt-3",
                "submission_name": "third",
                "submitted_at": "2026-09-05T12:00:03Z",
                "metrics": {"answer_accuracy": 0.5, "evidence_f1": 0.625},
            },
        )

        class LimitStore(RecordingStore):
            def submit(self, *args, **kwargs):
                return TestReceipt(False, None, "", None, private_attempts)

        upload = test_file(
            [{"instance_id": "test-1", "answer": "new", "evidence": ["b1"]}]
        )
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        result = configured_service(store=LimitStore()).submit_for_split(
            "test", upload, TEST_META, PROFILE
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(
            [attempt["receipt"] for attempt in result["attempts"]],
            ["receipt-1", "receipt-2", "receipt-3"],
        )
        self.assertEqual(result["attempts"][0]["answer_accuracy"], 1.0)
        self.assertEqual(result["attempts"][0]["evidence_f1"], 0.75)
        self.assertEqual(result["attempts"][1]["score"], "withheld")
        self.assertEqual(result["attempts"][2]["score"], "withheld")
        serialized = json.dumps(result)
        for withheld in ("0.125", "0.25", "0.5", "0.625"):
            self.assertNotIn(withheld, serialized)

    def test_history_is_bound_to_oauth_and_suppresses_later_metrics(self):
        store = RecordingStore()
        service = configured_service(store=store)

        result = service.history_for_oauth(PROFILE)

        self.assertEqual(store.history_identity, OAuthIdentity.from_profile(PROFILE))
        self.assertEqual(
            result,
            [
                {
                    "accepted": True,
                    "attempt": 1,
                    "receipt": "receipt-1",
                    "answer_accuracy": 1.0,
                    "evidence_f1": 1.0,
                    "submission_name": "first",
                    "accepted_at": "2026-09-05T12:00:01Z",
                },
                {
                    "accepted": True,
                    "attempt": 2,
                    "receipt": "receipt-2",
                    "score": "withheld",
                    "submission_name": "second",
                    "accepted_at": "2026-09-05T12:00:02Z",
                },
            ],
        )
        self.assertNotIn("secret-test-id", json.dumps(result))
        self.assertNotIn("private prediction", json.dumps(result))

    def test_service_genericizes_private_loader_and_store_failures(self):
        secrets = "secret@example.org private-answer score=0.25"

        for service in (
            configured_service(loader=lambda now: (_ for _ in ()).throw(RuntimeError(secrets))),
            configured_service(store=FailingStore(secrets)),
        ):
            upload = test_file()
            self.addCleanup(Path(upload.name).unlink, missing_ok=True)
            with self.subTest(service=service):
                with self.assertRaises(SubmissionError) as caught:
                    service.submit_for_split("test", upload, TEST_META, PROFILE)
                self.assertEqual(
                    str(caught.exception), "Test submission is temporarily unavailable."
                )
                self.assertNotIn(secrets, str(caught.exception))

    def test_gradio_handler_returns_only_attempt_appropriate_feedback(self):
        for attempt, expected in (
            (1, {"answer_accuracy", "evidence_f1"}),
            (3, {"score"}),
        ):
            with self.subTest(attempt=attempt):
                upload = test_file()
                self.addCleanup(Path(upload.name).unlink, missing_ok=True)
                service = configured_service(store=RecordingStore(attempts=(attempt,)))

                with patch.object(app, "_SUBMISSION_SERVICE", service):
                    result = app.submit_for_split("test", upload, TEST_META, PROFILE)

                metric_keys = {
                    key
                    for key in result
                    if key in {"answer_accuracy", "evidence_f1", "score"}
                }
                self.assertEqual(metric_keys, expected)
                self.assertNotIn("evidence_exact_match", result)
                self.assertNotIn("examples", result)
                self.assertNotIn("per_example", result)

    def test_gradio_handler_genericizes_private_exception_details(self):
        upload = test_file()
        self.addCleanup(Path(upload.name).unlink, missing_ok=True)
        secret = "private@example.org private-gold-answer"
        service = configured_service(
            loader=lambda now: (_ for _ in ()).throw(RuntimeError(secret))
        )

        with patch.object(app, "_SUBMISSION_SERVICE", service):
            with self.assertRaises(gradio_error_type()) as caught:
                app.submit_for_split("test", upload, TEST_META, PROFILE)

        self.assertNotIn(secret, str(caught.exception))


class FailingStore:
    def __init__(self, message):
        self.message = message

    def submit(self, *args, **kwargs):
        raise TestStoreError(self.message)


def gradio_error_type():
    return type(app.gr.Error("fixture"))


class InMemoryConfigHub:
    download_root = None

    def __init__(self, release, gold, tasks):
        self.files = {
            ("private/repo", "sealed/release.json"): json.dumps(release).encode(
                "utf-8"
            ),
            ("private/repo", "sealed/gold.jsonl"): gold,
            ("public/repo", "test/tasks.jsonl"): tasks,
        }
        self.repo_calls = []
        self.download_calls = []

    def repo_info(self, repo_id, *, repo_type, revision):
        self.repo_calls.append((repo_id, revision))
        return SimpleNamespace(sha="private-sha")

    def hf_hub_download(self, repo_id, path, *, repo_type, revision):
        self.download_calls.append((repo_id, path, revision))
        target = self.download_root / f"{repo_id}-{path}".replace("/", "-")
        target.write_bytes(self.files[(repo_id, path)])
        return str(target)


class HubTestConfigLoaderTests(unittest.TestCase):
    def setUp(self):
        self.downloads = tempfile.TemporaryDirectory()
        self.addCleanup(self.downloads.cleanup)
        InMemoryConfigHub.download_root = Path(self.downloads.name)
        self.gold = b'{"instance_id":"test-1","answer":"42","evidence":["b1"]}\n'
        self.tasks = (
            b'{"instance_id":"test-1","user_query":"What is the answer?",'
            b'"document_pdf":"test/documents/test-1.pdf"}\n'
        )
        self.release = {
            "release_id": "trusted-release",
            "task_manifest_sha256": hashlib.sha256(self.tasks).hexdigest(),
            "gold_sha256": hashlib.sha256(self.gold).hexdigest(),
            "public_revision": "f" * 40,
            "open_at": "2026-09-01T00:00:00Z",
            "close_at": "2026-10-01T00:00:00Z",
            "enabled": True,
            "max_attempts": 3,
        }

    def test_loader_reads_fixed_server_paths_at_one_sha_and_verifies_gold(self):
        hub = InMemoryConfigHub(self.release, self.gold, self.tasks)

        config = HubTestConfigLoader(
            hub,
            "private/repo",
            public_api=hub,
            public_repo_id="public/repo",
            task_manifest_path="test/tasks.jsonl",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            expected_policy=self._expected_policy(),
            enabled=True,
        )(NOW)

        self.assertEqual(config.policy.release_id, "trusted-release")
        self.assertEqual(config.labels, TEST_LABELS)
        self.assertEqual(config.scoring_gold_sha256, hashlib.sha256(self.gold).hexdigest())
        self.assertEqual(config.private_revision, "private-sha")
        self.assertEqual(config.public_revision, "f" * 40)
        self.assertEqual(config.public_repo_id, "public/repo")
        self.assertEqual(config.task_manifest_path, "test/tasks.jsonl")
        self.assertEqual(hub.repo_calls, [("private/repo", "main")])
        self.assertEqual(
            hub.download_calls,
            [
                ("private/repo", "sealed/release.json", "private-sha"),
                ("private/repo", "sealed/gold.jsonl", "private-sha"),
                ("public/repo", "test/tasks.jsonl", "f" * 40),
            ],
        )

    def test_loader_rejects_policy_that_differs_from_deployment_pin_before_upload_read(self):
        hub = InMemoryConfigHub(self.release, self.gold, self.tasks)
        expected = self._expected_policy()
        for changes in (
            {"release_id": "deployment-release"},
            {"task_manifest_sha256": "c" * 64},
            {"gold_sha256": "d" * 64},
            {"open_at": dt.datetime(2026, 9, 2, tzinfo=dt.timezone.utc)},
            {"close_at": dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)},
        ):
            with self.subTest(changes=changes):
                expected_policy = TestReleasePolicy(
                    release_id=changes.get("release_id", expected.release_id),
                    task_manifest_sha256=changes.get(
                        "task_manifest_sha256", expected.task_manifest_sha256
                    ),
                    gold_sha256=changes.get("gold_sha256", expected.gold_sha256),
                    open_at=changes.get("open_at", expected.open_at),
                    close_at=changes.get("close_at", expected.close_at),
                    enabled=True,
                    max_attempts=3,
                )
                loader = HubTestConfigLoader(
                    hub,
                    "private/repo",
                    public_api=hub,
                    public_repo_id="public/repo",
                    task_manifest_path="test/tasks.jsonl",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    expected_policy=expected_policy,
                    enabled=True,
                )
                service = SubmissionService(
                    validation_submitter=lambda file_obj, metadata: None,
                    test_store=None,
                    test_config_loader=loader,
                    now_provider=lambda: NOW,
                )
                unreadable = FileProbe()

                with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
                    service.submit_for_split("test", unreadable, TEST_META, PROFILE)

                self.assertFalse(unreadable.was_read)

        invalid_attempt_hub = InMemoryConfigHub(
            {**self.release, "max_attempts": 4}, self.gold, self.tasks
        )
        invalid_attempt_loader = HubTestConfigLoader(
            invalid_attempt_hub,
            "private/repo",
            public_api=invalid_attempt_hub,
            public_repo_id="public/repo",
            task_manifest_path="test/tasks.jsonl",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            expected_policy=expected,
            enabled=True,
        )
        invalid_attempt_service = SubmissionService(
            validation_submitter=lambda file_obj, metadata: None,
            test_store=None,
            test_config_loader=invalid_attempt_loader,
            now_provider=lambda: NOW,
        )
        unreadable = FileProbe()
        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            invalid_attempt_service.submit_for_split("test", unreadable, TEST_META, PROFILE)
        self.assertFalse(unreadable.was_read)

    def test_loader_rejects_malformed_server_digest(self):
        release = {**self.release, "task_manifest_sha256": "not-a-sha256"}
        hub = InMemoryConfigHub(release, self.gold, self.tasks)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            self._loader(hub)(NOW)

    def test_loader_rejects_public_task_content_digest_mismatch(self):
        release = {**self.release, "task_manifest_sha256": "a" * 64}
        hub = InMemoryConfigHub(release, self.gold, self.tasks)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            self._loader(hub)(NOW)

    def test_loader_rejects_non_exact_public_task_schema(self):
        tasks = (
            b'{"instance_id":"test-1","user_query":"question",'
            b'"document_pdf":"test/documents/test-1.pdf","answer":"leak"}\n'
        )
        release = {**self.release, "task_manifest_sha256": hashlib.sha256(tasks).hexdigest()}
        hub = InMemoryConfigHub(release, self.gold, tasks)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            self._loader(hub)(NOW)

    def test_loader_rejects_public_task_and_gold_id_mismatch(self):
        tasks = (
            b'{"instance_id":"test-2","user_query":"question",'
            b'"document_pdf":"test/documents/test-2.pdf"}\n'
        )
        release = {**self.release, "task_manifest_sha256": hashlib.sha256(tasks).hexdigest()}
        hub = InMemoryConfigHub(release, self.gold, tasks)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            self._loader(hub)(NOW)

    def test_loader_rejects_noncanonical_or_traversing_document_paths(self):
        invalid_paths = (
            "documents/test-1.pdf",
            "test/documents/../private/test-1.pdf",
            "private/test-1.pdf",
            "test/documents/test-2.pdf",
        )
        for document_pdf in invalid_paths:
            with self.subTest(document_pdf=document_pdf):
                tasks = json.dumps(
                    {
                        "instance_id": "test-1",
                        "user_query": "question",
                        "document_pdf": document_pdf,
                    },
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                release = {
                    **self.release,
                    "task_manifest_sha256": hashlib.sha256(tasks).hexdigest(),
                }
                hub = InMemoryConfigHub(release, self.gold, tasks)

                with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
                    self._loader(hub)(NOW)

    def test_loader_rejects_traversal_bearing_instance_id_with_matching_path(self):
        tasks = (
            b'{"instance_id":"../../private","user_query":"question",'
            b'"document_pdf":"test/documents/../../private.pdf"}\n'
        )
        gold = b'{"instance_id":"../../private","answer":"42","evidence":["b1"]}\n'
        release = {
            **self.release,
            "task_manifest_sha256": hashlib.sha256(tasks).hexdigest(),
            "gold_sha256": hashlib.sha256(gold).hexdigest(),
        }
        hub = InMemoryConfigHub(release, gold, tasks)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            self._loader(hub)(NOW)

    def test_loader_rejects_ids_outside_safe_opaque_component_contract(self):
        invalid_ids = (
            "",
            ".",
            "..",
            "bad/id",
            "bad\\id",
            "bad id",
            "-leading",
            "_leading",
        )
        for instance_id in invalid_ids:
            with self.subTest(instance_id=instance_id):
                tasks = json.dumps(
                    {
                        "instance_id": instance_id,
                        "user_query": "question",
                        "document_pdf": f"test/documents/{instance_id}.pdf",
                    },
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                gold = json.dumps(
                    {"instance_id": instance_id, "answer": "42", "evidence": ["b1"]},
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                release = {
                    **self.release,
                    "task_manifest_sha256": hashlib.sha256(tasks).hexdigest(),
                    "gold_sha256": hashlib.sha256(gold).hexdigest(),
                }
                hub = InMemoryConfigHub(release, gold, tasks)

                with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
                    self._loader(hub)(NOW)

    def test_loader_accepts_safe_opaque_instance_ids(self):
        task_rows = [
            {
                "instance_id": instance_id,
                "user_query": "question",
                "document_pdf": f"test/documents/{instance_id}.pdf",
            }
            for instance_id in ("task_000001", "opaque-id", "Task.2026_01")
        ]
        gold_rows = [
            {"instance_id": row["instance_id"], "answer": "42", "evidence": ["b1"]}
            for row in task_rows
        ]
        tasks = b"".join(
            json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in task_rows
        )
        gold = b"".join(
            json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in gold_rows
        )
        release = {
            **self.release,
            "task_manifest_sha256": hashlib.sha256(tasks).hexdigest(),
            "gold_sha256": hashlib.sha256(gold).hexdigest(),
        }
        hub = InMemoryConfigHub(release, gold, tasks)

        config = self._loader(hub, release)(NOW)

        self.assertEqual(
            [row["instance_id"] for row in config.labels],
            ["task_000001", "opaque-id", "Task.2026_01"],
        )

    def test_loader_rejects_gold_digest_mismatch_without_details(self):
        release = {**self.release, "gold_sha256": "c" * 64}
        hub = InMemoryConfigHub(release, self.gold, self.tasks)

        with self.assertRaises(SubmissionError) as caught:
            self._loader(hub)(NOW)

        self.assertEqual(str(caught.exception), "Test submission is temporarily unavailable.")
        self.assertNotIn("c" * 64, str(caught.exception))

    def test_disabled_loader_performs_no_repository_io(self):
        hub = InMemoryConfigHub(self.release, self.gold, self.tasks)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            HubTestConfigLoader(
                hub,
                "private/repo",
                public_api=hub,
                public_repo_id="public/repo",
                task_manifest_path="test/tasks.jsonl",
                release_config_path="sealed/release.json",
                gold_config_path="sealed/gold.jsonl",
                expected_policy=self._expected_policy(),
                enabled=False,
            )(NOW)

        self.assertEqual(hub.repo_calls, [])
        self.assertEqual(hub.download_calls, [])

    def _expected_policy(self, release=None):
        release = self.release if release is None else release
        return TestReleasePolicy(
            release_id=release["release_id"],
            task_manifest_sha256=release["task_manifest_sha256"],
            gold_sha256=release["gold_sha256"],
            open_at=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
            close_at=dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc),
            enabled=True,
            max_attempts=3,
        )

    def _loader(self, hub, release=None):
        return HubTestConfigLoader(
            hub,
            "private/repo",
            public_api=hub,
            public_repo_id="public/repo",
            task_manifest_path="test/tasks.jsonl",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            expected_policy=self._expected_policy(release),
            enabled=True,
        )


if __name__ == "__main__":
    unittest.main()
