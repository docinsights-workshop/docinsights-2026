import datetime as dt
import hashlib
import json
import tempfile
import unittest
import warnings
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
    def __init__(self, attempts=(1,)):
        self.attempts = iter(attempts)
        self.submissions = []
        self.history_identity = None

    def submit(self, identity, metadata, predictions, metrics, now):
        self.submissions.append(
            {
                "identity": identity,
                "metadata": metadata,
                "predictions": predictions,
                "metrics": metrics,
                "now": now,
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
    loader = loader or (lambda now: TrustedTestConfig(TRUSTED_POLICY, TEST_LABELS))
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
                "team": "Fixture Team",
                "participant_names": "Alice Example, Bob Example",
                "submission_name": "final-run",
            },
        )
        self.assertEqual(recorded["metrics"]["answer_accuracy"], 1.0)
        self.assertEqual(recorded["metrics"]["evidence_f1"], 1.0)
        self.assertEqual(recorded["predictions"], TEST_ROWS)
        self.assertEqual(recorded["now"], NOW)

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

    def __init__(self, release, gold):
        self.files = {
            "private/test_release.json": json.dumps(release).encode("utf-8"),
            "private/test_labels.jsonl": gold,
        }
        self.repo_calls = 0
        self.download_calls = []

    def repo_info(self, repo_id, *, repo_type, revision):
        self.repo_calls += 1
        return SimpleNamespace(sha="trusted-sha")

    def hf_hub_download(self, repo_id, path, *, repo_type, revision):
        self.download_calls.append((path, revision))
        target = self.download_root / path.replace("/", "-")
        target.write_bytes(self.files[path])
        return str(target)


class HubTestConfigLoaderTests(unittest.TestCase):
    def setUp(self):
        self.downloads = tempfile.TemporaryDirectory()
        self.addCleanup(self.downloads.cleanup)
        InMemoryConfigHub.download_root = Path(self.downloads.name)
        self.gold = b'{"instance_id":"test-1","answer":"42","evidence":["b1"]}\n'
        self.release = {
            "release_id": "trusted-release",
            "task_manifest_sha256": "a" * 64,
            "gold_sha256": hashlib.sha256(self.gold).hexdigest(),
            "open_at": "2026-09-01T00:00:00Z",
            "close_at": "2026-10-01T00:00:00Z",
            "enabled": True,
            "max_attempts": 3,
        }

    def test_loader_reads_fixed_server_paths_at_one_sha_and_verifies_gold(self):
        hub = InMemoryConfigHub(self.release, self.gold)

        config = HubTestConfigLoader(hub, "private/repo", enabled=True)(NOW)

        self.assertEqual(config.policy.release_id, "trusted-release")
        self.assertEqual(config.labels, TEST_LABELS)
        self.assertEqual(hub.repo_calls, 1)
        self.assertEqual(
            hub.download_calls,
            [
                ("private/test_release.json", "trusted-sha"),
                ("private/test_labels.jsonl", "trusted-sha"),
            ],
        )

    def test_loader_rejects_malformed_server_digest(self):
        release = {**self.release, "task_manifest_sha256": "not-a-sha256"}
        hub = InMemoryConfigHub(release, self.gold)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            HubTestConfigLoader(hub, "private/repo", enabled=True)(NOW)

    def test_loader_rejects_gold_digest_mismatch_without_details(self):
        release = {**self.release, "gold_sha256": "c" * 64}
        hub = InMemoryConfigHub(release, self.gold)

        with self.assertRaises(SubmissionError) as caught:
            HubTestConfigLoader(hub, "private/repo", enabled=True)(NOW)

        self.assertEqual(str(caught.exception), "Test submission is temporarily unavailable.")
        self.assertNotIn("c" * 64, str(caught.exception))

    def test_disabled_loader_performs_no_repository_io(self):
        hub = InMemoryConfigHub(self.release, self.gold)

        with self.assertRaisesRegex(SubmissionError, "temporarily unavailable"):
            HubTestConfigLoader(hub, "private/repo", enabled=False)(NOW)

        self.assertEqual(hub.repo_calls, 0)
        self.assertEqual(hub.download_calls, [])


if __name__ == "__main__":
    unittest.main()
