import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
from app import load_test_deployment_config
from submission_service import SubmissionService


VALID_RELEASE = {
    "TEST_SUBMISSIONS_ENABLED": "true",
    "TEST_PUBLIC_LEADERBOARD_ENABLED": "true",
    "TEST_RELEASE_ID": "docsem-test-2026-09",
    "TEST_TASK_MANIFEST_SHA256": "a" * 64,
    "TEST_GOLD_SHA256": "b" * 64,
    "TEST_OPEN_AT": "2026-09-05T00:00:00Z",
    "TEST_CLOSE_AT": "2026-09-10T00:00:00Z",
    "TEST_MAX_ATTEMPTS": "3",
    "TEST_RELEASE_CONFIG_PATH": "private/test_release.json",
    "TEST_GOLD_CONFIG_PATH": "private/test_labels.jsonl",
    "TEST_TASKS_FILE": "test/tasks.jsonl",
}


def assert_anonymous_validation_fixture(test_case):
    upload = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    test_case.addCleanup(Path(upload.name).unlink, missing_ok=True)
    upload.write(json.dumps({"instance_id": "val-1", "answer": "42", "evidence": ["b1"]}))
    upload.close()
    service = SubmissionService(
        validation_submitter=app._legacy_validation_submitter,
        test_store=None,
        test_config_loader=lambda now: (_ for _ in ()).throw(AssertionError()),
    )
    with (
        patch.object(
            app,
            "_load_gold_rows",
            return_value=[{"instance_id": "val-1", "answer": "42", "evidence": ["b1"]}],
        ),
        patch.object(app, "_persist_submission", return_value="legacy persistence"),
    ):
        result = service.submit_for_split(
            "validation",
            SimpleNamespace(name=upload.name),
            {
                "team": "Fixture Team",
                "contact": "lead@example.org",
                "submission_name": "baseline",
                "participant_names": "Alice Example",
            },
            None,
        )
    test_case.assertEqual(
        result["value"],
        {
            "answer_accuracy": 1.0,
            "evidence_exact_match": 1.0,
            "evidence_f1": 1.0,
            "examples": 1,
            "message": "legacy persistence",
        },
    )


class TestDeploymentConfigTests(unittest.TestCase):
    def test_defaults_disable_test_surfaces(self):
        config = load_test_deployment_config({})

        self.assertFalse(config.submissions_enabled)
        self.assertFalse(config.public_leaderboard_enabled)
        self.assertEqual(config.max_attempts, 3)

    def test_each_missing_required_release_value_disables_requested_test_surfaces(self):
        for missing in (
            "TEST_RELEASE_ID",
            "TEST_TASK_MANIFEST_SHA256",
            "TEST_GOLD_SHA256",
            "TEST_OPEN_AT",
            "TEST_CLOSE_AT",
        ):
            with self.subTest(missing=missing):
                environment = dict(VALID_RELEASE)
                del environment[missing]

                config = load_test_deployment_config(environment)

                self.assertFalse(config.submissions_enabled)
                self.assertFalse(config.public_leaderboard_enabled)
                self.assertEqual(config.max_attempts, 3)
                assert_anonymous_validation_fixture(self)

    def test_missing_server_config_paths_disable_test_and_preserve_validation(self):
        for missing in ("TEST_RELEASE_CONFIG_PATH", "TEST_GOLD_CONFIG_PATH"):
            with self.subTest(missing=missing):
                environment = dict(VALID_RELEASE)
                del environment[missing]

                config = load_test_deployment_config(environment)

                self.assertFalse(config.submissions_enabled)
                self.assertFalse(config.public_leaderboard_enabled)
                assert_anonymous_validation_fixture(self)

    def test_noncanonical_safe_server_paths_disable_test_and_preserve_validation(self):
        for key, value in (
            ("TEST_RELEASE_CONFIG_PATH", "sealed/release.json"),
            ("TEST_GOLD_CONFIG_PATH", "sealed/gold.jsonl"),
            ("TEST_TASKS_FILE", "test/other-tasks.jsonl"),
        ):
            with self.subTest(key=key, value=value):
                config = load_test_deployment_config({**VALID_RELEASE, key: value})

                self.assertFalse(config.submissions_enabled)
                self.assertFalse(config.public_leaderboard_enabled)
                assert_anonymous_validation_fixture(self)

    def test_malformed_or_non_utc_windows_disable_requested_test_surfaces(self):
        for key, value in (
            ("TEST_OPEN_AT", "2026-09-05 00:00:00Z"),
            ("TEST_OPEN_AT", "2026-09-05T00:00:00+00:00"),
            ("TEST_CLOSE_AT", "not-a-time"),
        ):
            with self.subTest(key=key, value=value):
                environment = {**VALID_RELEASE, key: value}
                config = load_test_deployment_config(environment)

                self.assertFalse(config.submissions_enabled)
                self.assertFalse(config.public_leaderboard_enabled)
                assert_anonymous_validation_fixture(self)

    def test_malformed_release_id_or_digest_disables_requested_test_surfaces(self):
        for key, value in (
            ("TEST_RELEASE_ID", "../candidate"),
            ("TEST_TASK_MANIFEST_SHA256", "A" * 64),
            ("TEST_GOLD_SHA256", "not-a-sha256"),
        ):
            with self.subTest(key=key, value=value):
                config = load_test_deployment_config({**VALID_RELEASE, key: value})

                self.assertFalse(config.submissions_enabled)
                self.assertFalse(config.public_leaderboard_enabled)
                assert_anonymous_validation_fixture(self)

    def test_closed_or_non_three_attempt_configuration_disables_test_surfaces(self):
        for environment in (
            {
                **VALID_RELEASE,
                "TEST_OPEN_AT": "2026-09-10T00:00:00Z",
                "TEST_CLOSE_AT": "2026-09-10T00:00:00Z",
            },
            {**VALID_RELEASE, "TEST_MAX_ATTEMPTS": "4"},
        ):
            with self.subTest(environment=environment):
                config = load_test_deployment_config(environment)

                self.assertFalse(config.submissions_enabled)
                self.assertFalse(config.public_leaderboard_enabled)
                self.assertEqual(config.max_attempts, 3)
                assert_anonymous_validation_fixture(self)

    def test_complete_explicit_release_configuration_enables_requested_surfaces(self):
        config = load_test_deployment_config(VALID_RELEASE)

        self.assertTrue(config.submissions_enabled)
        self.assertTrue(config.public_leaderboard_enabled)
        self.assertEqual(config.release_id, "docsem-test-2026-09")
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.feedback_policy, "first-attempt-only")
        self.assertEqual(config.task_manifest_path, "test/tasks.jsonl")


if __name__ == "__main__":
    unittest.main()
