import datetime as dt
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

import app
from submission_service import SubmissionService, TrustedTestConfig


PROFILE_A = {
    "sub": "subject-a",
    "email": "alice@example.org",
    "email_verified": True,
    "name": "Alice Example",
    "preferred_username": "alice",
    "profile": "https://huggingface.co/alice",
    "picture": "https://huggingface.co/avatars/alice.svg",
}

PRIVATE_HEAD = "a" * 40
TASK_DIGEST = "b" * 64
GOLD_DIGEST = "c" * 64
SCORER_REVISION = "d" * 40
SCORER_DIGEST = "e" * 64
SOURCE_REVISION = "f" * 40


def trusted_test_config(policy):
    return TrustedTestConfig(
        policy=policy,
        labels=[],
        scoring_gold_sha256=policy.gold_sha256,
        private_revision="d" * 40,
        public_revision="e" * 40,
        public_repo_id="public/docsem",
        task_manifest_path="test/tasks.jsonl",
    )


def canonical_json(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def finalized_artifacts(*, rows=None):
    projection = {
        "schema_version": 2,
        "split": "test",
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_DIGEST,
        "rows": rows
        or [
            {
                "rank": 1,
                "hf_username": "alice<script>",
                "team": "Team <Alpha>",
                "submission_name": "best & final",
                "selected_attempt": 2,
                "joint_accuracy": 0.625,
                "answer_accuracy": 0.75,
                "evidence_f1": 0.5,
            }
        ],
    }
    projection_bytes = canonical_json(projection)
    projection_digest = hashlib.sha256(projection_bytes).hexdigest()
    audit = {
        "schema_version": 1,
        "split": "test",
        "release_id": "docsem-test-2026",
        "source_revision": SOURCE_REVISION,
        "finalized_at": "2026-09-11T12:00:01Z",
        "close_at": "2026-09-11T12:00:00Z",
        "task_manifest_sha256": TASK_DIGEST,
        "gold_sha256": GOLD_DIGEST,
        "scorer_revision": SCORER_REVISION,
        "scorer_code_sha256": SCORER_DIGEST,
        "public_projection_sha256": projection_digest,
        "eligible_attempt_count": 1,
        "excluded_attempt_count": 0,
        "selected_account_count": len(projection["rows"]),
        "input_manifest_sha256": "1" * 64,
        "input_records": [],
        "projection_issue_codes": [],
        "eligible_attempts": [
            {
                "account_key": "0" * 64,
                "submission_id": "fixture-submission",
                "attempt_number": 2,
                "record_sha256": "2" * 64,
                "selected": True,
                "joint_accuracy": 0.625,
                "answer_accuracy": 0.75,
                "evidence_f1": 0.5,
                "rescored_metrics_sha256": "3" * 64,
            }
        ],
        "excluded_attempts": [],
        "applied_audit_records": [],
        "metric_absolute_tolerance": 1e-12,
    }
    audit_bytes = canonical_json(audit)
    release = {
        "schema_version": 1,
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_DIGEST,
        "gold_sha256": GOLD_DIGEST,
        "enabled": False,
        "finalized": True,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "open_at": "2026-09-05T00:00:00Z",
        "close_at": "2026-09-11T12:00:00Z",
        "public_revision": "2" * 40,
        "public_repo_id": "public/docsem",
        "task_manifest_path": "test/tasks.jsonl",
        "finalized_at": "2026-09-11T12:00:01Z",
        "finalization_source_revision": SOURCE_REVISION,
        "finalization_scorer_revision": SCORER_REVISION,
        "finalization_scorer_sha256": SCORER_DIGEST,
        "final_projection_sha256": projection_digest,
        "finalization_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
    }
    return {
        "private/test_release.json": canonical_json(release),
        "projections/test/public_final.json": projection_bytes,
        "private/test_finalization_audit.json": audit_bytes,
    }


def provisional_artifacts(*, rows=None):
    release = {
        "schema_version": 1,
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_DIGEST,
        "gold_sha256": GOLD_DIGEST,
        "enabled": True,
        "finalized": False,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "open_at": "2026-09-05T00:00:00Z",
        "close_at": "2026-09-11T12:00:00Z",
        "public_revision": "2" * 40,
        "public_repo_id": "public/docsem",
        "task_manifest_path": "test/tasks.jsonl",
    }
    projection = {
        "schema_version": 3,
        "split": "test",
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_DIGEST,
        "rows": rows
        or [
            {
                "rank": 1,
                "hf_username": "alice<script>",
                "team": "Team <Alpha>",
            }
        ],
    }
    return {
        "private/test_release.json": canonical_json(release),
        "projections/test/public_provisional.json": canonical_json(projection),
    }


def resign_final_artifacts(artifacts):
    """Update only fixture hashes so a semantic mutation reaches its validator."""

    resigned = dict(artifacts)
    projection_digest = hashlib.sha256(
        resigned["projections/test/public_final.json"]
    ).hexdigest()
    audit = json.loads(resigned["private/test_finalization_audit.json"])
    audit["public_projection_sha256"] = projection_digest
    resigned["private/test_finalization_audit.json"] = canonical_json(audit)
    release = json.loads(resigned["private/test_release.json"])
    release["final_projection_sha256"] = projection_digest
    release["finalization_audit_sha256"] = hashlib.sha256(
        resigned["private/test_finalization_audit.json"]
    ).hexdigest()
    resigned["private/test_release.json"] = canonical_json(release)
    return resigned


class FinalLeaderboardHub:
    def __init__(self, *, private=True, sha=PRIVATE_HEAD):
        self.private = private
        self.sha = sha
        self.calls = []

    def repo_info(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(private=self.private, sha=self.sha)


class AccountHistoryStore:
    def __init__(self):
        self.requested_identities = []

    def account_history(self, identity):
        self.requested_identities.append(
            (identity.identity_kind, identity.identity_subject)
        )
        if identity.identity_subject == "subject-a":
            return [
                {
                    "attempt_number": 1,
                    "submission_id": "receipt-a1",
                    "submission_name": "alice-first",
                    "submitted_at": "2026-09-05T12:00:01Z",
                    "metrics": {
                        "joint_accuracy": 0.625,
                        "answer_accuracy": 1.0,
                        "evidence_f1": 0.75,
                        "per_example": [{"instance_id": "secret-a"}],
                    },
                },
                {
                    "attempt_number": 2,
                    "submission_id": "receipt-a2",
                    "submission_name": "alice-second",
                    "submitted_at": "2026-09-05T12:00:02Z",
                    "metrics": {
                        "joint_accuracy": 0.125,
                        "answer_accuracy": 0.25,
                        "evidence_f1": 0.5,
                        "per_example": [{"instance_id": "secret-a"}],
                    },
                },
            ]
        return [
            {
                "attempt_number": 1,
                "submission_id": "receipt-b1",
                "submission_name": "bob-only",
                "submitted_at": "2026-09-05T12:00:03Z",
                "metrics": {
                    "joint_accuracy": 0.0,
                    "answer_accuracy": 0.125,
                    "evidence_f1": 0.25,
                },
            }
        ]


def configured_service(*, validation_submitter=lambda file_obj, metadata: {"ok": True}):
    store = AccountHistoryStore()
    service = SubmissionService(
        validation_submitter=validation_submitter,
        test_store=store,
        test_config_loader=lambda now: None,
    )
    return service, store


def final_deployment(**overrides):
    values = {
        "submissions_enabled": False,
        "public_leaderboard_enabled": True,
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_DIGEST,
        "gold_sha256": GOLD_DIGEST,
        "open_at": dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
        "close_at": dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
        "release_config_path": "private/test_release.json",
        "gold_config_path": "private/test_labels.jsonl",
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "task_manifest_path": "test/tasks.jsonl",
    }
    values.update(overrides)
    return app.TestDeploymentConfig(**values)


class PortalBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def endpoint(self, api_name):
        matches = [
            (index, block_fn)
            for index, block_fn in app.demo.fns.items()
            if block_fn.api_name == api_name
        ]
        self.assertEqual(len(matches), 1, f"missing generated endpoint {api_name}")
        return matches[0]

    async def invoke(self, api_name, inputs, profile=None):
        index, _ = self.endpoint(api_name)
        session = {"oauth_info": {"userinfo": profile}} if profile else {}
        return await app.demo.process_api(
            index,
            inputs,
            request=gr.Request(session=session, session_hash="portal-test"),
            session_hash="portal-test",
        )

    async def test_split_selection_adapts_instructions_contact_and_test_history(self):
        test_response = await self.invoke("select_split", [app.TEST_SPLIT_LABEL])
        test_updates = test_response["data"]
        self.assertIn(
            "Sign in with Hugging Face (recommended)", test_updates[0]["value"]
        )
        self.assertIn("Signed-out users", test_updates[0]["value"])
        self.assertIn("keyed to that email", test_updates[0]["value"])
        self.assertIn("Test submissions are not open yet", test_updates[0]["value"])
        self.assertTrue(test_updates[1]["visible"])
        self.assertEqual(test_updates[2]["value"], "Submit test predictions")
        self.assertFalse(test_updates[2]["interactive"])
        self.assertTrue(test_updates[3]["visible"])

        validation_response = await self.invoke(
            "select_split", [app.VALIDATION_SPLIT_LABEL]
        )
        validation_updates = validation_response["data"]
        self.assertIn("Submit validation predictions", validation_updates[0]["value"])
        self.assertTrue(validation_updates[1]["visible"])
        self.assertEqual(validation_updates[2]["value"], "Validate and score")
        self.assertTrue(validation_updates[2]["interactive"])
        self.assertFalse(validation_updates[3]["visible"])

    def test_validation_maintenance_state_disables_only_submission_ui(self):
        with patch.object(app, "VALIDATION_SUBMISSIONS_ENABLED", False):
            intro, contact, submit, history = app.split_ui(app.VALIDATION_SPLIT_LABEL)
        with patch.object(app, "_load_leaderboard_rows", return_value=[]):
            heading, leaderboard, refresh = app.leaderboard_view(
                app.VALIDATION_LEADERBOARD_LABEL
            )

        self.assertIn("paused for maintenance", intro["value"])
        self.assertTrue(contact["visible"])
        self.assertFalse(submit["interactive"])
        self.assertFalse(history["visible"])
        self.assertIn("Validation leaderboard", heading["value"])
        self.assertIn("No scored submissions yet", leaderboard["value"])
        self.assertTrue(refresh["visible"])

    async def test_public_test_inputs_notice_keeps_scoring_closed(self):
        public_dataset_url = (
            "https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data"
        )
        initial_portal = "\n".join(
            str(component["props"].get("value", ""))
            for component in app.demo.get_config_file()["components"]
        )

        self.assertIn(public_dataset_url, initial_portal)
        self.assertIn("public test tasks and PDFs are available", initial_portal)
        self.assertIn("Test submissions are not open yet", initial_portal)
        self.assertRegex(
            initial_portal,
            r"after the private\s+scoring key is installed and verified",
        )

        response = await self.invoke("select_split", [app.TEST_SPLIT_LABEL])
        instructions, _, submit_button, _ = response["data"]

        self.assertIn(public_dataset_url, instructions["value"])
        self.assertIn("public test tasks and PDFs are available", instructions["value"])
        self.assertIn("Test submissions are not open yet", instructions["value"])
        self.assertIn(
            "after the private scoring key is installed and verified",
            instructions["value"],
        )
        self.assertFalse(submit_button["interactive"])

    def test_test_release_notice_follows_authoritative_disabled_open_and_closed_state(
        self,
    ):
        deployment = app.TestDeploymentConfig(
            submissions_enabled=True,
            public_leaderboard_enabled=False,
            release_id="docsem-test-2026",
            task_manifest_sha256="a" * 64,
            gold_sha256="b" * 64,
            open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
            close_at=dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
            release_config_path="private/test_release.json",
            gold_config_path="private/test_labels.jsonl",
        )
        disabled = app._test_release_notice_html(
            dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
            deployment=deployment,
            submissions_enabled=False,
            write_token="server-token",
        )
        open_notice = app._test_release_notice_html(
            dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
            deployment=deployment,
            submissions_enabled=True,
            write_token="server-token",
            authoritative_loader=lambda now: trusted_test_config(
                deployment.expected_policy
            ),
        )
        closed = app._test_release_notice_html(
            deployment.close_at,
            deployment=deployment,
            submissions_enabled=True,
            write_token="server-token",
        )
        disabled_after_close = app._test_release_notice_html(
            deployment.close_at,
            deployment=deployment,
            submissions_enabled=False,
            write_token=None,
        )

        self.assertIn("Test submissions are not open yet.", disabled)
        self.assertNotIn("Test submissions are open.", disabled)
        self.assertIn("Test submissions are open.", open_notice)
        self.assertNotIn("Test submissions are not open yet.", open_notice)
        self.assertIn("Test submissions are closed.", closed)
        self.assertNotIn("Test submissions are open.", closed)
        self.assertIn("Test submissions are closed.", disabled_after_close)
        self.assertNotIn("Test submissions are not open yet.", disabled_after_close)
        for required_link in (
            "https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data",
            app.WORKSHOP_URL,
            app.PARTICIPANT_GUIDE_URL,
        ):
            with self.subTest(required_link=required_link):
                self.assertIn(required_link, open_notice)

    def test_countdown_uses_exact_deployment_close_instant_and_aoe_boundary(self):
        close_at = dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc)

        self.assertEqual(
            app._deadline_labels(close_at),
            (
                "September 10, 2026 at 11:59:59 PM Anywhere on Earth",
                "September 11, 2026 at 12:00:00 UTC",
            ),
        )
        self.assertEqual(
            app._countdown_text(
                close_at,
                dt.datetime(2026, 9, 10, 9, 57, 56, tzinfo=dt.timezone.utc),
            ),
            "1 day, 2 hours, 2 minutes, 4 seconds remaining",
        )
        self.assertEqual(
            app._countdown_text(close_at, close_at), "Test submissions are closed."
        )
        self.assertEqual(
            app._countdown_text(close_at, close_at + dt.timedelta(seconds=1)),
            "Test submissions are closed.",
        )

    def test_test_policy_copy_explains_private_feedback_and_rank_contract(self):
        notice = app._test_release_notice_html(
            dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
            deployment=app.TestDeploymentConfig(
                submissions_enabled=True,
                public_leaderboard_enabled=False,
                release_id="docsem-test-2026",
                task_manifest_sha256="a" * 64,
                gold_sha256="b" * 64,
                open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
                close_at=dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
                release_config_path="private/test_release.json",
                gold_config_path="private/test_labels.jsonl",
            ),
            submissions_enabled=True,
            write_token="server-token",
            authoritative_loader=lambda now: trusted_test_config(
                app.TestReleasePolicy(
                    release_id="docsem-test-2026",
                    task_manifest_sha256="a" * 64,
                    gold_sha256="b" * 64,
                    open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
                    close_at=dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
                    enabled=True,
                )
            ),
        )

        self.assertIn("3 accepted test submissions per identity", notice)
        self.assertIn("Hugging Face account or normalized contact email", notice)
        self.assertIn("alternate anonymous emails", notice)
        self.assertIn("Attempt 1", notice)
        self.assertIn("private to that submitting identity", notice)
        self.assertIn("Joint Exact Accuracy", notice)
        self.assertIn("Answer Exact Accuracy", notice)
        self.assertIn("Evidence F1 (macro)", notice)
        self.assertIn("Attempts 2–3", notice)
        self.assertIn("withheld", notice)
        self.assertIn("provisional public ranks use only attempt 1", notice)
        self.assertIn("no metrics", notice)
        self.assertIn("best of all 3 eligible attempts", notice)

    def test_open_copy_requires_authoritative_private_release_verification(self):
        deployment = app.TestDeploymentConfig(
            submissions_enabled=True,
            public_leaderboard_enabled=False,
            release_id="docsem-test-2026",
            task_manifest_sha256="a" * 64,
            gold_sha256="b" * 64,
            open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
            close_at=dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
            release_config_path="private/test_release.json",
            gold_config_path="private/test_labels.jsonl",
        )
        now = dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc)

        def unavailable(_):
            raise RuntimeError("private release unavailable")

        mismatched_policy = app.TestReleasePolicy(
            release_id="different-release",
            task_manifest_sha256="a" * 64,
            gold_sha256="b" * 64,
            open_at=deployment.open_at,
            close_at=deployment.close_at,
            enabled=True,
        )
        unavailable_notice = app._test_release_notice_html(
            now,
            deployment=deployment,
            submissions_enabled=True,
            write_token="server-token",
            authoritative_loader=unavailable,
        )
        mismatched_notice = app._test_release_notice_html(
            now,
            deployment=deployment,
            submissions_enabled=True,
            write_token="server-token",
            authoritative_loader=lambda current: trusted_test_config(mismatched_policy),
        )
        verified_notice = app._test_release_notice_html(
            now,
            deployment=deployment,
            submissions_enabled=True,
            write_token="server-token",
            authoritative_loader=lambda current: trusted_test_config(
                deployment.expected_policy
            ),
        )

        for refused in (unavailable_notice, mismatched_notice):
            self.assertIn("Test submissions are not open yet.", refused)
            self.assertNotIn("Test submissions are open.", refused)
        self.assertIn("Test submissions are open.", verified_notice)

    async def test_test_ui_requires_write_token_and_current_open_server_window(self):
        policy = app.TestReleasePolicy(
            release_id="configured-release",
            task_manifest_sha256="a" * 64,
            gold_sha256="b" * 64,
            open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
            close_at=dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
            enabled=True,
        )
        cases = (
            (
                "missing write token",
                None,
                dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
                False,
            ),
            (
                "closed server window",
                "server-write-token",
                policy.close_at,
                False,
            ),
            (
                "open server window",
                "server-write-token",
                dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
                True,
            ),
        )
        for name, write_token, now, expected_open in cases:
            with self.subTest(name=name):
                deployment = app.TestDeploymentConfig(
                    submissions_enabled=True,
                    public_leaderboard_enabled=False,
                    release_id=policy.release_id,
                    task_manifest_sha256=policy.task_manifest_sha256,
                    gold_sha256=policy.gold_sha256,
                    open_at=policy.open_at,
                    close_at=policy.close_at,
                    release_config_path="private/test_release.json",
                    gold_config_path="private/test_labels.jsonl",
                )
                with (
                    patch.object(app, "TEST_SUBMISSIONS_ENABLED", True),
                    patch.object(app, "WRITE_TOKEN", write_token),
                    patch.object(app, "TEST_DEPLOYMENT", deployment),
                    patch.object(
                        app,
                        "_SUBMISSION_SERVICE",
                        SimpleNamespace(
                            test_config_loader=lambda current: trusted_test_config(
                                policy
                            )
                        ),
                    ),
                    patch.object(app, "_server_now", return_value=now, create=True),
                ):
                    response = await self.invoke("select_split", [app.TEST_SPLIT_LABEL])

                updates = response["data"]
                self.assertEqual(updates[2]["interactive"], expected_open)
                expected_copy = (
                    "Test submissions are open."
                    if expected_open
                    else (
                        "Test submissions are closed."
                        if now >= policy.close_at
                        else "Test submissions are not open yet."
                    )
                )
                self.assertIn(expected_copy, updates[0]["value"])

    async def test_validation_endpoint_remains_anonymous_and_uses_legacy_metadata(self):
        captured = {}

        def validation_submitter(file_obj, metadata):
            captured.update(metadata)
            return {"split": "validation", "accepted": True}

        service, _ = configured_service(validation_submitter=validation_submitter)
        with patch.object(app, "_SUBMISSION_SERVICE", service):
            response = await self.invoke(
                "submit_predictions",
                [
                    app.VALIDATION_SPLIT_LABEL,
                    None,
                    "Team A",
                    "Alice Example",
                    "lead@example.org",
                    "baseline",
                ],
            )

        self.assertEqual(captured["contact"], "lead@example.org")
        self.assertEqual(response["data"][0]["value"].root["split"], "validation")

    async def test_signed_out_test_submit_and_history_require_contact_email(self):
        for api_name, inputs in (
            (
                "submit_predictions",
                [app.TEST_SPLIT_LABEL, None, "Team A", "Alice", "", "final"],
            ),
            ("my_test_submissions", [""]),
        ):
            with self.subTest(api_name=api_name):
                with self.assertRaisesRegex(Exception, "valid contact email"):
                    await self.invoke(api_name, inputs)

    async def test_history_endpoint_is_bound_to_injected_subject_and_masks_email(self):
        service, store = configured_service()
        with patch.object(app, "_SUBMISSION_SERVICE", service):
            response = await self.invoke(
                "my_test_submissions", ["ignored@example.org"], PROFILE_A
            )

        serialized = json.dumps(response["data"])
        self.assertEqual(store.requested_identities, [("huggingface", "subject-a")])
        self.assertIn("receipt-a1", serialized)
        self.assertIn("receipt-a2", serialized)
        self.assertIn("100.00%", serialized)
        self.assertIn("75.00%", serialized)
        self.assertIn("62.50%", serialized)
        self.assertIn("a***@example.org", serialized)
        self.assertIn("Score withheld until finalization", serialized)
        self.assertNotIn("receipt-b1", serialized)
        self.assertNotIn("subject-a", serialized)
        self.assertNotIn("subject-b", serialized)
        self.assertNotIn("alice@example.org", serialized)
        self.assertNotIn("25.00%", serialized)
        self.assertNotIn("50.00%", serialized)
        self.assertNotIn("secret-a", serialized)

    async def test_anonymous_history_endpoint_uses_normalized_contact_without_exposure(
        self,
    ):
        service, store = configured_service()
        with patch.object(app, "_SUBMISSION_SERVICE", service):
            response = await self.invoke(
                "my_test_submissions", [" Anonymous@Example.ORG "]
            )

        serialized = json.dumps(response["data"])
        self.assertEqual(
            store.requested_identities,
            [("email", "anonymous@example.org")],
        )
        self.assertIn("receipt-b1", serialized)
        self.assertNotIn("anonymous@example.org", serialized)

    def test_history_missing_joint_is_visibly_not_yet_computed(self):
        rendered = app._test_history_html(
            [
                {
                    "attempt": 1,
                    "receipt": "legacy-receipt",
                    "submission_name": "legacy",
                    "accepted_at": "2026-09-05T12:00:00Z",
                    "answer_accuracy": 0.75,
                    "evidence_f1": 0.625,
                }
            ],
            "l***@example.org",
        )

        self.assertIn("Joint Exact Accuracy Not yet computed", rendered)
        self.assertNotIn("Joint Exact Accuracy 0.00%", rendered)

    async def test_later_attempt_submit_update_serializes_no_metrics(self):
        class LaterAttemptService:
            def submit_for_split(self, split, file_obj, metadata, oauth_profile):
                self.profile = oauth_profile
                return {
                    "accepted": True,
                    "attempt": 2,
                    "receipt": "receipt-a2",
                    "score": "withheld",
                    "accepted_at": "2026-09-05T12:00:02Z",
                }

        service = LaterAttemptService()
        with patch.object(app, "_SUBMISSION_SERVICE", service):
            response = await self.invoke(
                "submit_predictions",
                [app.TEST_SPLIT_LABEL, None, "Team A", "Alice", "", "final"],
                PROFILE_A,
            )

        serialized = json.dumps(response["data"][0]["value"].root)
        self.assertEqual(service.profile.get("sub"), "subject-a")
        self.assertIn("receipt-a2", serialized)
        self.assertIn('"score": "withheld"', serialized)
        self.assertNotIn("answer_accuracy", serialized)
        self.assertNotIn("evidence_f1", serialized)
        self.assertNotIn("evidence_exact_match", serialized)
        self.assertNotIn("per_example", serialized)

    def test_disabled_final_leaderboard_is_notice_only_without_private_fetch(self):
        with (
            patch.object(app, "TEST_PUBLIC_LEADERBOARD_ENABLED", False),
            patch.object(app, "TEST_PROVISIONAL_LEADERBOARD_ENABLED", False),
            patch.object(
                app,
                "_load_final_test_projection",
                side_effect=AssertionError("disabled view must not fetch"),
                create=True,
            ),
        ):
            heading, content, refresh = app.leaderboard_view(
                app.FINAL_TEST_LEADERBOARD_LABEL
            )

        self.assertIn("Final test leaderboard", heading["value"])
        self.assertIn("not available yet", content["value"])
        self.assertNotIn("<table", content["value"].casefold())
        self.assertFalse(refresh["visible"])

    def test_provisional_loader_reads_only_release_and_rank_projection_at_one_head(
        self,
    ):
        hub = FinalLeaderboardHub()
        artifacts = provisional_artifacts()
        reads = []

        projection = app._load_provisional_test_projection(
            api=hub,
            artifact_reader=lambda path, revision: (
                reads.append((path, revision)) or artifacts[path]
            ),
            deployment=final_deployment(
                public_leaderboard_enabled=False,
                provisional_leaderboard_enabled=True,
            ),
            repo_id="private/docsem",
            token="private-token-sentinel",
            now=dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
        )

        self.assertEqual(
            projection["rows"],
            [{"rank": 1, "hf_username": "alice<script>", "team": "Team <Alpha>"}],
        )
        self.assertEqual(
            reads,
            [
                ("private/test_release.json", PRIVATE_HEAD),
                ("projections/test/public_provisional.json", PRIVATE_HEAD),
            ],
        )
        self.assertEqual(len(hub.calls), 1)

    def test_provisional_rank_table_has_no_metric_or_private_fields(self):
        projection = json.loads(
            provisional_artifacts()["projections/test/public_provisional.json"]
        )

        rendered = app.provisional_test_leaderboard_html(projection)

        self.assertIn("alice&lt;script&gt;", rendered)
        self.assertIn("Team &lt;Alpha&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        for forbidden in (
            "accuracy",
            "evidence",
            "score",
            "email",
            "identity_subject",
            "contact_email",
            "participant_names",
            "predictions",
            "submission_name",
        ):
            self.assertNotIn(forbidden, rendered.casefold())

    def test_provisional_loader_rejects_hidden_fields_and_rank_drift(self):
        base = provisional_artifacts()
        deployment = final_deployment(
            public_leaderboard_enabled=False,
            provisional_leaderboard_enabled=True,
        )
        for label, mutate in (
            (
                "hidden metric",
                lambda value: value["rows"][0].update({"joint_accuracy": 1.0}),
            ),
            ("private email", lambda value: value["rows"][0].update({"email": "x@y"})),
            ("rank drift", lambda value: value["rows"][0].update({"rank": 2})),
        ):
            with self.subTest(label=label):
                artifacts = dict(base)
                projection = json.loads(
                    artifacts["projections/test/public_provisional.json"]
                )
                mutate(projection)
                artifacts["projections/test/public_provisional.json"] = canonical_json(
                    projection
                )
                with self.assertRaises(app.FinalLeaderboardError):
                    app._load_provisional_test_projection(
                        api=FinalLeaderboardHub(),
                        artifact_reader=lambda path, revision, values=artifacts: values[
                            path
                        ],
                        deployment=deployment,
                        repo_id="private/docsem",
                        token="private-token-sentinel",
                        now=dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
                    )

    def test_provisional_view_renders_rank_only_until_final_flag_is_enabled(self):
        projection = json.loads(
            provisional_artifacts()["projections/test/public_provisional.json"]
        )
        with (
            patch.object(app, "TEST_PUBLIC_LEADERBOARD_ENABLED", False),
            patch.object(app, "TEST_PROVISIONAL_LEADERBOARD_ENABLED", True),
            patch.object(
                app, "_load_provisional_test_projection", return_value=projection
            ),
            patch.object(
                app,
                "_load_final_test_projection",
                side_effect=AssertionError("provisional view must not load final"),
            ),
        ):
            heading, content, refresh = app.leaderboard_view(
                app.FINAL_TEST_LEADERBOARD_LABEL
            )

        self.assertIn("Provisional test leaderboard", heading["value"])
        self.assertIn("alice&lt;script&gt;", content["value"])
        self.assertNotIn("62.50%", content["value"])
        self.assertTrue(refresh["visible"])

    def test_validation_leaderboard_view_preserves_legacy_rows(self):
        legacy_rows = [
            {
                "team": "Legacy Team",
                "submission_name": "baseline",
                "attempts": 2,
                "answer_accuracy": 0.625,
                "evidence_f1": 0.5,
                "submitted_at": "2026-09-03T12:00:00Z",
                "contact": "private@example.org",
            }
        ]
        with patch.object(app, "_load_leaderboard_rows", return_value=legacy_rows):
            heading, content, refresh = app.leaderboard_view(
                app.VALIDATION_LEADERBOARD_LABEL
            )

        self.assertIn("Validation leaderboard", heading["value"])
        self.assertIn("Legacy Team", content["value"])
        self.assertIn("62.50%", content["value"])
        self.assertIn("Not yet computed", content["value"])
        self.assertNotIn("private@example.org", content["value"])
        self.assertTrue(refresh["visible"])

    def test_finalized_loader_reads_only_three_artifacts_at_exact_private_head(self):
        hub = FinalLeaderboardHub()
        artifacts = finalized_artifacts()
        reads = []

        def reader(path, revision):
            reads.append((path, revision))
            return artifacts[path]

        deployment = final_deployment()
        projection = app._load_final_test_projection(
            api=hub,
            artifact_reader=reader,
            deployment=deployment,
            repo_id="private/docsem",
            token="private-token-sentinel",
        )

        self.assertEqual(projection["rows"][0]["selected_attempt"], 2)
        self.assertEqual(
            reads,
            [
                ("private/test_release.json", PRIVATE_HEAD),
                ("projections/test/public_final.json", PRIVATE_HEAD),
                ("private/test_finalization_audit.json", PRIVATE_HEAD),
            ],
        )
        self.assertEqual(len(hub.calls), 1)
        self.assertEqual(hub.calls[0]["repo_id"], "private/docsem")
        self.assertEqual(hub.calls[0]["repo_type"], "dataset")
        self.assertEqual(hub.calls[0]["token"], "private-token-sentinel")

    def test_final_loader_rejects_every_release_setting_mismatch_without_client_paths(
        self,
    ):
        cases = (
            (
                "future configured window",
                {
                    "open_at": dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc),
                },
                True,
            ),
            ("two-attempt deployment", {"max_attempts": 2}, False),
            (
                "different safe release path",
                {"release_config_path": "sealed/release.json"},
                False,
            ),
            (
                "different safe gold path",
                {"gold_config_path": "sealed/gold.jsonl"},
                False,
            ),
            (
                "different feedback policy",
                {"feedback_policy": "all-attempts"},
                False,
            ),
            (
                "different task path",
                {"task_manifest_path": "test/other-tasks.jsonl"},
                False,
            ),
        )

        for label, overrides, needs_release_read in cases:
            with self.subTest(label=label):
                hub = FinalLeaderboardHub()
                reads = []
                artifacts = finalized_artifacts()

                def reader(path, revision):
                    reads.append((path, revision))
                    return artifacts[path]

                with self.assertRaises(app.FinalLeaderboardError):
                    app._load_final_test_projection(
                        api=hub,
                        artifact_reader=reader,
                        deployment=final_deployment(**overrides),
                        repo_id="private/docsem",
                        token="private-token-sentinel",
                    )

                if needs_release_read:
                    self.assertEqual(len(hub.calls), 1)
                    self.assertEqual(
                        reads,
                        [
                            ("private/test_release.json", PRIVATE_HEAD),
                            ("projections/test/public_final.json", PRIVATE_HEAD),
                            ("private/test_finalization_audit.json", PRIVATE_HEAD),
                        ],
                    )
                else:
                    self.assertEqual(hub.calls, [])
                    self.assertEqual(reads, [])

    def test_final_test_table_escapes_rows_and_contains_only_public_fields(self):
        projection_bytes = finalized_artifacts()["projections/test/public_final.json"]
        projection = json.loads(projection_bytes)

        rendered = app.final_test_leaderboard_html(projection)

        self.assertIn("DocSem final test leaderboard", rendered)
        self.assertIn("alice&lt;script&gt;", rendered)
        self.assertIn("Team &lt;Alpha&gt;", rendered)
        self.assertIn("best &amp; final", rendered)
        self.assertIn("62.50%", rendered)
        self.assertIn("75.00%", rendered)
        self.assertIn("50.00%", rendered)
        self.assertNotIn("<script>", rendered)
        for private_name in (
            "email",
            "identity_subject",
            "contact_email",
            "participant_names",
            "predictions",
            "per_example",
        ):
            self.assertNotIn(private_name, rendered)

    def test_enabled_final_view_renders_only_after_verified_finalization(self):
        projection = json.loads(
            finalized_artifacts()["projections/test/public_final.json"]
        )
        with (
            patch.object(app, "TEST_PUBLIC_LEADERBOARD_ENABLED", True),
            patch.object(app, "_load_final_test_projection", return_value=projection),
        ):
            heading, content, refresh = app.leaderboard_view(
                app.FINAL_TEST_LEADERBOARD_LABEL
            )

        self.assertIn("Final test leaderboard", heading["value"])
        self.assertIn("<table", content["value"].casefold())
        self.assertIn("Team &lt;Alpha&gt;", content["value"])
        self.assertTrue(refresh["visible"])

    def test_final_loader_fails_closed_on_visibility_state_digest_or_schema_drift(self):
        base = finalized_artifacts()
        deployment = final_deployment()

        cases = []
        cases.append(("public repository", FinalLeaderboardHub(private=False), base))
        cases.append(("mutable revision", FinalLeaderboardHub(sha="main"), base))

        unfinalized = dict(base)
        release = json.loads(unfinalized["private/test_release.json"])
        release["finalized"] = False
        unfinalized["private/test_release.json"] = canonical_json(release)
        cases.append(("unfinalized release", FinalLeaderboardHub(), unfinalized))

        wrong_digest = dict(base)
        projection = json.loads(wrong_digest["projections/test/public_final.json"])
        projection["rows"][0]["answer_accuracy"] = 0.5
        wrong_digest["projections/test/public_final.json"] = canonical_json(projection)
        cases.append(("projection digest", FinalLeaderboardHub(), wrong_digest))

        extra_private_field = dict(base)
        projection = json.loads(
            extra_private_field["projections/test/public_final.json"]
        )
        projection["rows"][0]["contact_email"] = "secret@example.org"
        extra_private_field["projections/test/public_final.json"] = canonical_json(
            projection
        )
        cases.append(
            (
                "private row field",
                FinalLeaderboardHub(),
                resign_final_artifacts(extra_private_field),
            )
        )

        missing_joint = dict(base)
        projection = json.loads(missing_joint["projections/test/public_final.json"])
        projection["rows"][0].pop("joint_accuracy")
        missing_joint["projections/test/public_final.json"] = canonical_json(projection)
        cases.append(
            (
                "missing final joint metric",
                FinalLeaderboardHub(),
                resign_final_artifacts(missing_joint),
            )
        )

        invalid_audit_attempt = dict(base)
        audit = json.loads(
            invalid_audit_attempt["private/test_finalization_audit.json"]
        )
        audit["eligible_attempts"][0].pop("joint_accuracy")
        invalid_audit_attempt["private/test_finalization_audit.json"] = canonical_json(
            audit
        )
        cases.append(
            (
                "missing audit joint metric",
                FinalLeaderboardHub(),
                resign_final_artifacts(invalid_audit_attempt),
            )
        )

        for label, field, value in (
            ("non-contiguous rank", "rank", 2),
            ("attempt outside quota", "selected_attempt", 4),
            ("metric outside bounds", "answer_accuracy", 1.01),
            ("non-float metric", "evidence_f1", 1),
            ("unbounded public text", "team", "x" * 4097),
            ("control character", "submission_name", "bad\nname"),
        ):
            mutated = dict(base)
            projection = json.loads(mutated["projections/test/public_final.json"])
            projection["rows"][0][field] = value
            mutated["projections/test/public_final.json"] = canonical_json(projection)
            cases.append(
                (label, FinalLeaderboardHub(), resign_final_artifacts(mutated))
            )

        audit_count_mismatch = dict(base)
        audit = json.loads(audit_count_mismatch["private/test_finalization_audit.json"])
        audit["selected_account_count"] = 2
        audit_count_mismatch["private/test_finalization_audit.json"] = canonical_json(
            audit
        )
        release = json.loads(audit_count_mismatch["private/test_release.json"])
        release["finalization_audit_sha256"] = hashlib.sha256(
            audit_count_mismatch["private/test_finalization_audit.json"]
        ).hexdigest()
        audit_count_mismatch["private/test_release.json"] = canonical_json(release)
        cases.append(("audit row count", FinalLeaderboardHub(), audit_count_mismatch))

        for label, hub, artifacts in cases:
            with self.subTest(label=label):
                with self.assertRaises(app.FinalLeaderboardError):
                    app._load_final_test_projection(
                        api=hub,
                        artifact_reader=lambda path, revision, data=artifacts: data[
                            path
                        ],
                        deployment=deployment,
                        repo_id="private/docsem",
                        token="private-token-sentinel",
                    )

    def test_initial_config_has_leaderboard_selector_but_no_test_table_or_rows(self):
        serialized = json.dumps(app.demo.get_config_file())
        named_endpoints = {
            block_fn.api_name
            for block_fn in app.demo.fns.values()
            if block_fn.api_name is not False
        }

        self.assertIn(app.VALIDATION_LEADERBOARD_LABEL, serialized)
        self.assertIn(app.FINAL_TEST_LEADERBOARD_LABEL, serialized)
        self.assertNotIn("DocSem final test leaderboard", serialized)
        self.assertNotIn("alice&lt;script&gt;", serialized)
        self.assertNotIn("final_test_leaderboard", named_endpoints)
        self.assertNotIn("test_score", named_endpoints)
        self.assertNotIn("test_rank", named_endpoints)


if __name__ == "__main__":
    unittest.main()
