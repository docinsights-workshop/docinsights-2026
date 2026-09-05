import datetime as dt
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

import app
from submission_service import SubmissionService


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


def canonical_json(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def finalized_artifacts(*, rows=None):
    projection = {
        "schema_version": 1,
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
        "finalized_at": "2026-09-11T00:00:00Z",
        "close_at": "2026-09-10T00:00:00Z",
        "task_manifest_sha256": TASK_DIGEST,
        "gold_sha256": GOLD_DIGEST,
        "scorer_revision": SCORER_REVISION,
        "scorer_code_sha256": SCORER_DIGEST,
        "public_projection_sha256": projection_digest,
        "eligible_attempt_count": 3,
        "excluded_attempt_count": 0,
        "selected_account_count": len(projection["rows"]),
        "input_manifest_sha256": "1" * 64,
        "input_records": [],
        "projection_issue_codes": [],
        "eligible_attempts": [],
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
        "close_at": "2026-09-10T00:00:00Z",
        "public_revision": "2" * 40,
        "public_repo_id": "public/docsem",
        "task_manifest_path": "test/tasks.jsonl",
        "finalized_at": "2026-09-11T00:00:00Z",
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
        self.requested_subjects = []

    def account_history(self, identity):
        self.requested_subjects.append(identity.sub)
        if identity.sub == "subject-a":
            return [
                {
                    "attempt_number": 1,
                    "submission_id": "receipt-a1",
                    "submission_name": "alice-first",
                    "submitted_at": "2026-09-05T12:00:01Z",
                    "metrics": {
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
                "metrics": {"answer_accuracy": 0.125, "evidence_f1": 0.25},
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
        "close_at": dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc),
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
        self.assertIn("Sign in with Hugging Face", test_updates[0]["value"])
        self.assertIn("Test submissions are not open yet", test_updates[0]["value"])
        self.assertFalse(test_updates[1]["visible"])
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

    async def test_test_ui_requires_write_token_and_current_open_server_window(self):
        policy = app.TestReleasePolicy(
            release_id="configured-release",
            task_manifest_sha256="a" * 64,
            gold_sha256="b" * 64,
            open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
            close_at=dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc),
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
                with (
                    patch.object(app, "TEST_SUBMISSIONS_ENABLED", True),
                    patch.object(app, "WRITE_TOKEN", write_token),
                    patch.object(
                        app,
                        "TEST_DEPLOYMENT",
                        SimpleNamespace(expected_policy=policy),
                    ),
                    patch.object(app, "_server_now", return_value=now, create=True),
                ):
                    response = await self.invoke("select_split", [app.TEST_SPLIT_LABEL])

                updates = response["data"]
                self.assertEqual(updates[2]["interactive"], expected_open)
                self.assertIn(
                    "Test submissions are open."
                    if expected_open
                    else "Test submissions are not open yet.",
                    updates[0]["value"],
                )

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

    async def test_unauthenticated_test_submit_and_history_fail_closed(self):
        for api_name, inputs in (
            (
                "submit_predictions",
                [app.TEST_SPLIT_LABEL, None, "Team A", "Alice", "", "final"],
            ),
            ("my_test_submissions", []),
        ):
            with self.subTest(api_name=api_name):
                with self.assertRaisesRegex(Exception, "Sign in with Hugging Face"):
                    await self.invoke(api_name, inputs)

    async def test_history_endpoint_is_bound_to_injected_subject_and_masks_email(self):
        service, store = configured_service()
        with patch.object(app, "_SUBMISSION_SERVICE", service):
            response = await self.invoke("my_test_submissions", [], PROFILE_A)

        serialized = json.dumps(response["data"])
        self.assertEqual(store.requested_subjects, ["subject-a"])
        self.assertIn("receipt-a1", serialized)
        self.assertIn("receipt-a2", serialized)
        self.assertIn("100.00%", serialized)
        self.assertIn("75.00%", serialized)
        self.assertIn("a***@example.org", serialized)
        self.assertIn("Score withheld until finalization", serialized)
        self.assertNotIn("receipt-b1", serialized)
        self.assertNotIn("subject-a", serialized)
        self.assertNotIn("subject-b", serialized)
        self.assertNotIn("alice@example.org", serialized)
        self.assertNotIn("25.00%", serialized)
        self.assertNotIn("50.00%", serialized)
        self.assertNotIn("secret-a", serialized)

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
                    "open_at": dt.datetime(2026, 9, 12, tzinfo=dt.timezone.utc),
                    "close_at": dt.datetime(2026, 9, 17, tzinfo=dt.timezone.utc),
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
        self.assertIn("75.00%", rendered)
        self.assertIn("50.00%", rendered)
        self.assertNotIn("<script>", rendered)
        for private_name in (
            "email",
            "hf_subject",
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
        projection["rows"][0]["verified_email"] = "secret@example.org"
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
