import datetime as dt
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


if __name__ == "__main__":
    unittest.main()
