import datetime as dt
import unittest

from test_policy import (
    OAuthIdentity,
    TestPolicyError,
    TestReleasePolicy,
    account_key,
    canonical_submission_hash,
    participant_test_response,
    select_best_attempt,
)


METRICS = {
    "answer_accuracy": 0.812345,
    "evidence_f1": 0.654321,
    "evidence_exact_match": 0.5,
    "examples": 2,
    "per_example": [{"instance_id": "private", "answer_exact_match": 0.0}],
}

FIXTURE_ATTEMPTS = [
    {
        "submission_id": "later-id",
        "submitted_at": "2026-09-05T10:00:00Z",
        "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
    },
    {
        "submission_id": "expected-id",
        "submitted_at": "2026-09-05T09:00:00Z",
        "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
    },
    {
        "submission_id": "higher-answer",
        "submitted_at": "2026-09-05T08:00:00Z",
        "metrics": {"answer_accuracy": 0.89, "evidence_f1": 0.99},
    },
]


class TestPolicyTests(unittest.TestCase):
    def test_account_key_uses_stable_subject_not_email(self):
        first = OAuthIdentity(sub="stable-1", username="u", email="a@example.org")
        changed = OAuthIdentity(sub="stable-1", username="u2", email="b@example.org")

        self.assertEqual(account_key(first), account_key(changed))

    def test_missing_verified_email_is_rejected(self):
        with self.assertRaisesRegex(TestPolicyError, "verified email"):
            OAuthIdentity.from_profile({"sub": "s", "preferred_username": "u"})

    def test_disabled_or_closed_policy_rejects_before_scoring(self):
        policy = TestReleasePolicy.disabled()

        with self.assertRaisesRegex(TestPolicyError, "not open"):
            policy.require_open(now=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc))

    def test_profile_normalizes_email_and_rejects_unverified_email(self):
        identity = OAuthIdentity.from_profile(
            {
                "sub": " stable-1 ",
                "preferred_username": " participant ",
                "email": "A@Example.ORG",
                "email_verified": True,
            }
        )

        self.assertEqual(identity.sub, "stable-1")
        self.assertEqual(identity.username, "participant")
        self.assertEqual(identity.email, "a@example.org")
        with self.assertRaisesRegex(TestPolicyError, "verified email"):
            OAuthIdentity.from_profile(
                {
                    "sub": "s",
                    "preferred_username": "u",
                    "email": "a@example.org",
                    "email_verified": False,
                }
            )

    def test_active_policy_requires_complete_utc_window(self):
        policy = TestReleasePolicy(
            release_id="release-1",
            task_manifest_sha256="task-digest",
            gold_sha256="gold-digest",
            open_at=dt.datetime(2026, 9, 5, 8, tzinfo=dt.timezone.utc),
            close_at=dt.datetime(2026, 9, 6, 8, tzinfo=dt.timezone.utc),
            enabled=True,
        )

        self.assertTrue(
            policy.require_open(now=dt.datetime(2026, 9, 5, 9, tzinfo=dt.timezone.utc))
        )
        with self.assertRaisesRegex(TestPolicyError, "not open"):
            policy.require_open(now=dt.datetime(2026, 9, 6, 8, tzinfo=dt.timezone.utc))
        with self.assertRaisesRegex(TestPolicyError, "UTC"):
            policy.require_open(now=dt.datetime(2026, 9, 5, 9))

        with self.assertRaisesRegex(TestPolicyError, "configuration"):
            TestReleasePolicy(
                release_id="",
                task_manifest_sha256="task-digest",
                gold_sha256="gold-digest",
                open_at=dt.datetime(2026, 9, 5, 8, tzinfo=dt.timezone.utc),
                close_at=dt.datetime(2026, 9, 6, 8, tzinfo=dt.timezone.utc),
                enabled=True,
            )

    def test_canonical_hash_is_stable_for_payload_order_and_mapping_order(self):
        identity = OAuthIdentity(sub="stable-1", username="u", email="a@example.org")
        first = [
            {"instance_id": "two", "answer": " Final Answer: 42 ", "evidence": ["B2", "a1"]},
            {"instance_id": "one", "answer": "yes", "evidence": ["a"]},
        ]
        reordered = [
            {"evidence": ["a"], "answer": "yes", "instance_id": "one"},
            {"answer": "42", "instance_id": "two", "evidence": ["a1", "b2"]},
        ]

        self.assertEqual(
            canonical_submission_hash(first, "test", "r1", identity),
            canonical_submission_hash(reordered, "test", "r1", identity),
        )
        self.assertEqual(
            canonical_submission_hash(first, "test", "r1", identity),
            "9c1ee6bb4ed181c6de5014df91c7d899fe76c1d6ecc8cbe35b4d57f5ddeddd47",
        )
        self.assertNotEqual(
            canonical_submission_hash(first, "test", "r2", identity),
            canonical_submission_hash(first, "test", "r1", identity),
        )

    def test_attempt_one_feedback_has_only_public_aggregates(self):
        response = participant_test_response(1, METRICS, "receipt-1")

        self.assertEqual(
            set(response),
            {"accepted", "attempt", "receipt", "answer_accuracy", "evidence_f1"},
        )
        self.assertNotIn("per_example", response)

    def test_attempt_two_feedback_withholds_every_metric(self):
        response = participant_test_response(2, METRICS, "receipt-2")

        self.assertEqual(
            response,
            {"accepted": True, "attempt": 2, "receipt": "receipt-2", "score": "withheld"},
        )

    def test_attempt_three_feedback_withholds_every_metric(self):
        response = participant_test_response(3, METRICS, "receipt-3")

        self.assertEqual(response["score"], "withheld")
        self.assertNotIn("answer_accuracy", response)
        self.assertNotIn("evidence_f1", response)

    def test_best_attempt_uses_accuracy_f1_time_and_id(self):
        self.assertEqual(select_best_attempt(FIXTURE_ATTEMPTS)["submission_id"], "expected-id")

    def test_best_attempt_orders_aware_timestamps_by_utc_instant(self):
        attempts = [
            {
                "submission_id": "offset-earlier",
                "accepted_at": "2026-09-05T10:00:00+01:00",
                "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
            },
            {
                "submission_id": "utc-later",
                "accepted_at": "2026-09-05T09:30:00Z",
                "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
            },
        ]

        self.assertEqual(select_best_attempt(attempts)["submission_id"], "offset-earlier")

    def test_best_attempt_rejects_missing_timestamp(self):
        with self.assertRaisesRegex(TestPolicyError, "timestamp"):
            select_best_attempt(
                [
                    {
                        "submission_id": "missing-time",
                        "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
                    }
                ]
            )

    def test_best_attempt_rejects_malformed_timestamp(self):
        with self.assertRaisesRegex(TestPolicyError, "timestamp"):
            select_best_attempt(
                [
                    {
                        "submission_id": "malformed-time",
                        "accepted_at": "not-a-timestamp",
                        "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
                    }
                ]
            )

    def test_best_attempt_rejects_empty_attempts(self):
        with self.assertRaisesRegex(TestPolicyError, "attempt"):
            select_best_attempt([])


if __name__ == "__main__":
    unittest.main()
