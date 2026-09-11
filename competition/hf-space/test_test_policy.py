import datetime as dt
import os
import subprocess
import sys
import unittest

from test_policy import (
    TEST_ATTEMPT_COOLDOWN_SECONDS,
    TestIdentity,
    TestPolicyError,
    TestReleasePolicy,
    account_key,
    canonical_submission_hash,
    normalize_contact_email,
    next_eligible_at,
    participant_test_response,
    select_best_attempt,
)


METRICS = {
    "joint_accuracy": 0.54321,
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
        "metrics": {"joint_accuracy": 0.7, "answer_accuracy": 0.9, "evidence_f1": 0.8},
    },
    {
        "submission_id": "expected-id",
        "submitted_at": "2026-09-05T09:00:00Z",
        "metrics": {"joint_accuracy": 0.7, "answer_accuracy": 0.9, "evidence_f1": 0.8},
    },
    {
        "submission_id": "higher-answer",
        "submitted_at": "2026-09-05T08:00:00Z",
        "metrics": {
            "joint_accuracy": 0.6,
            "answer_accuracy": 0.99,
            "evidence_f1": 0.99,
        },
    },
]


class TestPolicyTests(unittest.TestCase):
    def test_operator_deadline_applies_to_actual_admission_and_closes_at_boundary(self):
        code = '''
import datetime as dt
from test_policy import TestReleasePolicy, OFFICIAL_TEST_CLOSE_AT
from test_store import _require_open, _ReleaseClosed
close = dt.datetime(2026, 9, 11, 15, 5, tzinfo=dt.timezone.utc)
assert OFFICIAL_TEST_CLOSE_AT == close
policy = TestReleasePolicy('release', 'a'*64, 'b'*64,
    dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc), close)
_require_open(policy, close - dt.timedelta(microseconds=1))
try:
    _require_open(policy, close)
except _ReleaseClosed:
    pass
else:
    raise AssertionError('deadline did not close admission')
'''
        result = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True,
            env={**os.environ, 'TEST_CLOSE_AT': '2026-09-11T15:05:00Z'},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_operator_deadline_cannot_extend_the_original_window(self):
        import test_policy
        from unittest.mock import patch
        for value in ('garbage', '2026-09-11T15:05:00', '2026-99-99T15:05:00Z'):
            with self.subTest(value=value), patch.dict(os.environ, {'TEST_CLOSE_AT': value}):
                self.assertEqual(test_policy._configured_test_close_at(),
                    dt.datetime(2026, 9, 11, 12, 0, tzinfo=dt.timezone.utc))

    def test_six_hour_cooldown_constant_and_next_eligible_timestamp_are_exact(self):
        self.assertEqual(TEST_ATTEMPT_COOLDOWN_SECONDS, 21_600)
        self.assertEqual(
            next_eligible_at(
                [
                    {
                        "submitted_at": "2026-09-05T10:00:00Z",
                        "submission_id": "first",
                    },
                    {
                        "submitted_at": "2026-09-05T18:30:00+01:00",
                        "submission_id": "second",
                    },
                ]
            ),
            "2026-09-05T23:30:00Z",
        )
        self.assertIsNone(next_eligible_at([]))

    def test_huggingface_identity_uses_stable_subject_and_verified_profile_email(self):
        first = TestIdentity.from_profile(
            {
                "sub": " stable-1 ",
                "preferred_username": " user-one ",
                "email": "A@Example.ORG",
                "email_verified": True,
            }
        )
        changed = TestIdentity.from_profile(
            {
                "sub": "stable-1",
                "preferred_username": "user-two",
                "email": "b@example.org",
                "email_verified": True,
            }
        )

        self.assertEqual(account_key(first), account_key(changed))
        self.assertEqual(
            account_key(first),
            "84b7a751be0df88d96101dcd5fa572beea884f895c2d3aa2bad3dfbf2e9a7a35",
        )
        self.assertEqual(
            first,
            TestIdentity(
                identity_kind="huggingface",
                identity_subject="stable-1",
                hf_username="user-one",
                contact_email="a@example.org",
                email_verified=True,
            ),
        )

    def test_email_identity_kind_is_rejected(self):
        with self.assertRaisesRegex(TestPolicyError, "identity is invalid"):
            TestIdentity(
                identity_kind="email",
                identity_subject="person+paper@example.org",
                hf_username="Not signed in",
                contact_email="person+paper@example.org",
                email_verified=False,
            )

    def test_verified_profile_email_normalization_uses_conservative_syntax(self):
        self.assertEqual(normalize_contact_email(" A@Example.org "), "a@example.org")
        for value in (
            "missing-at.example.org",
            "two@@example.org",
            ".lead@example.org",
            "trail.@example.org",
            "two..dots@example.org",
            "a@localhost",
            "a@-example.org",
            "a@example-.org",
            "a@exa_mple.org",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TestPolicyError, "valid contact email"):
                    normalize_contact_email(value)

    def test_missing_verified_email_is_rejected(self):
        with self.assertRaisesRegex(TestPolicyError, "verified email"):
            TestIdentity.from_profile({"sub": "s", "preferred_username": "u"})

    def test_disabled_or_closed_policy_rejects_before_scoring(self):
        policy = TestReleasePolicy.disabled()

        with self.assertRaisesRegex(TestPolicyError, "not open"):
            policy.require_open(now=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc))

    def test_profile_normalizes_email_and_rejects_unverified_email(self):
        identity = TestIdentity.from_profile(
            {
                "sub": " stable-1 ",
                "preferred_username": " participant ",
                "email": "A@Example.ORG",
                "email_verified": True,
            }
        )

        self.assertEqual(identity.identity_subject, "stable-1")
        self.assertEqual(identity.hf_username, "participant")
        self.assertEqual(identity.contact_email, "a@example.org")
        with self.assertRaisesRegex(TestPolicyError, "verified email"):
            TestIdentity.from_profile(
                {
                    "sub": "s",
                    "preferred_username": "u",
                    "email": "a@example.org",
                    "email_verified": False,
                }
            )

    def test_profile_requires_email_verified_to_be_explicitly_true(self):
        for profile in (
            {
                "sub": "s",
                "preferred_username": "u",
                "email": "a@example.org",
            },
            {
                "sub": "s",
                "preferred_username": "u",
                "email": "a@example.org",
                "email_verified": "true",
            },
        ):
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(TestPolicyError, "verified email"):
                    TestIdentity.from_profile(profile)

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
        identity = TestIdentity.from_profile(
            {
                "sub": "stable-1",
                "preferred_username": "u",
                "email": "a@example.org",
                "email_verified": True,
            }
        )
        metadata = {
            "team": "Team One",
            "participant_names": "Alice Example",
            "submission_name": "Run One",
        }
        first = [
            {
                "instance_id": "two",
                "answer": " Final Answer: 42 ",
                "evidence": ["B2", "a1"],
            },
            {"instance_id": "one", "answer": "yes", "evidence": ["a"]},
        ]
        reordered = [
            {"evidence": ["a"], "answer": "yes", "instance_id": "one"},
            {"answer": "42", "instance_id": "two", "evidence": ["a1", "b2"]},
        ]

        self.assertEqual(
            canonical_submission_hash(first, "test", "r1", identity, metadata),
            canonical_submission_hash(reordered, "test", "r1", identity, metadata),
        )
        self.assertEqual(
            canonical_submission_hash(first, "test", "r1", identity, metadata),
            "6d847b5dca10df9ce7a976d9454ef2253d69c277df4338e4edcf04c56113726c",
        )
        self.assertNotEqual(
            canonical_submission_hash(first, "test", "r2", identity, metadata),
            canonical_submission_hash(first, "test", "r1", identity, metadata),
        )
        self.assertNotEqual(
            canonical_submission_hash(first, "test", "r1", identity, metadata),
            canonical_submission_hash(
                first,
                "test",
                "r1",
                identity,
                {**metadata, "submission_name": "Run Two"},
            ),
        )

    def test_attempt_one_feedback_has_only_public_aggregates(self):
        response = participant_test_response(1, METRICS, "receipt-1")

        self.assertEqual(
            set(response),
            {
                "accepted",
                "attempt",
                "receipt",
                "joint_accuracy",
                "answer_accuracy",
                "evidence_f1",
            },
        )
        self.assertNotIn("per_example", response)

    def test_canonical_hash_keeps_null_abstention_distinct_from_string_none(self):
        identity = TestIdentity.from_profile(
            {
                "sub": "stable-1",
                "preferred_username": "user",
                "email": "user@example.org",
                "email_verified": True,
            }
        )
        metadata = {
            "team": "Team",
            "participant_names": "Alice",
            "submission_name": "Run",
        }
        null_rows = [{"instance_id": "one", "answer": None, "evidence": []}]
        text_rows = [{"instance_id": "one", "answer": "none", "evidence": []}]

        self.assertNotEqual(
            canonical_submission_hash(null_rows, "test", "release", identity, metadata),
            canonical_submission_hash(text_rows, "test", "release", identity, metadata),
        )

    def test_attempt_two_feedback_withholds_every_metric(self):
        response = participant_test_response(2, METRICS, "receipt-2")

        self.assertEqual(
            response,
            {
                "accepted": True,
                "attempt": 2,
                "receipt": "receipt-2",
                "score": "withheld",
            },
        )

    def test_attempt_three_feedback_withholds_every_metric(self):
        response = participant_test_response(3, METRICS, "receipt-3")

        self.assertEqual(response["score"], "withheld")
        self.assertNotIn("answer_accuracy", response)
        self.assertNotIn("evidence_f1", response)

    def test_best_attempt_uses_joint_accuracy_then_answer_f1_time_and_id(self):
        self.assertEqual(
            select_best_attempt(FIXTURE_ATTEMPTS)["submission_id"], "expected-id"
        )

    def test_best_attempt_uses_submission_id_as_last_tie_breaker(self):
        attempts = [
            {
                "submission_id": submission_id,
                "submitted_at": "2026-09-05T09:00:00Z",
                "metrics": {
                    "joint_accuracy": 0.7,
                    "answer_accuracy": 0.9,
                    "evidence_f1": 0.8,
                },
            }
            for submission_id in ("z-id", "a-id")
        ]

        self.assertEqual(select_best_attempt(attempts)["submission_id"], "a-id")

    def test_best_attempt_orders_aware_timestamps_by_utc_instant(self):
        attempts = [
            {
                "submission_id": "offset-earlier",
                "accepted_at": "2026-09-05T10:00:00+01:00",
                "metrics": {
                    "joint_accuracy": 0.7,
                    "answer_accuracy": 0.9,
                    "evidence_f1": 0.8,
                },
            },
            {
                "submission_id": "utc-later",
                "accepted_at": "2026-09-05T09:30:00Z",
                "metrics": {
                    "joint_accuracy": 0.7,
                    "answer_accuracy": 0.9,
                    "evidence_f1": 0.8,
                },
            },
        ]

        self.assertEqual(
            select_best_attempt(attempts)["submission_id"], "offset-earlier"
        )

    def test_best_attempt_rejects_missing_timestamp(self):
        with self.assertRaisesRegex(TestPolicyError, "timestamp"):
            select_best_attempt(
                [
                    {
                        "submission_id": "missing-time",
                        "metrics": {
                            "joint_accuracy": 0.7,
                            "answer_accuracy": 0.9,
                            "evidence_f1": 0.8,
                        },
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
                        "metrics": {
                            "joint_accuracy": 0.7,
                            "answer_accuracy": 0.9,
                            "evidence_f1": 0.8,
                        },
                    }
                ]
            )

    def test_best_attempt_rejects_empty_attempts(self):
        with self.assertRaisesRegex(TestPolicyError, "attempt"):
            select_best_attempt([])


if __name__ == "__main__":
    unittest.main()
