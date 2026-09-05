"""Deployment-copy parity checks for the organizer's local ledger contract."""

from __future__ import annotations

import importlib
import math
import sys
import unittest
from pathlib import Path


SPACE_ROOT = Path(__file__).resolve().parent
PARTICIPANT_ROOT = SPACE_ROOT.parent / "hf-space"
if str(PARTICIPANT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARTICIPANT_ROOT))

import scoring as participant_scoring  # noqa: E402
import test_contract as participant_contract  # noqa: E402
import test_policy as participant_policy  # noqa: E402


def _outcome(callable_, *args):
    try:
        return ("ok", callable_(*args))
    except ValueError as exc:
        return ("error", str(exc))


class OrganizerContractParityTests(unittest.TestCase):
    def organizer_contract(self):
        module_path = SPACE_ROOT / "organizer_contract.py"
        self.assertTrue(
            module_path.is_file(),
            "organizer Space bundle lacks its package-local ledger contract",
        )
        importlib.invalidate_caches()
        return importlib.import_module("organizer_contract")

    def test_shared_limits_and_validation_boundaries_match_participant_contract(self):
        """Catches organizer/participant resource-limit or schema drift."""

        organizer = self.organizer_contract()
        names = (
            "TEST_PREDICTION_KEYS",
            "MAX_TEST_ROWS",
            "MAX_INSTANCE_ID_CHARACTERS",
            "MAX_ANSWER_CHARACTERS",
            "MAX_EVIDENCE_IDS",
            "MAX_EVIDENCE_ID_CHARACTERS",
            "MAX_PRIVATE_TEXT_CHARACTERS",
            "MAX_PARTICIPANT_NAMES_CHARACTERS",
            "MAX_REPOSITORY_ID_CHARACTERS",
            "MAX_LEDGER_FILE_BYTES",
            "PRIVATE_TEXT_LIMITS",
            "PUBLIC_TEXT_FIELDS",
            "ADJUDICATION_ACTIONS",
            "ACCOUNT_ADJUDICATION_ACTIONS",
            "ATTEMPT_ADJUDICATION_ACTIONS",
        )
        self.assertEqual(
            {name: getattr(organizer, name) for name in names},
            {name: getattr(participant_contract, name) for name in names},
        )

        private_text_cases = (
            ("r" * 4_096, "release_id"),
            ("r" * 4_097, "release_id"),
            ("p" * 500, "participant_names"),
            ("p" * 501, "participant_names"),
            (" ", "team"),
        )
        for value, field in private_text_cases:
            with self.subTest(kind="private-text", field=field, size=len(value)):
                self.assertEqual(
                    _outcome(organizer.bounded_private_text, value, field),
                    _outcome(participant_contract.bounded_private_text, value, field),
                )

        repository_cases = (
            "a" * 127 + "/" + "b" * 128,
            "a" * 128 + "/" + "b" * 128,
            "owner/repo",
            "owner/not/a/repo",
        )
        for repo_id in repository_cases:
            with self.subTest(kind="repository", size=len(repo_id)):
                self.assertEqual(
                    _outcome(organizer.repository_id, repo_id),
                    _outcome(participant_contract.repository_id, repo_id),
                )

        boundary = [
            {
                "instance_id": "i" * 256,
                "answer": "a" * 4_096,
                "evidence": ["e" * 256] * 128,
            }
        ]
        invalid_rows = (
            [{**boundary[0], "extra": "no"}],
            [{**boundary[0], "instance_id": "i" * 257}],
            [{**boundary[0], "answer": "a" * 4_097}],
            [{**boundary[0], "evidence": ["e"] * 129}],
            [{**boundary[0], "evidence": ["e" * 257]}],
        )
        for rows in (boundary, *invalid_rows):
            with self.subTest(kind="predictions", shape=tuple(rows[0])):
                self.assertEqual(
                    _outcome(organizer.validate_test_predictions, rows),
                    _outcome(participant_contract.validate_test_predictions, rows),
                )

    def test_private_and_public_text_safety_match_participant_contract(self):
        """Catches controls or private paths accepted by only one deployed copy."""

        organizer = self.organizer_contract()
        cases = (
            "safe display name",
            " leading and trailing ",
            "line\nbreak",
            "tab\tspoof",
            "nul\0spoof",
            "delete\x7fspoof",
            "private/test_labels.jsonl",
            "attempts/test/account/record.json",
            "x" * 4_097,
        )
        for value in cases:
            with self.subTest(value=repr(value[:40])):
                self.assertEqual(
                    organizer.is_valid_public_text(value),
                    participant_contract.is_valid_public_text(value),
                )
        for field in participant_contract.PRIVATE_TEXT_LIMITS:
            with self.subTest(field=field):
                value = "safe\nspoof"
                self.assertEqual(
                    _outcome(organizer.bounded_private_text, value, field),
                    _outcome(participant_contract.bounded_private_text, value, field),
                )
                self.assertEqual(
                    _outcome(organizer.bounded_private_text, value, field)[0],
                    "error",
                )

    def test_adjudication_target_and_ordered_state_match_participant_contract(self):
        """Catches organizer/finalizer drift in audit-action state transitions."""

        organizer = self.organizer_contract()
        attempt_id = "11111111-1111-4111-8111-111111111111"
        cases = (
            ("note", None),
            ("note", attempt_id),
            ("exclude_account", None),
            ("reinstate_account", None),
            ("exclude_attempt", attempt_id),
            ("reinstate_attempt", attempt_id),
            ("unknown", None),
            ([], None),
            ("exclude_account", attempt_id),
            ("exclude_attempt", None),
            ("note", "not-a-uuid"),
        )
        for action, submission_id in cases:
            with self.subTest(action=action, submission_id=submission_id):
                self.assertEqual(
                    _outcome(
                        organizer.validate_adjudication_target,
                        action,
                        submission_id,
                    ),
                    _outcome(
                        participant_contract.validate_adjudication_target,
                        action,
                        submission_id,
                    ),
                )

        events = (
            {
                "record_id": "z-exclude",
                "created_at": "2026-10-01T02:00:00Z",
                "action": "exclude_attempt",
                "account_key": "account-a",
                "submission_id": attempt_id,
            },
            {
                "record_id": "a-reinstate",
                "created_at": "2026-10-01T02:00:00Z",
                "action": "reinstate_attempt",
                "account_key": "account-a",
                "submission_id": attempt_id,
            },
            {
                "record_id": "account-exclude",
                "created_at": "2026-10-01T01:00:00Z",
                "action": "exclude_account",
                "account_key": "account-a",
                "submission_id": None,
            },
            {
                "record_id": "account-reinstate",
                "created_at": "2026-10-01T01:30:00Z",
                "action": "reinstate_account",
                "account_key": "account-a",
                "submission_id": None,
            },
        )
        organizer_state = organizer.ordered_decision_state(events)
        participant_state = participant_contract.ordered_decision_state(events)
        self.assertEqual(organizer_state, participant_state)
        self.assertEqual(organizer_state[0], frozenset())
        self.assertEqual(organizer_state[1], frozenset({("account-a", attempt_id)}))
        self.assertEqual(
            organizer_state[2],
            (
                "account-exclude",
                "account-reinstate",
                "a-reinstate",
                "z-exclude",
            ),
        )

    def test_normalization_and_canonical_submission_hash_match_participant_policy(self):
        """Catches a standalone organizer accepting a different immutable hash."""

        organizer = self.organizer_contract()
        normalization_cases = (
            (" Final Answer:  0042  ", "0042"),
            ("ANSWER:\nHello   WORLD", "hello world"),
            (42, "42"),
        )
        for raw, expected in normalization_cases:
            with self.subTest(raw=raw):
                self.assertEqual(organizer.normalize_answer(raw), expected)
                self.assertEqual(
                    organizer.normalize_answer(raw),
                    participant_scoring.normalize_answer(raw),
                )

        predictions = [
            {
                "instance_id": " task-b ",
                "answer": "Final Answer:  0042  ",
                "evidence": [" B02 ", "b01", "B02"],
            },
            {
                "evidence": ["X2", "x1"],
                "answer": "ANSWER: Hello   WORLD",
                "instance_id": "task-a",
            },
        ]
        organizer_identity = organizer.OAuthIdentity(
            "subject-1", "user", "u@example.org"
        )
        participant_identity = participant_policy.OAuthIdentity(
            "subject-1", "user", "u@example.org"
        )
        organizer_hash = organizer.canonical_submission_hash(
            predictions, " Test ", " release-α ", organizer_identity
        )
        participant_hash = participant_policy.canonical_submission_hash(
            predictions, " Test ", " release-α ", participant_identity
        )
        self.assertEqual(
            organizer_hash,
            "9cbe8b01a046be6ffd52883e979686ab4e0e7598df79ec801a65ecd4af947bf8",
        )
        self.assertEqual(organizer_hash, participant_hash)

    def test_best_attempt_metric_and_timestamp_boundaries_match_participant_policy(
        self,
    ):
        """Catches ranking drift in score, UTC instant, and submission-id ties."""

        organizer = self.organizer_contract()
        attempts = [
            {
                "submission_id": "b",
                "submitted_at": "2026-09-05T13:00:00+01:00",
                "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
            },
            {
                "submission_id": "a",
                "submitted_at": "2026-09-05T12:00:00Z",
                "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.8},
            },
            {
                "submission_id": "later-better-evidence",
                "submitted_at": "2026-09-07T12:00:00Z",
                "metrics": {"answer_accuracy": 0.9, "evidence_f1": 0.9},
            },
        ]
        self.assertEqual(
            organizer.select_best_attempt(attempts)["submission_id"],
            "later-better-evidence",
        )
        self.assertEqual(
            organizer.select_best_attempt(attempts)["submission_id"],
            participant_policy.select_best_attempt(attempts)["submission_id"],
        )
        tied = attempts[:2]
        self.assertEqual(
            organizer.select_best_attempt(tied)["submission_id"],
            "a",
        )
        self.assertEqual(
            organizer.select_best_attempt(tied)["submission_id"],
            participant_policy.select_best_attempt(tied)["submission_id"],
        )

        invalid_attempt_sets = (
            [
                {
                    "submission_id": "missing-time",
                    "metrics": {"answer_accuracy": 1.0, "evidence_f1": 1.0},
                }
            ],
            [
                {
                    "submission_id": "naive-time",
                    "submitted_at": "2026-09-05T12:00:00",
                    "metrics": {"answer_accuracy": 1.0, "evidence_f1": 1.0},
                }
            ],
            [
                {
                    "submission_id": "nan-score",
                    "submitted_at": "2026-09-05T12:00:00Z",
                    "metrics": {"answer_accuracy": math.nan, "evidence_f1": 1.0},
                }
            ],
        )
        for invalid in invalid_attempt_sets:
            with self.subTest(submission_id=invalid[0]["submission_id"]):
                self.assertEqual(
                    _outcome(organizer.select_best_attempt, invalid),
                    _outcome(participant_policy.select_best_attempt, invalid),
                )


if __name__ == "__main__":
    unittest.main()
