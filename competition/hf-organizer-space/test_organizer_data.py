import datetime as dt
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub.errors import EntryNotFoundError

from organizer_data import (
    ADJUDICATION_PREFIX,
    EXCLUSION_PREFIX,
    MAX_FILE_BYTES,
    MAX_SNAPSHOT_FILES,
    OrganizerDataError,
    load_snapshot,
    organizer_rows,
    verify_snapshot,
)

# Participant-producer integration tests intentionally load the producer from
# the repository.  Production organizer modules must remain package-local.
_PARTICIPANT_SPACE = Path(__file__).resolve().parents[1] / "hf-space"
if str(_PARTICIPANT_SPACE) not in sys.path:
    sys.path.insert(0, str(_PARTICIPANT_SPACE))

from scoring import score_predictions  # noqa: E402
import test_contract  # noqa: E402
from test_policy import OAuthIdentity, account_key  # noqa: E402
from test_store import HubTestStore, TestStoreError  # noqa: E402


REVISION = "f" * 40
TASK_DIGEST = "1" * 64
GOLD_DIGEST = "2" * 64
ACCOUNT_A = hashlib.sha256(b"subject-a").hexdigest()
ACCOUNT_B = hashlib.sha256(b"subject-b").hexdigest()
PRIVATE_TOKEN = "organizer-read-token-sentinel"
HASH_A1 = "b6f5a9d8a5677110b6258d277266f894c07a1f77b101b025b07cd5d1f479df7d"
HASH_A2 = "a206eb6e77e5293c16ec434605c724e05507cca91a1a11647bff797877f39931"
HASH_B1 = "1eb7b1f4aff10fe6baed1bef45188b49ae6d8688021ea41a484efdfa36ba3cd5"
ID_A1 = "11111111-1111-4111-8111-111111111111"
ID_A2 = "22222222-2222-4222-8222-222222222222"
ID_B1 = "33333333-3333-4333-8333-333333333333"
RELEASE_STATE = {
    "schema_version": 2,
    "split": "test",
    "release_id": "docsem-test-2026",
    "task_manifest_sha256": TASK_DIGEST,
    "gold_sha256": GOLD_DIGEST,
}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def attempt(
    account,
    number,
    submission_id,
    submission_hash,
    answer,
    evidence,
    *,
    team="Shared Team",
):
    label = f"a-{number}" if account == ACCOUNT_A else f"b-{number}"
    return {
        **RELEASE_STATE,
        "submission_id": submission_id,
        "account_key": account,
        "hf_subject": "subject-a" if account == ACCOUNT_A else "subject-b",
        "hf_username": "user-a" if account == ACCOUNT_A else "user-b",
        "verified_email": "user-a@example.org"
        if account == ACCOUNT_A
        else "user-b@example.org",
        "scoring_gold_sha256": GOLD_DIGEST,
        "scoring_private_revision": "3" * 40,
        "scoring_public_revision": "4" * 40,
        "scoring_public_repo_id": "public/docsem",
        "scoring_task_manifest_path": "test/tasks.jsonl",
        "team": team,
        "participant_names": "Participant A"
        if account == ACCOUNT_A
        else "Participant B",
        "submission_name": f"run-{'a' if account == ACCOUNT_A else 'b'}-{number}",
        "submitted_at": f"2026-09-0{number}T12:00:00Z",
        "submission_hash": submission_hash,
        "attempt_number": number,
        "metrics": {
            "answer_accuracy": answer,
            "evidence_f1": evidence,
            "evidence_exact_match": 1.0 if evidence > 0.0 else 0.0,
            "examples": 2,
            "per_example": [
                {
                    "instance_id": "task-1",
                    "answer_exact_match": answer,
                    "evidence_exact_match": 1.0 if evidence > 0.0 else 0.0,
                    "evidence_f1": evidence,
                },
                {
                    "instance_id": "task-2",
                    "answer_exact_match": answer,
                    "evidence_exact_match": 1.0 if evidence > 0.0 else 0.0,
                    "evidence_f1": evidence,
                },
            ],
        },
        "predictions": [
            {
                "instance_id": "task-1",
                "answer": f"raw-private-prediction-sentinel-{label}",
                "evidence": ["block-1"],
            },
            {
                "instance_id": "task-2",
                "answer": f"another-private-prediction-{label}",
                "evidence": ["block-2"],
            },
        ],
    }


def account_projection(account, attempts):
    best = min(
        attempts,
        key=lambda item: (
            -item["metrics"]["answer_accuracy"],
            -item["metrics"]["evidence_f1"],
            item["submitted_at"],
            item["submission_id"],
        ),
    )
    return {
        **RELEASE_STATE,
        "account_key": account,
        "attempts": [
            {
                **RELEASE_STATE,
                "submission_id": item["submission_id"],
                "attempt_number": item["attempt_number"],
                "record_sha256": hashlib.sha256(json_bytes(item)).hexdigest(),
            }
            for item in attempts
        ],
        "best_submission_id": best["submission_id"],
    }


def organizer_projection(grouped):
    accounts = []
    for account, attempts in sorted(grouped.items()):
        best = min(
            attempts,
            key=lambda item: (
                -item["metrics"]["answer_accuracy"],
                -item["metrics"]["evidence_f1"],
                item["submitted_at"],
                item["submission_id"],
            ),
        )
        accounts.append(
            {
                **RELEASE_STATE,
                "account_key": account,
                "attempt_count": len(attempts),
                "best_submission_id": best["submission_id"],
                "hf_subject": best["hf_subject"],
                "hf_username": best["hf_username"],
                "verified_email": best["verified_email"],
                "team": best["team"],
                "participant_names": best["participant_names"],
                "submission_name": best["submission_name"],
                "submitted_at": best["submitted_at"],
                "attempt_number": best["attempt_number"],
                "metrics": best["metrics"],
            }
        )
    return {**RELEASE_STATE, "accounts": accounts}


def fixture_files():
    a1 = attempt(ACCOUNT_A, 1, ID_A1, HASH_A1, 0.0, 0.80)
    a2 = attempt(ACCOUNT_A, 2, ID_A2, HASH_A2, 1.0, 0.60)
    b1 = attempt(ACCOUNT_B, 1, ID_B1, HASH_B1, 1.0, 0.90)
    grouped = {ACCOUNT_A: [a1, a2], ACCOUNT_B: [b1]}
    release = {
        "schema_version": 1,
        "release_id": RELEASE_STATE["release_id"],
        "task_manifest_sha256": TASK_DIGEST,
        "gold_sha256": GOLD_DIGEST,
        "enabled": True,
        "finalized": False,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "open_at": "2026-09-01T00:00:00Z",
        "close_at": "2026-10-01T00:00:00Z",
        "public_revision": "4" * 40,
        "public_repo_id": "public/docsem",
        "task_manifest_path": "test/tasks.jsonl",
    }
    files = {
        "private/test_release.json": json_bytes(release),
        "private/test_labels.jsonl": b'{"answer":"gold-must-never-be-read"}\n',
        "leaderboard/leaderboard.json": b'{"legacy_validation":"must-be-ignored"}\n',
        "submissions/legacy.json": b'{"split":"validation"}\n',
        "projections/test/organizer_leaderboard.json": json_bytes(
            organizer_projection(grouped)
        ),
    }
    for account, attempts in grouped.items():
        files[f"projections/test/accounts/{account}.json"] = json_bytes(
            account_projection(account, attempts)
        )
        for item in attempts:
            files[f"attempts/test/{account}/{item['submission_id']}.json"] = json_bytes(
                item
            )
    files[f"{EXCLUSION_PREFIX}exclude-smoke.json"] = json_bytes(
        {
            **RELEASE_STATE,
            "record_id": "exclude-smoke",
            "account_key": ACCOUNT_A,
            "created_at": "2026-09-04T12:00:00Z",
            "reason_code": "organizer-smoke-test",
        }
    )
    files[f"{ADJUDICATION_PREFIX}appeal-reviewed.json"] = json_bytes(
        {
            **RELEASE_STATE,
            "record_id": "appeal-reviewed",
            "account_key": ACCOUNT_A,
            "submission_id": ID_A2,
            "created_at": "2026-09-05T12:00:00Z",
            "action": "note",
            "reason_code": "technical-review-complete",
        }
    )
    return files


class FakeHub:
    def __init__(self, files=None, *, revision=REVISION, private=True):
        self.files = dict(files or fixture_files())
        self.revision = revision
        self.private = private
        self.commits = (REVISION, "3" * 40)
        self.forbidden_downloads = {
            "private/test_labels.jsonl",
            "leaderboard/leaderboard.json",
            "submissions/legacy.json",
        }

    def repo_info(self, repo_id, *, repo_type, revision, token):
        if revision != REVISION or token != PRIVATE_TOKEN:
            raise RuntimeError("unpinned or unauthorized")
        return SimpleNamespace(sha=self.revision, private=self.private)

    def list_repo_files(self, repo_id, *, repo_type, revision, token):
        if revision != REVISION or token != PRIVATE_TOKEN:
            raise RuntimeError("unpinned or unauthorized")
        return sorted(self.files)

    def list_repo_commits(self, repo_id, *, repo_type, revision, token):
        if revision != REVISION or token != PRIVATE_TOKEN:
            raise RuntimeError("unpinned or unauthorized")
        return [SimpleNamespace(commit_id=commit) for commit in self.commits]

    def hf_hub_download(
        self,
        repo_id,
        filename,
        *,
        repo_type,
        revision,
        token,
        cache_dir,
    ):
        if revision != REVISION or token != PRIVATE_TOKEN:
            raise RuntimeError("unpinned or unauthorized")
        if filename in self.forbidden_downloads:
            raise AssertionError("private labels or legacy validation were read")
        destination = Path(cache_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)


class ProducerHub:
    """Minimal private-Hub fixture used to exercise the real ledger producer."""

    def __init__(self, root, files):
        self.root = Path(root)
        self.files = dict(files)
        self.sha = "producer-sha-0"
        self.create_calls = []

    def repo_info(self, repo_id, *, repo_type, revision):
        return SimpleNamespace(sha=self.sha)

    def hf_hub_download(self, repo_id, filename, *, repo_type, revision):
        if revision != self.sha or filename not in self.files:
            raise EntryNotFoundError("not found")
        destination = self.root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)

    def create_commit(
        self,
        *,
        repo_id,
        repo_type,
        revision,
        parent_commit,
        operations,
        commit_message,
    ):
        if parent_commit != self.sha:
            raise AssertionError("producer fixture requires the exact parent")
        for operation in operations:
            self.files[operation.path_in_repo] = operation.path_or_fileobj
        self.create_calls.append(
            tuple(operation.path_in_repo for operation in operations)
        )
        self.sha = "producer-sha-1"
        return SimpleNamespace(oid=self.sha)


class OrganizerSnapshotTests(unittest.TestCase):
    def load(self, hub=None):
        return load_snapshot(
            "private/docsem",
            REVISION,
            PRIVATE_TOKEN,
            api=hub or FakeHub(),
        )

    def produce_real_store_attempt(self, *, participant_names_length=500):
        """Return immutable files emitted by the participant HubTestStore."""
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        release_id = "r" * 4_096
        identifier = "i" * 256
        answer = "a" * 4_096
        evidence_id = "e" * 256
        labels = [
            {"instance_id": identifier, "answer": answer, "evidence": [evidence_id]}
        ]
        predictions = [dict(labels[0])]
        gold = (json.dumps(labels[0], separators=(",", ":")) + "\n").encode()
        gold_digest = hashlib.sha256(gold).hexdigest()
        release = {
            "schema_version": 1,
            "release_id": release_id,
            "task_manifest_sha256": TASK_DIGEST,
            "gold_sha256": gold_digest,
            "enabled": True,
            "finalized": False,
            "max_attempts": 3,
            "feedback_policy": "first-attempt-only",
            "open_at": "2026-09-01T00:00:00Z",
            "close_at": "2026-10-01T00:00:00Z",
            "public_revision": "4" * 40,
            "public_repo_id": "public/docsem",
            "task_manifest_path": "test/tasks.jsonl",
        }
        hub = ProducerHub(
            workspace.name,
            {
                "private/test_release.json": json_bytes(release),
                "private/test_labels.jsonl": gold,
            },
        )
        identity = OAuthIdentity(
            sub="s" * 4_096,
            username="u" * 4_096,
            email="e" * 4_084 + "@example.org",
        )
        metadata = {
            "release_id": release_id,
            "task_manifest_sha256": TASK_DIGEST,
            "scoring_gold_sha256": gold_digest,
            "scoring_private_revision": "3" * 40,
            "scoring_public_revision": "4" * 40,
            "scoring_public_repo_id": "public/docsem",
            "scoring_task_manifest_path": "test/tasks.jsonl",
            "team": "t" * 4_096,
            "participant_names": "p" * participant_names_length,
            "submission_name": "n" * 4_096,
        }
        metrics = score_predictions(predictions, labels)
        store = HubTestStore(
            hub,
            repo_id="private/docsem",
            release_config_path="private/test_release.json",
            gold_config_path="private/test_labels.jsonl",
            now_provider=lambda: dt.datetime(2026, 9, 5, 12, 0, tzinfo=dt.timezone.utc),
        )
        receipt = store.submit(identity, metadata, predictions, metrics)
        return hub.files, identity, receipt

    def make_real_store(self):
        """Return a real store over an in-memory private-Hub fixture."""
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        gold = b'{"instance_id":"task-0","answer":"42","evidence":["b1"]}\n'
        gold_digest = hashlib.sha256(gold).hexdigest()
        release = {
            "schema_version": 1,
            "release_id": "docsem-test-2026",
            "task_manifest_sha256": TASK_DIGEST,
            "gold_sha256": gold_digest,
            "enabled": True,
            "finalized": False,
            "max_attempts": 3,
            "feedback_policy": "first-attempt-only",
            "open_at": "2026-09-01T00:00:00Z",
            "close_at": "2026-10-01T00:00:00Z",
            "public_revision": "4" * 40,
            "public_repo_id": "public/docsem",
            "task_manifest_path": "test/tasks.jsonl",
        }
        hub = ProducerHub(
            workspace.name,
            {
                "private/test_release.json": json_bytes(release),
                "private/test_labels.jsonl": gold,
            },
        )
        identity = OAuthIdentity("subject-a", "user-a", "user-a@example.org")
        metadata = {
            "release_id": release["release_id"],
            "task_manifest_sha256": TASK_DIGEST,
            "scoring_gold_sha256": gold_digest,
            "scoring_private_revision": "3" * 40,
            "scoring_public_revision": "4" * 40,
            "scoring_public_repo_id": "public/docsem",
            "scoring_task_manifest_path": "test/tasks.jsonl",
            "team": "Shared Team",
            "participant_names": "Participant A",
            "submission_name": "run-a-1",
        }
        store = HubTestStore(
            hub,
            repo_id="private/docsem",
            release_config_path="private/test_release.json",
            gold_config_path="private/test_labels.jsonl",
            now_provider=lambda: dt.datetime(2026, 9, 5, 12, 0, tzinfo=dt.timezone.utc),
        )
        return hub, store, identity, metadata

    def test_real_participant_producer_records_verify_at_exact_text_boundaries(self):
        """Catches drift between producer records and OrganizerSnapshot validation."""
        files, identity, receipt = self.produce_real_store_attempt()
        snapshot = self.load(FakeHub(files))
        report = verify_snapshot(snapshot)

        self.assertTrue(report.valid)
        self.assertEqual(report.account_count, 1)
        self.assertEqual(report.attempt_count, 1)
        rows = organizer_rows(snapshot)
        self.assertEqual(rows[0]["account_key"], account_key(identity))
        self.assertEqual(rows[0]["submission_id"], receipt.submission_id)
        self.assertEqual(len(rows[0]["team"]), 4_096)
        self.assertEqual(len(rows[0]["participant_names"]), 500)
        self.assertEqual(len(rows[0]["submission_name"]), 4_096)

    def test_organizer_rejects_participant_names_above_producer_boundary(self):
        """Catches the organizer accepting records the participant producer forbids."""
        files, identity, receipt = self.produce_real_store_attempt()
        key = account_key(identity)
        attempt_path = f"attempts/test/{key}/{receipt.submission_id}.json"
        record = json.loads(files[attempt_path])
        record["participant_names"] = "p" * 501
        files[attempt_path] = json_bytes(record)

        account_path = f"projections/test/accounts/{key}.json"
        account = json.loads(files[account_path])
        account["attempts"][0]["record_sha256"] = hashlib.sha256(
            files[attempt_path]
        ).hexdigest()
        files[account_path] = json_bytes(account)

        organizer_path = "projections/test/organizer_leaderboard.json"
        projection = json.loads(files[organizer_path])
        projection["accounts"][0]["participant_names"] = "p" * 501
        files[organizer_path] = json_bytes(projection)

        report = verify_snapshot(self.load(FakeHub(files)))
        self.assertFalse(report.valid)
        self.assertIn("attempt_invalid", report.issue_codes)

    def test_real_store_rejects_extra_prediction_fields_without_emitting_records(self):
        """Catches direct producer calls bypassing the exact test-row schema."""
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        gold = b'{"instance_id":"task-1","answer":"42","evidence":["b1"]}\n'
        gold_digest = hashlib.sha256(gold).hexdigest()
        release = {
            "schema_version": 1,
            "release_id": "docsem-test-2026",
            "task_manifest_sha256": TASK_DIGEST,
            "gold_sha256": gold_digest,
            "enabled": True,
            "finalized": False,
            "max_attempts": 3,
            "feedback_policy": "first-attempt-only",
            "open_at": "2026-09-01T00:00:00Z",
            "close_at": "2026-10-01T00:00:00Z",
            "public_revision": "4" * 40,
            "public_repo_id": "public/docsem",
            "task_manifest_path": "test/tasks.jsonl",
        }
        hub = ProducerHub(
            workspace.name,
            {
                "private/test_release.json": json_bytes(release),
                "private/test_labels.jsonl": gold,
            },
        )
        identity = OAuthIdentity("subject-a", "user-a", "user-a@example.org")
        metadata = {
            "release_id": release["release_id"],
            "task_manifest_sha256": TASK_DIGEST,
            "scoring_gold_sha256": gold_digest,
            "scoring_private_revision": "3" * 40,
            "scoring_public_revision": "4" * 40,
            "scoring_public_repo_id": "public/docsem",
            "scoring_task_manifest_path": "test/tasks.jsonl",
            "team": "Shared Team",
            "participant_names": "Participant A",
            "submission_name": "run-a-1",
        }
        predictions = [
            {
                "instance_id": "task-1",
                "answer": "42",
                "evidence": ["b1"],
                "unexpected": "participant-controlled-extra",
            }
        ]
        metrics = score_predictions(predictions, [json.loads(gold)])
        store = HubTestStore(
            hub,
            repo_id="private/docsem",
            release_config_path="private/test_release.json",
            gold_config_path="private/test_labels.jsonl",
            now_provider=lambda: dt.datetime(2026, 9, 5, 12, 0, tzinfo=dt.timezone.utc),
        )

        with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
            store.submit(identity, metadata, predictions, metrics)

        self.assertEqual(hub.create_calls, [])
        self.assertFalse(any(path.startswith("attempts/test/") for path in hub.files))

    def test_real_store_rejects_attempt_above_reader_file_bound_without_mutation(self):
        """Catches the producer committing a record the organizer cannot read."""
        hub, store, identity, metadata = self.make_real_store()
        labels = [{"instance_id": "task-0", "answer": "42", "evidence": ["b1"]}]
        first_predictions = [dict(labels[0])]
        first_metrics = score_predictions(first_predictions, labels)
        first = store.submit(identity, metadata, first_predictions, first_metrics)
        before_files = dict(hub.files)
        oversized_predictions = [
            {
                "instance_id": f"task-{index:04d}",
                "answer": "a" * 4_096,
                "evidence": ["b1"],
            }
            for index in range(4_100)
        ]

        with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
            store.submit(identity, metadata, oversized_predictions, first_metrics)

        self.assertEqual(len(hub.create_calls), 1)
        self.assertEqual(hub.files, before_files)
        snapshot = self.load(FakeHub(hub.files))
        report = verify_snapshot(snapshot)
        self.assertTrue(report.valid)
        self.assertEqual(report.attempt_count, 1)
        self.assertEqual(
            organizer_rows(snapshot)[0]["submission_id"], first.submission_id
        )

    def test_real_store_and_reader_accept_an_attempt_at_exact_file_byte_limit(self):
        """Catches an accidental inclusive rejection at the shared byte ceiling."""
        hub, store, identity, metadata = self.make_real_store()
        predictions = [
            {
                "instance_id": f"task-{index:04d}",
                "answer": "a" * 4_096,
                "evidence": ["b"],
            }
            for index in range(3_800)
        ]
        metrics = {
            "answer_accuracy": 1.0,
            "evidence_exact_match": 1.0,
            "evidence_f1": 1.0,
            "examples": len(predictions),
            "per_example": [
                {
                    "instance_id": row["instance_id"],
                    "answer_exact_match": 1.0,
                    "evidence_exact_match": 1.0,
                    "evidence_f1": 1.0,
                }
                for row in predictions
            ],
        }
        first = store.submit(identity, metadata, predictions, metrics)
        key = account_key(identity)
        first_path = f"attempts/test/{key}/{first.submission_id}.json"
        remaining = test_contract.MAX_LEDGER_FILE_BYTES - len(hub.files[first_path])
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 255 * len(predictions))

        boundary_predictions = []
        for row in predictions:
            extra = min(remaining, 255)
            remaining -= extra
            boundary_predictions.append(
                {
                    "instance_id": row["instance_id"],
                    "answer": "z" * 4_096,
                    "evidence": ["b" + "e" * extra],
                }
            )
        self.assertEqual(remaining, 0)

        second = store.submit(identity, metadata, boundary_predictions, metrics)
        second_path = f"attempts/test/{key}/{second.submission_id}.json"
        self.assertEqual(
            len(hub.files[second_path]), test_contract.MAX_LEDGER_FILE_BYTES
        )

        snapshot = self.load(FakeHub(hub.files))
        report = verify_snapshot(snapshot)
        self.assertTrue(report.valid)
        self.assertEqual(report.attempt_count, 2)

    def test_reader_accepts_a_file_at_the_shared_serialized_byte_boundary(self):
        """Catches producer and organizer ledger-file limits drifting apart."""
        self.assertEqual(
            MAX_FILE_BYTES,
            getattr(test_contract, "MAX_LEDGER_FILE_BYTES", None),
        )
        files = fixture_files()
        release = files["private/test_release.json"]
        self.assertLess(len(release), MAX_FILE_BYTES)
        files["private/test_release.json"] = release + b" " * (
            MAX_FILE_BYTES - len(release)
        )

        report = verify_snapshot(self.load(FakeHub(files)))

        self.assertTrue(report.valid)

    def test_reconstructs_all_accounts_attempts_best_exclusions_and_adjudications(self):
        """Catches trusting a projection or collapsing accounts that share a team."""
        snapshot = self.load()
        audit = verify_snapshot(snapshot)
        rows = organizer_rows(snapshot)

        self.assertTrue(audit.valid)
        self.assertEqual(audit.account_count, 2)
        self.assertEqual(audit.attempt_count, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [(row["account_key"], row["attempt_number"]) for row in rows],
            [(ACCOUNT_A, 1), (ACCOUNT_A, 2), (ACCOUNT_B, 1)],
        )
        a_rows = [row for row in rows if row["account_key"] == ACCOUNT_A]
        self.assertEqual([row["selected_best"] for row in a_rows], [False, False])
        self.assertTrue(all(row["excluded"] for row in a_rows))
        self.assertTrue(all(row["exclusion_count"] == 1 for row in a_rows))
        self.assertTrue(all(row["adjudication_count"] == 1 for row in a_rows))
        self.assertEqual(
            {row["account_key"] for row in rows if row["team"] == "Shared Team"},
            {ACCOUNT_A, ACCOUNT_B},
        )
        self.assertFalse(rows[-1]["excluded"])
        self.assertNotIn("predictions", rows[0])

    def test_adjudications_recompute_eligibility_and_best_attempt_in_shared_order(self):
        """Catches the organizer view disagreeing with finalization decisions."""

        files = fixture_files()
        events = (
            {
                **RELEASE_STATE,
                "record_id": "reinstate-account-a",
                "account_key": ACCOUNT_A,
                "created_at": "2026-09-05T13:00:00Z",
                "action": "reinstate_account",
                "reason_code": "appeal-granted",
            },
            {
                **RELEASE_STATE,
                "record_id": "a-reinstate-best",
                "account_key": ACCOUNT_A,
                "submission_id": ID_A2,
                "created_at": "2026-09-05T14:00:00Z",
                "action": "reinstate_attempt",
                "reason_code": "ordered-first",
            },
            {
                **RELEASE_STATE,
                "record_id": "z-exclude-best",
                "account_key": ACCOUNT_A,
                "submission_id": ID_A2,
                "created_at": "2026-09-05T14:00:00Z",
                "action": "exclude_attempt",
                "reason_code": "ordered-last",
            },
            {
                **RELEASE_STATE,
                "record_id": "exclude-account-b",
                "account_key": ACCOUNT_B,
                "created_at": "2026-09-05T13:00:00Z",
                "action": "exclude_account",
                "reason_code": "temporary-review",
            },
            {
                **RELEASE_STATE,
                "record_id": "reinstate-account-b",
                "account_key": ACCOUNT_B,
                "created_at": "2026-09-05T15:00:00Z",
                "action": "reinstate_account",
                "reason_code": "review-complete",
            },
        )
        for event in events:
            files[f"{ADJUDICATION_PREFIX}{event['record_id']}.json"] = json_bytes(event)

        snapshot = self.load(FakeHub(files))
        report = verify_snapshot(snapshot)
        rows = organizer_rows(snapshot)

        self.assertTrue(report.valid)
        a1, a2 = [row for row in rows if row["account_key"] == ACCOUNT_A]
        b1 = [row for row in rows if row["account_key"] == ACCOUNT_B][0]
        self.assertFalse(a1["account_excluded"])
        self.assertFalse(a1["attempt_excluded"])
        self.assertFalse(a1["excluded"])
        self.assertTrue(a1["selected_best"])
        self.assertFalse(a2["account_excluded"])
        self.assertTrue(a2["attempt_excluded"])
        self.assertTrue(a2["excluded"])
        self.assertFalse(a2["selected_best"])
        self.assertFalse(b1["account_excluded"])
        self.assertFalse(b1["attempt_excluded"])
        self.assertFalse(b1["excluded"])
        self.assertTrue(b1["selected_best"])
        self.assertEqual(a1["exclusion_count"], 1)
        self.assertEqual(a1["adjudication_count"], 4)
        self.assertEqual(b1["adjudication_count"], 2)

    def test_adjudication_action_and_target_shape_match_finalizer_contract(self):
        """Catches invalid organizer audit records that finalization must refuse."""

        invalid_records = (
            {
                "record_id": "unknown-action",
                "action": "unknown",
                "submission_id": None,
            },
            {
                "record_id": "non-string-action",
                "action": [],
                "submission_id": None,
            },
            {
                "record_id": "account-with-attempt",
                "action": "exclude_account",
                "submission_id": ID_A1,
            },
            {
                "record_id": "attempt-without-target",
                "action": "exclude_attempt",
                "submission_id": None,
            },
            {
                "record_id": "note-with-bad-target",
                "action": "note",
                "submission_id": "not-a-uuid",
            },
        )
        for overrides in invalid_records:
            with self.subTest(record_id=overrides["record_id"]):
                files = fixture_files()
                record = {
                    **RELEASE_STATE,
                    "record_id": overrides["record_id"],
                    "account_key": ACCOUNT_A,
                    "created_at": "2026-09-06T12:00:00Z",
                    "action": overrides["action"],
                    "reason_code": "invalid-fixture",
                }
                if overrides["submission_id"] is not None:
                    record["submission_id"] = overrides["submission_id"]
                files[f"{ADJUDICATION_PREFIX}{record['record_id']}.json"] = json_bytes(
                    record
                )

                report = verify_snapshot(self.load(FakeHub(files)))

                self.assertFalse(report.valid)
                self.assertIn("adjudication_invalid", report.issue_codes)

    def test_sensitive_dataclass_repr_is_aggregate_only(self):
        """Catches tokens, emails, or private predictions leaking through repr."""
        snapshot = self.load()
        rendered = repr(snapshot) + repr(verify_snapshot(snapshot))
        for private_value in (
            PRIVATE_TOKEN,
            "user-a@example.org",
            "raw-private-prediction-sentinel",
            "gold-must-never-be-read",
        ):
            self.assertNotIn(private_value, rendered)
        self.assertIn("attempt_count=3", rendered)

    def test_disabled_release_without_attempts_is_a_valid_empty_snapshot(self):
        """Catches readiness mode requiring nonexistent attempt projections."""
        release = {
            "schema_version": 1,
            "release_id": RELEASE_STATE["release_id"],
            "task_manifest_sha256": TASK_DIGEST,
            "gold_sha256": GOLD_DIGEST,
            "enabled": False,
            "finalized": False,
            "max_attempts": 3,
            "feedback_policy": "first-attempt-only",
        }
        snapshot = self.load(
            FakeHub({"private/test_release.json": json_bytes(release)})
        )
        audit = verify_snapshot(snapshot)
        self.assertTrue(audit.valid)
        self.assertEqual(audit.account_count, 0)
        self.assertEqual(organizer_rows(snapshot), [])

    def test_requires_exact_pinned_private_repository_revision(self):
        """Catches mutable branch reads, SHA drift, and public repository access."""
        cases = (
            ("main", FakeHub(), PRIVATE_TOKEN),
            (REVISION, FakeHub(revision="e" * 40), PRIVATE_TOKEN),
            (REVISION, FakeHub(private=False), PRIVATE_TOKEN),
            (REVISION, FakeHub(), ""),
        )
        for revision, hub, token in cases:
            with self.subTest(
                revision=revision, private=hub.private, token=bool(token)
            ):
                with self.assertRaises(OrganizerDataError):
                    load_snapshot("private/docsem", revision, token, api=hub)

    def test_missing_or_mutated_attempt_records_fail_audit_and_rows(self):
        """Catches missing referenced files and record identity changed after admission."""
        mutations = []
        missing = fixture_files()
        del missing[f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"]
        mutations.append(missing)

        changed = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"
        value = json.loads(changed[path])
        value["submission_id"] = "different-id"
        changed[path] = json_bytes(value)
        mutations.append(changed)

        for files in mutations:
            with self.subTest(paths=len(files)):
                snapshot = self.load(FakeHub(files))
                self.assertFalse(verify_snapshot(snapshot).valid)
                with self.assertRaises(OrganizerDataError):
                    organizer_rows(snapshot)

    def test_duplicate_ids_or_hashes_and_noncontiguous_attempts_fail(self):
        """Catches ledger aliases, replayed payloads, and quota-number gaps."""
        mutations = []

        duplicate_id = fixture_files()
        b_path = f"attempts/test/{ACCOUNT_B}/{ID_B1}.json"
        b_record = json.loads(duplicate_id.pop(b_path))
        b_record["submission_id"] = ID_A1
        duplicate_id[f"attempts/test/{ACCOUNT_B}/{ID_A1}.json"] = json_bytes(b_record)
        b_projection = json.loads(
            duplicate_id[f"projections/test/accounts/{ACCOUNT_B}.json"]
        )
        b_projection["attempts"][0]["submission_id"] = ID_A1
        b_projection["best_submission_id"] = ID_A1
        duplicate_id[f"projections/test/accounts/{ACCOUNT_B}.json"] = json_bytes(
            b_projection
        )
        organizer = json.loads(
            duplicate_id["projections/test/organizer_leaderboard.json"]
        )
        organizer["accounts"][1]["best_submission_id"] = ID_A1
        organizer["accounts"][1]["submission_id"] = ID_A1
        duplicate_id["projections/test/organizer_leaderboard.json"] = json_bytes(
            organizer
        )
        mutations.append(duplicate_id)

        duplicate_hash = fixture_files()
        b_record = json.loads(duplicate_hash[b_path])
        b_record["submission_hash"] = HASH_A1
        duplicate_hash[b_path] = json_bytes(b_record)
        mutations.append(duplicate_hash)

        gap = fixture_files()
        a2_path = f"attempts/test/{ACCOUNT_A}/{ID_A2}.json"
        a2 = json.loads(gap[a2_path])
        a2["attempt_number"] = 3
        gap[a2_path] = json_bytes(a2)
        mutations.append(gap)

        for files in mutations:
            with self.subTest(mutation=len(mutations)):
                snapshot = self.load(FakeHub(files))
                self.assertFalse(verify_snapshot(snapshot).valid)

    def test_release_scoring_and_projection_mismatches_fail_audit(self):
        """Catches mixed releases/evaluators and stale cached leaderboard projections."""
        mutations = []

        inexact_schema = fixture_files()
        release = json.loads(inexact_schema["private/test_release.json"])
        release["schema_version"] = 1.0
        inexact_schema["private/test_release.json"] = json_bytes(release)
        mutations.append(inexact_schema)

        wrong_digest = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"
        value = json.loads(wrong_digest[path])
        value["scoring_gold_sha256"] = "9" * 64
        wrong_digest[path] = json_bytes(value)
        mutations.append(wrong_digest)

        stale_account = fixture_files()
        path = f"projections/test/accounts/{ACCOUNT_A}.json"
        value = json.loads(stale_account[path])
        value["best_submission_id"] = ID_A1
        stale_account[path] = json_bytes(value)
        mutations.append(stale_account)

        stale_organizer = fixture_files()
        path = "projections/test/organizer_leaderboard.json"
        value = json.loads(stale_organizer[path])
        value["accounts"][0]["metrics"]["answer_accuracy"] = 0.01
        stale_organizer[path] = json_bytes(value)
        mutations.append(stale_organizer)

        malformed_metrics = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"
        value = json.loads(malformed_metrics[path])
        value["metrics"]["answer_accuracy"] = float("nan")
        malformed_metrics[path] = json.dumps(value, allow_nan=True).encode()
        mutations.append(malformed_metrics)

        malformed_exact_match = fixture_files()
        value = json.loads(malformed_exact_match[path])
        value["metrics"]["evidence_exact_match"] = -1
        malformed_exact_match[path] = json_bytes(value)
        mutations.append(malformed_exact_match)

        malformed_timestamp = fixture_files()
        value = json.loads(malformed_timestamp[path])
        value["submitted_at"] = "2026-09-01T12:00:00"
        malformed_timestamp[path] = json_bytes(value)
        mutations.append(malformed_timestamp)

        for files in mutations:
            snapshot = self.load(FakeHub(files))
            audit = verify_snapshot(snapshot)
            self.assertFalse(audit.valid)
            self.assertTrue(audit.issue_codes)
            self.assertNotIn("raw-private", repr(audit))

    def test_attempt_and_projection_schema_versions_are_exact_integers(self):
        """Catches JSON floats masquerading as schema-version integers."""
        mutations = []
        attempt_schema = fixture_files()
        attempt_path = f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"
        value = json.loads(attempt_schema[attempt_path])
        value["schema_version"] = 2.0
        attempt_schema[attempt_path] = json_bytes(value)
        mutations.append(attempt_schema)

        projection_schema = fixture_files()
        projection_path = f"projections/test/accounts/{ACCOUNT_A}.json"
        value = json.loads(projection_schema[projection_path])
        value["schema_version"] = 2.0
        projection_schema[projection_path] = json_bytes(value)
        mutations.append(projection_schema)

        expected_codes = ("attempt_invalid", "account_projection_mismatch")
        for files, expected_code in zip(mutations, expected_codes):
            report = verify_snapshot(self.load(FakeHub(files)))
            self.assertFalse(report.valid)
            self.assertIn(expected_code, report.issue_codes)

    def test_per_example_metrics_and_prediction_inventory_are_validated(self):
        """Catches malformed detailed metrics or incomplete immutable payloads."""
        mutations = []

        malformed_detail = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"
        value = json.loads(malformed_detail[path])
        value["metrics"]["per_example"][0]["evidence_f1"] = 1.5
        malformed_detail[path] = json_bytes(value)
        mutations.append(malformed_detail)

        incomplete_predictions = fixture_files()
        value = json.loads(incomplete_predictions[path])
        value["predictions"].pop()
        incomplete_predictions[path] = json_bytes(value)
        mutations.append(incomplete_predictions)

        private_detail = fixture_files()
        value = json.loads(private_detail[path])
        value["metrics"]["per_example"][0]["gold_answer"] = "private-gold-sentinel"
        private_detail[path] = json_bytes(value)
        mutations.append(private_detail)

        fractional_exact = fixture_files()
        value = json.loads(fractional_exact[path])
        value["metrics"]["per_example"][0]["answer_exact_match"] = 0.5
        fractional_exact[path] = json_bytes(value)
        mutations.append(fractional_exact)

        for files in mutations:
            snapshot = self.load(FakeHub(files))
            report = verify_snapshot(snapshot)
            self.assertFalse(report.valid)
            self.assertIn("attempt_invalid", report.issue_codes)
            with self.assertRaises(OrganizerDataError):
                organizer_rows(snapshot)

    def test_submission_hash_is_recomputed_from_committed_predictions(self):
        """Catches an immutable record whose canonical payload hash was altered."""
        files = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"
        value = json.loads(files[path])
        value["submission_hash"] = "8" * 64
        files[path] = json_bytes(value)
        report = verify_snapshot(self.load(FakeHub(files)))
        self.assertFalse(report.valid)
        self.assertIn("attempt_invalid", report.issue_codes)

    def test_scoring_revisions_and_manifest_path_are_pinned(self):
        """Catches mutable evaluator refs and traversal-bearing scoring paths."""
        mutations = []
        path = f"attempts/test/{ACCOUNT_A}/{ID_A1}.json"
        for field, replacement in (
            ("scoring_private_revision", "main"),
            ("scoring_private_revision", "9" * 40),
            ("scoring_public_revision", "not-a-sha"),
            ("scoring_public_repo_id", "attacker/other-dataset"),
            ("scoring_task_manifest_path", "../../private/test_labels.jsonl"),
            ("submitted_at", "2027-01-01T00:00:00Z"),
        ):
            files = fixture_files()
            value = json.loads(files[path])
            value[field] = replacement
            files[path] = json_bytes(value)
            mutations.append(files)
        for files in mutations:
            report = verify_snapshot(self.load(FakeHub(files)))
            self.assertFalse(report.valid)
            self.assertIn("attempt_invalid", report.issue_codes)

    def test_account_projection_requires_exact_attempt_record_digest(self):
        """Catches references that are not bound to immutable record bytes."""
        files = fixture_files()
        path = f"projections/test/accounts/{ACCOUNT_A}.json"
        value = json.loads(files[path])
        del value["attempts"][0]["record_sha256"]
        files[path] = json_bytes(value)
        report = verify_snapshot(self.load(FakeHub(files)))
        self.assertFalse(report.valid)
        self.assertIn("account_projection_mismatch", report.issue_codes)

    def test_projection_counts_and_attempt_numbers_are_exact_integers(self):
        """Catches JSON floats masquerading as projection ordinals or counts."""
        mutations = []

        account_reference = fixture_files()
        path = f"projections/test/accounts/{ACCOUNT_A}.json"
        value = json.loads(account_reference[path])
        value["attempts"][0]["attempt_number"] = 1.0
        account_reference[path] = json_bytes(value)
        mutations.append(account_reference)

        organizer_count = fixture_files()
        path = "projections/test/organizer_leaderboard.json"
        value = json.loads(organizer_count[path])
        value["accounts"][0]["attempt_count"] = 2.0
        organizer_count[path] = json_bytes(value)
        mutations.append(organizer_count)

        organizer_number = fixture_files()
        value = json.loads(organizer_number[path])
        value["accounts"][0]["attempt_number"] = 2.0
        organizer_number[path] = json_bytes(value)
        mutations.append(organizer_number)

        for files in mutations:
            self.assertFalse(verify_snapshot(self.load(FakeHub(files))).valid)

    def test_malformed_account_keys_return_a_generic_failed_audit(self):
        """Catches unhashable private JSON values escaping as TypeError."""
        mutations = []
        exclusion = fixture_files()
        path = f"{EXCLUSION_PREFIX}exclude-smoke.json"
        value = json.loads(exclusion[path])
        value["account_key"] = [ACCOUNT_A]
        exclusion[path] = json_bytes(value)
        mutations.append(exclusion)

        organizer = fixture_files()
        path = "projections/test/organizer_leaderboard.json"
        value = json.loads(organizer[path])
        value["accounts"][0]["account_key"] = [ACCOUNT_A]
        organizer[path] = json_bytes(value)
        mutations.append(organizer)

        for files in mutations:
            report = verify_snapshot(self.load(FakeHub(files)))
            self.assertFalse(report.valid)
            self.assertTrue(report.issue_codes)

    def test_malformed_optional_audit_record_fails_without_private_detail(self):
        """Catches invalid append-only audit records being silently ignored."""
        files = fixture_files()
        path = f"{ADJUDICATION_PREFIX}appeal-reviewed.json"
        files[path] = b'{"reason_code":"secret-reason-sentinel"}\n'
        snapshot = self.load(FakeHub(files))
        audit = verify_snapshot(snapshot)
        self.assertFalse(audit.valid)
        self.assertNotIn("secret-reason-sentinel", repr(audit))

    def test_unknown_test_ledger_path_and_snapshot_bounds_fail_closed(self):
        """Catches namespace smuggling and unbounded repository inventories."""
        unknown = fixture_files()
        unknown["attempts/test/unexpected.json"] = b"{}\n"
        with self.assertRaises(OrganizerDataError):
            self.load(FakeHub(unknown))

        oversized_inventory = fixture_files()
        for number in range(MAX_SNAPSHOT_FILES + 1):
            oversized_inventory[f"unrelated/{number}.txt"] = b"x"
        with self.assertRaises(OrganizerDataError):
            self.load(FakeHub(oversized_inventory))

        oversized_file = fixture_files()
        oversized_file["private/test_release.json"] = b"{" + b"x" * MAX_FILE_BYTES
        with self.assertRaises(OrganizerDataError):
            self.load(FakeHub(oversized_file))

    def test_errors_never_echo_token_or_repository_private_payload(self):
        """Catches dependency exception text leaking credentials or private values."""

        class FailingHub(FakeHub):
            def repo_info(self, *args, **kwargs):
                raise RuntimeError(PRIVATE_TOKEN + " raw-private-prediction-sentinel")

        with self.assertRaises(OrganizerDataError) as caught:
            self.load(FailingHub())
        rendered = str(caught.exception)
        self.assertNotIn(PRIVATE_TOKEN, rendered)
        self.assertNotIn("raw-private-prediction-sentinel", rendered)


if __name__ == "__main__":
    unittest.main()
