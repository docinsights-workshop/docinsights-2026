import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

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


REVISION = "f" * 40
TASK_DIGEST = "1" * 64
GOLD_DIGEST = "2" * 64
ACCOUNT_A = hashlib.sha256(b"subject-a").hexdigest()
ACCOUNT_B = hashlib.sha256(b"subject-b").hexdigest()
PRIVATE_TOKEN = "organizer-read-token-sentinel"
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
            "evidence_exact_match": evidence,
            "examples": 2,
            "per_example": [
                {"instance_id": "task-1", "answer_exact_match": answer},
                {"instance_id": "task-2", "answer_exact_match": answer},
            ],
        },
        "predictions": [
            {
                "instance_id": "task-1",
                "answer": "raw-private-prediction-sentinel",
                "evidence": ["block-1"],
            }
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
    a1 = attempt(ACCOUNT_A, 1, "submission-a1", "5" * 64, 0.50, 0.80)
    a2 = attempt(ACCOUNT_A, 2, "submission-a2", "6" * 64, 0.75, 0.60)
    b1 = attempt(ACCOUNT_B, 1, "submission-b1", "7" * 64, 0.70, 0.90)
    grouped = {ACCOUNT_A: [a1, a2], ACCOUNT_B: [b1]}
    release = {
        **RELEASE_STATE,
        "enabled": True,
        "finalized": False,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
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
            "submission_id": "submission-a2",
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


class OrganizerSnapshotTests(unittest.TestCase):
    def load(self, hub=None):
        return load_snapshot(
            "private/docsem",
            REVISION,
            PRIVATE_TOKEN,
            api=hub or FakeHub(),
        )

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
        self.assertEqual([row["selected_best"] for row in a_rows], [False, True])
        self.assertTrue(all(row["excluded"] for row in a_rows))
        self.assertTrue(all(row["exclusion_count"] == 1 for row in a_rows))
        self.assertTrue(all(row["adjudication_count"] == 1 for row in a_rows))
        self.assertEqual(
            {row["account_key"] for row in rows if row["team"] == "Shared Team"},
            {ACCOUNT_A, ACCOUNT_B},
        )
        self.assertFalse(rows[-1]["excluded"])
        self.assertNotIn("predictions", rows[0])

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
            **RELEASE_STATE,
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
        del missing[f"attempts/test/{ACCOUNT_A}/submission-a1.json"]
        mutations.append(missing)

        changed = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/submission-a1.json"
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
        b_path = f"attempts/test/{ACCOUNT_B}/submission-b1.json"
        b_record = json.loads(duplicate_id.pop(b_path))
        b_record["submission_id"] = "submission-a1"
        duplicate_id[f"attempts/test/{ACCOUNT_B}/submission-a1.json"] = json_bytes(
            b_record
        )
        b_projection = json.loads(
            duplicate_id[f"projections/test/accounts/{ACCOUNT_B}.json"]
        )
        b_projection["attempts"][0]["submission_id"] = "submission-a1"
        b_projection["best_submission_id"] = "submission-a1"
        duplicate_id[f"projections/test/accounts/{ACCOUNT_B}.json"] = json_bytes(
            b_projection
        )
        organizer = json.loads(
            duplicate_id["projections/test/organizer_leaderboard.json"]
        )
        organizer["accounts"][1]["best_submission_id"] = "submission-a1"
        organizer["accounts"][1]["submission_id"] = "submission-a1"
        duplicate_id["projections/test/organizer_leaderboard.json"] = json_bytes(
            organizer
        )
        mutations.append(duplicate_id)

        duplicate_hash = fixture_files()
        b_record = json.loads(duplicate_hash[b_path])
        b_record["submission_hash"] = "5" * 64
        duplicate_hash[b_path] = json_bytes(b_record)
        mutations.append(duplicate_hash)

        gap = fixture_files()
        a2_path = f"attempts/test/{ACCOUNT_A}/submission-a2.json"
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

        wrong_digest = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/submission-a1.json"
        value = json.loads(wrong_digest[path])
        value["scoring_gold_sha256"] = "9" * 64
        wrong_digest[path] = json_bytes(value)
        mutations.append(wrong_digest)

        stale_account = fixture_files()
        path = f"projections/test/accounts/{ACCOUNT_A}.json"
        value = json.loads(stale_account[path])
        value["best_submission_id"] = "submission-a1"
        stale_account[path] = json_bytes(value)
        mutations.append(stale_account)

        stale_organizer = fixture_files()
        path = "projections/test/organizer_leaderboard.json"
        value = json.loads(stale_organizer[path])
        value["accounts"][0]["metrics"]["answer_accuracy"] = 0.01
        stale_organizer[path] = json_bytes(value)
        mutations.append(stale_organizer)

        malformed_metrics = fixture_files()
        path = f"attempts/test/{ACCOUNT_A}/submission-a1.json"
        value = json.loads(malformed_metrics[path])
        value["metrics"]["answer_accuracy"] = float("nan")
        malformed_metrics[path] = json.dumps(value, allow_nan=True).encode()
        mutations.append(malformed_metrics)

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
