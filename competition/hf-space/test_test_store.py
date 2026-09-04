import datetime as dt
import hashlib
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from requests import Response

from test_policy import OAuthIdentity, account_key
from test_store import HubTestStore, TestStoreError


NOW = dt.datetime(2026, 9, 5, 12, 0, tzinfo=dt.timezone.utc)
IDENTITY = OAuthIdentity(
    sub="oauth-subject-private",
    username="private-user",
    email="private@example.org",
)
TASK_DIGEST = "a" * 64
GOLD = b'{"instance_id":"test-1","answer":"withheld gold","evidence":["b1"]}\n'
GOLD_DIGEST = hashlib.sha256(GOLD).hexdigest()
META = {
    "release_id": "docsem-test-2026",
    "task_manifest_sha256": TASK_DIGEST,
    "scoring_gold_sha256": GOLD_DIGEST,
    "scoring_private_revision": "sha-before-scoring",
    "scoring_public_revision": "f" * 40,
    "scoring_public_repo_id": "public/repo",
    "scoring_task_manifest_path": "test/tasks.jsonl",
    "team": "Private Team",
    "participant_names": "Private Participant",
    "submission_name": "private run",
}
PREDICTIONS = [
    {"instance_id": "test-1", "answer": "private prediction", "evidence": ["b1"]}
]
METRICS = {
    "answer_accuracy": 0.25,
    "evidence_exact_match": 1.0,
    "evidence_f1": 0.75,
    "examples": 1,
    "per_example": [{"instance_id": "test-1", "answer_exact_match": 0.0}],
}


def release_bytes(
    *,
    enabled=True,
    open_at="2026-09-01T00:00:00Z",
    close_at="2026-10-01T00:00:00Z",
    gold_digest=GOLD_DIGEST,
):
    value = {
        "enabled": enabled,
        "max_attempts": 3,
    }
    if enabled:
        value.update(
            {
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_DIGEST,
                "gold_sha256": gold_digest,
                "open_at": open_at,
                "close_at": close_at,
            }
        )
    return json.dumps(value).encode("utf-8")


def conflict_error(message="stale parent"):
    response = Response()
    response.status_code = 409
    return HfHubHTTPError(message, response=response)


def outage_error(message="private outage detail"):
    response = Response()
    response.status_code = 503
    return HfHubHTTPError(message, response=response)


class InMemoryHub:
    """A SHA-versioned Hub fake that applies exact-parent commits atomically."""

    download_root = None

    def __init__(self, *, files=None, create_barrier=None, conflicts=0):
        initial = {
            "private/test_release.json": release_bytes(),
            "private/test_labels.jsonl": GOLD,
        }
        initial.update(files or {})
        self._lock = threading.Lock()
        self._counter = 0
        self._sha = "sha-0"
        self._snapshots = {self._sha: dict(initial)}
        self.create_barrier = create_barrier
        self._barrier_waits_remaining = create_barrier.parties if create_barrier else 0
        self.conflicts = conflicts
        self.create_calls = []
        self.download_calls = []
        self.repo_info_calls = 0
        self.repo_error = None
        self.download_error = None
        self.create_error = None

    @property
    def files(self):
        with self._lock:
            return dict(self._snapshots[self._sha])

    def replace_json(self, path, **changes):
        with self._lock:
            updated = dict(self._snapshots[self._sha])
            value = json.loads(updated[path].decode("utf-8"))
            value.update(changes)
            updated[path] = json.dumps(value).encode("utf-8")
            self._advance(updated)

    def replace_organizer_row(self, key, **changes):
        with self._lock:
            updated = dict(self._snapshots[self._sha])
            value = json.loads(updated["projections/test/organizer_leaderboard.json"])
            row = next(row for row in value["accounts"] if row["account_key"] == key)
            row.update(changes)
            updated["projections/test/organizer_leaderboard.json"] = json.dumps(value).encode(
                "utf-8"
            )
            self._advance(updated)

    def repo_info(self, repo_id, *, repo_type, revision):
        with self._lock:
            self.repo_info_calls += 1
            if self.repo_error is not None:
                raise self.repo_error
            return SimpleNamespace(sha=self._sha)

    def hf_hub_download(self, repo_id, filename, *, repo_type, revision):
        with self._lock:
            if self.download_error is not None:
                raise self.download_error
            self.download_calls.append((revision, filename))
            snapshot = self._snapshots.get(revision)
            if snapshot is None or filename not in snapshot:
                raise EntryNotFoundError("not found")
            self._counter += 1
            target = self.download_root / f"download-{id(self)}-{self._counter}"
            target.write_bytes(snapshot[filename])
            return str(target)

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
        operations = list(operations)
        with self._lock:
            barrier = self.create_barrier if self._barrier_waits_remaining else None
            if barrier is not None:
                self._barrier_waits_remaining -= 1
            if self._barrier_waits_remaining == 0:
                self.create_barrier = None
        if barrier is not None:
            barrier.wait(timeout=5)
        with self._lock:
            call = {
                "parent_commit": parent_commit,
                "operations": operations,
                "operation_ids": tuple(id(operation) for operation in operations),
                "commit_message": commit_message,
            }
            self.create_calls.append(call)
            if self.create_error is not None:
                raise self.create_error
            if self.conflicts:
                self.conflicts -= 1
                self._advance(dict(self._snapshots[self._sha]))
                raise conflict_error()
            if parent_commit != self._sha:
                raise conflict_error()
            updated = dict(self._snapshots[self._sha])
            for operation in operations:
                content = operation.path_or_fileobj
                if not isinstance(content, bytes):
                    raise AssertionError("test fake expects byte-backed commit operations")
                updated[operation.path_in_repo] = content
            self._advance(updated)
            return SimpleNamespace(oid=self._sha)

    def _advance(self, files):
        next_number = int(self._sha.split("-")[-1]) + 1
        self._sha = f"sha-{next_number}"
        self._snapshots[self._sha] = files


class HubTestStoreTests(unittest.TestCase):
    def setUp(self):
        self.downloads = tempfile.TemporaryDirectory()
        self.addCleanup(self.downloads.cleanup)
        InMemoryHub.download_root = Path(self.downloads.name)

    def test_three_concurrent_attempts_commit_and_fourth_is_rejected(self):
        hub = InMemoryHub(create_barrier=threading.Barrier(4))
        store = HubTestStore(hub, repo_id="private/repo")

        def submit(index):
            predictions = [
                {
                    "instance_id": "test-1",
                    "answer": f"distinct private prediction {index}",
                    "evidence": ["b1"],
                }
            ]
            return store.submit(IDENTITY, META, predictions, METRICS, NOW)

        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(submit, range(4)))

        self.assertEqual(sorted(r.attempt for r in receipts if r.accepted), [1, 2, 3])
        self.assertEqual(sum(not r.accepted for r in receipts), 1)
        self.assertEqual(len(store.account_history(IDENTITY)), 3)

    def test_exact_retry_returns_existing_receipt(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")

        first = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
        replay = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertEqual(first, replay)
        self.assertEqual(first.submission_id, replay.submission_id)
        self.assertEqual(len(store.account_history(IDENTITY)), 1)
        self.assertEqual(len(hub.create_calls), 1)

    def test_canonical_duplicate_hash_is_idempotent(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")
        first = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
        equivalent = [
            {
                "evidence": ["B1", "b1"],
                "answer": "  PRIVATE   prediction ",
                "instance_id": " test-1 ",
            }
        ]

        replay = store.submit(IDENTITY, META, equivalent, METRICS, NOW)

        self.assertEqual(replay.submission_id, first.submission_id)
        self.assertEqual(len(hub.create_calls), 1)

    def test_acceptance_is_one_exact_parent_commit_with_test_only_paths(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertTrue(receipt.accepted)
        self.assertEqual(receipt.attempt, 1)
        call = hub.create_calls[0]
        self.assertEqual(call["parent_commit"], "sha-0")
        self.assertEqual(call["commit_message"], "Accept DocSem test attempt 1")
        paths = {operation.path_in_repo for operation in call["operations"]}
        key = account_key(IDENTITY)
        self.assertEqual(
            paths,
            {
                f"attempts/test/{key}/{receipt.submission_id}.json",
                f"projections/test/accounts/{key}.json",
                "projections/test/organizer_leaderboard.json",
            },
        )

    def test_snapshot_reads_are_pinned_to_one_base_sha(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")

        store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertGreaterEqual(len(hub.download_calls), 4)
        self.assertEqual({revision for revision, _ in hub.download_calls}, {"sha-0"})

    def test_conflicts_reload_rederive_and_use_fresh_operations(self):
        hub = InMemoryHub(conflicts=2)
        store = HubTestStore(hub, repo_id="private/repo")

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertTrue(receipt.accepted)
        self.assertEqual(receipt.attempt, 1)
        self.assertEqual(
            [call["parent_commit"] for call in hub.create_calls],
            ["sha-0", "sha-1", "sha-2"],
        )
        operation_ids = [set(call["operation_ids"]) for call in hub.create_calls]
        self.assertTrue(operation_ids[0].isdisjoint(operation_ids[1]))
        self.assertTrue(operation_ids[1].isdisjoint(operation_ids[2]))
        self.assertEqual(hub.repo_info_calls, 3)

    def test_disabled_and_closed_releases_fail_before_commit(self):
        cases = {
            "disabled": release_bytes(enabled=False),
            "closed": release_bytes(close_at="2026-09-05T12:00:00Z"),
        }
        for name, release in cases.items():
            with self.subTest(name=name):
                hub = InMemoryHub(files={"private/test_release.json": release})
                store = HubTestStore(hub, repo_id="private/repo")
                with self.assertRaisesRegex(TestStoreError, "not open"):
                    store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
                self.assertEqual(hub.create_calls, [])

    def test_missing_gold_fails_closed_without_commit(self):
        hub = InMemoryHub()
        del hub._snapshots["sha-0"]["private/test_labels.jsonl"]
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertEqual(hub.create_calls, [])

    def test_release_task_and_gold_digest_mismatches_fail_closed(self):
        cases = {
            "release": {**META, "release_id": "wrong-release"},
            "task": {**META, "task_manifest_sha256": "b" * 64},
        }
        for name, metadata in cases.items():
            with self.subTest(name=name):
                hub = InMemoryHub()
                store = HubTestStore(hub, repo_id="private/repo")
                with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
                    store.submit(IDENTITY, metadata, PREDICTIONS, METRICS, NOW)
                self.assertEqual(hub.create_calls, [])

        hub = InMemoryHub(files={"private/test_release.json": release_bytes(gold_digest="c" * 64)})
        store = HubTestStore(hub, repo_id="private/repo")
        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
        self.assertEqual(hub.create_calls, [])

    def test_gold_change_after_scoring_is_rejected_before_commit(self):
        changed_gold = (
            b'{"instance_id":"test-1","answer":"changed gold",'
            b'"evidence":["b2"]}\n'
        )
        changed_digest = hashlib.sha256(changed_gold).hexdigest()
        hub = InMemoryHub(
            files={
                "private/test_release.json": release_bytes(gold_digest=changed_digest),
                "private/test_labels.jsonl": changed_gold,
            }
        )
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertEqual(hub.create_calls, [])

    def test_unrelated_private_head_advance_does_not_invalidate_scored_gold(self):
        hub = InMemoryHub()
        with hub._lock:
            updated = dict(hub._snapshots[hub._sha])
            updated["unrelated/audit.json"] = b"{}"
            hub._advance(updated)
        store = HubTestStore(hub, repo_id="private/repo")

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertTrue(receipt.accepted)
        self.assertEqual(hub.create_calls[0]["parent_commit"], "sha-1")

    def test_attempt_record_retains_scoring_snapshot_audit_metadata(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")

        store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
        record = store.account_history(IDENTITY)[0]

        self.assertEqual(
            {
                key: record[key]
                for key in (
                    "scoring_gold_sha256",
                    "scoring_private_revision",
                    "scoring_public_revision",
                    "scoring_public_repo_id",
                    "scoring_task_manifest_path",
                )
            },
            {
                "scoring_gold_sha256": GOLD_DIGEST,
                "scoring_private_revision": "sha-before-scoring",
                "scoring_public_revision": "f" * 40,
                "scoring_public_repo_id": "public/repo",
                "scoring_task_manifest_path": "test/tasks.jsonl",
            },
        )

    def test_existing_attempt_with_unbound_scoring_gold_fails_closed(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")
        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
        key = account_key(IDENTITY)
        hub.replace_json(
            f"attempts/test/{key}/{receipt.submission_id}.json",
            scoring_gold_sha256="c" * 64,
        )
        commit_count = len(hub.create_calls)

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(
                IDENTITY,
                META,
                [{"instance_id": "test-1", "answer": "new", "evidence": ["b1"]}],
                METRICS,
                NOW,
            )

        self.assertEqual(len(hub.create_calls), commit_count)

    def test_old_release_attempt_is_rejected_instead_of_counted(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")
        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
        key = account_key(IDENTITY)
        hub.replace_json(
            f"attempts/test/{key}/{receipt.submission_id}.json",
            release_id="old-release",
        )
        commit_count = len(hub.create_calls)

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(
                IDENTITY,
                META,
                [{"instance_id": "test-1", "answer": "new answer", "evidence": ["b1"]}],
                METRICS,
                NOW,
            )

        self.assertEqual(len(hub.create_calls), commit_count)

    def test_mismatched_account_and_organizer_projections_fail_closed(self):
        key = account_key(IDENTITY)
        cases = ("account", "organizer")
        for projection in cases:
            with self.subTest(projection=projection):
                hub = InMemoryHub()
                store = HubTestStore(hub, repo_id="private/repo")
                store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)
                if projection == "account":
                    hub.replace_json(
                        f"projections/test/accounts/{key}.json",
                        gold_sha256="d" * 64,
                    )
                else:
                    hub.replace_organizer_row(key, task_manifest_sha256="e" * 64)
                commit_count = len(hub.create_calls)

                with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
                    store.submit(
                        IDENTITY,
                        META,
                        [
                            {
                                "instance_id": "test-1",
                                "answer": "new answer",
                                "evidence": ["b1"],
                            }
                        ],
                        METRICS,
                        NOW,
                    )

                self.assertEqual(len(hub.create_calls), commit_count)

    def test_invalid_account_is_rejected_with_value_free_error(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaisesRegex(TestStoreError, "could not be accepted") as caught:
            store.submit(object(), META, PREDICTIONS, METRICS, NOW)

        self.assertNotIn("object", str(caught.exception))
        self.assertEqual(hub.repo_info_calls, 0)

    def test_subject_only_identity_is_rejected_before_repository_access(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")
        incomplete = OAuthIdentity(sub="subject-only", username="", email="")

        with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
            store.submit(incomplete, META, PREDICTIONS, METRICS, NOW)

        self.assertEqual(hub.repo_info_calls, 0)
        self.assertEqual(hub.create_calls, [])

    def test_fourth_distinct_attempt_is_rejected_without_persistence(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")
        receipts = []
        for index in range(4):
            predictions = [
                {"instance_id": "test-1", "answer": f"answer-{index}", "evidence": ["b1"]}
            ]
            receipts.append(store.submit(IDENTITY, META, predictions, METRICS, NOW))

        self.assertEqual([r.accepted for r in receipts], [True, True, True, False])
        self.assertIsNone(receipts[-1].attempt)
        self.assertEqual(receipts[-1].submission_id, "")
        self.assertEqual(len(hub.create_calls), 3)
        self.assertEqual(len(store.account_history(IDENTITY)), 3)

    def test_repo_outage_is_not_retried_and_is_genericized(self):
        hub = InMemoryHub()
        hub.create_error = outage_error(
            "private@example.org oauth-subject-private private prediction score=0.25"
        )
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable") as caught:
            store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertEqual(len(hub.create_calls), 1)
        self.assertEqual(str(caught.exception), "Test submission is temporarily unavailable.")

    def test_read_side_http_409_is_not_treated_as_a_cas_conflict(self):
        hub = InMemoryHub()
        hub.repo_error = conflict_error("read-side conflict")
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertEqual(hub.repo_info_calls, 1)
        self.assertEqual(hub.create_calls, [])

    def test_exhausted_conflicts_fail_without_writing_an_attempt(self):
        hub = InMemoryHub(conflicts=5)
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        self.assertEqual(len(hub.create_calls), 5)
        self.assertFalse(any(path.startswith("attempts/test/") for path in hub.files))

    def test_non_http_failures_are_genericized_without_sensitive_values(self):
        hub = InMemoryHub()
        hub.repo_error = ValueError(
            "private@example.org oauth-subject-private private prediction score=0.25"
        )
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaises(TestStoreError) as caught:
            store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        message = str(caught.exception)
        for private_value in (
            IDENTITY.email,
            IDENTITY.sub,
            PREDICTIONS[0]["answer"],
            "0.25",
            "score",
        ):
            self.assertNotIn(private_value, message)

    def test_commit_messages_do_not_contain_private_values(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")

        store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        message = hub.create_calls[0]["commit_message"]
        for private_value in (
            IDENTITY.email,
            IDENTITY.sub,
            PREDICTIONS[0]["answer"],
            "0.25",
            "score",
        ):
            self.assertNotIn(private_value, message)

    def test_account_history_reads_current_immutable_records(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")
        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS, NOW)

        history = store.account_history(IDENTITY)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["submission_id"], receipt.submission_id)
        self.assertEqual(history[0]["attempt_number"], 1)
        self.assertEqual(history[0]["predictions"], PREDICTIONS)


if __name__ == "__main__":
    unittest.main()
