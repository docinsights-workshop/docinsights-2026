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
TEST_CLOSE = dt.datetime(2026, 9, 11, 12, 0, tzinfo=dt.timezone.utc)
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
    "scoring_private_revision": "e" * 40,
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
    "joint_accuracy": 0.25,
    "answer_accuracy": 0.25,
    "evidence_exact_match": 1.0,
    "evidence_f1": 0.75,
    "examples": 1,
    "per_example": [
        {
            "instance_id": "test-1",
            "answer_exact_match": 0.0,
            "evidence_exact_match": 1.0,
            "evidence_f1": 0.75,
            "joint_exact_match": 0.0,
        }
    ],
}


def provisional_bytes(rows=()):
    return (
        json.dumps(
            {
                "schema_version": 3,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_DIGEST,
                "rows": list(rows),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def release_bytes(
    *,
    enabled=True,
    open_at="2026-09-01T00:00:00Z",
    close_at="2026-09-11T12:00:00Z",
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
            "sealed/release.json": release_bytes(),
            "sealed/gold.jsonl": GOLD,
            "projections/test/public_provisional.json": provisional_bytes(),
        }
        initial.update(files or {})
        self._lock = threading.Lock()
        self._counter = 0
        self._sha = "sha-0"
        self._snapshots = {self._sha: dict(initial)}
        self._parents = {self._sha: None}
        self.create_barrier = create_barrier
        self._barrier_waits_remaining = create_barrier.parties if create_barrier else 0
        self.conflicts = conflicts
        self.create_calls = []
        self.download_calls = []
        self.repo_info_calls = 0
        self.repo_error = None
        self.download_error = None
        self.download_hook = None
        self.create_error = None
        self.create_error_after_apply = None
        self.mutate_after_apply = None
        self.return_descendant = False
        self.commit_history_calls = []

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
            updated["projections/test/organizer_leaderboard.json"] = json.dumps(
                value
            ).encode("utf-8")
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
            if self.download_hook is not None:
                self.download_hook(revision, filename)
            snapshot = self._snapshots.get(revision)
            if snapshot is None or filename not in snapshot:
                raise EntryNotFoundError("not found")
            self._counter += 1
            target = self.download_root / f"download-{id(self)}-{self._counter}"
            target.write_bytes(snapshot[filename])
            return str(target)

    def list_repo_commits(self, repo_id, *, repo_type, revision):
        with self._lock:
            self.commit_history_calls.append((repo_id, repo_type, revision))
            if revision not in self._snapshots:
                raise EntryNotFoundError("not found")
            commit_ids = []
            current = revision
            while current is not None:
                commit_ids.append(SimpleNamespace(commit_id=current))
                current = self._parents[current]
            return commit_ids

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
                    raise AssertionError(
                        "test fake expects byte-backed commit operations"
                    )
                updated[operation.path_in_repo] = content
            if self.mutate_after_apply is not None:
                self.mutate_after_apply(updated, operations)
            self._advance(updated)
            if self.return_descendant:
                self._advance(dict(self._snapshots[self._sha]))
            if self.create_error_after_apply is not None:
                error = self.create_error_after_apply
                self.create_error_after_apply = None
                raise error
            return SimpleNamespace(oid=self._sha)

    def _advance(self, files):
        parent = self._sha
        next_number = int(self._sha.split("-")[-1]) + 1
        self._sha = f"sha-{next_number}"
        self._snapshots[self._sha] = files
        self._parents[self._sha] = parent


class HubTestStoreTests(unittest.TestCase):
    def test_missing_server_config_paths_fail_before_repository_io(self):
        hub = InMemoryHub()
        store = HubTestStore(hub, repo_id="private/repo")

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(hub.repo_info_calls, 0)
        self.assertEqual(hub.download_calls, [])

    def setUp(self):
        self.downloads = tempfile.TemporaryDirectory()
        self.addCleanup(self.downloads.cleanup)
        InMemoryHub.download_root = Path(self.downloads.name)

    def test_direct_store_rejects_noncanonical_or_oversized_predictions_before_io(self):
        """Catches callers bypassing the participant service's test payload gate."""
        cases = {
            "empty rows": [],
            "too many rows": [PREDICTIONS[0]] * 10_001,
            "extra row field": [{**PREDICTIONS[0], "gold_answer": "untrusted"}],
            "empty instance id": [{**PREDICTIONS[0], "instance_id": ""}],
            "long instance id": [{**PREDICTIONS[0], "instance_id": "i" * 257}],
            "duplicate instance id": [PREDICTIONS[0], dict(PREDICTIONS[0])],
            "long answer": [{**PREDICTIONS[0], "answer": "a" * 4_097}],
            "empty evidence": [{**PREDICTIONS[0], "evidence": []}],
            "too many evidence ids": [
                {**PREDICTIONS[0], "evidence": [f"b{index}" for index in range(129)]}
            ],
            "empty evidence id": [{**PREDICTIONS[0], "evidence": [""]}],
            "long evidence id": [{**PREDICTIONS[0], "evidence": ["b" * 257]}],
        }

        for name, predictions in cases.items():
            with self.subTest(name=name):
                hub = InMemoryHub()
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )

                with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
                    store.submit(IDENTITY, META, predictions, METRICS)

                self.assertEqual(hub.repo_info_calls, 0)
                self.assertEqual(hub.create_calls, [])

    def test_direct_store_bounds_identity_and_private_metadata_before_io(self):
        """Catches unbounded OAuth or participant metadata reaching JSON persistence."""
        identity_cases = {
            "subject": OAuthIdentity("s" * 4_097, IDENTITY.username, IDENTITY.email),
            "username": OAuthIdentity(IDENTITY.sub, "u" * 4_097, IDENTITY.email),
            "email": OAuthIdentity(IDENTITY.sub, IDENTITY.username, "e" * 4_097),
        }
        metadata_cases = {
            "team": {**META, "team": "t" * 4_097},
            "participant names": {**META, "participant_names": "p" * 501},
            "submission name": {**META, "submission_name": "n" * 4_097},
            "release id": {**META, "release_id": "r" * 4_097},
        }

        for name, identity in identity_cases.items():
            with self.subTest(identity=name):
                hub = InMemoryHub()
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )
                with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
                    store.submit(identity, META, PREDICTIONS, METRICS)
                self.assertEqual(hub.repo_info_calls, 0)
                self.assertEqual(hub.create_calls, [])

        for name, metadata in metadata_cases.items():
            with self.subTest(metadata=name):
                hub = InMemoryHub()
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )
                with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
                    store.submit(IDENTITY, metadata, PREDICTIONS, METRICS)
                self.assertEqual(hub.repo_info_calls, 0)
                self.assertEqual(hub.create_calls, [])

    def test_direct_store_rejects_ascii_controls_in_persisted_metadata_before_io(self):
        """Catches control characters crossing the immutable-ledger boundary."""

        identity_cases = {
            "subject NUL": OAuthIdentity(
                f"{IDENTITY.sub}\0x", IDENTITY.username, IDENTITY.email
            ),
            "username newline": OAuthIdentity(
                IDENTITY.sub, f"{IDENTITY.username}\nspoof", IDENTITY.email
            ),
            "email tab": OAuthIdentity(
                IDENTITY.sub, IDENTITY.username, f"{IDENTITY.email}\tspoof"
            ),
        }
        metadata_cases = {
            "release id": {**META, "release_id": "release\nspoof"},
            "team": {**META, "team": "Team\tSpoof"},
            "participant names": {
                **META,
                "participant_names": "Participant\rSpoof",
            },
            "submission name": {
                **META,
                "submission_name": "submission\x7fspoof",
            },
        }

        for name, identity in identity_cases.items():
            with self.subTest(identity=name):
                hub = InMemoryHub()
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )
                with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
                    store.submit(identity, META, PREDICTIONS, METRICS)
                self.assertEqual(hub.repo_info_calls, 0)
                self.assertEqual(hub.create_calls, [])

        for name, metadata in metadata_cases.items():
            with self.subTest(metadata=name):
                hub = InMemoryHub()
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )
                with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
                    store.submit(IDENTITY, metadata, PREDICTIONS, METRICS)
                self.assertEqual(hub.repo_info_calls, 0)
                self.assertEqual(hub.create_calls, [])

    def test_three_concurrent_attempts_commit_and_fourth_is_rejected(self):
        hub = InMemoryHub(create_barrier=threading.Barrier(4))
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        def submit(index):
            predictions = [
                {
                    "instance_id": "test-1",
                    "answer": f"distinct private prediction {index}",
                    "evidence": ["b1"],
                }
            ]
            return store.submit(IDENTITY, META, predictions, METRICS)

        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(submit, range(4)))

        self.assertEqual(sorted(r.attempt for r in receipts if r.accepted), [1, 2, 3])
        self.assertEqual(sum(not r.accepted for r in receipts), 1)
        self.assertEqual(len(store.account_history(IDENTITY)), 3)

    def test_exact_retry_returns_existing_receipt(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        first = store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        replay = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(first, replay)
        self.assertEqual(first.submission_id, replay.submission_id)
        self.assertEqual(len(store.account_history(IDENTITY)), 1)
        self.assertEqual(len(hub.create_calls), 1)

    def test_exact_retry_refuses_participant_metadata_drift(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        first = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        for field, value in (
            ("team", "Changed Team"),
            ("participant_names", "Changed Participant"),
            ("submission_name", "changed run"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
                    store.submit(
                        IDENTITY,
                        {**META, field: value},
                        PREDICTIONS,
                        METRICS,
                    )

        self.assertEqual(len(hub.create_calls), 1)
        self.assertEqual(
            store.account_history(IDENTITY)[0]["submission_id"], first.submission_id
        )

    def test_exact_retry_lookup_returns_immutable_record_without_commit(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        first = store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        commit_count = len(hub.create_calls)

        existing = store.find_exact_attempt(IDENTITY, META, PREDICTIONS)

        self.assertEqual(existing["submission_id"], first.submission_id)
        self.assertEqual(existing["metrics"], METRICS)
        self.assertEqual(len(hub.create_calls), commit_count)

    def test_canonical_duplicate_hash_is_idempotent(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        first = store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        equivalent = [
            {
                "evidence": ["B1", "b1"],
                "answer": "  PRIVATE   prediction ",
                "instance_id": " test-1 ",
            }
        ]

        replay = store.submit(IDENTITY, META, equivalent, METRICS)

        self.assertEqual(replay.submission_id, first.submission_id)
        self.assertEqual(len(hub.create_calls), 1)

    def test_acceptance_is_one_exact_parent_commit_with_test_only_paths(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

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
                "projections/test/public_provisional.json",
            },
        )

        attempt = json.loads(
            hub.files[f"attempts/test/{key}/{receipt.submission_id}.json"]
        )
        account = json.loads(hub.files[f"projections/test/accounts/{key}.json"])
        organizer = json.loads(hub.files["projections/test/organizer_leaderboard.json"])
        provisional = json.loads(hub.files["projections/test/public_provisional.json"])
        self.assertEqual(
            [value["schema_version"] for value in (attempt, account, organizer)],
            [3, 3, 3],
        )
        self.assertEqual(
            set(provisional),
            {
                "schema_version",
                "split",
                "release_id",
                "task_manifest_sha256",
                "rows",
            },
        )
        self.assertEqual(provisional["schema_version"], 3)
        self.assertEqual(
            provisional["rows"],
            [{"rank": 1, "hf_username": "private-user", "team": "Private Team"}],
        )
        self.assertEqual(set(provisional["rows"][0]), {"rank", "hf_username", "team"})

    def test_later_attempt_never_writes_or_changes_provisional_ranks(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        before = hub.files["projections/test/public_provisional.json"]

        receipt = store.submit(
            IDENTITY,
            {**META, "submission_name": "private run 2"},
            [{"instance_id": "test-1", "answer": "better", "evidence": ["b1"]}],
            {**METRICS, "joint_accuracy": 1.0, "answer_accuracy": 1.0},
        )

        self.assertEqual(receipt.attempt, 2)
        self.assertNotIn(
            "projections/test/public_provisional.json",
            {
                operation.path_in_repo
                for operation in hub.create_calls[-1]["operations"]
            },
        )
        self.assertEqual(hub.files["projections/test/public_provisional.json"], before)

    def test_new_account_recomputes_provisional_from_immutable_attempt_one_records(
        self,
    ):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        other = OAuthIdentity(
            sub="oauth-subject-other",
            username="other-user",
            email="other@example.org",
        )

        store.submit(
            other,
            {**META, "team": "Other Team"},
            PREDICTIONS,
            {**METRICS, "joint_accuracy": 0.75, "answer_accuracy": 0.75},
        )

        provisional = json.loads(hub.files["projections/test/public_provisional.json"])
        self.assertEqual(
            provisional["rows"],
            [
                {"rank": 1, "hf_username": "other-user", "team": "Other Team"},
                {
                    "rank": 2,
                    "hf_username": "private-user",
                    "team": "Private Team",
                },
            ],
        )

    def test_account_projection_binds_each_attempt_to_exact_record_bytes(self):
        """Catches projections that cannot detect an immutable-record rewrite."""
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        key = account_key(IDENTITY)
        attempt_path = f"attempts/test/{key}/{receipt.submission_id}.json"
        projection_path = f"projections/test/accounts/{key}.json"
        projection = json.loads(hub.files[projection_path])
        self.assertEqual(
            projection["attempts"][0]["record_sha256"],
            hashlib.sha256(hub.files[attempt_path]).hexdigest(),
        )

    def test_snapshot_reads_are_pinned_to_one_base_sha(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertGreaterEqual(len(hub.download_calls), 8)
        self.assertEqual(
            {revision for revision, _ in hub.download_calls}, {"sha-0", "sha-1"}
        )
        readback_paths = {
            path for revision, path in hub.download_calls if revision == "sha-1"
        }
        self.assertEqual(
            readback_paths,
            {
                "sealed/release.json",
                "sealed/gold.jsonl",
                next(
                    operation.path_in_repo
                    for operation in hub.create_calls[0]["operations"]
                    if operation.path_in_repo.startswith("attempts/test/")
                ),
                f"projections/test/accounts/{account_key(IDENTITY)}.json",
                "projections/test/organizer_leaderboard.json",
                "projections/test/public_provisional.json",
            },
        )

    def test_conflicts_reload_rederive_and_use_fresh_operations(self):
        hub = InMemoryHub(conflicts=2)
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

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

    def test_request_at_hard_close_is_rejected_before_commit(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: TEST_CLOSE,
        )

        with self.assertRaisesRegex(TestStoreError, "not open"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(hub.create_calls, [])
        self.assertFalse(any(path.startswith("attempts/test/") for path in hub.files))

    def test_private_policy_cannot_extend_the_hard_close(self):
        hub = InMemoryHub(
            files={
                "sealed/release.json": release_bytes(close_at="2026-09-12T12:00:00Z")
            }
        )
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: TEST_CLOSE,
        )

        with self.assertRaisesRegex(TestStoreError, "not open"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(hub.create_calls, [])

    def test_conflict_retry_resamples_time_and_rejects_when_hard_close_crosses(self):
        hub = InMemoryHub(conflicts=1)
        clock_values = iter((TEST_CLOSE - dt.timedelta(microseconds=1), TEST_CLOSE))
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: next(clock_values),
        )

        with self.assertRaisesRegex(TestStoreError, "not open"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(len(hub.create_calls), 1)
        self.assertFalse(any(path.startswith("attempts/test/") for path in hub.files))

    def test_first_attempt_resamples_after_cross_account_projection_reads(self):
        hub = InMemoryHub()
        first_store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        first_store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        committed_before = len(hub.create_calls)
        clock = [TEST_CLOSE - dt.timedelta(microseconds=1)]
        existing_key = account_key(IDENTITY)

        def cross_close_during_reconstruction(revision, filename):
            if filename == f"projections/test/accounts/{existing_key}.json":
                clock[0] = TEST_CLOSE

        hub.download_hook = cross_close_during_reconstruction
        other = OAuthIdentity(
            sub="oauth-subject-other",
            username="other-user",
            email="other@example.org",
        )
        second_store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: clock[0],
        )

        with self.assertRaisesRegex(TestStoreError, "not open"):
            second_store.submit(
                other,
                {**META, "team": "Other Team"},
                PREDICTIONS,
                METRICS,
            )

        self.assertEqual(len(hub.create_calls), committed_before)

    def test_accepted_at_uses_the_fresh_precommit_clock_instant(self):
        accepted_at = NOW + dt.timedelta(seconds=7)
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: accepted_at,
        )

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        record = store.account_history(IDENTITY)[0]

        expected = "2026-09-05T12:00:07Z"
        self.assertEqual(receipt.accepted_at, expected)
        self.assertEqual(record["submitted_at"], expected)

    def test_successful_preclose_commit_is_acknowledged_without_postcommit_clock(self):
        clock_values = iter((TEST_CLOSE - dt.timedelta(microseconds=1),))
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: next(clock_values),
        )

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertTrue(receipt.accepted)

    def test_disabled_and_closed_releases_fail_before_commit(self):
        cases = {
            "disabled": release_bytes(enabled=False),
            "closed": release_bytes(close_at="2026-09-05T12:00:00Z"),
        }
        for name, release in cases.items():
            with self.subTest(name=name):
                hub = InMemoryHub(files={"sealed/release.json": release})
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )
                with self.assertRaisesRegex(TestStoreError, "not open"):
                    store.submit(IDENTITY, META, PREDICTIONS, METRICS)
                self.assertEqual(hub.create_calls, [])

    def test_missing_gold_fails_closed_without_commit(self):
        hub = InMemoryHub()
        del hub._snapshots["sha-0"]["sealed/gold.jsonl"]
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(hub.create_calls, [])

    def test_release_task_and_gold_digest_mismatches_fail_closed(self):
        cases = {
            "release": {**META, "release_id": "wrong-release"},
            "task": {**META, "task_manifest_sha256": "b" * 64},
        }
        for name, metadata in cases.items():
            with self.subTest(name=name):
                hub = InMemoryHub()
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )
                with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
                    store.submit(IDENTITY, metadata, PREDICTIONS, METRICS)
                self.assertEqual(hub.create_calls, [])

        hub = InMemoryHub(
            files={"sealed/release.json": release_bytes(gold_digest="c" * 64)}
        )
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)
        self.assertEqual(hub.create_calls, [])

    def test_gold_change_after_scoring_is_rejected_before_commit(self):
        changed_gold = (
            b'{"instance_id":"test-1","answer":"changed gold","evidence":["b2"]}\n'
        )
        changed_digest = hashlib.sha256(changed_gold).hexdigest()
        hub = InMemoryHub(
            files={
                "sealed/release.json": release_bytes(gold_digest=changed_digest),
                "sealed/gold.jsonl": changed_gold,
            }
        )
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(hub.create_calls, [])

    def test_unrelated_private_head_advance_does_not_invalidate_scored_gold(self):
        hub = InMemoryHub()
        with hub._lock:
            updated = dict(hub._snapshots[hub._sha])
            updated["unrelated/audit.json"] = b"{}"
            hub._advance(updated)
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertTrue(receipt.accepted)
        self.assertEqual(hub.create_calls[0]["parent_commit"], "sha-1")

    def test_attempt_record_retains_scoring_snapshot_audit_metadata(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        store.submit(IDENTITY, META, PREDICTIONS, METRICS)
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
                "scoring_private_revision": "e" * 40,
                "scoring_public_revision": "f" * 40,
                "scoring_public_repo_id": "public/repo",
                "scoring_task_manifest_path": "test/tasks.jsonl",
            },
        )

    def test_existing_attempt_with_unbound_scoring_gold_fails_closed(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)
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
            )

        self.assertEqual(len(hub.create_calls), commit_count)

    def test_old_release_attempt_is_rejected_instead_of_counted(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)
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
            )

        self.assertEqual(len(hub.create_calls), commit_count)

    def test_mismatched_account_and_organizer_projections_fail_closed(self):
        key = account_key(IDENTITY)
        cases = ("account", "organizer")
        for projection in cases:
            with self.subTest(projection=projection):
                hub = InMemoryHub()
                store = HubTestStore(
                    hub,
                    repo_id="private/repo",
                    release_config_path="sealed/release.json",
                    gold_config_path="sealed/gold.jsonl",
                    now_provider=lambda: NOW,
                )
                store.submit(IDENTITY, META, PREDICTIONS, METRICS)
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
                    )

                self.assertEqual(len(hub.create_calls), commit_count)

    def test_invalid_account_is_rejected_with_value_free_error(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaisesRegex(TestStoreError, "could not be accepted") as caught:
            store.submit(object(), META, PREDICTIONS, METRICS)

        self.assertNotIn("object", str(caught.exception))
        self.assertEqual(hub.repo_info_calls, 0)

    def test_subject_only_identity_is_rejected_before_repository_access(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        incomplete = OAuthIdentity(sub="subject-only", username="", email="")

        with self.assertRaisesRegex(TestStoreError, "could not be accepted"):
            store.submit(incomplete, META, PREDICTIONS, METRICS)

        self.assertEqual(hub.repo_info_calls, 0)
        self.assertEqual(hub.create_calls, [])

    def test_fourth_distinct_attempt_is_rejected_without_persistence(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        receipts = []
        for index in range(4):
            predictions = [
                {
                    "instance_id": "test-1",
                    "answer": f"answer-{index}",
                    "evidence": ["b1"],
                }
            ]
            receipts.append(store.submit(IDENTITY, META, predictions, METRICS))

        self.assertEqual([r.accepted for r in receipts], [True, True, True, False])
        self.assertIsNone(receipts[-1].attempt)
        self.assertEqual(receipts[-1].submission_id, "")
        self.assertEqual(
            [attempt["attempt_number"] for attempt in receipts[-1].existing_attempts],
            [1, 2, 3],
        )
        self.assertEqual(len(hub.create_calls), 3)
        self.assertEqual(len(store.account_history(IDENTITY)), 3)

    def test_uncertain_postcommit_error_deduplicates_by_canonical_hash(self):
        hub = InMemoryHub()
        hub.create_error_after_apply = outage_error(
            "private@example.org oauth-subject-private private prediction score=0.25"
        )
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertTrue(receipt.accepted)
        self.assertEqual(len(hub.create_calls), 1)
        self.assertEqual(len(store.account_history(IDENTITY)), 1)

    def test_postcommit_tamper_never_returns_an_accepted_receipt(self):
        hub = InMemoryHub()

        def tamper(updated, operations):
            path = next(
                operation.path_in_repo
                for operation in operations
                if operation.path_in_repo.startswith("attempts/test/")
            )
            record = json.loads(updated[path])
            record["metrics"]["joint_accuracy"] = 0.99
            updated[path] = json.dumps(record).encode("utf-8")

        hub.mutate_after_apply = tamper
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(len(hub.create_calls), 1)

    def test_returned_revision_must_be_direct_child_even_when_bytes_match(self):
        hub = InMemoryHub()
        hub.return_descendant = True
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(len(hub.create_calls), 1)

    def test_postcommit_reload_rejects_invalid_unchanged_provisional_state(self):
        hub = InMemoryHub()
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        def tamper_inherited_provisional(updated, operations):
            projection = json.loads(updated["projections/test/public_provisional.json"])
            projection["rows"][0]["joint_accuracy"] = 1.0
            updated["projections/test/public_provisional.json"] = json.dumps(
                projection
            ).encode("utf-8")

        hub.mutate_after_apply = tamper_inherited_provisional
        committed_before = len(hub.create_calls)

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(
                IDENTITY,
                {**META, "submission_name": "private run 2"},
                [{"instance_id": "test-1", "answer": "second", "evidence": ["b1"]}],
                METRICS,
            )

        self.assertEqual(len(hub.create_calls), committed_before + 1)

    def test_read_side_http_409_is_not_treated_as_a_cas_conflict(self):
        hub = InMemoryHub()
        hub.repo_error = conflict_error("read-side conflict")
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(hub.repo_info_calls, 1)
        self.assertEqual(hub.create_calls, [])

    def test_exhausted_conflicts_fail_without_writing_an_attempt(self):
        hub = InMemoryHub(conflicts=5)
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaisesRegex(TestStoreError, "temporarily unavailable"):
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        self.assertEqual(len(hub.create_calls), 5)
        self.assertFalse(any(path.startswith("attempts/test/") for path in hub.files))

    def test_non_http_failures_are_genericized_without_sensitive_values(self):
        hub = InMemoryHub()
        hub.repo_error = ValueError(
            "private@example.org oauth-subject-private private prediction score=0.25"
        )
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        with self.assertRaises(TestStoreError) as caught:
            store.submit(IDENTITY, META, PREDICTIONS, METRICS)

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
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )

        store.submit(IDENTITY, META, PREDICTIONS, METRICS)

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
        store = HubTestStore(
            hub,
            repo_id="private/repo",
            release_config_path="sealed/release.json",
            gold_config_path="sealed/gold.jsonl",
            now_provider=lambda: NOW,
        )
        receipt = store.submit(IDENTITY, META, PREDICTIONS, METRICS)

        history = store.account_history(IDENTITY)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["submission_id"], receipt.submission_id)
        self.assertEqual(history[0]["attempt_number"], 1)
        self.assertEqual(history[0]["predictions"], PREDICTIONS)


if __name__ == "__main__":
    unittest.main()
