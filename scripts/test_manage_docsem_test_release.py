import contextlib
import datetime as dt
import hashlib
import importlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    manager = importlib.import_module("manage_docsem_test_release")
except ModuleNotFoundError:
    manager = None


UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 5, 20, 0, tzinfo=UTC)
OPEN_AT = "2026-09-05T20:05:00Z"
CLOSE_AT = "2026-09-11T12:00:00Z"
PRIVATE_HEAD = "a" * 40
NEXT_HEAD = "b" * 40
PUBLIC_HEAD = "c" * 40
RELEASE_ID = "docsem-test-synthetic-r1"


def canonical(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


TASK_ROWS = [
    {
        "instance_id": "task_010001",
        "user_query": "First query",
        "document_pdf": "test/documents/task_010001.pdf",
    },
    {
        "instance_id": "task_010002",
        "user_query": "Second query",
        "document_pdf": "test/documents/task_010002.pdf",
    },
]
TASK_BYTES = b"".join(canonical(row) for row in TASK_ROWS)
TASK_DIGEST = hashlib.sha256(TASK_BYTES).hexdigest()
SORTED_IDS_DIGEST = hashlib.sha256(b"task_010001\ntask_010002\n").hexdigest()
GOLD_ROWS = [
    {"instance_id": "task_010001", "answer": "private-a", "evidence": ["b1"]},
    {"instance_id": "task_010002", "answer": "private-b", "evidence": ["b2"]},
]
GOLD_BYTES = b"".join(canonical(row) for row in GOLD_ROWS)
GOLD_DIGEST = hashlib.sha256(GOLD_BYTES).hexdigest()
PDF_DIGEST = "d" * 64


def base_release(**changes):
    value = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "counts": {"tasks": 2, "pdfs": 2, "labels": 2},
        "sorted_ids_sha256": SORTED_IDS_DIGEST,
        "task_manifest_sha256": TASK_DIGEST,
        "gold_sha256": GOLD_DIGEST,
        "pdf_inventory_sha256": PDF_DIGEST,
        "visibility_audit": {"method": "synthetic-safe-audit"},
        "enabled": False,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "finalized": False,
    }
    value.update(changes)
    return value


def active_release(**changes):
    value = base_release(
        enabled=True,
        open_at=OPEN_AT,
        close_at=CLOSE_AT,
        public_revision=PUBLIC_HEAD,
        public_repo_id="owner/public",
        task_manifest_path="test/tasks.jsonl",
    )
    value.update(changes)
    return value


def provisional(rows=None):
    return {
        "schema_version": 3,
        "split": "test",
        "release_id": RELEASE_ID,
        "task_manifest_sha256": TASK_DIGEST,
        "rows": [] if rows is None else rows,
    }


def public_release():
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "counts": {"tasks": 2, "pdfs": 2},
        "sorted_ids_sha256": SORTED_IDS_DIGEST,
        "task_manifest_sha256": TASK_DIGEST,
        "pdf_inventory_sha256": PDF_DIGEST,
    }


def finalized_release(public_final, audit):
    return active_release(
        enabled=False,
        finalized=True,
        finalized_at="2026-09-11T12:00:01Z",
        finalization_source_revision="e" * 40,
        finalization_scorer_revision="f" * 40,
        finalization_scorer_sha256="1" * 64,
        final_projection_sha256=hashlib.sha256(public_final).hexdigest(),
        finalization_audit_sha256=hashlib.sha256(audit).hexdigest(),
    )


class FakeHub:
    def __init__(self, release=None, *, paths=None):
        self.identity_value = SimpleNamespace(username="owner", role="write")
        self.public_head = PUBLIC_HEAD
        self.private_head = PRIVATE_HEAD
        private = {
            "README.md": b"preserve me\n",
            "private/val_labels.jsonl": b"private validation\n",
            "private/test_labels.jsonl": GOLD_BYTES,
            "private/test_release.json": canonical(release or base_release()),
        }
        private.update(paths or {})
        self.snapshots = {PRIVATE_HEAD: private}
        self.parents = {PRIVATE_HEAD: None}
        self.commit_calls = []
        self.commit_error = None
        self.postwrite_corruption = None
        self.read_calls = []

    def identity(self, token):
        return self.identity_value

    def repository_state(self, repository, token):
        if repository == "owner/public":
            return SimpleNamespace(revision=self.public_head, private=False)
        return SimpleNamespace(revision=self.private_head, private=True)

    def list_paths(self, repository, revision, token):
        if repository == "owner/public":
            return ("README.md", "test/release.json", "test/tasks.jsonl")
        return tuple(sorted(self.snapshots[revision]))

    def read_files(self, repository, revision, paths, token):
        self.read_calls.append((repository, revision, tuple(paths)))
        if repository == "owner/public":
            values = {
                "test/release.json": canonical(public_release()),
                "test/tasks.jsonl": TASK_BYTES,
            }
        else:
            values = self.snapshots[revision]
        return {path: values[path] for path in paths}

    def create_commit(self, repository, expected_parent, files, message, token):
        self.commit_calls.append(
            {
                "repository": repository,
                "expected_parent": expected_parent,
                "files": dict(files),
                "message": message,
            }
        )
        if self.commit_error is not None:
            raise self.commit_error
        if expected_parent != self.private_head:
            raise manager.ParentConflictError("synthetic conflict")
        updated = dict(self.snapshots[expected_parent])
        updated.update(files)
        if self.postwrite_corruption is not None:
            self.postwrite_corruption(updated)
        self.snapshots[NEXT_HEAD] = updated
        self.parents[NEXT_HEAD] = expected_parent
        self.private_head = NEXT_HEAD
        return NEXT_HEAD

    def ancestry(self, repository, revision, token):
        values = []
        current = revision
        while current is not None:
            values.append(current)
            current = self.parents.get(current)
        return tuple(values)


class FeaturePresenceTest(unittest.TestCase):
    def test_guarded_release_manager_module_exists(self):
        self.assertIsNotNone(manager, "guarded release manager is not implemented")


@unittest.skipIf(manager is None, "feature module is not implemented yet")
class ManageReleaseTests(unittest.TestCase):
    def setUp(self):
        self.constants = mock.patch.multiple(
            manager,
            EXPECTED_OWNER="owner",
            PUBLIC_REPOSITORY="owner/public",
            PRIVATE_REPOSITORY="owner/private",
            PUBLIC_REVISION=PUBLIC_HEAD,
            RELEASE_ID=RELEASE_ID,
            EXPECTED_COUNT=2,
            TASK_MANIFEST_SHA256=TASK_DIGEST,
            GOLD_SHA256=GOLD_DIGEST,
            SORTED_IDS_SHA256=SORTED_IDS_DIGEST,
            PDF_INVENTORY_SHA256=PDF_DIGEST,
        )
        self.constants.start()
        self.addCleanup(self.constants.stop)

    def run_manager(self, hub, mode, **kwargs):
        return manager.manage_release(
            mode=mode,
            expected_private_head=PRIVATE_HEAD,
            expected_state=kwargs.pop("expected_state", None),
            open_at=kwargs.pop("open_at", None),
            hub=hub,
            token="secret-never-print",
            now=kwargs.pop("now", NOW),
            **kwargs,
        )

    def test_activate_is_one_exact_parent_commit_of_policy_and_empty_projection(self):
        hub = FakeHub()
        before = dict(hub.snapshots[PRIVATE_HEAD])

        receipt = self.run_manager(hub, "activate", open_at=OPEN_AT)

        self.assertEqual(receipt["status"], "activated")
        self.assertEqual(receipt["state"], "open")
        self.assertEqual(receipt["previous_revision"], PRIVATE_HEAD)
        self.assertEqual(receipt["revision"], NEXT_HEAD)
        self.assertNotIn("secret-never-print", json.dumps(receipt))
        self.assertEqual(len(hub.commit_calls), 1)
        call = hub.commit_calls[0]
        self.assertEqual(call["expected_parent"], PRIVATE_HEAD)
        self.assertEqual(
            set(call["files"]),
            {
                "private/test_release.json",
                "projections/test/public_provisional.json",
            },
        )
        self.assertEqual(call["files"]["private/test_release.json"], canonical(active_release()))
        self.assertEqual(
            call["files"]["projections/test/public_provisional.json"],
            canonical(provisional()),
        )
        after = hub.snapshots[NEXT_HEAD]
        for path, payload in before.items():
            if path != "private/test_release.json":
                self.assertEqual(after[path], payload)

    def test_activate_requires_a_fresh_future_utc_open_before_the_fixed_close(self):
        cases = (None, "2026-09-05T20:00:00Z", "2026-09-11T12:00:00Z", "bad")
        for open_at in cases:
            with self.subTest(open_at=open_at):
                hub = FakeHub()
                with self.assertRaises(manager.ReleaseError):
                    self.run_manager(hub, "activate", open_at=open_at)
                self.assertEqual(hub.commit_calls, [])

    def test_activate_refuses_every_preexisting_governed_artifact(self):
        paths = (
            "attempts/test/" + "1" * 64 + "/attempt.json",
            "projections/test/accounts/" + "1" * 64 + ".json",
            "projections/test/organizer_leaderboard.json",
            "projections/test/public_provisional.json",
            "projections/test/public_final.json",
            "private/test_finalization_audit.json",
            "exclusions/test/record.json",
            "adjudications/test/record.json",
        )
        for path in paths:
            with self.subTest(path=path):
                hub = FakeHub(paths={path: b"{}\n"})
                with self.assertRaises(manager.ReleaseError):
                    self.run_manager(hub, "activate", open_at=OPEN_AT)
                self.assertEqual(hub.commit_calls, [])

    def test_activate_refuses_noninstalled_release_state(self):
        for release in (
            active_release(),
            base_release(finalized=True),
            {**base_release(), "unexpected": True},
            base_release(counts={"tasks": 1, "pdfs": 2, "labels": 2}),
        ):
            with self.subTest(release=release):
                hub = FakeHub(release)
                with self.assertRaises(manager.ReleaseError):
                    self.run_manager(hub, "activate", open_at=OPEN_AT)
                self.assertEqual(hub.commit_calls, [])

    def test_close_at_cutoff_changes_only_enabled_and_preserves_every_path(self):
        existing = {
            "projections/test/public_provisional.json": canonical(
                provisional([{"rank": 1, "hf_username": "u", "team": "T"}])
            ),
            "attempts/test/" + "1" * 64 + "/123.json": b"private attempt\n",
            "projections/test/accounts/" + "1" * 64 + ".json": b"private account\n",
            "projections/test/organizer_leaderboard.json": b"private organizer\n",
        }
        hub = FakeHub(active_release(), paths=existing)
        before = dict(hub.snapshots[PRIVATE_HEAD])

        receipt = self.run_manager(
            hub,
            "close",
            now=dt.datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        )

        self.assertEqual(receipt["status"], "closed")
        self.assertEqual(len(hub.commit_calls), 1)
        call = hub.commit_calls[0]
        self.assertEqual(set(call["files"]), {"private/test_release.json"})
        expected = active_release(enabled=False)
        self.assertEqual(call["files"]["private/test_release.json"], canonical(expected))
        after = hub.snapshots[NEXT_HEAD]
        self.assertEqual(set(after), set(before))
        for path, payload in before.items():
            if path != "private/test_release.json":
                self.assertEqual(after[path], payload)

    def test_close_before_exclusive_cutoff_refuses_without_write(self):
        hub = FakeHub(active_release(), paths={
            "projections/test/public_provisional.json": canonical(provisional())
        })
        with self.assertRaises(manager.ReleaseError):
            self.run_manager(
                hub,
                "close",
                now=dt.datetime(2026, 9, 11, 11, 59, 59, tzinfo=UTC),
            )
        self.assertEqual(hub.commit_calls, [])

    def test_verify_only_accepts_exact_installed_open_and_closed_states(self):
        fixtures = {
            "installed": FakeHub(),
            "open": FakeHub(
                active_release(),
                paths={"projections/test/public_provisional.json": canonical(provisional())},
            ),
            "closed": FakeHub(
                active_release(enabled=False),
                paths={"projections/test/public_provisional.json": canonical(provisional())},
            ),
        }
        times = {
            "installed": NOW,
            "open": dt.datetime(2026, 9, 6, tzinfo=UTC),
            "closed": dt.datetime(2026, 9, 11, 12, 0, 1, tzinfo=UTC),
        }
        for state, hub in fixtures.items():
            with self.subTest(state=state):
                receipt = self.run_manager(
                    hub, "verify", expected_state=state, now=times[state]
                )
                self.assertEqual(receipt["status"], "verified")
                self.assertEqual(receipt["state"], state)
                self.assertEqual(hub.commit_calls, [])

    def test_verify_rejects_inexact_provisional_schema_or_private_public_data(self):
        bad_values = (
            {**provisional(), "extra": 1},
            {**provisional(), "schema_version": 1},
            provisional([{"rank": 1, "hf_username": "u", "team": "T", "score": 1.0}]),
            provisional([{"rank": 2, "hf_username": "u", "team": "T"}]),
        )
        for value in bad_values:
            with self.subTest(value=value):
                hub = FakeHub(
                    active_release(),
                    paths={"projections/test/public_provisional.json": canonical(value)},
                )
                with self.assertRaises(manager.ReleaseError):
                    self.run_manager(
                        hub,
                        "verify",
                        expected_state="open",
                        now=dt.datetime(2026, 9, 6, tzinfo=UTC),
                    )

    def test_verify_finalized_binds_final_artifacts_and_hashes(self):
        final = canonical(
            {
                "schema_version": 1,
                "split": "test",
                "release_id": RELEASE_ID,
                "task_manifest_sha256": TASK_DIGEST,
                "rows": [
                    {
                        "rank": 1,
                        "hf_username": "final-user",
                        "team": "Final Team",
                        "submission_name": "best run",
                        "selected_attempt": 2,
                        "joint_accuracy": 0.6,
                        "answer_accuracy": 0.75,
                        "evidence_f1": 0.8,
                    }
                ],
            }
        )
        audit = canonical(
            {
                "schema_version": 1,
                "split": "test",
                "release_id": RELEASE_ID,
                "source_revision": "e" * 40,
                "finalized_at": "2026-09-11T12:00:01Z",
                "close_at": CLOSE_AT,
                "task_manifest_sha256": TASK_DIGEST,
                "gold_sha256": GOLD_DIGEST,
                "scorer_revision": "f" * 40,
                "scorer_code_sha256": "1" * 64,
                "public_projection_sha256": hashlib.sha256(final).hexdigest(),
                "selected_account_count": 1,
            }
        )
        hub = FakeHub(
            finalized_release(final, audit),
            paths={
                "projections/test/public_provisional.json": canonical(provisional()),
                "projections/test/public_final.json": final,
                "private/test_finalization_audit.json": audit,
            },
        )
        receipt = self.run_manager(
            hub,
            "verify",
            expected_state="finalized",
            now=dt.datetime(2026, 9, 11, 12, 0, 2, tzinfo=UTC),
        )
        self.assertEqual(receipt["state"], "finalized")

        hub.snapshots[PRIVATE_HEAD]["projections/test/public_final.json"] += b" "
        with self.assertRaises(manager.ReleaseError):
            self.run_manager(
                hub,
                "verify",
                expected_state="finalized",
                now=dt.datetime(2026, 9, 11, 12, 0, 2, tzinfo=UTC),
            )

    def test_every_mode_refuses_wrong_head_identity_visibility_or_anchors(self):
        mutators = (
            lambda hub: setattr(hub, "private_head", "9" * 40),
            lambda hub: setattr(
                hub, "identity_value", SimpleNamespace(username="other", role="write")
            ),
            lambda hub: setattr(
                hub, "identity_value", SimpleNamespace(username="owner", role="read")
            ),
            lambda hub: setattr(hub, "public_head", "8" * 40),
            lambda hub: setattr(hub, "public_head", PUBLIC_HEAD),
        )
        for index, mutate in enumerate(mutators):
            with self.subTest(index=index):
                hub = FakeHub()
                mutate(hub)
                if index == 4:
                    hub.snapshots[PRIVATE_HEAD]["private/test_labels.jsonl"] += b" "
                with self.assertRaises(manager.ReleaseError):
                    self.run_manager(hub, "verify", expected_state="installed")

        hub = FakeHub()
        original = hub.repository_state
        hub.repository_state = lambda repo, token: (
            SimpleNamespace(revision=PRIVATE_HEAD, private=False)
            if repo == "owner/private"
            else original(repo, token)
        )
        with self.assertRaises(manager.ReleaseError):
            self.run_manager(hub, "verify", expected_state="installed")

    def test_parent_conflict_is_not_retried(self):
        hub = FakeHub()
        hub.commit_error = manager.ParentConflictError("stale")
        with self.assertRaises(manager.ConcurrentUpdateError):
            self.run_manager(hub, "activate", open_at=OPEN_AT)
        self.assertEqual(len(hub.commit_calls), 1)

    def test_ambiguous_commit_or_postwrite_failure_is_uncertain_and_not_retried(self):
        hub = FakeHub()
        hub.commit_error = RuntimeError("secret transport detail")
        with self.assertRaises(manager.PublicationUncertainError) as caught:
            self.run_manager(hub, "activate", open_at=OPEN_AT)
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(len(hub.commit_calls), 1)

        hub = FakeHub()
        hub.postwrite_corruption = lambda files: files.__setitem__(
            "projections/test/public_provisional.json", b"{}\n"
        )
        with self.assertRaises(manager.PublicationUncertainError):
            self.run_manager(hub, "activate", open_at=OPEN_AT)
        self.assertEqual(len(hub.commit_calls), 1)

    def test_main_emits_only_sanitized_json_and_has_distinct_safe_exit_codes(self):
        hub = FakeHub()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = manager.main(
                [
                    "--verify-only",
                    "--expect-state",
                    "installed",
                    "--expected-private-head",
                    PRIVATE_HEAD,
                ],
                hub=hub,
                token="secret-never-print",
                now=NOW,
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "verified")
        self.assertNotIn("secret-never-print", output.getvalue())

        hub = FakeHub()
        hub.commit_error = RuntimeError("server included secret-never-print")
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = manager.main(
                [
                    "--activate",
                    "--open-at",
                    OPEN_AT,
                    "--expected-private-head",
                    PRIVATE_HEAD,
                ],
                hub=hub,
                token="secret-never-print",
                now=NOW,
            )
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(error.getvalue())["status"], "uncertain")
        self.assertNotIn("secret-never-print", error.getvalue())


if __name__ == "__main__":
    unittest.main()
