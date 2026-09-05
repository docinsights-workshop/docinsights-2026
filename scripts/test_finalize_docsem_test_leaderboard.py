import copy
import datetime as dt
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from finalize_docsem_test_leaderboard import (
    AUDIT_PATH,
    GOLD_PATH,
    PUBLIC_FINAL_PATH,
    RELEASE_PATH,
    SCORING_PATH,
    FinalizationError,
    FinalizationSnapshot,
    SnapshotRecord,
    audit_public_projection,
    build_finalization,
    commit_finalization,
    load_finalization_snapshot,
)
from scoring import score_predictions as fixture_score_predictions
from test_policy import OAuthIdentity, canonical_submission_hash


UTC = dt.timezone.utc
SOURCE_REVISION = "a" * 40
FINAL_REVISION = "b" * 40
SCORER_REVISION = "9" * 40
PRIVATE_REVISION = "3" * 40
PUBLIC_REVISION = "4" * 40
TASK_SHA = "1" * 64
CLOSE = dt.datetime(2026, 10, 1, tzinfo=UTC)
NOW = dt.datetime(2026, 10, 2, tzinfo=UTC)
LABELS = (
    {"instance_id": "task-1", "answer": "yes", "evidence": ["b1", "b2"]},
    {"instance_id": "task-2", "answer": "2", "evidence": ["b3"]},
)


def canonical_json(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def canonical_jsonl(rows):
    return b"".join(canonical_json(row) for row in rows)


GOLD_BYTES = canonical_jsonl(LABELS)
GOLD_SHA = hashlib.sha256(GOLD_BYTES).hexdigest()
SCORER_SHA = hashlib.sha256(SCORING_PATH.read_bytes()).hexdigest()


def release(*, finalized=False, enabled=False):
    value = {
        "schema_version": 1,
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_SHA,
        "gold_sha256": GOLD_SHA,
        "enabled": enabled,
        "finalized": finalized,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "open_at": "2026-09-01T00:00:00Z",
        "close_at": "2026-10-01T00:00:00Z",
        "public_revision": PUBLIC_REVISION,
        "public_repo_id": "public/docsem",
        "task_manifest_path": "test/tasks.jsonl",
    }
    return value


def predictions(*, first="yes", second="2", first_evidence=None, second_evidence=None):
    return [
        {
            "instance_id": "task-1",
            "answer": first,
            "evidence": first_evidence or ["b1", "b2"],
        },
        {
            "instance_id": "task-2",
            "answer": second,
            "evidence": second_evidence or ["b3"],
        },
    ]


def score(rows):
    # Fixtures call the frozen scorer directly, independently of the
    # finalizer's injected scorer dispatch.
    return fixture_score_predictions(rows, list(LABELS))


def attempt(
    subject,
    number,
    submission_id,
    submitted_at,
    rows,
    *,
    username=None,
    team="Team",
    submission_name=None,
    overrides=None,
):
    identity = OAuthIdentity(subject, username or subject, f"{subject}@example.org")
    submission_hash = canonical_submission_hash(
        rows, "test", "docsem-test-2026", identity
    )
    account = hashlib.sha256(subject.encode()).hexdigest()
    value = {
        "schema_version": 2,
        "split": "test",
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_SHA,
        "gold_sha256": GOLD_SHA,
        "submission_id": submission_id,
        "account_key": account,
        "hf_subject": subject,
        "hf_username": username or subject,
        "verified_email": f"{subject}@example.org",
        "scoring_gold_sha256": GOLD_SHA,
        "scoring_private_revision": PRIVATE_REVISION,
        "scoring_public_revision": PUBLIC_REVISION,
        "scoring_public_repo_id": "public/docsem",
        "scoring_task_manifest_path": "test/tasks.jsonl",
        "team": team,
        "participant_names": f"Participant {subject}",
        "submission_name": submission_name or f"submission-{number}",
        "submitted_at": submitted_at,
        "submission_hash": submission_hash,
        "attempt_number": number,
        "metrics": score(rows),
        "predictions": copy.deepcopy(rows),
    }
    value.update(overrides or {})
    path = f"attempts/test/{account}/{submission_id}.json"
    return SnapshotRecord.from_value(path, value, committed=True)


def snapshot(records, *, release_value=None, exclusions=(), adjudications=()):
    return FinalizationSnapshot(
        repo_id="private/docsem",
        revision=SOURCE_REVISION,
        private=True,
        ancestor_revisions=(SOURCE_REVISION, PRIVATE_REVISION),
        release=release_value or release(),
        release_bytes=canonical_json(release_value or release()),
        labels=LABELS,
        gold_bytes=GOLD_BYTES,
        attempts=tuple(records),
        exclusions=tuple(exclusions),
        adjudications=tuple(adjudications),
        projection_issue_codes=(),
        scorer_revision=SCORER_REVISION,
        scorer_code_sha256=SCORER_SHA,
    )


class FinalizationPlanTests(unittest.TestCase):
    def test_selects_best_of_three_and_ranks_by_metrics_time_then_id(self):
        a1 = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(first="wrong-first", second="wrong-one"),
            username="user-a",
        )
        a2 = attempt(
            "account-a",
            2,
            "22222222-2222-4222-8222-222222222222",
            "2026-09-03T12:00:00Z",
            predictions(second="wrong-two", first_evidence=["b1"]),
            username="user-a",
        )
        a3 = attempt(
            "account-a",
            3,
            "33333333-3333-4333-8333-333333333333",
            "2026-09-04T12:00:00Z",
            predictions(second="wrong-three"),
            username="user-a",
            submission_name="best-a",
        )
        b1 = attempt(
            "account-b",
            1,
            "44444444-4444-4444-8444-444444444444",
            "2026-09-01T12:00:00Z",
            predictions(second="wrong-b"),
            username="user-b",
            submission_name="best-b",
        )

        plan = build_finalization(snapshot((a1, a2, a3, b1)), NOW)

        self.assertEqual(
            [row["hf_username"] for row in plan.public_projection["rows"]],
            ["user-b", "user-a"],
        )
        self.assertEqual(
            [row["selected_attempt"] for row in plan.public_projection["rows"]],
            [1, 3],
        )
        self.assertEqual(
            [row["rank"] for row in plan.public_projection["rows"]], [1, 2]
        )
        self.assertEqual(plan.eligible_attempt_count, 4)
        self.assertEqual(plan.selected_account_count, 2)

    def test_submission_id_breaks_an_exact_within_account_tie(self):
        later_id = attempt(
            "account-a",
            1,
            "ffffffff-ffff-4fff-8fff-ffffffffffff",
            "2026-09-02T12:00:00Z",
            predictions(second="wrong-a"),
            submission_name="later-id",
        )
        earlier_id = attempt(
            "account-a",
            2,
            "00000000-0000-4000-8000-000000000001",
            "2026-09-02T12:00:00Z",
            predictions(second="wrong-b"),
            submission_name="earlier-id",
        )

        plan = build_finalization(snapshot((later_id, earlier_id)), NOW)

        self.assertEqual(
            plan.public_projection["rows"][0]["submission_name"], "earlier-id"
        )
        self.assertEqual(plan.public_projection["rows"][0]["selected_attempt"], 2)

    def test_input_record_order_does_not_change_any_finalization_hash(self):
        first = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        second = attempt(
            "account-b",
            1,
            "22222222-2222-4222-8222-222222222222",
            "2026-09-03T12:00:00Z",
            predictions(second="wrong"),
        )

        forward = build_finalization(snapshot((first, second)), NOW)
        reverse = build_finalization(snapshot((second, first)), NOW)

        self.assertEqual(forward.projection_sha256, reverse.projection_sha256)
        self.assertEqual(forward.audit_sha256, reverse.audit_sha256)
        self.assertEqual(forward.release_sha256, reverse.release_sha256)

    def test_invalid_and_uncommitted_attempts_are_recorded_and_excluded(self):
        valid = attempt(
            "valid",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        bad_cases = [
            ("post_cutoff", {"submitted_at": "2026-10-01T00:00:00Z"}),
            ("post_cutoff", {"submitted_at": "2026-10-01T00:00:01Z"}),
            ("wrong_release", {"release_id": "other-release"}),
            ("wrong_task", {"task_manifest_sha256": "8" * 64}),
            ("wrong_gold", {"gold_sha256": "7" * 64}),
            ("wrong_provenance", {"scoring_public_revision": "6" * 40}),
            ("malformed", {"attempt_number": "one"}),
        ]
        records = [valid]
        for index, (_, override) in enumerate(bad_cases, start=2):
            records.append(
                attempt(
                    f"bad-{index}",
                    1,
                    str(uuid.UUID(int=index, version=4)),
                    "2026-09-03T12:00:00Z",
                    predictions(),
                    overrides=override,
                )
            )
        uncommitted = attempt(
            "uncommitted",
            1,
            "99999999-9999-4999-8999-999999999999",
            "2026-09-03T12:00:00Z",
            predictions(),
        )
        uncommitted = SnapshotRecord(
            path=uncommitted.path,
            sha256=uncommitted.sha256,
            committed=False,
            value=uncommitted.value,
        )
        records.append(uncommitted)

        plan = build_finalization(snapshot(tuple(records)), NOW)

        reasons = {
            item["reason_code"] for item in plan.audit_manifest["excluded_attempts"]
        }
        self.assertEqual(
            reasons,
            {
                "post_cutoff",
                "wrong_release",
                "wrong_task",
                "wrong_gold",
                "wrong_provenance",
                "malformed",
                "uncommitted",
            },
        )
        self.assertEqual(plan.eligible_attempt_count, 1)
        self.assertEqual(len(plan.public_projection["rows"]), 1)

    def test_duplicate_ids_hashes_and_attempt_numbers_exclude_every_collision(self):
        same_id = "11111111-1111-4111-8111-111111111111"
        first = attempt("dup-a", 1, same_id, "2026-09-02T12:00:00Z", predictions())
        second = attempt("dup-b", 1, same_id, "2026-09-02T12:00:00Z", predictions())
        repeated_number = attempt(
            "dup-a",
            1,
            "22222222-2222-4222-8222-222222222222",
            "2026-09-03T12:00:00Z",
            predictions(second="different-wrong"),
        )

        plan = build_finalization(snapshot((first, second, repeated_number)), NOW)

        reason_sets = [
            set(item["reason_codes"])
            for item in plan.audit_manifest["excluded_attempts"]
        ]
        self.assertTrue(
            any("duplicate_submission_id" in value for value in reason_sets)
        )
        self.assertTrue(
            any("duplicate_attempt_number" in value for value in reason_sets)
        )
        self.assertEqual(plan.eligible_attempt_count, 0)

    def test_exclusions_and_append_only_adjudications_are_applied_in_order(self):
        a = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        b = attempt(
            "account-b",
            1,
            "22222222-2222-4222-8222-222222222222",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        exclusion = SnapshotRecord.from_value(
            "exclusions/test/smoke.json",
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "record_id": "smoke",
                "account_key": a.value["account_key"],
                "created_at": "2026-10-01T01:00:00Z",
                "reason_code": "smoke-account",
            },
        )
        reinstate = SnapshotRecord.from_value(
            "adjudications/test/reinstate.json",
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "record_id": "reinstate",
                "account_key": a.value["account_key"],
                "created_at": "2026-10-01T02:00:00Z",
                "action": "reinstate_account",
                "reason_code": "reviewed",
            },
        )
        exclude_attempt = SnapshotRecord.from_value(
            "adjudications/test/exclude-b.json",
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "record_id": "exclude-b",
                "account_key": b.value["account_key"],
                "submission_id": b.value["submission_id"],
                "created_at": "2026-10-01T03:00:00Z",
                "action": "exclude_attempt",
                "reason_code": "invalid-entry",
            },
        )

        plan = build_finalization(
            snapshot(
                (a, b),
                exclusions=(exclusion,),
                adjudications=(reinstate, exclude_attempt),
            ),
            NOW,
        )

        self.assertEqual(
            [row["hf_username"] for row in plan.public_projection["rows"]],
            ["account-a"],
        )
        self.assertEqual(
            [
                item["record_id"]
                for item in plan.audit_manifest["applied_audit_records"]
            ],
            ["smoke", "reinstate", "exclude-b"],
        )

    def test_note_adjudication_may_reference_one_attempt_without_changing_eligibility(
        self,
    ):
        record = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        note = SnapshotRecord.from_value(
            "adjudications/test/review-note.json",
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "record_id": "review-note",
                "account_key": record.value["account_key"],
                "submission_id": record.value["submission_id"],
                "created_at": "2026-10-01T03:00:00Z",
                "action": "note",
                "reason_code": "technical-review-complete",
            },
        )

        plan = build_finalization(
            snapshot((record,), adjudications=(note,)),
            NOW,
        )

        self.assertEqual(plan.eligible_attempt_count, 1)
        self.assertEqual(plan.selected_account_count, 1)
        self.assertEqual(
            plan.audit_manifest["applied_audit_records"][0]["submission_id"],
            record.value["submission_id"],
        )

    def test_adjudication_must_target_an_existing_account_and_attempt(self):
        record = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        invalid = SnapshotRecord.from_value(
            "adjudications/test/missing-target.json",
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "record_id": "missing-target",
                "account_key": record.value["account_key"],
                "submission_id": "99999999-9999-4999-8999-999999999999",
                "created_at": "2026-10-01T03:00:00Z",
                "action": "exclude_attempt",
                "reason_code": "invalid-entry",
            },
        )

        with self.assertRaises(FinalizationError):
            build_finalization(snapshot((record,), adjudications=(invalid,)), NOW)

    def test_stored_score_mismatch_refuses_finalization(self):
        value = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        changed = copy.deepcopy(value.value)
        changed["metrics"]["answer_accuracy"] = 0.0
        corrupt = SnapshotRecord.from_value(value.path, changed, committed=True)

        with self.assertRaisesRegex(FinalizationError, "independent rescoring"):
            build_finalization(snapshot((corrupt,)), NOW)

    def test_requires_closed_disabled_release_and_utc_now(self):
        record = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        cases = (
            (snapshot((record,)), dt.datetime(2026, 9, 30, tzinfo=UTC)),
            (snapshot((record,), release_value=release(enabled=True)), NOW),
            (snapshot((record,)), dt.datetime(2026, 10, 2)),
        )
        for value, current in cases:
            with self.subTest(current=current, enabled=value.release.get("enabled")):
                with self.assertRaises(FinalizationError):
                    build_finalization(value, current)

    def test_public_projection_is_exactly_allowlisted_and_rejects_positive_controls(
        self,
    ):
        record = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        clean = build_finalization(snapshot((record,)), NOW).public_projection
        self.assertTrue(audit_public_projection(clean))
        self.assertEqual(
            set(clean["rows"][0]),
            {
                "rank",
                "hf_username",
                "team",
                "submission_name",
                "selected_attempt",
                "answer_accuracy",
                "evidence_f1",
            },
        )
        injections = (
            ("verified_email", "private@example.org"),
            ("hf_subject", "oauth-private-sub"),
            ("participant_names", "Private Person"),
            ("predictions", [{"answer": "private"}]),
            ("per_example", [{"instance_id": "private"}]),
            ("unselected_scores", [0.1]),
        )
        for field, value in injections:
            mutated = copy.deepcopy(clean)
            mutated["rows"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(FinalizationError):
                    audit_public_projection(mutated)
        private_path = copy.deepcopy(clean)
        private_path["rows"][0]["team"] = "private/test_labels.jsonl"
        with self.assertRaises(FinalizationError):
            audit_public_projection(private_path)

    def test_dry_run_summary_contains_only_counts_hashes_and_revisions(self):
        record = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
            team="secret-team-sentinel",
        )
        plan = build_finalization(snapshot((record,)), NOW)
        rendered = json.dumps(plan.dry_run_summary(), sort_keys=True)

        self.assertEqual(
            set(plan.dry_run_summary()),
            {
                "source_revision",
                "scorer_revision",
                "eligible_attempt_count",
                "excluded_attempt_count",
                "selected_account_count",
                "projection_sha256",
                "audit_sha256",
                "release_sha256",
                "already_finalized",
            },
        )
        self.assertNotIn("secret-team-sentinel", rendered)
        self.assertNotIn("example.org", rendered)
        self.assertNotIn("private/", rendered)


class FakeHub:
    def __init__(self, root, files, *, private=True):
        self.root = Path(root)
        self.files = dict(files)
        self.private = private
        self.sha = SOURCE_REVISION
        self.commits = [SOURCE_REVISION, PRIVATE_REVISION]
        self.create_calls = []
        self.download_calls = []

    def repo_info(self, repo_id, *, repo_type, revision, token):
        if revision not in ("main", self.sha):
            raise RuntimeError("wrong revision")
        return SimpleNamespace(sha=self.sha, private=self.private)

    def list_repo_files(self, repo_id, *, repo_type, revision, token):
        if revision != self.sha:
            raise RuntimeError("wrong revision")
        return sorted(self.files)

    def list_repo_commits(self, repo_id, *, repo_type, revision, token):
        return [SimpleNamespace(commit_id=item) for item in self.commits]

    def hf_hub_download(
        self, repo_id, filename, *, repo_type, revision, token, cache_dir=None
    ):
        if revision != self.sha or filename not in self.files:
            raise RuntimeError("missing")
        self.download_calls.append((revision, filename))
        target_root = Path(cache_dir or self.root)
        target = target_root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
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
        token,
    ):
        if parent_commit != self.sha:
            raise AssertionError("exact parent required")
        self.create_calls.append(
            (parent_commit, tuple(item.path_in_repo for item in operations))
        )
        for operation in operations:
            payload = operation.path_or_fileobj
            if hasattr(payload, "read"):
                payload = payload.read()
            self.files[operation.path_in_repo] = bytes(payload)
        self.sha = FINAL_REVISION
        self.commits.insert(0, FINAL_REVISION)
        return SimpleNamespace(oid=FINAL_REVISION)


def hub_files(record):
    release_value = release()
    attempt_bytes = canonical_json(record.value)
    account = record.value["account_key"]
    account_projection = {
        "schema_version": 2,
        "split": "test",
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_SHA,
        "gold_sha256": GOLD_SHA,
        "account_key": account,
        "attempts": [
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "submission_id": record.value["submission_id"],
                "attempt_number": 1,
                "record_sha256": hashlib.sha256(attempt_bytes).hexdigest(),
            }
        ],
        "best_submission_id": record.value["submission_id"],
    }
    organizer_projection = {
        "schema_version": 2,
        "split": "test",
        "release_id": "docsem-test-2026",
        "task_manifest_sha256": TASK_SHA,
        "gold_sha256": GOLD_SHA,
        "accounts": [
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "account_key": account,
                "attempt_count": 1,
                "best_submission_id": record.value["submission_id"],
                "hf_subject": record.value["hf_subject"],
                "hf_username": record.value["hf_username"],
                "verified_email": record.value["verified_email"],
                "team": record.value["team"],
                "participant_names": record.value["participant_names"],
                "submission_name": record.value["submission_name"],
                "submitted_at": record.value["submitted_at"],
                "attempt_number": record.value["attempt_number"],
                "metrics": record.value["metrics"],
            }
        ],
    }
    return {
        RELEASE_PATH: canonical_json(release_value),
        GOLD_PATH: GOLD_BYTES,
        record.path: attempt_bytes,
        f"projections/test/accounts/{account}.json": canonical_json(account_projection),
        "projections/test/organizer_leaderboard.json": canonical_json(
            organizer_projection
        ),
    }


class LoadAndCommitTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.record = attempt(
            "account-a",
            1,
            "11111111-1111-4111-8111-111111111111",
            "2026-09-02T12:00:00Z",
            predictions(),
        )
        self.hub = FakeHub(self.workspace.name, hub_files(self.record))

    def load(self, *, revision=SOURCE_REVISION, hub=None, scorer_sha=SCORER_SHA):
        return load_finalization_snapshot(
            "private/docsem",
            revision,
            "private-token-sentinel",
            scorer_revision=SCORER_REVISION,
            scorer_code_sha256=scorer_sha,
            scorer_bytes_at_revision=lambda _: SCORING_PATH.read_bytes(),
            api=hub or self.hub,
        )

    def test_loader_requires_exact_sha_private_visibility_gold_and_scorer_code(self):
        loaded = self.load()
        self.assertTrue(loaded.private)
        self.assertEqual(len(loaded.attempts), 1)
        self.assertTrue(loaded.attempts[0].committed)

        with self.assertRaises(FinalizationError):
            self.load(revision="main")
        with self.assertRaises(FinalizationError):
            self.load(
                hub=FakeHub(self.workspace.name, hub_files(self.record), private=False)
            )
        with self.assertRaises(FinalizationError):
            self.load(scorer_sha="0" * 64)
        with self.assertRaises(FinalizationError):
            load_finalization_snapshot(
                "private/docsem",
                SOURCE_REVISION,
                "private-token-sentinel",
                scorer_revision=SCORER_REVISION,
                scorer_code_sha256=SCORER_SHA,
                scorer_bytes_at_revision=lambda _: b"different scorer bytes",
                api=self.hub,
            )
        corrupt = hub_files(self.record)
        corrupt[GOLD_PATH] += b" "
        with self.assertRaises(FinalizationError):
            self.load(hub=FakeHub(self.workspace.name, corrupt))

    def test_projection_reference_is_the_committed_attempt_boundary(self):
        files = hub_files(self.record)
        projection_path = (
            f"projections/test/accounts/{self.record.value['account_key']}.json"
        )
        projection = json.loads(files[projection_path])
        projection["attempts"][0]["record_sha256"] = "0" * 64
        files[projection_path] = canonical_json(projection)

        loaded = self.load(hub=FakeHub(self.workspace.name, files))

        self.assertFalse(loaded.attempts[0].committed)
        self.assertIn("account_projection_invalid", loaded.projection_issue_codes)
        with self.assertRaisesRegex(FinalizationError, "projection"):
            build_finalization(loaded, NOW)

    def test_projection_with_wrong_release_cannot_mark_an_attempt_committed(self):
        files = hub_files(self.record)
        projection_path = (
            f"projections/test/accounts/{self.record.value['account_key']}.json"
        )
        projection = json.loads(files[projection_path])
        projection["release_id"] = "other-release"
        projection["attempts"][0]["release_id"] = "other-release"
        files[projection_path] = canonical_json(projection)

        loaded = self.load(hub=FakeHub(self.workspace.name, files))

        self.assertFalse(loaded.attempts[0].committed)
        self.assertIn("account_projection_invalid", loaded.projection_issue_codes)
        with self.assertRaisesRegex(FinalizationError, "projection"):
            build_finalization(loaded, NOW)

    def test_projection_issue_refuses_finalization(self):
        files = hub_files(self.record)
        projection = json.loads(files["projections/test/organizer_leaderboard.json"])
        projection["accounts"] = []
        files["projections/test/organizer_leaderboard.json"] = canonical_json(
            projection
        )
        loaded = self.load(hub=FakeHub(self.workspace.name, files))

        self.assertIn("organizer_projection_mismatch", loaded.projection_issue_codes)
        with self.assertRaisesRegex(FinalizationError, "projection"):
            build_finalization(loaded, NOW)

    def test_missing_attempt_referenced_as_projected_best_refuses_finalization(self):
        files = hub_files(self.record)
        del files[self.record.path]

        loaded = self.load(hub=FakeHub(self.workspace.name, files))

        self.assertIn("account_projection_invalid", loaded.projection_issue_codes)
        with self.assertRaisesRegex(FinalizationError, "projection"):
            build_finalization(loaded, NOW)

    def test_projection_bijection_rejects_orphan_attempt_and_wrong_derived_best(self):
        second = attempt(
            "account-a",
            2,
            "22222222-2222-4222-8222-222222222222",
            "2026-09-03T12:00:00Z",
            predictions(second="wrong"),
        )

        orphaned = hub_files(self.record)
        orphaned[second.path] = canonical_json(second.value)
        loaded = self.load(hub=FakeHub(self.workspace.name, orphaned))
        self.assertIn("account_projection_invalid", loaded.projection_issue_codes)
        with self.assertRaisesRegex(FinalizationError, "projection"):
            build_finalization(loaded, NOW)

        wrong_best = hub_files(self.record)
        second_bytes = canonical_json(second.value)
        wrong_best[second.path] = second_bytes
        account_path = (
            f"projections/test/accounts/{self.record.value['account_key']}.json"
        )
        account_projection = json.loads(wrong_best[account_path])
        account_projection["attempts"].append(
            {
                "schema_version": 2,
                "split": "test",
                "release_id": "docsem-test-2026",
                "task_manifest_sha256": TASK_SHA,
                "gold_sha256": GOLD_SHA,
                "submission_id": second.value["submission_id"],
                "attempt_number": 2,
                "record_sha256": hashlib.sha256(second_bytes).hexdigest(),
            }
        )
        account_projection["best_submission_id"] = second.value["submission_id"]
        wrong_best[account_path] = canonical_json(account_projection)
        organizer = json.loads(
            wrong_best["projections/test/organizer_leaderboard.json"]
        )
        row = organizer["accounts"][0]
        row.update(
            {
                "attempt_count": 2,
                "best_submission_id": second.value["submission_id"],
                "hf_subject": second.value["hf_subject"],
                "hf_username": second.value["hf_username"],
                "verified_email": second.value["verified_email"],
                "team": second.value["team"],
                "participant_names": second.value["participant_names"],
                "submission_name": second.value["submission_name"],
                "submitted_at": second.value["submitted_at"],
                "attempt_number": second.value["attempt_number"],
                "metrics": second.value["metrics"],
            }
        )
        wrong_best["projections/test/organizer_leaderboard.json"] = canonical_json(
            organizer
        )

        loaded = self.load(hub=FakeHub(self.workspace.name, wrong_best))
        self.assertIn("account_projection_invalid", loaded.projection_issue_codes)
        with self.assertRaisesRegex(FinalizationError, "projection"):
            build_finalization(loaded, NOW)

    def test_organizer_projection_must_match_selected_record_fields(self):
        files = hub_files(self.record)
        organizer = json.loads(files["projections/test/organizer_leaderboard.json"])
        organizer["accounts"][0]["team"] = "stale-team"
        files["projections/test/organizer_leaderboard.json"] = canonical_json(organizer)

        loaded = self.load(hub=FakeHub(self.workspace.name, files))

        self.assertIn("organizer_projection_mismatch", loaded.projection_issue_codes)
        with self.assertRaisesRegex(FinalizationError, "projection"):
            build_finalization(loaded, NOW)

    def test_cas_requires_both_confirmations_exact_parent_and_writes_three_files_once(
        self,
    ):
        loaded = self.load()
        plan = build_finalization(loaded, NOW)
        self.hub.download_calls.clear()
        for yes, maintenance in ((False, True), (True, False)):
            with self.subTest(yes=yes, maintenance=maintenance):
                with self.assertRaises(FinalizationError):
                    commit_finalization(
                        self.hub,
                        loaded,
                        plan,
                        token="private-token-sentinel",
                        expected_private_sha=SOURCE_REVISION,
                        now=NOW,
                        yes=yes,
                        maintenance_confirmed=maintenance,
                    )
        self.assertEqual(self.hub.create_calls, [])

        revision = commit_finalization(
            self.hub,
            loaded,
            plan,
            token="private-token-sentinel",
            expected_private_sha=SOURCE_REVISION,
            now=NOW + dt.timedelta(microseconds=1),
            yes=True,
            maintenance_confirmed=True,
        )

        self.assertEqual(revision, FINAL_REVISION)
        self.assertEqual(len(self.hub.create_calls), 1)
        self.assertEqual(
            set(self.hub.create_calls[0][1]),
            {PUBLIC_FINAL_PATH, AUDIT_PATH, RELEASE_PATH},
        )
        self.assertEqual(
            self.hub.download_calls,
            [
                (SOURCE_REVISION, RELEASE_PATH),
                (FINAL_REVISION, PUBLIC_FINAL_PATH),
                (FINAL_REVISION, AUDIT_PATH),
                (FINAL_REVISION, RELEASE_PATH),
            ],
        )
        finalized_release = json.loads(self.hub.files[RELEASE_PATH])
        self.assertTrue(finalized_release["finalized"])
        self.assertFalse(finalized_release["enabled"])
        self.assertEqual(finalized_release["finalized_at"], "2026-10-02T00:00:00Z")

    def test_commit_time_cannot_precede_the_immutable_plan_time(self):
        loaded = self.load()
        plan = build_finalization(loaded, NOW)

        with self.assertRaisesRegex(FinalizationError, "plan time"):
            commit_finalization(
                self.hub,
                loaded,
                plan,
                token="private-token-sentinel",
                expected_private_sha=SOURCE_REVISION,
                now=NOW - dt.timedelta(microseconds=1),
                yes=True,
                maintenance_confirmed=True,
            )
        self.assertEqual(self.hub.create_calls, [])

    def test_post_commit_verification_refuses_unconfirmed_write_states(self):
        class AckWithoutWriteHub(FakeHub):
            def create_commit(self, **kwargs):
                self.create_calls.append((kwargs["parent_commit"], ()))
                return SimpleNamespace(oid=FINAL_REVISION)

        class WrongArtifactHub(FakeHub):
            def create_commit(self, **kwargs):
                result = super().create_commit(**kwargs)
                self.files[PUBLIC_FINAL_PATH] = b"{}\n"
                return result

        class ParentRevisionAckHub(FakeHub):
            def create_commit(self, **kwargs):
                super().create_commit(**kwargs)
                return SimpleNamespace(oid=SOURCE_REVISION)

        for hub_type in (AckWithoutWriteHub, WrongArtifactHub, ParentRevisionAckHub):
            with self.subTest(hub=hub_type.__name__):
                hub = hub_type(self.workspace.name, hub_files(self.record))
                loaded = self.load(hub=hub)
                plan = build_finalization(loaded, NOW)

                with self.assertRaisesRegex(
                    FinalizationError, "may have succeeded.*current private revision"
                ):
                    commit_finalization(
                        hub,
                        loaded,
                        plan,
                        token="private-token-sentinel",
                        expected_private_sha=SOURCE_REVISION,
                        now=NOW,
                        yes=True,
                        maintenance_confirmed=True,
                    )

    def test_finalized_rerun_is_idempotent_and_mismatch_refuses(self):
        loaded = self.load()
        plan = build_finalization(loaded, NOW)
        commit_finalization(
            self.hub,
            loaded,
            plan,
            token="private-token-sentinel",
            expected_private_sha=SOURCE_REVISION,
            now=NOW,
            yes=True,
            maintenance_confirmed=True,
        )
        finalized = self.load(revision=FINAL_REVISION)
        repeat = build_finalization(finalized, NOW + dt.timedelta(days=1))

        result = commit_finalization(
            self.hub,
            finalized,
            repeat,
            token="private-token-sentinel",
            expected_private_sha=FINAL_REVISION,
            now=NOW + dt.timedelta(days=1),
            yes=False,
            maintenance_confirmed=False,
        )
        self.assertIsNone(result)
        self.assertTrue(repeat.already_finalized)
        self.assertEqual(len(self.hub.create_calls), 1)

        changed = copy.deepcopy(finalized.existing_public_final)
        changed["rows"][0]["answer_accuracy"] = 0.0
        corrupt = FinalizationSnapshot(
            **{
                **finalized.constructor_fields(),
                "existing_public_final": changed,
            }
        )
        with self.assertRaises(FinalizationError):
            build_finalization(corrupt, NOW + dt.timedelta(days=1))

    def test_head_move_or_reopened_release_refuses_before_commit(self):
        loaded = self.load()
        plan = build_finalization(loaded, NOW)
        self.hub.sha = "c" * 40
        with self.assertRaises(FinalizationError):
            commit_finalization(
                self.hub,
                loaded,
                plan,
                token="private-token-sentinel",
                expected_private_sha=SOURCE_REVISION,
                now=NOW,
                yes=True,
                maintenance_confirmed=True,
            )
        self.assertEqual(self.hub.create_calls, [])

        self.hub.sha = SOURCE_REVISION
        reopened = copy.deepcopy(self.hub.files)
        reopened_release = json.loads(reopened[RELEASE_PATH])
        reopened_release["enabled"] = True
        reopened[RELEASE_PATH] = canonical_json(reopened_release)
        self.hub.files = reopened
        with self.assertRaises(FinalizationError):
            commit_finalization(
                self.hub,
                loaded,
                plan,
                token="private-token-sentinel",
                expected_private_sha=SOURCE_REVISION,
                now=NOW,
                yes=True,
                maintenance_confirmed=True,
            )
        self.assertEqual(self.hub.create_calls, [])


if __name__ == "__main__":
    unittest.main()
