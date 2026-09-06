"""Atomic private-Hub persistence for DocSem test attempts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from test_contract import (
    MAX_LEDGER_FILE_BYTES,
    bounded_private_text,
    is_valid_public_text,
    repository_id,
    revision_digest,
    sha256_digest,
    validate_test_predictions,
)
from test_policy import (
    OFFICIAL_TEST_CLOSE_AT,
    TEST_ATTEMPT_COOLDOWN_SECONDS,
    TestIdentity,
    TestPolicyError,
    TestReleasePolicy,
    account_key,
    canonical_submission_hash,
    next_eligible_at,
    rank_attempts,
    select_best_attempt,
)


ORGANIZER_PATH = "projections/test/organizer_leaderboard.json"
PROVISIONAL_PATH = "projections/test/public_provisional.json"
LEDGER_SCHEMA_VERSION = 3
MAX_COMMIT_ATTEMPTS = 5
IMMUTABLE_RETRY_METADATA_FIELDS = (
    "team",
    "participant_names",
    "submission_name",
)
RELEASE_STATE_FIELDS = frozenset(
    {"schema_version", "split", "release_id", "task_manifest_sha256", "gold_sha256"}
)
ATTEMPT_RECORD_FIELDS = RELEASE_STATE_FIELDS | {
    "submission_id",
    "account_key",
    "identity_kind",
    "identity_subject",
    "hf_username",
    "contact_email",
    "email_verified",
    "scoring_gold_sha256",
    "scoring_private_revision",
    "scoring_public_revision",
    "scoring_public_repo_id",
    "scoring_task_manifest_path",
    "team",
    "participant_names",
    "submission_name",
    "submitted_at",
    "submission_hash",
    "attempt_number",
    "metrics",
    "predictions",
}
ACCOUNT_PROJECTION_FIELDS = RELEASE_STATE_FIELDS | {
    "account_key",
    "attempts",
    "best_submission_id",
}
ACCOUNT_ATTEMPT_REFERENCE_FIELDS = RELEASE_STATE_FIELDS | {
    "submission_id",
    "attempt_number",
    "record_sha256",
}
ORGANIZER_PROJECTION_FIELDS = RELEASE_STATE_FIELDS | {"accounts"}
ORGANIZER_ACCOUNT_FIELDS = RELEASE_STATE_FIELDS | {
    "account_key",
    "attempt_count",
    "best_submission_id",
    "identity_kind",
    "identity_subject",
    "hf_username",
    "contact_email",
    "email_verified",
    "team",
    "participant_names",
    "submission_name",
    "submitted_at",
    "attempt_number",
    "metrics",
}


class TestStoreError(RuntimeError):
    """Value-free public failure raised by private test persistence."""


class TestCooldownError(TestStoreError):
    """Safe participant refusal for a distinct attempt inside the cooldown."""

    def __init__(self, next_eligible_at_value: str):
        self.next_eligible_at = next_eligible_at_value
        super().__init__(
            "A distinct test attempt may be submitted at or after "
            f"{next_eligible_at_value}. Exact retries remain available."
        )


@dataclass(frozen=True)
class TestReceipt:
    accepted: bool
    attempt: int | None
    submission_id: str
    accepted_at: str | None
    existing_attempts: tuple[dict, ...] = ()
    matched_attempt: dict | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class _Snapshot:
    sha: str
    policy: TestReleasePolicy
    gold: bytes
    attempts: tuple[dict, ...]
    attempt_record_sha256: Mapping[str, str]
    organizer: dict
    provisional: dict
    provisional_raw: bytes


class _Unavailable(Exception):
    pass


class _InvalidSubmission(Exception):
    pass


class _ReleaseClosed(Exception):
    pass


class HubTestStore:
    """Persist test attempts with one exact-parent Hugging Face commit."""

    def __init__(
        self,
        api,
        repo_id: str,
        *,
        release_config_path: str | None = None,
        gold_config_path: str | None = None,
        now_provider: Callable[[], dt.datetime] | None = None,
    ):
        self.api = api
        self.repo_id = str(repo_id or "").strip()
        self.release_config_path = str(release_config_path or "").strip()
        self.gold_config_path = str(gold_config_path or "").strip()
        self.now_provider = now_provider or (lambda: dt.datetime.now(dt.timezone.utc))

    def submit(self, identity, metadata, predictions, metrics) -> TestReceipt:
        try:
            key = _complete_identity_key(identity)
            normalized_metadata = _submission_metadata(metadata)
            validate_test_predictions(predictions)
            normalized_predictions = _json_copy(predictions)
            normalized_metrics = _json_copy(metrics)
            _validate_metrics(normalized_metrics, normalized_predictions)
        except Exception:
            raise TestStoreError("Test submission could not be accepted.") from None

        candidate_id = str(uuid.uuid4())
        for _ in range(MAX_COMMIT_ATTEMPTS):
            try:
                snapshot = self._load_snapshot(key)
                _verify_release(snapshot, normalized_metadata)
                _require_complete_prediction_ids(normalized_predictions, snapshot.gold)
                submission_hash = canonical_submission_hash(
                    normalized_predictions,
                    "test",
                    snapshot.policy.release_id,
                    identity,
                    normalized_metadata,
                )
                existing = _find_submission(snapshot.attempts, submission_hash)
                if existing is not None:
                    _require_retry_metadata(existing, normalized_metadata)
                    self._validate_complete_snapshot(snapshot, key)
                    return _accepted_receipt(existing)
                if len(snapshot.attempts) >= snapshot.policy.max_attempts:
                    return TestReceipt(
                        False,
                        None,
                        "",
                        None,
                        tuple(_json_copy(snapshot.attempts)),
                    )

                attempt_number = len(snapshot.attempts) + 1
                first_attempts = None
                if attempt_number == 1:
                    # Cross-account reconstruction may require many pinned Hub
                    # reads. Complete it before the final admission timestamp so
                    # that slow I/O cannot carry an old decision past the close.
                    first_attempts = self._load_attempt_one_records(snapshot)
                    if snapshot.provisional != _provisional_projection(
                        snapshot.policy, first_attempts
                    ):
                        raise _Unavailable()

                # Resample the authoritative Space-server clock immediately before
                # constructing and issuing every exact-parent CAS. No network I/O
                # occurs between this check and create_commit(). Conflict retries
                # therefore cannot carry a stale decision across the hard close.
                commit_now = self.now_provider()
                _require_open(snapshot.policy, commit_now)
                _require_cooldown(snapshot.attempts, commit_now)
                accepted_at = _accepted_at(commit_now)
                record = _attempt_record(
                    identity=identity,
                    key=key,
                    metadata=normalized_metadata,
                    predictions=normalized_predictions,
                    metrics=normalized_metrics,
                    policy=snapshot.policy,
                    submission_id=candidate_id,
                    submission_hash=submission_hash,
                    attempt_number=attempt_number,
                    accepted_at=accepted_at,
                )
                record_bytes = _bounded_json_bytes(record)
                attempts = [*snapshot.attempts, record]
                attempt_record_sha256 = {
                    **snapshot.attempt_record_sha256,
                    candidate_id: hashlib.sha256(record_bytes).hexdigest(),
                }
                best = select_best_attempt(attempts)
                account_projection = _account_projection(
                    key,
                    snapshot.policy,
                    attempts,
                    best,
                    attempt_record_sha256,
                )
                organizer_projection = _organizer_projection(
                    snapshot.organizer,
                    key,
                    snapshot.policy,
                    attempts,
                    best,
                )
                account_projection_bytes = _bounded_json_bytes(account_projection)
                organizer_projection_bytes = _bounded_json_bytes(organizer_projection)
                provisional_projection_bytes = None
                if first_attempts is not None:
                    provisional_projection_bytes = _bounded_json_bytes(
                        _provisional_projection(
                            snapshot.policy,
                            [*first_attempts, record],
                        )
                    )

                operations = _commit_operations(
                    key,
                    candidate_id,
                    record_bytes,
                    account_projection_bytes,
                    organizer_projection_bytes,
                    provisional_projection_bytes,
                )
                try:
                    committed = self.api.create_commit(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        revision="main",
                        parent_commit=snapshot.sha,
                        operations=operations,
                        commit_message=f"Accept DocSem test attempt {attempt_number}",
                    )
                except HfHubHTTPError as exc:
                    if _is_parent_conflict(exc) or _is_uncertain_commit_error(exc):
                        continue
                    raise
                committed_revision = getattr(committed, "oid", None)
                if not isinstance(committed_revision, str) or not committed_revision:
                    # A write with no authoritative revision is uncertain. Reloading
                    # lets the canonical submission hash reconcile it without a
                    # duplicate attempt.
                    continue
                self._verify_commit_readback(
                    committed_revision,
                    {
                        operation.path_in_repo: operation.path_or_fileobj
                        for operation in operations
                    },
                )
                self._verify_direct_child(snapshot.sha, committed_revision)
                committed_snapshot = self._load_snapshot_at(key, committed_revision)
                self._validate_complete_snapshot(committed_snapshot, key)
                confirmed = _find_submission(
                    committed_snapshot.attempts, submission_hash
                )
                if confirmed is None or confirmed.get("submission_id") != candidate_id:
                    raise _Unavailable()
                _require_retry_metadata(confirmed, normalized_metadata)
                return _accepted_receipt(confirmed)
            except HfHubHTTPError:
                raise TestStoreError(
                    "Test submission is temporarily unavailable."
                ) from None
            except _ReleaseClosed:
                raise TestStoreError("Test submissions are not open.") from None
            except _InvalidSubmission:
                raise TestStoreError("Test submission could not be accepted.") from None
            except TestCooldownError:
                raise
            except Exception:
                raise TestStoreError(
                    "Test submission is temporarily unavailable."
                ) from None
        raise TestStoreError("Test submission is temporarily unavailable.")

    def _verify_commit_readback(self, revision: str, expected: Mapping[str, bytes]):
        for path, raw in expected.items():
            if not isinstance(raw, bytes) or self._read_required(path, revision) != raw:
                raise _Unavailable()

    def _verify_direct_child(self, parent: str, revision: str):
        commits = self.api.list_repo_commits(
            self.repo_id,
            repo_type="dataset",
            revision=revision,
        )
        if (
            not isinstance(commits, list)
            or len(commits) < 2
            or getattr(commits[0], "commit_id", None) != revision
            or getattr(commits[1], "commit_id", None) != parent
        ):
            raise _Unavailable()

    def _validate_complete_snapshot(self, snapshot: _Snapshot, current_key: str):
        first_attempts = []
        expected_accounts = []
        seen = set()
        for account in snapshot.organizer.get("accounts", []):
            key = _projection_account_key(account, seen)
            seen.add(key)
            if key == current_key:
                attempts = list(snapshot.attempts)
            else:
                attempts, _ = self._load_account_attempts(
                    key, snapshot.sha, snapshot.policy
                )
            if not attempts:
                raise _Unavailable()
            for attempt in attempts:
                _require_complete_prediction_ids(
                    attempt.get("predictions"), snapshot.gold
                )
            best = select_best_attempt(attempts)
            expected_accounts.append(
                _organizer_account(key, snapshot.policy, attempts, best)
            )
            first_attempts.append(attempts[0])
        if snapshot.attempts and current_key not in seen:
            raise _Unavailable()
        expected_accounts.sort(key=lambda account: str(account["account_key"]))
        if snapshot.organizer != {
            **_release_state(snapshot.policy),
            "accounts": expected_accounts,
        }:
            raise _Unavailable()
        if snapshot.provisional != _provisional_projection(
            snapshot.policy, first_attempts
        ):
            raise _Unavailable()

    def _load_attempt_one_records(self, snapshot: _Snapshot) -> list[dict]:
        records = []
        seen = set()
        for account in snapshot.organizer.get("accounts", []):
            key = _projection_account_key(account, seen)
            seen.add(key)
            attempts, _ = self._load_account_attempts(
                key, snapshot.sha, snapshot.policy
            )
            if not attempts:
                raise _Unavailable()
            best = select_best_attempt(attempts)
            if account != _organizer_account(key, snapshot.policy, attempts, best):
                raise _Unavailable()
            records.append(attempts[0])
        return records

    def account_history(self, identity) -> list[dict]:
        try:
            key = _complete_identity_key(identity)
        except Exception:
            raise TestStoreError(
                "Test submission history is temporarily unavailable."
            ) from None
        try:
            self._require_config_paths()
            sha = self._head_sha()
            policy = _release_policy(self._read_required(self.release_config_path, sha))
            attempts, _ = self._load_account_attempts(key, sha, policy)
            return [dict(attempt) for attempt in attempts]
        except Exception:
            raise TestStoreError(
                "Test submission history is temporarily unavailable."
            ) from None

    def find_exact_attempt(self, identity, metadata, predictions) -> dict | None:
        """Return an immutable matching attempt before any repeat scoring occurs."""

        try:
            key = _complete_identity_key(identity)
            normalized_metadata = _submission_metadata(metadata)
            validate_test_predictions(predictions)
            normalized_predictions = _json_copy(predictions)
        except Exception:
            raise TestStoreError("Test submission could not be accepted.") from None
        try:
            snapshot = self._load_snapshot(key)
            _verify_release(snapshot, normalized_metadata)
            submission_hash = canonical_submission_hash(
                normalized_predictions,
                "test",
                snapshot.policy.release_id,
                identity,
                normalized_metadata,
            )
            existing = _find_submission(snapshot.attempts, submission_hash)
            if existing is not None:
                _require_retry_metadata(existing, normalized_metadata)
                self._validate_complete_snapshot(snapshot, key)
            else:
                _require_cooldown(snapshot.attempts, self.now_provider())
            return _json_copy(existing) if existing is not None else None
        except _InvalidSubmission:
            raise TestStoreError("Test submission could not be accepted.") from None
        except TestCooldownError:
            raise
        except Exception:
            raise TestStoreError(
                "Test submission is temporarily unavailable."
            ) from None

    def _load_snapshot(self, key: str) -> _Snapshot:
        self._require_config_paths()
        return self._load_snapshot_at(key, self._head_sha())

    def _load_snapshot_at(self, key: str, sha: str) -> _Snapshot:
        self._require_config_paths()
        release_raw = self._read_required(self.release_config_path, sha)
        policy = _release_policy(release_raw)
        if not policy.enabled:
            raise _ReleaseClosed()
        gold = self._read_required(self.gold_config_path, sha)
        _validate_gold(gold)
        attempts, attempt_record_sha256 = self._load_account_attempts(key, sha, policy)
        organizer = self._read_json_optional(
            ORGANIZER_PATH,
            sha,
            {**_release_state(policy), "accounts": []},
        )
        _validate_organizer_projection(organizer, policy)
        provisional_raw = self._read_required(PROVISIONAL_PATH, sha)
        provisional = _decode_json(provisional_raw)
        _validate_provisional_projection(provisional, policy)
        return _Snapshot(
            sha,
            policy,
            gold,
            tuple(attempts),
            attempt_record_sha256,
            organizer,
            provisional,
            provisional_raw,
        )

    def _require_config_paths(self) -> None:
        if not self.release_config_path or not self.gold_config_path:
            raise _Unavailable()

    def _head_sha(self) -> str:
        info = self.api.repo_info(
            self.repo_id,
            repo_type="dataset",
            revision="main",
        )
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not sha:
            raise _Unavailable()
        return sha

    def _load_account_attempts(
        self,
        key: str,
        sha: str,
        policy: TestReleasePolicy,
    ) -> tuple[list[dict], dict[str, str]]:
        account_path = f"projections/test/accounts/{key}.json"
        projection = self._read_json_optional(account_path, sha, None)
        if projection is None:
            return [], {}
        _validate_release_state(projection, policy)
        if (
            set(projection) != ACCOUNT_PROJECTION_FIELDS
            or projection.get("account_key") != key
        ):
            raise _Unavailable()
        references = projection.get("attempts")
        if not isinstance(references, list):
            raise _Unavailable()
        attempts = []
        attempt_record_sha256 = {}
        previous_submitted = None
        for expected_number, reference in enumerate(references, start=1):
            if not isinstance(reference, Mapping):
                raise _Unavailable()
            _validate_release_state(reference, policy)
            if set(reference) != ACCOUNT_ATTEMPT_REFERENCE_FIELDS:
                raise _Unavailable()
            submission_id = reference.get("submission_id")
            if not isinstance(submission_id, str) or not submission_id:
                raise _Unavailable()
            path = f"attempts/test/{key}/{submission_id}.json"
            record_raw = self._read_required(path, sha)
            record = _decode_json(record_raw)
            _validate_release_state(record, policy)
            _validate_scoring_state(record, policy)
            _validate_attempt_contract(record)
            record_sha256 = hashlib.sha256(record_raw).hexdigest()
            if (
                not isinstance(record, dict)
                or record.get("account_key") != key
                or record.get("submission_id") != submission_id
                or record.get("attempt_number") != expected_number
                or reference.get("attempt_number") != expected_number
                or reference.get("record_sha256") != record_sha256
            ):
                raise _Unavailable()
            submitted = _accepted_datetime(record.get("submitted_at"))
            if (
                previous_submitted is not None
                and (submitted - previous_submitted).total_seconds()
                < TEST_ATTEMPT_COOLDOWN_SECONDS
            ):
                raise _Unavailable()
            previous_submitted = submitted
            attempts.append(record)
            attempt_record_sha256[submission_id] = record_sha256
        if (
            not attempts
            or projection.get("best_submission_id")
            != select_best_attempt(attempts)["submission_id"]
        ):
            raise _Unavailable()
        return attempts, attempt_record_sha256

    def _read_required(self, path: str, sha: str) -> bytes:
        try:
            local_path = self.api.hf_hub_download(
                self.repo_id,
                path,
                repo_type="dataset",
                revision=sha,
            )
            return Path(local_path).read_bytes()
        except EntryNotFoundError:
            raise _Unavailable() from None

    def _read_json_required(self, path: str, sha: str):
        return _decode_json(self._read_required(path, sha))

    def _read_json_optional(self, path: str, sha: str, default):
        try:
            local_path = self.api.hf_hub_download(
                self.repo_id,
                path,
                repo_type="dataset",
                revision=sha,
            )
            return _decode_json(Path(local_path).read_bytes())
        except EntryNotFoundError:
            return default


def _complete_identity_key(identity) -> str:
    if not isinstance(identity, TestIdentity):
        raise _InvalidSubmission()
    try:
        TestIdentity(
            identity.identity_kind,
            identity.identity_subject,
            identity.hf_username,
            identity.contact_email,
            identity.email_verified,
        )
    except (ValueError, TestPolicyError):
        raise _InvalidSubmission()
    return account_key(identity)


def _submission_metadata(metadata) -> dict:
    if not isinstance(metadata, Mapping):
        raise _InvalidSubmission()
    try:
        task_manifest_path = metadata.get("scoring_task_manifest_path")
        if task_manifest_path != "test/tasks.jsonl":
            raise ValueError()
        return {
            "release_id": bounded_private_text(
                metadata.get("release_id"), "release_id"
            ),
            "task_manifest_sha256": sha256_digest(metadata.get("task_manifest_sha256")),
            "scoring_gold_sha256": sha256_digest(metadata.get("scoring_gold_sha256")),
            "scoring_private_revision": revision_digest(
                metadata.get("scoring_private_revision")
            ),
            "scoring_public_revision": revision_digest(
                metadata.get("scoring_public_revision")
            ),
            "scoring_public_repo_id": repository_id(
                metadata.get("scoring_public_repo_id")
            ),
            "scoring_task_manifest_path": task_manifest_path,
            "team": bounded_private_text(metadata.get("team"), "team"),
            "participant_names": bounded_private_text(
                metadata.get("participant_names"), "participant_names"
            ),
            "submission_name": bounded_private_text(
                metadata.get("submission_name"), "submission_name"
            ),
        }
    except ValueError:
        raise _InvalidSubmission() from None


def _validate_attempt_contract(record) -> None:
    try:
        if not isinstance(record, Mapping) or set(record) != ATTEMPT_RECORD_FIELDS:
            raise ValueError()
        for field in (
            "release_id",
            "identity_subject",
            "hf_username",
            "contact_email",
            "team",
            "participant_names",
            "submission_name",
        ):
            bounded_private_text(record.get(field), field)
        predictions = record.get("predictions")
        validate_test_predictions(predictions)
        _validate_metrics(record.get("metrics"), predictions)
        sha256_digest(record.get("submission_hash"))
        if (
            type(record.get("attempt_number")) is not int
            or record["attempt_number"] < 1
        ):
            raise ValueError()
        identity = TestIdentity(
            record.get("identity_kind"),
            record.get("identity_subject"),
            record.get("hf_username"),
            record.get("contact_email"),
            record.get("email_verified"),
        )
        if account_key(identity) != record.get("account_key"):
            raise ValueError()
        expected_hash = canonical_submission_hash(
            predictions,
            "test",
            record.get("release_id"),
            identity,
            record,
        )
        if record.get("submission_hash") != expected_hash:
            raise ValueError()
    except (ValueError, TestPolicyError):
        raise _Unavailable() from None


def _validate_metrics(metrics, predictions) -> None:
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "joint_accuracy",
        "answer_accuracy",
        "evidence_exact_match",
        "evidence_f1",
        "examples",
        "per_example",
    }:
        raise ValueError()
    for metric_name in (
        "joint_accuracy",
        "answer_accuracy",
        "evidence_exact_match",
        "evidence_f1",
    ):
        value = metrics.get(metric_name)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError()
    examples = metrics.get("examples")
    per_example = metrics.get("per_example")
    if (
        type(examples) is not int
        or examples <= 0
        or not isinstance(per_example, list)
        or len(per_example) != examples
        or examples != len(predictions)
    ):
        raise ValueError()
    expected_ids = {row["instance_id"].strip() for row in predictions}
    actual_ids = set()
    for row in per_example:
        if not isinstance(row, Mapping) or set(row) != {
            "instance_id",
            "answer_exact_match",
            "evidence_exact_match",
            "evidence_f1",
            "joint_exact_match",
        }:
            raise ValueError()
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in actual_ids:
            raise ValueError()
        instance_id = instance_id.strip()
        if not instance_id or instance_id in actual_ids:
            raise ValueError()
        actual_ids.add(instance_id)
        for metric_name in (
            "answer_exact_match",
            "evidence_exact_match",
            "evidence_f1",
            "joint_exact_match",
        ):
            value = row.get(metric_name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError()
    if actual_ids != expected_ids:
        raise ValueError()


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise _InvalidSubmission() from None


def _parse_datetime(value):
    if not isinstance(value, str) or not value.strip():
        raise _Unavailable()
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        raise _Unavailable() from None


def _accepted_datetime(value) -> dt.datetime:
    parsed = _parse_datetime(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _Unavailable()
    return parsed.astimezone(dt.timezone.utc)


def _release_policy(raw: bytes) -> TestReleasePolicy:
    value = _decode_json(raw)
    if not isinstance(value, dict):
        raise _Unavailable()
    try:
        enabled = value.get("enabled", True)
        return TestReleasePolicy(
            release_id=value.get("release_id"),
            task_manifest_sha256=value.get("task_manifest_sha256"),
            gold_sha256=value.get("gold_sha256"),
            open_at=_parse_datetime(value.get("open_at")) if enabled else None,
            close_at=_parse_datetime(value.get("close_at")) if enabled else None,
            enabled=enabled,
            max_attempts=value.get("max_attempts", 3),
        )
    except TestPolicyError:
        raise _Unavailable() from None


def _decode_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _Unavailable() from None


def _validate_gold(raw: bytes):
    try:
        rows = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _Unavailable() from None
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise _Unavailable()


def _require_complete_prediction_ids(predictions, gold_raw: bytes):
    try:
        gold_rows = [
            json.loads(line)
            for line in gold_raw.decode("utf-8").splitlines()
            if line.strip()
        ]
        gold_ids = [row["instance_id"] for row in gold_rows]
        prediction_ids = [row["instance_id"] for row in predictions]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise _Unavailable() from None
    if len(gold_ids) != len(set(gold_ids)) or prediction_ids != gold_ids:
        raise _InvalidSubmission()


def _require_open(policy: TestReleasePolicy, now):
    try:
        policy.require_open(now)
    except TestPolicyError as exc:
        if str(exc) == "Test submissions are not open.":
            raise _ReleaseClosed() from None
        raise _Unavailable() from None
    if now >= OFFICIAL_TEST_CLOSE_AT:
        raise _ReleaseClosed()


def _require_cooldown(attempts, now):
    if not attempts:
        return
    eligible_text = next_eligible_at(attempts)
    eligible = _parse_datetime(eligible_text)
    if (
        not isinstance(now, dt.datetime)
        or now.tzinfo is None
        or now.utcoffset() is None
    ):
        raise _InvalidSubmission()
    if now.astimezone(dt.timezone.utc) < eligible:
        raise TestCooldownError(eligible_text)


def _verify_release(snapshot: _Snapshot, metadata: Mapping):
    if metadata.get("release_id") != snapshot.policy.release_id:
        raise _Unavailable()
    if metadata.get("task_manifest_sha256") != snapshot.policy.task_manifest_sha256:
        raise _Unavailable()
    if metadata.get("scoring_gold_sha256") != snapshot.policy.gold_sha256:
        raise _Unavailable()
    if hashlib.sha256(snapshot.gold).hexdigest() != snapshot.policy.gold_sha256:
        raise _Unavailable()


def _find_submission(attempts, submission_hash):
    matches = [
        attempt
        for attempt in attempts
        if attempt.get("submission_hash") == submission_hash
    ]
    if len(matches) > 1:
        raise _Unavailable()
    return matches[0] if matches else None


def _require_retry_metadata(record: Mapping, metadata: Mapping) -> None:
    if any(
        record.get(field) != metadata.get(field)
        for field in IMMUTABLE_RETRY_METADATA_FIELDS
    ):
        raise _InvalidSubmission()


def _projection_account_key(account, seen) -> str:
    key = account.get("account_key") if isinstance(account, Mapping) else None
    if (
        not isinstance(key, str)
        or len(key) != 64
        or any(character not in "0123456789abcdef" for character in key)
        or key in seen
    ):
        raise _Unavailable()
    return key


def _accepted_at(now) -> str:
    if (
        not isinstance(now, dt.datetime)
        or now.tzinfo is None
        or now.utcoffset() is None
    ):
        raise _InvalidSubmission()
    return now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _attempt_record(
    *,
    identity: TestIdentity,
    key: str,
    metadata: Mapping,
    predictions,
    metrics,
    policy: TestReleasePolicy,
    submission_id: str,
    submission_hash: str,
    attempt_number: int,
    accepted_at: str,
) -> dict:
    return {
        **_release_state(policy),
        "submission_id": submission_id,
        "account_key": key,
        "identity_kind": identity.identity_kind,
        "identity_subject": identity.identity_subject,
        "hf_username": identity.hf_username,
        "contact_email": identity.contact_email,
        "email_verified": identity.email_verified,
        "scoring_gold_sha256": metadata["scoring_gold_sha256"],
        "scoring_private_revision": metadata["scoring_private_revision"],
        "scoring_public_revision": metadata["scoring_public_revision"],
        "scoring_public_repo_id": metadata["scoring_public_repo_id"],
        "scoring_task_manifest_path": metadata["scoring_task_manifest_path"],
        "team": metadata["team"],
        "participant_names": metadata["participant_names"],
        "submission_name": metadata["submission_name"],
        "submitted_at": accepted_at,
        "submission_hash": submission_hash,
        "attempt_number": attempt_number,
        "metrics": metrics,
        "predictions": predictions,
    }


def _account_projection(
    key, policy, attempts, best, attempt_record_sha256: Mapping[str, str]
) -> dict:
    return {
        **_release_state(policy),
        "account_key": key,
        "attempts": [
            {
                **_release_state(policy),
                "submission_id": attempt["submission_id"],
                "attempt_number": attempt["attempt_number"],
                "record_sha256": attempt_record_sha256[attempt["submission_id"]],
            }
            for attempt in attempts
        ],
        "best_submission_id": best["submission_id"],
    }


def _organizer_projection(current, key, policy, attempts, best) -> dict:
    accounts = [
        account
        for account in current.get("accounts", [])
        if isinstance(account, Mapping) and account.get("account_key") != key
    ]
    accounts.append(_organizer_account(key, policy, attempts, best))
    accounts.sort(key=lambda account: str(account["account_key"]))
    return {**_release_state(policy), "accounts": accounts}


def _organizer_account(key, policy, attempts, best) -> dict:
    return {
        **_release_state(policy),
        "account_key": key,
        "attempt_count": len(attempts),
        "best_submission_id": best["submission_id"],
        "identity_kind": best["identity_kind"],
        "identity_subject": best["identity_subject"],
        "hf_username": best["hf_username"],
        "contact_email": best["contact_email"],
        "email_verified": best["email_verified"],
        "team": best["team"],
        "participant_names": best["participant_names"],
        "submission_name": best["submission_name"],
        "submitted_at": best["submitted_at"],
        "attempt_number": best["attempt_number"],
        "metrics": best["metrics"],
    }


def _release_state(policy: TestReleasePolicy) -> dict:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "split": "test",
        "release_id": policy.release_id,
        "task_manifest_sha256": policy.task_manifest_sha256,
        "gold_sha256": policy.gold_sha256,
    }


def _validate_release_state(value, policy: TestReleasePolicy):
    if not isinstance(value, Mapping):
        raise _Unavailable()
    if any(
        value.get(field) != expected
        for field, expected in _release_state(policy).items()
    ):
        raise _Unavailable()


def _validate_scoring_state(value, policy: TestReleasePolicy):
    try:
        if sha256_digest(value.get("scoring_gold_sha256")) != policy.gold_sha256:
            raise ValueError()
        revision_digest(value.get("scoring_private_revision"))
        revision_digest(value.get("scoring_public_revision"))
        repository_id(value.get("scoring_public_repo_id"))
        if value.get("scoring_task_manifest_path") != "test/tasks.jsonl":
            raise ValueError()
    except ValueError:
        raise _Unavailable() from None


def _validate_organizer_projection(value, policy: TestReleasePolicy):
    _validate_release_state(value, policy)
    if set(value) != ORGANIZER_PROJECTION_FIELDS:
        raise _Unavailable()
    accounts = value.get("accounts")
    if not isinstance(accounts, list):
        raise _Unavailable()
    for account in accounts:
        _validate_release_state(account, policy)
        if not isinstance(account, Mapping) or set(account) != ORGANIZER_ACCOUNT_FIELDS:
            raise _Unavailable()


def _provisional_projection(policy: TestReleasePolicy, first_attempts) -> dict:
    ranked = rank_attempts(list(first_attempts)) if first_attempts else []
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "split": "test",
        "release_id": policy.release_id,
        "task_manifest_sha256": policy.task_manifest_sha256,
        "rows": [
            {
                "rank": rank,
                "hf_username": attempt["hf_username"],
                "team": attempt["team"],
            }
            for rank, attempt in enumerate(ranked, start=1)
        ],
    }


def _validate_provisional_projection(value, policy: TestReleasePolicy):
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema_version",
            "split",
            "release_id",
            "task_manifest_sha256",
            "rows",
        }
        or value.get("schema_version") != LEDGER_SCHEMA_VERSION
        or type(value.get("schema_version")) is not int
        or value.get("split") != "test"
        or value.get("release_id") != policy.release_id
        or value.get("task_manifest_sha256") != policy.task_manifest_sha256
        or not isinstance(value.get("rows"), list)
    ):
        raise _Unavailable()
    for expected_rank, row in enumerate(value["rows"], start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"rank", "hf_username", "team"}
            or type(row.get("rank")) is not int
            or row.get("rank") != expected_rank
            or not is_valid_public_text(row.get("hf_username"))
            or not is_valid_public_text(row.get("team"))
        ):
            raise _Unavailable()


def _json_bytes(value) -> bytes:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (serialized + "\n").encode("utf-8")


def _bounded_json_bytes(value) -> bytes:
    raw = _json_bytes(value)
    if len(raw) > MAX_LEDGER_FILE_BYTES:
        raise _InvalidSubmission()
    return raw


def _commit_operations(
    key,
    submission_id,
    record_bytes,
    account_projection_bytes,
    organizer_projection_bytes,
    provisional_projection_bytes=None,
):
    operations = [
        CommitOperationAdd(
            path_in_repo=f"attempts/test/{key}/{submission_id}.json",
            path_or_fileobj=record_bytes,
        ),
        CommitOperationAdd(
            path_in_repo=f"projections/test/accounts/{key}.json",
            path_or_fileobj=account_projection_bytes,
        ),
        CommitOperationAdd(
            path_in_repo=ORGANIZER_PATH,
            path_or_fileobj=organizer_projection_bytes,
        ),
    ]
    if provisional_projection_bytes is not None:
        operations.append(
            CommitOperationAdd(
                path_in_repo=PROVISIONAL_PATH,
                path_or_fileobj=provisional_projection_bytes,
            )
        )
    return operations


def _accepted_receipt(record) -> TestReceipt:
    return TestReceipt(
        True,
        int(record["attempt_number"]),
        str(record["submission_id"]),
        str(record["submitted_at"]),
        matched_attempt=_json_copy(record),
    )


def _is_parent_conflict(exc: HfHubHTTPError) -> bool:
    return getattr(getattr(exc, "response", None), "status_code", None) == 409


def _is_uncertain_commit_error(exc: HfHubHTTPError) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status is None or status >= 500
