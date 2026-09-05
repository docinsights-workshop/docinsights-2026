"""Atomic private-Hub persistence for DocSem test attempts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from test_contract import (
    MAX_LEDGER_FILE_BYTES,
    bounded_private_text,
    repository_id,
    revision_digest,
    sha256_digest,
    validate_test_predictions,
)
from test_policy import (
    OAuthIdentity,
    TestPolicyError,
    TestReleasePolicy,
    account_key,
    canonical_submission_hash,
    select_best_attempt,
)


ORGANIZER_PATH = "projections/test/organizer_leaderboard.json"
MAX_COMMIT_ATTEMPTS = 5


class TestStoreError(RuntimeError):
    """Value-free public failure raised by private test persistence."""


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
        except Exception:
            raise TestStoreError("Test submission could not be accepted.") from None

        candidate_id = str(uuid.uuid4())
        for _ in range(MAX_COMMIT_ATTEMPTS):
            try:
                snapshot = self._load_snapshot(key)
                plan_now = self.now_provider()
                _require_open(snapshot.policy, plan_now)
                _verify_release(snapshot, normalized_metadata)
                submission_hash = canonical_submission_hash(
                    normalized_predictions,
                    "test",
                    snapshot.policy.release_id,
                    identity,
                )
                existing = _find_submission(snapshot.attempts, submission_hash)
                if existing is not None:
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

                # Take a new authoritative instant for every CAS attempt. A request
                # that was open while planning but crosses the deadline must never
                # materialize an attempt record or reach create_commit().
                commit_now = self.now_provider()
                _require_open(snapshot.policy, commit_now)
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

                operations = _commit_operations(
                    key,
                    candidate_id,
                    record_bytes,
                    account_projection_bytes,
                    organizer_projection_bytes,
                )
                try:
                    self.api.create_commit(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        revision="main",
                        parent_commit=snapshot.sha,
                        operations=operations,
                        commit_message=f"Accept DocSem test attempt {attempt_number}",
                    )
                except HfHubHTTPError as exc:
                    if _is_parent_conflict(exc):
                        continue
                    raise
                return TestReceipt(True, attempt_number, candidate_id, accepted_at)
            except HfHubHTTPError:
                raise TestStoreError(
                    "Test submission is temporarily unavailable."
                ) from None
            except _ReleaseClosed:
                raise TestStoreError("Test submissions are not open.") from None
            except _InvalidSubmission:
                raise TestStoreError("Test submission could not be accepted.") from None
            except Exception:
                raise TestStoreError(
                    "Test submission is temporarily unavailable."
                ) from None
        raise TestStoreError("Test submission is temporarily unavailable.")

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
            )
            existing = _find_submission(snapshot.attempts, submission_hash)
            return _json_copy(existing) if existing is not None else None
        except Exception:
            raise TestStoreError(
                "Test submission is temporarily unavailable."
            ) from None

    def _load_snapshot(self, key: str) -> _Snapshot:
        self._require_config_paths()
        sha = self._head_sha()
        release_raw = self._read_required(self.release_config_path, sha)
        gold = self._read_required(self.gold_config_path, sha)
        policy = _release_policy(release_raw)
        _validate_gold(gold)
        attempts, attempt_record_sha256 = self._load_account_attempts(key, sha, policy)
        organizer = self._read_json_optional(
            ORGANIZER_PATH,
            sha,
            {**_release_state(policy), "accounts": []},
        )
        _validate_organizer_projection(organizer, policy)
        return _Snapshot(
            sha,
            policy,
            gold,
            tuple(attempts),
            attempt_record_sha256,
            organizer,
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
        if projection.get("account_key") != key:
            raise _Unavailable()
        references = projection.get("attempts")
        if not isinstance(references, list):
            raise _Unavailable()
        attempts = []
        attempt_record_sha256 = {}
        for expected_number, reference in enumerate(references, start=1):
            if not isinstance(reference, Mapping):
                raise _Unavailable()
            _validate_release_state(reference, policy)
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
            attempts.append(record)
            attempt_record_sha256[submission_id] = record_sha256
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
    if not isinstance(identity, OAuthIdentity):
        raise _InvalidSubmission()
    try:
        bounded_private_text(identity.sub, "hf_subject")
        bounded_private_text(identity.username, "hf_username")
        bounded_private_text(identity.email, "verified_email")
    except ValueError:
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
        for field in (
            "release_id",
            "hf_subject",
            "hf_username",
            "verified_email",
            "team",
            "participant_names",
            "submission_name",
        ):
            bounded_private_text(record.get(field), field)
        validate_test_predictions(record.get("predictions"))
    except ValueError:
        raise _Unavailable() from None


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


def _require_open(policy: TestReleasePolicy, now):
    try:
        policy.require_open(now)
    except TestPolicyError as exc:
        if str(exc) == "Test submissions are not open.":
            raise _ReleaseClosed() from None
        raise _Unavailable() from None


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
    identity: OAuthIdentity,
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
        "hf_subject": identity.sub,
        "hf_username": identity.username,
        "verified_email": identity.email,
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
    accounts.append(
        {
            **_release_state(policy),
            "account_key": key,
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
    accounts.sort(key=lambda account: str(account["account_key"]))
    return {**_release_state(policy), "accounts": accounts}


def _release_state(policy: TestReleasePolicy) -> dict:
    return {
        "schema_version": 2,
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
    accounts = value.get("accounts")
    if not isinstance(accounts, list):
        raise _Unavailable()
    for account in accounts:
        _validate_release_state(account, policy)


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
):
    return [
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
