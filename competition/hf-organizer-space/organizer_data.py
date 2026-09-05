"""Pinned, read-only reconstruction of the private DocSem test ledger.

The immutable attempt files are authoritative.  Account and organizer
projections are checked against reconstructed state and are never used as the
source of leaderboard rows.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import sys
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi


_HF_SPACE = Path(__file__).resolve().parents[1] / "hf-space"
if str(_HF_SPACE) not in sys.path:
    sys.path.insert(0, str(_HF_SPACE))

from test_policy import (  # noqa: E402
    OAuthIdentity,
    TestPolicyError,
    canonical_submission_hash,
    select_best_attempt,
)


RELEASE_POLICY_PATH = "private/test_release.json"
ATTEMPT_PREFIX = "attempts/test/"
ACCOUNT_PROJECTION_PREFIX = "projections/test/accounts/"
ORGANIZER_PROJECTION_PATH = "projections/test/organizer_leaderboard.json"
EXCLUSION_PREFIX = "exclusions/test/"
ADJUDICATION_PREFIX = "adjudications/test/"

MAX_SNAPSHOT_FILES = 4096
MAX_SELECTED_FILES = 4096
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
MAX_SNAPSHOT_COMMITS = 10_000
MAX_ATTEMPTS = 3
MAX_ROWS_PER_ATTEMPT = 10_000
MAX_INSTANCE_ID_CHARACTERS = 256
MAX_ANSWER_CHARACTERS = 4096
MAX_EVIDENCE_IDS = 128
MAX_EVIDENCE_ID_CHARACTERS = 256
RELEASE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 2

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ACCOUNT = _SHA256
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_ATTEMPT_PATH = re.compile(
    r"attempts/test/(?P<account>[0-9a-f]{64})/(?P<record>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json"
)
_ACCOUNT_PATH = re.compile(r"projections/test/accounts/(?P<account>[0-9a-f]{64})\.json")
_EXCLUSION_PATH = re.compile(
    r"exclusions/test/(?P<record>[A-Za-z0-9][A-Za-z0-9._-]{0,127})\.json"
)
_ADJUDICATION_PATH = re.compile(
    r"adjudications/test/(?P<record>[A-Za-z0-9][A-Za-z0-9._-]{0,127})\.json"
)


class OrganizerDataError(RuntimeError):
    """Value-free failure raised for an unavailable or invalid private view."""


@dataclass(frozen=True, repr=False)
class _LoadedRecord:
    path: str
    sha256: str
    value: object = field(repr=False, compare=False)


@dataclass(frozen=True, repr=False)
class OrganizerSnapshot:
    """Sensitive in-memory snapshot whose repr exposes aggregate state only."""

    repo_id: str
    revision: str
    ancestor_revisions: tuple[str, ...] = field(repr=False, compare=False)
    release: object = field(repr=False, compare=False)
    attempts: tuple[_LoadedRecord, ...] = field(repr=False, compare=False)
    account_projections: tuple[_LoadedRecord, ...] = field(repr=False, compare=False)
    organizer_projection: _LoadedRecord | None = field(repr=False, compare=False)
    exclusions: tuple[_LoadedRecord, ...] = field(repr=False, compare=False)
    adjudications: tuple[_LoadedRecord, ...] = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            "OrganizerSnapshot("
            f"revision={self.revision!r}, "
            f"ancestor_count={len(self.ancestor_revisions)}, "
            f"attempt_count={len(self.attempts)}, "
            f"account_projection_count={len(self.account_projections)}, "
            f"exclusion_count={len(self.exclusions)}, "
            f"adjudication_count={len(self.adjudications)})"
        )


@dataclass(frozen=True, repr=False)
class AuditReport:
    valid: bool
    issue_codes: tuple[str, ...]
    revision: str
    account_count: int
    attempt_count: int
    exclusion_count: int
    adjudication_count: int

    def __repr__(self) -> str:
        return (
            "AuditReport("
            f"valid={self.valid!r}, "
            f"issue_count={len(self.issue_codes)}, "
            f"account_count={self.account_count}, "
            f"attempt_count={self.attempt_count}, "
            f"exclusion_count={self.exclusion_count}, "
            f"adjudication_count={self.adjudication_count})"
        )


def load_snapshot(repo_id, revision, token, *, api=None) -> OrganizerSnapshot:
    """Load only allowlisted test-ledger files from one exact private repo SHA."""

    try:
        repository = _repository_id(repo_id)
        pinned_revision = _pinned_revision(revision)
        read_token = _read_token(token)
        hub = api if api is not None else HfApi(token=read_token)
        info = hub.repo_info(
            repository,
            repo_type="dataset",
            revision=pinned_revision,
            token=read_token,
        )
        if (
            getattr(info, "sha", None) != pinned_revision
            or getattr(info, "private", None) is not True
        ):
            raise OrganizerDataError("Organizer snapshot is unavailable.")
        commit_infos = list(
            hub.list_repo_commits(
                repository,
                repo_type="dataset",
                revision=pinned_revision,
                token=read_token,
            )
        )
        ancestor_revisions = tuple(
            getattr(commit, "commit_id", None) for commit in commit_infos
        )
        if (
            not ancestor_revisions
            or len(ancestor_revisions) > MAX_SNAPSHOT_COMMITS
            or len(set(ancestor_revisions)) != len(ancestor_revisions)
            or pinned_revision not in ancestor_revisions
            or any(not _revision_digest(item) for item in ancestor_revisions)
        ):
            raise OrganizerDataError("Organizer snapshot is unavailable.")

        paths = list(
            hub.list_repo_files(
                repository,
                repo_type="dataset",
                revision=pinned_revision,
                token=read_token,
            )
        )
        selected = _selected_paths(paths)
        if RELEASE_POLICY_PATH not in selected:
            raise OrganizerDataError("Organizer snapshot is unavailable.")

        loaded = {}
        total_bytes = 0
        with tempfile.TemporaryDirectory(prefix="docsem-organizer-") as cache_name:
            cache = Path(cache_name)
            cache.chmod(0o700)
            for path in sorted(selected):
                local_name = hub.hf_hub_download(
                    repository,
                    path,
                    repo_type="dataset",
                    revision=pinned_revision,
                    token=read_token,
                    cache_dir=str(cache),
                )
                raw = _bounded_file(Path(local_name))
                total_bytes += len(raw)
                if total_bytes > MAX_SNAPSHOT_BYTES:
                    raise OrganizerDataError("Organizer snapshot is unavailable.")
                loaded[path] = _LoadedRecord(
                    path=path,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    value=_decode_object(raw),
                )

        release = loaded.pop(RELEASE_POLICY_PATH).value
        organizer = loaded.pop(ORGANIZER_PROJECTION_PATH, None)
        return OrganizerSnapshot(
            repo_id=repository,
            revision=pinned_revision,
            ancestor_revisions=ancestor_revisions,
            release=release,
            attempts=tuple(
                record
                for path, record in sorted(loaded.items())
                if _ATTEMPT_PATH.fullmatch(path)
            ),
            account_projections=tuple(
                record
                for path, record in sorted(loaded.items())
                if _ACCOUNT_PATH.fullmatch(path)
            ),
            organizer_projection=organizer,
            exclusions=tuple(
                record
                for path, record in sorted(loaded.items())
                if _EXCLUSION_PATH.fullmatch(path)
            ),
            adjudications=tuple(
                record
                for path, record in sorted(loaded.items())
                if _ADJUDICATION_PATH.fullmatch(path)
            ),
        )
    except OrganizerDataError:
        raise OrganizerDataError("Organizer snapshot is unavailable.") from None
    except Exception:
        raise OrganizerDataError("Organizer snapshot is unavailable.") from None


def verify_snapshot(snapshot) -> AuditReport:
    """Audit immutable records and require both stored projections to match."""

    if not isinstance(snapshot, OrganizerSnapshot):
        return AuditReport(False, ("snapshot_invalid",), "", 0, 0, 0, 0)
    try:
        return _verify_snapshot(snapshot)
    except Exception:
        return AuditReport(
            False,
            ("snapshot_invalid",),
            snapshot.revision,
            0,
            len(snapshot.attempts),
            len(snapshot.exclusions),
            len(snapshot.adjudications),
        )


def _verify_snapshot(snapshot: OrganizerSnapshot) -> AuditReport:
    """Internal verifier; the public boundary sanitizes all payload failures."""

    issues: set[str] = set()
    state = _release_state(snapshot.release, issues)
    grouped: dict[str, list[tuple[_LoadedRecord, Mapping]]] = defaultdict(list)
    submission_ids: set[str] = set()
    submission_hashes: set[str] = set()

    for loaded in snapshot.attempts:
        path_match = _ATTEMPT_PATH.fullmatch(loaded.path)
        record = loaded.value
        if path_match is None or not isinstance(record, Mapping):
            issues.add("attempt_invalid")
            continue
        account = path_match.group("account")
        submission_id = path_match.group("record")
        if not _valid_attempt(
            record,
            state,
            snapshot.release,
            snapshot.ancestor_revisions,
            account,
            submission_id,
        ):
            issues.add("attempt_invalid")
        if submission_id in submission_ids:
            issues.add("attempt_duplicate_id")
        submission_ids.add(submission_id)
        submission_hash = record.get("submission_hash")
        if isinstance(submission_hash, str):
            if submission_hash in submission_hashes:
                issues.add("attempt_duplicate_hash")
            submission_hashes.add(submission_hash)
        grouped[account].append((loaded, record))

    for account, entries in grouped.items():
        entries.sort(key=lambda item: _attempt_sort_number(item[1]))
        numbers = [item[1].get("attempt_number") for item in entries]
        if numbers != list(range(1, len(entries) + 1)) or len(entries) > MAX_ATTEMPTS:
            issues.add("attempt_numbering_invalid")

    projections = {}
    for loaded in snapshot.account_projections:
        match = _ACCOUNT_PATH.fullmatch(loaded.path)
        if match is None or match.group("account") in projections:
            issues.add("account_projection_invalid")
            continue
        projections[match.group("account")] = loaded.value

    if set(projections) != set(grouped):
        issues.add("account_projection_mismatch")
    for account in sorted(set(projections) & set(grouped)):
        if not _account_projection_matches(
            account, grouped[account], projections[account], state
        ):
            issues.add("account_projection_mismatch")

    if grouped:
        if snapshot.organizer_projection is None or not _organizer_projection_matches(
            grouped,
            snapshot.organizer_projection.value,
            state,
        ):
            issues.add("organizer_projection_mismatch")
    elif (
        snapshot.organizer_projection is not None
        and not _organizer_projection_matches(
            grouped,
            snapshot.organizer_projection.value,
            state,
        )
    ):
        issues.add("organizer_projection_mismatch")

    exclusion_accounts: set[str] = set()
    exclusion_ids: set[str] = set()
    for loaded in snapshot.exclusions:
        match = _EXCLUSION_PATH.fullmatch(loaded.path)
        record = loaded.value
        if (
            match is None
            or not _valid_audit_record(record, state, match.group("record"), grouped)
            or not isinstance(record, Mapping)
            or not _nonempty(record.get("reason_code"))
        ):
            issues.add("exclusion_invalid")
            continue
        if match.group("record") in exclusion_ids:
            issues.add("exclusion_duplicate")
        exclusion_ids.add(match.group("record"))
        exclusion_accounts.add(str(record["account_key"]))

    adjudication_ids: set[str] = set()
    for loaded in snapshot.adjudications:
        match = _ADJUDICATION_PATH.fullmatch(loaded.path)
        record = loaded.value
        valid = match is not None and _valid_audit_record(
            record,
            state,
            match.group("record") if match else "",
            grouped,
        )
        if valid and isinstance(record, Mapping):
            valid = _nonempty(record.get("action")) and _nonempty(
                record.get("reason_code")
            )
            submission_id = record.get("submission_id")
            if submission_id is not None:
                account_entries = grouped.get(str(record.get("account_key")), ())
                valid = valid and any(
                    entry.get("submission_id") == submission_id
                    for _, entry in account_entries
                )
        if not valid:
            issues.add("adjudication_invalid")
            continue
        record_id = match.group("record")
        if record_id in adjudication_ids:
            issues.add("adjudication_duplicate")
        adjudication_ids.add(record_id)

    return AuditReport(
        valid=not issues,
        issue_codes=tuple(sorted(issues)),
        revision=snapshot.revision,
        account_count=len(grouped),
        attempt_count=len(snapshot.attempts),
        exclusion_count=len(snapshot.exclusions),
        adjudication_count=len(snapshot.adjudications),
    )


def organizer_rows(snapshot) -> list[dict]:
    """Return detailed private attempt rows reconstructed from immutable files."""

    audit = verify_snapshot(snapshot)
    if not audit.valid:
        raise OrganizerDataError("Organizer snapshot failed integrity verification.")

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for loaded in snapshot.attempts:
        match = _ATTEMPT_PATH.fullmatch(loaded.path)
        grouped[match.group("account")].append(loaded.value)

    excluded = {
        str(loaded.value["account_key"])
        for loaded in snapshot.exclusions
        if isinstance(loaded.value, Mapping)
    }
    exclusion_counts = defaultdict(int)
    adjudication_counts = defaultdict(int)
    for loaded in snapshot.exclusions:
        exclusion_counts[str(loaded.value["account_key"])] += 1
    for loaded in snapshot.adjudications:
        adjudication_counts[str(loaded.value["account_key"])] += 1

    rows = []
    for account in sorted(grouped):
        attempts = sorted(
            grouped[account], key=lambda item: int(item["attempt_number"])
        )
        best = select_best_attempt(attempts)
        for attempt in attempts:
            metrics = attempt["metrics"]
            per_example = [
                {
                    "instance_id": row["instance_id"],
                    "answer_exact_match": row["answer_exact_match"],
                    "evidence_exact_match": row["evidence_exact_match"],
                    "evidence_f1": row["evidence_f1"],
                }
                for row in metrics["per_example"]
            ]
            rows.append(
                {
                    "account_key": account,
                    "submission_id": attempt["submission_id"],
                    "submission_hash": attempt["submission_hash"],
                    "attempt_number": attempt["attempt_number"],
                    "selected_best": attempt["submission_id"] == best["submission_id"],
                    "excluded": account in excluded,
                    "exclusion_count": exclusion_counts[account],
                    "adjudication_count": adjudication_counts[account],
                    "hf_subject": attempt["hf_subject"],
                    "hf_username": attempt["hf_username"],
                    "verified_email": attempt["verified_email"],
                    "team": attempt["team"],
                    "participant_names": attempt["participant_names"],
                    "submission_name": attempt["submission_name"],
                    "submitted_at": attempt["submitted_at"],
                    "release_id": attempt["release_id"],
                    "task_manifest_sha256": attempt["task_manifest_sha256"],
                    "gold_sha256": attempt["gold_sha256"],
                    "scoring_private_revision": attempt["scoring_private_revision"],
                    "scoring_public_revision": attempt["scoring_public_revision"],
                    "answer_accuracy": metrics["answer_accuracy"],
                    "evidence_f1": metrics["evidence_f1"],
                    "evidence_exact_match": metrics.get("evidence_exact_match"),
                    "examples": metrics.get("examples"),
                    "per_example": per_example,
                }
            )
    return rows


def _repository_id(value) -> str:
    if not _valid_repository_id(value):
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    return value.strip()


def _valid_repository_id(value) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 256:
        return False
    parts = value.strip().split("/")
    return len(parts) == 2 and all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) is not None
        for part in parts
    )


def _pinned_revision(value) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    return value


def _read_token(value) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    return value


def _safe_repo_path(path) -> bool:
    return (
        isinstance(path, str)
        and path
        and len(path) <= 512
        and not path.startswith("/")
        and "\\" not in path
        and ".." not in path.split("/")
        and all(part not in ("", ".") for part in path.split("/"))
    )


def _selected_paths(paths) -> set[str]:
    if len(paths) > MAX_SNAPSHOT_FILES or any(
        not _safe_repo_path(path) for path in paths
    ):
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    if len(set(paths)) != len(paths):
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    selected = set()
    governed_prefixes = (
        ATTEMPT_PREFIX,
        ACCOUNT_PROJECTION_PREFIX,
        "projections/test/organizer_",
        EXCLUSION_PREFIX,
        ADJUDICATION_PREFIX,
    )
    for path in paths:
        if path == RELEASE_POLICY_PATH or path == ORGANIZER_PROJECTION_PATH:
            selected.add(path)
            continue
        if any(
            pattern.fullmatch(path)
            for pattern in (
                _ATTEMPT_PATH,
                _ACCOUNT_PATH,
                _EXCLUSION_PATH,
                _ADJUDICATION_PATH,
            )
        ):
            selected.add(path)
            continue
        if path.startswith(governed_prefixes):
            raise OrganizerDataError("Organizer snapshot is unavailable.")
    if len(selected) > MAX_SELECTED_FILES:
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    return selected


def _bounded_file(path: Path) -> bytes:
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise OrganizerDataError("Organizer snapshot is unavailable.")
        raw = path.read_bytes()
    except OrganizerDataError:
        raise
    except OSError:
        raise OrganizerDataError("Organizer snapshot is unavailable.") from None
    if len(raw) > MAX_FILE_BYTES:
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    return raw


def _decode_object(raw: bytes):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise OrganizerDataError("Organizer snapshot is unavailable.")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    except OrganizerDataError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise OrganizerDataError("Organizer snapshot is unavailable.") from None
    if not isinstance(value, dict):
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    return value


def _release_state(release, issues: set[str]) -> dict | None:
    if not isinstance(release, Mapping):
        issues.add("release_invalid")
        return None
    state = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "split": "test",
        "release_id": release.get("release_id"),
        "task_manifest_sha256": release.get("task_manifest_sha256"),
        "gold_sha256": release.get("gold_sha256"),
    }
    split = release.get("split", "test")
    if (
        type(release.get("schema_version")) is not int
        or release.get("schema_version") != RELEASE_SCHEMA_VERSION
        or split != "test"
        or not _nonempty(state["release_id"])
        or not _digest(state["task_manifest_sha256"])
        or not _digest(state["gold_sha256"])
        or type(release.get("max_attempts")) is not int
        or release.get("max_attempts") != MAX_ATTEMPTS
        or release.get("feedback_policy") != "first-attempt-only"
        or not isinstance(release.get("enabled"), bool)
        or not isinstance(release.get("finalized"), bool)
    ):
        issues.add("release_invalid")
        return None
    has_window = any(release.get(name) is not None for name in ("open_at", "close_at"))
    if release.get("enabled") is True or release.get("finalized") is True or has_window:
        open_at = _parse_timestamp(release.get("open_at"))
        close_at = _parse_timestamp(release.get("close_at"))
        if (
            open_at is None
            or close_at is None
            or close_at <= open_at
            or not _revision_digest(release.get("public_revision"))
            or not _valid_repository_id(release.get("public_repo_id"))
            or release.get("task_manifest_path") != "test/tasks.jsonl"
        ):
            issues.add("release_invalid")
            return None
    return state


def _matches_state(value, state) -> bool:
    if state is None or not isinstance(value, Mapping):
        return False
    return all(
        value.get(key) == expected
        and (key != "schema_version" or type(value.get(key)) is int)
        for key, expected in state.items()
    )


def _valid_attempt(
    record, state, release, ancestor_revisions, account, submission_id
) -> bool:
    if not _matches_state(record, state):
        return False
    number = record.get("attempt_number")
    metrics = record.get("metrics")
    predictions = record.get("predictions")
    if (
        record.get("account_key") != account
        or record.get("submission_id") != submission_id
        or not _valid_uuid4(submission_id)
        or not isinstance(number, int)
        or isinstance(number, bool)
        or not 1 <= number <= MAX_ATTEMPTS
        or not _digest(record.get("submission_hash"))
        or record.get("scoring_gold_sha256") != state["gold_sha256"]
        or not _revision_digest(record.get("scoring_private_revision"))
        or record.get("scoring_private_revision") not in ancestor_revisions
        or not _revision_digest(record.get("scoring_public_revision"))
        or not isinstance(release, Mapping)
        or record.get("scoring_public_revision") != release.get("public_revision")
        or record.get("scoring_public_repo_id") != release.get("public_repo_id")
        or record.get("scoring_task_manifest_path") != release.get("task_manifest_path")
        or not _valid_timestamp(record.get("submitted_at"))
        or not _valid_metrics(metrics)
        or not _valid_predictions(predictions, metrics)
    ):
        return False
    submitted_at = _parse_timestamp(record.get("submitted_at"))
    open_at = _parse_timestamp(release.get("open_at"))
    close_at = _parse_timestamp(release.get("close_at"))
    if (
        submitted_at is None
        or open_at is None
        or close_at is None
        or not open_at <= submitted_at < close_at
    ):
        return False
    for name in (
        "hf_subject",
        "hf_username",
        "verified_email",
        "team",
        "participant_names",
        "submission_name",
    ):
        if not _nonempty(record.get(name)):
            return False
    expected_key = hashlib.sha256(str(record["hf_subject"]).encode("utf-8")).hexdigest()
    if expected_key != account:
        return False
    try:
        identity = OAuthIdentity(
            sub=record["hf_subject"],
            username=record["hf_username"],
            email=record["verified_email"],
        )
        expected_hash = canonical_submission_hash(
            predictions,
            record["split"],
            record["release_id"],
            identity,
        )
    except (TestPolicyError, TypeError, ValueError):
        return False
    return record.get("submission_hash") == expected_hash


def _valid_metrics(metrics) -> bool:
    aggregate_fields = {
        "answer_accuracy",
        "evidence_exact_match",
        "evidence_f1",
        "examples",
        "per_example",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != aggregate_fields:
        return False
    for name in ("answer_accuracy", "evidence_exact_match", "evidence_f1"):
        value = metrics.get(name)
        if type(value) is not float:
            return False
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            return False
    examples = metrics.get("examples")
    if (
        not isinstance(examples, int)
        or isinstance(examples, bool)
        or examples < 1
        or examples > MAX_ROWS_PER_ATTEMPT
    ):
        return False
    per_example = metrics.get("per_example", [])
    if not isinstance(per_example, list) or len(per_example) != examples:
        return False
    identifiers = set()
    sums = {
        "answer_accuracy": 0.0,
        "evidence_exact_match": 0.0,
        "evidence_f1": 0.0,
    }
    for row in per_example:
        detail_fields = {
            "instance_id",
            "answer_exact_match",
            "evidence_exact_match",
            "evidence_f1",
        }
        if not isinstance(row, Mapping) or set(row) != detail_fields:
            return False
        identifier = row.get("instance_id")
        if (
            not _bounded_string(identifier, MAX_INSTANCE_ID_CHARACTERS)
            or identifier in identifiers
        ):
            return False
        identifiers.add(identifier)
        for name in ("answer_exact_match", "evidence_exact_match", "evidence_f1"):
            value = row.get(name)
            if (
                type(value) is not float
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                return False
            if name != "evidence_f1" and value not in (0.0, 1.0):
                return False
            aggregate_name = "answer_accuracy" if name == "answer_exact_match" else name
            sums[aggregate_name] += float(value)
    for name, total in sums.items():
        if round(total / examples, 6) != float(metrics[name]):
            return False
    return True


def _valid_predictions(predictions, metrics) -> bool:
    if not isinstance(predictions, list) or not isinstance(metrics, Mapping):
        return False
    examples = metrics.get("examples")
    if len(predictions) != examples or len(predictions) > MAX_ROWS_PER_ATTEMPT:
        return False
    prediction_ids = set()
    for row in predictions:
        if not isinstance(row, Mapping) or set(row) != {
            "instance_id",
            "answer",
            "evidence",
        }:
            return False
        identifier = row.get("instance_id")
        answer = row.get("answer")
        evidence = row.get("evidence")
        if (
            not _bounded_string(identifier, MAX_INSTANCE_ID_CHARACTERS)
            or identifier in prediction_ids
            or not isinstance(answer, str)
            or len(answer) > MAX_ANSWER_CHARACTERS
            or not isinstance(evidence, list)
            or not 1 <= len(evidence) <= MAX_EVIDENCE_IDS
            or any(
                not _bounded_string(item, MAX_EVIDENCE_ID_CHARACTERS)
                for item in evidence
            )
        ):
            return False
        prediction_ids.add(identifier)
    metric_ids = {
        row.get("instance_id")
        for row in metrics.get("per_example", ())
        if isinstance(row, Mapping)
    }
    return prediction_ids == metric_ids


def _valid_timestamp(value) -> bool:
    return _parse_timestamp(value) is not None


def _parse_timestamp(value) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed


def _attempt_sort_number(record) -> int:
    value = record.get("attempt_number") if isinstance(record, Mapping) else None
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else MAX_ATTEMPTS + 1
    )


def _select_best(entries):
    try:
        return select_best_attempt([record for _, record in entries])
    except (TestPolicyError, TypeError, ValueError):
        return None


def _account_projection_matches(account, entries, projection, state) -> bool:
    if (
        not _matches_state(projection, state)
        or projection.get("account_key") != account
    ):
        return False
    references = projection.get("attempts")
    if not isinstance(references, list) or len(references) != len(entries):
        return False
    ordered = sorted(entries, key=lambda item: _attempt_sort_number(item[1]))
    for (loaded, attempt), reference in zip(ordered, references):
        if (
            not _matches_state(reference, state)
            or reference.get("submission_id") != attempt.get("submission_id")
            or type(reference.get("attempt_number")) is not int
            or reference.get("attempt_number") != attempt.get("attempt_number")
        ):
            return False
        record_digest = reference.get("record_sha256")
        if not _digest(record_digest) or record_digest != loaded.sha256:
            return False
    best = _select_best(ordered)
    return best is not None and projection.get("best_submission_id") == best.get(
        "submission_id"
    )


def _organizer_projection_matches(grouped, projection, state) -> bool:
    if not _matches_state(projection, state):
        return False
    accounts = projection.get("accounts")
    if not isinstance(accounts, list):
        return False
    by_account = {}
    for row in accounts:
        if not isinstance(row, Mapping) or not _matches_state(row, state):
            return False
        account = row.get("account_key")
        if not isinstance(account, str) or _ACCOUNT.fullmatch(account) is None:
            return False
        if account in by_account:
            return False
        by_account[account] = row
    if set(by_account) != set(grouped):
        return False
    for account, entries in grouped.items():
        best = _select_best(entries)
        if best is None:
            return False
        row = by_account[account]
        if (
            type(row.get("attempt_count")) is not int
            or type(row.get("attempt_number")) is not int
            or not _valid_metrics(row.get("metrics"))
        ):
            return False
        expected = {
            "attempt_count": len(entries),
            "best_submission_id": best.get("submission_id"),
            "hf_subject": best.get("hf_subject"),
            "hf_username": best.get("hf_username"),
            "verified_email": best.get("verified_email"),
            "team": best.get("team"),
            "participant_names": best.get("participant_names"),
            "submission_name": best.get("submission_name"),
            "submitted_at": best.get("submitted_at"),
            "attempt_number": best.get("attempt_number"),
            "metrics": best.get("metrics"),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            return False
    return True


def _valid_audit_record(record, state, record_id, grouped) -> bool:
    if not isinstance(record, Mapping):
        return False
    account = record.get("account_key")
    return (
        _matches_state(record, state)
        and record.get("record_id") == record_id
        and isinstance(account, str)
        and _ACCOUNT.fullmatch(account) is not None
        and account in grouped
        and _valid_timestamp(record.get("created_at"))
    )


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4096


def _bounded_string(value, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _valid_uuid4(value) -> bool:
    if not isinstance(value, str) or _UUID.fullmatch(value) is None:
        return False
    try:
        return str(uuid.UUID(value)) == value and uuid.UUID(value).version == 4
    except ValueError:
        return False


def _digest(value) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _revision_digest(value) -> bool:
    return isinstance(value, str) and _REVISION.fullmatch(value) is not None
