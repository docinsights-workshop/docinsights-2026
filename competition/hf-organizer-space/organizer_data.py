"""Pinned, read-only reconstruction of the private DocSem test ledger.

The immutable attempt files are authoritative.  Account and organizer
projections are checked against reconstructed state and are never used as the
source of leaderboard rows.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi


_HF_SPACE = Path(__file__).resolve().parents[1] / "hf-space"
if str(_HF_SPACE) not in sys.path:
    sys.path.insert(0, str(_HF_SPACE))

from test_policy import TestPolicyError, select_best_attempt  # noqa: E402


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
MAX_ATTEMPTS = 3
MAX_ROWS_PER_ATTEMPT = 10_000

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ACCOUNT = _SHA256
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ATTEMPT_PATH = re.compile(
    r"attempts/test/(?P<account>[0-9a-f]{64})/(?P<record>[A-Za-z0-9][A-Za-z0-9._-]{0,127})\.json"
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
        if not _valid_attempt(record, state, account, submission_id):
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
            metrics = copy.deepcopy(attempt["metrics"])
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
                    "per_example": metrics.get("per_example", []),
                }
            )
    return rows


def _repository_id(value) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 256:
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    result = value.strip()
    if result.startswith("/") or ".." in result.split("/") or result.count("/") != 1:
        raise OrganizerDataError("Organizer snapshot is unavailable.")
    return result


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
        "schema_version": release.get("schema_version"),
        "split": "test",
        "release_id": release.get("release_id"),
        "task_manifest_sha256": release.get("task_manifest_sha256"),
        "gold_sha256": release.get("gold_sha256"),
    }
    split = release.get("split", "test")
    if (
        state["schema_version"] != 2
        or isinstance(state["schema_version"], bool)
        or split != "test"
        or not _nonempty(state["release_id"])
        or not _digest(state["task_manifest_sha256"])
        or not _digest(state["gold_sha256"])
        or release.get("max_attempts") != MAX_ATTEMPTS
        or isinstance(release.get("max_attempts"), bool)
        or release.get("feedback_policy") != "first-attempt-only"
        or not isinstance(release.get("enabled"), bool)
        or not isinstance(release.get("finalized"), bool)
    ):
        issues.add("release_invalid")
        return None
    return state


def _matches_state(value, state) -> bool:
    return (
        state is not None
        and isinstance(value, Mapping)
        and all(value.get(key) == expected for key, expected in state.items())
    )


def _valid_attempt(record, state, account, submission_id) -> bool:
    if not _matches_state(record, state):
        return False
    number = record.get("attempt_number")
    metrics = record.get("metrics")
    predictions = record.get("predictions")
    if (
        record.get("account_key") != account
        or record.get("submission_id") != submission_id
        or not isinstance(number, int)
        or isinstance(number, bool)
        or not 1 <= number <= MAX_ATTEMPTS
        or not _digest(record.get("submission_hash"))
        or record.get("scoring_gold_sha256") != state["gold_sha256"]
        or not _valid_timestamp(record.get("submitted_at"))
        or not _valid_metrics(metrics)
        or not isinstance(predictions, list)
        or len(predictions) > MAX_ROWS_PER_ATTEMPT
    ):
        return False
    for name in (
        "hf_subject",
        "hf_username",
        "verified_email",
        "scoring_private_revision",
        "scoring_public_revision",
        "scoring_public_repo_id",
        "scoring_task_manifest_path",
        "team",
        "participant_names",
        "submission_name",
    ):
        if not _nonempty(record.get(name)):
            return False
    expected_key = hashlib.sha256(str(record["hf_subject"]).encode("utf-8")).hexdigest()
    return expected_key == account


def _valid_metrics(metrics) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    for name in ("answer_accuracy", "evidence_f1"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
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
    return isinstance(per_example, list) and len(per_example) <= MAX_ROWS_PER_ATTEMPT


def _valid_timestamp(value) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        return False
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0)


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
            or reference.get("attempt_number") != attempt.get("attempt_number")
        ):
            return False
        record_digest = reference.get("record_sha256")
        if record_digest is not None and record_digest != loaded.sha256:
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
    return (
        _matches_state(record, state)
        and isinstance(record, Mapping)
        and record.get("record_id") == record_id
        and record.get("account_key") in grouped
        and _valid_timestamp(record.get("created_at"))
    )


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4096


def _digest(value) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None
