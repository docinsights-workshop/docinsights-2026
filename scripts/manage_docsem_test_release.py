#!/usr/bin/env python3
"""Guard the DocSem held-out test release state transitions on private HF.

The command is read-only unless ``--activate`` or ``--close`` is selected.
Both mutations use one exact-parent commit and stop after any ambiguous result;
there is deliberately no automatic write retry.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import sys
from types import SimpleNamespace


PUBLIC_REPOSITORY = "amitbcp/docinsights-2026-shared-task-data"
PRIVATE_REPOSITORY = "amitbcp/docinsights-2026-shared-task-submissions"
PUBLIC_REVISION = "d9e1a394b46d2ac0a4dd87e12dd4a917a69f46e2"
RELEASE_ID = "docsem-test-a4205880-r1"
EXPECTED_OWNER = "amitbcp"
EXPECTED_ROLE = "write"
EXPECTED_COUNT = 1_730
TASK_MANIFEST_PATH = "test/tasks.jsonl"
PUBLIC_RELEASE_PATH = "test/release.json"
PRIVATE_RELEASE_PATH = "private/test_release.json"
PRIVATE_GOLD_PATH = "private/test_labels.jsonl"
PROVISIONAL_PATH = "projections/test/public_provisional.json"
PUBLIC_FINAL_PATH = "projections/test/public_final.json"
FINALIZATION_AUDIT_PATH = "private/test_finalization_audit.json"
TASK_MANIFEST_SHA256 = (
    "5fe8fbb8169b0c2b396fe155d263db36f4fa34b02a0cedd9075423b0bd3fc40d"
)
GOLD_SHA256 = (
    "67f91982261dbc38fc8ab0dea2402f470c8d0634f654dd8a1319e9699987a8f4"
)
SORTED_IDS_SHA256 = (
    "e30896a0540726d0dafab507d0c4d6408030ef86a702bcbd127526e49c06b3a9"
)
PDF_INVENTORY_SHA256 = (
    "3fce062d7485c44c3986df8945eac069703adfd8c52571578763458e11748238"
)
CLOSE_AT = os.environ.get("TEST_CLOSE_AT", "2026-09-11T12:00:00Z")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PATHS = 100_000
MAX_HISTORY = 10_000
MAX_PUBLIC_ROWS = 30_000

_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_BASE_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "counts",
        "sorted_ids_sha256",
        "task_manifest_sha256",
        "gold_sha256",
        "pdf_inventory_sha256",
        "visibility_audit",
        "enabled",
        "max_attempts",
        "feedback_policy",
        "finalized",
    }
)
_WINDOW_FIELDS = frozenset(
    {
        "open_at",
        "close_at",
        "public_revision",
        "public_repo_id",
        "task_manifest_path",
    }
)
_FINALIZATION_FIELDS = frozenset(
    {
        "finalized_at",
        "finalization_source_revision",
        "finalization_scorer_revision",
        "finalization_scorer_sha256",
        "final_projection_sha256",
        "finalization_audit_sha256",
    }
)
_PROVISIONAL_FIELDS = frozenset(
    {"schema_version", "split", "release_id", "task_manifest_sha256", "rows"}
)
_PROVISIONAL_ROW_FIELDS = frozenset({"rank", "hf_username", "team"})
_FINAL_ROW_FIELDS = frozenset(
    {
        "rank",
        "hf_username",
        "team",
        "submission_name",
        "selected_attempt",
        "joint_accuracy",
        "answer_accuracy",
        "evidence_f1",
    }
)
_ACTIVATION_FORBIDDEN_EXACT = frozenset(
    {
        "projections/test/organizer_leaderboard.json",
        PROVISIONAL_PATH,
        PUBLIC_FINAL_PATH,
        FINALIZATION_AUDIT_PATH,
    }
)
_ACTIVATION_FORBIDDEN_PREFIXES = (
    "attempts/test/",
    "projections/test/accounts/",
    "exclusions/test/",
    "adjudications/test/",
    "finalization/test/",
    "finalizations/test/",
)


class ReleaseError(RuntimeError):
    """A release invariant failed before any ambiguous write."""


class ParentConflictError(RuntimeError):
    """The exact-parent backend write was rejected as stale."""


class ConcurrentUpdateError(ReleaseError):
    """Another exact-parent writer advanced the private repository."""


class PublicationUncertainError(ReleaseError):
    """A mutation may have landed and must not be retried blindly."""


@dataclass(frozen=True, repr=False)
class _Snapshot:
    revision: str
    paths: frozenset[str]
    release: Mapping[str, object]
    release_bytes: bytes
    gold_bytes: bytes
    public_task_ids: tuple[str, ...]
    selected_bytes: Mapping[str, bytes]

    def __repr__(self) -> str:
        return f"_Snapshot(revision={self.revision!r}, paths={len(self.paths)}, sealed=True)"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError):
        raise ReleaseError("A release artifact is not canonical JSON.") from None


def _decode_json(raw: bytes, description: str):
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE_BYTES:
        raise ReleaseError(f"{description} is unavailable.")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        raise ReleaseError(f"{description} is invalid.") from None


def _decode_jsonl(raw: bytes, description: str) -> tuple[Mapping, ...]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE_BYTES:
        raise ReleaseError(f"{description} is unavailable.")
    try:
        lines = raw.splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError()
        rows = tuple(_decode_json(line, description) for line in lines)
    except ReleaseError:
        raise
    except Exception:
        raise ReleaseError(f"{description} is invalid.") from None
    if any(not isinstance(row, Mapping) for row in rows):
        raise ReleaseError(f"{description} is invalid.")
    return rows


def _parse_utc(value: object) -> dt.datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ReleaseError("The release window is invalid.")
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc
        )
    except ValueError:
        raise ReleaseError("The release window is invalid.") from None
    return parsed


def _require_utc_now(value: dt.datetime | None) -> dt.datetime:
    current = dt.datetime.now(dt.timezone.utc) if value is None else value
    if (
        not isinstance(current, dt.datetime)
        or current.tzinfo is None
        or current.utcoffset() != dt.timedelta(0)
    ):
        raise ReleaseError("The release check requires a UTC instant.")
    return current


def _safe_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _safe_public_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= 4096
        and _CONTROL.search(value) is None
        and not any(
            marker in value.replace("\\", "/").casefold()
            for marker in (
                "private/",
                "attempts/test/",
                "projections/test/accounts/",
                "exclusions/test/",
                "adjudications/test/",
            )
        )
    )


def _expected_public_release() -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "counts": {"tasks": EXPECTED_COUNT, "pdfs": EXPECTED_COUNT},
        "sorted_ids_sha256": SORTED_IDS_SHA256,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "pdf_inventory_sha256": PDF_INVENTORY_SHA256,
    }


def _validate_public_tasks(raw: bytes) -> tuple[str, ...]:
    if _sha256(raw) != TASK_MANIFEST_SHA256:
        raise ReleaseError("The pinned public task manifest digest differs.")
    rows = _decode_jsonl(raw, "The pinned public task manifest")
    identifiers: list[str] = []
    for row in rows:
        instance_id = row.get("instance_id")
        if (
            set(row) != {"instance_id", "user_query", "document_pdf"}
            or not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(row.get("user_query"), str)
            or not row["user_query"].strip()
            or row.get("document_pdf") != f"test/documents/{instance_id}.pdf"
        ):
            raise ReleaseError("The pinned public task manifest schema differs.")
        identifiers.append(instance_id)
    if (
        len(identifiers) != EXPECTED_COUNT
        or identifiers != sorted(set(identifiers))
        or _sha256("".join(f"{value}\n" for value in identifiers).encode("utf-8"))
        != SORTED_IDS_SHA256
    ):
        raise ReleaseError("The pinned public task inventory differs.")
    return tuple(identifiers)


def _validate_gold(raw: bytes, task_ids: Sequence[str]) -> None:
    if _sha256(raw) != GOLD_SHA256:
        raise ReleaseError("The private scoring-key digest differs.")
    rows = _decode_jsonl(raw, "The private scoring key")
    identifiers = []
    for row in rows:
        instance_id = row.get("instance_id")
        evidence = row.get("evidence")
        if (
            set(row) != {"instance_id", "answer", "evidence"}
            or not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) or not item for item in evidence)
            or len(set(evidence)) != len(evidence)
        ):
            raise ReleaseError("The private scoring-key schema differs.")
        identifiers.append(instance_id)
    if tuple(identifiers) != tuple(task_ids) or len(identifiers) != EXPECTED_COUNT:
        raise ReleaseError("The private scoring-key inventory differs.")


def _validate_release_anchors(release: Mapping) -> None:
    if (
        type(release.get("schema_version")) is not int
        or release.get("schema_version") != 1
        or release.get("release_id") != RELEASE_ID
        or release.get("counts")
        != {"tasks": EXPECTED_COUNT, "pdfs": EXPECTED_COUNT, "labels": EXPECTED_COUNT}
        or release.get("sorted_ids_sha256") != SORTED_IDS_SHA256
        or release.get("task_manifest_sha256") != TASK_MANIFEST_SHA256
        or release.get("gold_sha256") != GOLD_SHA256
        or release.get("pdf_inventory_sha256") != PDF_INVENTORY_SHA256
        or not isinstance(release.get("visibility_audit"), Mapping)
        or not release.get("visibility_audit")
        or type(release.get("enabled")) is not bool
        or type(release.get("finalized")) is not bool
        or type(release.get("max_attempts")) is not int
        or release.get("max_attempts") != 3
        or release.get("feedback_policy") != "first-attempt-only"
    ):
        raise ReleaseError("The private release anchors differ.")


def _validate_window(release: Mapping) -> tuple[dt.datetime, dt.datetime]:
    if (
        release.get("public_revision") != PUBLIC_REVISION
        or release.get("public_repo_id") != PUBLIC_REPOSITORY
        or release.get("task_manifest_path") != TASK_MANIFEST_PATH
        or release.get("close_at") != CLOSE_AT
    ):
        raise ReleaseError("The private release window anchors differ.")
    opened = _parse_utc(release.get("open_at"))
    closed = _parse_utc(release.get("close_at"))
    if opened >= closed:
        raise ReleaseError("The private release window is invalid.")
    return opened, closed


def _validate_provisional(raw: bytes, *, require_empty: bool = False) -> None:
    value = _decode_json(raw, "The provisional public projection")
    if (
        not isinstance(value, Mapping)
        or set(value) != _PROVISIONAL_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 3
        or value.get("split") != "test"
        or value.get("release_id") != RELEASE_ID
        or value.get("task_manifest_sha256") != TASK_MANIFEST_SHA256
        or not isinstance(value.get("rows"), list)
        or len(value["rows"]) > MAX_PUBLIC_ROWS
        or raw != _canonical_json(value)
    ):
        raise ReleaseError("The provisional public projection differs.")
    if require_empty and value["rows"]:
        raise ReleaseError("The activation projection is not empty.")
    for rank, row in enumerate(value["rows"], start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != _PROVISIONAL_ROW_FIELDS
            or type(row.get("rank")) is not int
            or row.get("rank") != rank
            or not _safe_public_text(row.get("hf_username"))
            or not _safe_public_text(row.get("team"))
        ):
            raise ReleaseError("The provisional public projection differs.")


def _validate_final_projection(raw: bytes) -> Mapping:
    value = _decode_json(raw, "The final public projection")
    if (
        not isinstance(value, Mapping)
        or set(value) != _PROVISIONAL_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 2
        or value.get("split") != "test"
        or value.get("release_id") != RELEASE_ID
        or value.get("task_manifest_sha256") != TASK_MANIFEST_SHA256
        or not isinstance(value.get("rows"), list)
        or len(value["rows"]) > MAX_PUBLIC_ROWS
        or raw != _canonical_json(value)
    ):
        raise ReleaseError("The final public projection differs.")
    for rank, row in enumerate(value["rows"], start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != _FINAL_ROW_FIELDS
            or type(row.get("rank")) is not int
            or row.get("rank") != rank
            or type(row.get("selected_attempt")) is not int
            or not 1 <= row["selected_attempt"] <= 3
            or not all(
                _safe_public_text(row.get(name))
                for name in ("hf_username", "team", "submission_name")
            )
        ):
            raise ReleaseError("The final public projection differs.")
        for name in ("joint_accuracy", "answer_accuracy", "evidence_f1"):
            metric = row.get(name)
            if type(metric) is not float or not 0.0 <= metric <= 1.0:
                raise ReleaseError("The final public projection differs.")
    return value


def _activation_inventory_is_empty(paths: Sequence[str]) -> bool:
    return not any(
        path in _ACTIVATION_FORBIDDEN_EXACT
        or any(path.startswith(prefix) for prefix in _ACTIVATION_FORBIDDEN_PREFIXES)
        for path in paths
    )


def _verify_state(snapshot: _Snapshot, expected: str, current: dt.datetime) -> None:
    release = snapshot.release
    _validate_release_anchors(release)
    if expected == "installed":
        if (
            set(release) != _BASE_RELEASE_FIELDS
            or release.get("enabled") is not False
            or release.get("finalized") is not False
            or not _activation_inventory_is_empty(snapshot.paths)
        ):
            raise ReleaseError("The private release is not in installed state.")
        return

    if expected not in {"open", "closed", "finalized"}:
        raise ReleaseError("The expected release state is invalid.")
    expected_fields = _BASE_RELEASE_FIELDS | _WINDOW_FIELDS
    if expected == "finalized":
        expected_fields |= _FINALIZATION_FIELDS
    if set(release) != expected_fields:
        raise ReleaseError("The private release schema differs.")
    _opened, closed = _validate_window(release)
    if PROVISIONAL_PATH not in snapshot.selected_bytes:
        raise ReleaseError("The provisional public projection is absent.")
    _validate_provisional(snapshot.selected_bytes[PROVISIONAL_PATH])

    if expected == "open":
        if (
            release.get("enabled") is not True
            or release.get("finalized") is not False
            or current >= closed
            or PUBLIC_FINAL_PATH in snapshot.paths
            or FINALIZATION_AUDIT_PATH in snapshot.paths
        ):
            raise ReleaseError("The private release is not in open state.")
        return
    if expected == "closed":
        if (
            release.get("enabled") is not False
            or release.get("finalized") is not False
            or current < closed
            or PUBLIC_FINAL_PATH in snapshot.paths
            or FINALIZATION_AUDIT_PATH in snapshot.paths
        ):
            raise ReleaseError("The private release is not in closed state.")
        return

    if release.get("enabled") is not False or release.get("finalized") is not True:
        raise ReleaseError("The private release is not in finalized state.")
    finalized = _parse_utc(release.get("finalized_at"))
    if finalized < closed or current < finalized:
        raise ReleaseError("The private release is not in finalized state.")
    for name in ("finalization_source_revision", "finalization_scorer_revision"):
        if _REVISION.fullmatch(str(release.get(name, ""))) is None:
            raise ReleaseError("The private finalization anchors differ.")
    for name in (
        "finalization_scorer_sha256",
        "final_projection_sha256",
        "finalization_audit_sha256",
    ):
        if _SHA256.fullmatch(str(release.get(name, ""))) is None:
            raise ReleaseError("The private finalization anchors differ.")
    if (
        PUBLIC_FINAL_PATH not in snapshot.selected_bytes
        or FINALIZATION_AUDIT_PATH not in snapshot.selected_bytes
    ):
        raise ReleaseError("The private finalization artifacts are absent.")
    final_bytes = snapshot.selected_bytes[PUBLIC_FINAL_PATH]
    audit_bytes = snapshot.selected_bytes[FINALIZATION_AUDIT_PATH]
    if (
        _sha256(final_bytes) != release["final_projection_sha256"]
        or _sha256(audit_bytes) != release["finalization_audit_sha256"]
    ):
        raise ReleaseError("The private finalization artifact digests differ.")
    final_projection = _validate_final_projection(final_bytes)
    audit = _decode_json(audit_bytes, "The private finalization audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version") != 1
        or audit.get("split") != "test"
        or audit.get("release_id") != RELEASE_ID
        or audit.get("task_manifest_sha256") != TASK_MANIFEST_SHA256
        or audit.get("gold_sha256") != GOLD_SHA256
        or audit.get("finalized_at") != release.get("finalized_at")
        or audit.get("close_at") != CLOSE_AT
        or audit.get("source_revision")
        != release.get("finalization_source_revision")
        or audit.get("scorer_revision")
        != release.get("finalization_scorer_revision")
        or audit.get("scorer_code_sha256")
        != release.get("finalization_scorer_sha256")
        or audit.get("public_projection_sha256")
        != release.get("final_projection_sha256")
        or type(audit.get("selected_account_count")) is not int
        or audit.get("selected_account_count") != len(final_projection["rows"])
    ):
        raise ReleaseError("The private finalization audit differs.")


def _validate_identity(value) -> None:
    if (
        getattr(value, "username", None) != EXPECTED_OWNER
        or getattr(value, "role", None) != EXPECTED_ROLE
    ):
        raise ReleaseError(
            "The authenticated Hugging Face identity is not the classic-write owner."
        )


def _validated_paths(values) -> frozenset[str]:
    try:
        paths = tuple(values)
    except TypeError:
        raise ReleaseError("The private repository inventory is unavailable.") from None
    if (
        not paths
        or len(paths) > MAX_PATHS
        or len(set(paths)) != len(paths)
        or any(not _safe_path(path) for path in paths)
    ):
        raise ReleaseError("The private repository inventory is invalid.")
    return frozenset(paths)


def _load_snapshot(hub, token: str, expected_head: str) -> _Snapshot:
    private_state = hub.repository_state(PRIVATE_REPOSITORY, token)
    if (
        getattr(private_state, "revision", None) != expected_head
        or getattr(private_state, "private", None) is not True
    ):
        raise ReleaseError("The private repository head or visibility differs.")
    public_state = hub.repository_state(PUBLIC_REPOSITORY, token)
    if (
        getattr(public_state, "revision", None) != PUBLIC_REVISION
        or getattr(public_state, "private", None) is not False
    ):
        raise ReleaseError("The public release head or visibility differs.")

    public_paths = _validated_paths(
        hub.list_paths(PUBLIC_REPOSITORY, PUBLIC_REVISION, token)
    )
    if not {TASK_MANIFEST_PATH, PUBLIC_RELEASE_PATH}.issubset(public_paths):
        raise ReleaseError("The pinned public release artifacts are absent.")
    public = hub.read_files(
        PUBLIC_REPOSITORY,
        PUBLIC_REVISION,
        (PUBLIC_RELEASE_PATH, TASK_MANIFEST_PATH),
        token,
    )
    if set(public) != {PUBLIC_RELEASE_PATH, TASK_MANIFEST_PATH}:
        raise ReleaseError("The pinned public release artifacts are incomplete.")
    public_release = _decode_json(public[PUBLIC_RELEASE_PATH], "The public release")
    if public_release != _expected_public_release():
        raise ReleaseError("The pinned public release anchors differ.")
    task_ids = _validate_public_tasks(public[TASK_MANIFEST_PATH])

    paths = _validated_paths(hub.list_paths(PRIVATE_REPOSITORY, expected_head, token))
    required = {PRIVATE_RELEASE_PATH, PRIVATE_GOLD_PATH}
    if not required.issubset(paths):
        raise ReleaseError("The private release artifacts are absent.")
    selected = required | {
        path
        for path in (PROVISIONAL_PATH, PUBLIC_FINAL_PATH, FINALIZATION_AUDIT_PATH)
        if path in paths
    }
    private = hub.read_files(
        PRIVATE_REPOSITORY, expected_head, tuple(sorted(selected)), token
    )
    if set(private) != selected:
        raise ReleaseError("The private release artifacts are incomplete.")
    release = _decode_json(private[PRIVATE_RELEASE_PATH], "The private release")
    if not isinstance(release, Mapping):
        raise ReleaseError("The private release is invalid.")
    if private[PRIVATE_RELEASE_PATH] != _canonical_json(release):
        raise ReleaseError("The private release is not canonical.")
    _validate_gold(private[PRIVATE_GOLD_PATH], task_ids)
    return _Snapshot(
        expected_head,
        paths,
        release,
        private[PRIVATE_RELEASE_PATH],
        private[PRIVATE_GOLD_PATH],
        task_ids,
        private,
    )


def _empty_provisional() -> dict[str, object]:
    return {
        "schema_version": 3,
        "split": "test",
        "release_id": RELEASE_ID,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "rows": [],
    }


def _recheck_after_conflict(hub, token: str) -> None:
    """Read the new head once; never convert it into an automatic retry."""
    try:
        state = hub.repository_state(PRIVATE_REPOSITORY, token)
        revision = getattr(state, "revision", None)
        if (
            getattr(state, "private", None) is not True
            or not isinstance(revision, str)
            or _REVISION.fullmatch(revision) is None
        ):
            raise ValueError()
    except Exception:
        # The outward result remains a sanitized conflict either way.
        return


def _require_write_boundary(hub, token: str, expected_private_head: str) -> None:
    """Recheck mutable identity/HEAD/visibility anchors before the only write."""

    _validate_identity(hub.identity(token))
    private_state = hub.repository_state(PRIVATE_REPOSITORY, token)
    if (
        getattr(private_state, "revision", None) != expected_private_head
        or getattr(private_state, "private", None) is not True
    ):
        raise ConcurrentUpdateError(
            "The private repository changed; re-evaluate from its new exact head."
        )
    public_state = hub.repository_state(PUBLIC_REPOSITORY, token)
    if (
        getattr(public_state, "revision", None) != PUBLIC_REVISION
        or getattr(public_state, "private", None) is not False
    ):
        raise ReleaseError("The public release head or visibility changed.")


def _commit_and_verify(
    *,
    hub,
    token: str,
    before: _Snapshot,
    files: Mapping[str, bytes],
    message: str,
    expected_state: str,
    now: dt.datetime,
) -> str:
    _require_write_boundary(hub, token, before.revision)
    try:
        returned = hub.create_commit(
            PRIVATE_REPOSITORY, before.revision, files, message, token
        )
    except ParentConflictError:
        _recheck_after_conflict(hub, token)
        raise ConcurrentUpdateError(
            "The private repository changed; re-evaluate from its new exact head."
        ) from None
    except Exception:
        raise PublicationUncertainError(
            "The private mutation may have landed; inspect the current head before any retry."
        ) from None

    try:
        if (
            not isinstance(returned, str)
            or _REVISION.fullmatch(returned) is None
            or returned == before.revision
        ):
            raise ValueError()
        state = hub.repository_state(PRIVATE_REPOSITORY, token)
        if (
            getattr(state, "revision", None) != returned
            or getattr(state, "private", None) is not True
        ):
            raise ValueError()
        ancestry = tuple(hub.ancestry(PRIVATE_REPOSITORY, returned, token))
        if (
            len(ancestry) < 2
            or len(ancestry) > MAX_HISTORY
            or ancestry[0] != returned
            or ancestry[1] != before.revision
            or len(set(ancestry)) != len(ancestry)
        ):
            raise ValueError()
        after = _load_snapshot(hub, token, returned)
        expected_paths = before.paths | frozenset(files)
        if after.paths != expected_paths:
            raise ValueError()
        for path, payload in files.items():
            if after.selected_bytes.get(path) != payload:
                raise ValueError()
        if after.gold_bytes != before.gold_bytes:
            raise ValueError()
        _verify_state(after, expected_state, now)
    except Exception:
        raise PublicationUncertainError(
            "The private mutation may have landed but exact readback failed; do not retry blindly."
        ) from None
    return returned


def _receipt(status: str, state: str, revision: str, previous: str | None = None):
    result = {
        "status": status,
        "state": state,
        "release_id": RELEASE_ID,
        "counts": {"tasks": EXPECTED_COUNT, "pdfs": EXPECTED_COUNT, "labels": EXPECTED_COUNT},
        "revision": revision,
        "public_revision": PUBLIC_REVISION,
        "writes": 0 if status == "verified" else 1,
        "retries": 0,
        "labels_exposed": False,
    }
    if previous is not None:
        result["previous_revision"] = previous
    return result


def manage_release(
    *,
    mode: str,
    expected_private_head: str,
    hub,
    token: str,
    now: dt.datetime | None = None,
    open_at: str | None = None,
    expected_state: str | None = None,
) -> dict[str, object]:
    """Verify or perform one guarded DocSem test release state transition."""

    if mode not in {"activate", "close", "verify"}:
        raise ReleaseError("The release-manager mode is invalid.")
    if _REVISION.fullmatch(str(expected_private_head or "")) is None:
        raise ReleaseError("An exact private repository head is required.")
    if not isinstance(token, str) or not token or len(token) > 4096 or "\n" in token:
        raise ReleaseError("A secure Hugging Face credential is required.")
    current = _require_utc_now(now)
    _validate_identity(hub.identity(token))
    snapshot = _load_snapshot(hub, token, expected_private_head)

    if mode == "verify":
        if open_at is not None or expected_state not in {
            "installed",
            "open",
            "closed",
            "finalized",
        }:
            raise ReleaseError("Verify-only requires one exact expected state.")
        _verify_state(snapshot, expected_state, current)
        return _receipt("verified", expected_state, expected_private_head)

    if expected_state is not None:
        raise ReleaseError("Mutation modes do not accept an expected-state override.")
    if mode == "activate":
        _verify_state(snapshot, "installed", current)
        opened = _parse_utc(open_at)
        closed = _parse_utc(CLOSE_AT)
        if opened <= current or opened >= closed:
            raise ReleaseError("Activation requires a fresh future UTC open time.")
        release = dict(snapshot.release)
        release.update(
            {
                "enabled": True,
                "open_at": open_at,
                "close_at": CLOSE_AT,
                "public_revision": PUBLIC_REVISION,
                "public_repo_id": PUBLIC_REPOSITORY,
                "task_manifest_path": TASK_MANIFEST_PATH,
            }
        )
        release_bytes = _canonical_json(release)
        projection_bytes = _canonical_json(_empty_provisional())
        _validate_provisional(projection_bytes, require_empty=True)
        files = {
            PRIVATE_RELEASE_PATH: release_bytes,
            PROVISIONAL_PATH: projection_bytes,
        }
        revision = _commit_and_verify(
            hub=hub,
            token=token,
            before=snapshot,
            files=files,
            message=f"Activate {RELEASE_ID}",
            expected_state="open",
            now=current,
        )
        return _receipt("activated", "open", revision, expected_private_head)

    if open_at is not None:
        raise ReleaseError("Close does not accept a new open time.")
    # The policy must still be the exact enabled/open schema, while the wall
    # clock is necessarily at or beyond its exclusive cutoff for this mode.
    _verify_state(
        snapshot,
        "open",
        min(current, _parse_utc(CLOSE_AT) - dt.timedelta(microseconds=1)),
    )
    _opened, closed = _validate_window(snapshot.release)
    if current < closed:
        raise ReleaseError("The exclusive test cutoff has not been reached.")
    release = dict(snapshot.release)
    release["enabled"] = False
    release_bytes = _canonical_json(release)
    revision = _commit_and_verify(
        hub=hub,
        token=token,
        before=snapshot,
        files={PRIVATE_RELEASE_PATH: release_bytes},
        message=f"Close {RELEASE_ID}",
        expected_state="closed",
        now=current,
    )
    return _receipt("closed", "closed", revision, expected_private_head)


class HuggingFaceBackend:
    """Small in-memory-read Hub adapter with exact-parent commit support."""

    @staticmethod
    def _imports():
        try:
            from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_url
            from huggingface_hub.utils import build_hf_headers, get_session
        except ImportError as exc:
            raise ReleaseError("The pinned Hugging Face client is unavailable.") from exc
        return CommitOperationAdd, HfApi, hf_hub_url, build_hf_headers, get_session

    def local_token(self) -> str:
        try:
            from huggingface_hub import get_token

            return get_token() or ""
        except Exception as exc:
            raise ReleaseError("Local Hugging Face authentication is unavailable.") from exc

    def identity(self, token: str):
        _operation, api_type, _url, _headers, _session = self._imports()
        try:
            value = api_type(token=token).whoami(token=token)
            return SimpleNamespace(
                username=value["name"], role=value["auth"]["accessToken"]["role"]
            )
        except Exception:
            raise ReleaseError("The Hugging Face identity could not be verified.") from None

    def repository_state(self, repository: str, token: str):
        _operation, api_type, _url, _headers, _session = self._imports()
        try:
            value = api_type(token=token).repo_info(
                repository, repo_type="dataset", revision="main", token=token
            )
            return SimpleNamespace(revision=value.sha, private=value.private)
        except Exception:
            raise ReleaseError("A Hugging Face repository state is unavailable.") from None

    def list_paths(self, repository: str, revision: str, token: str):
        _operation, api_type, _url, _headers, _session = self._imports()
        try:
            return tuple(
                api_type(token=token).list_repo_files(
                    repository,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                )
            )
        except Exception:
            raise ReleaseError("A Hugging Face repository inventory is unavailable.") from None

    def read_files(self, repository: str, revision: str, paths, token: str):
        _operation, _api, url_for, headers_for, session_for = self._imports()
        result = {}
        try:
            for path in paths:
                if not _safe_path(path):
                    raise ReleaseError("A requested Hugging Face path is unsafe.")
                response = session_for().get(
                    url_for(
                        repo_id=repository,
                        filename=path,
                        repo_type="dataset",
                        revision=revision,
                    ),
                    headers=headers_for(token=token),
                    stream=True,
                    timeout=(10, 60),
                )
                try:
                    response.raise_for_status()
                    payload = bytearray()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        payload.extend(chunk)
                        if len(payload) > MAX_FILE_BYTES:
                            raise ReleaseError("A Hugging Face artifact is too large.")
                    result[path] = bytes(payload)
                finally:
                    response.close()
            return result
        except ReleaseError:
            raise
        except Exception:
            raise ReleaseError("Hugging Face artifacts could not be read.") from None

    def create_commit(self, repository, expected_parent, files, message, token):
        operation_type, api_type, _url, _headers, _session = self._imports()
        operations = [
            operation_type(path_in_repo=path, path_or_fileobj=files[path])
            for path in sorted(files)
        ]
        try:
            result = api_type(token=token).create_commit(
                repo_id=repository,
                repo_type="dataset",
                revision="main",
                parent_commit=expected_parent,
                operations=operations,
                commit_message=message,
                token=token,
            )
            return getattr(result, "oid", None) or getattr(result, "commit_id", None)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {409, 412}:
                raise ParentConflictError("The exact parent changed.") from None
            raise

    def ancestry(self, repository: str, revision: str, token: str):
        _operation, api_type, _url, _headers, _session = self._imports()
        try:
            commits = api_type(token=token).list_repo_commits(
                repository,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
            return tuple(getattr(item, "commit_id", None) for item in commits)
        except Exception:
            raise ReleaseError("The private repository ancestry is unavailable.") from None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guard activation, close, and verification of the DocSem test release."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--activate", action="store_true")
    mode.add_argument("--close", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-private-head", required=True)
    parser.add_argument("--open-at")
    parser.add_argument(
        "--expect-state", choices=("installed", "open", "closed", "finalized")
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    hub=None,
    token: str | None = None,
    now: dt.datetime | None = None,
) -> int:
    args = parse_args(argv)
    backend = HuggingFaceBackend() if hub is None else hub
    credential = (
        token
        or os.environ.get("DOCSEM_PRIVATE_HF_TOKEN")
        or os.environ.get("DOCSEM_HF_WRITE_TOKEN")
        or os.environ.get("HF_WRITE_TOKEN")
        or (backend.local_token() if callable(getattr(backend, "local_token", None)) else "")
    )
    mode = "activate" if args.activate else "close" if args.close else "verify"
    try:
        result = manage_release(
            mode=mode,
            expected_private_head=args.expected_private_head,
            expected_state=args.expect_state,
            open_at=args.open_at,
            hub=backend,
            token=credential or "",
            now=now,
        )
    except ConcurrentUpdateError as exc:
        print(json.dumps({"status": "conflict", "error": str(exc)}), file=sys.stderr)
        return 4
    except PublicationUncertainError as exc:
        print(json.dumps({"status": "uncertain", "error": str(exc)}), file=sys.stderr)
        return 3
    except ReleaseError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
