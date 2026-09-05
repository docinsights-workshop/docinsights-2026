#!/usr/bin/env python3
"""Stage and publish only the disabled private half of one DocSem release.

The default operation is a read-only dry run.  This module has no GitHub,
public-upload, PDF-copy, activation, validation-label, or submission write path.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence

from prepare_docsem_test_release import (
    MAX_PDF_BYTES,
    MAX_PUBLIC_CHECKSUM_BYTES,
    MAX_PUBLIC_MANIFEST_BYTES,
    MAX_PUBLIC_TASKS_BYTES,
    ValidationError,
    _read_bounded_regular_file,
    _validate_labels,
    _visibility_audit_contract,
    _write_new_file,
)
from publish_docsem_test_release import (
    Artifact,
    HistorySnapshot,
    HfRepositoryState,
    HuggingFaceBackend as _GuardedHuggingFaceBackend,
    MAX_HISTORY_COMMITS,
    MAX_HISTORY_METADATA_BYTES,
    ReleaseError,
    RemoteMovedError,
    _canonical_json,
    _parse_json_document,
    _parse_jsonl,
    _read_regular_file,
    _safe_relative_path,
    _scan_public_history,
    _sha256,
    _state_for_namespace,
    _validate_revision,
    _validated_hf_state,
    _walk_exact,
)


RELEASE_ID = "docsem-test-a4205880-r1"
PUBLIC_HF_REPOSITORY = "amitbcp/docinsights-2026-shared-task-data"
PRIVATE_HF_REPOSITORY = "amitbcp/docinsights-2026-shared-task-submissions"
PUBLIC_REVISION = "d9e1a394b46d2ac0a4dd87e12dd4a917a69f46e2"
PUBLIC_STAGE = Path("/private/tmp/docsem-public-test-stage-a4205880-r1")
PRIVATE_LABEL_SOURCE = Path(
    "/private/tmp/docsem-private-source-a4205880-r1/labels.jsonl"
)
PRIVATE_LABELS_SHA256 = (
    "67f91982261dbc38fc8ab0dea2402f470c8d0634f654dd8a1319e9699987a8f4"
)
EXPECTED_COUNT = 1_730
SORTED_IDS_SHA256 = "e30896a0540726d0dafab507d0c4d6408030ef86a702bcbd127526e49c06b3a9"
TASK_MANIFEST_SHA256 = (
    "5fe8fbb8169b0c2b396fe155d263db36f4fa34b02a0cedd9075423b0bd3fc40d"
)
PDF_INVENTORY_SHA256 = (
    "3fce062d7485c44c3986df8945eac069703adfd8c52571578763458e11748238"
)
CONFIRMATION = "PUBLISH_DISABLED_PRIVATE_TEST_RELEASE"
LEGACY_HISTORY_POLICY = "legacy-private-label-cycle-v1"
LEGACY_HISTORY_CONFIRMATION = "ACKNOWLEDGE_RETAINED_LEGACY_PRIVATE_LABEL_HISTORY"
LEGACY_HISTORY_METADATA_SHA256 = (
    "17107b3da2db03b98356ba11ead006be38d85dad06f8f9b9e20a3bd02a5d1215"
)
EXPECTED_OWNER = "amitbcp"
EXPECTED_ROLE = "write"
MAX_PRIVATE_LABEL_BYTES = 64 * 1024 * 1024
MAX_PRIVATE_RELEASE_BYTES = 2 * 1024 * 1024
MAX_GIT_PATH_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_GIT_CONTROL_OUTPUT_BYTES = 4 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")

_PRIVATE_MANIFEST_KEYS = frozenset(
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
_PUBLIC_METADATA_PATHS = (
    "test/SHA256SUMS",
    "test/release.json",
    "test/tasks.jsonl",
)
_PRIVATE_PATHS = (
    "private/test_labels.jsonl",
    "private/test_release.json",
)

__all__ = [
    "Artifact",
    "HistorySnapshot",
    "HfRepositoryState",
    "ReleaseError",
    "RemoteMovedError",
    "PublicationUncertainError",
    "ReleaseConfig",
    "HfIdentity",
    "RemoteFile",
    "HuggingFaceBackend",
    "prepare_stage",
    "run_private_continuation",
    "main",
]


class PublicationUncertainError(ReleaseError):
    """A private write may have landed and must never be retried automatically."""


@dataclass(frozen=True)
class ReleaseConfig:
    public_stage: Path
    private_label_source: Path
    private_stage: Path
    private_hf_base: str


@dataclass(frozen=True)
class HfIdentity:
    username: str
    role: str


@dataclass(frozen=True)
class RemoteFile:
    size: int
    sha256: str | None


@dataclass(frozen=True, repr=False)
class PrivateHistoryEvent:
    revision: str
    parents: tuple[str, ...]
    timestamp: str
    status: str
    path: str
    subject: str

    def __repr__(self) -> str:
        return (
            "PrivateHistoryEvent(revision=<sanitized>, status="
            f"{self.status!r}, path=<sanitized>)"
        )


@dataclass(frozen=True, repr=False)
class PrivateHistoryAudit:
    head: str
    events: tuple[PrivateHistoryEvent, ...]
    reachable: frozenset[str]
    shallow: bool
    blob_objects_fetched: bool
    parent_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"PrivateHistoryAudit(event_count={len(self.events)}, sanitized=True)"


def _event_metadata(event: PrivateHistoryEvent) -> dict[str, object]:
    return {
        "revision": event.revision,
        "parents": list(event.parents),
        "timestamp": event.timestamp,
        "status": event.status,
        "path": event.path,
    }


@dataclass(frozen=True, repr=False)
class _PublicSnapshot:
    manifest: Mapping[str, object]
    task_ids: tuple[str, ...]
    paths: frozenset[str]
    metadata: Mapping[str, bytes]
    digests: Mapping[str, str]
    sizes: Mapping[str, int]
    fingerprint: tuple[tuple[str, int, int, int, int, int], ...]

    def __repr__(self) -> str:
        return f"_PublicSnapshot(count={len(self.task_ids)}, sealed=True)"


@dataclass(frozen=True, repr=False)
class _PrivateSnapshot:
    artifacts: Mapping[str, Artifact]
    manifest: Mapping[str, object]

    def __repr__(self) -> str:
        return "_PrivateSnapshot(counts=sanitized, sealed=True)"


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ReleaseError(
            "A local release path could not be inspected safely."
        ) from exc
    return True


def _entry_fingerprint(path: Path) -> tuple[int, int, int, int, int, int]:
    try:
        item = path.lstat()
    except OSError as exc:
        raise ReleaseError("A release input is absent or unreadable.") from exc
    return (
        item.st_mode,
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _tree_fingerprint(root: Path) -> tuple[tuple[str, int, int, int, int, int], ...]:
    files, directories = _walk_exact(root, "Public test staging")
    result = []
    for relative in sorted(files | directories):
        mode, device, inode, size, mtime, ctime = _entry_fingerprint(root / relative)
        result.append((relative, mode, device, inode, size, mtime ^ ctime))
    return tuple(result)


def _validate_local_config(config: ReleaseConfig) -> None:
    if not isinstance(config, ReleaseConfig):
        raise ReleaseError("The private continuation configuration is invalid.")
    if Path(config.public_stage).absolute() != PUBLIC_STAGE.absolute():
        raise ReleaseError("The approved public stage was not selected.")
    if Path(config.private_label_source).absolute() != PRIVATE_LABEL_SOURCE.absolute():
        raise ReleaseError("The approved private label source was not selected.")
    stage = Path(config.private_stage).absolute()
    public_root = Path(config.public_stage).absolute()
    private_source = Path(config.private_label_source).absolute()
    input_roots = (public_root, private_source.parent, private_source)
    if any(
        stage == input_path
        or stage in input_path.parents
        or input_path in stage.parents
        for input_path in input_roots
    ):
        raise ReleaseError("The private stage is not separate from its inputs.")
    try:
        parent_mode = stage.parent.lstat().st_mode
    except OSError as exc:
        raise ReleaseError(
            "The private staging parent is absent or unreadable."
        ) from exc
    if not stat.S_ISDIR(parent_mode) or stat.S_ISLNK(parent_mode):
        raise ReleaseError("The private staging parent is not a real directory.")


def _validate_remote_config(config: ReleaseConfig) -> None:
    _validate_local_config(config)
    _validate_revision(config.private_hf_base, "Private Hugging Face base")


def _parse_checksums(payload: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ReleaseError("The public checksum manifest is malformed.") from exc
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ReleaseError("The public checksum manifest is malformed.")
        value, path = line[:64], line[66:]
        if not _SHA.fullmatch(value) or not _safe_relative_path(path) or path in result:
            raise ReleaseError("The public checksum manifest is malformed.")
        result[path] = value
    expected = b"".join(
        f"{result[path]}  {path}\n".encode("ascii") for path in sorted(result)
    )
    if not result or payload != expected:
        raise ReleaseError("The public checksum manifest is not canonical.")
    return result


def _audit_local_public(
    root: Path,
    public_auditor: Callable[[Path], Mapping[str, object]] | None,
) -> _PublicSnapshot:
    initial = _tree_fingerprint(root)
    audited = None
    if public_auditor is not None:
        try:
            audited = public_auditor(root)
        except Exception as exc:
            raise ReleaseError("The pinned public staging audit failed.") from exc
    files, directories = _walk_exact(root, "Public test staging")
    if directories != {"test", "test/documents"}:
        raise ReleaseError(
            "The public staging directory inventory differs from its anchor."
        )
    pdf_paths = sorted(path for path in files if path.startswith("test/documents/"))
    expected_files = set(_PUBLIC_METADATA_PATHS) | set(pdf_paths)
    if files != expected_files or len(pdf_paths) != EXPECTED_COUNT:
        raise ReleaseError("The public staging file inventory differs from its anchor.")

    try:
        task_bytes = _read_bounded_regular_file(
            root / "test/tasks.jsonl", MAX_PUBLIC_TASKS_BYTES, "Public task manifest"
        )
        release_bytes = _read_bounded_regular_file(
            root / "test/release.json",
            MAX_PUBLIC_MANIFEST_BYTES,
            "Public release manifest",
        )
        checksum_bytes = _read_bounded_regular_file(
            root / "test/SHA256SUMS", MAX_PUBLIC_CHECKSUM_BYTES, "Public checksums"
        )
    except ValidationError as exc:
        raise ReleaseError("The public staging metadata is unsafe.") from exc
    manifest = _parse_json_document(release_bytes, "Public release manifest")
    expected_manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "counts": {"tasks": EXPECTED_COUNT, "pdfs": EXPECTED_COUNT},
        "sorted_ids_sha256": SORTED_IDS_SHA256,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "pdf_inventory_sha256": PDF_INVENTORY_SHA256,
    }
    if manifest != expected_manifest or (
        audited is not None and audited != expected_manifest
    ):
        raise ReleaseError(
            "The public release manifest differs from its approved anchor."
        )
    task_rows = _parse_jsonl(task_bytes, "Public task manifest")
    task_ids: list[str] = []
    for row in task_rows:
        instance_id = row.get("instance_id")
        if (
            set(row) != {"instance_id", "user_query", "document_pdf"}
            or not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(row.get("user_query"), str)
            or not str(row["user_query"]).strip()
            or row.get("document_pdf") != f"test/documents/{instance_id}.pdf"
        ):
            raise ReleaseError("The public task manifest schema is invalid.")
        task_ids.append(instance_id)
    if (
        len(task_ids) != EXPECTED_COUNT
        or task_ids != sorted(set(task_ids))
        or _sha256(task_bytes) != TASK_MANIFEST_SHA256
        or _sha256("".join(f"{item}\n" for item in task_ids).encode())
        != SORTED_IDS_SHA256
        or pdf_paths != [f"test/documents/{item}.pdf" for item in task_ids]
    ):
        raise ReleaseError("The public task IDs differ from their approved anchor.")

    checksums = _parse_checksums(checksum_bytes)
    expected_checksum_paths = {
        "tasks.jsonl",
        "release.json",
        *(path.removeprefix("test/") for path in pdf_paths),
    }
    if (
        set(checksums) != expected_checksum_paths
        or checksums.get("tasks.jsonl") != _sha256(task_bytes)
        or checksums.get("release.json") != _sha256(release_bytes)
    ):
        raise ReleaseError("The public checksums differ from their approved anchor.")
    pdf_inventory = b"".join(
        f"{Path(path).name}  {checksums[path.removeprefix('test/')]}\n".encode()
        for path in pdf_paths
    )
    if _sha256(pdf_inventory) != PDF_INVENTORY_SHA256:
        raise ReleaseError("The public PDF digest differs from its approved anchor.")
    for path in pdf_paths:
        try:
            payload = _read_bounded_regular_file(
                root / path, MAX_PDF_BYTES, "Pinned public PDF"
            )
        except ValidationError as exc:
            raise ReleaseError("A pinned public PDF is unsafe.") from exc
        if _sha256(payload) != checksums[path.removeprefix("test/")]:
            raise ReleaseError("A pinned public PDF differs from its checksum.")
        del payload
    final = _tree_fingerprint(root)
    if final != initial:
        raise ReleaseError("The public stage changed while it was audited.")
    digests = {
        "test/tasks.jsonl": _sha256(task_bytes),
        "test/release.json": _sha256(release_bytes),
        "test/SHA256SUMS": _sha256(checksum_bytes),
        **{path: checksums[path.removeprefix("test/")] for path in pdf_paths},
    }
    sizes = {path: (root / path).stat().st_size for path in expected_files}
    return _PublicSnapshot(
        manifest,
        tuple(task_ids),
        frozenset(expected_files),
        {
            "test/tasks.jsonl": task_bytes,
            "test/release.json": release_bytes,
            "test/SHA256SUMS": checksum_bytes,
        },
        digests,
        sizes,
        final,
    )


def _validate_label_bytes(payload: bytes, task_ids: Sequence[str]) -> None:
    rows = _parse_jsonl(payload, "Private test labels")
    try:
        _validate_labels([dict(row) for row in rows])
    except ValidationError as exc:
        raise ReleaseError("The private test label schema is invalid.") from exc
    if [row["instance_id"] for row in rows] != list(task_ids):
        raise ReleaseError("The private label IDs differ from the public task order.")


def _private_manifest(public: _PublicSnapshot, gold_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "counts": {
            "tasks": EXPECTED_COUNT,
            "pdfs": EXPECTED_COUNT,
            "labels": EXPECTED_COUNT,
        },
        "sorted_ids_sha256": SORTED_IDS_SHA256,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "gold_sha256": gold_sha256,
        "pdf_inventory_sha256": PDF_INVENTORY_SHA256,
        "visibility_audit": _visibility_audit_contract(),
        "enabled": False,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "finalized": False,
    }


def _validate_private_bytes(
    label_bytes: bytes,
    release_bytes: bytes,
    public: _PublicSnapshot,
) -> Mapping[str, object]:
    _validate_label_bytes(label_bytes, public.task_ids)
    manifest = _parse_json_document(release_bytes, "Private release policy")
    if (
        set(manifest) != _PRIVATE_MANIFEST_KEYS
        or type(manifest.get("schema_version")) is not int
        or type(manifest.get("max_attempts")) is not int
        or manifest != _private_manifest(public, _sha256(label_bytes))
    ):
        raise ReleaseError(
            "The private release policy differs from its approved contract."
        )
    return manifest


def _fsync_directory(path: Path) -> None:
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        os.fsync(descriptor)
    except OSError as exc:
        raise ReleaseError(
            "A private staging directory could not be made durable."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically install a directory without replacing an existing entry."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if hasattr(library, "renameatx_np"):
        result = library.renameatx_np(
            -2, source_bytes, -2, destination_bytes, 0x00000004
        )
    elif hasattr(library, "renameat2"):
        result = library.renameat2(-100, source_bytes, -100, destination_bytes, 1)
    else:
        if _path_entry_exists(destination):
            raise FileExistsError(errno.EEXIST, "destination exists")
        os.rename(source, destination)
        return
    if result != 0:
        value = ctypes.get_errno()
        raise OSError(value, os.strerror(value))


def _remove_owned_stage(path: Path) -> None:
    """Remove only a just-created continuation stage with its exact inventory."""
    try:
        if not path.exists():
            return
        files, directories = _walk_exact(path, "Private temporary staging")
        if files <= set(_PRIVATE_PATHS) and directories <= {"private"}:
            for relative in files:
                (path / relative).unlink()
            if (path / "private").exists():
                (path / "private").rmdir()
            path.rmdir()
    except Exception:
        # Never broaden cleanup if an owned temporary tree was unexpectedly changed.
        return


def _receipt(
    *,
    status: str,
    private_base: str | None = None,
    private_revision: str | None = None,
    legacy_result: str | None = None,
) -> dict[str, object]:
    revisions = {"public": PUBLIC_REVISION}
    if private_base is not None:
        revisions["private_base"] = private_base
    if private_revision is not None:
        revisions["private_returned"] = private_revision
    result = {
        "status": status,
        "release_id": RELEASE_ID,
        "counts": {
            "tasks": EXPECTED_COUNT,
            "pdfs": EXPECTED_COUNT,
            "labels": EXPECTED_COUNT,
        },
        "digests": {
            "sorted_ids_sha256": SORTED_IDS_SHA256,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "pdf_inventory_sha256": PDF_INVENTORY_SHA256,
            "gold_sha256": PRIVATE_LABELS_SHA256,
        },
        "revisions": revisions,
        "activation": "not-performed",
    }
    if legacy_result is not None:
        result["legacy_history"] = {
            "policy_id": LEGACY_HISTORY_POLICY,
            "result": legacy_result,
            "legacy_event_count": 2,
            "metadata_sha256": LEGACY_HISTORY_METADATA_SHA256,
            "blob_objects_fetched": False,
            "blob_contents_read": False,
            "legacy_content_compared": False,
            "legacy_history_retained": True,
        }
        result["operation"] = {"overwrites": 0, "deletes": 0, "retries": 0}
    return result


def prepare_stage(
    config: ReleaseConfig,
    *,
    public_auditor: Callable[[Path], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Atomically create only the disabled private stage."""
    _validate_local_config(config)
    destination = Path(config.private_stage).absolute()
    if _path_entry_exists(destination):
        raise ReleaseError("The private staging destination must be absent.")
    public = _audit_local_public(Path(config.public_stage), public_auditor)
    source = Path(config.private_label_source)
    source_before = _entry_fingerprint(source)
    try:
        label_bytes = _read_bounded_regular_file(
            source, MAX_PRIVATE_LABEL_BYTES, "Private test labels"
        )
    except ValidationError as exc:
        raise ReleaseError("The approved private label source is unsafe.") from exc
    if (
        _entry_fingerprint(source) != source_before
        or _sha256(label_bytes) != PRIVATE_LABELS_SHA256
    ):
        raise ReleaseError(
            "The approved private label source changed or has the wrong digest."
        )
    _validate_label_bytes(label_bytes, public.task_ids)
    release_bytes = _canonical_json(_private_manifest(public, _sha256(label_bytes)))
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=".docsem-private-stage-", dir=destination.parent)
        )
    except OSError as exc:
        raise ReleaseError(
            "A private staging workspace could not be created safely."
        ) from exc
    installed = False
    durable = False
    try:
        temporary.chmod(0o700)
        private = temporary / "private"
        private.mkdir(mode=0o700)
        _write_new_file(private / "test_labels.jsonl", label_bytes, 0o600)
        _write_new_file(private / "test_release.json", release_bytes, 0o600)
        _fsync_directory(private)
        _fsync_directory(temporary)
        if _entry_fingerprint(source) != source_before:
            raise ReleaseError(
                "The approved private label source changed during staging."
            )
        _rename_noreplace(temporary, destination)
        installed = True
        _fsync_directory(destination.parent)
        durable = True
    except ValidationError as exc:
        raise ReleaseError(
            "A private staging file could not be created safely."
        ) from exc
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError(
            "The private stage could not be installed atomically."
        ) from exc
    finally:
        if not durable and installed:
            _remove_owned_stage(destination)
        elif not installed:
            _remove_owned_stage(temporary)
    return _receipt(status="prepared")


def _load_private_stage(
    config: ReleaseConfig, public: _PublicSnapshot
) -> _PrivateSnapshot:
    root = Path(config.private_stage)
    try:
        root_mode = root.lstat().st_mode
        private_mode = (root / "private").lstat().st_mode
    except OSError as exc:
        raise ReleaseError(
            "The prepared private stage is absent or unreadable."
        ) from exc
    if (
        not stat.S_ISDIR(root_mode)
        or stat.S_ISLNK(root_mode)
        or stat.S_IMODE(root_mode) != 0o700
        or not stat.S_ISDIR(private_mode)
        or stat.S_ISLNK(private_mode)
        or stat.S_IMODE(private_mode) != 0o700
    ):
        raise ReleaseError("The prepared private stage permissions are unsafe.")
    files, directories = _walk_exact(root, "Private test staging")
    if files != set(_PRIVATE_PATHS) or directories != {"private"}:
        raise ReleaseError("The prepared private stage inventory is not exact.")
    payloads = {}
    limits = {
        "private/test_labels.jsonl": MAX_PRIVATE_LABEL_BYTES,
        "private/test_release.json": MAX_PRIVATE_RELEASE_BYTES,
    }
    for relative in _PRIVATE_PATHS:
        path = root / relative
        if stat.S_IMODE(path.lstat().st_mode) != 0o600:
            raise ReleaseError("The prepared private stage permissions are unsafe.")
        payloads[relative] = _read_regular_file(
            path, limits[relative], "Prepared private release file", exact_mode=0o600
        )
    manifest = _validate_private_bytes(
        payloads["private/test_labels.jsonl"],
        payloads["private/test_release.json"],
        public,
    )
    artifacts = {
        relative: Artifact(root / relative, len(payload), _sha256(payload))
        for relative, payload in payloads.items()
    }
    return _PrivateSnapshot(artifacts, manifest)


@contextmanager
def _sealed_private_operations(private: _PrivateSnapshot):
    """Yield immutable, privately permissioned copies only for the CAS duration."""
    try:
        temporary = tempfile.TemporaryDirectory(prefix="docsem-private-cas-")
    except OSError as exc:
        raise ReleaseError(
            "A private CAS workspace could not be created safely."
        ) from exc
    with temporary as name:
        root = Path(name)
        try:
            root.chmod(0o700)
            operations = {}
            for index, relative in enumerate(_PRIVATE_PATHS):
                payload = private.artifacts[relative].read_bytes()
                path = root / str(index)
                _write_new_file(path, payload, 0o400)
                operations[relative] = Artifact(path, len(payload), _sha256(payload))
            _fsync_directory(root)
        except ValidationError as exc:
            raise ReleaseError(
                "A private CAS artifact could not be sealed safely."
            ) from exc
        except OSError as exc:
            raise ReleaseError(
                "A private CAS artifact could not be sealed safely."
            ) from exc
        yield operations


def _validate_identity(identity: HfIdentity) -> None:
    if (
        not isinstance(identity, HfIdentity)
        or identity.username != EXPECTED_OWNER
        or identity.role != EXPECTED_ROLE
    ):
        raise ReleaseError(
            "The authenticated Hugging Face identity is not the approved classic-write owner."
        )


def _event_key(event: PrivateHistoryEvent) -> tuple[object, ...]:
    return (
        event.revision,
        event.parents,
        event.timestamp,
        event.status,
        event.path,
    )


def _validate_publication_events(
    events: Sequence[PrivateHistoryEvent],
    *,
    audit: PrivateHistoryAudit,
    expected_parent: str | None,
) -> str:
    if len(events) != 2:
        raise ReleaseError(
            "Private history does not contain one exact publication event."
        )
    labels, release = events
    expected_subject = f"Install {RELEASE_ID} disabled"
    if (
        labels.revision != release.revision
        or labels.parents != release.parents
        or labels.timestamp != release.timestamp
        or labels.status != "A"
        or release.status != "A"
        or (labels.path, release.path) != _PRIVATE_PATHS
        or labels.subject != expected_subject
        or release.subject != expected_subject
        or len(labels.parents) != 1
        or labels.parents[0] not in audit.reachable
        or (expected_parent is not None and labels.parents != (expected_parent,))
        or labels.revision not in audit.reachable
    ):
        raise ReleaseError("Private history publication metadata is not exact.")
    return labels.revision


def _validate_private_history(
    audit: PrivateHistoryAudit,
    *,
    expected_head: str,
    current_status: str,
    legacy_policy: str | None,
    expected_publication_parent: str | None = None,
) -> str:
    if (
        not isinstance(audit, PrivateHistoryAudit)
        or audit.head != expected_head
        or audit.shallow is not False
        or audit.blob_objects_fetched is not False
        or expected_head not in audit.reachable
        or not 1 <= len(audit.reachable) <= MAX_HISTORY_COMMITS
        or any(not re.fullmatch(r"[0-9a-f]{40}", item) for item in audit.reachable)
        or set(audit.parent_map) != set(audit.reachable)
        or any(not isinstance(parents, tuple) for parents in audit.parent_map.values())
        or any(
            parent not in audit.reachable
            for parents in audit.parent_map.values()
            for parent in parents
        )
        or len(audit.events) > 4
    ):
        raise ReleaseError("Private history audit is incomplete or unsafe.")
    for event in audit.events:
        try:
            parsed_time = datetime.strptime(
                event.timestamp, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError) as exc:
            raise ReleaseError("Private history event metadata is malformed.") from exc
        if (
            not isinstance(event, PrivateHistoryEvent)
            or not re.fullmatch(r"[0-9a-f]{40}", event.revision)
            or not isinstance(event.parents, tuple)
            or any(
                not re.fullmatch(r"[0-9a-f]{40}", parent) for parent in event.parents
            )
            or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", event.timestamp
            )
            or event.status not in {"A", "D", "M", "R", "C", "T"}
            or not _safe_relative_path(event.path)
            or not event.path.startswith("private/test_")
            or not isinstance(event.subject, str)
            or len(event.subject.encode("utf-8")) > 1024
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in event.subject
            )
            or event.revision not in audit.reachable
            or audit.parent_map.get(event.revision) != event.parents
            or parsed_time.strftime("%Y-%m-%dT%H:%M:%SZ") != event.timestamp
        ):
            raise ReleaseError("Private history event metadata is malformed.")

    legacy = audit.events[:2]
    has_legacy_shape = (
        len(legacy) == 2
        and legacy[0].status == "A"
        and legacy[0].path == "private/test_labels.jsonl"
        and legacy[0].parents == ()
        and legacy[1].status == "D"
        and legacy[1].path == "private/test_labels.jsonl"
        and legacy[1].parents == (legacy[0].revision,)
        and legacy[0].revision != legacy[1].revision
        and legacy[0].timestamp < legacy[1].timestamp
    )
    has_legacy = (
        has_legacy_shape
        and _sha256(_canonical_json([_event_metadata(event) for event in legacy]))
        == LEGACY_HISTORY_METADATA_SHA256
    )
    if legacy_policy is None:
        if has_legacy:
            raise ReleaseError(
                "Retained legacy private test history requires the exact policy."
            )
        if current_status == "pending":
            if audit.events:
                raise ReleaseError(
                    "A removed or superseded private test namespace exists in history."
                )
            return "matched-empty"
        _validate_publication_events(
            audit.events,
            audit=audit,
            expected_parent=expected_publication_parent,
        )
        return "matched-postpublish"
    if legacy_policy != LEGACY_HISTORY_POLICY or not has_legacy:
        raise ReleaseError(
            "The retained legacy private history profile does not match."
        )
    if current_status == "pending":
        if len(audit.events) != 2:
            raise ReleaseError(
                "The retained legacy private history event stream is not exact."
            )
        return "matched-prepublish"
    if len(audit.events) != 4:
        raise ReleaseError(
            "The retained legacy private history event stream is not exact."
        )
    if audit.events[2].revision in {legacy[0].revision, legacy[1].revision}:
        raise ReleaseError("The current publication revision reuses legacy history.")
    _validate_publication_events(
        audit.events[2:],
        audit=audit,
        expected_parent=expected_publication_parent,
    )
    return "matched-postpublish"


def _audit_public_remote(
    hub,
    token: str,
    public: _PublicSnapshot,
) -> None:
    if _tree_fingerprint(Path(PUBLIC_STAGE)) != public.fingerprint:
        raise ReleaseError("The public stage changed after its audit.")
    try:
        state = _validated_hf_state(
            hub.repository_state(PUBLIC_HF_REPOSITORY, token),
            expected_revision=PUBLIC_REVISION,
            expected_private=False,
            description="Public Hugging Face repository",
        )
        paths = tuple(hub.list_paths(PUBLIC_HF_REPOSITORY, state.revision, token))
        test_paths = {path for path in paths if path.startswith("test/")}
        if test_paths != set(public.paths):
            raise ReleaseError(
                "The public test inventory differs from the approved anchor."
            )
        metadata = hub.read_files(
            PUBLIC_HF_REPOSITORY, state.revision, _PUBLIC_METADATA_PATHS, token
        )
        if metadata != public.metadata:
            raise ReleaseError("The public metadata differs from the approved anchor.")
        inventory = hub.file_inventory(PUBLIC_HF_REPOSITORY, state.revision, token)
        test_inventory = {
            path: item for path, item in inventory.items() if path.startswith("test/")
        }
        if set(test_inventory) != set(public.paths):
            raise ReleaseError(
                "The public file inventory differs from the approved anchor."
            )
        for path, expected_digest in public.digests.items():
            item = test_inventory[path]
            if (
                not isinstance(item, RemoteFile)
                or type(item.size) is not int
                or item.size != public.sizes[path]
                or item.sha256 != expected_digest
            ):
                raise ReleaseError(
                    "A public file digest differs from the approved anchor."
                )
        _scan_public_history(hub.history_snapshots(PUBLIC_HF_REPOSITORY, token), "hf")
    except (ReleaseError, RemoteMovedError):
        raise
    except Exception as exc:
        raise ReleaseError(
            "The public release anchor could not be verified safely."
        ) from exc


def _inspect_private(
    config: ReleaseConfig,
    hub,
    token: str,
    private: _PrivateSnapshot,
    *,
    legacy_policy: str | None,
    expected_publication_parent: str | None = None,
) -> tuple[str, frozenset[str], str]:
    try:
        _validate_identity(hub.identity(token))
        state = _validated_hf_state(
            hub.repository_state(PRIVATE_HF_REPOSITORY, token),
            expected_revision=config.private_hf_base,
            expected_private=True,
            description="Private Hugging Face repository",
        )
        paths = tuple(hub.list_paths(PRIVATE_HF_REPOSITORY, state.revision, token))
        if any(not _safe_relative_path(path) for path in paths):
            raise ReleaseError("The private repository contains an unsafe path.")
        status = _state_for_namespace(
            paths,
            "private/test_",
            private.artifacts,
            lambda names: hub.read_files(
                PRIVATE_HF_REPOSITORY, state.revision, names, token
            ),
        )
        history_result = _validate_private_history(
            hub.history_audit(PRIVATE_HF_REPOSITORY, state.revision, token),
            expected_head=state.revision,
            current_status=status,
            legacy_policy=legacy_policy,
            expected_publication_parent=expected_publication_parent,
        )
        non_test = frozenset(
            path for path in paths if not path.startswith("private/test_")
        )
        return status, non_test, history_result
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError(
            "The private repository could not be inspected safely."
        ) from exc


def _reconcile_private(
    hub,
    token: str,
    revision: str,
    non_test: frozenset[str],
    private: _PrivateSnapshot,
    public: _PublicSnapshot,
) -> None:
    try:
        _validated_hf_state(
            hub.repository_state(PRIVATE_HF_REPOSITORY, token),
            expected_revision=revision,
            expected_private=True,
            description="Private Hugging Face repository",
        )
        paths = tuple(hub.list_paths(PRIVATE_HF_REPOSITORY, revision, token))
        current_non_test = frozenset(
            path for path in paths if not path.startswith("private/test_")
        )
        if current_non_test != non_test:
            raise ReleaseError("The private non-test path inventory changed.")
        if {path for path in paths if path.startswith("private/test_")} != set(
            _PRIVATE_PATHS
        ):
            raise ReleaseError("The installed private test inventory is not exact.")
        remote = hub.read_files(PRIVATE_HF_REPOSITORY, revision, _PRIVATE_PATHS, token)
        if set(remote) != set(_PRIVATE_PATHS) or any(
            remote[path] != private.artifacts[path].read_bytes()
            for path in _PRIVATE_PATHS
        ):
            raise ReleaseError("The installed private release differs from staging.")
        _validate_private_bytes(
            remote["private/test_labels.jsonl"],
            remote["private/test_release.json"],
            public,
        )
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError(
            "The installed private release could not be reconciled safely."
        ) from exc


def run_private_continuation(
    config: ReleaseConfig,
    *,
    hf_backend,
    token: str,
    publish: bool = False,
    confirmation: str | None = None,
    public_auditor: Callable[[Path], Mapping[str, object]] | None = None,
    legacy_history_policy: str | None = None,
    legacy_history_confirmation: str | None = None,
) -> dict[str, object]:
    """Dry-run or exact-parent publish the two-file disabled private release."""
    _validate_remote_config(config)
    if not isinstance(token, str) or not token:
        raise ReleaseError("A secure Hugging Face credential is required.")
    if publish and confirmation != CONFIRMATION:
        raise ReleaseError(
            "Private publication requires the exact confirmation phrase."
        )
    if legacy_history_policy not in {None, LEGACY_HISTORY_POLICY}:
        raise ReleaseError("The retained legacy history policy is invalid.")
    if legacy_history_policy is None and legacy_history_confirmation is not None:
        raise ReleaseError("A legacy confirmation requires the exact closed policy.")
    if (
        legacy_history_confirmation is not None
        and legacy_history_confirmation != LEGACY_HISTORY_CONFIRMATION
    ):
        raise ReleaseError("The retained legacy history confirmation is invalid.")
    if (
        publish
        and legacy_history_policy == LEGACY_HISTORY_POLICY
        and legacy_history_confirmation != LEGACY_HISTORY_CONFIRMATION
    ):
        raise ReleaseError(
            "Legacy-history publication requires its exact confirmation."
        )
    # Preparation already ran the full PDF auditor.  Dry-run/publication recheck
    # the immutable metadata, path/stat inventory, and remote PDF digests without
    # creating another local PDF-audit workspace.
    public = _audit_local_public(Path(config.public_stage), None)
    private = _load_private_stage(config, public)
    _audit_public_remote(hf_backend, token, public)
    status, non_test, history_result = _inspect_private(
        config,
        hf_backend,
        token,
        private,
        legacy_policy=legacy_history_policy,
    )
    if not publish:
        return _receipt(
            status=status,
            private_base=config.private_hf_base,
            legacy_result=(
                history_result
                if legacy_history_policy == LEGACY_HISTORY_POLICY
                else None
            ),
        )
    if status == "already-published":
        returned = config.private_hf_base
    else:
        # Recheck both immutable anchors immediately before the only write.
        _audit_public_remote(hf_backend, token, public)
        boundary_status, boundary_non_test, boundary_history = _inspect_private(
            config,
            hf_backend,
            token,
            private,
            legacy_policy=legacy_history_policy,
        )
        if (
            boundary_status != "pending"
            or boundary_non_test != non_test
            or boundary_history != history_result
        ):
            raise ReleaseError("Private state changed at the publication boundary.")
        try:
            with _sealed_private_operations(private) as operations:
                _validated_hf_state(
                    hf_backend.repository_state(PUBLIC_HF_REPOSITORY, token),
                    expected_revision=PUBLIC_REVISION,
                    expected_private=False,
                    description="Public Hugging Face repository",
                )
                returned = hf_backend.publish(
                    PRIVATE_HF_REPOSITORY,
                    config.private_hf_base,
                    operations,
                    f"Install {RELEASE_ID} disabled",
                    token,
                    expected_private=True,
                )
            returned = _validate_revision(returned, "Private Hugging Face publication")
        except RemoteMovedError as exc:
            raise ReleaseError(
                "The exact-parent private publication was refused without retry."
            ) from exc
        except Exception as exc:
            raise PublicationUncertainError(
                "The private publication outcome is uncertain; do not retry or compensate."
            ) from exc
    try:
        _reconcile_private(hf_backend, token, returned, non_test, private, public)
        post_config = ReleaseConfig(
            config.public_stage,
            config.private_label_source,
            config.private_stage,
            returned,
        )
        post_status, post_non_test, post_history = _inspect_private(
            post_config,
            hf_backend,
            token,
            private,
            legacy_policy=legacy_history_policy,
            expected_publication_parent=(
                config.private_hf_base if status == "pending" else None
            ),
        )
        if post_status != "already-published" or post_non_test != non_test:
            raise ReleaseError(
                "Private state changed during post-publication history audit."
            )
        _audit_public_remote(hf_backend, token, public)
    except Exception as exc:
        if status == "pending":
            raise PublicationUncertainError(
                "The private publication may have landed; do not retry or compensate."
            ) from exc
        if isinstance(exc, ReleaseError):
            raise
        raise ReleaseError("Idempotent private verification failed safely.") from exc
    return _receipt(
        status="already-published" if status == "already-published" else "published",
        private_base=config.private_hf_base,
        private_revision=returned,
        legacy_result=(
            post_history if legacy_history_policy == LEGACY_HISTORY_POLICY else None
        ),
    )


class HuggingFaceBackend(_GuardedHuggingFaceBackend):
    """Guarded HF adapter with identity and path-only private history inspection."""

    def identity(self, token: str) -> HfIdentity:
        _CommitOperationAdd, HfApi, _download = self._imports()
        try:
            value = HfApi(token=token).whoami(token=token)
            return HfIdentity(value["name"], value["auth"]["accessToken"]["role"])
        except Exception as exc:
            raise ReleaseError(
                "The Hugging Face identity could not be verified."
            ) from exc

    @staticmethod
    def _read_imports():
        try:
            from huggingface_hub import hf_hub_url
            from huggingface_hub.utils import build_hf_headers, get_session
        except ImportError as exc:
            raise ReleaseError(
                "The pinned Hugging Face client is unavailable."
            ) from exc
        return hf_hub_url, build_hf_headers, get_session

    def local_token(self) -> str:
        try:
            from huggingface_hub import get_token

            return get_token() or ""
        except Exception as exc:
            raise ReleaseError(
                "The local Hugging Face authentication could not be loaded."
            ) from exc

    def read_files(
        self, repository: str, revision: str, paths: Sequence[str], token: str
    ) -> Mapping[str, bytes]:
        """Read bounded metadata/private artifacts in memory without a disk cache."""
        hf_hub_url, build_hf_headers, get_session = self._read_imports()
        result = {}
        try:
            for path in paths:
                if not _safe_relative_path(path) or path.endswith(".pdf"):
                    raise ReleaseError("A requested Hugging Face path is unsafe.")
                limit = (
                    MAX_PRIVATE_LABEL_BYTES
                    if path == "private/test_labels.jsonl"
                    else MAX_PRIVATE_RELEASE_BYTES
                    if path == "private/test_release.json"
                    else MAX_HISTORY_METADATA_BYTES
                )
                response = get_session().get(
                    hf_hub_url(
                        repo_id=repository,
                        filename=path,
                        repo_type="dataset",
                        revision=revision,
                    ),
                    headers=build_hf_headers(token=token),
                    stream=True,
                    timeout=(10, 60),
                )
                try:
                    response.raise_for_status()
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) > limit:
                        raise ReleaseError(
                            "A Hugging Face artifact exceeds its size limit."
                        )
                    payload = bytearray()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        payload.extend(chunk)
                        if len(payload) > limit:
                            raise ReleaseError(
                                "A Hugging Face artifact exceeds its size limit."
                            )
                    result[path] = bytes(payload)
                finally:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
        except ReleaseError:
            raise
        except Exception as exc:
            raise ReleaseError(
                "Hugging Face release bytes could not be read safely."
            ) from exc
        return result

    def file_inventory(
        self, repository: str, revision: str, token: str
    ) -> Mapping[str, RemoteFile]:
        _CommitOperationAdd, HfApi, _download = self._imports()
        try:
            entries = HfApi(token=token).list_repo_tree(
                repo_id=repository,
                repo_type="dataset",
                revision=revision,
                path_in_repo="test",
                recursive=True,
                expand=True,
            )
            result = {}
            for entry in entries:
                path = getattr(entry, "rfilename", None) or getattr(entry, "path", None)
                size = getattr(entry, "size", None)
                if (
                    not isinstance(path, str)
                    or not path.startswith("test/")
                    or type(size) is not int
                ):
                    continue
                lfs = getattr(entry, "lfs", None)
                value = getattr(lfs, "sha256", None)
                if value is None and isinstance(lfs, Mapping):
                    value = lfs.get("sha256")
                result[path] = RemoteFile(size, value)
            metadata = self.read_files(
                repository, revision, _PUBLIC_METADATA_PATHS, token
            )
            for path, payload in metadata.items():
                result[path] = RemoteFile(len(payload), _sha256(payload))
            return result
        except ReleaseError:
            raise
        except Exception as exc:
            raise ReleaseError(
                "The public Hugging Face inventory could not be inspected."
            ) from exc

    @staticmethod
    def _git_environment(token: str) -> dict[str, str]:
        if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
            raise ReleaseError("The private history credential is invalid.")
        authorization = base64.b64encode(
            f"{EXPECTED_OWNER}:{token}".encode("utf-8")
        ).decode("ascii")
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "/usr/bin/false",
            "SSH_ASKPASS": "/usr/bin/false",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_CONFIG_COUNT": "3",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "",
            "GIT_CONFIG_KEY_1": "core.askPass",
            "GIT_CONFIG_VALUE_1": "",
            "GIT_CONFIG_KEY_2": "http.extraHeader",
            "GIT_CONFIG_VALUE_2": f"Authorization: Basic {authorization}",
        }

    @staticmethod
    def _run_history_git(
        arguments: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        max_output: int = MAX_GIT_CONTROL_OUTPUT_BYTES,
    ) -> bytes:
        output = None
        process = None
        try:
            output = tempfile.TemporaryFile(dir=cwd)
            process = subprocess.Popen(
                ["git", *arguments],
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_FSIZE, (max_output, max_output)
                ),
            )
            return_code = process.wait(timeout=300)
            output.seek(0, os.SEEK_END)
            size = output.tell()
            if return_code != 0 or size > max_output:
                raise ReleaseError("The private history metadata fetch failed safely.")
            output.seek(0)
            return output.read()
        except subprocess.TimeoutExpired as exc:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait()
            raise ReleaseError(
                "The private history metadata fetch failed safely."
            ) from exc
        except ReleaseError:
            raise
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReleaseError(
                "The private history metadata fetch failed safely."
            ) from exc
        finally:
            if output is not None:
                output.close()

    def _verify_filter_capability(
        self,
        root: Path,
        remote: str,
        expected_head: str,
        environment: Mapping[str, str],
    ) -> None:
        trace = root / "packet.trace"
        traced = dict(environment)
        traced["GIT_TRACE_PACKET"] = str(trace)
        output = self._run_history_git(
            ("ls-remote", "--heads", remote, "refs/heads/main"),
            cwd=root,
            environment=traced,
        )
        expected = f"{expected_head}\trefs/heads/main\n".encode("ascii")
        if output != expected:
            raise RemoteMovedError("Private history moved from its expected head.")
        try:
            trace_bytes = _read_regular_file(
                trace,
                MAX_GIT_CONTROL_OUTPUT_BYTES,
                "Private Git capability trace",
            )
            trace.unlink()
        except ReleaseError:
            raise
        except OSError as exc:
            raise ReleaseError(
                "Private Git capabilities could not be verified."
            ) from exc
        capability = any(
            (b"git<" in line or b"upload-pack>" in line)
            and b"fetch=" in line
            and re.search(rb"(?:^|[ =])filter(?:$|[ =])", line)
            for line in trace_bytes.splitlines()
        )
        if not capability:
            raise ReleaseError(
                "The private Git server did not advertise no-blob fetch support."
            )

    def _verify_no_blob_objects(
        self, repository: Path, environment: Mapping[str, str]
    ) -> None:
        objects = repository / "objects"
        try:
            indexes = tuple(sorted((objects / "pack").glob("*.idx")))
            loose = tuple(
                path
                for directory in objects.iterdir()
                if len(directory.name) == 2
                and all(character in "0123456789abcdef" for character in directory.name)
                and directory.is_dir()
                for path in directory.iterdir()
                if path.is_file()
            )
        except OSError as exc:
            raise ReleaseError(
                "Private history objects could not be inspected safely."
            ) from exc
        for index in indexes:
            output = self._run_history_git(
                ("verify-pack", "-v", str(index)),
                cwd=repository,
                environment=environment,
                max_output=MAX_GIT_PATH_OUTPUT_BYTES,
            )
            try:
                lines = output.decode("ascii").splitlines()
            except UnicodeDecodeError as exc:
                raise ReleaseError("Private history objects are malformed.") from exc
            for line in lines:
                fields = line.split()
                if (
                    len(fields) >= 2
                    and re.fullmatch(r"[0-9a-f]{40}", fields[0])
                    and fields[1] == "blob"
                ):
                    raise ReleaseError(
                        "Private history fetch unexpectedly contained file contents."
                    )
        for path in loose:
            object_id = path.parent.name + path.name
            object_type = self._run_history_git(
                ("cat-file", "-t", object_id),
                cwd=repository,
                environment=environment,
            ).strip()
            if object_type == b"blob":
                raise ReleaseError(
                    "Private history fetch unexpectedly contained file contents."
                )

    def _history_snapshot_from_remote(
        self, remote: str, expected_head: str, token: str
    ) -> tuple[frozenset[str], PrivateHistoryAudit]:
        _validate_revision(expected_head, "Private history head")
        environment = self._git_environment(token)
        try:
            temporary = tempfile.TemporaryDirectory(prefix="docsem-private-history-")
        except OSError as exc:
            raise ReleaseError(
                "A private history workspace could not be created safely."
            ) from exc
        with temporary as name:
            root = Path(name)
            root.chmod(0o700)
            self._verify_filter_capability(root, remote, expected_head, environment)
            repository = root / "repository.git"
            repository.mkdir(mode=0o700)
            self._run_history_git(
                ("init", "--bare", "."), cwd=repository, environment=environment
            )
            self._run_history_git(
                (
                    "fetch",
                    "--filter=blob:none",
                    "--no-tags",
                    "--force",
                    remote,
                    "refs/heads/main",
                ),
                cwd=repository,
                environment=environment,
            )
            fetched = (
                self._run_history_git(
                    ("rev-parse", "FETCH_HEAD"), cwd=repository, environment=environment
                )
                .decode("ascii")
                .strip()
            )
            if fetched != expected_head:
                raise RemoteMovedError("Private history moved from its expected head.")
            self._verify_no_blob_objects(repository, environment)
            shallow_value = self._run_history_git(
                ("rev-parse", "--is-shallow-repository"),
                cwd=repository,
                environment=environment,
            ).strip()
            if shallow_value not in {b"true", b"false"}:
                raise ReleaseError("Private history depth could not be verified.")
            count_text = (
                self._run_history_git(
                    ("rev-list", "--count", "FETCH_HEAD"),
                    cwd=repository,
                    environment=environment,
                )
                .decode("ascii")
                .strip()
            )
            if (
                not count_text.isdigit()
                or not 1 <= int(count_text) <= MAX_HISTORY_COMMITS
            ):
                raise ReleaseError(
                    "Private history exceeds the bounded reconciliation limit."
                )
            output = self._run_history_git(
                (
                    "log",
                    "--full-history",
                    "-m",
                    "--no-renames",
                    "--no-ext-diff",
                    "--format=",
                    "--name-only",
                    "-z",
                    "FETCH_HEAD",
                ),
                cwd=repository,
                environment=environment,
                max_output=MAX_GIT_PATH_OUTPUT_BYTES,
            )
            self._verify_no_blob_objects(repository, environment)
            try:
                paths = frozenset(
                    raw.decode("utf-8") for raw in output.split(b"\0") if raw
                )
            except UnicodeDecodeError as exc:
                raise ReleaseError("Private history contains an unsafe path.") from exc
            if any(not _safe_relative_path(path) for path in paths):
                raise ReleaseError("Private history contains an unsafe path.")

            graph_output = self._run_history_git(
                ("rev-list", "--parents", "--reverse", "--topo-order", "FETCH_HEAD"),
                cwd=repository,
                environment=environment,
                max_output=MAX_GIT_PATH_OUTPUT_BYTES,
            )
            try:
                graph_rows = tuple(
                    tuple(line.decode("ascii").split())
                    for line in graph_output.splitlines()
                    if line
                )
            except UnicodeDecodeError as exc:
                raise ReleaseError("Private history graph is malformed.") from exc
            if len(graph_rows) != int(count_text):
                raise ReleaseError("Private history graph is incomplete.")
            parent_map: dict[str, tuple[str, ...]] = {}
            order: list[str] = []
            for row in graph_rows:
                if not row or any(
                    not re.fullmatch(r"[0-9a-f]{40}", item) for item in row
                ):
                    raise ReleaseError("Private history graph is malformed.")
                revision, parents = row[0], tuple(row[1:])
                if revision in parent_map:
                    raise ReleaseError(
                        "Private history graph contains a duplicate commit."
                    )
                parent_map[revision] = parents
                order.append(revision)
            reachable = frozenset(order)
            if any(
                parent not in reachable
                for parents in parent_map.values()
                for parent in parents
            ):
                raise ReleaseError("Private history graph is incomplete.")

            namespace_paths = sorted(
                path for path in paths if path.startswith("private/test_")
            )
            if any(path not in _PRIVATE_PATHS for path in namespace_paths):
                raise ReleaseError(
                    "Private history contains an unapproved test namespace path."
                )
            touched: dict[str, set[str]] = {}
            for path in namespace_paths:
                revisions = self._run_history_git(
                    (
                        "log",
                        "--reverse",
                        "--topo-order",
                        "--full-history",
                        "-m",
                        "--no-renames",
                        "--no-ext-diff",
                        "--format=%H",
                        "FETCH_HEAD",
                        "--",
                        path,
                    ),
                    cwd=repository,
                    environment=environment,
                    max_output=MAX_GIT_PATH_OUTPUT_BYTES,
                )
                for raw_revision in revisions.splitlines():
                    try:
                        revision = raw_revision.decode("ascii")
                    except UnicodeDecodeError as exc:
                        raise ReleaseError(
                            "Private history event metadata is malformed."
                        ) from exc
                    if revision not in reachable:
                        raise ReleaseError("Private history event is not reachable.")
                    touched.setdefault(revision, set()).add(path)
                    if sum(len(value) for value in touched.values()) > 16:
                        raise ReleaseError(
                            "Private history contains too many test events."
                        )
            if sum(len(value) for value in touched.values()) > 16:
                raise ReleaseError("Private history contains too many test events.")

            events: list[PrivateHistoryEvent] = []
            for revision in order:
                if revision not in touched:
                    continue
                metadata = self._run_history_git(
                    ("log", "-1", "--format=%ct", revision),
                    cwd=repository,
                    environment=environment,
                )
                seconds = metadata.strip()
                if not seconds.isdigit():
                    raise ReleaseError("Private history commit metadata is malformed.")
                try:
                    timestamp = datetime.fromtimestamp(
                        int(seconds), tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (OverflowError, OSError, ValueError) as exc:
                    raise ReleaseError(
                        "Private history commit metadata is malformed."
                    ) from exc
                for path in sorted(touched[revision]):
                    changes = self._run_history_git(
                        (
                            "diff-tree",
                            "-m",
                            "--root",
                            "--no-commit-id",
                            "--name-status",
                            "-r",
                            "--no-renames",
                            "--no-ext-diff",
                            "-z",
                            revision,
                            "--",
                            path,
                        ),
                        cwd=repository,
                        environment=environment,
                    )
                    parts = tuple(item for item in changes.split(b"\0") if item)
                    if len(parts) != 2:
                        raise ReleaseError(
                            "Private history change metadata is malformed."
                        )
                    try:
                        status = parts[0].decode("ascii")
                        changed_path = parts[1].decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ReleaseError(
                            "Private history change metadata is malformed."
                        ) from exc
                    if changed_path != path:
                        raise ReleaseError(
                            "Private history change path is inconsistent."
                        )
                    events.append(
                        PrivateHistoryEvent(
                            revision=revision,
                            parents=parent_map[revision],
                            timestamp=timestamp,
                            status=status,
                            path=path,
                            subject="",
                        )
                    )
            if (
                len(events) >= 2
                and events[-2].revision == events[-1].revision
                and events[-2].status == events[-1].status == "A"
                and (events[-2].path, events[-1].path) == _PRIVATE_PATHS
            ):
                subject_bytes = self._run_history_git(
                    ("log", "-1", "--format=%s", events[-1].revision),
                    cwd=repository,
                    environment=environment,
                ).rstrip(b"\n")
                try:
                    subject = subject_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ReleaseError(
                        "Private history publication metadata is malformed."
                    ) from exc
                events[-2:] = [
                    PrivateHistoryEvent(
                        event.revision,
                        event.parents,
                        event.timestamp,
                        event.status,
                        event.path,
                        subject,
                    )
                    for event in events[-2:]
                ]
            self._verify_no_blob_objects(repository, environment)
            self._verify_filter_capability(root, remote, expected_head, environment)
            audit = PrivateHistoryAudit(
                head=fetched,
                events=tuple(events),
                reachable=reachable,
                shallow=shallow_value == b"true",
                blob_objects_fetched=False,
                parent_map=parent_map,
            )
            return paths, audit

    def _history_path_names_from_remote(
        self, remote: str, expected_head: str, token: str
    ) -> frozenset[str]:
        paths, _audit = self._history_snapshot_from_remote(remote, expected_head, token)
        return paths

    def _history_audit_from_remote(
        self, remote: str, expected_head: str, token: str
    ) -> PrivateHistoryAudit:
        _paths, audit = self._history_snapshot_from_remote(remote, expected_head, token)
        return audit

    def history_path_names(
        self, repository: str, expected_head: str, token: str
    ) -> frozenset[str]:
        if repository != PRIVATE_HF_REPOSITORY:
            raise ReleaseError("The private history repository is invalid.")
        return self._history_path_names_from_remote(
            f"https://huggingface.co/datasets/{repository}", expected_head, token
        )

    def history_audit(
        self, repository: str, expected_head: str, token: str
    ) -> PrivateHistoryAudit:
        if repository != PRIVATE_HF_REPOSITORY:
            raise ReleaseError("The private history repository is invalid.")
        return self._history_audit_from_remote(
            f"https://huggingface.co/datasets/{repository}", expected_head, token
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or dry-run the disabled private DocSem test continuation."
    )
    parser.add_argument("--public-stage", type=Path, default=PUBLIC_STAGE)
    parser.add_argument(
        "--private-label-source", type=Path, default=PRIVATE_LABEL_SOURCE
    )
    parser.add_argument("--private-stage", type=Path, required=True)
    parser.add_argument("--private-hf-base", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-stage", action="store_true")
    mode.add_argument("--publish", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--legacy-history-policy")
    parser.add_argument("--confirm-legacy-history")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    hf_backend=None,
    token: str | None = None,
    public_auditor: Callable[[Path], Mapping[str, object]] | None = None,
) -> int:
    args = parse_args(argv)
    config = ReleaseConfig(
        args.public_stage,
        args.private_label_source,
        args.private_stage,
        args.private_hf_base,
    )
    try:
        if args.prepare_stage:
            if (
                args.legacy_history_policy is not None
                or args.confirm_legacy_history is not None
            ):
                raise ReleaseError("Legacy history options are invalid for staging.")
            result = prepare_stage(config, public_auditor=public_auditor)
        else:
            hub = hf_backend or HuggingFaceBackend()
            credential = (
                token
                or os.environ.get("DOCSEM_PRIVATE_HF_TOKEN")
                or os.environ.get("DOCSEM_HF_WRITE_TOKEN")
                or os.environ.get("HF_WRITE_TOKEN")
                or (
                    hub.local_token()
                    if callable(getattr(hub, "local_token", None))
                    else ""
                )
            )
            result = run_private_continuation(
                config,
                hf_backend=hub,
                token=credential or "",
                publish=args.publish,
                confirmation=args.confirm,
                public_auditor=public_auditor,
                legacy_history_policy=args.legacy_history_policy,
                legacy_history_confirmation=args.confirm_legacy_history,
            )
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
