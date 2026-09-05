#!/usr/bin/env python3
"""Plan, publish, and reconcile one audited DocSem held-out test release.

Dry-run is the default.  Publication is an explicit, non-force sequence that
first installs a disabled private scoring release, then the canonical GitHub
payload, and finally the public Hugging Face payload.  This module never
activates submissions or publishes a leaderboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from prepare_docsem_hf_dataset import render_test_ready_dataset_card
from prepare_docsem_test_release import (
    MAX_PDF_BYTES,
    MAX_PUBLIC_CHECKSUM_BYTES,
    MAX_PUBLIC_MANIFEST_BYTES,
    MAX_PUBLIC_TASKS_BYTES,
    audit_public_payload,
)


CANONICAL_SOURCE_CHECKOUT = Path("/Users/aamita/Oracle/amitbcp/gsm-sem")
CANONICAL_GITHUB_REPOSITORY = "https://github.com/oracle-samples/gsm-sem.git"
PUBLIC_HF_REPOSITORY = "amitbcp/docinsights-2026-shared-task-data"
PRIVATE_HF_REPOSITORY = "amitbcp/docinsights-2026-shared-task-submissions"
CANONICAL_SOURCE_BRANCH = "main"

MAX_PRIVATE_LABEL_BYTES = 64 * 1024 * 1024
MAX_PRIVATE_RELEASE_BYTES = 2 * 1024 * 1024
MAX_DATASET_CARD_BYTES = 16 * 1024 * 1024
MAX_HISTORY_COMMITS = 10_000
MAX_HISTORY_METADATA_BYTES = 64 * 1024 * 1024

_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RELEASE_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".gz",
    ".7z",
    ".rar",
)
_FORBIDDEN_PUBLIC_PATH_PART = re.compile(
    r"(?:^|[._-])(answers?|evidence|gold|ground[_-]?truth|labels?|mapping|organizer|private|solutions?)(?:$|[._-])",
    re.IGNORECASE,
)
_FORBIDDEN_PUBLIC_FIELD = re.compile(
    r"(?:^|_)(answer|evidence|gold|label|source_mapping|organizer_note|private)(?:$|_)",
    re.IGNORECASE,
)
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
_PUBLIC_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "release_id",
        "counts",
        "sorted_ids_sha256",
        "task_manifest_sha256",
        "pdf_inventory_sha256",
    }
)


class ReleaseError(RuntimeError):
    """A sanitized, operator-actionable refusal."""


class RemoteMovedError(ReleaseError):
    """The exact parent revision changed before a compare-and-swap write."""


class PartialPublicationError(ReleaseError):
    """A safe prefix of the disabled publication sequence completed."""


@dataclass(frozen=True)
class SourceState:
    checkout: Path
    repository: str
    branch: str
    head: str
    remote_revision: str
    dirty_tracked: bool
    dirty_untracked: bool
    behind: bool
    diverged: bool


@dataclass(frozen=True)
class HistorySnapshot:
    revision: str
    paths: tuple[str, ...]
    metadata: Mapping[str, bytes]


@dataclass(frozen=True, repr=False)
class Artifact:
    snapshot_path: Path
    size: int
    sha256: str

    def read_bytes(self) -> bytes:
        payload = _read_regular_file(
            self.snapshot_path,
            self.size,
            "Sealed release artifact",
            exact_size=self.size,
        )
        if _sha256(payload) != self.sha256:
            raise ReleaseError("A sealed release artifact changed before use.")
        return payload

    def __repr__(self) -> str:
        return f"Artifact(size={self.size}, sha256={self.sha256!r})"


@dataclass(frozen=True)
class ReleaseConfig:
    public_stage: Path
    private_stage: Path
    release_card: Path
    card_template: Path
    source_checkout: Path
    source_repository: str
    source_branch: str
    source_base: str
    github_remote_base: str
    public_hf_base: str
    private_hf_base: str


@dataclass(frozen=True, repr=False)
class _CapturedRelease:
    public_root: Path
    public: Mapping[str, Artifact]
    private: Mapping[str, Artifact]
    release_card: Artifact
    public_manifest: Mapping[str, object]
    private_manifest: Mapping[str, object]

    def __repr__(self) -> str:
        counts = self.private_manifest.get("counts", {})
        return f"_CapturedRelease(counts={counts!r}, sealed=True)"


class GitBackend(Protocol):
    def inspect_source(
        self, checkout: Path, repository: str, branch: str
    ) -> SourceState: ...

    def current_revision(self, repository: str, branch: str) -> str: ...

    def list_paths(self, repository: str, revision: str) -> Sequence[str]: ...

    def read_files(
        self, repository: str, revision: str, paths: Sequence[str]
    ) -> Mapping[str, bytes]: ...

    def publish(
        self,
        repository: str,
        branch: str,
        expected_parent: str,
        operations: Mapping[str, Artifact],
        message: str,
    ) -> str: ...

    def history_snapshots(
        self, repository: str, branch: str
    ) -> Sequence[HistorySnapshot]: ...


class HfBackend(Protocol):
    def current_revision(self, repository: str, token: str) -> str: ...

    def list_paths(
        self, repository: str, revision: str, token: str
    ) -> Sequence[str]: ...

    def read_files(
        self, repository: str, revision: str, paths: Sequence[str], token: str
    ) -> Mapping[str, bytes]: ...

    def publish(
        self,
        repository: str,
        expected_parent: str,
        operations: Mapping[str, Artifact],
        message: str,
        token: str,
    ) -> str: ...

    def history_snapshots(
        self, repository: str, token: str
    ) -> Sequence[HistorySnapshot]: ...


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_rows(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json(row) for row in rows)


def _safe_relative_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith(("/", "\\"))
        and "\\" not in path
        and all(part not in {"", ".", ".."} for part in path.split("/"))
        and all(ord(character) >= 32 and ord(character) != 127 for character in path)
    )


def _read_regular_file(
    path: Path,
    max_bytes: int,
    description: str,
    *,
    exact_mode: int | None = None,
    exact_size: int | None = None,
) -> bytes:
    descriptor = None
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or not 0 <= initial.st_size <= max_bytes:
            raise ReleaseError(f"{description} is not a bounded regular file.")
        if exact_mode is not None and stat.S_IMODE(initial.st_mode) != exact_mode:
            raise ReleaseError(f"{description} permissions are unsafe.")
        if exact_size is not None and initial.st_size != exact_size:
            raise ReleaseError(f"{description} changed before use.")
        chunks = []
        remaining = initial.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ReleaseError(f"{description} changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ReleaseError(f"{description} changed while being read.")
        final = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, item) != getattr(final, item) for item in stable):
            raise ReleaseError(f"{description} changed while being read.")
        return b"".join(chunks)
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError(f"{description} is absent or unreadable.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_directory(
    path: Path, description: str, *, exact_mode: int | None = None
) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ReleaseError(f"{description} is absent or unreadable.") from exc
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ReleaseError(f"{description} is not a real directory.")
    if exact_mode is not None and stat.S_IMODE(mode) != exact_mode:
        raise ReleaseError(f"{description} permissions are unsafe.")


def _walk_exact(root: Path, description: str) -> tuple[set[str], set[str]]:
    _require_directory(root, description)
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path) -> None:
        try:
            entries = tuple(os.scandir(directory))
        except OSError as exc:
            raise ReleaseError(f"{description} cannot be inspected safely.") from exc
        for entry in entries:
            relative = (directory / entry.name).relative_to(root).as_posix()
            if not _safe_relative_path(relative):
                raise ReleaseError(f"{description} contains an unsafe path.")
            try:
                item_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ReleaseError(
                    f"{description} cannot be inspected safely."
                ) from exc
            if entry.is_symlink():
                raise ReleaseError(f"{description} contains a linked entry.")
            if stat.S_ISDIR(item_stat.st_mode):
                directories.add(relative)
                visit(directory / entry.name)
            elif stat.S_ISREG(item_stat.st_mode):
                files.add(relative)
            else:
                raise ReleaseError(f"{description} contains a special entry.")

    visit(root)
    return files, directories


def _snapshot_file(
    source: Path, destination: Path, max_bytes: int, description: str
) -> Artifact:
    payload = _read_regular_file(source, max_bytes, description)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    try:
        with destination.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        destination.chmod(0o400)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise ReleaseError("A release artifact could not be sealed safely.") from exc
    return Artifact(destination, len(payload), _sha256(payload))


def _parse_json_document(payload: bytes, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{description} is malformed.") from exc
    if not isinstance(value, dict) or payload != _canonical_json(value):
        raise ReleaseError(f"{description} is not canonical.")
    return value


def _parse_jsonl(payload: bytes, description: str) -> list[Mapping[str, object]]:
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{description} is malformed.") from exc
    if (
        not rows
        or any(not isinstance(row, dict) for row in rows)
        or payload != _canonical_rows(rows)
    ):
        raise ReleaseError(f"{description} is not canonical.")
    return rows


def _capture_public(
    public_stage: Path,
    snapshot_root: Path,
    public_auditor: Callable[[Path], Mapping[str, object]],
) -> tuple[Path, Mapping[str, Artifact], Mapping[str, object], list[str]]:
    try:
        initial_audit = public_auditor(public_stage)
    except Exception as exc:
        raise ReleaseError("The public test staging audit failed.") from exc
    files, directories = _walk_exact(public_stage, "Public test staging")
    if directories != {"test", "test/documents"}:
        raise ReleaseError("Public test staging has an unexpected directory inventory.")
    fixed = {"test/tasks.jsonl", "test/release.json", "test/SHA256SUMS"}
    pdf_paths = sorted(files - fixed)
    if (
        not pdf_paths
        or any(
            not path.startswith("test/documents/")
            or not path.endswith(".pdf")
            or Path(path).name != path.removeprefix("test/documents/")
            for path in pdf_paths
        )
        or files != fixed | set(pdf_paths)
    ):
        raise ReleaseError("Public test staging has an unexpected file inventory.")

    sealed_root = snapshot_root / "public"
    artifacts: dict[str, Artifact] = {}
    limits = {
        "test/tasks.jsonl": MAX_PUBLIC_TASKS_BYTES,
        "test/release.json": MAX_PUBLIC_MANIFEST_BYTES,
        "test/SHA256SUMS": MAX_PUBLIC_CHECKSUM_BYTES,
    }
    for relative in sorted(files):
        artifacts[relative] = _snapshot_file(
            public_stage / relative,
            sealed_root / relative,
            limits.get(relative, MAX_PDF_BYTES),
            "Public test staging file",
        )

    try:
        sealed_audit = public_auditor(sealed_root)
    except Exception as exc:
        raise ReleaseError("The sealed public test staging audit failed.") from exc
    manifest = _parse_json_document(
        artifacts["test/release.json"].read_bytes(), "Public release manifest"
    )
    if (
        manifest != initial_audit
        or manifest != sealed_audit
        or set(manifest) != _PUBLIC_MANIFEST_KEYS
    ):
        raise ReleaseError(
            "Public release manifest does not match the audited payload."
        )
    counts = manifest.get("counts")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("release_id"), str)
        or not _RELEASE_ID.fullmatch(str(manifest["release_id"]))
        or not isinstance(counts, dict)
        or set(counts) != {"tasks", "pdfs"}
        or any(type(counts.get(key)) is not int or counts[key] <= 0 for key in counts)
        or any(
            not isinstance(manifest.get(key), str)
            or not _SHA256.fullmatch(str(manifest[key]))
            for key in (
                "sorted_ids_sha256",
                "task_manifest_sha256",
                "pdf_inventory_sha256",
            )
        )
    ):
        raise ReleaseError("Public release manifest schema is invalid.")

    task_bytes = artifacts["test/tasks.jsonl"].read_bytes()
    rows = _parse_jsonl(task_bytes, "Public test task manifest")
    task_ids = []
    for row in rows:
        if set(row) != {"instance_id", "user_query", "document_pdf"}:
            raise ReleaseError("Public test task schema is invalid.")
        instance_id = row.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or Path(instance_id).name != instance_id
            or not isinstance(row.get("user_query"), str)
            or not str(row["user_query"]).strip()
            or row.get("document_pdf") != f"test/documents/{instance_id}.pdf"
        ):
            raise ReleaseError("Public test task row is invalid.")
        task_ids.append(instance_id)
    if task_ids != sorted(set(task_ids)):
        raise ReleaseError("Public test task IDs are not unique and sorted.")
    expected_pdfs = [f"test/documents/{item}.pdf" for item in task_ids]
    if pdf_paths != expected_pdfs:
        raise ReleaseError("Public test task and PDF inventories differ.")

    pdf_inventory = b"".join(
        f"{Path(path).name}  {artifacts[path].sha256}\n".encode("ascii")
        for path in pdf_paths
    )
    if (
        counts != {"tasks": len(rows), "pdfs": len(pdf_paths)}
        or manifest["task_manifest_sha256"] != _sha256(task_bytes)
        or manifest["sorted_ids_sha256"]
        != _sha256("".join(f"{item}\n" for item in task_ids).encode("utf-8"))
        or manifest["pdf_inventory_sha256"] != _sha256(pdf_inventory)
    ):
        raise ReleaseError("Public test staging hashes do not reconcile.")

    checksum_targets = [
        "test/release.json",
        "test/tasks.jsonl",
        *pdf_paths,
    ]
    expected_checksums = b"".join(
        f"{artifacts[path].sha256}  {path.removeprefix('test/')}\n".encode("ascii")
        for path in sorted(checksum_targets)
    )
    if artifacts["test/SHA256SUMS"].read_bytes() != expected_checksums:
        raise ReleaseError("Public test checksums do not reconcile.")
    return sealed_root, artifacts, manifest, task_ids


def _capture_private(
    private_stage: Path,
    snapshot_root: Path,
    public_manifest: Mapping[str, object],
    task_ids: Sequence[str],
) -> tuple[Mapping[str, Artifact], Mapping[str, object]]:
    _require_directory(private_stage, "Private test staging", exact_mode=0o700)
    _require_directory(
        private_stage / "private", "Private staging directory", exact_mode=0o700
    )
    files, directories = _walk_exact(private_stage, "Private test staging")
    expected_files = {"private/test_labels.jsonl", "private/test_release.json"}
    if files != expected_files or directories != {"private"}:
        raise ReleaseError("Private test staging inventory is not exact.")
    for relative in expected_files:
        mode = (private_stage / relative).lstat().st_mode
        if stat.S_IMODE(mode) != 0o600:
            raise ReleaseError("Private test staging file permissions are unsafe.")

    artifacts = {
        "private/test_labels.jsonl": _snapshot_file(
            private_stage / "private/test_labels.jsonl",
            snapshot_root / "private/private/test_labels.jsonl",
            MAX_PRIVATE_LABEL_BYTES,
            "Private test labels",
        ),
        "private/test_release.json": _snapshot_file(
            private_stage / "private/test_release.json",
            snapshot_root / "private/private/test_release.json",
            MAX_PRIVATE_RELEASE_BYTES,
            "Private test release policy",
        ),
    }
    label_bytes = artifacts["private/test_labels.jsonl"].read_bytes()
    rows = _parse_jsonl(label_bytes, "Private test labels")
    label_ids = []
    for row in rows:
        if set(row) != {"instance_id", "answer", "evidence"}:
            raise ReleaseError("Private test label schema is invalid.")
        instance_id = row.get("instance_id")
        evidence = row.get("evidence")
        answer = row.get("answer")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(evidence, list)
            or not evidence
            or len(evidence) > 1024
            or any(
                not isinstance(item, str) or not re.fullmatch(r"b[0-9]+", item)
                for item in evidence
            )
            or len(set(evidence)) != len(evidence)
        ):
            raise ReleaseError("Private test label row is invalid.")
        label_ids.append(instance_id)
    if label_ids != list(task_ids):
        raise ReleaseError("Private label and public task IDs differ.")

    private_manifest = _parse_json_document(
        artifacts["private/test_release.json"].read_bytes(),
        "Private test release policy",
    )
    counts = private_manifest.get("counts")
    if (
        set(private_manifest) != _PRIVATE_MANIFEST_KEYS
        or private_manifest.get("schema_version") != 1
        or private_manifest.get("release_id") != public_manifest.get("release_id")
        or not isinstance(counts, dict)
        or counts
        != {
            "tasks": len(task_ids),
            "pdfs": len(task_ids),
            "labels": len(label_ids),
        }
        or private_manifest.get("sorted_ids_sha256")
        != public_manifest.get("sorted_ids_sha256")
        or private_manifest.get("task_manifest_sha256")
        != public_manifest.get("task_manifest_sha256")
        or private_manifest.get("pdf_inventory_sha256")
        != public_manifest.get("pdf_inventory_sha256")
        or private_manifest.get("gold_sha256") != _sha256(label_bytes)
        or type(private_manifest.get("max_attempts")) is not int
        or private_manifest.get("max_attempts") != 3
        or private_manifest.get("feedback_policy") != "first-attempt-only"
        or private_manifest.get("enabled") is not False
        or private_manifest.get("finalized") is not False
        or not isinstance(private_manifest.get("visibility_audit"), dict)
    ):
        raise ReleaseError(
            "Private test release policy does not match the public release."
        )
    return artifacts, private_manifest


def _capture_release(
    config: ReleaseConfig,
    snapshot_root: Path,
    public_auditor: Callable[[Path], Mapping[str, object]],
    card_renderer: Callable[..., bytes],
) -> _CapturedRelease:
    try:
        public_resolved = Path(config.public_stage).resolve(strict=True)
        private_resolved = Path(config.private_stage).resolve(strict=True)
    except OSError as exc:
        raise ReleaseError("Release staging roots are absent or unreadable.") from exc
    if (
        public_resolved == private_resolved
        or public_resolved in private_resolved.parents
        or private_resolved in public_resolved.parents
    ):
        raise ReleaseError("Public and private staging roots must be separate.")

    sealed_public_root, public, public_manifest, task_ids = _capture_public(
        public_resolved, snapshot_root, public_auditor
    )
    private, private_manifest = _capture_private(
        private_resolved, snapshot_root, public_manifest, task_ids
    )
    template = _snapshot_file(
        Path(config.card_template),
        snapshot_root / "card-template.md",
        MAX_DATASET_CARD_BYTES,
        "Dataset card template",
    )
    card = _snapshot_file(
        Path(config.release_card),
        snapshot_root / "release-card.md",
        MAX_DATASET_CARD_BYTES,
        "Test-ready dataset card",
    )
    try:
        expected_card = card_renderer(
            sealed_public_root,
            card_template_path=template.snapshot_path,
        )
    except Exception as exc:
        raise ReleaseError(
            "The test-ready dataset card could not be regenerated."
        ) from exc
    if not isinstance(expected_card, bytes) or card.read_bytes() != expected_card:
        raise ReleaseError(
            "The test-ready dataset card does not match the audited release."
        )
    return _CapturedRelease(
        public_root=sealed_public_root,
        public=public,
        private=private,
        release_card=card,
        public_manifest=public_manifest,
        private_manifest=private_manifest,
    )


def _validate_config(config: ReleaseConfig) -> None:
    if Path(config.source_checkout) != CANONICAL_SOURCE_CHECKOUT:
        raise ReleaseError("The canonical source checkout was not selected.")
    if config.source_repository != CANONICAL_GITHUB_REPOSITORY:
        raise ReleaseError("The canonical GitHub repository was not selected.")
    if not config.source_branch:
        raise ReleaseError("The source branch must be specified explicitly.")
    if config.source_branch != CANONICAL_SOURCE_BRANCH:
        raise ReleaseError("The canonical source branch was not selected.")
    for value in (
        config.source_base,
        config.github_remote_base,
        config.public_hf_base,
        config.private_hf_base,
    ):
        if not isinstance(value, str) or not _REVISION.fullmatch(value):
            raise ReleaseError(
                "Every expected remote base must be an exact commit revision."
            )


def _github_operations(release: _CapturedRelease) -> dict[str, Artifact]:
    return {f"docsem/{path}": artifact for path, artifact in release.public.items()}


def _public_hf_operations(release: _CapturedRelease) -> dict[str, Artifact]:
    return {"README.md": release.release_card, **release.public}


def _private_hf_operations(release: _CapturedRelease) -> dict[str, Artifact]:
    return dict(release.private)


def _artifact_hashes(operations: Mapping[str, Artifact]) -> dict[str, str]:
    return {path: operations[path].sha256 for path in sorted(operations)}


def _state_for_namespace(
    paths: Sequence[str],
    prefix: str,
    expected: Mapping[str, Artifact],
    reader: Callable[[Sequence[str]], Mapping[str, bytes]],
    *,
    companion_path: str | None = None,
) -> str:
    current_namespace = {path for path in paths if path.startswith(prefix)}
    expected_namespace = {path for path in expected if path.startswith(prefix)}
    if not current_namespace:
        if companion_path is not None and companion_path in paths:
            companion = reader((companion_path,)).get(companion_path)
            if companion == expected[companion_path].read_bytes():
                raise ReleaseError(
                    "A release metadata file was published without its payload."
                )
        return "pending"
    if current_namespace != expected_namespace:
        raise ReleaseError(
            "A target repository contains a conflicting partial test release."
        )
    paths_to_read = sorted(expected_namespace)
    if companion_path is not None:
        if companion_path not in paths:
            raise ReleaseError("A target repository is missing release metadata.")
        paths_to_read.append(companion_path)
    remote = reader(paths_to_read)
    if set(remote) != set(paths_to_read) or any(
        remote[path] != expected[path].read_bytes() for path in paths_to_read
    ):
        raise ReleaseError("A target repository contains a conflicting test release.")
    return "already-published"


def _inspect_targets(
    config: ReleaseConfig,
    release: _CapturedRelease,
    git_backend: GitBackend,
    hf_backend: HfBackend,
    public_token: str,
    private_token: str,
) -> tuple[dict[str, str], SourceState]:
    try:
        source = git_backend.inspect_source(
            config.source_checkout,
            config.source_repository,
            config.source_branch,
        )
        github_revision = git_backend.current_revision(
            config.source_repository, config.source_branch
        )
        public_revision = hf_backend.current_revision(
            PUBLIC_HF_REPOSITORY, public_token
        )
        private_revision = hf_backend.current_revision(
            PRIVATE_HF_REPOSITORY, private_token
        )
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError("Repository state could not be inspected safely.") from exc
    if (
        Path(source.checkout) != CANONICAL_SOURCE_CHECKOUT
        or source.repository != CANONICAL_GITHUB_REPOSITORY
        or source.branch != CANONICAL_SOURCE_BRANCH
        or source.head != config.source_base
        or source.remote_revision != config.github_remote_base
        or config.source_base != config.github_remote_base
        or github_revision != config.github_remote_base
        or source.dirty_tracked
        or source.dirty_untracked
        or source.behind
        or source.diverged
    ):
        raise ReleaseError(
            "The canonical source checkout is dirty, stale, divergent, or mismatched."
        )
    if (
        public_revision != config.public_hf_base
        or private_revision != config.private_hf_base
    ):
        raise ReleaseError("A Hugging Face target moved from its expected base.")

    github_ops = _github_operations(release)
    public_ops = _public_hf_operations(release)
    private_ops = _private_hf_operations(release)
    try:
        github_paths = git_backend.list_paths(
            config.source_repository, config.github_remote_base
        )
        public_paths = hf_backend.list_paths(
            PUBLIC_HF_REPOSITORY, config.public_hf_base, public_token
        )
        private_paths = hf_backend.list_paths(
            PRIVATE_HF_REPOSITORY, config.private_hf_base, private_token
        )
        statuses = {
            "private_hugging_face": _state_for_namespace(
                private_paths,
                "private/test_",
                private_ops,
                lambda names: hf_backend.read_files(
                    PRIVATE_HF_REPOSITORY,
                    config.private_hf_base,
                    names,
                    private_token,
                ),
            ),
            "github": _state_for_namespace(
                github_paths,
                "docsem/test/",
                github_ops,
                lambda names: git_backend.read_files(
                    config.source_repository,
                    config.github_remote_base,
                    names,
                ),
            ),
            "public_hugging_face": _state_for_namespace(
                public_paths,
                "test/",
                public_ops,
                lambda names: hf_backend.read_files(
                    PUBLIC_HF_REPOSITORY,
                    config.public_hf_base,
                    names,
                    public_token,
                ),
                companion_path="README.md",
            ),
        }
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError(
            "Target repository contents could not be inspected safely."
        ) from exc
    return statuses, source


def _publication_plan(
    config: ReleaseConfig,
    release: _CapturedRelease,
    statuses: Mapping[str, str],
) -> dict[str, object]:
    github_ops = _github_operations(release)
    public_ops = _public_hf_operations(release)
    private_ops = _private_hf_operations(release)
    private_manifest = release.private_manifest
    return {
        "mode": "dry-run",
        "source": {
            "checkout": str(config.source_checkout),
            "repository": config.source_repository,
            "branch": config.source_branch,
            "base_revision": config.source_base,
        },
        "targets": {
            "github": {
                "repository": config.source_repository,
                "base_revision": config.github_remote_base,
            },
            "public_hugging_face": {
                "repository": PUBLIC_HF_REPOSITORY,
                "base_revision": config.public_hf_base,
            },
            "private_hugging_face": {
                "repository": PRIVATE_HF_REPOSITORY,
                "base_revision": config.private_hf_base,
            },
        },
        "release": {
            "release_id": release.public_manifest["release_id"],
            "counts": dict(private_manifest["counts"]),
            "sorted_ids_sha256": private_manifest["sorted_ids_sha256"],
            "task_manifest_sha256": private_manifest["task_manifest_sha256"],
            "pdf_inventory_sha256": private_manifest["pdf_inventory_sha256"],
            "gold_sha256": private_manifest["gold_sha256"],
            "dataset_card_sha256": release.release_card.sha256,
        },
        "operations": [
            {
                "target": "private_hugging_face",
                "status": statuses["private_hugging_face"],
                "paths": sorted(private_ops),
                "sha256": _artifact_hashes(private_ops),
                "policy": "exact-parent CAS; disabled and not finalized",
            },
            {
                "target": "github",
                "status": statuses["github"],
                "paths": sorted(github_ops),
                "sha256": _artifact_hashes(github_ops),
                "policy": "temporary clone; fast-forward push; never force",
            },
            {
                "target": "public_hugging_face",
                "status": statuses["public_hugging_face"],
                "paths": sorted(public_ops),
                "sha256": _artifact_hashes(public_ops),
                "policy": "exact-parent CAS; public test bytes and card together",
            },
        ],
        "safe_order": [
            "1. Publish the private disabled release and hidden labels by exact-parent CAS.",
            "2. Publish only docsem/test public inputs to canonical GitHub by non-force fast-forward.",
            "3. Publish the byte-identical public test payload and release card to Hugging Face by exact-parent CAS.",
            "4. Reconcile all three committed revisions and public reachable history; do not activate.",
        ],
        "partial_failure_recovery": [
            "Keep test submissions disabled after every partial result.",
            "Record completed sanitized revisions; never delete or rewrite a successful private disabled release.",
            "Refresh or fast-forward the canonical checkout, supply every new exact base, and regenerate the dry-run plan.",
            "Resume only when an existing target is byte-identical; conflicting partial releases fail closed.",
        ],
        "activation": "not-performed",
    }


def _validate_revision(value: str, description: str) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ReleaseError(f"{description} did not return an exact revision.")
    return value


def _partial_error(completed: Sequence[str]) -> PartialPublicationError:
    names = ",".join(completed)
    return PartialPublicationError(
        f"Publication stopped after safe completed targets: {names}. "
        "Test activation was not attempted; refresh exact bases and regenerate the plan."
    )


def _publish_target(
    name: str,
    expected_base: str,
    current: Callable[[], str],
    publish: Callable[[], str],
    completed: list[str],
) -> str:
    try:
        if current() != expected_base:
            raise RemoteMovedError("The target moved before publication.")
        revision = _validate_revision(publish(), f"{name} publication")
    except Exception as exc:
        if completed:
            raise _partial_error(completed) from exc
        raise ReleaseError(
            "Publication stopped before any target completed; exact remote state changed or a write failed."
        ) from exc
    completed.append(name)
    return revision


def _contains_forbidden_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            not isinstance(key, str)
            or _FORBIDDEN_PUBLIC_FIELD.search(key)
            or _contains_forbidden_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def _history_path_forbidden(path: str, namespace: str) -> bool:
    lowered = path.lower()
    if lowered.endswith(_ARCHIVE_SUFFIXES) and (
        path.startswith(namespace) or "/test/" in path or lowered.startswith("val/")
    ):
        return True
    if path.startswith(namespace) and _FORBIDDEN_PUBLIC_PATH_PART.search(
        Path(path).name
    ):
        return True
    if lowered in {
        "val/labels.jsonl",
        "validation/labels.jsonl",
        "docsem/val/labels.jsonl",
        "docsem/validation/labels.jsonl",
    }:
        return True
    return False


def _scan_public_history(snapshots: Sequence[HistorySnapshot], namespace: str) -> None:
    if len(snapshots) > MAX_HISTORY_COMMITS:
        raise ReleaseError("Public history exceeds the bounded reconciliation limit.")
    metadata_bytes = 0
    for snapshot in snapshots:
        if not _REVISION.fullmatch(snapshot.revision):
            raise ReleaseError("Public history contains an invalid revision.")
        paths = tuple(snapshot.paths)
        if any(not _safe_relative_path(path) for path in paths):
            raise ReleaseError("Public history contains an unsafe path.")
        if any(_history_path_forbidden(path, namespace) for path in paths):
            raise ReleaseError(
                "Public history contains a forbidden test or validation artifact."
            )
        for path, payload in snapshot.metadata.items():
            if path not in paths or not path.startswith(namespace):
                continue
            metadata_bytes += len(payload)
            if metadata_bytes > MAX_HISTORY_METADATA_BYTES:
                raise ReleaseError(
                    "Public history metadata exceeds the reconciliation limit."
                )
            try:
                if path.endswith(".jsonl"):
                    values = [
                        json.loads(line)
                        for line in payload.decode("utf-8").splitlines()
                    ]
                elif path.endswith(".json"):
                    values = json.loads(payload.decode("utf-8"))
                else:
                    continue
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReleaseError("Public history metadata is malformed.") from exc
            if _contains_forbidden_field(values):
                raise ReleaseError(
                    "Public history metadata contains a forbidden field."
                )


def _verify_remote_files(
    paths: Sequence[str],
    expected: Mapping[str, Artifact],
    reader: Callable[[Sequence[str]], Mapping[str, bytes]],
    namespace: str,
    *,
    companion_path: str | None = None,
) -> None:
    namespace_paths = {path for path in paths if path.startswith(namespace)}
    expected_namespace = {path for path in expected if path.startswith(namespace)}
    if namespace_paths != expected_namespace:
        raise ReleaseError(
            "A reconciled public/private release inventory is incomplete or extra."
        )
    requested = sorted(expected_namespace)
    if companion_path is not None:
        if companion_path not in paths:
            raise ReleaseError("A reconciled release metadata file is missing.")
        requested.append(companion_path)
    remote = reader(requested)
    if set(remote) != set(requested) or any(
        remote[path] != expected[path].read_bytes() for path in requested
    ):
        raise ReleaseError("A reconciled release byte or digest differs from staging.")


def _reconcile(
    config: ReleaseConfig,
    release: _CapturedRelease,
    revisions: Mapping[str, str],
    git_backend: GitBackend,
    hf_backend: HfBackend,
    public_token: str,
    private_token: str,
) -> dict[str, object]:
    github_ops = _github_operations(release)
    public_ops = _public_hf_operations(release)
    private_ops = _private_hf_operations(release)
    try:
        if (
            git_backend.current_revision(config.source_repository, config.source_branch)
            != revisions["github"]
        ):
            raise RemoteMovedError("GitHub moved before reconciliation.")
        if (
            hf_backend.current_revision(PUBLIC_HF_REPOSITORY, public_token)
            != revisions["public_hugging_face"]
        ):
            raise RemoteMovedError("Public Hugging Face moved before reconciliation.")
        if (
            hf_backend.current_revision(PRIVATE_HF_REPOSITORY, private_token)
            != revisions["private_hugging_face"]
        ):
            raise RemoteMovedError("Private Hugging Face moved before reconciliation.")

        github_paths = git_backend.list_paths(
            config.source_repository, revisions["github"]
        )
        _verify_remote_files(
            github_paths,
            github_ops,
            lambda names: git_backend.read_files(
                config.source_repository, revisions["github"], names
            ),
            "docsem/test/",
        )
        public_paths = hf_backend.list_paths(
            PUBLIC_HF_REPOSITORY, revisions["public_hugging_face"], public_token
        )
        _verify_remote_files(
            public_paths,
            public_ops,
            lambda names: hf_backend.read_files(
                PUBLIC_HF_REPOSITORY,
                revisions["public_hugging_face"],
                names,
                public_token,
            ),
            "test/",
            companion_path="README.md",
        )
        private_paths = hf_backend.list_paths(
            PRIVATE_HF_REPOSITORY, revisions["private_hugging_face"], private_token
        )
        _verify_remote_files(
            private_paths,
            private_ops,
            lambda names: hf_backend.read_files(
                PRIVATE_HF_REPOSITORY,
                revisions["private_hugging_face"],
                names,
                private_token,
            ),
            "private/test_",
        )
        _scan_public_history(
            git_backend.history_snapshots(
                config.source_repository, config.source_branch
            ),
            "docsem/test/",
        )
        _scan_public_history(
            hf_backend.history_snapshots(PUBLIC_HF_REPOSITORY, public_token),
            "test/",
        )
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError("Post-publication reconciliation failed safely.") from exc

    return {
        "mode": "published-and-reconciled",
        "release_id": release.public_manifest["release_id"],
        "counts": dict(release.private_manifest["counts"]),
        "hashes": {
            "task_manifest_sha256": release.private_manifest["task_manifest_sha256"],
            "pdf_inventory_sha256": release.private_manifest["pdf_inventory_sha256"],
            "gold_sha256": release.private_manifest["gold_sha256"],
            "dataset_card_sha256": release.release_card.sha256,
        },
        "revisions": dict(revisions),
        "activation": "not-performed",
    }


def run_release(
    *,
    config: ReleaseConfig,
    git_backend: GitBackend,
    hf_backend: HfBackend,
    public_token: str,
    private_token: str,
    publish: bool = False,
    confirmation: str | None = None,
    public_auditor: Callable[[Path], Mapping[str, object]] = audit_public_payload,
    card_renderer: Callable[..., bytes] = render_test_ready_dataset_card,
) -> dict[str, object]:
    """Return a sanitized dry-run plan or publish and reconcile it."""
    _validate_config(config)
    if (
        not isinstance(public_token, str)
        or not public_token
        or not isinstance(private_token, str)
        or not private_token
    ):
        raise ReleaseError(
            "Hugging Face credentials must come from the secure runtime environment."
        )
    if publish and confirmation != "PUBLISH":
        raise ReleaseError("Publication requires the exact confirmation word PUBLISH.")
    try:
        temporary = tempfile.TemporaryDirectory(prefix="docsem-publication-snapshot-")
    except OSError as exc:
        raise ReleaseError(
            "A private publication snapshot could not be created."
        ) from exc
    with temporary as temporary_name:
        snapshot_root = Path(temporary_name)
        snapshot_root.chmod(0o700)
        release = _capture_release(
            config,
            snapshot_root,
            public_auditor,
            card_renderer,
        )
        statuses, _source = _inspect_targets(
            config,
            release,
            git_backend,
            hf_backend,
            public_token,
            private_token,
        )
        plan = _publication_plan(config, release, statuses)
        if not publish:
            return plan

        completed: list[str] = []
        revisions: dict[str, str] = {}
        private_ops = _private_hf_operations(release)
        if statuses["private_hugging_face"] == "already-published":
            revisions["private_hugging_face"] = config.private_hf_base
            completed.append("private_hugging_face")
        else:
            revisions["private_hugging_face"] = _publish_target(
                "private_hugging_face",
                config.private_hf_base,
                lambda: hf_backend.current_revision(
                    PRIVATE_HF_REPOSITORY, private_token
                ),
                lambda: hf_backend.publish(
                    PRIVATE_HF_REPOSITORY,
                    config.private_hf_base,
                    private_ops,
                    f"Stage disabled DocSem test release {release.public_manifest['release_id']}",
                    private_token,
                ),
                completed,
            )

        github_ops = _github_operations(release)
        if statuses["github"] == "already-published":
            revisions["github"] = config.github_remote_base
            completed.append("github")
        else:
            revisions["github"] = _publish_target(
                "github",
                config.github_remote_base,
                lambda: git_backend.current_revision(
                    config.source_repository, config.source_branch
                ),
                lambda: git_backend.publish(
                    config.source_repository,
                    config.source_branch,
                    config.github_remote_base,
                    github_ops,
                    f"Release DocSem held-out test inputs {release.public_manifest['release_id']}",
                ),
                completed,
            )

        public_ops = _public_hf_operations(release)
        if statuses["public_hugging_face"] == "already-published":
            revisions["public_hugging_face"] = config.public_hf_base
            completed.append("public_hugging_face")
        else:
            revisions["public_hugging_face"] = _publish_target(
                "public_hugging_face",
                config.public_hf_base,
                lambda: hf_backend.current_revision(PUBLIC_HF_REPOSITORY, public_token),
                lambda: hf_backend.publish(
                    PUBLIC_HF_REPOSITORY,
                    config.public_hf_base,
                    public_ops,
                    f"Release DocSem held-out test inputs {release.public_manifest['release_id']}",
                    public_token,
                ),
                completed,
            )

        try:
            return _reconcile(
                config,
                release,
                revisions,
                git_backend,
                hf_backend,
                public_token,
                private_token,
            )
        except Exception as exc:
            raise _partial_error(completed) from exc


def _run_command(arguments: Sequence[str], *, cwd: Path | None = None) -> bytes:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseError("A guarded Git operation failed.") from exc


def _normalize_repository(value: str) -> str:
    normalized = value.removesuffix("/").removesuffix(".git")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.lower()


class SubprocessGitBackend:
    """Documented Git transport with a disposable mirror and non-force push."""

    def __init__(self) -> None:
        self._mirror_container: tempfile.TemporaryDirectory[str] | None = None
        self._mirror: Path | None = None
        self._repository: str | None = None

    def close(self) -> None:
        if self._mirror_container is not None:
            self._mirror_container.cleanup()
            self._mirror_container = None
            self._mirror = None

    def inspect_source(
        self, checkout: Path, repository: str, branch: str
    ) -> SourceState:
        checkout = Path(checkout)
        _require_directory(checkout, "Canonical source checkout")
        root = Path(
            _run_command(["git", "rev-parse", "--show-toplevel"], cwd=checkout)
            .decode()
            .strip()
        )
        actual_branch = (
            _run_command(["git", "branch", "--show-current"], cwd=checkout)
            .decode()
            .strip()
        )
        head = _run_command(["git", "rev-parse", "HEAD"], cwd=checkout).decode().strip()
        remote_url = (
            _run_command(["git", "remote", "get-url", "origin"], cwd=checkout)
            .decode()
            .strip()
        )
        remote_revision = self.current_revision(repository, branch)
        status_bytes = _run_command(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=checkout,
        )
        entries = [entry for entry in status_bytes.split(b"\0") if entry]
        untracked = any(entry.startswith(b"??") for entry in entries)
        tracked = any(not entry.startswith(b"??") for entry in entries)
        return SourceState(
            checkout=root,
            repository=(
                CANONICAL_GITHUB_REPOSITORY
                if _normalize_repository(remote_url)
                == _normalize_repository(repository)
                else remote_url
            ),
            branch=actual_branch,
            head=head,
            remote_revision=remote_revision,
            dirty_tracked=tracked,
            dirty_untracked=untracked,
            behind=head != remote_revision,
            diverged=False,
        )

    def current_revision(self, repository: str, branch: str) -> str:
        output = _run_command(
            ["git", "ls-remote", "--heads", repository, f"refs/heads/{branch}"]
        ).decode("utf-8")
        rows = [line.split() for line in output.splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0]) != 2:
            raise ReleaseError(
                "The canonical GitHub branch could not be resolved exactly."
            )
        return _validate_revision(rows[0][0], "GitHub branch")

    def _ensure_mirror(self, repository: str, revision: str | None = None) -> Path:
        if self._mirror is None:
            self._mirror_container = tempfile.TemporaryDirectory(
                prefix="docsem-git-mirror-"
            )
            self._mirror = Path(self._mirror_container.name) / "repository.git"
            _run_command(["git", "clone", "--mirror", repository, str(self._mirror)])
            self._repository = repository
        elif self._repository != repository:
            raise ReleaseError("The Git backend cannot mix repositories.")
        if revision is not None:
            try:
                _run_command(
                    ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                    cwd=self._mirror,
                )
            except ReleaseError:
                _run_command(["git", "fetch", "origin"], cwd=self._mirror)
                _run_command(
                    ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                    cwd=self._mirror,
                )
        return self._mirror

    def list_paths(self, repository: str, revision: str) -> Sequence[str]:
        mirror = self._ensure_mirror(repository, revision)
        output = _run_command(
            ["git", "ls-tree", "-r", "--name-only", revision], cwd=mirror
        )
        return tuple(line for line in output.decode("utf-8").splitlines() if line)

    def read_files(
        self, repository: str, revision: str, paths: Sequence[str]
    ) -> Mapping[str, bytes]:
        mirror = self._ensure_mirror(repository, revision)
        result = {}
        for path in paths:
            if not _safe_relative_path(path):
                raise ReleaseError("A requested Git path is unsafe.")
            result[path] = _run_command(
                ["git", "show", f"{revision}:{path}"], cwd=mirror
            )
        return result

    def publish(
        self,
        repository: str,
        branch: str,
        expected_parent: str,
        operations: Mapping[str, Artifact],
        message: str,
    ) -> str:
        if self.current_revision(repository, branch) != expected_parent:
            raise RemoteMovedError("GitHub moved before the non-force push.")
        with tempfile.TemporaryDirectory(prefix="docsem-github-publish-") as temporary:
            checkout = Path(temporary) / "checkout"
            _run_command(
                [
                    "git",
                    "clone",
                    "--branch",
                    branch,
                    "--single-branch",
                    repository,
                    str(checkout),
                ]
            )
            if (
                _run_command(["git", "rev-parse", "HEAD"], cwd=checkout)
                .decode()
                .strip()
                != expected_parent
            ):
                raise RemoteMovedError(
                    "GitHub moved while the temporary clone was created."
                )
            for path, artifact in operations.items():
                if not path.startswith("docsem/test/") or not _safe_relative_path(path):
                    raise ReleaseError(
                        "GitHub publication contains a path outside docsem/test."
                    )
                destination = checkout / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ReleaseError(
                        "The GitHub target already contains an unplanned test path."
                    )
                with destination.open("xb") as output:
                    output.write(artifact.read_bytes())
            _run_command(["git", "add", "--", *sorted(operations)], cwd=checkout)
            staged = set(
                _run_command(["git", "diff", "--cached", "--name-only"], cwd=checkout)
                .decode("utf-8")
                .splitlines()
            )
            if staged != set(operations):
                raise ReleaseError("The GitHub staged path allowlist is not exact.")
            _run_command(
                [
                    "git",
                    "-c",
                    "user.name=DocInsights Release Automation",
                    "-c",
                    "user.email=docinsights-release@users.noreply.github.com",
                    "commit",
                    "-m",
                    message,
                ],
                cwd=checkout,
            )
            revision = (
                _run_command(["git", "rev-parse", "HEAD"], cwd=checkout)
                .decode()
                .strip()
            )
            _run_command(
                ["git", "push", "origin", f"HEAD:refs/heads/{branch}"],
                cwd=checkout,
            )
            return _validate_revision(revision, "GitHub publication")

    def history_snapshots(
        self, repository: str, branch: str
    ) -> Sequence[HistorySnapshot]:
        revision = self.current_revision(repository, branch)
        mirror = self._ensure_mirror(repository, revision)
        revisions = _run_command(
            ["git", "rev-list", f"refs/heads/{branch}"], cwd=mirror
        )
        values = [line for line in revisions.decode("ascii").splitlines() if line]
        if len(values) > MAX_HISTORY_COMMITS:
            raise ReleaseError(
                "GitHub history exceeds the bounded reconciliation limit."
            )
        snapshots = []
        for value in values:
            paths = tuple(self.list_paths(repository, value))
            metadata_paths = [
                path
                for path in paths
                if path.startswith("docsem/test/")
                and path.endswith((".json", ".jsonl"))
            ]
            metadata = self.read_files(repository, value, metadata_paths)
            snapshots.append(HistorySnapshot(value, paths, metadata))
        return tuple(snapshots)


class HuggingFaceBackend:
    """Hugging Face Hub adapter using exact-parent create_commit operations."""

    @staticmethod
    def _imports():
        try:
            from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        except ImportError as exc:
            raise ReleaseError(
                "The pinned Hugging Face client is unavailable."
            ) from exc
        return CommitOperationAdd, HfApi, hf_hub_download

    def current_revision(self, repository: str, token: str) -> str:
        _CommitOperationAdd, HfApi, _download = self._imports()
        try:
            value = (
                HfApi(token=token)
                .repo_info(repo_id=repository, repo_type="dataset")
                .sha
            )
        except Exception as exc:
            raise ReleaseError(
                "A Hugging Face repository revision could not be resolved."
            ) from exc
        return _validate_revision(value, "Hugging Face repository")

    def list_paths(self, repository: str, revision: str, token: str) -> Sequence[str]:
        _CommitOperationAdd, HfApi, _download = self._imports()
        try:
            return tuple(
                HfApi(token=token).list_repo_files(
                    repo_id=repository,
                    repo_type="dataset",
                    revision=revision,
                )
            )
        except Exception as exc:
            raise ReleaseError(
                "A Hugging Face repository tree could not be inspected."
            ) from exc

    def read_files(
        self, repository: str, revision: str, paths: Sequence[str], token: str
    ) -> Mapping[str, bytes]:
        _CommitOperationAdd, _HfApi, download = self._imports()
        result = {}
        try:
            with tempfile.TemporaryDirectory(prefix="docsem-hf-read-") as cache:
                for path in paths:
                    if not _safe_relative_path(path):
                        raise ReleaseError("A requested Hugging Face path is unsafe.")
                    downloaded = Path(
                        download(
                            repo_id=repository,
                            repo_type="dataset",
                            filename=path,
                            revision=revision,
                            token=token,
                            cache_dir=cache,
                        )
                    )
                    resolved = downloaded.resolve(strict=True)
                    size = resolved.stat().st_size
                    limit = (
                        MAX_PDF_BYTES
                        if path.endswith(".pdf")
                        else MAX_HISTORY_METADATA_BYTES
                    )
                    if not resolved.is_file() or size > limit:
                        raise ReleaseError(
                            "A downloaded Hugging Face artifact is invalid."
                        )
                    result[path] = resolved.read_bytes()
        except ReleaseError:
            raise
        except Exception as exc:
            raise ReleaseError(
                "Hugging Face release bytes could not be downloaded."
            ) from exc
        return result

    def publish(
        self,
        repository: str,
        expected_parent: str,
        operations: Mapping[str, Artifact],
        message: str,
        token: str,
    ) -> str:
        CommitOperationAdd, HfApi, _download = self._imports()
        if self.current_revision(repository, token) != expected_parent:
            raise RemoteMovedError("Hugging Face moved before exact-parent commit.")
        additions = [
            CommitOperationAdd(
                path_in_repo=path,
                path_or_fileobj=str(operations[path].snapshot_path),
            )
            for path in sorted(operations)
        ]
        try:
            info = HfApi(token=token).create_commit(
                repo_id=repository,
                repo_type="dataset",
                revision="main",
                parent_commit=expected_parent,
                operations=additions,
                commit_message=message,
            )
        except Exception as exc:
            raise ReleaseError(
                "The exact-parent Hugging Face commit was refused."
            ) from exc
        return _validate_revision(info.oid, "Hugging Face publication")

    def history_snapshots(
        self, repository: str, token: str
    ) -> Sequence[HistorySnapshot]:
        _CommitOperationAdd, HfApi, _download = self._imports()
        try:
            commits = tuple(
                HfApi(token=token).list_repo_commits(
                    repo_id=repository,
                    repo_type="dataset",
                )
            )
        except Exception as exc:
            raise ReleaseError(
                "Hugging Face public history could not be enumerated."
            ) from exc
        if len(commits) > MAX_HISTORY_COMMITS:
            raise ReleaseError(
                "Hugging Face history exceeds the bounded reconciliation limit."
            )
        snapshots = []
        for commit in commits:
            revision = _validate_revision(commit.commit_id, "Hugging Face history")
            paths = tuple(self.list_paths(repository, revision, token))
            metadata_paths = [
                path
                for path in paths
                if path.startswith("test/") and path.endswith((".json", ".jsonl"))
            ]
            metadata = self.read_files(repository, revision, metadata_paths, token)
            snapshots.append(HistorySnapshot(revision, paths, metadata))
        return tuple(snapshots)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or explicitly publish one audited DocSem test release. "
            "Credentials are accepted only from the runtime environment or Git credential store."
        )
    )
    parser.add_argument("--public-stage", type=Path, required=True)
    parser.add_argument("--private-stage", type=Path, required=True)
    parser.add_argument("--release-card", type=Path, required=True)
    parser.add_argument("--card-template", type=Path, required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-base", required=True)
    parser.add_argument("--github-remote-base", required=True)
    parser.add_argument("--public-hf-base", required=True)
    parser.add_argument("--private-hf-base", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--confirm",
        help="Required with --publish and must be exactly PUBLISH.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    git_backend: GitBackend | None = None,
    hf_backend: HfBackend | None = None,
    public_token: str | None = None,
    private_token: str | None = None,
    public_auditor: Callable[[Path], Mapping[str, object]] = audit_public_payload,
    card_renderer: Callable[..., bytes] = render_test_ready_dataset_card,
) -> int:
    args = parse_args(argv)
    config = ReleaseConfig(
        public_stage=args.public_stage,
        private_stage=args.private_stage,
        release_card=args.release_card,
        card_template=args.card_template,
        source_checkout=CANONICAL_SOURCE_CHECKOUT,
        source_repository=CANONICAL_GITHUB_REPOSITORY,
        source_branch=args.source_branch,
        source_base=args.source_base,
        github_remote_base=args.github_remote_base,
        public_hf_base=args.public_hf_base,
        private_hf_base=args.private_hf_base,
    )
    git = git_backend or SubprocessGitBackend()
    hf = hf_backend or HuggingFaceBackend()
    private_credential = (
        private_token
        or os.environ.get("DOCSEM_PRIVATE_HF_TOKEN")
        or os.environ.get("HF_WRITE_TOKEN")
    )
    public_credential = (
        public_token or os.environ.get("DOCSEM_PUBLIC_HF_TOKEN") or private_credential
    )
    try:
        result = run_release(
            config=config,
            git_backend=git,
            hf_backend=hf,
            public_token=public_credential or "",
            private_token=private_credential or "",
            publish=args.publish,
            confirmation=args.confirm,
            public_auditor=public_auditor,
            card_renderer=card_renderer,
        )
    except ReleaseError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        return 2
    finally:
        close = getattr(git, "close", None)
        if callable(close):
            close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
