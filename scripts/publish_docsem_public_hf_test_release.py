#!/usr/bin/env python3
"""Guarded, public-only publisher for the DocSem held-out test inputs.

This tool is deliberately dry-run by default.  It has no private-repository
configuration, no token argument, and no code path which reads labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Mapping, Protocol, Sequence

from prepare_docsem_hf_dataset import render_test_ready_dataset_card
from prepare_docsem_test_release import audit_public_payload

SOURCE_CHECKOUT = Path("/private/tmp/gsm-sem-docsem-test-release-a420588")
SOURCE_TASK_ROOT = SOURCE_CHECKOUT / "docsem/test"
SOURCE_HEAD = "41c675bda8fa93662675f3dcd90fb7af4000cc22"
SOURCE_PARENT = "a4205880bfdd47aa3683050cd4a6ddf923fadffb"
SOURCE_MANIFEST_SHA256 = "3872e0beb953f91a4fc89558a0981fcf12553505bc744e80b68f53ae130e9d83"
RELEASE_ID = "docsem-test-a4205880-r1"
PUBLIC_HF_REPOSITORY = "amitbcp/docinsights-2026-shared-task-data"
PUBLIC_HF_ORIGINAL_BASE = "e6c9c75bea7575a64279072dcdf0f6050fef9e9f"
PUBLIC_HF_GITATTRIBUTES_SHA256 = "9778c3c37d9a3a1cbfa1e28d446d10c85aea6b2b135a4a3210650089fb573301"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_README = REPO_ROOT / "competition/hf-dataset/README.md"
TRACKED_INSTRUCTIONS = REPO_ROOT / "competition/hf-dataset/INSTRUCTIONS.md"
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_FORBIDDEN = re.compile(r"(?:label|answer|evidence|gold|private|source.?mapping|archive|link|solution)", re.I)
_TEST_PDF_LFS_SUFFIX = b" filter=lfs diff=lfs merge=lfs -text\n"
_MAX_GITATTRIBUTES_BYTES = 1024 * 1024
MAX_HISTORY_COMMITS = 10_000
MAX_HISTORY_METADATA_BYTES = 64 * 1024 * 1024
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".7z", ".rar")
_SPLIT_COMPONENTS = frozenset({"val", "validation", "test"})
_CONTENT_EXTENSION = re.compile(
    r"\.(?:pdf|jsonl?|zip|tar|tgz|gz|7z|rar)(?=\.|$)", re.IGNORECASE
)
_FORBIDDEN_PUBLIC_PATH_PART = re.compile(
    r"(?:^|[._-])(?:answers?|evidence|gold|ground[_-]?truth|labels?|mapping|"
    r"organizer|private|solutions?|archives?)(?:$|[._-])",
    re.IGNORECASE,
)
_FORBIDDEN_PUBLIC_FIELD = re.compile(
    r"(?:^|[_-])(?:answers?|evidence|gold|ground[_-]?truth|labels?|solutions?|"
    r"source[_-]?mapping|organizer[_-]?note|private)(?:$|[_-])",
    re.IGNORECASE,
)


class ReleaseError(RuntimeError):
    """A sanitized fail-closed release refusal."""


class RemoteMovedError(ReleaseError):
    """The caller-supplied Hugging Face base is no longer current."""


@dataclass(frozen=True)
class SourceState:
    checkout: Path
    head: str
    parent: str
    dirty: bool
    manifest_sha256: str


@dataclass(frozen=True)
class RemoteState:
    revision: str
    private: bool


@dataclass(frozen=True)
class RemoteFile:
    size: int
    sha256: str


@dataclass(frozen=True)
class LocalFile:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseConfig:
    source_root: Path
    stage: Path
    train_tasks: Path
    validation_tasks: Path
    public_hf_base: str


class SourceInspector(Protocol):
    def inspect(self, checkout: Path) -> SourceState: ...


class HfBackend(Protocol):
    def state(self) -> RemoteState: ...
    def inventory(self, revision: str) -> Mapping[str, RemoteFile]: ...
    def read(self, revision: str, paths: Sequence[str]) -> Mapping[str, bytes]: ...
    def history_snapshots(self) -> Sequence[tuple[str, Mapping[str, bytes]]]: ...
    def upload_large_folder(self, stage: Path, expected_parent: str) -> None: ...
    def commit_docs(self, files: Mapping[str, bytes], expected_parent: str) -> str: ...


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _read(path: Path, description: str) -> bytes:
    try:
        item = path.lstat()
        if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
            raise ReleaseError(f"{description} is unsafe.")
        with path.open("rb") as handle:
            return handle.read()
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError(f"{description} is unavailable.") from exc


def _file_digest(path: Path, description: str) -> LocalFile:
    try:
        item = path.lstat()
        if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
            raise ReleaseError(f"{description} is unsafe.")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        final = path.stat()
        if final.st_ino != item.st_ino or final.st_size != item.st_size:
            raise ReleaseError(f"{description} changed while being read.")
        return LocalFile(path, item.st_size, digest.hexdigest())
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError(f"{description} is unavailable.") from exc


def _walk(root: Path, description: str) -> tuple[set[str], set[str]]:
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise ReleaseError(f"{description} is unavailable.") from exc
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ReleaseError(f"{description} is unsafe.")
    files, directories = set(), set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ReleaseError(f"{description} is unavailable.") from exc
        if _FORBIDDEN.search(relative) or path.is_symlink() or not relative or ".." in relative.split("/"):
            raise ReleaseError(f"{description} contains a forbidden path.")
        if stat.S_ISDIR(mode): directories.add(relative)
        elif stat.S_ISREG(mode): files.add(relative)
        else: raise ReleaseError(f"{description} contains a special file.")
    return files, directories


def _require_source_identity(source: SourceInspector, config: ReleaseConfig) -> None:
    state = source.inspect(SOURCE_CHECKOUT)
    if not isinstance(state, SourceState) or (
        Path(state.checkout) != SOURCE_CHECKOUT or state.head != SOURCE_HEAD or
        state.parent != SOURCE_PARENT or state.dirty or
        state.manifest_sha256 != SOURCE_MANIFEST_SHA256
    ):
        raise ReleaseError("The selected public source is not the approved clean release.")
    try:
        if Path(config.source_root).resolve() != SOURCE_TASK_ROOT.resolve():
            raise ReleaseError("The staged source root is not the approved checkout path.")
    except OSError as exc:
        raise ReleaseError("The approved source root is unavailable.") from exc


def _task_rows(path: Path, description: str) -> list[dict]:
    payload = _read(path, description)
    try: rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ReleaseError(f"{description} is malformed.") from exc
    if not rows or any(not isinstance(row, dict) for row in rows): raise ReleaseError(f"{description} is malformed.")
    return rows


def _ids(path: Path, description: str) -> set[str]:
    values = set()
    for row in _task_rows(path, description):
        value = row.get("instance_id")
        if not isinstance(value, str) or not _ID.fullmatch(value): raise ReleaseError(f"{description} is malformed.")
        values.add(value)
    return values


def prepare_stage(config: ReleaseConfig, *, source: SourceInspector) -> dict:
    """Explicit local write action; dry-run release planning never calls this."""
    _require_source_identity(source, config)
    files, directories = _walk(Path(config.source_root), "Public source")
    if directories != {"documents"} or any(
        name != "tasks.jsonl" and not name.startswith("documents/")
        for name in files
    ):
        raise ReleaseError("Public source inventory is not exact.")
    # Deliberately read only the approved public task manifest.  The source
    # release manifest is identity-attested by SourceInspector and is never
    # emitted; it is not parsed as a task/label side channel.
    rows = _task_rows(Path(config.source_root) / "tasks.jsonl", "Public tasks")
    ids = []
    for row in rows:
        if set(row) != {"instance_id", "user_query", "document_pdf"}:
            raise ReleaseError("Public task schema is invalid.")
        item = row.get("instance_id")
        if not isinstance(item, str) or not _ID.fullmatch(item) or not isinstance(row.get("user_query"), str) or not row["user_query"].strip() or row.get("document_pdf") != f"documents/{item}.pdf":
            raise ReleaseError("Public task schema is invalid.")
        ids.append(item)
    if ids != sorted(set(ids)) or set(ids) & (_ids(config.train_tasks, "Training tasks") | _ids(config.validation_tasks, "Validation tasks")):
        raise ReleaseError("Public task IDs are unsafe.")
    document_files = {f"documents/{item}.pdf" for item in ids}
    if files != {"tasks.jsonl"} | document_files:
        raise ReleaseError("Public task PDFs are not an exact bijection.")

    normalized = [{**row, "document_pdf": f"test/documents/{row['instance_id']}.pdf"} for row in rows]
    task_bytes = b"".join(_json(row) for row in normalized)
    source_documents = [config.source_root / "documents" / f"{item}.pdf" for item in ids]
    source_pdfs = {f"test/documents/{path.name}": _file_digest(path, "Public PDF") for path in source_documents}
    pdf_digests = {Path(name).name: item.sha256 for name, item in source_pdfs.items()}
    manifest = {
        "schema_version": 1, "release_id": RELEASE_ID,
        "counts": {"tasks": len(ids), "pdfs": len(ids)},
        "sorted_ids_sha256": _sha("".join(f"{item}\n" for item in ids).encode()),
        "task_manifest_sha256": _sha(task_bytes),
        "pdf_inventory_sha256": _sha(b"".join(f"{name}  {digest}\n".encode() for name, digest in sorted(pdf_digests.items()))),
    }
    manifest_bytes = _json(manifest)
    checksums = {"tasks.jsonl": _sha(task_bytes), "release.json": _sha(manifest_bytes), **{f"documents/{name}": digest for name, digest in pdf_digests.items()}}
    checksum_bytes = b"".join(f"{digest}  {name}\n".encode() for name, digest in sorted(checksums.items()))
    metadata = {"test/tasks.jsonl": task_bytes, "test/release.json": manifest_bytes, "test/SHA256SUMS": checksum_bytes}

    stage = Path(config.stage)
    if stage.exists():
        try:
            audit_public_payload(stage)
            current = {p.relative_to(stage).as_posix(): _read(p, "Existing public stage") for p in stage.rglob("*") if p.is_file() and not p.as_posix().endswith(".pdf")}
        except Exception as exc: raise ReleaseError("Existing public stage is ambiguous.") from exc
        if current != metadata: raise ReleaseError("Existing public stage differs from the approved payload.")
        for remote_name, source_pdf in source_pdfs.items():
            staged = _file_digest(stage / remote_name, "Existing public PDF")
            if (staged.size, staged.sha256) != (source_pdf.size, source_pdf.sha256):
                raise ReleaseError("Existing public stage differs from the approved payload.")
        return manifest
    if os.stat(config.source_root).st_dev != os.stat(stage.parent).st_dev:
        raise ReleaseError("Public stage is not on the source device for hardlinking.")
    stage.mkdir(mode=0o700)
    try:
        (stage / "test/documents").mkdir(parents=True, mode=0o700)
        (stage / "test/tasks.jsonl").write_bytes(task_bytes); (stage / "test/release.json").write_bytes(manifest_bytes); (stage / "test/SHA256SUMS").write_bytes(checksum_bytes)
        for source_path in source_documents:
            target = stage / "test/documents" / source_path.name
            try: os.link(source_path, target)
            except OSError as exc: raise ReleaseError("Hardlinks are unavailable for public PDFs.") from exc
            if source_path.stat().st_dev != target.stat().st_dev or source_path.stat().st_ino != target.stat().st_ino:
                raise ReleaseError("Public PDF hardlink verification failed.")
        audit_public_payload(stage)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def _audited_stage_manifest(config: ReleaseConfig, *, source: SourceInspector) -> dict:
    """Read-only stage audit for dry-runs and publication.

    Source data is read one file at a time; no PDF corpus is materialized.
    """
    _require_source_identity(source, config)
    stage = Path(config.stage)
    if not stage.exists():
        raise ReleaseError("No prepared audited public stage exists; run --prepare-stage first.")
    try:
        manifest = audit_public_payload(stage)
    except Exception as exc:
        raise ReleaseError("Prepared public stage audit failed.") from exc
    rows = _task_rows(SOURCE_TASK_ROOT / "tasks.jsonl", "Public tasks")
    normalized = [{**row, "document_pdf": f"test/documents/{row['instance_id']}.pdf"} for row in rows]
    if manifest.get("task_manifest_sha256") != _sha(b"".join(_json(row) for row in normalized)):
        raise ReleaseError("Prepared public stage does not match the approved source.")
    for row in rows:
        source_pdf = _file_digest(SOURCE_TASK_ROOT / row["document_pdf"], "Public PDF")
        staged_pdf = _file_digest(stage / f"test/documents/{row['instance_id']}.pdf", "Prepared public PDF")
        if (source_pdf.size, source_pdf.sha256) != (staged_pdf.size, staged_pdf.sha256):
            raise ReleaseError("Prepared public stage does not match the approved source.")
    return manifest


def _release_docs() -> dict[str, bytes]:
    base_readme = _read(TRACKED_README, "Tracked dataset card")
    base_instructions = _read(TRACKED_INSTRUCTIONS, "Tracked participant instructions")
    marker = b"  - split: validation\n    path: val/tasks.jsonl\n"
    if base_readme.count(marker) != 1:
        raise ReleaseError("Tracked dataset card cannot be rendered deterministically.")
    readme = base_readme.replace(marker, marker + b"  - split: test\n    path: test/tasks.jsonl\n", 1)
    replacements = {
        b"- **Held-out test:** not released in the current public payload. After the organizers publish an audited release, the `test` task split will contain tasks and PDFs without labels.": b"- **Held-out test:** 1,730 public test tasks and PDFs are available without labels; submissions and scoring remain closed.",
        b"No official held-out test payload is present in this revision, so the tracked dataset-card configuration intentionally declares only the existing train and validation task files. The release generator adds the `test` configuration to a deterministic release-card payload only after an explicitly selected public staging tree passes the complete audit. It must not be populated from similarly named local directories or archives.": b"The audited release exposes the public `test` task split with 1,730 tasks and byte-identical PDFs. The labels configuration remains train-only; validation and test labels are not public.",
        b"- Declared the held-out `test` task-split contract and participant policy. The official test files are not included in this revision and the test submission window is not open.": b"- Released the audited public held-out test inputs. Test submission and scoring remain closed.",
        b"available only after the audited release is published.": b"available in this audited public release.",
        b"once released.": b"in this audited release.",
    }
    for old, new in replacements.items():
        if old in readme:
            readme = readme.replace(old, new)
    readme = readme.replace(
        b"Held-out test: not released in the current public payload.",
        b"Held-out test: 1,730 public test tasks and PDFs are available; submissions and scoring remain closed.",
    ).replace(
        b"No official held-out test payload is present in this revision.",
        b"The audited public test payload contains 1,730 tasks and PDFs; labels config remains train-only.",
    )
    start = b"### Held-out test submission policy\n"
    end = b"The public release contains only tasks, PDFs, checksums, and sanitized release metadata."
    if start in readme and end in readme:
        prefix, remainder = readme.split(start, 1)
        _, suffix = remainder.split(end, 1)
        readme = prefix + start + b"\nPublic test inputs are available, but test submissions and scoring remain closed until a separate organizer announcement.\n\n" + end + suffix
    readme = readme.replace(
        b"- `evidence` must be a non-empty list of visible PDF block IDs such as `b01`.",
        b"- For train and validation, `evidence` is a non-empty list of visible PDF block IDs such as `b01`; for test, copy the opaque token before the colon exactly, including punctuation.",
    ).replace(
        b"Only answer accuracy and evidence F1 are returned for the first held-out test attempt; later-attempt metrics and all per-example test results remain organizer-only until finalization.",
        b"Test submissions and scoring remain closed; no held-out test metrics or per-example results are returned in this release.",
    ).replace(
        b"This Hugging Face package mirrors the source release's 908 training tasks, 908 training labels, 217 validation tasks, and all 1,125 PDFs. The PDF files are byte-identical. The Hugging Face task manifests only prefix `document_pdf` with `train/` or `val/` so files resolve directly from this repository's root.",
        b"This Hugging Face package contains 908 training tasks and labels, 217 validation tasks, and 1,730 public test tasks with PDFs. Test labels are not installed or public; the labels config remains train-only. PDFs are byte-identical and task paths resolve from this repository root.",
    ).replace(
        b"Validation and test labels are not included in this public dataset. They remain in the access-restricted organizer evaluation repository and are used only by the submission services.",
        b"Validation labels remain organizer-only. Test labels are not installed in this public release or a submission service; public test inputs are available while submissions and scoring remain closed.",
    ).replace(
        b"# After the official release, reloading this config will also provide tasks[\"test\"].",
        b"# This audited release provides tasks[\"test\"] public inputs; test submissions and scoring remain closed.",
    )
    readme += (
        b"\n## Held-out test release\n\n"
        b"Public held-out test inputs are available in the `test` task split: 1,730 test tasks and PDFs; submissions remain closed and scoring remains closed. "
        b"This audited release is `docsem-test-a4205880-r1`, derived from upstream `a4205880`. "
        b"Collision-remapped test filenames preserve byte-identical PDFs. This manifest does not claim those remapped names are on GitHub main.\n"
    )
    instructions = base_instructions.replace(
        b"Every content block begins with a visible identifier in the form `b01: <block content>`; use these identifiers when reporting evidence.",
        b"For train and validation, visible block identifiers such as `b01:` remain the evidence convention. For test, copy the opaque visible token immediately before the colon exactly, including punctuation; do not infer a closed token grammar.",
    ).replace(
        b"Every content block begins with `b01: content`.",
        b"Train and validation use visible block identifiers; test uses the opaque token before the colon exactly, including punctuation.",
    ).replace(
        b"The public package contains labelled train data and unlabelled validation inputs.\nValidation labels remain private and are used only by the official submission portal.",
        b"The public package contains labelled train data, unlabelled validation inputs, and 1,730 unlabelled public test inputs. Validation labels remain private. Test labels are not installed, and test submissions/scoring remain closed.",
    ) + (
        b"\n## Held-out test evidence\n\n"
        b"For test instances, copy the opaque visible token immediately before the block colon exactly, including punctuation. "
        b"Do not infer a closed token grammar. Test submissions remain closed.\n"
    )
    docs = {"README.md": readme, "INSTRUCTIONS.md": instructions}
    _audit_release_docs(docs)
    return docs


def _audit_release_docs(docs: Mapping[str, bytes]) -> None:
    required = {
        "README.md": (b"1,730 test tasks and PDFs", b"labels config remains train-only", b"submissions remain closed", b"scoring remains closed"),
        "INSTRUCTIONS.md": (b"before the colon exactly, including punctuation", b"Test labels are not installed", b"test submissions/scoring remain closed"),
    }
    forbidden = (
        b"After the official release", b"all 1,125 PDFs", b"used only by the submission services",
        b"Validation and test labels are not included in this public dataset", b"Every content block begins with a visible identifier",
    )
    if set(docs) != set(required) or any(item not in docs[name] for name, items in required.items() for item in items) or any(item in payload for payload in docs.values() for item in forbidden):
        raise ReleaseError("Generated release documentation is internally inconsistent.")


def _make_upload_root(stage: Path) -> Path:
    """Make an uploader cache outside the audited stage without copying PDFs."""
    try:
        root = stage.parent / f".{stage.name}-upload-{RELEASE_ID}"
        if root == stage or stage in root.parents:
            raise ReleaseError("Upload cache overlaps the audited stage.")
        if not root.exists(): root.mkdir(mode=0o700)
        _reconcile_upload_root(root, stage)
        return root
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("A separate resumable upload cache could not be prepared.") from exc


def _reconcile_upload_root(root: Path, stage: Path) -> None:
    """Audit a reusable uploader cache before it can be uploaded.

    Only exact stage files and the uploader-owned .cache/.huggingface subtree
    may exist. Matching incomplete payload files are completed without
    overwriting an existing path.
    """
    if root.is_symlink() or not root.is_dir(): raise ReleaseError("Existing upload cache is unsafe.")
    expected = {p.relative_to(stage).as_posix(): p for p in (stage / "test").rglob("*") if p.is_file()}
    seen = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ReleaseError("Existing upload cache contains an unsafe entry.")
        if relative == ".cache":
            if not stat.S_ISDIR(mode): raise ReleaseError("Existing upload cache contains an unsafe cache entry.")
            continue
        if relative.startswith(".cache/"):
            if relative != ".cache/huggingface" and not relative.startswith(".cache/huggingface/"):
                raise ReleaseError("Existing upload cache contains an unexpected cache entry.")
            if relative == ".cache/huggingface" and not stat.S_ISDIR(mode):
                raise ReleaseError("Existing upload cache contains an unsafe cache entry.")
            continue
        if stat.S_ISDIR(mode):
            if relative not in {"test", "test/documents"}:
                raise ReleaseError("Existing upload cache contains an unexpected directory.")
            continue
        if relative not in expected:
            raise ReleaseError("Existing upload cache contains an unexpected payload file.")
        source = _file_digest(expected[relative], "Prepared public payload")
        current = _file_digest(path, "Existing upload payload")
        if (source.size, source.sha256) != (current.size, current.sha256):
            raise ReleaseError("Existing upload cache differs from the audited stage.")
        seen.add(relative)
    (root / "test/documents").mkdir(parents=True, exist_ok=True)
    for relative, source in expected.items():
        destination = root / relative
        if destination.exists(): continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith("test/documents/"):
            os.link(source, destination)
            if source.stat().st_ino != destination.stat().st_ino:
                raise ReleaseError("Upload cache PDF hardlink verification failed.")
        else:
            shutil.copyfile(source, destination)


def _safe_relative_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith(("/", "\\"))
        and "\\" not in path
        and all(part not in {"", ".", ".."} for part in path.split("/"))
        and all(ord(character) >= 32 and ord(character) != 127 for character in path)
    )


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


def _split_sensitive_metadata_path(path: str) -> bool:
    lowered = path.lower()
    components = lowered.split("/")
    return (
        bool(_SPLIT_COMPONENTS.intersection(components[:-1]))
        and lowered.endswith((".json", ".jsonl"))
        and len(_CONTENT_EXTENSION.findall(components[-1])) == 1
    )


def _history_path_forbidden(path: str) -> bool:
    lowered = path.lower()
    components = lowered.split("/")
    filename = components[-1]
    extension_markers = _CONTENT_EXTENSION.findall(filename)
    if "private" in components:
        return True
    if any(
        re.search(rf"{re.escape(suffix)}(?=\.|$)", filename)
        for suffix in _ARCHIVE_SUFFIXES
    ):
        return True
    if len(extension_markers) > 1:
        return True
    if extension_markers and not re.search(r"\.(?:pdf|jsonl?)\Z", filename):
        return True
    if path == "train/labels.jsonl":
        return False
    return any(_FORBIDDEN_PUBLIC_PATH_PART.search(component) for component in components)


def _forbidden_public_path(path: str) -> bool:
    lower = path.lower()
    if lower == "train/labels.jsonl": return False
    return bool(_FORBIDDEN.search(path)) and (lower.startswith(("val/", "validation/", "test/", "private/")) or "private" in lower)


def _scan_history(hf: HfBackend) -> None:
    snapshots = hf.history_snapshots()
    if len(snapshots) > MAX_HISTORY_COMMITS:
        raise ReleaseError("Public history exceeds the bounded reconciliation limit.")
    metadata_bytes = 0
    for revision, paths in snapshots:
        if not _REVISION.fullmatch(revision) or not isinstance(paths, Mapping):
            raise ReleaseError("Public history contains an invalid snapshot.")
        names = tuple(paths)
        if any(not _safe_relative_path(path) for path in names):
            raise ReleaseError("Public history contains an unsafe path.")
        if any(_history_path_forbidden(path) for path in names):
            raise ReleaseError("Public history contains a forbidden validation/test path.")
        for path in names:
            if not _split_sensitive_metadata_path(path):
                continue
            payload = paths[path]
            if not isinstance(payload, bytes):
                raise ReleaseError("Public history metadata is invalid.")
            metadata_bytes += len(payload)
            if metadata_bytes > MAX_HISTORY_METADATA_BYTES:
                raise ReleaseError("Public history metadata exceeds the reconciliation limit.")
            try:
                if path.lower().endswith(".jsonl"):
                    value = [
                        json.loads(line)
                        for line in payload.decode("utf-8").splitlines()
                    ]
                else:
                    value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReleaseError("Public history metadata is malformed.") from exc
            if _contains_forbidden_field(value):
                raise ReleaseError("Public history metadata contains a forbidden field.")


def _expect_state(hf: HfBackend, base: str) -> RemoteState:
    state = hf.state()
    if not isinstance(state, RemoteState) or state.private is not False: raise ReleaseError("The Hugging Face target is not public.")
    if not _REVISION.fullmatch(state.revision): raise ReleaseError("The Hugging Face target revision is invalid.")
    if state.revision != base: raise RemoteMovedError("The Hugging Face target moved from the expected base.")
    return state


def _public_state(hf: HfBackend) -> RemoteState:
    state = hf.state()
    if not isinstance(state, RemoteState) or state.private is not False:
        raise ReleaseError("The Hugging Face target is not public.")
    if not _REVISION.fullmatch(state.revision):
        raise ReleaseError("The Hugging Face target revision is invalid.")
    return state


def _stage_expectations(stage: Path) -> tuple[dict[str, bytes], dict[str, RemoteFile]]:
    metadata = {name: _read(stage / name, "Prepared public metadata") for name in ("test/tasks.jsonl", "test/release.json", "test/SHA256SUMS")}
    remote = {name: RemoteFile(len(payload), _sha(payload)) for name, payload in metadata.items()}
    for path in (stage / "test/documents").glob("*.pdf"):
        item = _file_digest(path, "Prepared public PDF")
        remote[f"test/documents/{path.name}"] = RemoteFile(item.size, item.sha256)
    return metadata, remote


def _test_status(hf: HfBackend, revision: str, metadata: Mapping[str, bytes], expected: Mapping[str, RemoteFile]) -> str:
    inventory = dict(hf.inventory(revision)); names = {name for name in inventory if name.startswith("test/")}
    if not names: return "missing"
    if not names.issubset(expected): raise ReleaseError("Remote test inventory is not exact.")
    if any(
        inventory[name] != expected[name]
        for name in expected
        if name.startswith("test/documents/") and name in inventory
    ) or any(inventory[name].size != len(payload) for name, payload in metadata.items() if name in inventory):
        raise ReleaseError("Remote test file hash or size differs.")
    present_metadata = tuple(name for name in metadata if name in names)
    if hf.read(revision, present_metadata) != {name: metadata[name] for name in present_metadata}:
        raise ReleaseError("Remote test metadata differs.")
    return "complete" if names == set(expected) else "partial"


def _remote_bytes(hf: HfBackend, revision: str, paths: Sequence[str], description: str) -> dict[str, bytes]:
    try:
        payloads = hf.read(revision, paths)
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError(f"{description} is unavailable.") from exc
    if not isinstance(payloads, Mapping) or set(payloads) != set(paths) or any(
        not isinstance(payload, bytes) for payload in payloads.values()
    ):
        raise ReleaseError(f"{description} is invalid.")
    return dict(payloads)


def _audit_test_pdf_attributes(
    hf: HfBackend,
    revision: str,
    inventory: Mapping[str, RemoteFile],
    expected: Mapping[str, RemoteFile],
) -> None:
    info = inventory.get(".gitattributes")
    if not isinstance(info, RemoteFile) or info.size < 0 or info.size > _MAX_GITATTRIBUTES_BYTES:
        raise ReleaseError("Public Git attributes are invalid.")
    payload = _remote_bytes(hf, revision, (".gitattributes",), "Public Git attributes")[".gitattributes"]
    if len(payload) != info.size or len(payload) > _MAX_GITATTRIBUTES_BYTES:
        raise ReleaseError("Public Git attributes are invalid.")
    expected_rules = {
        path.encode("utf-8") + _TEST_PDF_LFS_SUFFIX
        for path in inventory
        if path.startswith("test/documents/") and path.endswith(".pdf") and path in expected
    }
    rules, baseline = [], []
    for line in payload.splitlines(keepends=True):
        if line in expected_rules:
            rules.append(line)
        else:
            baseline.append(line)
    if len(rules) != len(expected_rules) or set(rules) != expected_rules:
        raise ReleaseError("Public test PDF Git LFS rules are not exact.")
    if _sha(b"".join(baseline)) != PUBLIC_HF_GITATTRIBUTES_SHA256:
        raise ReleaseError("Public Git attributes differ from the approved baseline.")


def _original_public_inventory(hf: HfBackend, expected: Mapping[str, RemoteFile]) -> dict[str, RemoteFile]:
    try:
        baseline = dict(hf.inventory(PUBLIC_HF_ORIGINAL_BASE))
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError("Original public Hugging Face inventory is unavailable.") from exc
    if any(path.startswith("test/") or _forbidden_public_path(path) for path in baseline):
        raise ReleaseError("Original public Hugging Face inventory is invalid.")
    _audit_test_pdf_attributes(hf, PUBLIC_HF_ORIGINAL_BASE, baseline, expected)
    return baseline


def _audit_non_test_state(
    hf: HfBackend,
    revision: str,
    inventory: Mapping[str, RemoteFile],
    baseline: Mapping[str, RemoteFile],
    expected: Mapping[str, RemoteFile],
    docs: Mapping[str, bytes],
) -> bool:
    current = {name: info for name, info in inventory.items() if not name.startswith("test/")}
    if set(current) != set(baseline):
        raise ReleaseError("Public non-test inventory differs from the approved baseline.")
    for name, info in baseline.items():
        if name not in {".gitattributes", "README.md", "INSTRUCTIONS.md"} and current.get(name) != info:
            raise ReleaseError("A non-test public file differs from the approved baseline.")
    _audit_test_pdf_attributes(hf, revision, inventory, expected)
    if all(current.get(name) == baseline.get(name) for name in docs):
        return False
    if _remote_bytes(hf, revision, tuple(docs), "Public release documentation") != docs:
        raise ReleaseError("Public release documentation is neither approved baseline nor exact release content.")
    return True


def run_release(config: ReleaseConfig, *, source: SourceInspector, hf: HfBackend, publish: bool = False, confirm: str | None = None, expected_complete_base: str | None = None) -> dict:
    if not _REVISION.fullmatch(config.public_hf_base): raise ReleaseError("An exact public Hugging Face base is required.")
    if publish and confirm != "PUBLISH": raise ReleaseError("Publishing requires --confirm PUBLISH.")
    manifest = _audited_stage_manifest(config, source=source)
    metadata, expected = _stage_expectations(Path(config.stage))
    state = _expect_state(hf, expected_complete_base or config.public_hf_base)
    _scan_history(hf)
    baseline = _original_public_inventory(hf, expected)
    before = dict(hf.inventory(state.revision))
    if any(_forbidden_public_path(path) for path in before): raise ReleaseError("Public target contains a forbidden validation/test path.")
    docs = _release_docs()
    test = _test_status(hf, state.revision, metadata, expected)
    docs_complete = _audit_non_test_state(hf, state.revision, before, baseline, expected, docs)
    result = {"mode": "dry-run", "release_id": RELEASE_ID, "counts": dict(manifest["counts"]), "aggregate_digests": {key: manifest[key] for key in ("sorted_ids_sha256", "task_manifest_sha256", "pdf_inventory_sha256")}, "base_revision": state.revision}
    if not publish: return result
    if test == "complete" and docs_complete:
        return {**result, "mode": "already-complete", "revision": state.revision}
    current = state.revision
    if test in {"missing", "partial"}:
        upload_root = _make_upload_root(Path(config.stage))
        _expect_state(hf, current)
        hf.upload_large_folder(upload_root, current)
        current = _public_state(hf).revision
        after_upload = dict(hf.inventory(current))
        if _audit_non_test_state(hf, current, after_upload, baseline, expected, docs):
            raise ReleaseError("Release documentation changed during test upload.")
        _scan_history(hf)
        if _test_status(hf, current, metadata, expected) != "complete":
            return {**result, "mode": "partial-upload", "revision": current}
    else:
        current = state.revision
    # Exact-parent CAS for both documents is intentionally the final operation.
    _expect_state(hf, current)
    final = hf.commit_docs(docs, current)
    if not _REVISION.fullmatch(final): raise ReleaseError("Documentation publication did not return an exact revision.")
    if _expect_state(hf, final).revision != final: raise ReleaseError("Final public revision changed unexpectedly.")
    final_inventory = dict(hf.inventory(final)); _scan_history(hf); _test_status(hf, final, metadata, expected)
    if not _audit_non_test_state(hf, final, final_inventory, baseline, expected, docs):
        raise ReleaseError("Release documentation differs after publication.")
    if any(_forbidden_public_path(path) for path in final_inventory): raise ReleaseError("Final public target contains a forbidden path.")
    if set(final_inventory) != set(baseline) | set(expected):
        raise ReleaseError("Final public inventory changed unexpectedly.")
    return {**result, "mode": "published", "revision": final}


class LocalSourceInspector:
    def inspect(self, checkout: Path) -> SourceState:
        def git(*args: str) -> str: return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()
        try:
            return SourceState(checkout, git("rev-parse", "HEAD"), git("rev-parse", "HEAD^"), bool(git("status", "--porcelain")), _sha(_read(SOURCE_TASK_ROOT / "tasks.jsonl", "Source manifest")))
        except (OSError, subprocess.CalledProcessError) as exc: raise ReleaseError("Public source checkout cannot be inspected.") from exc


class HuggingFaceHubBackend:
    """Public dataset adapter using the Hub's resumable large-folder API.

    Authentication is intentionally delegated to the locally configured Hub
    client.  This class never accepts, stores, or prints an access token.
    """

    def __init__(self, repository: str = PUBLIC_HF_REPOSITORY):
        self.repository = repository
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ReleaseError("The locally authenticated Hugging Face client is unavailable.") from exc
        self.api = HfApi()

    def state(self) -> RemoteState:
        try:
            info = self.api.repo_info(self.repository, repo_type="dataset")
            return RemoteState(info.sha, bool(info.private))
        except Exception as exc: raise ReleaseError("Public Hugging Face target cannot be inspected.") from exc

    def _paths(self, revision: str):
        try: return tuple(self.api.list_repo_tree(self.repository, repo_type="dataset", revision=revision, recursive=True, expand=True))
        except Exception as exc: raise ReleaseError("Public Hugging Face inventory cannot be inspected.") from exc

    def _download(self, revision: str, path: str) -> bytes:
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(self.repository, path, repo_type="dataset", revision=revision)).read_bytes()
        except Exception as exc: raise ReleaseError("Public Hugging Face file cannot be verified.") from exc

    def inventory(self, revision: str) -> Mapping[str, RemoteFile]:
        inventory = {}
        for entry in self._paths(revision):
            # RepoFolder has a tree_id but no byte size; never download it.
            if getattr(entry, "size", None) is None:
                continue
            path = getattr(entry, "path", None) or getattr(entry, "rfilename", None)
            if not isinstance(path, str): continue
            lfs = getattr(entry, "lfs", None)
            if lfs and getattr(lfs, "sha256", None): inventory[path] = RemoteFile(int(getattr(lfs, "size", 0)), lfs.sha256)
            else:
                inventory[path] = RemoteFile(int(entry.size), str(getattr(entry, "blob_id", "")))
        return inventory

    def read(self, revision: str, paths: Sequence[str]) -> Mapping[str, bytes]:
        return {path: self._download(revision, path) for path in paths}

    def history_snapshots(self) -> Sequence[tuple[str, Mapping[str, bytes]]]:
        try:
            commits = tuple(self.api.list_repo_commits(self.repository, repo_type="dataset"))
            if len(commits) > MAX_HISTORY_COMMITS:
                raise ReleaseError("Public history exceeds the bounded reconciliation limit.")
            snapshots = []
            metadata_bytes = 0
            for commit in commits:
                revision = getattr(commit, "commit_id", None)
                if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
                    raise ReleaseError("Public history contains an invalid revision.")
                inventory = dict(self.inventory(revision))
                paths = tuple(inventory)
                if any(not _safe_relative_path(path) for path in paths):
                    raise ReleaseError("Public history contains an unsafe path.")
                metadata_paths = tuple(
                    path for path in paths if _split_sensitive_metadata_path(path)
                )
                for path in metadata_paths:
                    info = inventory[path]
                    if not isinstance(info, RemoteFile) or info.size < 0:
                        raise ReleaseError("Public history metadata inventory is invalid.")
                    metadata_bytes += info.size
                    if metadata_bytes > MAX_HISTORY_METADATA_BYTES:
                        raise ReleaseError("Public history metadata exceeds the reconciliation limit.")
                metadata = self.read(revision, metadata_paths) if metadata_paths else {}
                if set(metadata) != set(metadata_paths) or any(
                    not isinstance(metadata[path], bytes)
                    or len(metadata[path]) != inventory[path].size
                    for path in metadata_paths
                ):
                    raise ReleaseError("Public history metadata is invalid.")
                snapshots.append((
                    revision,
                    {
                        path: metadata[path] if path in metadata else b""
                        for path in paths
                    },
                ))
            return tuple(snapshots)
        except ReleaseError:
            raise
        except Exception as exc: raise ReleaseError("Public Hugging Face history cannot be inspected.") from exc

    def upload_large_folder(self, stage: Path, expected_parent: str) -> None:
        _expect_state(self, expected_parent)
        try:
            self.api.upload_large_folder(
                self.repository,
                repo_type="dataset",
                folder_path=str(stage),
                allow_patterns="test/**",
                ignore_patterns=[".cache/**", "**/.cache/**"],
                num_workers=2,
            )
        except Exception as exc: raise ReleaseError("Public test upload failed.") from exc

    def commit_docs(self, files: Mapping[str, bytes], expected_parent: str) -> str:
        _expect_state(self, expected_parent)
        try:
            from huggingface_hub import CommitOperationAdd
            result = self.api.create_commit(self.repository, repo_type="dataset", parent_commit=expected_parent, commit_message=f"Publish {RELEASE_ID} documentation", operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=io.BytesIO(payload)) for name, payload in files.items()])
            return result.oid
        except Exception as exc: raise ReleaseError("Public release documentation CAS failed.") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run public-only DocSem held-out test release")
    parser.add_argument("--stage", type=Path, required=True); parser.add_argument("--train-tasks", type=Path, required=True); parser.add_argument("--validation-tasks", type=Path, required=True)
    parser.add_argument("--public-hf-base", required=True); parser.add_argument("--prepare-stage", action="store_true"); parser.add_argument("--publish", action="store_true"); parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    config = ReleaseConfig(SOURCE_TASK_ROOT, args.stage, args.train_tasks, args.validation_tasks, args.public_hf_base)
    try:
        inspector = LocalSourceInspector()
        if args.prepare_stage:
            result = {"mode": "prepared", "release_id": RELEASE_ID, "counts": prepare_stage(config, source=inspector)["counts"]}
        else:
            result = run_release(config, source=inspector, hf=HuggingFaceHubBackend(), publish=args.publish, confirm=args.confirm)
    except ReleaseError as exc:
        print(str(exc), file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
