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
SOURCE_HEAD = "41c675bda8fa93662675f3dcd90fb7af4000cc22"
SOURCE_PARENT = "a4205880bfdd47aa3683050cd4a6ddf923fadffb"
SOURCE_MANIFEST_SHA256 = "3872e0beb953f91a4fc89558a0981fcf12553505bc744e80b68f53ae130e9d83"
RELEASE_ID = "docsem-test-a4205880-r1"
PUBLIC_HF_REPOSITORY = "amitbcp/docinsights-2026-shared-task-data"
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_FORBIDDEN = re.compile(r"(?:label|answer|evidence|gold|private|source.?mapping|archive|link|solution)", re.I)


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
    def upload_large_folder(self, stage: Path, expected_parent: str) -> str: ...
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


def _require_source_identity(source: SourceInspector) -> None:
    state = source.inspect(SOURCE_CHECKOUT)
    if not isinstance(state, SourceState) or (
        Path(state.checkout) != SOURCE_CHECKOUT or state.head != SOURCE_HEAD or
        state.parent != SOURCE_PARENT or state.dirty or
        state.manifest_sha256 != SOURCE_MANIFEST_SHA256
    ):
        raise ReleaseError("The selected public source is not the approved clean release.")


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


def _build_stage(config: ReleaseConfig) -> tuple[dict, dict[str, bytes]]:
    files, directories = _walk(Path(config.source_root), "Public source")
    if directories != {"documents"} or any(
        name not in {"tasks.jsonl", "release.json"} and not name.startswith("documents/")
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
    if files != {"tasks.jsonl", "release.json"} | document_files:
        raise ReleaseError("Public task PDFs are not an exact bijection.")

    normalized = [{**row, "document_pdf": f"test/documents/{row['instance_id']}.pdf"} for row in rows]
    task_bytes = b"".join(_json(row) for row in normalized)
    source_documents = [config.source_root / "documents" / f"{item}.pdf" for item in ids]
    pdf_digests = {path.name: _sha(_read(path, "Public PDF")) for path in source_documents}
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
    expected = {"test/tasks.jsonl": task_bytes, "test/release.json": manifest_bytes, "test/SHA256SUMS": checksum_bytes}
    expected.update({f"test/documents/{path.name}": _read(path, "Public PDF") for path in source_documents})

    stage = Path(config.stage)
    if stage.exists():
        try:
            audit_public_payload(stage)
            current = {p.relative_to(stage).as_posix(): _read(p, "Existing public stage") for p in stage.rglob("*") if p.is_file()}
        except Exception as exc: raise ReleaseError("Existing public stage is ambiguous.") from exc
        if current != expected: raise ReleaseError("Existing public stage differs from the approved payload.")
        return manifest, expected
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
    return manifest, expected


def _release_docs(stage: Path, base_readme: bytes, base_instructions: bytes) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="docsem-card-") as temporary:
        template = Path(temporary) / "README.md"; template.write_bytes(base_readme)
        readme = render_test_ready_dataset_card(stage, card_template_path=template)
    readme += (
        b"\n## Held-out test release\n\n"
        b"Public held-out test inputs are available in the `test` task split; submissions remain closed. "
        b"This audited release is `docsem-test-a4205880-r1`, derived from upstream `a4205880`. "
        b"Collision-remapped test filenames preserve byte-identical PDFs. This manifest does not claim those remapped names are on GitHub main.\n"
    )
    instructions = base_instructions + (
        b"\n## Held-out test evidence\n\n"
        b"For test instances, copy the opaque visible token immediately before the block colon exactly, including punctuation. "
        b"Do not infer a closed token grammar. Test submissions remain closed.\n"
    )
    return {"README.md": readme, "INSTRUCTIONS.md": instructions}


def _forbidden_public_path(path: str) -> bool:
    lower = path.lower()
    if lower == "train/labels.jsonl": return False
    return bool(_FORBIDDEN.search(path)) and (lower.startswith(("val/", "validation/", "test/", "private/")) or "private" in lower)


def _scan_history(hf: HfBackend) -> None:
    for _, paths in hf.history_snapshots():
        if any(_forbidden_public_path(path) for path in paths):
            raise ReleaseError("Public history contains a forbidden validation/test path.")


def _expect_state(hf: HfBackend, base: str) -> RemoteState:
    state = hf.state()
    if not isinstance(state, RemoteState) or state.private is not False: raise ReleaseError("The Hugging Face target is not public.")
    if not _REVISION.fullmatch(state.revision): raise ReleaseError("The Hugging Face target revision is invalid.")
    if state.revision != base: raise RemoteMovedError("The Hugging Face target moved from the expected base.")
    return state


def _test_status(hf: HfBackend, revision: str, expected: Mapping[str, bytes]) -> str:
    inventory = dict(hf.inventory(revision)); names = {name for name in inventory if name.startswith("test/")}
    if not names: return "missing"
    if names != set(expected): raise ReleaseError("Remote test inventory is not exact.")
    if any(inventory[name] != RemoteFile(len(data), _sha(data)) for name, data in expected.items()): raise ReleaseError("Remote test file hash or size differs.")
    metadata = hf.read(revision, ("test/tasks.jsonl", "test/release.json", "test/SHA256SUMS"))
    if metadata != {name: expected[name] for name in metadata} or len(metadata) != 3: raise ReleaseError("Remote test metadata differs.")
    return "complete"


def run_release(config: ReleaseConfig, *, source: SourceInspector, hf: HfBackend, publish: bool = False, confirm: str | None = None, expected_complete_base: str | None = None) -> dict:
    if not _REVISION.fullmatch(config.public_hf_base): raise ReleaseError("An exact public Hugging Face base is required.")
    if publish and confirm != "PUBLISH": raise ReleaseError("Publishing requires --confirm PUBLISH.")
    _require_source_identity(source)
    manifest, expected = _build_stage(config)
    state = _expect_state(hf, expected_complete_base or config.public_hf_base)
    _scan_history(hf)
    before = dict(hf.inventory(state.revision))
    if any(_forbidden_public_path(path) for path in before): raise ReleaseError("Public target contains a forbidden validation/test path.")
    base_docs = hf.read(state.revision, ("README.md", "INSTRUCTIONS.md"))
    if set(base_docs) != {"README.md", "INSTRUCTIONS.md"}: raise ReleaseError("Public documentation templates are unavailable.")
    test = _test_status(hf, state.revision, expected)
    if test == "complete" and (
        RELEASE_ID.encode() in base_docs["README.md"]
        and b"immediately before the block colon" in base_docs["INSTRUCTIONS.md"]
    ):
        # A prior completed public-only release is already the exact terminal
        # documentation state.  Do not render its release text a second time.
        docs, docs_complete = dict(base_docs), True
    else:
        docs = _release_docs(Path(config.stage), base_docs["README.md"], base_docs["INSTRUCTIONS.md"])
        docs_complete = False
    result = {"mode": "dry-run", "release_id": RELEASE_ID, "counts": dict(manifest["counts"]), "aggregate_digests": {key: manifest[key] for key in ("sorted_ids_sha256", "task_manifest_sha256", "pdf_inventory_sha256")}, "base_revision": state.revision}
    if not publish: return result
    if test == "complete" and docs_complete:
        return {**result, "mode": "already-complete", "revision": state.revision}
    current = state.revision
    if test == "missing":
        current = hf.upload_large_folder(Path(config.stage), current)
        if not _REVISION.fullmatch(current): raise ReleaseError("Test upload did not return an exact revision.")
        after_upload = dict(hf.inventory(current))
        if {name: info for name, info in after_upload.items() if not name.startswith("test/")} != {name: info for name, info in before.items() if not name.startswith("test/")}:
            raise ReleaseError("A non-test path changed before documentation publication.")
        _scan_history(hf); _test_status(hf, current, expected)
    else:
        current = state.revision
    # Exact-parent CAS for both documents is intentionally the final operation.
    final = hf.commit_docs(docs, current)
    if not _REVISION.fullmatch(final): raise ReleaseError("Documentation publication did not return an exact revision.")
    final_inventory = dict(hf.inventory(final)); _scan_history(hf); _test_status(hf, final, expected)
    if hf.read(final, ("README.md", "INSTRUCTIONS.md")) != docs: raise ReleaseError("Release documentation differs after publication.")
    if any(_forbidden_public_path(path) for path in final_inventory): raise ReleaseError("Final public target contains a forbidden path.")
    return {**result, "mode": "published", "revision": final}


class LocalSourceInspector:
    def __init__(self, source_manifest: Path): self.source_manifest = source_manifest
    def inspect(self, checkout: Path) -> SourceState:
        def git(*args: str) -> str: return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()
        try:
            return SourceState(checkout, git("rev-parse", "HEAD"), git("rev-parse", "HEAD^"), bool(git("status", "--porcelain")), _sha(_read(self.source_manifest, "Source manifest")))
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
            path = getattr(entry, "path", None) or getattr(entry, "rfilename", None)
            if not isinstance(path, str): continue
            lfs = getattr(entry, "lfs", None)
            if lfs and getattr(lfs, "sha256", None): inventory[path] = RemoteFile(int(getattr(lfs, "size", 0)), lfs.sha256)
            else:
                payload = self._download(revision, path); inventory[path] = RemoteFile(len(payload), _sha(payload))
        return inventory

    def read(self, revision: str, paths: Sequence[str]) -> Mapping[str, bytes]:
        return {path: self._download(revision, path) for path in paths}

    def history_snapshots(self) -> Sequence[tuple[str, Mapping[str, bytes]]]:
        try:
            commits = self.api.list_repo_commits(self.repository, repo_type="dataset")
            return tuple((commit.commit_id, {name: b"" for name in self.inventory(commit.commit_id)}) for commit in commits)
        except ReleaseError: raise
        except Exception as exc: raise ReleaseError("Public Hugging Face history cannot be inspected.") from exc

    def upload_large_folder(self, stage: Path, expected_parent: str) -> str:
        if self.state().revision != expected_parent: raise RemoteMovedError("The public Hugging Face base moved before test upload.")
        try:
            result = self.api.upload_large_folder(self.repository, repo_type="dataset", folder_path=str(stage / "test"), path_in_repo="test")
            return result.oid
        except Exception as exc: raise ReleaseError("Public test upload failed.") from exc

    def commit_docs(self, files: Mapping[str, bytes], expected_parent: str) -> str:
        if self.state().revision != expected_parent: raise RemoteMovedError("The public Hugging Face base moved before documentation CAS.")
        try:
            from huggingface_hub import CommitOperationAdd
            result = self.api.create_commit(self.repository, repo_type="dataset", parent_commit=expected_parent, commit_message=f"Publish {RELEASE_ID} documentation", operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=io.BytesIO(payload)) for name, payload in files.items()])
            return result.oid
        except Exception as exc: raise ReleaseError("Public release documentation CAS failed.") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run public-only DocSem held-out test release")
    parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True); parser.add_argument("--train-tasks", type=Path, required=True); parser.add_argument("--validation-tasks", type=Path, required=True)
    parser.add_argument("--public-hf-base", required=True); parser.add_argument("--publish", action="store_true"); parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    config = ReleaseConfig(args.source_root, args.stage, args.train_tasks, args.validation_tasks, args.public_hf_base)
    try:
        result = run_release(config, source=LocalSourceInspector(args.source_manifest), hf=HuggingFaceHubBackend(), publish=args.publish, confirm=args.confirm)
    except ReleaseError as exc:
        print(str(exc), file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
