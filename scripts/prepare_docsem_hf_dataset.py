#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from prepare_docsem_test_release import (
    MAX_PDF_BYTES,
    MAX_PUBLIC_CHECKSUM_BYTES,
    MAX_PUBLIC_MANIFEST_BYTES,
    MAX_PUBLIC_TASKS_BYTES,
    ValidationError,
    audit_public_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = Path("/Users/aamita/Oracle/amitbcp/gsm-sem/docsem")
PUBLIC_LICENSE = PUBLIC_SOURCE_ROOT.parent / "LICENSE.txt"
ORGANIZER_SOURCE_ROOT = Path(
    "/Users/aamita/Oracle/amitbcp/docsem-workshop-final-public"
)
TARGET_ROOT = REPO_ROOT / "competition/hf-dataset"
PRIVATE_ROOT = REPO_ROOT / "competition/hf-submissions"
DATASET_CARD_TEMPLATE_PATH = REPO_ROOT / "competition/hf-dataset/README.md"

MAX_SOURCE_METADATA_BYTES = 64 * 1024 * 1024
MAX_SOURCE_TEXT_BYTES = 16 * 1024 * 1024
_TEST_TASK_CONFIG_ENTRY = "  - split: test\n    path: test/tasks.jsonl\n"
_VALIDATION_TASK_CONFIG_ENTRY = "  - split: validation\n    path: val/tasks.jsonl\n"
_TRAIN_LABELS_CONFIG = (
    "- config_name: labels\n"
    "  data_files:\n"
    "  - split: train\n"
    "    path: train/labels.jsonl\n"
)

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_CLOEXEC
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_CLOEXEC
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class _BackupPreservedError(ValidationError):
    """An install failed and the only recoverable prior copy remains in backup."""


@dataclass(frozen=True, repr=False)
class _SourcePDF:
    snapshot_path: Path
    name: str
    size: int
    sha256: str

    def __repr__(self):
        return f"_SourcePDF(size={self.size}, sealed=True)"


@dataclass(frozen=True, repr=False)
class _OrdinarySourceSnapshot:
    train_tasks: tuple[dict, ...]
    train_labels: tuple[dict, ...]
    val_tasks: tuple[dict, ...]
    val_labels: tuple[dict, ...]
    train_pdfs: tuple[_SourcePDF, ...]
    val_pdfs: tuple[_SourcePDF, ...]
    instructions: bytes
    license_text: bytes
    base_dataset_card: bytes

    def __repr__(self):
        return (
            "_OrdinarySourceSnapshot("
            f"train_tasks={len(self.train_tasks)}, "
            f"train_labels={len(self.train_labels)}, "
            f"val_tasks={len(self.val_tasks)}, "
            f"val_labels={len(self.val_labels)}, "
            f"train_pdfs={len(self.train_pdfs)}, "
            f"val_pdfs={len(self.val_pdfs)})"
        )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_identity(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_source_file(path, max_bytes, description):
    descriptor = None
    try:
        descriptor = os.open(path, _FILE_OPEN_FLAGS)
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or not 0 <= initial.st_size <= max_bytes:
            raise ValidationError(f"{description} is not a bounded regular file.")
        remaining = initial.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValidationError(f"{description} changed while being captured.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValidationError(f"{description} changed while being captured.")
        final = os.fstat(descriptor)
        if _stable_identity(initial) != _stable_identity(final):
            raise ValidationError(f"{description} changed while being captured.")
        return b"".join(chunks)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"{description} is absent or unreadable.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _parse_source_jsonl(path, description):
    payload = _read_source_file(path, MAX_SOURCE_METADATA_BYTES, description)
    try:
        return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{description} is not valid UTF-8 JSONL.") from exc


def _capture_task_rows(split):
    rows = []
    seen = set()
    for row in _parse_source_jsonl(
        PUBLIC_SOURCE_ROOT / split / "tasks.jsonl",
        f"Public {split} task manifest",
    ):
        if not isinstance(row, dict):
            raise ValidationError(f"Public {split} task manifest has an invalid row.")
        instance_id = row.get("instance_id")
        query = row.get("user_query")
        source_pdf = row.get("document_pdf")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or Path(instance_id).name != instance_id
            or not isinstance(query, str)
            or source_pdf != f"documents/{instance_id}.pdf"
            or instance_id in seen
        ):
            raise ValidationError(f"Public {split} task manifest has an invalid row.")
        seen.add(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "user_query": query,
                "document_pdf": f"{split}/{source_pdf}",
            }
        )
    if not rows:
        raise ValidationError(f"Public {split} task manifest is empty.")
    return tuple(rows)


def _capture_label_rows(source_root, split, description):
    rows = []
    seen = set()
    for row in _parse_source_jsonl(source_root / split / "labels.jsonl", description):
        if not isinstance(row, dict):
            raise ValidationError(f"{description} has an invalid row.")
        instance_id = row.get("instance_id")
        evidence = row.get("evidence")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id in seen
            or "answer" not in row
            or not isinstance(evidence, list)
        ):
            raise ValidationError(f"{description} has an invalid row.")
        seen.add(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "answer": str(row["answer"]),
                "evidence": [str(value) for value in evidence],
            }
        )
    if not rows:
        raise ValidationError(f"{description} is empty.")
    return tuple(rows)


def _capture_source_file_to_snapshot(source_path, snapshot_path, max_bytes, description):
    source_descriptor = destination_descriptor = None
    try:
        source_descriptor = os.open(source_path, _FILE_OPEN_FLAGS)
        initial = os.fstat(source_descriptor)
        if not stat.S_ISREG(initial.st_mode) or not 0 <= initial.st_size <= max_bytes:
            raise ValidationError(f"{description} is not a bounded regular file.")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        snapshot_path.parent.chmod(0o700)
        destination_descriptor = os.open(
            snapshot_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        remaining = initial.st_size
        copied = 0
        while remaining:
            chunk = os.read(source_descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValidationError(f"{description} changed while being captured.")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("short snapshot write")
                view = view[written:]
            copied += len(chunk)
            remaining -= len(chunk)
        if os.read(source_descriptor, 1):
            raise ValidationError(f"{description} changed while being captured.")
        final = os.fstat(source_descriptor)
        if _stable_identity(initial) != _stable_identity(final):
            raise ValidationError(f"{description} changed while being captured.")
        os.fsync(destination_descriptor)
        snapshot_path.chmod(0o400)
        return copied, digest.hexdigest()
    except ValidationError:
        snapshot_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        snapshot_path.unlink(missing_ok=True)
        raise ValidationError(f"{description} could not be captured safely.") from exc
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)


def _capture_pdf_inventory(split, expected_ids, snapshot_root):
    source_directory = PUBLIC_SOURCE_ROOT / split / "documents"
    try:
        directory_mode = source_directory.lstat().st_mode
        if not stat.S_ISDIR(directory_mode) or stat.S_ISLNK(directory_mode):
            raise ValidationError(f"Public {split} PDF directory is not a real directory.")
        entries = sorted(os.scandir(source_directory), key=lambda entry: entry.name)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"Public {split} PDF directory is absent or unreadable.") from exc

    snapshots = []
    seen_ids = set()
    for entry in entries:
        if (
            not entry.name.endswith(".pdf")
            or not entry.is_file(follow_symlinks=False)
            or Path(entry.name).name != entry.name
        ):
            raise ValidationError(f"Public {split} PDF inventory has an invalid entry.")
        instance_id = Path(entry.name).stem
        if instance_id in seen_ids:
            raise ValidationError(f"Public {split} PDF inventory has duplicate IDs.")
        snapshot_path = Path(snapshot_root) / split / entry.name
        size, digest = _capture_source_file_to_snapshot(
            Path(entry.path),
            snapshot_path,
            MAX_PDF_BYTES,
            f"Public {split} PDF",
        )
        snapshots.append(
            _SourcePDF(
                snapshot_path=snapshot_path,
                name=entry.name,
                size=size,
                sha256=digest,
            )
        )
        seen_ids.add(instance_id)
    if seen_ids != set(expected_ids):
        raise ValidationError(f"Public {split} task IDs and PDF IDs differ.")
    return tuple(snapshots)


def _normalize_base_dataset_card(card_bytes):
    try:
        text = card_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("Dataset card template is not UTF-8.") from exc
    parts = text.split("---\n", 2)
    if len(parts) != 3 or parts[0]:
        raise ValidationError("Dataset card template has invalid front matter.")
    front_matter = parts[1]
    if front_matter.count(_VALIDATION_TASK_CONFIG_ENTRY) != 1:
        raise ValidationError("Dataset card template lacks the validation task config.")
    if front_matter.count(_TEST_TASK_CONFIG_ENTRY) > 1:
        raise ValidationError("Dataset card template has duplicate test task configs.")
    front_matter = front_matter.replace(_TEST_TASK_CONFIG_ENTRY, "")
    if (
        front_matter.count(_TRAIN_LABELS_CONFIG) != 1
        or front_matter.count("config_name: labels") != 1
        or front_matter.count("labels.jsonl") != 1
    ):
        raise ValidationError("Dataset card template labels config is not train-only.")
    return ("---\n" + front_matter + "---\n" + parts[2]).encode("utf-8")


def _capture_ordinary_sources(snapshot_root):
    snapshot_root = Path(snapshot_root)
    snapshot_root.mkdir(parents=True, mode=0o700)
    snapshot_root.chmod(0o700)
    train_tasks = _capture_task_rows("train")
    val_tasks = _capture_task_rows("val")
    train_labels = _capture_label_rows(
        PUBLIC_SOURCE_ROOT,
        "train",
        "Public train labels",
    )
    val_labels = _capture_label_rows(
        ORGANIZER_SOURCE_ROOT,
        "val",
        "Private validation labels",
    )
    train_ids = {row["instance_id"] for row in train_tasks}
    val_ids = {row["instance_id"] for row in val_tasks}
    if train_ids != {row["instance_id"] for row in train_labels}:
        raise ValidationError("Train task IDs and label IDs differ.")
    if val_ids != {row["instance_id"] for row in val_labels}:
        raise ValidationError("Validation task IDs and label IDs differ.")
    train_pdfs = _capture_pdf_inventory("train", train_ids, snapshot_root)
    val_pdfs = _capture_pdf_inventory("val", val_ids, snapshot_root)
    instructions = _read_source_file(
        PUBLIC_SOURCE_ROOT / "PARTICIPANT_INSTRUCTIONS.md",
        MAX_SOURCE_TEXT_BYTES,
        "Participant instructions",
    )
    license_text = _read_source_file(
        PUBLIC_LICENSE,
        MAX_SOURCE_TEXT_BYTES,
        "Public license",
    )
    base_dataset_card = _normalize_base_dataset_card(
        _read_source_file(
            DATASET_CARD_TEMPLATE_PATH,
            MAX_SOURCE_TEXT_BYTES,
            "Dataset card template",
        )
    )
    return _OrdinarySourceSnapshot(
        train_tasks=train_tasks,
        train_labels=train_labels,
        val_tasks=val_tasks,
        val_labels=val_labels,
        train_pdfs=train_pdfs,
        val_pdfs=val_pdfs,
        instructions=instructions,
        license_text=license_text,
        base_dataset_card=base_dataset_card,
    )


def _open_real_directory(path, *, dir_fd=None):
    descriptor = None
    try:
        descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS, dir_fd=dir_fd)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValidationError("Public staging contains an invalid directory.")
        return descriptor
    except ValidationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ValidationError("Public staging is absent or unreadable.") from exc


def _read_regular_at(directory_fd, name, max_bytes):
    descriptor = None
    try:
        descriptor = os.open(name, _FILE_OPEN_FLAGS, dir_fd=directory_fd)
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or not 0 <= initial.st_size <= max_bytes:
            raise ValidationError("Public staging contains an invalid file.")
        remaining = initial.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValidationError("Public staging changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValidationError("Public staging changed while being read.")
        final = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, field) != getattr(final, field) for field in stable_fields):
            raise ValidationError("Public staging changed while being read.")
        return b"".join(chunks)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Public staging is absent or unreadable.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _capture_audited_public_test(staging_root):
    """Capture only the already-audited public test allowlist."""
    root = Path(staging_root)
    audited_manifest = audit_public_payload(root)
    root_fd = test_fd = documents_fd = None
    try:
        root_fd = _open_real_directory(root)
        test_fd = _open_real_directory("test", dir_fd=root_fd)
        documents_fd = _open_real_directory("documents", dir_fd=test_fd)
        tasks_bytes = _read_regular_at(test_fd, "tasks.jsonl", MAX_PUBLIC_TASKS_BYTES)
        release_bytes = _read_regular_at(
            test_fd,
            "release.json",
            MAX_PUBLIC_MANIFEST_BYTES,
        )
        checksum_bytes = _read_regular_at(
            test_fd,
            "SHA256SUMS",
            MAX_PUBLIC_CHECKSUM_BYTES,
        )
        try:
            task_rows = [json.loads(line) for line in tasks_bytes.decode("utf-8").splitlines()]
            captured_manifest = json.loads(release_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("Public staging metadata changed after audit.") from exc
        if (
            captured_manifest != audited_manifest
            or hashlib.sha256(tasks_bytes).hexdigest()
            != audited_manifest["task_manifest_sha256"]
            or len(task_rows) != audited_manifest["counts"]["tasks"]
        ):
            raise ValidationError("Public staging metadata changed after audit.")

        payload = {
            "test/tasks.jsonl": tasks_bytes,
            "test/release.json": release_bytes,
            "test/SHA256SUMS": checksum_bytes,
        }
        pdf_digests = []
        for row in task_rows:
            instance_id = row.get("instance_id") if isinstance(row, dict) else None
            document_pdf = row.get("document_pdf") if isinstance(row, dict) else None
            expected_path = f"test/documents/{instance_id}.pdf"
            if not isinstance(instance_id, str) or document_pdf != expected_path:
                raise ValidationError("Public staging task paths changed after audit.")
            filename = f"{instance_id}.pdf"
            if Path(filename).name != filename:
                raise ValidationError("Public staging task paths changed after audit.")
            pdf_bytes = _read_regular_at(documents_fd, filename, MAX_PDF_BYTES)
            payload[expected_path] = pdf_bytes
            pdf_digests.append((filename, hashlib.sha256(pdf_bytes).hexdigest()))

        inventory_bytes = b"".join(
            f"{name}  {digest}\n".encode("ascii")
            for name, digest in sorted(pdf_digests)
        )
        if (
            len(pdf_digests) != audited_manifest["counts"]["pdfs"]
            or hashlib.sha256(inventory_bytes).hexdigest()
            != audited_manifest["pdf_inventory_sha256"]
        ):
            raise ValidationError("Public staging PDFs changed after audit.")
        _verify_captured_public_test(payload, audited_manifest)
        return payload, audited_manifest
    finally:
        for descriptor in (documents_fd, test_fd, root_fd):
            if descriptor is not None:
                os.close(descriptor)


def _verify_captured_public_test(payload, audited_manifest):
    expected_keys = {
        "test/tasks.jsonl",
        "test/release.json",
        "test/SHA256SUMS",
    } | {key for key in payload if key.startswith("test/documents/") and key.endswith(".pdf")}
    if set(payload) != expected_keys:
        raise ValidationError("Captured public test payload has an invalid inventory.")
    try:
        captured_manifest = json.loads(payload["test/release.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Captured public test manifest is invalid.") from exc
    if captured_manifest != audited_manifest:
        raise ValidationError("Captured public test manifest changed after audit.")
    checksum_targets = sorted(
        key.removeprefix("test/") for key in payload if key != "test/SHA256SUMS"
    )
    expected_checksum_bytes = b"".join(
        (f"{hashlib.sha256(payload[f'test/{name}']).hexdigest()}  {name}\n").encode("ascii")
        for name in checksum_targets
    )
    if payload.get("test/SHA256SUMS") != expected_checksum_bytes:
        raise ValidationError("Captured public checksum inventory changed after audit.")


def _render_test_ready_dataset_card(base_card_bytes, captured_payload, audited_manifest):
    _verify_captured_public_test(captured_payload, audited_manifest)
    base_card = _normalize_base_dataset_card(base_card_bytes)
    text = base_card.decode("utf-8")
    front_matter, body = text.removeprefix("---\n").split("---\n", 1)
    if front_matter.count(_VALIDATION_TASK_CONFIG_ENTRY) != 1:
        raise ValidationError("Dataset card template lacks the validation task config.")
    release_front_matter = front_matter.replace(
        _VALIDATION_TASK_CONFIG_ENTRY,
        _VALIDATION_TASK_CONFIG_ENTRY + _TEST_TASK_CONFIG_ENTRY,
        1,
    )
    return ("---\n" + release_front_matter + "---\n" + body).encode("utf-8")


def render_test_ready_dataset_card(staging_root, *, card_template_path=None):
    """Return release-card bytes only after capturing an explicit audited test tree."""
    captured_payload, audited_manifest = _capture_audited_public_test(staging_root)
    template = (
        DATASET_CARD_TEMPLATE_PATH if card_template_path is None else Path(card_template_path)
    )
    base_card = _read_source_file(
        template,
        MAX_SOURCE_TEXT_BYTES,
        "Dataset card template",
    )
    return _render_test_ready_dataset_card(
        base_card,
        captured_payload,
        audited_manifest,
    )


def _write_public_test_snapshot(root, payload):
    try:
        manifest = json.loads(payload["test/release.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Captured public test manifest is invalid.") from exc
    _verify_captured_public_test(payload, manifest)
    test_root = root / "test"
    documents_root = test_root / "documents"
    documents_root.mkdir(parents=True, mode=0o755)
    test_root.chmod(0o755)
    documents_root.chmod(0o755)
    for relative_name, content in sorted(payload.items()):
        destination = root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(content)
        destination.chmod(0o644)


def _remove_generated_test_output(target_root):
    destination = Path(target_root) / "test"
    try:
        mode = destination.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValidationError("Generated test output could not be inspected safely.") from exc
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        shutil.rmtree(destination)
    else:
        destination.unlink()


def _require_real_dataset_root(target_root):
    target = Path(target_root)
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValidationError("Dataset output root could not be inspected safely.") from exc
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ValidationError("Dataset output root is not a real directory.")


def _install_public_test_snapshot(target_root, payload):
    try:
        manifest = json.loads(payload["test/release.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Captured public test manifest is invalid.") from exc
    _verify_captured_public_test(payload, manifest)

    target = Path(target_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target_mode = target.lstat().st_mode
        if not stat.S_ISDIR(target_mode) or stat.S_ISLNK(target_mode):
            raise ValidationError("Dataset output root is not a real directory.")
    else:
        target.mkdir()

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".docsem-hf-public-test-",
            dir=target.parent,
        )
    )
    backup_root = None
    prior_test = target / "test"
    installed = False
    restored = False
    try:
        temporary_root.chmod(0o755)
        _write_public_test_snapshot(temporary_root, payload)
        audit_public_payload(temporary_root)
        try:
            prior_mode = prior_test.lstat().st_mode
        except FileNotFoundError:
            prior_mode = None
        except OSError as exc:
            raise ValidationError("Prior generated test output could not be inspected.") from exc
        if prior_mode is not None:
            if not stat.S_ISDIR(prior_mode) or stat.S_ISLNK(prior_mode):
                raise ValidationError("Prior generated test output is not a real directory.")
            backup_root = Path(
                tempfile.mkdtemp(
                    prefix=".docsem-hf-public-test-backup-",
                    dir=target.parent,
                )
            )
            backup_root.chmod(0o700)
            os.replace(prior_test, backup_root / "test")
        try:
            os.replace(temporary_root / "test", prior_test)
            installed = True
        except OSError:
            if backup_root is not None and (backup_root / "test").exists():
                try:
                    os.replace(backup_root / "test", prior_test)
                    restored = True
                except OSError as restore_exc:
                    raise ValidationError(
                        f"Audited public test payload could not be installed; prior snapshot remains at {backup_root}."
                    ) from restore_exc
            raise
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Audited public test payload could not be installed.") from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        if backup_root is not None and (installed or restored):
            shutil.rmtree(backup_root, ignore_errors=True)


def _write_bytes(path, payload, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
    path.chmod(mode)


def _copy_captured_pdfs(target_root, split, snapshots):
    target_directory = Path(target_root) / split / "documents"
    if target_directory.exists():
        raise ValidationError(f"Generated {split} PDF directory already exists.")
    target_directory.mkdir(parents=True, exist_ok=True)
    for snapshot in snapshots:
        source_descriptor = destination_descriptor = None
        destination = target_directory / snapshot.name
        try:
            source_descriptor = os.open(snapshot.snapshot_path, _FILE_OPEN_FLAGS)
            initial = os.fstat(source_descriptor)
            if not stat.S_ISREG(initial.st_mode) or initial.st_size != snapshot.size:
                raise ValidationError(f"Captured {split} PDF snapshot is invalid.")
            destination_descriptor = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            digest = hashlib.sha256()
            remaining = snapshot.size
            while remaining:
                chunk = os.read(source_descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValidationError(f"Captured {split} PDF snapshot is incomplete.")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        raise OSError("short generated PDF write")
                    view = view[written:]
                remaining -= len(chunk)
            if os.read(source_descriptor, 1):
                raise ValidationError(f"Captured {split} PDF snapshot is oversized.")
            final = os.fstat(source_descriptor)
            if (
                _stable_identity(initial) != _stable_identity(final)
                or digest.hexdigest() != snapshot.sha256
            ):
                raise ValidationError(f"Captured {split} PDF snapshot changed.")
            os.fsync(destination_descriptor)
            destination.chmod(0o644)
        except ValidationError:
            destination.unlink(missing_ok=True)
            raise
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise ValidationError(f"Captured {split} PDF could not be copied safely.") from exc
        finally:
            if destination_descriptor is not None:
                os.close(destination_descriptor)
            if source_descriptor is not None:
                os.close(source_descriptor)


def _materialize_public_generation(root, ordinary, dataset_card, captured_test):
    root = Path(root)
    root.mkdir(mode=0o755)
    write_jsonl(root / "train" / "tasks.jsonl", ordinary.train_tasks)
    write_jsonl(root / "train" / "labels.jsonl", ordinary.train_labels)
    write_jsonl(root / "val" / "tasks.jsonl", ordinary.val_tasks)

    _copy_captured_pdfs(root, "train", ordinary.train_pdfs)
    _copy_captured_pdfs(root, "val", ordinary.val_pdfs)

    sample_submission = [
        {
            "instance_id": row["instance_id"],
            "answer": "0",
            "evidence": ["b01"],
        }
        for row in ordinary.val_tasks[:5]
    ]
    write_jsonl(root / "examples" / "sample_val_submission.jsonl", sample_submission)
    _write_bytes(root / "INSTRUCTIONS.md", ordinary.instructions)
    _write_bytes(root / "LICENSE.txt", ordinary.license_text)
    if captured_test is not None:
        _write_public_test_snapshot(root, captured_test)
    _write_bytes(root / "README.md", dataset_card)


def _ensure_real_directory(path, mode, created_directories, changed_directory_modes):
    path = Path(path)
    try:
        current_mode = path.lstat().st_mode
    except FileNotFoundError:
        path.mkdir(mode=mode)
        created_directories.append(path)
        return
    except OSError as exc:
        raise ValidationError("Generated output directory could not be inspected safely.") from exc
    if not stat.S_ISDIR(current_mode) or stat.S_ISLNK(current_mode):
        raise ValidationError("Generated output directory is not a real directory.")
    prior_permissions = stat.S_IMODE(current_mode)
    if prior_permissions != mode:
        path.chmod(mode)
        changed_directory_modes.append((path, prior_permissions))


def _validate_install_path(path, *, directory):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValidationError("Generated output path could not be inspected safely.") from exc
    expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected or stat.S_ISLNK(mode):
        raise ValidationError("Generated output path has an unsafe type.")
    return True


def _install_path_with_backup(staged_path, destination, backup_path, *, directory):
    staged_path = Path(staged_path)
    destination = Path(destination)
    backup_path = Path(backup_path)
    if not _validate_install_path(staged_path, directory=directory):
        raise ValidationError("Generated staged output is missing.")
    had_prior = _validate_install_path(destination, directory=directory)
    if had_prior:
        os.replace(destination, backup_path)
    try:
        os.replace(staged_path, destination)
    except OSError:
        if had_prior:
            try:
                os.replace(backup_path, destination)
            except OSError as restore_exc:
                raise _BackupPreservedError(
                    f"Generated output install failed; prior output remains at {backup_path}."
                ) from restore_exc
        raise
    return (staged_path, destination, backup_path, had_prior)


def _rollback_installed_paths(records):
    for staged_path, destination, backup_path, had_prior in reversed(records):
        try:
            os.replace(destination, staged_path)
            if had_prior:
                os.replace(backup_path, destination)
        except OSError as exc:
            raise _BackupPreservedError(
                f"Generated output rollback failed; prior output remains at {backup_path}."
            ) from exc


def _install_generated_outputs(staged_public, staged_val_labels, staged_leaderboard):
    target = Path(TARGET_ROOT)
    private_root = Path(PRIVATE_ROOT)
    created_directories = []
    changed_directory_modes = []
    created_files = []
    public_backup = private_backup = None
    records = []
    rollback_complete = False
    try:
        _ensure_real_directory(
            private_root,
            0o700,
            created_directories,
            changed_directory_modes,
        )
        _ensure_real_directory(
            private_root / "private",
            0o700,
            created_directories,
            changed_directory_modes,
        )
        leaderboard_directory = private_root / "leaderboard"
        if staged_leaderboard is not None or leaderboard_directory.exists():
            _ensure_real_directory(
                leaderboard_directory,
                0o700,
                created_directories,
                changed_directory_modes,
            )
        submissions = private_root / "submissions"
        _ensure_real_directory(
            submissions,
            0o700,
            created_directories,
            changed_directory_modes,
        )
        keep_file = submissions / ".gitkeep"
        if not keep_file.exists():
            with keep_file.open("xb"):
                pass
            created_files.append(keep_file)
            keep_file.chmod(0o600)

        public_backup = Path(
            tempfile.mkdtemp(
                prefix=".docsem-hf-public-backup-",
                dir=target.parent,
            )
        )
        private_backup = Path(
            tempfile.mkdtemp(
                prefix=".docsem-hf-private-backup-",
                dir=private_root.parent,
            )
        )
        public_backup.chmod(0o700)
        private_backup.chmod(0o700)

        records.append(
            _install_path_with_backup(
                staged_public,
                target,
                public_backup / "dataset",
                directory=True,
            )
        )
        records.append(
            _install_path_with_backup(
                staged_val_labels,
                private_root / "private/val_labels.jsonl",
                private_backup / "val_labels.jsonl",
                directory=False,
            )
        )
        if staged_leaderboard is not None:
            records.append(
                _install_path_with_backup(
                    staged_leaderboard,
                    private_root / "leaderboard/leaderboard.json",
                    private_backup / "leaderboard.json",
                    directory=False,
                )
            )

    except (OSError, ValidationError) as exc:
        preserve_backups = isinstance(exc, _BackupPreservedError)
        mode_restore_error = None
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        try:
            _rollback_installed_paths(records)
            rollback_complete = not preserve_backups
        finally:
            for path, prior_permissions in reversed(changed_directory_modes):
                try:
                    path.chmod(prior_permissions)
                except OSError as restore_exc:
                    if mode_restore_error is None:
                        mode_restore_error = restore_exc
            for path in reversed(created_directories):
                try:
                    path.rmdir()
                except OSError:
                    pass
        if mode_restore_error is not None:
            raise ValidationError(
                "Generated output rollback could not restore prior directory permissions."
            ) from mode_restore_error
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("Generated outputs could not be installed atomically.") from exc
    else:
        rollback_complete = True
    finally:
        if rollback_complete:
            if public_backup is not None:
                shutil.rmtree(public_backup, ignore_errors=True)
            if private_backup is not None:
                shutil.rmtree(private_backup, ignore_errors=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the public DocSem HF dataset and private validation labels."
    )
    parser.add_argument(
        "--reset-leaderboard",
        action="store_true",
        help="Reset the local generated leaderboard payload to an empty list.",
    )
    parser.add_argument(
        "--test-public-staging",
        type=Path,
        help=(
            "Explicit audited public staging root containing only test tasks, PDFs, "
            "checksums, and the sanitized release manifest."
        ),
    )
    return parser.parse_args(argv)


def generate_dataset(*, reset_leaderboard=False, test_public_staging=None):
    _require_real_dataset_root(TARGET_ROOT)
    captured_test = None
    captured_manifest = None
    snapshot_container = public_container = private_container = None
    try:
        if test_public_staging is not None:
            captured_test, captured_manifest = _capture_audited_public_test(
                test_public_staging
            )

        snapshot_container = Path(
            tempfile.mkdtemp(
                prefix=".docsem-hf-ordinary-snapshot-",
                dir=Path(TARGET_ROOT).parent,
            )
        )
        snapshot_container.chmod(0o700)

        # Capture every ordinary input, including exact PDF bytes, before any
        # generated output is mutated. PDF bytes remain in a private on-disk
        # snapshot and are never retained as one aggregate in memory.
        ordinary = _capture_ordinary_sources(snapshot_container / "ordinary")
        dataset_card = ordinary.base_dataset_card
        if captured_test is not None:
            dataset_card = _render_test_ready_dataset_card(
                ordinary.base_dataset_card,
                captured_test,
                captured_manifest,
            )

        public_container = Path(
            tempfile.mkdtemp(
                prefix=".docsem-hf-public-generation-",
                dir=Path(TARGET_ROOT).parent,
            )
        )
        public_container.chmod(0o700)
        staged_public = public_container / "dataset"
        _materialize_public_generation(
            staged_public,
            ordinary,
            dataset_card,
            captured_test,
        )

        private_container = Path(
            tempfile.mkdtemp(
                prefix=".docsem-hf-private-generation-",
                dir=Path(PRIVATE_ROOT).parent,
            )
        )
        private_container.chmod(0o700)
        staged_val_labels = private_container / "val_labels.jsonl"
        write_jsonl(staged_val_labels, ordinary.val_labels)
        staged_val_labels.chmod(0o600)
        staged_leaderboard = None
        if reset_leaderboard:
            staged_leaderboard = private_container / "leaderboard.json"
            _write_bytes(staged_leaderboard, b"[]\n", mode=0o600)

        _install_generated_outputs(
            staged_public,
            staged_val_labels,
            staged_leaderboard,
        )
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Dataset outputs could not be generated safely.") from exc
    finally:
        for temporary_root in (
            private_container,
            public_container,
            snapshot_container,
        ):
            if temporary_root is not None:
                shutil.rmtree(temporary_root, ignore_errors=True)

    summary = {
        "train_tasks": len(ordinary.train_tasks),
        "train_labels": len(ordinary.train_labels),
        "val_tasks_public": len(ordinary.val_tasks),
        "val_labels_private": len(ordinary.val_labels),
        "leaderboard_reset": reset_leaderboard,
    }
    if captured_manifest is not None:
        summary.update(
            {
                "test_tasks_public": captured_manifest["counts"]["tasks"],
                "test_release_id": captured_manifest["release_id"],
            }
        )
    return summary


def main():
    args = parse_args()
    summary = generate_dataset(
        reset_leaderboard=args.reset_leaderboard,
        test_public_staging=args.test_public_staging,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
