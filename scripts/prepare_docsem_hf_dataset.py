#!/usr/bin/env python3
import argparse
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


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def copy_pdfs(split):
    source_dir = PUBLIC_SOURCE_ROOT / split / "documents"
    target_dir = TARGET_ROOT / split / "documents"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(source_dir.glob("*.pdf")):
        shutil.copy2(pdf, target_dir / pdf.name)


def public_tasks(split):
    rows = []
    for row in read_jsonl(PUBLIC_SOURCE_ROOT / split / "tasks.jsonl"):
        rows.append(
            {
                "instance_id": row["instance_id"],
                "user_query": row["user_query"],
                "document_pdf": f"{split}/{row['document_pdf']}",
            }
        )
    return rows


def labels(source_root, split):
    return [
        {
            "instance_id": row["instance_id"],
            "answer": str(row["answer"]),
            "evidence": [str(value) for value in row["evidence"]],
        }
        for row in read_jsonl(source_root / split / "labels.jsonl")
    ]


def validate_public_split(split, label_rows=None):
    tasks = public_tasks(split)
    task_ids = {row["instance_id"] for row in tasks}
    pdf_ids = {
        path.stem
        for path in (PUBLIC_SOURCE_ROOT / split / "documents").glob("*.pdf")
    }
    if task_ids != pdf_ids:
        raise ValueError(f"{split} task ids and PDF ids differ")
    if label_rows is not None:
        label_ids = {row["instance_id"] for row in label_rows}
        if task_ids != label_ids:
            raise ValueError(f"{split} task ids and label ids differ")
    return tasks


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
        return payload, audited_manifest
    finally:
        for descriptor in (documents_fd, test_fd, root_fd):
            if descriptor is not None:
                os.close(descriptor)


def _write_public_test_snapshot(root, payload):
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
    target = Path(target_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target_mode = target.lstat().st_mode
        if not stat.S_ISDIR(target_mode) or stat.S_ISLNK(target_mode):
            raise ValidationError("Dataset output root is not a real directory.")
    else:
        target.mkdir()

    temporary_root = Path(
        tempfile.mkdtemp(prefix=".docsem-hf-public-test-", dir=target.parent)
    )
    try:
        temporary_root.chmod(0o755)
        _write_public_test_snapshot(temporary_root, payload)
        audit_public_payload(temporary_root)
        _remove_generated_test_output(target)
        os.replace(temporary_root / "test", target / "test")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Audited public test payload could not be installed.") from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


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
    if test_public_staging is not None:
        captured_test, captured_manifest = _capture_audited_public_test(
            test_public_staging
        )
    else:
        _remove_generated_test_output(TARGET_ROOT)

    train_labels = labels(PUBLIC_SOURCE_ROOT, "train")
    train_tasks = validate_public_split("train", train_labels)
    val_labels = labels(ORGANIZER_SOURCE_ROOT, "val")
    val_tasks = validate_public_split("val", val_labels)

    write_jsonl(TARGET_ROOT / "train" / "tasks.jsonl", train_tasks)
    write_jsonl(TARGET_ROOT / "train" / "labels.jsonl", train_labels)
    write_jsonl(TARGET_ROOT / "val" / "tasks.jsonl", val_tasks)

    copy_pdfs("train")
    copy_pdfs("val")

    sample_submission = [
        {
            "instance_id": row["instance_id"],
            "answer": "0",
            "evidence": ["b01"],
        }
        for row in val_tasks[:5]
    ]
    write_jsonl(TARGET_ROOT / "examples" / "sample_val_submission.jsonl", sample_submission)

    shutil.copy2(
        PUBLIC_SOURCE_ROOT / "PARTICIPANT_INSTRUCTIONS.md",
        TARGET_ROOT / "INSTRUCTIONS.md",
    )
    shutil.copy2(PUBLIC_LICENSE, TARGET_ROOT / "LICENSE.txt")

    write_jsonl(PRIVATE_ROOT / "private" / "val_labels.jsonl", val_labels)
    (PRIVATE_ROOT / "submissions").mkdir(parents=True, exist_ok=True)
    (PRIVATE_ROOT / "submissions" / ".gitkeep").write_text("", encoding="utf-8")
    if reset_leaderboard:
        (PRIVATE_ROOT / "leaderboard").mkdir(parents=True, exist_ok=True)
        (PRIVATE_ROOT / "leaderboard" / "leaderboard.json").write_text(
            "[]\n", encoding="utf-8"
        )

    if captured_test is not None:
        _install_public_test_snapshot(TARGET_ROOT, captured_test)

    summary = {
        "train_tasks": len(train_tasks),
        "train_labels": len(train_labels),
        "val_tasks_public": len(val_tasks),
        "val_labels_private": len(val_labels),
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
