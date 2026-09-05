"""Validate one explicitly selected DocSem held-out test source.

This module deliberately validates only a caller-provided directory.  It does
not discover, select, stage, publish, or activate any local test material.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import tempfile
from typing import Iterable

try:
    import fitz
except ImportError:
    fitz = None


SCHEMA_VERSION = 1
TASK_KEYS = frozenset(("instance_id", "user_query", "document_pdf"))
LABEL_KEYS = frozenset(("instance_id", "answer", "evidence"))
BLOCK_ID = re.compile(r"b[0-9]+$")
OCR_HEADER = re.compile(r"^[ \t]*(b[0-9]+):")

PYMUPDF_VERSION = "1.26.3"
TESSERACT_VERSION = "5.5.1"
TESSERACT_BINARY = "tesseract"
OCR_METHOD = "pymupdf-raster-tesseract-cli"
RENDER_DPI = 300
OCR_LANGUAGE = "eng"
OCR_PAGE_SEGMENTATION_MODE = 6

MAX_PDF_BYTES = 16 * 1024 * 1024
MAX_PAGES = 16
MAX_PAGE_WIDTH_POINTS = 1000
MAX_PAGE_HEIGHT_POINTS = 1500
MAX_RENDER_PIXELS_PER_PAGE = 12_000_000
MAX_RENDER_PIXELS_TOTAL = 96_000_000
MAX_RASTER_BYTES = 32 * 1024 * 1024
MAX_OCR_OUTPUT_BYTES = 1024 * 1024
MAX_OCR_LINE_CHARS = 4096
MAX_SUBPROCESS_LOG_BYTES = 64 * 1024
OCR_TIMEOUT_SECONDS = 30
OCR_VERSION_TIMEOUT_SECONDS = 5


class ValidationError(ValueError):
    """Raised for a structural source error without exposing private rows."""


@dataclass(frozen=True)
class ValidatedTestSource:
    """Validated source values used by later explicit staging steps."""

    source_root: Path
    ids: tuple[str, ...]
    task_rows: tuple[dict, ...]
    label_rows: tuple[dict, ...]
    pdf_paths: tuple[Path, ...]


def _fail(message: str) -> None:
    raise ValidationError(message)


def _read_jsonl(path: Path, kind: str) -> list[dict]:
    if not path.is_file():
        _fail(f"Required {kind} file is absent.")
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                _fail(f"{kind.capitalize()} file contains a blank row.")
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                _fail(f"{kind.capitalize()} rows must be JSON objects.")
            rows.append(parsed)
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{kind.capitalize()} file is not UTF-8.") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{kind.capitalize()} file contains invalid JSON.") from exc
    if not rows:
        _fail(f"{kind.capitalize()} file is empty.")
    return rows


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and "/" not in value
        and "\\" not in value
        and "\n" not in value
        and "\r" not in value
        and "\x00" not in value
        and value not in {".", ".."}
    )


def _validate_tasks(rows: list[dict]) -> dict[str, dict]:
    tasks = {}
    for row in rows:
        if set(row) != TASK_KEYS:
            _fail("Task row schema is not exact.")
        instance_id = row["instance_id"]
        if not _valid_identifier(instance_id):
            _fail("Task instance ID is invalid.")
        if not isinstance(row["user_query"], str) or not row["user_query"].strip():
            _fail("Task query is invalid.")
        if row["document_pdf"] != f"documents/{instance_id}.pdf":
            _fail("Task document path does not match its instance ID.")
        if instance_id in tasks:
            _fail("Task instance IDs are not unique.")
        tasks[instance_id] = row
    return tasks


def _validate_labels(rows: list[dict]) -> dict[str, dict]:
    labels = {}
    for row in rows:
        if set(row) != LABEL_KEYS:
            _fail("Private label row schema is not exact.")
        instance_id = row["instance_id"]
        if not _valid_identifier(instance_id):
            _fail("Private label instance ID is invalid.")
        if not isinstance(row["answer"], str) or not row["answer"].strip():
            _fail("Private label answer is invalid.")
        evidence = row["evidence"]
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(block, str) or not BLOCK_ID.fullmatch(block) for block in evidence)
            or len(set(evidence)) != len(evidence)
        ):
            _fail("Private label evidence must be unique non-empty block IDs.")
        if instance_id in labels:
            _fail("Private label instance IDs are not unique.")
        labels[instance_id] = row
    return labels


def _visibility_audit_contract() -> dict:
    """Return the sanitized, exact visibility-audit contract."""
    return {
        "method": OCR_METHOD,
        "pymupdf_version": PYMUPDF_VERSION,
        "tesseract_version": TESSERACT_VERSION,
        "render_dpi": RENDER_DPI,
        "colorspace": "grayscale",
        "ocr_language": OCR_LANGUAGE,
        "page_segmentation_mode": OCR_PAGE_SEGMENTATION_MODE,
        "max_pdf_bytes": MAX_PDF_BYTES,
        "max_pages": MAX_PAGES,
        "max_page_width_points": MAX_PAGE_WIDTH_POINTS,
        "max_page_height_points": MAX_PAGE_HEIGHT_POINTS,
        "max_render_pixels_per_page": MAX_RENDER_PIXELS_PER_PAGE,
        "max_render_pixels_total": MAX_RENDER_PIXELS_TOTAL,
        "max_raster_bytes": MAX_RASTER_BYTES,
        "max_ocr_output_bytes": MAX_OCR_OUTPUT_BYTES,
        "ocr_timeout_seconds_per_page": OCR_TIMEOUT_SECONDS,
    }


def _subprocess_environment() -> dict[str, str]:
    """Expose no caller secrets to the OCR subprocess."""
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_THREAD_LIMIT": "1",
    }
    tessdata_prefix = os.environ.get("TESSDATA_PREFIX")
    if tessdata_prefix:
        environment["TESSDATA_PREFIX"] = tessdata_prefix
    return environment


def _limit_child_file_size(max_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))


def _run_bounded_process(
    arguments: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    max_file_bytes: int,
) -> int:
    """Run one OCR command with bounded files, time, and inherited state."""
    process = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                cwd=str(stdout_path.parent),
                env=_subprocess_environment(),
                close_fds=True,
                start_new_session=True,
                preexec_fn=lambda: _limit_child_file_size(max_file_bytes),
            )
            try:
                return process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise ValidationError("OCR process exceeded its time limit.") from exc
    except ValidationError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.wait()
        raise ValidationError("OCR process could not run safely.") from exc


def _read_bounded_regular_file(path: Path, max_bytes: int, description: str) -> bytes:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise ValidationError(f"{description} is absent or unreadable.") from exc
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_bytes:
        _fail(f"{description} is malformed or exceeds its size limit.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"{description} is absent or unreadable.") from exc


def _resolve_ocr_runtime() -> str:
    if fitz is None:
        _fail("PDF renderer is unavailable.")
    if getattr(fitz, "VersionBind", None) != PYMUPDF_VERSION:
        _fail("PDF renderer version does not match the tested contract.")
    binary = shutil.which(TESSERACT_BINARY)
    if binary is None:
        _fail("OCR backend is unavailable.")
    with tempfile.TemporaryDirectory(prefix="docsem-ocr-version-") as temporary_root:
        root = Path(temporary_root)
        stdout_path = root / "stdout"
        stderr_path = root / "stderr"
        return_code = _run_bounded_process(
            [binary, "--version"],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=OCR_VERSION_TIMEOUT_SECONDS,
            max_file_bytes=MAX_SUBPROCESS_LOG_BYTES,
        )
        if return_code != 0:
            _fail("OCR backend version check failed.")
        stdout = _read_bounded_regular_file(
            stdout_path,
            MAX_SUBPROCESS_LOG_BYTES,
            "OCR backend version output",
        )
        _read_bounded_regular_file(
            stderr_path,
            MAX_SUBPROCESS_LOG_BYTES,
            "OCR backend version diagnostics",
        )
    try:
        version_lines = stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ValidationError("OCR backend version output is malformed.") from exc
    if not version_lines or version_lines[0] != f"tesseract {TESSERACT_VERSION}":
        _fail("OCR backend version does not match the tested contract.")
    return binary


def _ocr_headers_from_raster(raster: bytes, binary: str) -> set[str]:
    if len(raster) > MAX_RASTER_BYTES:
        _fail("Rendered page image exceeds its size limit.")
    with tempfile.TemporaryDirectory(prefix="docsem-ocr-page-") as temporary_root:
        root = Path(temporary_root)
        raster_path = root / "page.png"
        output_base = root / "ocr"
        stdout_path = root / "stdout"
        stderr_path = root / "stderr"
        raster_path.write_bytes(raster)
        raster_path.chmod(0o600)
        return_code = _run_bounded_process(
            [
                binary,
                str(raster_path),
                str(output_base),
                "--dpi",
                str(RENDER_DPI),
                "--psm",
                str(OCR_PAGE_SEGMENTATION_MODE),
                "-l",
                OCR_LANGUAGE,
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=OCR_TIMEOUT_SECONDS,
            max_file_bytes=max(MAX_OCR_OUTPUT_BYTES, MAX_SUBPROCESS_LOG_BYTES),
        )
        if return_code != 0:
            _fail("OCR backend rejected a rendered page.")
        payload = _read_bounded_regular_file(
            output_base.with_suffix(".txt"),
            MAX_OCR_OUTPUT_BYTES,
            "OCR output",
        )
        _read_bounded_regular_file(stdout_path, MAX_SUBPROCESS_LOG_BYTES, "OCR standard output")
        _read_bounded_regular_file(stderr_path, MAX_SUBPROCESS_LOG_BYTES, "OCR diagnostics")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValidationError("OCR output is malformed.") from exc
    if any(ord(character) < 32 and character not in "\t\n\r\f" for character in text):
        _fail("OCR output is malformed.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "")
    lines = normalized.split("\n")
    if any(len(line) > MAX_OCR_LINE_CHARS for line in lines):
        _fail("OCR output contains an oversized line.")
    return {
        match.group(1)
        for line in lines
        if (match := OCR_HEADER.match(line)) is not None
    }


def _render_visible_pdf_evidence_blocks(path: Path, binary: str) -> set[str]:
    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_size > MAX_PDF_BYTES:
            _fail("PDF is not a bounded regular file.")
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                _fail("PDF is unreadable.")
        matched_blocks = set()
        with fitz.open(str(path)) as document:
            if document.needs_pass:
                _fail("PDF is encrypted and cannot be inspected.")
            if type(document.page_count) is not int or not 0 < document.page_count <= MAX_PAGES:
                _fail("PDF page count exceeds its limit.")
            total_pixels = 0
            for page in document:
                width_points = float(page.rect.width)
                height_points = float(page.rect.height)
                if (
                    not math.isfinite(width_points)
                    or not math.isfinite(height_points)
                    or width_points <= 0
                    or height_points <= 0
                    or width_points > MAX_PAGE_WIDTH_POINTS
                    or height_points > MAX_PAGE_HEIGHT_POINTS
                ):
                    _fail("PDF page dimensions exceed their limits.")
                expected_width = math.ceil(width_points * RENDER_DPI / 72)
                expected_height = math.ceil(height_points * RENDER_DPI / 72)
                expected_pixels = expected_width * expected_height
                total_pixels += expected_pixels
                if (
                    expected_pixels > MAX_RENDER_PIXELS_PER_PAGE
                    or total_pixels > MAX_RENDER_PIXELS_TOTAL
                ):
                    _fail("PDF raster allocation exceeds its limit.")
                pixmap = page.get_pixmap(
                    dpi=RENDER_DPI,
                    colorspace=fitz.csGRAY,
                    alpha=False,
                )
                if (
                    type(pixmap.width) is not int
                    or type(pixmap.height) is not int
                    or pixmap.width <= 0
                    or pixmap.height <= 0
                    or pixmap.width * pixmap.height > MAX_RENDER_PIXELS_PER_PAGE
                ):
                    _fail("PDF renderer produced an invalid page image.")
                matched_blocks.update(_ocr_headers_from_raster(pixmap.tobytes("png"), binary))
        return matched_blocks
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("PDF is unreadable.") from exc


def _validate_pdfs(source_root: Path, tasks: dict[str, dict], labels: dict[str, dict]) -> tuple[Path, ...]:
    ocr_binary = _resolve_ocr_runtime()
    documents_root = source_root / "documents"
    if not documents_root.is_dir():
        _fail("Documents directory is absent.")
    pdf_paths = tuple(sorted(documents_root.glob("*.pdf"), key=lambda path: path.name))
    if len(pdf_paths) != len(list(documents_root.iterdir())):
        _fail("Documents directory contains a non-PDF entry.")
    pdf_ids = {path.stem for path in pdf_paths}
    if set(tasks) != pdf_ids:
        _fail("Task IDs and PDF stems are not a bijection.")
    paths_by_id = {path.stem: path for path in pdf_paths}
    for instance_id in sorted(tasks):
        required_blocks = set(labels[instance_id]["evidence"])
        if not required_blocks.issubset(
            _render_visible_pdf_evidence_blocks(paths_by_id[instance_id], ocr_binary)
        ):
            _fail("PDF does not visibly contain every evidence block ID.")
    return pdf_paths


def validate_source(
    source_root: str | Path,
    train_ids: Iterable[str],
    val_ids: Iterable[str],
) -> ValidatedTestSource:
    """Fail closed unless an explicit source directory meets the release contract."""
    root = Path(source_root)
    if root.suffix.lower() == ".zip" or not root.is_dir():
        _fail("Source must be an explicitly selected directory, not an archive.")

    tasks = _validate_tasks(_read_jsonl(root / "tasks.jsonl", "task"))
    labels = _validate_labels(_read_jsonl(root / "labels.jsonl", "private label"))
    if set(tasks) != set(labels):
        _fail("Task and private label IDs do not match exactly.")

    existing_ids = set(train_ids) | set(val_ids)
    if set(tasks) & existing_ids:
        _fail("Held-out IDs overlap train or validation IDs.")

    pdf_paths = _validate_pdfs(root, tasks, labels)
    ids = tuple(sorted(tasks))
    return ValidatedTestSource(
        source_root=root.resolve(),
        ids=ids,
        task_rows=tuple(tasks[instance_id] for instance_id in ids),
        label_rows=tuple(labels[instance_id] for instance_id in ids),
        pdf_paths=pdf_paths,
    )


def _canonical_rows(rows: Iterable[dict]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pdf_inventory_digest(paths: Iterable[Path]) -> str:
    entries = []
    for path in paths:
        entries.append(f"{path.name}  {_sha256(path.read_bytes())}\n".encode("ascii"))
    return _sha256(b"".join(entries))


def build_release_manifest(validated: ValidatedTestSource, release_id: str) -> dict:
    """Return a deterministic, sanitized manifest with private values only hashed."""
    if not isinstance(release_id, str) or not release_id.strip() or release_id != release_id.strip():
        raise ValidationError("Release ID is invalid.")
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "counts": {
            "tasks": len(validated.task_rows),
            "pdfs": len(validated.pdf_paths),
            "labels": len(validated.label_rows),
        },
        "sorted_ids_sha256": _sha256("".join(f"{instance_id}\n" for instance_id in validated.ids).encode("utf-8")),
        "tasks_sha256": _sha256(_canonical_rows(validated.task_rows)),
        "pdf_inventory_sha256": _pdf_inventory_digest(validated.pdf_paths),
        "private_labels_sha256": _sha256(_canonical_rows(validated.label_rows)),
        "visibility_audit": _visibility_audit_contract(),
    }
