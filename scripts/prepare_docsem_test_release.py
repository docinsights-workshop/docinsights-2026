"""Validate one explicitly selected DocSem held-out test source.

This module deliberately validates only a caller-provided directory.  It does
not discover, select, stage, publish, or activate any local test material.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import select
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import time
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
RELEASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

PYMUPDF_VERSION = "1.26.3"
TESSERACT_VERSION = "5.5.1"
TESSERACT_BINARY = "/opt/homebrew/Cellar/tesseract/5.5.1/bin/tesseract"
TESSERACT_BINARY_SHA256 = (
    "6517c9cf1b17280201af3e48880517bbfafd24b5876aacb75d5643bafff1c414"
)
TESSDATA_ROOT: str | None = None
TESSDATA_ROOT_ENV = "DOCSEM_TESSDATA_ROOT"
TESSERACT_TRAINEDDATA_SHA256 = (
    "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
)
OCR_METHOD = "pymupdf-raster-tesseract-cli"
RENDER_DPI = 300
OCR_LANGUAGE = "eng"
OCR_PAGE_SEGMENTATION_MODE = 6

MAX_TESSERACT_BINARY_BYTES = 16 * 1024 * 1024
MAX_TRAINEDDATA_BYTES = 8 * 1024 * 1024
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
PAGE_WORKFLOW_TIMEOUT_SECONDS = 45
PAGE_WORKER_CPU_SECONDS = 40
PAGE_WORKER_OPEN_FILES = 64
MAX_EVIDENCE_IDS_PER_TASK = 1024
MAX_PUBLIC_TASKS_BYTES = 16 * 1024 * 1024
MAX_PUBLIC_MANIFEST_BYTES = 1024 * 1024
MAX_PUBLIC_CHECKSUM_BYTES = 4 * 1024 * 1024
MAX_SOURCE_TASKS_BYTES = 16 * 1024 * 1024
MAX_PRIVATE_LABELS_BYTES = 16 * 1024 * 1024
MAX_PDF_XREF_OBJECTS = 100_000
MAX_PDF_KEYS_PER_OBJECT = 256
MAX_PDF_STRUCTURE_VALUE_CHARS = 1024 * 1024
MAX_PDF_STRUCTURE_TOTAL_CHARS = 8 * 1024 * 1024

PUBLIC_MANIFEST_RELATIVE_PATH = "test/release.json"
PUBLIC_TASKS_RELATIVE_PATH = "test/tasks.jsonl"
PUBLIC_CHECKSUMS_RELATIVE_PATH = "test/SHA256SUMS"
PUBLIC_DOCUMENTS_RELATIVE_PATH = "test/documents"

_PUBLIC_MANIFEST_KEYS = frozenset(
    (
        "schema_version",
        "release_id",
        "counts",
        "sorted_ids_sha256",
        "task_manifest_sha256",
        "pdf_inventory_sha256",
    )
)
_PUBLIC_FORBIDDEN_FIELD = re.compile(
    r"(^|[_-])(answers?|evidence|labels?|gold|private|source[_-]?mappings?|"
    r"source[_-]?(file|path|document)|organizer|notes?|correctness|per[_-]?example)($|[_-])",
    re.IGNORECASE,
)
_EMBEDDED_FORBIDDEN_FIELD = re.compile(
    rb'''["']\s*(?:answers?|evidence|labels?|gold|private|source[_-]?mappings?|'''
    rb'''source[_-]?(?:file|path|document)|organizer|notes?|correctness|per[_-]?example)'''
    rb'''\s*["']\s*:''',
    re.IGNORECASE,
)
_ARCHIVE_MAGICS = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"7z\xbc\xaf'\x1c",
    b"Rar!\x1a\x07",
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PDF_NAME_TOKEN = re.compile(r"/([A-Za-z0-9_.+#-]+)")
_PDF_FORBIDDEN_KEYS = frozenset(
    {
        "AA",
        "AcroForm",
        "AF",
        "Collection",
        "EmbeddedFiles",
        "ImportData",
        "JavaScript",
        "JS",
        "Launch",
        "OpenAction",
        "RichMedia",
        "RichMediaContent",
        "SubmitForm",
        "XFA",
    }
)
_PDF_FORBIDDEN_TYPE_NAMES = frozenset(
    {
        "EmbeddedFile",
        "FileAttachment",
        "Filespec",
        "GoToE",
        "GoToR",
        "Movie",
        "Rendition",
        "ResetForm",
        "RichMediaExecute",
        "Screen",
        "Sound",
        "Thread",
        "Trans",
        "UseAttachments",
        "Widget",
    }
)
_PDF_FORBIDDEN_NAME_TREE_KEYS = frozenset({"EmbeddedFiles", "JavaScript"})

_PAGE_RESULT_HEADER = struct.Struct(">Q")
_DOCUMENT_PROBE_RESULT = struct.Struct(">I")


class ValidationError(ValueError):
    """Raised for a structural source error without exposing private rows."""


@dataclass(frozen=True)
class ValidatedPDF:
    """One bounded source PDF sealed by its validation-time digest."""

    source_path: Path
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ValidatedTestSource:
    """Validated source values used by later explicit staging steps."""

    source_root: Path
    ids: tuple[str, ...]
    task_rows: tuple[dict, ...] = field(repr=False)
    label_rows: tuple[dict, ...] = field(repr=False)
    pdfs: tuple[ValidatedPDF, ...]
    canonical_task_bytes: bytes = field(repr=False)
    canonical_label_bytes: bytes = field(repr=False)
    tasks_sha256: str
    private_labels_sha256: str
    pdf_inventory_sha256: str

    @property
    def pdf_paths(self) -> tuple[Path, ...]:
        """Preserve the read-only path view used by callers and tests."""
        return tuple(pdf.source_path for pdf in self.pdfs)


@dataclass(frozen=True)
class OCRRuntime:
    """Pinned OCR executable and validation-owned English model root."""

    binary: str
    tessdata_root: Path
    traineddata_sha256: str


def _fail(message: str) -> None:
    raise ValidationError(message)


def _read_jsonl(path: Path, kind: str) -> list[dict]:
    rows = []
    try:
        max_bytes = (
            MAX_PRIVATE_LABELS_BYTES if kind == "private label" else MAX_SOURCE_TASKS_BYTES
        )
        payload = _read_bounded_regular_file(
            path,
            max_bytes,
            f"Required {kind} file",
        )
        for line in payload.decode("utf-8", errors="strict").splitlines():
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
            or len(evidence) > MAX_EVIDENCE_IDS_PER_TASK
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
        "tesseract_binary_sha256": TESSERACT_BINARY_SHA256,
        "traineddata_sha256": TESSERACT_TRAINEDDATA_SHA256,
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
        "page_workflow_timeout_seconds": PAGE_WORKFLOW_TIMEOUT_SECONDS,
        "page_worker_cpu_seconds": PAGE_WORKER_CPU_SECONDS,
        "page_worker_open_files": PAGE_WORKER_OPEN_FILES,
        "max_tesseract_binary_bytes": MAX_TESSERACT_BINARY_BYTES,
        "max_traineddata_bytes": MAX_TRAINEDDATA_BYTES,
        "max_evidence_ids_per_task": MAX_EVIDENCE_IDS_PER_TASK,
    }


def _subprocess_environment(tessdata_root: Path | None = None) -> dict[str, str]:
    """Expose no caller secrets to the OCR subprocess."""
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_THREAD_LIMIT": "1",
    }
    if tessdata_root is not None:
        environment["TESSDATA_PREFIX"] = str(tessdata_root)
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
    tessdata_root: Path | None = None,
    isolated_process_group: bool = True,
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
                env=_subprocess_environment(tessdata_root),
                close_fds=True,
                start_new_session=isolated_process_group,
                preexec_fn=lambda: _limit_child_file_size(max_file_bytes),
            )
            try:
                return process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                if isolated_process_group:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    os.kill(process.pid, signal.SIGKILL)
                process.wait()
                raise ValidationError("OCR process exceeded its time limit.") from exc
    except ValidationError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None and process.poll() is None:
            try:
                if isolated_process_group:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    os.kill(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.wait()
        raise ValidationError("OCR process could not run safely.") from exc


def _read_bounded_regular_file(path: Path, max_bytes: int, description: str) -> bytes:
    """Read one stable, bounded regular file without following its final link."""
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > max_bytes:
            _fail(f"{description} is malformed or exceeds its size limit.")
        remaining = initial.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"{description} changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{description} changed while being read.")
        final = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, name) != getattr(final, name) for name in stable_fields):
            _fail(f"{description} changed while being read.")
        return b"".join(chunks)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"{description} is absent or unreadable.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _traineddata_source_path(binary: str) -> Path:
    """Select one explicit or conventional English traineddata path."""
    if TESSDATA_ROOT is not None:
        return Path(TESSDATA_ROOT) / f"{OCR_LANGUAGE}.traineddata"
    configured_root = os.environ.get(TESSDATA_ROOT_ENV)
    if configured_root:
        return Path(configured_root) / f"{OCR_LANGUAGE}.traineddata"

    resolved_binary = Path(binary).resolve()
    candidates = [
        Path("/opt/homebrew/share/tessdata"),
        resolved_binary.parent.parent / "share" / "tessdata",
        Path("/usr/local/share/tessdata"),
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/share/tessdata"),
    ]
    seen = set()
    for root in candidates:
        candidate = root / f"{OCR_LANGUAGE}.traineddata"
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            candidate.lstat()
        except OSError:
            continue
        return candidate
    _fail("Pinned OCR language data is unavailable.")


def _read_pinned_traineddata(path: Path) -> bytes:
    """Read one bounded regular model without following its final symlink."""
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size <= 0
            or file_stat.st_size > MAX_TRAINEDDATA_BYTES
        ):
            _fail("Pinned OCR language data is malformed or exceeds its size limit.")
        remaining = file_stat.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail("Pinned OCR language data changed while being verified.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail("Pinned OCR language data changed while being verified.")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Pinned OCR language data is unavailable or unsafe.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != TESSERACT_TRAINEDDATA_SHA256:
        _fail("Pinned OCR language data digest does not match the tested contract.")
    return payload


def _read_pinned_tesseract_binary(path: Path) -> bytes:
    """Read the exact executable without following links or trusting PATH."""
    if not path.is_absolute():
        _fail("Pinned OCR executable path is not absolute.")
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size <= 0
            or file_stat.st_size > MAX_TESSERACT_BINARY_BYTES
            or not file_stat.st_mode & 0o111
        ):
            _fail("Pinned OCR executable is malformed or exceeds its size limit.")
        remaining = file_stat.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail("Pinned OCR executable changed while being verified.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail("Pinned OCR executable changed while being verified.")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Pinned OCR executable is unavailable or unsafe.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != TESSERACT_BINARY_SHA256:
        _fail("Pinned OCR executable digest does not match the tested contract.")
    return payload


@contextmanager
def _resolve_ocr_runtime():
    _require_pdf_renderer()
    source_binary = Path(TESSERACT_BINARY)
    binary_payload = _read_pinned_tesseract_binary(source_binary)
    traineddata = _read_pinned_traineddata(_traineddata_source_path(str(source_binary)))
    with tempfile.TemporaryDirectory(prefix="docsem-pinned-ocr-") as temporary_root:
        root = Path(temporary_root)
        root.chmod(0o700)
        controlled_binary = root / "tesseract"
        with controlled_binary.open("xb") as output:
            output.write(binary_payload)
        controlled_binary.chmod(0o500)
        controlled_tessdata = root / "tessdata"
        controlled_tessdata.mkdir(mode=0o700)
        controlled_model = controlled_tessdata / f"{OCR_LANGUAGE}.traineddata"
        with controlled_model.open("xb") as output:
            output.write(traineddata)
        controlled_model.chmod(0o600)
        version_root = root / "version"
        version_root.mkdir(mode=0o700)
        stdout_path = version_root / "stdout"
        stderr_path = version_root / "stderr"
        return_code = _run_bounded_process(
            [str(controlled_binary), "--version"],
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
        yield OCRRuntime(
            binary=str(controlled_binary),
            tessdata_root=controlled_tessdata,
            traineddata_sha256=TESSERACT_TRAINEDDATA_SHA256,
        )


def _require_pdf_renderer() -> None:
    """Require the exact public PDF API used by bounded structural workers."""
    if fitz is None:
        _fail("PDF renderer is unavailable.")
    if getattr(fitz, "VersionBind", None) != PYMUPDF_VERSION:
        _fail("PDF renderer version does not match the tested contract.")
    if not hasattr(os, "fork") or not hasattr(os, "setsid"):
        _fail("Bounded PDF page workers are unavailable.")


def _ocr_headers_from_raster(
    raster: bytes,
    runtime: OCRRuntime,
    temporary_root: Path,
) -> set[str]:
    if len(raster) > MAX_RASTER_BYTES:
        _fail("Rendered page image exceeds its size limit.")
    raster_path = temporary_root / "page.png"
    output_base = temporary_root / "ocr"
    stdout_path = temporary_root / "stdout"
    stderr_path = temporary_root / "stderr"
    with raster_path.open("xb") as output:
        output.write(raster)
    raster_path.chmod(0o600)
    return_code = _run_bounded_process(
        [
            runtime.binary,
            str(raster_path),
            str(output_base),
            "--tessdata-dir",
            str(runtime.tessdata_root),
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
        tessdata_root=runtime.tessdata_root,
        isolated_process_group=False,
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


def _set_page_worker_limit(limit: int, value: int) -> None:
    _, hard = resource.getrlimit(limit)
    bounded = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(limit, (bounded, bounded))


def _apply_page_worker_limits() -> None:
    """Constrain renderer and nested OCR work before touching the PDF."""
    _set_page_worker_limit(resource.RLIMIT_CPU, PAGE_WORKER_CPU_SECONDS)
    _set_page_worker_limit(resource.RLIMIT_FSIZE, MAX_RASTER_BYTES)
    _set_page_worker_limit(resource.RLIMIT_NOFILE, PAGE_WORKER_OPEN_FILES)
    _set_page_worker_limit(resource.RLIMIT_CORE, 0)


def _silence_page_worker(result_fd: int) -> None:
    """Keep renderer/OCR diagnostics private and close unrelated descriptors."""
    devnull = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    finally:
        os.close(devnull)
    max_fd = min(int(os.sysconf("SC_OPEN_MAX")), 4096)
    os.closerange(3, result_fd)
    os.closerange(result_fd + 1, max_fd)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("page worker result pipe closed")
        offset += written


def _decode_pdf_name(value: str) -> str:
    """Decode PDF ``#xx`` name escapes used to disguise structural keys."""
    output = bytearray()
    index = 0
    encoded = value.encode("ascii", errors="strict")
    while index < len(encoded):
        if index + 2 < len(encoded) and encoded[index] == ord("#"):
            try:
                output.append(int(encoded[index + 1 : index + 3], 16))
                index += 3
                continue
            except ValueError:
                pass
        output.append(encoded[index])
        index += 1
    return output.decode("ascii", errors="strict")


def _pdf_value_names(value: str) -> set[str]:
    return {_decode_pdf_name(match.group(1)) for match in _PDF_NAME_TOKEN.finditer(value)}


def _audit_pdf_structure(document) -> None:
    """Reject attachments and active content using bounded public PyMuPDF APIs."""
    embedded_names = document.embfile_names()
    if not isinstance(embedded_names, list) or embedded_names:
        _fail("PDF contains an embedded file or malformed attachment inventory.")

    xref_count = document.xref_length()
    if type(xref_count) is not int or not 0 < xref_count <= MAX_PDF_XREF_OBJECTS:
        _fail("PDF object inventory exceeds its limit.")
    total_value_chars = 0
    for xref in range(1, xref_count):
        keys = document.xref_get_keys(xref)
        if not isinstance(keys, (list, tuple)) or len(keys) > MAX_PDF_KEYS_PER_OBJECT:
            _fail("PDF object dictionary is malformed or exceeds its limit.")
        for encoded_key in keys:
            if not isinstance(encoded_key, str):
                _fail("PDF object dictionary contains an invalid key.")
            key = _decode_pdf_name(encoded_key)
            if key in _PDF_FORBIDDEN_KEYS:
                _fail("PDF contains an attachment or active-content structure.")
            value_type, value = document.xref_get_key(xref, encoded_key)
            if not isinstance(value_type, str) or not isinstance(value, str):
                _fail("PDF object dictionary contains an invalid value.")
            if len(value) > MAX_PDF_STRUCTURE_VALUE_CHARS:
                _fail("PDF object dictionary value exceeds its limit.")
            total_value_chars += len(value)
            if total_value_chars > MAX_PDF_STRUCTURE_TOTAL_CHARS:
                _fail("PDF structural metadata exceeds its limit.")
            value_names = _pdf_value_names(value) if value_type in {"array", "dict", "name"} else set()
            if (
                key in {"PageMode", "S", "Subtype", "Type"}
                and value_names & _PDF_FORBIDDEN_TYPE_NAMES
            ) or (
                key == "Names"
                and value_names & _PDF_FORBIDDEN_NAME_TREE_KEYS
            ):
                _fail("PDF contains an attachment or active-content structure.")


def _probe_pdf_document(path: Path) -> int:
    """Open a PDF and return its bounded page count inside a worker."""
    with fitz.open(str(path)) as document:
        if document.needs_pass:
            _fail("PDF is encrypted and cannot be inspected.")
        _audit_pdf_structure(document)
        page_count = document.page_count
        if type(page_count) is not int or not 0 < page_count <= MAX_PAGES:
            _fail("PDF page count exceeds its limit.")
        return page_count


def _render_and_ocr_page(
    path: Path,
    page_number: int,
    runtime: OCRRuntime,
    required_blocks: tuple[str, ...],
    temporary_root: Path,
) -> tuple[int, bytes]:
    """Run the complete bounded page workflow inside its worker."""
    with fitz.open(str(path)) as document:
        if document.needs_pass:
            _fail("PDF is encrypted and cannot be inspected.")
        if type(document.page_count) is not int or not 0 < document.page_count <= MAX_PAGES:
            _fail("PDF page count exceeds its limit.")
        if page_number < 0 or page_number >= document.page_count:
            _fail("PDF page selection is invalid.")
        page = document.load_page(page_number)
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
        if expected_pixels > MAX_RENDER_PIXELS_PER_PAGE:
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
        raster = pixmap.tobytes("png")
    headers = _ocr_headers_from_raster(raster, runtime, temporary_root)
    return expected_pixels, bytes(block in headers for block in required_blocks)


def _terminate_page_worker(pid: int, *, child_already_reaped: bool = False) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        if not child_already_reaped:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if not child_already_reaped:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


def _wait_for_page_worker(
    pid: int,
    result_fd: int,
    max_result_bytes: int,
) -> tuple[bytes, int]:
    """Collect one fixed-size result without permitting an unbounded wait."""
    os.set_blocking(result_fd, False)
    deadline = time.monotonic() + PAGE_WORKFLOW_TIMEOUT_SECONDS
    payload = bytearray()
    status = None
    eof = False
    while status is None or not eof:
        while not eof and len(payload) <= max_result_bytes:
            try:
                chunk = os.read(result_fd, max_result_bytes + 1 - len(payload))
            except BlockingIOError:
                break
            if not chunk:
                eof = True
                break
            payload.extend(chunk)
        if len(payload) > max_result_bytes:
            _terminate_page_worker(pid)
            _fail("PDF page worker returned an oversized result.")
        if status is None:
            try:
                waited_pid, child_status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                _fail("PDF page worker state is invalid.")
            if waited_pid == pid:
                status = child_status
        if status is not None and eof:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_page_worker(pid)
            _fail("PDF page workflow exceeded its time limit.")
        select.select(
            [] if eof else [result_fd],
            [],
            [],
            min(remaining, 0.01),
        )
    return bytes(payload), status


def _run_bounded_page_workflow(
    path: Path,
    page_number: int,
    runtime: OCRRuntime,
    required_blocks: tuple[str, ...],
) -> tuple[int, bytes]:
    expected_result_bytes = _PAGE_RESULT_HEADER.size + len(required_blocks)
    with tempfile.TemporaryDirectory(prefix="docsem-ocr-page-") as temporary_root:
        root = Path(temporary_root)
        root.chmod(0o700)
        result_read_fd, result_write_fd = os.pipe()
        try:
            pid = os.fork()
        except OSError as exc:
            os.close(result_read_fd)
            os.close(result_write_fd)
            raise ValidationError("PDF page worker could not start safely.") from exc
        if pid == 0:
            os.close(result_read_fd)
            try:
                os.setsid()
                _silence_page_worker(result_write_fd)
                _apply_page_worker_limits()
                pixel_count, flags = _render_and_ocr_page(
                    path,
                    page_number,
                    runtime,
                    required_blocks,
                    root,
                )
                result = _PAGE_RESULT_HEADER.pack(pixel_count) + flags
                if len(result) != expected_result_bytes:
                    os._exit(71)
                _write_all(result_write_fd, result)
                os.close(result_write_fd)
                os._exit(0)
            except BaseException:
                try:
                    os.close(result_write_fd)
                except OSError:
                    pass
                os._exit(70)

        os.close(result_write_fd)
        try:
            try:
                payload, status = _wait_for_page_worker(
                    pid,
                    result_read_fd,
                    expected_result_bytes,
                )
            except ValidationError:
                raise
            except BaseException as exc:
                _terminate_page_worker(pid)
                raise ValidationError("PDF page worker could not be supervised safely.") from exc
        finally:
            os.close(result_read_fd)
        if (
            not os.WIFEXITED(status)
            or os.WEXITSTATUS(status) != 0
            or len(payload) != expected_result_bytes
        ):
            _terminate_page_worker(pid, child_already_reaped=True)
            _fail("PDF page workflow failed safely.")
        pixel_count = _PAGE_RESULT_HEADER.unpack(payload[: _PAGE_RESULT_HEADER.size])[0]
        return pixel_count, payload[_PAGE_RESULT_HEADER.size :]


def _run_bounded_document_probe(path: Path) -> int:
    """Supervise PDF open and page-count inspection with a wall deadline."""
    expected_result_bytes = _DOCUMENT_PROBE_RESULT.size
    result_read_fd, result_write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError as exc:
        os.close(result_read_fd)
        os.close(result_write_fd)
        raise ValidationError("PDF document probe could not start safely.") from exc
    if pid == 0:
        os.close(result_read_fd)
        try:
            os.setsid()
            _silence_page_worker(result_write_fd)
            _apply_page_worker_limits()
            result = _DOCUMENT_PROBE_RESULT.pack(_probe_pdf_document(path))
            _write_all(result_write_fd, result)
            os.close(result_write_fd)
            os._exit(0)
        except BaseException:
            try:
                os.close(result_write_fd)
            except OSError:
                pass
            os._exit(70)

    os.close(result_write_fd)
    try:
        try:
            payload, status = _wait_for_page_worker(
                pid,
                result_read_fd,
                expected_result_bytes,
            )
        except ValidationError:
            raise
        except BaseException as exc:
            _terminate_page_worker(pid)
            raise ValidationError("PDF document probe could not be supervised safely.") from exc
    finally:
        os.close(result_read_fd)
    if (
        not os.WIFEXITED(status)
        or os.WEXITSTATUS(status) != 0
        or len(payload) != expected_result_bytes
    ):
        _terminate_page_worker(pid, child_already_reaped=True)
        _fail("PDF document probe failed safely.")
    page_count = _DOCUMENT_PROBE_RESULT.unpack(payload)[0]
    if not 0 < page_count <= MAX_PAGES:
        _fail("PDF document probe returned an invalid result.")
    return page_count


def _render_visible_pdf_evidence_blocks(
    path: Path,
    runtime: OCRRuntime,
    required_blocks: set[str],
) -> set[str]:
    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_size > MAX_PDF_BYTES:
            _fail("PDF is not a bounded regular file.")
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                _fail("PDF is unreadable.")
        ordered_blocks = tuple(sorted(required_blocks))
        matched = bytearray(len(ordered_blocks))
        page_count = _run_bounded_document_probe(path)
        total_pixels = 0
        for page_number in range(page_count):
            expected_pixels, flags = _run_bounded_page_workflow(
                path,
                page_number,
                runtime,
                ordered_blocks,
            )
            if expected_pixels > MAX_RENDER_PIXELS_PER_PAGE or len(flags) != len(ordered_blocks):
                _fail("PDF page worker returned an invalid result.")
            total_pixels += expected_pixels
            if total_pixels > MAX_RENDER_PIXELS_TOTAL:
                _fail("PDF raster allocation exceeds its limit.")
            for index, present in enumerate(flags):
                if present not in (0, 1):
                    _fail("PDF page worker returned an invalid result.")
                matched[index] = matched[index] or present
        return {
            block
            for index, block in enumerate(ordered_blocks)
            if matched[index]
        }
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("PDF is unreadable.") from exc


def _validate_pdfs(
    source_root: Path,
    tasks: dict[str, dict],
    labels: dict[str, dict],
) -> tuple[ValidatedPDF, ...]:
    documents_root = source_root / "documents"
    try:
        directory_stat = documents_root.lstat()
        entries = tuple(os.scandir(documents_root))
    except OSError as exc:
        raise ValidationError("Documents directory is absent or unreadable.") from exc
    if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
        _fail("Documents directory is not a real directory.")
    pdf_paths = []
    for entry in entries:
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValidationError("Documents directory contains an unreadable entry.") from exc
        if (
            entry.is_symlink()
            or not stat.S_ISREG(entry_stat.st_mode)
            or not entry.name.endswith(".pdf")
        ):
            _fail("Documents directory contains a non-PDF entry.")
        pdf_paths.append(documents_root / entry.name)
    pdf_paths = tuple(sorted(pdf_paths, key=lambda path: path.name))
    pdf_ids = {path.stem for path in pdf_paths}
    if set(tasks) != pdf_ids:
        _fail("Task IDs and PDF stems are not a bijection.")
    paths_by_id = {path.stem: path for path in pdf_paths}
    sealed_pdfs = []
    with _resolve_ocr_runtime() as runtime:
        try:
            temporary_directory = tempfile.TemporaryDirectory(prefix="docsem-validated-pdf-")
        except OSError as exc:
            raise ValidationError("PDF validation workspace could not be created safely.") from exc
        with temporary_directory as temporary_root:
            root = Path(temporary_root)
            root.chmod(0o700)
            for index, instance_id in enumerate(sorted(tasks)):
                source_path = paths_by_id[instance_id]
                payload = _read_bounded_regular_file(
                    source_path,
                    MAX_PDF_BYTES,
                    "Source PDF",
                )
                if not payload.startswith(b"%PDF-"):
                    _fail("PDF is unreadable.")
                digest = _sha256(payload)
                exact_path = root / f"{index}.pdf"
                _write_new_file(exact_path, payload, 0o600)
                required_blocks = set(labels[instance_id]["evidence"])
                if not required_blocks.issubset(
                    _render_visible_pdf_evidence_blocks(
                        exact_path,
                        runtime,
                        required_blocks,
                    )
                ):
                    _fail("PDF does not visibly contain every evidence block ID.")
                sealed_pdfs.append(
                    ValidatedPDF(
                        source_path=source_path.absolute(),
                        name=source_path.name,
                        size=len(payload),
                        sha256=digest,
                    )
                )
                exact_path.unlink()
                del payload
    return tuple(sealed_pdfs)


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

    pdfs = _validate_pdfs(root, tasks, labels)
    ids = tuple(sorted(tasks))
    task_rows = tuple(tasks[instance_id] for instance_id in ids)
    label_rows = tuple(labels[instance_id] for instance_id in ids)
    task_bytes = _canonical_rows(task_rows)
    label_bytes = _canonical_rows(label_rows)
    return ValidatedTestSource(
        source_root=root.resolve(),
        ids=ids,
        task_rows=task_rows,
        label_rows=label_rows,
        pdfs=pdfs,
        canonical_task_bytes=task_bytes,
        canonical_label_bytes=label_bytes,
        tasks_sha256=_sha256(task_bytes),
        private_labels_sha256=_sha256(label_bytes),
        pdf_inventory_sha256=_sealed_pdf_inventory_digest(pdfs),
    )


def _canonical_rows(rows: Iterable[dict]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inventory_digest(entries: Iterable[tuple[str, str]]) -> str:
    return _sha256(
        b"".join(
            f"{name}  {digest}\n".encode("ascii")
            for name, digest in sorted(entries)
        )
    )


def _sealed_pdf_inventory_digest(pdfs: Iterable[ValidatedPDF]) -> str:
    return _inventory_digest((pdf.name, pdf.sha256) for pdf in pdfs)


def _pdf_inventory_digest(paths: Iterable[Path]) -> str:
    return _inventory_digest(
        (
            path.name,
            _sha256(_read_bounded_regular_file(path, MAX_PDF_BYTES, "PDF inventory entry")),
        )
        for path in paths
    )


def _verify_validated_snapshot(validated: ValidatedTestSource) -> None:
    """Recompute all mutable rows and verify immutable validation-time seals."""
    if (
        not isinstance(validated, ValidatedTestSource)
        or type(validated.ids) is not tuple
        or type(validated.task_rows) is not tuple
        or type(validated.label_rows) is not tuple
        or type(validated.pdfs) is not tuple
        or type(validated.canonical_task_bytes) is not bytes
        or type(validated.canonical_label_bytes) is not bytes
    ):
        _fail("Validated source snapshot is malformed.")
    try:
        tasks = _validate_tasks(list(validated.task_rows))
        labels = _validate_labels(list(validated.label_rows))
    except (TypeError, KeyError) as exc:
        raise ValidationError("Validated source rows are malformed.") from exc
    ids = tuple(sorted(tasks))
    if ids != validated.ids or set(tasks) != set(labels):
        _fail("Validated source row IDs are inconsistent.")
    task_bytes = _canonical_rows(tasks[instance_id] for instance_id in ids)
    label_bytes = _canonical_rows(labels[instance_id] for instance_id in ids)
    if (
        task_bytes != validated.canonical_task_bytes
        or label_bytes != validated.canonical_label_bytes
        or type(validated.tasks_sha256) is not str
        or not _SHA256_HEX.fullmatch(validated.tasks_sha256)
        or type(validated.private_labels_sha256) is not str
        or not _SHA256_HEX.fullmatch(validated.private_labels_sha256)
        or _sha256(task_bytes) != validated.tasks_sha256
        or _sha256(label_bytes) != validated.private_labels_sha256
    ):
        _fail("Validated source rows changed after validation.")

    expected_names = tuple(f"{instance_id}.pdf" for instance_id in ids)
    if tuple(pdf.name for pdf in validated.pdfs) != expected_names:
        _fail("Validated source PDF inventory is inconsistent.")
    for pdf in validated.pdfs:
        if (
            not isinstance(pdf, ValidatedPDF)
            or not isinstance(pdf.source_path, Path)
            or type(pdf.name) is not str
            or type(pdf.size) is not int
            or not 0 < pdf.size <= MAX_PDF_BYTES
            or type(pdf.sha256) is not str
            or not _SHA256_HEX.fullmatch(pdf.sha256)
        ):
            _fail("Validated source PDF seal is malformed.")
    if (
        type(validated.pdf_inventory_sha256) is not str
        or not _SHA256_HEX.fullmatch(validated.pdf_inventory_sha256)
        or _sealed_pdf_inventory_digest(validated.pdfs)
        != validated.pdf_inventory_sha256
    ):
        _fail("Validated source PDF inventory seal is inconsistent.")


def build_release_manifest(validated: ValidatedTestSource, release_id: str) -> dict:
    """Return a deterministic, sanitized manifest with private values only hashed."""
    _verify_validated_snapshot(validated)
    if not isinstance(release_id, str) or not RELEASE_ID.fullmatch(release_id):
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
        "tasks_sha256": validated.tasks_sha256,
        "pdf_inventory_sha256": validated.pdf_inventory_sha256,
        "private_labels_sha256": validated.private_labels_sha256,
        "visibility_audit": _visibility_audit_contract(),
    }


def _canonical_json_document(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_new_file(path: Path, payload: bytes, mode: int) -> None:
    """Create one file without following an existing link and set its exact mode."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags, mode)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise ValidationError("A staging file could not be created safely.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _safe_relative_file_name(value: str) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.startswith(("/", "\\"))
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _walk_payload(root: Path) -> tuple[set[str], set[str]]:
    """Return regular files/directories without following any link or special node."""
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ValidationError("Public payload root is absent or unreadable.") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        _fail("Public payload root is not a real directory.")

    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path) -> None:
        try:
            entries = tuple(os.scandir(directory))
        except OSError as exc:
            raise ValidationError("Public payload tree is unreadable.") from exc
        for entry in entries:
            relative = (directory / entry.name).relative_to(root).as_posix()
            if not _safe_relative_file_name(relative):
                _fail("Public payload contains an unsafe path.")
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValidationError("Public payload tree is unreadable.") from exc
            if entry.is_symlink():
                _fail("Public payload contains a symbolic link.")
            if stat.S_ISDIR(entry_stat.st_mode):
                directories.add(relative)
                visit(directory / entry.name)
            elif stat.S_ISREG(entry_stat.st_mode):
                files.add(relative)
            else:
                _fail("Public payload contains a special file.")

    visit(root)
    return files, directories


def _contains_forbidden_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            not isinstance(key, str)
            or _PUBLIC_FORBIDDEN_FIELD.search(key)
            or _contains_forbidden_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def _public_manifest_schema_is_exact(manifest: object) -> bool:
    if not isinstance(manifest, dict) or set(manifest) != _PUBLIC_MANIFEST_KEYS:
        return False
    counts = manifest.get("counts")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
        or type(manifest.get("release_id")) is not str
        or not RELEASE_ID.fullmatch(manifest["release_id"])
        or not isinstance(counts, dict)
        or set(counts) != {"tasks", "pdfs"}
        or any(type(counts.get(key)) is not int or counts[key] <= 0 for key in counts)
    ):
        return False
    return all(
        type(manifest.get(key)) is str and bool(_SHA256_HEX.fullmatch(manifest[key]))
        for key in (
            "sorted_ids_sha256",
            "task_manifest_sha256",
            "pdf_inventory_sha256",
        )
    )


def _normalize_high_entropy_sentinels(values: Iterable[str | bytes]) -> tuple[bytes, ...]:
    """Accept only explicit canaries unlikely to collide with normal task text."""
    sentinels = []
    for value in values:
        if not isinstance(value, (str, bytes)):
            _fail("Public audit sentinel is malformed.")
        encoded = value if isinstance(value, bytes) else value.encode("utf-8")
        if len(encoded) < 16 or len(set(encoded)) < 8:
            _fail("Public audit sentinel is not sufficiently distinctive.")
        sentinels.append(encoded)
    return tuple(sentinels)


def _public_pdf_inventory_digest(paths: Iterable[Path]) -> str:
    return _pdf_inventory_digest(sorted(paths, key=lambda path: path.name))


def audit_public_payload(
    path: str | Path,
    *,
    forbidden_values: Iterable[str | bytes] = (),
) -> dict:
    """Fail closed unless a staged public payload is exact and label-free.

    ``forbidden_values`` accepts optional high-entropy release canaries only.
    Normal answer and evidence scalars are intentionally not substring-scanned
    because they can legitimately occur in public queries or source PDFs.
    """
    root = Path(path)
    sentinels = _normalize_high_entropy_sentinels(forbidden_values)
    files, directories = _walk_payload(root)
    if directories != {"test", PUBLIC_DOCUMENTS_RELATIVE_PATH}:
        _fail("Public payload directory inventory is not exact.")

    fixed_files = {
        PUBLIC_TASKS_RELATIVE_PATH,
        PUBLIC_MANIFEST_RELATIVE_PATH,
        PUBLIC_CHECKSUMS_RELATIVE_PATH,
    }
    pdf_files = files - fixed_files
    if (
        not pdf_files
        or any(
            not name.startswith(f"{PUBLIC_DOCUMENTS_RELATIVE_PATH}/")
            or not name.lower().endswith(".pdf")
            for name in pdf_files
        )
        or files != fixed_files | pdf_files
    ):
        _fail("Public payload file inventory is not exact.")

    task_path = root / PUBLIC_TASKS_RELATIVE_PATH
    task_bytes = _read_bounded_regular_file(
        task_path,
        MAX_PUBLIC_TASKS_BYTES,
        "Public task manifest",
    )
    try:
        task_rows = [json.loads(line) for line in task_bytes.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Public task manifest is malformed.") from exc
    if not task_rows or any(not isinstance(row, dict) or set(row) != TASK_KEYS for row in task_rows):
        _fail("Public task manifest schema is not exact.")
    if _contains_forbidden_field(task_rows):
        _fail("Public task manifest contains a forbidden field.")

    task_ids: list[str] = []
    for row in task_rows:
        instance_id = row["instance_id"]
        if (
            not _valid_identifier(instance_id)
            or not isinstance(row["user_query"], str)
            or not row["user_query"].strip()
            or row["document_pdf"]
            != f"{PUBLIC_DOCUMENTS_RELATIVE_PATH}/{instance_id}.pdf"
            or not _safe_relative_file_name(row["document_pdf"])
        ):
            _fail("Public task manifest contains an unsafe row.")
        task_ids.append(instance_id)
    if task_ids != sorted(set(task_ids)):
        _fail("Public task manifest IDs are not unique and sorted.")
    if task_bytes != _canonical_rows(task_rows):
        _fail("Public task manifest is not canonical.")
    expected_pdf_files = {
        f"{PUBLIC_DOCUMENTS_RELATIVE_PATH}/{instance_id}.pdf"
        for instance_id in task_ids
    }
    if pdf_files != expected_pdf_files:
        _fail("Public tasks and PDFs are not a bijection.")

    public_manifest_path = root / PUBLIC_MANIFEST_RELATIVE_PATH
    public_manifest_bytes = _read_bounded_regular_file(
        public_manifest_path,
        MAX_PUBLIC_MANIFEST_BYTES,
        "Public release manifest",
    )
    try:
        public_manifest = json.loads(public_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Public release manifest is malformed.") from exc
    if (
        not _public_manifest_schema_is_exact(public_manifest)
        or _contains_forbidden_field(public_manifest)
        or public_manifest_bytes != _canonical_json_document(public_manifest)
    ):
        _fail("Public release manifest schema is not exact.")

    pdf_paths = [root / name for name in sorted(pdf_files)]
    _require_pdf_renderer()
    pdf_digests = {}
    try:
        temporary_directory = tempfile.TemporaryDirectory(prefix="docsem-public-pdf-audit-")
    except OSError as exc:
        raise ValidationError("Public PDF audit workspace could not be created safely.") from exc
    with temporary_directory as temporary_root:
        audit_root = Path(temporary_root)
        audit_root.chmod(0o700)
        for index, pdf_path in enumerate(pdf_paths):
            payload = _read_bounded_regular_file(pdf_path, MAX_PDF_BYTES, "Public PDF")
            if (
                not payload.startswith(b"%PDF-")
                or not payload.rstrip().endswith(b"%%EOF")
                or payload.startswith(_ARCHIVE_MAGICS)
                or _EMBEDDED_FORBIDDEN_FIELD.search(payload)
            ):
                _fail("Public PDF contains an unsafe embedded payload.")
            exact_path = audit_root / f"{index}.pdf"
            _write_new_file(exact_path, payload, 0o600)
            _run_bounded_document_probe(exact_path)
            pdf_digests[pdf_path.name] = _sha256(payload)
            exact_path.unlink()
            del payload

    expected_public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": public_manifest.get("release_id"),
        "counts": {"tasks": len(task_rows), "pdfs": len(pdf_paths)},
        "sorted_ids_sha256": _sha256(
            "".join(f"{instance_id}\n" for instance_id in task_ids).encode("utf-8")
        ),
        "task_manifest_sha256": _sha256(task_bytes),
        "pdf_inventory_sha256": _inventory_digest(pdf_digests.items()),
    }
    if public_manifest != expected_public_manifest:
        _fail("Public release manifest hashes do not match the payload.")

    checksum_path = root / PUBLIC_CHECKSUMS_RELATIVE_PATH
    checksum_bytes = _read_bounded_regular_file(
        checksum_path,
        MAX_PUBLIC_CHECKSUM_BYTES,
        "Public checksum inventory",
    )
    checksum_targets = sorted(
        {PUBLIC_TASKS_RELATIVE_PATH, PUBLIC_MANIFEST_RELATIVE_PATH} | pdf_files
    )
    fixed_digests = {
        PUBLIC_TASKS_RELATIVE_PATH: _sha256(task_bytes),
        PUBLIC_MANIFEST_RELATIVE_PATH: _sha256(public_manifest_bytes),
    }
    expected_checksums = b"".join(
        (
            f"{fixed_digests[name] if name in fixed_digests else pdf_digests[Path(name).name]}  "
            f"{name.removeprefix('test/')}\n"
        ).encode("ascii")
        for name in checksum_targets
    )
    if checksum_bytes != expected_checksums:
        _fail("Public checksum inventory does not match the payload.")

    public_metadata = task_bytes + public_manifest_bytes + checksum_bytes
    if any(value in public_metadata for value in sentinels):
        _fail("Public metadata contains a private release sentinel.")

    return public_manifest


def _normalized_public_tasks(validated: ValidatedTestSource) -> tuple[dict, ...]:
    tasks_by_id = {row.get("instance_id"): row for row in validated.task_rows}
    if tuple(sorted(tasks_by_id)) != validated.ids or len(tasks_by_id) != len(validated.task_rows):
        _fail("Validated source task IDs are inconsistent.")
    rows = []
    for instance_id in validated.ids:
        row = tasks_by_id[instance_id]
        if set(row) != TASK_KEYS or row.get("document_pdf") != f"documents/{instance_id}.pdf":
            _fail("Validated source task schema is inconsistent.")
        rows.append(
            {
                "instance_id": instance_id,
                "user_query": row["user_query"],
                "document_pdf": f"{PUBLIC_DOCUMENTS_RELATIVE_PATH}/{instance_id}.pdf",
            }
        )
    return tuple(rows)


def _prepare_output_root(path: Path, mode: int) -> None:
    path.mkdir(mode=mode)
    path.chmod(mode)


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValidationError("A staging destination could not be inspected safely.") from exc
    return True


def stage_release(
    validated: ValidatedTestSource,
    public_root: str | Path,
    private_root: str | Path,
    release_id: str,
) -> dict:
    """Stage deterministic public and private payloads without publishing them."""
    if not isinstance(validated, ValidatedTestSource):
        _fail("A validated test source is required for staging.")
    source_manifest = build_release_manifest(validated, release_id)
    public_destination = Path(public_root).absolute()
    private_destination = Path(private_root).absolute()
    source_root = validated.source_root.absolute()
    try:
        public_resolved = public_destination.resolve(strict=False)
        private_resolved = private_destination.resolve(strict=False)
        source_resolved = source_root.resolve(strict=True)
    except OSError as exc:
        raise ValidationError("Staging paths could not be resolved safely.") from exc
    if (
        public_resolved == private_resolved
        or public_resolved == source_resolved
        or private_resolved == source_resolved
        or public_resolved in private_resolved.parents
        or private_resolved in public_resolved.parents
        or public_resolved in source_resolved.parents
        or private_resolved in source_resolved.parents
        or source_resolved in public_resolved.parents
        or source_resolved in private_resolved.parents
    ):
        _fail("Public, private, and source roots must be separate.")
    if _path_entry_exists(public_destination) or _path_entry_exists(private_destination):
        _fail("Staging destinations must not already exist.")

    tasks_by_id = _validate_tasks(list(validated.task_rows))
    labels_by_id = _validate_labels(list(validated.label_rows))
    if tuple(sorted(tasks_by_id)) != validated.ids or set(tasks_by_id) != set(labels_by_id):
        _fail("Validated source rows are inconsistent.")
    normalized_tasks = _normalized_public_tasks(validated)
    task_bytes = _canonical_rows(normalized_tasks)
    label_bytes = validated.canonical_label_bytes
    pdfs_by_name = {pdf.name: pdf for pdf in validated.pdfs}
    expected_pdf_names = {f"{instance_id}.pdf" for instance_id in validated.ids}
    if set(pdfs_by_name) != expected_pdf_names or len(pdfs_by_name) != len(validated.pdfs):
        _fail("Validated source PDF inventory is inconsistent.")

    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "counts": {"tasks": len(normalized_tasks), "pdfs": len(pdfs_by_name)},
        "sorted_ids_sha256": source_manifest["sorted_ids_sha256"],
        "task_manifest_sha256": _sha256(task_bytes),
        "pdf_inventory_sha256": source_manifest["pdf_inventory_sha256"],
    }
    public_manifest_bytes = _canonical_json_document(public_manifest)
    private_manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "counts": source_manifest["counts"],
        "sorted_ids_sha256": source_manifest["sorted_ids_sha256"],
        "task_manifest_sha256": _sha256(task_bytes),
        "gold_sha256": _sha256(label_bytes),
        "pdf_inventory_sha256": source_manifest["pdf_inventory_sha256"],
        "visibility_audit": source_manifest["visibility_audit"],
        "enabled": False,
        "max_attempts": 3,
        "feedback_policy": "first-attempt-only",
        "finalized": False,
    }

    try:
        for parent in (public_destination.parent, private_destination.parent):
            parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValidationError("Staging parent directories could not be prepared safely.") from exc
    public_temporary = None
    private_temporary = None
    public_committed = False
    private_committed = False
    try:
        public_temporary = Path(
            tempfile.mkdtemp(prefix=".docsem-public-stage-", dir=public_destination.parent)
        )
        private_temporary = Path(
            tempfile.mkdtemp(prefix=".docsem-private-stage-", dir=private_destination.parent)
        )
        public_temporary.chmod(0o700)
        private_temporary.chmod(0o700)
        public_test = public_temporary / "test"
        public_documents = public_test / "documents"
        private_directory = private_temporary / "private"
        _prepare_output_root(public_test, 0o755)
        _prepare_output_root(public_documents, 0o755)
        _prepare_output_root(private_directory, 0o700)

        _write_new_file(public_test / "tasks.jsonl", task_bytes, 0o644)
        _write_new_file(public_test / "release.json", public_manifest_bytes, 0o644)
        staged_digests = {
            "tasks.jsonl": _sha256(task_bytes),
            "release.json": _sha256(public_manifest_bytes),
        }
        for name in sorted(pdfs_by_name):
            sealed_pdf = pdfs_by_name[name]
            pdf_bytes = _read_bounded_regular_file(
                sealed_pdf.source_path,
                MAX_PDF_BYTES,
                "Validated source PDF",
            )
            digest = _sha256(pdf_bytes)
            if len(pdf_bytes) != sealed_pdf.size or digest != sealed_pdf.sha256:
                _fail("Validated source PDF changed after validation.")
            _write_new_file(public_documents / name, pdf_bytes, 0o644)
            staged_digests[f"documents/{name}"] = digest
            del pdf_bytes

        checksum_targets = sorted(
            ["tasks.jsonl", "release.json"]
            + [f"documents/{name}" for name in pdfs_by_name]
        )
        checksum_bytes = b"".join(
            f"{staged_digests[name]}  {name}\n".encode("ascii")
            for name in checksum_targets
        )
        _write_new_file(public_test / "SHA256SUMS", checksum_bytes, 0o644)

        _write_new_file(private_directory / "test_labels.jsonl", label_bytes, 0o600)
        _write_new_file(
            private_directory / "test_release.json",
            _canonical_json_document(private_manifest),
            0o600,
        )
        audit_public_payload(public_temporary)

        public_temporary.chmod(0o755)
        os.replace(private_temporary, private_destination)
        private_committed = True
        os.replace(public_temporary, public_destination)
        public_committed = True
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Staging payloads could not be committed safely.") from exc
    finally:
        if public_temporary is not None and not public_committed:
            shutil.rmtree(public_temporary, ignore_errors=True)
        if private_temporary is not None and not private_committed:
            shutil.rmtree(private_temporary, ignore_errors=True)
        if private_committed and not public_committed:
            shutil.rmtree(private_destination, ignore_errors=True)

    return private_manifest
