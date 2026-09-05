"""Validate one explicitly selected DocSem held-out test source.

This module deliberately validates only a caller-provided directory.  It does
not discover, select, stage, publish, or activate any local test material.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import select
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

_PAGE_RESULT_HEADER = struct.Struct(">Q")
_DOCUMENT_PROBE_RESULT = struct.Struct(">I")


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


@dataclass(frozen=True)
class OCRRuntime:
    """Pinned OCR executable and validation-owned English model root."""

    binary: str
    tessdata_root: Path
    traineddata_sha256: str


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
    if fitz is None:
        _fail("PDF renderer is unavailable.")
    if getattr(fitz, "VersionBind", None) != PYMUPDF_VERSION:
        _fail("PDF renderer version does not match the tested contract.")
    if not hasattr(os, "fork") or not hasattr(os, "setsid"):
        _fail("Bounded PDF page workers are unavailable.")
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


def _probe_pdf_document(path: Path) -> int:
    """Open a PDF and return its bounded page count inside a worker."""
    with fitz.open(str(path)) as document:
        if document.needs_pass:
            _fail("PDF is encrypted and cannot be inspected.")
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


def _validate_pdfs(source_root: Path, tasks: dict[str, dict], labels: dict[str, dict]) -> tuple[Path, ...]:
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
    with _resolve_ocr_runtime() as runtime:
        for instance_id in sorted(tasks):
            required_blocks = set(labels[instance_id]["evidence"])
            if not required_blocks.issubset(
                _render_visible_pdf_evidence_blocks(
                    paths_by_id[instance_id],
                    runtime,
                    required_blocks,
                )
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
