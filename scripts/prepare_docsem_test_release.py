"""Validate one explicitly selected DocSem held-out test source.

This module deliberately validates only a caller-provided directory.  It does
not discover, select, stage, publish, or activate any local test material.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

try:
    import fitz
    from pymupdf import _mupdf as _fitz_raw
    from pymupdf import mupdf as _fitz_mupdf
except ImportError:
    fitz = None
    _fitz_raw = None
    _fitz_mupdf = None


SCHEMA_VERSION = 1
TASK_KEYS = frozenset(("instance_id", "user_query", "document_pdf"))
LABEL_KEYS = frozenset(("instance_id", "answer", "evidence"))
BLOCK_ID = re.compile(r"b[0-9]+$")
GLYPH_RENDER_SCALE = 4
GLYPH_CORE_EDGE = 2


class ValidationError(ValueError):
    """Raised for a structural source error without exposing private rows."""


if _fitz_mupdf is not None:

    class _TextOperationOmittingDevice(_fitz_mupdf.FzDevice2):
        """Replay a MuPDF page while suppressing one paint operation."""

        def __init__(self, target: object, omitted_seqno: int | None):
            super().__init__()
            self._target = target
            self._omitted_seqno = omitted_seqno
            self._seqno = 0
            for callback in (
                "fill_path",
                "stroke_path",
                "clip_path",
                "clip_stroke_path",
                "fill_text",
                "stroke_text",
                "clip_text",
                "clip_stroke_text",
                "ignore_text",
                "fill_shade",
                "fill_image",
                "fill_image_mask",
                "clip_image_mask",
                "pop_clip",
                "begin_mask",
                "end_mask",
                "begin_group",
                "end_group",
                "begin_tile",
                "end_tile",
                "render_flags",
                "set_default_colorspaces",
                "begin_layer",
                "end_layer",
                "begin_structure",
                "end_structure",
                "begin_metatext",
                "end_metatext",
            ):
                getattr(self, f"use_virtual_{callback}")()

        def _forward(self, operation: str, *args):
            callback = getattr(_fitz_raw, f"ll_fz_{operation}")
            return callback(self._target.m_internal, *args)

        def _paint(self, operation: str, *args) -> None:
            if self._seqno != self._omitted_seqno:
                self._forward(operation, *args)
            self._seqno += 1

        def fill_path(self, _context, *args):
            self._paint("fill_path", *args)

        def stroke_path(self, _context, *args):
            self._paint("stroke_path", *args)

        def clip_path(self, _context, *args):
            self._forward("clip_path", *args)

        def clip_stroke_path(self, _context, *args):
            self._forward("clip_stroke_path", *args)

        def fill_text(self, _context, *args):
            self._paint("fill_text", *args)

        def stroke_text(self, _context, *args):
            self._paint("stroke_text", *args)

        def clip_text(self, _context, *args):
            self._forward("clip_text", *args)

        def clip_stroke_text(self, _context, *args):
            self._forward("clip_stroke_text", *args)

        def ignore_text(self, _context, *args):
            self._paint("ignore_text", *args)

        def fill_shade(self, _context, *args):
            self._paint("fill_shade", *args)

        def fill_image(self, _context, *args):
            self._paint("fill_image", *args)

        def fill_image_mask(self, _context, *args):
            self._paint("fill_image_mask", *args)

        def clip_image_mask(self, _context, *args):
            self._forward("clip_image_mask", *args)

        def pop_clip(self, _context):
            self._forward("pop_clip")

        def begin_mask(self, _context, *args):
            self._forward("begin_mask", *args)

        def end_mask(self, _context, transfer_function):
            self._forward("end_mask_tr", transfer_function)

        def begin_group(self, _context, *args):
            self._forward("begin_group", *args)

        def end_group(self, _context):
            self._forward("end_group")

        def begin_tile(self, _context, *args):
            return self._forward("begin_tile_id", *args)

        def end_tile(self, _context):
            self._forward("end_tile")

        def render_flags(self, _context, *args):
            self._forward("render_flags", *args)

        def set_default_colorspaces(self, _context, *args):
            self._forward("set_default_colorspaces", *args)

        def begin_layer(self, _context, *args):
            self._forward("begin_layer", *args)

        def end_layer(self, _context):
            self._forward("end_layer")

        def begin_structure(self, _context, *args):
            self._forward("begin_structure", *args)

        def end_structure(self, _context):
            self._forward("end_structure")

        def begin_metatext(self, _context, *args):
            self._forward("begin_metatext", *args)

        def end_metatext(self, _context):
            self._forward("end_metatext")

else:
    _TextOperationOmittingDevice = None


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


def _numeric_rect(value: object, description: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        _fail(f"PDF {description} is malformed.")
    try:
        left, top, right, bottom = (float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"PDF {description} is malformed.") from exc
    if not all(math.isfinite(part) for part in (left, top, right, bottom)) or left >= right or top >= bottom:
        _fail(f"PDF {description} is malformed.")
    return left, top, right, bottom


def _validate_pixmap(pixmap: object) -> None:
    if (
        not isinstance(pixmap.width, int)
        or not isinstance(pixmap.height, int)
        or not isinstance(pixmap.n, int)
        or not isinstance(pixmap.stride, int)
        or pixmap.width <= 0
        or pixmap.height <= 0
        or pixmap.n < 3
        or pixmap.stride < pixmap.width * pixmap.n
        or len(pixmap.samples) < pixmap.stride * pixmap.height
    ):
        _fail("PDF renderer produced an invalid page image.")


def _trace_rgb(trace: dict) -> tuple[int, int, int]:
    color = trace.get("color")
    if not isinstance(color, (tuple, list)) or len(color) != 3:
        _fail("PDF text color is malformed.")
    try:
        values = tuple(float(component) for component in color)
    except (TypeError, ValueError) as exc:
        raise ValidationError("PDF text color is malformed.") from exc
    if not all(math.isfinite(component) and 0 <= component <= 1 for component in values):
        _fail("PDF text color is malformed.")
    return tuple(int(component * 255) for component in values)


def _pixmaps_have_contributed_color_core(
    rendered: object,
    replayed_with_text_operation: object,
    without_text_operation: object,
    expected_rgb: tuple[int, int, int],
) -> bool:
    if expected_rgb == (255, 255, 255):
        return False
    _validate_pixmap(rendered)
    _validate_pixmap(replayed_with_text_operation)
    _validate_pixmap(without_text_operation)
    geometry = (rendered.width, rendered.height, rendered.n, rendered.stride)
    if geometry != (
        replayed_with_text_operation.width,
        replayed_with_text_operation.height,
        replayed_with_text_operation.n,
        replayed_with_text_operation.stride,
    ) or geometry != (
        without_text_operation.width,
        without_text_operation.height,
        without_text_operation.n,
        without_text_operation.stride,
    ):
        _fail("PDF differential renderer produced incompatible page images.")
    rendered_samples = rendered.samples
    replayed_samples = replayed_with_text_operation.samples
    baseline_samples = without_text_operation.samples
    for y in range(rendered.height - GLYPH_CORE_EDGE + 1):
        for x in range(rendered.width - GLYPH_CORE_EDGE + 1):
            if all(
                (
                    tuple(
                        rendered_samples[
                            (y + delta_y) * rendered.stride
                            + (x + delta_x) * rendered.n
                            + component
                        ]
                        for component in range(3)
                    )
                    == expected_rgb
                    and tuple(
                        replayed_samples[
                            (y + delta_y) * replayed_with_text_operation.stride
                            + (x + delta_x) * replayed_with_text_operation.n
                            + component
                        ]
                        for component in range(3)
                    )
                    != tuple(
                        baseline_samples[
                            (y + delta_y) * without_text_operation.stride
                            + (x + delta_x) * without_text_operation.n
                            + component
                        ]
                        for component in range(3)
                    )
                )
                for delta_y in range(GLYPH_CORE_EDGE)
                for delta_x in range(GLYPH_CORE_EDGE)
            ):
                return True
    return False


def _render_bbox_omitting_operation(
    page: object,
    bbox: object,
    omitted_seqno: int | None,
) -> object:
    if _TextOperationOmittingDevice is None or _fitz_mupdf is None:
        _fail("PDF differential renderer is unavailable.")
    left, top, right, bottom = _numeric_rect(bbox, "text bounding box")
    matrix = fitz.Matrix(GLYPH_RENDER_SCALE, GLYPH_RENDER_SCALE)
    device_bounds = (fitz.Rect(left, top, right, bottom) * matrix).irect
    pixmap = fitz.Pixmap(fitz.csRGB, device_bounds, False)
    pixmap.clear_with(255)
    draw_device = _fitz_mupdf.fz_new_draw_device(
        _fitz_mupdf.FzMatrix(GLYPH_RENDER_SCALE, 0, 0, GLYPH_RENDER_SCALE, 0, 0),
        pixmap.this,
    )
    replay_device = _TextOperationOmittingDevice(draw_device, omitted_seqno)
    try:
        _fitz_mupdf.fz_run_page(
            page.this,
            replay_device,
            _fitz_mupdf.FzMatrix(),
            _fitz_mupdf.FzCookie(),
        )
    finally:
        try:
            _fitz_mupdf.fz_close_device(replay_device)
        finally:
            _fitz_mupdf.fz_close_device(draw_device)
    _validate_pixmap(pixmap)
    return pixmap


def _bbox_has_text_operation_glyph_core(
    page: object,
    bbox: object,
    expected_rgb: tuple[int, int, int],
    text_seqno: int,
) -> bool:
    left, top, right, bottom = _numeric_rect(bbox, "text bounding box")
    rendered = page.get_pixmap(
        matrix=fitz.Matrix(GLYPH_RENDER_SCALE, GLYPH_RENDER_SCALE),
        clip=fitz.Rect(left, top, right, bottom),
        alpha=False,
        colorspace=fitz.csRGB,
    )
    replayed_with_text_operation = _render_bbox_omitting_operation(page, bbox, None)
    without_text_operation = _render_bbox_omitting_operation(page, bbox, text_seqno)
    return _pixmaps_have_contributed_color_core(
        rendered,
        replayed_with_text_operation,
        without_text_operation,
        expected_rgb,
    )


def _text_paint_seqno(trace: dict, bbox_log: list) -> int:
    seqno = trace.get("seqno")
    if type(seqno) is not int or seqno < 0 or seqno >= len(bbox_log):
        _fail("PDF text paint sequence is malformed.")
    record = bbox_log[seqno]
    expected_kind = {0: "fill-text", 1: "stroke-text"}.get(trace.get("type"))
    if (
        expected_kind is None
        or not isinstance(record, (tuple, list))
        or len(record) < 2
        or record[0] != expected_kind
    ):
        _fail("PDF text paint sequence is malformed.")
    _numeric_rect(record[1], "text paint bounding box")
    return seqno


def _trace_characters(trace: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
    raw_characters = trace.get("chars")
    if not isinstance(raw_characters, (tuple, list)) or not raw_characters:
        _fail("PDF text character geometry is malformed.")
    characters = []
    for character in raw_characters:
        if not isinstance(character, (tuple, list)) or len(character) < 4 or not isinstance(character[0], int):
            _fail("PDF text character geometry is malformed.")
        try:
            text = chr(character[0])
        except ValueError as exc:
            raise ValidationError("PDF text character geometry is malformed.") from exc
        characters.append((text, _numeric_rect(character[3], "character bounding box")))
    return characters


def _block_match_spans(text: str, block_id: str):
    return re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(block_id)}(?![A-Za-z0-9_])", text)


def _matched_blocks_with_rendered_characters(
    trace: dict,
    page: object,
    page_rect: tuple[float, float, float, float],
    bbox_log: list,
    evidence_ids: set[str],
) -> set[str]:
    _numeric_rect(trace.get("bbox"), "text bounding box")
    expected_rgb = _trace_rgb(trace)
    text_seqno = _text_paint_seqno(trace, bbox_log)
    characters = _trace_characters(trace)
    text = "".join(character[0] for character in characters)
    page_left, page_top, page_right, page_bottom = page_rect
    matched = set()
    for block_id in evidence_ids:
        for match in _block_match_spans(text, block_id):
            matched_characters = characters[match.start() : match.end()]
            if all(
                left >= page_left
                and top >= page_top
                and right <= page_right
                and bottom <= page_bottom
                and _bbox_has_text_operation_glyph_core(
                    page,
                    (left, top, right, bottom),
                    expected_rgb,
                    text_seqno,
                )
                for _, (left, top, right, bottom) in matched_characters
            ):
                matched.add(block_id)
    return matched


def _render_visible_pdf_evidence_blocks(path: Path, evidence_ids: set[str]) -> set[str]:
    if fitz is None:
        _fail("PDF renderer is unavailable.")
    try:
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                _fail("PDF is unreadable.")
        matched_blocks = set()
        with fitz.open(str(path)) as document:
            if document.needs_pass:
                _fail("PDF is encrypted and cannot be inspected.")
            for page in document:
                page_rect = _numeric_rect(tuple(page.rect), "page bounds")
                bbox_log = page.get_bboxlog()
                if not isinstance(bbox_log, list):
                    _fail("PDF renderer paint log is malformed.")
                for trace in page.get_texttrace():
                    if trace.get("type") not in {0, 1} or float(trace.get("opacity", 0)) <= 0:
                        continue
                    matched_blocks.update(
                        _matched_blocks_with_rendered_characters(
                            trace,
                            page,
                            page_rect,
                            bbox_log,
                            evidence_ids,
                        )
                    )
        return matched_blocks
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
    for instance_id in sorted(tasks):
        required_blocks = set(labels[instance_id]["evidence"])
        if _render_visible_pdf_evidence_blocks(paths_by_id[instance_id], required_blocks) != required_blocks:
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
    }
