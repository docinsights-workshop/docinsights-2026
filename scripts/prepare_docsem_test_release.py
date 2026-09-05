"""Validate one explicitly selected DocSem held-out test source.

This module deliberately validates only a caller-provided directory.  It does
not discover, select, stage, publish, or activate any local test material.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

try:
    import fitz
except ImportError:
    fitz = None


SCHEMA_VERSION = 1
TASK_KEYS = frozenset(("instance_id", "user_query", "document_pdf"))
LABEL_KEYS = frozenset(("instance_id", "answer", "evidence"))
BLOCK_ID = re.compile(r"b[0-9]+$")


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


def _render_visible_pdf_text(path: Path) -> str:
    if fitz is None:
        _fail("PDF renderer is unavailable.")
    try:
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                _fail("PDF is unreadable.")
        visible_runs = []
        with fitz.open(str(path)) as document:
            if document.needs_pass:
                _fail("PDF is encrypted and cannot be inspected.")
            for page in document:
                pixmap = page.get_pixmap(alpha=False)
                if pixmap.width <= 0 or pixmap.height <= 0:
                    _fail("PDF renderer produced an invalid page image.")
                for trace in page.get_texttrace():
                    if trace.get("type") not in {0, 1, 2} or float(trace.get("opacity", 0)) <= 0:
                        continue
                    visible_runs.append("".join(chr(character[0]) for character in trace["chars"]))
        if not visible_runs:
            _fail("PDF contains no renderer-visible text.")
        return "\n".join(visible_runs)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("PDF is unreadable.") from exc


def _has_visible_block(text: str, block_id: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(block_id)}(?![A-Za-z0-9_])",
            text,
        )
    )


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
        text = _render_visible_pdf_text(paths_by_id[instance_id])
        if any(not _has_visible_block(text, block_id) for block_id in labels[instance_id]["evidence"]):
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
