"""Fixture-based tests for fail-closed DocSem held-out test source validation."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
import select
import shutil
import signal
import stat
import time
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import prepare_docsem_test_release as release_module
from prepare_docsem_test_release import (
    ValidationError,
    build_release_manifest,
    validate_source,
)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_pdf_with_visible_text(
    path,
    text,
    *,
    rendering_mode=0,
    font_size=12,
    x=72,
    y=720,
    fill_color=None,
    underlay_content="",
    font_name="F1",
):
    """Write a tiny self-contained PDF whose text extractor sees ``text``."""
    color = "" if fill_color is None else f"{fill_color[0]} {fill_color[1]} {fill_color[2]} rg "
    content = (
        f"{underlay_content}{color}BT /{font_name} {font_size} Tf "
        f"{rendering_mode} Tr {x} {y} Td ({text}) Tj ET"
    ).encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            + f"/Resources << /Font << /{font_name} 4 0 R >> >> ".encode("ascii")
            + b"/Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    )
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets[1:]))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(output)


def replace_with_image_only_pdf(path, header_text):
    """Rasterize one readable header into a PDF with no extraction text."""
    fitz = release_module.fitz
    source_path = path.with_name(f"{path.stem}-vector.pdf")
    write_pdf_with_visible_text(source_path, header_text, font_size=24)
    try:
        with fitz.open(source_path) as vector_document:
            page_rect = vector_document[0].rect
            image_bytes = vector_document[0].get_pixmap(
                dpi=150,
                colorspace=fitz.csGRAY,
                alpha=False,
            ).tobytes("png")
        image_document = fitz.open()
        try:
            page = image_document.new_page(width=page_rect.width, height=page_rect.height)
            page.insert_image(page.rect, stream=image_bytes)
            image_document.save(path)
        finally:
            image_document.close()
    finally:
        source_path.unlink(missing_ok=True)


def duplicate_pdf_page(path):
    """Turn a one-page fixture into a two-page fixture using public APIs."""
    fitz = release_module.fitz
    replacement = path.with_name(f"{path.stem}-two-pages.pdf")
    with fitz.open(path) as source:
        output = fitz.open()
        try:
            output.insert_pdf(source)
            output.insert_pdf(source)
            output.save(replacement)
        finally:
            output.close()
    replacement.replace(path)


def add_embedded_file(path, name, payload):
    """Attach compressed organizer-only bytes through PyMuPDF's public API."""
    fitz = release_module.fitz
    replacement = path.with_name(f"{path.stem}-with-attachment.pdf")
    with fitz.open(path) as document:
        document.embfile_add(name, payload, filename=name)
        document.save(replacement, deflate=True)
    replacement.replace(path)


def add_catalog_javascript(path):
    """Add an active catalog action through PyMuPDF's public API."""
    fitz = release_module.fitz
    replacement = path.with_name(f"{path.stem}-with-action.pdf")
    with fitz.open(path) as document:
        document.xref_set_key(
            document.pdf_catalog(),
            "OpenAction",
            "<</S/JavaScript/JS(app.alert\\(1\\))>>",
        )
        document.save(replacement, deflate=True)
    replacement.replace(path)


def refresh_public_release_hashes(public_root):
    """Reconcile public hashes so structural PDF tests cannot fail on drift alone."""
    test_root = public_root / "test"
    pdf_paths = sorted((test_root / "documents").glob("*.pdf"), key=lambda item: item.name)
    manifest_path = test_root / "release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pdf_inventory_sha256"] = release_module._public_pdf_inventory_digest(pdf_paths)
    manifest_path.write_bytes(release_module._canonical_json_document(manifest))
    refresh_public_checksums(public_root)


def refresh_public_checksums(public_root):
    """Reconcile checksums without changing the release manifest under test."""
    test_root = public_root / "test"
    pdf_paths = sorted((test_root / "documents").glob("*.pdf"), key=lambda item: item.name)
    targets = sorted(
        ["tasks.jsonl", "release.json"]
        + [f"documents/{pdf_path.name}" for pdf_path in pdf_paths]
    )
    (test_root / "SHA256SUMS").write_bytes(
        b"".join(
            (
                f"{hashlib.sha256((test_root / name).read_bytes()).hexdigest()}  {name}\n"
            ).encode("ascii")
            for name in targets
        )
    )


@contextmanager
def temporary_module_values(**values):
    """Temporarily replace dependency/runtime settings at their public boundary."""
    missing = object()
    originals = {name: getattr(release_module, name, missing) for name in values}
    try:
        for name, value in values.items():
            setattr(release_module, name, value)
        yield
    finally:
        for name, value in originals.items():
            if value is missing:
                delattr(release_module, name)
            else:
                setattr(release_module, name, value)


def write_fake_tesseract(path, *, version="5.5.1", ocr_body="exit 0"):
    """Create a real subprocess boundary with one controlled OCR behavior."""
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        f"  printf '%s\\n' 'tesseract {version}'\n"
        "  exit 0\n"
        "fi\n"
        f"{ocr_body}\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


class _RendererMustNotRun:
    """Renderer sentinel used to prove runtime validation precedes rendering."""

    VersionBind = release_module.PYMUPDF_VERSION

    def __init__(self, marker_path):
        self._marker_path = marker_path

    def open(self, *_args, **_kwargs):
        self._marker_path.write_text("rendering began", encoding="utf-8")
        raise ValidationError("PDF renderer was invoked.")


class _SlowPixmapPage:
    def __init__(self, page, delay_seconds, completion_marker):
        self._page = page
        self._delay_seconds = delay_seconds
        self._completion_marker = completion_marker

    def __getattr__(self, name):
        return getattr(self._page, name)

    def get_pixmap(self, *args, **kwargs):
        time.sleep(self._delay_seconds)
        self._completion_marker.write_text("renderer completed", encoding="utf-8")
        return self._page.get_pixmap(*args, **kwargs)


class _SlowPixmapDocument:
    def __init__(self, document, delay_seconds, completion_marker):
        self._document = document
        self._delay_seconds = delay_seconds
        self._completion_marker = completion_marker

    def __enter__(self):
        self._document.__enter__()
        return self

    def __exit__(self, *args):
        return self._document.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._document, name)

    def __iter__(self):
        for page in self._document:
            yield _SlowPixmapPage(page, self._delay_seconds, self._completion_marker)

    def load_page(self, page_number):
        return _SlowPixmapPage(
            self._document.load_page(page_number),
            self._delay_seconds,
            self._completion_marker,
        )


class _SlowPixmapRenderer:
    def __init__(self, renderer, delay_seconds, completion_marker):
        self._renderer = renderer
        self._delay_seconds = delay_seconds
        self._completion_marker = completion_marker
        self.VersionBind = renderer.VersionBind
        self.csGRAY = renderer.csGRAY

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def open(self, *args, **kwargs):
        return _SlowPixmapDocument(
            self._renderer.open(*args, **kwargs),
            self._delay_seconds,
            self._completion_marker,
        )


class _SlowOpenRenderer:
    """Renderer whose PDF open cannot stall the supervising parent."""

    def __init__(self, renderer, delay_seconds, completion_marker):
        self._renderer = renderer
        self._delay_seconds = delay_seconds
        self._completion_marker = completion_marker
        self.VersionBind = renderer.VersionBind
        self.csGRAY = renderer.csGRAY

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def open(self, *args, **kwargs):
        time.sleep(self._delay_seconds)
        self._completion_marker.write_text("PDF open completed", encoding="utf-8")
        return self._renderer.open(*args, **kwargs)


class _SlowPageCountDocument:
    """Document whose metadata probe cannot stall the supervising parent."""

    def __init__(self, document, delay_seconds, completion_marker):
        self._document = document
        self._delay_seconds = delay_seconds
        self._completion_marker = completion_marker

    def __enter__(self):
        self._document.__enter__()
        return self

    def __exit__(self, *args):
        return self._document.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._document, name)

    @property
    def page_count(self):
        time.sleep(self._delay_seconds)
        self._completion_marker.write_text("page count completed", encoding="utf-8")
        return self._document.page_count


class _SlowPageCountRenderer:
    def __init__(self, renderer, delay_seconds, completion_marker):
        self._renderer = renderer
        self._delay_seconds = delay_seconds
        self._completion_marker = completion_marker
        self.VersionBind = renderer.VersionBind
        self.csGRAY = renderer.csGRAY

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def open(self, *args, **kwargs):
        return _SlowPageCountDocument(
            self._renderer.open(*args, **kwargs),
            self._delay_seconds,
            self._completion_marker,
        )


class SourceFixture:
    """Creates complete synthetic test sources without touching project data."""

    def __init__(self, root):
        self.root = root

    def create(self):
        (self.root / "documents").mkdir(parents=True)
        self.tasks = [
            {
                "instance_id": f"synthetic-{number}",
                "user_query": f"Synthetic query {number}",
                "document_pdf": f"documents/synthetic-{number}.pdf",
            }
            for number in range(1, 4)
        ]
        self.labels = [
            {
                "instance_id": f"synthetic-{number}",
                "answer": f"private-synthetic-answer-{number}",
                "evidence": [f"b{number:02d}"],
            }
            for number in range(1, 4)
        ]
        write_jsonl(self.root / "tasks.jsonl", self.tasks)
        write_jsonl(self.root / "labels.jsonl", self.labels)
        for number in range(1, 4):
            write_pdf_with_visible_text(
                self.root / "documents" / f"synthetic-{number}.pdf",
                f"b{number:02d}: Synthetic fixture visible block",
            )
        return self

    def write_tasks(self):
        write_jsonl(self.root / "tasks.jsonl", self.tasks)

    def write_labels(self):
        write_jsonl(self.root / "labels.jsonl", self.labels)


class PrepareDocsemTestReleaseTests(unittest.TestCase):
    def make_source(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        return SourceFixture(Path(tempdir.name) / "official-source").create()

    def assert_rejected(self, source, *, train_ids=(), val_ids=()):
        with self.assertRaises(ValidationError) as caught:
            validate_source(source.root, train_ids, val_ids)
        self.assertNotIn("private-synthetic-answer", str(caught.exception))

    def test_accepts_only_the_complete_exact_source_contract(self):
        """Catches accepting schema drift, split leakage, or unverifiable PDFs."""
        source = self.make_source()

        validated = validate_source(source.root, {"train-only"}, {"validation-only"})

        self.assertEqual(validated.ids, ("synthetic-1", "synthetic-2", "synthetic-3"))
        self.assertEqual(validated.task_rows, tuple(source.tasks))
        self.assertEqual(validated.label_rows, tuple(source.labels))
        self.assertEqual(
            tuple(path.name for path in validated.pdf_paths),
            ("synthetic-1.pdf", "synthetic-2.pdf", "synthetic-3.pdf"),
        )

    def test_manifest_is_deterministic_and_contains_only_sanitized_digests(self):
        """Catches manifests that leak labels or depend on incoming row order."""
        source = self.make_source()
        validated = validate_source(source.root, (), ())

        manifest = build_release_manifest(validated, "synthetic-release-v1")
        source.tasks.reverse()
        source.labels.reverse()
        source.write_tasks()
        source.write_labels()
        reordered = build_release_manifest(validate_source(source.root, (), ()), "synthetic-release-v1")

        self.assertEqual(manifest, reordered)
        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "release_id",
                "counts",
                "sorted_ids_sha256",
                "tasks_sha256",
                "pdf_inventory_sha256",
                "private_labels_sha256",
                "visibility_audit",
            },
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["counts"], {"tasks": 3, "pdfs": 3, "labels": 3})
        self.assertEqual(
            manifest["sorted_ids_sha256"],
            hashlib.sha256(b"synthetic-1\nsynthetic-2\nsynthetic-3\n").hexdigest(),
        )
        serialized = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("private-synthetic-answer", serialized)
        self.assertNotIn('"evidence"', serialized)
        self.assertNotIn('"answer"', serialized)
        self.assertEqual(
            manifest["visibility_audit"],
            {
                "method": "pymupdf-raster-tesseract-cli",
                "pymupdf_version": "1.26.3",
                "tesseract_version": "5.5.1",
                "tesseract_binary_sha256": "6517c9cf1b17280201af3e48880517bbfafd24b5876aacb75d5643bafff1c414",
                "traineddata_sha256": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
                "render_dpi": 300,
                "colorspace": "grayscale",
                "ocr_language": "eng",
                "page_segmentation_mode": 6,
                "max_pdf_bytes": 16777216,
                "max_pages": 16,
                "max_page_width_points": 1000,
                "max_page_height_points": 1500,
                "max_render_pixels_per_page": 12000000,
                "max_render_pixels_total": 96000000,
                "max_raster_bytes": 33554432,
                "max_ocr_output_bytes": 1048576,
                "ocr_timeout_seconds_per_page": 30,
                "page_workflow_timeout_seconds": 45,
                "page_worker_cpu_seconds": 40,
                "page_worker_open_files": 64,
                "max_tesseract_binary_bytes": 16777216,
                "max_traineddata_bytes": 8388608,
                "max_evidence_ids_per_task": 1024,
            },
        )
        self.assertNotIn("b01", serialized)

    def test_validated_snapshot_repr_does_not_expose_private_rows(self):
        """Catches private label bytes or rows leaking through dataclass repr."""
        source = self.make_source()

        validated = validate_source(source.root, (), ())

        rendered = repr(validated)
        self.assertNotIn("private-synthetic-answer", rendered)
        self.assertNotIn("'evidence'", rendered)

    def test_rejects_duplicate_task_ids(self):
        """Catches a release whose task identifiers are not unique."""
        source = self.make_source()
        source.tasks.append(dict(source.tasks[0]))
        source.write_tasks()
        self.assert_rejected(source)

    def test_rejects_ids_that_would_make_the_manifest_ambiguous(self):
        """Catches control characters that could forge a sorted-ID digest line."""
        source = self.make_source()
        source.tasks[0]["instance_id"] = "synthetic-1\nforged-id"
        source.tasks[0]["document_pdf"] = "documents/synthetic-1\nforged-id.pdf"
        source.labels[0]["instance_id"] = "synthetic-1\nforged-id"
        (source.root / "documents" / "synthetic-1.pdf").rename(
            source.root / "documents" / "synthetic-1\nforged-id.pdf"
        )
        source.write_tasks()
        source.write_labels()
        self.assert_rejected(source)

    def test_rejects_missing_or_extra_pdfs(self):
        """Catches a task/PDF mapping that is not a bijection."""
        source = self.make_source()
        (source.root / "documents" / "synthetic-1.pdf").unlink()
        self.assert_rejected(source)

        source = self.make_source()
        (source.root / "documents" / "extra.pdf").write_bytes(b"not relevant")
        self.assert_rejected(source)

    def test_rejects_missing_or_extra_labels(self):
        """Catches private gold that is not exactly aligned to tasks."""
        source = self.make_source()
        source.labels.pop()
        source.write_labels()
        self.assert_rejected(source)

        source = self.make_source()
        source.labels.append(
            {"instance_id": "extra", "answer": "private-extra", "evidence": ["b99"]}
        )
        source.write_labels()
        self.assert_rejected(source)

    def test_rejects_overlapping_train_or_validation_ids(self):
        """Catches held-out IDs that collide with an existing public split."""
        source = self.make_source()
        self.assert_rejected(source, train_ids={"synthetic-1"})
        self.assert_rejected(source, val_ids={"synthetic-2"})

    def test_rejects_public_fields_in_tasks_and_nonexact_label_schema(self):
        """Catches answer/evidence leakage and label schema drift."""
        source = self.make_source()
        source.tasks[0]["answer"] = "private-synthetic-answer-1"
        source.write_tasks()
        self.assert_rejected(source)

        source = self.make_source()
        source.labels[0]["source_mapping"] = "private"
        source.write_labels()
        self.assert_rejected(source)

    def test_rejects_empty_or_non_block_evidence(self):
        """Catches empty evidence and evidence values that cannot name a block."""
        source = self.make_source()
        source.labels[0]["evidence"] = []
        source.write_labels()
        self.assert_rejected(source)

        source = self.make_source()
        source.labels[0]["evidence"] = ["not a block"]
        source.write_labels()
        self.assert_rejected(source)

    def test_rejects_an_evidence_set_larger_than_the_bounded_worker_protocol(self):
        """Catches unbounded page-worker result allocation from private labels."""
        source = self.make_source()
        source.labels[0]["evidence"] = [
            f"b{number}" for number in range(release_module.MAX_EVIDENCE_IDS_PER_TASK + 1)
        ]
        source.write_labels()
        marker = source.root / "render-called"

        with temporary_module_values(fitz=_RendererMustNotRun(marker)):
            self.assert_rejected(source)

        self.assertFalse(marker.exists())

    def test_rejects_unreadable_pdf_or_missing_visible_evidence_block(self):
        """Catches corrupt PDFs and evidence identifiers absent from visible text."""
        source = self.make_source()
        (source.root / "documents" / "synthetic-1.pdf").write_bytes(b"not a PDF")
        self.assert_rejected(source)

        source = self.make_source()
        source.labels[0]["evidence"] = ["b99"]
        source.write_labels()
        self.assert_rejected(source)

    def test_accepts_an_image_only_pdf_with_a_readable_evidence_header(self):
        """Catches depending on extraction text instead of rendered page pixels."""
        source = self.make_source()
        pdf_path = source.root / "documents" / "synthetic-1.pdf"
        replace_with_image_only_pdf(pdf_path, "b01: Raster evidence header")
        with release_module.fitz.open(pdf_path) as document:
            self.assertEqual(document[0].get_text().strip(), "")

        try:
            validated = validate_source(source.root, (), ())
        except ValidationError as exc:
            self.fail(f"Image-only readable evidence was rejected: {type(exc).__name__}")

        self.assertEqual(validated.ids, ("synthetic-1", "synthetic-2", "synthetic-3"))

    def test_rejects_a_clipped_two_pixel_evidence_fragment(self):
        """Catches treating a rendered fragment as a readable evidence header."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01: clipped header",
            font_size=24,
            x=72,
            y=720,
            underlay_content="72 719 0.48 0.48 re W n ",
        )
        self.assert_rejected(source)

    def test_rejects_evidence_text_hidden_by_a_nonrendering_pdf_mode(self):
        """Catches extraction-only checks that mistake invisible text for rendered evidence."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01: Synthetic fixture hidden block",
            rendering_mode=3,
        )
        self.assert_rejected(source)

    def test_rejects_evidence_text_rendered_outside_the_page(self):
        """Catches a visible-mode trace whose evidence bbox never reaches page pixels."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01: Synthetic fixture off-page block",
            y=900,
        )
        self.assert_rejected(source)

    def test_rejects_evidence_suffix_clipped_after_a_visible_trace_prefix(self):
        """Catches whole-trace ink being used to validate an off-page evidence suffix."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "visible-prefix-b01:",
            x=540,
        )
        self.assert_rejected(source)

    def test_rejects_unrelated_ink_under_white_evidence_characters(self):
        """Catches unrelated black pixels being mistaken for white evidence glyphs."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01:",
            x=500,
            fill_color=(1, 1, 1),
            underlay_content="0 0 0 rg 495 728 35 2 re f ",
        )
        self.assert_rejected(source)

    def test_rejects_unrelated_ink_under_clipped_mid_gray_evidence_characters(self):
        """Catches loose color matching that credits ink from a clipped gray ID."""
        underlays = (
            ("black", 0, 728, 2),
            ("near-gray-antialias", 0.499, 727.5, 0.3),
        )
        for name, underlay_color, underlay_y, underlay_height in underlays:
            with self.subTest(underlay=name):
                source = self.make_source()
                write_pdf_with_visible_text(
                    source.root / "documents" / "synthetic-1.pdf",
                    "b01:",
                    x=500,
                    fill_color=(0.5, 0.5, 0.5),
                    underlay_content=(
                        f"{underlay_color} {underlay_color} {underlay_color} rg "
                        f"495 {underlay_y} 35 {underlay_height} re f "
                        "0 0 1 1 re W n "
                    ),
                )
                self.assert_rejected(source)

    def test_rejects_same_color_underlay_beneath_clipped_evidence_characters(self):
        """Catches crediting same-color paint from an unrelated draw operation."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01:",
            x=500,
            fill_color=(0.5, 0.5, 0.5),
            underlay_content=(
                "0.5 0.5 0.5 rg 495 728 35 2 re f "
                "0 0 1 1 re W n "
            ),
        )
        self.assert_rejected(source)

    def test_accepts_visible_antialiased_mid_gray_evidence_at_small_text_size(self):
        """Catches strict raster proof that rejects a normal small gray glyph core."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01:",
            font_size=8,
            fill_color=(0.5, 0.5, 0.5),
        )

        validated = validate_source(source.root, (), ())

        self.assertEqual(validated.ids, ("synthetic-1", "synthetic-2", "synthetic-3"))

    def test_accepts_a_benign_resource_name_that_matches_an_active_key(self):
        """Catches mistaking an ordinary resource name for associated-file metadata."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01: ordinary resource name",
            font_name="AF",
        )

        validated = validate_source(source.root, (), ())

        self.assertEqual(validated.ids, ("synthetic-1", "synthetic-2", "synthetic-3"))

    def test_fails_closed_when_the_pdf_renderer_is_unavailable(self):
        """Catches falling back to extraction-only validation without a renderer."""
        source = self.make_source()
        had_renderer = hasattr(release_module, "fitz")
        original_renderer = getattr(release_module, "fitz", None)
        release_module.fitz = None
        try:
            self.assert_rejected(source)
        finally:
            if had_renderer:
                release_module.fitz = original_renderer
            else:
                delattr(release_module, "fitz")

    def test_fails_closed_when_tesseract_is_missing_or_the_wrong_version(self):
        """Catches an implicit OCR fallback or an untested backend runtime."""
        source = self.make_source()
        with temporary_module_values(TESSERACT_BINARY="/definitely/not/tesseract"):
            self.assert_rejected(source)

        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        wrong_version = Path(tempdir.name) / "tesseract-wrong-version"
        write_fake_tesseract(wrong_version, version="5.4.0")
        with temporary_module_values(
            TESSERACT_BINARY=str(wrong_version),
            TESSERACT_BINARY_SHA256=hashlib.sha256(wrong_version.read_bytes()).hexdigest(),
        ):
            self.assert_rejected(source)

    def test_ignores_a_path_poisoned_tesseract_that_forges_version_and_headers(self):
        """Catches selecting an attacker-controlled executable from caller PATH."""
        source = self.make_source()
        source.labels[0]["evidence"] = ["b99"]
        source.write_labels()
        fake_root = source.root / "fake-path"
        fake_root.mkdir()
        marker = fake_root / "fake-was-executed"
        fake_binary = fake_root / "tesseract"
        write_fake_tesseract(
            fake_binary,
            ocr_body=(
                f"printf '%s' 'executed' > '{marker}'; "
                "printf '%s\\n' 'b99: forged header' > \"$2.txt\""
            ),
        )

        original_path = os.environ.get("PATH")
        os.environ["PATH"] = (
            str(fake_root) if original_path is None else f"{fake_root}{os.pathsep}{original_path}"
        )
        try:
            self.assert_rejected(source)
        finally:
            if original_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = original_path

        self.assertFalse(marker.exists())

    def test_rejects_missing_or_wrong_traineddata_before_pdf_rendering(self):
        """Catches rendering before the pinned English OCR model is verified."""
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        fake_binary = root / "tesseract"
        write_fake_tesseract(fake_binary)

        missing_root = root / "missing-tessdata"
        missing_render_marker = root / "missing-render-called"
        source = self.make_source()
        with temporary_module_values(
            TESSERACT_BINARY=str(fake_binary),
            TESSERACT_BINARY_SHA256=hashlib.sha256(fake_binary.read_bytes()).hexdigest(),
            TESSDATA_ROOT=str(missing_root),
            fitz=_RendererMustNotRun(missing_render_marker),
        ):
            self.assert_rejected(source)
        self.assertFalse(missing_render_marker.exists())

        wrong_root = root / "wrong-tessdata"
        wrong_root.mkdir()
        (wrong_root / "eng.traineddata").write_bytes(b"unapproved-traineddata")
        wrong_render_marker = root / "wrong-render-called"
        source = self.make_source()
        with temporary_module_values(
            TESSERACT_BINARY=str(fake_binary),
            TESSERACT_BINARY_SHA256=hashlib.sha256(fake_binary.read_bytes()).hexdigest(),
            TESSDATA_ROOT=str(wrong_root),
            fitz=_RendererMustNotRun(wrong_render_marker),
        ):
            self.assert_rejected(source)
        self.assertFalse(wrong_render_marker.exists())

    def test_rejects_a_traineddata_fifo_without_blocking(self):
        """Catches a blocking model-file open before its regular-file check."""
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        fifo_path = Path(tempdir.name) / "eng.traineddata"
        os.mkfifo(fifo_path, mode=0o600)
        result_read_fd, result_write_fd = os.pipe()
        started = time.monotonic()
        pid = os.fork()
        if pid == 0:
            os.close(result_read_fd)
            try:
                release_module._read_pinned_traineddata(fifo_path)
            except ValidationError:
                os.write(result_write_fd, b"rejected")
                os._exit(0)
            except BaseException:
                os.write(result_write_fd, b"error")
                os._exit(2)
            os.write(result_write_fd, b"accepted")
            os._exit(3)

        os.close(result_write_fd)
        try:
            readable, _, _ = select.select([result_read_fd], [], [], 0.25)
            if not readable:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                self.fail("traineddata FIFO blocked validation")
            result = os.read(result_read_fd, 32)
            _, status = os.waitpid(pid, 0)
        finally:
            os.close(result_read_fd)

        self.assertEqual(result, b"rejected")
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(os.WEXITSTATUS(status), 0)
        self.assertLess(time.monotonic() - started, 0.3)

    def test_accepts_a_custom_root_only_with_the_exact_traineddata_digest(self):
        """Catches trusting a model by pathname instead of its pinned bytes."""
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        fake_binary = root / "tesseract"
        write_fake_tesseract(
            fake_binary,
            ocr_body=(
                "[ \"$3\" = '--tessdata-dir' ] || exit 11; "
                "[ \"$4\" = \"$TESSDATA_PREFIX\" ] || exit 12; "
                "[ -f \"$TESSDATA_PREFIX/eng.traineddata\" ] || exit 13; "
                "printf '%s\\n' 'b01: visible' 'b02: visible' 'b03: visible' "
                "> \"$2.txt\""
            ),
        )
        tessdata_root = root / "custom-tessdata"
        tessdata_root.mkdir()
        traineddata = b"synthetic-pinned-English-model"
        (tessdata_root / "eng.traineddata").write_bytes(traineddata)
        expected_digest = hashlib.sha256(traineddata).hexdigest()
        source = self.make_source()

        with temporary_module_values(
            TESSERACT_BINARY=str(fake_binary),
            TESSERACT_BINARY_SHA256=hashlib.sha256(fake_binary.read_bytes()).hexdigest(),
            TESSDATA_ROOT=str(tessdata_root),
            TESSERACT_TRAINEDDATA_SHA256=expected_digest,
        ):
            validated = validate_source(source.root, (), ())
            manifest = build_release_manifest(validated, "synthetic-release-v1")

        self.assertEqual(validated.ids, ("synthetic-1", "synthetic-2", "synthetic-3"))
        self.assertEqual(
            manifest["visibility_audit"].get("traineddata_sha256"),
            expected_digest,
        )
        self.assertNotIn(str(tessdata_root), json.dumps(manifest, sort_keys=True))

    def test_kills_a_slow_page_renderer_within_the_worker_wall_limit(self):
        """Catches unbounded rendering or PNG encoding in the parent process."""
        source = self.make_source()
        completion_marker = source.root / "slow-renderer-completed"
        slow_renderer = _SlowPixmapRenderer(
            release_module.fitz,
            0.4,
            completion_marker,
        )

        with release_module._resolve_ocr_runtime() as runtime:
            started = time.monotonic()
            with temporary_module_values(
                fitz=slow_renderer,
                PAGE_WORKFLOW_TIMEOUT_SECONDS=0.05,
            ), self.assertRaises(ValidationError):
                release_module._render_visible_pdf_evidence_blocks(
                    source.root / "documents" / "synthetic-1.pdf",
                    runtime,
                    {"b01"},
                )
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.3)
        time.sleep(0.45)
        self.assertFalse(completion_marker.exists())

    def test_kills_a_slow_pdf_open_within_the_worker_wall_limit(self):
        """Catches a PyMuPDF open call running outside a killable boundary."""
        source = self.make_source()
        completion_marker = source.root / "slow-open-completed"
        slow_renderer = _SlowOpenRenderer(release_module.fitz, 0.4, completion_marker)

        with release_module._resolve_ocr_runtime() as runtime:
            started = time.monotonic()
            with temporary_module_values(
                fitz=slow_renderer,
                PAGE_WORKFLOW_TIMEOUT_SECONDS=0.05,
            ), self.assertRaises(ValidationError):
                release_module._render_visible_pdf_evidence_blocks(
                    source.root / "documents" / "synthetic-1.pdf",
                    runtime,
                    {"b01"},
                )
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.3)
        time.sleep(0.45)
        self.assertFalse(completion_marker.exists())

    def test_kills_a_slow_page_count_probe_within_the_worker_wall_limit(self):
        """Catches a PyMuPDF page-count read running in the parent process."""
        source = self.make_source()
        completion_marker = source.root / "slow-page-count-completed"
        slow_renderer = _SlowPageCountRenderer(release_module.fitz, 0.4, completion_marker)

        with release_module._resolve_ocr_runtime() as runtime:
            started = time.monotonic()
            with temporary_module_values(
                fitz=slow_renderer,
                PAGE_WORKFLOW_TIMEOUT_SECONDS=0.05,
            ), self.assertRaises(ValidationError):
                release_module._render_visible_pdf_evidence_blocks(
                    source.root / "documents" / "synthetic-1.pdf",
                    runtime,
                    {"b01"},
                )
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.3)
        time.sleep(0.45)
        self.assertFalse(completion_marker.exists())

    def test_rejects_pdf_and_raster_resource_excess(self):
        """Catches unbounded file, page, geometry, and raster allocations."""
        cases = []

        source = self.make_source()
        cases.append((source, {"MAX_PDF_BYTES": 64}))

        source = self.make_source()
        duplicate_pdf_page(source.root / "documents" / "synthetic-1.pdf")
        cases.append((source, {"MAX_PAGES": 1}))

        source = self.make_source()
        cases.append((source, {"MAX_PAGE_WIDTH_POINTS": 100}))

        source = self.make_source()
        cases.append((source, {"MAX_RENDER_PIXELS_PER_PAGE": 1000}))

        source = self.make_source()
        cases.append((source, {"MAX_RENDER_PIXELS_TOTAL": 1000}))

        source = self.make_source()
        cases.append((source, {"MAX_RASTER_BYTES": 100}))

        for number, (bounded_source, settings) in enumerate(cases, start=1):
            with self.subTest(resource_case=number), temporary_module_values(**settings):
                self.assert_rejected(bounded_source)

    def test_rejects_ocr_timeout_malformed_and_oversized_output(self):
        """Catches trusting a stalled or structurally unsafe OCR subprocess."""
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        fake_root = Path(tempdir.name)
        cases = (
            (
                "timeout",
                "sleep 1",
                {"OCR_TIMEOUT_SECONDS": 0.05},
            ),
            (
                "malformed",
                "printf '\\377' > \"$2.txt\"",
                {},
            ),
            (
                "oversized",
                "printf '%s\\n' 'b01: output-over-limit' > \"$2.txt\"",
                {"MAX_OCR_OUTPUT_BYTES": 8},
            ),
            (
                "not-line-anchored",
                "printf '%s\\n' 'prefix b01: not a header' > \"$2.txt\"",
                {},
            ),
        )
        for name, body, settings in cases:
            with self.subTest(ocr_case=name):
                source = self.make_source()
                fake_binary = fake_root / f"tesseract-{name}"
                write_fake_tesseract(fake_binary, ocr_body=body)
                with temporary_module_values(
                    TESSERACT_BINARY=str(fake_binary),
                    TESSERACT_BINARY_SHA256=hashlib.sha256(fake_binary.read_bytes()).hexdigest(),
                    **settings,
                ):
                    self.assert_rejected(source)

    def test_validation_never_emits_ocr_or_private_label_text(self):
        """Catches subprocess output or private values escaping to visible logs."""
        source = self.make_source()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            validate_source(source.root, (), ())

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn("private-synthetic-answer", stdout.getvalue() + stderr.getvalue())

    def test_rejects_a_zip_instead_of_an_explicit_source_directory(self):
        """Catches treating an archive as an implicitly selected release source."""
        source = self.make_source()
        archive = source.root.parent / "official-source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.write(source.root / "tasks.jsonl", "tasks.jsonl")
        with self.assertRaises(ValidationError):
            validate_source(archive, (), ())

    def test_rejects_compressed_embedded_files_and_active_pdf_actions(self):
        """Catches hidden attachments and executable PDF structures at validation."""
        cases = (
            ("json", "private-labels.json", b'{"answer":"organizer-only"}'),
            ("zip", "private-labels.zip", b"PK\x03\x04compressed-private-payload"),
        )
        for name, attachment_name, payload in cases:
            with self.subTest(attachment=name):
                source = self.make_source()
                add_embedded_file(
                    source.root / "documents/synthetic-1.pdf",
                    attachment_name,
                    payload,
                )
                self.assert_rejected(source)

        source = self.make_source()
        add_catalog_javascript(source.root / "documents/synthetic-1.pdf")
        self.assert_rejected(source)

    def test_staging_rejects_mutation_after_validation(self):
        """Catches staging rows or PDF bytes different from the validated snapshot."""
        mutations = (
            (
                "nested-label-row",
                lambda validated: validated.label_rows[0]["evidence"].__setitem__(0, "b99"),
            ),
            (
                "task-row",
                lambda validated: validated.task_rows[0].__setitem__(
                    "user_query", "Changed after validation"
                ),
            ),
            (
                "pdf-bytes",
                lambda validated: write_pdf_with_visible_text(
                    validated.pdf_paths[0],
                    "b01: Valid-looking but changed after validation",
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(mutation=name):
                source = self.make_source()
                validated = validate_source(source.root, (), ())
                mutate(validated)
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                root = Path(temporary.name)
                with self.assertRaises(ValidationError):
                    release_module.stage_release(
                        validated,
                        root / "public",
                        root / "private",
                        "synthetic-release-v1",
                    )

    def test_stages_short_grounded_answer_and_evidence_strings(self):
        """Catches treating ordinary answer/evidence substrings as metadata leaks."""
        source = self.make_source()
        source.tasks[0]["user_query"] = "Does b01 support answer 1?"
        source.labels[0]["answer"] = "1"
        source.write_tasks()
        source.write_labels()
        validated = validate_source(source.root, (), ())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        manifest = release_module.stage_release(
            validated,
            root / "public",
            root / "private",
            "synthetic-release-v1",
        )

        self.assertEqual(manifest["counts"]["tasks"], 3)
        release_module.audit_public_payload(root / "public")

    def test_second_temporary_directory_failure_is_clean_and_normalized(self):
        """Catches a leaked public temp tree when private temp allocation fails."""
        source = self.make_source()
        validated = validate_source(source.root, (), ())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        real_mkdtemp = tempfile.mkdtemp
        created = []

        def fail_second_mkdtemp(*args, **kwargs):
            if not created:
                result = real_mkdtemp(*args, **kwargs)
                created.append(Path(result))
                return result
            raise OSError("synthetic private temp allocation failure")

        with patch.object(
            release_module.tempfile,
            "mkdtemp",
            side_effect=fail_second_mkdtemp,
        ), self.assertRaises(ValidationError):
            release_module.stage_release(
                validated,
                root / "public",
                root / "private",
                "synthetic-release-v1",
            )

        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists())
        self.assertFalse((root / "public").exists())
        self.assertFalse((root / "private").exists())

    def test_stages_exact_deterministic_public_and_private_payloads(self):
        """Catches nondeterminism, private metadata leakage, and altered PDFs."""
        source = self.make_source()
        validated = validate_source(source.root, (), ())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        manifests = []
        staged_trees = []
        for number in (1, 2):
            public_root = root / f"public-{number}"
            private_root = root / f"private-{number}"
            manifests.append(
                release_module.stage_release(
                    validated,
                    public_root,
                    private_root,
                    "synthetic-release-v1",
                )
            )
            public_files = {
                path.relative_to(public_root).as_posix()
                for path in public_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                public_files,
                {
                    "test/SHA256SUMS",
                    "test/release.json",
                    "test/tasks.jsonl",
                    "test/documents/synthetic-1.pdf",
                    "test/documents/synthetic-2.pdf",
                    "test/documents/synthetic-3.pdf",
                },
            )
            self.assertEqual(
                {
                    path.relative_to(private_root).as_posix()
                    for path in private_root.rglob("*")
                    if path.is_file()
                },
                {"private/test_labels.jsonl", "private/test_release.json"},
            )

            task_bytes = (public_root / "test/tasks.jsonl").read_bytes()
            task_rows = [json.loads(line) for line in task_bytes.splitlines()]
            self.assertEqual(
                task_rows,
                [
                    {
                        "document_pdf": f"test/documents/synthetic-{item}.pdf",
                        "instance_id": f"synthetic-{item}",
                        "user_query": f"Synthetic query {item}",
                    }
                    for item in range(1, 4)
                ],
            )
            for pdf in validated.pdf_paths:
                self.assertEqual(
                    (public_root / "test/documents" / pdf.name).read_bytes(),
                    pdf.read_bytes(),
                )

            public_manifest = json.loads((public_root / "test/release.json").read_text())
            serialized_public = json.dumps(public_manifest, sort_keys=True)
            self.assertNotIn("private", serialized_public.casefold())
            self.assertNotIn("answer", serialized_public.casefold())
            self.assertNotIn("evidence", serialized_public.casefold())
            self.assertNotIn("gold", serialized_public.casefold())
            self.assertEqual(
                public_manifest["task_manifest_sha256"],
                hashlib.sha256(task_bytes).hexdigest(),
            )

            private_files = (
                private_root / "private/test_labels.jsonl",
                private_root / "private/test_release.json",
            )
            self.assertTrue(
                all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in private_files)
            )
            self.assertTrue(
                all(
                    stat.S_IMODE(path.stat().st_mode) == 0o700
                    for path in (private_root, private_root / "private")
                )
            )
            self.assertEqual(
                (private_root / "private/test_labels.jsonl").read_bytes(),
                release_module._canonical_rows(validated.label_rows),
            )
            private_manifest = json.loads(
                (private_root / "private/test_release.json").read_text()
            )
            self.assertEqual(private_manifest, manifests[-1])
            self.assertEqual(private_manifest["enabled"], False)
            self.assertEqual(private_manifest["finalized"], False)
            self.assertEqual(private_manifest["max_attempts"], 3)
            self.assertEqual(private_manifest["feedback_policy"], "first-attempt-only")

            release_module.audit_public_payload(public_root)
            staged_trees.append(
                {
                    path.relative_to(public_root).as_posix(): path.read_bytes()
                    for path in public_root.rglob("*")
                    if path.is_file()
                }
            )

        self.assertEqual(manifests[0], manifests[1])
        self.assertEqual(staged_trees[0], staged_trees[1])

    def test_public_payload_audit_has_positive_leakage_controls(self):
        """Catches forbidden paths, fields, values, archives, and unsafe nodes."""
        source = self.make_source()
        validated = validate_source(source.root, (), ())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        good_public = root / "good-public"
        release_module.stage_release(
            validated,
            good_public,
            root / "good-private",
            "synthetic-release-v1",
        )

        def copied(name):
            target = root / name
            shutil.copytree(good_public, target)
            return target

        mutations = []

        public = copied("label-filename")
        (public / "test/labels.jsonl").write_text("{}\n", encoding="utf-8")
        mutations.append(("label filename", public, {}))

        public = copied("answer-field")
        rows = [json.loads(line) for line in (public / "test/tasks.jsonl").read_text().splitlines()]
        rows[0]["answer"] = "leaked"
        write_jsonl(public / "test/tasks.jsonl", rows)
        mutations.append(("answer field", public, {}))

        public = copied("evidence-field")
        manifest = json.loads((public / "test/release.json").read_text())
        manifest["evidence"] = ["b01"]
        (public / "test/release.json").write_text(json.dumps(manifest), encoding="utf-8")
        mutations.append(("evidence field", public, {}))

        for name in ("private", "source_mapping", "organizer_notes"):
            public = copied(name)
            (public / "test" / f"{name}.json").write_text("{}", encoding="utf-8")
            mutations.append((name, public, {}))

        public = copied("archive")
        with zipfile.ZipFile(public / "test/payload.zip", "w") as archive:
            archive.writestr("labels.jsonl", "{}\n")
        mutations.append(("archive", public, {}))

        public = copied("symlink")
        (public / "test/alias.pdf").symlink_to("documents/synthetic-1.pdf")
        mutations.append(("symlink", public, {}))

        public = copied("special")
        os.mkfifo(public / "test/unexpected.pipe", mode=0o600)
        mutations.append(("special file", public, {}))

        public = copied("unexpected-directory")
        (public / "test/extra").mkdir()
        mutations.append(("unexpected directory", public, {}))

        public = copied("unsafe-task-path")
        rows = [json.loads(line) for line in (public / "test/tasks.jsonl").read_text().splitlines()]
        rows[0]["document_pdf"] = "test/documents/../synthetic-1.pdf"
        write_jsonl(public / "test/tasks.jsonl", rows)
        mutations.append(("unsafe task path", public, {}))

        public = copied("embedded-content")
        pdf = public / "test/documents/synthetic-1.pdf"
        pdf.write_bytes(pdf.read_bytes() + b'\n{"answer":"leaked"}\n')
        mutations.append(("embedded label content", public, {}))

        public = copied("private-value")
        rows = [json.loads(line) for line in (public / "test/tasks.jsonl").read_text().splitlines()]
        rows[0]["user_query"] = "private-synthetic-answer-1"
        write_jsonl(public / "test/tasks.jsonl", rows)
        mutations.append(
            (
                "private value",
                public,
                {"forbidden_values": ("private-synthetic-answer-1",)},
            )
        )

        for name, public, keyword_arguments in mutations:
            with self.subTest(leakage_case=name), self.assertRaises(ValidationError):
                release_module.audit_public_payload(public, **keyword_arguments)

    def test_public_audit_rejects_embedded_files_after_hashes_are_reconciled(self):
        """Catches structural attachments even when every public digest matches."""
        source = self.make_source()
        validated = validate_source(source.root, (), ())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        public = root / "public"
        release_module.stage_release(
            validated,
            public,
            root / "private",
            "synthetic-release-v1",
        )

        for name, attachment_name, payload in (
            ("json", "private-labels.json", b'{"answer":"organizer-only"}'),
            ("zip", "private-labels.zip", b"PK\x03\x04compressed-private-payload"),
        ):
            with self.subTest(attachment=name):
                mutated = root / f"public-{name}"
                shutil.copytree(public, mutated)
                add_embedded_file(
                    mutated / "test/documents/synthetic-1.pdf",
                    attachment_name,
                    payload,
                )
                refresh_public_release_hashes(mutated)
                with self.assertRaises(ValidationError):
                    release_module.audit_public_payload(mutated)

    def test_public_manifest_rejects_nonexact_types_keys_and_digests(self):
        """Catches bool-as-int and malformed nested or digest manifest values."""
        source = self.make_source()
        validated = validate_source(source.root, (), ())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        good_public = root / "good-public"
        release_module.stage_release(
            validated,
            good_public,
            root / "private",
            "synthetic-release-v1",
        )

        mutations = (
            ("bool schema version", lambda manifest: manifest.__setitem__("schema_version", True)),
            ("nested count key", lambda manifest: manifest["counts"].__setitem__("labels", 3)),
            (
                "uppercase digest",
                lambda manifest: manifest.__setitem__(
                    "sorted_ids_sha256", manifest["sorted_ids_sha256"].upper()
                ),
            ),
            (
                "non-string digest",
                lambda manifest: manifest.__setitem__("task_manifest_sha256", 7),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(manifest_case=name):
                public = root / f"manifest-{name.replace(' ', '-')}"
                shutil.copytree(good_public, public)
                manifest_path = public / "test/release.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_bytes(release_module._canonical_json_document(manifest))
                refresh_public_checksums(public)
                with self.assertRaises(ValidationError):
                    release_module.audit_public_payload(public)

    def test_staging_is_quiet_and_does_not_modify_existing_public_splits(self):
        """Catches staging that logs private rows or rewrites train/validation."""
        source = self.make_source()
        validated = validate_source(source.root, (), ())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        existing = root / "dataset"
        for split in ("train", "validation"):
            (existing / split / "documents").mkdir(parents=True)
            (existing / split / "tasks.jsonl").write_text(
                f'{{"instance_id":"{split}-1"}}\n', encoding="utf-8"
            )
            (existing / split / "documents" / f"{split}-1.pdf").write_bytes(
                f"unchanged-{split}".encode("ascii")
            )

        def snapshot(path):
            return {
                item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
                for item in path.rglob("*")
                if item.is_file()
            }

        before = snapshot(existing)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            release_module.stage_release(
                validated,
                root / "staging/public",
                root / "staging/private",
                "synthetic-release-v1",
            )
        after = snapshot(existing)

        self.assertEqual(after, before)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn("private-synthetic-answer", stdout.getvalue() + stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
