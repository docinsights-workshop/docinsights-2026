"""Fixture-based tests for fail-closed DocSem held-out test source validation."""

import hashlib
import json
import tempfile
import unittest
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
):
    """Write a tiny self-contained PDF whose text extractor sees ``text``."""
    color = "" if fill_color is None else f"{fill_color[0]} {fill_color[1]} {fill_color[2]} rg "
    content = (
        f"{underlay_content}{color}BT /F1 {font_size} Tf {rendering_mode} Tr {x} {y} Td ({text}) Tj ET"
    ).encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
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
                f"Synthetic fixture visible block b{number:02d}",
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

    def test_rejects_unreadable_pdf_or_missing_visible_evidence_block(self):
        """Catches corrupt PDFs and evidence identifiers absent from visible text."""
        source = self.make_source()
        (source.root / "documents" / "synthetic-1.pdf").write_bytes(b"not a PDF")
        self.assert_rejected(source)

        source = self.make_source()
        source.labels[0]["evidence"] = ["b99"]
        source.write_labels()
        self.assert_rejected(source)

    def test_rejects_evidence_text_hidden_by_a_nonrendering_pdf_mode(self):
        """Catches extraction-only checks that mistake invisible text for rendered evidence."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "Synthetic fixture hidden block b01",
            rendering_mode=3,
        )
        self.assert_rejected(source)

    def test_rejects_evidence_text_rendered_outside_the_page(self):
        """Catches a visible-mode trace whose evidence bbox never reaches page pixels."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "Synthetic fixture off-page block b01",
            y=900,
        )
        self.assert_rejected(source)

    def test_rejects_evidence_suffix_clipped_after_a_visible_trace_prefix(self):
        """Catches whole-trace ink being used to validate an off-page evidence suffix."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "visible-prefix-b01",
            x=540,
        )
        self.assert_rejected(source)

    def test_rejects_unrelated_ink_under_white_evidence_characters(self):
        """Catches unrelated black pixels being mistaken for white evidence glyphs."""
        source = self.make_source()
        write_pdf_with_visible_text(
            source.root / "documents" / "synthetic-1.pdf",
            "b01",
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
                    "b01",
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
            "b01",
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
            "b01",
            font_size=8,
            fill_color=(0.5, 0.5, 0.5),
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

    def test_rejects_a_zip_instead_of_an_explicit_source_directory(self):
        """Catches treating an archive as an implicitly selected release source."""
        source = self.make_source()
        archive = source.root.parent / "official-source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.write(source.root / "tasks.jsonl", "tasks.jsonl")
        with self.assertRaises(ValidationError):
            validate_source(archive, (), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
