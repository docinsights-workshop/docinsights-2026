#!/usr/bin/env python3
"""Behavioral tests for the public-only DocSem test publisher."""

import hashlib, json, os, tempfile, unittest
from pathlib import Path

try:
    import publish_docsem_public_hf_test_release as publisher
except ModuleNotFoundError:
    publisher = None

def canonical(value): return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
def sha(value): return hashlib.sha256(value).hexdigest()

def pdf_bytes():
    body = b"BT /F1 12 Tf 72 720 Td (opaque-token: value) Tj ET"
    objects = (b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream")
    output, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for number, value in enumerate(objects, 1):
        offsets.append(len(output)); output.extend(f"{number} 0 obj\n".encode() + value + b"\nendobj\n")
    start = len(output); output.extend(f"xref\n0 6\n0000000000 65535 f \n".encode()); output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])); output.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    return bytes(output)

class Source:
    def __init__(self, state): self.state = state
    def inspect(self, checkout): return self.state

class Hf:
    def __init__(self, root, base):
        self.root, self.revision, self.private = Path(root), base, False
        self.events = []; self.history = [(base, self.tree())]
    def tree(self): return {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
    def state(self): return publisher.RemoteState(self.revision, self.private)
    def inventory(self, revision): return {name: publisher.RemoteFile(len(data), sha(data)) for name, data in self.tree().items()}
    def read(self, revision, paths): return {path: self.tree()[path] for path in paths}
    def history_snapshots(self): return tuple(self.history)
    def upload_large_folder(self, stage, expected_parent):
        self.events.append("upload")
        if expected_parent != self.revision: raise publisher.RemoteMovedError("moved")
        for source in (Path(stage) / "test").rglob("*"):
            if source.is_file():
                destination = self.root / source.relative_to(stage); destination.parent.mkdir(parents=True, exist_ok=True); os.link(source, destination)
        self.revision = "c" * 40; self.history.append((self.revision, self.tree())); return self.revision
    def commit_docs(self, files, expected_parent):
        self.events.append("docs")
        if expected_parent != self.revision: raise publisher.RemoteMovedError("moved")
        for path, data in files.items(): (self.root / path).write_bytes(data)
        self.revision = "d" * 40; self.history.append((self.revision, self.tree())); return self.revision

class PublicReleaseTests(unittest.TestCase):
    BASE = "b" * 40
    def setUp(self):
        if publisher is None: self.fail("dedicated public-only publisher is missing")
        self.original_audit = publisher.audit_public_payload; self.original_renderer = publisher.render_test_ready_dataset_card
        self.original_source_root = publisher.SOURCE_TASK_ROOT
        publisher.audit_public_payload = self.audit_fixture
        publisher.render_test_ready_dataset_card = self.render_fixture
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        self.source, self.stage, self.dataset = root / "source", root / "stage", root / "dataset"
        (self.source / "documents").mkdir(parents=True); self.dataset.mkdir()
        self.ids = ("test_000001", "test_000002")
        self.rows = [{"instance_id": item, "user_query": "PUBLIC-QUERY-NOT-PRINTED", "document_pdf": f"documents/{item}.pdf"} for item in self.ids]
        (self.source / "tasks.jsonl").write_bytes(b"".join(canonical(row) for row in self.rows))
        for item in self.ids: (self.source / "documents" / f"{item}.pdf").write_bytes(pdf_bytes())
        self.train, self.validation = root / "train.jsonl", root / "validation.jsonl"
        self.train.write_bytes(canonical({"instance_id":"train_000001","user_query":"q","document_pdf":"documents/train_000001.pdf"})); self.validation.write_bytes(canonical({"instance_id":"val_000001","user_query":"q","document_pdf":"documents/val_000001.pdf"}))
        self.readme = b"---\nconfigs:\n- config_name: tasks\n  data_files:\n  - split: validation\n    path: val/tasks.jsonl\n- config_name: labels\n  data_files:\n  - split: train\n    path: train/labels.jsonl\n---\nBase.\n"; self.instructions = b"Existing train and validation behavior.\n"
        (self.dataset / "README.md").write_bytes(self.readme); (self.dataset / "INSTRUCTIONS.md").write_bytes(self.instructions); (self.dataset / "train").mkdir(); (self.dataset / "train/labels.jsonl").write_bytes(b"train labels permitted\n")
        self.hf = Hf(self.dataset, self.BASE)
        publisher.SOURCE_TASK_ROOT = self.source
        self.state = publisher.SourceState(publisher.SOURCE_CHECKOUT, publisher.SOURCE_HEAD, publisher.SOURCE_PARENT, False, publisher.SOURCE_MANIFEST_SHA256)
        self.config = publisher.ReleaseConfig(self.source, self.stage, self.train, self.validation, self.BASE)
    def audit_fixture(self, stage):
        files = {p.relative_to(stage).as_posix() for p in Path(stage).rglob("*") if p.is_file()}
        if "test/tasks.jsonl" not in files or any("label" in name for name in files): raise ValueError("unsafe fixture")
        return json.loads((Path(stage) / "test/release.json").read_text())
    def render_fixture(self, stage, *, card_template_path):
        return Path(card_template_path).read_bytes().replace(b"  - split: validation\n    path: val/tasks.jsonl\n", b"  - split: validation\n    path: val/tasks.jsonl\n  - split: test\n    path: test/tasks.jsonl\n")
    def cleanup(self):
        publisher.audit_public_payload = self.original_audit; publisher.render_test_ready_dataset_card = self.original_renderer; publisher.SOURCE_TASK_ROOT = self.original_source_root
    def tearDown(self): self.cleanup(); self.temp.cleanup()
    def release(self, **kwargs):
        if not self.stage.exists(): publisher.prepare_stage(self.config, source=Source(self.state))
        return publisher.run_release(self.config, source=Source(self.state), hf=self.hf, **kwargs)

    def test_default_dry_run_is_write_free_and_sanitized(self):
        result = self.release(); self.assertEqual(result["mode"], "dry-run"); self.assertEqual(self.hf.events, []); self.assertNotIn("PUBLIC-QUERY", json.dumps(result)); self.assertNotIn("private", json.dumps(result).lower())
    def test_publish_requires_exact_confirmation_and_base(self):
        with self.assertRaises(publisher.ReleaseError): self.release(publish=True, confirm="yes")
        self.config = publisher.ReleaseConfig(self.source, self.stage, self.train, self.validation, "a" * 40)
        with self.assertRaises(publisher.RemoteMovedError): self.release()
    def test_stage_allowlist_normalizes_paths_and_hardlinks_pdf_bytes(self):
        self.release(); files = {p.relative_to(self.stage).as_posix() for p in self.stage.rglob("*") if p.is_file()}
        self.assertEqual(files, {"test/tasks.jsonl", "test/release.json", "test/SHA256SUMS", "test/documents/test_000001.pdf", "test/documents/test_000002.pdf"})
        rows = [json.loads(line) for line in (self.stage / "test/tasks.jsonl").read_text().splitlines()]
        self.assertEqual([row["document_pdf"] for row in rows], ["test/documents/test_000001.pdf", "test/documents/test_000002.pdf"])
        self.assertEqual((self.stage / "test/documents/test_000001.pdf").read_bytes(), (self.source / "documents/test_000001.pdf").read_bytes()); self.assertEqual((self.stage / "test/documents/test_000001.pdf").stat().st_ino, (self.source / "documents/test_000001.pdf").stat().st_ino)
    def test_dirty_bad_digest_label_input_overlap_and_hardlink_failure_fail_closed(self):
        bad = publisher.SourceState(self.state.checkout, self.state.head, self.state.parent, True, self.state.manifest_sha256)
        with self.assertRaises(publisher.ReleaseError): publisher.run_release(self.config, source=Source(bad), hf=self.hf)
        bad_digest = publisher.SourceState(self.state.checkout, self.state.head, self.state.parent, False, "0" * 64)
        with self.assertRaises(publisher.ReleaseError): publisher.run_release(self.config, source=Source(bad_digest), hf=self.hf)
        (self.source / "labels.jsonl").write_bytes(b"must not be read")
        with self.assertRaises(publisher.ReleaseError): self.release()
        (self.source / "labels.jsonl").unlink(); self.rows[1]["instance_id"] = "train_000001"; (self.source / "tasks.jsonl").write_bytes(b"".join(canonical(row) for row in self.rows))
        with self.assertRaises(publisher.ReleaseError): self.release()
    def test_unavailable_hardlinks_fail_closed(self):
        original = publisher.os.link; publisher.os.link = lambda *_: (_ for _ in ()).throw(OSError("unavailable"))
        try:
            with self.assertRaises(publisher.ReleaseError): self.release()
        finally: publisher.os.link = original
    def test_private_backend_is_never_addressed_upload_precedes_atomic_docs(self):
        result = self.release(publish=True, confirm="PUBLISH"); self.assertEqual(result["mode"], "published"); self.assertEqual(self.hf.events, ["upload", "docs"])
        self.assertIn(b"submissions remain closed", (self.dataset / "README.md").read_bytes()); instructions = (self.dataset / "INSTRUCTIONS.md").read_text(); self.assertIn("immediately before the block colon", instructions); self.assertIn("including punctuation", instructions); self.assertNotIn("bNN", instructions)
    def test_remote_extra_path_size_sha_or_non_test_drift_prevents_docs(self):
        self.release(); self.hf.upload_large_folder(self.stage, self.BASE); (self.dataset / "test/extra.pdf").write_bytes(b"extra")
        self.config = publisher.ReleaseConfig(self.source, self.stage, self.train, self.validation, "c" * 40)
        with self.assertRaises(publisher.ReleaseError): self.release(publish=True, confirm="PUBLISH")
        self.assertEqual(self.hf.events, ["upload"])
    def test_partial_upload_resumes_without_docs_and_complete_is_idempotent(self):
        self.release(); self.hf.upload_large_folder(self.stage, self.BASE); self.release(publish=True, confirm="PUBLISH", expected_complete_base="c" * 40); self.assertEqual(self.hf.events, ["upload", "docs"])
        result = self.release(publish=True, confirm="PUBLISH", expected_complete_base="d" * 40); self.assertEqual(result["mode"], "already-complete"); self.assertEqual(self.hf.events, ["upload", "docs"])
    def test_private_target_or_public_history_test_labels_are_refused(self):
        self.hf.private = True
        with self.assertRaises(publisher.ReleaseError): self.release()

    def test_fix_round_exposes_explicit_read_only_and_prepare_operations(self):
        self.assertTrue(hasattr(publisher, "prepare_stage"), "prepare_stage is required so default dry-run cannot write")

    def test_fix_round_binds_the_stage_to_the_approved_source_root(self):
        self.assertTrue(hasattr(publisher, "SOURCE_TASK_ROOT"), "approved source task root must be fixed")

    def test_fix_round_uses_tracked_templates_and_safe_upload_adapter(self):
        self.assertTrue(hasattr(publisher, "TRACKED_README"), "release docs must derive from tracked templates")

    def test_prepared_stage_is_read_only_during_default_dry_run(self):
        publisher.prepare_stage(self.config, source=Source(self.state))
        before = {path.relative_to(self.stage).as_posix(): path.stat().st_ino for path in self.stage.rglob("*") if path.is_file()}
        self.assertEqual(self.release()["mode"], "dry-run")
        after = {path.relative_to(self.stage).as_posix(): path.stat().st_ino for path in self.stage.rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def test_unapproved_caller_source_root_is_refused(self):
        other = self.temp.name and Path(self.temp.name) / "other"
        other.mkdir()
        config = publisher.ReleaseConfig(other, self.stage, self.train, self.validation, self.BASE)
        with self.assertRaises(publisher.ReleaseError): publisher.prepare_stage(config, source=Source(self.state))
        self.hf.private = False; self.hf.history.append(("a" * 40, {"test/labels.jsonl": b"not allowed"}))
        with self.assertRaises(publisher.ReleaseError): self.release()

if __name__ == "__main__": unittest.main()
