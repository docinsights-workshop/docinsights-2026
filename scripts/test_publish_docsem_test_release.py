#!/usr/bin/env python3
"""Tests for the guarded DocSem test-release publisher."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import prepare_docsem_test_release as preparer
import publish_docsem_test_release as publisher


def _canonical_json(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_rows(rows):
    return b"".join(_canonical_json(row) for row in rows)


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


class FakeGitBackend:
    def __init__(self, source_state, *, tree=None, event_log=None):
        self.source_state = source_state
        self.revision = source_state.remote_revision
        self.tree = dict(tree or {"docsem/README.md": b"existing source docs\n"})
        self.history = [(self.revision, dict(self.tree))]
        self.writes = []
        self.event_log = event_log if event_log is not None else []
        self.move_before_commit = False
        self.fail_message = None
        self.tamper_after_commit = None

    def inspect_source(self, checkout, repository, branch):
        return self.source_state

    def current_revision(self, repository, branch):
        return self.revision

    def list_paths(self, repository, revision):
        if revision != self.revision:
            raise RuntimeError("unknown revision")
        return tuple(sorted(self.tree))

    def read_files(self, repository, revision, paths):
        if revision != self.revision:
            raise RuntimeError("unknown revision")
        return {path: self.tree[path] for path in paths}

    def publish(self, repository, branch, expected_parent, operations, message):
        self.event_log.append("github")
        if self.fail_message:
            raise RuntimeError(self.fail_message)
        if self.move_before_commit:
            self.revision = "9" * 40
        if self.revision != expected_parent:
            raise publisher.RemoteMovedError("GitHub base moved.")
        captured = {
            path: artifact.read_bytes() for path, artifact in operations.items()
        }
        self.writes.append(
            {
                "repository": repository,
                "branch": branch,
                "expected_parent": expected_parent,
                "paths": tuple(sorted(captured)),
                "files": captured,
                "message": message,
            }
        )
        self.tree.update(captured)
        self.revision = "d" * 40
        if self.tamper_after_commit:
            self.tree[self.tamper_after_commit] = b"tampered\n"
        self.history.append((self.revision, dict(self.tree)))
        return self.revision

    def history_snapshots(self, repository, branch):
        return tuple(
            publisher.HistorySnapshot(
                revision=revision,
                paths=tuple(sorted(tree)),
                metadata={
                    path: payload
                    for path, payload in tree.items()
                    if publisher._split_sensitive_metadata_path(path)
                },
            )
            for revision, tree in self.history
        )


class FakeHfBackend:
    def __init__(self, revisions, trees=None, *, event_log=None, private_flags=None):
        self.revisions = dict(revisions)
        self.private_flags = dict(
            private_flags
            or {
                publisher.PUBLIC_HF_REPOSITORY: False,
                publisher.PRIVATE_HF_REPOSITORY: True,
            }
        )
        self.trees = {key: dict(value) for key, value in (trees or {}).items()}
        for repo in revisions:
            self.trees.setdefault(repo, {})
        self.history = {
            repo: [(revision, dict(self.trees[repo]))]
            for repo, revision in self.revisions.items()
        }
        self.writes = []
        self.event_log = event_log if event_log is not None else []
        self.move_before_commit = set()
        self.flip_visibility_before_commit = set()
        self.fail_message = None
        self.tamper_after_commit = None

    def repository_state(self, repository, token):
        return publisher.HfRepositoryState(
            revision=self.revisions[repository],
            private=self.private_flags[repository],
        )

    def list_paths(self, repository, revision, token):
        if revision != self.revisions[repository]:
            raise RuntimeError("unknown revision")
        return tuple(sorted(self.trees[repository]))

    def read_files(self, repository, revision, paths, token):
        if revision != self.revisions[repository]:
            raise RuntimeError("unknown revision")
        return {path: self.trees[repository][path] for path in paths}

    def publish(
        self,
        repository,
        expected_parent,
        operations,
        message,
        token,
        *,
        expected_private,
    ):
        stage = (
            "private_hf"
            if repository == publisher.PRIVATE_HF_REPOSITORY
            else "public_hf"
        )
        self.event_log.append(stage)
        if self.fail_message:
            raise RuntimeError(self.fail_message)
        if repository in self.flip_visibility_before_commit:
            self.private_flags[repository] = not self.private_flags[repository]
        if self.private_flags[repository] is not expected_private:
            raise publisher.ReleaseError("Hugging Face visibility changed.")
        if repository in self.move_before_commit:
            self.revisions[repository] = "8" * 40
        if self.revisions[repository] != expected_parent:
            raise publisher.RemoteMovedError("Hugging Face base moved.")
        captured = {
            path: artifact.read_bytes() for path, artifact in operations.items()
        }
        self.writes.append(
            {
                "repository": repository,
                "expected_parent": expected_parent,
                "paths": tuple(sorted(captured)),
                "files": captured,
                "message": message,
            }
        )
        self.trees[repository].update(captured)
        new_revision = ("e" if stage == "private_hf" else "f") * 40
        self.revisions[repository] = new_revision
        if self.tamper_after_commit and self.tamper_after_commit[0] == repository:
            self.trees[repository][self.tamper_after_commit[1]] = b"tampered\n"
        self.history[repository].append((new_revision, dict(self.trees[repository])))
        return new_revision

    def history_snapshots(self, repository, token):
        return tuple(
            publisher.HistorySnapshot(
                revision=revision,
                paths=tuple(sorted(tree)),
                metadata={
                    path: payload
                    for path, payload in tree.items()
                    if publisher._split_sensitive_metadata_path(path)
                },
            )
            for revision, tree in self.history[repository]
        )


class PublicationFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.public_root = self.root / "public"
        self.private_root = self.root / "private"
        self.card_template = self.root / "base-README.md"
        self.release_card = self.root / "release-README.md"
        (self.public_root / "test/documents").mkdir(parents=True)
        (self.private_root / "private").mkdir(parents=True)
        self.private_root.chmod(0o700)
        (self.private_root / "private").chmod(0o700)

        self.task_rows = [
            {
                "instance_id": "test_000001",
                "user_query": "Question one?",
                "document_pdf": "test/documents/test_000001.pdf",
            },
            {
                "instance_id": "test_000002",
                "user_query": "Question two?",
                "document_pdf": "test/documents/test_000002.pdf",
            },
        ]
        self.label_rows = [
            {"instance_id": "test_000001", "answer": "17", "evidence": ["b01"]},
            {"instance_id": "test_000002", "answer": "29", "evidence": ["b02"]},
        ]
        self.pdfs = {
            "test_000001.pdf": b"%PDF-1.4\nfirst synthetic fixture\n%%EOF\n",
            "test_000002.pdf": b"%PDF-1.4\nsecond synthetic fixture\n%%EOF\n",
        }
        self.tasks_bytes = _canonical_rows(self.task_rows)
        self.labels_bytes = _canonical_rows(self.label_rows)
        pdf_inventory = b"".join(
            f"{name}  {_digest(payload)}\n".encode("ascii")
            for name, payload in sorted(self.pdfs.items())
        )
        self.release_id = "docsem-test-2026-09-04"
        self.public_manifest = {
            "schema_version": 1,
            "release_id": self.release_id,
            "counts": {"tasks": 2, "pdfs": 2},
            "sorted_ids_sha256": _digest(b"test_000001\ntest_000002\n"),
            "task_manifest_sha256": _digest(self.tasks_bytes),
            "pdf_inventory_sha256": _digest(pdf_inventory),
        }
        self.public_manifest_bytes = _canonical_json(self.public_manifest)
        self.private_manifest = {
            "schema_version": 1,
            "release_id": self.release_id,
            "counts": {"tasks": 2, "pdfs": 2, "labels": 2},
            "sorted_ids_sha256": self.public_manifest["sorted_ids_sha256"],
            "task_manifest_sha256": self.public_manifest["task_manifest_sha256"],
            "gold_sha256": _digest(self.labels_bytes),
            "pdf_inventory_sha256": self.public_manifest["pdf_inventory_sha256"],
            "visibility_audit": {
                **preparer._visibility_audit_contract(),
            },
            "enabled": False,
            "max_attempts": 3,
            "feedback_policy": "first-attempt-only",
            "finalized": False,
        }

        (self.public_root / "test/tasks.jsonl").write_bytes(self.tasks_bytes)
        (self.public_root / "test/release.json").write_bytes(self.public_manifest_bytes)
        for name, payload in self.pdfs.items():
            (self.public_root / "test/documents" / name).write_bytes(payload)
        checksums = {
            "tasks.jsonl": _digest(self.tasks_bytes),
            "release.json": _digest(self.public_manifest_bytes),
            **{
                f"documents/{name}": _digest(payload)
                for name, payload in self.pdfs.items()
            },
        }
        self.checksum_bytes = b"".join(
            f"{digest}  {name}\n".encode("ascii")
            for name, digest in sorted(checksums.items())
        )
        (self.public_root / "test/SHA256SUMS").write_bytes(self.checksum_bytes)

        labels_path = self.private_root / "private/test_labels.jsonl"
        release_path = self.private_root / "private/test_release.json"
        labels_path.write_bytes(self.labels_bytes)
        release_path.write_bytes(_canonical_json(self.private_manifest))
        labels_path.chmod(0o600)
        release_path.chmod(0o600)

        self.base_card_bytes = (
            b"---\nconfigs:\n- config_name: tasks\n  data_files:\n"
            b"  - split: validation\n    path: val/tasks.jsonl\n"
            b"- config_name: labels\n  data_files:\n"
            b"  - split: train\n    path: train/labels.jsonl\n---\nBase card.\n"
        )
        self.release_card_bytes = self.base_card_bytes.replace(
            b"  - split: validation\n    path: val/tasks.jsonl\n",
            b"  - split: validation\n    path: val/tasks.jsonl\n"
            b"  - split: test\n    path: test/tasks.jsonl\n",
        )
        self.card_template.write_bytes(self.base_card_bytes)
        self.release_card.write_bytes(self.release_card_bytes)

    def public_auditor(self, path):
        self.audit_paths = getattr(self, "audit_paths", []) + [Path(path)]
        return self.public_manifest

    def card_renderer(self, staging_root, *, card_template_path=None):
        self.render_args = (Path(staging_root), Path(card_template_path))
        return self.release_card_bytes


class PublishDocsemTestReleaseTests(unittest.TestCase):
    SOURCE_BASE = "a" * 40
    PUBLIC_BASE = "b" * 40
    PRIVATE_BASE = "c" * 40

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = PublicationFixture(self.temporary.name)
        self.event_log = []
        self.source_state = publisher.SourceState(
            checkout=publisher.CANONICAL_SOURCE_CHECKOUT,
            repository=publisher.CANONICAL_GITHUB_REPOSITORY,
            branch="main",
            head=self.SOURCE_BASE,
            remote_revision=self.SOURCE_BASE,
            dirty_tracked=False,
            dirty_untracked=False,
            behind=False,
            diverged=False,
        )
        self.git = FakeGitBackend(self.source_state, event_log=self.event_log)
        self.hf = FakeHfBackend(
            {
                publisher.PUBLIC_HF_REPOSITORY: self.PUBLIC_BASE,
                publisher.PRIVATE_HF_REPOSITORY: self.PRIVATE_BASE,
            },
            trees={
                publisher.PUBLIC_HF_REPOSITORY: {
                    "README.md": self.fixture.base_card_bytes,
                    "train/labels.jsonl": b"public training labels are allowed\n",
                    "val/tasks.jsonl": _canonical_rows(
                        [
                            {
                                "instance_id": "val_000001",
                                "user_query": "Public validation question?",
                                "document_pdf": "val/documents/val_000001.pdf",
                            }
                        ]
                    ),
                },
                publisher.PRIVATE_HF_REPOSITORY: {
                    "private/val_labels.jsonl": b"private validation labels\n"
                },
            },
            event_log=self.event_log,
        )
        self.config = publisher.ReleaseConfig(
            public_stage=self.fixture.public_root,
            private_stage=self.fixture.private_root,
            release_card=self.fixture.release_card,
            card_template=self.fixture.card_template,
            source_checkout=publisher.CANONICAL_SOURCE_CHECKOUT,
            source_repository=publisher.CANONICAL_GITHUB_REPOSITORY,
            source_branch="main",
            source_base=self.SOURCE_BASE,
            github_remote_base=self.SOURCE_BASE,
            public_hf_base=self.PUBLIC_BASE,
            private_hf_base=self.PRIVATE_BASE,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def run_release(self, **kwargs):
        options = {
            "config": self.config,
            "git_backend": self.git,
            "hf_backend": self.hf,
            "public_token": "public-token",
            "private_token": "private-token",
            "public_auditor": self.fixture.public_auditor,
            "card_renderer": self.fixture.card_renderer,
        }
        options.update(kwargs)
        return publisher.run_release(**options)

    def test_default_dry_run_names_exact_targets_operations_hashes_and_recovery(self):
        result = self.run_release()

        self.assertEqual(result["mode"], "dry-run")
        self.assertEqual(
            result["source"],
            {
                "checkout": str(publisher.CANONICAL_SOURCE_CHECKOUT),
                "repository": publisher.CANONICAL_GITHUB_REPOSITORY,
                "branch": "main",
                "base_revision": self.SOURCE_BASE,
            },
        )
        self.assertEqual(
            result["targets"],
            {
                "github": {
                    "repository": publisher.CANONICAL_GITHUB_REPOSITORY,
                    "base_revision": self.SOURCE_BASE,
                },
                "public_hugging_face": {
                    "repository": publisher.PUBLIC_HF_REPOSITORY,
                    "base_revision": self.PUBLIC_BASE,
                },
                "private_hugging_face": {
                    "repository": publisher.PRIVATE_HF_REPOSITORY,
                    "base_revision": self.PRIVATE_BASE,
                },
            },
        )
        public_paths = (
            "test/SHA256SUMS",
            "test/documents/test_000001.pdf",
            "test/documents/test_000002.pdf",
            "test/release.json",
            "test/tasks.jsonl",
        )
        self.assertEqual(result["operations"][0]["target"], "private_hugging_face")
        self.assertEqual(
            tuple(result["operations"][0]["paths"]),
            ("private/test_labels.jsonl", "private/test_release.json"),
        )
        self.assertEqual(
            tuple(result["operations"][1]["paths"]),
            tuple(f"docsem/{path}" for path in public_paths),
        )
        self.assertEqual(
            tuple(result["operations"][2]["paths"]),
            ("README.md",) + public_paths,
        )
        self.assertEqual(result["release"]["release_id"], self.fixture.release_id)
        self.assertEqual(
            result["release"]["counts"], {"tasks": 2, "pdfs": 2, "labels": 2}
        )
        self.assertEqual(
            result["release"]["task_manifest_sha256"], _digest(self.fixture.tasks_bytes)
        )
        self.assertEqual(
            result["release"]["gold_sha256"], _digest(self.fixture.labels_bytes)
        )
        self.assertEqual(
            result["release"]["dataset_card_sha256"],
            _digest(self.fixture.release_card_bytes),
        )
        self.assertIn("private disabled release", result["safe_order"][0])
        self.assertTrue(
            any("regenerate" in item for item in result["partial_failure_recovery"])
        )
        self.assertEqual(self.git.writes, [])
        self.assertEqual(self.hf.writes, [])
        self.assertEqual(
            self.fixture.audit_paths[0], self.fixture.public_root.resolve()
        )
        self.assertEqual(len(self.fixture.audit_paths), 2)
        self.assertNotEqual(self.fixture.audit_paths[1], self.fixture.public_root)

    def test_source_and_remote_state_must_be_exact(self):
        cases = {
            "unspecified source branch": replace(self.config, source_branch=""),
            "wrong source branch": replace(self.config, source_branch="release"),
            "wrong source base": replace(self.config, source_base="0" * 40),
            "unspecified source base": replace(self.config, source_base=""),
            "remote movement": replace(self.config, github_remote_base="7" * 40),
            "public HF movement": replace(self.config, public_hf_base="7" * 40),
            "private HF movement": replace(self.config, private_hf_base="7" * 40),
        }
        for name, config in cases.items():
            with self.subTest(name=name), self.assertRaises(publisher.ReleaseError):
                self.run_release(config=config)

        for name, state in {
            "tracked changes": replace(self.source_state, dirty_tracked=True),
            "untracked changes": replace(self.source_state, dirty_untracked=True),
            "behind": replace(self.source_state, behind=True),
            "diverged": replace(self.source_state, diverged=True),
            "head mismatch": replace(self.source_state, head="1" * 40),
            "remote mismatch": replace(self.source_state, remote_revision="2" * 40),
            "wrong repository": replace(self.source_state, repository="other/repo"),
        }.items():
            with self.subTest(name=name), self.assertRaises(publisher.ReleaseError):
                bad_git = FakeGitBackend(state)
                self.run_release(git_backend=bad_git)

    def test_hugging_face_visibility_is_checked_in_dry_run_and_each_cas(self):
        self.hf.private_flags[publisher.PUBLIC_HF_REPOSITORY] = True
        with self.assertRaisesRegex(publisher.ReleaseError, "visibility"):
            self.run_release()
        self.assertEqual(self.event_log, [])

        self.setUp_fresh_fixture()
        self.hf.private_flags[publisher.PRIVATE_HF_REPOSITORY] = False
        with self.assertRaisesRegex(publisher.ReleaseError, "visibility"):
            self.run_release()
        self.assertEqual(self.event_log, [])

        self.setUp_fresh_fixture()
        self.hf.flip_visibility_before_commit.add(publisher.PRIVATE_HF_REPOSITORY)
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="PUBLISH")
        self.assertEqual(self.event_log, ["private_hf"])
        self.assertEqual(self.hf.writes, [])

    def test_stage_pair_card_and_private_modes_fail_closed(self):
        mutations = []

        def missing_manifest():
            (self.fixture.public_root / "test/release.json").unlink()

        mutations.append(("missing manifest", missing_manifest))

        def hash_mismatch():
            (self.fixture.public_root / "test/tasks.jsonl").write_bytes(b"{}\n")

        mutations.append(("hash mismatch", hash_mismatch))

        def public_label_file():
            (self.fixture.public_root / "test/labels.jsonl").write_bytes(b"secret\n")

        mutations.append(("public labels", public_label_file))

        def public_extra_file():
            (self.fixture.public_root / "test/notes.txt").write_bytes(b"extra\n")

        mutations.append(("public extra", public_extra_file))

        def public_archive():
            (self.fixture.public_root / "test/data.zip").write_bytes(b"PK\x03\x04")

        mutations.append(("public archive", public_archive))

        def private_bad_mode():
            (self.fixture.private_root / "private/test_labels.jsonl").chmod(0o644)

        mutations.append(("private mode", private_bad_mode))

        def private_enabled():
            manifest = dict(self.fixture.private_manifest)
            manifest["enabled"] = True
            path = self.fixture.private_root / "private/test_release.json"
            path.write_bytes(_canonical_json(manifest))
            path.chmod(0o600)

        mutations.append(("private enabled", private_enabled))

        def private_schema_version_bool():
            manifest = dict(self.fixture.private_manifest)
            manifest["schema_version"] = True
            path = self.fixture.private_root / "private/test_release.json"
            path.write_bytes(_canonical_json(manifest))
            path.chmod(0o600)

        mutations.append(("private schema version type", private_schema_version_bool))

        def private_visibility_contract_extra():
            manifest = dict(self.fixture.private_manifest)
            manifest["visibility_audit"] = dict(manifest["visibility_audit"])
            manifest["visibility_audit"]["unapproved"] = True
            path = self.fixture.private_root / "private/test_release.json"
            path.write_bytes(_canonical_json(manifest))
            path.chmod(0o600)

        mutations.append(
            ("private visibility contract", private_visibility_contract_extra)
        )

        def private_hash_mismatch():
            path = self.fixture.private_root / "private/test_labels.jsonl"
            path.write_bytes(self.fixture.labels_bytes + b"{}\n")
            path.chmod(0o600)

        mutations.append(("private hash mismatch", private_hash_mismatch))

        def private_invalid_answer_with_matching_digest():
            rows = [dict(row) for row in self.fixture.label_rows]
            rows[0]["answer"] = {"hidden": "17"}
            self._replace_private_labels_and_digest(rows)

        mutations.append(
            ("private answer schema", private_invalid_answer_with_matching_digest)
        )

        def private_duplicate_evidence_with_matching_digest():
            rows = [dict(row) for row in self.fixture.label_rows]
            rows[0]["evidence"] = ["b01", "b01"]
            self._replace_private_labels_and_digest(rows)

        mutations.append(
            ("private evidence schema", private_duplicate_evidence_with_matching_digest)
        )

        def card_mismatch():
            self.fixture.release_card.write_bytes(b"wrong card\n")

        mutations.append(("card mismatch", card_mismatch))

        for name, mutate in mutations:
            with self.subTest(name=name):
                self.temporary.cleanup()
                self.temporary = tempfile.TemporaryDirectory()
                self.fixture = PublicationFixture(self.temporary.name)
                self.config = replace(
                    self.config,
                    public_stage=self.fixture.public_root,
                    private_stage=self.fixture.private_root,
                    release_card=self.fixture.release_card,
                    card_template=self.fixture.card_template,
                )
                mutate()
                with self.assertRaises(publisher.ReleaseError):
                    self.run_release()

    def _replace_private_labels_and_digest(self, rows):
        labels_bytes = _canonical_rows(rows)
        labels = self.fixture.private_root / "private/test_labels.jsonl"
        policy = self.fixture.private_root / "private/test_release.json"
        manifest = dict(self.fixture.private_manifest)
        manifest["gold_sha256"] = _digest(labels_bytes)
        labels.write_bytes(labels_bytes)
        policy.write_bytes(_canonical_json(manifest))
        labels.chmod(0o600)
        policy.chmod(0o600)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_linked_public_or_private_input_is_rejected(self):
        public_tasks = self.fixture.public_root / "test/tasks.jsonl"
        saved_tasks = self.fixture.root / "saved-tasks.jsonl"
        public_tasks.replace(saved_tasks)
        public_tasks.symlink_to(saved_tasks)
        with self.assertRaises(publisher.ReleaseError):
            self.run_release()

        public_tasks.unlink()
        saved_tasks.replace(public_tasks)
        labels = self.fixture.private_root / "private/test_labels.jsonl"
        saved_labels = self.fixture.root / "saved-labels.jsonl"
        labels.replace(saved_labels)
        labels.symlink_to(saved_labels)
        with self.assertRaises(publisher.ReleaseError):
            self.run_release()

    def test_publish_requires_exact_confirmation_and_uses_safe_order_and_paths(self):
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="yes")
        self.assertEqual(self.event_log, [])

        receipt = self.run_release(publish=True, confirmation="PUBLISH")

        self.assertEqual(self.event_log, ["private_hf", "github", "public_hf"])
        self.assertEqual(receipt["mode"], "published-and-reconciled")
        self.assertEqual(receipt["revisions"]["github"], "d" * 40)
        self.assertEqual(receipt["revisions"]["private_hugging_face"], "e" * 40)
        self.assertEqual(receipt["revisions"]["public_hugging_face"], "f" * 40)
        self.assertEqual(self.hf.writes[0]["expected_parent"], self.PRIVATE_BASE)
        self.assertEqual(self.git.writes[0]["expected_parent"], self.SOURCE_BASE)
        self.assertEqual(self.hf.writes[1]["expected_parent"], self.PUBLIC_BASE)
        private_release = json.loads(
            self.hf.writes[0]["files"]["private/test_release.json"].decode("utf-8")
        )
        self.assertIs(private_release["enabled"], False)
        self.assertIs(private_release["finalized"], False)
        self.assertNotIn(
            "activation", " ".join(write["message"] for write in self.hf.writes)
        )

    def test_remote_movement_aborts_without_overwriting_or_continuing(self):
        self.hf.move_before_commit.add(publisher.PRIVATE_HF_REPOSITORY)
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="PUBLISH")
        self.assertEqual(self.event_log, ["private_hf"])
        self.assertEqual(self.git.writes, [])

        self.setUp_fresh_fixture()
        self.git.move_before_commit = True
        with self.assertRaisesRegex(
            publisher.ReleaseError,
            rf"private_hugging_face={('e' * 40)}",
        ):
            self.run_release(publish=True, confirmation="PUBLISH")
        self.assertEqual(self.event_log, ["private_hf", "github"])
        self.assertEqual(len(self.hf.writes), 1)

    def test_reconciliation_detects_remote_byte_and_reachable_history_leaks(self):
        self.hf.tamper_after_commit = (
            publisher.PUBLIC_HF_REPOSITORY,
            "test/tasks.jsonl",
        )
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="PUBLISH")

        self.setUp_fresh_fixture()
        self.git.tamper_after_commit = "docsem/test/documents/test_000001.pdf"
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="PUBLISH")

        self.setUp_fresh_fixture()
        self.hf.history[publisher.PUBLIC_HF_REPOSITORY].append(
            (
                "7" * 40,
                {
                    "test/tasks.jsonl": _canonical_rows(
                        [
                            {
                                "instance_id": "leak",
                                "user_query": "q",
                                "document_pdf": "test/documents/leak.pdf",
                                "answer": "secret",
                            }
                        ]
                    )
                },
            )
        )
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="PUBLISH")

        self.setUp_fresh_fixture()
        self.git.history.append(("6" * 40, {"docsem/test/labels.jsonl": b"secret\n"}))
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="PUBLISH")

        self.setUp_fresh_fixture()
        self.git.history.append(
            (
                "5" * 40,
                {
                    "docsem/test/ground_truth.jsonl": _canonical_json(
                        {"instance_id": "x"}
                    )
                },
            )
        )
        with self.assertRaises(publisher.ReleaseError):
            self.run_release(publish=True, confirmation="PUBLISH")

    def test_dry_run_scans_entire_public_trees_and_reachable_history(self):
        forbidden_cases = (
            ("github private tree", "github", "private/audit.json", b"{}\n"),
            (
                "github validation gold",
                "github",
                "docsem/validation/gold.jsonl",
                b"{}\n",
            ),
            ("hf test answers", "hf", "test/answers.jsonl", b"{}\n"),
            ("hf archive", "hf", "legacy/data.zip", b"PK\x03\x04"),
            ("hf polyglot", "hf", "test/tasks.jsonl.pdf", b"%PDF\n"),
            ("hf disguised metadata", "hf", "test/tasks.jsonl.txt", b"{}\n"),
        )
        for name, target, path, payload in forbidden_cases:
            with self.subTest(name=name):
                self.setUp_fresh_fixture()
                if target == "github":
                    prior_tree = dict(self.git.tree)
                    prior_tree[path] = payload
                    self.git.history.append(("4" * 40, prior_tree))
                else:
                    repository = publisher.PUBLIC_HF_REPOSITORY
                    prior_tree = dict(self.hf.trees[repository])
                    prior_tree[path] = payload
                    self.hf.history[repository].append(("4" * 40, prior_tree))
                with self.assertRaises(publisher.ReleaseError):
                    self.run_release()
                self.assertEqual(self.event_log, [])

        self.setUp_fresh_fixture()
        repository = publisher.PUBLIC_HF_REPOSITORY
        self.hf.trees[repository]["legacy/data.zip"] = b"PK\x03\x04"
        with self.assertRaises(publisher.ReleaseError):
            self.run_release()

        self.setUp_fresh_fixture()
        self.git.history.append(
            (
                "3" * 40,
                {
                    "docsem/val/tasks.jsonl": _canonical_rows(
                        [
                            {
                                "instance_id": "val_leak",
                                "user_query": "q",
                                "document_pdf": "val/documents/val_leak.pdf",
                                "answer": "secret",
                            }
                        ]
                    )
                },
            )
        )
        with self.assertRaisesRegex(publisher.ReleaseError, "forbidden field"):
            self.run_release()

        self.setUp_fresh_fixture()
        repository = publisher.PUBLIC_HF_REPOSITORY
        prior_tree = dict(self.hf.trees[repository])
        prior_tree["test/tasks.jsonl"] = _canonical_rows(
            [
                {
                    "instance_id": "test_leak",
                    "user_query": "q",
                    "document_pdf": "test/documents/test_leak.pdf",
                    "solutions": ["secret"],
                }
            ]
        )
        self.hf.history[repository].append(("1" * 40, prior_tree))
        with self.assertRaisesRegex(publisher.ReleaseError, "forbidden field"):
            self.run_release()

    def test_only_canonical_training_label_path_is_allowed_publicly(self):
        self.git.tree["docsem/train/labels.jsonl"] = b"public training labels\n"
        self.run_release()

        self.setUp_fresh_fixture()
        repository = publisher.PUBLIC_HF_REPOSITORY
        self.hf.trees[repository]["legacy/train_labels.jsonl"] = b"public\n"
        self.hf.history[repository].append(("2" * 40, dict(self.hf.trees[repository])))
        with self.assertRaises(publisher.ReleaseError):
            self.run_release()

        self.setUp_fresh_fixture()
        self.git.tree["train/labels.jsonl"] = b"misplaced public labels\n"
        with self.assertRaises(publisher.ReleaseError):
            self.run_release()

    def test_card_template_must_equal_public_hf_base_readme(self):
        self.fixture.card_template.write_bytes(b"---\nprivate: secret\n---\nbody\n")
        self.fixture.release_card.write_bytes(self.fixture.release_card_bytes)
        with self.assertRaisesRegex(publisher.ReleaseError, "README"):
            self.run_release()

        self.setUp_fresh_fixture()
        self.fixture.release_card.write_bytes(
            self.fixture.release_card_bytes + b"Private organizer content.\n"
        )
        with self.assertRaises(publisher.ReleaseError):
            self.run_release()

    def setUp_fresh_fixture(self):
        self.temporary.cleanup()
        self.setUp()

    def test_generic_remote_errors_do_not_echo_tokens_or_private_values(self):
        sentinel = "PRIVATE-ANSWER-DO-NOT-PRINT"
        self.hf.fail_message = (
            f"backend exploded: {sentinel} public-token private-token"
        )
        with self.assertRaises(publisher.ReleaseError) as caught:
            self.run_release(publish=True, confirmation="PUBLISH")
        message = str(caught.exception)
        self.assertNotIn(sentinel, message)
        self.assertNotIn("public-token", message)
        self.assertNotIn("private-token", message)

    def test_cli_output_is_sanitized_json_and_default_is_dry_run(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = publisher.main(
                [
                    "--public-stage",
                    str(self.fixture.public_root),
                    "--private-stage",
                    str(self.fixture.private_root),
                    "--release-card",
                    str(self.fixture.release_card),
                    "--card-template",
                    str(self.fixture.card_template),
                    "--source-branch",
                    "main",
                    "--source-base",
                    self.SOURCE_BASE,
                    "--github-remote-base",
                    self.SOURCE_BASE,
                    "--public-hf-base",
                    self.PUBLIC_BASE,
                    "--private-hf-base",
                    self.PRIVATE_BASE,
                ],
                git_backend=self.git,
                hf_backend=self.hf,
                public_token="public-token",
                private_token="private-token",
                public_auditor=self.fixture.public_auditor,
                card_renderer=self.fixture.card_renderer,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["mode"], "dry-run")
        self.assertNotIn("Question one", output.getvalue())
        self.assertNotIn('"answer"', output.getvalue())
        self.assertNotIn('"evidence"', output.getvalue())
        self.assertEqual(self.git.writes, [])
        self.assertEqual(self.hf.writes, [])

    def test_cli_supports_documented_docsem_hf_write_token(self):
        self.hf.required_token = "documented-token"
        original_state = self.hf.repository_state

        def checked_state(repository, token):
            if token != self.hf.required_token:
                raise publisher.ReleaseError("wrong credential")
            return original_state(repository, token)

        self.hf.repository_state = checked_state
        output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"DOCSEM_HF_WRITE_TOKEN": "documented-token"},
                clear=True,
            ),
            contextlib.redirect_stdout(output),
        ):
            exit_code = publisher.main(
                [
                    "--public-stage",
                    str(self.fixture.public_root),
                    "--private-stage",
                    str(self.fixture.private_root),
                    "--release-card",
                    str(self.fixture.release_card),
                    "--card-template",
                    str(self.fixture.card_template),
                    "--source-branch",
                    "main",
                    "--source-base",
                    self.SOURCE_BASE,
                    "--github-remote-base",
                    self.SOURCE_BASE,
                    "--public-hf-base",
                    self.PUBLIC_BASE,
                    "--private-hf-base",
                    self.PRIVATE_BASE,
                ],
                git_backend=self.git,
                hf_backend=self.hf,
                public_auditor=self.fixture.public_auditor,
                card_renderer=self.fixture.card_renderer,
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "dry-run")

    def test_real_hf_backend_collects_all_split_sensitive_metadata(self):
        revision = "1" * 40
        tree = {
            "README.md": b"card\n",
            "train/tasks.jsonl": b"{}\n",
            "train/labels.jsonl": b"{}\n",
            "val/tasks.jsonl": b'{"instance_id":"v"}\n',
            "validation/release.json": b"{}\n",
            "test/tasks.jsonl": b'{"instance_id":"t"}\n',
        }

        class Api:
            def __init__(self, token=None):
                self.token = token

            def list_repo_commits(self, **_kwargs):
                return [SimpleNamespace(commit_id=revision)]

            def list_repo_files(self, **_kwargs):
                return tuple(tree)

            def repo_info(self, **_kwargs):
                return SimpleNamespace(sha=revision, private=False)

        def download(*, filename, cache_dir, **_kwargs):
            destination = Path(cache_dir) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(tree[filename])
            return str(destination)

        backend = publisher.HuggingFaceBackend()
        with mock.patch.object(
            publisher.HuggingFaceBackend,
            "_imports",
            return_value=(object, Api, download),
        ):
            state = backend.repository_state("owner/data", "token")
            snapshots = backend.history_snapshots("owner/data", "token")
        self.assertEqual(
            state,
            publisher.HfRepositoryState(revision=revision, private=False),
        )
        self.assertEqual(
            set(snapshots[0].metadata),
            {"val/tasks.jsonl", "validation/release.json", "test/tasks.jsonl"},
        )

    def test_real_git_backend_uses_fast_forward_push_and_reads_reachable_history(self):
        root = self.fixture.root / "git-integration"
        remote = root / "remote.git"
        seed = root / "seed"
        root.mkdir()
        subprocess.run(
            ["git", "init", "--bare", "--initial-branch=main", str(remote)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "init", "--initial-branch=main", str(seed)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (seed / "docsem").mkdir()
        (seed / "docsem/README.md").write_text("existing\n", encoding="utf-8")
        subprocess.run(["git", "add", "docsem/README.md"], cwd=seed, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-m",
                "seed",
            ],
            cwd=seed,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "remote", "add", "origin", str(remote)], cwd=seed, check=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=seed,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=seed,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        sealed = root / "tasks.jsonl"
        sealed.write_bytes(self.fixture.tasks_bytes)
        sealed.chmod(0o400)
        artifact = publisher.Artifact(
            sealed, len(self.fixture.tasks_bytes), _digest(self.fixture.tasks_bytes)
        )

        hooks_template = root / "hooks-template"
        hooks = hooks_template / "hooks"
        hooks.mkdir(parents=True)
        marker = root / "hook-ran"
        for name in ("pre-commit", "pre-push"):
            hook = hooks / name
            hook.write_text(
                f"#!/bin/sh\nprintf hook > {marker}\nexit 97\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)

        backend = publisher.SubprocessGitBackend()
        try:
            with mock.patch.dict(os.environ, {"GIT_TEMPLATE_DIR": str(hooks_template)}):
                revision = backend.publish(
                    str(remote),
                    "main",
                    base,
                    {"docsem/test/tasks.jsonl": artifact},
                    "fixture release",
                )
            self.assertNotEqual(revision, base)
            self.assertFalse(marker.exists())
            self.assertEqual(backend.current_revision(str(remote), "main"), revision)
            self.assertIn(
                "docsem/test/tasks.jsonl", backend.list_paths(str(remote), revision)
            )
            self.assertEqual(
                backend.read_files(str(remote), revision, ["docsem/test/tasks.jsonl"]),
                {"docsem/test/tasks.jsonl": self.fixture.tasks_bytes},
            )
            history = backend.history_snapshots(str(remote), "main")
            self.assertGreaterEqual(len(history), 2)
            self.assertEqual(history[0].revision, revision)
        finally:
            backend.close()

    def test_git_backend_rejects_attributes_and_linked_parent_components(self):
        for name, entries in (
            (
                "attributes",
                {
                    ".gitattributes": b"*.jsonl filter=evil\n",
                    "docsem/README.md": b"x\n",
                },
            ),
            (
                "linked parent",
                {"docsem/README.md": b"x\n", "docsem/test": ("symlink", "../escape")},
            ),
        ):
            with self.subTest(name=name):
                root, remote, base, artifact = self._git_backend_fixture(entries)
                backend = publisher.SubprocessGitBackend()
                try:
                    with self.assertRaises(publisher.ReleaseError):
                        backend.publish(
                            str(remote),
                            "main",
                            base,
                            {"docsem/test/tasks.jsonl": artifact},
                            "fixture release",
                        )
                    self.assertEqual(
                        backend.current_revision(str(remote), "main"), base
                    )
                finally:
                    backend.close()
                    root.cleanup()

    def test_git_blob_verifier_rejects_index_and_commit_mismatch(self):
        root = tempfile.TemporaryDirectory()
        checkout = Path(root.name)
        subprocess.run(
            ["git", "init", "--initial-branch=main", str(checkout)], check=True
        )
        target = checkout / "docsem/test/tasks.jsonl"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"wrong\n")
        subprocess.run(
            ["git", "add", "--", "docsem/test/tasks.jsonl"], cwd=checkout, check=True
        )
        sealed = checkout / "sealed"
        sealed.write_bytes(b"expected\n")
        sealed.chmod(0o400)
        artifact = publisher.Artifact(sealed, 9, _digest(b"expected\n"))
        with self.assertRaises(publisher.ReleaseError):
            publisher._verify_git_artifacts(
                checkout,
                ":",
                {"docsem/test/tasks.jsonl": artifact},
                "index",
            )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-m",
                "wrong blob",
            ],
            cwd=checkout,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self.assertRaises(publisher.ReleaseError):
            publisher._verify_git_artifacts(
                checkout,
                "HEAD",
                {"docsem/test/tasks.jsonl": artifact},
                "committed",
            )
        root.cleanup()

    def _git_backend_fixture(self, entries):
        root = tempfile.TemporaryDirectory()
        base_root = Path(root.name)
        remote = base_root / "remote.git"
        seed = base_root / "seed"
        subprocess.run(
            ["git", "init", "--bare", "--initial-branch=main", str(remote)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "init", "--initial-branch=main", str(seed)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for path, payload in entries.items():
            destination = seed / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, tuple) and payload[0] == "symlink":
                destination.symlink_to(payload[1])
            else:
                destination.write_bytes(payload)
        subprocess.run(["git", "add", "--all"], cwd=seed, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-m",
                "seed",
            ],
            cwd=seed,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "remote", "add", "origin", str(remote)], cwd=seed, check=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=seed,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=seed,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        sealed = base_root / "tasks.jsonl"
        sealed.write_bytes(self.fixture.tasks_bytes)
        sealed.chmod(0o400)
        artifact = publisher.Artifact(
            sealed,
            len(self.fixture.tasks_bytes),
            _digest(self.fixture.tasks_bytes),
        )
        return root, remote, base, artifact


if __name__ == "__main__":
    unittest.main(verbosity=2)
