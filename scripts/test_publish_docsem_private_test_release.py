#!/usr/bin/env python3
"""Synthetic behavioral tests for the disabled private DocSem continuation."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

import prepare_docsem_test_release as preparer

try:
    import publish_docsem_private_test_release as publisher
except ModuleNotFoundError:
    publisher = None


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def rows(values: list[dict]) -> bytes:
    return b"".join(canonical(value) for value in values)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FakeHub:
    def __init__(self, public_tree: dict[str, bytes], private_tree: dict[str, bytes]):
        self.public_revision = "a" * 40
        self.private_revision = "b" * 40
        self.public_private = False
        self.private_private = True
        self.username = "amitbcp"
        self.role = "write"
        self.trees = {
            (publisher.PUBLIC_HF_REPOSITORY, self.public_revision): dict(public_tree),
            (publisher.PRIVATE_HF_REPOSITORY, self.private_revision): dict(
                private_tree
            ),
        }
        self.public_history = [self.public_revision]
        self.private_history = [self.private_revision]
        self.reads: list[tuple[str, str, tuple[str, ...]]] = []
        self.writes: list[dict] = []
        self.forbidden_reads = {
            "private/validation_labels.jsonl",
            "private/submissions/raw.jsonl",
            "private/attempts/state.json",
            "private/projections/leaderboard.json",
        }
        self.move_before_publish = False
        self.post_write_tamper = False
        self.raise_after_commit = False
        self.identity_calls = 0
        self.public_visibility_flip_on_second_identity = False
        self.history_events_override = None
        self.history_reachable_override = None
        self.public_history_audit_override = None
        self.public_history_audit_calls = 0
        self.public_history_audit_flip_on_call = None
        self.include_legacy_public_metadata = False

    def identity(self, token):
        self.identity_calls += 1
        if self.public_visibility_flip_on_second_identity and self.identity_calls == 2:
            self.public_private = True
        return publisher.HfIdentity(self.username, self.role)

    def repository_state(self, repository, token):
        if repository == publisher.PUBLIC_HF_REPOSITORY:
            return publisher.HfRepositoryState(
                self.public_revision, self.public_private
            )
        return publisher.HfRepositoryState(self.private_revision, self.private_private)

    def list_paths(self, repository, revision, token):
        return tuple(sorted(self.trees[(repository, revision)]))

    def file_inventory(self, repository, revision, token):
        return {
            path: publisher.RemoteFile(len(payload), digest(payload))
            for path, payload in self.trees[(repository, revision)].items()
        }

    def read_files(self, repository, revision, paths, token):
        paths = tuple(paths)
        self.reads.append((repository, revision, paths))
        if self.forbidden_reads.intersection(paths):
            raise AssertionError("private sentinel path was read")
        tree = self.trees[(repository, revision)]
        return {path: tree[path] for path in paths}

    def history_snapshots(self, repository, token):
        snapshots = []
        for revision in self.public_history:
            tree = self.trees[(repository, revision)]
            paths = tuple(sorted(tree))
            metadata_paths = tuple(
                path
                for path in paths
                if (
                    path.endswith((".json", ".jsonl"))
                    and any(
                        part in {"val", "validation", "test"}
                        for part in path.split("/")[:-1]
                    )
                )
                or (
                    self.include_legacy_public_metadata
                    and path == "data/dev_labels.jsonl"
                )
            )
            snapshots.append(
                publisher.HistorySnapshot(
                    revision,
                    paths,
                    {path: tree[path] for path in metadata_paths},
                )
            )
        return tuple(snapshots)

    def public_history_audit(self, repository, expected_head, token):
        self.public_history_audit_calls += 1
        if expected_head != self.public_revision:
            raise publisher.ReleaseError("public history head mismatch")
        audit = self.public_history_audit_override
        if audit is None:
            return publisher.PublicHistoryAudit(
                head=expected_head,
                graph=((expected_head, ()),),
                events=(),
                presence=((expected_head, False),),
                shallow=False,
                blob_objects_fetched=False,
            )
        if self.public_history_audit_flip_on_call == self.public_history_audit_calls:
            graph = tuple(audit.graph)
            return replace(audit, graph=tuple(reversed(graph)))
        return audit

    def history_path_names(self, repository, expected_head, token):
        if expected_head != self.private_revision:
            raise publisher.ReleaseError("history head mismatch")
        return frozenset(
            path
            for revision in self.private_history
            for path in self.trees[(repository, revision)]
        )

    def history_audit(self, repository, expected_head, token):
        if self.history_events_override is None:
            events = tuple(
                publisher.PrivateHistoryEvent(
                    revision=revision,
                    parents=(),
                    timestamp="2026-01-01T00:00:00Z",
                    status="M",
                    path=path,
                    subject="SYNTHETIC-HISTORICAL-EVENT",
                )
                for revision in reversed(self.private_history)
                for path in sorted(self.trees[(repository, revision)])
                if path.startswith("private/test_")
            )
        else:
            events = tuple(self.history_events_override)
        reachable = frozenset(
            self.history_reachable_override
            if self.history_reachable_override is not None
            else {event.revision for event in events} | {expected_head}
        )
        parent_map = {event.revision: event.parents for event in events}
        for revision in reachable:
            if revision not in parent_map:
                parent_map[revision] = (
                    (self.LEGACY_DELETE,)
                    if hasattr(self, "LEGACY_DELETE")
                    and self.LEGACY_DELETE in reachable
                    and revision == expected_head
                    else ()
                )
        return publisher.PrivateHistoryAudit(
            head=expected_head,
            events=events,
            reachable=reachable,
            shallow=False,
            blob_objects_fetched=False,
            parent_map=parent_map,
        )

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
        if self.move_before_publish:
            self.private_revision = "c" * 40
            self.trees[(repository, self.private_revision)] = dict(
                self.trees[(repository, expected_parent)]
            )
            raise publisher.ReleaseError("remote moved")
        if self.private_revision != expected_parent or expected_private is not True:
            raise publisher.ReleaseError("parent mismatch")
        current = dict(self.trees[(repository, expected_parent)])
        installed = {
            path: artifact.read_bytes() for path, artifact in operations.items()
        }
        self.writes.append(
            {
                "repository": repository,
                "parent": expected_parent,
                "paths": tuple(sorted(operations)),
                "message": message,
                "files": installed,
                "sources": tuple(
                    operations[path].snapshot_path for path in sorted(operations)
                ),
            }
        )
        current.update(installed)
        if self.post_write_tamper:
            current["private/test_release.json"] += b" "
        self.private_revision = "d" * 40
        self.trees[(repository, self.private_revision)] = current
        self.private_history.insert(0, self.private_revision)
        publication_events = (
            publisher.PrivateHistoryEvent(
                revision=self.private_revision,
                parents=(expected_parent,),
                timestamp="2026-09-05T20:00:00Z",
                status="A",
                path="private/test_labels.jsonl",
                subject=message,
            ),
            publisher.PrivateHistoryEvent(
                revision=self.private_revision,
                parents=(expected_parent,),
                timestamp="2026-09-05T20:00:00Z",
                status="A",
                path="private/test_release.json",
                subject=message,
            ),
        )
        self.history_events_override = (
            tuple(self.history_events_override or ()) + publication_events
        )
        self.history_reachable_override = set(
            self.history_reachable_override or {expected_parent}
        ) | {self.private_revision}
        if self.raise_after_commit:
            raise publisher.ReleaseError("PRIVATE-UNCERTAIN-RESPONSE")
        return self.private_revision


class PinnedDefaultTests(unittest.TestCase):
    def test_default_private_label_source_is_the_exact_approved_file(self):
        self.assertEqual(
            publisher.PRIVATE_LABEL_SOURCE,
            Path("/private/tmp/docsem-private-source-a4205880-r1/labels.jsonl"),
        )
        self.assertFalse(hasattr(publisher, "LEGACY_ADD_REVISION"))
        self.assertFalse(hasattr(publisher, "LEGACY_DELETE_REVISION"))
        self.assertEqual(
            publisher.LEGACY_HISTORY_POLICY, "legacy-private-label-cycle-v1"
        )
        self.assertEqual(
            publisher.LEGACY_PUBLIC_HISTORY_POLICY,
            "legacy-public-development-label-cycle-v1",
        )
        self.assertEqual(
            publisher.LEGACY_PUBLIC_HISTORY_METADATA_SHA256,
            "d14363ea4836a87af514ee3658314a588d407d015313bb237602caf40fafc20c",
        )


class PrivateContinuationTests(unittest.TestCase):
    LEGACY_POLICY = "legacy-private-label-cycle-v1"
    LEGACY_CONFIRM = "ACKNOWLEDGE_RETAINED_LEGACY_PRIVATE_LABEL_HISTORY"
    LEGACY_ADD = "1" * 40
    LEGACY_DELETE = "2" * 40
    PUBLIC_HISTORY_POLICY = "legacy-public-development-label-cycle-v1"
    PUBLIC_HISTORY_CONFIRM = "ACKNOWLEDGE_RETAINED_PUBLIC_DEVELOPMENT_LABEL_HISTORY"
    PUBLIC_ADD = "4" * 40
    PUBLIC_PRESENT = "5" * 40
    PUBLIC_DELETE = "6" * 40

    def setUp(self):
        if publisher is None:
            self.fail("private-only continuation publisher is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.public_stage = self.root / "public-stage"
        self.private_source = self.root / "private-source" / "test_labels.jsonl"
        self.private_stage = self.root / "private-stage"
        (self.public_stage / "test/documents").mkdir(parents=True)
        self.private_source.parent.mkdir()
        self.ids = ["test_010001", "test_010002"]
        self.task_rows = [
            {
                "instance_id": item,
                "user_query": f"synthetic query {index}",
                "document_pdf": f"test/documents/{item}.pdf",
            }
            for index, item in enumerate(self.ids)
        ]
        self.label_rows = [
            {
                "instance_id": item,
                "answer": f"PRIVATE-ANSWER-{index}-Z9x7Q4m2",
                "evidence": [f"A{index}#9"],
            }
            for index, item in enumerate(self.ids)
        ]
        self.task_bytes = rows(self.task_rows)
        self.label_bytes = rows(self.label_rows)
        (self.public_stage / "test/tasks.jsonl").write_bytes(self.task_bytes)
        pdf_digests = {}
        for index, item in enumerate(self.ids):
            payload = f"%PDF-1.4 synthetic {index}\n%%EOF\n".encode()
            path = self.public_stage / f"test/documents/{item}.pdf"
            path.write_bytes(payload)
            pdf_digests[path.name] = digest(payload)
        self.sorted_ids_digest = digest(
            "".join(f"{item}\n" for item in self.ids).encode()
        )
        self.pdf_digest = digest(
            b"".join(
                f"{name}  {value}\n".encode()
                for name, value in sorted(pdf_digests.items())
            )
        )
        self.manifest = {
            "schema_version": 1,
            "release_id": "docsem-test-synthetic-r1",
            "counts": {"tasks": 2, "pdfs": 2},
            "sorted_ids_sha256": self.sorted_ids_digest,
            "task_manifest_sha256": digest(self.task_bytes),
            "pdf_inventory_sha256": self.pdf_digest,
        }
        manifest_bytes = canonical(self.manifest)
        (self.public_stage / "test/release.json").write_bytes(manifest_bytes)
        checksum_items = {
            "release.json": digest(manifest_bytes),
            "tasks.jsonl": digest(self.task_bytes),
            **{f"documents/{name}": value for name, value in pdf_digests.items()},
        }
        self.checksum_bytes = b"".join(
            f"{value}  {name}\n".encode()
            for name, value in sorted(checksum_items.items())
        )
        (self.public_stage / "test/SHA256SUMS").write_bytes(self.checksum_bytes)
        self.private_source.write_bytes(self.label_bytes)

        self.legacy_metadata_digest = digest(
            canonical(
                [
                    {
                        "revision": self.LEGACY_ADD,
                        "parents": [],
                        "timestamp": "2020-01-01T00:00:00Z",
                        "status": "A",
                        "path": "private/test_labels.jsonl",
                    },
                    {
                        "revision": self.LEGACY_DELETE,
                        "parents": [self.LEGACY_ADD],
                        "timestamp": "2020-01-01T00:01:00Z",
                        "status": "D",
                        "path": "private/test_labels.jsonl",
                    },
                ]
            )
        )
        self.public_history_metadata_digest = digest(
            canonical(
                {
                    "schema_version": 1,
                    "head": "a" * 40,
                    "graph": [
                        {"revision": self.PUBLIC_ADD, "parents": []},
                        {
                            "revision": self.PUBLIC_PRESENT,
                            "parents": [self.PUBLIC_ADD],
                        },
                        {
                            "revision": self.PUBLIC_DELETE,
                            "parents": [self.PUBLIC_PRESENT],
                        },
                        {"revision": "a" * 40, "parents": [self.PUBLIC_DELETE]},
                    ],
                    "events": [
                        {
                            "revision": self.PUBLIC_ADD,
                            "parents": [],
                            "timestamp": "2020-01-01T00:00:00Z",
                            "status": "A",
                            "path": "data/dev_labels.jsonl",
                        },
                        {
                            "revision": self.PUBLIC_DELETE,
                            "parents": [self.PUBLIC_PRESENT],
                            "timestamp": "2020-01-01T00:02:00Z",
                            "status": "D",
                            "path": "data/dev_labels.jsonl",
                        },
                    ],
                    "presence": [
                        {"revision": self.PUBLIC_ADD, "present": True},
                        {"revision": self.PUBLIC_PRESENT, "present": True},
                        {"revision": self.PUBLIC_DELETE, "present": False},
                        {"revision": "a" * 40, "present": False},
                    ],
                }
            )
        )

        self.patches = [
            mock.patch.object(publisher, "PUBLIC_STAGE", self.public_stage),
            mock.patch.object(publisher, "PRIVATE_LABEL_SOURCE", self.private_source),
            mock.patch.object(publisher, "RELEASE_ID", self.manifest["release_id"]),
            mock.patch.object(publisher, "EXPECTED_COUNT", 2),
            mock.patch.object(publisher, "PUBLIC_REVISION", "a" * 40),
            mock.patch.object(publisher, "SORTED_IDS_SHA256", self.sorted_ids_digest),
            mock.patch.object(
                publisher, "TASK_MANIFEST_SHA256", digest(self.task_bytes)
            ),
            mock.patch.object(publisher, "PDF_INVENTORY_SHA256", self.pdf_digest),
            mock.patch.object(
                publisher, "PRIVATE_LABELS_SHA256", digest(self.label_bytes)
            ),
            mock.patch.object(
                publisher,
                "LEGACY_HISTORY_METADATA_SHA256",
                self.legacy_metadata_digest,
            ),
            mock.patch.object(
                publisher,
                "LEGACY_PUBLIC_HISTORY_METADATA_SHA256",
                self.public_history_metadata_digest,
                create=True,
            ),
        ]
        for patcher in self.patches:
            patcher.start()

        public_tree = {
            path.relative_to(self.public_stage).as_posix(): path.read_bytes()
            for path in self.public_stage.rglob("*")
            if path.is_file()
        }
        public_tree["README.md"] = b"public metadata\n"
        self.private_sentinel = b"VALIDATION-SENTINEL-V8q2Km9X\n"
        self.private_tree = {
            "README.md": b"private repository\n",
            "private/validation_labels.jsonl": self.private_sentinel,
            "private/submissions/raw.jsonl": b"SUBMISSION-SENTINEL-R7y4Lp2Q\n",
            "private/attempts/state.json": b"ATTEMPT-SENTINEL-W9m3Hd6T\n",
            "private/projections/leaderboard.json": b"PROJECTION-SENTINEL-X4n8Jk1P\n",
        }
        self.hub = FakeHub(public_tree, self.private_tree)
        self.config = publisher.ReleaseConfig(
            self.public_stage,
            self.private_source,
            self.private_stage,
            "b" * 40,
        )

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def audit(self, path):
        self.assertEqual(Path(path), self.public_stage)
        return dict(self.manifest)

    def prepare(self):
        return publisher.prepare_stage(self.config, public_auditor=self.audit)

    def release_call(self, **kwargs):
        return publisher.run_private_continuation(
            self.config,
            hf_backend=self.hub,
            token="CLASSIC-WRITE-TOKEN-SENTINEL-N7q2",
            public_auditor=self.audit,
            **kwargs,
        )

    def exact_legacy_events(self):
        return (
            publisher.PrivateHistoryEvent(
                revision=self.LEGACY_ADD,
                parents=(),
                timestamp="2020-01-01T00:00:00Z",
                status="A",
                path="private/test_labels.jsonl",
                subject="SYNTHETIC-LEGACY-SUBJECT-PRIVATE",
            ),
            publisher.PrivateHistoryEvent(
                revision=self.LEGACY_DELETE,
                parents=(self.LEGACY_ADD,),
                timestamp="2020-01-01T00:01:00Z",
                status="D",
                path="private/test_labels.jsonl",
                subject="SYNTHETIC-DELETE-SUBJECT-PRIVATE",
            ),
        )

    def configure_exact_legacy_history(self):
        self.hub.history_events_override = self.exact_legacy_events()
        self.hub.history_reachable_override = {
            self.LEGACY_ADD,
            self.LEGACY_DELETE,
            self.hub.private_revision,
        }

    def exact_public_history_audit(self):
        return publisher.PublicHistoryAudit(
            head=self.hub.public_revision,
            graph=(
                (self.PUBLIC_ADD, ()),
                (self.PUBLIC_PRESENT, (self.PUBLIC_ADD,)),
                (self.PUBLIC_DELETE, (self.PUBLIC_PRESENT,)),
                (self.hub.public_revision, (self.PUBLIC_DELETE,)),
            ),
            events=(
                publisher.PublicHistoryEvent(
                    revision=self.PUBLIC_ADD,
                    parents=(),
                    timestamp="2020-01-01T00:00:00Z",
                    status="A",
                    path="data/dev_labels.jsonl",
                ),
                publisher.PublicHistoryEvent(
                    revision=self.PUBLIC_DELETE,
                    parents=(self.PUBLIC_PRESENT,),
                    timestamp="2020-01-01T00:02:00Z",
                    status="D",
                    path="data/dev_labels.jsonl",
                ),
            ),
            presence=(
                (self.PUBLIC_ADD, True),
                (self.PUBLIC_PRESENT, True),
                (self.PUBLIC_DELETE, False),
                (self.hub.public_revision, False),
            ),
            shallow=False,
            blob_objects_fetched=False,
        )

    def configure_exact_public_legacy_history(self):
        retired_path = "data/dev_labels.jsonl"
        base_tree = dict(
            self.hub.trees[(publisher.PUBLIC_HF_REPOSITORY, self.hub.public_revision)]
        )
        sentinel = b"LEGACY-PUBLIC-CONTENT-MUST-NOT-BE-READ\n"
        for revision, present in (
            (self.PUBLIC_ADD, True),
            (self.PUBLIC_PRESENT, True),
            (self.PUBLIC_DELETE, False),
        ):
            tree = dict(base_tree)
            if present:
                tree[retired_path] = sentinel
            self.hub.trees[(publisher.PUBLIC_HF_REPOSITORY, revision)] = tree
        self.hub.public_history = [
            self.hub.public_revision,
            self.PUBLIC_DELETE,
            self.PUBLIC_PRESENT,
            self.PUBLIC_ADD,
        ]
        self.hub.public_history_audit_override = self.exact_public_history_audit()

    @staticmethod
    def public_audit_digest(audit):
        return digest(
            canonical(
                {
                    "schema_version": 1,
                    "head": audit.head,
                    "graph": [
                        {"revision": revision, "parents": list(parents)}
                        for revision, parents in audit.graph
                    ],
                    "events": [
                        {
                            "revision": event.revision,
                            "parents": list(event.parents),
                            "timestamp": event.timestamp,
                            "status": event.status,
                            "path": event.path,
                        }
                        for event in audit.events
                    ],
                    "presence": [
                        {"revision": revision, "present": present}
                        for revision, present in audit.presence
                    ],
                }
            )
        )

    def test_prepare_stage_installs_only_exact_private_files_with_safe_modes(self):
        before = {
            path.relative_to(self.public_stage): path.read_bytes()
            for path in self.public_stage.rglob("*")
            if path.is_file()
        }
        result = self.prepare()
        files = {
            path.relative_to(self.private_stage).as_posix()
            for path in self.private_stage.rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            files, {"private/test_labels.jsonl", "private/test_release.json"}
        )
        self.assertEqual(
            (self.private_stage / "private/test_labels.jsonl").read_bytes(),
            self.label_bytes,
        )
        self.assertEqual(stat.S_IMODE(self.private_stage.stat().st_mode), 0o700)
        self.assertEqual(
            stat.S_IMODE((self.private_stage / "private").stat().st_mode), 0o700
        )
        for path in files:
            self.assertEqual(
                stat.S_IMODE((self.private_stage / path).stat().st_mode), 0o600
            )
        self.assertFalse(
            any(path.suffix == ".pdf" for path in self.private_stage.rglob("*"))
        )
        self.assertEqual(
            before,
            {
                path.relative_to(self.public_stage): path.read_bytes()
                for path in self.public_stage.rglob("*")
                if path.is_file()
            },
        )
        self.assertEqual(result["activation"], "not-performed")
        self.assertNotIn("mode", result)
        self.assertNotIn("answer", json.dumps(result).lower())
        self.assertNotIn("evidence", json.dumps(result).lower())

    def test_prepare_stage_rejects_nonexact_source_and_rolls_back(self):
        cases = {
            "noncanonical": b'{"instance_id":"test_010001", "answer":"x","evidence":["A0#9"]}\n',
            "wrong-schema": rows(
                [{**self.label_rows[0], "extra": True}, self.label_rows[1]]
            ),
            "empty-answer": rows(
                [{**self.label_rows[0], "answer": " "}, self.label_rows[1]]
            ),
            "invalid-evidence": rows(
                [
                    {**self.label_rows[0], "evidence": ["too_long_token"]},
                    self.label_rows[1],
                ]
            ),
            "duplicate-evidence": rows(
                [
                    {**self.label_rows[0], "evidence": ["A0#9", "A0#9"]},
                    self.label_rows[1],
                ]
            ),
            "reordered": rows(list(reversed(self.label_rows))),
            "missing": rows(self.label_rows[:1]),
            "extra": rows(
                self.label_rows + [{**self.label_rows[0], "instance_id": "test_010003"}]
            ),
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                self.private_source.write_bytes(payload)
                with self.assertRaises(publisher.ReleaseError):
                    self.prepare()
                self.assertFalse(self.private_stage.exists())
        self.private_source.write_bytes(self.label_bytes)
        with mock.patch.object(publisher, "PRIVATE_LABELS_SHA256", "0" * 64):
            with self.assertRaises(publisher.ReleaseError):
                self.prepare()
        self.assertFalse(self.private_stage.exists())

    def test_prepare_stage_rejects_link_special_file_and_detected_mutation(self):
        real = self.root / "labels-real.jsonl"
        real.write_bytes(self.label_bytes)
        self.private_source.unlink()
        self.private_source.symlink_to(real)
        with self.assertRaises(publisher.ReleaseError):
            self.prepare()
        self.private_source.unlink()
        os.mkfifo(self.private_source)
        with self.assertRaises(publisher.ReleaseError):
            self.prepare()
        self.private_source.unlink()
        self.private_source.mkdir()
        with self.assertRaises(publisher.ReleaseError):
            self.prepare()
        self.private_source.rmdir()
        self.private_source.write_bytes(self.label_bytes)
        original = publisher._read_bounded_regular_file

        def mutate_after_read(path, *args, **kwargs):
            payload = original(path, *args, **kwargs)
            if Path(path) == self.private_source:
                self.private_source.write_bytes(payload + b" ")
            return payload

        with mock.patch.object(
            publisher, "_read_bounded_regular_file", side_effect=mutate_after_read
        ):
            with self.assertRaises(publisher.ReleaseError):
                self.prepare()
        self.assertFalse(self.private_stage.exists())

    def test_prepare_stage_rejects_public_anchor_inventory_and_checksum_drift(self):
        mutations = (
            (
                "release",
                lambda: setattr(
                    self, "manifest", {**self.manifest, "release_id": "wrong"}
                ),
            ),
            (
                "extra-path",
                lambda: (self.public_stage / "test/extra.json").write_bytes(b"{}\n"),
            ),
            (
                "checksum",
                lambda: (self.public_stage / "test/SHA256SUMS").write_bytes(
                    b"0" * 64 + b"  tasks.jsonl\n"
                ),
            ),
            (
                "task-order",
                lambda: (self.public_stage / "test/tasks.jsonl").write_bytes(
                    rows(list(reversed(self.task_rows)))
                ),
            ),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                original_manifest = dict(self.manifest)
                original_tasks = (self.public_stage / "test/tasks.jsonl").read_bytes()
                original_checksums = (
                    self.public_stage / "test/SHA256SUMS"
                ).read_bytes()
                mutation()
                with self.assertRaises(publisher.ReleaseError):
                    self.prepare()
                self.assertFalse(self.private_stage.exists())
                self.manifest = original_manifest
                (self.public_stage / "test/tasks.jsonl").write_bytes(original_tasks)
                (self.public_stage / "test/SHA256SUMS").write_bytes(original_checksums)
                (self.public_stage / "test/extra.json").unlink(missing_ok=True)

    def test_prepare_stage_rejects_every_pinned_public_manifest_digest_or_count_change(
        self,
    ):
        path = self.public_stage / "test/release.json"
        base = json.loads(path.read_text())
        mutations = (
            lambda value: value["counts"].update(tasks=3),
            lambda value: value.update(sorted_ids_sha256="0" * 64),
            lambda value: value.update(task_manifest_sha256="0" * 64),
            lambda value: value.update(pdf_inventory_sha256="0" * 64),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = json.loads(json.dumps(base))
                mutation(value)
                path.write_bytes(canonical(value))
                with self.assertRaises(publisher.ReleaseError):
                    self.prepare()
                self.assertFalse(self.private_stage.exists())
        path.write_bytes(canonical(base))

    def test_prepare_stage_rejects_linked_special_or_mutating_public_metadata(self):
        path = self.public_stage / "test/tasks.jsonl"
        original = path.read_bytes()
        target = self.root / "public-task-target"
        target.write_bytes(original)
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(publisher.ReleaseError):
            self.prepare()
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(publisher.ReleaseError):
            self.prepare()
        path.unlink()
        path.write_bytes(original)

        def mutate_public(root):
            path.write_bytes(original + b" ")
            return dict(self.manifest)

        with self.assertRaises(publisher.ReleaseError):
            publisher.prepare_stage(self.config, public_auditor=mutate_public)
        self.assertFalse(self.private_stage.exists())
        path.write_bytes(original)

    def test_prepare_stage_is_atomic_and_requires_absent_destination(self):
        self.private_stage.mkdir()
        marker = self.private_stage / "owner-data"
        marker.write_bytes(b"preserve")
        with self.assertRaises(publisher.ReleaseError):
            self.prepare()
        self.assertEqual(marker.read_bytes(), b"preserve")
        marker.unlink()
        self.private_stage.rmdir()
        with mock.patch.object(
            publisher, "_rename_noreplace", side_effect=OSError("synthetic")
        ):
            with self.assertRaises(publisher.ReleaseError):
                self.prepare()
        self.assertFalse(self.private_stage.exists())
        self.assertFalse(any(self.root.glob(".docsem-private-stage-*")))

    def test_private_stage_rejects_all_input_ancestor_and_descendant_overlaps(self):
        before = {
            path.relative_to(self.public_stage): path.read_bytes()
            for path in self.public_stage.rglob("*")
            if path.is_file()
        }
        cases = (
            self.public_stage,
            self.public_stage / "nested-stage",
            self.private_source,
            self.private_source.parent,
            self.private_source.parent / "nested-stage",
            self.root,
        )
        for stage in cases:
            with self.subTest(stage=stage):
                config = publisher.ReleaseConfig(
                    self.public_stage,
                    self.private_source,
                    stage,
                    "b" * 40,
                )
                with self.assertRaises(publisher.ReleaseError):
                    publisher.prepare_stage(config, public_auditor=self.audit)
        self.assertEqual(
            before,
            {
                path.relative_to(self.public_stage): path.read_bytes()
                for path in self.public_stage.rglob("*")
                if path.is_file()
            },
        )

    def test_default_cli_prepare_uses_no_renderer_or_ocr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(
            preparer,
            "_run_bounded_document_probe",
            side_effect=AssertionError("renderer must not run"),
        ) as renderer:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = publisher.main(
                    [
                        "--public-stage",
                        str(self.public_stage),
                        "--private-label-source",
                        str(self.private_source),
                        "--private-stage",
                        str(self.private_stage),
                        "--prepare-stage",
                    ]
                )
        self.assertEqual(code, 0)
        renderer.assert_not_called()

    def test_prepare_stage_rolls_back_install_when_parent_fsync_fails(self):
        original = publisher._fsync_directory
        calls = 0

        def fail_after_install(path):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise publisher.ReleaseError("synthetic durability failure")
            return original(path)

        with mock.patch.object(
            publisher, "_fsync_directory", side_effect=fail_after_install
        ):
            with self.assertRaises(publisher.ReleaseError):
                self.prepare()
        self.assertFalse(self.private_stage.exists())

    def test_prepare_stage_normalizes_all_local_creation_failures(self):
        with mock.patch.object(
            publisher.tempfile, "mkdtemp", side_effect=OSError("PRIVATE-SENTINEL")
        ):
            with self.assertRaises(publisher.ReleaseError):
                self.prepare()
        with mock.patch.object(
            publisher,
            "_write_new_file",
            side_effect=publisher.ValidationError("PRIVATE-SENTINEL"),
        ):
            with self.assertRaises(publisher.ReleaseError):
                self.prepare()
        self.assertFalse(self.private_stage.exists())

    def test_dry_run_requires_existing_exact_stage_and_writes_nothing(self):
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()
        self.assertEqual(self.hub.writes, [])
        self.prepare()
        result = self.release_call()
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["activation"], "not-performed")
        self.assertNotIn("mode", result)
        self.assertEqual(self.hub.writes, [])

    def test_dry_run_does_not_repeat_the_temp_writing_pdf_audit(self):
        self.prepare()
        forbidden = mock.Mock(side_effect=AssertionError("must not run"))
        result = publisher.run_private_continuation(
            self.config,
            hf_backend=self.hub,
            token="token",
            public_auditor=forbidden,
        )
        self.assertNotIn("mode", result)
        forbidden.assert_not_called()

    def test_exact_private_manifest_rejects_schema_policy_and_digest_mutation(self):
        self.prepare()
        path = self.private_stage / "private/test_release.json"
        base = json.loads(path.read_text())
        mutations = (
            lambda item: item.update(extra=True),
            lambda item: item.update(schema_version=True),
            lambda item: item.update(enabled=True),
            lambda item: item.update(finalized=True),
            lambda item: item.update(max_attempts=True),
            lambda item: item.update(gold_sha256="0" * 64),
            lambda item: item.update(task_manifest_sha256="0" * 64),
            lambda item: item.update(visibility_audit={}),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = json.loads(json.dumps(base))
                mutation(value)
                path.write_bytes(canonical(value))
                with self.assertRaises(publisher.ReleaseError):
                    self.release_call()
        path.write_bytes(canonical(base))

    def test_dry_run_rejects_linked_or_special_private_stage_entries(self):
        self.prepare()
        path = self.private_stage / "private/test_release.json"
        original = path.read_bytes()
        target = self.root / "private-policy-target"
        target.write_bytes(original)
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()

    def test_publish_requires_confirmation_classic_owner_exact_parent_and_private_visibility(
        self,
    ):
        self.prepare()
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(publish=True, confirmation=None)
        for attribute, value in (
            ("username", "other-owner"),
            ("role", "fineGrained"),
            ("private_private", False),
        ):
            with self.subTest(attribute=attribute):
                original = getattr(self.hub, attribute)
                setattr(self.hub, attribute, value)
                with self.assertRaises(publisher.ReleaseError):
                    self.release_call()
                setattr(self.hub, attribute, original)
        bad = publisher.ReleaseConfig(
            self.public_stage, self.private_source, self.private_stage, "short"
        )
        with self.assertRaises(publisher.ReleaseError):
            publisher.run_private_continuation(
                bad,
                hf_backend=self.hub,
                token="token",
                public_auditor=self.audit,
            )

    def test_public_revision_visibility_metadata_inventory_and_history_drift_refuse(
        self,
    ):
        self.prepare()
        cases = []
        cases.append(("visibility", lambda: setattr(self.hub, "public_private", True)))
        cases.append(
            ("revision", lambda: setattr(self.hub, "public_revision", "e" * 40))
        )
        cases.append(
            (
                "metadata",
                lambda: self.hub.trees[
                    (publisher.PUBLIC_HF_REPOSITORY, self.hub.public_revision)
                ].__setitem__("test/release.json", b"{}\n"),
            )
        )
        cases.append(
            (
                "inventory",
                lambda: self.hub.trees[
                    (publisher.PUBLIC_HF_REPOSITORY, self.hub.public_revision)
                ].__setitem__("test/documents/extra.pdf", b"%PDF\n%%EOF\n"),
            )
        )
        for name, mutation in cases:
            with self.subTest(name=name):
                public_revision = self.hub.public_revision
                tree = dict(
                    self.hub.trees.get(
                        (publisher.PUBLIC_HF_REPOSITORY, public_revision), {}
                    )
                )
                private = self.hub.public_private
                mutation()
                with self.assertRaises(publisher.ReleaseError):
                    self.release_call()
                self.hub.public_revision = public_revision
                self.hub.public_private = private
                self.hub.trees[(publisher.PUBLIC_HF_REPOSITORY, public_revision)] = tree
        tree = self.hub.trees[
            (publisher.PUBLIC_HF_REPOSITORY, self.hub.public_revision)
        ]
        tree["test/labels.jsonl"] = rows(self.label_rows)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()

    def test_public_reachable_history_label_leak_refuses(self):
        self.prepare()
        old_revision = "8" * 40
        current = self.hub.trees[
            (publisher.PUBLIC_HF_REPOSITORY, self.hub.public_revision)
        ]
        self.hub.trees[(publisher.PUBLIC_HF_REPOSITORY, old_revision)] = {
            **current,
            "test/gold.jsonl": rows(self.label_rows),
        }
        self.hub.public_history.append(old_revision)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()

    def test_legacy_public_history_defaults_to_reject_and_closed_profile_is_read_only(
        self,
    ):
        self.prepare()
        self.configure_exact_public_legacy_history()
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()

        result = self.release_call(
            legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY
        )

        self.assertEqual(result["status"], "pending")
        self.assertEqual(
            result["public_history"],
            {
                "policy_id": self.PUBLIC_HISTORY_POLICY,
                "metadata_sha256": publisher.LEGACY_PUBLIC_HISTORY_METADATA_SHA256,
                "event_count": 2,
                "presence_count": 2,
                "history_retained": True,
                "remediation": False,
                "content_read": False,
            },
        )
        self.assertEqual(self.hub.writes, [])
        self.assertFalse(
            any(
                "data/dev_labels.jsonl" in paths
                for _repository, _revision, paths in self.hub.reads
            )
        )
        self.assertNotIn("LEGACY-PUBLIC-CONTENT-MUST-NOT-BE-READ", json.dumps(result))

    def test_legacy_public_profile_rejects_each_field_graph_event_and_presence_mutation(
        self,
    ):
        self.prepare()
        self.configure_exact_public_legacy_history()
        base = self.exact_public_history_audit()
        extra_event = publisher.PublicHistoryEvent(
            revision=self.hub.public_revision,
            parents=(self.PUBLIC_DELETE,),
            timestamp="2020-01-01T00:03:00Z",
            status="A",
            path="data/dev_labels.jsonl",
        )
        mutations = (
            replace(base, head="f" * 40),
            replace(base, shallow=True),
            replace(base, blob_objects_fetched=True),
            replace(base, graph=base.graph[:-1]),
            replace(base, graph=tuple(reversed(base.graph))),
            replace(
                base,
                graph=(*base.graph[:-1], (self.hub.public_revision, ("f" * 40,))),
            ),
            replace(base, graph=(base.graph[0], base.graph[0], *base.graph[2:])),
            replace(base, events=tuple(reversed(base.events))),
            replace(base, events=base.events[:1]),
            replace(base, events=(*base.events, extra_event)),
            replace(base, events=(replace(base.events[0], status="D"), base.events[1])),
            replace(
                base,
                events=(
                    replace(base.events[0], path="data/other_labels.jsonl"),
                    base.events[1],
                ),
            ),
            replace(
                base,
                events=(
                    replace(base.events[0], timestamp="2020-02-31T00:00:00Z"),
                    base.events[1],
                ),
            ),
            replace(
                base,
                events=(
                    base.events[0],
                    replace(base.events[1], revision="f" * 40),
                ),
            ),
            replace(
                base,
                events=(
                    base.events[0],
                    replace(base.events[1], parents=(self.PUBLIC_ADD,)),
                ),
            ),
            replace(base, presence=base.presence[:-1]),
            replace(base, presence=tuple(reversed(base.presence))),
            replace(
                base,
                presence=(
                    (self.PUBLIC_ADD, False),
                    *base.presence[1:],
                ),
            ),
            replace(
                base,
                presence=(
                    *base.presence[:-1],
                    ("f" * 40, False),
                ),
            ),
            replace(
                base,
                presence=(*base.presence, ("f" * 40, False)),
            ),
        )
        for audit in mutations:
            with self.subTest(audit=audit):
                self.hub.public_history_audit_override = audit
                with mock.patch.object(
                    publisher,
                    "LEGACY_PUBLIC_HISTORY_METADATA_SHA256",
                    self.public_audit_digest(audit),
                ):
                    with self.assertRaises(publisher.ReleaseError):
                        self.release_call(
                            legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY
                        )
        self.assertEqual(self.hub.writes, [])

    def test_legacy_public_profile_checks_snapshot_presence_and_other_forbidden_paths(
        self,
    ):
        self.prepare()
        self.configure_exact_public_legacy_history()
        tree = self.hub.trees[(publisher.PUBLIC_HF_REPOSITORY, self.PUBLIC_PRESENT)]
        tree.pop("data/dev_labels.jsonl")
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY)

        self.configure_exact_public_legacy_history()
        self.hub.trees[(publisher.PUBLIC_HF_REPOSITORY, self.PUBLIC_PRESENT)][
            "data/other_labels.jsonl"
        ] = b"OTHER-FORBIDDEN-PUBLIC-LABEL-PATH\n"
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY)

        self.configure_exact_public_legacy_history()
        self.hub.include_legacy_public_metadata = True
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY)
        self.assertEqual(self.hub.writes, [])

    def test_public_history_policy_and_confirmation_are_independent_and_closed(self):
        self.prepare()
        self.configure_exact_public_legacy_history()
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_public_history_policy="allow")
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                legacy_public_history_confirmation=self.PUBLIC_HISTORY_CONFIRM
            )
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
                legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY,
            )
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
                legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY,
                legacy_public_history_confirmation="wrong",
            )
        self.assertEqual(self.hub.writes, [])

    def test_both_legacy_profiles_require_both_confirmations_and_only_private_cas(self):
        self.prepare()
        self.configure_exact_legacy_history()
        self.configure_exact_public_legacy_history()
        before_public = {
            key: dict(value)
            for key, value in self.hub.trees.items()
            if key[0] == publisher.PUBLIC_HF_REPOSITORY
        }
        common = {
            "publish": True,
            "confirmation": "PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            "legacy_history_policy": self.LEGACY_POLICY,
            "legacy_public_history_policy": self.PUBLIC_HISTORY_POLICY,
        }
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                **common,
                legacy_history_confirmation=self.LEGACY_CONFIRM,
            )
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                **common,
                legacy_public_history_confirmation=self.PUBLIC_HISTORY_CONFIRM,
            )
        result = self.release_call(
            **common,
            legacy_history_confirmation=self.LEGACY_CONFIRM,
            legacy_public_history_confirmation=self.PUBLIC_HISTORY_CONFIRM,
        )
        self.assertEqual(result["status"], "published")
        self.assertEqual(len(self.hub.writes), 1)
        self.assertEqual(
            self.hub.writes[0]["paths"],
            ("private/test_labels.jsonl", "private/test_release.json"),
        )
        after_public = {
            key: dict(value)
            for key, value in self.hub.trees.items()
            if key[0] == publisher.PUBLIC_HF_REPOSITORY
        }
        self.assertEqual(after_public, before_public)
        self.assertEqual(result["public_history"]["content_read"], False)

    def test_public_history_post_cas_drift_is_uncertain_and_never_retried(self):
        self.prepare()
        self.configure_exact_public_legacy_history()
        self.hub.public_history_audit_flip_on_call = 3
        with self.assertRaises(publisher.PublicationUncertainError):
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
                legacy_public_history_policy=self.PUBLIC_HISTORY_POLICY,
                legacy_public_history_confirmation=self.PUBLIC_HISTORY_CONFIRM,
            )
        self.assertEqual(len(self.hub.writes), 1)
        self.assertGreaterEqual(self.hub.public_history_audit_calls, 3)

    def test_cli_accepts_closed_public_history_options_without_printing_legacy_body(
        self,
    ):
        self.prepare()
        self.configure_exact_public_legacy_history()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = publisher.main(
                [
                    "--public-stage",
                    str(self.public_stage),
                    "--private-label-source",
                    str(self.private_source),
                    "--private-stage",
                    str(self.private_stage),
                    "--private-hf-base",
                    "b" * 40,
                    "--legacy-public-history-policy",
                    self.PUBLIC_HISTORY_POLICY,
                ],
                hf_backend=self.hub,
                token="TOKEN-SENTINEL",
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["public_history"]["presence_count"], 2)
        visible = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn("LEGACY-PUBLIC-CONTENT-MUST-NOT-BE-READ", visible)

    def test_private_partial_different_and_deleted_historical_namespace_refuse(self):
        self.prepare()
        staged = {
            path: (self.private_stage / path).read_bytes()
            for path in ("private/test_labels.jsonl", "private/test_release.json")
        }
        tree = self.hub.trees[
            (publisher.PRIVATE_HF_REPOSITORY, self.hub.private_revision)
        ]
        tree["private/test_labels.jsonl"] = staged["private/test_labels.jsonl"]
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()
        tree["private/test_release.json"] = staged["private/test_release.json"]
        tree["private/test_labels.jsonl"] += b" "
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()
        tree.pop("private/test_labels.jsonl")
        tree.pop("private/test_release.json")
        old_revision = "9" * 40
        self.hub.trees[(publisher.PRIVATE_HF_REPOSITORY, old_revision)] = {
            **self.private_tree,
            "private/test_release.json": staged["private/test_release.json"],
        }
        self.hub.private_history.append(old_revision)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()

    def test_private_superseded_historical_namespace_refuses_even_if_current_is_exact(
        self,
    ):
        self.prepare()
        current = self.hub.trees[
            (publisher.PRIVATE_HF_REPOSITORY, self.hub.private_revision)
        ]
        for path in ("private/test_labels.jsonl", "private/test_release.json"):
            current[path] = (self.private_stage / path).read_bytes()
        old_revision = "7" * 40
        self.hub.trees[(publisher.PRIVATE_HF_REPOSITORY, old_revision)] = {
            **self.private_tree,
            "private/test_legacy.json": b"{}\n",
        }
        self.hub.private_history.append(old_revision)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()

    def test_legacy_history_defaults_to_reject_but_named_profile_matches_dry_run(self):
        self.prepare()
        self.configure_exact_legacy_history()
        with self.assertRaises(publisher.ReleaseError):
            self.release_call()
        result = self.release_call(legacy_history_policy=self.LEGACY_POLICY)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(
            result["legacy_history"],
            {
                "policy_id": self.LEGACY_POLICY,
                "result": "matched-prepublish",
                "legacy_event_count": 2,
                "metadata_sha256": publisher.LEGACY_HISTORY_METADATA_SHA256,
                "blob_objects_fetched": False,
                "blob_contents_read": False,
                "legacy_content_compared": False,
                "legacy_history_retained": True,
            },
        )
        self.assertEqual(
            result["operation"], {"overwrites": 0, "deletes": 0, "retries": 0}
        )
        self.assertEqual(self.hub.writes, [])
        private_reads = [
            path
            for repository, _revision, paths in self.hub.reads
            if repository == publisher.PRIVATE_HF_REPOSITORY
            for path in paths
        ]
        self.assertEqual(private_reads, [])

    def test_legacy_profile_rejects_every_event_stream_mutation(self):
        self.prepare()
        base = self.exact_legacy_events()
        extra = publisher.PrivateHistoryEvent(
            revision="5" * 40,
            parents=(self.LEGACY_DELETE,),
            timestamp="2020-01-01T00:02:00Z",
            status="M",
            path="private/test_labels.jsonl",
            subject="PRIVATE-EXTRA-SUBJECT",
        )
        mutations = (
            (replace(base[0], status="D"), base[1]),
            (replace(base[0], path="private/test_release.json"), base[1]),
            (replace(base[0], revision="3" * 40), base[1]),
            (replace(base[0], parents=("2" * 40,)), base[1]),
            (replace(base[0], parents=("3" * 40, "4" * 40)), base[1]),
            (replace(base[0], timestamp="2020-01-01T00:00:01Z"), base[1]),
            (replace(base[0], timestamp="2020-02-31T00:00:00Z"), base[1]),
            (base[0], replace(base[1], revision="3" * 40)),
            (base[0], replace(base[1], parents=("4" * 40,))),
            (base[0], replace(base[1], parents=(self.LEGACY_ADD, "4" * 40))),
            (base[0], replace(base[1], timestamp="2020-01-01T00:01:01Z")),
            (base[0], replace(base[1], status="M")),
            (base[0], replace(base[1], path="private/test_release.json")),
            tuple(reversed(base)),
            base[:1],
            (*base, extra),
        )
        for events in mutations:
            with self.subTest(events=events):
                self.hub.history_events_override = tuple(events)
                self.hub.history_reachable_override = {
                    event.revision for event in events
                } | {self.hub.private_revision}
                with self.assertRaises(publisher.ReleaseError):
                    self.release_call(legacy_history_policy=self.LEGACY_POLICY)
        self.configure_exact_legacy_history()
        self.hub.history_reachable_override.remove(self.LEGACY_ADD)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_history_policy=self.LEGACY_POLICY)
        self.configure_exact_legacy_history()
        self.hub.history_reachable_override.remove(self.LEGACY_DELETE)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_history_policy=self.LEGACY_POLICY)

    def test_legacy_options_are_closed_and_publish_requires_both_confirmations(self):
        self.prepare()
        self.configure_exact_legacy_history()
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_history_policy="allow")
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(legacy_history_confirmation=self.LEGACY_CONFIRM)
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
                legacy_history_policy=self.LEGACY_POLICY,
            )
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
                legacy_history_policy=self.LEGACY_POLICY,
                legacy_history_confirmation="wrong",
            )
        self.assertEqual(self.hub.writes, [])
        invalid_stage = self.root / "invalid-legacy-prepare"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = publisher.main(
                [
                    "--public-stage",
                    str(self.public_stage),
                    "--private-label-source",
                    str(self.private_source),
                    "--private-stage",
                    str(invalid_stage),
                    "--prepare-stage",
                    "--legacy-history-policy",
                    self.LEGACY_POLICY,
                ]
            )
        self.assertEqual(code, 2)
        self.assertFalse(invalid_stage.exists())

    def test_legacy_audit_rejects_head_depth_blob_and_graph_malformations(self):
        self.configure_exact_legacy_history()
        audit = self.hub.history_audit(
            publisher.PRIVATE_HF_REPOSITORY, self.hub.private_revision, "token"
        )
        malformed = (
            replace(audit, head="f" * 40),
            replace(audit, shallow=True),
            replace(audit, blob_objects_fetched=True),
            replace(audit, parent_map={}),
            replace(
                audit,
                reachable=frozenset(f"{index:040x}" for index in range(10_001)),
            ),
        )
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(publisher.ReleaseError):
                    publisher._validate_private_history(
                        value,
                        expected_head=self.hub.private_revision,
                        current_status="pending",
                        legacy_policy=self.LEGACY_POLICY,
                    )

    def test_legacy_publish_adds_two_files_then_postscan_and_idempotency_match(self):
        self.prepare()
        self.configure_exact_legacy_history()
        result = self.release_call(
            publish=True,
            confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            legacy_history_policy=self.LEGACY_POLICY,
            legacy_history_confirmation=self.LEGACY_CONFIRM,
        )
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["legacy_history"]["result"], "matched-postpublish")
        self.assertEqual(len(self.hub.writes), 1)
        second_config = publisher.ReleaseConfig(
            self.public_stage, self.private_source, self.private_stage, "d" * 40
        )
        second = publisher.run_private_continuation(
            second_config,
            hf_backend=self.hub,
            token="token",
            publish=True,
            confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            legacy_history_policy=self.LEGACY_POLICY,
            legacy_history_confirmation=self.LEGACY_CONFIRM,
        )
        self.assertEqual(second["status"], "already-published")
        self.assertEqual(second["legacy_history"]["result"], "matched-postpublish")
        self.assertEqual(len(self.hub.writes), 1)
        visible = json.dumps(second)
        for secret in (
            "SYNTHETIC-LEGACY-SUBJECT-PRIVATE",
            "SYNTHETIC-DELETE-SUBJECT-PRIVATE",
            "PRIVATE-ANSWER",
        ):
            self.assertNotIn(secret, visible)

    def test_legacy_postpublish_rejects_message_change_or_later_test_event(self):
        self.prepare()
        self.configure_exact_legacy_history()
        self.release_call(
            publish=True,
            confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            legacy_history_policy=self.LEGACY_POLICY,
            legacy_history_confirmation=self.LEGACY_CONFIRM,
        )
        post = tuple(self.hub.history_events_override)
        second_config = publisher.ReleaseConfig(
            self.public_stage, self.private_source, self.private_stage, "d" * 40
        )
        later = publisher.PrivateHistoryEvent(
            revision="e" * 40,
            parents=("d" * 40,),
            timestamp="2026-09-05T20:01:00Z",
            status="M",
            path="private/test_release.json",
            subject="PRIVATE-LATER-EVENT",
        )
        for events in (
            post[:2],
            (*post[:-1], replace(post[-1], subject="PRIVATE-WRONG-MESSAGE")),
            (*post[:2], post[3], post[2]),
            (
                *post[:2],
                replace(post[2], revision=self.LEGACY_ADD),
                replace(post[3], revision=self.LEGACY_ADD),
            ),
            (
                *post[:2],
                replace(post[2], parents=("f" * 40,)),
                replace(post[3], parents=("f" * 40,)),
            ),
            (*post, later),
        ):
            with self.subTest(events=events):
                self.hub.history_events_override = events
                self.hub.history_reachable_override = {
                    event.revision for event in events
                } | {"d" * 40}
                with self.assertRaises(publisher.ReleaseError):
                    publisher.run_private_continuation(
                        second_config,
                        hf_backend=self.hub,
                        token="token",
                        legacy_history_policy=self.LEGACY_POLICY,
                    )

    def test_publish_is_exact_two_add_cas_preserves_non_test_inventory_and_is_idempotent(
        self,
    ):
        self.prepare()
        before = dict(
            self.hub.trees[(publisher.PRIVATE_HF_REPOSITORY, self.hub.private_revision)]
        )
        result = self.release_call(
            publish=True,
            confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
        )
        self.assertEqual(result["activation"], "not-performed")
        self.assertNotIn("mode", result)
        self.assertEqual(len(self.hub.writes), 1)
        write = self.hub.writes[0]
        self.assertEqual(write["repository"], publisher.PRIVATE_HF_REPOSITORY)
        self.assertEqual(write["parent"], "b" * 40)
        self.assertEqual(
            write["paths"],
            ("private/test_labels.jsonl", "private/test_release.json"),
        )
        self.assertEqual(
            write["message"],
            f"Install {self.manifest['release_id']} disabled",
        )
        self.assertTrue(
            all(self.private_stage not in source.parents for source in write["sources"])
        )
        after = self.hub.trees[(publisher.PRIVATE_HF_REPOSITORY, "d" * 40)]
        self.assertEqual({path: after[path] for path in before}, before)
        private_reads = [
            path
            for repository, _revision, paths in self.hub.reads
            if repository == publisher.PRIVATE_HF_REPOSITORY
            for path in paths
        ]
        self.assertEqual(
            set(private_reads),
            {"private/test_labels.jsonl", "private/test_release.json"},
        )
        second_config = publisher.ReleaseConfig(
            self.public_stage,
            self.private_source,
            self.private_stage,
            "d" * 40,
        )
        second = publisher.run_private_continuation(
            second_config,
            hf_backend=self.hub,
            token="token",
            publish=True,
            confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            public_auditor=self.audit,
        )
        self.assertEqual(second["status"], "already-published")
        self.assertEqual(len(self.hub.writes), 1)

    def test_remote_movement_and_post_write_tamper_refuse_without_retry(self):
        self.prepare()
        self.hub.move_before_publish = True
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            )
        self.assertEqual(self.hub.writes, [])
        self.hub.move_before_publish = False
        self.hub.private_revision = "b" * 40
        self.hub.post_write_tamper = True
        with self.assertRaises(publisher.PublicationUncertainError) as caught:
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            )
        self.assertEqual(len(self.hub.writes), 1)
        self.assertIn("may have landed", str(caught.exception))
        self.assertNotIn("PRIVATE", str(caught.exception).replace("Private", ""))

    def test_lost_publish_response_is_uncertain_and_never_retried(self):
        self.prepare()
        self.hub.raise_after_commit = True
        with self.assertRaises(publisher.PublicationUncertainError) as caught:
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            )
        self.assertEqual(len(self.hub.writes), 1)
        self.assertIn("outcome is uncertain", str(caught.exception))
        self.assertNotIn("PRIVATE-UNCERTAIN-RESPONSE", str(caught.exception))

    def test_cli_reports_uncertain_landed_write_distinctly_and_safely(self):
        self.prepare()
        self.configure_exact_legacy_history()
        self.hub.raise_after_commit = True
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = publisher.main(
                [
                    "--public-stage",
                    str(self.public_stage),
                    "--private-label-source",
                    str(self.private_source),
                    "--private-stage",
                    str(self.private_stage),
                    "--private-hf-base",
                    "b" * 40,
                    "--publish",
                    "--confirm",
                    "PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
                    "--legacy-history-policy",
                    self.LEGACY_POLICY,
                    "--confirm-legacy-history",
                    self.LEGACY_CONFIRM,
                ],
                hf_backend=self.hub,
                token="TOKEN-SENTINEL-K4p9Vr2M",
            )
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "uncertain")
        self.assertEqual(len(self.hub.writes), 1)
        visible = stdout.getvalue() + stderr.getvalue()
        for secret in (
            "PRIVATE-UNCERTAIN-RESPONSE",
            "PRIVATE-ANSWER",
            "TOKEN-SENTINEL",
        ):
            self.assertNotIn(secret, visible)

    def test_public_visibility_change_at_final_write_boundary_prevents_cas(self):
        self.prepare()
        self.hub.public_visibility_flip_on_second_identity = True
        with self.assertRaises(publisher.ReleaseError):
            self.release_call(
                publish=True,
                confirmation="PUBLISH_DISABLED_PRIVATE_TEST_RELEASE",
            )
        self.assertEqual(self.hub.writes, [])

    def test_cli_output_and_errors_are_sanitized(self):
        self.prepare()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = publisher.main(
                [
                    "--public-stage",
                    str(self.public_stage),
                    "--private-label-source",
                    str(self.private_source),
                    "--private-stage",
                    str(self.private_stage),
                    "--private-hf-base",
                    "b" * 40,
                ],
                hf_backend=self.hub,
                token="TOKEN-SENTINEL-K4p9Vr2M",
                public_auditor=self.audit,
            )
        self.assertEqual(code, 0)
        visible = stdout.getvalue() + stderr.getvalue()
        for secret in (
            "PRIVATE-ANSWER",
            "A0#9",
            "TOKEN-SENTINEL",
            self.private_sentinel.decode().strip(),
        ):
            self.assertNotIn(secret, visible)
        self.hub.username = "PRIVATE-USER-SENTINEL-J8q3"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = publisher.main(
                [
                    "--public-stage",
                    str(self.public_stage),
                    "--private-label-source",
                    str(self.private_source),
                    "--private-stage",
                    str(self.private_stage),
                    "--private-hf-base",
                    "b" * 40,
                ],
                hf_backend=self.hub,
                token="TOKEN-SENTINEL-K4p9Vr2M",
                public_auditor=self.audit,
            )
        self.assertEqual(code, 2)
        self.assertNotIn("PRIVATE-USER-SENTINEL", stderr.getvalue())

    def test_cli_can_use_the_locally_authenticated_token_without_exposing_it(self):
        self.prepare()
        self.hub.local_token = lambda: "LOCAL-TOKEN-SENTINEL-Q7m2"
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = publisher.main(
                    [
                        "--public-stage",
                        str(self.public_stage),
                        "--private-label-source",
                        str(self.private_source),
                        "--private-stage",
                        str(self.private_stage),
                        "--private-hf-base",
                        "b" * 40,
                    ],
                    hf_backend=self.hub,
                    public_auditor=self.audit,
                )
        self.assertEqual(code, 0)
        self.assertNotIn("LOCAL-TOKEN-SENTINEL", stdout.getvalue() + stderr.getvalue())

    def test_concrete_remote_reads_are_bounded_in_memory_without_download_cache(self):
        class Response:
            headers = {"Content-Length": "3"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                return iter((b"abc",))

        class Session:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return Response()

        session = Session()
        backend = publisher.HuggingFaceBackend()
        imports = (
            lambda **kwargs: "https://example.invalid/file",
            lambda **kwargs: {"authorization": "Bearer private"},
            lambda: session,
        )
        with (
            mock.patch.object(
                backend,
                "_imports",
                side_effect=AssertionError("disk download forbidden"),
            ),
            mock.patch.object(
                backend, "_read_imports", return_value=imports, create=True
            ),
        ):
            value = backend.read_files(
                "owner/data", "a" * 40, ["test/release.json"], "token"
            )
        self.assertEqual(value, {"test/release.json": b"abc"})
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.calls[0][1]["stream"])

    def test_concrete_file_inventory_skips_repo_folders(self):
        class Lfs:
            sha256 = "1" * 64

        class File:
            path = "test/documents/test_010001.pdf"
            size = 123
            lfs = Lfs()

        class Folder:
            path = "test/documents"

        class Api:
            def list_repo_tree(self, **kwargs):
                return (Folder(), File())

        backend = publisher.HuggingFaceBackend()
        metadata = {
            path: self.hub.trees[
                (publisher.PUBLIC_HF_REPOSITORY, self.hub.public_revision)
            ][path]
            for path in publisher._PUBLIC_METADATA_PATHS
        }
        with (
            mock.patch.object(
                backend,
                "_imports",
                return_value=(object, lambda token: Api(), object),
            ),
            mock.patch.object(backend, "read_files", return_value=metadata),
        ):
            inventory = backend.file_inventory(
                publisher.PUBLIC_HF_REPOSITORY, "a" * 40, "token"
            )
        self.assertEqual(
            set(inventory),
            set(publisher._PUBLIC_METADATA_PATHS) | {"test/documents/test_010001.pdf"},
        )
        self.assertNotIn("test/documents", inventory)

    def test_private_history_uses_one_no_blob_path_only_git_fetch(self):
        remote = self.root / "history.git"
        work = self.root / "history-work"
        subprocess.run(
            ["git", "init", "--bare", str(remote)], check=True, capture_output=True
        )
        subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(work), "config", "user.name", "Synthetic"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(work),
                "config",
                "user.email",
                "synthetic@example.invalid",
            ],
            check=True,
        )
        (work / "private").mkdir()
        (work / "private/test_labels.jsonl").write_bytes(b"SYNTHETIC-OLD\n")
        subprocess.run(["git", "-C", str(work), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", "old"],
            check=True,
            capture_output=True,
        )
        (work / "private/test_labels.jsonl").unlink()
        (work / "README.md").write_bytes(b"current\n")
        subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", "current"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(work), "branch", "-M", "main"], check=True)
        subprocess.run(
            ["git", "-C", str(work), "push", str(remote), "main"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(remote), "config", "uploadpack.allowFilter", "true"],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(work), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        backend = publisher.HuggingFaceBackend()
        paths = backend._history_path_names_from_remote(
            remote.as_uri(), head, "TOKEN-SENTINEL-V6q2"
        )
        self.assertIn("private/test_labels.jsonl", paths)
        self.assertIn("README.md", paths)
        audit = backend._history_audit_from_remote(
            remote.as_uri(), head, "TOKEN-SENTINEL-V6q2"
        )
        self.assertEqual(audit.head, head)
        self.assertEqual(
            [(event.status, event.path) for event in audit.events],
            [
                ("A", "private/test_labels.jsonl"),
                ("D", "private/test_labels.jsonl"),
            ],
        )
        self.assertEqual(audit.events[0].parents, ())
        self.assertEqual(audit.events[1].parents, (audit.events[0].revision,))
        self.assertFalse(audit.shallow)
        self.assertFalse(audit.blob_objects_fetched)
        self.assertEqual([event.subject for event in audit.events], ["", ""])
        (work / "private/test_labels.jsonl").write_bytes(b"SYNTHETIC-NEW\n")
        (work / "private/test_release.json").write_bytes(b"{}\n")
        subprocess.run(["git", "-C", str(work), "add", "."], check=True)
        message = f"Install {publisher.RELEASE_ID} disabled"
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", message],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(work), "push", str(remote), "main"],
            check=True,
            capture_output=True,
        )
        new_head = subprocess.run(
            ["git", "-C", str(work), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        post = backend._history_audit_from_remote(
            remote.as_uri(), new_head, "TOKEN-SENTINEL-V6q2"
        )
        self.assertEqual(
            [(event.status, event.path) for event in post.events],
            [
                ("A", "private/test_labels.jsonl"),
                ("D", "private/test_labels.jsonl"),
                ("A", "private/test_labels.jsonl"),
                ("A", "private/test_release.json"),
            ],
        )
        self.assertEqual([event.subject for event in post.events[:2]], ["", ""])
        self.assertEqual(
            [event.subject for event in post.events[2:]], [message, message]
        )
        self.assertEqual(post.events[2].revision, post.events[3].revision)
        self.assertEqual(post.events[2].parents, post.events[3].parents)
        self.assertEqual(post.events[2].timestamp, post.events[3].timestamp)
        environment = backend._git_environment("TOKEN-SENTINEL-V6q2")
        self.assertNotIn("TOKEN-SENTINEL-V6q2", repr(environment))
        self.assertIn("Authorization: Basic ", environment["GIT_CONFIG_VALUE_2"])
        with self.assertRaises(publisher.ReleaseError):
            backend._verify_no_blob_objects(remote, environment)
        subprocess.run(
            ["git", "-C", str(remote), "config", "uploadpack.allowFilter", "false"],
            check=True,
        )
        with self.assertRaises(publisher.ReleaseError):
            backend._history_path_names_from_remote(
                remote.as_uri(), new_head, "TOKEN-SENTINEL-V6q2"
            )

    def test_public_history_audit_uses_only_no_blob_path_and_tree_metadata(self):
        remote = self.root / "public-history.git"
        work = self.root / "public-history-work"
        subprocess.run(
            ["git", "init", "--bare", str(remote)], check=True, capture_output=True
        )
        subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(work), "config", "user.name", "Synthetic"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(work),
                "config",
                "user.email",
                "synthetic@example.invalid",
            ],
            check=True,
        )
        (work / "data").mkdir()
        (work / "data/dev_labels.jsonl").write_bytes(
            b"SYNTHETIC-PUBLIC-LABEL-BODY-MUST-NOT-BE-READ\n"
        )
        subprocess.run(["git", "-C", str(work), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", "synthetic add"],
            check=True,
            capture_output=True,
        )
        (work / "README.md").write_bytes(b"present snapshot\n")
        subprocess.run(["git", "-C", str(work), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", "synthetic present"],
            check=True,
            capture_output=True,
        )
        (work / "data/dev_labels.jsonl").unlink()
        subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", "synthetic delete"],
            check=True,
            capture_output=True,
        )
        (work / "README.md").write_bytes(b"current snapshot\n")
        subprocess.run(["git", "-C", str(work), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(work), "commit", "-m", "synthetic current"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(work), "branch", "-M", "main"], check=True)
        subprocess.run(
            ["git", "-C", str(work), "push", str(remote), "main"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(remote), "config", "uploadpack.allowFilter", "true"],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(work), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        backend = publisher.HuggingFaceBackend()
        commands = []
        original = backend._run_history_git

        def observed(arguments, **kwargs):
            commands.append(tuple(arguments))
            return original(arguments, **kwargs)

        with mock.patch.object(backend, "_run_history_git", side_effect=observed):
            audit = backend._public_history_audit_from_remote(
                remote.as_uri(), head, "TOKEN-SENTINEL-PUBLIC-V6q2"
            )
        self.assertEqual(audit.head, head)
        self.assertEqual(len(audit.graph), 4)
        self.assertEqual(
            [(event.status, event.path) for event in audit.events],
            [
                ("A", "data/dev_labels.jsonl"),
                ("D", "data/dev_labels.jsonl"),
            ],
        )
        self.assertEqual(
            [present for _revision, present in audit.presence],
            [True, True, False, False],
        )
        self.assertFalse(audit.shallow)
        self.assertFalse(audit.blob_objects_fetched)
        self.assertFalse(any(command and command[0] == "show" for command in commands))
        self.assertNotIn("SYNTHETIC-PUBLIC-LABEL-BODY", repr(audit))


if __name__ == "__main__":
    unittest.main(verbosity=2)
