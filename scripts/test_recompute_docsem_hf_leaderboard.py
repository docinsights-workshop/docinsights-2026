#!/usr/bin/env python3
import io
import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "hf-space"))
sys.path.insert(0, str(ROOT / "scripts"))

import recompute_docsem_hf_leaderboard as recompute  # noqa: E402
from recompute_docsem_hf_leaderboard import (  # noqa: E402
    _parse_corrections,
    _parser,
    _read_remote_bytes,
    apply_label_corrections,
    migrate_joint_metric_payload,
    recompute_submission_payload,
)


class RecomputeTests(unittest.TestCase):
    def test_remote_reader_preserves_exact_label_bytes(self):
        raw = (
            b'{"instance_id":"task_1","answer":"10","evidence":["b01"]}\r\n'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.jsonl"
            path.write_bytes(raw)
            with patch.object(recompute, "hf_hub_download", return_value=str(path)):
                loaded = _read_remote_bytes(
                    "private/repo",
                    "private/val_labels.jsonl",
                    "token",
                    revision="base123",
                    cache_dir=Path(temp_dir) / "cache",
                )

        self.assertEqual(loaded, raw)

    def _run_main_fixture(
        self,
        *,
        yes,
        maintenance_confirmed=True,
        joint_metric_migration=False,
        expected_submission_count=None,
        submission_paths=None,
        submission_markers=None,
        labels_before_bytes=None,
        labels_after_bytes=None,
    ):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        corrections_path = Path(temp_dir.name) / "corrections.json"
        corrections_path.write_text(
            json.dumps(
                {
                    "task_1": {
                        "expected": "10",
                        "replacement": "11",
                    }
                }
            ),
            encoding="utf-8",
        )
        args = SimpleNamespace(
            corrections_file=None if joint_metric_migration else corrections_path,
            joint_metric_migration=joint_metric_migration,
            expected_submission_count=expected_submission_count,
            repo_id="private/repo",
            gold_file="private/val_labels.jsonl",
            leaderboard_file="leaderboard/leaderboard.json",
            yes=yes,
            maintenance_confirmed=maintenance_confirmed,
        )
        labels_text = '{"instance_id":"task_1","answer":"10","evidence":["b01"]}\n'
        labels_before_bytes = labels_before_bytes or labels_text.encode("utf-8")
        labels_after_bytes = labels_after_bytes or labels_before_bytes
        submission_payload = {
            "leaderboard": {
                "team": "Private Team",
                "contact": "secret@example.org",
                "submission_name": "private-run",
                "submitted_at": "2026-09-03T12:00:00Z",
                "answer_accuracy": 1.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 1,
            },
            "metrics": {
                "answer_accuracy": 1.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 1,
                "per_example": [
                    {
                        "instance_id": "task_1",
                        "answer_exact_match": 1.0,
                        "evidence_exact_match": 1.0,
                        "evidence_f1": 1.0,
                    }
                ],
            },
            "predictions": [
                {"instance_id": "task_1", "answer": "10", "evidence": ["b01"]}
            ],
        }
        submission_paths = submission_paths or ["submissions/private.json"]
        submission_markers = submission_markers or {}
        submission_payloads = {}
        for repo_path in submission_paths:
            payload = copy.deepcopy(submission_payload)
            if repo_path in submission_markers:
                payload["private_fixture_marker"] = submission_markers[repo_path]
            submission_payloads[repo_path] = payload

        api = MagicMock()
        api.repo_info.return_value = SimpleNamespace(sha="base123")
        api.list_repo_files.return_value = [
            "private/val_labels.jsonl",
            *submission_paths,
        ]
        sealed_operations = {}

        def create_commit(**kwargs):
            for operation in kwargs["operations"]:
                source = operation.path_or_fileobj
                if hasattr(source, "read"):
                    position = source.tell()
                    source.seek(0)
                    payload_bytes = source.read()
                    source.seek(position)
                else:
                    payload_bytes = Path(source).read_bytes()
                sealed_operations[operation.path_in_repo] = payload_bytes
            return SimpleNamespace(
                commit_url="https://example.invalid/commit",
                oid="next123",
            )

        api.create_commit.side_effect = create_commit
        api.sealed_operations = sealed_operations

        def read_remote(_repo_id, filename, _token, **_kwargs):
            if filename == "private/val_labels.jsonl":
                return labels_text
            return json.dumps(submission_payloads[filename])

        def read_remote_bytes(_repo_id, filename, _token, **kwargs):
            if filename != "private/val_labels.jsonl":
                return read_remote(_repo_id, filename, _token, **kwargs).encode("utf-8")
            return (
                labels_after_bytes
                if kwargs["revision"] == "next123"
                else labels_before_bytes
            )

        output = io.StringIO()
        with (
            patch.object(recompute, "_parser") as parser,
            patch.object(recompute, "get_token", return_value="token"),
            patch.object(recompute, "HfApi", return_value=api),
            patch.object(
                recompute, "_read_remote_text", side_effect=read_remote
            ) as reader,
            patch.object(
                recompute, "_read_remote_bytes", side_effect=read_remote_bytes
            ) as byte_reader,
            redirect_stdout(output),
        ):
            parser.return_value.parse_args.return_value = args
            recompute.main()
        api.byte_reader = byte_reader
        return output.getvalue(), api, reader

    def test_cli_modes_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parser().parse_args(
                    [
                        "--corrections-file",
                        "corrections.json",
                        "--joint-metric-migration",
                    ]
                )

    def test_joint_migration_adds_only_joint_fields(self):
        labels = [
            {"instance_id": "task_1", "answer": "10", "evidence": ["b01"]},
            {"instance_id": "task_2", "answer": "20", "evidence": ["b02"]},
        ]
        payload = {
            "leaderboard": {
                "team": "Example Team",
                "contact": "lead@example.org",
                "participant_names": "Alice Example",
                "submission_name": "run-1",
                "submitted_at": "2026-08-31T12:00:00Z",
                "answer_accuracy": 0.5,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 2,
                "legacy_leaderboard_extension": "preserved",
            },
            "metrics": {
                "answer_accuracy": 0.5,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 2,
                "legacy_metric_extension": "preserved",
                "per_example": [
                    {
                        "instance_id": "task_1",
                        "answer_exact_match": 1.0,
                        "evidence_exact_match": 1.0,
                        "evidence_f1": 1.0,
                        "legacy_example_extension": "preserved",
                    },
                    {
                        "instance_id": "task_2",
                        "answer_exact_match": 0.0,
                        "evidence_exact_match": 1.0,
                        "evidence_f1": 1.0,
                    },
                ],
            },
            "predictions": [
                {
                    "instance_id": "task_1",
                    "answer": "10",
                    "evidence": ["b01"],
                    "legacy_prediction_extension": "preserved",
                },
                {"instance_id": "task_2", "answer": "21", "evidence": ["b02"]},
            ],
            "legacy_payload_extension": {"preserved": True},
        }
        original = copy.deepcopy(payload)

        updated = migrate_joint_metric_payload(payload, labels)

        self.assertEqual(payload, original)
        self.assertEqual(updated["predictions"], original["predictions"])
        self.assertEqual(
            updated["legacy_payload_extension"], original["legacy_payload_extension"]
        )
        self.assertEqual(
            updated["leaderboard"]["legacy_leaderboard_extension"], "preserved"
        )
        self.assertEqual(updated["metrics"]["legacy_metric_extension"], "preserved")
        self.assertEqual(
            updated["metrics"]["per_example"][0]["legacy_example_extension"],
            "preserved",
        )
        for field in (
            "answer_accuracy",
            "evidence_exact_match",
            "evidence_f1",
            "examples",
        ):
            self.assertEqual(updated["metrics"][field], original["metrics"][field])
            self.assertEqual(
                updated["leaderboard"][field], original["leaderboard"][field]
            )
        self.assertEqual(updated["metrics"]["joint_accuracy"], 0.5)
        self.assertEqual(updated["leaderboard"]["joint_accuracy"], 0.5)
        self.assertEqual(
            [row["joint_exact_match"] for row in updated["metrics"]["per_example"]],
            [1.0, 0.0],
        )

    def test_joint_migration_rejects_prior_metric_drift_without_private_values(self):
        labels = [
            {
                "instance_id": "private_task_sentinel",
                "answer": "10",
                "evidence": ["b01"],
            }
        ]
        payload = {
            "leaderboard": {
                "team": "Private Team",
                "contact": "secret@example.org",
                "answer_accuracy": 0.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 1,
            },
            "metrics": {
                "answer_accuracy": 0.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 1,
                "per_example": [
                    {
                        "instance_id": "private_task_sentinel",
                        "answer_exact_match": 0.0,
                        "evidence_exact_match": 1.0,
                        "evidence_f1": 1.0,
                    }
                ],
            },
            "predictions": [
                {
                    "instance_id": "private_task_sentinel",
                    "answer": "10",
                    "evidence": ["b01"],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "prior metrics") as caught:
            migrate_joint_metric_payload(payload, labels)

        error = str(caught.exception)
        self.assertNotIn("private_task_sentinel", error)
        self.assertNotIn("secret@example.org", error)
        self.assertNotIn("Private Team", error)

    def test_joint_migration_sanitizes_scoring_errors(self):
        labels = [
            {
                "instance_id": "private_missing_task_sentinel",
                "answer": "private_answer_sentinel",
                "evidence": ["private_evidence_sentinel"],
            }
        ]
        payload = {
            "leaderboard": {},
            "metrics": {"per_example": []},
            "predictions": [
                {
                    "instance_id": "private_extra_task_sentinel",
                    "answer": "private_prediction_sentinel",
                    "evidence": ["private_prediction_evidence_sentinel"],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "could not be re-scored") as caught:
            migrate_joint_metric_payload(payload, labels)

        error = str(caught.exception)
        for private_value in (
            "private_missing_task_sentinel",
            "private_answer_sentinel",
            "private_evidence_sentinel",
            "private_extra_task_sentinel",
            "private_prediction_sentinel",
            "private_prediction_evidence_sentinel",
        ):
            self.assertNotIn(private_value, error)

    def test_joint_migration_requires_exact_expected_submission_count(self):
        with self.assertRaisesRegex(RuntimeError, "expected submission count"):
            self._run_main_fixture(
                yes=True,
                joint_metric_migration=True,
                expected_submission_count=2,
            )

    def test_joint_migration_commit_is_atomic_omits_gold_and_proves_unchanged_sha(self):
        output, api, reader = self._run_main_fixture(
            yes=True,
            joint_metric_migration=True,
            expected_submission_count=1,
        )

        api.create_commit.assert_called_once()
        kwargs = api.create_commit.call_args.kwargs
        self.assertEqual(kwargs["parent_commit"], "base123")
        paths = [operation.path_in_repo for operation in kwargs["operations"]]
        self.assertEqual(
            paths,
            ["submissions/private.json", "leaderboard/leaderboard.json"],
        )
        self.assertNotIn("private/val_labels.jsonl", paths)
        gold_reads = [
            call
            for call in api.byte_reader.call_args_list
            if call.args[1] == "private/val_labels.jsonl"
        ]
        self.assertEqual(
            [call.kwargs["revision"] for call in gold_reads],
            ["base123", "next123"],
        )
        self.assertIn('"label_sha256_unchanged": "pending_verification"', output)
        self.assertIn('"label_sha256_unchanged": true', output)
        self.assertIn('"submissions_scanned": 1', output)
        self.assertNotIn("task_1", output)
        self.assertNotIn("secret@example.org", output)

    def test_joint_migration_seals_nested_same_basename_submissions_independently(self):
        paths = [
            "submissions/first/shared.json",
            "submissions/second/shared.json",
        ]
        markers = {
            paths[0]: "first-private-marker",
            paths[1]: "second-private-marker",
        }

        _output, api, _reader = self._run_main_fixture(
            yes=True,
            joint_metric_migration=True,
            expected_submission_count=2,
            submission_paths=paths,
            submission_markers=markers,
        )

        self.assertEqual(
            [
                json.loads(api.sealed_operations[path])["private_fixture_marker"]
                for path in paths
            ],
            ["first-private-marker", "second-private-marker"],
        )

    def test_joint_migration_rejects_an_exact_label_byte_change_after_commit(self):
        before = (
            b'{"instance_id":"task_1","answer":"10","evidence":["b01"]}\r\n'
        )
        after = before.replace(b"\r\n", b"\n")

        with self.assertRaisesRegex(RuntimeError, "integrity verification"):
            self._run_main_fixture(
                yes=True,
                joint_metric_migration=True,
                expected_submission_count=1,
                labels_before_bytes=before,
                labels_after_bytes=after,
            )

    def test_corrections_replace_only_named_validation_answers(self):
        labels = [
            {"instance_id": "task_1", "answer": "10", "evidence": ["b01"]},
            {"instance_id": "task_2", "answer": "20", "evidence": ["b02"]},
        ]

        corrected = apply_label_corrections(
            labels,
            {"task_2": {"expected": "20", "replacement": "-20"}},
        )

        self.assertEqual(corrected[0]["answer"], "10")
        self.assertEqual(corrected[1]["answer"], "-20")
        self.assertEqual(corrected[1]["evidence"], ["b02"])

    def test_corrections_reject_an_unexpected_current_answer(self):
        labels = [
            {"instance_id": "task_1", "answer": "10", "evidence": ["b01"]},
        ]

        with self.assertRaisesRegex(
            ValueError, "does not match the expected value"
        ) as caught:
            apply_label_corrections(
                labels,
                {"task_1": {"expected": "9", "replacement": "11"}},
            )
        self.assertNotIn("task_1", str(caught.exception))
        self.assertNotIn("9", str(caught.exception))
        self.assertNotIn("10", str(caught.exception))

    def test_corrections_reject_duplicate_gold_ids(self):
        labels = [
            {
                "instance_id": "private_duplicate_sentinel",
                "answer": "10",
                "evidence": ["b01"],
            },
            {
                "instance_id": "private_duplicate_sentinel",
                "answer": "10",
                "evidence": ["b01"],
            },
        ]

        with self.assertRaisesRegex(
            ValueError, "Duplicate validation label IDs"
        ) as caught:
            apply_label_corrections(
                labels,
                {
                    "private_duplicate_sentinel": {
                        "expected": "10",
                        "replacement": "11",
                    }
                },
            )
        self.assertNotIn("private_duplicate_sentinel", str(caught.exception))

    def test_corrections_reject_unknown_ids_without_disclosing_them(self):
        labels = [
            {"instance_id": "task_1", "answer": "10", "evidence": ["b01"]},
        ]

        with self.assertRaisesRegex(
            ValueError, "unknown validation instance IDs"
        ) as caught:
            apply_label_corrections(
                labels,
                {
                    "private_unknown_sentinel": {
                        "expected": "10",
                        "replacement": "11",
                    }
                },
            )
        self.assertNotIn("private_unknown_sentinel", str(caught.exception))

    def test_corrections_file_requires_expected_and_replacement_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corrections.json"
            path.write_text(
                '{"task_1":{"expected":10,"replacement":-20}}',
                encoding="utf-8",
            )

            corrections = _parse_corrections(path)

        self.assertEqual(
            corrections,
            {"task_1": {"expected": "10", "replacement": "-20"}},
        )

    def test_dry_run_uses_one_pinned_snapshot_and_redacts_private_details(self):
        output, api, reader = self._run_main_fixture(yes=False)

        api.repo_info.assert_called_once_with(
            "private/repo",
            repo_type="dataset",
            revision="main",
            token="token",
        )
        api.list_repo_files.assert_called_once_with(
            "private/repo",
            repo_type="dataset",
            revision="base123",
            token="token",
        )
        for call in reader.call_args_list:
            self.assertEqual(call.kwargs["revision"], "base123")
            self.assertIn("cache_dir", call.kwargs)
        self.assertIn('"correction_count": 1', output)
        self.assertIn('"submissions_scanned": 1', output)
        self.assertNotIn("task_1", output)
        self.assertNotIn("secret@example.org", output)
        self.assertNotIn('"replacement": "11"', output)

    def test_commit_is_compare_and_swap_against_the_audited_snapshot(self):
        _output, api, _reader = self._run_main_fixture(yes=True)

        api.create_commit.assert_called_once()
        self.assertEqual(
            api.create_commit.call_args.kwargs.get("parent_commit"),
            "base123",
        )
        self.assertEqual(api.create_commit.call_args.kwargs.get("revision"), "main")

    def test_commit_requires_confirmed_submission_maintenance(self):
        with self.assertRaisesRegex(RuntimeError, "maintenance gate"):
            self._run_main_fixture(yes=True, maintenance_confirmed=False)

    def test_recompute_updates_metrics_and_preserves_submission_metadata(self):
        labels = [
            {"instance_id": "task_1", "answer": "-10", "evidence": ["b01"]},
            {"instance_id": "task_2", "answer": "20", "evidence": ["b02"]},
        ]
        payload = {
            "leaderboard": {
                "team": "Example Team",
                "contact": "lead@example.org",
                "submission_name": "run-1",
                "submitted_at": "2026-08-31T12:00:00Z",
                "answer_accuracy": 1.0,
                "evidence_f1": 1.0,
            },
            "metrics": {
                "answer_accuracy": 1.0,
                "evidence_exact_match": 1.0,
                "evidence_f1": 1.0,
                "examples": 2,
            },
            "predictions": [
                {"instance_id": "task_1", "answer": "10", "evidence": ["b01"]},
                {"instance_id": "task_2", "answer": "20", "evidence": ["b02"]},
            ],
        }

        updated = recompute_submission_payload(payload, labels)

        self.assertEqual(updated["metrics"]["answer_accuracy"], 0.5)
        self.assertEqual(updated["metrics"]["evidence_f1"], 1.0)
        self.assertEqual(updated["leaderboard"]["team"], "Example Team")
        self.assertEqual(updated["leaderboard"]["contact"], "lead@example.org")
        self.assertEqual(updated["leaderboard"]["submission_name"], "run-1")
        self.assertEqual(updated["leaderboard"]["answer_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
