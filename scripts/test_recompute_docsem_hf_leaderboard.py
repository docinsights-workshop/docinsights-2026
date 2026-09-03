#!/usr/bin/env python3
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "hf-space"))
sys.path.insert(0, str(ROOT / "scripts"))

import recompute_docsem_hf_leaderboard as recompute  # noqa: E402
from recompute_docsem_hf_leaderboard import (  # noqa: E402
    _parse_corrections,
    apply_label_corrections,
    recompute_submission_payload,
)


class RecomputeTests(unittest.TestCase):
    def _run_main_fixture(self, *, yes, maintenance_confirmed=True):
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
            corrections_file=corrections_path,
            repo_id="private/repo",
            gold_file="private/val_labels.jsonl",
            leaderboard_file="leaderboard/leaderboard.json",
            yes=yes,
            maintenance_confirmed=maintenance_confirmed,
        )
        labels_text = (
            '{"instance_id":"task_1","answer":"10","evidence":["b01"]}\n'
        )
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
        api = MagicMock()
        api.repo_info.return_value = SimpleNamespace(sha="base123")
        api.list_repo_files.return_value = [
            "private/val_labels.jsonl",
            "submissions/private.json",
        ]
        api.create_commit.return_value = SimpleNamespace(
            commit_url="https://example.invalid/commit"
        )

        def read_remote(_repo_id, filename, _token, **_kwargs):
            if filename == "private/val_labels.jsonl":
                return labels_text
            return json.dumps(submission_payload)

        output = io.StringIO()
        with (
            patch.object(recompute, "_parser") as parser,
            patch.object(recompute, "get_token", return_value="token"),
            patch.object(recompute, "HfApi", return_value=api),
            patch.object(recompute, "_read_remote_text", side_effect=read_remote) as reader,
            redirect_stdout(output),
        ):
            parser.return_value.parse_args.return_value = args
            recompute.main()
        return output.getvalue(), api, reader

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

        with self.assertRaisesRegex(ValueError, "does not match the expected value") as caught:
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

        with self.assertRaisesRegex(ValueError, "Duplicate validation label IDs") as caught:
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

        with self.assertRaisesRegex(ValueError, "unknown validation instance IDs") as caught:
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
