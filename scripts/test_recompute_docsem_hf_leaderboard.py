#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition" / "hf-space"))
sys.path.insert(0, str(ROOT / "scripts"))

from recompute_docsem_hf_leaderboard import (  # noqa: E402
    apply_label_corrections,
    recompute_submission_payload,
)


class RecomputeTests(unittest.TestCase):
    def test_corrections_replace_only_named_validation_answers(self):
        labels = [
            {"instance_id": "task_1", "answer": "10", "evidence": ["b01"]},
            {"instance_id": "task_2", "answer": "20", "evidence": ["b02"]},
        ]

        corrected = apply_label_corrections(labels, {"task_2": "-20"})

        self.assertEqual(corrected[0]["answer"], "10")
        self.assertEqual(corrected[1]["answer"], "-20")

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
